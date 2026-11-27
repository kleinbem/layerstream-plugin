#![allow(clippy::unwrap_used, clippy::undocumented_unsafe_blocks)]

use std::ffi::{CStr, CString, c_char, c_void};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::ptr;

use layerstream_ffi::{
    layerstream_cancel_token_cancel, layerstream_cancel_token_free, layerstream_cancel_token_new,
    layerstream_clear_error, layerstream_engine_chat, layerstream_engine_chat_stream,
    layerstream_engine_free, layerstream_engine_generate, layerstream_engine_generate_stream,
    layerstream_engine_kv_clear, layerstream_engine_kv_trim, layerstream_engine_load_state,
    layerstream_engine_open, layerstream_engine_save_state, layerstream_engine_stats_json,
    layerstream_inspect_gguf_json, layerstream_json_object_gbnf, layerstream_json_schema_to_gbnf,
    layerstream_last_error, layerstream_parse_assistant_output_json, layerstream_probe_gpu_json,
    layerstream_string_free, layerstream_version,
};
use serde_json::Value as JsonValue;

fn take_c_string(ptr: *mut c_char) -> String {
    assert!(!ptr.is_null(), "expected non-null C string");
    let s = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .expect("valid UTF-8")
        .to_string();
    unsafe { layerstream_string_free(ptr) };
    s
}

fn last_error_string() -> Option<String> {
    let p = layerstream_last_error();
    if p.is_null() {
        None
    } else {
        Some(
            unsafe { CStr::from_ptr(p) }
                .to_str()
                .expect("valid UTF-8 error")
                .to_string(),
        )
    }
}

struct TempPath(PathBuf);
impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn write_synthetic_llama_gguf(tag: &str) -> TempPath {
    const N_VOCAB: u64 = 64;
    const N_EMBD: u64 = 64;
    const N_HEAD: u32 = 2;
    const N_HEAD_KV: u32 = 2;
    const N_FF: u64 = 128;
    const N_LAYERS: u32 = 2;
    const ALIGN: usize = 32;

    let write_str = |buf: &mut Vec<u8>, s: &str| {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    };
    let kv_str = |buf: &mut Vec<u8>, k: &str, v: &str| {
        write_str(buf, k);
        buf.extend_from_slice(&8u32.to_le_bytes());
        write_str(buf, v);
    };
    let kv_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
        write_str(buf, k);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    };
    let kv_f32 = |buf: &mut Vec<u8>, k: &str, v: f32| {
        write_str(buf, k);
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    };
    let kv_bool = |buf: &mut Vec<u8>, k: &str, v: bool| {
        write_str(buf, k);
        buf.extend_from_slice(&7u32.to_le_bytes());
        buf.push(u8::from(v));
    };

    let mut seed: u64 = 0x5EED_C0DE_1234_5678;
    let mut next_f32 = |scale: f32| -> f32 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let u = ((seed >> 40) as f32) / ((1u32 << 24) as f32);
        (u * 2.0 - 1.0) * scale
    };

    struct TensorSpec {
        name: String,
        dims: Vec<u64>,
        bytes: Vec<u8>,
        rel_offset: u64,
    }
    let mut tensors = Vec::new();
    let mut push_tensor = |name: String, dims: Vec<u64>, is_norm: bool| {
        let nelems: usize = dims.iter().map(|&d| d as usize).product();
        let mut bytes = Vec::with_capacity(nelems * 4);
        for _ in 0..nelems {
            let v = if is_norm {
                1.0 + next_f32(0.05)
            } else {
                next_f32(0.15)
            };
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        tensors.push(TensorSpec {
            name,
            dims,
            bytes,
            rel_offset: 0,
        });
    };

    push_tensor("token_embd.weight".into(), vec![N_EMBD, N_VOCAB], false);
    push_tensor("output_norm.weight".into(), vec![N_EMBD], true);
    push_tensor("output.weight".into(), vec![N_EMBD, N_VOCAB], false);
    for il in 0..N_LAYERS {
        push_tensor(format!("blk.{il}.attn_norm.weight"), vec![N_EMBD], true);
        push_tensor(
            format!("blk.{il}.attn_q.weight"),
            vec![N_EMBD, N_EMBD],
            false,
        );
        push_tensor(
            format!("blk.{il}.attn_k.weight"),
            vec![N_EMBD, N_EMBD],
            false,
        );
        push_tensor(
            format!("blk.{il}.attn_v.weight"),
            vec![N_EMBD, N_EMBD],
            false,
        );
        push_tensor(
            format!("blk.{il}.attn_output.weight"),
            vec![N_EMBD, N_EMBD],
            false,
        );
        push_tensor(format!("blk.{il}.ffn_norm.weight"), vec![N_EMBD], true);
        push_tensor(
            format!("blk.{il}.ffn_gate.weight"),
            vec![N_EMBD, N_FF],
            false,
        );
        push_tensor(format!("blk.{il}.ffn_up.weight"), vec![N_EMBD, N_FF], false);
        push_tensor(
            format!("blk.{il}.ffn_down.weight"),
            vec![N_FF, N_EMBD],
            false,
        );
    }

    let mut cur_off = 0usize;
    for t in &mut tensors {
        cur_off = (cur_off + ALIGN - 1) & !(ALIGN - 1);
        t.rel_offset = cur_off as u64;
        cur_off += t.bytes.len();
    }

    let mut kv_buf = Vec::new();
    let mut kv_count = 0u64;
    let arch = "llama";
    kv_str(&mut kv_buf, "general.architecture", arch);
    kv_count += 1;
    kv_str(&mut kv_buf, "general.name", "synthetic-ffi-test");
    kv_count += 1;
    kv_u32(&mut kv_buf, "llama.block_count", N_LAYERS);
    kv_count += 1;
    kv_u32(&mut kv_buf, "llama.context_length", 256);
    kv_count += 1;
    kv_u32(&mut kv_buf, "llama.embedding_length", N_EMBD as u32);
    kv_count += 1;
    kv_u32(&mut kv_buf, "llama.feed_forward_length", N_FF as u32);
    kv_count += 1;
    kv_u32(&mut kv_buf, "llama.attention.head_count", N_HEAD);
    kv_count += 1;
    kv_u32(&mut kv_buf, "llama.attention.head_count_kv", N_HEAD_KV);
    kv_count += 1;
    kv_f32(&mut kv_buf, "llama.attention.layer_norm_rms_epsilon", 1e-5);
    kv_count += 1;
    kv_f32(&mut kv_buf, "llama.rope.freq_base", 10_000.0);
    kv_count += 1;
    kv_str(&mut kv_buf, "tokenizer.ggml.model", "gpt2");
    kv_count += 1;
    kv_str(&mut kv_buf, "tokenizer.ggml.pre", "default");
    kv_count += 1;
    write_str(&mut kv_buf, "tokenizer.ggml.merges");
    kv_buf.extend_from_slice(&9u32.to_le_bytes());
    kv_buf.extend_from_slice(&8u32.to_le_bytes());
    kv_buf.extend_from_slice(&0u64.to_le_bytes());
    kv_count += 1;
    write_str(&mut kv_buf, "tokenizer.ggml.tokens");
    kv_buf.extend_from_slice(&9u32.to_le_bytes());
    kv_buf.extend_from_slice(&8u32.to_le_bytes());
    kv_buf.extend_from_slice(&N_VOCAB.to_le_bytes());
    for i in 0..N_VOCAB {
        let tok = match i {
            0 => "<bos>".to_string(),
            1 => "<eos>".to_string(),
            2 => " ".to_string(),
            _ => {
                let ch = (b'a' + ((i - 3) % 26) as u8) as char;
                if i < 29 {
                    ch.to_string()
                } else {
                    format!("{ch}{i}")
                }
            }
        };
        write_str(&mut kv_buf, &tok);
    }
    kv_count += 1;
    write_str(&mut kv_buf, "tokenizer.ggml.token_type");
    kv_buf.extend_from_slice(&9u32.to_le_bytes());
    kv_buf.extend_from_slice(&5u32.to_le_bytes());
    kv_buf.extend_from_slice(&N_VOCAB.to_le_bytes());
    for i in 0..N_VOCAB {
        let tt: i32 = if i < 2 { 3 } else { 1 };
        kv_buf.extend_from_slice(&tt.to_le_bytes());
    }
    kv_count += 1;
    kv_u32(&mut kv_buf, "tokenizer.ggml.bos_token_id", 0);
    kv_count += 1;
    kv_u32(&mut kv_buf, "tokenizer.ggml.eos_token_id", 1);
    kv_count += 1;
    kv_bool(&mut kv_buf, "tokenizer.ggml.add_bos_token", true);
    kv_count += 1;

    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&kv_count.to_le_bytes());
    header.extend_from_slice(&kv_buf);
    for t in &tensors {
        write_str(&mut header, &t.name);
        header.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for &d in &t.dims {
            header.extend_from_slice(&d.to_le_bytes());
        }
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&t.rel_offset.to_le_bytes());
    }
    let aligned_header = (header.len() + ALIGN - 1) & !(ALIGN - 1);
    header.resize(aligned_header, 0);

    let path =
        std::env::temp_dir().join(format!("layerstream-ffi-{tag}-{}.gguf", std::process::id()));
    let mut f = std::fs::File::create(&path).expect("create synthetic gguf");
    f.write_all(&header).unwrap();
    let mut written = 0usize;
    for t in &tensors {
        let pad = (t.rel_offset as usize).saturating_sub(written);
        if pad > 0 {
            f.write_all(&vec![0u8; pad]).unwrap();
            written += pad;
        }
        f.write_all(&t.bytes).unwrap();
        written += t.bytes.len();
    }
    f.flush().unwrap();
    TempPath(path)
}

#[test]
fn standalone_c_abi_helpers_and_null_guards() {
    let ver = unsafe { CStr::from_ptr(layerstream_version()) }
        .to_str()
        .expect("utf8 version");
    assert_eq!(ver, env!("CARGO_PKG_VERSION"));

    // NULL-safe frees and cancel
    unsafe {
        layerstream_string_free(ptr::null_mut());
        layerstream_cancel_token_cancel(ptr::null());
        layerstream_cancel_token_free(ptr::null_mut());
        layerstream_engine_free(ptr::null_mut());
    }

    // GPU probe (CPU-only returns valid JSON)
    let gpu_json: JsonValue =
        serde_json::from_str(&take_c_string(layerstream_probe_gpu_json(false))).unwrap();
    assert_eq!(gpu_json["available"], false);

    // GBNF helpers
    let obj_gbnf = take_c_string(layerstream_json_object_gbnf());
    assert!(obj_gbnf.starts_with("root ::= ws "));
    let schema =
        CString::new(r#"{"type":"object","properties":{"ok":{"type":"boolean"}}}"#).unwrap();
    let gbnf = take_c_string(unsafe { layerstream_json_schema_to_gbnf(schema.as_ptr()) });
    assert!(gbnf.starts_with("root ::= ws "));

    // Assistant output parsing
    let raw = CString::new(
        r#"<think>plan</think>Hi<tool_call>{"name":"ping","arguments":{"x":1}}</tool_call>"#,
    )
    .unwrap();
    let parsed: JsonValue = serde_json::from_str(&take_c_string(unsafe {
        layerstream_parse_assistant_output_json(raw.as_ptr())
    }))
    .unwrap();
    assert_eq!(parsed["content"], "Hi");
    assert_eq!(parsed["reasoning_content"], "plan");
    assert_eq!(parsed["tool_calls"][0]["function"]["name"], "ping");

    // Error handling & clear_error
    assert!(unsafe { layerstream_inspect_gguf_json(ptr::null()) }.is_null());
    assert!(last_error_string().unwrap().contains("must not be NULL"));
    layerstream_clear_error();
    assert!(last_error_string().is_none());

    assert!(unsafe { layerstream_engine_stats_json(ptr::null()) }.is_null());
    assert!(
        unsafe {
            layerstream_engine_generate(ptr::null_mut(), ptr::null(), ptr::null(), ptr::null())
        }
        .is_null()
    );
    assert!(
        unsafe {
            layerstream_engine_generate_stream(
                ptr::null_mut(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                None,
                ptr::null_mut(),
            )
        }
        .is_null()
    );
    assert!(
        unsafe { layerstream_engine_chat(ptr::null_mut(), ptr::null(), ptr::null(), ptr::null()) }
            .is_null()
    );
    assert!(
        unsafe {
            layerstream_engine_chat_stream(
                ptr::null_mut(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                None,
                ptr::null_mut(),
            )
        }
        .is_null()
    );
    assert_eq!(unsafe { layerstream_engine_kv_clear(ptr::null_mut()) }, -1);
    assert_eq!(
        unsafe { layerstream_engine_kv_trim(ptr::null_mut(), 0) },
        -1
    );
    assert_eq!(
        unsafe { layerstream_engine_save_state(ptr::null_mut(), ptr::null()) },
        -1
    );
    assert_eq!(
        unsafe { layerstream_engine_load_state(ptr::null_mut(), ptr::null()) },
        -1
    );
}

#[test]
fn engine_end_to_end_c_abi_lifecycle() {
    let gguf = write_synthetic_llama_gguf("lifecycle");
    let model_c = CString::new(gguf.0.to_str().unwrap()).unwrap();

    // Inspect GGUF
    let info: JsonValue = serde_json::from_str(&take_c_string(unsafe {
        layerstream_inspect_gguf_json(model_c.as_ptr())
    }))
    .unwrap();
    assert_eq!(info["architecture"], "llama");
    assert_eq!(info["n_layers"], 2);

    // Open engine (CPU-only, tiny ctx)
    let open_opts =
        CString::new(r#"{"ctx_size": 128, "use_gpu": false, "prefer_direct_io": false}"#).unwrap();
    let engine = unsafe { layerstream_engine_open(model_c.as_ptr(), open_opts.as_ptr()) };
    assert!(!engine.is_null(), "open failed: {:?}", last_error_string());

    // Stats
    let stats: JsonValue = serde_json::from_str(&take_c_string(unsafe {
        layerstream_engine_stats_json(engine)
    }))
    .unwrap();
    assert_eq!(stats["loaded"], true);
    assert_eq!(stats["n_layers"], 2);

    // Generate (non-streaming)
    let prompt = CString::new("abc").unwrap();
    let gen_opts = CString::new(r#"{"max_tokens": 4, "temperature": 0.0}"#).unwrap();
    let gen_out: JsonValue = serde_json::from_str(&take_c_string(unsafe {
        layerstream_engine_generate(engine, prompt.as_ptr(), gen_opts.as_ptr(), ptr::null())
    }))
    .unwrap();
    assert!(gen_out["summary"]["completion_tokens"].as_u64().unwrap() >= 1);

    // Generate stream with callback returning false after 1 token
    extern "C" fn stop_after_one(
        _chunk: *const c_char,
        _len: usize,
        user_data: *mut c_void,
    ) -> bool {
        let count = unsafe { &mut *(user_data as *mut usize) };
        *count += 1;
        false
    }
    let mut seen: usize = 0;
    let stream_sum: JsonValue = serde_json::from_str(&take_c_string(unsafe {
        layerstream_engine_generate_stream(
            engine,
            prompt.as_ptr(),
            gen_opts.as_ptr(),
            ptr::null(),
            Some(stop_after_one),
            (&mut seen as *mut usize).cast::<c_void>(),
        )
    }))
    .unwrap();
    assert_eq!(seen, 1);
    assert!(stream_sum["completion_tokens"].as_u64().unwrap() >= 1);

    // Pre-cancelled CancelToken returns finish_reason == "cancelled"
    let tok = layerstream_cancel_token_new();
    unsafe { layerstream_cancel_token_cancel(tok) };
    let cancelled: JsonValue = serde_json::from_str(&take_c_string(unsafe {
        layerstream_engine_generate(engine, prompt.as_ptr(), gen_opts.as_ptr(), tok)
    }))
    .unwrap();
    assert_eq!(cancelled["summary"]["finish_reason"], "cancelled");
    unsafe { layerstream_cancel_token_free(tok) };

    // Chat (non-streaming with parse_output: true) & chat_stream
    let msgs = CString::new(r#"[{"role":"user","content":"abc"}]"#).unwrap();
    let chat_opts = CString::new(r#"{"max_tokens": 3, "parse_output": true}"#).unwrap();
    let chat_out: JsonValue = serde_json::from_str(&take_c_string(unsafe {
        layerstream_engine_chat(engine, msgs.as_ptr(), chat_opts.as_ptr(), ptr::null())
    }))
    .unwrap();
    assert!(chat_out["parsed"].is_object());

    let chat_stream_sum: JsonValue = serde_json::from_str(&take_c_string(unsafe {
        layerstream_engine_chat_stream(
            engine,
            msgs.as_ptr(),
            gen_opts.as_ptr(),
            ptr::null(),
            None,
            ptr::null_mut(),
        )
    }))
    .unwrap();
    assert!(chat_stream_sum["prompt_tokens"].as_u64().unwrap() >= 1);

    // KV trim, save_state, load_state, kv_clear
    let state_path = TempPath(
        std::env::temp_dir().join(format!("layerstream-ffi-state-{}.bin", std::process::id())),
    );
    let state_c = CString::new(state_path.0.to_str().unwrap()).unwrap();
    let saved = unsafe { layerstream_engine_save_state(engine, state_c.as_ptr()) };
    assert!(saved > 0, "save_state failed: {:?}", last_error_string());
    assert_eq!(unsafe { layerstream_engine_kv_clear(engine) }, 0);
    let loaded = unsafe { layerstream_engine_load_state(engine, state_c.as_ptr()) };
    assert!(loaded > 0, "load_state failed: {:?}", last_error_string());
    assert!(unsafe { layerstream_engine_kv_trim(engine, 1) } >= 0);
    assert_eq!(unsafe { layerstream_engine_kv_trim(engine, -1) }, -1);

    unsafe { layerstream_engine_free(engine) };
}

#[test]
fn c11_header_compiles_cleanly_with_werror() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include_dir = manifest_dir.join("include");
    let smoke_c = manifest_dir.join("examples").join("smoke.c");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let out = Command::new(&cc)
        .args(["-fsyntax-only", "-std=c11", "-Wall", "-Wextra", "-Werror"])
        .arg(format!("-I{}", include_dir.display()))
        .arg(&smoke_c)
        .output();
    match out {
        Ok(res) => assert!(
            res.status.success(),
            "C11 header check failed:\n{}",
            String::from_utf8_lossy(&res.stderr)
        ),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            // Skip syntax check only if no C compiler is installed on PATH.
        }
        Err(err) => panic!("failed to invoke {cc}: {err}"),
    }
}
