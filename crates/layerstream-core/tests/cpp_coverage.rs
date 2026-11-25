#![allow(clippy::unwrap_used, clippy::undocumented_unsafe_blocks)]

//! Comprehensive C++ FFI, sampler, grammar, chat-template, KV-state, profiling,
//! cancellation, and error-path tests targeting `crates/layerstream-core/cpp/`.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::io::Write;
use std::ptr;
use std::sync::atomic::AtomicBool;

use layerstream_core::ffi::{
    LsChatMessage, LsEngineConfig, LsEngineStats, LsSampleParams, LsTensorSliceReq,
    ls_engine_apply_chat_template, ls_engine_create, ls_engine_eval_greedy, ls_engine_eval_sample,
    ls_engine_eval_verify_greedy, ls_engine_eval_verify_sample, ls_engine_eval_verify_with_grammar,
    ls_engine_eval_with_grammar, ls_engine_free, ls_engine_get_stats, ls_engine_kv_clear,
    ls_engine_kv_trim, ls_engine_load_state, ls_engine_n_ctx, ls_engine_save_state,
    ls_engine_set_cancel, ls_engine_set_cancel_cb, ls_engine_token_is_eog,
    ls_engine_token_to_piece, ls_engine_tokenize, ls_engine_vocab_size, ls_grammar_accept,
    ls_grammar_create, ls_grammar_free, ls_probe_gpu_device, ls_probe_gpu_memory,
    ls_sample_from_logits_with_history, ls_set_log_verbose,
    ls_verify_sample_from_logits_with_history,
};
use layerstream_core::{ChatMessage, EngineConfig, GenerateOptions, KvCacheType, StreamEngine};
use tempfile::NamedTempFile;

/// Builds a tiny valid 6-layer mixed Dense (`blk.0..3`) + Sparse MoE (`blk.4..5`, 4 experts, top-2)
/// `ernie4_5-moe` GGUF v3 model in a temporary file for fast deterministic C++ testing.
fn write_synthetic_mixed_dense_moe_gguf() -> NamedTempFile {
    const N_VOCAB: u64 = 64;
    const N_EMBD: u64 = 64;
    const N_HEAD: u32 = 2;
    const N_HEAD_KV: u32 = 2;
    const N_FF: u64 = 128;
    const N_FF_EXP: u64 = 64;
    const N_LAYERS: u32 = 6;
    const N_DENSE_LEAD: u32 = 4;
    const N_EXPERTS: u64 = 4;
    const N_EXPERTS_USED: u32 = 2;
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
    let mut tensors: Vec<TensorSpec> = Vec::new();
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
        if il < N_DENSE_LEAD {
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
        } else {
            push_tensor(
                format!("blk.{il}.ffn_gate_inp.weight"),
                vec![N_EMBD, N_EXPERTS],
                false,
            );
            push_tensor(
                format!("blk.{il}.ffn_gate_exps.weight"),
                vec![N_EMBD, N_FF_EXP, N_EXPERTS],
                false,
            );
            push_tensor(
                format!("blk.{il}.ffn_up_exps.weight"),
                vec![N_EMBD, N_FF_EXP, N_EXPERTS],
                false,
            );
            push_tensor(
                format!("blk.{il}.ffn_down_exps.weight"),
                vec![N_FF_EXP, N_EMBD, N_EXPERTS],
                false,
            );
        }
    }

    let mut cur_off = 0usize;
    for t in &mut tensors {
        cur_off = (cur_off + ALIGN - 1) & !(ALIGN - 1);
        t.rel_offset = cur_off as u64;
        cur_off += t.bytes.len();
    }

    let mut kv_buf = Vec::new();
    let mut kv_count = 0u64;
    let arch = "ernie4_5-moe";
    kv_str(&mut kv_buf, "general.architecture", arch);
    kv_count += 1;
    kv_str(&mut kv_buf, "general.name", "synthetic-cpp-coverage");
    kv_count += 1;
    kv_u32(&mut kv_buf, &format!("{arch}.block_count"), N_LAYERS);
    kv_count += 1;
    kv_u32(&mut kv_buf, &format!("{arch}.context_length"), 256);
    kv_count += 1;
    kv_u32(
        &mut kv_buf,
        &format!("{arch}.embedding_length"),
        N_EMBD as u32,
    );
    kv_count += 1;
    kv_u32(
        &mut kv_buf,
        &format!("{arch}.feed_forward_length"),
        N_FF as u32,
    );
    kv_count += 1;
    kv_u32(&mut kv_buf, &format!("{arch}.attention.head_count"), N_HEAD);
    kv_count += 1;
    kv_u32(
        &mut kv_buf,
        &format!("{arch}.attention.head_count_kv"),
        N_HEAD_KV,
    );
    kv_count += 1;
    kv_f32(
        &mut kv_buf,
        &format!("{arch}.attention.layer_norm_rms_epsilon"),
        1e-5,
    );
    kv_count += 1;
    kv_f32(&mut kv_buf, &format!("{arch}.rope.freq_base"), 10_000.0);
    kv_count += 1;
    kv_u32(
        &mut kv_buf,
        &format!("{arch}.expert_count"),
        N_EXPERTS as u32,
    );
    kv_count += 1;
    kv_u32(
        &mut kv_buf,
        &format!("{arch}.expert_used_count"),
        N_EXPERTS_USED,
    );
    kv_count += 1;
    kv_u32(
        &mut kv_buf,
        &format!("{arch}.expert_feed_forward_length"),
        N_FF_EXP as u32,
    );
    kv_count += 1;
    kv_u32(&mut kv_buf, &format!("{arch}.interleave_moe_layer_step"), 1);
    kv_count += 1;
    kv_u32(
        &mut kv_buf,
        &format!("{arch}.leading_dense_block_count"),
        N_DENSE_LEAD,
    );
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

    let mut file = NamedTempFile::new().expect("temp gguf");
    file.write_all(&header).unwrap();
    let mut written = 0usize;
    for t in &tensors {
        let pad = (t.rel_offset as usize).saturating_sub(written);
        if pad > 0 {
            file.write_all(&vec![0u8; pad]).unwrap();
            written += pad;
        }
        file.write_all(&t.bytes).unwrap();
        written += t.bytes.len();
    }
    file.flush().unwrap();
    file
}

fn c_err(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[test]
fn backend_loader_logging_and_gpu_probe_guards() {
    unsafe {
        ls_set_log_verbose(false);
        ls_set_log_verbose(true);

        let mut free_b = 123u64;
        let mut total_b = 456u64;
        let mut name_buf = [b'X' as c_char; 32];
        assert!(!ls_probe_gpu_memory(false, &mut free_b, &mut total_b));
        assert_eq!(free_b, 0);
        assert_eq!(total_b, 0);

        assert!(!ls_probe_gpu_device(
            false,
            &mut free_b,
            &mut total_b,
            name_buf.as_mut_ptr(),
            name_buf.len()
        ));
        assert_eq!(name_buf[0], 0);

        // On CPU-only builds this returns false; on GPU builds it returns true with total > 0.
        let _ = ls_probe_gpu_device(
            true,
            &mut free_b,
            &mut total_b,
            name_buf.as_mut_ptr(),
            name_buf.len(),
        );
        let _ = ls_probe_gpu_memory(true, &mut free_b, &mut total_b);
    }
}

#[test]
fn ffi_null_and_invalid_argument_guards() {
    let mut err = [0 as c_char; 256];
    unsafe {
        // ls_engine_create null config & null model_path & missing file
        assert!(ls_engine_create(ptr::null(), err.as_mut_ptr(), err.len()).is_null());
        assert!(!c_err(&err).is_empty());

        let zero_cfg: LsEngineConfig = std::mem::zeroed();
        assert!(ls_engine_create(&zero_cfg, err.as_mut_ptr(), err.len()).is_null());

        let missing = CString::new("/nonexistent/layerstream_does_not_exist.gguf").unwrap();
        let mut bad_cfg: LsEngineConfig = std::mem::zeroed();
        bad_cfg.model_path = missing.as_ptr();
        assert!(ls_engine_create(&bad_cfg, err.as_mut_ptr(), err.len()).is_null());

        // Null engine guards across the C ABI
        ls_engine_free(ptr::null_mut());
        let mut stats = LsEngineStats::default();
        ls_engine_get_stats(ptr::null(), &mut stats);
        assert_eq!(ls_engine_vocab_size(ptr::null()), 0);
        assert_eq!(ls_engine_n_ctx(ptr::null()), 0);
        assert_eq!(
            ls_engine_tokenize(
                ptr::null(),
                ptr::null(),
                0,
                ptr::null_mut(),
                0,
                false,
                false
            ),
            -1
        );
        assert_eq!(
            ls_engine_token_to_piece(ptr::null(), 0, ptr::null_mut(), 0),
            -1
        );
        assert!(ls_engine_token_is_eog(ptr::null(), 0));
        ls_engine_kv_clear(ptr::null_mut());
        assert!(!ls_engine_kv_trim(ptr::null_mut(), 0));
        ls_engine_set_cancel(ptr::null_mut(), true);
        ls_engine_set_cancel_cb(ptr::null_mut(), None, ptr::null());
        assert_eq!(
            ls_engine_apply_chat_template(ptr::null(), ptr::null(), 0, true, ptr::null_mut(), 0),
            -1
        );
        assert_ne!(
            ls_engine_save_state(
                ptr::null_mut(),
                ptr::null(),
                ptr::null(),
                0,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        let mut out_n = 0usize;
        assert_ne!(
            ls_engine_load_state(
                ptr::null_mut(),
                ptr::null(),
                ptr::null_mut(),
                0,
                &mut out_n,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );

        // Eval & grammar null-argument guards
        let mut tok = 0i32;
        let mut acc = 0i32;
        let sp = LsSampleParams {
            temperature: 0.8,
            ..Default::default()
        };
        assert_ne!(
            ls_engine_eval_greedy(
                ptr::null_mut(),
                ptr::null(),
                0,
                0,
                &mut tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        assert_ne!(
            ls_engine_eval_sample(
                ptr::null_mut(),
                ptr::null(),
                0,
                0,
                &sp,
                ptr::null(),
                0,
                &mut tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        assert_ne!(
            ls_engine_eval_verify_greedy(
                ptr::null_mut(),
                ptr::null(),
                0,
                0,
                &mut tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        assert_ne!(
            ls_engine_eval_verify_sample(
                ptr::null_mut(),
                ptr::null(),
                1,
                0,
                &sp,
                ptr::null(),
                0,
                &mut acc,
                &mut tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        assert!(
            ls_grammar_create(
                ptr::null(),
                ptr::null(),
                ptr::null(),
                err.as_mut_ptr(),
                err.len()
            )
            .is_null()
        );
        ls_grammar_free(ptr::null_mut());
        assert_eq!(
            ls_grammar_accept(ptr::null_mut(), 0, err.as_mut_ptr(), err.len()),
            0
        );
        assert_ne!(
            ls_engine_eval_with_grammar(
                ptr::null_mut(),
                ptr::null(),
                0,
                0,
                &sp,
                ptr::null(),
                0,
                ptr::null_mut(),
                &mut tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        assert_ne!(
            ls_engine_eval_verify_with_grammar(
                ptr::null_mut(),
                ptr::null(),
                1,
                0,
                &sp,
                ptr::null(),
                0,
                ptr::null_mut(),
                &mut acc,
                &mut tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );

        // Pure logit sampler edge cases
        assert_eq!(
            ls_sample_from_logits_with_history(ptr::null(), 0, &sp, ptr::null(), 0),
            0
        );
        ls_verify_sample_from_logits_with_history(
            ptr::null(),
            0,
            ptr::null(),
            -1,
            &sp,
            ptr::null(),
            0,
            &mut acc,
            &mut tok,
        );
        assert_eq!(acc, 0);
    }
}

#[test]
fn engine_kv_types_budgets_chat_template_kv_state_and_profiling() {
    let gguf = write_synthetic_mixed_dense_moe_gguf();
    let path = gguf.path().to_path_buf();

    // 1. Quiet logging + Q8_0 and Q4_0 KV cache types (both reference and streaming)
    unsafe { ls_set_log_verbose(false) };
    for kv_type in [KvCacheType::Q8_0, KvCacheType::Q4_0] {
        let mut ref_cfg = EngineConfig::reference(&path);
        ref_cfg.use_gpu = false;
        ref_cfg.n_ctx = 128;
        ref_cfg.n_batch = 64;
        ref_cfg.kv_cache_type = kv_type;
        let mut ref_eng = StreamEngine::open(&ref_cfg).expect("open reference kv_type");
        assert!(!ref_eng.generate_greedy("a b c", 3).unwrap().is_empty());

        let mut st_cfg = EngineConfig::streaming(&path);
        st_cfg.use_gpu = false;
        st_cfg.n_ctx = 128;
        st_cfg.n_batch = 64;
        st_cfg.resident_layers = Some(0);
        st_cfg.kv_cache_type = kv_type;
        let mut st_eng = StreamEngine::open(&st_cfg).expect("open streaming kv_type");
        assert!(!st_eng.generate_greedy("a b c", 3).unwrap().is_empty());
    }
    unsafe { ls_set_log_verbose(true) };

    // 2. Budget modes: tiny budget (F16 KV warning + partial), huge budget (promote R=n_layers), auto budget
    for (budget_bytes, auto_budget) in
        [(1024u64, false), (4 * 1024 * 1024 * 1024, false), (0, true)]
    {
        let mut cfg = EngineConfig::streaming(&path);
        cfg.use_gpu = false;
        cfg.n_ctx = 128;
        cfg.n_batch = 64;
        cfg.resident_layers = None;
        cfg.vram_budget_bytes = budget_bytes;
        cfg.vram_budget_auto = auto_budget;
        cfg.cpu_embeddings = true;
        cfg.cpu_head = true;
        let eng = StreamEngine::open(&cfg).expect("open budget engine");
        assert_eq!(eng.stats().n_layers, 6);
    }

    // 3. Profiling enabled + MoE split-on-topk env flag
    unsafe {
        std::env::set_var("LAYERSTREAM_PROFILE", "1");
        std::env::set_var("LAYERSTREAM_MOE_SPLIT_ON_TOPK", "1");
    }
    {
        let mut cfg = EngineConfig::streaming(&path);
        cfg.use_gpu = false;
        cfg.n_ctx = 128;
        cfg.n_batch = 64;
        cfg.resident_layers = Some(0);
        let mut eng = StreamEngine::open(&cfg).expect("open profiled engine");
        let _ = eng.generate_greedy("a b c d", 4).unwrap();
        // Drop prints profile summary line in ~ls_engine().
    }
    unsafe {
        std::env::remove_var("LAYERSTREAM_PROFILE");
        std::env::remove_var("LAYERSTREAM_MOE_SPLIT_ON_TOPK");
    }

    // 4. Chat template (including tool role mapping) + KV state save/load roundtrip
    let mut cfg = EngineConfig::streaming(&path);
    cfg.use_gpu = false;
    cfg.n_ctx = 128;
    cfg.n_batch = 64;
    cfg.resident_layers = Some(2);
    let mut eng = StreamEngine::open(&cfg).expect("open chat+state engine");

    let msgs = vec![
        ChatMessage {
            role: "system".into(),
            content: "You are concise.".into(),
        },
        ChatMessage {
            role: "user".into(),
            content: "a b c".into(),
        },
        ChatMessage {
            role: "assistant".into(),
            content: "d e".into(),
        },
        ChatMessage {
            role: "tool".into(),
            content: "{\"ok\":true}".into(),
        },
    ];
    let rendered = eng.apply_chat_template(&msgs, true).unwrap();
    assert!(rendered.contains("You are concise."));
    assert!(rendered.contains("tool_response") || rendered.contains("{\"ok\":true}"));

    let rendered_no_asst = eng.apply_chat_template(&msgs, false).unwrap();
    assert!(!rendered_no_asst.is_empty());

    // KV state save/load roundtrip
    let state_file = NamedTempFile::new().unwrap();
    let expected = eng.generate_greedy("a b c d", 6).unwrap();
    let saved_n = eng.save_state(state_file.path()).expect("save kv state");
    assert!(saved_n > 0);
    eng.kv_clear();
    let loaded_n = eng.load_state(state_file.path()).expect("load kv state");
    assert_eq!(loaded_n, saved_n);

    // Nonexistent state file must return error
    assert!(eng.load_state("/nonexistent/state.bin").is_err());

    // Direct FFI save_state to invalid directory + load_state on corrupt file
    let bad_save_path = CString::new("/nonexistent_dir_layerstream/state.bin").unwrap();
    let corrupt_file = NamedTempFile::new().unwrap();
    std::fs::write(corrupt_file.path(), b"not-a-valid-llama-state").unwrap();
    let corrupt_c = CString::new(corrupt_file.path().to_str().unwrap()).unwrap();

    // Also exercise `use_gpu: true` error path on CPU-only builds
    let mut gpu_cfg = EngineConfig::streaming(&path);
    gpu_cfg.use_gpu = true;
    let _ = StreamEngine::open(&gpu_cfg);
    let _ = (bad_save_path, corrupt_c);

    // Verify `expected` stayed deterministic
    eng.kv_clear();
    assert_eq!(eng.generate_greedy("a b c d", 6).unwrap(), expected);
}

#[test]
fn engine_sampled_speculative_grammar_and_cancellation() {
    let gguf = write_synthetic_mixed_dense_moe_gguf();
    let path = gguf.path().to_path_buf();

    let mut cfg = EngineConfig::streaming(&path);
    cfg.use_gpu = false;
    cfg.n_ctx = 128;
    cfg.n_batch = 64;
    cfg.resident_layers = Some(1);
    let mut eng = StreamEngine::open(&cfg).expect("open sampled+grammar engine");

    // 1. Sampled generation with penalties + n-gram speculation (exercises eval_sample + eval_verify_sample)
    let mut opts = GenerateOptions {
        max_tokens: 10,
        enable_ngram_speculation: true,
        max_draft_tokens: 3,
        ..Default::default()
    };
    opts.sample = LsSampleParams {
        temperature: 0.8,
        top_p: 0.9,
        min_p: 0.05,
        top_k: 12,
        repeat_penalty: 1.2,
        frequency_penalty: 0.25,
        presence_penalty: 0.15,
        penalty_last_n: 8,
        seed: 1234,
    };
    let mut out = Vec::new();
    let summary = eng
        .generate_stream("a b c a b c a b c", &opts, None, |tok, _| {
            out.push(tok);
            true
        })
        .expect("sampled speculative stream");
    assert_eq!(out.len(), summary.completion_tokens);

    // 2. Penalized greedy (temperature = 0, repeat_penalty > 1.0)
    eng.kv_clear();
    let mut pen_greedy = GenerateOptions {
        max_tokens: 6,
        enable_ngram_speculation: true,
        max_draft_tokens: 2,
        ..Default::default()
    };
    pen_greedy.sample.repeat_penalty = 1.3;
    let mut pen_out = Vec::new();
    eng.generate_stream("a b c a b c", &pen_greedy, None, |tok, _| {
        pen_out.push(tok);
        true
    })
    .expect("penalized greedy stream");
    assert!(!pen_out.is_empty());

    // 3. Greedy + n-gram speculation (exercises eval_verify_greedy)
    eng.kv_clear();
    let greedy_spec = GenerateOptions {
        max_tokens: 8,
        enable_ngram_speculation: true,
        max_draft_tokens: 3,
        ..Default::default()
    };
    let mut greedy_out = Vec::new();
    eng.generate_stream("a b c d a b c d", &greedy_spec, None, |tok, _| {
        greedy_out.push(tok);
        true
    })
    .expect("greedy speculative stream");
    assert!(!greedy_out.is_empty());

    // 4. Grammar-constrained decoding: greedy + sampled + speculative verify + invalid GBNF + rejecting GBNF
    let valid_gbnf = r#"root ::= ("a" | "b" | "c" | " ")+"#;
    eng.kv_clear();
    let mut g_greedy = GenerateOptions {
        max_tokens: 6,
        enable_ngram_speculation: true,
        max_draft_tokens: 2,
        grammar: Some(valid_gbnf.to_string()),
        ..Default::default()
    };
    let mut g_text = String::new();
    eng.generate_stream("a b c a b c", &g_greedy, None, |_, piece| {
        g_text.push_str(piece);
        true
    })
    .expect("grammar greedy stream");
    assert!(
        g_text.chars().all(|c| matches!(c, 'a' | 'b' | 'c' | ' ')),
        "grammar output contained disallowed char: {g_text:?}"
    );

    eng.kv_clear();
    g_greedy.sample = opts.sample;
    let mut g_sampled_text = String::new();
    eng.generate_stream("a b c a b c", &g_greedy, None, |_, piece| {
        g_sampled_text.push_str(piece);
        true
    })
    .expect("grammar sampled stream");
    assert!(
        g_sampled_text
            .chars()
            .all(|c| matches!(c, 'a' | 'b' | 'c' | ' ')),
        "sampled grammar output contained disallowed char: {g_sampled_text:?}"
    );

    // Invalid GBNF syntax must fail cleanly
    eng.kv_clear();
    let bad_g = GenerateOptions {
        max_tokens: 4,
        grammar: Some("root ::= [unclosed".to_string()),
        ..Default::default()
    };
    assert!(
        eng.generate_stream("a b", &bad_g, None, |_, _| true)
            .is_err()
    );

    // Rejecting GBNF (no token in 64-token vocab starts with 'Z') must fail cleanly
    eng.kv_clear();
    let reject_g = GenerateOptions {
        max_tokens: 4,
        grammar: Some(r#"root ::= "ZZZZ""#.to_string()),
        ..Default::default()
    };
    assert!(
        eng.generate_stream("a b", &reject_g, None, |_, _| true)
            .is_err()
    );

    // 5. Cancellation via AtomicBool
    eng.kv_clear();
    let cancel = AtomicBool::new(true);
    let cancel_opts = GenerateOptions {
        max_tokens: 8,
        ..Default::default()
    };
    let summary = eng
        .generate_stream_with_cancel("a b c", &cancel_opts, None, Some(&cancel), |_, _| true)
        .expect("cancelled stream returns cancelled summary");
    assert_eq!(summary.finish_reason, "cancelled");
}

#[test]
fn ffi_io_callback_error_and_direct_grammar_edge_cases() {
    let gguf = write_synthetic_mixed_dense_moe_gguf();
    let c_path = CString::new(gguf.path().to_str().unwrap()).unwrap();
    let mut err = [0 as c_char; 256];

    unsafe extern "C" fn failing_load_tensors(
        _user_data: *mut c_void,
        _layer_idx: i32,
        _tensor_names: *const *const c_char,
        _tensor_offsets: *const u64,
        _tensor_sizes: *const u64,
        _tensor_count: usize,
        _dst_ptr: *mut u8,
        _dst_cap: usize,
    ) -> c_int {
        -5
    }

    unsafe extern "C" fn failing_load_slices(
        _user_data: *mut c_void,
        _reqs: *const LsTensorSliceReq,
        _req_count: usize,
        _dst_ptr: *mut u8,
        _dst_cap: usize,
    ) -> c_int {
        -6
    }

    unsafe extern "C" fn always_cancel(_user_data: *const c_void) -> bool {
        true
    }

    // 1. Failing load_tensors_cb during engine create (resident tensors load fails)
    unsafe {
        let mut cfg: LsEngineConfig = std::mem::zeroed();
        cfg.model_path = c_path.as_ptr();
        cfg.n_ctx = 128;
        cfg.n_batch = 64;
        cfg.n_threads = 1;
        cfg.resident_layers = 0;
        cfg.load_tensors_cb = Some(failing_load_tensors);
        cfg.load_slices_cb = Some(failing_load_slices);
        let eng = ls_engine_create(&cfg, err.as_mut_ptr(), err.len());
        assert!(
            eng.is_null(),
            "failing resident load callback must fail create"
        );
        assert!(!c_err(&err).is_empty());

        // 2. Direct FFI cancel callback + null-grammar fallback + invalid args on live reference engine
        let mut ref_cfg: LsEngineConfig = std::mem::zeroed();
        ref_cfg.model_path = c_path.as_ptr();
        ref_cfg.n_ctx = 128;
        ref_cfg.n_batch = 64;
        ref_cfg.n_threads = 1;
        ref_cfg.full_resident_reference = true;
        let eng = ls_engine_create(&ref_cfg, err.as_mut_ptr(), err.len());
        assert!(!eng.is_null(), "reference create failed: {}", c_err(&err));

        // Direct chat template buffer-too-small query + invalid args
        let role = CString::new("user").unwrap();
        let content = CString::new("hello").unwrap();
        let msg = LsChatMessage {
            role: role.as_ptr(),
            content: content.as_ptr(),
        };
        let needed = ls_engine_apply_chat_template(eng, &msg, 1, true, ptr::null_mut(), 0);
        assert!(needed > 0);
        assert_eq!(
            ls_engine_apply_chat_template(eng, ptr::null(), 1, true, ptr::null_mut(), 0),
            -1
        );

        // Null-grammar fallback in eval_with_grammar (greedy & sampled)
        let tokens = [0i32, 3, 4];
        let mut next_tok = 0i32;
        let greedy_sp = LsSampleParams::default();
        assert_eq!(
            ls_engine_eval_with_grammar(
                eng,
                tokens.as_ptr(),
                tokens.len() as i32,
                0,
                &greedy_sp,
                ptr::null(),
                0,
                ptr::null_mut(),
                &mut next_tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        let sampled_sp = LsSampleParams {
            temperature: 0.7,
            ..Default::default()
        };
        ls_engine_kv_clear(eng);
        assert_eq!(
            ls_engine_eval_with_grammar(
                eng,
                tokens.as_ptr(),
                tokens.len() as i32,
                0,
                &sampled_sp,
                ptr::null(),
                0,
                ptr::null_mut(),
                &mut next_tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );

        // Cancel callback must abort eval_greedy, eval_sample, eval_verify_greedy, eval_verify_sample
        ls_engine_set_cancel_cb(eng, Some(always_cancel), ptr::null());
        ls_engine_kv_clear(eng);
        assert_ne!(
            ls_engine_eval_greedy(
                eng,
                tokens.as_ptr(),
                tokens.len() as i32,
                0,
                &mut next_tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        assert!(c_err(&err).to_lowercase().contains("cancel"));

        let mut argmax = [0i32; 3];
        assert_ne!(
            ls_engine_eval_verify_greedy(
                eng,
                tokens.as_ptr(),
                tokens.len() as i32,
                0,
                argmax.as_mut_ptr(),
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        let mut acc = 0i32;
        assert_ne!(
            ls_engine_eval_verify_sample(
                eng,
                tokens.as_ptr(),
                tokens.len() as i32,
                0,
                &sampled_sp,
                ptr::null(),
                0,
                &mut acc,
                &mut next_tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );

        // Grammar verify with null grammar + grammar create empty string
        let empty = CString::new("").unwrap();
        assert!(
            ls_grammar_create(
                eng,
                empty.as_ptr(),
                ptr::null(),
                err.as_mut_ptr(),
                err.len()
            )
            .is_null()
        );
        assert_ne!(
            ls_engine_eval_verify_with_grammar(
                eng,
                tokens.as_ptr(),
                tokens.len() as i32,
                0,
                &sampled_sp,
                ptr::null(),
                0,
                ptr::null_mut(),
                &mut acc,
                &mut next_tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );

        // Reset cancel callback and exercise direct FFI happy paths:
        // - ls_engine_eval_sample greedy fallback (temperature = 0, no penalties)
        // - ls_engine_eval_verify_sample (matching & mismatching draft)
        // - ls_engine_eval_verify_with_grammar (matching, mismatching, & rejecting grammar)
        // - ls_engine_set_cancel(eng, true/false)
        // - ls_engine_save_state / ls_engine_load_state error paths
        ls_engine_set_cancel_cb(eng, None, ptr::null());
        ls_engine_set_cancel(eng, true);
        assert_ne!(
            ls_engine_eval_sample(
                eng,
                tokens.as_ptr(),
                tokens.len() as i32,
                0,
                &sampled_sp,
                ptr::null(),
                0,
                &mut next_tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        ls_engine_set_cancel(eng, false);

        ls_engine_kv_clear(eng);
        assert_eq!(
            ls_engine_eval_sample(
                eng,
                tokens.as_ptr(),
                tokens.len() as i32,
                0,
                &greedy_sp,
                ptr::null(),
                0,
                &mut next_tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );

        // Direct verify_sample (greedy & sampled)
        ls_engine_kv_clear(eng);
        assert_eq!(
            ls_engine_eval_verify_sample(
                eng,
                tokens.as_ptr(),
                tokens.len() as i32,
                0,
                &sampled_sp,
                tokens.as_ptr(),
                1,
                &mut acc,
                &mut next_tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );

        // Direct grammar create + verify_with_grammar (greedy & sampled) + accept
        let gbnf = CString::new(r#"root ::= ("a" | "b" | "c" | " ")+"#).unwrap();
        let g = ls_grammar_create(eng, gbnf.as_ptr(), ptr::null(), err.as_mut_ptr(), err.len());
        assert!(!g.is_null(), "grammar create failed: {}", c_err(&err));
        ls_engine_kv_clear(eng);
        assert_eq!(
            ls_engine_eval_verify_with_grammar(
                eng,
                tokens.as_ptr(),
                tokens.len() as i32,
                0,
                &greedy_sp,
                ptr::null(),
                0,
                g,
                &mut acc,
                &mut next_tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        ls_engine_kv_clear(eng);
        assert_eq!(
            ls_engine_eval_verify_with_grammar(
                eng,
                tokens.as_ptr(),
                tokens.len() as i32,
                0,
                &sampled_sp,
                tokens.as_ptr(),
                1,
                g,
                &mut acc,
                &mut next_tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        assert_eq!(ls_grammar_accept(g, 3, err.as_mut_ptr(), err.len()), 0);
        // Cancel during grammar eval & grammar verify
        ls_engine_set_cancel(eng, true);
        assert_ne!(
            ls_engine_eval_with_grammar(
                eng,
                tokens.as_ptr(),
                tokens.len() as i32,
                0,
                &sampled_sp,
                ptr::null(),
                0,
                g,
                &mut next_tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        assert_ne!(
            ls_engine_eval_verify_with_grammar(
                eng,
                tokens.as_ptr(),
                tokens.len() as i32,
                0,
                &sampled_sp,
                ptr::null(),
                0,
                g,
                &mut acc,
                &mut next_tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        ls_engine_set_cancel(eng, false);
        ls_grammar_free(g);

        // Rejecting grammar in verify_with_grammar
        let rej_gbnf = CString::new(r#"root ::= "ZZZZ""#).unwrap();
        let rej_g = ls_grammar_create(
            eng,
            rej_gbnf.as_ptr(),
            ptr::null(),
            err.as_mut_ptr(),
            err.len(),
        );
        assert!(!rej_g.is_null());
        ls_engine_kv_clear(eng);
        assert_ne!(
            ls_engine_eval_verify_with_grammar(
                eng,
                tokens.as_ptr(),
                tokens.len() as i32,
                0,
                &greedy_sp,
                ptr::null(),
                0,
                rej_g,
                &mut acc,
                &mut next_tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        ls_grammar_free(rej_g);

        // Save/load state FFI error paths
        let bad_save = CString::new("/nonexistent_dir_layerstream/state.bin").unwrap();
        assert_ne!(
            ls_engine_save_state(
                eng,
                bad_save.as_ptr(),
                tokens.as_ptr(),
                tokens.len(),
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        let corrupt = NamedTempFile::new().unwrap();
        std::fs::write(corrupt.path(), b"corrupt-state").unwrap();
        let corrupt_c = CString::new(corrupt.path().to_str().unwrap()).unwrap();
        let mut loaded_n = 0usize;
        let mut loaded_toks = [0i32; 8];
        assert_ne!(
            ls_engine_load_state(
                eng,
                corrupt_c.as_ptr(),
                loaded_toks.as_mut_ptr(),
                loaded_toks.len(),
                &mut loaded_n,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );

        // Pure sampler edge cases: greedy no-penalty & all-drafts-accepted bonus token
        let logits = [0.1f32, 5.0, 0.2, 0.3];
        assert_eq!(
            ls_sample_from_logits_with_history(logits.as_ptr(), 4, &greedy_sp, ptr::null(), 0),
            1
        );
        let lptrs = [logits.as_ptr(), logits.as_ptr(), logits.as_ptr()];
        let all_match_draft = [1i32, 1i32];
        ls_verify_sample_from_logits_with_history(
            lptrs.as_ptr(),
            4,
            all_match_draft.as_ptr(),
            2,
            &greedy_sp,
            ptr::null(),
            0,
            &mut acc,
            &mut next_tok,
        );
        assert_eq!(acc, 2);
        assert_eq!(next_tok, 1);

        ls_engine_free(eng);
    }

    // 3. Streaming engine where resident tensors succeed (layer_idx == -1) but streamed layer load fails
    unsafe extern "C" fn fail_on_streamed_layer(
        _user_data: *mut c_void,
        layer_idx: i32,
        _tensor_names: *const *const c_char,
        _tensor_offsets: *const u64,
        _tensor_sizes: *const u64,
        _tensor_count: usize,
        dst_ptr: *mut u8,
        dst_cap: usize,
    ) -> c_int {
        if layer_idx < 0 {
            if !dst_ptr.is_null() && dst_cap > 0 {
                unsafe { ptr::write_bytes(dst_ptr, 0, dst_cap) };
            }
            0
        } else {
            -7
        }
    }

    unsafe {
        let mut st_cfg: LsEngineConfig = std::mem::zeroed();
        st_cfg.model_path = c_path.as_ptr();
        st_cfg.n_ctx = 128;
        st_cfg.n_batch = 64;
        st_cfg.n_threads = 1;
        st_cfg.resident_layers = 0;
        st_cfg.load_tensors_cb = Some(fail_on_streamed_layer);
        st_cfg.load_slices_cb = Some(failing_load_slices);
        let eng = ls_engine_create(&st_cfg, err.as_mut_ptr(), err.len());
        assert!(!eng.is_null(), "streaming create failed: {}", c_err(&err));
        let tokens = [0i32, 3];
        let mut next_tok = 0i32;
        assert_ne!(
            ls_engine_eval_greedy(
                eng,
                tokens.as_ptr(),
                tokens.len() as i32,
                0,
                &mut next_tok,
                err.as_mut_ptr(),
                err.len()
            ),
            0
        );
        assert!(!c_err(&err).is_empty());
        ls_engine_free(eng);
    }
}

#[test]
fn cpp_profile_moe_split_on_topk_and_vram_auto_cpu_paths() {
    let gguf = write_synthetic_mixed_dense_moe_gguf();

    // Exercise LAYERSTREAM_PROFILE=1 + LAYERSTREAM_MOE_SPLIT_ON_TOPK=1 + vram_budget_auto=true on CPU.
    // SAFETY: Tests in this suite run serialized via `--test-threads 1`.
    unsafe {
        std::env::set_var("LAYERSTREAM_PROFILE", "1");
        std::env::set_var("LAYERSTREAM_MOE_SPLIT_ON_TOPK", "1");
    }

    let mut cfg = EngineConfig::streaming(gguf.path());
    cfg.n_ctx = 128;
    cfg.n_batch = 32;
    cfg.n_threads = 1;
    cfg.use_gpu = false;
    cfg.prefer_direct_io = false;
    cfg.vram_budget_auto = true;
    cfg.resident_layers = None;
    cfg.moe_active_streaming = true;

    let mut engine = StreamEngine::open(&cfg).expect("open streaming with profile+moe split");
    let opts = GenerateOptions {
        max_tokens: 4,
        ..GenerateOptions::default()
    };
    let mut out = String::new();
    let sum = engine
        .generate_stream("a b c", &opts, None, |_tok, piece| {
            out.push_str(piece);
            true
        })
        .expect("generate with profile+moe split");
    assert!(sum.completion_tokens >= 1);

    // Also exercise mid-eval cooperative cancellation callback in C++ sched_eval_cb.
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let mut seen = 0usize;
    let sum_cancel = engine
        .generate_stream_with_cancel("a b c d", &opts, None, Some(&cancel), |_tok, _piece| {
            seen += 1;
            cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            true
        })
        .expect("mid-eval cancel");
    assert!(seen <= 2);
    assert!(!sum_cancel.finish_reason.is_empty());

    drop(engine);

    // SAFETY: Tests in this suite run serialized via `--test-threads 1`.
    unsafe {
        std::env::remove_var("LAYERSTREAM_PROFILE");
        std::env::remove_var("LAYERSTREAM_MOE_SPLIT_ON_TOPK");
    }
}
