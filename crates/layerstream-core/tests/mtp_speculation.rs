#![allow(clippy::unwrap_used)]

//! End-to-end test for built-in MTP (Multi-Token Prediction) self-speculation on a tiny
//! synthetic `glm4moe` GGUF (`2` trunk layers + `1` NextN/MTP head = `3` total blocks).
//! Verifies exact greedy parity between non-speculative reference mode and streaming mode
//! using only the model's embedded MTP head (`enable_ngram_speculation = false`, no draft GGUF).

use std::io::Write;
use std::ptr;

use layerstream_core::ffi::{ls_engine_mtp_accept, ls_engine_mtp_draft};
use layerstream_core::{EngineConfig, GenerateOptions, StreamEngine};
use tempfile::NamedTempFile;

fn write_synthetic_glm4moe_mtp_gguf() -> NamedTempFile {
    const N_VOCAB: u64 = 64;
    const N_EMBD: u64 = 64;
    const N_HEAD: u32 = 2;
    const N_HEAD_KV: u32 = 2;
    const N_FF: u64 = 128;
    const N_FF_EXP: u64 = 64;
    const N_TRUNK_LAYERS: u32 = 2;
    const N_NEXTN_LAYERS: u32 = 1;
    const N_ALL_LAYERS: u32 = N_TRUNK_LAYERS + N_NEXTN_LAYERS;
    const N_DENSE_LEAD: u32 = 1;
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

    let mut seed: u64 = 0x4D54_505F_5345_4544;
    let mut next_f32 = |scale: f32| -> f32 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let u = ((seed >> 40) as f32) / ((1u32 << 24) as f32);
        (u * 2.0 - 1.0) * scale
    };

    struct TSpec {
        name: String,
        dims: Vec<u64>,
        bytes: Vec<u8>,
        rel_offset: u64,
    }
    let mut tensors: Vec<TSpec> = Vec::new();
    let mut push_t = |name: String, dims: Vec<u64>, is_norm: bool| {
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
        tensors.push(TSpec {
            name,
            dims,
            bytes,
            rel_offset: 0,
        });
    };

    push_t("token_embd.weight".into(), vec![N_EMBD, N_VOCAB], false);
    push_t("output_norm.weight".into(), vec![N_EMBD], true);
    push_t("output.weight".into(), vec![N_EMBD, N_VOCAB], false);

    for il in 0..N_ALL_LAYERS {
        push_t(format!("blk.{il}.attn_norm.weight"), vec![N_EMBD], true);
        push_t(
            format!("blk.{il}.attn_q.weight"),
            vec![N_EMBD, N_EMBD],
            false,
        );
        push_t(
            format!("blk.{il}.attn_k.weight"),
            vec![N_EMBD, N_EMBD],
            false,
        );
        push_t(
            format!("blk.{il}.attn_v.weight"),
            vec![N_EMBD, N_EMBD],
            false,
        );
        push_t(
            format!("blk.{il}.attn_output.weight"),
            vec![N_EMBD, N_EMBD],
            false,
        );
        push_t(
            format!("blk.{il}.post_attention_norm.weight"),
            vec![N_EMBD],
            true,
        );

        if il < N_DENSE_LEAD {
            push_t(
                format!("blk.{il}.ffn_gate.weight"),
                vec![N_EMBD, N_FF],
                false,
            );
            push_t(format!("blk.{il}.ffn_up.weight"), vec![N_EMBD, N_FF], false);
            push_t(
                format!("blk.{il}.ffn_down.weight"),
                vec![N_FF, N_EMBD],
                false,
            );
        } else {
            push_t(
                format!("blk.{il}.ffn_gate_inp.weight"),
                vec![N_EMBD, N_EXPERTS],
                false,
            );
            push_t(format!("blk.{il}.exp_probs_b.bias"), vec![N_EXPERTS], false);
            push_t(
                format!("blk.{il}.ffn_gate_exps.weight"),
                vec![N_EMBD, N_FF_EXP, N_EXPERTS],
                false,
            );
            push_t(
                format!("blk.{il}.ffn_up_exps.weight"),
                vec![N_EMBD, N_FF_EXP, N_EXPERTS],
                false,
            );
            push_t(
                format!("blk.{il}.ffn_down_exps.weight"),
                vec![N_FF_EXP, N_EMBD, N_EXPERTS],
                false,
            );
        }

        if il >= N_TRUNK_LAYERS {
            push_t(
                format!("blk.{il}.nextn.eh_proj.weight"),
                vec![2 * N_EMBD, N_EMBD],
                false,
            );
            push_t(format!("blk.{il}.nextn.enorm.weight"), vec![N_EMBD], true);
            push_t(format!("blk.{il}.nextn.hnorm.weight"), vec![N_EMBD], true);
        }
    }

    let mut cur_off = 0usize;
    for t in &mut tensors {
        cur_off = (cur_off + ALIGN - 1) & !(ALIGN - 1);
        t.rel_offset = cur_off as u64;
        cur_off += t.bytes.len();
    }

    let mut kv = Vec::new();
    let mut kv_n = 0u64;
    let arch = "glm4moe";
    kv_str(&mut kv, "general.architecture", arch);
    kv_n += 1;
    kv_str(&mut kv, "general.name", "synthetic-glm4moe-mtp");
    kv_n += 1;
    kv_u32(&mut kv, &format!("{arch}.block_count"), N_ALL_LAYERS);
    kv_n += 1;
    kv_u32(
        &mut kv,
        &format!("{arch}.nextn_predict_layers"),
        N_NEXTN_LAYERS,
    );
    kv_n += 1;
    kv_u32(&mut kv, &format!("{arch}.context_length"), 256);
    kv_n += 1;
    kv_u32(&mut kv, &format!("{arch}.embedding_length"), N_EMBD as u32);
    kv_n += 1;
    kv_u32(&mut kv, &format!("{arch}.feed_forward_length"), N_FF as u32);
    kv_n += 1;
    kv_u32(&mut kv, &format!("{arch}.attention.head_count"), N_HEAD);
    kv_n += 1;
    kv_u32(
        &mut kv,
        &format!("{arch}.attention.head_count_kv"),
        N_HEAD_KV,
    );
    kv_n += 1;
    kv_f32(
        &mut kv,
        &format!("{arch}.attention.layer_norm_rms_epsilon"),
        1e-5,
    );
    kv_n += 1;
    kv_f32(&mut kv, &format!("{arch}.rope.freq_base"), 10_000.0);
    kv_n += 1;
    kv_u32(&mut kv, &format!("{arch}.expert_count"), N_EXPERTS as u32);
    kv_n += 1;
    kv_u32(
        &mut kv,
        &format!("{arch}.expert_used_count"),
        N_EXPERTS_USED,
    );
    kv_n += 1;
    kv_u32(
        &mut kv,
        &format!("{arch}.expert_feed_forward_length"),
        N_FF_EXP as u32,
    );
    kv_n += 1;
    kv_u32(&mut kv, &format!("{arch}.expert_shared_count"), 0);
    kv_n += 1;
    kv_u32(
        &mut kv,
        &format!("{arch}.leading_dense_block_count"),
        N_DENSE_LEAD,
    );
    kv_n += 1;

    kv_str(&mut kv, "tokenizer.ggml.model", "gpt2");
    kv_n += 1;
    kv_str(&mut kv, "tokenizer.ggml.pre", "default");
    kv_n += 1;
    write_str(&mut kv, "tokenizer.ggml.merges");
    kv.extend_from_slice(&9u32.to_le_bytes());
    kv.extend_from_slice(&8u32.to_le_bytes());
    kv.extend_from_slice(&0u64.to_le_bytes());
    kv_n += 1;
    write_str(&mut kv, "tokenizer.ggml.tokens");
    kv.extend_from_slice(&9u32.to_le_bytes());
    kv.extend_from_slice(&8u32.to_le_bytes());
    kv.extend_from_slice(&N_VOCAB.to_le_bytes());
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
        write_str(&mut kv, &tok);
    }
    kv_n += 1;
    write_str(&mut kv, "tokenizer.ggml.token_type");
    kv.extend_from_slice(&9u32.to_le_bytes());
    kv.extend_from_slice(&5u32.to_le_bytes());
    kv.extend_from_slice(&N_VOCAB.to_le_bytes());
    for i in 0..N_VOCAB {
        let tt: i32 = if i < 2 { 3 } else { 1 };
        kv.extend_from_slice(&tt.to_le_bytes());
    }
    kv_n += 1;
    kv_u32(&mut kv, "tokenizer.ggml.bos_token_id", 0);
    kv_n += 1;
    kv_u32(&mut kv, "tokenizer.ggml.eos_token_id", 1);
    kv_n += 1;
    kv_bool(&mut kv, "tokenizer.ggml.add_bos_token", true);
    kv_n += 1;

    let mut hdr = Vec::new();
    hdr.extend_from_slice(b"GGUF");
    hdr.extend_from_slice(&3u32.to_le_bytes());
    hdr.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    hdr.extend_from_slice(&kv_n.to_le_bytes());
    hdr.extend_from_slice(&kv);
    for t in &tensors {
        write_str(&mut hdr, &t.name);
        hdr.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for &d in &t.dims {
            hdr.extend_from_slice(&d.to_le_bytes());
        }
        hdr.extend_from_slice(&0u32.to_le_bytes());
        hdr.extend_from_slice(&t.rel_offset.to_le_bytes());
    }
    let aligned = (hdr.len() + ALIGN - 1) & !(ALIGN - 1);
    hdr.resize(aligned, 0);

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&hdr).unwrap();
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

#[test]
fn mtp_self_speculation_matches_reference_greedy_parity_and_null_guards() {
    // Null-pointer FFI guard checks
    let mut err_buf = [0 as std::ffi::c_char; 128];
    let mut dummy = [0i32; 4];
    // SAFETY: null pointer guard test.
    unsafe {
        assert_eq!(
            ls_engine_mtp_draft(
                ptr::null_mut(),
                0,
                0,
                4,
                dummy.as_mut_ptr(),
                err_buf.as_mut_ptr(),
                err_buf.len()
            ),
            -1
        );
        ls_engine_mtp_accept(ptr::null_mut(), 0);
    }

    let gguf = write_synthetic_glm4moe_mtp_gguf();

    // 1. Reference engine (100% resident, MTP enabled)
    let mut ref_cfg = EngineConfig::reference(gguf.path());
    ref_cfg.n_ctx = 256;
    ref_cfg.n_batch = 64;
    ref_cfg.n_threads = 2;
    ref_cfg.use_gpu = false;
    let mut ref_eng = StreamEngine::open(&ref_cfg).unwrap();
    assert_eq!(ref_eng.stats().mtp_layers, 1);
    assert!(ref_eng.stats().can_speculate);

    // Non-speculative baseline tokens via single-token eval_greedy loop
    let prompt_toks = ref_eng.tokenize("abc def", true, false).unwrap();
    let mut ref_tokens = Vec::new();
    let mut next = ref_eng.eval_greedy(&prompt_toks, 0).unwrap();
    for pos in (prompt_toks.len() as i32..).take(8) {
        if ref_eng.token_is_eog(next) {
            break;
        }
        ref_tokens.push(next);
        next = ref_eng.eval_greedy(&[next], pos).unwrap();
    }
    assert!(!ref_tokens.is_empty());

    // 2. Streaming engine (0 resident trunk layers, resident MTP head, n-gram OFF -> pure MTP drafting!)
    let mut stream_cfg = EngineConfig::streaming(gguf.path());
    stream_cfg.n_ctx = 256;
    stream_cfg.n_batch = 64;
    stream_cfg.n_threads = 2;
    stream_cfg.resident_layers = Some(0);
    stream_cfg.use_gpu = false;
    stream_cfg.prefer_direct_io = false;
    let mut stream_eng = StreamEngine::open(&stream_cfg).unwrap();
    assert_eq!(stream_eng.stats().mtp_layers, 1);
    assert_eq!(stream_eng.stats().n_layers, 2);
    assert!(stream_eng.stats().can_speculate);

    let opts = GenerateOptions {
        max_tokens: ref_tokens.len(),
        enable_ngram_speculation: false,
        max_draft_tokens: 3,
        ..GenerateOptions::default()
    };
    let mut mtp_tokens = Vec::new();
    let summary = stream_eng
        .generate_stream("abc def", &opts, None, |tok, _piece| {
            mtp_tokens.push(tok);
            true
        })
        .unwrap();

    assert_eq!(
        mtp_tokens, ref_tokens,
        "MTP self-speculation must preserve exact greedy token parity with reference"
    );
    assert!(
        summary.speculation.sweeps > 0,
        "expected MTP speculative sweeps > 0"
    );
    assert!(
        summary.speculation.drafted_tokens > 0,
        "expected MTP drafted_tokens > 0"
    );

    // Exercise kv_trim and kv_clear with active MTP context
    let _ = stream_eng.kv_trim(1);
    stream_eng.kv_clear();
}
