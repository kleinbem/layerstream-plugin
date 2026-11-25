#![allow(clippy::unwrap_used)]

//! End-to-end Rust coverage tests for `layerstream_core::speculator` and `layerstream_core::sdk`.

use std::io::Write;
use std::sync::atomic::AtomicBool;

use layerstream_core::EngineConfig;
#[cfg(feature = "hf")]
use layerstream_core::sdk::{
    CheckOptions, LoadedSession, OpenSessionOptions, RecommendOptions, check_model,
    inspect_gguf_file, recommend_repo,
};
use layerstream_core::speculator::{DraftModelProposer, NgramProposer};
#[cfg(feature = "hf")]
use layerstream_core::{GenerateOptions, KvCacheType};
use tempfile::NamedTempFile;

fn write_tiny_llama_gguf(n_vocab: u64, alt_vocab: bool) -> NamedTempFile {
    const N_EMBD: u64 = 64;
    const N_HEAD: u32 = 2;
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

    let mut seed: u64 = 0xCAFE_BABE_1234_5678;
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

    push_t("token_embd.weight".into(), vec![N_EMBD, n_vocab], false);
    push_t("output_norm.weight".into(), vec![N_EMBD], true);
    push_t("output.weight".into(), vec![N_EMBD, n_vocab], false);
    for il in 0..N_LAYERS {
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
        push_t(format!("blk.{il}.ffn_norm.weight"), vec![N_EMBD], true);
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
    }

    let mut cur_off = 0usize;
    for t in &mut tensors {
        cur_off = (cur_off + ALIGN - 1) & !(ALIGN - 1);
        t.rel_offset = cur_off as u64;
        cur_off += t.bytes.len();
    }

    let mut kv = Vec::new();
    let mut kv_n = 0u64;
    kv_str(&mut kv, "general.architecture", "llama");
    kv_n += 1;
    kv_str(&mut kv, "general.name", "synthetic-sdk-test");
    kv_n += 1;
    kv_u32(&mut kv, "llama.block_count", N_LAYERS);
    kv_n += 1;
    kv_u32(&mut kv, "llama.context_length", 128);
    kv_n += 1;
    kv_u32(&mut kv, "llama.embedding_length", N_EMBD as u32);
    kv_n += 1;
    kv_u32(&mut kv, "llama.feed_forward_length", N_FF as u32);
    kv_n += 1;
    kv_u32(&mut kv, "llama.attention.head_count", N_HEAD);
    kv_n += 1;
    kv_u32(&mut kv, "llama.attention.head_count_kv", N_HEAD);
    kv_n += 1;
    kv_f32(&mut kv, "llama.attention.layer_norm_rms_epsilon", 1e-5);
    kv_n += 1;
    kv_f32(&mut kv, "llama.rope.freq_base", 10_000.0);
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
    kv.extend_from_slice(&n_vocab.to_le_bytes());
    for i in 0..n_vocab {
        let tok = match i {
            0 => "<bos>".to_string(),
            1 => "<eos>".to_string(),
            2 => " ".to_string(),
            _ => {
                let ch = (b'a' + ((i - 3) % 26) as u8) as char;
                if alt_vocab {
                    format!("ALT_{ch}_{i}")
                } else if i < 29 {
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
    kv.extend_from_slice(&n_vocab.to_le_bytes());
    for i in 0..n_vocab {
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
fn ngram_and_draft_model_proposer_cover_sync_trim_cancel_and_bounds() {
    // NgramProposer edge cases
    let ng = NgramProposer::default();
    assert!(ng.propose(&[1, 2], 4).is_empty());
    assert!(ng.propose(&[1, 2, 3, 4], 0).is_empty());
    let ng_bad = NgramProposer {
        min_n: 5,
        max_n: 2,
        max_draft: 4,
    };
    assert!(ng_bad.propose(&[1, 2, 3, 4, 5, 6], 4).is_empty());
    let matched = ng.propose(&[3, 4, 5, 6, 7, 3, 4], 3);
    assert_eq!(matched, vec![5, 6, 7]);

    // DraftModelProposer on synthetic GGUF
    let gguf = write_tiny_llama_gguf(64, false);
    let mut cfg = EngineConfig::reference(gguf.path());
    cfg.n_ctx = 64;
    cfg.n_batch = 16;
    cfg.n_threads = 2;
    cfg.use_gpu = false;
    let mut proposer = DraftModelProposer::open(cfg, 4).unwrap();
    assert_eq!(proposer.engine().vocab_size(), 64);

    // Empty history -> empty proposal
    assert!(proposer.propose(&[], 4).unwrap().is_empty());

    // Pre-cancelled -> empty proposal
    let cancel = AtomicBool::new(true);
    assert!(
        proposer
            .propose_with_cancel(&[0, 3, 4], 3, Some(&cancel))
            .unwrap()
            .is_empty()
    );

    // Normal draft proposal (limit=0 uses max_draft)
    let drafted = proposer.propose(&[0, 3, 4, 5], 0).unwrap();
    assert!(!drafted.is_empty());

    // Calling again with identical history -> no delta -> empty
    assert!(proposer.propose(&[0, 3, 4, 5], 3).unwrap().is_empty());

    // Extending history reuses prefix
    let drafted2 = proposer.propose(&[0, 3, 4, 5, 6], 3).unwrap();
    assert!(!drafted2.is_empty());

    // Prefix divergence triggers kv_trim(common)
    let _ = proposer.propose(&[0, 3, 7, 8], 2).unwrap();

    // Complete divergence (common == 0) triggers reset
    let _ = proposer.propose(&[3, 4, 5], 2).unwrap();

    // Out-of-vocab token returns Ok(vec![]) cleanly
    assert!(proposer.propose(&[0, 9999], 2).unwrap().is_empty());

    // History >= n_ctx returns Ok(vec![]) cleanly
    let long_hist = vec![3i32; proposer.engine().n_ctx() as usize];
    assert!(proposer.propose(&long_hist, 2).unwrap().is_empty());

    // trim_to(k) and trim_to(0)
    proposer.trim_to(2);
    proposer.trim_to(0);
    proposer.reset();
}

#[cfg(feature = "hf")]
#[test]
fn sdk_loaded_session_inspect_and_check_model_end_to_end() {
    let target = write_tiny_llama_gguf(64, false);
    let draft = write_tiny_llama_gguf(64, false);
    let incompat = write_tiny_llama_gguf(64, true);

    // inspect_gguf_file (valid + missing)
    let info = inspect_gguf_file(target.path()).unwrap();
    assert_eq!(info.architecture, "llama");
    assert_eq!(info.n_layers, 2);
    assert_eq!(info.shard_count, 1);
    assert!(inspect_gguf_file(std::path::Path::new("/nonexistent/model.gguf")).is_err());

    // LoadedSession::open with explicit draft model + vram_budget_bytes
    let opts = OpenSessionOptions {
        draft_model: Some(draft.path().to_string_lossy().into_owned()),
        vram_budget_bytes: 512 * 1024 * 1024,
        resident_layers: Some(0),
        ctx_size: 128,
        batch_size: 32,
        threads: Some(2),
        kv_cache_type: KvCacheType::F16,
        use_gpu: false,
        prefer_direct_io: false,
        max_draft_tokens: 3,
        ..OpenSessionOptions::default()
    };
    let mut sess = LoadedSession::open(&target.path().to_string_lossy(), opts).unwrap();
    let snap = sess.stats_snapshot();
    assert!(snap.loaded);
    assert!(!snap.isolated_worker);
    assert!(snap.draft_model_path.is_some());
    assert_eq!(snap.n_layers, 2);

    // Generate using the attached draft speculator!
    let gen_opts = GenerateOptions {
        max_tokens: 4,
        ..GenerateOptions::default()
    };
    let mut out = String::new();
    let summary = sess
        .engine
        .generate_stream("abc", &gen_opts, sess.draft.as_mut(), |_tok, piece| {
            out.push_str(piece);
            true
        })
        .unwrap();
    assert!(summary.completion_tokens > 0);
    assert!(!out.is_empty());

    // Incompatible explicit draft fails closed
    let bad_opts = OpenSessionOptions {
        draft_model: Some(incompat.path().to_string_lossy().into_owned()),
        ctx_size: 64,
        batch_size: 32,
        use_gpu: false,
        ..OpenSessionOptions::default()
    };
    assert!(LoadedSession::open(&target.path().to_string_lossy(), bad_opts).is_err());

    // check_model without draft (CPU)
    let rep = check_model(
        &target.path().to_string_lossy(),
        CheckOptions {
            use_gpu: false,
            allow_slow: true,
            ..CheckOptions::default()
        },
    )
    .unwrap();
    assert!(rep.ok);
    assert_eq!(rep.verdict, "OK");
    assert_eq!(rep.gpu_backend.as_deref(), Some("CPU"));

    // check_model with compatible draft (CPU) -> GOOD verdict
    let rep_draft = check_model(
        &target.path().to_string_lossy(),
        CheckOptions {
            draft_model: Some(draft.path().to_string_lossy().into_owned()),
            use_gpu: false,
            allow_slow: true,
            ..CheckOptions::default()
        },
    )
    .unwrap();
    assert!(rep_draft.ok);
    assert_eq!(rep_draft.verdict, "GOOD");
    assert!(rep_draft.draft_model_path.is_some());

    // check_model with incompatible draft -> error
    assert!(
        check_model(
            &target.path().to_string_lossy(),
            CheckOptions {
                draft_model: Some(incompat.path().to_string_lossy().into_owned()),
                use_gpu: false,
                ..CheckOptions::default()
            },
        )
        .is_err()
    );

    // recommend_repo with invalid repo spec fails cleanly before network
    assert!(recommend_repo("not-a-valid-repo", RecommendOptions::default()).is_err());
}
