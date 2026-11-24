use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use layerstream_core::{
    EngineConfig, GenerateOptions, GgufIndex, LayerIoPool, NgramProposer, Residency, StreamEngine,
};
use tempfile::NamedTempFile;

/// Builds a minimal valid synthetic GGUF v3 file with 2 layers (`blk.0.*`, `blk.1.*`)
/// to verify header parsing, 4K-aligned span coalescing, and O_DIRECT/buffered pread.
fn write_synthetic_gguf() -> (NamedTempFile, Vec<u8>, Vec<u8>) {
    let mut f = NamedTempFile::new().expect("temp gguf");
    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&3u32.to_le_bytes()); // version 3
    header.extend_from_slice(&2u64.to_le_bytes()); // 2 tensors
    header.extend_from_slice(&2u64.to_le_bytes()); // 2 kv pairs

    let write_str = |buf: &mut Vec<u8>, s: &str| {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    };

    // kv 1: general.architecture = "llama"
    write_str(&mut header, "general.architecture");
    header.extend_from_slice(&8u32.to_le_bytes()); // STRING
    write_str(&mut header, "llama");

    // kv 2: llama.block_count = 2
    write_str(&mut header, "llama.block_count");
    header.extend_from_slice(&4u32.to_le_bytes()); // UINT32
    header.extend_from_slice(&2u32.to_le_bytes());

    let payload0: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
    let payload1: Vec<u8> = (0..2048).map(|i| ((i * 7 + 3) % 251) as u8).collect();

    // tensor 0: blk.0.attn_q.weight at rel_offset = 0
    write_str(&mut header, "blk.0.attn_q.weight");
    header.extend_from_slice(&1u32.to_le_bytes());
    header.extend_from_slice(&256u64.to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes()); // F32
    header.extend_from_slice(&0u64.to_le_bytes());

    // tensor 1: blk.1.attn_q.weight at rel_offset = 1024
    write_str(&mut header, "blk.1.attn_q.weight");
    header.extend_from_slice(&1u32.to_le_bytes());
    header.extend_from_slice(&512u64.to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes()); // F32
    header.extend_from_slice(&1024u64.to_le_bytes());

    // Pad header to 32-byte default alignment.
    let aligned_header = (header.len() + 31) & !31;
    header.resize(aligned_header, 0);

    f.write_all(&header).unwrap();
    f.write_all(&payload0).unwrap();
    f.write_all(&payload1).unwrap();
    f.flush().unwrap();

    (f, payload0, payload1)
}

#[test]
fn gguf_index_and_coalesced_pread_roundtrip() {
    let (file, p0, p1) = write_synthetic_gguf();
    let index = GgufIndex::open(file.path()).expect("open synthetic gguf");
    assert_eq!(index.meta.architecture, "llama");
    assert_eq!(index.meta.n_layers, 2);
    assert_eq!(index.tensors.len(), 2);

    let spans = index
        .plan_coalesced_spans(&[
            ("blk.0.attn_q.weight", 0, p0.len()),
            ("blk.1.attn_q.weight", p0.len(), p1.len()),
        ])
        .expect("plan spans");
    // Adjacent tensors must coalesce into a single 4K-aligned span.
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].aligned_offset % 4096, 0);

    let pool = LayerIoPool::new(index, true).expect("io pool");
    let mut dst = vec![0u8; p0.len() + p1.len()];
    pool.load_tensors_into(
        &[
            ("blk.0.attn_q.weight", 0, p0.len()),
            ("blk.1.attn_q.weight", p0.len(), p1.len()),
        ],
        &mut dst,
    )
    .expect("load tensors");

    assert_eq!(&dst[..p0.len()], p0.as_slice());
    assert_eq!(&dst[p0.len()..], p1.as_slice());

    // Verify arbitrary sub-tensor slice coalescing & parallel slice loading (MoE path).
    let mut sub_dst = vec![0u8; 512];
    pool.load_slices_into(
        &[
            ("blk.0.attn_q.weight", 128, 0, 256, 0),
            ("blk.1.attn_q.weight", 512, 256, 256, 1),
        ],
        &mut sub_dst,
    )
    .expect("load sub-tensor slices");
    assert_eq!(&sub_dst[0..256], &p0[128..384]);
    assert_eq!(&sub_dst[256..512], &p1[512..768]);
}

#[tokio::test]
async fn residency_evicts_immediately_when_idle_timeout_is_zero() {
    let residency: Residency<u32> = Residency::new(Duration::ZERO, 2);
    assert!(!residency.is_loaded());
    {
        let mut lease = residency.acquire().await.unwrap();
        let val = lease.get_or_load("model-a", || Ok(42)).unwrap();
        assert_eq!(*val, 42);
    }
    // Immediate drop frees slot back to unloaded (0 MB VRAM).
    assert!(!residency.is_loaded());
}

#[test]
fn ngram_proposer_matches_longest_suffix() {
    let proposer = NgramProposer::default();
    let history = vec![10, 20, 30, 40, 50, 60, 99, 20, 30, 40];
    let draft = proposer.propose(&history, 4);
    assert_eq!(draft, vec![50, 60, 99, 20]);
}

/// Builds a tiny valid 6-layer mixed Dense (`blk.0..3`) + Sparse MoE (`blk.4..5`, 4 experts, top-2)
/// `ernie4_5-moe` GGUF v3 model in a temporary file for fast deterministic parity testing across
/// CPU, Vulkan, CUDA, and Metal without requiring external model downloads.
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
        buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
        write_str(buf, v);
    };
    let kv_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
        write_str(buf, k);
        buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
        buf.extend_from_slice(&v.to_le_bytes());
    };
    let kv_f32 = |buf: &mut Vec<u8>, k: &str, v: f32| {
        write_str(buf, k);
        buf.extend_from_slice(&6u32.to_le_bytes()); // FLOAT32
        buf.extend_from_slice(&v.to_le_bytes());
    };
    let kv_bool = |buf: &mut Vec<u8>, k: &str, v: bool| {
        write_str(buf, k);
        buf.extend_from_slice(&7u32.to_le_bytes()); // BOOL
        buf.push(u8::from(v));
    };

    // Deterministic non-degenerate F32 weights via xorshift64.
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

    // Build GGUF v3 header.
    let mut kv_buf = Vec::new();
    let mut kv_count = 0u64;
    let arch = "ernie4_5-moe";
    kv_str(&mut kv_buf, "general.architecture", arch);
    kv_count += 1;
    kv_str(&mut kv_buf, "general.name", "synthetic-mixed-dense-moe");
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

    // Minimal BPE tokenizer with 64 ASCII tokens (including empty merges array).
    kv_str(&mut kv_buf, "tokenizer.ggml.model", "gpt2");
    kv_count += 1;
    kv_str(&mut kv_buf, "tokenizer.ggml.pre", "default");
    kv_count += 1;
    write_str(&mut kv_buf, "tokenizer.ggml.merges");
    kv_buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
    kv_buf.extend_from_slice(&8u32.to_le_bytes()); // elem = STRING
    kv_buf.extend_from_slice(&0u64.to_le_bytes()); // 0 merges
    kv_count += 1;
    write_str(&mut kv_buf, "tokenizer.ggml.tokens");
    kv_buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
    kv_buf.extend_from_slice(&8u32.to_le_bytes()); // elem = STRING
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
    kv_buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
    kv_buf.extend_from_slice(&5u32.to_le_bytes()); // elem = INT32
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
        header.extend_from_slice(&0u32.to_le_bytes()); // GGML_TYPE_F32
        header.extend_from_slice(&t.rel_offset.to_le_bytes());
    }
    let aligned_header = (header.len() + ALIGN - 1) & !(ALIGN - 1);
    header.resize(aligned_header, 0);

    let mut file = NamedTempFile::new().expect("temp mixed dense+moe gguf");
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

/// Regression gate for P0-1, P0-2, and P2-10: verifies exact greedy parity on a mixed
/// Dense (blk.0..3) + Sparse MoE (blk.4..5) model across R=0, R=2, and stream_head=true.
#[test]
fn synthetic_mixed_dense_moe_matches_reference_parity() {
    let gguf = write_synthetic_mixed_dense_moe_gguf();
    let path = gguf.path().to_path_buf();
    let use_gpu = std::env::var("LAYERSTREAM_TEST_GPU").as_deref() == Ok("1");
    let prompt = "a b c d e f a b c";
    let max_tokens = 12;

    let mut ref_cfg = EngineConfig::reference(&path);
    ref_cfg.use_gpu = use_gpu;
    ref_cfg.n_ctx = 128;
    ref_cfg.n_batch = 64;
    let mut reference = StreamEngine::open(&ref_cfg).expect("open mixed dense+moe reference");
    let expected = reference
        .generate_greedy(prompt, max_tokens)
        .expect("reference generate on mixed dense+moe");
    assert_eq!(expected.len(), max_tokens);
    drop(reference);

    // 1. R=0: 4 streamed dense layers (ping-pong) + 2 streamed sparse MoE layers (expert_slot).
    let mut r0_cfg = EngineConfig::streaming(&path);
    r0_cfg.use_gpu = use_gpu;
    r0_cfg.n_ctx = 128;
    r0_cfg.n_batch = 64;
    r0_cfg.resident_layers = Some(0);
    let mut eng_r0 = StreamEngine::open(&r0_cfg).expect("open mixed dense+moe R=0");
    let st0 = eng_r0.stats();
    assert_eq!(st0.n_layers, 6);
    assert_eq!(st0.moe_layers, 2);
    assert!(st0.moe_active_streaming);
    assert!(
        st0.slot_bytes > 0,
        "expected ping-pong slot for dense layers"
    );
    assert!(
        st0.expert_slot_bytes > 0,
        "expected expert slot for sparse MoE layers"
    );
    let out_r0 = eng_r0
        .generate_greedy(prompt, max_tokens)
        .expect("mixed R=0 generate");
    assert_eq!(
        out_r0, expected,
        "mixed dense+MoE R=0 diverged from reference"
    );
    // Second cold pass exercises hot-expert RAM LRU cache hits while staying bit-exact.
    eng_r0.kv_clear();
    let out_r0_cached = eng_r0
        .generate_greedy(prompt, max_tokens)
        .expect("mixed R=0 cached generate");
    assert_eq!(
        out_r0_cached, expected,
        "mixed dense+MoE R=0 with warm expert LRU cache diverged"
    );
    drop(eng_r0);

    // 2. R=2: blk.0..1 resident, blk.2..3 streamed dense, blk.4..5 streamed sparse MoE.
    let mut r2_cfg = EngineConfig::streaming(&path);
    r2_cfg.use_gpu = use_gpu;
    r2_cfg.n_ctx = 128;
    r2_cfg.n_batch = 64;
    r2_cfg.resident_layers = Some(2);
    let mut eng_r2 = StreamEngine::open(&r2_cfg).expect("open mixed dense+moe R=2");
    assert_eq!(eng_r2.stats().resident_layers, 2);
    let out_r2 = eng_r2
        .generate_greedy(prompt, max_tokens)
        .expect("mixed R=2 generate");
    assert_eq!(
        out_r2, expected,
        "mixed dense+MoE R=2 diverged from reference"
    );
    drop(eng_r2);

    // 3. R=0 + stream_head=true: dense layers + sparse MoE + streamed LM head.
    let mut sh_cfg = EngineConfig::streaming(&path);
    sh_cfg.use_gpu = use_gpu;
    sh_cfg.n_ctx = 128;
    sh_cfg.n_batch = 64;
    sh_cfg.resident_layers = Some(0);
    sh_cfg.stream_head = true;
    let mut eng_sh = StreamEngine::open(&sh_cfg).expect("open mixed dense+moe stream_head");
    let out_sh = eng_sh
        .generate_greedy(prompt, max_tokens)
        .expect("mixed stream_head generate");
    assert_eq!(
        out_sh, expected,
        "mixed dense+MoE stream_head diverged from reference"
    );
}

/// Real model parity gate: set `LAYERSTREAM_TEST_GGUF=/path/to/model.gguf` to verify
/// 100% identical greedy tokens between stock resident llama.cpp, 2-slot streaming
/// (R=0 and hybrid R=2), and speculative n-gram verification!
#[test]
fn streaming_matches_stock_llama_cpp_greedy_parity() {
    let Ok(model_path) = std::env::var("LAYERSTREAM_TEST_GGUF") else {
        eprintln!("skipping real-model parity test (set LAYERSTREAM_TEST_GGUF to run)");
        return;
    };
    let path = PathBuf::from(model_path);
    let use_gpu = std::env::var("LAYERSTREAM_TEST_GPU").as_deref() == Ok("1");
    let prompt = "Repeat this exact sequence of words: alpha beta gamma delta epsilon alpha beta gamma delta epsilon alpha beta";
    let followup = format!("{prompt} gamma delta");
    let max_tokens = 24;

    let spec_opts = GenerateOptions {
        max_tokens,
        enable_ngram_speculation: true,
        max_draft_tokens: 5,
        ..Default::default()
    };

    let mut ref_cfg = EngineConfig::reference(&path);
    ref_cfg.use_gpu = use_gpu;
    let mut reference = StreamEngine::open(&ref_cfg).expect("open reference engine");
    let expected = reference
        .generate_greedy(prompt, max_tokens)
        .expect("reference generate");
    reference.kv_clear();
    let mut expected_spec = Vec::new();
    reference
        .generate_stream(prompt, &spec_opts, None, |tok, _| {
            expected_spec.push(tok);
            true
        })
        .expect("reference speculative generate");
    reference.kv_clear();
    let cold_followup = reference
        .generate_greedy(&followup, 16)
        .expect("cold followup generate");
    drop(reference);

    // Test 1: Pure 2-slot streaming (0 resident layers, no speculation).
    let mut stream_cfg = EngineConfig::streaming(&path);
    stream_cfg.use_gpu = use_gpu;
    stream_cfg.resident_layers = Some(0);
    let mut streamed = StreamEngine::open(&stream_cfg).expect("open streaming engine (R=0)");
    let stats0 = streamed.stats();
    assert_eq!(stats0.resident_layers, 0);
    assert!(stats0.streamed_layers > 0);
    let actual_r0 = streamed
        .generate_greedy(prompt, max_tokens)
        .expect("streamed generate (R=0)");
    assert_eq!(
        actual_r0, expected,
        "R=0 streamed tokens diverged from stock llama.cpp"
    );
    assert!(streamed.stats().total_layer_loads > 0);

    // Test 2: Pure 2-slot streaming + zero-VRAM n-gram speculative verification
    // (compared apples-to-apples against stock reference running the same batched verify).
    streamed.kv_clear();
    let mut actual_spec = Vec::new();
    let summary = streamed
        .generate_stream(prompt, &spec_opts, None, |tok, _| {
            actual_spec.push(tok);
            true
        })
        .expect("speculative streamed generate");
    assert_eq!(
        actual_spec, expected_spec,
        "speculative streamed tokens diverged from stock llama.cpp"
    );
    eprintln!(
        "Speculation summary: tokens={} sweeps={} drafted={} accepted={}",
        summary.completion_tokens,
        summary.speculation.sweeps,
        summary.speculation.drafted_tokens,
        summary.speculation.accepted_tokens
    );
    assert!(
        summary.speculation.accepted_tokens > 0,
        "expected n-gram speculation to accept tokens on repetitive prompt"
    );
    assert!(
        summary.speculation.sweeps < summary.completion_tokens,
        "expected fewer NVMe sweeps ({}) than generated tokens ({})",
        summary.speculation.sweeps,
        summary.completion_tokens
    );
    drop(streamed);

    // Test 3: Hybrid partial-residency (first 2 layers pinned resident, rest streamed).
    if stats0.n_layers > 3 {
        let mut hybrid_cfg = EngineConfig::streaming(&path);
        hybrid_cfg.use_gpu = use_gpu;
        hybrid_cfg.resident_layers = Some(2);
        let mut hybrid = StreamEngine::open(&hybrid_cfg).expect("open hybrid engine (R=2)");
        assert_eq!(hybrid.stats().resident_layers, 2);
        let actual_r2 = hybrid
            .generate_greedy(prompt, max_tokens)
            .expect("hybrid generate (R=2)");
        assert_eq!(
            actual_r2, expected,
            "R=2 hybrid tokens diverged from stock llama.cpp"
        );

        // Test 4: Multi-turn KV prefix reuse must match cold greedy output identically.
        let warm_followup = hybrid
            .generate_greedy(&followup, 16)
            .expect("warm prefix-reused followup generate");
        assert_eq!(
            warm_followup, cold_followup,
            "multi-turn KV prefix reuse diverged from cold reference"
        );

        // Test 5: Sampled lossless speculative verification at temperature > 0.
        let mut sampled_opts = GenerateOptions {
            max_tokens: 24,
            enable_ngram_speculation: true,
            max_draft_tokens: 5,
            ..Default::default()
        };
        sampled_opts.sample.temperature = 0.7;
        sampled_opts.sample.top_p = 0.9;
        sampled_opts.sample.top_k = 40;
        let mut sampled_out = Vec::new();
        let sampled_summary = hybrid
            .generate_stream(prompt, &sampled_opts, None, |tok, _| {
                sampled_out.push(tok);
                true
            })
            .expect("sampled speculative generate");
        eprintln!(
            "Sampled (temp=0.7) speculation: tokens={} sweeps={} drafted={} accepted={}",
            sampled_summary.completion_tokens,
            sampled_summary.speculation.sweeps,
            sampled_summary.speculation.drafted_tokens,
            sampled_summary.speculation.accepted_tokens
        );
        assert_eq!(sampled_out.len(), sampled_summary.completion_tokens);
        assert!(
            sampled_summary.speculation.accepted_tokens > 0,
            "expected sampled speculative rejection sampling to accept tokens at temp=0.7"
        );
        drop(hybrid);
    }

    // Test 6: Ultra-low-VRAM floor (--cpu-embeddings + --cpu-head / --stream-head) parity.
    let mut low_vram_cfg = EngineConfig::streaming(&path);
    low_vram_cfg.use_gpu = use_gpu;
    low_vram_cfg.resident_layers = Some(0);
    low_vram_cfg.cpu_embeddings = true;
    low_vram_cfg.stream_head = true;
    let mut low_vram = StreamEngine::open(&low_vram_cfg).expect("open ultra-low-vram engine");
    let lv_stats = low_vram.stats();
    if lv_stats.using_gpu && !lv_stats.is_unified_memory {
        assert!(
            lv_stats.cpu_resident_bytes > 0 || lv_stats.head_streamed,
            "expected embeddings/head offloaded from discrete GPU resident buffer"
        );
    }
    let actual_low_vram = low_vram
        .generate_greedy(prompt, max_tokens)
        .expect("ultra-low-vram generate");
    assert_eq!(
        actual_low_vram, expected,
        "ultra-low-vram (--cpu-embeddings --stream-head) tokens diverged from stock llama.cpp"
    );
    drop(low_vram);

    // Test 7: Misuse guardrails — auto VRAM budget + over-budget clamping + draft tokenizer check.
    let mut auto_cfg = EngineConfig::streaming(&path);
    auto_cfg.use_gpu = use_gpu;
    auto_cfg.resident_layers = None;
    auto_cfg.vram_budget_auto = true;
    let mut auto_eng = StreamEngine::open(&auto_cfg).expect("open auto-budget engine");
    assert!(
        auto_eng.check_draft_compatible(&auto_eng).is_ok(),
        "identical tokenizer must pass draft compatibility check"
    );
    let actual_auto = auto_eng
        .generate_greedy(prompt, 12)
        .expect("auto-budget generate");
    assert_eq!(
        actual_auto,
        expected[..12],
        "auto-budget tokens diverged from stock llama.cpp"
    );
    drop(auto_eng);

    let mut over_cfg = EngineConfig::streaming(&path);
    over_cfg.use_gpu = use_gpu;
    over_cfg.resident_layers = None;
    // Deliberately absurd 1 TiB budget must clamp safely to available GPU memory without OOM.
    over_cfg.vram_budget_bytes = 1024 * 1024 * 1024 * 1024;
    let over_eng = StreamEngine::open(&over_cfg).expect("over-budget must clamp safely");
    assert!(over_eng.stats().resident_layers <= over_eng.stats().n_layers);
}
