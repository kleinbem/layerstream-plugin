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
    let max_tokens = 24;

    let mut ref_cfg = EngineConfig::reference(&path);
    ref_cfg.use_gpu = use_gpu;
    let mut reference = StreamEngine::open(&ref_cfg).expect("open reference engine");
    let expected = reference
        .generate_greedy(prompt, max_tokens)
        .expect("reference generate");
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

    // Test 2: Pure 2-slot streaming + zero-VRAM n-gram speculative verification.
    let spec_opts = GenerateOptions {
        max_tokens,
        enable_ngram_speculation: true,
        max_draft_tokens: 5,
        ..Default::default()
    };
    let mut actual_spec = Vec::new();
    let summary = streamed
        .generate_stream(prompt, &spec_opts, None, |tok, _| {
            actual_spec.push(tok);
            true
        })
        .expect("speculative streamed generate");
    assert_eq!(
        actual_spec, expected,
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
    }
}
