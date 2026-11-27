//! Pure-Rust hot-path microbenchmarks and latency regression guards.
//!
//! Runs quickly (~100 ms total) as part of `cargo test` so regressions in
//! GGUF parsing, 4K span coalescing, TinyLFU expert-cache hits, JSON-Schema
//! GBNF compilation, and SSE tag streaming are caught automatically.

use std::collections::HashMap;
use std::time::Instant;

use layerstream_core::{
    GgufIndex, LayerIoPool, ModelMeta, StreamTagParser, TensorLocation, json_schema_to_gbnf,
};
use serde_json::json;

fn median_us(mut samples: Vec<u128>) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2] as f64 / 1000.0
}

#[test]
fn hot_path_microbenchmarks_within_latency_budgets() {
    // 1. 4K span coalescing across 16 scattered expert slices.
    let mut tensors = HashMap::new();
    for i in 0..16u64 {
        tensors.insert(
            format!("blk.0.ffn_gate_exps.{i}"),
            TensorLocation {
                name: format!("blk.0.ffn_gate_exps.{i}"),
                shard_idx: 0,
                abs_offset: 4096 + i * 262_144,
                ggml_type: 0,
                dims: vec![16384], // 64 KiB each
            },
        );
    }
    let dir = std::env::temp_dir().join(format!("ls-microbench-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let shard = dir.join("bench.gguf");
    std::fs::write(&shard, vec![0x5Au8; 8 * 1024 * 1024]).expect("write shard");

    let idx = GgufIndex {
        shards: vec![shard],
        shard_sizes: vec![8 * 1024 * 1024],
        meta: ModelMeta {
            architecture: "qwen3moe".to_string(),
            name: Some("bench".to_string()),
            n_layers: 1,
            context_length: Some(4096),
            total_file_bytes: 8 * 1024 * 1024,
        },
        tensors,
    };

    let reqs: Vec<(&str, u64, usize, usize, u32)> = (0..8usize)
        .map(|i| {
            let name: &'static str = match i {
                0 => "blk.0.ffn_gate_exps.0",
                1 => "blk.0.ffn_gate_exps.2",
                2 => "blk.0.ffn_gate_exps.4",
                3 => "blk.0.ffn_gate_exps.6",
                4 => "blk.0.ffn_gate_exps.8",
                5 => "blk.0.ffn_gate_exps.10",
                6 => "blk.0.ffn_gate_exps.12",
                _ => "blk.0.ffn_gate_exps.14",
            };
            (name, 0u64, i * 16384, 16384usize, i as u32)
        })
        .collect();

    let mut coalesce_ns = Vec::with_capacity(200);
    for _ in 0..200 {
        let t0 = Instant::now();
        let spans = idx.plan_coalesced_sub_slices(&reqs).expect("plan");
        assert_eq!(spans.len(), 8);
        coalesce_ns.push(t0.elapsed().as_nanos());
    }
    let coalesce_us = median_us(coalesce_ns);

    // 2. TinyLFU hot-expert cache hit path (8 x 16 KiB = 128 KiB from RAM).
    let pool =
        LayerIoPool::new_with_options(idx, false, Some(2), Some(16 * 1024 * 1024)).expect("pool");
    let mut dst = vec![0u8; 8 * 16384];
    pool.load_slices_into(&reqs, &mut dst).expect("warm");
    let mut cache_hit_ns = Vec::with_capacity(200);
    for _ in 0..200 {
        let t0 = Instant::now();
        pool.load_slices_into(&reqs, &mut dst).expect("hit");
        cache_hit_ns.push(t0.elapsed().as_nanos());
    }
    let cache_hit_us = median_us(cache_hit_ns);

    // 3. JSON Schema -> GBNF compilation (typical Pydantic tool schema).
    let schema = json!({
        "type": "object",
        "properties": {
            "thought": {"type": "string", "minLength": 1, "maxLength": 256},
            "action": {"enum": ["search", "read", "reply"]},
            "args": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"}
                },
                "required": ["query"]
            }
        },
        "required": ["thought", "action", "args"]
    });
    let mut gbnf_ns = Vec::with_capacity(50);
    for _ in 0..50 {
        let t0 = Instant::now();
        let g = json_schema_to_gbnf(&schema).expect("gbnf");
        assert!(g.contains("root ::="));
        gbnf_ns.push(t0.elapsed().as_nanos());
    }
    let gbnf_us = median_us(gbnf_ns);

    // 4. StreamTagParser streaming throughput (100 token chunks).
    let chunks = [
        "<think>",
        "checking ",
        "weather ",
        "</think>",
        "Sure! ",
        "<tool_call>",
        "{\"name\":\"wx\",",
        "\"arguments\":{\"q\":\"NYC\"}}",
        "</tool_call>",
    ];
    let mut stream_ns = Vec::with_capacity(200);
    for _ in 0..200 {
        let t0 = Instant::now();
        let mut p = StreamTagParser::new();
        for c in &chunks {
            let _ = p.push(c);
        }
        let _ = p.flush();
        assert!(p.has_tool_calls());
        stream_ns.push(t0.elapsed().as_nanos());
    }
    let stream_us = median_us(stream_ns);

    eprintln!(
        "microbench median_us: coalesce={coalesce_us:.1}us cache_hit_128k={cache_hit_us:.1}us gbnf_compile={gbnf_us:.1}us stream_tag={stream_us:.1}us"
    );

    // Generous regression ceilings (works under debug + ASan/TSan CI runners too).
    assert!(
        coalesce_us < 50_000.0,
        "coalesce regression: {coalesce_us:.1}us"
    );
    assert!(
        cache_hit_us < 50_000.0,
        "cache_hit regression: {cache_hit_us:.1}us"
    );
    assert!(
        gbnf_us < 200_000.0,
        "gbnf_compile regression: {gbnf_us:.1}us"
    );
    assert!(
        stream_us < 50_000.0,
        "stream_tag regression: {stream_us:.1}us"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
