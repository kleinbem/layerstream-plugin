//! Reproducible benchmark harness for `layerstream-core` (JSON output).
//!
//! Measures cold load time, TTFT, decode tok/s, VmHWM peak RSS, `/proc/meminfo`
//! page-cache delta, NVMe bytes read, and speculative sweep efficiency across
//! three standardized workload archetypes.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use layerstream_core::{
    DraftModelProposer, EngineConfig, GenerateOptions, KvCacheType, StreamEngine,
};

struct Workload {
    id: &'static str,
    prompt: &'static str,
    max_tokens: usize,
}

const WORKLOADS: &[Workload] = &[
    Workload {
        id: "prompt_heavy",
        prompt: "You are reviewing a systems architecture document. Read the following technical specification carefully and summarize the core memory-management mechanism in one concise sentence.\n\n\
Specification:\n\
Modern large language models are structured as sequential stacks of transformer blocks where intermediate hidden states are several orders of magnitude smaller than layer weight matrices. On memory-constrained edge accelerators such as unified-memory ARM SoCs with 8 GiB LPDDR5 and PCIe NVMe storage, a 14-billion-parameter 4-bit quantized checkpoint requires roughly 9 GiB of weights plus key-value cache and compute workspace, exceeding total physical memory. Standard inference engines either fail during tensor allocation or rely on operating system virtual-memory paging via mmap, which pollutes the Linux page cache, triggers erratic swap thrashing, and serializes I/O behind synchronous page faults. Layer-streaming inference avoids full residency by allocating a small fixed pool of reusable compute buffers sized to the largest single transformer block, keeping embeddings, output projections, and optionally early hot layers resident, and asynchronously streaming remaining blocks from NVMe using page-aligned O_DIRECT reads overlapped with GPU compute. On unified-memory architectures where the CPU and GPU share physical DRAM, zero-copy mapped host buffers eliminate redundant host-to-device DMA copies and halve peak staging memory. Prompt prefill runs layer-major over multi-hundred-token batches so an entire prompt requires only one sequential storage sweep. Autoregressive decode remains storage-bandwidth-bound at one token per full sweep unless paired with speculative verification, where a zero-VRAM n-gram lookup proposer or a tiny resident draft model proposes multiple candidate tokens that are verified in parallel within a single storage pass, multiplying effective decode throughput without altering greedy target output distribution.\n\n\
Summary:",
        max_tokens: 32,
    },
    Workload {
        id: "code_edit",
        prompt: "Repeat the following Rust struct verbatim and add a single field `pub timeout_secs: u64,` right before the closing brace:\n\n\
```rust\n\
#[derive(Debug, Clone)]\n\
pub struct StreamConfig {\n\
    pub model_path: PathBuf,\n\
    pub n_ctx: u32,\n\
    pub n_batch: u32,\n\
    pub resident_layers: Option<u32>,\n\
    pub vram_budget_bytes: u64,\n\
    pub use_gpu: bool,\n\
    pub prefer_direct_io: bool,\n\
}\n\
```\n\n\
Updated struct:\n```rust\n",
        max_tokens: 48,
    },
    Workload {
        id: "open_ended",
        prompt: "Explain clearly how page-aligned O_DIRECT NVMe reads and asynchronous ping-pong GPU buffers interact when streaming transformer layers on an 8 GB edge device:",
        max_tokens: 48,
    },
];

fn read_proc_kb(path: &str, key: &str) -> u64 {
    let Ok(content) = fs::read_to_string(path) else {
        return 0;
    };
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix(key)
            && let Some(val) = rest.split_whitespace().next()
            && let Ok(kb) = val.parse::<u64>()
        {
            return kb;
        }
    }
    0
}

fn page_cache_mib() -> f64 {
    read_proc_kb("/proc/meminfo", "Cached:") as f64 / 1024.0
}

fn peak_rss_mib() -> f64 {
    read_proc_kb("/proc/self/status", "VmHWM:") as f64 / 1024.0
}

fn current_rss_mib() -> f64 {
    read_proc_kb("/proc/self/status", "VmRSS:") as f64 / 1024.0
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path: Option<PathBuf> = None;
    let mut draft_path: Option<PathBuf> = None;
    let mut mode = "stream_r0".to_string();
    let mut resident_layers: Option<u32> = Some(0);
    let mut vram_budget_mb: u64 = 0;
    let mut use_gpu = true;
    let mut enable_ngram = false;
    let mut max_draft_tokens: usize = 5;
    let mut n_ctx: u32 = 2048;
    let mut n_batch: u32 = 512;
    let mut kv_cache_type = KvCacheType::F16;
    let mut reference = false;
    let mut workload_filter: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = Some(PathBuf::from(&args[i]));
            }
            "--draft-model" => {
                i += 1;
                draft_path = Some(PathBuf::from(&args[i]));
            }
            "--mode" => {
                i += 1;
                mode = args[i].clone();
            }
            "--resident-layers" => {
                i += 1;
                resident_layers = Some(args[i].parse().context("parse --resident-layers")?);
            }
            "--vram-budget-mb" => {
                i += 1;
                vram_budget_mb = args[i].parse().context("parse --vram-budget-mb")?;
                resident_layers = None;
            }
            "--no-gpu" => use_gpu = false,
            "--ngram" => enable_ngram = true,
            "--max-draft" => {
                i += 1;
                max_draft_tokens = args[i].parse().context("parse --max-draft")?;
            }
            "--ctx" => {
                i += 1;
                n_ctx = args[i].parse().context("parse --ctx")?;
            }
            "--batch" => {
                i += 1;
                n_batch = args[i].parse().context("parse --batch")?;
            }
            "--kv-cache-type" => {
                i += 1;
                kv_cache_type = match args[i].as_str() {
                    "f16" => KvCacheType::F16,
                    "q8_0" => KvCacheType::Q8_0,
                    "q4_0" => KvCacheType::Q4_0,
                    other => bail!("unsupported --kv-cache-type: {other} (use f16|q8_0|q4_0)"),
                };
            }
            "--reference" => reference = true,
            "--workload" => {
                i += 1;
                workload_filter = Some(args[i].clone());
            }
            other => bail!("unknown argument: {other}"),
        }
        i += 1;
    }

    let model_path = model_path.context("--model <path.gguf> is required")?;

    let cache_before_load = page_cache_mib();
    let t_load = Instant::now();

    let mut cfg = if reference {
        EngineConfig::reference(&model_path)
    } else {
        EngineConfig::streaming(&model_path)
    };
    cfg.n_ctx = n_ctx;
    cfg.n_batch = n_batch;
    cfg.kv_cache_type = kv_cache_type;
    cfg.use_gpu = use_gpu;
    cfg.resident_layers = resident_layers;
    cfg.vram_budget_bytes = vram_budget_mb.saturating_mul(1024 * 1024);

    let mut engine = StreamEngine::open(&cfg)?;
    let mut draft = match &draft_path {
        Some(dp) => {
            let mut dcfg = EngineConfig::reference(dp);
            dcfg.n_ctx = n_ctx;
            dcfg.n_batch = n_batch;
            dcfg.use_gpu = use_gpu;
            Some(DraftModelProposer::open(dcfg, max_draft_tokens)?)
        }
        None => None,
    };
    let load_time_s = t_load.elapsed().as_secs_f64();
    let stats = engine.stats();

    let mut results = Vec::new();
    for wl in WORKLOADS {
        if let Some(ref f) = workload_filter
            && wl.id != f
        {
            continue;
        }

        let cache_before = page_cache_mib();
        let io_before = engine.io_stats();
        let t0 = Instant::now();
        let mut ttft_s: Option<f64> = None;
        let mut text = String::new();

        let opts = GenerateOptions {
            max_tokens: wl.max_tokens,
            enable_ngram_speculation: enable_ngram,
            max_draft_tokens,
            ..Default::default()
        };

        let summary = engine.generate_stream(wl.prompt, &opts, draft.as_mut(), |_tok, piece| {
            if ttft_s.is_none() {
                ttft_s = Some(t0.elapsed().as_secs_f64());
            }
            text.push_str(piece);
            true
        })?;

        let total_s = t0.elapsed().as_secs_f64();
        let ttft = ttft_s.unwrap_or(total_s);
        let decode_s = (total_s - ttft).max(1e-6);
        let decode_toks = summary.completion_tokens.saturating_sub(1);
        let decode_tps = if decode_toks > 0 {
            decode_toks as f64 / decode_s
        } else {
            summary.completion_tokens as f64 / total_s.max(1e-6)
        };
        let total_tps = summary.completion_tokens as f64 / total_s.max(1e-6);
        let io_after = engine.io_stats();
        let io_bytes_delta = io_after.bytes_read.saturating_sub(io_before.bytes_read);
        let cache_after = page_cache_mib();

        let record = serde_json::json!({
            "engine": "layerstream-plugin",
            "mode": mode,
            "model": model_path.file_name().and_then(|s| s.to_str()).unwrap_or("unknown"),
            "draft_model": draft_path.as_ref().and_then(|p| p.file_name()).and_then(|s| s.to_str()),
            "workload": wl.id,
            "load_time_s": round3(load_time_s),
            "ttft_s": round3(ttft),
            "decode_time_s": round3(decode_s),
            "total_time_s": round3(total_s),
            "prompt_tokens": summary.prompt_tokens,
            "completion_tokens": summary.completion_tokens,
            "decode_tok_per_s": round3(decode_tps),
            "total_tok_per_s": round3(total_tps),
            "sweeps": summary.speculation.sweeps,
            "drafted_tokens": summary.speculation.drafted_tokens,
            "accepted_tokens": summary.speculation.accepted_tokens,
            "io_mib_read": round2(io_bytes_delta as f64 / (1024.0 * 1024.0)),
            "direct_io_active": io_after.direct_io_active,
            "peak_rss_mib": round1(peak_rss_mib()),
            "current_rss_mib": round1(current_rss_mib()),
            "page_cache_delta_mib": round1(cache_after - cache_before),
            "page_cache_delta_since_start_mib": round1(cache_after - cache_before_load),
            "n_layers": stats.n_layers,
            "resident_layers": stats.resident_layers,
            "streamed_layers": stats.streamed_layers,
            "resident_mib": round1(stats.resident_bytes as f64 / (1024.0 * 1024.0)),
            "slot_mib": round1(stats.slot_bytes as f64 / (1024.0 * 1024.0)),
            "uma_zero_copy": stats.is_uma_zero_copy,
            "using_gpu": stats.using_gpu,
            "can_speculate": stats.can_speculate,
            "output_preview": text.chars().take(160).collect::<String>(),
        });
        results.push(record);
    }

    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}
fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}
