#![recursion_limit = "256"]
//! Reproducible benchmark harness for `layerstream-core` (JSON output).
//!
//! Measures cold load time, TTFT, decode tok/s, VmHWM peak RSS, `/proc/meminfo`
//! MemAvailable/swap/page-cache deltas, kernel `pread` bandwidth, MoE expert-cache
//! hit rates, INA3221 idle + active power, and speculative sweep efficiency across
//! three standardized workload archetypes (with optional `--chat`, `--warmup`, `--reps N`).

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use layerstream_core::{
    ChatMessage, DraftModelProposer, EngineConfig, GenerateOptions, KvCacheType, StreamEngine,
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

fn mem_avail_mib() -> f64 {
    read_proc_kb("/proc/meminfo", "MemAvailable:") as f64 / 1024.0
}

fn swap_used_mib() -> f64 {
    let total = read_proc_kb("/proc/meminfo", "SwapTotal:");
    let free = read_proc_kb("/proc/meminfo", "SwapFree:");
    total.saturating_sub(free) as f64 / 1024.0
}

fn peak_rss_mib() -> f64 {
    read_proc_kb("/proc/self/status", "VmHWM:") as f64 / 1024.0
}

fn current_rss_mib() -> f64 {
    read_proc_kb("/proc/self/status", "VmRSS:") as f64 / 1024.0
}

fn read_ina3221_watts() -> Option<f64> {
    let entries = fs::read_dir("/sys/class/hwmon").ok()?;
    for entry in entries.flatten() {
        let dir = entry.path();
        let Ok(name) = fs::read_to_string(dir.join("name")) else {
            continue;
        };
        if name.trim() != "ina3221" {
            continue;
        }
        if let Ok(uw_s) = fs::read_to_string(dir.join("power1_input"))
            && let Ok(uw) = uw_s.trim().parse::<f64>()
        {
            return Some(uw / 1_000_000.0);
        }
        if let (Ok(mv_s), Ok(ma_s)) = (
            fs::read_to_string(dir.join("in1_input")),
            fs::read_to_string(dir.join("curr1_input")),
        ) && let (Ok(mv), Ok(ma)) = (mv_s.trim().parse::<f64>(), ma_s.trim().parse::<f64>())
        {
            return Some((mv * ma) / 1_000_000.0);
        }
    }
    None
}

fn sample_idle_watts(duration_ms: u64) -> Option<f64> {
    read_ina3221_watts()?;
    let deadline = Instant::now() + Duration::from_millis(duration_ms);
    let mut samples = Vec::new();
    while Instant::now() < deadline {
        if let Some(w) = read_ina3221_watts() {
            samples.push(w);
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    if samples.is_empty() {
        None
    } else {
        Some(samples.iter().sum::<f64>() / samples.len() as f64)
    }
}

struct PowerSampler {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<Vec<f64>>>,
}

impl PowerSampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let handle = if read_ina3221_watts().is_some() {
            let s = stop.clone();
            Some(std::thread::spawn(move || {
                let mut samples = Vec::new();
                while !s.load(Ordering::Relaxed) {
                    if let Some(w) = read_ina3221_watts() {
                        samples.push(w);
                    }
                    std::thread::sleep(Duration::from_millis(250));
                }
                samples
            }))
        } else {
            None
        };
        Self { stop, handle }
    }

    fn finish(mut self) -> Option<f64> {
        self.stop.store(true, Ordering::Relaxed);
        let samples = self.handle.take()?.join().ok()?;
        if samples.is_empty() {
            None
        } else {
            Some(samples.iter().sum::<f64>() / samples.len() as f64)
        }
    }
}

fn current_git_rev() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()?;
    if out.status.success() {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            return Some(s);
        }
    }
    None
}

fn median_f64(mut vals: Vec<f64>) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        vals[mid]
    } else {
        0.5 * (vals[mid - 1] + vals[mid])
    }
}

fn min_f64(vals: &[f64]) -> f64 {
    vals.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max_f64(vals: &[f64]) -> f64 {
    vals.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

struct RepMetrics {
    ttft_s: f64,
    decode_s: f64,
    total_s: f64,
    prompt_tokens: usize,
    completion_tokens: usize,
    decode_tps: f64,
    total_tps: f64,
    sweeps: usize,
    drafted_tokens: usize,
    accepted_tokens: usize,
    io_mib_read: f64,
    pread_mib_s: f64,
    effective_io_mib_s: f64,
    cache_hits: u64,
    cache_misses: u64,
    cache_mib_saved: f64,
    avg_watts: Option<f64>,
    mem_avail_before: f64,
    mem_avail_after: f64,
    swap_before: f64,
    swap_after: f64,
    page_cache_delta: f64,
    output_preview: String,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path: Option<PathBuf> = None;
    let mut draft_path: Option<PathBuf> = None;
    let mut mode = "stream_r0".to_string();
    let mut resident_layers: Option<u32> = Some(0);
    let mut vram_budget_mb: u64 = 0;
    let mut vram_budget_auto = false;
    let mut use_gpu = true;
    let mut enable_ngram = false;
    let mut max_draft_tokens: usize = 5;
    let mut n_ctx: u32 = 2048;
    let mut n_batch: u32 = 512;
    let mut n_threads: Option<u32> = None;
    let mut kv_cache_type = KvCacheType::F16;
    let mut kv_cache_type_str = "f16".to_string();
    let mut temperature: f32 = 0.0;
    let mut seed: u32 = 42;
    let mut cpu_embeddings = false;
    let mut cpu_head = false;
    let mut stream_head = false;
    let mut moe_active_streaming = true;
    let mut multi_turn = false;
    let mut reference = false;
    let mut apply_chat = false;
    let mut warmup = false;
    let mut reps: usize = 1;
    let mut keep_expert_cache = false;
    let mut workload_filter: Option<String> = None;
    let mut max_tokens_override: Option<usize> = None;

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
                if args[i].eq_ignore_ascii_case("auto") {
                    vram_budget_auto = true;
                    vram_budget_mb = 0;
                } else {
                    vram_budget_auto = false;
                    vram_budget_mb = args[i].parse().context("parse --vram-budget-mb")?;
                }
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
            "--threads" => {
                i += 1;
                n_threads = Some(args[i].parse().context("parse --threads")?);
            }
            "--kv-cache-type" => {
                i += 1;
                kv_cache_type_str = args[i].clone();
                kv_cache_type = match args[i].as_str() {
                    "f16" => KvCacheType::F16,
                    "q8_0" => KvCacheType::Q8_0,
                    "q4_0" => KvCacheType::Q4_0,
                    other => bail!("unsupported --kv-cache-type: {other} (use f16|q8_0|q4_0)"),
                };
            }
            "--temperature" => {
                i += 1;
                temperature = args[i].parse().context("parse --temperature")?;
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().context("parse --seed")?;
            }
            "--cpu-embeddings" => cpu_embeddings = true,
            "--cpu-head" => cpu_head = true,
            "--stream-head" => stream_head = true,
            "--no-moe-active-streaming" => moe_active_streaming = false,
            "--multi-turn" => multi_turn = true,
            "--reference" => reference = true,
            "--chat" => apply_chat = true,
            "--warmup" => warmup = true,
            "--reps" => {
                i += 1;
                reps = args[i].parse::<usize>().context("parse --reps")?.max(1);
            }
            "--keep-expert-cache" => keep_expert_cache = true,
            "--workload" => {
                i += 1;
                workload_filter = Some(args[i].clone());
            }
            "--max-tokens" => {
                i += 1;
                max_tokens_override = Some(args[i].parse().context("parse --max-tokens")?);
            }
            other => bail!("unknown argument: {other}"),
        }
        i += 1;
    }

    let model_path = model_path.context("--model <path.gguf> is required")?;
    let git_rev = current_git_rev();
    let idle_watts = sample_idle_watts(1000);
    let mem_avail_start = mem_avail_mib();
    let cache_before_load = page_cache_mib();
    let t_load = Instant::now();

    let mut cfg = if reference {
        EngineConfig::reference(&model_path)
    } else {
        EngineConfig::streaming(&model_path)
    };
    cfg.n_ctx = n_ctx;
    cfg.n_batch = n_batch;
    if let Some(t) = n_threads {
        cfg.n_threads = t;
    }
    let resolved_threads = cfg.n_threads;
    cfg.kv_cache_type = kv_cache_type;
    cfg.use_gpu = use_gpu;
    cfg.cpu_embeddings = cpu_embeddings;
    cfg.cpu_head = cpu_head;
    cfg.stream_head = stream_head;
    cfg.moe_active_streaming = moe_active_streaming;
    cfg.resident_layers = resident_layers;
    cfg.vram_budget_auto = vram_budget_auto;
    let draft_reserve = draft_path.as_ref().map_or(0, |dp| {
        let draft_bytes = fs::metadata(dp).map(|m| m.len()).unwrap_or(0);
        draft_bytes.saturating_add(128 * 1024 * 1024)
    });
    cfg.vram_budget_bytes = if vram_budget_auto {
        draft_reserve
    } else if vram_budget_mb > 0 {
        vram_budget_mb
            .saturating_mul(1024 * 1024)
            .saturating_sub(draft_reserve)
    } else {
        0
    };

    let mut engine = StreamEngine::open(&cfg)?;
    let mut draft = match &draft_path {
        Some(dp) => {
            let mut dcfg = EngineConfig::reference(dp);
            dcfg.n_ctx = n_ctx;
            dcfg.n_batch = n_batch;
            dcfg.kv_cache_type = kv_cache_type;
            dcfg.use_gpu = use_gpu;
            Some(DraftModelProposer::open(dcfg, max_draft_tokens)?)
        }
        None => None,
    };
    let load_time_s = t_load.elapsed().as_secs_f64();
    let mem_avail_loaded = mem_avail_mib();
    let stats = engine.stats();

    let mut results = Vec::new();
    for wl in WORKLOADS {
        if let Some(ref f) = workload_filter
            && wl.id != f
        {
            continue;
        }

        let prompt_text = if apply_chat {
            engine
                .apply_chat_template(
                    &[ChatMessage {
                        role: "user".to_string(),
                        content: wl.prompt.to_string(),
                    }],
                    true,
                )
                .unwrap_or_else(|_| wl.prompt.to_string())
        } else {
            wl.prompt.to_string()
        };

        let mut opts = GenerateOptions {
            max_tokens: max_tokens_override.unwrap_or(wl.max_tokens),
            enable_ngram_speculation: enable_ngram,
            max_draft_tokens,
            ..Default::default()
        };
        opts.sample.temperature = temperature;
        opts.sample.seed = seed;

        if warmup {
            engine.kv_clear();
            if let Some(ref mut d) = draft {
                d.reset();
            }
            let _ = engine.generate_stream(&prompt_text, &opts, draft.as_mut(), |_, _| true)?;
        }

        let mut rep_runs: Vec<RepMetrics> = Vec::with_capacity(reps);
        for rep_idx in 0..reps {
            engine.kv_clear();
            if let Some(ref mut d) = draft {
                d.reset();
            }
            if !keep_expert_cache && !warmup {
                engine.clear_expert_cache();
            }
            if temperature > 0.0 {
                opts.sample.seed = seed.wrapping_add(rep_idx as u32);
            }

            let cache_before = page_cache_mib();
            let avail_before = mem_avail_mib();
            let swap_before = swap_used_mib();
            let io_before = engine.io_stats();
            let power_sampler = PowerSampler::start();
            let t0 = Instant::now();
            let mut ttft_s: Option<f64> = None;
            let mut text = String::new();

            let summary =
                engine.generate_stream(&prompt_text, &opts, draft.as_mut(), |_tok, piece| {
                    if ttft_s.is_none() {
                        ttft_s = Some(t0.elapsed().as_secs_f64());
                    }
                    text.push_str(piece);
                    true
                })?;

            let total_s = t0.elapsed().as_secs_f64();
            let avg_watts = power_sampler.finish();
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
            let pread_us_delta = io_after.pread_us.saturating_sub(io_before.pread_us);
            let cache_hits = io_after.cache_hits.saturating_sub(io_before.cache_hits);
            let cache_misses = io_after.cache_misses.saturating_sub(io_before.cache_misses);
            let cache_bytes_saved = io_after
                .cache_bytes_served
                .saturating_sub(io_before.cache_bytes_served);
            let io_mib = io_bytes_delta as f64 / (1024.0 * 1024.0);
            let pread_mib_s = if pread_us_delta > 0 {
                io_mib / (pread_us_delta as f64 / 1_000_000.0)
            } else {
                0.0
            };
            let effective_io_mib_s = io_mib / total_s.max(1e-6);
            let cache_after = page_cache_mib();

            rep_runs.push(RepMetrics {
                ttft_s: ttft,
                decode_s,
                total_s,
                prompt_tokens: summary.prompt_tokens,
                completion_tokens: summary.completion_tokens,
                decode_tps,
                total_tps,
                sweeps: summary.speculation.sweeps,
                drafted_tokens: summary.speculation.drafted_tokens,
                accepted_tokens: summary.speculation.accepted_tokens,
                io_mib_read: io_mib,
                pread_mib_s,
                effective_io_mib_s,
                cache_hits,
                cache_misses,
                cache_mib_saved: cache_bytes_saved as f64 / (1024.0 * 1024.0),
                avg_watts,
                mem_avail_before: avail_before,
                mem_avail_after: mem_avail_mib(),
                swap_before,
                swap_after: swap_used_mib(),
                page_cache_delta: cache_after - cache_before,
                output_preview: text.chars().take(160).collect::<String>(),
            });
        }

        let last = rep_runs.last().expect("at least one rep");
        let stats_after = engine.stats();
        let io_final = engine.io_stats();
        let cache_after_wl = page_cache_mib();

        let decode_tps_vals: Vec<f64> = rep_runs.iter().map(|r| r.decode_tps).collect();
        let total_tps_vals: Vec<f64> = rep_runs.iter().map(|r| r.total_tps).collect();
        let ttft_vals: Vec<f64> = rep_runs.iter().map(|r| r.ttft_s).collect();
        let total_s_vals: Vec<f64> = rep_runs.iter().map(|r| r.total_s).collect();
        let decode_s_vals: Vec<f64> = rep_runs.iter().map(|r| r.decode_s).collect();
        let pread_vals: Vec<f64> = rep_runs.iter().map(|r| r.pread_mib_s).collect();
        let eff_io_vals: Vec<f64> = rep_runs.iter().map(|r| r.effective_io_mib_s).collect();
        let watts_vals: Vec<f64> = rep_runs.iter().filter_map(|r| r.avg_watts).collect();

        let med_decode_tps = median_f64(decode_tps_vals.clone());
        let med_total_tps = median_f64(total_tps_vals.clone());
        let med_ttft = median_f64(ttft_vals.clone());
        let med_total_s = median_f64(total_s_vals);
        let med_decode_s = median_f64(decode_s_vals);
        let med_pread_mib_s = median_f64(pread_vals);
        let med_eff_io_mib_s = median_f64(eff_io_vals);
        let med_watts = if watts_vals.is_empty() {
            None
        } else {
            Some(median_f64(watts_vals))
        };
        let net_watts = med_watts.and_then(|w| idle_watts.map(|iw| (w - iw).max(0.0)));
        let joules_per_tok =
            med_watts.map(|w| (w * med_total_s) / last.completion_tokens.max(1) as f64);
        let net_joules_per_tok =
            net_watts.map(|nw| (nw * med_total_s) / last.completion_tokens.max(1) as f64);
        let cache_total = last.cache_hits + last.cache_misses;
        let cache_hit_rate = if cache_total > 0 {
            Some(round3(last.cache_hits as f64 / cache_total as f64))
        } else {
            None
        };

        let record = serde_json::json!({
            "engine": "layerstream-plugin",
            "engine_version": env!("CARGO_PKG_VERSION"),
            "git_rev": git_rev,
            "mode": mode,
            "model": model_path.file_name().and_then(|s| s.to_str()).unwrap_or("unknown"),
            "draft_model": draft_path.as_ref().and_then(|p| p.file_name()).and_then(|s| s.to_str()),
            "workload": wl.id,
            "chat_template_applied": apply_chat,
            "warmup": warmup,
            "reps": reps,
            "temperature": round3(temperature as f64),
            "seed": seed,
            "threads": resolved_threads,
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "kv_cache_type": kv_cache_type_str,
            "vram_budget_auto": vram_budget_auto,
            "vram_budget_mb": vram_budget_mb,
            "load_time_s": round3(load_time_s),
            "ttft_s": round3(med_ttft),
            "ttft_s_min": round3(min_f64(&ttft_vals)),
            "ttft_s_max": round3(max_f64(&ttft_vals)),
            "decode_time_s": round3(med_decode_s),
            "total_time_s": round3(med_total_s),
            "prompt_tokens": last.prompt_tokens,
            "completion_tokens": last.completion_tokens,
            "decode_tok_per_s": round3(med_decode_tps),
            "decode_tok_per_s_min": round3(min_f64(&decode_tps_vals)),
            "decode_tok_per_s_max": round3(max_f64(&decode_tps_vals)),
            "total_tok_per_s": round3(med_total_tps),
            "total_tok_per_s_min": round3(min_f64(&total_tps_vals)),
            "total_tok_per_s_max": round3(max_f64(&total_tps_vals)),
            "sweeps": last.sweeps,
            "drafted_tokens": last.drafted_tokens,
            "accepted_tokens": last.accepted_tokens,
            "io_mib_read": round2(last.io_mib_read),
            "pread_mib_s": round1(med_pread_mib_s),
            "effective_io_mib_s": round1(med_eff_io_mib_s),
            "expert_cache_hits": last.cache_hits,
            "expert_cache_misses": last.cache_misses,
            "expert_cache_hit_rate": cache_hit_rate,
            "expert_cache_mib_saved": round2(last.cache_mib_saved),
            "direct_io_active": io_final.direct_io_active,
            "peak_rss_mib": round1(peak_rss_mib()),
            "current_rss_mib": round1(current_rss_mib()),
            "mem_avail_start_mib": round1(mem_avail_start),
            "mem_avail_loaded_mib": round1(mem_avail_loaded),
            "mem_avail_before_mib": round1(last.mem_avail_before),
            "mem_avail_after_mib": round1(last.mem_avail_after),
            "swap_delta_mib": round1(last.swap_after - last.swap_before),
            "idle_watts": idle_watts.map(round2),
            "avg_watts": med_watts.map(round2),
            "net_watts": net_watts.map(round2),
            "joules_per_tok": joules_per_tok.map(round2),
            "net_joules_per_tok": net_joules_per_tok.map(round2),
            "page_cache_delta_mib": round1(last.page_cache_delta),
            "page_cache_delta_since_start_mib": round1(cache_after_wl - cache_before_load),
            "n_layers": stats.n_layers,
            "resident_layers": stats.resident_layers,
            "streamed_layers": stats.streamed_layers,
            "resident_mib": round1(stats.resident_bytes as f64 / (1024.0 * 1024.0)),
            "cpu_resident_mib": round1(stats.cpu_resident_bytes as f64 / (1024.0 * 1024.0)),
            "slot_mib": round1(stats.slot_bytes as f64 / (1024.0 * 1024.0)),
            "expert_slot_mib": round1(stats.expert_slot_bytes as f64 / (1024.0 * 1024.0)),
            "head_streamed": stats.head_streamed,
            "moe_layers": stats.moe_layers,
            "moe_active_streaming": stats.moe_active_streaming,
            "moe_experts_loaded": stats_after.moe_experts_loaded,
            "moe_experts_total": stats_after.moe_experts_total,
            "uma_zero_copy": stats.is_uma_zero_copy,
            "is_unified_memory": stats.is_unified_memory,
            "using_gpu": stats.using_gpu,
            "can_speculate": stats.can_speculate,
            "output_preview": last.output_preview,
        });
        results.push(record);
    }

    if multi_turn {
        engine.kv_clear();
        if let Some(ref mut d) = draft {
            d.reset();
        }
        let base_prompt = format!(
            "{0}\n\nAdditional Context Section A:\n{0}\n\nAdditional Context Section B:\n{0}\n\nAdditional Context Section C:\n{0}\n\nSummary:",
            WORKLOADS[0].prompt
        );
        let mut turn1_text = String::new();
        let mut opts1 = GenerateOptions {
            max_tokens: 16,
            enable_ngram_speculation: enable_ngram,
            max_draft_tokens,
            ..Default::default()
        };
        opts1.sample.temperature = temperature;
        opts1.sample.seed = seed;
        let t1 = Instant::now();
        let mut ttft1: Option<f64> = None;
        let s1 = engine.generate_stream(&base_prompt, &opts1, draft.as_mut(), |_, piece| {
            if ttft1.is_none() {
                ttft1 = Some(t1.elapsed().as_secs_f64());
            }
            turn1_text.push_str(piece);
            true
        })?;
        let total1 = t1.elapsed().as_secs_f64();

        let followup = format!(
            "{base_prompt}{turn1_text}\n\nFollow-up: In one short sentence, why does O_DIRECT prevent page-cache thrashing?\nAnswer:"
        );
        let mut turn2_text = String::new();
        let t2 = Instant::now();
        let mut ttft2: Option<f64> = None;
        let s2 = engine.generate_stream(&followup, &opts1, draft.as_mut(), |_, piece| {
            if ttft2.is_none() {
                ttft2 = Some(t2.elapsed().as_secs_f64());
            }
            turn2_text.push_str(piece);
            true
        })?;
        let total2 = t2.elapsed().as_secs_f64();

        results.push(serde_json::json!({
            "engine": "layerstream-plugin",
            "mode": format!("{mode}_multiturn"),
            "model": model_path.file_name().and_then(|s| s.to_str()).unwrap_or("unknown"),
            "workload": "multi_turn_prefix_reuse",
            "turn1_prompt_tokens": s1.prompt_tokens,
            "turn1_ttft_s": round3(ttft1.unwrap_or(total1)),
            "turn1_total_s": round3(total1),
            "turn2_prompt_tokens": s2.prompt_tokens,
            "turn2_ttft_s": round3(ttft2.unwrap_or(total2)),
            "turn2_total_s": round3(total2),
            "ttft_speedup_x": round2(ttft1.unwrap_or(total1) / ttft2.unwrap_or(total2).max(1e-6)),
            "turn2_output_preview": turn2_text.chars().take(160).collect::<String>(),
        }));
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
