use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, bail};
use clap::Args;
use layerstream_core::{KvCacheType, StreamEngine, hf, probe_storage_mib_per_s};
use layerstream_server::{ModelRegistry, ServerConfig};

const MIN_STREAMING_STORAGE_MIB_S: f64 = 150.0;
const OVERSIZE_GUARD_BYTES: u64 = 1024 * 1024 * 1024; // 1 GiB

#[derive(Debug, Clone, Args)]
pub struct EngineArgs {
    /// GGUF model path or `hf://org/repo:file.gguf` (optionally prefixed with `alias=`). Repeatable.
    #[arg(long = "model", value_name = "SPEC")]
    pub models: Vec<String>,

    /// Directory containing `.gguf` models to advertise lazily (0 MB model/KV weights idle).
    #[arg(long, env = "LAYERSTREAM_MODELS_DIR")]
    pub models_dir: Option<PathBuf>,

    /// Optional tiny resident draft GGUF (local path or `hf://org/repo:file.gguf`) for speculative verification.
    #[arg(long, env = "LAYERSTREAM_DRAFT_MODEL", value_name = "PATH_OR_HF")]
    pub draft_model: Option<String>,

    /// Local cache directory for `hf://` model pulls (defaults to `$XDG_CACHE_HOME/layerstream/models`).
    #[arg(long, env = "LAYERSTREAM_CACHE_DIR")]
    pub cache_dir: Option<PathBuf>,

    /// Context window size in tokens.
    #[arg(long, default_value_t = 4096)]
    pub ctx_size: u32,

    /// VRAM budget in MiB for automatic hybrid partial-residency (`auto` probes free GPU memory safely, `0` = minimum VRAM).
    #[arg(long, default_value = "auto", value_name = "AUTO_OR_MB")]
    pub vram_budget_mb: String,

    /// KV cache quantization type: f16, q8_0 (2x smaller, recommended), or q4_0 (4x smaller).
    #[arg(long, default_value = "q8_0")]
    pub kv_cache_type: String,

    /// Seconds of idle time before freeing model/KV GPU buffers and terminating the GPU worker
    /// (default: 60; 0 = free immediately after each request so host runtimes reclaim 100% of VRAM/UMA).
    #[arg(long)]
    pub idle_timeout_secs: Option<u64>,

    /// Run GPU inference in-process instead of an isolated on-demand worker subprocess
    /// (retains GPU driver/CUDA context memory after idle eviction).
    #[arg(long)]
    pub in_process: bool,

    /// Allow streaming oversize models even on slow storage (<150 MiB/s) or non-O_DIRECT mounts (for batch jobs).
    #[arg(long)]
    pub allow_slow: bool,

    // Hidden internal/CI flags kept out of normal `--help` to prevent footguns.
    #[arg(long, default_value_t = 512, hide = true)]
    pub batch_size: u32,

    #[arg(long, hide = true)]
    pub threads: Option<u32>,

    #[arg(long, hide = true)]
    pub no_gpu: bool,

    #[arg(long, hide = true)]
    pub low_vram: bool,

    #[arg(long, default_value_t = 8, hide = true)]
    pub max_draft_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct DeferredPullPlan {
    pub models: Vec<String>,
    pub models_dir: Option<PathBuf>,
    pub draft_model: Option<String>,
    pub cache_root: PathBuf,
    pub use_gpu: bool,
    pub allow_slow: bool,
}

impl DeferredPullPlan {
    pub fn execute(&self) -> Result<ModelRegistry> {
        let mut resolved_models = Vec::with_capacity(self.models.len());
        for spec in &self.models {
            resolved_models.push(hf::resolve_model_spec(spec, &self.cache_root)?);
        }
        let resolved_draft = match self.draft_model.as_deref() {
            Some(spec) => Some(hf::resolve_path_or_hf(spec, &self.cache_root)?),
            None => None,
        };
        let registry = ModelRegistry::from_sources(
            &resolved_models,
            self.models_dir.as_deref(),
            resolved_draft,
        )?;
        run_preflight_checks(&registry, self.use_gpu, self.allow_slow, false)?;
        Ok(registry)
    }
}

impl EngineArgs {
    /// Returns true if any configured `--model` or `--draft-model` requires a network `hf://` pull.
    pub fn has_uncached_hf(&self) -> bool {
        let cache_root = self.cache_dir.clone().unwrap_or_else(hf::default_cache_dir);
        if self
            .models
            .iter()
            .any(|spec| !hf::is_model_spec_cached(spec, &cache_root))
        {
            return true;
        }
        if let Some(draft) = self.draft_model.as_deref()
            && !hf::is_path_or_hf_cached(draft, &cache_root)
        {
            return true;
        }
        false
    }

    /// Returns true if at least one model source (`--model` or `--models-dir`) was configured.
    pub fn has_model_sources(&self) -> bool {
        !self.models.is_empty() || self.models_dir.is_some()
    }

    pub fn into_server_config(self) -> Result<ServerConfig> {
        let (mut cfg, deferred) = self.into_server_config_with_deferred(false, None)?;
        if let Some(plan) = deferred {
            cfg.registry = plan.execute()?;
        }
        Ok(cfg)
    }

    /// Builds a `ServerConfig`; when `defer_uncached_hf == true` (Mesh-LLM plugin mode),
    /// uncached `hf://` pulls are deferred to a background task so the plugin connects to
    /// `MESH_LLM_PLUGIN_ENDPOINT` immediately without tripping the host's 10 s connect timeout.
    ///
    /// Note: intentionally does NOT probe GPU drivers at startup so the coordinator process stays
    /// at ~10 MiB RSS / 0 B GPU driver context while idle. Explicit GPU checks happen in `check`
    /// / `recommend` and lazily inside the GPU worker on first inference request.
    pub fn into_server_config_with_deferred(
        self,
        defer_uncached_hf: bool,
        managed_idle_timeout_secs: Option<u64>,
    ) -> Result<(ServerConfig, Option<DeferredPullPlan>)> {
        let use_gpu = !self.no_gpu;

        let cache_root = self.cache_dir.clone().unwrap_or_else(hf::default_cache_dir);

        let n_threads = self.threads.unwrap_or_else(|| {
            if use_gpu {
                // Keep CPU cores free for kernel NVMe/dm-crypt BIOs and HTTP serving when GPU executes layers.
                2
            } else {
                std::thread::available_parallelism()
                    .map(|n| n.get().min(8) as u32)
                    .unwrap_or(4)
            }
        });

        let kv_cache_type: KvCacheType = self
            .kv_cache_type
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid --kv-cache-type: {e}"))?;

        let (vram_budget_auto, vram_budget_bytes) =
            if self.vram_budget_mb.eq_ignore_ascii_case("auto") {
                (true, 0u64)
            } else {
                let mb: u64 = self.vram_budget_mb.parse().map_err(|_| {
                anyhow::anyhow!(
                    "invalid --vram-budget-mb '{}' (expected 'auto' or non-negative integer MiB)",
                    self.vram_budget_mb
                )
            })?;
                (false, mb.saturating_mul(1024 * 1024))
            };

        let effective_idle_timeout_secs = self
            .idle_timeout_secs
            .or(managed_idle_timeout_secs)
            .unwrap_or(60);

        let needs_defer = defer_uncached_hf && self.has_uncached_hf();
        let plan = DeferredPullPlan {
            models: self.models,
            models_dir: self.models_dir,
            draft_model: self.draft_model,
            cache_root,
            use_gpu,
            allow_slow: self.allow_slow,
        };
        let (registry, deferred) = if needs_defer {
            (ModelRegistry::default(), Some(plan))
        } else {
            (plan.execute()?, None)
        };

        Ok((
            ServerConfig {
                registry,
                n_ctx: self.ctx_size,
                n_batch: self.batch_size,
                n_threads,
                resident_layers: None,
                vram_budget_bytes,
                vram_budget_auto,
                kv_cache_type,
                use_gpu,
                prefer_direct_io: true,
                cpu_embeddings: self.low_vram,
                cpu_head: false,
                stream_head: self.low_vram,
                moe_active_streaming: true,
                idle_timeout: Duration::from_secs(effective_idle_timeout_secs),
                enable_ngram_speculation: true,
                max_draft_tokens: self.max_draft_tokens,
                max_queued_requests: 8,
            },
            deferred,
        ))
    }
}

/// Validates storage throughput, GPU memory, and draft pairings at startup and logs a clear
/// performance verdict so users know up front whether a setup will be fast or slow (and why).
pub fn run_preflight_checks(
    registry: &ModelRegistry,
    use_gpu: bool,
    allow_slow: bool,
    print_report: bool,
) -> Result<()> {
    // Only initialize the GPU driver during explicit `check` reports; in `serve` / `plugin`
    // startup keep cold GPU memory strictly at 0 MB (no ~325 MiB CUDA primary context pinned
    // while idle) so resident runtimes like Mesh-LLM Skippy have full headroom.
    let gpu_mem = if print_report {
        StreamEngine::probe_gpu_memory(use_gpu)
    } else {
        None
    };
    if print_report {
        match gpu_mem {
            Some((free_b, total_b)) => {
                println!(
                    "GPU memory: {} MiB free / {} MiB total",
                    free_b / (1024 * 1024),
                    total_b / (1024 * 1024)
                );
            }
            None if use_gpu => {
                bail!("no usable GPU device found (CUDA/Vulkan/Metal)");
            }
            None => {
                println!("GPU memory: disabled (--no-gpu test mode)");
            }
        }
    }

    for entry in registry.list() {
        let size_bytes = entry.meta.total_file_bytes;
        let size_gib = size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let fits_in_vram = gpu_mem
            .map(|(free_b, _)| size_bytes.saturating_add(768 * 1024 * 1024) < free_b)
            .unwrap_or(false);

        let probe_res = probe_storage_mib_per_s(&entry.path, true).ok();
        let (mib_s, direct_ok) = probe_res.unwrap_or((0.0, false));
        let is_oversize = size_bytes >= OVERSIZE_GUARD_BYTES && !fits_in_vram;

        if is_oversize
            && use_gpu
            && probe_res.is_some()
            && mib_s > 0.0
            && mib_s < MIN_STREAMING_STORAGE_MIB_S
        {
            let msg = format!(
                "storage for '{}' ({size_gib:.1} GiB) measured only {mib_s:.0} MiB/s \
                 (minimum recommended {MIN_STREAMING_STORAGE_MIB_S:.0} MiB/s NVMe; \
                 estimated decode would be <0.02 tok/s). Move the GGUF to local NVMe \
                 or pass --allow-slow for offline batch use.",
                entry.id
            );
            if allow_slow {
                tracing::warn!("{msg}");
            } else {
                bail!("{msg}");
            }
        }

        if is_oversize && !direct_ok {
            tracing::warn!(
                model = %entry.id,
                path = %entry.path.display(),
                "O_DIRECT/F_NOCACHE unavailable on this mount (e.g. tmpfs/FUSE/NFS); OS page cache may grow during streaming"
            );
        }

        let est_stream_mib = (size_bytes as f64 / (1024.0 * 1024.0)).saturating_sub_f64(
            gpu_mem.map_or(0.0, |(f, _)| (f as f64 * 0.45) / (1024.0 * 1024.0)),
        );
        let base_tps = if est_stream_mib > 1.0 && mib_s > 1.0 {
            mib_s / est_stream_mib
        } else {
            0.0
        };

        let verdict = if fits_in_vram {
            format!(
                "GOOD (100% resident in VRAM, {size_gib:.2} GiB — no SSD streaming needed; native mesh-llm/Skippy also works directly)"
            )
        } else if entry.draft_path.is_some() {
            format!(
                "GOOD ({size_gib:.2} GiB streamed + draft speculator, storage={mib_s:.0} MiB/s, direct_io={direct_ok}, est ~{:.2}-{:.2} tok/s)",
                base_tps * 2.5,
                base_tps * 5.0
            )
        } else if size_bytes >= 4 * 1024 * 1024 * 1024 {
            format!(
                "SLOW ({size_gib:.2} GiB streamed without --draft-model, storage={mib_s:.0} MiB/s, est ~{base_tps:.2} tok/s; add a small same-family --draft-model for 3-5x speedup or use a sparse MoE GGUF)"
            )
        } else {
            format!("OK ({size_gib:.2} GiB, storage={mib_s:.0} MiB/s, direct_io={direct_ok})")
        };

        if print_report {
            println!("Model '{}': {verdict}", entry.id);
        } else if verdict.starts_with("SLOW") {
            tracing::warn!(model = %entry.id, "{verdict}");
        } else {
            tracing::info!(model = %entry.id, "{verdict}");
        }
    }
    Ok(())
}

trait SaturatingSubF64 {
    fn saturating_sub_f64(self, rhs: f64) -> f64;
}

impl SaturatingSubF64 for f64 {
    fn saturating_sub_f64(self, rhs: f64) -> f64 {
        (self - rhs).max(0.0)
    }
}
