use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::Args;
use layerstream_server::{ModelRegistry, ServerConfig};

#[derive(Debug, Clone, Args)]
pub struct EngineArgs {
    /// GGUF model path (or `alias=/path/to/model.gguf`). Repeatable.
    #[arg(long = "model", value_name = "SPEC")]
    pub models: Vec<String>,

    /// Directory containing `.gguf` models to advertise lazily (0 MB VRAM until requested).
    #[arg(long, env = "LAYERSTREAM_MODELS_DIR")]
    pub models_dir: Option<PathBuf>,

    /// Optional tiny resident draft GGUF for speculative multi-token verification.
    #[arg(long, env = "LAYERSTREAM_DRAFT_MODEL")]
    pub draft_model: Option<PathBuf>,

    /// Context window size in tokens.
    #[arg(long, default_value_t = 4096)]
    pub ctx_size: u32,

    /// Prompt prefill batch chunk size (layer-major prefill runs 1 NVMe sweep per chunk).
    #[arg(long, default_value_t = 512)]
    pub batch_size: u32,

    /// CPU thread count.
    #[arg(long)]
    pub threads: Option<u32>,

    /// Pin the first N layers (`blk.0..N-1`) in resident memory; stream the rest.
    #[arg(long)]
    pub resident_layers: Option<u32>,

    /// Optional VRAM budget in MiB for automatic hybrid partial-residency (0 = minimum VRAM).
    #[arg(long, default_value_t = 0)]
    pub vram_budget_mb: u64,

    /// Seconds of idle time before freeing 100% of GPU memory (0 = free immediately after each request).
    #[arg(long, default_value_t = 60)]
    pub idle_timeout_secs: u64,

    /// Disable GPU offloading and stream strictly on CPU.
    #[arg(long)]
    pub no_gpu: bool,

    /// Disable Linux O_DIRECT and use buffered pread.
    #[arg(long)]
    pub no_direct_io: bool,

    /// Disable zero-VRAM n-gram speculative verification.
    #[arg(long)]
    pub no_ngram: bool,

    /// Maximum draft tokens proposed per NVMe verification sweep.
    #[arg(long, default_value_t = 5)]
    pub max_draft_tokens: usize,
}

impl EngineArgs {
    pub fn into_server_config(self) -> Result<ServerConfig> {
        let registry = ModelRegistry::from_sources(
            &self.models,
            self.models_dir.as_deref(),
            self.draft_model,
        )?;

        let n_threads = self.threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get().min(8) as u32)
                .unwrap_or(4)
        });

        Ok(ServerConfig {
            registry,
            n_ctx: self.ctx_size,
            n_batch: self.batch_size,
            n_threads,
            resident_layers: self.resident_layers,
            vram_budget_bytes: self.vram_budget_mb.saturating_mul(1024 * 1024),
            use_gpu: !self.no_gpu,
            prefer_direct_io: !self.no_direct_io,
            idle_timeout: Duration::from_secs(self.idle_timeout_secs),
            enable_ngram_speculation: !self.no_ngram,
            max_draft_tokens: self.max_draft_tokens,
            max_queued_requests: 8,
        })
    }
}
