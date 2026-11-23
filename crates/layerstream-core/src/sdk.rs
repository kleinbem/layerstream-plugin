//! Shared high-level SDK facade (`LoadedSession`, `inspect_gguf_file`, `pull_model`,
//! `recommend_repo`, `check_model`) eliminating duplicated business logic across the
//! Python, Node.js, and CLI bindings.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use serde::Serialize;

use crate::engine::{EngineConfig, GenerateSummary, KvCacheType, StreamEngine};
use crate::gguf_index::{GgufIndex, check_gguf_draft_compatible};
use crate::hf::{
    self, PretrainedResolveOptions, ProgressCallback, QuantRecommendation, resolve_pretrained_model,
};
use crate::io_reader::{IoStats, probe_dir_storage_mib_per_s, probe_storage_mib_per_s};
use crate::speculator::DraftModelProposer;

pub const MIN_STREAMING_STORAGE_MIB_S: f64 = 150.0;
pub const OVERSIZE_GUARD_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Clone, Default)]
pub struct OpenSessionOptions {
    pub draft_model: Option<String>,
    pub quant: Option<String>,
    pub revision: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub vram_budget_auto: bool,
    pub vram_budget_bytes: u64,
    pub resident_layers: Option<u32>,
    pub ctx_size: u32,
    pub batch_size: u32,
    pub threads: Option<u32>,
    pub kv_cache_type: KvCacheType,
    pub use_gpu: bool,
    pub prefer_direct_io: bool,
    pub cpu_embeddings: bool,
    pub cpu_head: bool,
    pub stream_head: bool,
    pub moe_active_streaming: bool,
    pub max_draft_tokens: usize,
    pub on_progress: Option<ProgressCallback>,
}

pub struct LoadedSession {
    pub engine: StreamEngine,
    pub draft: Option<DraftModelProposer>,
    pub model_path: PathBuf,
    pub draft_model_path: Option<PathBuf>,
    pub source_uri: Option<String>,
    pub resolved_repo: Option<String>,
    pub selected_quant: Option<String>,
    pub default_max_draft_tokens: usize,
    pub last_summary: Option<GenerateSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EngineStatsSnapshot {
    pub loaded: bool,
    pub isolated_worker: bool,
    pub backend_name: String,
    pub cached_kv_tokens: usize,
    pub model_path: String,
    pub draft_model_path: Option<String>,
    pub source_uri: Option<String>,
    pub resolved_repo: Option<String>,
    pub selected_quant: Option<String>,
    pub n_ctx: u32,
    pub vocab_size: i32,
    pub n_layers: u32,
    pub resident_layers: u32,
    pub streamed_layers: u32,
    pub resident_bytes: u64,
    pub cpu_resident_bytes: u64,
    pub slot_bytes: u64,
    pub expert_slot_bytes: u64,
    pub total_layer_loads: u64,
    pub moe_layers: u32,
    pub moe_experts_loaded: u64,
    pub moe_experts_total: u64,
    pub head_streamed: bool,
    pub moe_active_streaming: bool,
    pub is_uma_zero_copy: bool,
    pub is_unified_memory: bool,
    pub using_gpu: bool,
    pub can_speculate: bool,
    pub io: IoStats,
}

impl LoadedSession {
    pub fn open(model: &str, opts: OpenSessionOptions) -> Result<Self> {
        let ctx_size = if opts.ctx_size > 0 {
            opts.ctx_size
        } else {
            4096
        };
        let batch_size = if opts.batch_size > 0 {
            opts.batch_size
        } else {
            512
        };
        let max_draft_tokens = opts.max_draft_tokens.max(1);
        let n_threads = opts.threads.unwrap_or_else(|| {
            if opts.use_gpu {
                2
            } else {
                std::thread::available_parallelism()
                    .map(|n| n.get().min(8) as u32)
                    .unwrap_or(4)
            }
        });
        let vram_opt = if opts.vram_budget_bytes > 0 {
            Some(opts.vram_budget_bytes)
        } else {
            None
        };

        let target_resolve = PretrainedResolveOptions {
            revision: opts.revision.clone(),
            quant: opts.quant,
            cache_dir: opts.cache_dir.clone(),
            vram_budget_bytes: vram_opt,
            use_gpu: opts.use_gpu,
            has_draft: opts.draft_model.is_some(),
            on_progress: opts.on_progress.clone(),
        };
        let resolved_target = resolve_pretrained_model(model, &target_resolve)?;

        let is_auto_draft = opts
            .draft_model
            .as_deref()
            .is_some_and(|s| s.trim().eq_ignore_ascii_case("auto"));

        let (mut draft_proposer, mut draft_path) = if let Some(draft_spec) = opts.draft_model {
            if is_auto_draft {
                match hf::resolve_auto_draft_model(
                    &resolved_target,
                    opts.cache_dir,
                    vram_opt,
                    opts.use_gpu,
                    opts.on_progress,
                )? {
                    Some(resolved_draft) => {
                        let mut draft_cfg = EngineConfig::reference(&resolved_draft.local_path);
                        draft_cfg.n_ctx = ctx_size;
                        draft_cfg.n_batch = batch_size.clamp(16, 64);
                        draft_cfg.n_threads = n_threads.max(1);
                        draft_cfg.kv_cache_type = opts.kv_cache_type;
                        draft_cfg.use_gpu = opts.use_gpu;
                        draft_cfg.prefer_direct_io = opts.prefer_direct_io;
                        match DraftModelProposer::open(draft_cfg, max_draft_tokens) {
                            Ok(proposer) => (Some(proposer), Some(resolved_draft.local_path)),
                            Err(_) => (None, None),
                        }
                    }
                    None => (None, None),
                }
            } else {
                let draft_resolve = PretrainedResolveOptions {
                    revision: opts.revision,
                    quant: None,
                    cache_dir: opts.cache_dir,
                    vram_budget_bytes: None,
                    use_gpu: opts.use_gpu,
                    has_draft: false,
                    on_progress: opts.on_progress,
                };
                let resolved_draft = resolve_pretrained_model(&draft_spec, &draft_resolve)?;
                let mut draft_cfg = EngineConfig::reference(&resolved_draft.local_path);
                draft_cfg.n_ctx = ctx_size;
                draft_cfg.n_batch = batch_size.clamp(16, 64);
                draft_cfg.n_threads = n_threads.max(1);
                draft_cfg.kv_cache_type = opts.kv_cache_type;
                draft_cfg.use_gpu = opts.use_gpu;
                draft_cfg.prefer_direct_io = opts.prefer_direct_io;
                let proposer = DraftModelProposer::open(draft_cfg, max_draft_tokens)?;
                (Some(proposer), Some(resolved_draft.local_path))
            }
        } else {
            (None, None)
        };

        let mut cfg = EngineConfig::streaming(&resolved_target.local_path);
        cfg.n_ctx = ctx_size;
        cfg.n_batch = batch_size;
        cfg.n_threads = n_threads.max(1);
        cfg.resident_layers = opts.resident_layers;
        cfg.vram_budget_bytes = opts.vram_budget_bytes;
        cfg.vram_budget_auto = opts.vram_budget_auto && opts.resident_layers.is_none();
        cfg.kv_cache_type = opts.kv_cache_type;
        cfg.use_gpu = opts.use_gpu;
        cfg.prefer_direct_io = opts.prefer_direct_io;
        cfg.cpu_embeddings = opts.cpu_embeddings;
        cfg.cpu_head = opts.cpu_head;
        cfg.stream_head = opts.stream_head;
        cfg.moe_active_streaming = opts.moe_active_streaming;

        let engine = StreamEngine::open(&cfg)?;
        if let Some(ref proposer) = draft_proposer
            && let Err(reason) = engine.check_draft_compatible(proposer.engine())
        {
            if is_auto_draft {
                tracing::warn!(
                    reason = %reason,
                    "draft_model='auto' failed live tokenizer probe; falling back to n-gram speculation"
                );
                draft_proposer = None;
                draft_path = None;
            } else {
                bail!("incompatible draft model tokenizer: {reason}");
            }
        }

        Ok(Self {
            engine,
            draft: draft_proposer,
            model_path: resolved_target.local_path,
            draft_model_path: draft_path,
            source_uri: resolved_target.source_uri,
            resolved_repo: resolved_target.resolved_repo,
            selected_quant: resolved_target.selected_quant,
            default_max_draft_tokens: max_draft_tokens,
            last_summary: None,
        })
    }

    pub fn stats_snapshot(&self) -> EngineStatsSnapshot {
        let st = self.engine.stats();
        let io = self.engine.io_stats();
        EngineStatsSnapshot {
            loaded: true,
            isolated_worker: false,
            backend_name: st.backend_name_str(),
            cached_kv_tokens: self.engine.cached_kv_len(),
            model_path: self.model_path.to_string_lossy().into_owned(),
            draft_model_path: self
                .draft_model_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
            source_uri: self.source_uri.clone(),
            resolved_repo: self.resolved_repo.clone(),
            selected_quant: self.selected_quant.clone(),
            n_ctx: self.engine.n_ctx(),
            vocab_size: self.engine.vocab_size(),
            n_layers: st.n_layers,
            resident_layers: st.resident_layers,
            streamed_layers: st.streamed_layers,
            resident_bytes: st.resident_bytes,
            cpu_resident_bytes: st.cpu_resident_bytes,
            slot_bytes: st.slot_bytes,
            expert_slot_bytes: st.expert_slot_bytes,
            total_layer_loads: st.total_layer_loads,
            moe_layers: st.moe_layers,
            moe_experts_loaded: st.moe_experts_loaded,
            moe_experts_total: st.moe_experts_total,
            head_streamed: st.head_streamed,
            moe_active_streaming: st.moe_active_streaming,
            is_uma_zero_copy: st.is_uma_zero_copy,
            is_unified_memory: st.is_unified_memory,
            using_gpu: st.using_gpu,
            can_speculate: st.can_speculate,
            io,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufInspection {
    pub architecture: String,
    pub name: Option<String>,
    pub n_layers: u32,
    pub context_length: Option<u32>,
    pub total_file_bytes: u64,
    pub shard_count: usize,
    pub tensor_count: usize,
}

pub fn inspect_gguf_file(path: &Path) -> Result<GgufInspection> {
    if !path.is_file() {
        bail!("local GGUF model not found: '{}'", path.display());
    }
    let idx = GgufIndex::open(path)?;
    Ok(GgufInspection {
        architecture: idx.meta.architecture,
        name: idx.meta.name,
        n_layers: idx.meta.n_layers,
        context_length: idx.meta.context_length,
        total_file_bytes: idx.meta.total_file_bytes,
        shard_count: idx.shards.len(),
        tensor_count: idx.tensors.len(),
    })
}

pub fn pull_model(
    spec: &str,
    cache_dir: Option<PathBuf>,
    quant: Option<String>,
    revision: Option<String>,
    on_progress: Option<ProgressCallback>,
) -> Result<PathBuf> {
    let root = cache_dir.unwrap_or_else(hf::default_cache_dir);
    let trimmed = spec.trim();
    if trimmed.starts_with("hf://")
        && let Ok(mut parsed) = hf::HfSpec::parse(trimmed)
    {
        if let Some(rev) = revision.as_deref().filter(|s| !s.trim().is_empty())
            && !trimmed.contains('@')
        {
            parsed.revision = rev.trim().to_string();
        }
        let uri = if parsed.revision == "main" {
            format!("hf://{}:{}", parsed.repo_id, parsed.filename)
        } else {
            format!(
                "hf://{}:{}@{}",
                parsed.repo_id, parsed.filename, parsed.revision
            )
        };
        hf::ensure_hf_downloaded_with_progress(&uri, &root, on_progress.as_ref())
    } else if !trimmed.starts_with("hf://") && trimmed.contains('/') {
        let opts = PretrainedResolveOptions {
            revision,
            quant,
            cache_dir: Some(root),
            vram_budget_bytes: None,
            use_gpu: true,
            has_draft: false,
            on_progress,
        };
        Ok(resolve_pretrained_model(trimmed, &opts)?.local_path)
    } else {
        hf::ensure_hf_downloaded_with_progress(trimmed, &root, on_progress.as_ref())
    }
}

#[derive(Debug, Clone, Default)]
pub struct RecommendOptions {
    pub revision: Option<String>,
    pub free_ram_gb: Option<f64>,
    pub nvme_mib_s: Option<f64>,
    pub has_draft: bool,
    pub cache_dir: Option<PathBuf>,
    pub use_gpu: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantRecommendationReport {
    pub resolved_repo: String,
    pub quant: String,
    pub primary_file: String,
    pub total_bytes: u64,
    pub shard_count: u32,
    pub hf_uri: String,
    pub verdict: String,
    pub pinned_pct: u8,
    pub est_load_s: f64,
    pub est_tps_low: f64,
    pub est_tps_high: f64,
    pub recommended: bool,
    pub score: i32,
}

pub fn recommend_repo(
    repo: &str,
    opts: RecommendOptions,
) -> Result<(String, Vec<QuantRecommendationReport>)> {
    let (repo_id, rev) = hf::parse_repo_and_revision(repo, opts.revision.as_deref())?;
    let mut effective_repo = repo_id.clone();
    let mut candidates = hf::list_hf_gguf_candidates(&repo_id, &rev)?;

    if candidates.is_empty() && !repo_id.to_ascii_lowercase().ends_with("-gguf") {
        let sibling = format!("{repo_id}-GGUF");
        if let Ok(sib_cands) = hf::list_hf_gguf_candidates(&sibling, &rev)
            && !sib_cands.is_empty()
        {
            effective_repo = sibling;
            candidates = sib_cands;
        }
    }
    if candidates.is_empty() {
        bail!("no .gguf files found in Hugging Face repository '{repo_id}@{rev}'");
    }

    let free_vram_bytes = if let Some(gb) = opts.free_ram_gb {
        (gb.max(0.5) * 1024.0 * 1024.0 * 1024.0) as u64
    } else if opts.use_gpu
        && let Some((free, _)) = StreamEngine::probe_gpu_memory(true)
    {
        free
    } else {
        4 * 1024 * 1024 * 1024
    };

    let storage_bw = if let Some(bw) = opts.nvme_mib_s {
        bw.max(10.0)
    } else {
        let probe_dir = opts.cache_dir.unwrap_or_else(hf::default_cache_dir);
        let _ = std::fs::create_dir_all(&probe_dir);
        probe_dir_storage_mib_per_s(&probe_dir).unwrap_or(1500.0)
    };

    let recs: Vec<QuantRecommendation> = hf::recommend_candidates(
        &effective_repo,
        &rev,
        &candidates,
        free_vram_bytes,
        storage_bw,
        opts.has_draft,
    );
    let reports = recs
        .into_iter()
        .map(|r| QuantRecommendationReport {
            resolved_repo: effective_repo.clone(),
            quant: r.candidate.quant,
            primary_file: r.candidate.primary_file,
            total_bytes: r.candidate.total_bytes,
            shard_count: r.candidate.shard_count,
            hf_uri: r.hf_uri,
            verdict: r.verdict,
            pinned_pct: r.pinned_pct,
            est_load_s: r.est_load_s,
            est_tps_low: r.est_tps_low,
            est_tps_high: r.est_tps_high,
            recommended: r.recommended,
            score: r.score,
        })
        .collect();
    Ok((effective_repo, reports))
}

#[derive(Debug, Clone, Default)]
pub struct CheckOptions {
    pub draft_model: Option<String>,
    pub quant: Option<String>,
    pub revision: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub use_gpu: bool,
    pub allow_slow: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PreflightReport {
    pub ok: bool,
    pub verdict: String,
    pub summary: String,
    pub model_path: String,
    pub draft_model_path: Option<String>,
    pub architecture: String,
    pub n_layers: u32,
    pub context_length: Option<u32>,
    pub total_file_bytes: u64,
    pub storage_mib_s: f64,
    pub direct_io: bool,
    pub gpu_backend: Option<String>,
    pub gpu_free_bytes: Option<u64>,
    pub gpu_total_bytes: Option<u64>,
    pub fits_in_vram: bool,
    pub est_tps_low: f64,
    pub est_tps_high: f64,
}

pub fn check_model(model: &str, opts: CheckOptions) -> Result<PreflightReport> {
    let resolve_opts = PretrainedResolveOptions {
        revision: opts.revision.clone(),
        quant: opts.quant,
        cache_dir: opts.cache_dir.clone(),
        vram_budget_bytes: None,
        use_gpu: opts.use_gpu,
        has_draft: opts.draft_model.is_some(),
        on_progress: None,
    };
    let resolved_target = resolve_pretrained_model(model, &resolve_opts)?;
    let idx = GgufIndex::open(&resolved_target.local_path)?;

    let (draft_path, draft_reserve) = if let Some(d_spec) = opts.draft_model {
        if d_spec.trim().eq_ignore_ascii_case("auto") {
            match hf::resolve_auto_draft_model(
                &resolved_target,
                opts.cache_dir,
                None,
                opts.use_gpu,
                None,
            )? {
                Some(resolved_draft) => {
                    let draft_idx = GgufIndex::open(&resolved_draft.local_path)?;
                    let reserve = draft_idx
                        .meta
                        .total_file_bytes
                        .saturating_add(64 * 1024 * 1024);
                    (Some(resolved_draft.local_path), reserve)
                }
                None => (None, 0u64),
            }
        } else {
            let d_opts = PretrainedResolveOptions {
                revision: opts.revision,
                quant: None,
                cache_dir: opts.cache_dir,
                vram_budget_bytes: None,
                use_gpu: opts.use_gpu,
                has_draft: false,
                on_progress: None,
            };
            let resolved_draft = resolve_pretrained_model(&d_spec, &d_opts)?;
            check_gguf_draft_compatible(&resolved_target.local_path, &resolved_draft.local_path)
                .map_err(|e| anyhow::anyhow!("incompatible draft model tokenizer: {e}"))?;
            let draft_idx = GgufIndex::open(&resolved_draft.local_path)?;
            let reserve = draft_idx
                .meta
                .total_file_bytes
                .saturating_add(64 * 1024 * 1024);
            (Some(resolved_draft.local_path), reserve)
        }
    } else {
        (None, 0u64)
    };

    let gpu_dev = if opts.use_gpu {
        let probed = StreamEngine::probe_gpu_device(true);
        if probed.is_none() {
            bail!("no usable GPU device found (CUDA/Vulkan/Metal); disable use_gpu for CPU mode");
        }
        probed
    } else {
        None
    };
    let gpu_mem = gpu_dev.as_ref().map(|(f, t, _)| (*f, *t));

    let size_bytes = idx.meta.total_file_bytes;
    let size_gib = size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let fits_in_vram = gpu_mem
        .map(|(free_b, _)| {
            size_bytes
                .saturating_add(draft_reserve)
                .saturating_add(768 * 1024 * 1024)
                < free_b
        })
        .unwrap_or(false);

    let probe_res = probe_storage_mib_per_s(&resolved_target.local_path, true).ok();
    let (mib_s, direct_ok) = probe_res.unwrap_or((0.0, false));
    let is_oversize = size_bytes >= OVERSIZE_GUARD_BYTES && !fits_in_vram;

    if is_oversize && opts.use_gpu && !opts.allow_slow {
        if probe_res.is_none() || !mib_s.is_finite() || mib_s <= 0.0 {
            bail!(
                "could not measure storage throughput for '{}' ({size_gib:.1} GiB) \
                 required for out-of-core GPU streaming; pass allow_slow to bypass preflight.",
                resolved_target.local_path.display()
            );
        }
        if !direct_ok {
            bail!(
                "storage throughput for '{}' ({size_gib:.1} GiB) is unverified because \
                 direct I/O (O_DIRECT/F_NOCACHE) was unavailable (measured {mib_s:.0} MiB/s from page cache). \
                 Move the GGUF to a direct-I/O filesystem or pass allow_slow.",
                resolved_target.local_path.display()
            );
        }
        if mib_s < MIN_STREAMING_STORAGE_MIB_S {
            bail!(
                "storage for '{}' ({size_gib:.1} GiB) measured only {mib_s:.0} MiB/s \
                 (minimum recommended {MIN_STREAMING_STORAGE_MIB_S:.0} MiB/s NVMe). \
                 Move the GGUF to local NVMe or pass allow_slow.",
                resolved_target.local_path.display()
            );
        }
    }

    let pinned_mib = gpu_mem.map_or(0.0, |(f, _)| {
        (f.saturating_sub(draft_reserve) as f64 * 0.45) / (1024.0 * 1024.0)
    });
    let est_stream_mib = ((size_bytes as f64 / (1024.0 * 1024.0)) - pinned_mib).max(0.0);
    let base_tps = if est_stream_mib > 1.0 && mib_s > 1.0 {
        mib_s / est_stream_mib
    } else {
        0.0
    };

    let (verdict, est_tps_low, est_tps_high, summary) = if fits_in_vram {
        (
            "GOOD",
            15.0,
            50.0,
            format!("GOOD (100% resident in VRAM, {size_gib:.2} GiB)"),
        )
    } else if draft_path.is_some() {
        let lo = (base_tps * 2.5 * 100.0).round() / 100.0;
        let hi = (base_tps * 5.0 * 100.0).round() / 100.0;
        (
            "GOOD",
            lo,
            hi,
            format!(
                "GOOD ({size_gib:.2} GiB streamed + draft speculator, storage={mib_s:.0} MiB/s, direct_io={direct_ok}, est ~{lo:.2}-{hi:.2} tok/s)"
            ),
        )
    } else if size_bytes >= 4 * 1024 * 1024 * 1024 {
        let tps = (base_tps * 100.0).round() / 100.0;
        (
            "SLOW",
            tps,
            tps,
            format!(
                "SLOW ({size_gib:.2} GiB streamed without draft_model, storage={mib_s:.0} MiB/s, est ~{tps:.2} tok/s)"
            ),
        )
    } else {
        let tps = (base_tps * 100.0).round() / 100.0;
        (
            "OK",
            tps,
            tps,
            format!("OK ({size_gib:.2} GiB, storage={mib_s:.0} MiB/s, direct_io={direct_ok})"),
        )
    };

    Ok(PreflightReport {
        ok: true,
        verdict: verdict.to_string(),
        summary,
        model_path: resolved_target.local_path.to_string_lossy().into_owned(),
        draft_model_path: draft_path.map(|p| p.to_string_lossy().into_owned()),
        architecture: idx.meta.architecture,
        n_layers: idx.meta.n_layers,
        context_length: idx.meta.context_length,
        total_file_bytes: size_bytes,
        storage_mib_s: mib_s,
        direct_io: direct_ok,
        gpu_backend: gpu_dev
            .as_ref()
            .map(|(_, _, b)| b.clone())
            .or_else(|| (!opts.use_gpu).then(|| "CPU".to_string())),
        gpu_free_bytes: gpu_dev.as_ref().map(|(f, _, _)| *f),
        gpu_total_bytes: gpu_dev.as_ref().map(|(_, t, _)| *t),
        fits_in_vram,
        est_tps_low,
        est_tps_high,
    })
}
