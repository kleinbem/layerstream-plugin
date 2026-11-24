//! Hugging Face GGUF resolution wired to `StreamEngine::probe_gpu_memory`.

use std::path::PathBuf;

use anyhow::Result;

pub use hf_gguf::*;

/// Resolves a model specification into a local `.gguf` file path, using `StreamEngine`
/// GPU memory probing when `opts.use_gpu` is set and `opts.vram_budget_bytes` is unset.
pub fn resolve_pretrained_model(
    spec: &str,
    opts: &PretrainedResolveOptions,
) -> Result<ResolvedModel> {
    hf_gguf::resolve_pretrained_model_with_probe(spec, opts, || {
        crate::StreamEngine::probe_gpu_memory(true).map(|(free, _)| free)
    })
}

/// Resolves `draft_model="auto"` for an already-resolved target GGUF model, using
/// `StreamEngine` GPU memory probing when `use_gpu` is set and `vram_budget_bytes` is unset.
pub fn resolve_auto_draft_model(
    target: &ResolvedModel,
    cache_dir: Option<PathBuf>,
    vram_budget_bytes: Option<u64>,
    use_gpu: bool,
    on_progress: Option<ProgressCallback>,
) -> Result<Option<ResolvedModel>> {
    hf_gguf::resolve_auto_draft_model_with_probe(
        target,
        cache_dir,
        vram_budget_bytes,
        use_gpu,
        on_progress,
        || crate::StreamEngine::probe_gpu_memory(true).map(|(free, _)| free),
    )
}
