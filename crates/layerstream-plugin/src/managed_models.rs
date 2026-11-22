//! Host-managed `[[models]]` (`provider = "layerstream"`) support via Mesh-LLM's
//! `model_assets.v1` contract.
//!
//! When operators configure `[[models]]` entries naming `layerstream` in `mesh-llm`'s
//! `config.toml`, the host injects `MESH_LLM_PLUGIN_MODELS` and serves asset preparation
//! requests (`PrepareModelAssetsRequest`) backed by the host's Hugging Face credentials,
//! source policy (`repo` / `revision` / `local_dir` / `HF_HUB_OFFLINE`), and shared HF cache.

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use layerstream_server::ModelRegistry;
use mesh_llm_plugin::PluginContext;
use mesh_llm_plugin::models::{
    AssetPlan, ManagedModelConfig, ModelLifecycle, configured_models_from_env,
};

use crate::config::run_preflight_checks;
use layerstream_core::hf;

const SUPPORTED_OPTIONS: &[&str] = &["file", "files", "alias", "draft_model", "draft_file"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManagedDraftSource {
    /// Fall back to CLI `--draft-model` (if configured).
    DefaultCli,
    /// Draft GGUF fetched from the same host-managed bundle via `options.draft_file`.
    AssetFile(String),
    /// External local path or `hf://...` URI via `options.draft_model`.
    ExternalSpec(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedModelPlan {
    pub configured_model: String,
    pub advertised_id: String,
    pub asset_plan: AssetPlan,
    pub primary_file: String,
    pub draft_source: ManagedDraftSource,
    pub lifecycle: ModelLifecycle,
}

/// Reads `MESH_LLM_PLUGIN_MODELS` (if present) and validates every entry into an `AssetPlan`.
pub fn load_managed_model_plans() -> Result<Vec<ManagedModelPlan>> {
    let configs = configured_models_from_env()?;
    build_managed_model_plans(&configs)
}

/// Validates and converts host-supplied `ManagedModelConfig` entries into `ManagedModelPlan`s.
pub fn build_managed_model_plans(configs: &[ManagedModelConfig]) -> Result<Vec<ManagedModelPlan>> {
    let mut plans = Vec::with_capacity(configs.len());
    let mut seen_ids = BTreeSet::new();

    for config in configs {
        let plan = plan_managed_model(config)?;
        if !seen_ids.insert(plan.advertised_id.clone()) {
            bail!(
                "duplicate managed model ID '{}'; set distinct [models.options].alias values",
                plan.advertised_id
            );
        }
        plans.push(plan);
    }

    Ok(plans)
}

/// Returns the smallest positive `lifecycle.idle_timeout_secs` across managed models, if any.
///
/// `ModelLifecycle::default().idle_timeout_secs` is `0` when omitted, so `0` is treated as
/// unspecified rather than disabling layerstream's default idle eviction (operators who want
/// immediate post-request eviction can pass `--idle-timeout-secs 0` explicitly).
pub fn effective_managed_idle_timeout_secs(plans: &[ManagedModelPlan]) -> Option<u64> {
    plans
        .iter()
        .map(|p| p.lifecycle.idle_timeout_secs)
        .filter(|&secs| secs > 0)
        .min()
}

fn plan_managed_model(config: &ManagedModelConfig) -> Result<ManagedModelPlan> {
    if !config.request_defaults.is_empty() {
        let keys = config
            .request_defaults
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        bail!(
            "managed model '{}': models.request_defaults is not supported by layerstream (got: {keys})",
            config.model
        );
    }

    for key in config.options.keys() {
        if !SUPPORTED_OPTIONS.contains(&key.as_str()) {
            bail!(
                "managed model '{}': unsupported option '{key}' (supported options: {})",
                config.model,
                SUPPORTED_OPTIONS.join(", ")
            );
        }
    }

    let opt_file = optional_non_empty_str(config, "file")?;
    let opt_files = optional_non_empty_str_list(config, "files")?;
    if opt_file.is_some() && opt_files.is_some() {
        bail!(
            "managed model '{}': options.file and options.files cannot both be set",
            config.model
        );
    }

    let opt_alias = optional_non_empty_str(config, "alias")?;
    let opt_draft_model = optional_non_empty_str(config, "draft_model")?;
    let opt_draft_file = optional_non_empty_str(config, "draft_file")?;
    if opt_draft_model.is_some() && opt_draft_file.is_some() {
        bail!(
            "managed model '{}': options.draft_model and options.draft_file cannot both be set",
            config.model
        );
    }

    let (mut files, primary_file) = if let Some(explicit_files) = opt_files {
        let primary = select_primary_gguf(&explicit_files).with_context(|| {
            format!(
                "managed model '{}': options.files must include at least one .gguf file",
                config.model
            )
        })?;
        (explicit_files, primary)
    } else if let Some(file) = opt_file {
        ensure_gguf_filename(&config.model, "options.file", &file)?;
        let expanded = hf::expand_gguf_shard_filename(&file);
        let primary = expanded[0].clone();
        (expanded, primary)
    } else {
        let variant = config.variant.trim();
        if is_gguf_filename(variant) {
            let expanded = hf::expand_gguf_shard_filename(variant);
            let primary = expanded[0].clone();
            (expanded, primary)
        } else {
            bail!(
                "managed model '{}': specify a GGUF filename via variant = \"<file>.gguf\" \
                 or [models.options] file = \"<file>.gguf\" (or files = [...]); got variant = {:?}",
                config.model,
                config.variant
            );
        }
    };

    let draft_source = if let Some(draft_file) = opt_draft_file {
        ensure_gguf_filename(&config.model, "options.draft_file", &draft_file)?;
        let expanded = hf::expand_gguf_shard_filename(&draft_file);
        let primary_draft = expanded[0].clone();
        for shard in expanded {
            if !files.contains(&shard) {
                files.push(shard);
            }
        }
        ManagedDraftSource::AssetFile(primary_draft)
    } else if let Some(draft_model) = opt_draft_model {
        ManagedDraftSource::ExternalSpec(draft_model)
    } else {
        ManagedDraftSource::DefaultCli
    };

    let (model_repo, model_rev) = config
        .model
        .rsplit_once('@')
        .map_or((config.model.as_str(), None), |(repo, rev)| {
            (repo, Some(rev))
        });
    let repo = config
        .source
        .repo
        .as_deref()
        .unwrap_or(model_repo)
        .trim()
        .to_string();
    let revision = config
        .source
        .revision
        .as_deref()
        .or(model_rev)
        .unwrap_or("main")
        .trim()
        .to_string();

    let asset_plan = AssetPlan {
        model: config.model.clone(),
        repo,
        revision,
        files,
    };
    asset_plan.validate().with_context(|| {
        if config.source.local_dir.is_some() {
            format!(
                "invalid managed model '{}': mesh-llm model_assets.v1 requires an 'owner/name' \
                 model ID (for example 'local/{}') when using [models.source].local_dir",
                config.model, config.model
            )
        } else {
            format!(
                "invalid managed model '{}': set model = \"owner/repo\" (or [models.source].repo = \"owner/repo\") \
                 and valid relative GGUF path(s)",
                config.model
            )
        }
    })?;

    let advertised_id = opt_alias.unwrap_or_else(|| config.model.clone());
    Ok(ManagedModelPlan {
        configured_model: config.model.clone(),
        advertised_id,
        asset_plan,
        primary_file,
        draft_source,
        lifecycle: config.lifecycle.clone(),
    })
}

fn optional_non_empty_str(config: &ManagedModelConfig, key: &str) -> Result<Option<String>> {
    let Some(value) = config.options.get(key) else {
        return Ok(None);
    };
    let s = value.as_str().ok_or_else(|| {
        anyhow!(
            "managed model '{}': options.{key} must be a string",
            config.model
        )
    })?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        bail!(
            "managed model '{}': options.{key} must not be empty",
            config.model
        );
    }
    Ok(Some(trimmed.to_string()))
}

fn optional_non_empty_str_list(
    config: &ManagedModelConfig,
    key: &str,
) -> Result<Option<Vec<String>>> {
    let Some(value) = config.options.get(key) else {
        return Ok(None);
    };
    let arr = value.as_array().ok_or_else(|| {
        anyhow!(
            "managed model '{}': options.{key} must be an array of strings",
            config.model
        )
    })?;
    if arr.is_empty() {
        bail!(
            "managed model '{}': options.{key} must not be empty",
            config.model
        );
    }
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        let s = entry.as_str().ok_or_else(|| {
            anyhow!(
                "managed model '{}': options.{key} entries must be strings",
                config.model
            )
        })?;
        let trimmed = s.trim();
        if trimmed.is_empty() {
            bail!(
                "managed model '{}': options.{key} entries must not be empty",
                config.model
            );
        }
        out.push(trimmed.to_string());
    }
    Ok(Some(out))
}

fn is_gguf_filename(name: &str) -> bool {
    name.len() > 5 && name.to_ascii_lowercase().ends_with(".gguf")
}

fn ensure_gguf_filename(model: &str, field: &str, value: &str) -> Result<()> {
    if !is_gguf_filename(value) {
        bail!("managed model '{model}': {field} must end with '.gguf' (got {value:?})");
    }
    Ok(())
}

fn is_secondary_shard_name(name: &str) -> bool {
    let file_name = name.rsplit('/').next().unwrap_or(name);
    let Some(stem) = file_name
        .strip_suffix(".gguf")
        .or_else(|| file_name.strip_suffix(".GGUF"))
    else {
        return false;
    };
    if let Some((prefix_and_idx, total_str)) = stem.rsplit_once("-of-")
        && total_str.len() == 5
        && total_str.parse::<u32>().is_ok()
        && let Some((_, idx_str)) = prefix_and_idx.rsplit_once('-')
        && idx_str.len() == 5
        && let Ok(idx) = idx_str.parse::<u32>()
    {
        return idx > 1;
    }
    false
}

fn select_primary_gguf(files: &[String]) -> Result<String> {
    let ggufs: Vec<&String> = files.iter().filter(|f| is_gguf_filename(f)).collect();
    if ggufs.is_empty() {
        bail!("no .gguf file found in asset list");
    }
    if let Some(primary) = ggufs.iter().find(|f| !is_secondary_shard_name(f)) {
        return Ok((*primary).clone());
    }
    Ok(ggufs[0].clone())
}

/// Prepares all host-managed models via `context.prepare_model_assets`, registers their
/// resolved GGUF headers (0 MB VRAM) into `registry`, and runs storage preflight checks.
#[allow(clippy::too_many_arguments)]
pub async fn prepare_managed_models(
    context: &PluginContext<'static>,
    plans: &[ManagedModelPlan],
    mut registry: ModelRegistry,
    default_draft_spec: Option<String>,
    cache_root: PathBuf,
    use_gpu: bool,
    allow_slow: bool,
    on_status: impl Fn(String),
) -> Result<ModelRegistry> {
    let default_draft_path = match default_draft_spec {
        Some(spec) => {
            let cache = cache_root.clone();
            Some(
                tokio::task::spawn_blocking(move || hf::resolve_path_or_hf(&spec, &cache))
                    .await
                    .context("join draft resolution task")??,
            )
        }
        None => None,
    };

    let total = plans.len();
    for (idx, plan) in plans.iter().enumerate() {
        if !plan.lifecycle.prefetch {
            tracing::info!(
                model = %plan.configured_model,
                "preparing managed model asset bundle during initialization (layerstream stays at 0 MB VRAM idle until inference)"
            );
        }
        on_status(format!(
            "preparing managed model {}/{total}: {}",
            idx + 1,
            plan.advertised_id
        ));

        let assets = context
            .prepare_model_assets(&plan.asset_plan)
            .await
            .with_context(|| {
                format!(
                    "prepare host-managed model '{}' ({}@{})",
                    plan.configured_model, plan.asset_plan.repo, plan.asset_plan.revision
                )
            })?;

        let primary_path = assets
            .files
            .get(&plan.primary_file)
            .cloned()
            .with_context(|| {
                format!(
                    "host asset response for '{}' omitted primary GGUF '{}'",
                    plan.configured_model, plan.primary_file
                )
            })?;

        let draft_path = match &plan.draft_source {
            ManagedDraftSource::AssetFile(file) => {
                let path = assets.files.get(file).cloned().with_context(|| {
                    format!(
                        "host asset response for '{}' omitted draft GGUF '{file}'",
                        plan.configured_model
                    )
                })?;
                Some(path)
            }
            ManagedDraftSource::ExternalSpec(spec) => {
                let spec = spec.clone();
                let cache = cache_root.clone();
                Some(
                    tokio::task::spawn_blocking(move || hf::resolve_path_or_hf(&spec, &cache))
                        .await
                        .context("join managed draft resolution task")??,
                )
            }
            ManagedDraftSource::DefaultCli => default_draft_path.clone(),
        };

        let advertised_id = plan.advertised_id.clone();
        let resolved_rev = assets.revision.clone();
        let primary_for_log = primary_path.clone();
        registry = tokio::task::spawn_blocking(move || -> Result<ModelRegistry> {
            registry.register_file(&primary_path, Some(advertised_id), draft_path)?;
            Ok(registry)
        })
        .await
        .context("join managed model registration task")??;

        tracing::info!(
            model = %plan.advertised_id,
            configured = %plan.configured_model,
            revision = %resolved_rev,
            path = %primary_for_log.display(),
            files = assets.files.len(),
            "registered host-managed GGUF via model_assets.v1 (0 MB VRAM idle)"
        );
    }

    let registry = tokio::task::spawn_blocking(move || -> Result<ModelRegistry> {
        run_preflight_checks(&registry, use_gpu, allow_slow, false)?;
        Ok(registry)
    })
    .await
    .context("join preflight check task")??;

    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse_config(value: serde_json::Value) -> ManagedModelConfig {
        serde_json::from_value(value).expect("valid ManagedModelConfig JSON")
    }

    #[test]
    fn plans_variant_gguf_with_default_main_revision() {
        let cfg = parse_config(json!({
            "model": "unsloth/Qwen3-30B-A3B-GGUF",
            "provider": "layerstream",
            "variant": "Qwen3-30B-A3B-Q4_K_M.gguf"
        }));
        let plans = build_managed_model_plans(&[cfg]).unwrap();
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].advertised_id, "unsloth/Qwen3-30B-A3B-GGUF");
        assert_eq!(plans[0].asset_plan.repo, "unsloth/Qwen3-30B-A3B-GGUF");
        assert_eq!(plans[0].asset_plan.revision, "main");
        assert_eq!(plans[0].asset_plan.files, vec!["Qwen3-30B-A3B-Q4_K_M.gguf"]);
        assert_eq!(plans[0].primary_file, "Qwen3-30B-A3B-Q4_K_M.gguf");
    }

    #[test]
    fn expands_multipart_shards_from_options_file_and_at_revision() {
        let cfg = parse_config(json!({
            "model": "bartowski/Llama-70B-GGUF@rev123",
            "provider": "layerstream",
            "options": {
                "file": "Q4_K_M/llama-00002-of-00003.gguf",
                "alias": "llama-70b"
            }
        }));
        let plans = build_managed_model_plans(&[cfg]).unwrap();
        assert_eq!(plans[0].advertised_id, "llama-70b");
        assert_eq!(plans[0].asset_plan.model, "bartowski/Llama-70B-GGUF@rev123");
        assert_eq!(plans[0].asset_plan.repo, "bartowski/Llama-70B-GGUF");
        assert_eq!(plans[0].asset_plan.revision, "rev123");
        assert_eq!(
            plans[0].asset_plan.files,
            vec![
                "Q4_K_M/llama-00001-of-00003.gguf",
                "Q4_K_M/llama-00002-of-00003.gguf",
                "Q4_K_M/llama-00003-of-00003.gguf",
            ]
        );
        assert_eq!(plans[0].primary_file, "Q4_K_M/llama-00001-of-00003.gguf");
    }

    #[test]
    fn source_overrides_and_draft_file_bundle_into_same_plan() {
        let cfg = parse_config(json!({
            "model": "qwen3-30b",
            "provider": "layerstream",
            "variant": "q4_k_m",
            "options": {
                "file": "Qwen3-30B-A3B-Q4_K_M.gguf",
                "draft_file": "Qwen3-0.6B-Q4_K_M.gguf"
            },
            "source": {
                "repo": "unsloth/Qwen3-30B-A3B-GGUF",
                "revision": "abcdef0123456789"
            },
            "lifecycle": {
                "load": "on_demand",
                "idle_timeout_secs": 30,
                "prefetch": true
            }
        }));
        let plans = build_managed_model_plans(&[cfg]).unwrap();
        assert_eq!(plans[0].advertised_id, "qwen3-30b");
        assert_eq!(plans[0].asset_plan.repo, "unsloth/Qwen3-30B-A3B-GGUF");
        assert_eq!(plans[0].asset_plan.revision, "abcdef0123456789");
        assert_eq!(
            plans[0].asset_plan.files,
            vec!["Qwen3-30B-A3B-Q4_K_M.gguf", "Qwen3-0.6B-Q4_K_M.gguf"]
        );
        assert_eq!(
            plans[0].draft_source,
            ManagedDraftSource::AssetFile("Qwen3-0.6B-Q4_K_M.gguf".into())
        );
        assert_eq!(effective_managed_idle_timeout_secs(&plans), Some(30));
    }

    #[test]
    fn explicit_options_files_selects_primary_shard() {
        let cfg = parse_config(json!({
            "model": "org/repo",
            "provider": "layerstream",
            "options": {
                "files": [
                    "model-00002-of-00002.gguf",
                    "model-00001-of-00002.gguf"
                ]
            }
        }));
        let plans = build_managed_model_plans(&[cfg]).unwrap();
        assert_eq!(plans[0].primary_file, "model-00001-of-00002.gguf");
    }

    #[test]
    fn rejects_default_fp32_variant_without_gguf_file() {
        let cfg = parse_config(json!({
            "model": "org/repo",
            "provider": "layerstream"
        }));
        let err = build_managed_model_plans(&[cfg]).unwrap_err();
        assert!(err.to_string().contains("specify a GGUF filename"), "{err}");
    }

    #[test]
    fn rejects_unknown_options_and_request_defaults_and_conflicts() {
        let unknown_opt = parse_config(json!({
            "model": "org/repo",
            "provider": "layerstream",
            "variant": "m.gguf",
            "options": { "bogus": true }
        }));
        assert!(
            build_managed_model_plans(&[unknown_opt])
                .unwrap_err()
                .to_string()
                .contains("unsupported option 'bogus'")
        );

        let req_defaults = parse_config(json!({
            "model": "org/repo",
            "provider": "layerstream",
            "variant": "m.gguf",
            "request_defaults": { "temperature": 0.2 }
        }));
        assert!(
            build_managed_model_plans(&[req_defaults])
                .unwrap_err()
                .to_string()
                .contains("request_defaults")
        );

        let both_files = parse_config(json!({
            "model": "org/repo",
            "provider": "layerstream",
            "options": {
                "file": "a.gguf",
                "files": ["a.gguf"]
            }
        }));
        assert!(
            build_managed_model_plans(&[both_files])
                .unwrap_err()
                .to_string()
                .contains("cannot both be set")
        );

        let traversal = parse_config(json!({
            "model": "org/repo",
            "provider": "layerstream",
            "options": { "file": "../escape.gguf" }
        }));
        assert!(build_managed_model_plans(&[traversal]).is_err());

        let local_bad_id = parse_config(json!({
            "model": "my-model",
            "provider": "layerstream",
            "variant": "m.gguf",
            "source": { "local_dir": "/tmp/models" }
        }));
        let err = build_managed_model_plans(&[local_bad_id]).unwrap_err();
        assert!(err.to_string().contains("local/my-model"), "{err:#}");
    }

    #[test]
    fn rejects_duplicate_advertised_ids() {
        let a = parse_config(json!({
            "model": "org/repo-a",
            "provider": "layerstream",
            "variant": "a.gguf",
            "options": { "alias": "shared-id" }
        }));
        let b = parse_config(json!({
            "model": "org/repo-b",
            "provider": "layerstream",
            "variant": "b.gguf",
            "options": { "alias": "shared-id" }
        }));
        let err = build_managed_model_plans(&[a, b]).unwrap_err();
        assert!(err.to_string().contains("duplicate managed model ID"));
    }

    #[test]
    fn idle_timeout_ignores_zero_default_and_picks_minimum_positive() {
        let default_zero = parse_config(json!({
            "model": "org/a",
            "provider": "layerstream",
            "variant": "a.gguf"
        }));
        let plans = build_managed_model_plans(std::slice::from_ref(&default_zero)).unwrap();
        assert_eq!(effective_managed_idle_timeout_secs(&plans), None);

        let t60 = parse_config(json!({
            "model": "org/b",
            "provider": "layerstream",
            "variant": "b.gguf",
            "lifecycle": { "load": "on_demand", "idle_timeout_secs": 60, "prefetch": true }
        }));
        let t15 = parse_config(json!({
            "model": "org/c",
            "provider": "layerstream",
            "variant": "c.gguf",
            "lifecycle": { "load": "on_demand", "idle_timeout_secs": 15, "prefetch": true }
        }));
        let plans = build_managed_model_plans(&[default_zero, t60, t15]).unwrap();
        assert_eq!(effective_managed_idle_timeout_secs(&plans), Some(15));
    }
}
