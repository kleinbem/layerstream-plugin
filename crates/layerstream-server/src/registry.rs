//! Multi-model GGUF registry (reads only GGUF headers at startup -> 0 MB VRAM).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use layerstream_core::{GgufIndex, ModelMeta};

#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub id: String,
    pub path: PathBuf,
    pub draft_path: Option<PathBuf>,
    pub meta: ModelMeta,
}

#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    models: BTreeMap<String, ModelEntry>,
}

impl ModelRegistry {
    pub fn from_sources(
        explicit_models: &[String],
        models_dir: Option<&Path>,
        default_draft: Option<PathBuf>,
    ) -> Result<Self> {
        let mut registry = Self::default();

        for spec in explicit_models {
            let (explicit_id, raw_path) = match spec.split_once('=') {
                Some((alias, path)) if !alias.is_empty() && !path.is_empty() => {
                    (Some(alias.trim().to_string()), PathBuf::from(path.trim()))
                }
                _ => (None, PathBuf::from(spec.trim())),
            };
            registry.register_file(&raw_path, explicit_id, default_draft.clone())?;
        }

        if let Some(dir) = models_dir {
            if !dir.is_dir() {
                bail!(
                    "--models-dir does not exist or is not a directory: {}",
                    dir.display()
                );
            }
            let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
                .with_context(|| format!("read models dir {}", dir.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.is_file()
                        && p.extension().and_then(|e| e.to_str()) == Some("gguf")
                        && !is_secondary_shard(p)
                })
                .collect();
            entries.sort();
            for path in entries {
                if let Err(err) = registry.register_file(&path, None, default_draft.clone()) {
                    tracing::warn!("skipping {}: {err:#}", path.display());
                }
            }
        }

        Ok(registry)
    }

    pub fn register_file(
        &mut self,
        path: &Path,
        explicit_id: Option<String>,
        draft_path: Option<PathBuf>,
    ) -> Result<()> {
        let canonical = std::fs::canonicalize(path)
            .with_context(|| format!("resolve model path {}", path.display()))?;
        let index = GgufIndex::open(&canonical)?;
        let id = explicit_id.unwrap_or_else(|| derive_model_id(&canonical));
        self.models.insert(
            id.clone(),
            ModelEntry {
                id,
                path: canonical,
                draft_path,
                meta: index.meta,
            },
        );
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&ModelEntry> {
        self.models.get(id)
    }

    pub fn first(&self) -> Option<&ModelEntry> {
        self.models.values().next()
    }

    pub fn list(&self) -> Vec<&ModelEntry> {
        self.models.values().collect()
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

fn derive_model_id(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("layerstream-model");
    // Strip `-00001-of-0000N` suffix from multi-part GGUF stems.
    if let Some((prefix_and_idx, total_str)) = stem.rsplit_once("-of-")
        && total_str.len() == 5
        && total_str.parse::<u32>().is_ok()
        && let Some((prefix, idx_str)) = prefix_and_idx.rsplit_once('-')
        && idx_str.len() == 5
        && idx_str.parse::<u32>().is_ok()
    {
        return prefix.to_string();
    }
    stem.to_string()
}

fn is_secondary_shard(path: &Path) -> bool {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
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
