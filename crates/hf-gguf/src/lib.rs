//! Direct Hugging Face (`hf://org/repo:file.gguf`) model pull and local cache.
//!
//! Supported URI formats:
//! - `hf://org/repo:filename.gguf`
//! - `hf://org/repo:subdir/filename.gguf@revision`
//! - `hf://org/repo/filename.gguf`
//! - `alias=hf://org/repo:filename.gguf` (via `resolve_model_spec`)
//!
//! Features:
//! - Resumable HTTP `Range` downloads via atomic `<file>.part` rename.
//! - Multi-part GGUF shard auto-detection (`*-00001-of-0000N.gguf` downloads all N shards).
//! - Authentication via `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or `~/.cache/huggingface/token`.
//! - All progress logs go strictly to stderr (preserving Mesh-LLM's stdout protobuf channel).

use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Error returned by Hugging Face GGUF discovery, quant recommendation, and download operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HfError {
    /// Invalid `hf://` URI, repo ID, filename, or model specification.
    #[error("{0}")]
    InvalidSpec(String),
    /// Requested local file or Hugging Face GGUF artifact was not found.
    #[error("{0}")]
    NotFound(String),
    /// HTTP transport or non-success status error.
    #[error("{0}")]
    Http(String),
    /// Failed to parse JSON metadata from the Hugging Face Hub API.
    #[error("{context}: {source}")]
    Json {
        context: String,
        #[source]
        source: serde_json::Error,
    },
    /// Local filesystem or lock I/O error.
    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
    /// Download integrity, resume, or size validation failure.
    #[error("{0}")]
    DownloadFailed(String),
    /// Download was aborted by a caller-supplied progress callback.
    #[error("{0}")]
    Cancelled(String),
}

impl HfError {
    pub(crate) fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }
}

impl From<std::io::Error> for HfError {
    fn from(source: std::io::Error) -> Self {
        Self::Io {
            context: "I/O error".to_string(),
            source,
        }
    }
}

/// Convenience `Result` alias for `hf-gguf`.
pub type Result<T> = std::result::Result<T, HfError>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DownloadProgress {
    pub repo_id: String,
    pub filename: String,
    pub shard_index: u32,
    pub shard_count: u32,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub mib_per_s: f64,
    pub cached: bool,
    pub done: bool,
}

/// Callback invoked during Hugging Face model downloads. Returning `Err` aborts the download.
pub type ProgressCallback = Arc<dyn Fn(&DownloadProgress) -> Result<()> + Send + Sync>;

const HF_PREFIX: &str = "hf://";
const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
const DEFAULT_READ_TIMEOUT_SECS: u64 = 60;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfSpec {
    pub repo_id: String,
    pub filename: String,
    pub revision: String,
}

impl HfSpec {
    /// Parses `hf://org/repo:path/to/file.gguf[@revision]` or `hf://org/repo/file.gguf[@revision]`.
    pub fn parse(uri: &str) -> Result<Self> {
        let Some(rest) = uri.strip_prefix(HF_PREFIX) else {
            return Err(HfError::InvalidSpec(format!("not an hf:// URI: {uri}")));
        };
        let (body, revision) = match rest.rsplit_once('@') {
            Some((b, rev)) if !rev.is_empty() && !rev.contains('/') => (b, rev.to_string()),
            _ => (rest, "main".to_string()),
        };

        let mut revision = revision;
        let (repo_id, filename) = if let Some((repo, file)) = body.split_once(':') {
            (repo.trim_matches('/'), file.trim_start_matches('/'))
        } else {
            // Support `hf://org/repo/filename.gguf` (and `hf://org/repo/resolve/<rev>/file.gguf`).
            let mut parts = body.splitn(3, '/');
            let org = parts.next().unwrap_or("");
            let repo = parts.next().unwrap_or("");
            let tail = parts.next().unwrap_or("");
            if org.is_empty() || repo.is_empty() || tail.is_empty() {
                return Err(HfError::InvalidSpec(format!(
                    "invalid hf:// spec '{uri}'; expected hf://org/repo:filename.gguf"
                )));
            }
            let repo_slice = &body[..org.len() + 1 + repo.len()];
            let clean_file = if let Some(after_resolve) = tail.strip_prefix("resolve/") {
                if let Some((rev_part, f)) = after_resolve.split_once('/') {
                    if !rev_part.is_empty() && revision == "main" {
                        revision = rev_part.to_string();
                    }
                    f
                } else {
                    tail
                }
            } else {
                tail
            };
            (repo_slice, clean_file.trim_start_matches('/'))
        };

        if !repo_id.contains('/')
            || repo_id.starts_with('/')
            || repo_id.ends_with('/')
            || repo_id.contains("..")
            || repo_id.contains('\\')
            || repo_id.contains(':')
            || repo_id
                .split('/')
                .any(|seg| seg.is_empty() || seg == "." || seg == "..")
        {
            return Err(HfError::InvalidSpec(format!(
                "invalid Hugging Face repo id '{repo_id}' in '{uri}' (expected org/repo)"
            )));
        }
        if filename.is_empty()
            || filename.contains("..")
            || filename.contains('\\')
            || filename.contains(':')
            || filename
                .split('/')
                .any(|seg| seg.is_empty() || seg == "." || seg == "..")
        {
            return Err(HfError::InvalidSpec(format!(
                "invalid Hugging Face filename '{filename}' in '{uri}'"
            )));
        }

        Ok(Self {
            repo_id: repo_id.to_string(),
            filename: filename.to_string(),
            revision,
        })
    }

    pub fn download_url(&self) -> String {
        let endpoint = std::env::var("HF_ENDPOINT")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_HF_ENDPOINT.to_string());
        format!(
            "{}/{}/resolve/{}/{}",
            endpoint.trim_end_matches('/'),
            self.repo_id,
            self.revision,
            self.filename
        )
    }

    pub fn cache_path(&self, cache_root: &Path) -> PathBuf {
        let repo_dir = self.repo_id.replace('/', "--");
        let rev_dir: String = self
            .revision
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        // Append an 8-hex digest of the exact `(repo_id, revision)` pair so repos whose
        // sanitized names match (e.g. `org/a--b` vs `org--a/b`) never share a cache directory.
        let mut hasher = Sha256::new();
        hasher.update(self.repo_id.as_bytes());
        hasher.update([0u8]);
        hasher.update(self.revision.as_bytes());
        let digest = hasher.finalize();
        let disambiguator = format!(
            "{:02x}{:02x}{:02x}{:02x}",
            digest[0], digest[1], digest[2], digest[3]
        );
        let mut p = cache_root.join(format!("{repo_dir}@{rev_dir}-{disambiguator}"));
        for seg in self.filename.split('/') {
            if !seg.is_empty() {
                p.push(seg);
            }
        }
        p
    }

    /// Expands multi-part shard specs `*-00001-of-0000N.gguf` into all N shard specs.
    pub fn expand_shards(&self) -> Vec<Self> {
        expand_gguf_shard_filename(&self.filename)
            .into_iter()
            .map(|filename| Self {
                repo_id: self.repo_id.clone(),
                filename,
                revision: self.revision.clone(),
            })
            .collect()
    }
}

/// Expands multi-part GGUF filenames `*-00001-of-0000N.gguf` (or any shard index
/// `*-XXXXX-of-0000N.gguf`) into all `N` ordered shard paths (`00001..=N`).
pub fn expand_gguf_shard_filename(filename: &str) -> Vec<String> {
    let (dir_prefix, file_name) = match filename.rsplit_once('/') {
        Some((d, f)) => (Some(d), f),
        None => (None, filename),
    };

    if let Some(stem) = file_name.strip_suffix(".gguf")
        && let Some((prefix_and_idx, total_str)) = stem.rsplit_once("-of-")
        && total_str.len() == 5
        && let Ok(total) = total_str.parse::<u32>()
        && total > 1
        && let Some((prefix, idx_str)) = prefix_and_idx.rsplit_once('-')
        && idx_str.len() == 5
        && idx_str.parse::<u32>().is_ok()
    {
        return (1..=total)
            .map(|idx| {
                let shard_file = format!("{prefix}-{idx:05}-of-{total:05}.gguf");
                match dir_prefix {
                    Some(d) => format!("{d}/{shard_file}"),
                    None => shard_file,
                }
            })
            .collect();
    }

    vec![filename.to_string()]
}

// Host-managed `[[models]]` (`provider = "layerstream"`) use `model_assets.v1` and share
// mesh-llm's Hugging Face cache directly. Standalone CLI / `--model hf://...` args fall back
// to `LAYERSTREAM_CACHE_DIR` / XDG cache (tracking: Mesh-LLM/mesh-llm#3192,
// upstream: Mesh-LLM/mesh-llm#3215).
pub fn default_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LAYERSTREAM_CACHE_DIR")
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir);
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME")
        && !xdg.trim().is_empty()
    {
        return PathBuf::from(xdg).join("layerstream").join("models");
    }
    #[cfg(windows)]
    if let Ok(local) = std::env::var("LOCALAPPDATA")
        && !local.trim().is_empty()
    {
        return PathBuf::from(local).join("layerstream").join("models");
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        return PathBuf::from(home)
            .join(".cache")
            .join("layerstream")
            .join("models");
    }
    PathBuf::from(".layerstream-cache").join("models")
}

/// Returns `true` if `spec` is a local path OR an `hf://` URI whose files are already cached.
pub fn is_model_spec_cached(spec: &str, cache_root: &Path) -> bool {
    let target = match spec.split_once('=') {
        Some((_, rhs)) if rhs.starts_with(HF_PREFIX) => rhs,
        _ => spec,
    };
    is_path_or_hf_cached(target, cache_root)
}

/// Returns candidate Hugging Face Hub cache directories (`.../hub`) in priority order.
fn hf_hub_cache_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let mut push_unique = |p: PathBuf| {
        if !roots.contains(&p) {
            roots.push(p);
        }
    };
    if let Ok(dir) = std::env::var("HF_HUB_CACHE")
        && !dir.trim().is_empty()
    {
        push_unique(PathBuf::from(dir.trim()));
    }
    if let Ok(dir) = std::env::var("HF_HOME")
        && !dir.trim().is_empty()
    {
        push_unique(PathBuf::from(dir.trim()).join("hub"));
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME")
        && !xdg.trim().is_empty()
    {
        push_unique(PathBuf::from(xdg.trim()).join("huggingface").join("hub"));
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        push_unique(
            PathBuf::from(home.trim())
                .join(".cache")
                .join("huggingface")
                .join("hub"),
        );
    }
    #[cfg(windows)]
    if let Ok(profile) = std::env::var("USERPROFILE")
        && !profile.trim().is_empty()
    {
        push_unique(
            PathBuf::from(profile.trim())
                .join(".cache")
                .join("huggingface")
                .join("hub"),
        );
    }
    roots
}

fn is_hex_commit_sha(rev: &str) -> bool {
    rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Resolves all shards of `spec` in a single standard Hugging Face Hub cache root (`.../hub`),
/// returning the ordered local paths only when every shard exists and is a complete GGUF file.
fn find_shards_in_hf_hub_root(
    spec: &HfSpec,
    shards: &[HfSpec],
    hub_root: &Path,
) -> Option<Vec<PathBuf>> {
    let model_dir = hub_root.join(format!("models--{}", spec.repo_id.replace('/', "--")));
    if !model_dir.is_dir() {
        return None;
    }
    let commit = if is_hex_commit_sha(&spec.revision) {
        spec.revision.clone()
    } else {
        let mut ref_path = model_dir.join("refs");
        for seg in spec.revision.split('/') {
            if seg.is_empty() || seg == "." || seg == ".." {
                return None;
            }
            ref_path.push(seg);
        }
        let raw = fs::read_to_string(&ref_path).ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty()
            || trimmed.contains('/')
            || trimmed.contains('\\')
            || trimmed == "."
            || trimmed == ".."
        {
            return None;
        }
        trimmed.to_string()
    };

    let snap_dir = model_dir.join("snapshots").join(commit);
    if !snap_dir.is_dir() {
        return None;
    }

    let mut resolved = Vec::with_capacity(shards.len());
    for shard in shards {
        let mut p = snap_dir.clone();
        for seg in shard.filename.split('/') {
            if !seg.is_empty() {
                p.push(seg);
            }
        }
        if !is_complete_cached_file(&p) {
            return None;
        }
        resolved.push(p);
    }
    if resolved.is_empty() {
        None
    } else {
        Some(resolved)
    }
}

/// Looks up `spec` (and all its shards) in the user's standard Hugging Face Hub cache.
fn find_shards_in_hf_hub_cache(spec: &HfSpec, shards: &[HfSpec]) -> Option<Vec<PathBuf>> {
    for root in hf_hub_cache_roots() {
        if let Some(paths) = find_shards_in_hf_hub_root(spec, shards, &root) {
            return Some(paths);
        }
    }
    None
}

/// Returns `true` if `spec` is a local path OR an `hf://` URI whose shards are all cached.
pub fn is_path_or_hf_cached(spec: &str, cache_root: &Path) -> bool {
    if !spec.starts_with(HF_PREFIX) {
        return true;
    }
    let Ok(parsed) = HfSpec::parse(spec) else {
        return false;
    };
    let shards = parsed.expand_shards();
    if find_shards_in_hf_hub_cache(&parsed, &shards).is_some() {
        return true;
    }
    shards
        .iter()
        .all(|shard| is_complete_cached_file(&shard.cache_path(cache_root)))
}

/// Resolves a `--model` argument (`path`, `alias=path`, `hf://...`, or `alias=hf://...`).
/// Local paths pass through unchanged; `hf://` URIs are pulled into `cache_root` if missing.
pub fn resolve_model_spec(spec: &str, cache_root: &Path) -> Result<String> {
    if let Some((alias, target)) = spec.split_once('=') {
        if target.starts_with(HF_PREFIX) {
            let local = ensure_hf_downloaded(target, cache_root)?;
            return Ok(format!("{alias}={}", local.display()));
        }
        return Ok(spec.to_string());
    }

    if spec.starts_with(HF_PREFIX) {
        let local = ensure_hf_downloaded(spec, cache_root)?;
        return Ok(local.to_string_lossy().into_owned());
    }

    Ok(spec.to_string())
}

/// Resolves a `--draft-model` argument (local path or `hf://...` URI).
pub fn resolve_path_or_hf(spec: &str, cache_root: &Path) -> Result<PathBuf> {
    if spec.starts_with(HF_PREFIX) {
        ensure_hf_downloaded(spec, cache_root)
    } else {
        Ok(PathBuf::from(spec))
    }
}

/// Ensures the requested `hf://` GGUF (and all sibling shards if multi-part) exists in `cache_root`.
pub fn ensure_hf_downloaded(uri: &str, cache_root: &Path) -> Result<PathBuf> {
    ensure_hf_downloaded_with_progress(uri, cache_root, None)
}

/// Same as `ensure_hf_downloaded`, invoking `on_progress` on cache hit, periodic chunks, and completion.
pub fn ensure_hf_downloaded_with_progress(
    uri: &str,
    cache_root: &Path,
    on_progress: Option<&ProgressCallback>,
) -> Result<PathBuf> {
    let spec = HfSpec::parse(uri)?;
    let shards = spec.expand_shards();
    let shard_count = shards.len().max(1) as u32;

    if let Some(hub_paths) = find_shards_in_hf_hub_cache(&spec, &shards) {
        for (idx, (shard, hub_path)) in shards.iter().zip(hub_paths.iter()).enumerate() {
            let shard_index = (idx + 1) as u32;
            tracing::info!(
                uri = %format!("hf://{}:{}@{}", shard.repo_id, shard.filename, shard.revision),
                path = %hub_path.display(),
                "reusing Hugging Face Hub cache snapshot"
            );
            if let Some(cb) = on_progress {
                let sz = fs::metadata(hub_path).map(|m| m.len()).unwrap_or(0);
                cb(&DownloadProgress {
                    repo_id: shard.repo_id.clone(),
                    filename: shard.filename.clone(),
                    shard_index,
                    shard_count,
                    downloaded_bytes: sz,
                    total_bytes: Some(sz),
                    mib_per_s: 0.0,
                    cached: true,
                    done: true,
                })?;
            }
        }
        return hub_paths
            .into_iter()
            .next()
            .ok_or_else(|| HfError::DownloadFailed("empty shard expansion".into()));
    }

    let token = load_hf_token();
    let mut primary_path = None;

    for (idx, shard) in shards.iter().enumerate() {
        let shard_index = (idx + 1) as u32;
        let dest = shard.cache_path(cache_root);
        if idx == 0 {
            primary_path = Some(dest.clone());
        }
        if is_complete_cached_file(&dest) {
            tracing::info!(
                uri = %format!("hf://{}:{}@{}", shard.repo_id, shard.filename, shard.revision),
                path = %dest.display(),
                "using cached Hugging Face GGUF"
            );
            if let Some(cb) = on_progress {
                let sz = fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
                cb(&DownloadProgress {
                    repo_id: shard.repo_id.clone(),
                    filename: shard.filename.clone(),
                    shard_index,
                    shard_count,
                    downloaded_bytes: sz,
                    total_bytes: Some(sz),
                    mib_per_s: 0.0,
                    cached: true,
                    done: true,
                })?;
            }
            continue;
        }
        download_with_resume(
            shard,
            &dest,
            token.as_deref(),
            shard_index,
            shard_count,
            on_progress,
        )?;
    }

    primary_path.ok_or_else(|| HfError::DownloadFailed("empty shard expansion".into()))
}

fn is_complete_cached_file(path: &Path) -> bool {
    let part = part_path_for(path);
    if part.exists() {
        return false;
    }
    fs::metadata(path)
        .map(|m| m.is_file() && m.len() > 0)
        .unwrap_or(false)
}

fn part_path_for(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_os_string();
    s.push(".part");
    PathBuf::from(s)
}

fn meta_path_for(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_os_string();
    s.push(".part.meta");
    PathBuf::from(s)
}

fn lock_path_for(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_os_string();
    s.push(".part.lock");
    PathBuf::from(s)
}

struct PartLock {
    _file: File,
}

impl PartLock {
    fn acquire(lock_path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(lock_path)
            .map_err(|e| HfError::io(format!("open download lock {}", lock_path.display()), e))?;
        file.lock().map_err(|e| {
            HfError::io(
                format!("acquire exclusive lock on {}", lock_path.display()),
                e,
            )
        })?;
        Ok(Self { _file: file })
    }
}

#[derive(Debug, Default, Clone)]
struct PartMeta {
    etag: Option<String>,
    sha256: Option<String>,
}

fn read_part_meta(path: &Path) -> PartMeta {
    let Ok(text) = fs::read_to_string(path) else {
        return PartMeta::default();
    };
    let mut meta = PartMeta::default();
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("etag=") {
            let trimmed = v.trim();
            if !trimmed.is_empty() {
                meta.etag = Some(trimmed.to_string());
            }
        } else if let Some(v) = line.strip_prefix("sha256=") {
            let trimmed = v.trim();
            if !trimmed.is_empty() {
                meta.sha256 = Some(trimmed.to_string());
            }
        }
    }
    meta
}

fn write_part_meta(path: &Path, meta: &PartMeta) {
    let mut body = String::new();
    if let Some(etag) = &meta.etag {
        body.push_str(&format!("etag={etag}\n"));
    }
    if let Some(sha) = &meta.sha256 {
        body.push_str(&format!("sha256={sha}\n"));
    }
    let _ = fs::write(path, body);
}

fn extract_header_sha256(headers: &ureq::http::HeaderMap, key: &str) -> Option<String> {
    let raw = headers.get(key).and_then(|v| v.to_str().ok())?;
    let cleaned = raw
        .trim()
        .trim_start_matches("W/")
        .trim_matches('"')
        .to_ascii_lowercase();
    if cleaned.len() == 64 && cleaned.bytes().all(|b| b.is_ascii_hexdigit()) {
        Some(cleaned)
    } else {
        None
    }
}

fn url_origin(url: &str) -> Option<String> {
    let parsed = url.parse::<ureq::http::Uri>().ok()?;
    let scheme = parsed.scheme_str()?;
    let authority = parsed.authority()?.as_str();
    Some(format!("{scheme}://{authority}"))
}

fn resolve_redirect_url(base_url: &str, location: &str) -> Result<String> {
    let loc = location.trim();
    if loc.starts_with("http://") || loc.starts_with("https://") {
        return Ok(loc.to_string());
    }
    let origin = url_origin(base_url).ok_or_else(|| {
        HfError::Http(format!(
            "cannot resolve relative redirect '{loc}' against '{base_url}'"
        ))
    })?;
    if loc.starts_with('/') {
        Ok(format!("{origin}{loc}"))
    } else {
        Ok(format!("{origin}/{loc}"))
    }
}

fn extract_expected_sha256(
    headers: &ureq::http::HeaderMap,
    hub_redirected: bool,
    hub_sha256: Option<&str>,
) -> Option<String> {
    if let Some(sha) = hub_sha256 {
        return Some(sha.to_string());
    }
    if let Some(sha) = extract_header_sha256(headers, "x-linked-etag") {
        return Some(sha);
    }
    // Never treat a redirected CDN's ETag (e.g., HF Xet-bridge CAS hash) as file SHA-256.
    if !hub_redirected {
        return extract_header_sha256(headers, "etag");
    }
    None
}

fn extract_etag(headers: &ureq::http::HeaderMap) -> Option<String> {
    headers
        .get("etag")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn content_range_matches_start(headers: &ureq::http::HeaderMap, expected_start: u64) -> bool {
    let Some(raw) = headers.get("content-range").and_then(|v| v.to_str().ok()) else {
        return false;
    };
    let Some(rest) = raw.trim().strip_prefix("bytes ") else {
        return false;
    };
    let Some((start_s, _)) = rest.split_once('-') else {
        return false;
    };
    start_s.trim().parse::<u64>().ok() == Some(expected_start)
}

fn hash_file_prefix(path: &Path, limit: u64) -> Result<Sha256> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 256 * 1024];
    let mut rem = limit;
    while rem > 0 {
        let to_read = (rem as usize).min(buf.len());
        let n = f.read(&mut buf[..to_read])?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        rem -= n as u64;
    }
    Ok(hasher)
}

fn to_hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn load_hf_token() -> Option<String> {
    for var in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(tok) = std::env::var(var) {
            let trimmed = tok.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    let token_path = if let Ok(hf_home) = std::env::var("HF_HOME")
        && !hf_home.trim().is_empty()
    {
        PathBuf::from(hf_home).join("token")
    } else if let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("token")
    } else {
        return None;
    };
    fs::read_to_string(token_path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

std::thread_local! {
    static ACTIVE_THREAD_PART_LOCKS: std::cell::RefCell<std::collections::HashSet<PathBuf>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

struct ThreadPartLockGuard(PathBuf);

impl Drop for ThreadPartLockGuard {
    fn drop(&mut self) {
        ACTIVE_THREAD_PART_LOCKS.with(|set| {
            set.borrow_mut().remove(&self.0);
        });
    }
}

fn render_tty_progress_bar(ev: &DownloadProgress) -> bool {
    let fname = ev.filename.rsplit('/').next().unwrap_or(&ev.filename);
    let shard_tag = if ev.shard_count > 1 {
        format!("[{}/{}] ", ev.shard_index, ev.shard_count)
    } else {
        String::new()
    };
    let line = if let Some(total) = ev.total_bytes.filter(|&t| t > 0) {
        let frac = (ev.downloaded_bytes as f64 / total as f64).clamp(0.0, 1.0);
        let width = 20usize;
        let filled = ((frac * width as f64).round() as usize).min(width);
        let bar = format!("{}{}", "#".repeat(filled), "-".repeat(width - filled));
        format!(
            "\r\x1b[K[layerstream-hf] {shard_tag}{fname} [{bar}] {:5.1}% ({:.2}/{:.2} GiB, {:.1} MiB/s)",
            frac * 100.0,
            ev.downloaded_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            total as f64 / (1024.0 * 1024.0 * 1024.0),
            ev.mib_per_s,
        )
    } else {
        format!(
            "\r\x1b[K[layerstream-hf] {shard_tag}{fname} {:.2} GiB ({:.1} MiB/s)",
            ev.downloaded_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            ev.mib_per_s,
        )
    };
    let mut err = std::io::stderr();
    err.write_all(line.as_bytes()).is_ok() && err.flush().is_ok()
}

fn emit_download_progress(ev: &DownloadProgress, tty_stderr: bool) {
    if tty_stderr {
        let _ = render_tty_progress_bar(ev);
    } else if let Some(total) = ev.total_bytes {
        let pct = (ev.downloaded_bytes as f64 / total.max(1) as f64) * 100.0;
        eprintln!(
            "[layerstream-hf] {}: {:.2} / {:.2} GiB ({pct:.1}%, {:.1} MiB/s)",
            ev.filename,
            ev.downloaded_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            total as f64 / (1024.0 * 1024.0 * 1024.0),
            ev.mib_per_s,
        );
    } else {
        eprintln!(
            "[layerstream-hf] {}: {:.2} GiB ({:.1} MiB/s)",
            ev.filename,
            ev.downloaded_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            ev.mib_per_s,
        );
    }
}

fn download_with_resume(
    spec: &HfSpec,
    dest: &Path,
    token: Option<&str>,
    shard_index: u32,
    shard_count: u32,
    on_progress: Option<&ProgressCallback>,
) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| HfError::io(format!("create cache directory {}", parent.display()), e))?;
    }

    let lock_path = lock_path_for(dest);
    let lock_key = lock_path
        .parent()
        .and_then(|p| p.canonicalize().ok())
        .and_then(|p| lock_path.file_name().map(|name| p.join(name)))
        .unwrap_or_else(|| lock_path.clone());
    let inserted = ACTIVE_THREAD_PART_LOCKS.with(|set| set.borrow_mut().insert(lock_key.clone()));
    if !inserted {
        return Err(HfError::DownloadFailed(format!(
            "re-entrant download of '{}' from inside progress callback would deadlock",
            dest.display()
        )));
    }
    let _thread_reentry_guard = ThreadPartLockGuard(lock_key);
    let _lock = PartLock::acquire(&lock_path)?;

    // Another concurrent process may have completed the download while we waited for the lock.
    if is_complete_cached_file(dest) {
        if let Some(cb) = on_progress {
            let sz = fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
            cb(&DownloadProgress {
                repo_id: spec.repo_id.clone(),
                filename: spec.filename.clone(),
                shard_index,
                shard_count,
                downloaded_bytes: sz,
                total_bytes: Some(sz),
                mib_per_s: 0.0,
                cached: true,
                done: true,
            })?;
        }
        return Ok(());
    }

    let part_path = part_path_for(dest);
    let meta_path = meta_path_for(dest);
    let mut existing_len = fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
    let saved_meta = read_part_meta(&meta_path);

    // Never resume a partial file without a strong ETag or a trusted SHA-256 digest: weak ETags
    // (`W/"..."`) are omitted from `If-Range`, so resuming on a weak ETag alone without a digest
    // could silently splice bytes from two incompatible GGUF representations.
    let has_strong_etag = saved_meta
        .etag
        .as_deref()
        .is_some_and(|e| !e.trim_start().starts_with("W/"));
    let can_resume = existing_len > 0 && (has_strong_etag || saved_meta.sha256.is_some());
    let mut resume_etag = if can_resume {
        saved_meta.etag.clone()
    } else {
        if existing_len > 0 {
            let _ = fs::remove_file(&part_path);
            let _ = fs::remove_file(&meta_path);
            existing_len = 0;
        }
        None
    };

    let url = spec.download_url();
    let initial_origin = url_origin(&url);
    let read_timeout_secs = std::env::var("LAYERSTREAM_HF_READ_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_READ_TIMEOUT_SECS)
        .max(1);

    // Handle redirects manually so we capture the Hub's 302 `X-Linked-Etag` (authoritative
    // LFS SHA-256) before following redirects to CDNs such as Xet-bridge whose 200 `ETag`
    // is a 64-hex CAS identifier rather than the file's SHA-256.
    let agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .max_redirects(0)
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(read_timeout_secs)))
        .timeout_recv_body(Some(Duration::from_secs(read_timeout_secs)))
        .build()
        .new_agent();

    let send_get = |range_from: u64,
                    if_range: Option<&str>|
     -> Result<(ureq::http::Response<ureq::Body>, bool, Option<String>)> {
        let mut current_url = url.clone();
        let mut redirected = false;
        let mut hub_sha256: Option<String> = None;

        for _hop in 0..10 {
            let mut req = agent.get(&current_url).header(
                "User-Agent",
                concat!("layerstream-plugin/", env!("CARGO_PKG_VERSION")),
            );
            // Send Authorization only to the original Hub origin (never to redirected CDNs).
            if let Some(tok) = token
                && url_origin(&current_url) == initial_origin
            {
                req = req.header("Authorization", &format!("Bearer {tok}"));
            }
            if range_from > 0 {
                req = req.header("Range", &format!("bytes={range_from}-"));
                // RFC 9110 §13.1.5 forbids weak ETags (`W/"..."`) in `If-Range`.
                if let Some(etag) = if_range
                    && !etag.trim_start().starts_with("W/")
                {
                    req = req.header("If-Range", etag);
                }
            }
            let resp = req
                .call()
                .map_err(|e| HfError::Http(format!("HTTP GET failed for {current_url}: {e}")))?;
            let code = resp.status().as_u16();
            if matches!(code, 301 | 302 | 303 | 307 | 308)
                && let Some(loc) = resp.headers().get("location").and_then(|v| v.to_str().ok())
            {
                if hub_sha256.is_none() {
                    hub_sha256 = extract_header_sha256(resp.headers(), "x-linked-etag");
                }
                redirected = true;
                current_url = resolve_redirect_url(&current_url, loc)?;
                continue;
            }
            return Ok((resp, redirected, hub_sha256));
        }
        Err(HfError::Http(format!(
            "too many redirects downloading {url}"
        )))
    };

    let (mut resp, mut hub_redirected, mut hub_sha256) =
        send_get(existing_len, resume_etag.as_deref())?;
    if resp.status().as_u16() == 416 && existing_len > 0 {
        let _ = fs::remove_file(&part_path);
        let _ = fs::remove_file(&meta_path);
        existing_len = 0;
        resume_etag = None;
        (resp, hub_redirected, hub_sha256) = send_get(0, None)?;
    }

    let mut status = resp.status().as_u16();
    if status == 206
        && existing_len > 0
        && !content_range_matches_start(resp.headers(), existing_len)
    {
        // Server returned a mismatched Content-Range start offset: restart cleanly from byte 0.
        tracing::warn!(
            uri = %format!("hf://{}:{}", spec.repo_id, spec.filename),
            "mismatched Content-Range on resume; restarting download cleanly from byte 0"
        );
        let _ = fs::remove_file(&part_path);
        let _ = fs::remove_file(&meta_path);
        existing_len = 0;
        (resp, hub_redirected, hub_sha256) = send_get(0, None)?;
        status = resp.status().as_u16();
    }

    if (existing_len == 0 && status != 200) || (existing_len > 0 && status != 200 && status != 206)
    {
        return Err(HfError::Http(format!(
            "Hugging Face download failed for hf://{}:{} (HTTP {status} from {url})",
            spec.repo_id, spec.filename
        )));
    }

    let resp_etag = extract_etag(resp.headers());
    let expected_sha256 =
        extract_expected_sha256(resp.headers(), hub_redirected, hub_sha256.as_deref())
            .or(saved_meta.sha256.clone());

    // If the server returned 206 with a different ETag (ignoring If-Range), restart from byte 0!
    let mut resuming = status == 206 && existing_len > 0;
    if resuming
        && let (Some(prev), Some(curr)) = (&resume_etag, &resp_etag)
        && prev != curr
    {
        tracing::warn!(
            prev_etag = %prev,
            curr_etag = %curr,
            "ETag changed during resume; restarting download from byte 0 to prevent splice corruption"
        );
        let _ = fs::remove_file(&part_path);
        let _ = fs::remove_file(&meta_path);
        existing_len = 0;
        resuming = false;
        (resp, hub_redirected, hub_sha256) = send_get(0, None)?;
        status = resp.status().as_u16();
        if status != 200 {
            return Err(HfError::Http(format!(
                "Hugging Face restart download failed for hf://{}:{} (HTTP {status} from {url})",
                spec.repo_id, spec.filename
            )));
        }
    }

    let final_etag = if resuming {
        extract_etag(resp.headers()).or(resp_etag)
    } else {
        extract_etag(resp.headers())
    };
    let final_sha256 = if resuming {
        extract_expected_sha256(resp.headers(), hub_redirected, hub_sha256.as_deref())
            .or(expected_sha256)
    } else {
        extract_expected_sha256(resp.headers(), hub_redirected, hub_sha256.as_deref())
    };

    let content_len = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());
    let range_total = if resuming {
        resp.headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .and_then(|cr| cr.rsplit_once('/'))
            .and_then(|(_, tot)| tot.trim().parse::<u64>().ok())
    } else {
        None
    };
    let total_expected =
        range_total.or_else(|| content_len.map(|cl| if resuming { existing_len + cl } else { cl }));

    let (mut file, mut hasher): (File, Option<Sha256>) = if resuming {
        tracing::info!(
            uri = %format!("hf://{}:{}", spec.repo_id, spec.filename),
            resume_mib = existing_len / (1024 * 1024),
            "resuming Hugging Face GGUF download"
        );
        let h = if final_sha256.is_some() {
            Some(hash_file_prefix(&part_path, existing_len)?)
        } else {
            None
        };
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&part_path)
            .map_err(|e| HfError::io(format!("open {} for resume", part_path.display()), e))?;
        (f, h)
    } else {
        existing_len = 0;
        let _ = fs::remove_file(&meta_path);
        let f = File::create(&part_path)
            .map_err(|e| HfError::io(format!("create {}", part_path.display()), e))?;
        write_part_meta(
            &meta_path,
            &PartMeta {
                etag: final_etag.clone(),
                sha256: final_sha256.clone(),
            },
        );
        tracing::info!(
            uri = %format!("hf://{}:{}", spec.repo_id, spec.filename),
            total_mib = total_expected.map(|b| b / (1024 * 1024)),
            "downloading Hugging Face GGUF"
        );
        (f, final_sha256.as_ref().map(|_| Sha256::new()))
    };
    if resuming {
        write_part_meta(
            &meta_path,
            &PartMeta {
                etag: final_etag,
                sha256: final_sha256.clone(),
            },
        );
    }

    let mut reader = resp.into_body().into_reader();
    let mut buf = vec![0u8; 1024 * 1024];
    let mut downloaded = existing_len;
    let start = Instant::now();
    let mut last_log = Instant::now();
    let mut last_cb = Instant::now();
    let mut session_bytes = 0u64;
    let tty_stderr = on_progress.is_none()
        && std::io::stderr().is_terminal()
        && std::env::var("LAYERSTREAM_NO_PROGRESS_BAR").is_err();

    if let Some(cb) = on_progress {
        cb(&DownloadProgress {
            repo_id: spec.repo_id.clone(),
            filename: spec.filename.clone(),
            shard_index,
            shard_count,
            downloaded_bytes: downloaded,
            total_bytes: total_expected,
            mib_per_s: 0.0,
            cached: false,
            done: false,
        })?;
    }

    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| HfError::io(format!("network read error downloading {url}"), e))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| HfError::io(format!("write error for {}", part_path.display()), e))?;
        if let Some(h) = hasher.as_mut() {
            h.update(&buf[..n]);
        }
        downloaded += n as u64;
        session_bytes += n as u64;

        let log_interval = if tty_stderr {
            Duration::from_millis(120)
        } else {
            Duration::from_secs(2)
        };
        if let Some(cb) = on_progress {
            if last_cb.elapsed() >= Duration::from_millis(100) {
                let elapsed_s = start.elapsed().as_secs_f64().max(1e-3);
                let mib_s = (session_bytes as f64 / (1024.0 * 1024.0)) / elapsed_s;
                cb(&DownloadProgress {
                    repo_id: spec.repo_id.clone(),
                    filename: spec.filename.clone(),
                    shard_index,
                    shard_count,
                    downloaded_bytes: downloaded,
                    total_bytes: total_expected,
                    mib_per_s: mib_s,
                    cached: false,
                    done: false,
                })?;
                last_cb = Instant::now();
            }
        } else if last_log.elapsed() >= log_interval {
            let elapsed_s = start.elapsed().as_secs_f64().max(1e-3);
            let mib_s = (session_bytes as f64 / (1024.0 * 1024.0)) / elapsed_s;
            emit_download_progress(
                &DownloadProgress {
                    repo_id: spec.repo_id.clone(),
                    filename: spec.filename.clone(),
                    shard_index,
                    shard_count,
                    downloaded_bytes: downloaded,
                    total_bytes: total_expected,
                    mib_per_s: mib_s,
                    cached: false,
                    done: false,
                },
                tty_stderr,
            );
            last_log = Instant::now();
        }
    }

    file.flush()?;
    drop(file);

    if let Some(expected) = total_expected
        && downloaded != expected
    {
        if tty_stderr {
            eprintln!();
        }
        return Err(HfError::DownloadFailed(format!(
            "truncated download for hf://{}:{} (got {} bytes, expected {})",
            spec.repo_id, spec.filename, downloaded, expected
        )));
    }

    if let (Some(expected_hex), Some(h)) = (&final_sha256, hasher) {
        let actual_hex = to_hex_lower(&h.finalize());
        if actual_hex != *expected_hex {
            if tty_stderr {
                eprintln!();
            }
            let _ = fs::remove_file(&part_path);
            let _ = fs::remove_file(&meta_path);
            return Err(HfError::DownloadFailed(format!(
                "SHA-256 mismatch for hf://{}:{} (expected {}, got {}); removed corrupt partial file",
                spec.repo_id, spec.filename, expected_hex, actual_hex
            )));
        }
    }

    fs::rename(&part_path, dest).map_err(|e| {
        HfError::io(
            format!(
                "rename completed download {} -> {}",
                part_path.display(),
                dest.display()
            ),
            e,
        )
    })?;
    let _ = fs::remove_file(&meta_path);

    let elapsed_s = start.elapsed().as_secs_f64().max(1e-3);
    let final_mib_s = (session_bytes as f64 / (1024.0 * 1024.0)) / elapsed_s;
    if let Some(cb) = on_progress {
        cb(&DownloadProgress {
            repo_id: spec.repo_id.clone(),
            filename: spec.filename.clone(),
            shard_index,
            shard_count,
            downloaded_bytes: downloaded,
            total_bytes: Some(downloaded),
            mib_per_s: final_mib_s,
            cached: false,
            done: true,
        })?;
    } else if tty_stderr {
        eprintln!(
            "\r\x1b[K[layerstream-hf] ready: {} ({:.2} GiB, {final_mib_s:.1} MiB/s)",
            dest.display(),
            downloaded as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else {
        eprintln!(
            "[layerstream-hf] ready: {} ({:.2} GiB)",
            dest.display(),
            downloaded as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    Ok(())
}

mod hub;
pub use hub::*;

#[cfg(test)]
mod tests;
