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
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
            bail!("not an hf:// URI: {uri}");
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
                bail!("invalid hf:// spec '{uri}'; expected hf://org/repo:filename.gguf");
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
            bail!("invalid Hugging Face repo id '{repo_id}' in '{uri}' (expected org/repo)");
        }
        if filename.is_empty()
            || filename.contains("..")
            || filename.contains('\\')
            || filename.contains(':')
            || filename
                .split('/')
                .any(|seg| seg.is_empty() || seg == "." || seg == "..")
        {
            bail!("invalid Hugging Face filename '{filename}' in '{uri}'");
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
        let mut p = cache_root.join(format!("{repo_dir}@{rev_dir}"));
        for seg in self.filename.split('/') {
            if !seg.is_empty() {
                p.push(seg);
            }
        }
        p
    }

    /// Expands multi-part shard specs `*-00001-of-0000N.gguf` into all N shard specs.
    pub fn expand_shards(&self) -> Vec<Self> {
        let (dir_prefix, file_name) = match self.filename.rsplit_once('/') {
            Some((d, f)) => (Some(d), f),
            None => (None, self.filename.as_str()),
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
                    let full_name = match dir_prefix {
                        Some(d) => format!("{d}/{shard_file}"),
                        None => shard_file,
                    };
                    Self {
                        repo_id: self.repo_id.clone(),
                        filename: full_name,
                        revision: self.revision.clone(),
                    }
                })
                .collect();
        }

        vec![self.clone()]
    }
}

// tracking: Mesh-LLM/mesh-llm#3192 (mesh-llm 0.80.0 does not yet pass a host model/cache
// directory env var to spawned plugins, so fall back to LAYERSTREAM_CACHE_DIR / XDG cache).
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

/// Returns `true` if `spec` is a local path OR an `hf://` URI whose shards are all cached.
pub fn is_path_or_hf_cached(spec: &str, cache_root: &Path) -> bool {
    if !spec.starts_with(HF_PREFIX) {
        return true;
    }
    let Ok(parsed) = HfSpec::parse(spec) else {
        return false;
    };
    parsed
        .expand_shards()
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
    let spec = HfSpec::parse(uri)?;
    let shards = spec.expand_shards();
    let token = load_hf_token();
    let mut primary_path = None;

    for (idx, shard) in shards.iter().enumerate() {
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
            continue;
        }
        download_with_resume(shard, &dest, token.as_deref())?;
    }

    primary_path.context("empty shard expansion")
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
    #[cfg(not(unix))]
    path: PathBuf,
}

impl PartLock {
    fn acquire(lock_path: &Path) -> Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(lock_path)
                .with_context(|| format!("open download lock {}", lock_path.display()))?;
            let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
            if rc != 0 {
                bail!(
                    "failed to acquire flock on {}: {}",
                    lock_path.display(),
                    std::io::Error::last_os_error()
                );
            }
            Ok(Self { _file: file })
        }
        #[cfg(not(unix))]
        {
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(false)
                .open(lock_path)
                .with_context(|| format!("open download lock {}", lock_path.display()))?;
            Ok(Self {
                _file: file,
                path: lock_path.to_path_buf(),
            })
        }
    }
}

#[cfg(not(unix))]
impl Drop for PartLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
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
        anyhow::anyhow!("cannot resolve relative redirect '{loc}' against '{base_url}'")
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

fn download_with_resume(spec: &HfSpec, dest: &Path, token: Option<&str>) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create cache directory {}", parent.display()))?;
    }

    let lock_path = lock_path_for(dest);
    let _lock = PartLock::acquire(&lock_path)?;

    // Another concurrent process may have completed the download while we waited for the lock.
    if is_complete_cached_file(dest) {
        return Ok(());
    }

    let part_path = part_path_for(dest);
    let meta_path = meta_path_for(dest);
    let mut existing_len = fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
    let saved_meta = read_part_meta(&meta_path);

    // Never resume a partial file without a matching saved ETag: if the upstream file changed
    // between runs, resuming blindly would silently splice two incompatible GGUF versions.
    let mut resume_etag = if existing_len > 0 && saved_meta.etag.is_some() {
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
                .with_context(|| format!("HTTP GET failed for {current_url}"))?;
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
        bail!("too many redirects downloading {url}")
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

    if status != 200 && status != 206 {
        bail!(
            "Hugging Face download failed for hf://{}:{} (HTTP {status} from {url})",
            spec.repo_id,
            spec.filename
        );
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
            bail!(
                "Hugging Face restart download failed for hf://{}:{} (HTTP {status} from {url})",
                spec.repo_id,
                spec.filename
            );
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
            .with_context(|| format!("open {} for resume", part_path.display()))?;
        (f, h)
    } else {
        existing_len = 0;
        let _ = fs::remove_file(&meta_path);
        let f =
            File::create(&part_path).with_context(|| format!("create {}", part_path.display()))?;
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
    let mut session_bytes = 0u64;

    loop {
        let n = reader
            .read(&mut buf)
            .with_context(|| format!("network read error downloading {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .with_context(|| format!("write error for {}", part_path.display()))?;
        if let Some(h) = hasher.as_mut() {
            h.update(&buf[..n]);
        }
        downloaded += n as u64;
        session_bytes += n as u64;

        if last_log.elapsed() >= Duration::from_secs(2) {
            let elapsed_s = start.elapsed().as_secs_f64().max(1e-3);
            let mib_s = (session_bytes as f64 / (1024.0 * 1024.0)) / elapsed_s;
            if let Some(total) = total_expected {
                let pct = (downloaded as f64 / total.max(1) as f64) * 100.0;
                eprintln!(
                    "[layerstream-hf] {}: {:.2} / {:.2} GiB ({pct:.1}%, {mib_s:.1} MiB/s)",
                    spec.filename,
                    downloaded as f64 / (1024.0 * 1024.0 * 1024.0),
                    total as f64 / (1024.0 * 1024.0 * 1024.0),
                );
            } else {
                eprintln!(
                    "[layerstream-hf] {}: {:.2} GiB ({mib_s:.1} MiB/s)",
                    spec.filename,
                    downloaded as f64 / (1024.0 * 1024.0 * 1024.0),
                );
            }
            last_log = Instant::now();
        }
    }

    file.flush()?;
    drop(file);

    if let Some(expected) = total_expected
        && downloaded != expected
    {
        bail!(
            "truncated download for hf://{}:{} (got {} bytes, expected {})",
            spec.repo_id,
            spec.filename,
            downloaded,
            expected
        );
    }

    if let (Some(expected_hex), Some(h)) = (&final_sha256, hasher) {
        let actual_hex = to_hex_lower(&h.finalize());
        if actual_hex != *expected_hex {
            let _ = fs::remove_file(&part_path);
            let _ = fs::remove_file(&meta_path);
            bail!(
                "SHA-256 mismatch for hf://{}:{} (expected {}, got {}); removed corrupt partial file",
                spec.repo_id,
                spec.filename,
                expected_hex,
                actual_hex
            );
        }
    }

    fs::rename(&part_path, dest).with_context(|| {
        format!(
            "rename completed download {} -> {}",
            part_path.display(),
            dest.display()
        )
    })?;
    let _ = fs::remove_file(&meta_path);

    eprintln!(
        "[layerstream-hf] ready: {} ({:.2} GiB)",
        dest.display(),
        downloaded as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}

/// Parses a Hugging Face repository reference (`org/repo`, `org/repo@rev`,
/// `hf://org/repo[@rev]`, or `https://huggingface.co/org/repo`) into `(repo_id, revision)`.
pub fn parse_repo_and_revision(
    input: &str,
    rev_override: Option<&str>,
) -> Result<(String, String)> {
    let trimmed = input.trim();
    let stripped = trimmed
        .strip_prefix(HF_PREFIX)
        .or_else(|| trimmed.strip_prefix("https://huggingface.co/"))
        .or_else(|| trimmed.strip_prefix("http://huggingface.co/"))
        .unwrap_or(trimmed)
        .trim_matches('/');

    let (body, at_rev) = match stripped.rsplit_once('@') {
        Some((b, r)) if !r.is_empty() && !r.contains('/') => (b, Some(r.to_string())),
        _ => (stripped, None),
    };

    let mut parts = body.split('/');
    let org = parts.next().unwrap_or("");
    let repo = parts.next().unwrap_or("");
    if org.is_empty() || repo.is_empty() || org.contains("..") || repo.contains("..") {
        bail!("invalid Hugging Face repo '{input}' (expected org/repo)");
    }
    let rev = rev_override
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string)
        .or(at_rev)
        .unwrap_or_else(|| "main".to_string());
    Ok((format!("{org}/{repo}"), rev))
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfTreeEntry {
    #[serde(rename = "type")]
    pub entry_type: Option<String>,
    pub path: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub lfs: Option<HfLfsInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfLfsInfo {
    #[serde(default)]
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct HfGgufCandidate {
    pub primary_file: String,
    pub quant: String,
    pub total_bytes: u64,
    pub shard_count: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantRecommendation {
    pub candidate: HfGgufCandidate,
    pub hf_uri: String,
    pub verdict: String,
    pub pinned_pct: u8,
    pub est_load_s: f64,
    pub est_tps_low: f64,
    pub est_tps_high: f64,
    pub recommended: bool,
    pub score: i32,
}

fn extract_quant_label(path: &str) -> String {
    let fname = path.rsplit('/').next().unwrap_or(path);
    let stem = fname.strip_suffix(".gguf").unwrap_or(fname);
    // Strip trailing shard suffix `-00001-of-0000N` before scanning tokens.
    let clean = strip_shard_suffix(stem).map(|(p, _, _)| p).unwrap_or(stem);
    let upper = clean.to_ascii_uppercase();
    // Match well-known GGUF quant identifiers from most specific to least specific.
    const KNOWN: &[&str] = &[
        "UD-IQ1_S",
        "UD-IQ1_M",
        "UD-IQ2_XXS",
        "UD-IQ2_XS",
        "UD-IQ2_M",
        "UD-IQ3_XXS",
        "UD-Q2_K_XL",
        "UD-Q3_K_XL",
        "UD-Q4_K_XL",
        "UD-Q5_K_XL",
        "UD-Q6_K_XL",
        "UD-Q8_K_XL",
        "IQ1_S",
        "IQ1_M",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ2_M",
        "IQ3_XXS",
        "IQ3_XS",
        "IQ3_S",
        "IQ3_M",
        "IQ4_XS",
        "IQ4_NL",
        "Q2_K_XL",
        "Q2_K_L",
        "Q2_K_S",
        "Q2_K",
        "Q3_K_XL",
        "Q3_K_L",
        "Q3_K_M",
        "Q3_K_S",
        "Q4_K_XL",
        "Q4_K_L",
        "Q4_K_M",
        "Q4_K_S",
        "Q4_1",
        "Q4_0",
        "Q5_K_XL",
        "Q5_K_L",
        "Q5_K_M",
        "Q5_K_S",
        "Q5_1",
        "Q5_0",
        "Q6_K_XL",
        "Q6_K_L",
        "Q6_K",
        "Q8_K_XL",
        "Q8_0",
        "MXFP4",
        "BF16",
        "FP16",
        "F16",
        "F32",
    ];
    for q in KNOWN {
        if upper.contains(q) {
            return (*q).to_string();
        }
    }
    clean
        .rsplit(['-', '_', '.'])
        .next()
        .unwrap_or("GGUF")
        .to_ascii_uppercase()
}

fn strip_shard_suffix(stem: &str) -> Option<(&str, u32, u32)> {
    let (prefix_and_idx, total_str) = stem.rsplit_once("-of-")?;
    if total_str.len() != 5 {
        return None;
    }
    let total: u32 = total_str.parse().ok()?;
    if total <= 1 {
        return None;
    }
    let (prefix, idx_str) = prefix_and_idx.rsplit_once('-')?;
    if idx_str.len() != 5 {
        return None;
    }
    let idx: u32 = idx_str.parse().ok()?;
    Some((prefix, idx, total))
}

fn quant_quality_rank(quant: &str) -> i32 {
    let (ud_bonus, base) = match quant.strip_prefix("UD-") {
        Some(rest) => (3, rest),
        None => (0, quant),
    };
    let rank = match base {
        "Q4_K_M" | "Q4_K_XL" => 100,
        "Q5_K_M" | "Q5_K_L" | "Q5_K_XL" => 98,
        "IQ4_XS" | "IQ4_NL" | "Q4_K_S" | "Q4_K_L" => 95,
        "Q6_K" | "Q6_K_L" | "Q6_K_XL" => 94,
        "Q8_0" | "Q8_K_XL" => 92,
        "Q5_K_S" | "Q5_0" | "Q5_1" | "Q4_0" | "Q4_1" => 88,
        "Q3_K_M" | "Q3_K_L" | "Q3_K_XL" => 80,
        "IQ3_M" | "IQ3_S" | "Q3_K_S" => 74,
        "IQ3_XS" | "IQ3_XXS" => 66,
        "Q2_K_XL" | "Q2_K_L" => 62,
        "Q2_K" | "Q2_K_S" | "IQ2_M" => 58,
        "BF16" | "FP16" | "F16" => 52, // huge footprint penalty on edge GPUs
        _ => 50,
    };
    rank + ud_bonus
}

/// Groups Hugging Face tree entries into logical GGUF checkpoints, combining multi-part shards
/// (`*-00001-of-0000N.gguf`) and filtering out `mmproj` vision projector files.
pub fn group_gguf_tree_entries(entries: &[HfTreeEntry]) -> Vec<HfGgufCandidate> {
    let mut grouped: BTreeMap<String, (String, u64, u32)> = BTreeMap::new();
    for entry in entries {
        if entry.entry_type.as_deref().is_some_and(|t| t != "file") {
            continue;
        }
        let lower = entry.path.to_ascii_lowercase();
        if !lower.ends_with(".gguf") || lower.contains("mmproj") {
            continue;
        }
        let bytes = entry
            .lfs
            .as_ref()
            .map(|l| l.size)
            .filter(|&s| s > 0)
            .unwrap_or(entry.size);
        let (dir_prefix, fname) = match entry.path.rsplit_once('/') {
            Some((d, f)) => (Some(d), f),
            None => (None, entry.path.as_str()),
        };
        let stem = fname.strip_suffix(".gguf").unwrap_or(fname);
        if let Some((prefix, idx, total)) = strip_shard_suffix(stem) {
            let key = match dir_prefix {
                Some(d) => format!("{d}/{prefix}"),
                None => prefix.to_string(),
            };
            let shard1_name = format!("{prefix}-00001-of-{total:05}.gguf");
            let primary = match dir_prefix {
                Some(d) => format!("{d}/{shard1_name}"),
                None => shard1_name,
            };
            let slot = grouped.entry(key).or_insert((primary, 0, total));
            slot.1 = slot.1.saturating_add(bytes);
            if idx == 1 {
                slot.0 = entry.path.clone();
            }
        } else {
            grouped.insert(entry.path.clone(), (entry.path.clone(), bytes, 1));
        }
    }

    let mut out: Vec<HfGgufCandidate> = grouped
        .into_values()
        .map(|(primary_file, total_bytes, shard_count)| HfGgufCandidate {
            quant: extract_quant_label(&primary_file),
            primary_file,
            total_bytes,
            shard_count,
        })
        .collect();
    out.sort_by_key(|c| c.total_bytes);
    out
}

/// Fetches the recursive file tree of a Hugging Face repo and returns logical GGUF checkpoints.
pub fn list_hf_gguf_candidates(repo_id: &str, revision: &str) -> Result<Vec<HfGgufCandidate>> {
    let endpoint = std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_HF_ENDPOINT.to_string());
    let url = format!(
        "{}/api/models/{}/tree/{}?recursive=true",
        endpoint.trim_end_matches('/'),
        repo_id,
        revision
    );

    let agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_body(Some(Duration::from_secs(30)))
        .http_status_as_error(false)
        .build()
        .new_agent();

    let mut req = agent.get(&url).header("User-Agent", "layerstream-plugin");
    if let Some(tok) = load_hf_token() {
        req = req.header("Authorization", &format!("Bearer {tok}"));
    }

    let resp = req
        .call()
        .with_context(|| format!("query Hugging Face API {url}"))?;
    let status = resp.status().as_u16();
    if status != 200 {
        bail!("Hugging Face API returned HTTP {status} for {repo_id}@{revision} ({url})");
    }
    let body = resp
        .into_body()
        .read_to_string()
        .context("read Hugging Face tree JSON")?;
    let entries: Vec<HfTreeEntry> =
        serde_json::from_str(&body).context("parse Hugging Face tree JSON")?;
    Ok(group_gguf_tree_entries(&entries))
}

/// Evaluates and ranks GGUF candidates against probed (or user-specified) GPU/UMA memory
/// and sustained NVMe `O_DIRECT` bandwidth.
pub fn recommend_candidates(
    repo_id: &str,
    revision: &str,
    candidates: &[HfGgufCandidate],
    free_vram_bytes: u64,
    nvme_mib_s: f64,
    has_draft: bool,
) -> Vec<QuantRecommendation> {
    let free_gib = free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let nvme_gib_s = (nvme_mib_s / 1024.0).max(0.05);
    let draft_reserve = if has_draft { 0.65 } else { 0.0 };
    let headroom = (free_gib * 0.14).max(1.0);

    let mut recs: Vec<QuantRecommendation> = candidates
        .iter()
        .map(|cand| {
            let model_gib = cand.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let kv_and_slots = (model_gib * 0.08).clamp(0.8, 2.2);
            let usable = (free_gib - headroom - kv_and_slots - draft_reserve).max(0.0);
            let fits_resident = model_gib > 0.0 && model_gib <= usable;
            let pinned_frac = if fits_resident {
                1.0
            } else if model_gib > 0.0 {
                (usable / model_gib).clamp(0.0, 0.85)
            } else {
                0.0
            };
            let streamed_gib = (model_gib * (1.0 - pinned_frac)).max(0.0);
            let est_load_s = model_gib.min(usable + 0.8) / nvme_gib_s;

            let (verdict, est_tps_low, est_tps_high, fit_bonus) = if fits_resident {
                ("RESIDENT".to_string(), 15.0, 50.0, 200)
            } else if nvme_mib_s < 150.0 {
                ("TOO_SLOW_DISK".to_string(), 0.01, 0.05, -200)
            } else {
                let sweep_s = ((streamed_gib / nvme_gib_s) * 1.15).max(0.15);
                let mult_lo = if has_draft { 2.4 } else { 1.0 };
                let mult_hi = if has_draft { 5.2 } else { 1.0 };
                let lo = mult_lo / sweep_s;
                let hi = mult_hi / sweep_s;
                if lo >= 1.0 {
                    ("GOOD".to_string(), lo, hi, 140)
                } else if hi >= 0.5 {
                    ("SLOW".to_string(), lo, hi, 60)
                } else {
                    ("VERY_SLOW".to_string(), lo, hi, -40)
                }
            };

            let hf_uri = if revision == "main" {
                format!("hf://{repo_id}:{}", cand.primary_file)
            } else {
                format!("hf://{repo_id}:{}@{revision}", cand.primary_file)
            };

            let score = fit_bonus + quant_quality_rank(&cand.quant);
            QuantRecommendation {
                candidate: cand.clone(),
                hf_uri,
                verdict,
                pinned_pct: (pinned_frac * 100.0).round() as u8,
                est_load_s: (est_load_s * 10.0).round() / 10.0,
                est_tps_low: (est_tps_low * 100.0).round() / 100.0,
                est_tps_high: (est_tps_high * 100.0).round() / 100.0,
                recommended: false,
                score,
            }
        })
        .collect();

    if let Some((best_idx, _)) = recs
        .iter()
        .enumerate()
        .max_by_key(|(_, r)| (r.score, r.candidate.total_bytes))
    {
        recs[best_idx].recommended = true;
    }
    recs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn parses_colon_slash_and_revision_specs() {
        let s1 = HfSpec::parse(
            "hf://bartowski/Qwen2.5-14B-Instruct-GGUF:Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        )
        .unwrap();
        assert_eq!(s1.repo_id, "bartowski/Qwen2.5-14B-Instruct-GGUF");
        assert_eq!(s1.filename, "Qwen2.5-14B-Instruct-Q4_K_M.gguf");
        assert_eq!(s1.revision, "main");
        assert!(s1.download_url().ends_with(
            "/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
        ));

        let s2 = HfSpec::parse("hf://unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M/model.gguf@v1.0").unwrap();
        assert_eq!(s2.repo_id, "unsloth/Qwen3-30B-A3B-GGUF");
        assert_eq!(s2.filename, "Q4_K_M/model.gguf");
        assert_eq!(s2.revision, "v1.0");

        let s3 = HfSpec::parse("hf://org/repo/model.gguf").unwrap();
        assert_eq!(s3.repo_id, "org/repo");
        assert_eq!(s3.filename, "model.gguf");
    }

    #[test]
    fn expands_multipart_gguf_shards() {
        let s = HfSpec::parse("hf://org/big-gguf:Q4_K_M/big-00001-of-00003.gguf").unwrap();
        let shards = s.expand_shards();
        assert_eq!(shards.len(), 3);
        assert_eq!(shards[0].filename, "Q4_K_M/big-00001-of-00003.gguf");
        assert_eq!(shards[1].filename, "Q4_K_M/big-00002-of-00003.gguf");
        assert_eq!(shards[2].filename, "Q4_K_M/big-00003-of-00003.gguf");
    }

    #[test]
    fn rejects_path_traversal_and_bad_repo() {
        assert!(HfSpec::parse("hf://badrepo:file.gguf").is_err());
        assert!(HfSpec::parse("hf://org/repo:../escape.gguf").is_err());
        assert!(HfSpec::parse("/local/path.gguf").is_err());
    }

    #[test]
    fn revision_cache_paths_are_isolated() {
        let tmp = Path::new("/tmp/ls-cache");
        let a = HfSpec::parse("hf://org/repo:model.gguf@v1").unwrap();
        let b = HfSpec::parse("hf://org/repo:model.gguf@v2").unwrap();
        let m = HfSpec::parse("hf://org/repo:model.gguf").unwrap();
        assert_ne!(a.cache_path(tmp), b.cache_path(tmp));
        assert_ne!(a.cache_path(tmp), m.cache_path(tmp));
    }

    #[test]
    fn resumable_range_download_etag_sha256_and_offline_cache_hit() {
        let _env_guard = ENV_LOCK.lock().unwrap();
        use std::io::{BufRead, BufReader};
        use std::net::TcpListener;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let payload_v1: Vec<u8> = (0..32_768).map(|i| ((i * 31 + 7) % 251) as u8).collect();
        let payload_v2: Vec<u8> = (0..32_768).map(|i| ((i * 17 + 3) % 241) as u8).collect();
        let sha_v1 = to_hex_lower(&Sha256::digest(&payload_v1));
        let sha_v2 = to_hex_lower(&Sha256::digest(&payload_v2));
        let etag_v1 = format!("\"{sha_v1}\"");
        let etag_v2 = format!("\"{sha_v2}\"");

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_bg = hits.clone();
        let pv1 = payload_v1.clone();
        let pv2 = payload_v2.clone();
        let ev1 = etag_v1.clone();
        let ev2 = etag_v2.clone();

        // Request sequence:
        // 1: Range + matching If-Range(v1) -> 206 Partial Content of v1
        // 2: Range + oversized offset -> 416
        // 3: Full GET after 416 -> 200 OK of v1
        // 4: Range + stale If-Range(v1) while server now serves v2 -> 200 OK of v2 (no splice!)
        let server = std::thread::spawn(move || {
            for (req_no, stream) in listener.incoming().take(4).enumerate() {
                let Ok(mut sock) = stream else { break };
                hits_bg.fetch_add(1, Ordering::SeqCst);
                let mut reader = BufReader::new(sock.try_clone().unwrap());
                let mut range_start: Option<usize> = None;
                let mut if_range: Option<String> = None;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    let lower = line.to_ascii_lowercase();
                    if let Some(rest) = lower.strip_prefix("range: bytes=")
                        && let Some((start_s, _)) = rest.trim().split_once('-')
                        && let Ok(s) = start_s.parse::<usize>()
                    {
                        range_start = Some(s);
                    } else if let Some(rest) = lower.strip_prefix("if-range:") {
                        if_range = Some(rest.trim().to_string());
                    }
                }
                let (active_payload, active_etag) = if req_no < 3 {
                    (&pv1, &ev1)
                } else {
                    (&pv2, &ev2)
                };
                let etag_matches = if_range.as_deref() == Some(active_etag.as_str());
                match range_start {
                    Some(start) if start < active_payload.len() && etag_matches => {
                        let slice = &active_payload[start..];
                        let header = format!(
                            "HTTP/1.1 206 Partial Content\r\n\
                             ETag: {active_etag}\r\n\
                             Content-Range: bytes {}-{}/{}\r\n\
                             Content-Length: {}\r\n\
                             Connection: close\r\n\r\n",
                            start,
                            active_payload.len() - 1,
                            active_payload.len(),
                            slice.len()
                        );
                        sock.write_all(header.as_bytes()).unwrap();
                        sock.write_all(slice).unwrap();
                    }
                    Some(start) if start >= active_payload.len() && etag_matches => {
                        let header = "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                        sock.write_all(header.as_bytes()).unwrap();
                    }
                    _ => {
                        let header = format!(
                            "HTTP/1.1 200 OK\r\n\
                             ETag: {active_etag}\r\n\
                             Content-Length: {}\r\n\
                             Connection: close\r\n\r\n",
                            active_payload.len()
                        );
                        sock.write_all(header.as_bytes()).unwrap();
                        sock.write_all(active_payload).unwrap();
                    }
                }
                let _ = sock.flush();
            }
        });

        let tmp = std::env::temp_dir().join(format!("layerstream-hf-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        unsafe { std::env::set_var("HF_ENDPOINT", format!("http://{addr}")) };

        // 1. Pre-populate partial `.part` + matching `.part.meta` ETag -> verifies 206 resume + SHA-256.
        let uri = "hf://testorg/testrepo:model.gguf";
        let spec = HfSpec::parse(uri).unwrap();
        let dest = spec.cache_path(&tmp);
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(part_path_for(&dest), &payload_v1[..4096]).unwrap();
        write_part_meta(
            &meta_path_for(&dest),
            &PartMeta {
                etag: Some(etag_v1.clone()),
                sha256: Some(sha_v1.clone()),
            },
        );

        let local = ensure_hf_downloaded(uri, &tmp).unwrap();
        assert_eq!(local, dest);
        assert_eq!(fs::read(&local).unwrap(), payload_v1);
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        // 2. Second call must be a 0-network cache hit.
        let local2 = ensure_hf_downloaded(uri, &tmp).unwrap();
        assert_eq!(local2, dest);
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        // 3. Oversized stale `.part` with matching ETag triggers 416 recovery + clean download (2 requests).
        fs::remove_file(&dest).unwrap();
        fs::write(part_path_for(&dest), vec![0u8; payload_v1.len() + 1024]).unwrap();
        write_part_meta(
            &meta_path_for(&dest),
            &PartMeta {
                etag: Some(etag_v1.clone()),
                sha256: Some(sha_v1.clone()),
            },
        );
        let local3 = ensure_hf_downloaded(uri, &tmp).unwrap();
        assert_eq!(fs::read(&local3).unwrap(), payload_v1);
        assert_eq!(hits.load(Ordering::SeqCst), 3);

        // 4. Stale `.part` (v1 prefix + v1 ETag) against updated upstream v2 -> restarts cleanly (NEVER splices v1+v2!).
        fs::remove_file(&dest).unwrap();
        fs::write(part_path_for(&dest), &payload_v1[..4096]).unwrap();
        write_part_meta(
            &meta_path_for(&dest),
            &PartMeta {
                etag: Some(etag_v1),
                sha256: Some(sha_v1),
            },
        );
        let local4 = ensure_hf_downloaded(uri, &tmp).unwrap();
        assert_eq!(fs::read(&local4).unwrap(), payload_v2);
        assert_eq!(hits.load(Ordering::SeqCst), 4);

        unsafe { std::env::remove_var("HF_ENDPOINT") };
        let _ = fs::remove_dir_all(&tmp);
        server.join().unwrap();
    }

    #[test]
    fn stalls_time_out_cleanly() {
        let _env_guard = ENV_LOCK.lock().unwrap();
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut req = [0u8; 1024];
                let _ = sock.read(&mut req);
                let _ = sock.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 1048576\r\nConnection: close\r\n\r\n",
                );
                let _ = sock.flush();
                std::thread::sleep(Duration::from_secs(3));
            }
        });

        let tmp = std::env::temp_dir().join(format!("layerstream-hf-stall-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        unsafe {
            std::env::set_var("HF_ENDPOINT", format!("http://{addr}"));
            std::env::set_var("LAYERSTREAM_HF_READ_TIMEOUT_SECS", "1");
        }
        let err = ensure_hf_downloaded("hf://stallorg/stallrepo:model.gguf", &tmp);
        unsafe {
            std::env::remove_var("HF_ENDPOINT");
            std::env::remove_var("LAYERSTREAM_HF_READ_TIMEOUT_SECS");
        }
        assert!(err.is_err(), "stalled download must time out");
        let _ = fs::remove_dir_all(&tmp);
        let _ = server.join();
    }

    #[test]
    fn parses_repo_and_revision_forms() {
        assert_eq!(
            parse_repo_and_revision("Qwen/Qwen2.5-14B-Instruct-GGUF", None).unwrap(),
            ("Qwen/Qwen2.5-14B-Instruct-GGUF".into(), "main".into())
        );
        assert_eq!(
            parse_repo_and_revision("hf://bartowski/Llama-3-GGUF@v2", None).unwrap(),
            ("bartowski/Llama-3-GGUF".into(), "v2".into())
        );
        assert_eq!(
            parse_repo_and_revision("https://huggingface.co/unsloth/Qwen3-GGUF/", Some("rev1"))
                .unwrap(),
            ("unsloth/Qwen3-GGUF".into(), "rev1".into())
        );
        assert!(parse_repo_and_revision("invalid-no-slash", None).is_err());
    }

    #[test]
    fn groups_shards_skips_mmproj_and_ranks_quants() {
        let gib = 1024u64 * 1024 * 1024;
        let entries = vec![
            HfTreeEntry {
                entry_type: Some("file".into()),
                path: "mmproj-model-f16.gguf".into(),
                size: gib,
                lfs: None,
            },
            HfTreeEntry {
                entry_type: Some("file".into()),
                path: "model-q4_k_m-00001-of-00003.gguf".into(),
                size: 0,
                lfs: Some(HfLfsInfo { size: 3 * gib }),
            },
            HfTreeEntry {
                entry_type: Some("file".into()),
                path: "model-q4_k_m-00002-of-00003.gguf".into(),
                size: 0,
                lfs: Some(HfLfsInfo { size: 3 * gib }),
            },
            HfTreeEntry {
                entry_type: Some("file".into()),
                path: "model-q4_k_m-00003-of-00003.gguf".into(),
                size: 0,
                lfs: Some(HfLfsInfo { size: 2 * gib }),
            },
            HfTreeEntry {
                entry_type: Some("file".into()),
                path: "model-q8_0.gguf".into(),
                size: 16 * gib,
                lfs: None,
            },
        ];

        let cands = group_gguf_tree_entries(&entries);
        assert_eq!(cands.len(), 2, "mmproj skipped and 3 shards grouped");
        assert_eq!(cands[0].quant, "Q4_K_M");
        assert_eq!(cands[0].shard_count, 3);
        assert_eq!(cands[0].total_bytes, 8 * gib);
        assert_eq!(cands[0].primary_file, "model-q4_k_m-00001-of-00003.gguf");

        // On an 8 GB Jetson (5.2 GiB free, 2800 MiB/s), 8 GiB Q4_K_M should beat 16 GiB Q8_0.
        let recs = recommend_candidates(
            "Qwen/Qwen2.5-14B-Instruct-GGUF",
            "main",
            &cands,
            (5.2 * gib as f64) as u64,
            2800.0,
            true,
        );
        let best = recs.iter().find(|r| r.recommended).unwrap();
        assert_eq!(best.candidate.quant, "Q4_K_M");
        assert_eq!(best.verdict, "GOOD");
        assert_eq!(
            best.hf_uri,
            "hf://Qwen/Qwen2.5-14B-Instruct-GGUF:model-q4_k_m-00001-of-00003.gguf"
        );
    }

    #[test]
    fn xet_redirect_uses_hub_x_linked_etag_not_cdn_xet_etag() {
        let _env_guard = ENV_LOCK.lock().unwrap();
        use std::io::{BufRead, BufReader};
        use std::net::TcpListener;

        let payload: Vec<u8> = (0..16_384).map(|i| ((i * 29 + 11) % 251) as u8).collect();
        let real_sha256 = to_hex_lower(&Sha256::digest(&payload));
        let fake_xet_cas_etag = "0663b826e09042345d6e2069d9655cd3ae84812d8f39a5894bed7b24fc0e86cc";
        assert_ne!(real_sha256, fake_xet_cas_etag);

        // Simulate HF Hub 302 with `X-Linked-Etag` (real SHA-256) redirecting to a CDN 200
        // whose `ETag` is a different 64-hex Xet CAS hash (and verify Authorization is stripped).
        let cdn_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let cdn_addr = cdn_listener.local_addr().unwrap();
        let cdn_payload = payload.clone();
        let cdn_thread = std::thread::spawn(move || {
            let (mut sock, _) = cdn_listener.accept().unwrap();
            let mut reader = BufReader::new(sock.try_clone().unwrap());
            let mut saw_auth = false;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" || line == "\n" {
                    break;
                }
                if line.to_ascii_lowercase().starts_with("authorization:") {
                    saw_auth = true;
                }
            }
            assert!(
                !saw_auth,
                "Authorization header must not leak to redirected CDN"
            );
            let header = format!(
                "HTTP/1.1 200 OK\r\nETag: \"{fake_xet_cas_etag}\"\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                cdn_payload.len()
            );
            sock.write_all(header.as_bytes()).unwrap();
            sock.write_all(&cdn_payload).unwrap();
            let _ = sock.flush();
        });

        let hub_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let hub_addr = hub_listener.local_addr().unwrap();
        let hub_sha = real_sha256.clone();
        let hub_thread = std::thread::spawn(move || {
            let (mut sock, _) = hub_listener.accept().unwrap();
            let mut reader = BufReader::new(sock.try_clone().unwrap());
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" || line == "\n" {
                    break;
                }
            }
            let header = format!(
                "HTTP/1.1 302 Found\r\nLocation: http://{cdn_addr}/xet-bridge/blob\r\nX-Linked-Etag: \"{hub_sha}\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            sock.write_all(header.as_bytes()).unwrap();
            let _ = sock.flush();
        });

        let tmp = std::env::temp_dir().join(format!("layerstream-xet-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        unsafe {
            std::env::set_var("HF_ENDPOINT", format!("http://{hub_addr}"));
            std::env::set_var("HF_TOKEN", "hf_test_secret_token");
        }

        let res = ensure_hf_downloaded("hf://testorg/xetrepo:model.gguf", &tmp);

        unsafe {
            std::env::remove_var("HF_ENDPOINT");
            std::env::remove_var("HF_TOKEN");
        }
        hub_thread.join().unwrap();
        cdn_thread.join().unwrap();

        let local = res.expect("Xet redirect download must verify against Hub X-Linked-Etag");
        assert_eq!(fs::read(&local).unwrap(), payload);
        let _ = fs::remove_dir_all(&tmp);
    }
}
