//! Lightweight remote GGUF header inspection over a single HTTP `Range` request.
//!
//! Fetches only the leading 256 KiB of a Hugging Face GGUF file to extract
//! architecture, layer count, MoE expert configuration, and built-in MTP
//! (`nextn_predict_layers`) metadata *before* downloading multi-gigabyte weights.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::{DEFAULT_READ_TIMEOUT_SECS, HfError, HfSpec, Result, load_hf_token};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const REMOTE_HEADER_PROBE_BYTES: usize = 256 * 1024;
const REMOTE_HEADER_RETRY_BYTES: usize = 1024 * 1024;

/// Architecture and capability metadata extracted from a GGUF header without
/// downloading tensor weights.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteGgufMetadata {
    pub gguf_version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub architecture: String,
    pub name: Option<String>,
    pub block_count: Option<u32>,
    pub context_length: Option<u32>,
    pub embedding_length: Option<u32>,
    pub expert_count: Option<u32>,
    pub expert_used_count: Option<u32>,
    pub mtp_layers: u32,
    pub has_mtp: bool,
    pub is_moe: bool,
    pub is_hybrid_ssm: bool,
    pub file_type: Option<u32>,
    pub quantization_version: Option<u32>,
}

impl RemoteGgufMetadata {
    /// Returns `true` when the checkpoint has no built-in MTP heads and would
    /// benefit from an external draft model for speculative verification.
    pub fn needs_external_draft(&self) -> bool {
        !self.has_mtp
    }

    /// Formats a concise one-line human summary of the inspected GGUF capabilities.
    pub fn summary_line(&self) -> String {
        let mut parts = vec![format!("arch={}", self.architecture)];
        if let Some(layers) = self.block_count {
            parts.push(format!("layers={layers}"));
        }
        if let Some(ctx) = self.context_length {
            parts.push(format!("ctx={ctx}"));
        }
        if let Some(experts) = self.expert_count
            && experts > 0
        {
            if let Some(used) = self.expert_used_count {
                parts.push(format!("MoE={experts}x (top-{used})"));
            } else {
                parts.push(format!("MoE={experts}x"));
            }
        }
        if self.has_mtp {
            parts.push(format!(
                "MTP={} (built-in self-speculation, no external draft needed)",
                self.mtp_layers
            ));
        }
        if self.is_hybrid_ssm {
            parts.push("hybrid-SSM (n_rs_seq rollback)".to_string());
        }
        parts.join(" | ")
    }
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        if end > self.buf.len() {
            return None;
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Some(slice)
    }

    fn read_u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> Option<u16> {
        let b: [u8; 2] = self.take(2)?.try_into().ok()?;
        Some(u16::from_le_bytes(b))
    }

    fn read_u32(&mut self) -> Option<u32> {
        let b: [u8; 4] = self.take(4)?.try_into().ok()?;
        Some(u32::from_le_bytes(b))
    }

    fn read_u64(&mut self) -> Option<u64> {
        let b: [u8; 8] = self.take(8)?.try_into().ok()?;
        Some(u64::from_le_bytes(b))
    }

    fn read_str(&mut self) -> Option<String> {
        let len = usize::try_from(self.read_u64()?).ok()?;
        let bytes = self.take(len)?;
        Some(String::from_utf8_lossy(bytes).into_owned())
    }

    fn skip_scalar(&mut self, vtype: u32) -> Option<()> {
        match vtype {
            0 | 1 | 7 => {
                self.take(1)?;
            }
            2 | 3 => {
                self.take(2)?;
            }
            4..=6 => {
                self.take(4)?;
            }
            10..=12 => {
                self.take(8)?;
            }
            8 => {
                let len = usize::try_from(self.read_u64()?).ok()?;
                self.take(len)?;
            }
            9 => {
                let elem_type = self.read_u32()?;
                let count = usize::try_from(self.read_u64()?).ok()?;
                for _ in 0..count {
                    self.skip_scalar(elem_type)?;
                }
            }
            _ => return None,
        }
        Some(())
    }

    fn read_u32_like(&mut self, vtype: u32) -> Option<u32> {
        match vtype {
            0 => Some(u32::from(self.read_u8()?)),
            1 => u32::try_from(self.read_u8()? as i8).ok(),
            2 => Some(u32::from(self.read_u16()?)),
            3 => u32::try_from(self.read_u16()? as i16).ok(),
            4 => self.read_u32(),
            5 => u32::try_from(self.read_u32()? as i32).ok(),
            10 => u32::try_from(self.read_u64()?).ok(),
            11 => u32::try_from(self.read_u64()? as i64).ok(),
            _ => {
                self.skip_scalar(vtype)?;
                None
            }
        }
    }
}

/// Parses GGUF v2/v3 metadata from a leading header slice.
///
/// Tolerates truncated buffers (e.g., when a large `tokenizer.ggml.tokens`
/// array extends beyond a 256 KiB HTTP `Range` slice) because `general.*` and
/// `<arch>.*` keys precede tokenizer arrays in GGUF files.
pub fn parse_gguf_header_metadata(bytes: &[u8]) -> Result<RemoteGgufMetadata> {
    let mut cur = Cursor::new(bytes);
    let magic = cur
        .take(4)
        .ok_or_else(|| HfError::DownloadFailed("GGUF header truncated (< 4 bytes)".into()))?;
    if magic != GGUF_MAGIC {
        return Err(HfError::DownloadFailed(
            "remote file is not a valid GGUF binary (bad magic)".into(),
        ));
    }
    let gguf_version = cur
        .read_u32()
        .ok_or_else(|| HfError::DownloadFailed("GGUF header truncated reading version".into()))?;
    if !(2..=3).contains(&gguf_version) {
        return Err(HfError::DownloadFailed(format!(
            "unsupported GGUF version {gguf_version} (expected 2 or 3)"
        )));
    }
    let tensor_count = cur.read_u64().ok_or_else(|| {
        HfError::DownloadFailed("GGUF header truncated reading tensor_count".into())
    })?;
    let kv_count = cur
        .read_u64()
        .ok_or_else(|| HfError::DownloadFailed("GGUF header truncated reading kv_count".into()))?;

    let mut architecture = String::new();
    let mut name = None;
    let mut block_count = None;
    let mut context_length = None;
    let mut embedding_length = None;
    let mut expert_count = None;
    let mut expert_used_count = None;
    let mut mtp_layers = 0u32;
    let mut is_hybrid_ssm = false;
    let mut file_type = None;
    let mut quantization_version = None;

    for _ in 0..kv_count {
        let Some(key) = cur.read_str() else {
            break;
        };
        let Some(vtype) = cur.read_u32() else {
            break;
        };
        if key.contains(".ssm.") || key.contains(".time_mix") {
            is_hybrid_ssm = true;
        }
        match key.as_str() {
            "general.architecture" if vtype == 8 => {
                let Some(v) = cur.read_str() else {
                    break;
                };
                architecture = v;
            }
            "general.name" if vtype == 8 => {
                let Some(v) = cur.read_str() else {
                    break;
                };
                name = Some(v);
            }
            "general.file_type" => {
                file_type = cur.read_u32_like(vtype);
            }
            "general.quantization_version" => {
                quantization_version = cur.read_u32_like(vtype);
            }
            other if other.ends_with(".block_count") => {
                block_count = cur.read_u32_like(vtype);
            }
            other if other.ends_with(".context_length") => {
                context_length = cur.read_u32_like(vtype);
            }
            other if other.ends_with(".embedding_length") => {
                embedding_length = cur.read_u32_like(vtype);
            }
            other if other.ends_with(".expert_count") => {
                expert_count = cur.read_u32_like(vtype);
            }
            other if other.ends_with(".expert_used_count") => {
                expert_used_count = cur.read_u32_like(vtype);
            }
            other if other.ends_with(".nextn_predict_layers") => {
                if let Some(n) = cur.read_u32_like(vtype) {
                    mtp_layers = n;
                }
            }
            _ => {
                if cur.skip_scalar(vtype).is_none() {
                    break;
                }
            }
        }
    }

    if architecture.is_empty() {
        return Err(HfError::DownloadFailed(
            "GGUF header did not contain general.architecture within probe window".into(),
        ));
    }

    Ok(RemoteGgufMetadata {
        gguf_version,
        tensor_count,
        kv_count,
        architecture,
        name,
        block_count,
        context_length,
        embedding_length,
        expert_count,
        expert_used_count,
        mtp_layers,
        has_mtp: mtp_layers > 0,
        is_moe: expert_count.unwrap_or(0) > 0,
        is_hybrid_ssm,
        file_type,
        quantization_version,
    })
}

/// Inspects a local GGUF file's header (reading at most 1 MiB) without loading tensor weights.
pub fn inspect_local_gguf(path: &Path) -> Result<RemoteGgufMetadata> {
    let mut file = File::open(path)
        .map_err(|e| HfError::io(format!("open local GGUF {}", path.display()), e))?;
    let mut buf = Vec::with_capacity(64 * 1024);
    file.by_ref()
        .take(REMOTE_HEADER_PROBE_BYTES as u64)
        .read_to_end(&mut buf)
        .map_err(|e| HfError::io(format!("read local GGUF header {}", path.display()), e))?;
    match parse_gguf_header_metadata(&buf) {
        Ok(meta) if meta.block_count.is_some() => Ok(meta),
        _ => {
            file.take((REMOTE_HEADER_RETRY_BYTES - REMOTE_HEADER_PROBE_BYTES) as u64)
                .read_to_end(&mut buf)
                .map_err(|e| {
                    HfError::io(format!("read extended GGUF header {}", path.display()), e)
                })?;
            parse_gguf_header_metadata(&buf)
        }
    }
}

fn fetch_remote_header_bytes(spec: &HfSpec, max_bytes: usize) -> Result<Vec<u8>> {
    let url = spec.download_url();
    let token = load_hf_token();
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(DEFAULT_READ_TIMEOUT_SECS)))
        .http_status_as_error(false)
        .build()
        .new_agent();

    let mut req = agent
        .get(&url)
        .header("Range", &format!("bytes=0-{}", max_bytes - 1));
    if let Some(tok) = token.as_deref() {
        req = req.header("Authorization", &format!("Bearer {tok}"));
    }

    let resp = req
        .call()
        .map_err(|e| HfError::Http(format!("remote GGUF inspect request failed: {e}")))?;
    let status = resp.status().as_u16();
    if status == 404 {
        return Err(HfError::NotFound(format!(
            "remote GGUF not found: {} ({})",
            spec.repo_id, spec.filename
        )));
    }
    if status != 200 && status != 206 {
        return Err(HfError::Http(format!(
            "remote GGUF inspect HTTP {status} for {url}"
        )));
    }

    let mut buf = Vec::with_capacity(max_bytes.min(64 * 1024));
    resp.into_body()
        .into_reader()
        .take(max_bytes as u64)
        .read_to_end(&mut buf)
        .map_err(|e| HfError::io("reading remote GGUF header", e))?;
    Ok(buf)
}

/// Fetches the leading 256 KiB (retrying up to 1 MiB if metadata is truncated)
/// of a Hugging Face GGUF via HTTP `Range` and returns its parsed architecture,
/// MoE, hybrid-SSM, and MTP metadata without downloading model weights.
pub fn inspect_remote_gguf(spec: &HfSpec) -> Result<RemoteGgufMetadata> {
    let buf = fetch_remote_header_bytes(spec, REMOTE_HEADER_PROBE_BYTES)?;
    match parse_gguf_header_metadata(&buf) {
        Ok(meta) if meta.block_count.is_some() => Ok(meta),
        first_res if buf.len() < REMOTE_HEADER_PROBE_BYTES => first_res,
        _ => {
            let larger = fetch_remote_header_bytes(spec, REMOTE_HEADER_RETRY_BYTES)?;
            parse_gguf_header_metadata(&larger)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::thread;

    fn push_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn push_kv_str(out: &mut Vec<u8>, k: &str, v: &str) {
        push_str(out, k);
        out.extend_from_slice(&8u32.to_le_bytes());
        push_str(out, v);
    }

    fn push_kv_u32(out: &mut Vec<u8>, k: &str, v: u32) {
        push_str(out, k);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }

    fn synthetic_glm4moe_mtp_header() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&42u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&10u64.to_le_bytes()); // kv_count (last one truncated intentionally)
        push_kv_str(&mut buf, "general.architecture", "glm4moe");
        push_kv_str(&mut buf, "general.name", "GLM-4.5-Air-Test");
        push_kv_u32(&mut buf, "general.file_type", 15);
        push_kv_u32(&mut buf, "general.quantization_version", 2);
        push_kv_u32(&mut buf, "glm4moe.block_count", 48);
        push_kv_u32(&mut buf, "glm4moe.context_length", 131072);
        push_kv_u32(&mut buf, "glm4moe.embedding_length", 4096);
        push_kv_u32(&mut buf, "glm4moe.expert_count", 128);
        push_kv_u32(&mut buf, "glm4moe.expert_used_count", 8);
        push_kv_u32(&mut buf, "glm4moe.nextn_predict_layers", 1);
        buf
    }

    #[test]
    fn parses_mtp_moe_header_and_rejects_bad_magic() {
        let raw = synthetic_glm4moe_mtp_header();
        let meta = parse_gguf_header_metadata(&raw).expect("parse synthetic header");
        assert_eq!(meta.gguf_version, 3);
        assert_eq!(meta.architecture, "glm4moe");
        assert_eq!(meta.name.as_deref(), Some("GLM-4.5-Air-Test"));
        assert_eq!(meta.block_count, Some(48));
        assert_eq!(meta.context_length, Some(131072));
        assert_eq!(meta.embedding_length, Some(4096));
        assert_eq!(meta.expert_count, Some(128));
        assert_eq!(meta.expert_used_count, Some(8));
        assert_eq!(meta.mtp_layers, 1);
        assert!(meta.has_mtp);
        assert!(meta.is_moe);
        assert_eq!(meta.file_type, Some(15));
        assert_eq!(meta.quantization_version, Some(2));

        assert!(parse_gguf_header_metadata(b"NOTG").is_err());
        assert!(parse_gguf_header_metadata(b"GG").is_err());
    }

    #[test]
    fn inspect_remote_gguf_fetches_range_from_mock_endpoint() {
        let _env_guard = crate::tests::ENV_LOCK.lock().unwrap();
        let payload = synthetic_glm4moe_mtp_header();
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut reader = BufReader::new(stream.try_clone().expect("clone"));
            let mut line = String::new();
            let mut saw_range = false;
            loop {
                line.clear();
                if reader.read_line(&mut line).expect("read") == 0 || line == "\r\n" {
                    break;
                }
                if line.to_ascii_lowercase().starts_with("range: bytes=0-") {
                    saw_range = true;
                }
            }
            assert!(saw_range, "expected Range header");
            let resp = format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            );
            stream.write_all(resp.as_bytes()).expect("write hdr");
            stream.write_all(&payload).expect("write body");
        });

        let prev = std::env::var_os("HF_ENDPOINT");
        // SAFETY: isolated unit test mutates HF_ENDPOINT around a single synchronous call.
        unsafe {
            std::env::set_var("HF_ENDPOINT", format!("http://{addr}"));
        }
        let spec = HfSpec::parse("hf://zai-org/GLM-4.5-Air-GGUF:glm4.5-q4_k_m.gguf").unwrap();
        let meta = inspect_remote_gguf(&spec);
        match prev {
            Some(v) => {
                // SAFETY: restores original HF_ENDPOINT value saved above.
                unsafe { std::env::set_var("HF_ENDPOINT", v) }
            }
            None => {
                // SAFETY: removes test-only HF_ENDPOINT override set above.
                unsafe { std::env::remove_var("HF_ENDPOINT") }
            }
        }
        server.join().expect("server thread");

        let meta = meta.expect("inspect_remote_gguf");
        assert_eq!(meta.architecture, "glm4moe");
        assert!(meta.has_mtp);
        assert!(!meta.needs_external_draft());
        assert!(meta.is_moe);
        let summary = meta.summary_line();
        assert!(summary.contains("arch=glm4moe"));
        assert!(summary.contains("MoE=128x (top-8)"));
        assert!(summary.contains("MTP=1"));

        // Also verify inspect_local_gguf on a temporary file
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        std::fs::write(tmp.path(), synthetic_glm4moe_mtp_header()).expect("write tmp");
        let local_meta = inspect_local_gguf(tmp.path()).expect("inspect_local_gguf");
        assert_eq!(local_meta, meta);
    }

    #[test]
    fn skips_mixed_scalar_and_array_kvals_and_rejects_bad_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&8u64.to_le_bytes());
        // u8, u16, f32, u64, string, array[u32, 2], then general.architecture, then u64 block_count
        push_str(&mut buf, "skip.u8");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.push(1);
        push_str(&mut buf, "skip.u16");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        push_str(&mut buf, "skip.f32");
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend_from_slice(&1.0f32.to_le_bytes());
        push_str(&mut buf, "skip.u64");
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&99u64.to_le_bytes());
        push_kv_str(&mut buf, "skip.str", "ignored");
        push_str(&mut buf, "skip.arr");
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&7u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        push_kv_str(&mut buf, "general.architecture", "qwen2");
        push_str(&mut buf, "qwen2.block_count");
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&32u64.to_le_bytes());

        let meta = parse_gguf_header_metadata(&buf).expect("parse v2 with skipped types");
        assert_eq!(meta.gguf_version, 2);
        assert_eq!(meta.architecture, "qwen2");
        assert_eq!(meta.block_count, Some(32));
        assert!(!meta.has_mtp);
        assert!(!meta.is_moe);

        let mut bad_ver = b"GGUF".to_vec();
        bad_ver.extend_from_slice(&99u32.to_le_bytes());
        bad_ver.extend_from_slice(&0u64.to_le_bytes());
        bad_ver.extend_from_slice(&0u64.to_le_bytes());
        assert!(parse_gguf_header_metadata(&bad_ver).is_err());
    }
}
