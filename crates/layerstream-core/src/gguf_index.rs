//! Pure-Rust little-endian GGUF v2/v3 header parser & 4K-aligned coalesced layer span planner.
//!
//! Reads only GGUF metadata headers (single-file or multi-part `*-00001-of-0000N.gguf`)
//! so model discovery uses 0 MB of VRAM and ~milliseconds of startup time.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
pub const DIRECT_IO_ALIGN: u64 = 4096;
const DEFAULT_MERGE_GAP_BYTES: u64 = 64 * 1024;
const MAX_GGUF_TENSORS: usize = 1_000_000;
const MAX_GGUF_KV_COUNT: usize = 1_000_000;
const MAX_GGUF_STRING_BYTES: usize = 64 * 1024 * 1024;
const MAX_GGUF_TENSOR_DIMS: usize = 16;
const MAX_GGUF_ARRAY_ELEMS: u64 = 16_000_000;
/// Maximum vocabulary size padding tolerance between target and draft models of the same family
/// (e.g., Qwen2.5-14B/32B pads 151,936 base tokens to 152,064 with 128 trailing `[PAD]` entries).
pub const MAX_DRAFT_VOCAB_PAD_DELTA: usize = 256;

#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub architecture: String,
    pub name: Option<String>,
    pub n_layers: u32,
    pub context_length: Option<u32>,
    pub total_file_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct TensorLocation {
    pub name: String,
    pub shard_idx: usize,
    pub abs_offset: u64,
    pub ggml_type: u32,
    pub dims: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct SpanSlice {
    pub dst_offset: usize,
    pub span_offset: usize,
    pub len: usize,
    pub tensor_idx: u32,
}

#[derive(Debug, Clone)]
pub struct CoalescedSpan {
    pub shard_idx: usize,
    /// 4K-aligned file offset to read from.
    pub aligned_offset: u64,
    /// Exact byte length to read (4K-aligned except possibly truncated by EOF).
    pub aligned_len: usize,
    /// Valid bytes within `[0..aligned_len]` before EOF.
    pub valid_len: usize,
    pub slices: Vec<SpanSlice>,
}

#[derive(Debug, Clone)]
pub struct GgufIndex {
    pub shards: Vec<PathBuf>,
    pub shard_sizes: Vec<u64>,
    pub meta: ModelMeta,
    pub tensors: HashMap<String, TensorLocation>,
}

impl GgufIndex {
    /// Parses GGUF header(s) starting from `path`.
    /// Automatically detects multi-part GGUF shards named `*-00001-of-0000N.gguf`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let primary = path.as_ref().to_path_buf();
        let shards = discover_shards(&primary)?;

        let mut shard_sizes = Vec::with_capacity(shards.len());
        let mut tensors = HashMap::new();
        let mut kv_all = HashMap::new();
        let mut total_file_bytes = 0u64;

        for (shard_idx, shard_path) in shards.iter().enumerate() {
            let file = File::open(shard_path)
                .with_context(|| format!("failed to open GGUF shard {}", shard_path.display()))?;
            let file_len = file.metadata()?.len();
            shard_sizes.push(file_len);
            total_file_bytes += file_len;

            let mut reader = BufReader::new(file);
            let parsed = parse_single_header(&mut reader, shard_idx, file_len, false)
                .with_context(|| format!("failed to parse GGUF header {}", shard_path.display()))?;

            if shard_idx == 0 {
                kv_all = parsed.kv;
            }
            for t in parsed.tensors {
                tensors.insert(t.name.clone(), t);
            }
        }

        let architecture = kv_all
            .get("general.architecture")
            .and_then(KvValue::as_str)
            .unwrap_or("unknown")
            .to_string();
        let name = kv_all
            .get("general.name")
            .and_then(KvValue::as_str)
            .map(ToOwned::to_owned);

        let block_count_key = format!("{architecture}.block_count");
        let n_layers = match kv_all.get(&block_count_key).and_then(KvValue::as_u32) {
            Some(v) => v,
            None => infer_layer_count(&tensors)?,
        };

        let ctx_key = format!("{architecture}.context_length");
        let context_length = kv_all.get(&ctx_key).and_then(KvValue::as_u32);

        Ok(Self {
            shards,
            shard_sizes,
            meta: ModelMeta {
                architecture,
                name,
                n_layers,
                context_length,
                total_file_bytes,
            },
            tensors,
        })
    }

    /// Plans coalesced 4K-aligned file spans for a list of `(name, dst_offset, nbytes)` requests.
    pub fn plan_coalesced_spans(
        &self,
        requests: &[(&str, usize, usize)],
    ) -> Result<Vec<CoalescedSpan>> {
        let sub_reqs: Vec<(&str, u64, usize, usize, u32)> = requests
            .iter()
            .enumerate()
            .map(|(i, &(name, dst_offset, len))| (name, 0u64, dst_offset, len, i as u32))
            .collect();
        self.plan_coalesced_sub_slices(&sub_reqs)
    }

    /// Plans coalesced 4K-aligned file spans for arbitrary tensor byte sub-ranges:
    /// `(tensor_name, src_offset_in_tensor, dst_offset, len, tensor_idx)`.
    /// Adjacent or nearby slices (e.g. active MoE experts) within `DEFAULT_MERGE_GAP_BYTES`
    /// are automatically merged into a single 4K-aligned span.
    pub fn plan_coalesced_sub_slices(
        &self,
        requests: &[(&str, u64, usize, usize, u32)],
    ) -> Result<Vec<CoalescedSpan>> {
        #[derive(Clone)]
        struct Item {
            shard_idx: usize,
            abs_offset: u64,
            len: usize,
            dst_offset: usize,
            tensor_idx: u32,
        }

        let mut items = Vec::with_capacity(requests.len());
        for &(name, src_off, dst_offset, len, tensor_idx) in requests {
            let loc = self
                .tensors
                .get(name)
                .with_context(|| format!("tensor '{name}' missing from GGUF index"))?;
            let shard_size = self.shard_sizes[loc.shard_idx];
            let abs_offset = loc.abs_offset.saturating_add(src_off);
            if abs_offset.saturating_add(len as u64) > shard_size {
                bail!(
                    "tensor slice '{name}' [{}, {}) exceeds shard size {}",
                    abs_offset,
                    abs_offset.saturating_add(len as u64),
                    shard_size
                );
            }
            items.push(Item {
                shard_idx: loc.shard_idx,
                abs_offset,
                len,
                dst_offset,
                tensor_idx,
            });
        }

        items.sort_by_key(|it| (it.shard_idx, it.abs_offset));

        let mut spans = Vec::new();
        let mut i = 0;
        while i < items.len() {
            let shard_idx = items[i].shard_idx;
            let shard_size = self.shard_sizes[shard_idx];
            let group_start = items[i].abs_offset;
            let mut group_end = items[i].abs_offset + items[i].len as u64;
            let mut j = i + 1;

            while j < items.len() && items[j].shard_idx == shard_idx {
                let next_start = items[j].abs_offset;
                let next_end = next_start + items[j].len as u64;
                if next_start <= group_end.saturating_add(DEFAULT_MERGE_GAP_BYTES) {
                    group_end = group_end.max(next_end);
                    j += 1;
                } else {
                    break;
                }
            }

            let aligned_offset = align_down(group_start, DIRECT_IO_ALIGN);
            let aligned_end = align_up(group_end, DIRECT_IO_ALIGN);
            let clamped_end = aligned_end.min(shard_size);
            let valid_len = (clamped_end - aligned_offset) as usize;
            let aligned_len = (aligned_end - aligned_offset) as usize;

            let mut slices = Vec::with_capacity(j - i);
            for item in &items[i..j] {
                slices.push(SpanSlice {
                    dst_offset: item.dst_offset,
                    span_offset: (item.abs_offset - aligned_offset) as usize,
                    len: item.len,
                    tensor_idx: item.tensor_idx,
                });
            }

            spans.push(CoalescedSpan {
                shard_idx,
                aligned_offset,
                aligned_len,
                valid_len,
                slices,
            });

            i = j;
        }

        Ok(spans)
    }
}

fn infer_layer_count(tensors: &HashMap<String, TensorLocation>) -> Result<u32> {
    let mut max_idx: Option<u32> = None;
    for name in tensors.keys() {
        if let Some(rest) = name.strip_prefix("blk.")
            && let Some((num, _)) = rest.split_once('.')
            && let Ok(idx) = num.parse::<u32>()
        {
            max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
        }
    }
    match max_idx {
        None => Ok(0),
        Some(m) => m
            .checked_add(1)
            .context("GGUF block_count exceeds u32::MAX"),
    }
}

fn discover_shards(primary: &Path) -> Result<Vec<PathBuf>> {
    let Some(file_name) = primary.file_name().and_then(|s| s.to_str()) else {
        return Ok(vec![primary.to_path_buf()]);
    };

    // Detect `prefix-00001-of-0000N.gguf` multi-part naming convention.
    if let Some(stem) = file_name.strip_suffix(".gguf")
        && let Some((prefix_and_idx, total_str)) = stem.rsplit_once("-of-")
        && total_str.len() == 5
        && let Ok(total) = total_str.parse::<u32>()
        && total > 1
        && let Some((prefix, idx_str)) = prefix_and_idx.rsplit_once('-')
        && idx_str.len() == 5
        && idx_str.parse::<u32>().is_ok()
    {
        let parent = primary.parent().unwrap_or_else(|| Path::new("."));
        let mut shards = Vec::with_capacity(total as usize);
        for idx in 1..=total {
            let candidate = parent.join(format!("{prefix}-{idx:05}-of-{total:05}.gguf"));
            if !candidate.is_file() {
                bail!("missing multi-part GGUF shard: {}", candidate.display());
            }
            shards.push(candidate);
        }
        return Ok(shards);
    }

    Ok(vec![primary.to_path_buf()])
}

#[derive(Debug, Clone)]
enum KvValue {
    U32(u32),
    U64(u64),
    I32(i32),
    Str(String),
    Other,
}

impl KvValue {
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    fn as_u32(&self) -> Option<u32> {
        match *self {
            Self::U32(v) => Some(v),
            Self::U64(v) => u32::try_from(v).ok(),
            Self::I32(v) => u32::try_from(v).ok(),
            _ => None,
        }
    }
}

struct ParsedHeader {
    kv: HashMap<String, KvValue>,
    tensors: Vec<TensorLocation>,
    vocab_tokens: Option<Vec<String>>,
}

/// Validates at startup (from GGUF metadata headers alone, 0 MB VRAM) that `draft_path`
/// shares the exact same base vocabulary as `target_path`, allowing up to
/// `MAX_DRAFT_VOCAB_PAD_DELTA` trailing padded tokens (e.g. Qwen2.5-14B/32B `152,064`
/// vs `0.5B` `151,936`).
pub fn check_gguf_draft_compatible(
    target_path: impl AsRef<Path>,
    draft_path: impl AsRef<Path>,
) -> Result<()> {
    let target_tokens = read_gguf_vocab_tokens(target_path.as_ref())?;
    let draft_tokens = read_gguf_vocab_tokens(draft_path.as_ref())?;
    check_vocab_lists_compatible(&target_tokens, &draft_tokens).map_err(|msg| anyhow::anyhow!(msg))
}

pub fn check_vocab_lists_compatible(
    target_tokens: &[String],
    draft_tokens: &[String],
) -> std::result::Result<(), String> {
    if target_tokens.is_empty() || draft_tokens.is_empty() {
        return Err(format!(
            "missing tokenizer vocabulary in GGUF header (target={}, draft={})",
            target_tokens.len(),
            draft_tokens.len()
        ));
    }
    let delta = target_tokens.len().abs_diff(draft_tokens.len());
    if delta > MAX_DRAFT_VOCAB_PAD_DELTA {
        return Err(format!(
            "vocabulary size mismatch (target={}, draft={}, delta={} > max pad tolerance {})",
            target_tokens.len(),
            draft_tokens.len(),
            delta,
            MAX_DRAFT_VOCAB_PAD_DELTA
        ));
    }
    let shared = target_tokens.len().min(draft_tokens.len());
    for i in 0..shared {
        if target_tokens[i] != draft_tokens[i] {
            return Err(format!(
                "vocabulary token mismatch at id {i} (target={:?}, draft={:?})",
                target_tokens[i], draft_tokens[i]
            ));
        }
    }
    Ok(())
}

fn read_gguf_vocab_tokens(path: &Path) -> Result<Vec<String>> {
    let shards = discover_shards(path)?;
    let primary = &shards[0];
    let file = File::open(primary)
        .with_context(|| format!("failed to open GGUF for vocab check {}", primary.display()))?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let parsed = parse_single_header(&mut reader, 0, file_len, true)
        .with_context(|| format!("failed to parse GGUF header {}", primary.display()))?;
    parsed
        .vocab_tokens
        .context("GGUF header missing tokenizer.ggml.tokens array")
}

fn parse_single_header<R: Read + Seek>(
    r: &mut R,
    shard_idx: usize,
    file_len: u64,
    capture_vocab: bool,
) -> Result<ParsedHeader> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        bail!("invalid GGUF magic: {:?}", magic);
    }

    let version = read_u32(r)?;
    if !(2..=3).contains(&version) {
        bail!("unsupported GGUF version: {version}");
    }

    let tensor_count_u64 = read_u64(r)?;
    let kv_count_u64 = read_u64(r)?;
    if tensor_count_u64 > MAX_GGUF_TENSORS as u64 {
        bail!("GGUF tensor_count {tensor_count_u64} exceeds safety limit {MAX_GGUF_TENSORS}");
    }
    if kv_count_u64 > MAX_GGUF_KV_COUNT as u64 {
        bail!("GGUF kv_count {kv_count_u64} exceeds safety limit {MAX_GGUF_KV_COUNT}");
    }
    let tensor_count = tensor_count_u64 as usize;
    let kv_count = kv_count_u64 as usize;

    let mut kv = HashMap::with_capacity(kv_count.min(1024));
    let mut vocab_tokens: Option<Vec<String>> = None;
    for _ in 0..kv_count {
        let key = read_string(r, file_len)?;
        let val_type = read_u32(r)?;
        if capture_vocab && key == "tokenizer.ggml.tokens" && val_type == 9 {
            let elem_type = read_u32(r)?;
            let len = read_u64(r)?;
            if elem_type != 8 || len > MAX_GGUF_ARRAY_ELEMS {
                bail!("unexpected tokenizer.ggml.tokens array type={elem_type} len={len}");
            }
            let mut toks = Vec::with_capacity((len as usize).min(262_144));
            for _ in 0..len {
                toks.push(read_string(r, file_len)?);
            }
            vocab_tokens = Some(toks);
            kv.insert(key, KvValue::Other);
        } else {
            let val = read_kv_value(r, val_type, file_len)?;
            kv.insert(key, val);
        }
    }

    let alignment = kv
        .get("general.alignment")
        .and_then(KvValue::as_u32)
        .map(u64::from)
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("invalid GGUF general.alignment: {alignment} (must be non-zero power of two)");
    }

    if capture_vocab {
        return Ok(ParsedHeader {
            kv,
            tensors: Vec::new(),
            vocab_tokens,
        });
    }

    let mut raw_tensors = Vec::with_capacity(tensor_count.min(4096));
    for _ in 0..tensor_count {
        let name = read_string(r, file_len)?;
        let n_dims_u32 = read_u32(r)?;
        if n_dims_u32 as usize > MAX_GGUF_TENSOR_DIMS {
            bail!("GGUF tensor '{name}' n_dims {n_dims_u32} exceeds limit {MAX_GGUF_TENSOR_DIMS}");
        }
        let n_dims = n_dims_u32 as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(read_u64(r)?);
        }
        let ggml_type = read_u32(r)?;
        let rel_offset = read_u64(r)?;
        raw_tensors.push((name, dims, ggml_type, rel_offset));
    }

    let header_end = r.stream_position()?;
    let data_start = align_up(header_end, alignment);

    let mut tensors = Vec::with_capacity(raw_tensors.len());
    for (name, dims, ggml_type, rel_offset) in raw_tensors {
        let abs_offset = data_start
            .checked_add(rel_offset)
            .context("GGUF tensor offset overflow")?;
        tensors.push(TensorLocation {
            name,
            shard_idx,
            abs_offset,
            ggml_type,
            dims,
        });
    }

    Ok(ParsedHeader {
        kv,
        tensors,
        vocab_tokens,
    })
}

fn read_kv_value<R: Read + Seek>(r: &mut R, val_type: u32, file_len: u64) -> Result<KvValue> {
    read_kv_value_depth(r, val_type, file_len, 0)
}

const MAX_GGUF_ARRAY_DEPTH: u32 = 16;

fn read_kv_value_depth<R: Read + Seek>(
    r: &mut R,
    val_type: u32,
    file_len: u64,
    depth: u32,
) -> Result<KvValue> {
    if depth > MAX_GGUF_ARRAY_DEPTH {
        bail!("GGUF metadata array nesting depth exceeds limit {MAX_GGUF_ARRAY_DEPTH}");
    }
    match val_type {
        0 => {
            skip_bytes(r, 1, file_len)?;
            Ok(KvValue::Other)
        } // UINT8
        1 => {
            skip_bytes(r, 1, file_len)?;
            Ok(KvValue::Other)
        } // INT8
        2 => {
            skip_bytes(r, 2, file_len)?;
            Ok(KvValue::Other)
        } // UINT16
        3 => {
            skip_bytes(r, 2, file_len)?;
            Ok(KvValue::Other)
        } // INT16
        4 => Ok(KvValue::U32(read_u32(r)?)),        // UINT32
        5 => Ok(KvValue::I32(read_u32(r)? as i32)), // INT32
        6 => {
            skip_bytes(r, 4, file_len)?;
            Ok(KvValue::Other)
        } // FLOAT32
        7 => {
            skip_bytes(r, 1, file_len)?;
            Ok(KvValue::Other)
        } // BOOL
        8 => Ok(KvValue::Str(read_string(r, file_len)?)), // STRING
        9 => {
            // ARRAY
            let elem_type = read_u32(r)?;
            let len = read_u64(r)?;
            if len > MAX_GGUF_ARRAY_ELEMS {
                bail!("GGUF array length {len} exceeds safety limit {MAX_GGUF_ARRAY_ELEMS}");
            }
            if let Some(elem_size) = fixed_kv_type_size(elem_type) {
                let bytes = len
                    .checked_mul(elem_size)
                    .context("GGUF array byte size overflow")?;
                skip_bytes(r, bytes, file_len)?;
            } else {
                for _ in 0..len {
                    let _ = read_kv_value_depth(r, elem_type, file_len, depth + 1)?;
                }
            }
            Ok(KvValue::Other)
        }
        10 => Ok(KvValue::U64(read_u64(r)?)), // UINT64
        11 => {
            skip_bytes(r, 8, file_len)?;
            Ok(KvValue::Other)
        } // INT64
        12 => {
            skip_bytes(r, 8, file_len)?;
            Ok(KvValue::Other)
        } // FLOAT64
        other => bail!("unknown GGUF KV type id: {other}"),
    }
}

fn fixed_kv_type_size(val_type: u32) -> Option<u64> {
    match val_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string<R: Read + Seek>(r: &mut R, file_len: u64) -> Result<String> {
    let len_u64 = read_u64(r)?;
    let pos = r.stream_position()?;
    let remaining = file_len.saturating_sub(pos);
    if len_u64 > MAX_GGUF_STRING_BYTES as u64 || len_u64 > remaining {
        bail!(
            "GGUF string length {len_u64} exceeds bounds (max={MAX_GGUF_STRING_BYTES}, remaining={remaining})"
        );
    }
    let mut buf = vec![0u8; len_u64 as usize];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn skip_bytes<R: Seek>(r: &mut R, n: u64, file_len: u64) -> Result<()> {
    let pos = r.stream_position()?;
    let target = pos.checked_add(n).context("GGUF seek offset overflow")?;
    if target > file_len {
        bail!("GGUF skip past EOF ({target} > {file_len})");
    }
    r.seek(SeekFrom::Start(target))?;
    Ok(())
}

#[inline]
pub const fn align_down(v: u64, align: u64) -> u64 {
    v & !(align - 1)
}

#[inline]
pub const fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn gguf_header(version: u32, n_tensors: u64, n_kv: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(GGUF_MAGIC);
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&n_tensors.to_le_bytes());
        buf.extend_from_slice(&n_kv.to_le_bytes());
        buf
    }

    #[test]
    fn rejects_invalid_magic_version_and_truncated_headers() {
        let mut bad_magic = Cursor::new(
            b"NOTG\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                .to_vec(),
        );
        assert!(parse_single_header(&mut bad_magic, 0, 24, false).is_err());

        let mut old_ver = Cursor::new(gguf_header(1, 0, 0));
        assert!(parse_single_header(&mut old_ver, 0, 24, false).is_err());

        let mut future_ver = Cursor::new(gguf_header(99, 0, 0));
        assert!(parse_single_header(&mut future_ver, 0, 24, false).is_err());

        let mut truncated = Cursor::new(b"GGUF\x03\x00".to_vec());
        assert!(parse_single_header(&mut truncated, 0, 6, false).is_err());
    }

    #[test]
    fn rejects_unbounded_tensor_kv_and_string_counts() {
        let mut huge_tensors = Cursor::new(gguf_header(3, (MAX_GGUF_TENSORS as u64) + 1, 0));
        assert!(parse_single_header(&mut huge_tensors, 0, 24, false).is_err());

        let mut huge_kv = Cursor::new(gguf_header(3, 0, (MAX_GGUF_KV_COUNT as u64) + 1));
        assert!(parse_single_header(&mut huge_kv, 0, 24, false).is_err());

        // Header claiming 1 KV entry whose key string length exceeds remaining file bytes.
        let mut bad_str = gguf_header(3, 0, 1);
        bad_str.extend_from_slice(&1024u64.to_le_bytes());
        let len = bad_str.len() as u64;
        let mut cur = Cursor::new(bad_str);
        assert!(parse_single_header(&mut cur, 0, len, false).is_err());
    }

    #[test]
    fn align_helpers_and_span_coalescing_merge_nearby_and_split_distant() {
        assert_eq!(align_down(0, DIRECT_IO_ALIGN), 0);
        assert_eq!(align_down(4095, DIRECT_IO_ALIGN), 0);
        assert_eq!(align_down(4096, DIRECT_IO_ALIGN), 4096);
        assert_eq!(align_up(1, DIRECT_IO_ALIGN), 4096);
        assert_eq!(align_up(4096, DIRECT_IO_ALIGN), 4096);
        assert_eq!(align_up(4097, DIRECT_IO_ALIGN), 8192);

        let mut tensors = HashMap::new();
        tensors.insert(
            "t0".to_string(),
            TensorLocation {
                name: "t0".to_string(),
                shard_idx: 0,
                abs_offset: 100,
                ggml_type: 0,
                dims: vec![64],
            },
        );
        tensors.insert(
            "t1".to_string(),
            TensorLocation {
                name: "t1".to_string(),
                shard_idx: 0,
                abs_offset: 8_192,
                ggml_type: 0,
                dims: vec![64],
            },
        );
        tensors.insert(
            "t2".to_string(),
            TensorLocation {
                name: "t2".to_string(),
                shard_idx: 0,
                abs_offset: 1_000_000,
                ggml_type: 0,
                dims: vec![64],
            },
        );
        let idx = GgufIndex {
            shards: vec![PathBuf::from("dummy.gguf")],
            shard_sizes: vec![2_000_000],
            meta: ModelMeta {
                architecture: "llama".to_string(),
                name: None,
                n_layers: 1,
                context_length: Some(128),
                total_file_bytes: 2_000_000,
            },
            tensors,
        };

        let spans = idx
            .plan_coalesced_spans(&[("t0", 0, 256), ("t1", 256, 256), ("t2", 512, 256)])
            .expect("plan spans");
        assert_eq!(spans.len(), 2, "t0+t1 should coalesce, t2 should split");
        assert_eq!(spans[0].aligned_offset % DIRECT_IO_ALIGN, 0);
        assert_eq!(spans[0].slices.len(), 2);
        assert_eq!(spans[1].aligned_offset % DIRECT_IO_ALIGN, 0);
        assert_eq!(spans[1].slices.len(), 1);
    }
}
