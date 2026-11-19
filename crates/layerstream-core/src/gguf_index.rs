//! Pure-Rust GGUF v2/v3 header parser & 4K-aligned coalesced layer span planner.
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
            let parsed = parse_single_header(&mut reader, shard_idx)
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
        let n_layers = kv_all
            .get(&block_count_key)
            .and_then(KvValue::as_u32)
            .unwrap_or_else(|| infer_layer_count(&tensors));

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
        #[derive(Clone)]
        struct Item {
            shard_idx: usize,
            abs_offset: u64,
            len: usize,
            dst_offset: usize,
        }

        let mut items = Vec::with_capacity(requests.len());
        for &(name, dst_offset, len) in requests {
            let loc = self
                .tensors
                .get(name)
                .with_context(|| format!("tensor '{name}' missing from GGUF index"))?;
            let shard_size = self.shard_sizes[loc.shard_idx];
            if loc.abs_offset + (len as u64) > shard_size {
                bail!(
                    "tensor '{name}' [{}, {}) exceeds shard size {}",
                    loc.abs_offset,
                    loc.abs_offset + (len as u64),
                    shard_size
                );
            }
            items.push(Item {
                shard_idx: loc.shard_idx,
                abs_offset: loc.abs_offset,
                len,
                dst_offset,
            });
        }

        items.sort_by_key(|it| (it.shard_idx, it.abs_offset));

        let mut spans = Vec::new();
        let mut i = 0;
        while i < items.len() {
            let shard_idx = items[i].shard_idx;
            let shard_size = self.shard_sizes[shard_idx];
            let mut group_start = items[i].abs_offset;
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
                });
            }

            spans.push(CoalescedSpan {
                shard_idx,
                aligned_offset,
                aligned_len,
                valid_len,
                slices,
            });

            let _ = &mut group_start;
            i = j;
        }

        Ok(spans)
    }
}

fn infer_layer_count(tensors: &HashMap<String, TensorLocation>) -> u32 {
    let mut max_idx: Option<u32> = None;
    for name in tensors.keys() {
        if let Some(rest) = name.strip_prefix("blk.")
            && let Some((num, _)) = rest.split_once('.')
            && let Ok(idx) = num.parse::<u32>()
        {
            max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
        }
    }
    max_idx.map_or(0, |m| m + 1)
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
}

fn parse_single_header<R: Read + Seek>(r: &mut R, shard_idx: usize) -> Result<ParsedHeader> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        bail!("invalid GGUF magic: {:?}", magic);
    }

    let version = read_u32(r)?;
    if !(2..=3).contains(&version) {
        bail!("unsupported GGUF version: {version}");
    }

    let tensor_count = read_u64(r)? as usize;
    let kv_count = read_u64(r)? as usize;

    let mut kv = HashMap::with_capacity(kv_count.min(1024));
    for _ in 0..kv_count {
        let key = read_string(r)?;
        let val_type = read_u32(r)?;
        let val = read_kv_value(r, val_type)?;
        kv.insert(key, val);
    }

    let alignment = kv
        .get("general.alignment")
        .and_then(KvValue::as_u32)
        .map(u64::from)
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

    let mut raw_tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_string(r)?;
        let n_dims = read_u32(r)? as usize;
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

    let tensors = raw_tensors
        .into_iter()
        .map(|(name, dims, ggml_type, rel_offset)| TensorLocation {
            name,
            shard_idx,
            abs_offset: data_start + rel_offset,
            ggml_type,
            dims,
        })
        .collect();

    Ok(ParsedHeader { kv, tensors })
}

fn read_kv_value<R: Read + Seek>(r: &mut R, val_type: u32) -> Result<KvValue> {
    match val_type {
        0 => {
            skip_bytes(r, 1)?;
            Ok(KvValue::Other)
        } // UINT8
        1 => {
            skip_bytes(r, 1)?;
            Ok(KvValue::Other)
        } // INT8
        2 => {
            skip_bytes(r, 2)?;
            Ok(KvValue::Other)
        } // UINT16
        3 => {
            skip_bytes(r, 2)?;
            Ok(KvValue::Other)
        } // INT16
        4 => Ok(KvValue::U32(read_u32(r)?)),        // UINT32
        5 => Ok(KvValue::I32(read_u32(r)? as i32)), // INT32
        6 => {
            skip_bytes(r, 4)?;
            Ok(KvValue::Other)
        } // FLOAT32
        7 => {
            skip_bytes(r, 1)?;
            Ok(KvValue::Other)
        } // BOOL
        8 => Ok(KvValue::Str(read_string(r)?)),     // STRING
        9 => {
            // ARRAY
            let elem_type = read_u32(r)?;
            let len = read_u64(r)?;
            if let Some(elem_size) = fixed_kv_type_size(elem_type) {
                skip_bytes(r, len * elem_size)?;
            } else {
                for _ in 0..len {
                    let _ = read_kv_value(r, elem_type)?;
                }
            }
            Ok(KvValue::Other)
        }
        10 => Ok(KvValue::U64(read_u64(r)?)), // UINT64
        11 => {
            skip_bytes(r, 8)?;
            Ok(KvValue::Other)
        } // INT64
        12 => {
            skip_bytes(r, 8)?;
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

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn skip_bytes<R: Seek>(r: &mut R, n: u64) -> Result<()> {
    r.seek(SeekFrom::Current(n as i64))?;
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
