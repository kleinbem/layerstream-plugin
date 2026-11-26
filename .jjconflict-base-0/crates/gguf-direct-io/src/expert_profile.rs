//! Persistent hot-expert cache profiles for zero-cold-start MoE pre-warming.
//!
//! Inspired by hierarchical MoE engines that remember hot experts across restarts:
//! saves the `(shard_idx, file_offset, len)` tuples resident in the S3-FIFO expert
//! cache to a compact binary profile (`LSEP` magic) and pre-warms them sequentially
//! on subsequent opens so worker evictions do not pay cold-start NVMe misses.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;

const PROFILE_MAGIC: &[u8; 4] = b"LSEP";
const PROFILE_VERSION: u32 = 1;
const RECORD_BYTES: usize = 4 + 8 + 8; // shard_idx(u32) + file_offset(u64) + len(u64)

/// Serializable hot-expert slice descriptor (sorted by shard and file offset for
/// sequential NVMe pre-warming).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HotSliceEntry {
    pub shard_idx: u32,
    pub file_offset: u64,
    pub len: u64,
}

/// Writes a sorted list of hot expert slices atomically to `path`.
pub fn write_expert_profile(path: &Path, entries: &[HotSliceEntry]) -> io::Result<usize> {
    let mut sorted = entries.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = path.with_extension("lsep.tmp");
    let mut buf = Vec::with_capacity(12 + sorted.len() * RECORD_BYTES);
    buf.extend_from_slice(PROFILE_MAGIC);
    buf.extend_from_slice(&PROFILE_VERSION.to_le_bytes());
    buf.extend_from_slice(&(sorted.len() as u32).to_le_bytes());
    for e in &sorted {
        buf.extend_from_slice(&e.shard_idx.to_le_bytes());
        buf.extend_from_slice(&e.file_offset.to_le_bytes());
        buf.extend_from_slice(&e.len.to_le_bytes());
    }

    let mut f = File::create(&tmp_path)?;
    f.write_all(&buf)?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp_path, path)?;
    Ok(sorted.len())
}

/// Reads a previously saved hot-expert profile from `path`.
pub fn read_expert_profile(path: &Path) -> io::Result<Vec<HotSliceEntry>> {
    let mut f = File::open(path)?;
    let mut raw = Vec::new();
    f.read_to_end(&mut raw)?;
    if raw.len() < 12 || &raw[0..4] != PROFILE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid expert profile magic",
        ));
    }
    let version = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    if version != PROFILE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported expert profile version {version}"),
        ));
    }
    let count = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
    let expected = 12usize
        .checked_add(count.saturating_mul(RECORD_BYTES))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "profile size overflow"))?;
    if raw.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated or corrupt expert profile",
        ));
    }
    let mut out = Vec::with_capacity(count);
    let mut off = 12;
    for _ in 0..count {
        let shard_idx = u32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
        let file_offset = u64::from_le_bytes([
            raw[off + 4],
            raw[off + 5],
            raw[off + 6],
            raw[off + 7],
            raw[off + 8],
            raw[off + 9],
            raw[off + 10],
            raw[off + 11],
        ]);
        let len = u64::from_le_bytes([
            raw[off + 12],
            raw[off + 13],
            raw[off + 14],
            raw[off + 15],
            raw[off + 16],
            raw[off + 17],
            raw[off + 18],
            raw[off + 19],
        ]);
        out.push(HotSliceEntry {
            shard_idx,
            file_offset,
            len,
        });
        off += RECORD_BYTES;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_sorted_deduped_profile_and_rejects_corrupt_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nested/olmoe.lsep");
        let entries = vec![
            HotSliceEntry {
                shard_idx: 0,
                file_offset: 8192,
                len: 4096,
            },
            HotSliceEntry {
                shard_idx: 0,
                file_offset: 4096,
                len: 2048,
            },
            HotSliceEntry {
                shard_idx: 0,
                file_offset: 4096,
                len: 2048,
            },
        ];
        let written = write_expert_profile(&path, &entries).expect("write");
        assert_eq!(written, 2);

        let loaded = read_expert_profile(&path).expect("read");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].file_offset, 4096);
        assert_eq!(loaded[1].file_offset, 8192);

        let bad = dir.path().join("bad.lsep");
        fs::write(&bad, b"NOPE12345678").expect("write bad");
        assert!(read_expert_profile(&bad).is_err());

        let bad_ver = dir.path().join("bad_ver.lsep");
        let mut raw = b"LSEP".to_vec();
        raw.extend_from_slice(&99u32.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());
        fs::write(&bad_ver, raw).expect("write bad_ver");
        assert!(read_expert_profile(&bad_ver).is_err());
    }
}
