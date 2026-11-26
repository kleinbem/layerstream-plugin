//! Persistent hot-expert cache profiles for zero-cold-start MoE pre-warming.
//!
//! Inspired by hierarchical MoE engines that remember hot experts across restarts:
//! saves the `(shard_idx, file_offset, len)` tuples resident in the S3-FIFO expert
//! cache to a compact binary profile (`LSEP` magic) and pre-warms them sequentially
//! on subsequent opens so worker evictions do not pay cold-start NVMe misses.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const PROFILE_MAGIC: &[u8; 4] = b"LSEP";
const PROFILE_VERSION: u32 = 1;
const RECORD_BYTES: usize = 4 + 8 + 8; // shard_idx(u32) + file_offset(u64) + len(u64)
/// Upper bound on profile records (~20 MiB) to prevent OOM DoS on corrupt/hostile `.lsep` files.
const MAX_PROFILE_ENTRIES: usize = 1_000_000;
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Serializable hot-expert slice descriptor (sorted by shard and file offset for
/// sequential NVMe pre-warming).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HotSliceEntry {
    pub shard_idx: u32,
    pub file_offset: u64,
    pub len: u64,
}

fn create_unique_temp_file(parent: &Path, prefix: &str) -> io::Result<(PathBuf, File)> {
    let pid = std::process::id();
    for _ in 0..64 {
        let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{prefix}-{pid}-{seq}.tmp"));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(f) => return Ok((candidate, f)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "failed to create unique temporary expert profile file",
    ))
}

/// Writes a sorted list of hot expert slices atomically to `path`.
pub fn write_expert_profile(path: &Path, entries: &[HotSliceEntry]) -> io::Result<usize> {
    let mut sorted = entries.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() > MAX_PROFILE_ENTRIES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "expert profile entry count {} exceeds maximum {MAX_PROFILE_ENTRIES}",
                sorted.len()
            ),
        ));
    }

    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => {
            fs::create_dir_all(p)?;
            p
        }
        _ => Path::new("."),
    };

    let mut buf = Vec::with_capacity(12 + sorted.len() * RECORD_BYTES);
    buf.extend_from_slice(PROFILE_MAGIC);
    buf.extend_from_slice(&PROFILE_VERSION.to_le_bytes());
    buf.extend_from_slice(&(sorted.len() as u32).to_le_bytes());
    for e in &sorted {
        buf.extend_from_slice(&e.shard_idx.to_le_bytes());
        buf.extend_from_slice(&e.file_offset.to_le_bytes());
        buf.extend_from_slice(&e.len.to_le_bytes());
    }

    let (tmp_path, mut f) = create_unique_temp_file(parent, "layerstream-lsep")?;
    let write_res = (|| -> io::Result<()> {
        f.write_all(&buf)?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp_path, path)
    })();
    if write_res.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }
    write_res?;
    Ok(sorted.len())
}

/// Reads a previously saved hot-expert profile from `path`.
pub fn read_expert_profile(path: &Path) -> io::Result<Vec<HotSliceEntry>> {
    let mut f = File::open(path)?;
    // Read and validate the 12-byte header first so giant/corrupt files cannot trigger unbounded allocation.
    let mut header = [0u8; 12];
    f.read_exact(&mut header)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid expert profile magic"))?;
    if &header[0..4] != PROFILE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid expert profile magic",
        ));
    }
    let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    if version != PROFILE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported expert profile version {version}"),
        ));
    }
    let count = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
    if count > MAX_PROFILE_ENTRIES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expert profile entry count {count} exceeds maximum {MAX_PROFILE_ENTRIES}"),
        ));
    }
    let payload_len = count
        .checked_mul(RECORD_BYTES)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "profile size overflow"))?;
    let mut raw = vec![0u8; payload_len];
    f.read_exact(&mut raw).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated or corrupt expert profile",
        )
    })?;
    let mut trailing = [0u8; 1];
    if f.read(&mut trailing)? != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated or corrupt expert profile",
        ));
    }
    let mut out = Vec::with_capacity(count);
    let mut off = 0;
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

        // Oversized entry count (> MAX_PROFILE_ENTRIES) fails header check without large allocation.
        let huge_cnt = dir.path().join("huge.lsep");
        let mut huge = b"LSEP".to_vec();
        huge.extend_from_slice(&1u32.to_le_bytes());
        huge.extend_from_slice(&((MAX_PROFILE_ENTRIES as u32) + 1).to_le_bytes());
        fs::write(&huge_cnt, huge).expect("write huge");
        assert!(read_expert_profile(&huge_cnt).is_err());

        // Trailing bytes after declared count fail closed.
        let trailing = dir.path().join("trailing.lsep");
        let mut tr = b"LSEP".to_vec();
        tr.extend_from_slice(&1u32.to_le_bytes());
        tr.extend_from_slice(&0u32.to_le_bytes());
        tr.push(0xFF);
        fs::write(&trailing, tr).expect("write trailing");
        assert!(read_expert_profile(&trailing).is_err());

        // Short header and truncated payload fail closed cleanly.
        let short_hdr = dir.path().join("short.lsep");
        fs::write(&short_hdr, b"LS").expect("write short");
        assert!(read_expert_profile(&short_hdr).is_err());

        let trunc_body = dir.path().join("trunc.lsep");
        let mut tb = b"LSEP".to_vec();
        tb.extend_from_slice(&1u32.to_le_bytes());
        tb.extend_from_slice(&2u32.to_le_bytes());
        tb.extend_from_slice(&[0u8; 4]);
        fs::write(&trunc_body, tb).expect("write trunc");
        assert!(read_expert_profile(&trunc_body).is_err());
    }
}
