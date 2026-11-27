//! Fast startup sequential `O_DIRECT` storage throughput probes.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::gguf_index::{DIRECT_IO_ALIGN, align_up};
use crate::io_reader::{AlignedBuf, open_direct, pread_exact_or_eof};
use crate::{GgufIoError, Result};

/// Measures sequential read throughput `(mib_per_s, direct_io_active)` on `path` by reading
/// up to 16 MiB (4K-aligned). Intended for fast (~10-50 ms) startup preflight diagnostics.
pub fn probe_storage_mib_per_s(
    path: &std::path::Path,
    prefer_direct_io: bool,
) -> Result<(f64, bool)> {
    let buffered = File::open(path).map_err(|e| {
        GgufIoError::io(
            format!("failed to open {} for storage probe", path.display()),
            e,
        )
    })?;
    let file_len = buffered.metadata()?.len();
    if file_len < 4096 {
        return Ok((0.0, false));
    }
    let direct = if prefer_direct_io {
        open_direct(path)
    } else {
        None
    };
    let mut direct_active = direct.is_some();
    let probe_bytes = (file_len as usize).min(16 * 1024 * 1024);
    let aligned_len = align_up(probe_bytes as u64, DIRECT_IO_ALIGN) as usize;
    let mut buf = AlignedBuf::new(aligned_len)?;
    let slice = buf.as_mut_slice(aligned_len);
    let mut t0 = std::time::Instant::now();
    let n = if let Some(ref df) = direct {
        match pread_exact_or_eof(df, slice, 0) {
            Ok(n) => n,
            Err(_) => {
                // Some filesystems (e.g. FUSE/overlayfs) accept O_DIRECT on open() but fail on read().
                direct_active = false;
                t0 = std::time::Instant::now();
                pread_exact_or_eof(&buffered, &mut slice[..probe_bytes], 0)?
            }
        }
    } else {
        pread_exact_or_eof(&buffered, &mut slice[..probe_bytes], 0)?
    };
    let elapsed_s = t0.elapsed().as_secs_f64().max(1e-6);
    let mib_per_s = (n as f64 / (1024.0 * 1024.0)) / elapsed_s;
    Ok((mib_per_s, direct_active))
}

fn find_existing_probe_file(dir: &std::path::Path, depth: usize) -> Option<std::path::PathBuf> {
    const MIN_PROBE_BYTES: u64 = 16 * 1024 * 1024;
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if let Ok(meta) = entry.metadata() {
            if meta.is_file() && meta.len() >= MIN_PROBE_BYTES {
                return Some(path);
            }
            if meta.is_dir()
                && depth > 0
                && let Some(found) = find_existing_probe_file(&path, depth - 1)
            {
                return Some(found);
            }
        }
    }
    None
}

/// Probes sequential `O_DIRECT` read throughput (MiB/s) on the filesystem backing `dir`.
/// Reuses an existing >=16 MiB file when available, or briefly writes/removes a 16 MiB temp file.
pub fn probe_dir_storage_mib_per_s(dir: &std::path::Path) -> Option<f64> {
    if let Some(existing) = find_existing_probe_file(dir, 2)
        && let Ok((mib_s, _)) = probe_storage_mib_per_s(&existing, true)
        && mib_s >= 10.0
    {
        return Some(mib_s);
    }

    struct TempCleanup(std::path::PathBuf);
    impl Drop for TempCleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    static PROBE_SEQ: AtomicU64 = AtomicU64::new(0);
    let pid = std::process::id();
    let mut created: Option<(std::path::PathBuf, File)> = None;
    for _ in 0..32 {
        let seq = PROBE_SEQ.fetch_add(1, Ordering::Relaxed);
        let candidate = dir.join(format!(".layerstream-probe-{pid}-{seq}.tmp"));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(f) => {
                created = Some((candidate, f));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => return None,
        }
    }
    let (tmp_path, mut f) = created?;
    let _cleanup = TempCleanup(tmp_path.clone());
    let chunk = vec![0xA5u8; 1024 * 1024];
    for _ in 0..16 {
        f.write_all(&chunk).ok()?;
    }
    f.sync_all().ok()?;
    drop(f);

    probe_storage_mib_per_s(&tmp_path, true)
        .ok()
        .map(|(mib_s, _)| mib_s)
        .filter(|&bw| bw.is_finite() && bw > 0.0)
}
