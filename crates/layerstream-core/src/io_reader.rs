//! High-throughput coalesced 4K-aligned O_DIRECT pread reader for GGUF layer streaming.
//!
//! Optimizations:
//! - Caches planned `CoalescedSpan`s per `layer_idx` after sweep 1 (zero CStr/HashMap
//!   overhead in the hot decode loop).
//! - Splits large layer spans (>= 16 MiB) across `io_threads` (default 4) parallel
//!   4K-aligned `pread` calls using `std::thread::scope`, driving NVMe queue depth > 1
//!   and parallelizing dm-crypt/LUKS decryption across CPU cores.
//! - Uses `O_DIRECT` on Linux, `F_NOCACHE` on macOS, and `FILE_FLAG_NO_BUFFERING` on
//!   Windows to avoid polluting the host OS page cache.

use std::alloc::{Layout, alloc, dealloc};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};

use crate::gguf_index::{CoalescedSpan, DIRECT_IO_ALIGN, GgufIndex, align_up};

const PARALLEL_READ_MIN_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_IO_THREADS: usize = 4;

/// Page-aligned (4096-byte) owned byte buffer suitable for Linux `O_DIRECT` / Windows unbuffered I/O.
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    pub fn new(min_bytes: usize) -> Result<Self> {
        let len = align_up(
            min_bytes.max(DIRECT_IO_ALIGN as usize) as u64,
            DIRECT_IO_ALIGN,
        ) as usize;
        let layout = Layout::from_size_align(len, DIRECT_IO_ALIGN as usize)
            .context("invalid AlignedBuf layout")?;
        let raw = unsafe { alloc(layout) };
        let Some(ptr) = NonNull::new(raw) else {
            bail!("failed to allocate {len}-byte 4K-aligned IO buffer");
        };
        Ok(Self { ptr, len, layout })
    }

    pub fn ensure_capacity(&mut self, min_bytes: usize) -> Result<()> {
        if min_bytes <= self.len {
            return Ok(());
        }
        *self = Self::new(min_bytes)?;
        Ok(())
    }

    pub fn as_mut_slice(&mut self, len: usize) -> &mut [u8] {
        assert!(len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

struct ShardHandle {
    direct: Option<File>,
    buffered: File,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct IoStats {
    pub bytes_read: u64,
    pub span_reads: u64,
    pub direct_io_active: bool,
}

pub struct LayerIoPool {
    index: GgufIndex,
    shards: Vec<ShardHandle>,
    direct_io_active: bool,
    io_threads: usize,
    scratch: Mutex<AlignedBuf>,
    span_cache: Mutex<HashMap<i32, Vec<CoalescedSpan>>>,
    bytes_read: AtomicU64,
    span_reads: AtomicU64,
}

impl LayerIoPool {
    pub fn new(index: GgufIndex, prefer_direct_io: bool) -> Result<Self> {
        let mut shards = Vec::with_capacity(index.shards.len());
        let mut all_direct = prefer_direct_io;

        for path in &index.shards {
            let buffered = File::open(path)
                .with_context(|| format!("failed to open shard {}", path.display()))?;
            let direct = if prefer_direct_io {
                open_direct(path)
            } else {
                None
            };
            if direct.is_none() {
                all_direct = false;
            }
            shards.push(ShardHandle { direct, buffered });
        }

        let io_threads = std::thread::available_parallelism()
            .map(|n| n.get().clamp(1, DEFAULT_IO_THREADS))
            .unwrap_or(DEFAULT_IO_THREADS);

        Ok(Self {
            index,
            shards,
            direct_io_active: all_direct,
            io_threads,
            scratch: Mutex::new(AlignedBuf::new(4 * 1024 * 1024)?),
            span_cache: Mutex::new(HashMap::new()),
            bytes_read: AtomicU64::new(0),
            span_reads: AtomicU64::new(0),
        })
    }

    pub fn index(&self) -> &GgufIndex {
        &self.index
    }

    pub fn stats(&self) -> IoStats {
        IoStats {
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            span_reads: self.span_reads.load(Ordering::Relaxed),
            direct_io_active: self.direct_io_active,
        }
    }

    /// Loads a cached layer plan if already computed, avoiding CStr/HashMap work on sweeps 2..N.
    pub fn load_cached_layer_into(&self, layer_idx: i32, dst: &mut [u8]) -> Result<bool> {
        if layer_idx < 0 {
            return Ok(false);
        }
        let cached = {
            let guard = self
                .span_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("span_cache mutex poisoned"))?;
            guard.get(&layer_idx).cloned()
        };
        let Some(spans) = cached else {
            return Ok(false);
        };

        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| anyhow::anyhow!("IO scratch mutex poisoned"))?;
        for span in &spans {
            self.read_span_into(span, &mut scratch, dst)?;
        }
        Ok(true)
    }

    /// Plans, caches (for `layer_idx >= 0`), and loads all requested `(name, dst_offset, len)` tensors.
    pub fn load_layer_tensors_into(
        &self,
        layer_idx: i32,
        requests: &[(&str, usize, usize)],
        dst: &mut [u8],
    ) -> Result<()> {
        let spans = self.index.plan_coalesced_spans(requests)?;
        if layer_idx >= 0
            && let Ok(mut cache) = self.span_cache.lock()
        {
            cache.insert(layer_idx, spans.clone());
        }

        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| anyhow::anyhow!("IO scratch mutex poisoned"))?;
        for span in &spans {
            self.read_span_into(span, &mut scratch, dst)?;
        }
        Ok(())
    }

    /// Reads all requested `(name, dst_offset, len)` tensors into `dst` without caching.
    pub fn load_tensors_into(
        &self,
        requests: &[(&str, usize, usize)],
        dst: &mut [u8],
    ) -> Result<()> {
        self.load_layer_tensors_into(-1, requests, dst)
    }

    fn read_span_into(
        &self,
        span: &CoalescedSpan,
        scratch: &mut AlignedBuf,
        dst: &mut [u8],
    ) -> Result<()> {
        scratch.ensure_capacity(span.aligned_len)?;
        let handle = &self.shards[span.shard_idx];
        let read_buf = scratch.as_mut_slice(span.aligned_len);

        let mut used_direct = false;
        if let Some(direct_file) = &handle.direct {
            used_direct = self.read_parallel_or_single(
                direct_file,
                read_buf,
                span.aligned_offset,
                span.valid_len,
            );
        }

        if !used_direct {
            let valid_buf = &mut read_buf[..span.valid_len];
            pread_all(&handle.buffered, valid_buf, span.aligned_offset)?;
        }

        self.bytes_read
            .fetch_add(span.valid_len as u64, Ordering::Relaxed);
        self.span_reads.fetch_add(1, Ordering::Relaxed);

        for slice in &span.slices {
            let src = &read_buf[slice.span_offset..slice.span_offset + slice.len];
            let out = &mut dst[slice.dst_offset..slice.dst_offset + slice.len];
            out.copy_from_slice(src);
        }

        Ok(())
    }

    /// Drives NVMe queue depth and parallel dm-crypt decryption by splitting large
    /// 4K-aligned direct-IO spans across `io_threads` concurrent `pread` calls.
    fn read_parallel_or_single(
        &self,
        file: &File,
        buf: &mut [u8],
        base_offset: u64,
        valid_len: usize,
    ) -> bool {
        if self.io_threads <= 1 || valid_len < PARALLEL_READ_MIN_BYTES {
            return matches!(pread_exact_or_eof(file, buf, base_offset), Ok(n) if n >= valid_len);
        }

        let chunk_raw = buf.len().div_ceil(self.io_threads);
        let chunk_size = align_up(chunk_raw as u64, DIRECT_IO_ALIGN) as usize;
        if chunk_size == 0 || chunk_size >= buf.len() {
            return matches!(pread_exact_or_eof(file, buf, base_offset), Ok(n) if n >= valid_len);
        }

        let ok = std::sync::atomic::AtomicBool::new(true);
        std::thread::scope(|s| {
            for (idx, chunk) in buf.chunks_mut(chunk_size).enumerate() {
                let rel_off = idx * chunk_size;
                let file_off = base_offset + rel_off as u64;
                let expected = valid_len.saturating_sub(rel_off).min(chunk.len());
                if expected == 0 {
                    continue;
                }
                let ok_ref = &ok;
                s.spawn(move || match pread_exact_or_eof(file, chunk, file_off) {
                    Ok(n) if n >= expected => {}
                    _ => ok_ref.store(false, Ordering::Relaxed),
                });
            }
        });
        ok.load(Ordering::Relaxed)
    }
}

#[cfg(target_os = "linux")]
fn open_direct(path: &std::path::Path) -> Option<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .ok()
}

#[cfg(target_os = "macos")]
fn open_direct(path: &std::path::Path) -> Option<File> {
    use std::os::unix::io::AsRawFd;
    let file = OpenOptions::new().read(true).open(path).ok()?;
    let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
    if rc == 0 { Some(file) } else { None }
}

#[cfg(windows)]
fn open_direct(path: &std::path::Path) -> Option<File> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_NO_BUFFERING)
        .open(path)
        .ok()
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn open_direct(_path: &std::path::Path) -> Option<File> {
    None
}

#[cfg(unix)]
fn file_read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    file.read_at(buf, offset)
}

#[cfg(windows)]
fn file_read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    file.seek_read(buf, offset)
}

fn pread_exact_or_eof(file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<usize> {
    let mut total = 0usize;
    while !buf.is_empty() {
        match file_read_at(file, buf, offset) {
            Ok(0) => break,
            Ok(n) => {
                total += n;
                offset += n as u64;
                buf = &mut buf[n..];
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(total)
}

fn pread_all(file: &File, mut buf: &mut [u8], mut offset: u64) -> Result<()> {
    while !buf.is_empty() {
        match file_read_at(file, buf, offset) {
            Ok(0) => bail!("unexpected EOF during pread at offset {offset}"),
            Ok(n) => {
                offset += n as u64;
                buf = &mut buf[n..];
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
