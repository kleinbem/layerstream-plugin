//! Coalesced 4K-aligned O_DIRECT pread reader for GGUF layer streaming.
//!
//! Keeps host Linux page cache clean when sweeping models larger than RAM, and
//! falls back automatically to buffered pread on filesystems (e.g. tmpfs) where
//! O_DIRECT is unsupported.

use std::alloc::{Layout, alloc, dealloc};
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

/// Page-aligned (4096-byte) owned byte buffer suitable for Linux `O_DIRECT`.
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
    scratch: Mutex<AlignedBuf>,
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

        Ok(Self {
            index,
            shards,
            direct_io_active: all_direct,
            scratch: Mutex::new(AlignedBuf::new(4 * 1024 * 1024)?),
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

    /// Reads all requested `(name, dst_offset, len)` tensors into `dst` using
    /// coalesced 4K-aligned O_DIRECT pread calls.
    pub fn load_tensors_into(
        &self,
        requests: &[(&str, usize, usize)],
        dst: &mut [u8],
    ) -> Result<()> {
        let spans = self.index.plan_coalesced_spans(requests)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| anyhow::anyhow!("IO scratch mutex poisoned"))?;

        for span in &spans {
            self.read_span_into(span, &mut scratch, dst)?;
        }
        Ok(())
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
            match pread_exact_or_eof(direct_file, read_buf, span.aligned_offset) {
                Ok(n) if n >= span.valid_len => {
                    used_direct = true;
                }
                _ => {
                    // Fallback cleanly to buffered pread if a tail/FS rejects O_DIRECT.
                }
            }
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
