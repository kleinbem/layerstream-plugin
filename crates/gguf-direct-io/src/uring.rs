//! Linux `io_uring` batched asynchronous read engine for `gguf-direct-io`.
//!
//! Submits multiple 4K-aligned `O_DIRECT` chunks (for large dense layers) or
//! multiple scattered sparse MoE expert slices in a single `io_uring_enter`
//! syscall, saturating NVMe queue depth without OS thread wakeups.
//!
//! Automatically falls back to `None` (standard `pread` path) when:
//! - `LAYERSTREAM_IO_URING=0` is set, or
//! - the host kernel / container seccomp policy denies `io_uring_setup` (`EPERM` / `ENOSYS`).

use std::os::unix::io::RawFd;

use io_uring::{IoUring, opcode, types};

use crate::gguf_index::DIRECT_IO_ALIGN;
use crate::{GgufIoError, Result};

/// Minimum large-span size (4 MiB) before splitting across batched `io_uring` SQEs.
pub(crate) const URING_SPLIT_MIN_BYTES: usize = 4 * 1024 * 1024;
/// Target chunk size (2 MiB) per SQE when splitting a large sequential `O_DIRECT` span.
const URING_CHUNK_BYTES: usize = 2 * 1024 * 1024;
/// Maximum ring depth (power of two).
const URING_ENTRIES: u32 = 128;

/// Single disjoint read operation for `UringReader::read_batch`.
pub(crate) struct UringReadOp {
    pub fd: RawFd,
    pub ptr: *mut u8,
    pub len: usize,
    pub offset: u64,
    pub min_valid: usize,
}

pub(crate) struct UringReader {
    ring: IoUring,
}

// SAFETY: `UringReader` is always accessed behind `parking_lot::Mutex` inside `LayerIoPool`,
// ensuring exclusive single-threaded submission and completion reaping.
unsafe impl Send for UringReader {}

impl UringReader {
    /// Probes `io_uring_setup` unless disabled via `LAYERSTREAM_IO_URING=0`.
    /// Returns `None` gracefully on older kernels or restricted seccomp sandboxes.
    pub(crate) fn try_new() -> Option<Self> {
        if std::env::var("LAYERSTREAM_IO_URING")
            .ok()
            .as_deref()
            .is_some_and(|v| v == "0" || v.eq_ignore_ascii_case("false"))
        {
            return None;
        }
        match IoUring::new(URING_ENTRIES) {
            Ok(ring) => {
                tracing::debug!(
                    entries = URING_ENTRIES,
                    "enabled Linux io_uring batched reader"
                );
                Some(Self { ring })
            }
            Err(err) => {
                // coverage: only hit on kernels < 5.1 or seccomp sandboxes denying io_uring_setup.
                tracing::debug!(error = %err, "io_uring unavailable; falling back to pread");
                None
            }
        }
    }

    /// Splits a large 4K-aligned `O_DIRECT` span into concurrent `io_uring` read SQEs.
    /// Returns `true` if at least `valid_len` bytes were read cleanly, or `false` to fall back.
    pub(crate) fn read_aligned_span(
        &mut self,
        fd: RawFd,
        buf: &mut [u8],
        base_offset: u64,
        valid_len: usize,
    ) -> bool {
        if valid_len < URING_SPLIT_MIN_BYTES || buf.len() < valid_len {
            return false;
        }
        let align = DIRECT_IO_ALIGN as usize;
        let max_chunks = (URING_ENTRIES as usize / 2).max(1);
        let raw_chunk = buf.len().div_ceil(max_chunks).max(URING_CHUNK_BYTES);
        let chunk_size = (raw_chunk + align - 1) & !(align - 1);

        let mut ops = Vec::with_capacity(buf.len().div_ceil(chunk_size));
        let mut rel = 0usize;
        while rel < buf.len() {
            let len = (buf.len() - rel).min(chunk_size);
            let min_valid = valid_len.saturating_sub(rel).min(len);
            ops.push(UringReadOp {
                fd,
                // SAFETY: `rel < buf.len()` and `rel + len <= buf.len()`; all slices are disjoint.
                ptr: unsafe { buf.as_mut_ptr().add(rel) },
                len,
                offset: base_offset + rel as u64,
                min_valid,
            });
            rel += len;
        }

        self.read_batch(&ops).is_ok()
    }

    /// Submits a batch of disjoint read operations and reaps all completions,
    /// transparently retrying any short reads (or `EINTR` / `EAGAIN`).
    /// Returns total bytes read across all operations.
    pub(crate) fn read_batch(&mut self, ops: &[UringReadOp]) -> Result<u64> {
        if ops.is_empty() {
            return Ok(0);
        }

        struct Pending {
            fd: RawFd,
            ptr: *mut u8,
            remaining: usize,
            offset: u64,
            min_valid: usize,
            read_so_far: usize,
        }

        let mut states: Vec<Pending> = ops
            .iter()
            .map(|op| Pending {
                fd: op.fd,
                ptr: op.ptr,
                remaining: op.len,
                offset: op.offset,
                min_valid: op.min_valid,
                read_so_far: 0,
            })
            .collect();

        let mut queue: Vec<usize> = (0..states.len()).collect();
        let mut total_bytes = 0u64;
        let mut first_err: Option<GgufIoError> = None;

        while !queue.is_empty() {
            let batch_end = queue.len().min(URING_ENTRIES as usize);
            let batch: Vec<usize> = queue.drain(..batch_end).collect();
            let mut submitted = 0usize;

            for &idx in &batch {
                let st = &states[idx];
                if st.remaining == 0 {
                    continue;
                }
                let len_u32 = u32::try_from(st.remaining).unwrap_or(u32::MAX);
                let sqe = opcode::Read::new(types::Fd(st.fd), st.ptr, len_u32)
                    .offset(st.offset)
                    .build()
                    .user_data(idx as u64);
                // SAFETY: Every `Pending` buffer pointer remains valid and disjoint until
                // `submit_and_wait(submitted)` reaps its corresponding CQE below.
                unsafe {
                    if self.ring.submission().push(&sqe).is_err() {
                        // coverage: batch size is bounded by `URING_ENTRIES`, so SQ never overflows.
                        queue.push(idx);
                        break;
                    }
                }
                submitted += 1;
            }

            if submitted == 0 {
                break;
            }

            loop {
                match self.ring.submit_and_wait(submitted) {
                    Ok(_) => break,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        return Err(GgufIoError::io("io_uring submit_and_wait failed", e));
                    }
                }
            }

            let mut reaped = 0usize;
            while reaped < submitted {
                let mut cq = self.ring.completion();
                for cqe in &mut cq {
                    reaped += 1;
                    let idx = cqe.user_data() as usize;
                    let res = cqe.result();
                    if res < 0 {
                        let errno = -res;
                        if errno == libc::EINTR || errno == libc::EAGAIN {
                            queue.push(idx);
                        } else if first_err.is_none() {
                            let io_err = std::io::Error::from_raw_os_error(errno);
                            first_err = Some(GgufIoError::io(
                                format!("io_uring read failed at offset {}", states[idx].offset),
                                io_err,
                            ));
                        }
                        continue;
                    }
                    let n = res as usize;
                    let st = &mut states[idx];
                    st.read_so_far += n;
                    total_bytes = total_bytes.saturating_add(n as u64);
                    if n == 0 {
                        // EOF reached: valid if we satisfied `min_valid`, else error.
                        if st.read_so_far < st.min_valid && first_err.is_none() {
                            first_err = Some(GgufIoError::io(
                                format!("unexpected EOF in io_uring read at offset {}", st.offset),
                                std::io::Error::new(
                                    std::io::ErrorKind::UnexpectedEof,
                                    "short io_uring read",
                                ),
                            ));
                        }
                    } else if n < st.remaining {
                        // Short read: advance pointer/offset and resubmit if more valid bytes needed.
                        // SAFETY: `n < st.remaining`, so advancing `st.ptr` stays within the original slice.
                        st.ptr = unsafe { st.ptr.add(n) };
                        st.offset += n as u64;
                        st.remaining -= n;
                        if st.read_so_far < st.min_valid {
                            queue.push(idx);
                        }
                    }
                }
                drop(cq);
                if reaped < submitted {
                    // coverage: only hit if kernel returned early before all CQEs arrived.
                    let _ = self.ring.submit_and_wait(submitted - reaped);
                }
            }

            if let Some(err) = first_err {
                return Err(err);
            }
        }

        Ok(total_bytes)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::undocumented_unsafe_blocks)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::io::AsRawFd;

    #[test]
    fn uring_respects_disable_env_and_handles_batch_and_errors() {
        unsafe { std::env::set_var("LAYERSTREAM_IO_URING", "0") };
        assert!(UringReader::try_new().is_none());
        unsafe { std::env::remove_var("LAYERSTREAM_IO_URING") };

        let Some(mut reader) = UringReader::try_new() else {
            eprintln!("skipping io_uring runtime test: kernel/sandbox denied io_uring_setup");
            return;
        };

        // Empty batch returns 0.
        assert_eq!(reader.read_batch(&[]).unwrap(), 0);

        // Create a 5 MiB deterministic temp file to exercise both `read_batch` and `read_aligned_span`.
        let mut f = tempfile::NamedTempFile::new().unwrap();
        let size = 5 * 1024 * 1024;
        let data: Vec<u8> = (0..size).map(|i| ((i * 31 + 7) & 0xff) as u8).collect();
        f.write_all(&data).unwrap();
        f.flush().unwrap();
        let fd = f.as_file().as_raw_fd();

        // 1. Multi-op disjoint batch read.
        let mut b0 = vec![0u8; 4096];
        let mut b1 = vec![0u8; 8192];
        let ops = [
            UringReadOp {
                fd,
                ptr: b0.as_mut_ptr(),
                len: b0.len(),
                offset: 0,
                min_valid: b0.len(),
            },
            UringReadOp {
                fd,
                ptr: b1.as_mut_ptr(),
                len: b1.len(),
                offset: 1024 * 1024,
                min_valid: b1.len(),
            },
        ];
        let read_n = reader.read_batch(&ops).unwrap();
        assert_eq!(read_n, (b0.len() + b1.len()) as u64);
        assert_eq!(&b0[..], &data[..4096]);
        assert_eq!(&b1[..], &data[1024 * 1024..1024 * 1024 + 8192]);

        // 2. Large aligned span split (>= 4 MiB) + small/invalid guard checks.
        let mut small = vec![0u8; 4096];
        assert!(!reader.read_aligned_span(fd, &mut small, 0, 4096));
        assert!(!reader.read_aligned_span(fd, &mut small, 0, URING_SPLIT_MIN_BYTES));

        let mut big = vec![0u8; size];
        assert!(reader.read_aligned_span(fd, &mut big, 0, size));
        assert_eq!(big, data);

        // 3. Unexpected EOF error when reading past end of file with `min_valid > 0`.
        let mut eof_buf = vec![0u8; 4096];
        let eof_op = [UringReadOp {
            fd,
            ptr: eof_buf.as_mut_ptr(),
            len: eof_buf.len(),
            offset: size as u64,
            min_valid: 16,
        }];
        assert!(reader.read_batch(&eof_op).is_err());

        // 4. Invalid fd returns EBADF error.
        let bad_op = [UringReadOp {
            fd: -1,
            ptr: eof_buf.as_mut_ptr(),
            len: 64,
            offset: 0,
            min_valid: 64,
        }];
        assert!(reader.read_batch(&bad_op).is_err());
    }
}
