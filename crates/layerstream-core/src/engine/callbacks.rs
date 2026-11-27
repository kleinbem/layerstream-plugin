//! C-FFI I/O callback trampolines and streaming UTF-8 token piece reassembly.

use std::ffi::{CStr, c_char, c_int, c_void};

use crate::ffi::{LsSpanSlice, LsTensorSliceReq};
use crate::io_reader::LayerIoPool;

pub(super) unsafe extern "C" fn rust_load_tensors_cb(
    user_data: *mut c_void,
    layer_idx: i32,
    tensor_names: *const *const c_char,
    tensor_offsets: *const u64,
    tensor_sizes: *const u64,
    tensor_count: usize,
    dst_ptr: *mut u8,
    dst_cap: usize,
) -> c_int {
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user_data.is_null() || dst_ptr.is_null() {
            return -1;
        }
        // SAFETY: `user_data` is the boxed `LayerIoPool` owned by `StreamEngine` and outlives all C++ engine callbacks.
        let pool = unsafe { &*(user_data as *const LayerIoPool) };
        // SAFETY: C++ caller guarantees the destination pointer is valid and exclusively writable for the given capacity.
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, dst_cap) };

        // Fast path: reuse cached 4K-aligned CoalescedSpan plan on sweeps 2..N.
        match pool.load_cached_layer_into(layer_idx, dst) {
            Ok(true) => return 0,
            Ok(false) => {}
            Err(err) => {
                tracing::error!("layerstream cached IO callback failed: {err:#}");
                return -5;
            }
        }

        let mut requests = Vec::with_capacity(tensor_count);
        for i in 0..tensor_count {
            // SAFETY: `i < n_tensors`, and C++ caller guarantees the parallel metadata arrays have at least `n_tensors` elements.
            let c_name = unsafe { *tensor_names.add(i) };
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let Ok(name) = unsafe { CStr::from_ptr(c_name) }.to_str() else {
                return -2;
            };
            // SAFETY: `i < n_tensors`, and C++ caller guarantees the parallel metadata arrays have at least `n_tensors` elements.
            let off = unsafe { *tensor_offsets.add(i) } as usize;
            // SAFETY: `i < n_tensors`, and C++ caller guarantees the parallel metadata arrays have at least `n_tensors` elements.
            let sz = unsafe { *tensor_sizes.add(i) } as usize;
            if off.saturating_add(sz) > dst_cap {
                return -3;
            }
            requests.push((name, off, sz));
        }

        match pool.load_layer_tensors_into(layer_idx, &requests, dst) {
            Ok(()) => 0,
            Err(err) => {
                tracing::error!("layerstream IO callback failed: {err:#}");
                -4
            }
        }
    }));
    res.unwrap_or(-99)
}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe extern "C" fn rust_load_layer_direct_cb(
    user_data: *mut c_void,
    layer_idx: i32,
    tensor_names: *const *const c_char,
    tensor_offsets: *const u64,
    tensor_sizes: *const u64,
    tensor_count: usize,
    pin_ptr: *mut u8,
    pin_cap: usize,
    out_slices_ptr: *mut LsSpanSlice,
    max_slices: usize,
) -> c_int {
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user_data.is_null() || pin_ptr.is_null() || out_slices_ptr.is_null() {
            return -1;
        }
        // SAFETY: `user_data` is the boxed `LayerIoPool` owned by `StreamEngine` and outlives all C++ engine callbacks.
        let pool = unsafe { &*(user_data as *const LayerIoPool) };
        // SAFETY: C++ caller guarantees the destination pointer is valid and exclusively writable for the given capacity.
        let pin_buf = unsafe { std::slice::from_raw_parts_mut(pin_ptr, pin_cap) };
        // SAFETY: C++ caller guarantees the destination pointer is valid and exclusively writable for the given capacity.
        let out_slices = unsafe { std::slice::from_raw_parts_mut(out_slices_ptr, max_slices) };

        // Fast path: cached span plan on sweeps 2..N (zero CStr/Vec allocation!).
        match pool.load_layer_direct_into(layer_idx, None, pin_buf, out_slices) {
            Ok(n) if n > 0 => return n as c_int,
            Ok(_) => {}
            Err(err) => {
                tracing::error!("layerstream zero-copy cached IO callback failed: {err:#}");
                return -5;
            }
        }

        let mut requests = Vec::with_capacity(tensor_count);
        for i in 0..tensor_count {
            // SAFETY: `i < n_tensors`, and C++ caller guarantees the parallel metadata arrays have at least `n_tensors` elements.
            let c_name = unsafe { *tensor_names.add(i) };
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let Ok(name) = unsafe { CStr::from_ptr(c_name) }.to_str() else {
                return -2;
            };
            // SAFETY: `i < n_tensors`, and C++ caller guarantees the parallel metadata arrays have at least `n_tensors` elements.
            let off = unsafe { *tensor_offsets.add(i) } as usize;
            // SAFETY: `i < n_tensors`, and C++ caller guarantees the parallel metadata arrays have at least `n_tensors` elements.
            let sz = unsafe { *tensor_sizes.add(i) } as usize;
            requests.push((name, off, sz));
        }

        match pool.load_layer_direct_into(layer_idx, Some(&requests), pin_buf, out_slices) {
            Ok(n) => n as c_int,
            Err(err) => {
                tracing::error!("layerstream zero-copy IO callback failed: {err:#}");
                -4
            }
        }
    }));
    res.unwrap_or(-99)
}

pub(super) unsafe extern "C" fn rust_load_slices_cb(
    user_data: *mut c_void,
    reqs_ptr: *const LsTensorSliceReq,
    req_count: usize,
    dst_ptr: *mut u8,
    dst_cap: usize,
) -> c_int {
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user_data.is_null() || dst_ptr.is_null() || (reqs_ptr.is_null() && req_count > 0) {
            return -1;
        }
        // SAFETY: `user_data` is the boxed `LayerIoPool` owned by `StreamEngine` and outlives all C++ engine callbacks.
        let pool = unsafe { &*(user_data as *const LayerIoPool) };
        // SAFETY: C++ caller guarantees the destination pointer is valid and exclusively writable for the given capacity.
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, dst_cap) };
        let raw_reqs = if req_count == 0 {
            &[]
        } else {
            // SAFETY: C++ caller guarantees `reqs_ptr` points to `req_count` initialized `LsSliceReadReq` entries.
            unsafe { std::slice::from_raw_parts(reqs_ptr, req_count) }
        };

        let mut requests = Vec::with_capacity(req_count);
        for r in raw_reqs {
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let Ok(name) = unsafe { CStr::from_ptr(r.tensor_name) }.to_str() else {
                return -2;
            };
            let dst_off = r.dst_offset as usize;
            let len = r.len as usize;
            if dst_off.saturating_add(len) > dst_cap {
                return -3;
            }
            requests.push((name, r.src_offset_in_tensor, dst_off, len, r.tensor_idx));
        }

        match pool.load_slices_into(&requests, dst) {
            Ok(()) => 0,
            Err(err) => {
                tracing::error!("layerstream MoE slice IO callback failed: {err:#}");
                -4
            }
        }
    }));
    res.unwrap_or(-99)
}

/// Buffers incomplete trailing multi-byte UTF-8 sequences across token boundaries so byte-level
/// BPE tokenizers never emit spurious `U+FFFD` replacement characters mid-codepoint.
#[derive(Default)]
pub(super) struct Utf8PieceAssembler {
    pending: Vec<u8>,
}

impl Utf8PieceAssembler {
    pub(super) fn push(&mut self, bytes: &[u8]) -> String {
        if bytes.is_empty() {
            return String::new();
        }
        if self.pending.is_empty()
            && let Ok(s) = std::str::from_utf8(bytes)
        {
            return s.to_string();
        }
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    if !valid.is_empty() {
                        out.push_str(valid);
                    }
                    self.pending.clear();
                    break;
                }
                Err(err) => {
                    let valid_len = err.valid_up_to();
                    if valid_len > 0 {
                        let valid_str =
                            // SAFETY: `valid_len` is `err.valid_up_to()` from `std::str::from_utf8`, so `self.pending[..valid_len]` is valid UTF-8.
                            unsafe { std::str::from_utf8_unchecked(&self.pending[..valid_len]) };
                        out.push_str(valid_str);
                    }
                    match err.error_len() {
                        Some(bad_len) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid_len + bad_len);
                        }
                        None => {
                            // Incomplete trailing 2..=4 byte UTF-8 sequence; wait for next token.
                            if valid_len > 0 {
                                self.pending.drain(..valid_len);
                            }
                            break;
                        }
                    }
                }
            }
        }
        out
    }

    pub(super) fn flush(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let rem = std::mem::take(&mut self.pending);
        String::from_utf8_lossy(&rem).into_owned()
    }
}

#[cfg(test)]
mod utf8_tests {
    use super::Utf8PieceAssembler;

    #[test]
    fn reassembles_multibyte_utf8_across_token_pieces() {
        let mut asm = Utf8PieceAssembler::default();
        // "🦀" = [0xF0, 0x9F, 0xA6, 0x80], "é" = [0xC3, 0xA9]
        assert_eq!(asm.push(&[b'h', b'i', 0xF0]), "hi");
        assert_eq!(asm.push(&[0x9F]), "");
        assert_eq!(asm.push(&[0xA6]), "");
        assert_eq!(asm.push(&[0x80, 0xC3]), "🦀");
        assert_eq!(asm.push(&[0xA9, b'!']), "é!");
        assert_eq!(asm.flush(), "");
    }

    #[test]
    fn flushes_truncated_utf8_tail_as_replacement_char() {
        let mut asm = Utf8PieceAssembler::default();
        assert_eq!(asm.push(&[0xF0, 0x9F]), "");
        assert_eq!(asm.flush(), "\u{FFFD}");
    }
}
