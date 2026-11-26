//! High-throughput coalesced 4K-aligned O_DIRECT pread reader for GGUF layer streaming.
//!
//! Optimizations:
//! - Caches planned `CoalescedSpan`s per `layer_idx` after sweep 1 (zero CStr/HashMap
//!   overhead in the hot decode loop).
//! - Optionally splits large layer spans (>= 4 MiB) across `io_threads` (`DEFAULT_IO_THREADS = 1`
//!   to avoid Linux dm-crypt contention; `DEFAULT_MOE_IO_THREADS = 4` on macOS, `1` elsewhere;
//!   override via `LAYERSTREAM_IO_THREADS` / `LAYERSTREAM_MOE_IO_THREADS`) using `std::thread::scope`.
//! - Uses `O_DIRECT` on Linux, `F_NOCACHE` on macOS, and `FILE_FLAG_NO_BUFFERING` on
//!   Windows to avoid polluting the host OS page cache.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::{GgufIoError, Result};
use parking_lot::{Condvar, Mutex};
use quick_cache::{Weighter, unsync::Cache as QuickCache};

use crate::gguf_index::{CoalescedSpan, DIRECT_IO_ALIGN, GgufIndex, ResolvedTensorLoc, align_up};

/// C-ABI-compatible slice descriptor returned by zero-copy `load_layer_direct_into`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DirectSpanSlice {
    pub src_offset: u64,
    pub dst_offset: u64,
    pub len: u64,
    pub tensor_idx: u32,
}

const PARALLEL_READ_MIN_BYTES: usize = 4 * 1024 * 1024;
/// Default staging chunk size for fallback O_DIRECT pread (full-layer 256 MiB spans
/// maximize Linux kernel BIO queue depth through dm-crypt / NVMe).
/// Override with `LAYERSTREAM_IO_CHUNK_MB` (e.g. `16`, `64`, `256`).
const DEFAULT_SCRATCH_CHUNK_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_IO_THREADS: usize = 1;
/// Sparse MoE slices are small (~0.5–1 MiB) and scattered across tens of GiB.
/// Parallel workers keep the NVMe submission queue and Linux dm-crypt/kcryptd
/// workers saturated.
fn default_moe_io_threads() -> usize {
    #[cfg(target_os = "macos")]
    {
        4
    }
    #[cfg(target_os = "linux")]
    {
        std::thread::available_parallelism()
            .map(|n| n.get().clamp(2, 6))
            .unwrap_or(4)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        1
    }
}

/// Page-aligned (4096-byte) owned byte buffer suitable for Linux `O_DIRECT` / Windows unbuffered I/O.
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

// SAFETY: `AlignedBuf` exclusively owns its heap allocation and has no thread-local affinity.
unsafe impl Send for AlignedBuf {}
// SAFETY: Shared references (`&AlignedBuf`) do not expose interior mutability.
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    pub fn new(min_bytes: usize) -> Result<Self> {
        let base = min_bytes.max(DIRECT_IO_ALIGN as usize) as u64;
        // coverage: 64-bit alignment overflow and OOM `alloc_zeroed() == null` are unreachable in normal unit tests.
        let rounded = base
            .checked_add(DIRECT_IO_ALIGN - 1)
            .map(|v| v & !(DIRECT_IO_ALIGN - 1))
            .ok_or_else(|| {
                GgufIoError::AllocationFailed("AlignedBuf size overflow during 4K alignment".into())
            })?;
        let len = usize::try_from(rounded)
            .map_err(|_| GgufIoError::AllocationFailed("AlignedBuf size exceeds usize".into()))?;
        let layout = Layout::from_size_align(len, DIRECT_IO_ALIGN as usize).map_err(|e| {
            GgufIoError::AllocationFailed(format!("invalid AlignedBuf layout: {e}"))
        })?;
        // SAFETY: `layout` has non-zero size (`len >= DIRECT_IO_ALIGN`) and power-of-two alignment.
        let raw = unsafe { alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(raw) else {
            return Err(GgufIoError::AllocationFailed(format!(
                "failed to allocate {len}-byte 4K-aligned IO buffer"
            )));
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
        // SAFETY: `self.ptr` points to `self.len` initialized zeroed bytes owned exclusively by `&mut self`, and `len <= self.len`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was allocated via `alloc_zeroed` with `self.layout` and is deallocated once on drop.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

struct ShardHandle {
    direct: Option<File>,
    buffered: File,
    #[cfg(target_os = "linux")]
    sparse_buffered: File,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct IoStats {
    pub bytes_read: u64,
    pub span_reads: u64,
    pub pread_us: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_bytes_served: u64,
    pub direct_io_active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SliceKey {
    shard_idx: usize,
    file_offset: u64,
    len: usize,
}

#[derive(Clone, Copy)]
struct SliceByteWeighter;

impl Weighter<SliceKey, Vec<u8>> for SliceByteWeighter {
    fn weight(&self, _key: &SliceKey, val: &Vec<u8>) -> u64 {
        val.len() as u64
    }
}

/// Bounded scan-resistant S3-FIFO cache of raw immutable GGUF expert slices in host RAM.
///
/// Sparse MoE routing is heavily Zipfian, and on large MoE models (e.g., 30B-A3B) a single
/// token sweep can exceed the RAM cache budget. `quick_cache`'s S3-FIFO policy resists
/// cyclic scan thrashing while enforcing an exact byte-weight budget.
struct ExpertSliceCache {
    budget_bytes: Option<usize>,
    cache: Option<QuickCache<SliceKey, Vec<u8>, SliceByteWeighter>>,
}

impl ExpertSliceCache {
    fn new(explicit_budget_bytes: Option<usize>) -> Self {
        Self {
            budget_bytes: explicit_budget_bytes,
            cache: None,
        }
    }

    fn ensure_budget(&mut self) -> usize {
        if let Some(b) = self.budget_bytes {
            return b;
        }
        let auto_bytes = auto_expert_cache_budget_bytes();
        self.budget_bytes = Some(auto_bytes);
        if auto_bytes > 0 {
            tracing::info!(
                expert_cache_mib = auto_bytes / (1024 * 1024),
                "enabled bounded scan-resistant MoE hot-expert RAM cache"
            );
        }
        auto_bytes
    }

    fn ensure_cache(&mut self) -> Option<&mut QuickCache<SliceKey, Vec<u8>, SliceByteWeighter>> {
        let budget = self.ensure_budget();
        if budget == 0 {
            return None;
        }
        Some(self.cache.get_or_insert_with(|| {
            // Estimate ~256 KiB per expert slice (min 16 slots for small unit-test budgets).
            let est_items = (budget / (256 * 1024)).max(16);
            QuickCache::with_weighter(est_items, budget as u64, SliceByteWeighter)
        }))
    }

    fn get_and_touch(&mut self, key: SliceKey) -> Option<&[u8]> {
        self.cache.as_mut()?.get(&key).map(Vec::as_slice)
    }

    fn insert(&mut self, key: SliceKey, bytes: &[u8]) {
        let budget = self.ensure_budget();
        if budget == 0 || bytes.is_empty() || bytes.len() > budget / 2 {
            return;
        }
        if let Some(cache) = self.ensure_cache() {
            cache.insert(key, bytes.to_vec());
        }
    }

    #[cfg(test)]
    fn used_bytes(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.weight() as usize)
    }

    fn clear(&mut self) {
        if let Some(cache) = self.cache.as_mut() {
            cache.clear();
        }
    }

    fn export_hot_entries(&self) -> Vec<crate::expert_profile::HotSliceEntry> {
        let Some(cache) = self.cache.as_ref() else {
            return Vec::new();
        };
        cache
            .iter()
            .map(|(k, _)| crate::expert_profile::HotSliceEntry {
                shard_idx: k.shard_idx as u32,
                file_offset: k.file_offset,
                len: k.len as u64,
            })
            .collect()
    }
}

fn auto_expert_cache_budget_bytes() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if let Some(rest) = line.strip_prefix("MemAvailable:")
                    && let Some(val) = rest.split_whitespace().next()
                    && let Ok(kb) = val.parse::<u64>()
                {
                    let avail = kb.saturating_mul(1024);
                    // Keep >= 1.25 GiB headroom for OS, CUDA/Vulkan scratch, and services.
                    let reserve = 1280u64 * 1024 * 1024;
                    let cap = 2048u64 * 1024 * 1024;
                    let target = (avail.saturating_sub(reserve) * 3) / 5;
                    return target.min(cap) as usize;
                }
            }
        }
        0
    }
    #[cfg(not(target_os = "linux"))]
    {
        256 * 1024 * 1024
    }
}

/// Persistent scoped worker pool that eliminates per-layer OS thread spawn/join overhead.
///
/// The calling thread executes task `0`; `n_tasks - 1` parked background threads execute
/// tasks `1..n_tasks`. `run()` blocks until every background task finishes (even on panic)
/// before returning, so stack borrows remain valid for the entire dispatch.
struct ScopedWorkerPool {
    dispatch_lock: Mutex<()>,
    shared: Arc<PoolShared>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

struct PoolShared {
    state: Mutex<PoolState>,
    work_cv: Condvar,
    done_cv: Condvar,
}

struct PoolState {
    generation: u64,
    active_bg: usize,
    done_bg: usize,
    shutdown: bool,
    task_fn: Option<*const (dyn Fn(usize) + Sync + 'static)>,
}

// SAFETY: `task_fn` points to a `Sync` closure live on the dispatching thread's stack until
// `done_bg == active_bg`; `PoolState` is only accessed under `PoolShared::state`.
unsafe impl Send for PoolState {}

impl ScopedWorkerPool {
    fn new(max_threads: usize) -> Self {
        let bg_count = max_threads.saturating_sub(1);
        let shared = Arc::new(PoolShared {
            state: Mutex::new(PoolState {
                generation: 0,
                active_bg: 0,
                done_bg: 0,
                shutdown: false,
                task_fn: None,
            }),
            work_cv: Condvar::new(),
            done_cv: Condvar::new(),
        });
        let mut handles = Vec::with_capacity(bg_count);
        for bg_idx in 0..bg_count {
            let worker_id = bg_idx + 1;
            let sh = Arc::clone(&shared);
            let h = std::thread::Builder::new()
                .name(format!("ls-io-{worker_id}"))
                .spawn(move || {
                    let mut seen_gen = 0u64;
                    loop {
                        let task_ptr = {
                            let mut st = sh.state.lock();
                            while !st.shutdown && st.generation == seen_gen {
                                sh.work_cv.wait(&mut st);
                            }
                            if st.shutdown {
                                break;
                            }
                            seen_gen = st.generation;
                            if worker_id <= st.active_bg {
                                st.task_fn
                            } else {
                                None
                            }
                        };
                        if let Some(ptr) = task_ptr {
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                // SAFETY: `run()` holds `dispatch_lock` and waits until `done_bg == active_bg`
                                // before clearing `task_fn` and returning, so `ptr` is valid and `Sync`.
                                let f: &(dyn Fn(usize) + Sync) = unsafe { &*ptr };
                                f(worker_id);
                            }));
                            let mut st = sh.state.lock();
                            st.done_bg += 1;
                            if st.done_bg == st.active_bg {
                                sh.done_cv.notify_one();
                            }
                        }
                    }
                });
            if let Ok(handle) = h {
                handles.push(handle);
            }
        }
        Self {
            dispatch_lock: Mutex::new(()),
            shared,
            handles,
        }
    }

    fn run<F: Fn(usize) + Sync>(&self, n_tasks: usize, f: F) {
        if n_tasks == 0 {
            return;
        }
        let max_tasks = self.handles.len() + 1;
        let actual_tasks = n_tasks.min(max_tasks);
        if actual_tasks <= 1 {
            for idx in 0..n_tasks {
                f(idx);
            }
            return;
        }

        let _dispatch_guard = self.dispatch_lock.lock();
        let active_bg = actual_tasks - 1;
        let trait_obj: &(dyn Fn(usize) + Sync) = &f;
        // SAFETY: Erasing the lifetime to `'static` is sound because `WaitGuard` blocks until
        // all `active_bg` background workers finish and clears `task_fn` before `f` is dropped.
        let erased: *const (dyn Fn(usize) + Sync + 'static) =
            unsafe { std::mem::transmute::<*const (dyn Fn(usize) + Sync), _>(trait_obj) };

        {
            let mut st = self.shared.state.lock();
            st.active_bg = active_bg;
            st.done_bg = 0;
            st.task_fn = Some(erased);
            st.generation = st.generation.wrapping_add(1);
            self.shared.work_cv.notify_all();
        }

        struct WaitGuard<'a>(&'a PoolShared);
        impl Drop for WaitGuard<'_> {
            fn drop(&mut self) {
                let mut st = self.0.state.lock();
                while st.done_bg < st.active_bg {
                    self.0.done_cv.wait(&mut st);
                }
                st.task_fn = None;
                st.active_bg = 0;
            }
        }
        let _wait = WaitGuard(&self.shared);

        // Calling thread executes task 0 (and any overflow tasks if n_tasks > max_tasks).
        f(0);
        for extra in actual_tasks..n_tasks {
            f(extra);
        }
    }
}

impl Drop for ScopedWorkerPool {
    fn drop(&mut self) {
        {
            let mut st = self.shared.state.lock();
            st.shutdown = true;
            self.shared.work_cv.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

pub struct LayerIoPool {
    index: GgufIndex,
    shards: Vec<ShardHandle>,
    direct_io_active: bool,
    sparse_direct_io: bool,
    io_threads: usize,
    moe_io_threads: usize,
    max_chunk_bytes: usize,
    profile_enabled: bool,
    workers: ScopedWorkerPool,
    #[cfg(target_os = "linux")]
    uring: Option<Mutex<crate::uring::UringReader>>,
    scratch: Mutex<Option<AlignedBuf>>,
    moe_scratches: Mutex<Vec<AlignedBuf>>,
    span_cache: Mutex<HashMap<i32, Arc<[CoalescedSpan]>>>,
    expert_cache: Mutex<ExpertSliceCache>,
    bytes_read: AtomicU64,
    span_reads: AtomicU64,
    direct_zero_copy_loads: AtomicU64,
    pread_us: AtomicU64,
    memcpy_us: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    cache_bytes_served: AtomicU64,
    slow_storage_warned: AtomicBool,
}

impl LayerIoPool {
    pub fn new(index: GgufIndex, prefer_direct_io: bool) -> Result<Self> {
        let mut shards = Vec::with_capacity(index.shards.len());
        let mut all_direct = prefer_direct_io;

        for path in &index.shards {
            let buffered = File::open(path).map_err(|e| {
                GgufIoError::io(format!("failed to open shard {}", path.display()), e)
            })?;
            #[cfg(target_os = "linux")]
            let sparse_buffered = open_sparse_buffered(path).map_err(|e| {
                GgufIoError::io(format!("failed to open sparse shard {}", path.display()), e)
            })?;
            let direct = if prefer_direct_io {
                open_direct(path)
            } else {
                None
            };
            if direct.is_none() {
                all_direct = false;
            }
            shards.push(ShardHandle {
                direct,
                buffered,
                #[cfg(target_os = "linux")]
                sparse_buffered,
            });
        }

        if prefer_direct_io && !all_direct {
            tracing::warn!(
                "O_DIRECT unavailable for one or more GGUF shards (e.g. tmpfs/FUSE/network mount); \
                 falling back to buffered I/O (OS page cache may grow)"
            );
        }

        let io_threads = std::env::var("LAYERSTREAM_IO_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|n| n.clamp(1, 8))
            .unwrap_or(DEFAULT_IO_THREADS);

        let moe_io_threads = std::env::var("LAYERSTREAM_MOE_IO_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|n| n.clamp(1, 8))
            .unwrap_or_else(default_moe_io_threads);

        // On Linux, scattered sub-MiB O_DIRECT reads can trigger NVMe MSI-X IRQ
        // timeouts on embedded PCIe controllers (e.g. Tegra Orin + Crucial T500)
        // while buffered `POSIX_FADV_RANDOM` + immediate `POSIX_FADV_DONTNEED`
        // sustains ~1,150+ MiB/s with < 10 ms tail latency and ~0 MiB page-cache
        // growth. Keep full-layer sequential reads on O_DIRECT and allow forcing
        // O_DIRECT on sparse MoE reads via `LAYERSTREAM_SPARSE_DIRECT_IO=1`.
        #[cfg(target_os = "linux")]
        let default_sparse_direct = false;
        #[cfg(not(target_os = "linux"))]
        let default_sparse_direct = true;
        let sparse_direct_io = std::env::var("LAYERSTREAM_SPARSE_DIRECT_IO")
            .ok()
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(default_sparse_direct);

        let max_chunk_bytes = std::env::var("LAYERSTREAM_IO_CHUNK_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| (mb.clamp(4, 512)) * 1024 * 1024)
            .unwrap_or(DEFAULT_SCRATCH_CHUNK_BYTES);

        let profile_enabled = std::env::var("LAYERSTREAM_PROFILE")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);

        let explicit_expert_cache = std::env::var("LAYERSTREAM_EXPERT_CACHE_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb.saturating_mul(1024 * 1024));

        let workers = ScopedWorkerPool::new(io_threads.max(moe_io_threads));
        #[cfg(target_os = "linux")]
        let uring = crate::uring::UringReader::try_new().map(Mutex::new);

        Ok(Self {
            index,
            shards,
            direct_io_active: all_direct,
            sparse_direct_io,
            io_threads,
            moe_io_threads,
            max_chunk_bytes,
            profile_enabled,
            workers,
            #[cfg(target_os = "linux")]
            uring,
            scratch: Mutex::new(None),
            moe_scratches: Mutex::new(Vec::new()),
            span_cache: Mutex::new(HashMap::new()),
            expert_cache: Mutex::new(ExpertSliceCache::new(explicit_expert_cache)),
            bytes_read: AtomicU64::new(0),
            span_reads: AtomicU64::new(0),
            direct_zero_copy_loads: AtomicU64::new(0),
            pread_us: AtomicU64::new(0),
            memcpy_us: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            cache_bytes_served: AtomicU64::new(0),
            slow_storage_warned: AtomicBool::new(false),
        })
    }

    fn record_io_sample(&self, bytes: u64, elapsed_us: u64) {
        let total_bytes = self.bytes_read.fetch_add(bytes, Ordering::Relaxed) + bytes;
        let total_us = self.pread_us.fetch_add(elapsed_us, Ordering::Relaxed) + elapsed_us;
        // One-time slow-storage check after the first 128 MiB read: warns if running on
        // slow media (e.g., microSD, HDD, USB 2.0, or network share < 150 MiB/s).
        if total_bytes >= 128 * 1024 * 1024
            && total_us > 0
            && !self.slow_storage_warned.load(Ordering::Relaxed)
            && !self.slow_storage_warned.swap(true, Ordering::Relaxed)
        {
            let mib_s = (total_bytes as f64 / (1024.0 * 1024.0)) / (total_us as f64 / 1_000_000.0);
            if mib_s < 150.0 {
                tracing::warn!(
                    throughput_mib_s = format_args!("{mib_s:.0}"),
                    "slow storage detected ({mib_s:.0} MiB/s); layer streaming is designed for fast NVMe SSDs (>= 1,000 MiB/s) — move GGUF files to NVMe or pass --vram-budget-mb auto to pin hot layers"
                );
            }
        }
    }

    pub fn index(&self) -> &GgufIndex {
        &self.index
    }

    pub fn clear_expert_cache(&self) {
        self.expert_cache.lock().clear();
    }

    /// Saves the currently resident S3-FIFO hot-expert slice coordinates to `path`
    /// so a future session can pre-warm the cache sequentially from NVMe.
    pub fn save_expert_profile(&self, path: &std::path::Path) -> std::io::Result<usize> {
        let entries = self.expert_cache.lock().export_hot_entries();
        crate::expert_profile::write_expert_profile(path, &entries)
    }

    /// Pre-warms the S3-FIFO expert cache from a saved hot-expert profile using
    /// sequential reads sorted by `(shard_idx, file_offset)`. Returns the number
    /// of slices loaded into RAM.
    pub fn warm_expert_profile(&self, path: &std::path::Path) -> std::io::Result<usize> {
        let entries = crate::expert_profile::read_expert_profile(path)?;
        let mut loaded = 0usize;
        let mut scratch = Vec::new();
        for entry in entries {
            let shard_idx = entry.shard_idx as usize;
            let Ok(len) = usize::try_from(entry.len) else {
                continue;
            };
            if len == 0 || shard_idx >= self.shards.len() {
                continue;
            }
            if scratch.len() < len {
                scratch.resize(len, 0);
            }
            let file = &self.shards[shard_idx].buffered;
            if pread_all(file, &mut scratch[..len], entry.file_offset).is_ok() {
                self.expert_cache.lock().insert(
                    SliceKey {
                        shard_idx,
                        file_offset: entry.file_offset,
                        len,
                    },
                    &scratch[..len],
                );
                loaded += 1;
            }
        }
        Ok(loaded)
    }

    pub fn stats(&self) -> IoStats {
        IoStats {
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            span_reads: self.span_reads.load(Ordering::Relaxed),
            pread_us: self.pread_us.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            cache_bytes_served: self.cache_bytes_served.load(Ordering::Relaxed),
            direct_io_active: self.direct_io_active,
        }
    }

    /// Zero-copy fast path for discrete GPUs: performs a single 4K-aligned O_DIRECT `pread`
    /// straight into the caller's pinned staging buffer (`pin_buf`), eliminating the
    /// intermediate `scratch` allocation and CPU `memcpy`.
    ///
    /// Returns `Ok(n_slices)` (> 0) on success, or `Ok(0)` when the layer must fall back
    /// to `load_layer_tensors_into` (e.g. spans multiple shards or exceeds `pin_buf`).
    pub fn load_layer_direct_into(
        &self,
        layer_idx: i32,
        requests: Option<&[(&str, usize, usize)]>,
        pin_buf: &mut [u8],
        out_slices: &mut [DirectSpanSlice],
    ) -> Result<usize> {
        if layer_idx < 0 || !(pin_buf.as_ptr() as usize).is_multiple_of(DIRECT_IO_ALIGN as usize) {
            return Ok(0);
        }

        let spans = {
            let cached = self.span_cache.lock().get(&layer_idx).cloned();
            if let Some(s) = cached {
                s
            } else {
                let Some(reqs) = requests else {
                    return Ok(0);
                };
                let planned: Arc<[CoalescedSpan]> =
                    Arc::from(self.index.plan_coalesced_spans(reqs)?);
                self.span_cache
                    .lock()
                    .insert(layer_idx, Arc::clone(&planned));
                planned
            }
        };

        let total_aligned: usize = spans.iter().map(|s| s.aligned_len).sum();
        let total_slices: usize = spans.iter().map(|s| s.slices.len()).sum();
        if spans.is_empty() || total_aligned > pin_buf.len() || total_slices > out_slices.len() {
            return Ok(0);
        }

        let t_pread = std::time::Instant::now();
        let mut buf_off = 0usize;
        let mut slice_idx = 0usize;
        let mut span_bytes_total = 0u64;
        for span in spans.iter() {
            let handle = &self.shards[span.shard_idx];
            let read_buf = &mut pin_buf[buf_off..buf_off + span.aligned_len];

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
                pread_all(
                    &handle.buffered,
                    &mut read_buf[..span.valid_len],
                    span.aligned_offset,
                )?;
            }

            span_bytes_total += span.valid_len as u64;
            self.span_reads.fetch_add(1, Ordering::Relaxed);

            for s in &span.slices {
                out_slices[slice_idx] = DirectSpanSlice {
                    src_offset: (buf_off + s.span_offset) as u64,
                    dst_offset: s.dst_offset as u64,
                    len: s.len as u64,
                    tensor_idx: s.tensor_idx,
                };
                slice_idx += 1;
            }
            buf_off += span.aligned_len;
        }
        self.record_io_sample(span_bytes_total, t_pread.elapsed().as_micros() as u64);
        self.direct_zero_copy_loads.fetch_add(1, Ordering::Relaxed);
        Ok(slice_idx)
    }

    /// Loads a cached layer plan if already computed, avoiding CStr/HashMap work on sweeps 2..N.
    pub fn load_cached_layer_into(&self, layer_idx: i32, dst: &mut [u8]) -> Result<bool> {
        if layer_idx < 0 {
            return Ok(false);
        }
        let cached = self.span_cache.lock().get(&layer_idx).cloned();
        let Some(spans) = cached else {
            return Ok(false);
        };

        let mut scratch_guard = self.scratch.lock();
        let scratch = match scratch_guard.as_mut() {
            Some(buf) => buf,
            None => scratch_guard.insert(AlignedBuf::new(4 * 1024 * 1024)?),
        };
        for span in spans.iter() {
            self.read_span_into(span, scratch, dst)?;
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
        let spans: Arc<[CoalescedSpan]> = Arc::from(self.index.plan_coalesced_spans(requests)?);
        if layer_idx >= 0 {
            self.span_cache.lock().insert(layer_idx, Arc::clone(&spans));
        }

        if layer_idx < 0 {
            // One-shot startup load (e.g. resident layers): use an ephemeral scratch buffer
            // so its memory is returned to the OS immediately after initialization.
            let mut ephemeral = AlignedBuf::new(4 * 1024 * 1024)?;
            for span in spans.iter() {
                self.read_span_into(span, &mut ephemeral, dst)?;
            }
            return Ok(());
        }

        let mut scratch_guard = self.scratch.lock();
        let scratch = match scratch_guard.as_mut() {
            Some(buf) => buf,
            None => scratch_guard.insert(AlignedBuf::new(4 * 1024 * 1024)?),
        };
        for span in spans.iter() {
            self.read_span_into(span, scratch, dst)?;
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

    /// Reads arbitrary sub-tensor slices (e.g., active MoE experts) into `dst`.
    /// Serves hot experts directly from the bounded RAM LRU cache when present, and reads
    /// remaining misses via coalesced 4K-aligned `O_DIRECT` spans before inserting them.
    pub fn load_slices_into(
        &self,
        requests: &[(&str, u64, usize, usize, u32)],
        dst: &mut [u8],
    ) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }

        // Validate all destination ranges in the full request are in-bounds and
        // mutually disjoint before cache hits or reads write to `dst`.
        let mut req_ranges: Vec<(usize, usize)> = Vec::with_capacity(requests.len());
        for &(_, _, dst_off, len, _) in requests {
            let end = dst_off
                .checked_add(len)
                .ok_or_else(|| GgufIoError::OutOfBounds("slice dst_offset overflow".into()))?;
            if end > dst.len() {
                return Err(GgufIoError::OutOfBounds(format!(
                    "slice dst range [{dst_off}..{end}) exceeds dst_cap {}",
                    dst.len()
                )));
            }
            req_ranges.push((dst_off, end));
        }
        req_ranges.sort_unstable_by_key(|r| r.0);
        for w in req_ranges.windows(2) {
            if w[0].1 > w[1].0 {
                return Err(GgufIoError::OutOfBounds(format!(
                    "overlapping slice destination ranges [{}..{}) and [{}..{})",
                    w[0].0, w[0].1, w[1].0, w[1].1
                )));
            }
        }

        // Resolve each request's tensor location once so expert cache lookup and span planning
        // share the same `ResolvedTensorLoc` without double-hashing tensor names.
        let mut resolved_reqs: Vec<(ResolvedTensorLoc, u64, usize, usize, u32)> =
            Vec::with_capacity(requests.len());
        for &(name, src_off, dst_off, len, tidx) in requests {
            let loc = self
                .index
                .tensors
                .get(name)
                .ok_or_else(|| GgufIoError::MissingTensor(name.to_string()))?
                .resolved();
            resolved_reqs.push((loc, src_off, dst_off, len, tidx));
        }

        // 1. Check bounded hot-expert RAM cache first.
        let mut misses: Vec<(ResolvedTensorLoc, u64, usize, usize, u32)> = Vec::new();
        let mut miss_meta: Vec<(SliceKey, usize, usize)> = Vec::new();
        let cache_enabled = {
            let mut cache = self.expert_cache.lock();
            let budget = cache.ensure_budget();
            if budget > 0 {
                let mut hits = 0u64;
                let mut hit_bytes = 0u64;
                misses.reserve(resolved_reqs.len());
                miss_meta.reserve(resolved_reqs.len());
                for &(loc, src_off, dst_off, len, tidx) in &resolved_reqs {
                    let end = dst_off + len;
                    let key = SliceKey {
                        shard_idx: loc.shard_idx,
                        file_offset: loc.abs_offset.saturating_add(src_off),
                        len,
                    };
                    if let Some(cached) = cache.get_and_touch(key) {
                        dst[dst_off..end].copy_from_slice(cached);
                        hits += 1;
                        hit_bytes += len as u64;
                        continue;
                    }
                    miss_meta.push((key, dst_off, len));
                    misses.push((loc, src_off, dst_off, len, tidx));
                }
                if hits > 0 {
                    self.cache_hits.fetch_add(hits, Ordering::Relaxed);
                    self.cache_bytes_served
                        .fetch_add(hit_bytes, Ordering::Relaxed);
                }
                if !misses.is_empty() {
                    self.cache_misses
                        .fetch_add(misses.len() as u64, Ordering::Relaxed);
                }
                true
            } else {
                false
            }
        };

        let active_reqs: &[(ResolvedTensorLoc, u64, usize, usize, u32)] = if cache_enabled {
            if misses.is_empty() {
                return Ok(());
            }
            &misses
        } else {
            &resolved_reqs
        };

        let spans = self.index.plan_coalesced_sub_slices_resolved(active_reqs)?;
        #[cfg(target_os = "linux")]
        let mut handled_by_uring = false;
        #[cfg(target_os = "linux")]
        if !self.sparse_direct_io
            && spans.len() > 1
            && spans.iter().all(|s| s.slices.len() == 1)
            && let Some(uring_mu) = &self.uring
        {
            use std::os::unix::io::AsRawFd;
            let mut ops = Vec::with_capacity(spans.len());
            for span in &spans {
                let slice = &span.slices[0];
                let fd = self.shards[span.shard_idx].sparse_buffered.as_raw_fd();
                let offset = span.aligned_offset + slice.span_offset as u64;
                ops.push(crate::uring::UringReadOp {
                    fd,
                    // SAFETY: All slice destination ranges were validated in-bounds and disjoint above.
                    ptr: unsafe { dst.as_mut_ptr().add(slice.dst_offset) },
                    len: slice.len,
                    offset,
                    min_valid: slice.len,
                });
            }
            let t_wall = std::time::Instant::now();
            let read_res = uring_mu.lock().read_batch(&ops);
            if let Ok(bytes) = read_res {
                for span in &spans {
                    let handle = &self.shards[span.shard_idx];
                    evict_page_cache_range(
                        &handle.sparse_buffered,
                        span.aligned_offset,
                        span.valid_len,
                    );
                }
                self.span_reads
                    .fetch_add(spans.len() as u64, Ordering::Relaxed);
                self.record_io_sample(bytes, t_wall.elapsed().as_micros() as u64);
                handled_by_uring = true;
            }
        }
        #[cfg(not(target_os = "linux"))]
        let handled_by_uring = false;

        let n_workers = self.moe_io_threads.min(spans.len()).max(1);
        if handled_by_uring {
            // Completed in a single batched io_uring submission above.
        } else if n_workers <= 1 {
            let mut scratch_guard = self.scratch.lock();
            let scratch = match scratch_guard.as_mut() {
                Some(buf) => buf,
                None => scratch_guard.insert(AlignedBuf::new(4 * 1024 * 1024)?),
            };
            let t_wall = std::time::Instant::now();
            let mut batch_bytes = 0u64;
            for span in &spans {
                batch_bytes = batch_bytes.saturating_add(self.read_sparse_span_into_disjoint_ptr(
                    span,
                    scratch,
                    dst.as_mut_ptr(),
                    dst.len(),
                )?);
            }
            self.record_io_sample(batch_bytes, t_wall.elapsed().as_micros() as u64);
        } else {
            // Destination ranges for all requests were validated in-bounds and mutually disjoint
            // above, and `active_reqs` is a subset of `requests`.
            #[derive(Clone, Copy)]
            struct SendDstPtr(*mut u8, usize);
            impl SendDstPtr {
                #[inline]
                fn parts(self) -> (*mut u8, usize) {
                    (self.0, self.1)
                }
            }
            // SAFETY: Each parallel worker thread writes strictly to disjoint byte ranges within `dst`.
            unsafe impl Send for SendDstPtr {}
            // SAFETY: Disjoint destination ranges are verified during request validation above.
            unsafe impl Sync for SendDstPtr {}

            #[derive(Clone, Copy)]
            struct SendScratchPtr(*mut AlignedBuf);
            impl SendScratchPtr {
                #[inline]
                fn as_ptr(self) -> *mut AlignedBuf {
                    self.0
                }
            }
            // SAFETY: Worker `w` exclusively accesses `scratches[w]` (disjoint elements).
            unsafe impl Send for SendScratchPtr {}
            // SAFETY: Worker `w` exclusively accesses `scratches[w]` (disjoint elements).
            unsafe impl Sync for SendScratchPtr {}

            let mut scratches = self.moe_scratches.lock();
            while scratches.len() < n_workers {
                scratches.push(AlignedBuf::new(2 * 1024 * 1024)?);
            }

            let dst_send = SendDstPtr(dst.as_mut_ptr(), dst.len());
            let scratches_base = SendScratchPtr(scratches.as_mut_ptr());
            let chunk_len = spans.len().div_ceil(n_workers);
            let actual_workers = spans.len().div_ceil(chunk_len);
            let first_err: Mutex<Option<GgufIoError>> = Mutex::new(None);
            let batch_bytes = AtomicU64::new(0);
            let t_wall = std::time::Instant::now();
            self.workers.run(actual_workers, |w| {
                let start = w * chunk_len;
                let end = (start + chunk_len).min(spans.len());
                if start >= end {
                    return;
                }
                let (base, len) = dst_send.parts();
                // SAFETY: `w < actual_workers <= n_workers <= scratches.len()`, and each worker `w`
                // has unique exclusive access to `scratches[w]` while `scratches` lock is held.
                let scratch_buf = unsafe { &mut *scratches_base.as_ptr().add(w) };
                let mut local_bytes = 0u64;
                for span in &spans[start..end] {
                    match self.read_sparse_span_into_disjoint_ptr(span, scratch_buf, base, len) {
                        Ok(n) => local_bytes = local_bytes.saturating_add(n),
                        Err(e) => {
                            let mut g = first_err.lock();
                            if g.is_none() {
                                *g = Some(e);
                            }
                            return;
                        }
                    }
                }
                batch_bytes.fetch_add(local_bytes, Ordering::Relaxed);
            });
            self.record_io_sample(
                batch_bytes.load(Ordering::Relaxed),
                t_wall.elapsed().as_micros() as u64,
            );

            if let Some(e) = first_err.into_inner() {
                return Err(e);
            }
        }

        // 2. Populate the hot-expert cache with newly read misses.
        if cache_enabled && !miss_meta.is_empty() {
            let mut cache = self.expert_cache.lock();
            for (key, dst_off, len) in miss_meta {
                cache.insert(key, &dst[dst_off..dst_off + len]);
            }
        }
        Ok(())
    }

    /// Internal helper for sparse MoE disjoint slice reads (caller must verify
    /// disjointness and bounds). Returns the number of bytes read from storage.
    fn read_sparse_span_into_disjoint_ptr(
        &self,
        span: &CoalescedSpan,
        scratch: &mut AlignedBuf,
        dst_base: *mut u8,
        dst_len: usize,
    ) -> Result<u64> {
        let handle = &self.shards[span.shard_idx];

        #[cfg(target_os = "linux")]
        if !self.sparse_direct_io {
            // Single-slice fast path: pread straight into `dst` (zero scratch copy)
            // using `POSIX_FADV_RANDOM` + immediate `POSIX_FADV_DONTNEED`.
            if let [slice] = span.slices.as_slice() {
                let dst_end = slice.dst_offset.checked_add(slice.len).ok_or_else(|| {
                    GgufIoError::OutOfBounds("sparse slice dst_offset overflow".into())
                })?;
                if dst_end > dst_len {
                    return Err(GgufIoError::OutOfBounds(format!(
                        "sparse slice dst range [{}..{dst_end}) exceeds dst_cap {dst_len}",
                        slice.dst_offset
                    )));
                }
                let file_off = span.aligned_offset + slice.span_offset as u64;
                // SAFETY: `dst_end <= dst_len` was checked above and each span writes disjoint destination ranges in `dst_base`.
                let dst_slice = unsafe {
                    std::slice::from_raw_parts_mut(dst_base.add(slice.dst_offset), slice.len)
                };
                pread_all(&handle.sparse_buffered, dst_slice, file_off)?;
                evict_page_cache_range(
                    &handle.sparse_buffered,
                    span.aligned_offset,
                    span.valid_len,
                );
                self.span_reads.fetch_add(1, Ordering::Relaxed);
                return Ok(slice.len as u64);
            }
        }

        let max_chunk = span.aligned_len.min(self.max_chunk_bytes);
        scratch.ensure_capacity(max_chunk)?;

        let mut total_bytes = 0u64;
        let mut chunk_rel = 0usize;
        while chunk_rel < span.valid_len {
            let chunk_valid = (span.valid_len - chunk_rel).min(self.max_chunk_bytes);
            let chunk_aligned = align_up(chunk_valid as u64, DIRECT_IO_ALIGN) as usize;
            let file_off = span.aligned_offset + chunk_rel as u64;
            let read_buf = scratch.as_mut_slice(chunk_aligned);

            let mut used_direct = false;
            if self.sparse_direct_io
                && let Some(direct_file) = &handle.direct
            {
                used_direct = matches!(pread_exact_or_eof(direct_file, read_buf, file_off), Ok(n) if n >= chunk_valid);
            }
            if !used_direct {
                #[cfg(target_os = "linux")]
                {
                    pread_all(
                        &handle.sparse_buffered,
                        &mut read_buf[..chunk_valid],
                        file_off,
                    )?;
                    evict_page_cache_range(&handle.sparse_buffered, file_off, chunk_valid);
                }
                #[cfg(not(target_os = "linux"))]
                {
                    pread_all(&handle.buffered, &mut read_buf[..chunk_valid], file_off)?;
                }
            }
            total_bytes = total_bytes.saturating_add(chunk_valid as u64);
            self.span_reads.fetch_add(1, Ordering::Relaxed);

            let t_copy = self.profile_enabled.then(std::time::Instant::now);
            let chunk_end = chunk_rel + chunk_valid;
            for slice in &span.slices {
                let slice_end = slice.span_offset + slice.len;
                let ov_start = slice.span_offset.max(chunk_rel);
                let ov_end = slice_end.min(chunk_end);
                if ov_start < ov_end {
                    let len = ov_end - ov_start;
                    let src_off = ov_start - chunk_rel;
                    let dst_off = slice.dst_offset + (ov_start - slice.span_offset);
                    debug_assert!(dst_off + len <= dst_len);
                    // SAFETY: `src_off + len <= read_buf.len()`, `dst_off + len <= dst_len` (validated in span planning), and `read_buf` is disjoint from `dst_base`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            read_buf.as_ptr().add(src_off),
                            dst_base.add(dst_off),
                            len,
                        );
                    }
                }
            }
            if let Some(t0) = t_copy {
                self.memcpy_us
                    .fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            chunk_rel += chunk_valid;
        }
        Ok(total_bytes)
    }

    fn read_span_into(
        &self,
        span: &CoalescedSpan,
        scratch: &mut AlignedBuf,
        dst: &mut [u8],
    ) -> Result<()> {
        let handle = &self.shards[span.shard_idx];
        let max_chunk = span.aligned_len.min(self.max_chunk_bytes);
        scratch.ensure_capacity(max_chunk)?;

        let mut chunk_rel = 0usize;
        while chunk_rel < span.valid_len {
            let chunk_valid = (span.valid_len - chunk_rel).min(self.max_chunk_bytes);
            let chunk_aligned = align_up(chunk_valid as u64, DIRECT_IO_ALIGN) as usize;
            let file_off = span.aligned_offset + chunk_rel as u64;
            let read_buf = scratch.as_mut_slice(chunk_aligned);

            let t_pread = std::time::Instant::now();
            let mut used_direct = false;
            if let Some(direct_file) = &handle.direct {
                used_direct =
                    self.read_parallel_or_single(direct_file, read_buf, file_off, chunk_valid);
            }

            if !used_direct {
                let valid_buf = &mut read_buf[..chunk_valid];
                pread_all(&handle.buffered, valid_buf, file_off)?;
            }
            self.record_io_sample(chunk_valid as u64, t_pread.elapsed().as_micros() as u64);
            self.span_reads.fetch_add(1, Ordering::Relaxed);

            let t_copy = self.profile_enabled.then(std::time::Instant::now);
            let chunk_end = chunk_rel + chunk_valid;
            for slice in &span.slices {
                let slice_end = slice.span_offset + slice.len;
                let ov_start = slice.span_offset.max(chunk_rel);
                let ov_end = slice_end.min(chunk_end);
                if ov_start < ov_end {
                    let len = ov_end - ov_start;
                    let src_off = ov_start - chunk_rel;
                    let dst_off = slice.dst_offset + (ov_start - slice.span_offset);
                    let dst_end = dst_off.checked_add(len).ok_or_else(|| {
                        GgufIoError::OutOfBounds("serial span destination offset overflow".into())
                    })?;
                    if dst_end > dst.len() {
                        return Err(GgufIoError::OutOfBounds(format!(
                            "serial span slice [{dst_off}..{dst_end}) exceeds destination buffer ({} bytes)",
                            dst.len()
                        )));
                    }
                    dst[dst_off..dst_end].copy_from_slice(&read_buf[src_off..src_off + len]);
                }
            }
            if let Some(t0) = t_copy {
                self.memcpy_us
                    .fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
            }

            chunk_rel += chunk_valid;
        }

        Ok(())
    }

    /// Drives NVMe queue depth and parallel dm-crypt decryption by splitting large
    /// 4K-aligned direct-IO spans across batched Linux `io_uring` SQEs (when available)
    /// or `io_threads` concurrent `pread` calls.
    fn read_parallel_or_single(
        &self,
        file: &File,
        buf: &mut [u8],
        base_offset: u64,
        valid_len: usize,
    ) -> bool {
        #[cfg(target_os = "linux")]
        if valid_len >= crate::uring::URING_SPLIT_MIN_BYTES
            && let Some(uring_mu) = &self.uring
        {
            use std::os::unix::io::AsRawFd;
            if uring_mu
                .lock()
                .read_aligned_span(file.as_raw_fd(), buf, base_offset, valid_len)
            {
                return true;
            }
        }
        if self.io_threads <= 1 || valid_len < PARALLEL_READ_MIN_BYTES {
            return matches!(pread_exact_or_eof(file, buf, base_offset), Ok(n) if n >= valid_len);
        }

        let chunk_raw = buf.len().div_ceil(self.io_threads);
        let chunk_size = align_up(chunk_raw as u64, DIRECT_IO_ALIGN) as usize;
        if chunk_size == 0 || chunk_size >= buf.len() {
            return matches!(pread_exact_or_eof(file, buf, base_offset), Ok(n) if n >= valid_len);
        }

        #[derive(Clone, Copy)]
        struct SendBufPtr(*mut u8, usize);
        impl SendBufPtr {
            #[inline]
            fn parts(self) -> (*mut u8, usize) {
                (self.0, self.1)
            }
        }
        // SAFETY: Each worker `idx` writes strictly to its disjoint slice `[idx * chunk_size .. end)`.
        unsafe impl Send for SendBufPtr {}
        // SAFETY: Disjoint chunks are partitioned by `chunk_size` across unique worker indices.
        unsafe impl Sync for SendBufPtr {}

        let buf_ptr = SendBufPtr(buf.as_mut_ptr(), buf.len());
        let n_chunks = buf.len().div_ceil(chunk_size);
        let ok = AtomicBool::new(true);
        self.workers.run(n_chunks, |idx| {
            let (base, total_len) = buf_ptr.parts();
            let rel_off = idx * chunk_size;
            if rel_off >= total_len {
                return;
            }
            let chunk_len = (total_len - rel_off).min(chunk_size);
            let expected = valid_len.saturating_sub(rel_off).min(chunk_len);
            if expected == 0 {
                return;
            }
            let file_off = base_offset + rel_off as u64;
            // SAFETY: `[rel_off .. rel_off + chunk_len)` is within `buf` and disjoint across worker indices.
            let chunk = unsafe { std::slice::from_raw_parts_mut(base.add(rel_off), chunk_len) };
            match pread_exact_or_eof(file, chunk, file_off) {
                Ok(n) if n >= expected => {}
                _ => ok.store(false, Ordering::Relaxed),
            }
        });
        ok.load(Ordering::Relaxed)
    }
}

impl Drop for LayerIoPool {
    fn drop(&mut self) {
        if self.profile_enabled {
            let bytes = self.bytes_read.load(Ordering::Relaxed);
            let reads = self.span_reads.load(Ordering::Relaxed);
            let zc = self.direct_zero_copy_loads.load(Ordering::Relaxed);
            let pread_ms = self.pread_us.load(Ordering::Relaxed) as f64 / 1000.0;
            let copy_ms = self.memcpy_us.load(Ordering::Relaxed) as f64 / 1000.0;
            let hits = self.cache_hits.load(Ordering::Relaxed);
            let misses = self.cache_misses.load(Ordering::Relaxed);
            let cached_mib =
                self.cache_bytes_served.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
            if bytes > 0 || hits > 0 {
                let mib = bytes as f64 / (1024.0 * 1024.0);
                let bw = mib / (pread_ms / 1000.0).max(1e-6);
                let hit_pct = if hits + misses > 0 {
                    100.0 * hits as f64 / (hits + misses) as f64
                } else {
                    0.0
                };
                eprintln!(
                    "[layerstream-io-profile] read={mib:.1}MiB chunks={reads} zc_loads={zc} \
                     chunk_cap={}MiB threads={} moe_threads={} direct={} sparse_direct={} pread={pread_ms:.1}ms ({bw:.0} MiB/s) \
                     memcpy={copy_ms:.1}ms expert_cache={hits}/{} ({hit_pct:.1}% hit, {cached_mib:.1}MiB saved)",
                    self.max_chunk_bytes / (1024 * 1024),
                    self.io_threads,
                    self.moe_io_threads,
                    self.direct_io_active,
                    self.sparse_direct_io,
                    hits + misses
                );
            }
        }
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

#[cfg(target_os = "linux")]
fn open_sparse_buffered(path: &std::path::Path) -> std::io::Result<File> {
    use std::os::unix::io::AsRawFd;
    let file = File::open(path)?;
    // Disable kernel sequential read-ahead on this handle so scattered MoE expert
    // reads don't fetch unused adjacent experts or pollute the OS page cache.
    // SAFETY: `file.as_raw_fd()` is an open file descriptor owned by `file`.
    unsafe {
        let _ = libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM);
    }
    Ok(file)
}

#[cfg(target_os = "linux")]
fn evict_page_cache_range(file: &File, offset: u64, len: usize) {
    use std::os::unix::io::AsRawFd;
    if len == 0 {
        return;
    }
    if let (Ok(off), Ok(n)) = (libc::off_t::try_from(offset), libc::off_t::try_from(len)) {
        // SAFETY: `file.as_raw_fd()` is an open file descriptor and `off`/`n` fit in `off_t`.
        unsafe {
            let _ = libc::posix_fadvise(file.as_raw_fd(), off, n, libc::POSIX_FADV_DONTNEED);
        }
    }
}

#[cfg(target_os = "macos")]
fn open_direct(path: &std::path::Path) -> Option<File> {
    use std::os::unix::io::AsRawFd;
    let file = OpenOptions::new().read(true).open(path).ok()?;
    let fd = file.as_raw_fd();
    // SAFETY: `fd` is a valid open descriptor owned by `file`; `F_NOCACHE` takes a single `c_int`.
    let rc = unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
    // Disable speculative OS read-ahead so scattered MoE expert slices don't waste SSD bandwidth.
    // SAFETY: `fd` is a valid open descriptor owned by `file`; `F_RDAHEAD` takes a single `c_int`.
    unsafe {
        let _ = libc::fcntl(fd, libc::F_RDAHEAD, 0);
    }
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
            Ok(0) => {
                return Err(GgufIoError::io(
                    format!("unexpected EOF during pread at offset {offset}"),
                    std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!("unexpected EOF during pread at offset {offset}"),
                    ),
                ));
            }
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

    let tmp_path = dir.join(format!(".layerstream-probe-{}.tmp", std::process::id()));
    let _cleanup = TempCleanup(tmp_path.clone());
    let mut f = File::create(&tmp_path).ok()?;
    let chunk = vec![0xA5u8; 1024 * 1024];
    use std::io::Write;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_slice_cache_respects_budget_and_clears() {
        let mut cache = ExpertSliceCache::new(Some(12));
        let k1 = SliceKey {
            shard_idx: 0,
            file_offset: 0,
            len: 4,
        };
        let k2 = SliceKey {
            shard_idx: 0,
            file_offset: 4,
            len: 4,
        };
        let k3 = SliceKey {
            shard_idx: 0,
            file_offset: 8,
            len: 4,
        };
        let k4 = SliceKey {
            shard_idx: 0,
            file_offset: 12,
            len: 4,
        };

        cache.insert(k1, b"AAAA");
        cache.insert(k2, b"BBBB");
        cache.insert(k3, b"CCCC");
        assert!(cache.used_bytes() <= 12);
        assert_eq!(cache.get_and_touch(k1), Some(b"AAAA".as_slice()));

        // Oversized entries (> budget / 2) are rejected.
        let k_big = SliceKey {
            shard_idx: 0,
            file_offset: 64,
            len: 8,
        };
        cache.insert(k_big, b"12345678");
        assert_eq!(cache.get_and_touch(k_big), None);

        // Inserting k4 stays strictly within byte budget while keeping hot k1 resident.
        cache.insert(k4, b"DDDD");
        assert!(cache.used_bytes() <= 12);
        assert_eq!(cache.get_and_touch(k1), Some(b"AAAA".as_slice()));

        cache.clear();
        assert_eq!(cache.used_bytes(), 0);
        assert_eq!(cache.get_and_touch(k1), None);
    }

    #[test]
    fn expert_slice_cache_resists_cyclic_scan_thrashing() {
        // Capacity 80 bytes (20 items) scanning 30 items repeatedly: pure LRU gets 0% hits,
        // S3-FIFO keeps hot items resident across sweeps.
        let mut cache = ExpertSliceCache::new(Some(80));
        let keys: Vec<SliceKey> = (0..30)
            .map(|i| SliceKey {
                shard_idx: 0,
                file_offset: i * 4,
                len: 4,
            })
            .collect();
        let mut hits = 0usize;
        let mut total = 0usize;
        for sweep in 0..8 {
            for &k in &keys {
                if sweep >= 2 {
                    total += 1;
                }
                if cache.get_and_touch(k).is_some() {
                    if sweep >= 2 {
                        hits += 1;
                    }
                } else {
                    cache.insert(k, b"DATA");
                }
            }
        }
        assert!(
            hits * 100 / total >= 40,
            "expected >=40% hit rate under cyclic scan, got {hits}/{total}"
        );
    }

    #[test]
    fn expert_slice_cache_disabled_at_zero_budget() {
        let mut cache = ExpertSliceCache::new(Some(0));
        let k = SliceKey {
            shard_idx: 0,
            file_offset: 0,
            len: 4,
        };
        cache.insert(k, b"TEST");
        assert_eq!(cache.used_bytes(), 0);
        assert_eq!(cache.get_and_touch(k), None);
    }

    #[test]
    fn load_slices_into_reads_single_and_coalesced_spans_parallel_and_serial() {
        use crate::gguf_index::{ModelMeta, TensorLocation};
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();
        let shard_path = dir.path().join("synthetic.gguf");
        // 512 KiB synthetic shard with unique byte values per 4 KiB page.
        let mut payload = vec![0u8; 512 * 1024];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = ((i / 251) ^ (i % 251)) as u8;
        }
        std::fs::File::create(&shard_path)
            .unwrap()
            .write_all(&payload)
            .unwrap();

        let mut tensors = HashMap::new();
        // t0 and t1 are 8 KiB apart (< 64 KiB merge gap -> coalesced multi-slice span).
        tensors.insert(
            "t0".to_string(),
            TensorLocation {
                name: "t0".to_string(),
                shard_idx: 0,
                abs_offset: 4096,
                ggml_type: 0,
                dims: vec![8192],
            },
        );
        tensors.insert(
            "t1".to_string(),
            TensorLocation {
                name: "t1".to_string(),
                shard_idx: 0,
                abs_offset: 16384,
                ggml_type: 0,
                dims: vec![8192],
            },
        );
        // t2 and t3 are > 128 KiB apart (-> single-slice direct-to-dst fast path).
        tensors.insert(
            "t2".to_string(),
            TensorLocation {
                name: "t2".to_string(),
                shard_idx: 0,
                abs_offset: 160 * 1024,
                ggml_type: 0,
                dims: vec![12288],
            },
        );
        tensors.insert(
            "t3".to_string(),
            TensorLocation {
                name: "t3".to_string(),
                shard_idx: 0,
                abs_offset: 384 * 1024,
                ggml_type: 0,
                dims: vec![12288],
            },
        );

        let index = GgufIndex {
            meta: ModelMeta {
                architecture: "qwen3moe".to_string(),
                name: None,
                n_layers: 1,
                context_length: Some(2048),
                total_file_bytes: payload.len() as u64,
            },
            shards: vec![shard_path],
            shard_sizes: vec![payload.len() as u64],
            tensors,
        };

        let mut pool = LayerIoPool::new(index, false).unwrap();
        pool.sparse_direct_io = false;

        let reqs = [
            ("t0", 128u64, 0usize, 4096usize, 0u32),
            ("t1", 256u64, 4096usize, 4096usize, 1u32),
            ("t2", 512u64, 8192usize, 8192usize, 2u32),
            ("t3", 0u64, 16384usize, 8192usize, 3u32),
        ];

        for threads in [1usize, 4usize] {
            pool.moe_io_threads = threads;
            pool.clear_expert_cache();
            let mut dst = vec![0u8; 24576];
            pool.load_slices_into(&reqs, &mut dst).unwrap();

            assert_eq!(&dst[0..4096], &payload[4096 + 128..4096 + 128 + 4096]);
            assert_eq!(&dst[4096..8192], &payload[16384 + 256..16384 + 256 + 4096]);
            assert_eq!(
                &dst[8192..16384],
                &payload[160 * 1024 + 512..160 * 1024 + 512 + 8192]
            );
            assert_eq!(&dst[16384..24576], &payload[384 * 1024..384 * 1024 + 8192]);

            // Second read with warm expert cache serves all 4 slices from RAM.
            let before = pool.stats();
            let mut dst2 = vec![0u8; 24576];
            pool.load_slices_into(&reqs, &mut dst2).unwrap();
            let after = pool.stats();
            assert_eq!(dst2, dst);
            assert_eq!(after.cache_hits - before.cache_hits, 4);
            assert_eq!(after.bytes_read, before.bytes_read);
        }

        // Overlapping destination ranges must still fail closed.
        let bad = [
            ("t0", 0u64, 0usize, 4096usize, 0u32),
            ("t1", 0u64, 2048usize, 4096usize, 1u32),
        ];
        let mut dst = vec![0u8; 8192];
        assert!(pool.load_slices_into(&bad, &mut dst).is_err());
    }

    #[test]
    fn scoped_worker_pool_runs_parallel_and_overflow_batches() {
        let pool = ScopedWorkerPool::new(4);
        for n_tasks in [0usize, 1, 2, 4, 7] {
            let hits: Vec<AtomicU64> = (0..n_tasks).map(|_| AtomicU64::new(0)).collect();
            pool.run(n_tasks, |idx| {
                hits[idx].fetch_add(1, Ordering::Relaxed);
            });
            for h in &hits {
                assert_eq!(h.load(Ordering::Relaxed), 1);
            }
        }
    }

    #[test]
    fn load_layer_into_cached_layer_and_storage_probe_work() {
        use crate::gguf_index::{ModelMeta, TensorLocation};
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();
        let shard_path = dir.path().join("layer_test.gguf");
        let mut payload = vec![0u8; 64 * 1024];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = ((i * 13) ^ (i >> 5)) as u8;
        }
        std::fs::File::create(&shard_path)
            .unwrap()
            .write_all(&payload)
            .unwrap();

        let mut tensors = HashMap::new();
        tensors.insert(
            "blk.0.w0".to_string(),
            TensorLocation {
                name: "blk.0.w0".to_string(),
                shard_idx: 0,
                abs_offset: 4096,
                ggml_type: 0,
                dims: vec![1024], // 4096 bytes
            },
        );
        tensors.insert(
            "blk.0.w1".to_string(),
            TensorLocation {
                name: "blk.0.w1".to_string(),
                shard_idx: 0,
                abs_offset: 12288,
                ggml_type: 0,
                dims: vec![1024], // 4096 bytes
            },
        );

        let index = GgufIndex {
            meta: ModelMeta {
                architecture: "llama".to_string(),
                name: Some("test".to_string()),
                n_layers: 1,
                context_length: Some(512),
                total_file_bytes: payload.len() as u64,
            },
            shards: vec![shard_path.clone()],
            shard_sizes: vec![payload.len() as u64],
            tensors,
        };
        let pool = LayerIoPool::new(index, false).unwrap();

        // First call plans & caches layer 0 spans.
        let mut dst = vec![0u8; 8192];
        assert!(!pool.load_cached_layer_into(0, &mut dst).unwrap());
        pool.load_layer_tensors_into(
            0,
            &[("blk.0.w0", 0, 4096), ("blk.0.w1", 4096, 4096)],
            &mut dst,
        )
        .unwrap();
        assert_eq!(&dst[..4096], &payload[4096..8192]);
        assert_eq!(&dst[4096..8192], &payload[12288..16384]);

        // Uncached one-shot load (`layer_idx = -1`) also matches.
        let mut dst_oneshot = vec![0u8; 8192];
        pool.load_tensors_into(
            &[("blk.0.w0", 0, 4096), ("blk.0.w1", 4096, 4096)],
            &mut dst_oneshot,
        )
        .unwrap();
        assert_eq!(dst_oneshot, dst);

        // Second call hits cached layer spans directly.
        let mut dst2 = vec![0u8; 8192];
        assert!(pool.load_cached_layer_into(0, &mut dst2).unwrap());
        assert_eq!(dst2, dst);
        assert!(pool.stats().bytes_read > 0);
        assert!(pool.stats().span_reads >= 2);

        // Undersized destination buffer fails with typed OutOfBounds.
        let mut too_small = vec![0u8; 4096];
        let err = pool.load_cached_layer_into(0, &mut too_small).unwrap_err();
        assert!(matches!(err, GgufIoError::OutOfBounds(_)));

        // Storage probes succeed on real files/directories.
        let (mib_s, _direct) = probe_storage_mib_per_s(&shard_path, false).unwrap();
        assert!(mib_s >= 0.0);
        assert!(probe_dir_storage_mib_per_s(dir.path()).is_some());

        // `record_io_sample` triggers one-time slow-storage warning at >=128 MiB and <150 MiB/s.
        assert!(!pool.slow_storage_warned.load(Ordering::Relaxed));
        pool.record_io_sample(128 * 1024 * 1024, 2_000_000); // 64 MiB/s
        assert!(pool.slow_storage_warned.load(Ordering::Relaxed));
        pool.record_io_sample(128 * 1024 * 1024, 2_000_000); // idempotent second call
    }
}
