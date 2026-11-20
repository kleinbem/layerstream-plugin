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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Context, Result, bail};

use crate::ffi::LsSpanSlice;
use crate::gguf_index::{CoalescedSpan, DIRECT_IO_ALIGN, GgufIndex, align_up};

const PARALLEL_READ_MIN_BYTES: usize = 4 * 1024 * 1024;
/// Default staging chunk size for fallback O_DIRECT pread (full-layer 256 MiB spans
/// maximize Linux kernel BIO queue depth through dm-crypt / NVMe).
/// Override with `LAYERSTREAM_IO_CHUNK_MB` (e.g. `16`, `64`, `256`).
const DEFAULT_SCRATCH_CHUNK_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_IO_THREADS: usize = 1;
// Linux dm-crypt/LUKS2 contends across multi-threaded O_DIRECT reads on the same file,
// whereas macOS Apple Silicon's hardware NVMe AES controller benefits from parallel queue depth.
#[cfg(target_os = "macos")]
const DEFAULT_MOE_IO_THREADS: usize = 4;
#[cfg(not(target_os = "macos"))]
const DEFAULT_MOE_IO_THREADS: usize = 1;

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

struct LruEntry {
    key: SliceKey,
    data: Vec<u8>,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Bounded O(1) scan-resistant LFU-LRU cache of raw immutable GGUF expert slices in host RAM.
///
/// Sparse MoE routing is heavily Zipfian, and on large MoE models (e.g., 30B-A3B) a single
/// token sweep can exceed the RAM cache budget. Pure LRU scan-thrashes to ~0% hit rate when
/// `budget < 1 sweep`, so admission compares frequency counts (`freq[incoming] > freq[victim]`)
/// across the LRU tail window and decays counts periodically to adapt to topic shifts.
struct ExpertSliceLru {
    budget_bytes: Option<usize>,
    used_bytes: usize,
    head: Option<usize>,
    tail: Option<usize>,
    entries: Vec<Option<LruEntry>>,
    free_slots: Vec<usize>,
    map: HashMap<SliceKey, usize>,
    freq: HashMap<SliceKey, u16>,
    access_ops: u64,
}

impl ExpertSliceLru {
    const TAIL_SAMPLE_WINDOW: usize = 8;
    const FREQ_DECAY_INTERVAL: u64 = 32_768;

    fn new(explicit_budget_bytes: Option<usize>) -> Self {
        Self {
            budget_bytes: explicit_budget_bytes,
            used_bytes: 0,
            head: None,
            tail: None,
            entries: Vec::new(),
            free_slots: Vec::new(),
            map: HashMap::new(),
            freq: HashMap::new(),
            access_ops: 0,
        }
    }

    fn bump_freq(&mut self, key: SliceKey) -> u16 {
        self.access_ops = self.access_ops.wrapping_add(1);
        if self.access_ops.is_multiple_of(Self::FREQ_DECAY_INTERVAL) {
            self.freq.retain(|_, v| {
                *v >>= 1;
                *v > 0
            });
        }
        let count = self.freq.entry(key).or_insert(0);
        *count = count.saturating_add(1);
        *count
    }

    fn key_freq(&self, key: SliceKey) -> u16 {
        self.freq.get(&key).copied().unwrap_or(0)
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

    fn detach(&mut self, idx: usize) {
        let (prev, next) = {
            let e = self.entries[idx].as_ref().expect("valid LRU slot");
            (e.prev, e.next)
        };
        if let Some(p) = prev {
            self.entries[p].as_mut().expect("valid prev").next = next;
        } else {
            self.head = next;
        }
        if let Some(n) = next {
            self.entries[n].as_mut().expect("valid next").prev = prev;
        } else {
            self.tail = prev;
        }
        let e = self.entries[idx].as_mut().expect("valid LRU slot");
        e.prev = None;
        e.next = None;
    }

    fn push_front(&mut self, idx: usize) {
        let old_head = self.head;
        {
            let e = self.entries[idx].as_mut().expect("valid LRU slot");
            e.prev = None;
            e.next = old_head;
        }
        if let Some(h) = old_head {
            self.entries[h].as_mut().expect("valid head").prev = Some(idx);
        } else {
            self.tail = Some(idx);
        }
        self.head = Some(idx);
    }

    fn get_and_touch(&mut self, key: SliceKey) -> Option<&[u8]> {
        let idx = *self.map.get(&key)?;
        self.bump_freq(key);
        if self.head != Some(idx) {
            self.detach(idx);
            self.push_front(idx);
        }
        self.entries[idx].as_ref().map(|e| e.data.as_slice())
    }

    /// Picks the lowest-frequency victim within the LRU tail window (tie-broken by oldest).
    fn pick_victim(&self) -> Option<(usize, u16)> {
        let mut cur = self.tail;
        let mut best: Option<(usize, u16)> = None;
        let mut steps = 0usize;
        while let Some(idx) = cur {
            if steps >= Self::TAIL_SAMPLE_WINDOW {
                break;
            }
            let e = self.entries[idx].as_ref().expect("valid LRU slot");
            let f = self.key_freq(e.key);
            if best.is_none_or(|(_, bf)| f < bf) {
                best = Some((idx, f));
                if f <= 1 {
                    break;
                }
            }
            cur = e.prev;
            steps += 1;
        }
        best
    }

    fn evict_slot(&mut self, idx: usize) {
        self.detach(idx);
        if let Some(entry) = self.entries[idx].take() {
            self.map.remove(&entry.key);
            self.used_bytes = self.used_bytes.saturating_sub(entry.data.len());
        }
        self.free_slots.push(idx);
    }

    fn insert(&mut self, key: SliceKey, bytes: &[u8]) {
        let budget = self.ensure_budget();
        if budget == 0 || bytes.is_empty() || bytes.len() > budget / 2 {
            return;
        }
        let incoming_freq = self.bump_freq(key);
        if let Some(&idx) = self.map.get(&key) {
            if self.head != Some(idx) {
                self.detach(idx);
                self.push_front(idx);
            }
            return;
        }

        // Scan-resistant TinyLFU admission gate: once full, only admit if the incoming slice
        // is strictly hotter than the coldest candidate in the LRU tail window!
        if self.used_bytes.saturating_add(bytes.len()) > budget {
            let Some((_, victim_freq)) = self.pick_victim() else {
                return;
            };
            if incoming_freq <= victim_freq {
                return;
            }
        }

        while self.used_bytes.saturating_add(bytes.len()) > budget {
            let Some((victim_idx, victim_freq)) = self.pick_victim() else {
                return;
            };
            if incoming_freq <= victim_freq {
                return;
            }
            self.evict_slot(victim_idx);
        }
        let entry = LruEntry {
            key,
            data: bytes.to_vec(),
            prev: None,
            next: None,
        };
        let idx = if let Some(slot) = self.free_slots.pop() {
            self.entries[slot] = Some(entry);
            slot
        } else {
            let slot = self.entries.len();
            self.entries.push(Some(entry));
            slot
        };
        self.used_bytes += bytes.len();
        self.map.insert(key, idx);
        self.push_front(idx);
    }

    fn clear(&mut self) {
        self.map.clear();
        self.freq.clear();
        self.entries.clear();
        self.free_slots.clear();
        self.head = None;
        self.tail = None;
        self.used_bytes = 0;
        self.access_ops = 0;
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

pub struct LayerIoPool {
    index: GgufIndex,
    shards: Vec<ShardHandle>,
    direct_io_active: bool,
    io_threads: usize,
    moe_io_threads: usize,
    max_chunk_bytes: usize,
    profile_enabled: bool,
    scratch: Mutex<Option<AlignedBuf>>,
    moe_scratches: Mutex<Vec<AlignedBuf>>,
    span_cache: Mutex<HashMap<i32, Vec<CoalescedSpan>>>,
    expert_cache: Mutex<ExpertSliceLru>,
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
            .unwrap_or(DEFAULT_MOE_IO_THREADS);

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

        Ok(Self {
            index,
            shards,
            direct_io_active: all_direct,
            io_threads,
            moe_io_threads,
            max_chunk_bytes,
            profile_enabled,
            scratch: Mutex::new(None),
            moe_scratches: Mutex::new(Vec::new()),
            span_cache: Mutex::new(HashMap::new()),
            expert_cache: Mutex::new(ExpertSliceLru::new(explicit_expert_cache)),
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
        if let Ok(mut cache) = self.expert_cache.lock() {
            cache.clear();
        }
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
        out_slices: &mut [LsSpanSlice],
    ) -> Result<usize> {
        if layer_idx < 0 || !(pin_buf.as_ptr() as usize).is_multiple_of(DIRECT_IO_ALIGN as usize) {
            return Ok(0);
        }

        let spans = {
            let cached = self
                .span_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("span_cache mutex poisoned"))?
                .get(&layer_idx)
                .cloned();
            if let Some(s) = cached {
                s
            } else {
                let Some(reqs) = requests else {
                    return Ok(0);
                };
                let planned = self.index.plan_coalesced_spans(reqs)?;
                if let Ok(mut cache) = self.span_cache.lock() {
                    cache.insert(layer_idx, planned.clone());
                }
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
        for span in &spans {
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
                out_slices[slice_idx] = LsSpanSlice {
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

        let mut scratch_guard = self
            .scratch
            .lock()
            .map_err(|_| anyhow::anyhow!("IO scratch mutex poisoned"))?;
        let scratch = match scratch_guard.as_mut() {
            Some(buf) => buf,
            None => scratch_guard.insert(AlignedBuf::new(4 * 1024 * 1024)?),
        };
        for span in &spans {
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
        let spans = self.index.plan_coalesced_spans(requests)?;
        if layer_idx >= 0
            && let Ok(mut cache) = self.span_cache.lock()
        {
            cache.insert(layer_idx, spans.clone());
        }

        if layer_idx < 0 {
            // One-shot startup load (e.g. resident layers): use an ephemeral scratch buffer
            // so its memory is returned to the OS immediately after initialization.
            let mut ephemeral = AlignedBuf::new(4 * 1024 * 1024)?;
            for span in &spans {
                self.read_span_into(span, &mut ephemeral, dst)?;
            }
            return Ok(());
        }

        let mut scratch_guard = self
            .scratch
            .lock()
            .map_err(|_| anyhow::anyhow!("IO scratch mutex poisoned"))?;
        let scratch = match scratch_guard.as_mut() {
            Some(buf) => buf,
            None => scratch_guard.insert(AlignedBuf::new(4 * 1024 * 1024)?),
        };
        for span in &spans {
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

        // 1. Check bounded hot-expert RAM LRU cache first.
        let mut misses: Vec<(&str, u64, usize, usize, u32)> = Vec::new();
        let mut miss_meta: Vec<(SliceKey, usize, usize)> = Vec::new();
        let cache_enabled = {
            let mut cache = self
                .expert_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("expert_cache mutex poisoned"))?;
            let budget = cache.ensure_budget();
            if budget > 0 {
                let mut hits = 0u64;
                let mut hit_bytes = 0u64;
                misses.reserve(requests.len());
                miss_meta.reserve(requests.len());
                for &(name, src_off, dst_off, len, tidx) in requests {
                    let end = dst_off
                        .checked_add(len)
                        .context("slice dst_offset overflow")?;
                    if end > dst.len() {
                        bail!(
                            "slice dst range [{dst_off}..{end}) exceeds dst_cap {}",
                            dst.len()
                        );
                    }
                    if let Some(loc) = self.index.tensors.get(name) {
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
                    }
                    misses.push((name, src_off, dst_off, len, tidx));
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

        let active_reqs: &[(&str, u64, usize, usize, u32)] = if cache_enabled {
            if misses.is_empty() {
                return Ok(());
            }
            &misses
        } else {
            requests
        };

        let spans = self.index.plan_coalesced_sub_slices(active_reqs)?;
        let n_workers = self.moe_io_threads.min(spans.len()).max(1);
        if n_workers <= 1 {
            let mut scratch_guard = self
                .scratch
                .lock()
                .map_err(|_| anyhow::anyhow!("IO scratch mutex poisoned"))?;
            let scratch = match scratch_guard.as_mut() {
                Some(buf) => buf,
                None => scratch_guard.insert(AlignedBuf::new(4 * 1024 * 1024)?),
            };
            for span in &spans {
                self.read_span_into(span, scratch, dst)?;
            }
        } else {
            // Verify all destination ranges are in-bounds and mutually disjoint before
            // parallelizing copies into `dst`.
            let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(active_reqs.len());
            for span in &spans {
                for sl in &span.slices {
                    let end = sl
                        .dst_offset
                        .checked_add(sl.len)
                        .context("slice dst_offset overflow")?;
                    if end > dst.len() {
                        bail!(
                            "slice dst range [{}..{}) exceeds dst_cap {}",
                            sl.dst_offset,
                            end,
                            dst.len()
                        );
                    }
                    ranges.push((sl.dst_offset, end));
                }
            }
            ranges.sort_unstable_by_key(|r| r.0);
            for w in ranges.windows(2) {
                if w[0].1 > w[1].0 {
                    bail!(
                        "overlapping slice destination ranges [{}..{}) and [{}..{})",
                        w[0].0,
                        w[0].1,
                        w[1].0,
                        w[1].1
                    );
                }
            }

            #[derive(Clone, Copy)]
            struct SendDstPtr(*mut u8, usize);
            impl SendDstPtr {
                #[inline]
                fn parts(self) -> (*mut u8, usize) {
                    (self.0, self.1)
                }
            }
            unsafe impl Send for SendDstPtr {}
            unsafe impl Sync for SendDstPtr {}

            let mut scratches = self
                .moe_scratches
                .lock()
                .map_err(|_| anyhow::anyhow!("moe_scratches mutex poisoned"))?;
            while scratches.len() < n_workers {
                scratches.push(AlignedBuf::new(2 * 1024 * 1024)?);
            }

            let dst_ptr = SendDstPtr(dst.as_mut_ptr(), dst.len());
            let chunk_len = spans.len().div_ceil(n_workers);
            let first_err: Mutex<Option<anyhow::Error>> = Mutex::new(None);
            std::thread::scope(|s| {
                for (span_chunk, scratch_buf) in spans
                    .chunks(chunk_len)
                    .zip(scratches.iter_mut().take(n_workers))
                {
                    let err_slot = &first_err;
                    let dst_send = dst_ptr;
                    s.spawn(move || {
                        let (base, len) = dst_send.parts();
                        for span in span_chunk {
                            if let Err(e) =
                                self.read_span_into_disjoint_ptr(span, scratch_buf, base, len)
                            {
                                if let Ok(mut g) = err_slot.lock()
                                    && g.is_none()
                                {
                                    *g = Some(e);
                                }
                                return;
                            }
                        }
                    });
                }
            });

            if let Some(e) = first_err.into_inner().ok().flatten() {
                return Err(e);
            }
        }

        // 2. Populate the hot-expert LRU cache with newly read misses.
        if cache_enabled && !miss_meta.is_empty() {
            let mut cache = self
                .expert_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("expert_cache mutex poisoned"))?;
            for (key, dst_off, len) in miss_meta {
                cache.insert(key, &dst[dst_off..dst_off + len]);
            }
        }
        Ok(())
    }

    /// Internal helper for parallel disjoint slice writes (caller must verify disjointness and bounds).
    fn read_span_into_disjoint_ptr(
        &self,
        span: &CoalescedSpan,
        scratch: &mut AlignedBuf,
        dst_base: *mut u8,
        dst_len: usize,
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
                used_direct = matches!(pread_exact_or_eof(direct_file, read_buf, file_off), Ok(n) if n >= chunk_valid);
            }
            if !used_direct {
                pread_all(&handle.buffered, &mut read_buf[..chunk_valid], file_off)?;
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
                    debug_assert!(dst_off + len <= dst_len);
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
        Ok(())
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
                    dst[dst_off..dst_off + len].copy_from_slice(&read_buf[src_off..src_off + len]);
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
                     chunk_cap={}MiB threads={} direct={} pread={pread_ms:.1}ms ({bw:.0} MiB/s) \
                     memcpy={copy_ms:.1}ms expert_cache={hits}/{} ({hit_pct:.1}% hit, {cached_mib:.1}MiB saved)",
                    self.max_chunk_bytes / (1024 * 1024),
                    self.io_threads,
                    self.direct_io_active,
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

#[cfg(target_os = "macos")]
fn open_direct(path: &std::path::Path) -> Option<File> {
    use std::os::unix::io::AsRawFd;
    let file = OpenOptions::new().read(true).open(path).ok()?;
    let fd = file.as_raw_fd();
    let rc = unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
    // Disable speculative OS read-ahead so scattered MoE expert slices don't waste SSD bandwidth.
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

/// Measures sequential read throughput `(mib_per_s, direct_io_active)` on `path` by reading
/// up to 16 MiB (4K-aligned). Intended for fast (~10-50 ms) startup preflight diagnostics.
pub fn probe_storage_mib_per_s(
    path: &std::path::Path,
    prefer_direct_io: bool,
) -> Result<(f64, bool)> {
    let buffered = File::open(path)
        .with_context(|| format!("failed to open {} for storage probe", path.display()))?;
    let file_len = buffered.metadata()?.len();
    if file_len < 4096 {
        return Ok((0.0, false));
    }
    let direct = if prefer_direct_io {
        open_direct(path)
    } else {
        None
    };
    let direct_active = direct.is_some();
    let probe_bytes = (file_len as usize).min(16 * 1024 * 1024);
    let aligned_len = align_up(probe_bytes as u64, DIRECT_IO_ALIGN) as usize;
    let mut buf = AlignedBuf::new(aligned_len)?;
    let slice = buf.as_mut_slice(aligned_len);
    let t0 = std::time::Instant::now();
    let n = if let Some(ref df) = direct {
        pread_exact_or_eof(df, slice, 0)?
    } else {
        pread_exact_or_eof(&buffered, &mut slice[..probe_bytes], 0)?
    };
    let elapsed_s = t0.elapsed().as_secs_f64().max(1e-6);
    let mib_per_s = (n as f64 / (1024.0 * 1024.0)) / elapsed_s;
    Ok((mib_per_s, direct_active))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_slice_lru_evicts_least_recently_used_and_respects_budget() {
        let mut lru = ExpertSliceLru::new(Some(12));
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

        lru.insert(k1, b"AAAA");
        lru.insert(k2, b"BBBB");
        lru.insert(k3, b"CCCC");
        assert_eq!(lru.used_bytes, 12);

        // Touch k1 so k1 has freq=2 while k2 and k3 have freq=1.
        assert_eq!(lru.get_and_touch(k1), Some(b"AAAA".as_slice()));

        // First access to k4 (freq=1) does NOT thrash-evict existing entries (scan resistance).
        lru.insert(k4, b"DDDD");
        assert_eq!(lru.used_bytes, 12);
        assert_eq!(lru.get_and_touch(k4), None);
        assert_eq!(lru.get_and_touch(k1), Some(b"AAAA".as_slice()));

        // Second access to k4 (freq=2) beats the oldest freq=1 victim (k2) and is admitted!
        lru.insert(k4, b"DDDD");
        assert_eq!(lru.used_bytes, 12);
        assert_eq!(lru.get_and_touch(k2), None);
        assert_eq!(lru.get_and_touch(k1), Some(b"AAAA".as_slice()));
        assert_eq!(lru.get_and_touch(k3), Some(b"CCCC".as_slice()));
        assert_eq!(lru.get_and_touch(k4), Some(b"DDDD".as_slice()));

        lru.clear();
        assert_eq!(lru.used_bytes, 0);
        assert_eq!(lru.get_and_touch(k1), None);
    }

    #[test]
    fn expert_slice_lru_resists_cyclic_scan_thrashing() {
        // Capacity 12 (3 items) scanning 5 items repeatedly: pure LRU gets 0% hits,
        // scan-resistant TinyLFU-LRU keeps 3/5 = 60% hits across sweeps!
        let mut lru = ExpertSliceLru::new(Some(12));
        let keys: Vec<SliceKey> = (0..5)
            .map(|i| SliceKey {
                shard_idx: 0,
                file_offset: i * 4,
                len: 4,
            })
            .collect();
        let mut hits = 0usize;
        let mut total = 0usize;
        for sweep in 0..5 {
            for &k in &keys {
                if sweep > 0 {
                    total += 1;
                }
                if lru.get_and_touch(k).is_some() {
                    if sweep > 0 {
                        hits += 1;
                    }
                } else {
                    lru.insert(k, b"DATA");
                }
            }
        }
        assert!(
            hits * 100 / total >= 55,
            "expected >=55% hit rate under cyclic scan, got {hits}/{total}"
        );
    }

    #[test]
    fn expert_slice_lru_disabled_at_zero_budget() {
        let mut lru = ExpertSliceLru::new(Some(0));
        let k = SliceKey {
            shard_idx: 0,
            file_offset: 0,
            len: 4,
        };
        lru.insert(k, b"TEST");
        assert_eq!(lru.used_bytes, 0);
        assert_eq!(lru.get_and_touch(k), None);
    }
}
