//! Bounded scan-resistant S3-FIFO host-RAM cache for hot MoE expert slices.

use quick_cache::{Weighter, unsync::Cache as QuickCache};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SliceKey {
    pub(crate) shard_idx: usize,
    pub(crate) file_offset: u64,
    pub(crate) len: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct SliceByteWeighter;

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
pub(crate) struct ExpertSliceCache {
    budget_bytes: Option<usize>,
    cache: Option<QuickCache<SliceKey, Vec<u8>, SliceByteWeighter>>,
}

impl ExpertSliceCache {
    pub(crate) fn new(explicit_budget_bytes: Option<usize>) -> Self {
        Self {
            budget_bytes: explicit_budget_bytes,
            cache: None,
        }
    }

    pub(crate) fn ensure_budget(&mut self) -> usize {
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

    pub(crate) fn ensure_cache(
        &mut self,
    ) -> Option<&mut QuickCache<SliceKey, Vec<u8>, SliceByteWeighter>> {
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

    pub(crate) fn get_and_touch(&mut self, key: SliceKey) -> Option<&[u8]> {
        self.cache.as_mut()?.get(&key).map(Vec::as_slice)
    }

    pub(crate) fn insert(&mut self, key: SliceKey, bytes: &[u8]) {
        let budget = self.ensure_budget();
        if budget == 0 || bytes.is_empty() || bytes.len() > budget / 2 {
            return;
        }
        if let Some(cache) = self.ensure_cache() {
            cache.insert(key, bytes.to_vec());
        }
    }

    #[cfg(test)]
    pub(crate) fn used_bytes(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.weight() as usize)
    }

    pub(crate) fn clear(&mut self) {
        if let Some(cache) = self.cache.as_mut() {
            cache.clear();
        }
    }

    pub(crate) fn export_hot_entries(&self) -> Vec<crate::expert_profile::HotSliceEntry> {
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

/// Computes an adaptive TinyLFU hot-expert RAM cache budget from host total and
/// available RAM:
/// - **Edge (`<= 12 GiB` total):** reserves 1.5 GiB headroom and uses 60% of the
///   remaining available RAM up to 2 GiB (keeps tight MoE models like 30B-A3B
///   swap-free while letting smaller MoEs like OLMoE-1B-7B reach ~100% cache hit
///   rate when multi-GiB headroom is free).
/// - **Standard (`12..48 GiB` total):** reserves 2 GiB headroom and caps at 4 GiB.
/// - **High (`>= 48 GiB` total):** reserves 4 GiB headroom and caps at 32 GiB
///   (caches most/all hot experts on 64 GB unified-memory APUs & workstations).
pub fn auto_expert_cache_budget_for_meminfo(total_bytes: u64, avail_bytes: u64) -> usize {
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;
    let (reserve, cap, num, den) = if total_bytes == 0 || total_bytes <= 12 * GIB {
        (1536 * MIB, 2048 * MIB, 3, 5)
    } else if total_bytes < 48 * GIB {
        (2048 * MIB, 4096 * MIB, 3, 5)
    } else {
        (4096 * MIB, 32 * GIB, 1, 2)
    };
    let target = avail_bytes.saturating_sub(reserve).saturating_mul(num) / den;
    target.min(cap) as usize
}

fn auto_expert_cache_budget_bytes() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            let mut total_kb = 0u64;
            let mut avail_kb = 0u64;
            for line in content.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:")
                    && let Some(val) = rest.split_whitespace().next()
                    && let Ok(kb) = val.parse::<u64>()
                {
                    total_kb = kb;
                } else if let Some(rest) = line.strip_prefix("MemAvailable:")
                    && let Some(val) = rest.split_whitespace().next()
                    && let Ok(kb) = val.parse::<u64>()
                {
                    avail_kb = kb;
                }
            }
            if avail_kb > 0 {
                return auto_expert_cache_budget_for_meminfo(
                    total_kb.saturating_mul(1024),
                    avail_kb.saturating_mul(1024),
                );
            }
        }
        0
    }
    #[cfg(not(target_os = "linux"))]
    {
        256 * 1024 * 1024
    }
}
