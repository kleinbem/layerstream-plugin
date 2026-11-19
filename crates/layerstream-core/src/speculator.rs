//! Speculative token proposers (zero-VRAM n-gram lookup + optional resident draft GGUF).
//!
//! Each speculative pass verifies `K + 1` tokens in a single NVMe layer sweep,
//! multiplying decode throughput on I/O-bound models while preserving exact
//! greedy output parity with the target model.

use anyhow::Result;

use crate::engine::{EngineConfig, StreamEngine};

/// Zero-VRAM prompt/history n-gram lookup proposer.
#[derive(Debug, Clone)]
pub struct NgramProposer {
    pub min_n: usize,
    pub max_n: usize,
    pub max_draft: usize,
}

impl Default for NgramProposer {
    fn default() -> Self {
        Self {
            min_n: 2,
            max_n: 4,
            max_draft: 5,
        }
    }
}

impl NgramProposer {
    /// Searches backwards through `history` for the longest matching suffix n-gram
    /// (`max_n` down to `min_n`) and returns up to `limit` continuation tokens.
    pub fn propose(&self, history: &[i32], limit: usize) -> Vec<i32> {
        let cap = limit.min(self.max_draft);
        if cap == 0 || history.len() < self.min_n + 1 {
            return Vec::new();
        }

        for n in (self.min_n..=self.max_n).rev() {
            if history.len() <= n {
                continue;
            }
            let needle = &history[history.len() - n..];
            // Search backwards for the most recent earlier occurrence of `needle`.
            let search_end = history.len() - n;
            for start in (0..search_end).rev() {
                if &history[start..start + n] == needle {
                    let cont_start = start + n;
                    let avail = (history.len() - cont_start).min(cap);
                    if avail > 0 {
                        return history[cont_start..cont_start + avail].to_vec();
                    }
                }
            }
        }

        Vec::new()
    }
}

/// Optional tiny resident GGUF draft model sharing the GPU lease.
pub struct DraftModelProposer {
    draft: StreamEngine,
    draft_pos: i32,
    synced_len: usize,
    max_draft: usize,
}

impl DraftModelProposer {
    pub fn open(mut cfg: EngineConfig, max_draft: usize) -> Result<Self> {
        cfg.full_resident_reference = true;
        let draft = StreamEngine::open(&cfg)?;
        Ok(Self {
            draft,
            draft_pos: 0,
            synced_len: 0,
            max_draft: max_draft.max(1),
        })
    }

    pub fn reset(&mut self) {
        self.draft.kv_clear();
        self.draft_pos = 0;
        self.synced_len = 0;
    }

    /// Synchronizes draft KV state with the committed `history` and drafts up to `limit` tokens.
    pub fn propose(&mut self, history: &[i32], limit: usize) -> Result<Vec<i32>> {
        let cap = limit.min(self.max_draft);
        if cap == 0 || history.is_empty() {
            return Ok(Vec::new());
        }

        // If target trimmed/diverged or reset, roll draft KV back cleanly.
        if history.len() < self.synced_len {
            self.draft.kv_trim(history.len() as i32);
            self.draft_pos = history.len() as i32;
            self.synced_len = history.len();
        }

        // Feed any newly committed tokens (including the target's latest token) into draft KV.
        let mut next_draft = 0i32;
        if self.synced_len < history.len() {
            let delta = &history[self.synced_len..];
            for chunk in delta.chunks(256) {
                next_draft = self.draft.eval_greedy(chunk, self.draft_pos)?;
                self.draft_pos += chunk.len() as i32;
            }
            self.synced_len = history.len();
        } else {
            return Ok(Vec::new());
        }

        let mut proposed = Vec::with_capacity(cap);
        for _ in 0..cap {
            if self.draft.token_is_eog(next_draft) {
                break;
            }
            proposed.push(next_draft);
            if proposed.len() >= cap {
                break;
            }
            next_draft = self.draft.eval_greedy(&[next_draft], self.draft_pos)?;
            self.draft_pos += 1;
        }

        // Trim speculative draft tokens back out so next call syncs strictly from committed history.
        if self.draft_pos > self.synced_len as i32 {
            self.draft.kv_trim(self.synced_len as i32);
            self.draft_pos = self.synced_len as i32;
        }

        Ok(proposed)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SpeculationStats {
    pub sweeps: usize,
    pub drafted_tokens: usize,
    pub accepted_tokens: usize,
}
