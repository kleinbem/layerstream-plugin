use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};

use crate::ffi::{
    LsChatMessage, LsEngine, LsEngineConfig, LsEngineStats, LsSampleParams, LsSpanSlice,
    LsTensorSliceReq, ls_engine_apply_chat_template, ls_engine_create, ls_engine_eval_greedy,
    ls_engine_eval_sample, ls_engine_eval_verify_greedy, ls_engine_eval_verify_sample,
    ls_engine_free, ls_engine_get_stats, ls_engine_kv_clear, ls_engine_kv_trim, ls_engine_n_ctx,
    ls_engine_set_cancel_cb, ls_engine_token_is_eog, ls_engine_token_to_piece, ls_engine_tokenize,
    ls_engine_vocab_size, ls_probe_gpu_memory,
};
use crate::gguf_index::{GgufIndex, MAX_DRAFT_VOCAB_PAD_DELTA};
use crate::io_reader::{IoStats, LayerIoPool};
use crate::speculator::{DraftModelProposer, NgramProposer, SpeculationStats};

const RC_CANCELLED: c_int = -100;

#[derive(Debug, Clone)]
pub struct PromptTooLongError {
    pub prompt_tokens: usize,
    pub n_ctx: usize,
}

impl fmt::Display for PromptTooLongError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "prompt has {} tokens, which exceeds context window n_ctx={} (need at least 1 token for generation)",
            self.prompt_tokens, self.n_ctx
        )
    }
}

impl std::error::Error for PromptTooLongError {}

#[derive(Debug, Clone, Copy)]
pub struct CancelledError;

impl fmt::Display for CancelledError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "generation cancelled")
    }
}

impl std::error::Error for CancelledError {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KvCacheType {
    #[default]
    F16 = 0,
    Q8_0 = 1,
    Q4_0 = 2,
}

impl std::str::FromStr for KvCacheType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f16" => Ok(Self::F16),
            "q8_0" => Ok(Self::Q8_0),
            "q4_0" => Ok(Self::Q4_0),
            other => bail!("invalid kv_cache_type '{other}' (expected f16, q8_0, or q4_0)"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_path: PathBuf,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub resident_layers: Option<u32>,
    pub vram_budget_bytes: u64,
    /// Probe free GPU memory automatically and pick a safe hybrid residency budget.
    pub vram_budget_auto: bool,
    pub kv_cache_type: KvCacheType,
    pub use_gpu: bool,
    pub prefer_direct_io: bool,
    pub full_resident_reference: bool,
    /// Pin `token_embd.weight` in CPU RAM instead of GPU VRAM.
    pub cpu_embeddings: bool,
    /// Pin `output.weight` (LM head) in CPU RAM instead of GPU VRAM.
    pub cpu_head: bool,
    /// Stream `output.weight` through ping-pong GPU slots when <= slot_bytes (else CPU fallback).
    pub stream_head: bool,
    /// Enable per-expert active streaming for sparse MoE layers (default true).
    pub moe_active_streaming: bool,
}

impl EngineConfig {
    pub fn streaming(model_path: impl AsRef<Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            n_ctx: 4096,
            n_batch: 512,
            // Default to 2 CPU threads when GPU offloading is enabled so host cores stay free
            // for NVMe/dm-crypt BIO completion and HTTP serving.
            n_threads: 2,
            resident_layers: None,
            vram_budget_bytes: 0,
            vram_budget_auto: false,
            kv_cache_type: KvCacheType::F16,
            use_gpu: true,
            prefer_direct_io: true,
            full_resident_reference: false,
            cpu_embeddings: false,
            cpu_head: false,
            stream_head: false,
            moe_active_streaming: true,
        }
    }

    pub fn reference(model_path: impl AsRef<Path>) -> Self {
        Self {
            full_resident_reference: true,
            ..Self::streaming(model_path)
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_tokens: usize,
    pub sample: LsSampleParams,
    pub stop: Vec<String>,
    pub enable_ngram_speculation: bool,
    pub max_draft_tokens: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            sample: LsSampleParams::default(),
            stop: Vec::new(),
            enable_ngram_speculation: true,
            max_draft_tokens: 5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateSummary {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
    pub speculation: SpeculationStats,
}

pub struct StreamEngine {
    raw: NonNull<LsEngine>,
    /// Raw heap-allocated `LayerIoPool` owned by `StreamEngine` via `Box::into_raw` so
    /// moving `StreamEngine` never invalidates the C++ callback pointer under Stacked Borrows.
    /// Reconstructed and dropped in `Drop` strictly AFTER `ls_engine_free` joins the I/O worker.
    io_pool: Option<*mut LayerIoPool>,
    n_batch: usize,
    cached_kv_tokens: Vec<i32>,
}

unsafe impl Send for StreamEngine {}

impl StreamEngine {
    pub fn open(cfg: &EngineConfig) -> Result<Self> {
        #[cfg(unix)]
        let path_bytes = std::os::unix::ffi::OsStrExt::as_bytes(cfg.model_path.as_os_str());
        #[cfg(not(unix))]
        let path_lossy = cfg.model_path.to_string_lossy();
        #[cfg(not(unix))]
        let path_bytes = path_lossy.as_bytes();
        let c_path = CString::new(path_bytes).context("model_path contains interior NUL")?;

        let io_pool_raw: Option<*mut LayerIoPool> = if cfg.full_resident_reference {
            None
        } else {
            let index = GgufIndex::open(&cfg.model_path)?;
            let pool = LayerIoPool::new(index, cfg.prefer_direct_io)?;
            Some(Box::into_raw(Box::new(pool)))
        };

        let io_ptr = io_pool_raw
            .map(|p| p.cast::<c_void>())
            .unwrap_or(std::ptr::null_mut());

        let raw_cfg = LsEngineConfig {
            model_path: c_path.as_ptr(),
            n_ctx: cfg.n_ctx,
            n_batch: cfg.n_batch,
            n_threads: cfg.n_threads,
            resident_layers: cfg.resident_layers.map_or(-1, |v| v as i32),
            vram_budget_bytes: cfg.vram_budget_bytes,
            vram_budget_auto: cfg.vram_budget_auto,
            kv_cache_type: cfg.kv_cache_type as i32,
            use_gpu: cfg.use_gpu,
            full_resident_reference: cfg.full_resident_reference,
            cpu_embeddings: cfg.cpu_embeddings,
            cpu_head: cfg.cpu_head,
            stream_head: cfg.stream_head,
            moe_active_streaming: cfg.moe_active_streaming,
            io_user_data: io_ptr,
            load_tensors_cb: if io_pool_raw.is_some() {
                Some(rust_load_tensors_cb)
            } else {
                None
            },
            load_layer_direct_cb: if io_pool_raw.is_some() {
                Some(rust_load_layer_direct_cb)
            } else {
                None
            },
            load_slices_cb: if io_pool_raw.is_some() {
                Some(rust_load_slices_cb)
            } else {
                None
            },
        };

        let mut err_buf = [0 as c_char; 512];
        let ptr = unsafe { ls_engine_create(&raw_cfg, err_buf.as_mut_ptr(), err_buf.len()) };
        let Some(raw) = NonNull::new(ptr) else {
            if let Some(p) = io_pool_raw {
                drop(unsafe { Box::from_raw(p) });
            }
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("failed to create layerstream engine: {msg}");
        };

        Ok(Self {
            raw,
            io_pool: io_pool_raw,
            n_batch: cfg.n_batch.max(1) as usize,
            cached_kv_tokens: Vec::new(),
        })
    }

    pub fn stats(&self) -> LsEngineStats {
        let mut out = LsEngineStats::default();
        unsafe { ls_engine_get_stats(self.raw.as_ptr(), &mut out) };
        out
    }

    pub fn io_stats(&self) -> IoStats {
        self.io_pool
            .map(|p| unsafe { &*p }.stats())
            .unwrap_or_default()
    }

    pub fn clear_expert_cache(&self) {
        if let Some(p) = self.io_pool {
            unsafe { &*p }.clear_expert_cache();
        }
    }

    pub fn vocab_size(&self) -> i32 {
        unsafe { ls_engine_vocab_size(self.raw.as_ptr()) }
    }

    pub fn n_ctx(&self) -> u32 {
        unsafe { ls_engine_n_ctx(self.raw.as_ptr()) }
    }

    /// Probes available and total memory `(free_bytes, total_bytes)` on the primary GPU backend.
    pub fn probe_gpu_memory(use_gpu: bool) -> Option<(u64, u64)> {
        let mut free_b = 0u64;
        let mut total_b = 0u64;
        let ok = unsafe { ls_probe_gpu_memory(use_gpu, &mut free_b, &mut total_b) };
        if ok && total_b > 0 {
            Some((free_b, total_b))
        } else {
            None
        }
    }

    /// Verifies that a draft model shares the same base tokenizer vocabulary and token IDs
    /// as this target model. Allows up to `MAX_DRAFT_VOCAB_PAD_DELTA` (256) trailing padded
    /// entries (e.g., Qwen2.5-14B/32B `152,064` vs `0.5B` `151,936`).
    pub fn check_draft_compatible(&self, draft: &StreamEngine) -> std::result::Result<(), String> {
        let target_v = self.vocab_size();
        let draft_v = draft.vocab_size();
        if target_v <= 0 || draft_v <= 0 {
            return Err(format!(
                "invalid vocabulary size (target={target_v}, draft={draft_v})"
            ));
        }
        let delta = (target_v as usize).abs_diff(draft_v as usize);
        if delta > MAX_DRAFT_VOCAB_PAD_DELTA {
            return Err(format!(
                "vocabulary size mismatch (target={target_v}, draft={draft_v}, delta={delta} > {MAX_DRAFT_VOCAB_PAD_DELTA})"
            ));
        }
        const PROBE: &str = "Hello, world! 12345\nfn main() { println!(\"ok\"); }";
        let target_ids = self
            .tokenize(PROBE, false, false)
            .map_err(|e| format!("target probe tokenize failed: {e}"))?;
        let draft_ids = draft
            .tokenize(PROBE, false, false)
            .map_err(|e| format!("draft probe tokenize failed: {e}"))?;
        if target_ids != draft_ids {
            return Err(format!(
                "tokenizer IDs diverged on canonical probe (target={:?}, draft={:?})",
                &target_ids[..target_ids.len().min(6)],
                &draft_ids[..draft_ids.len().min(6)],
            ));
        }
        let shared = target_v.min(draft_v);
        for id in [0, 1, shared / 4, shared / 2, (shared * 3) / 4, shared - 1] {
            if id < 0 || id >= shared {
                continue;
            }
            let tp = self.token_to_piece(id).unwrap_or_default();
            let dp = draft.token_to_piece(id).unwrap_or_default();
            if tp != dp {
                return Err(format!(
                    "boundary token {id} mismatch (target={tp:?}, draft={dp:?})"
                ));
            }
        }
        Ok(())
    }

    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_assistant_prompt: bool,
    ) -> Result<String> {
        let c_strings: Vec<(CString, CString)> = messages
            .iter()
            .map(|m| {
                let role = CString::new(m.role.as_bytes())
                    .unwrap_or_else(|_| CString::new("user").unwrap());
                let content = CString::new(m.content.replace('\0', "")).unwrap();
                (role, content)
            })
            .collect();
        let raw_msgs: Vec<LsChatMessage> = c_strings
            .iter()
            .map(|(r, c)| LsChatMessage {
                role: r.as_ptr(),
                content: c.as_ptr(),
            })
            .collect();

        let mut buf = vec![0u8; 4096];
        let needed = unsafe {
            ls_engine_apply_chat_template(
                self.raw.as_ptr(),
                raw_msgs.as_ptr(),
                raw_msgs.len(),
                add_assistant_prompt,
                buf.as_mut_ptr().cast::<c_char>(),
                buf.len() as i32,
            )
        };
        if needed < 0 {
            bail!("ls_engine_apply_chat_template failed (code={needed})");
        }
        let needed_usize = needed as usize;
        if needed_usize >= buf.len() {
            buf.resize(needed_usize + 1, 0);
            let n2 = unsafe {
                ls_engine_apply_chat_template(
                    self.raw.as_ptr(),
                    raw_msgs.as_ptr(),
                    raw_msgs.len(),
                    add_assistant_prompt,
                    buf.as_mut_ptr().cast::<c_char>(),
                    buf.len() as i32,
                )
            };
            if n2 < 0 {
                bail!("ls_engine_apply_chat_template resize failed (code={n2})");
            }
            buf.truncate(n2 as usize);
        } else {
            buf.truncate(needed_usize);
        }
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
        let bytes = text.as_bytes();
        let cap = (bytes.len() + 16).min(i32::MAX as usize) as i32;
        let mut out = vec![0i32; cap as usize];
        let n = unsafe {
            ls_engine_tokenize(
                self.raw.as_ptr(),
                bytes.as_ptr().cast::<c_char>(),
                bytes.len(),
                out.as_mut_ptr(),
                cap,
                add_special,
                parse_special,
            )
        };
        if n < 0 {
            let needed = (-n) as usize;
            out.resize(needed, 0);
            let n2 = unsafe {
                ls_engine_tokenize(
                    self.raw.as_ptr(),
                    bytes.as_ptr().cast::<c_char>(),
                    bytes.len(),
                    out.as_mut_ptr(),
                    needed as i32,
                    add_special,
                    parse_special,
                )
            };
            if n2 < 0 {
                bail!("llama_tokenize failed with code {n2}");
            }
            out.truncate(n2 as usize);
            return Ok(out);
        }
        out.truncate(n as usize);
        Ok(out)
    }

    pub fn token_to_piece(&self, token: i32) -> Result<String> {
        let mut buf = vec![0u8; 64];
        let n = unsafe {
            ls_engine_token_to_piece(
                self.raw.as_ptr(),
                token,
                buf.as_mut_ptr().cast::<c_char>(),
                buf.len() as i32,
            )
        };
        if n < 0 {
            let needed = (-n) as usize;
            buf.resize(needed, 0);
            let n2 = unsafe {
                ls_engine_token_to_piece(
                    self.raw.as_ptr(),
                    token,
                    buf.as_mut_ptr().cast::<c_char>(),
                    buf.len() as i32,
                )
            };
            if n2 < 0 {
                bail!("llama_token_to_piece failed for token {token}");
            }
            buf.truncate(n2 as usize);
        } else {
            buf.truncate(n as usize);
        }
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    pub fn token_is_eog(&self, token: i32) -> bool {
        unsafe { ls_engine_token_is_eog(self.raw.as_ptr(), token) }
    }

    pub fn kv_clear(&mut self) {
        self.cached_kv_tokens.clear();
        unsafe { ls_engine_kv_clear(self.raw.as_ptr()) };
    }

    pub fn kv_trim(&mut self, pos_min: i32) -> bool {
        let ok = unsafe { ls_engine_kv_trim(self.raw.as_ptr(), pos_min) };
        if ok && pos_min >= 0 {
            self.cached_kv_tokens.truncate(pos_min as usize);
        }
        ok
    }

    pub fn eval_greedy(&mut self, tokens: &[i32], pos_start: i32) -> Result<i32> {
        let mut next = 0i32;
        let mut err_buf = [0 as c_char; 512];
        let rc = unsafe {
            ls_engine_eval_greedy(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                pos_start,
                &mut next,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc == RC_CANCELLED {
            return Err(CancelledError.into());
        }
        if rc != 0 {
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("eval_greedy failed (rc={rc}): {msg}");
        }
        Ok(next)
    }

    pub fn eval_sample(
        &mut self,
        tokens: &[i32],
        pos_start: i32,
        sample: &LsSampleParams,
    ) -> Result<i32> {
        let mut next = 0i32;
        let mut err_buf = [0 as c_char; 512];
        let rc = unsafe {
            ls_engine_eval_sample(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                pos_start,
                sample,
                &mut next,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc == RC_CANCELLED {
            return Err(CancelledError.into());
        }
        if rc != 0 {
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("eval_sample failed (rc={rc}): {msg}");
        }
        Ok(next)
    }

    pub fn eval_verify_greedy(&mut self, tokens: &[i32], pos_start: i32) -> Result<Vec<i32>> {
        let mut out = vec![0i32; tokens.len()];
        let mut err_buf = [0 as c_char; 512];
        let rc = unsafe {
            ls_engine_eval_verify_greedy(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                pos_start,
                out.as_mut_ptr(),
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc == RC_CANCELLED {
            return Err(CancelledError.into());
        }
        if rc != 0 {
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("eval_verify_greedy failed (rc={rc}): {msg}");
        }
        Ok(out)
    }

    /// Lossless speculative rejection sampling verification for `temperature > 0`.
    /// Returns `(accepted_draft_count, next_token)`.
    pub fn eval_verify_sample(
        &mut self,
        tokens: &[i32],
        pos_start: i32,
        sample: &LsSampleParams,
    ) -> Result<(usize, i32)> {
        let mut accepted = 0i32;
        let mut next_tok = 0i32;
        let mut err_buf = [0 as c_char; 512];
        let rc = unsafe {
            ls_engine_eval_verify_sample(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                pos_start,
                sample,
                &mut accepted,
                &mut next_tok,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc == RC_CANCELLED {
            return Err(CancelledError.into());
        }
        if rc != 0 {
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("eval_verify_sample failed (rc={rc}): {msg}");
        }
        Ok((accepted.max(0) as usize, next_tok))
    }

    /// Runs chunked layer-major prefill followed by greedy decode (without speculation).
    pub fn generate_greedy(&mut self, prompt: &str, max_tokens: usize) -> Result<Vec<i32>> {
        let mut opts = GenerateOptions {
            max_tokens,
            enable_ngram_speculation: false,
            ..Default::default()
        };
        opts.sample.temperature = 0.0;
        let mut out = Vec::with_capacity(max_tokens);
        self.generate_stream(prompt, &opts, None, |tok, _piece| {
            out.push(tok);
            true
        })?;
        Ok(out)
    }

    /// Runs chunked layer-major prefill + streaming generation with optional n-gram
    /// and/or resident draft-model speculative verification.
    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        opts: &GenerateOptions,
        draft_proposer: Option<&mut DraftModelProposer>,
        on_token: F,
    ) -> Result<GenerateSummary>
    where
        F: FnMut(i32, &str) -> bool,
    {
        self.generate_stream_with_cancel(prompt, opts, draft_proposer, None, on_token)
    }

    /// Same as `generate_stream`, plus an optional cooperative cancellation flag that
    /// aborts prefill and mid-layer `llama_decode` passes immediately on client disconnect
    /// or process shutdown.
    pub fn generate_stream_with_cancel<F>(
        &mut self,
        prompt: &str,
        opts: &GenerateOptions,
        mut draft_proposer: Option<&mut DraftModelProposer>,
        cancel: Option<&AtomicBool>,
        mut on_token: F,
    ) -> Result<GenerateSummary>
    where
        F: FnMut(i32, &str) -> bool,
    {
        let is_cancelled = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));
        if is_cancelled() {
            return Ok(GenerateSummary {
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: "cancelled".to_string(),
                speculation: SpeculationStats::default(),
            });
        }

        let prompt_tokens = self.tokenize(prompt, true, true)?;
        if prompt_tokens.is_empty() {
            bail!("prompt tokenized to 0 tokens");
        }
        let n_ctx = self.n_ctx() as usize;
        if n_ctx > 0 && prompt_tokens.len() >= n_ctx {
            return Err(PromptTooLongError {
                prompt_tokens: prompt_tokens.len(),
                n_ctx,
            }
            .into());
        }

        // Reuse longest matching KV prefix on multi-turn chat continuations
        // (avoids re-prefilling unchanged system prompts & prior turns off NVMe!).
        let can_reuse_kv = self.stats().can_speculate;
        let mut common_prefix = 0usize;
        if can_reuse_kv && !self.cached_kv_tokens.is_empty() {
            let max_shared = self
                .cached_kv_tokens
                .len()
                .min(prompt_tokens.len().saturating_sub(1));
            while common_prefix < max_shared
                && self.cached_kv_tokens[common_prefix] == prompt_tokens[common_prefix]
            {
                common_prefix += 1;
            }
        }
        if common_prefix > 0 && self.kv_trim(common_prefix as i32) {
            tracing::debug!(common_prefix, "reusing cached KV prefix across turn");
        } else {
            self.kv_clear();
            common_prefix = 0;
        }
        // Truncate cached_kv_tokens BEFORE prefill so any mid-prefill error or cancellation
        // never leaves cached_kv_tokens ahead of actual KV cache state.
        self.cached_kv_tokens.truncate(common_prefix);

        if let Some(dp) = draft_proposer.as_deref_mut() {
            dp.trim_to(common_prefix);
        }

        let _cancel_guard = CancelCbGuard::install(self.raw, cancel);

        let mut history = prompt_tokens.clone();
        let mut pos = common_prefix as i32;
        let mut next_tok = 0i32;
        let mut spec_stats = SpeculationStats::default();

        for chunk in prompt_tokens[common_prefix..].chunks(self.n_batch) {
            if is_cancelled() {
                return Ok(GenerateSummary {
                    prompt_tokens: prompt_tokens.len(),
                    completion_tokens: 0,
                    finish_reason: "cancelled".to_string(),
                    speculation: spec_stats,
                });
            }
            let eval_res = if opts.sample.temperature <= 0.0 {
                self.eval_greedy(chunk, pos)
            } else {
                self.eval_sample(chunk, pos, &opts.sample)
            };
            match eval_res {
                Ok(tok) => {
                    next_tok = tok;
                    self.cached_kv_tokens.extend_from_slice(chunk);
                    spec_stats.sweeps += 1;
                    pos += chunk.len() as i32;
                }
                Err(e) if e.downcast_ref::<CancelledError>().is_some() => {
                    // Partial/aborted chunk may have touched KV cache; clear so next turn stays clean.
                    self.kv_clear();
                    return Ok(GenerateSummary {
                        prompt_tokens: prompt_tokens.len(),
                        completion_tokens: 0,
                        finish_reason: "cancelled".to_string(),
                        speculation: spec_stats,
                    });
                }
                Err(e) => {
                    self.kv_clear();
                    return Err(e);
                }
            }
        }

        let target_vocab = self.vocab_size();
        let max_gen_tokens = if n_ctx > 0 {
            opts.max_tokens
                .min(n_ctx.saturating_sub(prompt_tokens.len()))
        } else {
            opts.max_tokens
        };
        let can_speculate = self.stats().can_speculate
            && (opts.enable_ngram_speculation || draft_proposer.is_some());

        let min_adaptive_k = if draft_proposer.is_some() {
            opts.max_draft_tokens.clamp(6, 10)
        } else {
            3
        };
        let mut adaptive_draft_k = opts.max_draft_tokens.max(min_adaptive_k);
        let max_adaptive_k = if draft_proposer.is_some() {
            (opts.max_draft_tokens * 3).clamp(12, 20)
        } else {
            (opts.max_draft_tokens * 2).clamp(4, 12)
        };

        let mut completion_tokens = 0usize;
        let mut finish_reason = "length".to_string();
        let mut stop_filter = StopFilter::new(&opts.stop);
        let mut last_emitted_tok = next_tok;

        'gen_loop: while completion_tokens < max_gen_tokens {
            if is_cancelled() {
                finish_reason = "cancelled".to_string();
                break;
            }
            if self.token_is_eog(next_tok) {
                finish_reason = "stop".to_string();
                break;
            }

            // Emit `next_tok` BEFORE calling `draft_proposer.propose(...)` so draft latency
            // never pads TTFT, and route through `StopFilter` so stop text is never leaked.
            let piece = self.token_to_piece(next_tok)?;
            completion_tokens += 1;
            history.push(next_tok);
            last_emitted_tok = next_tok;
            let (safe_emit, hit_stop_seq) = stop_filter.push(&piece);
            if (!safe_emit.is_empty() || opts.stop.is_empty()) && !on_token(next_tok, &safe_emit) {
                finish_reason = "cancelled".to_string();
                break 'gen_loop;
            }
            if hit_stop_seq {
                finish_reason = "stop".to_string();
                break 'gen_loop;
            }
            if completion_tokens >= max_gen_tokens {
                finish_reason = "length".to_string();
                break 'gen_loop;
            }

            // Remaining budget for this pass: `pos` will hold `next_tok`, and `draft` tokens
            // occupy `pos + 1 .. pos + 1 + draft.len()`. Ensure `pos + 1 + draft.len() <= n_ctx`.
            let remaining_by_max = max_gen_tokens - completion_tokens;
            let remaining_by_ctx = if n_ctx > 0 {
                n_ctx.saturating_sub((pos as usize) + 1)
            } else {
                remaining_by_max
            };
            let max_step_draft = remaining_by_max.min(remaining_by_ctx);
            if n_ctx > 0 && (pos as usize) >= n_ctx {
                finish_reason = "length".to_string();
                break 'gen_loop;
            }

            // Speculative multi-token path (exact greedy or unbiased rejection sampling in 1 sweep).
            if can_speculate && max_step_draft > 0 {
                // Cascade: when a neural draft model is active, only let n-gram override it
                // on strong verbatim matches (min_n=4, e.g. code edits/copying) so noisy
                // 2-token prose hits never preempt the smarter draft model.
                let ngram = if draft_proposer.is_some() {
                    NgramProposer {
                        min_n: 4,
                        max_n: 6,
                        max_draft: adaptive_draft_k,
                    }
                } else {
                    NgramProposer {
                        max_draft: adaptive_draft_k,
                        ..Default::default()
                    }
                };
                let mut draft = if opts.enable_ngram_speculation {
                    ngram.propose(&history, max_step_draft)
                } else {
                    Vec::new()
                };
                if draft.len() < 2
                    && let Some(dp) = draft_proposer.as_deref_mut()
                {
                    draft = dp.propose(&history, max_step_draft.min(adaptive_draft_k))?;
                }
                // Guard against padded-vocab draft models proposing an out-of-range token ID.
                if let Some(bad_idx) = draft.iter().position(|&t| t < 0 || t >= target_vocab) {
                    draft.truncate(bad_idx);
                }

                if !draft.is_empty() {
                    let mut batch = Vec::with_capacity(1 + draft.len());
                    batch.push(next_tok);
                    batch.extend_from_slice(&draft);

                    let verify_res = if opts.sample.temperature <= 0.0 {
                        self.eval_verify_greedy(&batch, pos).map(|verified| {
                            let mut acc = 0usize;
                            let mut next_after = verified[draft.len()];
                            for (i, &d_tok) in draft.iter().enumerate() {
                                if verified[i] == d_tok {
                                    acc += 1;
                                } else {
                                    next_after = verified[i];
                                    break;
                                }
                            }
                            (acc, next_after)
                        })
                    } else {
                        let mut step_sample = opts.sample;
                        step_sample.seed = step_sample
                            .seed
                            .wrapping_add((completion_tokens as u32).wrapping_mul(2654435761));
                        self.eval_verify_sample(&batch, pos, &step_sample)
                    };

                    let (accepted_n, corrected_or_bonus) = match verify_res {
                        Ok(v) => v,
                        Err(e) if e.downcast_ref::<CancelledError>().is_some() => {
                            self.kv_clear();
                            finish_reason = "cancelled".to_string();
                            break 'gen_loop;
                        }
                        Err(e) => {
                            self.kv_clear();
                            return Err(e);
                        }
                    };
                    spec_stats.sweeps += 1;
                    spec_stats.drafted_tokens += draft.len();

                    let mut accepted = 0usize;
                    let mut early_stop = false;
                    for &d_tok in draft.iter().take(accepted_n) {
                        accepted += 1;
                        spec_stats.accepted_tokens += 1;
                        if self.token_is_eog(d_tok) {
                            finish_reason = "stop".to_string();
                            early_stop = true;
                            break;
                        }
                        let d_piece = self.token_to_piece(d_tok)?;
                        completion_tokens += 1;
                        history.push(d_tok);
                        last_emitted_tok = d_tok;
                        let (safe_emit, hit_stop_seq) = stop_filter.push(&d_piece);
                        if (!safe_emit.is_empty() || opts.stop.is_empty())
                            && !on_token(d_tok, &safe_emit)
                        {
                            finish_reason = "cancelled".to_string();
                            early_stop = true;
                            break;
                        }
                        if hit_stop_seq {
                            finish_reason = "stop".to_string();
                            early_stop = true;
                            break;
                        }
                        if completion_tokens >= max_gen_tokens {
                            finish_reason = "length".to_string();
                            early_stop = true;
                            break;
                        }
                    }

                    // Always trim any unaccepted or post-stop tail from KV cache and verify `kv_trim`.
                    let valid_pos = pos + 1 + accepted as i32;
                    if accepted < draft.len() {
                        if !self.kv_trim(valid_pos) {
                            self.kv_clear();
                            bail!(
                                "kv_trim({valid_pos}) failed after speculative verification; \
                                 model memory does not support partial rollback"
                            );
                        }
                        if self.cached_kv_tokens.len() <= history.len() {
                            self.cached_kv_tokens
                                .extend_from_slice(&history[self.cached_kv_tokens.len()..]);
                        } else {
                            self.cached_kv_tokens = history.clone();
                        }
                        pos = valid_pos;
                        next_tok = corrected_or_bonus;
                    } else {
                        if self.cached_kv_tokens.len() <= history.len() {
                            self.cached_kv_tokens
                                .extend_from_slice(&history[self.cached_kv_tokens.len()..]);
                        } else {
                            self.cached_kv_tokens = history.clone();
                        }
                        pos += 1 + draft.len() as i32;
                        next_tok = corrected_or_bonus;
                    }

                    if early_stop {
                        break 'gen_loop;
                    }

                    // Adapt draft depth upward aggressively on full acceptance streaks
                    // (verifying extra tokens is nearly free compared to an NVMe sweep),
                    // and gently downward on misses without collapsing below `min_adaptive_k`.
                    if accepted == draft.len() {
                        let step = if draft_proposer.is_some() { 4 } else { 2 };
                        adaptive_draft_k = (adaptive_draft_k + step).min(max_adaptive_k);
                    } else if accepted == 0 {
                        adaptive_draft_k = (adaptive_draft_k.saturating_sub(1)).max(min_adaptive_k);
                    }
                    continue 'gen_loop;
                }
            }

            // Standard 1-token sweep fallback.
            let mut step_sample = opts.sample;
            step_sample.seed = step_sample.seed.wrapping_add(completion_tokens as u32);
            let eval_res = if step_sample.temperature <= 0.0 {
                self.eval_greedy(&[next_tok], pos)
            } else {
                self.eval_sample(&[next_tok], pos, &step_sample)
            };
            match eval_res {
                Ok(tok) => {
                    next_tok = tok;
                    if self.cached_kv_tokens.len() <= history.len() {
                        self.cached_kv_tokens
                            .extend_from_slice(&history[self.cached_kv_tokens.len()..]);
                    } else {
                        self.cached_kv_tokens = history.clone();
                    }
                    spec_stats.sweeps += 1;
                    pos += 1;
                }
                Err(e) if e.downcast_ref::<CancelledError>().is_some() => {
                    self.kv_clear();
                    finish_reason = "cancelled".to_string();
                    break 'gen_loop;
                }
                Err(e) => {
                    self.kv_clear();
                    return Err(e);
                }
            }
        }

        // Flush any safe held-back suffix bytes when generation ended without hitting a stop sequence.
        if finish_reason != "cancelled" {
            let tail = stop_filter.flush();
            if !tail.is_empty() && !on_token(last_emitted_tok, &tail) {
                finish_reason = "cancelled".to_string();
            }
        }

        Ok(GenerateSummary {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            finish_reason,
            speculation: spec_stats,
        })
    }
}

struct CancelCbGuard {
    raw: NonNull<LsEngine>,
    installed: bool,
}

impl CancelCbGuard {
    fn install(raw: NonNull<LsEngine>, cancel: Option<&AtomicBool>) -> Self {
        if let Some(flag) = cancel {
            unsafe {
                ls_engine_set_cancel_cb(
                    raw.as_ptr(),
                    Some(rust_cancel_check_cb),
                    (flag as *const AtomicBool).cast::<c_void>(),
                );
            }
            Self {
                raw,
                installed: true,
            }
        } else {
            Self {
                raw,
                installed: false,
            }
        }
    }
}

impl Drop for CancelCbGuard {
    fn drop(&mut self) {
        if self.installed {
            unsafe { ls_engine_set_cancel_cb(self.raw.as_ptr(), None, std::ptr::null()) };
        }
    }
}

unsafe extern "C" fn rust_cancel_check_cb(user_data: *const c_void) -> bool {
    if user_data.is_null() {
        return false;
    }
    let flag = unsafe { &*(user_data as *const AtomicBool) };
    flag.load(Ordering::Relaxed)
}

/// Sliding UTF-8-safe holdback buffer for OpenAI `stop` sequences.
/// Guarantees stop sequences are never emitted to streaming clients (even when split across
/// multiple tokens) and catches mid-piece matches (e.g. `"\n\nUser:"` with `stop: ["\n\n"]`).
pub struct StopFilter {
    stops: Vec<String>,
    max_stop_bytes: usize,
    pending: String,
}

impl StopFilter {
    pub fn new(stops: &[String]) -> Self {
        let stops: Vec<String> = stops.iter().filter(|s| !s.is_empty()).cloned().collect();
        let max_stop_bytes = stops.iter().map(String::len).max().unwrap_or(0);
        Self {
            stops,
            max_stop_bytes,
            pending: String::new(),
        }
    }

    /// Pushes a new piece and returns `(safe_prefix_to_emit, matched_stop)`.
    pub fn push(&mut self, piece: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (piece.to_string(), false);
        }
        self.pending.push_str(piece);

        let mut earliest_hit: Option<usize> = None;
        for stop in &self.stops {
            if let Some(idx) = self.pending.find(stop.as_str()) {
                earliest_hit = Some(earliest_hit.map_or(idx, |prev| prev.min(idx)));
            }
        }
        if let Some(hit_idx) = earliest_hit {
            let emit = self.pending[..hit_idx].to_string();
            self.pending.clear();
            return (emit, true);
        }

        let keep = self.max_stop_bytes.saturating_sub(1);
        if self.pending.len() <= keep {
            return (String::new(), false);
        }
        let mut split = self.pending.len() - keep;
        while split > 0 && !self.pending.is_char_boundary(split) {
            split -= 1;
        }
        if split == 0 {
            return (String::new(), false);
        }
        let emit = self.pending[..split].to_string();
        self.pending.drain(..split);
        (emit, false)
    }

    /// Flushes any remaining buffered bytes at natural end-of-generation (non-stop).
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

impl Drop for StreamEngine {
    fn drop(&mut self) {
        // Free C++ engine FIRST so `worker_thread` is joined before freeing `LayerIoPool`.
        unsafe { ls_engine_free(self.raw.as_ptr()) };
        if let Some(p) = self.io_pool.take() {
            drop(unsafe { Box::from_raw(p) });
        }
    }
}

unsafe extern "C" fn rust_load_tensors_cb(
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
        let pool = unsafe { &*(user_data as *const LayerIoPool) };
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
            let c_name = unsafe { *tensor_names.add(i) };
            let Ok(name) = unsafe { CStr::from_ptr(c_name) }.to_str() else {
                return -2;
            };
            let off = unsafe { *tensor_offsets.add(i) } as usize;
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
unsafe extern "C" fn rust_load_layer_direct_cb(
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
        let pool = unsafe { &*(user_data as *const LayerIoPool) };
        let pin_buf = unsafe { std::slice::from_raw_parts_mut(pin_ptr, pin_cap) };
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
            let c_name = unsafe { *tensor_names.add(i) };
            let Ok(name) = unsafe { CStr::from_ptr(c_name) }.to_str() else {
                return -2;
            };
            let off = unsafe { *tensor_offsets.add(i) } as usize;
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

unsafe extern "C" fn rust_load_slices_cb(
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
        let pool = unsafe { &*(user_data as *const LayerIoPool) };
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, dst_cap) };
        let raw_reqs = if req_count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(reqs_ptr, req_count) }
        };

        let mut requests = Vec::with_capacity(req_count);
        for r in raw_reqs {
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
