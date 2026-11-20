use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use anyhow::{Context, Result, bail};

use crate::ffi::{
    LsChatMessage, LsEngine, LsEngineConfig, LsEngineStats, LsSampleParams,
    ls_engine_apply_chat_template, ls_engine_create, ls_engine_eval_greedy, ls_engine_eval_sample,
    ls_engine_eval_verify_greedy, ls_engine_free, ls_engine_get_stats, ls_engine_kv_clear,
    ls_engine_kv_trim, ls_engine_token_is_eog, ls_engine_token_to_piece, ls_engine_tokenize,
};
use crate::gguf_index::GgufIndex;
use crate::io_reader::{IoStats, LayerIoPool};
use crate::speculator::{DraftModelProposer, NgramProposer, SpeculationStats};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KvCacheType {
    #[default]
    F16 = 0,
    Q8_0 = 1,
    Q4_0 = 2,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_path: PathBuf,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub resident_layers: Option<u32>,
    pub vram_budget_bytes: u64,
    pub kv_cache_type: KvCacheType,
    pub use_gpu: bool,
    pub prefer_direct_io: bool,
    pub full_resident_reference: bool,
}

impl EngineConfig {
    pub fn streaming(model_path: impl AsRef<Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            n_ctx: 4096,
            n_batch: 512,
            n_threads: std::thread::available_parallelism()
                .map(|n| n.get().min(8) as u32)
                .unwrap_or(4),
            resident_layers: None,
            vram_budget_bytes: 0,
            kv_cache_type: KvCacheType::F16,
            use_gpu: true,
            prefer_direct_io: true,
            full_resident_reference: false,
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
    io_pool: Option<Box<LayerIoPool>>,
    n_batch: usize,
    cached_kv_tokens: Vec<i32>,
}

unsafe impl Send for StreamEngine {}

impl StreamEngine {
    pub fn open(cfg: &EngineConfig) -> Result<Self> {
        let c_path = CString::new(cfg.model_path.to_string_lossy().as_bytes())
            .context("model_path contains interior NUL")?;

        let mut io_pool = if cfg.full_resident_reference {
            None
        } else {
            let index = GgufIndex::open(&cfg.model_path)?;
            Some(Box::new(LayerIoPool::new(index, cfg.prefer_direct_io)?))
        };

        let io_ptr = io_pool
            .as_mut()
            .map(|p| (&mut **p as *mut LayerIoPool).cast::<c_void>())
            .unwrap_or(std::ptr::null_mut());

        let raw_cfg = LsEngineConfig {
            model_path: c_path.as_ptr(),
            n_ctx: cfg.n_ctx,
            n_batch: cfg.n_batch,
            n_threads: cfg.n_threads,
            resident_layers: cfg.resident_layers.map_or(-1, |v| v as i32),
            vram_budget_bytes: cfg.vram_budget_bytes,
            kv_cache_type: cfg.kv_cache_type as i32,
            use_gpu: cfg.use_gpu,
            full_resident_reference: cfg.full_resident_reference,
            io_user_data: io_ptr,
            load_tensors_cb: if io_pool.is_some() {
                Some(rust_load_tensors_cb)
            } else {
                None
            },
            prefetch_start_cb: None,
            prefetch_wait_cb: None,
        };

        let mut err_buf = [0 as c_char; 512];
        let ptr = unsafe { ls_engine_create(&raw_cfg, err_buf.as_mut_ptr(), err_buf.len()) };
        let Some(raw) = NonNull::new(ptr) else {
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("failed to create layerstream engine: {msg}");
        };

        Ok(Self {
            raw,
            io_pool,
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
        self.io_pool.as_ref().map(|p| p.stats()).unwrap_or_default()
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
        if rc != 0 {
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("eval_verify_greedy failed (rc={rc}): {msg}");
        }
        Ok(out)
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
    ///
    /// `on_token(token_id, utf8_piece)` returns `false` to cancel generation immediately.
    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        opts: &GenerateOptions,
        mut draft_proposer: Option<&mut DraftModelProposer>,
        mut on_token: F,
    ) -> Result<GenerateSummary>
    where
        F: FnMut(i32, &str) -> bool,
    {
        let prompt_tokens = self.tokenize(prompt, true, true)?;
        if prompt_tokens.is_empty() {
            bail!("prompt tokenized to 0 tokens");
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
        if let Some(dp) = draft_proposer.as_deref_mut() {
            dp.reset();
        }

        let mut history = prompt_tokens.clone();
        let mut pos = common_prefix as i32;
        let mut next_tok = 0i32;
        let mut spec_stats = SpeculationStats::default();

        for chunk in prompt_tokens[common_prefix..].chunks(self.n_batch) {
            next_tok = if opts.sample.temperature <= 0.0 {
                self.eval_greedy(chunk, pos)?
            } else {
                self.eval_sample(chunk, pos, &opts.sample)?
            };
            spec_stats.sweeps += 1;
            pos += chunk.len() as i32;
        }
        self.cached_kv_tokens = prompt_tokens.clone();

        let can_speculate = self.stats().can_speculate
            && opts.sample.temperature <= 0.0
            && (opts.enable_ngram_speculation || draft_proposer.is_some());

        let mut adaptive_draft_k = opts.max_draft_tokens.max(1);
        let max_adaptive_k = (opts.max_draft_tokens * 2).clamp(4, 12);

        let mut completion_tokens = 0usize;
        let mut finish_reason = "length".to_string();
        let mut accumulated_text = String::new();

        'gen_loop: while completion_tokens < opts.max_tokens {
            if self.token_is_eog(next_tok) {
                finish_reason = "stop".to_string();
                break;
            }

            let remaining_after_current = opts.max_tokens - completion_tokens - 1;

            // Speculative multi-token path (exact greedy verification in 1 NVMe sweep).
            if can_speculate && remaining_after_current > 0 {
                let mut probe_history = history.clone();
                probe_history.push(next_tok);

                // Cascade: try zero-VRAM n-gram first (0 ms) on strong hits (>= 2 tokens),
                // then fall back to resident `--draft-model` with adaptive draft depth.
                let ngram = NgramProposer {
                    max_draft: adaptive_draft_k,
                    ..Default::default()
                };
                let mut draft = if opts.enable_ngram_speculation {
                    ngram.propose(&probe_history, remaining_after_current)
                } else {
                    Vec::new()
                };
                if draft.len() < 2
                    && let Some(dp) = draft_proposer.as_deref_mut()
                {
                    draft = dp.propose(
                        &probe_history,
                        remaining_after_current.min(adaptive_draft_k),
                    )?;
                }

                if !draft.is_empty() {
                    // Emit `next_tok` first (already verified by previous sweep).
                    let piece = self.token_to_piece(next_tok)?;
                    accumulated_text.push_str(&piece);
                    completion_tokens += 1;
                    history.push(next_tok);
                    if !on_token(next_tok, &piece) {
                        finish_reason = "cancelled".to_string();
                        break 'gen_loop;
                    }
                    if hit_stop(&accumulated_text, &opts.stop) {
                        finish_reason = "stop".to_string();
                        break 'gen_loop;
                    }

                    let mut batch = Vec::with_capacity(1 + draft.len());
                    batch.push(next_tok);
                    batch.extend_from_slice(&draft);

                    let verified = self.eval_verify_greedy(&batch, pos)?;
                    spec_stats.sweeps += 1;
                    spec_stats.drafted_tokens += draft.len();

                    let mut accepted = 0usize;
                    let mut diverged_next: Option<i32> = None;

                    for (i, &d_tok) in draft.iter().enumerate() {
                        let v_tok = verified[i];
                        if v_tok == d_tok {
                            accepted += 1;
                            if self.token_is_eog(d_tok) {
                                finish_reason = "stop".to_string();
                                break 'gen_loop;
                            }
                            let d_piece = self.token_to_piece(d_tok)?;
                            accumulated_text.push_str(&d_piece);
                            completion_tokens += 1;
                            history.push(d_tok);
                            if !on_token(d_tok, &d_piece) {
                                finish_reason = "cancelled".to_string();
                                break 'gen_loop;
                            }
                            if hit_stop(&accumulated_text, &opts.stop) {
                                finish_reason = "stop".to_string();
                                break 'gen_loop;
                            }
                            if completion_tokens >= opts.max_tokens {
                                finish_reason = "length".to_string();
                                break 'gen_loop;
                            }
                        } else {
                            diverged_next = Some(v_tok);
                            break;
                        }
                    }

                    spec_stats.accepted_tokens += accepted;
                    self.cached_kv_tokens = history.clone();

                    // Adapt draft depth upward on full acceptance streaks, downward on misses.
                    if accepted == draft.len() {
                        adaptive_draft_k = (adaptive_draft_k + 2).min(max_adaptive_k);
                    } else if accepted == 0 {
                        adaptive_draft_k = (adaptive_draft_k.saturating_sub(1)).max(3);
                    }

                    if let Some(corrected) = diverged_next {
                        // Trim rejected draft tail from KV cache.
                        let valid_pos = pos + 1 + accepted as i32;
                        self.kv_trim(valid_pos);
                        pos = valid_pos;
                        next_tok = corrected;
                    } else {
                        // All draft tokens accepted: bonus token is verified[draft.len()].
                        pos += 1 + draft.len() as i32;
                        next_tok = verified[draft.len()];
                    }
                    continue 'gen_loop;
                }
            }

            // Standard 1-token sweep fallback.
            let piece = self.token_to_piece(next_tok)?;
            accumulated_text.push_str(&piece);
            completion_tokens += 1;
            history.push(next_tok);
            if !on_token(next_tok, &piece) {
                finish_reason = "cancelled".to_string();
                break;
            }
            if hit_stop(&accumulated_text, &opts.stop) {
                finish_reason = "stop".to_string();
                break;
            }
            if completion_tokens >= opts.max_tokens {
                break;
            }

            let mut step_sample = opts.sample;
            step_sample.seed = step_sample.seed.wrapping_add(completion_tokens as u32);
            next_tok = if step_sample.temperature <= 0.0 {
                self.eval_greedy(&[next_tok], pos)?
            } else {
                self.eval_sample(&[next_tok], pos, &step_sample)?
            };
            self.cached_kv_tokens = history.clone();
            spec_stats.sweeps += 1;
            pos += 1;
        }

        Ok(GenerateSummary {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            finish_reason,
            speculation: spec_stats,
        })
    }
}

fn hit_stop(text: &str, stops: &[String]) -> bool {
    stops.iter().any(|s| !s.is_empty() && text.ends_with(s))
}

impl Drop for StreamEngine {
    fn drop(&mut self) {
        unsafe { ls_engine_free(self.raw.as_ptr()) };
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
}
