use super::*;

impl StreamEngine {
    pub fn eval_greedy(&mut self, tokens: &[i32], pos_start: i32) -> Result<i32> {
        let mut next = 0i32;
        let mut err_buf = [0 as c_char; 512];
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
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
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
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
        self.eval_sample_with_history(tokens, pos_start, sample, &[])
    }

    pub fn eval_sample_with_history(
        &mut self,
        tokens: &[i32],
        pos_start: i32,
        sample: &LsSampleParams,
        penalty_history: &[i32],
    ) -> Result<i32> {
        let mut next = 0i32;
        let mut err_buf = [0 as c_char; 512];
        let (pen_ptr, pen_len) = if penalty_history.is_empty() {
            (std::ptr::null(), 0)
        } else {
            (
                penalty_history.as_ptr(),
                penalty_history.len().min(i32::MAX as usize) as i32,
            )
        };
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let rc = unsafe {
            ls_engine_eval_sample(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                pos_start,
                sample,
                pen_ptr,
                pen_len,
                &mut next,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc == RC_CANCELLED {
            return Err(CancelledError.into());
        }
        if rc != 0 {
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
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
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
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
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("eval_verify_greedy failed (rc={rc}): {msg}");
        }
        Ok(out)
    }

    /// Lossless speculative rejection sampling verification for `temperature > 0` (or penalized greedy).
    /// Returns `(accepted_draft_count, next_token)`.
    pub fn eval_verify_sample(
        &mut self,
        tokens: &[i32],
        pos_start: i32,
        sample: &LsSampleParams,
    ) -> Result<(usize, i32)> {
        self.eval_verify_sample_with_history(tokens, pos_start, sample, &[])
    }

    pub fn eval_verify_sample_with_history(
        &mut self,
        tokens: &[i32],
        pos_start: i32,
        sample: &LsSampleParams,
        penalty_history: &[i32],
    ) -> Result<(usize, i32)> {
        let mut accepted = 0i32;
        let mut next_tok = 0i32;
        let mut err_buf = [0 as c_char; 512];
        let (pen_ptr, pen_len) = if penalty_history.is_empty() {
            (std::ptr::null(), 0)
        } else {
            (
                penalty_history.as_ptr(),
                penalty_history.len().min(i32::MAX as usize) as i32,
            )
        };
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let rc = unsafe {
            ls_engine_eval_verify_sample(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                pos_start,
                sample,
                pen_ptr,
                pen_len,
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
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("eval_verify_sample failed (rc={rc}): {msg}");
        }
        Ok((accepted.max(0) as usize, next_tok))
    }

    fn eval_step_with_grammar(
        &mut self,
        tokens: &[i32],
        pos_start: i32,
        sample: &LsSampleParams,
        penalty_history: &[i32],
        grammar: Option<&GrammarHandle>,
    ) -> Result<i32> {
        let Some(g) = grammar else {
            return if sample.temperature <= 0.0 && !sample.has_penalties() {
                self.eval_greedy(tokens, pos_start)
            } else {
                self.eval_sample_with_history(tokens, pos_start, sample, penalty_history)
            };
        };
        let mut next = 0i32;
        let mut err_buf = [0 as c_char; 512];
        let (pen_ptr, pen_len) = if penalty_history.is_empty() {
            (std::ptr::null(), 0)
        } else {
            (
                penalty_history.as_ptr(),
                penalty_history.len().min(i32::MAX as usize) as i32,
            )
        };
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let rc = unsafe {
            ls_engine_eval_with_grammar(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                pos_start,
                sample,
                pen_ptr,
                pen_len,
                g.raw.as_ptr(),
                &mut next,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc == RC_CANCELLED {
            return Err(CancelledError.into());
        }
        if rc != 0 {
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("eval_with_grammar failed (rc={rc}): {msg}");
        }
        Ok(next)
    }

    fn eval_verify_with_grammar(
        &mut self,
        tokens: &[i32],
        pos_start: i32,
        sample: &LsSampleParams,
        penalty_history: &[i32],
        grammar: &GrammarHandle,
    ) -> Result<(usize, i32)> {
        let mut accepted = 0i32;
        let mut next_tok = 0i32;
        let mut err_buf = [0 as c_char; 512];
        let (pen_ptr, pen_len) = if penalty_history.is_empty() {
            (std::ptr::null(), 0)
        } else {
            (
                penalty_history.as_ptr(),
                penalty_history.len().min(i32::MAX as usize) as i32,
            )
        };
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let rc = unsafe {
            ls_engine_eval_verify_with_grammar(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                pos_start,
                sample,
                pen_ptr,
                pen_len,
                grammar.raw.as_ptr(),
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
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("eval_verify_with_grammar failed (rc={rc}): {msg}");
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

        let mut grammar = match opts.grammar.as_deref() {
            None => None,
            Some(raw) => {
                let g_str = raw.trim();
                if g_str.is_empty() {
                    bail!("grammar string must not be empty");
                }
                Some(GrammarHandle::create(self.raw, g_str)?)
            }
        };

        // Reuse longest matching KV prefix on multi-turn chat continuations
        // (avoids re-prefilling unchanged system prompts & prior turns off NVMe!).
        let st = self.stats();
        let can_reuse_kv = st.can_speculate && st.rs_rollback_max == 0;
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

        let prefill_slice = &prompt_tokens[common_prefix..];
        let total_prefill_chunks = prefill_slice.chunks(self.n_batch).len();
        for (chunk_idx, chunk) in prefill_slice.chunks(self.n_batch).enumerate() {
            if is_cancelled() {
                return Ok(GenerateSummary {
                    prompt_tokens: prompt_tokens.len(),
                    completion_tokens: 0,
                    finish_reason: "cancelled".to_string(),
                    speculation: spec_stats,
                });
            }
            let is_last_prefill = chunk_idx + 1 == total_prefill_chunks;
            let eval_res = if is_last_prefill {
                self.eval_step_with_grammar(chunk, pos, &opts.sample, &history, grammar.as_ref())
            } else {
                // Intermediate prefill chunks discard their sampled output token, so run pure greedy.
                self.eval_greedy(chunk, pos)
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
        let has_mtp = st.mtp_layers > 0;
        let rs_rollback_cap = if st.rs_rollback_max > 0 {
            st.rs_rollback_max as usize
        } else {
            usize::MAX
        };
        let can_speculate = st.can_speculate
            && (opts.enable_ngram_speculation || draft_proposer.is_some() || has_mtp);

        let min_adaptive_k = if draft_proposer.is_some() || has_mtp {
            opts.max_draft_tokens.clamp(6, 10)
        } else {
            3
        }
        .min(rs_rollback_cap);
        let mut adaptive_draft_k = opts
            .max_draft_tokens
            .max(min_adaptive_k)
            .min(rs_rollback_cap);
        let max_adaptive_k = if draft_proposer.is_some() {
            (opts.max_draft_tokens * 3).clamp(12, 20)
        } else {
            (opts.max_draft_tokens * 2).clamp(4, 12)
        };

        let mut completion_tokens = 0usize;
        let mut finish_reason = "length".to_string();
        let mut utf8_assembler = Utf8PieceAssembler::default();
        let mut stop_filter = StopFilter::new(&opts.stop);
        let mut hit_stop_any = false;
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
            if let Some(ref mut g) = grammar {
                g.accept(next_tok)?;
            }

            // Emit `next_tok` BEFORE calling `draft_proposer.propose(...)` so draft latency
            // never pads TTFT, buffering incomplete multi-byte UTF-8 sequences across tokens
            // and routing through `StopFilter` so stop text is never leaked.
            let raw_bytes = self.token_to_piece_bytes(next_tok)?;
            let piece = utf8_assembler.push(&raw_bytes);
            completion_tokens += 1;
            history.push(next_tok);
            last_emitted_tok = next_tok;
            let (safe_emit, hit_stop_seq) = stop_filter.push(&piece);
            if (!safe_emit.is_empty()
                || (opts.stop.is_empty() && piece.is_empty() && raw_bytes.is_empty()))
                && !on_token(next_tok, &safe_emit)
            {
                finish_reason = "cancelled".to_string();
                break 'gen_loop;
            }
            if hit_stop_seq {
                hit_stop_any = true;
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
            let max_step_draft = remaining_by_max.min(remaining_by_ctx).min(rs_rollback_cap);
            if n_ctx > 0 && (pos as usize) >= n_ctx {
                finish_reason = "length".to_string();
                break 'gen_loop;
            }

            // Speculative multi-token path (exact greedy or unbiased rejection sampling in 1 sweep).
            if can_speculate && max_step_draft > 0 {
                // Cascade: when a neural draft model is active, only let n-gram override it
                // on strong verbatim matches (min_n=4, e.g. code edits/copying) so noisy
                // 2-token prose hits never preempt the smarter draft model.
                let ngram = if draft_proposer.is_some() || has_mtp {
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
                if draft.len() < 2 {
                    if let Some(dp) = draft_proposer.as_deref_mut() {
                        draft = dp.propose_with_cancel(
                            &history,
                            max_step_draft.min(adaptive_draft_k),
                            cancel,
                        )?;
                        if is_cancelled() {
                            finish_reason = "cancelled".to_string();
                            break;
                        }
                    } else if has_mtp {
                        let io = self.io_stats();
                        let mtp_cap = effective_mtp_draft_cap(
                            max_step_draft.min(adaptive_draft_k),
                            self.stats().moe_layers,
                            io.cache_hits,
                            io.cache_misses,
                        );
                        draft = if mtp_cap > 0 {
                            self.mtp_draft(next_tok, pos, mtp_cap)?
                        } else {
                            Vec::new()
                        };
                        if is_cancelled() {
                            finish_reason = "cancelled".to_string();
                            break;
                        }
                    }
                }
                // Guard against padded-vocab draft models proposing an out-of-range token ID.
                if let Some(bad_idx) = draft.iter().position(|&t| t < 0 || t >= target_vocab) {
                    draft.truncate(bad_idx);
                }

                if !draft.is_empty() {
                    let mut batch = Vec::with_capacity(1 + draft.len());
                    batch.push(next_tok);
                    batch.extend_from_slice(&draft);

                    let mut step_sample = opts.sample;
                    if step_sample.temperature > 0.0 {
                        step_sample.seed = step_sample
                            .seed
                            .wrapping_add((completion_tokens as u32).wrapping_mul(2654435761));
                    }

                    let verify_res = if let Some(ref g) = grammar {
                        self.eval_verify_with_grammar(&batch, pos, &step_sample, &history, g)
                    } else if step_sample.temperature <= 0.0 && !step_sample.has_penalties() {
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
                        self.eval_verify_sample_with_history(&batch, pos, &step_sample, &history)
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
                        spec_stats.accepted_tokens += 1;
                        if self.token_is_eog(d_tok) {
                            finish_reason = "stop".to_string();
                            early_stop = true;
                            break;
                        }
                        accepted += 1;
                        if let Some(ref mut g) = grammar {
                            g.accept(d_tok)?;
                        }
                        let d_bytes = self.token_to_piece_bytes(d_tok)?;
                        let d_piece = utf8_assembler.push(&d_bytes);
                        completion_tokens += 1;
                        history.push(d_tok);
                        last_emitted_tok = d_tok;
                        let (safe_emit, hit_stop_seq) = stop_filter.push(&d_piece);
                        if (!safe_emit.is_empty()
                            || (opts.stop.is_empty() && d_piece.is_empty() && d_bytes.is_empty()))
                            && !on_token(d_tok, &safe_emit)
                        {
                            finish_reason = "cancelled".to_string();
                            early_stop = true;
                            break;
                        }
                        if hit_stop_seq {
                            hit_stop_any = true;
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

                    if has_mtp {
                        self.mtp_accept(accepted);
                    }
                    if early_stop {
                        break 'gen_loop;
                    }

                    // Adapt draft depth upward aggressively on full acceptance streaks
                    // (verifying extra tokens is nearly free compared to an NVMe sweep),
                    // and gently downward on misses without collapsing below `min_adaptive_k`.
                    if accepted == draft.len() {
                        let step = if draft_proposer.is_some() || has_mtp {
                            4
                        } else {
                            2
                        };
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
            let eval_res = self.eval_step_with_grammar(
                &[next_tok],
                pos,
                &step_sample,
                &history,
                grammar.as_ref(),
            );
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

        // Flush any trailing incomplete UTF-8 bytes and held-back stop-filter suffix bytes
        // when generation ended without cancellation or an already-matched stop sequence.
        if finish_reason != "cancelled" && !hit_stop_any {
            let utf8_tail = utf8_assembler.flush();
            if !utf8_tail.is_empty() {
                let (safe_emit, hit_stop_seq) = stop_filter.push(&utf8_tail);
                if !safe_emit.is_empty() && !on_token(last_emitted_tok, &safe_emit) {
                    finish_reason = "cancelled".to_string();
                } else if hit_stop_seq {
                    hit_stop_any = true;
                    finish_reason = "stop".to_string();
                }
            }
        }
        if finish_reason != "cancelled" && !hit_stop_any {
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

struct GrammarHandle {
    raw: NonNull<LsGrammar>,
}

impl GrammarHandle {
    fn create(eng: NonNull<LsEngine>, grammar_str: &str) -> Result<Self> {
        let c_grammar =
            CString::new(grammar_str).context("GBNF grammar contains interior NUL byte")?;
        let mut err_buf = [0 as c_char; 512];
        // SAFETY: `eng` is a live `LsEngine`, `c_grammar` and `c"root"` are NUL-terminated, and `err_buf` is valid for its length.
        let ptr = unsafe {
            ls_grammar_create(
                eng.as_ptr(),
                c_grammar.as_ptr(),
                c"root".as_ptr(),
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        let Some(raw) = NonNull::new(ptr) else {
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("invalid grammar: {msg}");
        };
        Ok(Self { raw })
    }

    fn accept(&mut self, token: i32) -> Result<()> {
        let mut err_buf = [0 as c_char; 512];
        // SAFETY: `self.raw` is a live `LsEngine` handle and all passed pointers/slices remain valid for the duration of the FFI call.
        let rc = unsafe {
            ls_grammar_accept(
                self.raw.as_ptr(),
                token,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc != 0 {
            // SAFETY: C++ FFI wrote a NUL-terminated string into `err_buf` / `name_buf` or passed a valid C string pointer.
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("grammar state error on token {token}: {msg}");
        }
        Ok(())
    }
}

impl Drop for GrammarHandle {
    fn drop(&mut self) {
        // SAFETY: `self.raw` is a valid grammar handle owned exclusively by `self` and freed once on drop.
        unsafe { ls_grammar_free(self.raw.as_ptr()) };
    }
}

pub(crate) struct CancelCbGuard<'a> {
    raw: NonNull<LsEngine>,
    installed: bool,
    _flag: std::marker::PhantomData<Option<&'a AtomicBool>>,
}

impl StreamEngine {
    pub(crate) fn install_cancel_cb<'a>(
        &self,
        cancel: Option<&'a AtomicBool>,
    ) -> CancelCbGuard<'a> {
        CancelCbGuard::install(self.raw, cancel)
    }
}

impl<'a> CancelCbGuard<'a> {
    fn install(raw: NonNull<LsEngine>, cancel: Option<&'a AtomicBool>) -> Self {
        if let Some(flag) = cancel {
            // SAFETY: `self.raw` is a live `LsEngine`, and `CancelCbGuard` clears the callback before the `AtomicBool` borrow ends.
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
                _flag: std::marker::PhantomData,
            }
        } else {
            Self {
                raw,
                installed: false,
                _flag: std::marker::PhantomData,
            }
        }
    }
}

impl Drop for CancelCbGuard<'_> {
    fn drop(&mut self) {
        if self.installed {
            // SAFETY: `self.raw` is a live `LsEngine`, and `CancelCbGuard` clears the callback before the `AtomicBool` borrow ends.
            unsafe { ls_engine_set_cancel_cb(self.raw.as_ptr(), None, std::ptr::null()) };
        }
    }
}

unsafe extern "C" fn rust_cancel_check_cb(user_data: *const c_void) -> bool {
    if user_data.is_null() {
        return false;
    }
    // SAFETY: `user_data` was registered by `CancelCbGuard` pointing to a live `AtomicBool` for the duration of the C++ call.
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
