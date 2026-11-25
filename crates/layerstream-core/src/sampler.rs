//! Rust implementation of token sampling, speculative rejection verification,
//! and GBNF grammar-constrained decoding (migrated from `cpp/sampler_eval.cpp`).

use std::cell::RefCell;
use std::ffi::{CStr, c_char, c_int};
use std::ptr;

use crate::ffi::{LsEngine, LsSampleParams};

#[repr(C)]
pub struct LlamaVocab {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LlamaSampler {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LlamaTokenData {
    id: i32,
    logit: f32,
    p: f32,
}

#[repr(C)]
struct LlamaTokenDataArray {
    data: *mut LlamaTokenData,
    size: usize,
    selected: i64,
    sorted: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LlamaSamplerChainParams {
    no_perf: bool,
}

/// Opaque GBNF grammar sampler handle (owned by Rust, freed via `ls_grammar_free`).
pub struct LsGrammar {
    smpl: *mut LlamaSampler,
}

unsafe extern "C" {
    fn llama_vocab_n_tokens(vocab: *const LlamaVocab) -> i32;
    fn llama_vocab_is_eog(vocab: *const LlamaVocab, token: i32) -> bool;

    fn llama_sampler_accept(smpl: *mut LlamaSampler, token: i32);
    fn llama_sampler_apply(smpl: *mut LlamaSampler, cur_p: *mut LlamaTokenDataArray);
    fn llama_sampler_clone(smpl: *const LlamaSampler) -> *mut LlamaSampler;
    fn llama_sampler_free(smpl: *mut LlamaSampler);

    fn llama_sampler_chain_default_params() -> LlamaSamplerChainParams;
    fn llama_sampler_chain_init(params: LlamaSamplerChainParams) -> *mut LlamaSampler;
    fn llama_sampler_chain_add(chain: *mut LlamaSampler, smpl: *mut LlamaSampler);

    fn llama_sampler_init_greedy() -> *mut LlamaSampler;
    fn llama_sampler_init_top_k(k: i32) -> *mut LlamaSampler;
    fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut LlamaSampler;
    fn llama_sampler_init_min_p(p: f32, min_keep: usize) -> *mut LlamaSampler;
    fn llama_sampler_init_temp(t: f32) -> *mut LlamaSampler;
    fn llama_sampler_init_penalties(
        n_vocab: i32,
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> *mut LlamaSampler;

    // Bridge helpers in `cpp/stream_engine.cpp`
    fn ls_engine_run_decode_batch(
        eng: *mut LsEngine,
        tokens: *const i32,
        n_tokens: i32,
        pos_start: i32,
        logits_all: bool,
        prefetch_err_msg: *const c_char,
        decode_err_msg: *const c_char,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> c_int;
    fn ls_engine_get_logits_ith(eng: *const LsEngine, idx: i32) -> *const f32;
    fn ls_engine_vocab_ptr(eng: *const LsEngine) -> *const LlamaVocab;
    fn ls_llama_grammar_init_nothrow(
        vocab: *const LlamaVocab,
        grammar_str: *const c_char,
        root: *const c_char,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> *mut LlamaSampler;
    fn ls_llama_grammar_accept_nothrow(
        smpl: *mut LlamaSampler,
        token: i32,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> c_int;
    fn ls_llama_sampler_apply_nothrow(
        smpl: *mut LlamaSampler,
        cur_p: *mut LlamaTokenDataArray,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> c_int;
}

struct SamplerHandle(*mut LlamaSampler);

impl SamplerHandle {
    fn null() -> Self {
        Self(ptr::null_mut())
    }

    fn as_ptr(&self) -> *mut LlamaSampler {
        self.0
    }
}

impl Drop for SamplerHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: `self.0` was returned by `llama_sampler_*` and is owned uniquely by this handle.
            unsafe { llama_sampler_free(self.0) };
        }
    }
}

#[derive(Clone, Copy)]
struct ProbEntry {
    id: i32,
    val: f32,
}

/// Fast deterministic SplitMix64 PRNG producing 53-bit uniform doubles in `[0.0, 1.0)`.
struct SplitMixRng(u64);

impl SplitMixRng {
    fn new(seed: u32) -> Self {
        let mut s = u64::from(seed).wrapping_add(0x9E37_79B9_7F4A_7C15);
        s = (s ^ (s >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        s = (s ^ (s >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self(s ^ (s >> 31))
    }

    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

thread_local! {
    static CAND_BUF: RefCell<Vec<LlamaTokenData>> = const { RefCell::new(Vec::new()) };
    static DIST_BUF: RefCell<Vec<ProbEntry>> = const { RefCell::new(Vec::new()) };
}

fn write_error(buf: *mut c_char, cap: usize, msg: &str) {
    if buf.is_null() || cap == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let n = bytes.len().min(cap - 1);
    // SAFETY: Caller guarantees `buf` is valid for writes of `cap` bytes.
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), buf, n);
        *buf.add(n) = 0;
    }
}

fn argmax_logits(logits: &[f32]) -> i32 {
    let mut best_id: i32 = 0;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_id = i as i32;
        }
    }
    best_id
}

fn has_sample_penalties(sp: Option<&LsSampleParams>) -> bool {
    sp.is_some_and(LsSampleParams::has_penalties)
}

fn make_penalties_sampler(
    n_vocab: i32,
    sp: Option<&LsSampleParams>,
    base_tokens: &[i32],
    max_extra: i32,
) -> SamplerHandle {
    let Some(sp) = sp else {
        return SamplerHandle::null();
    };
    if !sp.has_penalties() || n_vocab <= 0 {
        return SamplerHandle::null();
    }
    let b_len = base_tokens.len() as i32;
    let max_total = (b_len + max_extra.max(0)).max(1);
    let eff_last_n = if sp.penalty_last_n < 0 {
        max_total
    } else {
        sp.penalty_last_n.min(max_total)
    };
    if eff_last_n <= 0 {
        return SamplerHandle::null();
    }
    let rp = if sp.repeat_penalty.is_finite() && sp.repeat_penalty > 0.0 {
        sp.repeat_penalty
    } else {
        1.0
    };
    let fp = if sp.frequency_penalty.is_finite() {
        sp.frequency_penalty
    } else {
        0.0
    };
    let pp = if sp.presence_penalty.is_finite() {
        sp.presence_penalty
    } else {
        0.0
    };

    // SAFETY: `llama_sampler_init_penalties` is a pure C API returning a newly allocated sampler.
    let smpl = unsafe { llama_sampler_init_penalties(n_vocab, eff_last_n, rp, fp, pp) };
    let handle = SamplerHandle(smpl);
    if !handle.as_ptr().is_null() && b_len > 0 {
        let start = (b_len - eff_last_n).max(0) as usize;
        for &tok in &base_tokens[start..] {
            if (0..n_vocab).contains(&tok) {
                // SAFETY: `handle.as_ptr()` is a live penalties sampler.
                unsafe { llama_sampler_accept(handle.as_ptr(), tok) };
            }
        }
    }
    handle
}

fn make_filter_chain(sp: Option<&LsSampleParams>) -> SamplerHandle {
    let Some(sp) = sp else {
        // SAFETY: `llama_sampler_init_greedy` returns a newly allocated greedy sampler.
        return SamplerHandle(unsafe { llama_sampler_init_greedy() });
    };
    if !sp.temperature.is_finite() || sp.temperature <= 0.0 {
        // SAFETY: `llama_sampler_init_greedy` returns a newly allocated greedy sampler.
        return SamplerHandle(unsafe { llama_sampler_init_greedy() });
    }
    // SAFETY: Standard llama.cpp sampler chain construction; chain takes ownership of added samplers.
    unsafe {
        let chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if chain.is_null() {
            return SamplerHandle::null();
        }
        if sp.top_k > 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(sp.top_k));
        }
        llama_sampler_chain_add(chain, llama_sampler_init_temp(sp.temperature.max(1e-5)));
        if sp.min_p.is_finite() && sp.min_p > 0.0 {
            llama_sampler_chain_add(chain, llama_sampler_init_min_p(sp.min_p, 1));
        }
        if sp.top_p.is_finite() && sp.top_p > 0.0 && sp.top_p < 1.0 {
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(sp.top_p, 1));
        }
        SamplerHandle(chain)
    }
}

enum StepResult {
    Greedy(i32),
    Dist,
    GrammarRejected,
    SamplerError(c_int),
}

#[allow(clippy::too_many_arguments)]
fn run_sampler_step(
    raw_logits: &[f32],
    pen_smpl: *mut LlamaSampler,
    grammar_smpl: *mut LlamaSampler,
    filter_smpl: *mut LlamaSampler,
    is_greedy: bool,
    cand_buf: &mut Vec<LlamaTokenData>,
    out_dist: &mut Vec<ProbEntry>,
    err_buf: *mut c_char,
    err_cap: usize,
) -> StepResult {
    cand_buf.clear();
    cand_buf.reserve(raw_logits.len());
    for (i, &raw) in raw_logits.iter().enumerate() {
        let logit = if raw.is_finite() {
            raw
        } else {
            f32::NEG_INFINITY
        };
        cand_buf.push(LlamaTokenData {
            id: i as i32,
            logit,
            p: 0.0,
        });
    }

    let mut cur_p = LlamaTokenDataArray {
        data: cand_buf.as_mut_ptr(),
        size: cand_buf.len(),
        selected: -1,
        sorted: false,
    };

    if !pen_smpl.is_null() {
        // SAFETY: `pen_smpl` is live and `cur_p` points into `cand_buf` for the duration of the call.
        unsafe { llama_sampler_apply(pen_smpl, &mut cur_p) };
    }
    if !grammar_smpl.is_null() {
        // SAFETY: `ls_llama_sampler_apply_nothrow` catches any C++ exceptions from grammar application.
        let rc =
            unsafe { ls_llama_sampler_apply_nothrow(grammar_smpl, &mut cur_p, err_buf, err_cap) };
        if rc != 0 {
            return StepResult::SamplerError(rc);
        }
        if cur_p.size == 0 {
            return StepResult::GrammarRejected;
        }
    }

    if is_greedy {
        let mut found = false;
        let mut best_id = 0i32;
        let mut best_val = f32::NEG_INFINITY;
        // SAFETY: `cur_p.data` points into `cand_buf` and `cur_p.size <= cand_buf.len()`.
        let slice = unsafe { std::slice::from_raw_parts(cur_p.data, cur_p.size) };
        for td in slice {
            if td.logit.is_finite() && td.logit > -1e29 && (!found || td.logit > best_val) {
                best_val = td.logit;
                best_id = td.id;
                found = true;
            }
        }
        if !found {
            if !grammar_smpl.is_null() {
                return StepResult::GrammarRejected;
            }
            return StepResult::Greedy(0);
        }
        return StepResult::Greedy(best_id);
    }

    if !filter_smpl.is_null() {
        // SAFETY: `filter_smpl` is live and `cur_p` points into `cand_buf`.
        unsafe { llama_sampler_apply(filter_smpl, &mut cur_p) };
    }

    out_dist.clear();
    out_dist.reserve(cur_p.size.min(256));
    let mut max_l = f32::NEG_INFINITY;
    // SAFETY: `cur_p.data` points into `cand_buf` and `cur_p.size <= cand_buf.len()`.
    let slice = unsafe { std::slice::from_raw_parts(cur_p.data, cur_p.size) };
    for td in slice {
        if td.logit.is_finite() && td.logit > -1e29 {
            if td.logit > max_l {
                max_l = td.logit;
            }
            out_dist.push(ProbEntry {
                id: td.id,
                val: td.logit,
            });
        }
    }
    if out_dist.is_empty() {
        if !grammar_smpl.is_null() {
            return StepResult::GrammarRejected;
        }
        out_dist.push(ProbEntry { id: 0, val: 1.0 });
        return StepResult::Dist;
    }

    let mut sum = 0.0f64;
    for e in out_dist.iter_mut() {
        let p = f64::from(e.val - max_l).exp();
        e.val = p as f32;
        sum += p;
    }
    if !sum.is_finite() || sum <= 0.0 {
        out_dist.truncate(1);
        out_dist[0].val = 1.0;
        return StepResult::Dist;
    }
    for e in out_dist.iter_mut() {
        e.val = (f64::from(e.val) / sum) as f32;
    }
    StepResult::Dist
}

fn sample_from_dist(dist: &[ProbEntry], rng: &mut SplitMixRng) -> i32 {
    let Some(last) = dist.last() else {
        return 0;
    };
    let r = rng.next_f64();
    let mut cum = 0.0f64;
    for e in dist {
        cum += f64::from(e.val);
        if r < cum {
            return e.id;
        }
    }
    last.id
}

/// # Safety
/// `logits` must point to at least `n_vocab` floats if non-null; `penalty_tokens` must point
/// to `n_penalty_tokens` elements if non-null and positive; `sample_params` may be null.
pub unsafe extern "C" fn ls_sample_from_logits_with_history(
    logits: *const f32,
    n_vocab: i32,
    sample_params: *const LsSampleParams,
    penalty_tokens: *const i32,
    n_penalty_tokens: i32,
) -> i32 {
    if logits.is_null() || n_vocab <= 0 {
        return 0;
    }
    // SAFETY: Caller guarantees `logits` is valid for `n_vocab` elements.
    let logits_slice = unsafe { std::slice::from_raw_parts(logits, n_vocab as usize) };
    // SAFETY: Caller guarantees `sample_params` is either null or points to a valid `LsSampleParams`.
    let sp = unsafe { sample_params.as_ref() };
    let is_greedy = sp.is_none_or(|p| p.temperature <= 0.0);
    if is_greedy && !has_sample_penalties(sp) {
        return argmax_logits(logits_slice);
    }
    let pen_slice = if !penalty_tokens.is_null() && n_penalty_tokens > 0 {
        // SAFETY: Caller guarantees `penalty_tokens` has `n_penalty_tokens` elements.
        unsafe { std::slice::from_raw_parts(penalty_tokens, n_penalty_tokens as usize) }
    } else {
        &[]
    };
    let pen = make_penalties_sampler(n_vocab, sp, pen_slice, 0);
    let flt = make_filter_chain(sp);
    CAND_BUF.with(|cb| {
        DIST_BUF.with(|db| {
            let mut cand_buf = cb.borrow_mut();
            let mut dist = db.borrow_mut();
            match run_sampler_step(
                logits_slice,
                pen.as_ptr(),
                ptr::null_mut(),
                flt.as_ptr(),
                is_greedy,
                &mut cand_buf,
                &mut dist,
                ptr::null_mut(),
                0,
            ) {
                StepResult::Greedy(id) => id,
                StepResult::Dist => {
                    let seed = sp.map_or(42, |p| p.seed);
                    let mut rng = SplitMixRng::new(seed);
                    sample_from_dist(&dist, &mut rng)
                }
                StepResult::GrammarRejected | StepResult::SamplerError(_) => 0,
            }
        })
    })
}

/// # Safety
/// Same pointer validity requirements as `ls_sample_from_logits_with_history`.
pub unsafe extern "C" fn ls_sample_from_logits(
    logits: *const f32,
    n_vocab: i32,
    sample_params: *const LsSampleParams,
) -> i32 {
    // SAFETY: Delegates directly with empty history.
    unsafe { ls_sample_from_logits_with_history(logits, n_vocab, sample_params, ptr::null(), 0) }
}

/// # Safety
/// `logits_per_pos` must contain `k + 1` pointers each valid for `n_vocab` floats;
/// `draft_tokens` must contain `k` ints; output pointers must be writable if non-null.
pub unsafe extern "C" fn ls_verify_sample_from_logits_with_history(
    logits_per_pos: *const *const f32,
    n_vocab: i32,
    draft_tokens: *const i32,
    k: i32,
    sample_params: *const LsSampleParams,
    penalty_tokens: *const i32,
    n_penalty_tokens: i32,
    out_accepted_count: *mut i32,
    out_next_token: *mut i32,
) {
    if logits_per_pos.is_null()
        || sample_params.is_null()
        || out_accepted_count.is_null()
        || out_next_token.is_null()
        || n_vocab <= 0
        || k < 0
        || (k > 0 && draft_tokens.is_null())
    {
        if !out_accepted_count.is_null() {
            // SAFETY: Checked non-null above.
            unsafe { *out_accepted_count = 0 };
        }
        if !out_next_token.is_null() {
            // SAFETY: Checked non-null above.
            unsafe { *out_next_token = 0 };
        }
        return;
    }

    let k_usize = k as usize;
    // SAFETY: Caller guarantees `logits_per_pos` has `k + 1` entries.
    let ptrs = unsafe { std::slice::from_raw_parts(logits_per_pos, k_usize + 1) };
    for &p in ptrs {
        if p.is_null() {
            // SAFETY: Output pointers verified non-null above.
            unsafe {
                *out_accepted_count = 0;
                *out_next_token = 0;
            }
            return;
        }
    }
    let drafts = if k > 0 {
        // SAFETY: Caller guarantees `draft_tokens` has `k` elements.
        unsafe { std::slice::from_raw_parts(draft_tokens, k_usize) }
    } else {
        &[]
    };
    // SAFETY: Checked non-null above.
    let sp = unsafe { &*sample_params };
    let pen_slice = if !penalty_tokens.is_null() && n_penalty_tokens > 0 {
        // SAFETY: Caller guarantees `penalty_tokens` has `n_penalty_tokens` elements.
        unsafe { std::slice::from_raw_parts(penalty_tokens, n_penalty_tokens as usize) }
    } else {
        &[]
    };

    let is_greedy = sp.temperature <= 0.0;
    let pen = make_penalties_sampler(n_vocab, Some(sp), pen_slice, k);
    let flt = make_filter_chain(Some(sp));

    CAND_BUF.with(|cb| {
        DIST_BUF.with(|db| {
            let mut cand_buf = cb.borrow_mut();
            let mut dist = db.borrow_mut();

            if is_greedy {
                for (i, &draft_tok) in drafts.iter().enumerate() {
                    // SAFETY: Verified non-null above and valid for `n_vocab` floats.
                    let logits = unsafe { std::slice::from_raw_parts(ptrs[i], n_vocab as usize) };
                    let best = match run_sampler_step(
                        logits,
                        pen.as_ptr(),
                        ptr::null_mut(),
                        flt.as_ptr(),
                        true,
                        &mut cand_buf,
                        &mut dist,
                        ptr::null_mut(),
                        0,
                    ) {
                        StepResult::Greedy(id) => id,
                        _ => 0,
                    };
                    if best != draft_tok {
                        // SAFETY: Output pointers verified non-null above.
                        unsafe {
                            *out_accepted_count = i as i32;
                            *out_next_token = best;
                        }
                        return;
                    }
                    if !pen.as_ptr().is_null() && (0..n_vocab).contains(&draft_tok) {
                        // SAFETY: `pen.as_ptr()` is a live penalties sampler.
                        unsafe { llama_sampler_accept(pen.as_ptr(), draft_tok) };
                    }
                }
                // SAFETY: Verified non-null above and valid for `n_vocab` floats.
                let bonus_logits =
                    unsafe { std::slice::from_raw_parts(ptrs[k_usize], n_vocab as usize) };
                let best = match run_sampler_step(
                    bonus_logits,
                    pen.as_ptr(),
                    ptr::null_mut(),
                    flt.as_ptr(),
                    true,
                    &mut cand_buf,
                    &mut dist,
                    ptr::null_mut(),
                    0,
                ) {
                    StepResult::Greedy(id) => id,
                    _ => 0,
                };
                // SAFETY: Output pointers verified non-null above.
                unsafe {
                    *out_accepted_count = k;
                    *out_next_token = best;
                }
                return;
            }

            let mut rng = SplitMixRng::new(sp.seed);
            for (i, &draft_tok) in drafts.iter().enumerate() {
                // SAFETY: Verified non-null above and valid for `n_vocab` floats.
                let logits = unsafe { std::slice::from_raw_parts(ptrs[i], n_vocab as usize) };
                run_sampler_step(
                    logits,
                    pen.as_ptr(),
                    ptr::null_mut(),
                    flt.as_ptr(),
                    false,
                    &mut cand_buf,
                    &mut dist,
                    ptr::null_mut(),
                    0,
                );
                let mut p_draft = 0.0f64;
                let mut draft_pos = None;
                for (j, e) in dist.iter().enumerate() {
                    if e.id == draft_tok {
                        p_draft = f64::from(e.val);
                        draft_pos = Some(j);
                        break;
                    }
                }
                // Lossless Leviathan rejection step for deterministic draft (q(d) = 1).
                if p_draft >= 1.0 - 1e-7 || rng.next_f64() < p_draft {
                    if !pen.as_ptr().is_null() && (0..n_vocab).contains(&draft_tok) {
                        // SAFETY: `pen.as_ptr()` is a live penalties sampler.
                        unsafe { llama_sampler_accept(pen.as_ptr(), draft_tok) };
                    }
                    continue;
                }
                if let Some(pos) = draft_pos {
                    dist[pos].val = 0.0;
                    let rem = (1.0 - p_draft).max(1e-12);
                    for e in dist.iter_mut() {
                        e.val = (f64::from(e.val) / rem) as f32;
                    }
                }
                // SAFETY: Output pointers verified non-null above.
                unsafe {
                    *out_accepted_count = i as i32;
                    *out_next_token = sample_from_dist(&dist, &mut rng);
                }
                return;
            }

            // SAFETY: Verified non-null above and valid for `n_vocab` floats.
            let bonus_logits =
                unsafe { std::slice::from_raw_parts(ptrs[k_usize], n_vocab as usize) };
            run_sampler_step(
                bonus_logits,
                pen.as_ptr(),
                ptr::null_mut(),
                flt.as_ptr(),
                false,
                &mut cand_buf,
                &mut dist,
                ptr::null_mut(),
                0,
            );
            // SAFETY: Output pointers verified non-null above.
            unsafe {
                *out_accepted_count = k;
                *out_next_token = sample_from_dist(&dist, &mut rng);
            }
        })
    });
}

/// # Safety
/// Same pointer validity requirements as `ls_verify_sample_from_logits_with_history`.
pub unsafe extern "C" fn ls_verify_sample_from_logits(
    logits_per_pos: *const *const f32,
    n_vocab: i32,
    draft_tokens: *const i32,
    k: i32,
    sample_params: *const LsSampleParams,
    out_accepted_count: *mut i32,
    out_next_token: *mut i32,
) {
    // SAFETY: Delegates directly with empty history.
    unsafe {
        ls_verify_sample_from_logits_with_history(
            logits_per_pos,
            n_vocab,
            draft_tokens,
            k,
            sample_params,
            ptr::null(),
            0,
            out_accepted_count,
            out_next_token,
        );
    }
}

/// # Safety
/// `eng` must be a valid `LsEngine` handle; `tokens`, `out_next_token`, and `err_buf` must be valid pointers.
pub unsafe extern "C" fn ls_engine_eval_greedy(
    eng: *mut LsEngine,
    tokens: *const i32,
    n_tokens: i32,
    pos_start: i32,
    out_next_token: *mut i32,
    err_buf: *mut c_char,
    err_cap: usize,
) -> c_int {
    if eng.is_null() || tokens.is_null() || n_tokens <= 0 || out_next_token.is_null() {
        write_error(
            err_buf,
            err_cap,
            "invalid arguments to ls_engine_eval_greedy",
        );
        return -1;
    }
    // SAFETY: `eng` and `tokens` are validated non-null; C string literals are NUL-terminated.
    let rc = unsafe {
        ls_engine_run_decode_batch(
            eng,
            tokens,
            n_tokens,
            pos_start,
            false,
            c"layer prefetch failed before llama_process".as_ptr(),
            c"llama_process failed".as_ptr(),
            err_buf,
            err_cap,
        )
    };
    if rc != 0 {
        return rc;
    }
    // SAFETY: Decode succeeded; logits pointer is valid for `n_vocab` floats.
    unsafe {
        let logits = ls_engine_get_logits_ith(eng, n_tokens - 1);
        if logits.is_null() {
            write_error(err_buf, err_cap, "llama_get_logits_ith returned null");
            return -4;
        }
        let n_vocab = llama_vocab_n_tokens(ls_engine_vocab_ptr(eng));
        let slice = std::slice::from_raw_parts(logits, n_vocab as usize);
        *out_next_token = argmax_logits(slice);
    }
    0
}

/// # Safety
/// `eng` must be a valid `LsEngine` handle; all non-null pointer arguments must match their lengths.
pub unsafe extern "C" fn ls_engine_eval_sample(
    eng: *mut LsEngine,
    tokens: *const i32,
    n_tokens: i32,
    pos_start: i32,
    sample_params: *const LsSampleParams,
    penalty_tokens: *const i32,
    n_penalty_tokens: i32,
    out_next_token: *mut i32,
    err_buf: *mut c_char,
    err_cap: usize,
) -> c_int {
    // SAFETY: Caller guarantees `sample_params` is null or valid.
    let sp = unsafe { sample_params.as_ref() };
    if sp.is_none_or(|p| p.temperature <= 0.0) && !has_sample_penalties(sp) {
        // SAFETY: Delegates directly to `ls_engine_eval_greedy`.
        return unsafe {
            ls_engine_eval_greedy(
                eng,
                tokens,
                n_tokens,
                pos_start,
                out_next_token,
                err_buf,
                err_cap,
            )
        };
    }
    if eng.is_null() || tokens.is_null() || n_tokens <= 0 || out_next_token.is_null() {
        write_error(
            err_buf,
            err_cap,
            "invalid arguments to ls_engine_eval_sample",
        );
        return -1;
    }
    // SAFETY: `eng` and `tokens` are validated non-null.
    let rc = unsafe {
        ls_engine_run_decode_batch(
            eng,
            tokens,
            n_tokens,
            pos_start,
            false,
            c"layer prefetch failed before sampled llama_process".as_ptr(),
            c"sampled llama_process failed".as_ptr(),
            err_buf,
            err_cap,
        )
    };
    if rc != 0 {
        return rc;
    }
    // SAFETY: Decode succeeded; logits and vocab pointers are valid.
    unsafe {
        let logits = ls_engine_get_logits_ith(eng, n_tokens - 1);
        if logits.is_null() {
            write_error(err_buf, err_cap, "llama_get_logits_ith returned null");
            return -4;
        }
        let n_vocab = llama_vocab_n_tokens(ls_engine_vocab_ptr(eng));
        *out_next_token = ls_sample_from_logits_with_history(
            logits,
            n_vocab,
            sample_params,
            penalty_tokens,
            n_penalty_tokens,
        );
    }
    0
}

/// # Safety
/// `eng` must be a valid `LsEngine` handle; `tokens` and `out_argmax_tokens` must hold `n_tokens` elements.
pub unsafe extern "C" fn ls_engine_eval_verify_greedy(
    eng: *mut LsEngine,
    tokens: *const i32,
    n_tokens: i32,
    pos_start: i32,
    out_argmax_tokens: *mut i32,
    err_buf: *mut c_char,
    err_cap: usize,
) -> c_int {
    if eng.is_null() || tokens.is_null() || n_tokens <= 0 || out_argmax_tokens.is_null() {
        write_error(
            err_buf,
            err_cap,
            "invalid arguments to ls_engine_eval_verify_greedy",
        );
        return -1;
    }
    // SAFETY: `eng` and `tokens` are validated non-null.
    let rc = unsafe {
        ls_engine_run_decode_batch(
            eng,
            tokens,
            n_tokens,
            pos_start,
            true,
            c"layer prefetch failed before verify llama_process".as_ptr(),
            c"verify llama_process failed".as_ptr(),
            err_buf,
            err_cap,
        )
    };
    if rc != 0 {
        return rc;
    }
    // SAFETY: Decode succeeded; `out_argmax_tokens` has `n_tokens` writable slots.
    unsafe {
        let n_vocab = llama_vocab_n_tokens(ls_engine_vocab_ptr(eng));
        for i in 0..n_tokens {
            let logits = ls_engine_get_logits_ith(eng, i);
            if logits.is_null() {
                write_error(
                    err_buf,
                    err_cap,
                    "llama_get_logits_ith returned null during verify",
                );
                return -4;
            }
            let slice = std::slice::from_raw_parts(logits, n_vocab as usize);
            *out_argmax_tokens.add(i as usize) = argmax_logits(slice);
        }
    }
    0
}

/// # Safety
/// `eng` must be a valid `LsEngine` handle; `tokens` must hold `n_tokens > 1` elements.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ls_engine_eval_verify_sample(
    eng: *mut LsEngine,
    tokens: *const i32,
    n_tokens: i32,
    pos_start: i32,
    sample_params: *const LsSampleParams,
    penalty_tokens: *const i32,
    n_penalty_tokens: i32,
    out_accepted_count: *mut i32,
    out_next_token: *mut i32,
    err_buf: *mut c_char,
    err_cap: usize,
) -> c_int {
    if eng.is_null()
        || tokens.is_null()
        || n_tokens <= 1
        || sample_params.is_null()
        || out_accepted_count.is_null()
        || out_next_token.is_null()
    {
        write_error(
            err_buf,
            err_cap,
            "invalid arguments to ls_engine_eval_verify_sample",
        );
        return -1;
    }
    // SAFETY: `eng` and `tokens` are validated non-null.
    let rc = unsafe {
        ls_engine_run_decode_batch(
            eng,
            tokens,
            n_tokens,
            pos_start,
            true,
            c"layer prefetch failed before sampled verify llama_process".as_ptr(),
            c"sampled verify llama_process failed".as_ptr(),
            err_buf,
            err_cap,
        )
    };
    if rc != 0 {
        return rc;
    }
    // SAFETY: Decode succeeded; collect `n_tokens` logits pointers and verify.
    unsafe {
        let n_vocab = llama_vocab_n_tokens(ls_engine_vocab_ptr(eng));
        let mut logits_ptrs = Vec::with_capacity(n_tokens as usize);
        for i in 0..n_tokens {
            let p = ls_engine_get_logits_ith(eng, i);
            if p.is_null() {
                write_error(
                    err_buf,
                    err_cap,
                    "llama_get_logits_ith returned null during sampled verify",
                );
                return -4;
            }
            logits_ptrs.push(p);
        }
        ls_verify_sample_from_logits_with_history(
            logits_ptrs.as_ptr(),
            n_vocab,
            tokens.add(1),
            n_tokens - 1,
            sample_params,
            penalty_tokens,
            n_penalty_tokens,
            out_accepted_count,
            out_next_token,
        );
    }
    0
}

/// # Safety
/// `eng` must be a valid `LsEngine` handle; `grammar_str` and `root_rule` must be NUL-terminated if non-null.
pub unsafe extern "C" fn ls_grammar_create(
    eng: *const LsEngine,
    grammar_str: *const c_char,
    root_rule: *const c_char,
    err_buf: *mut c_char,
    err_cap: usize,
) -> *mut LsGrammar {
    if eng.is_null() || grammar_str.is_null() {
        write_error(err_buf, err_cap, "grammar string must not be empty");
        return ptr::null_mut();
    }
    // SAFETY: `eng` and `grammar_str` are validated non-null.
    unsafe {
        let vocab = ls_engine_vocab_ptr(eng);
        if vocab.is_null() || CStr::from_ptr(grammar_str).to_bytes().is_empty() {
            write_error(err_buf, err_cap, "grammar string must not be empty");
            return ptr::null_mut();
        }
        let root = if !root_rule.is_null() && !CStr::from_ptr(root_rule).to_bytes().is_empty() {
            root_rule
        } else {
            c"root".as_ptr()
        };
        let smpl = ls_llama_grammar_init_nothrow(vocab, grammar_str, root, err_buf, err_cap);
        if smpl.is_null() {
            return ptr::null_mut();
        }
        Box::into_raw(Box::new(LsGrammar { smpl }))
    }
}

/// # Safety
/// `grammar` must be null or a pointer previously returned by `ls_grammar_create`.
pub unsafe extern "C" fn ls_grammar_free(grammar: *mut LsGrammar) {
    if grammar.is_null() {
        return;
    }
    // SAFETY: Reclaims the `Box<LsGrammar>` allocated in `ls_grammar_create`.
    unsafe {
        let boxed = Box::from_raw(grammar);
        if !boxed.smpl.is_null() {
            llama_sampler_free(boxed.smpl);
        }
    }
}

/// # Safety
/// `grammar` must be null or a live `LsGrammar` pointer.
pub unsafe extern "C" fn ls_grammar_accept(
    grammar: *mut LsGrammar,
    token: i32,
    err_buf: *mut c_char,
    err_cap: usize,
) -> c_int {
    if grammar.is_null() {
        return 0;
    }
    // SAFETY: `grammar` is non-null; `ls_llama_grammar_accept_nothrow` handles null `smpl` and C++ exceptions.
    unsafe { ls_llama_grammar_accept_nothrow((*grammar).smpl, token, err_buf, err_cap) }
}

/// # Safety
/// `eng` must be a valid `LsEngine` handle; `grammar` may be null (falls back to unconstrained eval).
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ls_engine_eval_with_grammar(
    eng: *mut LsEngine,
    tokens: *const i32,
    n_tokens: i32,
    pos_start: i32,
    sample_params: *const LsSampleParams,
    penalty_tokens: *const i32,
    n_penalty_tokens: i32,
    grammar: *mut LsGrammar,
    out_next_token: *mut i32,
    err_buf: *mut c_char,
    err_cap: usize,
) -> c_int {
    let grammar_smpl = if grammar.is_null() {
        ptr::null_mut()
    } else {
        // SAFETY: `grammar` is checked non-null before dereferencing.
        unsafe { (*grammar).smpl }
    };
    if grammar_smpl.is_null() {
        // SAFETY: Delegates directly to unconstrained eval.
        return unsafe {
            ls_engine_eval_sample(
                eng,
                tokens,
                n_tokens,
                pos_start,
                sample_params,
                penalty_tokens,
                n_penalty_tokens,
                out_next_token,
                err_buf,
                err_cap,
            )
        };
    }
    if eng.is_null() || tokens.is_null() || n_tokens <= 0 || out_next_token.is_null() {
        write_error(
            err_buf,
            err_cap,
            "invalid arguments to ls_engine_eval_with_grammar",
        );
        return -1;
    }
    // SAFETY: `eng` and `tokens` are validated non-null.
    let rc = unsafe {
        ls_engine_run_decode_batch(
            eng,
            tokens,
            n_tokens,
            pos_start,
            false,
            c"layer prefetch failed before grammar llama_process".as_ptr(),
            c"grammar llama_process failed".as_ptr(),
            err_buf,
            err_cap,
        )
    };
    if rc != 0 {
        return rc;
    }
    // SAFETY: Decode succeeded; logits, vocab, and optional pointers are valid.
    unsafe {
        let raw_logits = ls_engine_get_logits_ith(eng, n_tokens - 1);
        if raw_logits.is_null() {
            write_error(err_buf, err_cap, "llama_get_logits_ith returned null");
            return -4;
        }
        let n_vocab = llama_vocab_n_tokens(ls_engine_vocab_ptr(eng));
        let logits_slice = std::slice::from_raw_parts(raw_logits, n_vocab as usize);
        let sp = sample_params.as_ref();
        let is_greedy = sp.is_none_or(|p| p.temperature <= 0.0);
        let pen_slice = if !penalty_tokens.is_null() && n_penalty_tokens > 0 {
            std::slice::from_raw_parts(penalty_tokens, n_penalty_tokens as usize)
        } else {
            &[]
        };
        let pen = make_penalties_sampler(n_vocab, sp, pen_slice, 0);
        let flt = make_filter_chain(sp);
        CAND_BUF.with(|cb| {
            DIST_BUF.with(|db| {
                let mut cand_buf = cb.borrow_mut();
                let mut dist = db.borrow_mut();
                match run_sampler_step(
                    logits_slice,
                    pen.as_ptr(),
                    grammar_smpl,
                    flt.as_ptr(),
                    is_greedy,
                    &mut cand_buf,
                    &mut dist,
                    err_buf,
                    err_cap,
                ) {
                    StepResult::Greedy(id) => {
                        *out_next_token = id;
                        0
                    }
                    StepResult::Dist => {
                        let seed = sp.map_or(42, |p| p.seed);
                        let mut rng = SplitMixRng::new(seed);
                        *out_next_token = sample_from_dist(&dist, &mut rng);
                        0
                    }
                    StepResult::GrammarRejected => {
                        write_error(err_buf, err_cap, "grammar rejected all candidate tokens");
                        -5
                    }
                    StepResult::SamplerError(code) => code,
                }
            })
        })
    }
}

/// # Safety
/// `eng` and `grammar` must be valid live handles; `tokens` must hold `n_tokens > 1` elements.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ls_engine_eval_verify_with_grammar(
    eng: *mut LsEngine,
    tokens: *const i32,
    n_tokens: i32,
    pos_start: i32,
    sample_params: *const LsSampleParams,
    penalty_tokens: *const i32,
    n_penalty_tokens: i32,
    grammar: *mut LsGrammar,
    out_accepted_count: *mut i32,
    out_next_token: *mut i32,
    err_buf: *mut c_char,
    err_cap: usize,
) -> c_int {
    let grammar_smpl = if grammar.is_null() {
        ptr::null_mut()
    } else {
        // SAFETY: `grammar` is checked non-null before dereferencing.
        unsafe { (*grammar).smpl }
    };
    if eng.is_null()
        || tokens.is_null()
        || n_tokens <= 1
        || grammar_smpl.is_null()
        || out_accepted_count.is_null()
        || out_next_token.is_null()
    {
        write_error(
            err_buf,
            err_cap,
            "invalid arguments to ls_engine_eval_verify_with_grammar",
        );
        return -1;
    }
    // SAFETY: `eng` and `tokens` are validated non-null.
    let rc = unsafe {
        ls_engine_run_decode_batch(
            eng,
            tokens,
            n_tokens,
            pos_start,
            true,
            c"layer prefetch failed before grammar verify llama_process".as_ptr(),
            c"grammar verify llama_process failed".as_ptr(),
            err_buf,
            err_cap,
        )
    };
    if rc != 0 {
        return rc;
    }
    // SAFETY: Decode succeeded; all pointers validated above.
    unsafe {
        let vocab = ls_engine_vocab_ptr(eng);
        let n_vocab = llama_vocab_n_tokens(vocab);
        let k = n_tokens - 1;
        let drafts = std::slice::from_raw_parts(tokens.add(1), k as usize);
        let sp = sample_params.as_ref();
        let is_greedy = sp.is_none_or(|p| p.temperature <= 0.0);

        let step_guard = SamplerHandle(llama_sampler_clone(grammar_smpl));
        if step_guard.as_ptr().is_null() {
            write_error(
                err_buf,
                err_cap,
                "failed to clone grammar sampler for speculative verification",
            );
            return -5;
        }
        let pen_slice = if !penalty_tokens.is_null() && n_penalty_tokens > 0 {
            std::slice::from_raw_parts(penalty_tokens, n_penalty_tokens as usize)
        } else {
            &[]
        };
        let pen = make_penalties_sampler(n_vocab, sp, pen_slice, k);
        let flt = make_filter_chain(sp);
        let mut rng = SplitMixRng::new(sp.map_or(42, |p| p.seed));

        CAND_BUF.with(|cb| {
            DIST_BUF.with(|db| {
                let mut cand_buf = cb.borrow_mut();
                let mut dist = db.borrow_mut();

                for (i, &d_tok) in drafts.iter().enumerate() {
                    let raw_logits = ls_engine_get_logits_ith(eng, i as i32);
                    if raw_logits.is_null() {
                        write_error(
                            err_buf,
                            err_cap,
                            "llama_get_logits_ith returned null during grammar verify",
                        );
                        return -4;
                    }
                    let logits_slice = std::slice::from_raw_parts(raw_logits, n_vocab as usize);
                    match run_sampler_step(
                        logits_slice,
                        pen.as_ptr(),
                        step_guard.as_ptr(),
                        flt.as_ptr(),
                        is_greedy,
                        &mut cand_buf,
                        &mut dist,
                        err_buf,
                        err_cap,
                    ) {
                        StepResult::Greedy(best_id) => {
                            if best_id != d_tok {
                                *out_accepted_count = i as i32;
                                *out_next_token = best_id;
                                return 0;
                            }
                        }
                        StepResult::Dist => {
                            let mut p_draft = 0.0f64;
                            let mut draft_pos = None;
                            for (j, e) in dist.iter().enumerate() {
                                if e.id == d_tok {
                                    p_draft = f64::from(e.val);
                                    draft_pos = Some(j);
                                    break;
                                }
                            }
                            if !(p_draft >= 1.0 - 1e-7 || rng.next_f64() < p_draft) {
                                if let Some(pos) = draft_pos {
                                    dist[pos].val = 0.0;
                                    let rem = (1.0 - p_draft).max(1e-12);
                                    for e in dist.iter_mut() {
                                        e.val = (f64::from(e.val) / rem) as f32;
                                    }
                                }
                                *out_accepted_count = i as i32;
                                *out_next_token = sample_from_dist(&dist, &mut rng);
                                return 0;
                            }
                        }
                        StepResult::GrammarRejected => {
                            write_error(err_buf, err_cap, "grammar rejected all candidate tokens");
                            return -5;
                        }
                        StepResult::SamplerError(code) => return code,
                    }

                    if llama_vocab_is_eog(vocab, d_tok) {
                        *out_accepted_count = (i as i32) + 1;
                        *out_next_token = d_tok;
                        return 0;
                    }
                    let accept_rc = ls_llama_grammar_accept_nothrow(
                        step_guard.as_ptr(),
                        d_tok,
                        err_buf,
                        err_cap,
                    );
                    if accept_rc != 0 {
                        return accept_rc;
                    }
                    if !pen.as_ptr().is_null() && (0..n_vocab).contains(&d_tok) {
                        llama_sampler_accept(pen.as_ptr(), d_tok);
                    }
                }

                let raw_bonus = ls_engine_get_logits_ith(eng, k);
                if raw_bonus.is_null() {
                    write_error(
                        err_buf,
                        err_cap,
                        "llama_get_logits_ith returned null at bonus position",
                    );
                    return -4;
                }
                let bonus_slice = std::slice::from_raw_parts(raw_bonus, n_vocab as usize);
                match run_sampler_step(
                    bonus_slice,
                    pen.as_ptr(),
                    step_guard.as_ptr(),
                    flt.as_ptr(),
                    is_greedy,
                    &mut cand_buf,
                    &mut dist,
                    err_buf,
                    err_cap,
                ) {
                    StepResult::Greedy(best_id) => {
                        *out_next_token = best_id;
                        *out_accepted_count = k;
                        0
                    }
                    StepResult::Dist => {
                        *out_next_token = sample_from_dist(&dist, &mut rng);
                        *out_accepted_count = k;
                        0
                    }
                    StepResult::GrammarRejected => {
                        write_error(
                            err_buf,
                            err_cap,
                            "grammar rejected all candidate tokens at bonus position",
                        );
                        -5
                    }
                    StepResult::SamplerError(code) => code,
                }
            })
        })
    }
}
