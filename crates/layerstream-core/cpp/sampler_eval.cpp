#include "engine_internal.hpp"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace ls_internal {
int32_t argmax_logits(const float * logits, int32_t n_vocab) {
    int32_t best_id = 0;
    float best_val = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_id = i;
        }
    }
    return best_id;
}
}  // namespace ls_internal

namespace {

struct ProbEntry {
    int32_t id;
    float val;
};

struct SamplerHandle {
    llama_sampler * smpl = nullptr;
    SamplerHandle() = default;
    explicit SamplerHandle(llama_sampler * s) : smpl(s) {}
    SamplerHandle(const SamplerHandle &) = delete;
    SamplerHandle & operator=(const SamplerHandle &) = delete;
    SamplerHandle(SamplerHandle && other) noexcept : smpl(other.smpl) { other.smpl = nullptr; }
    SamplerHandle & operator=(SamplerHandle && other) noexcept {
        if (this != &other) {
            if (smpl) {
                llama_sampler_free(smpl);
            }
            smpl = other.smpl;
            other.smpl = nullptr;
        }
        return *this;
    }
    ~SamplerHandle() {
        if (smpl) {
            llama_sampler_free(smpl);
        }
    }
};

static bool has_sample_penalties(const ls_sample_params * sp) {
    if (!sp || sp->penalty_last_n == 0) {
        return false;
    }
    const bool has_rep = std::isfinite(sp->repeat_penalty) && sp->repeat_penalty > 0.0f &&
        std::fabs(sp->repeat_penalty - 1.0f) > 1e-6f;
    const bool has_freq = std::isfinite(sp->frequency_penalty) &&
        std::fabs(sp->frequency_penalty) > 1e-6f;
    const bool has_pres = std::isfinite(sp->presence_penalty) &&
        std::fabs(sp->presence_penalty) > 1e-6f;
    return has_rep || has_freq || has_pres;
}

// Reuses upstream llama.cpp `llama_sampler_init_penalties` for repeat / frequency / presence
// penalties across single-step and speculative verification passes.
static SamplerHandle make_penalties_sampler(
    int32_t n_vocab,
    const ls_sample_params * sp,
    const int32_t * base_tokens,
    int32_t n_base,
    int32_t max_extra
) {
    if (!has_sample_penalties(sp) || n_vocab <= 0) {
        return SamplerHandle{};
    }
    const int32_t b_len = (base_tokens && n_base > 0) ? n_base : 0;
    const int32_t max_total = std::max<int32_t>(b_len + std::max<int32_t>(max_extra, 0), 1);
    const int32_t eff_last_n = (sp->penalty_last_n < 0)
        ? max_total
        : std::min<int32_t>(sp->penalty_last_n, max_total);
    if (eff_last_n <= 0) {
        return SamplerHandle{};
    }
    const float rp = (std::isfinite(sp->repeat_penalty) && sp->repeat_penalty > 0.0f)
        ? sp->repeat_penalty
        : 1.0f;
    const float fp = std::isfinite(sp->frequency_penalty) ? sp->frequency_penalty : 0.0f;
    const float pp = std::isfinite(sp->presence_penalty) ? sp->presence_penalty : 0.0f;

    SamplerHandle h{llama_sampler_init_penalties(n_vocab, eff_last_n, rp, fp, pp)};
    if (h.smpl && b_len > 0) {
        const int32_t start = std::max<int32_t>(0, b_len - eff_last_n);
        for (int32_t i = start; i < b_len; ++i) {
            const int32_t tok = base_tokens[i];
            if (tok >= 0 && tok < n_vocab) {
                llama_sampler_accept(h.smpl, tok);
            }
        }
    }
    return h;
}

// Reuses upstream llama.cpp `llama_sampler_chain` (`top_k` -> `temp` -> `min_p` -> `top_p`)
// so single-step sampling, speculative verification, and GBNF grammar sampling share one codepath.
static SamplerHandle make_filter_chain(const ls_sample_params * sp) {
    if (!sp || !std::isfinite(sp->temperature) || sp->temperature <= 0.0f) {
        return SamplerHandle{llama_sampler_init_greedy()};
    }
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!chain) {
        return SamplerHandle{};
    }
    if (sp->top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(sp->top_k));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_temp(std::max(sp->temperature, 1e-5f)));
    if (std::isfinite(sp->min_p) && sp->min_p > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(sp->min_p, 1));
    }
    if (std::isfinite(sp->top_p) && sp->top_p > 0.0f && sp->top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(sp->top_p, 1));
    }
    return SamplerHandle{chain};
}

static bool run_sampler_step(
    const float * raw_logits,
    int32_t n_vocab,
    llama_sampler * pen_smpl,
    llama_sampler * grammar_smpl,
    llama_sampler * filter_smpl,
    bool is_greedy,
    std::vector<llama_token_data> & cand_buf,
    std::vector<ProbEntry> & out_dist,
    int32_t & out_greedy_token
) {
    if (!raw_logits || n_vocab <= 0) {
        return false;
    }
    cand_buf.resize(static_cast<size_t>(n_vocab));
    for (int32_t i = 0; i < n_vocab; ++i) {
        const float v = std::isfinite(raw_logits[i]) ? raw_logits[i] : -INFINITY;
        cand_buf[static_cast<size_t>(i)] = llama_token_data{i, v, 0.0f};
    }
    llama_token_data_array cur_p{cand_buf.data(), cand_buf.size(), -1, false};

    if (pen_smpl) {
        llama_sampler_apply(pen_smpl, &cur_p);
    }
    if (grammar_smpl) {
        llama_sampler_apply(grammar_smpl, &cur_p);
        if (cur_p.size == 0) {
            return false;
        }
    }

    if (is_greedy) {
        bool found = false;
        int32_t best_id = 0;
        float best_val = -INFINITY;
        for (size_t i = 0; i < cur_p.size; ++i) {
            const auto & td = cur_p.data[i];
            if (std::isfinite(td.logit) && td.logit > -1e29f && (!found || td.logit > best_val)) {
                best_val = td.logit;
                best_id = td.id;
                found = true;
            }
        }
        if (!found) {
            if (grammar_smpl) {
                return false;
            }
            out_greedy_token = 0;
            return true;
        }
        out_greedy_token = best_id;
        return true;
    }

    if (filter_smpl) {
        llama_sampler_apply(filter_smpl, &cur_p);
    }

    out_dist.clear();
    out_dist.reserve(std::min<size_t>(cur_p.size, 256));
    float max_l = -INFINITY;
    for (size_t i = 0; i < cur_p.size; ++i) {
        const auto & td = cur_p.data[i];
        if (std::isfinite(td.logit) && td.logit > -1e29f) {
            if (td.logit > max_l) {
                max_l = td.logit;
            }
            out_dist.push_back(ProbEntry{td.id, td.logit});
        }
    }
    if (out_dist.empty()) {
        if (grammar_smpl) {
            return false;
        }
        out_dist.push_back(ProbEntry{0, 1.0f});
        return true;
    }

    double sum = 0.0;
    for (auto & e : out_dist) {
        const double p = std::exp(static_cast<double>(e.val - max_l));
        e.val = static_cast<float>(p);
        sum += p;
    }
    if (!std::isfinite(sum) || sum <= 0.0) {
        out_dist.resize(1);
        out_dist[0].val = 1.0f;
        return true;
    }
    for (auto & e : out_dist) {
        e.val = static_cast<float>(static_cast<double>(e.val) / sum);
    }
    return true;
}

static int32_t sample_from_dist(const std::vector<ProbEntry> & dist, std::mt19937 & rng) {
    if (dist.empty()) {
        return 0;
    }
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const double r = uni(rng);
    double cum = 0.0;
    for (const auto & e : dist) {
        cum += static_cast<double>(e.val);
        if (r < cum) {
            return e.id;
        }
    }
    return dist.back().id;
}

}  // namespace

namespace ls_internal {
int report_eval_failure(
    const ls_engine * eng,
    int rc,
    const char * fallback_msg,
    char * err_buf,
    size_t err_cap
) {
    if (ls_internal::engine_is_cancelled(eng)) {
        ls_internal::set_error(err_buf, err_cap, "request cancelled");
        return -100;
    }
    if (!ls_internal::engine_eval_error_msg(eng).empty()) {
        ls_internal::set_error(err_buf, err_cap, ls_internal::engine_eval_error_msg(eng));
    } else {
        ls_internal::set_error(err_buf, err_cap, fallback_msg);
    }
    return rc != 0 ? rc : -3;
}

}  // namespace ls_internal

int32_t ls_sample_from_logits_with_history(
    const float * logits,
    int32_t n_vocab,
    const ls_sample_params * sample_params,
    const int32_t * penalty_tokens,
    int32_t n_penalty_tokens
) {
    if (!logits || n_vocab <= 0) {
        return 0;
    }
    const bool is_greedy = (!sample_params || sample_params->temperature <= 0.0f);
    if (is_greedy && !has_sample_penalties(sample_params)) {
        return ls_internal::argmax_logits(logits, n_vocab);
    }
    SamplerHandle pen = make_penalties_sampler(n_vocab, sample_params, penalty_tokens, n_penalty_tokens, 0);
    SamplerHandle flt = make_filter_chain(sample_params);
    std::vector<llama_token_data> cand_buf;
    std::vector<ProbEntry> dist;
    int32_t greedy_id = 0;
    if (!run_sampler_step(logits, n_vocab, pen.smpl, nullptr, flt.smpl, is_greedy, cand_buf, dist, greedy_id)) {
        return 0;
    }
    if (is_greedy) {
        return greedy_id;
    }
    std::mt19937 rng(sample_params->seed);
    return sample_from_dist(dist, rng);
}

int32_t ls_sample_from_logits(
    const float * logits,
    int32_t n_vocab,
    const ls_sample_params * sample_params
) {
    return ls_sample_from_logits_with_history(logits, n_vocab, sample_params, nullptr, 0);
}

void ls_verify_sample_from_logits_with_history(
    const float * const * logits_per_pos,
    int32_t n_vocab,
    const int32_t * draft_tokens,
    int32_t k,
    const ls_sample_params * sample_params,
    const int32_t * penalty_tokens,
    int32_t n_penalty_tokens,
    int32_t * out_accepted_count,
    int32_t * out_next_token
) {
    if (!logits_per_pos || !sample_params || !out_accepted_count || !out_next_token || n_vocab <= 0 || k < 0) {
        if (out_accepted_count) {
            *out_accepted_count = 0;
        }
        if (out_next_token) {
            *out_next_token = 0;
        }
        return;
    }
    for (int32_t i = 0; i <= k; ++i) {
        if (!logits_per_pos[i]) {
            *out_accepted_count = 0;
            *out_next_token = 0;
            return;
        }
    }
    const bool is_greedy = (sample_params->temperature <= 0.0f);
    SamplerHandle pen = make_penalties_sampler(n_vocab, sample_params, penalty_tokens, n_penalty_tokens, k);
    SamplerHandle flt = make_filter_chain(sample_params);
    std::vector<llama_token_data> cand_buf;
    std::vector<ProbEntry> dist;

    if (is_greedy) {
        int32_t best = 0;
        for (int32_t i = 0; i < k; ++i) {
            run_sampler_step(logits_per_pos[i], n_vocab, pen.smpl, nullptr, flt.smpl, true, cand_buf, dist, best);
            if (best != draft_tokens[i]) {
                *out_accepted_count = i;
                *out_next_token = best;
                return;
            }
            if (pen.smpl && draft_tokens[i] >= 0 && draft_tokens[i] < n_vocab) {
                llama_sampler_accept(pen.smpl, draft_tokens[i]);
            }
        }
        run_sampler_step(logits_per_pos[k], n_vocab, pen.smpl, nullptr, flt.smpl, true, cand_buf, dist, best);
        *out_accepted_count = k;
        *out_next_token = best;
        return;
    }

    std::mt19937 rng(sample_params->seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    int32_t dummy_greedy = 0;

    int32_t accepted = 0;
    for (int32_t i = 0; i < k; ++i) {
        run_sampler_step(logits_per_pos[i], n_vocab, pen.smpl, nullptr, flt.smpl, false, cand_buf, dist, dummy_greedy);
        const int32_t draft_tok = draft_tokens[i];

        double p_draft = 0.0;
        size_t draft_pos = dist.size();
        for (size_t j = 0; j < dist.size(); ++j) {
            if (dist[j].id == draft_tok) {
                p_draft = static_cast<double>(dist[j].val);
                draft_pos = j;
                break;
            }
        }

        // Lossless Leviathan rejection step for deterministic draft (q(d)=1):
        // accept d with prob p(d); otherwise sample from p'(v) = p(v)/(1 - p(d)) for v != d.
        if (p_draft >= 1.0 - 1e-7 || uni(rng) < p_draft) {
            ++accepted;
            if (pen.smpl && draft_tok >= 0 && draft_tok < n_vocab) {
                llama_sampler_accept(pen.smpl, draft_tok);
            }
            continue;
        }

        if (draft_pos < dist.size()) {
            dist[draft_pos].val = 0.0f;
            const double rem = std::max(1e-12, 1.0 - p_draft);
            for (auto & e : dist) {
                e.val = static_cast<float>(static_cast<double>(e.val) / rem);
            }
        }
        *out_accepted_count = accepted;
        *out_next_token = sample_from_dist(dist, rng);
        return;
    }

    run_sampler_step(logits_per_pos[k], n_vocab, pen.smpl, nullptr, flt.smpl, false, cand_buf, dist, dummy_greedy);
    *out_accepted_count = k;
    *out_next_token = sample_from_dist(dist, rng);
}

void ls_verify_sample_from_logits(
    const float * const * logits_per_pos,
    int32_t n_vocab,
    const int32_t * draft_tokens,
    int32_t k,
    const ls_sample_params * sample_params,
    int32_t * out_accepted_count,
    int32_t * out_next_token
) {
    ls_verify_sample_from_logits_with_history(
        logits_per_pos,
        n_vocab,
        draft_tokens,
        k,
        sample_params,
        nullptr,
        0,
        out_accepted_count,
        out_next_token
    );
}

namespace {

int run_decode_batch(
    ls_engine * eng,
    const int32_t * tokens,
    int32_t n_tokens,
    int32_t pos_start,
    bool logits_all,
    const char * prefetch_err_msg,
    const char * decode_err_msg,
    char * err_buf,
    size_t err_cap
) {
    llama_context * ctx = ls_internal::engine_ctx(eng);
    struct BatchExtGuard {
        llama_batch_ext * batch = nullptr;
        explicit BatchExtGuard(llama_context * c) : batch(llama_batch_ext_init(c)) {}
        BatchExtGuard(const BatchExtGuard &) = delete;
        BatchExtGuard & operator=(const BatchExtGuard &) = delete;
        ~BatchExtGuard() {
            if (batch) {
                llama_batch_ext_free(batch);
            }
        }
    } guard(ctx);

    if (!guard.batch) {
        ls_internal::set_error(err_buf, err_cap, "llama_batch_ext_init returned null");
        return -2;
    }
    for (int32_t i = 0; i < n_tokens; ++i) {
        const int32_t idx = llama_batch_ext_add_token(guard.batch, 0, tokens[i]);
        if (idx < 0) {
            ls_internal::set_error(err_buf, err_cap, "llama_batch_ext_add_token failed");
            return -2;
        }
        const llama_pos pos = pos_start + i;
        llama_batch_ext_set_pos(guard.batch, idx, &pos);
        llama_batch_ext_set_output_logits(guard.batch, idx, logits_all || (i == n_tokens - 1));
    }

    using Clock = std::chrono::steady_clock;
    const auto t0 = ls_internal::engine_profile_enabled(eng) ? Clock::now() : Clock::time_point{};
    ls_internal::engine_prepare_decode_pass(eng);
    if (ls_internal::engine_eval_failed(eng)) {
        return ls_internal::report_eval_failure(eng, -2, prefetch_err_msg, err_buf, err_cap);
    }

    const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, guard.batch);
    if (ls_internal::engine_profile_enabled(eng)) {
        const auto dt = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
        ls_internal::engine_record_decode_pass_us(eng, static_cast<uint64_t>(dt));
    }
    if (rc != 0 || ls_internal::engine_eval_failed(eng)) {
        return ls_internal::report_eval_failure(eng, rc, decode_err_msg, err_buf, err_cap);
    }
    return 0;
}

}  // namespace

int ls_engine_eval_greedy(
    ls_engine * eng,
    const int32_t * tokens,
    int32_t n_tokens,
    int32_t pos_start,
    int32_t * out_next_token,
    char * err_buf,
    size_t err_cap
) {
    if (!eng || !ls_internal::engine_ctx(eng) || !tokens || n_tokens <= 0 || !out_next_token) {
        ls_internal::set_error(err_buf, err_cap, "invalid arguments to ls_engine_eval_greedy");
        return -1;
    }

    if (const int rc = run_decode_batch(
            eng,
            tokens,
            n_tokens,
            pos_start,
            false,
            "layer prefetch failed before llama_process",
            "llama_process failed",
            err_buf,
            err_cap
        );
        rc != 0) {
        return rc;
    }

    const float * logits = llama_get_logits_ith(ls_internal::engine_ctx(eng), n_tokens - 1);
    if (!logits) {
        ls_internal::set_error(err_buf, err_cap, "llama_get_logits_ith returned null");
        return -4;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(ls_internal::engine_vocab(eng));
    *out_next_token = ls_internal::argmax_logits(logits, n_vocab);
    return 0;
}

int ls_engine_eval_sample(
    ls_engine * eng,
    const int32_t * tokens,
    int32_t n_tokens,
    int32_t pos_start,
    const ls_sample_params * sample_params,
    const int32_t * penalty_tokens,
    int32_t n_penalty_tokens,
    int32_t * out_next_token,
    char * err_buf,
    size_t err_cap
) {
    if ((!sample_params || sample_params->temperature <= 0.0f) && !has_sample_penalties(sample_params)) {
        return ls_engine_eval_greedy(eng, tokens, n_tokens, pos_start, out_next_token, err_buf, err_cap);
    }
    if (!eng || !ls_internal::engine_ctx(eng) || !tokens || n_tokens <= 0 || !out_next_token) {
        ls_internal::set_error(err_buf, err_cap, "invalid arguments to ls_engine_eval_sample");
        return -1;
    }

    if (const int rc = run_decode_batch(
            eng,
            tokens,
            n_tokens,
            pos_start,
            false,
            "layer prefetch failed before sampled llama_process",
            "sampled llama_process failed",
            err_buf,
            err_cap
        );
        rc != 0) {
        return rc;
    }

    const float * logits = llama_get_logits_ith(ls_internal::engine_ctx(eng), n_tokens - 1);
    if (!logits) {
        ls_internal::set_error(err_buf, err_cap, "llama_get_logits_ith returned null");
        return -4;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(ls_internal::engine_vocab(eng));
    *out_next_token = ls_sample_from_logits_with_history(
        logits,
        n_vocab,
        sample_params,
        penalty_tokens,
        n_penalty_tokens
    );
    return 0;
}

int ls_engine_eval_verify_greedy(
    ls_engine * eng,
    const int32_t * tokens,
    int32_t n_tokens,
    int32_t pos_start,
    int32_t * out_argmax_tokens,
    char * err_buf,
    size_t err_cap
) {
    if (!eng || !ls_internal::engine_ctx(eng) || !tokens || n_tokens <= 0 || !out_argmax_tokens) {
        ls_internal::set_error(err_buf, err_cap, "invalid arguments to ls_engine_eval_verify_greedy");
        return -1;
    }

    if (const int rc = run_decode_batch(
            eng,
            tokens,
            n_tokens,
            pos_start,
            true,
            "layer prefetch failed before verify llama_process",
            "verify llama_process failed",
            err_buf,
            err_cap
        );
        rc != 0) {
        return rc;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(ls_internal::engine_vocab(eng));
    for (int32_t i = 0; i < n_tokens; ++i) {
        const float * logits = llama_get_logits_ith(ls_internal::engine_ctx(eng), i);
        if (!logits) {
            ls_internal::set_error(err_buf, err_cap, "llama_get_logits_ith returned null during verify");
            return -4;
        }
        out_argmax_tokens[i] = ls_internal::argmax_logits(logits, n_vocab);
    }
    return 0;
}

int ls_engine_eval_verify_sample(
    ls_engine * eng,
    const int32_t * tokens,
    int32_t n_tokens,
    int32_t pos_start,
    const ls_sample_params * sample_params,
    const int32_t * penalty_tokens,
    int32_t n_penalty_tokens,
    int32_t * out_accepted_count,
    int32_t * out_next_token,
    char * err_buf,
    size_t err_cap
) {
    if (!eng || !ls_internal::engine_ctx(eng) || !tokens || n_tokens <= 1 || !sample_params || !out_accepted_count || !out_next_token) {
        ls_internal::set_error(err_buf, err_cap, "invalid arguments to ls_engine_eval_verify_sample");
        return -1;
    }

    if (const int rc = run_decode_batch(
            eng,
            tokens,
            n_tokens,
            pos_start,
            true,
            "layer prefetch failed before sampled verify llama_process",
            "sampled verify llama_process failed",
            err_buf,
            err_cap
        );
        rc != 0) {
        return rc;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(ls_internal::engine_vocab(eng));
    const int32_t k = n_tokens - 1;
    std::vector<const float *> logits_ptrs(static_cast<size_t>(n_tokens));
    for (int32_t i = 0; i < n_tokens; ++i) {
        logits_ptrs[static_cast<size_t>(i)] = llama_get_logits_ith(ls_internal::engine_ctx(eng), i);
        if (!logits_ptrs[static_cast<size_t>(i)]) {
            ls_internal::set_error(err_buf, err_cap, "llama_get_logits_ith returned null during sampled verify");
            return -4;
        }
    }
    ls_verify_sample_from_logits_with_history(
        logits_ptrs.data(),
        n_vocab,
        tokens + 1,
        k,
        sample_params,
        penalty_tokens,
        n_penalty_tokens,
        out_accepted_count,
        out_next_token
    );
    return 0;
}

struct ls_grammar {
    llama_sampler * smpl = nullptr;
    const llama_vocab * vocab = nullptr;
};

ls_grammar * ls_grammar_create(
    const ls_engine * eng,
    const char * grammar_str,
    const char * root_rule,
    char * err_buf,
    size_t err_cap
) {
    if (!eng || !ls_internal::engine_vocab(eng) || !grammar_str || grammar_str[0] == '\0') {
        ls_internal::set_error(err_buf, err_cap, "grammar string must not be empty");
        return nullptr;
    }
    const char * root = (root_rule && root_rule[0] != '\0') ? root_rule : "root";
    try {
        llama_sampler * smpl = llama_sampler_init_grammar(ls_internal::engine_vocab(eng), grammar_str, root);
        if (!smpl) {
            ls_internal::set_error(err_buf, err_cap, "failed to parse GBNF grammar");
            return nullptr;
        }
        auto * g = new (std::nothrow) ls_grammar();
        if (!g) {
            llama_sampler_free(smpl);
            ls_internal::set_error(err_buf, err_cap, "out of memory allocating ls_grammar");
            return nullptr;
        }
        g->smpl = smpl;
        g->vocab = ls_internal::engine_vocab(eng);
        return g;
    } catch (const std::exception & e) {
        ls_internal::set_error(err_buf, err_cap, std::string("GBNF grammar init exception: ") + e.what());
        return nullptr;
    } catch (...) {
        ls_internal::set_error(err_buf, err_cap, "unknown GBNF grammar init exception");
        return nullptr;
    }
}

void ls_grammar_free(ls_grammar * grammar) {
    if (!grammar) {
        return;
    }
    if (grammar->smpl) {
        llama_sampler_free(grammar->smpl);
        grammar->smpl = nullptr;
    }
    delete grammar;
}

int ls_grammar_accept(
    ls_grammar * grammar,
    int32_t token,
    char * err_buf,
    size_t err_cap
) {
    if (!grammar || !grammar->smpl) {
        return 0;
    }
    try {
        llama_sampler_accept(grammar->smpl, token);
        return 0;
    } catch (const std::exception & e) {
        ls_internal::set_error(err_buf, err_cap, std::string("grammar accept failed: ") + e.what());
        return -1;
    } catch (...) {
        ls_internal::set_error(err_buf, err_cap, "unknown grammar accept exception");
        return -1;
    }
}

int ls_engine_eval_with_grammar(
    ls_engine * eng,
    const int32_t * tokens,
    int32_t n_tokens,
    int32_t pos_start,
    const ls_sample_params * sample_params,
    const int32_t * penalty_tokens,
    int32_t n_penalty_tokens,
    ls_grammar * grammar,
    int32_t * out_next_token,
    char * err_buf,
    size_t err_cap
) {
    if (!grammar || !grammar->smpl) {
        if ((!sample_params || sample_params->temperature <= 0.0f) && !has_sample_penalties(sample_params)) {
            return ls_engine_eval_greedy(eng, tokens, n_tokens, pos_start, out_next_token, err_buf, err_cap);
        }
        return ls_engine_eval_sample(
            eng,
            tokens,
            n_tokens,
            pos_start,
            sample_params,
            penalty_tokens,
            n_penalty_tokens,
            out_next_token,
            err_buf,
            err_cap
        );
    }
    if (!eng || !ls_internal::engine_ctx(eng) || !tokens || n_tokens <= 0 || !out_next_token) {
        ls_internal::set_error(err_buf, err_cap, "invalid arguments to ls_engine_eval_with_grammar");
        return -1;
    }

    try {
        if (const int rc = run_decode_batch(
                eng,
                tokens,
                n_tokens,
                pos_start,
                false,
                "layer prefetch failed before grammar llama_process",
                "grammar llama_process failed",
                err_buf,
                err_cap
            );
            rc != 0) {
            return rc;
        }

        const float * raw_logits = llama_get_logits_ith(ls_internal::engine_ctx(eng), n_tokens - 1);
        if (!raw_logits) {
            ls_internal::set_error(err_buf, err_cap, "llama_get_logits_ith returned null");
            return -4;
        }

        const int32_t n_vocab = llama_vocab_n_tokens(ls_internal::engine_vocab(eng));
        const bool is_greedy = (!sample_params || sample_params->temperature <= 0.0f);
        SamplerHandle pen = make_penalties_sampler(n_vocab, sample_params, penalty_tokens, n_penalty_tokens, 0);
        SamplerHandle flt = make_filter_chain(sample_params);
        std::vector<llama_token_data> cand_buf;
        std::vector<ProbEntry> dist;
        int32_t greedy_id = 0;
        if (!run_sampler_step(
                raw_logits,
                n_vocab,
                pen.smpl,
                grammar->smpl,
                flt.smpl,
                is_greedy,
                cand_buf,
                dist,
                greedy_id
            )) {
            ls_internal::set_error(err_buf, err_cap, "grammar rejected all candidate tokens");
            return -5;
        }
        if (is_greedy) {
            *out_next_token = greedy_id;
        } else {
            std::mt19937 rng(sample_params->seed);
            *out_next_token = sample_from_dist(dist, rng);
        }
        return 0;
    } catch (const std::exception & e) {
        ls_internal::set_error(err_buf, err_cap, std::string("grammar eval failed: ") + e.what());
        return -6;
    } catch (...) {
        ls_internal::set_error(err_buf, err_cap, "unknown grammar eval exception");
        return -6;
    }
}

int ls_engine_eval_verify_with_grammar(
    ls_engine * eng,
    const int32_t * tokens,
    int32_t n_tokens,
    int32_t pos_start,
    const ls_sample_params * sample_params,
    const int32_t * penalty_tokens,
    int32_t n_penalty_tokens,
    ls_grammar * grammar,
    int32_t * out_accepted_count,
    int32_t * out_next_token,
    char * err_buf,
    size_t err_cap
) {
    if (!eng || !ls_internal::engine_ctx(eng) || !tokens || n_tokens <= 1 || !grammar || !grammar->smpl || !out_accepted_count || !out_next_token) {
        ls_internal::set_error(err_buf, err_cap, "invalid arguments to ls_engine_eval_verify_with_grammar");
        return -1;
    }

    try {
        if (const int rc = run_decode_batch(
                eng,
                tokens,
                n_tokens,
                pos_start,
                true,
                "layer prefetch failed before grammar verify llama_process",
                "grammar verify llama_process failed",
                err_buf,
                err_cap
            );
            rc != 0) {
            return rc;
        }

        const int32_t n_vocab = llama_vocab_n_tokens(ls_internal::engine_vocab(eng));
        const int32_t k = n_tokens - 1;
        const int32_t * draft_tokens = tokens + 1;
        const bool is_greedy = (!sample_params || sample_params->temperature <= 0.0f);

        SamplerHandle step_guard{llama_sampler_clone(grammar->smpl)};
        if (!step_guard.smpl) {
            ls_internal::set_error(err_buf, err_cap, "failed to clone grammar sampler for speculative verification");
            return -5;
        }
        SamplerHandle pen = make_penalties_sampler(n_vocab, sample_params, penalty_tokens, n_penalty_tokens, k);
        SamplerHandle flt = make_filter_chain(sample_params);

        std::vector<llama_token_data> cand_buf;
        std::vector<ProbEntry> dist;
        std::mt19937 rng(sample_params ? sample_params->seed : 42u);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        int32_t best_id = 0;

        for (int32_t i = 0; i < k; ++i) {
            const float * raw_logits = llama_get_logits_ith(ls_internal::engine_ctx(eng), i);
            if (!raw_logits) {
                ls_internal::set_error(err_buf, err_cap, "llama_get_logits_ith returned null during grammar verify");
                return -4;
            }
            if (!run_sampler_step(
                    raw_logits,
                    n_vocab,
                    pen.smpl,
                    step_guard.smpl,
                    flt.smpl,
                    is_greedy,
                    cand_buf,
                    dist,
                    best_id
                )) {
                ls_internal::set_error(err_buf, err_cap, "grammar rejected all candidate tokens");
                return -5;
            }
            const int32_t d_tok = draft_tokens[i];

            if (is_greedy) {
                if (best_id != d_tok) {
                    *out_accepted_count = i;
                    *out_next_token = best_id;
                    return 0;
                }
            } else {
                double p_draft = 0.0;
                size_t draft_pos = dist.size();
                for (size_t j = 0; j < dist.size(); ++j) {
                    if (dist[j].id == d_tok) {
                        p_draft = static_cast<double>(dist[j].val);
                        draft_pos = j;
                        break;
                    }
                }
                if (!(p_draft >= 1.0 - 1e-7 || uni(rng) < p_draft)) {
                    if (draft_pos < dist.size()) {
                        dist[draft_pos].val = 0.0f;
                        const double rem = std::max(1e-12, 1.0 - p_draft);
                        for (auto & e : dist) {
                            e.val = static_cast<float>(static_cast<double>(e.val) / rem);
                        }
                    }
                    *out_accepted_count = i;
                    *out_next_token = sample_from_dist(dist, rng);
                    return 0;
                }
            }

            if (llama_vocab_is_eog(ls_internal::engine_vocab(eng), d_tok)) {
                *out_accepted_count = i + 1;
                *out_next_token = d_tok;
                return 0;
            }
            llama_sampler_accept(step_guard.smpl, d_tok);
            if (pen.smpl && d_tok >= 0 && d_tok < n_vocab) {
                llama_sampler_accept(pen.smpl, d_tok);
            }
        }

        const float * raw_bonus = llama_get_logits_ith(ls_internal::engine_ctx(eng), k);
        if (!raw_bonus) {
            ls_internal::set_error(err_buf, err_cap, "llama_get_logits_ith returned null at bonus position");
            return -4;
        }
        if (!run_sampler_step(
                raw_bonus,
                n_vocab,
                pen.smpl,
                step_guard.smpl,
                flt.smpl,
                is_greedy,
                cand_buf,
                dist,
                best_id
            )) {
            ls_internal::set_error(err_buf, err_cap, "grammar rejected all candidate tokens at bonus position");
            return -5;
        }
        *out_next_token = is_greedy ? best_id : sample_from_dist(dist, rng);
        *out_accepted_count = k;
        return 0;
    } catch (const std::exception & e) {
        ls_internal::set_error(err_buf, err_cap, std::string("grammar verify failed: ") + e.what());
        return -6;
    } catch (...) {
        ls_internal::set_error(err_buf, err_cap, "unknown grammar verify exception");
        return -6;
    }
}
