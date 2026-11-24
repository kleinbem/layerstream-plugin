#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ls_engine ls_engine;
typedef struct ls_grammar ls_grammar;

/**
 * Callback invoked by C++ to populate all tensors of a layer (or resident set)
 * into a host buffer `dst_ptr` of size `dst_cap`.
 *
 * `tensor_names[i]` must be copied into `dst_ptr + tensor_offsets[i]` with exact
 * length `tensor_sizes[i]`.
 *
 * Returns 0 on success, non-zero on error.
 */
typedef int (*ls_load_tensors_cb)(
    void * user_data,
    int32_t layer_idx,
    const char * const * tensor_names,
    const uint64_t * tensor_offsets,
    const uint64_t * tensor_sizes,
    size_t tensor_count,
    uint8_t * dst_ptr,
    size_t dst_cap
);

typedef struct ls_span_slice {
    uint64_t src_offset;
    uint64_t dst_offset;
    uint64_t len;
    uint32_t tensor_idx;
} ls_span_slice;

typedef struct ls_tensor_slice_req {
    const char * tensor_name;
    uint64_t src_offset_in_tensor;
    uint64_t dst_offset;
    uint64_t len;
    uint32_t tensor_idx;
} ls_tensor_slice_req;

/**
 * Loads arbitrary byte sub-ranges of tensors (e.g., active MoE expert slices)
 * into a host buffer `dst_ptr` of size `dst_cap`.
 * Returns 0 on success, non-zero on error.
 */
typedef int (*ls_load_slices_cb)(
    void * user_data,
    const ls_tensor_slice_req * reqs,
    size_t req_count,
    uint8_t * dst_ptr,
    size_t dst_cap
);

/**
 * Optional zero-copy direct-to-pinned-buffer callback for discrete GPUs.
 * Reads the layer's 4K-aligned span via O_DIRECT straight into `pin_buf`
 * (zero intermediate scratch buffer, zero CPU memcpy!) and populates `out_slices`
 * so C++ issues per-tensor async H2D copies into `slot_buf[slot]`.
 *
 * Returns > 0 (slice count) on zero-copy success, 0 to fall back to `load_tensors_cb`,
 * or < 0 on fatal I/O error.
 */
typedef int (*ls_load_layer_direct_cb)(
    void * user_data,
    int32_t layer_idx,
    const char * const * tensor_names,
    const uint64_t * tensor_offsets,
    const uint64_t * tensor_sizes,
    size_t tensor_count,
    uint8_t * pin_buf,
    size_t pin_cap,
    ls_span_slice * out_slices,
    size_t max_slices
);

typedef struct ls_engine_config {
    const char * model_path;
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_threads;
    int32_t resident_layers;
    uint64_t vram_budget_bytes;
    /** Automatically probe free GPU memory and pick a safe hybrid residency budget. */
    bool vram_budget_auto;
    /** KV cache type: 0 = F16 (default), 1 = Q8_0 (2x smaller), 2 = Q4_0 (4x smaller). */
    int32_t kv_cache_type;
    bool use_gpu;
    bool full_resident_reference;
    /** Pin token_embd.weight in CPU RAM instead of GPU VRAM (discrete GPUs only). */
    bool cpu_embeddings;
    /** Pin output.weight (LM head) in CPU RAM instead of GPU VRAM (discrete GPUs only). */
    bool cpu_head;
    /** Stream output.weight through ping-pong GPU slots when <= slot_bytes (else CPU fallback on discrete GPUs). */
    bool stream_head;
    /** Enable per-expert active streaming for sparse MoE layers (default true). */
    bool moe_active_streaming;

    void * io_user_data;
    ls_load_tensors_cb load_tensors_cb;
    ls_load_layer_direct_cb load_layer_direct_cb;
    ls_load_slices_cb load_slices_cb;
} ls_engine_config;

typedef struct ls_engine_stats {
    uint32_t n_layers;
    uint32_t resident_layers;
    uint32_t streamed_layers;
    uint64_t resident_bytes;
    uint64_t cpu_resident_bytes;
    uint64_t slot_bytes;
    uint64_t expert_slot_bytes;
    uint64_t total_layer_loads;
    uint32_t moe_layers;
    uint64_t moe_experts_loaded;
    uint64_t moe_experts_total;
    bool head_streamed;
    bool moe_active_streaming;
    bool is_uma_zero_copy;
    bool is_unified_memory;
    bool using_gpu;
    bool can_speculate;
    char backend_name[32];
} ls_engine_stats;

typedef struct ls_chat_message {
    const char * role;
    const char * content;
} ls_chat_message;

typedef struct ls_sample_params {
    float temperature;
    float top_p;
    float min_p;
    int32_t top_k;
    float repeat_penalty;
    float frequency_penalty;
    float presence_penalty;
    int32_t penalty_last_n;
    uint32_t seed;
} ls_sample_params;

typedef bool (*ls_cancel_check_cb)(const void * user_data);

ls_engine * ls_engine_create(const ls_engine_config * cfg, char * err_buf, size_t err_cap);
void ls_engine_free(ls_engine * eng);
void ls_engine_get_stats(const ls_engine * eng, ls_engine_stats * out);
int32_t ls_engine_vocab_size(const ls_engine * eng);
uint32_t ls_engine_n_ctx(const ls_engine * eng);
void ls_engine_set_cancel(ls_engine * eng, bool cancel);
void ls_engine_set_cancel_cb(ls_engine * eng, ls_cancel_check_cb cb, const void * user_data);
bool ls_probe_gpu_memory(bool use_gpu, uint64_t * out_free_bytes, uint64_t * out_total_bytes);
bool ls_probe_gpu_device(
    bool use_gpu,
    uint64_t * out_free_bytes,
    uint64_t * out_total_bytes,
    char * out_backend_name,
    size_t backend_name_cap
);
/** Controls llama.cpp/ggml stderr verbosity (false suppresses INFO/DEBUG logs while keeping WARN/ERROR). */
void ls_set_log_verbose(bool verbose);

/** Pure sampling helper shared by eval_sample and tests (exact penalties -> top_k -> temp -> min_p -> top_p). */
int32_t ls_sample_from_logits(
    const float * logits,
    int32_t n_vocab,
    const ls_sample_params * sample_params
);

int32_t ls_sample_from_logits_with_history(
    const float * logits,
    int32_t n_vocab,
    const ls_sample_params * sample_params,
    const int32_t * penalty_tokens,
    int32_t n_penalty_tokens
);

/**
 * Pure speculative rejection sampling helper for tests (matches eval_verify_sample).
 * Buffer requirements:
 * - `logits_per_pos`: array of `k + 1` pointers, each pointing to `n_vocab` floats.
 * - `draft_tokens`: array of `k` token IDs (`k >= 0`).
 * - `out_accepted_count` and `out_next_token`: non-null output pointers.
 */
void ls_verify_sample_from_logits(
    const float * const * logits_per_pos,
    int32_t n_vocab,
    const int32_t * draft_tokens,
    int32_t k,
    const ls_sample_params * sample_params,
    int32_t * out_accepted_count,
    int32_t * out_next_token
);

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
);

/** Applies the model's embedded chat template (or ChatML fallback). Returns required bytes (negative on error). */
int32_t ls_engine_apply_chat_template(
    const ls_engine * eng,
    const ls_chat_message * messages,
    size_t n_messages,
    bool add_assistant_prompt,
    char * out_buf,
    int32_t out_cap
);

int32_t ls_engine_tokenize(
    const ls_engine * eng,
    const char * text,
    size_t text_len,
    int32_t * out_tokens,
    int32_t max_tokens,
    bool add_special,
    bool parse_special
);

int32_t ls_engine_token_to_piece(
    const ls_engine * eng,
    int32_t token,
    char * out_buf,
    int32_t buf_cap
);

bool ls_engine_token_is_eog(const ls_engine * eng, int32_t token);

void ls_engine_kv_clear(ls_engine * eng);
bool ls_engine_kv_trim(ls_engine * eng, int32_t pos_min);

/** Saves full KV memory state + `tokens[0..n_tokens)` to `path` via `llama_state_save_file`. */
int ls_engine_save_state(
    ls_engine * eng,
    const char * path,
    const int32_t * tokens,
    size_t n_tokens,
    char * err_buf,
    size_t err_cap
);

/** Restores full KV memory state + token sequence from `path` via `llama_state_load_file`. */
int ls_engine_load_state(
    ls_engine * eng,
    const char * path,
    int32_t * out_tokens,
    size_t cap_tokens,
    size_t * out_n_tokens,
    char * err_buf,
    size_t err_cap
);

int ls_engine_eval_greedy(
    ls_engine * eng,
    const int32_t * tokens,
    int32_t n_tokens,
    int32_t pos_start,
    int32_t * out_next_token,
    char * err_buf,
    size_t err_cap
);

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
);

int ls_engine_eval_verify_greedy(
    ls_engine * eng,
    const int32_t * tokens,
    int32_t n_tokens,
    int32_t pos_start,
    int32_t * out_argmax_tokens,
    char * err_buf,
    size_t err_cap
);

/**
 * Lossless speculative rejection sampling verification for temperature > 0 (or penalized greedy).
 * `tokens[0]` is the anchor token and `tokens[1..n_tokens-1]` are K draft tokens.
 * `penalty_tokens[0..n_penalty_tokens)` holds history up through `tokens[0]`.
 * Writes accepted draft count in `[0..K]` to `out_accepted_count` and the next
 * token (either the resampled correction token or the bonus token when all K
 * are accepted) to `out_next_token`.
 */
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
);

/** Compiles a GBNF grammar sampler against the engine's vocabulary. */
ls_grammar * ls_grammar_create(
    const ls_engine * eng,
    const char * grammar_str,
    const char * root_rule,
    char * err_buf,
    size_t err_cap
);

void ls_grammar_free(ls_grammar * grammar);

int ls_grammar_accept(
    ls_grammar * grammar,
    int32_t token,
    char * err_buf,
    size_t err_cap
);

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
);

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
);

#ifdef __cplusplus
}
#endif
