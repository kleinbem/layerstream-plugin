#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ls_engine ls_engine;

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

typedef int (*ls_prefetch_start_cb)(
    void * user_data,
    int32_t layer_idx,
    const char * const * tensor_names,
    const uint64_t * tensor_offsets,
    const uint64_t * tensor_sizes,
    size_t tensor_count,
    uint8_t * dst_ptr,
    size_t dst_cap
);

typedef int (*ls_prefetch_wait_cb)(
    void * user_data,
    int32_t layer_idx
);

typedef struct ls_engine_config {
    const char * model_path;
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_threads;
    int32_t resident_layers;
    uint64_t vram_budget_bytes;
    /** KV cache type: 0 = F16 (default), 1 = Q8_0 (2x smaller), 2 = Q4_0 (4x smaller). */
    int32_t kv_cache_type;
    bool use_gpu;
    bool full_resident_reference;

    void * io_user_data;
    ls_load_tensors_cb load_tensors_cb;
    ls_prefetch_start_cb prefetch_start_cb;
    ls_prefetch_wait_cb prefetch_wait_cb;
} ls_engine_config;

typedef struct ls_engine_stats {
    uint32_t n_layers;
    uint32_t resident_layers;
    uint32_t streamed_layers;
    uint64_t resident_bytes;
    uint64_t slot_bytes;
    uint64_t total_layer_loads;
    bool is_uma_zero_copy;
    bool using_gpu;
    bool can_speculate;
} ls_engine_stats;

typedef struct ls_chat_message {
    const char * role;
    const char * content;
} ls_chat_message;

typedef struct ls_sample_params {
    float temperature;
    float top_p;
    int32_t top_k;
    float repeat_penalty;
    uint32_t seed;
} ls_sample_params;

ls_engine * ls_engine_create(const ls_engine_config * cfg, char * err_buf, size_t err_cap);
void ls_engine_free(ls_engine * eng);
void ls_engine_get_stats(const ls_engine * eng, ls_engine_stats * out);

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

#ifdef __cplusplus
}
#endif
