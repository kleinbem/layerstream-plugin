#pragma once

#include "ggml-backend.h"
#include "llama.h"
#include "stream_engine.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace ls_internal {

// Implemented in backend_loader.cpp
void ensure_llama_initialized();
void set_error(char * buf, size_t cap, const std::string & msg);
int gpu_device_priority(ggml_backend_dev_t dev);
ggml_backend_dev_t select_primary_gpu_dev();
bool select_target_buft(
    bool use_gpu,
    ggml_backend_dev_t & out_gpu_dev,
    ggml_backend_buffer_type_t & out_buft,
    bool & out_using_gpu,
    bool & out_is_uma_zero_copy,
    bool & out_is_unified_memory,
    bool & out_is_cuda_backend,
    std::string & out_backend_name,
    std::string & out_err
);

// Implemented in stream_engine.cpp (keeps `struct ls_engine` encapsulated)
llama_context * engine_ctx(const ls_engine * eng);
const llama_vocab * engine_vocab(const ls_engine * eng);
void engine_prepare_decode_pass(ls_engine * eng);
bool engine_is_cancelled(const ls_engine * eng);
bool engine_eval_failed(const ls_engine * eng);
std::string engine_eval_error_msg(const ls_engine * eng);
bool engine_profile_enabled(const ls_engine * eng);
void engine_record_decode_pass_us(ls_engine * eng, uint64_t elapsed_us);

// Implemented in sampler_eval.cpp (shared with grammar_eval.cpp)
int32_t argmax_logits(const float * logits, int32_t n_vocab);
int report_eval_failure(
    const ls_engine * eng,
    int rc,
    const char * fallback_msg,
    char * err_buf,
    size_t err_cap
);

}  // namespace ls_internal
