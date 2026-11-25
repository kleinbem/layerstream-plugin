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

}  // namespace ls_internal
