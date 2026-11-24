#pragma once

// Compatibility isolation layer for internal llama.cpp headers (`llama-model.h`).
// Keep ALL direct `llama_model` / `llama_hparams` field access in this single file
// so upstream llama.cpp struct changes only require updating one place.

#include "ggml.h"
#include "llama-model.h"
#include "llama.h"

#include <cstdint>
#include <string>

namespace ls_llama {

inline const auto & tensors_by_name(const llama_model * model) {
    return model->tensors_by_name;
}

inline bool is_tok_embd_tensor(
    const llama_model * model,
    const ggml_tensor * t,
    const std::string & name
) {
    return t == model->tok_embd || name == "token_embd.weight";
}

inline bool is_output_head_tensor(
    const llama_model * model,
    const ggml_tensor * t,
    const std::string & name
) {
    return t == model->output || name == "output.weight";
}

// Compute KV-cache memory budget directly from public `llama_hparams` fields
// rather than calling non-exported internal `llama_hparams` methods (required
// for Windows DLL linking against `llama.dll`).
inline uint64_t estimate_kv_cache_bytes(
    const llama_model * model,
    uint32_t n_layers,
    uint32_t n_ctx,
    int kv_cache_type
) {
    const double bytes_per_elem = (kv_cache_type == 1) ? 1.0625
                                : (kv_cache_type == 2) ? 0.5625
                                                       : 2.0;
    uint64_t kv_bytes = 0;
    for (uint32_t il = 0; il < n_layers; ++il) {
        const uint64_t n_head_kv = (il < model->hparams.n_head_kv_arr.size())
            ? model->hparams.n_head_kv_arr[il]
            : model->hparams.n_head_kv_arr[0];
        const uint64_t k_dim = static_cast<uint64_t>(model->hparams.n_embd_head_k_full) * n_head_kv;
        const uint64_t v_dim = static_cast<uint64_t>(model->hparams.n_embd_head_v_full) * n_head_kv;
        kv_bytes += static_cast<uint64_t>(
            static_cast<double>((k_dim + v_dim) * static_cast<uint64_t>(n_ctx)) * bytes_per_elem
        );
    }
    return kv_bytes;
}

// Workaround: `llama_model_params.no_alloc = true` skips weight allocation, and `llama_context`
// inherits `model.no_alloc()` (`params.no_alloc`) to skip KV/compute allocation unless cleared
// before `llama_init_from_model` (tracking: kleinbem/layerstream-plugin#8,
// upstream: ggml-org/llama.cpp#32893).
inline void clear_model_no_alloc(llama_model * model) {
    struct LsModelNoAllocReset : llama_model {
        static void clear(llama_model * m) {
            constexpr auto member = &LsModelNoAllocReset::params;
            (m->*member).no_alloc = false;
        }
    };
    LsModelNoAllocReset::clear(model);
}

}  // namespace ls_llama
