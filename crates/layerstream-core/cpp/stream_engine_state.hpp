#pragma once

#include "cuda_shim.hpp"
#include "engine_internal.hpp"
#include "llama_compat.hpp"
#include "stream_engine.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace {

static size_t align_up(size_t value, size_t alignment) {
    if (alignment <= 1) {
        return value;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

// Parses "blk.<L>." prefix and returns layer index L, or -1 if not a layer tensor.
static int32_t parse_layer_index(const char * name) {
    if (!name || std::strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    const char * p = name + 4;
    if (*p < '0' || *p > '9') {
        return -1;
    }
    int32_t val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        ++p;
    }
    if (*p != '.') {
        return -1;
    }
    return val;
}

// Parses "ffn_moe_topk-<L>" and returns layer index L, or -1 if not a router top-K node.
static int32_t parse_moe_topk_layer(const char * name) {
    static constexpr const char kPrefix[] = "ffn_moe_topk-";
    static constexpr size_t kPrefixLen = sizeof(kPrefix) - 1;
    if (!name || std::strncmp(name, kPrefix, kPrefixLen) != 0) {
        return -1;
    }
    const char * p = name + kPrefixLen;
    if (*p < '0' || *p > '9') {
        return -1;
    }
    int32_t val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        ++p;
    }
    if (*p != '\0') {
        return -1;
    }
    return val;
}

// Returns true when `t` is a 3D MoE expert weight tensor (`blk.<L>.*_exps.*`)
// whose dim-2 stride lays out each expert contiguously in GGUF & backend memory.
static bool is_streamable_moe_expert_tensor(const std::string & name, const ggml_tensor * t) {
    if (!t || name.find("_exps.") == std::string::npos) {
        return false;
    }
    if (t->ne[2] <= 1 || t->ne[3] != 1 || t->nb[2] == 0) {
        return false;
    }
    return static_cast<uint64_t>(t->ne[2]) * static_cast<uint64_t>(t->nb[2]) == ggml_nbytes(t);
}

}  // namespace

struct TensorBinding {
    std::string name;
    ggml_tensor * tensor = nullptr;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    uint64_t alloc_size = 0;
};

struct LayerPlan {
    int32_t layer_idx = -1;
    int slot_idx = -1;
    bool is_moe_sparse = false;
    uint32_t n_experts = 0;
    uint64_t total_bytes = 0;
    std::vector<TensorBinding> tensors;
    std::vector<TensorBinding> moe_non_expert_tensors;
    std::vector<const char *> name_ptrs;
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> sizes;

    void finalize_arrays() {
        name_ptrs.clear();
        offsets.clear();
        sizes.clear();
        name_ptrs.reserve(tensors.size());
        offsets.reserve(tensors.size());
        sizes.reserve(tensors.size());
        for (const auto & tb : tensors) {
            name_ptrs.push_back(tb.name.c_str());
            offsets.push_back(tb.offset);
            sizes.push_back(tb.nbytes);
        }
    }
};

struct PrefetchJob {
    int32_t layer_idx = -1;
    int slot = 0;
};

struct MoeWorkRequest {
    int32_t layer_idx = -1;
    bool load_full = false;
    const std::vector<ls_tensor_slice_req> * reqs = nullptr;
    const std::vector<ls_tensor_slice_req> * packed_reqs = nullptr;
    // Output populated by worker thread (for non-CUDA discrete GPU upload on eval thread).
    std::vector<ls_span_slice> direct_slices;
    int n_direct = 0;
    int rc = 0;
};

struct ls_engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;

    bool full_resident_reference = false;
    bool using_gpu = false;
    bool is_uma_zero_copy = true;
    bool is_unified_memory = true;
    bool is_cuda_backend = false;
    bool can_speculate = false;
    uint32_t rs_rollback_max = 0;
    llama_context * ctx_mtp = nullptr;
    uint32_t mtp_layers = 0;
    int32_t mtp_n_embd = 0;
    bool mtp_is_mem_shared = false;
    bool mtp_chain_heads = false;
    std::vector<float> mtp_pending_h;
    std::vector<float> mtp_verify_h;
    int32_t mtp_verify_rows = 0;
    bool head_streamed = false;
    bool moe_active_streaming = false;
    std::string backend_name = "CPU";

    uint32_t n_layers = 0;
    uint32_t stream_end_layer = 0;
    uint32_t resident_layers = 0;
    uint32_t streamed_layers = 0;
    uint32_t streamed_dense_layers = 0;
    uint32_t streamed_moe_layers = 0;
    uint32_t moe_layers = 0;
    uint64_t resident_bytes = 0;
    uint64_t cpu_resident_bytes = 0;
    uint64_t slot_bytes = 0;
    uint64_t expert_slot_bytes = 0;
    std::atomic<uint64_t> total_layer_loads{0};
    std::atomic<uint64_t> moe_experts_loaded{0};
    std::atomic<uint64_t> moe_experts_total{0};
    std::atomic<bool> cancel_requested{false};
    ls_cancel_check_cb cancel_cb = nullptr;
    const void * cancel_cb_user_data = nullptr;

    bool is_cancelled() const {
        if (cancel_requested.load(std::memory_order_relaxed)) {
            return true;
        }
        return cancel_cb != nullptr && cancel_cb(cancel_cb_user_data);
    }

    // Real backend buffers owned by us (never freed by llama_model_free because no_alloc=true).
    ggml_backend_buffer_t resident_buf = nullptr;
    ggml_backend_buffer_t cpu_resident_buf = nullptr;
    ggml_backend_buffer_t slot_buf[2] = {nullptr, nullptr};
    ggml_backend_buffer_t expert_slot_buf = nullptr;

    // Double-buffered host staging buffers for discrete GPUs (page-aligned / cudaHostAlloc),
    // one per ping-pong slot so NVMe read(L+1) into `pin_buf[slot]` overlaps with
    // in-flight H2D DMA(L) from `pin_buf[1 - slot]` and GPU compute(L-1).
    // Exclusively written by `worker_thread` (never concurrently by eval thread).
    uint8_t * pin_buf[2] = {nullptr, nullptr};
    size_t pin_cap = 0;
#ifdef LAYERSTREAM_HAS_CUDA_RT
    bool pin_is_cuda_host = false;
    ls_cuda::Stream copy_stream = nullptr;
    ls_cuda::Event slot_event[2] = {nullptr, nullptr};
    bool slot_h2d_recorded[2] = {false, false}; // exclusively accessed on worker_thread
#endif

    // Layer plans indexed by layer index [0..stream_end_layer).
    // When `head_streamed` is true, virtual layer `n_layers` holds `output.weight`.
    std::vector<LayerPlan> layer_plans;
    LayerPlan resident_plan;
    LayerPlan cpu_resident_plan;

    // Next streamed dense layer chain (handles both pure-dense and mixed dense+MoE models).
    int32_t first_dense_stream_layer = -1;
    std::vector<int32_t> next_dense_layer;

    // O(1) map from any streamed dense ggml_tensor* to its layer index.
    std::unordered_map<const ggml_tensor *, int32_t> tensor_to_layer;
    // O(1) map from any streamed MoE 3D expert ggml_tensor* to its layer index.
    std::unordered_map<const ggml_tensor *, int32_t> tensor_to_moe_layer;

    // IO callbacks into Rust (invoked exclusively from `worker_thread` once running).
    void * io_user_data = nullptr;
    ls_load_tensors_cb load_tensors_cb = nullptr;
    ls_load_layer_direct_cb load_layer_direct_cb = nullptr;
    ls_load_slices_cb load_slices_cb = nullptr;

    // Per-slot state for async background worker (dense ping-pong layers).
    std::mutex slot_mu;
    std::condition_variable slot_cv;
    int32_t slot_loaded_layer[2] = {-1, -1};
    int32_t slot_inflight_layer[2] = {-1, -1};
    int slot_error[2] = {0, 0};
    // For non-CUDA discrete GPUs (e.g. Vulkan): worker fills `pin_buf` and marks
    // `slot_needs_h2d[slot] = true`; eval thread performs `ggml_backend_tensor_set`
    // inside `wait_slot_ready` while the GPU backend is idle/synchronized.
    bool slot_needs_h2d[2] = {false, false};
    ls_span_slice slot_h2d_direct_slices[2][128];
    int slot_h2d_n_direct[2] = {0, 0};

    // Single-owner I/O worker queue (handles both async dense prefetches and sync MoE loads).
    std::mutex worker_mu;
    std::condition_variable worker_cv;
    std::vector<PrefetchJob> pending_jobs;
    MoeWorkRequest * active_moe_req = nullptr;
    bool moe_io_done = false;
    bool moe_upload_done = false;
    bool stop_worker = false;
    std::thread worker_thread;

    void init_mtp_context(uint32_t n_ctx, uint32_t n_batch, uint32_t n_threads, int32_t kv_cache_type) {
        if (!model || !ctx) {
            return;
        }
        const int32_t n_mtp = llama_model_n_layer_nextn(model);
        if (n_mtp <= 0) {
            return;
        }
        const int32_t n_embd = llama_model_n_embd_out(model);
        if (n_embd <= 0) {
            return;
        }

        llama_context_params mp = llama_context_default_params();
        mp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        mp.n_ctx = n_ctx;
        mp.n_batch = n_batch;
        mp.n_ubatch = n_batch;
        mp.n_threads = static_cast<int32_t>(n_threads);
        mp.n_threads_batch = static_cast<int32_t>(n_threads);
        if (!using_gpu) {
            mp.offload_kqv = false;
            mp.op_offload = false;
        }
        if (kv_cache_type == 1) {
            mp.type_k = GGML_TYPE_Q8_0;
            mp.type_v = GGML_TYPE_Q8_0;
            mp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        } else if (kv_cache_type == 2) {
            mp.type_k = GGML_TYPE_Q4_0;
            mp.type_v = GGML_TYPE_Q4_0;
            mp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        } else {
            mp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        }
        mp.abort_callback = &ls_engine::abort_cb;
        mp.abort_callback_data = this;

        llama_context * mctx = llama_init_from_model(model, mp);
        if (!mctx) {
            return;
        }
        ctx_mtp = mctx;
        mtp_layers = static_cast<uint32_t>(n_mtp);
        mtp_n_embd = n_embd;
        mtp_is_mem_shared = (llama_get_ctx_other(ctx_mtp) == ctx);
        mtp_chain_heads = (n_mtp > 1 && !mtp_is_mem_shared);
        llama_set_embeddings_nextn(ctx, true, mtp_is_mem_shared);
        llama_set_embeddings_nextn(ctx_mtp, true, true);
        mtp_pending_h.assign(static_cast<size_t>(mtp_n_embd), 0.0f);
        mtp_verify_h.clear();
        mtp_verify_rows = 0;
    }

    void mtp_reset_state() {
        if (ctx_mtp && !mtp_is_mem_shared) {
            llama_memory_clear(llama_get_memory(ctx_mtp), true);
        }
        std::fill(mtp_pending_h.begin(), mtp_pending_h.end(), 0.0f);
        mtp_verify_rows = 0;
    }

    void mtp_on_target_decode(const int32_t * tokens, int32_t n_tokens, int32_t pos_start) {
        if (!ctx_mtp || mtp_n_embd <= 0 || n_tokens <= 0 || !tokens) {
            return;
        }
        const size_t n_embd_sz = static_cast<size_t>(mtp_n_embd);
        const size_t row_bytes = n_embd_sz * sizeof(float);

        if (!mtp_is_mem_shared) {
            const float * h_tgt = llama_get_embeddings_nextn(ctx);
            if (!h_tgt) {
                return;
            }
            llama_batch_ext * mb = llama_batch_ext_init(ctx_mtp);
            if (!mb) {
                return;
            }
            for (int32_t k = 0; k < n_tokens; ++k) {
                const int32_t idx = llama_batch_ext_add_token(mb, 0, tokens[k]);
                if (idx < 0) {
                    llama_batch_ext_free(mb);
                    return;
                }
                const llama_pos p = pos_start + k;
                llama_batch_ext_set_pos(mb, idx, &p);
                llama_batch_ext_set_output_logits(mb, idx, false);
                const float * h_row = (k == 0)
                    ? mtp_pending_h.data()
                    : h_tgt + static_cast<size_t>(k - 1) * n_embd_sz;
                llama_batch_ext_set_embd_state(mb, idx, llama_embd{h_row, 1, n_embd_sz});
            }

            llama_memory_t mem_mtp = llama_get_memory(ctx_mtp);
            const uint32_t heads = mtp_chain_heads ? mtp_layers : 1u;
            for (uint32_t head = 0; head < heads; ++head) {
                if (mtp_chain_heads) {
                    llama_memory_seq_rm(mem_mtp, 0, pos_start, -1);
                    llama_set_nextn_layer_offset(ctx_mtp, static_cast<int32_t>(head));
                }
                if (llama_process(ctx_mtp, LLAMA_PROCESS_TYPE_DECODE, mb) != 0) {
                    break;
                }
            }
            if (mtp_chain_heads) {
                llama_set_nextn_layer_offset(ctx_mtp, 0);
            }
            llama_batch_ext_free(mb);
        }

        mtp_verify_rows = n_tokens;
        mtp_verify_h.resize(static_cast<size_t>(n_tokens) * n_embd_sz);
        for (int32_t i = 0; i < n_tokens; ++i) {
            const float * h = llama_get_embeddings_nextn_ith(ctx, i);
            if (h) {
                std::memcpy(mtp_verify_h.data() + static_cast<size_t>(i) * n_embd_sz, h, row_bytes);
            }
        }
        if (n_tokens > 0) {
            std::memcpy(
                mtp_pending_h.data(),
                mtp_verify_h.data() + static_cast<size_t>(n_tokens - 1) * n_embd_sz,
                row_bytes
            );
        }
    }

    // Active decode pass tracking (accessed only from eval thread during llama_decode).
    int32_t active_stream_layer = -1;
    int32_t pending_release_layer = -1;
    int32_t highest_enqueued_layer = -1;
    int32_t seen_moe_topk_layer = -1;
    ggml_tensor * seen_moe_topk_tensor = nullptr;
    std::unordered_set<const ggml_tensor *> moe_weight_chain;
    int32_t pending_moe_layer = -1;
    ggml_tensor * pending_moe_topk_tensor = nullptr;
    int32_t loaded_moe_layer = -1;
    int32_t moe_split_completed_layer = -1;
    std::vector<uint8_t> loaded_moe_mask;
    bool eval_failed = false;
    std::string eval_error_msg;

    // Optional microsecond profiling (enabled via LAYERSTREAM_PROFILE=1).
    bool profile_enabled = false;
    std::atomic<uint64_t> prof_io_read_us{0};
    std::atomic<uint64_t> prof_h2d_sync_us{0};
    std::atomic<uint64_t> prof_cb_wait_us{0};
    std::atomic<uint64_t> prof_pass_count{0};
    std::atomic<uint64_t> prof_decode_total_us{0};

    bool uses_deferred_eval_h2d() const {
#ifdef LAYERSTREAM_HAS_CUDA_RT
        return !is_uma_zero_copy && (copy_stream == nullptr);
#else
        return !is_uma_zero_copy;
#endif
    }

    // Runs exclusively on `worker_thread`.
    int load_layer_into_slot(int32_t layer_idx, int slot, bool & out_needs_eval_h2d) {
        out_needs_eval_h2d = false;
        const auto & plan = layer_plans[layer_idx];
        if (plan.tensors.empty()) {
            return 0;
        }

        if (is_uma_zero_copy) {
            using Clock = std::chrono::steady_clock;
            auto t_io0 = profile_enabled ? Clock::now() : Clock::time_point{};
            auto * dst = static_cast<uint8_t *>(ggml_backend_buffer_get_base(slot_buf[slot]));
            int rc = load_tensors_cb(
                io_user_data,
                layer_idx,
                plan.name_ptrs.data(),
                plan.offsets.data(),
                plan.sizes.data(),
                plan.tensors.size(),
                dst,
                slot_bytes
            );
            if (rc != 0) {
                return rc;
            }
            for (const auto & tb : plan.tensors) {
                if (tb.alloc_size > tb.nbytes) {
                    std::memset(dst + tb.offset + tb.nbytes, 0, static_cast<size_t>(tb.alloc_size - tb.nbytes));
                }
            }
            if (profile_enabled) {
                prof_io_read_us.fetch_add(
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t_io0).count()),
                    std::memory_order_relaxed
                );
            }
        } else {
            using Clock = std::chrono::steady_clock;
            auto t_h2d_wait0 = profile_enabled ? Clock::now() : Clock::time_point{};
            uint8_t * slot_pin = pin_buf[slot];
#ifdef LAYERSTREAM_HAS_CUDA_RT
            // Only synchronize this SAME slot's previous H2D DMA (2 layers ago), allowing
            // NVMe read(L+1) into `pin_buf[slot]` to overlap with in-flight H2D(L) from `pin_buf[1 - slot]`.
            if (copy_stream != nullptr && slot_h2d_recorded[slot] && slot_event[slot] != nullptr) {
                if (ls_cuda::event_synchronize(slot_event[slot]) != ls_cuda::kSuccess) {
                    return -9;
                }
            }
#endif
            auto t_io0 = profile_enabled ? Clock::now() : Clock::time_point{};
            if (profile_enabled) {
                prof_h2d_sync_us.fetch_add(
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t_io0 - t_h2d_wait0).count()),
                    std::memory_order_relaxed
                );
            }
            ls_span_slice direct_slices[128] = {};
            int n_direct = 0;
            if (load_layer_direct_cb != nullptr) {
                n_direct = load_layer_direct_cb(
                    io_user_data,
                    layer_idx,
                    plan.name_ptrs.data(),
                    plan.offsets.data(),
                    plan.sizes.data(),
                    plan.tensors.size(),
                    slot_pin,
                    pin_cap,
                    direct_slices,
                    128
                );
                if (n_direct < 0) {
                    return n_direct;
                }
            }

            if (n_direct == 0) {
                int rc = load_tensors_cb(
                    io_user_data,
                    layer_idx,
                    plan.name_ptrs.data(),
                    plan.offsets.data(),
                    plan.sizes.data(),
                    plan.tensors.size(),
                    slot_pin,
                    pin_cap
                );
                if (rc != 0) {
                    return rc;
                }
            }
            if (profile_enabled) {
                auto t_io1 = Clock::now();
                prof_io_read_us.fetch_add(
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t_io1 - t_io0).count()),
                    std::memory_order_relaxed
                );
            }

#ifdef LAYERSTREAM_HAS_CUDA_RT
            if (copy_stream != nullptr) {
                auto * gpu_dst = static_cast<uint8_t *>(ggml_backend_buffer_get_base(slot_buf[slot]));
                bool submitted_h2d = false;
                auto fence_on_err = [&]() {
                    if (submitted_h2d) {
                        (void)ls_cuda::stream_synchronize(copy_stream);
                    }
                };
                if (n_direct > 0) {
                    for (int i = 0; i < n_direct; ++i) {
                        int ce = ls_cuda::memcpy_async(
                            gpu_dst + direct_slices[i].dst_offset,
                            slot_pin + direct_slices[i].src_offset,
                            direct_slices[i].len,
                            ls_cuda::kMemcpyHostToDevice,
                            copy_stream
                        );
                        if (ce != ls_cuda::kSuccess) {
                            fence_on_err();
                            return -10;
                        }
                        submitted_h2d = true;
                    }
                } else {
                    int ce = ls_cuda::memcpy_async(
                        gpu_dst,
                        slot_pin,
                        plan.total_bytes,
                        ls_cuda::kMemcpyHostToDevice,
                        copy_stream
                    );
                    if (ce != ls_cuda::kSuccess) {
                        return -10;
                    }
                    submitted_h2d = true;
                }
                // Zero any quantized row padding tails between heterogeneous layers reusing slot_buf.
                for (const auto & tb : plan.tensors) {
                    if (tb.alloc_size > tb.nbytes) {
                        if (ls_cuda::memset_async(
                                gpu_dst + tb.offset + tb.nbytes,
                                0,
                                static_cast<size_t>(tb.alloc_size - tb.nbytes),
                                copy_stream
                            ) != ls_cuda::kSuccess) {
                            fence_on_err();
                            return -10;
                        }
                    }
                }
                int ce = ls_cuda::event_record(slot_event[slot], copy_stream);
                if (ce != ls_cuda::kSuccess) {
                    fence_on_err();
                    return -11;
                }
                slot_h2d_recorded[slot] = true;
            } else
#endif
            {
                // Non-CUDA discrete backend (e.g. Vulkan): defer `ggml_backend_tensor_set` to
                // the eval thread inside `wait_slot_ready` so `ggml_backend_vk_context` is
                // never touched concurrently with `ggml_backend_graph_compute_async`.
                slot_h2d_n_direct[slot] = n_direct;
                if (n_direct > 0) {
                    std::memcpy(
                        slot_h2d_direct_slices[slot],
                        direct_slices,
                        static_cast<size_t>(n_direct) * sizeof(ls_span_slice)
                    );
                }
                out_needs_eval_h2d = true;
            }
        }

        total_layer_loads.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    // Runs exclusively on `worker_thread` for MoE I/O (and CUDA H2D).
    int execute_moe_work_on_worker(MoeWorkRequest & req) {
        const auto & plan = layer_plans[req.layer_idx];
        req.n_direct = 0;

        if (req.load_full) {
            if (is_uma_zero_copy) {
                auto * dst = static_cast<uint8_t *>(ggml_backend_buffer_get_base(expert_slot_buf));
                int rc = load_tensors_cb(
                    io_user_data,
                    req.layer_idx,
                    plan.name_ptrs.data(),
                    plan.offsets.data(),
                    plan.sizes.data(),
                    plan.tensors.size(),
                    dst,
                    expert_slot_bytes
                );
                if (rc != 0) {
                    return rc;
                }
                for (const auto & tb : plan.tensors) {
                    if (tb.alloc_size > tb.nbytes) {
                        std::memset(dst + tb.offset + tb.nbytes, 0, static_cast<size_t>(tb.alloc_size - tb.nbytes));
                    }
                }
                return 0;
            }

#ifdef LAYERSTREAM_HAS_CUDA_RT
            if (copy_stream != nullptr && slot_h2d_recorded[0] && slot_event[0] != nullptr) {
                if (ls_cuda::event_synchronize(slot_event[0]) != ls_cuda::kSuccess) {
                    return -21;
                }
            }
#endif
            req.direct_slices.resize(std::max<size_t>(plan.tensors.size() * 2, 64));
            int n_direct = 0;
            uint8_t * moe_pin = pin_buf[0];
            if (load_layer_direct_cb != nullptr) {
                n_direct = load_layer_direct_cb(
                    io_user_data,
                    req.layer_idx,
                    plan.name_ptrs.data(),
                    plan.offsets.data(),
                    plan.sizes.data(),
                    plan.tensors.size(),
                    moe_pin,
                    pin_cap,
                    req.direct_slices.data(),
                    req.direct_slices.size()
                );
                if (n_direct < 0) {
                    return n_direct;
                }
            }
            if (n_direct == 0) {
                int rc = load_tensors_cb(
                    io_user_data,
                    req.layer_idx,
                    plan.name_ptrs.data(),
                    plan.offsets.data(),
                    plan.sizes.data(),
                    plan.tensors.size(),
                    moe_pin,
                    pin_cap
                );
                if (rc != 0) {
                    return rc;
                }
            }
            req.n_direct = n_direct;

#ifdef LAYERSTREAM_HAS_CUDA_RT
            if (copy_stream != nullptr) {
                auto * gpu_dst = static_cast<uint8_t *>(ggml_backend_buffer_get_base(expert_slot_buf));
                bool submitted_h2d = false;
                auto fence_on_err = [&]() {
                    if (submitted_h2d) {
                        (void)ls_cuda::stream_synchronize(copy_stream);
                    }
                };
                if (n_direct > 0) {
                    for (int i = 0; i < n_direct; ++i) {
                        if (ls_cuda::memcpy_async(
                                gpu_dst + req.direct_slices[i].dst_offset,
                                moe_pin + req.direct_slices[i].src_offset,
                                req.direct_slices[i].len,
                                ls_cuda::kMemcpyHostToDevice,
                                copy_stream
                            ) != ls_cuda::kSuccess) {
                            fence_on_err();
                            return -22;
                        }
                        submitted_h2d = true;
                    }
                } else {
                    if (ls_cuda::memcpy_async(
                            gpu_dst,
                            moe_pin,
                            plan.total_bytes,
                            ls_cuda::kMemcpyHostToDevice,
                            copy_stream
                        ) != ls_cuda::kSuccess) {
                        return -22;
                    }
                    submitted_h2d = true;
                }
                for (const auto & tb : plan.tensors) {
                    if (tb.alloc_size > tb.nbytes) {
                        if (ls_cuda::memset_async(
                                gpu_dst + tb.offset + tb.nbytes,
                                0,
                                static_cast<size_t>(tb.alloc_size - tb.nbytes),
                                copy_stream
                            ) != ls_cuda::kSuccess) {
                            fence_on_err();
                            return -22;
                        }
                    }
                }
                if (ls_cuda::stream_synchronize(copy_stream) != ls_cuda::kSuccess) {
                    return -23;
                }
            }
#endif
            return 0;
        }

        // Sparse active-expert streaming.
        const auto & reqs = *req.reqs;
        if (is_uma_zero_copy) {
            auto * dst = static_cast<uint8_t *>(ggml_backend_buffer_get_base(expert_slot_buf));
            return load_slices_cb(io_user_data, reqs.data(), reqs.size(), dst, expert_slot_bytes);
        }

#ifdef LAYERSTREAM_HAS_CUDA_RT
        if (copy_stream != nullptr && slot_h2d_recorded[0] && slot_event[0] != nullptr) {
            if (ls_cuda::event_synchronize(slot_event[0]) != ls_cuda::kSuccess) {
                return -24;
            }
        }
#endif
        uint8_t * moe_pin = pin_buf[0];
        const auto & packed_reqs = *req.packed_reqs;
        int rc = load_slices_cb(io_user_data, packed_reqs.data(), packed_reqs.size(), moe_pin, pin_cap);
        if (rc != 0) {
            return rc;
        }

#ifdef LAYERSTREAM_HAS_CUDA_RT
        if (copy_stream != nullptr) {
            auto * gpu_dst = static_cast<uint8_t *>(ggml_backend_buffer_get_base(expert_slot_buf));
            bool submitted_h2d = false;
            for (size_t i = 0; i < reqs.size(); ++i) {
                if (ls_cuda::memcpy_async(
                        gpu_dst + reqs[i].dst_offset,
                        moe_pin + packed_reqs[i].dst_offset,
                        reqs[i].len,
                        ls_cuda::kMemcpyHostToDevice,
                        copy_stream
                    ) != ls_cuda::kSuccess) {
                    if (submitted_h2d) {
                        (void)ls_cuda::stream_synchronize(copy_stream);
                    }
                    return -25;
                }
                submitted_h2d = true;
            }
            if (ls_cuda::stream_synchronize(copy_stream) != ls_cuda::kSuccess) {
                return -26;
            }
        }
#endif
        return 0;
    }

    int load_moe_experts(int32_t layer_idx, const std::vector<int32_t> & active_experts, bool force_full) {
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layer_plans.size()) {
            return -20;
        }
        const auto & plan = layer_plans[layer_idx];
        if (!plan.is_moe_sparse || plan.tensors.empty() || !expert_slot_buf) {
            return 0;
        }

        if (loaded_moe_layer != layer_idx) {
            loaded_moe_layer = layer_idx;
            loaded_moe_mask.assign(plan.n_experts, 0);
        }

        std::vector<int32_t> missing_experts;
        if (!force_full) {
            missing_experts.reserve(active_experts.size());
            for (int32_t e : active_experts) {
                if (e >= 0 && static_cast<uint32_t>(e) < plan.n_experts && loaded_moe_mask[e] == 0) {
                    missing_experts.push_back(e);
                }
            }
            if (missing_experts.empty()) {
                moe_experts_total.fetch_add(plan.n_experts, std::memory_order_relaxed);
                return 0;
            }
        }

        // Switch to a single sequential full-layer read once > 50% of a layer's experts are
        // missing (e.g., long prompt prefills), while keeping 1-token decode and multi-token
        // speculative verification batches on the sparse + TinyLFU hot-expert cache path.
        const bool load_full = force_full ||
            (missing_experts.size() * 2 > static_cast<size_t>(plan.n_experts)) ||
            (load_slices_cb == nullptr);

        using Clock = std::chrono::steady_clock;
        auto t0 = profile_enabled ? Clock::now() : Clock::time_point{};

        std::vector<ls_tensor_slice_req> reqs;
        std::vector<ls_tensor_slice_req> packed_reqs;
        if (!load_full) {
            reqs.reserve(plan.tensors.size() * missing_experts.size());
            for (uint32_t tidx = 0; tidx < static_cast<uint32_t>(plan.tensors.size()); ++tidx) {
                const auto & tb = plan.tensors[tidx];
                const uint64_t exp_stride = static_cast<uint64_t>(tb.tensor->nb[2]);
                for (int32_t e : missing_experts) {
                    const uint64_t exp_off = static_cast<uint64_t>(e) * exp_stride;
                    reqs.push_back(ls_tensor_slice_req{
                        tb.name.c_str(),
                        exp_off,
                        tb.offset + exp_off,
                        exp_stride,
                        tidx,
                    });
                }
            }
            if (!is_uma_zero_copy) {
                packed_reqs.reserve(reqs.size());
                uint64_t packed_off = 0;
                for (const auto & r : reqs) {
                    packed_reqs.push_back(ls_tensor_slice_req{
                        r.tensor_name,
                        r.src_offset_in_tensor,
                        packed_off,
                        r.len,
                        r.tensor_idx,
                    });
                    packed_off += r.len;
                }
                if (packed_off > pin_cap) {
                    return -27;
                }
            }
        }

        MoeWorkRequest work;
        work.layer_idx = layer_idx;
        work.load_full = load_full;
        work.reqs = &reqs;
        work.packed_reqs = &packed_reqs;

        // Submit priority job to the single owner `worker_thread` so `pin_buf`,
        // `LayerIoPool`, and CUDA `copy_stream`/`last_h2d_slot` are never touched concurrently.
        // Deferred dense-slot invalidation (Vulkan) happens inside `worker_loop` right when
        // the worker claims this MoE request, after any preceding dense read has published.
        const bool deferred_h2d = uses_deferred_eval_h2d();
        {
            std::unique_lock<std::mutex> wlk(worker_mu);
            worker_cv.wait(wlk, [&]() { return active_moe_req == nullptr || stop_worker; });
            if (stop_worker) {
                return -28;
            }
            active_moe_req = &work;
            moe_io_done = false;
            moe_upload_done = false;
            worker_cv.notify_all();
            worker_cv.wait(wlk, [&]() { return moe_io_done || stop_worker; });
        }

        int rc = work.rc;
        if (deferred_h2d) {
            if (rc == 0) {
                // Perform non-CUDA (Vulkan) GPU uploads on the eval thread while GPU is idle
                // and `worker_thread` is paused waiting for `moe_upload_done`.
                if (load_full) {
                    if (work.n_direct > 0) {
                        for (int i = 0; i < work.n_direct; ++i) {
                            const uint32_t tidx = work.direct_slices[i].tensor_idx;
                            if (tidx < plan.tensors.size()) {
                                ggml_backend_tensor_set(
                                    plan.tensors[tidx].tensor,
                                    pin_buf[0] + work.direct_slices[i].src_offset,
                                    0,
                                    work.direct_slices[i].len
                                );
                            }
                        }
                    } else {
                        for (const auto & tb : plan.tensors) {
                            ggml_backend_tensor_set(tb.tensor, pin_buf[0] + tb.offset, 0, tb.nbytes);
                        }
                    }
                } else {
                    for (size_t i = 0; i < reqs.size(); ++i) {
                        const auto & tb = plan.tensors[reqs[i].tensor_idx];
                        ggml_backend_tensor_set(
                            tb.tensor,
                            pin_buf[0] + packed_reqs[i].dst_offset,
                            reqs[i].src_offset_in_tensor,
                            reqs[i].len
                        );
                    }
                }
            }

            // Release `worker_thread` and wait for it to clear `active_moe_req` before
            // `&work` leaves this stack frame (prevents lost-wakeup between consecutive MoE layers).
            std::unique_lock<std::mutex> wlk(worker_mu);
            moe_upload_done = true;
            worker_cv.notify_all();
            worker_cv.wait(wlk, [&]() { return active_moe_req == nullptr || stop_worker; });
        }

        if (rc != 0) {
            return rc;
        }

        if (load_full) {
            std::fill(loaded_moe_mask.begin(), loaded_moe_mask.end(), 1);
            moe_experts_loaded.fetch_add(plan.n_experts, std::memory_order_relaxed);
            moe_experts_total.fetch_add(plan.n_experts, std::memory_order_relaxed);
        } else {
            for (int32_t e : missing_experts) {
                loaded_moe_mask[e] = 1;
            }
            moe_experts_loaded.fetch_add(missing_experts.size(), std::memory_order_relaxed);
            moe_experts_total.fetch_add(plan.n_experts, std::memory_order_relaxed);
        }

        if (profile_enabled) {
            const uint64_t dt = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count()
            );
            prof_io_read_us.fetch_add(dt, std::memory_order_relaxed);
            prof_cb_wait_us.fetch_add(dt, std::memory_order_relaxed);
        }
        total_layer_loads.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    void worker_loop() {
        while (true) {
            PrefetchJob dense_job;
            MoeWorkRequest * moe_req = nullptr;
            {
                std::unique_lock<std::mutex> lk(worker_mu);
                worker_cv.wait(lk, [&]() {
                    if (stop_worker || (active_moe_req != nullptr && !moe_io_done)) {
                        return true;
                    }
                    if (!pending_jobs.empty()) {
                        if (!uses_deferred_eval_h2d()) {
                            return true;
                        }
                        // On non-CUDA discrete GPU (Vulkan), `pin_buf[0]` and `pin_buf[1]` are
                        // independent, so wake if ANY queued job targets a free staging slot.
                        std::lock_guard<std::mutex> slk(slot_mu);
                        for (const auto & job : pending_jobs) {
                            if (!slot_needs_h2d[job.slot]) {
                                return true;
                            }
                        }
                        return false;
                    }
                    return false;
                });
                if (stop_worker) {
                    return;
                }
                if (active_moe_req != nullptr && !moe_io_done) {
                    moe_req = active_moe_req;
                } else {
                    size_t pick = 0;
                    if (uses_deferred_eval_h2d()) {
                        std::lock_guard<std::mutex> slk(slot_mu);
                        for (size_t i = 0; i < pending_jobs.size(); ++i) {
                            if (!slot_needs_h2d[pending_jobs[i].slot]) {
                                pick = i;
                                break;
                            }
                        }
                    }
                    dense_job = pending_jobs[pick];
                    pending_jobs.erase(pending_jobs.begin() + static_cast<ptrdiff_t>(pick));
                }
            }

            if (moe_req != nullptr) {
                // MoE uses `pin_buf[0]`; invalidate only slot 0 if it holds an un-uploaded Vulkan dense layer.
                if (uses_deferred_eval_h2d()) {
                    std::lock_guard<std::mutex> slk(slot_mu);
                    if (slot_needs_h2d[0]) {
                        slot_loaded_layer[0] = -1;
                        slot_needs_h2d[0] = false;
                    }
                }
                moe_req->rc = execute_moe_work_on_worker(*moe_req);
                std::unique_lock<std::mutex> lk(worker_mu);
                moe_io_done = true;
                if (uses_deferred_eval_h2d()) {
                    worker_cv.notify_all();
                    // Wait for eval thread to finish deferred `ggml_backend_tensor_set`
                    // reading from `pin_buf` before touching `pin_buf` again.
                    worker_cv.wait(lk, [&]() { return moe_upload_done || stop_worker; });
                }
                active_moe_req = nullptr;
                worker_cv.notify_all();
                continue;
            }

            bool needs_eval_h2d = false;
            int rc = load_layer_into_slot(dense_job.layer_idx, dense_job.slot, needs_eval_h2d);

            {
                std::lock_guard<std::mutex> lk(slot_mu);
                slot_inflight_layer[dense_job.slot] = -1;
                if (rc == 0) {
                    slot_loaded_layer[dense_job.slot] = dense_job.layer_idx;
                    slot_needs_h2d[dense_job.slot] = needs_eval_h2d;
                    slot_error[dense_job.slot] = 0;
                } else {
                    slot_loaded_layer[dense_job.slot] = -1;
                    slot_needs_h2d[dense_job.slot] = false;
                    slot_error[dense_job.slot] = rc;
                }
            }
            slot_cv.notify_all();
        }
    }

    void enqueue_prefetch(int32_t layer_idx) {
        if (layer_idx < 0 || static_cast<uint32_t>(layer_idx) >= stream_end_layer) {
            return;
        }
        const int slot = layer_plans[layer_idx].slot_idx;
        if (slot < 0) {
            return;
        }

        {
            std::lock_guard<std::mutex> lk(slot_mu);
            if (slot_loaded_layer[slot] == layer_idx || slot_inflight_layer[slot] == layer_idx) {
                return;
            }
            slot_inflight_layer[slot] = layer_idx;
            slot_error[slot] = 0;
        }
        {
            std::lock_guard<std::mutex> lk(worker_mu);
            pending_jobs.push_back(PrefetchJob{layer_idx, slot});
        }
        worker_cv.notify_one();
    }

    bool wait_slot_ready(int32_t layer_idx) {
        if (layer_idx < 0 || static_cast<uint32_t>(layer_idx) >= stream_end_layer) {
            return true;
        }
        const int slot = layer_plans[layer_idx].slot_idx;
        if (slot < 0) {
            return true;
        }
        using Clock = std::chrono::steady_clock;
        auto t0 = profile_enabled ? Clock::now() : Clock::time_point{};

        // If the OTHER slot still holds an un-uploaded Vulkan `pin_buf` entry (e.g. a layer
        // was skipped), drop it so `worker_thread` isn't blocked from filling `pin_buf` for `slot`.
        if (uses_deferred_eval_h2d()) {
            const int other = 1 - slot;
            bool woke = false;
            {
                std::lock_guard<std::mutex> wlk(worker_mu);
                std::lock_guard<std::mutex> slk(slot_mu);
                if (slot_loaded_layer[slot] != layer_idx && slot_needs_h2d[other]) {
                    slot_loaded_layer[other] = -1;
                    slot_needs_h2d[other] = false;
                    woke = true;
                }
            }
            if (woke) {
                worker_cv.notify_all();
            }
        }

        // If neither loaded nor in flight (e.g. first layer or skipped layer), enqueue now.
        enqueue_prefetch(layer_idx);

        int err = 0;
        bool needs_h2d = false;
        {
            std::unique_lock<std::mutex> lk(slot_mu);
            slot_cv.wait(lk, [&]() {
                return slot_inflight_layer[slot] != layer_idx &&
                       (slot_loaded_layer[slot] == layer_idx || slot_error[slot] != 0);
            });
            err = slot_error[slot];
            needs_h2d = (err == 0 && slot_needs_h2d[slot]);
        }
        if (err != 0) {
            return false;
        }

        if (needs_h2d) {
            // Non-CUDA discrete GPU (e.g. Vulkan): perform `ggml_backend_tensor_set` safely
            // from `pin_buf[slot]` on the eval thread while the GPU backend is idle/synchronized.
            const auto & plan = layer_plans[layer_idx];
            const int n_direct = slot_h2d_n_direct[slot];
            if (n_direct > 0) {
                for (int i = 0; i < n_direct; ++i) {
                    const uint32_t tidx = slot_h2d_direct_slices[slot][i].tensor_idx;
                    if (tidx < plan.tensors.size()) {
                        ggml_backend_tensor_set(
                            plan.tensors[tidx].tensor,
                            pin_buf[slot] + slot_h2d_direct_slices[slot][i].src_offset,
                            0,
                            slot_h2d_direct_slices[slot][i].len
                        );
                    }
                }
            } else {
                for (const auto & tb : plan.tensors) {
                    ggml_backend_tensor_set(tb.tensor, pin_buf[slot] + tb.offset, 0, tb.nbytes);
                }
            }
            {
                std::lock_guard<std::mutex> wlk(worker_mu);
                std::lock_guard<std::mutex> slk(slot_mu);
                slot_needs_h2d[slot] = false;
            }
            worker_cv.notify_all();
        }

#ifdef LAYERSTREAM_HAS_CUDA_RT
        if (!is_uma_zero_copy && copy_stream != nullptr && slot_event[slot] != nullptr) {
            if (ls_cuda::event_synchronize(slot_event[slot]) != ls_cuda::kSuccess) {
                return false;
            }
        }
#endif
        if (profile_enabled) {
            prof_cb_wait_us.fetch_add(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count()),
                std::memory_order_relaxed
            );
        }
        return true;
    }

    int32_t inspect_node_stream_layer(const ggml_tensor * node) const {
        if (!node || tensor_to_layer.empty()) {
            return -1;
        }
        auto lookup = [&](const ggml_tensor * t) -> int32_t {
            if (!t) {
                return -1;
            }
            auto it = tensor_to_layer.find(t);
            if (it != tensor_to_layer.end()) {
                return it->second;
            }
            if (t->view_src) {
                auto vit = tensor_to_layer.find(t->view_src);
                if (vit != tensor_to_layer.end()) {
                    return vit->second;
                }
            }
            return -1;
        };

        int32_t found = lookup(node);
        if (found >= 0) {
            return found;
        }
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            int32_t sl = lookup(node->src[i]);
            if (sl >= 0) {
                return sl;
            }
        }
        return -1;
    }

    int32_t inspect_node_moe_layer(const ggml_tensor * node) const {
        if (!node || tensor_to_moe_layer.empty()) {
            return -1;
        }
        auto lookup = [&](const ggml_tensor * t) -> int32_t {
            if (!t) {
                return -1;
            }
            auto it = tensor_to_moe_layer.find(t);
            if (it != tensor_to_moe_layer.end()) {
                return it->second;
            }
            if (t->view_src) {
                auto vit = tensor_to_moe_layer.find(t->view_src);
                if (vit != tensor_to_moe_layer.end()) {
                    return vit->second;
                }
            }
            return -1;
        };

        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            int32_t ml = lookup(node->src[i]);
            if (ml >= 0) {
                return ml;
            }
        }
        return -1;
    }

    void prepare_decode_pass() {
        eval_failed = false;
        eval_error_msg.clear();
        seen_moe_topk_layer = -1;
        seen_moe_topk_tensor = nullptr;
        moe_weight_chain.clear();
        pending_moe_layer = -1;
        pending_moe_topk_tensor = nullptr;
        moe_split_completed_layer = -1;
        if (is_cancelled()) {
            eval_failed = true;
            eval_error_msg = "request cancelled";
            return;
        }
        if (streamed_dense_layers == 0) {
            return;
        }
        // Drain any stale inflight jobs from a prior cancelled/aborted pass.
        {
            std::unique_lock<std::mutex> lk(slot_mu);
            slot_cv.wait(lk, [&]() {
                return slot_inflight_layer[0] == -1 && slot_inflight_layer[1] == -1;
            });
        }
        const int32_t first_stream = first_dense_stream_layer;
        if (first_stream < 0) {
            return;
        }
        if (!wait_slot_ready(first_stream)) {
            eval_failed = true;
            eval_error_msg = "failed to load first streamed dense layer";
            return;
        }
        active_stream_layer = first_stream;
        pending_release_layer = -1;
        highest_enqueued_layer = first_stream;

        const int32_t second_stream = next_dense_layer[first_stream];
        if (second_stream >= 0) {
            enqueue_prefetch(second_stream);
            highest_enqueued_layer = second_stream;
        }
    }

    static bool abort_cb(void * user_data) {
        auto * self = static_cast<ls_engine *>(user_data);
        if (self->is_cancelled()) {
            self->eval_failed = true;
            if (self->eval_error_msg.empty()) {
                self->eval_error_msg = "request cancelled";
            }
            return true;
        }
        return self->eval_failed;
    }

    static bool cb_eval(ggml_tensor * t, bool ask, void * user_data) {
        auto * self = static_cast<ls_engine *>(user_data);
        if (self->is_cancelled()) {
            self->eval_failed = true;
            if (self->eval_error_msg.empty()) {
                self->eval_error_msg = "request cancelled";
            }
            // In ask=true, force a split so ask=false returns false to abort llama_decode immediately.
            return ask;
        }
        if (self->eval_failed) {
            return ask;
        }
        if (self->streamed_dense_layers == 0 && self->streamed_moe_layers == 0) {
            return false;
        }

        if (ask) {
            if (self->streamed_moe_layers > 0) {
                if (self->is_cuda_backend && self->seen_moe_topk_layer >= 0) {
                    const bool touches_topk =
                        (t->src[0] == self->seen_moe_topk_tensor ||
                         t->src[1] == self->seen_moe_topk_tensor);
                    const bool touches_chain =
                        (!self->moe_weight_chain.empty() &&
                         ((t->src[0] && self->moe_weight_chain.count(t->src[0]) > 0) ||
                          (t->src[1] && self->moe_weight_chain.count(t->src[1]) > 0)));
                    const bool is_weights_tail_op =
                        (t->op == GGML_OP_GET_ROWS || t->op == GGML_OP_RESHAPE ||
                         t->op == GGML_OP_SUM_ROWS || t->op == GGML_OP_CLAMP ||
                         t->op == GGML_OP_DIV || t->op == GGML_OP_SOFT_MAX ||
                         t->op == GGML_OP_SCALE);
                    const bool is_moe_mul = (self->inspect_node_moe_layer(t) == self->seen_moe_topk_layer);

                    if (touches_topk || (touches_chain && is_weights_tail_op)) {
                        // Still inside ggml-cuda's fusible topk_moe subgraph; keep going so
                        // ggml_cuda_op_topk_moe stays 100% fused (bit-identical to stock llama.cpp).
                        self->moe_weight_chain.insert(t);
                        return false;
                    } else if (!self->moe_weight_chain.empty() && !is_moe_mul) {
                        // First node strictly AFTER topk_moe finishes (typically `cur = reshape_3d`)
                        // and strictly BEFORE `mul_mat_id`: stop gv here so ask=false streams ONLY
                        // the active K experts into `expert_slot_buf`!
                        self->pending_moe_layer = self->seen_moe_topk_layer;
                        self->pending_moe_topk_tensor = self->seen_moe_topk_tensor;
                        self->seen_moe_topk_layer = -1;
                        self->seen_moe_topk_tensor = nullptr;
                        self->moe_weight_chain.clear();
                        return true;
                    }
                }

                const int32_t topk_layer = parse_moe_topk_layer(t->name);
                if (topk_layer >= 0 &&
                    static_cast<uint32_t>(topk_layer) < self->n_layers &&
                    self->layer_plans[topk_layer].is_moe_sparse &&
                    topk_layer >= static_cast<int32_t>(self->resident_layers)) {
                    const bool force_split_on_topk =
                        !self->is_cuda_backend ||
                        (std::getenv("LAYERSTREAM_MOE_SPLIT_ON_TOPK") != nullptr &&
                         std::getenv("LAYERSTREAM_MOE_SPLIT_ON_TOPK")[0] == '1');
                    if (force_split_on_topk) {
                        // On Vulkan, Metal, and CPU (where ggml_cuda_op_topk_moe does not exist),
                        // graph_optimize may reorder `ffn_norm` ahead of `ffn_moe_topk`, leaving no
                        // intervening non-MoE node before `MUL_MAT_ID`. Split directly on `ffn_moe_topk`
                        // so `ask=false` ALWAYS populates `expert_slot_buf` strictly before `MUL_MAT_ID`.
                        self->pending_moe_layer = topk_layer;
                        self->pending_moe_topk_tensor = t;
                        self->seen_moe_topk_layer = -1;
                        self->seen_moe_topk_tensor = nullptr;
                        self->moe_weight_chain.clear();
                        return true;
                    }
                    self->seen_moe_topk_layer = topk_layer;
                    self->seen_moe_topk_tensor = t;
                    self->moe_weight_chain.clear();
                    return false;
                }

                // Fail closed: NEVER mutate `expert_slot_buf` inside `ask=true` (that would
                // overwrite earlier layers' weights before GPU compute runs). If a streamed MoE
                // MUL_MAT_ID node is reached without a completed split for that layer, abort cleanly.
                const int32_t ml = self->inspect_node_moe_layer(t);
                if (ml >= 0 && ml != self->moe_split_completed_layer) {
                    self->seen_moe_topk_layer = -1;
                    self->seen_moe_topk_tensor = nullptr;
                    self->moe_weight_chain.clear();
                    self->eval_failed = true;
                    self->eval_error_msg =
                        "MoE router split was not reached before MUL_MAT_ID for layer " +
                        std::to_string(ml) +
                        " (refusing to corrupt expert_slot_buf; set LAYERSTREAM_MOE_SPLIT_ON_TOPK=1 or LAYERSTREAM_MOE_FULL_LAYER=1)";
                    return true;
                }
            }

            if (self->streamed_dense_layers == 0) {
                return false;
            }

            const int32_t op_layer = self->inspect_node_stream_layer(t);
            if (op_layer < 0 || op_layer == self->active_stream_layer) {
                return false;
            }

            // Fail closed if an unexpected non-sequential dense layer jump would alias the
            // currently executing layer's ping-pong slot inside gv = [j0 .. t].
            if (self->active_stream_layer >= 0 &&
                self->layer_plans[op_layer].slot_idx == self->layer_plans[self->active_stream_layer].slot_idx) {
                self->eval_failed = true;
                self->eval_error_msg =
                    "non-sequential dense layer transition (" +
                    std::to_string(self->active_stream_layer) + " -> " +
                    std::to_string(op_layer) + ") aliases the same ping-pong slot";
                return true;
            }

            // Transitioning to a new streamed dense layer: wait on its slot BEFORE
            // ggml_backend_sched executes gv = [j0 .. t]. Both the outgoing
            // layer's slot and incoming layer's slot remain valid during gv.
            if (!self->wait_slot_ready(op_layer)) {
                self->eval_failed = true;
                self->eval_error_msg = "failed to load streamed dense layer " + std::to_string(op_layer);
                return true;
            }
            self->pending_release_layer = self->active_stream_layer;
            self->active_stream_layer = op_layer;
            return true;
        } else {
            if (self->pending_moe_layer >= 0) {
                const int32_t ml = self->pending_moe_layer;
                ggml_tensor * topk_t = self->pending_moe_topk_tensor;
                self->pending_moe_layer = -1;
                self->pending_moe_topk_tensor = nullptr;

                const auto & plan = self->layer_plans[ml];
                std::vector<int32_t> active_experts;
                if (topk_t != nullptr && topk_t->type == GGML_TYPE_I32 && topk_t->ne[0] > 0 && topk_t->ne[1] > 0) {
                    const size_t n_k = static_cast<size_t>(topk_t->ne[0]);
                    const size_t n_tok = static_cast<size_t>(topk_t->ne[1]);
                    std::vector<int32_t> ids(n_k * n_tok);
                    ggml_backend_tensor_get_2d(
                        topk_t,
                        ids.data(),
                        0,
                        n_k * sizeof(int32_t),
                        n_tok,
                        topk_t->nb[1],
                        n_k * sizeof(int32_t)
                    );
                    std::vector<uint8_t> seen(plan.n_experts, 0);
                    active_experts.reserve(std::min<size_t>(ids.size(), plan.n_experts));
                    for (int32_t eid : ids) {
                        if (eid >= 0 && static_cast<uint32_t>(eid) < plan.n_experts && !seen[eid]) {
                            seen[eid] = 1;
                            active_experts.push_back(eid);
                        }
                    }
                    std::sort(active_experts.begin(), active_experts.end());
                }

                const bool force_full = active_experts.empty();
                if (self->load_moe_experts(ml, active_experts, force_full) != 0) {
                    self->eval_failed = true;
                    self->eval_error_msg = "failed to load MoE experts for layer " + std::to_string(ml);
                    return false;
                }
                self->moe_split_completed_layer = ml;
                return true;
            }

            // ggml_backend_synchronize(split_backend) has just completed.
            // `pending_release_layer`'s slot is now completely free; start
            // prefetching the next dense streamed layer into that freed slot.
            if (self->active_stream_layer >= 0 &&
                static_cast<size_t>(self->active_stream_layer) < self->next_dense_layer.size()) {
                const int32_t next_layer = self->next_dense_layer[self->active_stream_layer];
                if (next_layer >= 0 && next_layer > self->highest_enqueued_layer) {
                    self->enqueue_prefetch(next_layer);
                    self->highest_enqueued_layer = next_layer;
                } else if (next_layer < 0 &&
                           self->streamed_dense_layers >= 2 &&
                           self->first_dense_stream_layer >= 0 &&
                           self->layer_plans[self->active_stream_layer].slot_idx == 1) {
                    // Inter-sweep wraparound prefetch: the final streamed layer uses Slot 1,
                    // so Slot 0 is free now. Prefetch the first streamed layer while sampling
                    // and the draft model run!
                    self->enqueue_prefetch(self->first_dense_stream_layer);
                }
            }
            return true;
        }
    }
};

