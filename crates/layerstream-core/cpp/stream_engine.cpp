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

    // Single shared host staging buffer for discrete GPUs (page-aligned / cudaHostAlloc).
    // Exclusively written by `worker_thread` (never concurrently by eval thread) to guarantee
    // zero data races and zero aliasing &mut [u8] UB while saving ~166-385 MiB of locked RAM.
    uint8_t * pin_buf = nullptr;
    size_t pin_cap = 0;
#ifdef LAYERSTREAM_HAS_CUDA_RT
    bool pin_is_cuda_host = false;
    ls_cuda::Stream copy_stream = nullptr;
    ls_cuda::Event slot_event[2] = {nullptr, nullptr};
    int last_h2d_slot = -1; // exclusively accessed on worker_thread
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
#ifdef LAYERSTREAM_HAS_CUDA_RT
            // Ensure the previous H2D DMA from our shared pinned buffer has finished
            // before refilling it with the next layer's NVMe read.
            if (copy_stream != nullptr && last_h2d_slot >= 0 && slot_event[last_h2d_slot] != nullptr) {
                if (ls_cuda::event_synchronize(slot_event[last_h2d_slot]) != ls_cuda::kSuccess) {
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
                    pin_buf,
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
                    pin_buf,
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
                            pin_buf + direct_slices[i].src_offset,
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
                        pin_buf,
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
                last_h2d_slot = slot;
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
            if (copy_stream != nullptr && last_h2d_slot >= 0 && slot_event[last_h2d_slot] != nullptr) {
                if (ls_cuda::event_synchronize(slot_event[last_h2d_slot]) != ls_cuda::kSuccess) {
                    return -21;
                }
            }
#endif
            req.direct_slices.resize(std::max<size_t>(plan.tensors.size() * 2, 64));
            int n_direct = 0;
            if (load_layer_direct_cb != nullptr) {
                n_direct = load_layer_direct_cb(
                    io_user_data,
                    req.layer_idx,
                    plan.name_ptrs.data(),
                    plan.offsets.data(),
                    plan.sizes.data(),
                    plan.tensors.size(),
                    pin_buf,
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
                    pin_buf,
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
                                pin_buf + req.direct_slices[i].src_offset,
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
                            pin_buf,
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
        if (copy_stream != nullptr && last_h2d_slot >= 0 && slot_event[last_h2d_slot] != nullptr) {
            if (ls_cuda::event_synchronize(slot_event[last_h2d_slot]) != ls_cuda::kSuccess) {
                return -24;
            }
        }
#endif
        const auto & packed_reqs = *req.packed_reqs;
        int rc = load_slices_cb(io_user_data, packed_reqs.data(), packed_reqs.size(), pin_buf, pin_cap);
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
                        pin_buf + packed_reqs[i].dst_offset,
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
                                    pin_buf + work.direct_slices[i].src_offset,
                                    0,
                                    work.direct_slices[i].len
                                );
                            }
                        }
                    } else {
                        for (const auto & tb : plan.tensors) {
                            ggml_backend_tensor_set(tb.tensor, pin_buf + tb.offset, 0, tb.nbytes);
                        }
                    }
                } else {
                    for (size_t i = 0; i < reqs.size(); ++i) {
                        const auto & tb = plan.tensors[reqs[i].tensor_idx];
                        ggml_backend_tensor_set(
                            tb.tensor,
                            pin_buf + packed_reqs[i].dst_offset,
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
                        // On non-CUDA discrete GPU (Vulkan), wait until eval thread has uploaded
                        // any prior `pin_buf` dense layer before overwriting `pin_buf`.
                        std::lock_guard<std::mutex> slk(slot_mu);
                        return !slot_needs_h2d[0] && !slot_needs_h2d[1];
                    }
                    return false;
                });
                if (stop_worker) {
                    return;
                }
                if (active_moe_req != nullptr && !moe_io_done) {
                    moe_req = active_moe_req;
                } else {
                    dense_job = pending_jobs.front();
                    pending_jobs.erase(pending_jobs.begin());
                }
            }

            if (moe_req != nullptr) {
                // Any preceding dense prefetch on this worker has already finished and published;
                // invalidate deferred (un-uploaded) Vulkan dense slots before overwriting `pin_buf`.
                if (uses_deferred_eval_h2d()) {
                    std::lock_guard<std::mutex> slk(slot_mu);
                    for (int s = 0; s < 2; ++s) {
                        if (slot_needs_h2d[s]) {
                            slot_loaded_layer[s] = -1;
                            slot_needs_h2d[s] = false;
                        }
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
            // Non-CUDA discrete GPU (e.g. Vulkan): worker paused before overwriting `pin_buf`
            // because `slot_needs_h2d[slot]` is true; perform `ggml_backend_tensor_set` safely
            // on the eval thread while the GPU backend is idle and synchronized.
            const auto & plan = layer_plans[layer_idx];
            const int n_direct = slot_h2d_n_direct[slot];
            if (n_direct > 0) {
                for (int i = 0; i < n_direct; ++i) {
                    const uint32_t tidx = slot_h2d_direct_slices[slot][i].tensor_idx;
                    if (tidx < plan.tensors.size()) {
                        ggml_backend_tensor_set(
                            plan.tensors[tidx].tensor,
                            pin_buf + slot_h2d_direct_slices[slot][i].src_offset,
                            0,
                            slot_h2d_direct_slices[slot][i].len
                        );
                    }
                }
            } else {
                for (const auto & tb : plan.tensors) {
                    ggml_backend_tensor_set(tb.tensor, pin_buf + tb.offset, 0, tb.nbytes);
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

static uint8_t * alloc_aligned_4k(size_t size) {
    const size_t aligned = align_up(std::max<size_t>(size, 4096), 4096);
#ifdef _WIN32
    return static_cast<uint8_t *>(_aligned_malloc(aligned, 4096));
#else
    void * ptr = nullptr;
    if (posix_memalign(&ptr, 4096, aligned) != 0) {
        return nullptr;
    }
    return static_cast<uint8_t *>(ptr);
#endif
}

static void free_aligned_4k(uint8_t * ptr) {
    if (!ptr) {
        return;
    }
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

ls_engine * ls_engine_create(const ls_engine_config * cfg, char * err_buf, size_t err_cap) {
    if (!cfg || !cfg->model_path) {
        ls_internal::set_error(err_buf, err_cap, "invalid ls_engine_config: model_path is null");
        return nullptr;
    }

    ls_internal::ensure_llama_initialized();

    auto * eng = new ls_engine();
    eng->full_resident_reference = cfg->full_resident_reference;
    eng->io_user_data = cfg->io_user_data;
    eng->load_tensors_cb = cfg->load_tensors_cb;
    eng->load_layer_direct_cb = cfg->load_layer_direct_cb;
    eng->load_slices_cb = cfg->load_slices_cb;

    const uint32_t n_ctx = cfg->n_ctx > 0 ? cfg->n_ctx : 2048;
    const uint32_t n_batch = cfg->n_batch > 0 ? cfg->n_batch : 512;
    const uint32_t n_threads = cfg->n_threads > 0 ? cfg->n_threads : 4;

    // Static CPU-only device list used when `use_gpu == false` so llama.cpp never
    // opens Vulkan0/CUDA0 or allocates GPU compute buffers in CPU test mode.
    static ggml_backend_dev_t g_cpu_only_devs[2] = {nullptr, nullptr};
    static std::once_flag g_cpu_dev_once;
    std::call_once(g_cpu_dev_once, []() {
        g_cpu_only_devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        g_cpu_only_devs[1] = nullptr;
    });

    if (cfg->full_resident_reference) {
        ggml_backend_dev_t ref_gpu_dev = nullptr;
        ggml_backend_dev_t active_devs[3] = {nullptr, nullptr, nullptr};
        if (cfg->use_gpu) {
            ggml_backend_buffer_type_t dummy_buft = nullptr;
            bool dummy_gpu = false, dummy_uma = false, dummy_unified = false, dummy_cuda = false;
            std::string gpu_err;
            if (!ls_internal::select_target_buft(
                    true,
                    ref_gpu_dev,
                    dummy_buft,
                    dummy_gpu,
                    dummy_uma,
                    dummy_unified,
                    dummy_cuda,
                    eng->backend_name,
                    gpu_err
                )) {
                ls_internal::set_error(err_buf, err_cap, gpu_err);
                ls_engine_free(eng);
                return nullptr;
            }
            eng->is_unified_memory = dummy_unified;
            eng->is_cuda_backend = dummy_cuda;
        }
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = cfg->use_gpu ? 999 : 0;
        mparams.no_alloc = false;
        if (!cfg->use_gpu && g_cpu_only_devs[0] != nullptr) {
            mparams.devices = g_cpu_only_devs;
        } else if (cfg->use_gpu && ref_gpu_dev != nullptr) {
            active_devs[0] = ref_gpu_dev;
            active_devs[1] = g_cpu_only_devs[0];
            active_devs[2] = nullptr;
            mparams.devices = active_devs;
        }

        eng->model = llama_model_load_from_file(cfg->model_path, mparams);
        if (!eng->model) {
            ls_internal::set_error(err_buf, err_cap, "llama_model_load_from_file failed (reference mode)");
            ls_engine_free(eng);
            return nullptr;
        }
        eng->vocab = llama_model_get_vocab(eng->model);
        eng->n_layers = static_cast<uint32_t>(llama_model_n_layer(eng->model));
        eng->resident_layers = eng->n_layers;
        eng->streamed_layers = 0;
        eng->using_gpu = cfg->use_gpu && llama_supports_gpu_offload();

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = n_ctx;
        cparams.n_batch = n_batch;
        cparams.n_ubatch = n_batch;
        cparams.n_threads = static_cast<int32_t>(n_threads);
        cparams.n_threads_batch = static_cast<int32_t>(n_threads);
        if (!cfg->use_gpu) {
            cparams.offload_kqv = false;
            cparams.op_offload = false;
        }
        if (cfg->kv_cache_type == 1) {
            cparams.type_k = GGML_TYPE_Q8_0;
            cparams.type_v = GGML_TYPE_Q8_0;
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        } else if (cfg->kv_cache_type == 2) {
            cparams.type_k = GGML_TYPE_Q4_0;
            cparams.type_v = GGML_TYPE_Q4_0;
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        } else {
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        }
        // Reference mode is 100% resident: use llama.cpp's lightweight abort_callback for
        // cancellation instead of cb_eval so ggml_backend_sched stays on the zero-callback fast path.
        cparams.abort_callback = &ls_engine::abort_cb;
        cparams.abort_callback_data = eng;

        eng->ctx = llama_init_from_model(eng->model, cparams);
        if (!eng->ctx) {
            ls_internal::set_error(err_buf, err_cap, "llama_init_from_model failed (reference mode)");
            ls_engine_free(eng);
            return nullptr;
        }
        llama_memory_t mem = llama_get_memory(eng->ctx);
        eng->can_speculate = mem != nullptr && llama_memory_can_shift(mem) && llama_memory_seq_rm(mem, 0, 1, -1);
        return eng;
    }

    if (const char * p = std::getenv("LAYERSTREAM_PROFILE"); p && p[0] != '\0' && p[0] != '0') {
        eng->profile_enabled = true;
    }

    if (!cfg->load_tensors_cb) {
        ls_internal::set_error(err_buf, err_cap, "load_tensors_cb must be set in streaming mode");
        ls_engine_free(eng);
        return nullptr;
    }

    ggml_backend_dev_t chosen_gpu_dev = nullptr;
    ggml_backend_buffer_type_t target_buft = nullptr;
    std::string gpu_err;
    if (!ls_internal::select_target_buft(
            cfg->use_gpu,
            chosen_gpu_dev,
            target_buft,
            eng->using_gpu,
            eng->is_uma_zero_copy,
            eng->is_unified_memory,
            eng->is_cuda_backend,
            eng->backend_name,
            gpu_err
        )) {
        ls_internal::set_error(
            err_buf,
            err_cap,
            gpu_err.empty() ? "failed to select target ggml backend buffer type" : gpu_err
        );
        ls_engine_free(eng);
        return nullptr;
    }

    // 1. Load metadata and tensor graph with zero memory allocation.
    ggml_backend_dev_t active_devs[3] = {nullptr, nullptr, nullptr};
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = eng->using_gpu ? 999 : 0;
    mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    mparams.no_alloc = true;
    if (!eng->using_gpu && g_cpu_only_devs[0] != nullptr) {
        mparams.devices = g_cpu_only_devs;
    } else if (eng->using_gpu && chosen_gpu_dev != nullptr) {
        active_devs[0] = chosen_gpu_dev;
        active_devs[1] = g_cpu_only_devs[0];
        active_devs[2] = nullptr;
        mparams.devices = active_devs;
    }

    eng->model = llama_model_load_from_file(cfg->model_path, mparams);
    if (!eng->model) {
        ls_internal::set_error(err_buf, err_cap, "llama_model_load_from_file(no_alloc=true) failed");
        ls_engine_free(eng);
        return nullptr;
    }
    eng->vocab = llama_model_get_vocab(eng->model);
    eng->n_layers = static_cast<uint32_t>(llama_model_n_layer(eng->model));
    eng->layer_plans.resize(eng->n_layers);
    for (uint32_t i = 0; i < eng->n_layers; ++i) {
        eng->layer_plans[i].layer_idx = static_cast<int32_t>(i);
    }

    // 2. Partition all model tensors into non-layer tensors (GPU resident, CPU resident,
    // or streamed LM head) vs blk.{L}.* layers (dense or sparse MoE).
    // On unified-memory GPUs (Jetson Orin, Apple Silicon, iGPUs), moving embeddings/head to
    // CPU RAM allocates from the exact same physical DRAM pool and would trick auto-budget
    // into over-allocating resident layers -> ignore cpu_embeddings/cpu_head there.
    const bool allow_cpu_offload = eng->using_gpu && !eng->is_unified_memory;
    if (eng->using_gpu && eng->is_unified_memory && (cfg->cpu_embeddings || cfg->cpu_head)) {
        std::fprintf(
            stderr,
            "[layerstream-guard] unified-memory GPU shares system DRAM; keeping embeddings/head on GPU "
            "to prevent RAM over-budgeting\n"
        );
    }
    const size_t buft_align = std::max<size_t>(ggml_backend_buft_get_alignment(target_buft), 256);
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    const size_t cpu_align = std::max<size_t>(ggml_backend_buft_get_alignment(cpu_buft), 64);
    std::unordered_set<const ggml_tensor *> seen_ptrs;

    std::vector<TensorBinding> gpu_non_layer_tensors;
    std::vector<TensorBinding> cpu_non_layer_tensors;
    std::optional<TensorBinding> stream_head_candidate;

    for (const auto & kv : ls_llama::tensors_by_name(eng->model)) {
        const std::string & name = kv.first;
        ggml_tensor * t = kv.second;
        if (!t || !seen_ptrs.insert(t).second) {
            continue;
        }
        const int32_t layer_idx = parse_layer_index(name.c_str());
        TensorBinding tb;
        tb.name = name;
        tb.tensor = t;
        tb.nbytes = ggml_nbytes(t);
        if (layer_idx >= 0 && static_cast<uint32_t>(layer_idx) < eng->n_layers) {
            eng->layer_plans[layer_idx].tensors.push_back(std::move(tb));
        } else {
            const bool is_tok_embd = ls_llama::is_tok_embd_tensor(eng->model, t, name);
            const bool is_output_head = ls_llama::is_output_head_tensor(eng->model, t, name);

            if (is_tok_embd && is_output_head) {
                // Tied embeddings: used at both start and end of every pass, so never alias
                // into ping-pong slots; pin on CPU only on discrete GPUs when requested.
                if (allow_cpu_offload && (cfg->cpu_embeddings || cfg->cpu_head || cfg->stream_head)) {
                    cpu_non_layer_tensors.push_back(std::move(tb));
                } else {
                    gpu_non_layer_tensors.push_back(std::move(tb));
                }
            } else if (is_tok_embd) {
                if (allow_cpu_offload && cfg->cpu_embeddings) {
                    cpu_non_layer_tensors.push_back(std::move(tb));
                } else {
                    gpu_non_layer_tensors.push_back(std::move(tb));
                }
            } else if (is_output_head) {
                if (allow_cpu_offload && cfg->cpu_head) {
                    cpu_non_layer_tensors.push_back(std::move(tb));
                } else if (cfg->stream_head) {
                    stream_head_candidate = std::move(tb);
                } else {
                    gpu_non_layer_tensors.push_back(std::move(tb));
                }
            } else {
                gpu_non_layer_tensors.push_back(std::move(tb));
            }
        }
    }

    // Detect sparse MoE layers (`blk.<L>.*_exps.*` 3D expert tensors) and split their
    // tiny non-expert tensors (attention, norms, router, shared experts, 2D biases) into
    // permanent resident memory so only active K experts stream on demand.
    const bool allow_moe_sparse = cfg->moe_active_streaming &&
        !(std::getenv("LAYERSTREAM_MOE_FULL_LAYER") && std::getenv("LAYERSTREAM_MOE_FULL_LAYER")[0] == '1');
    for (auto & plan : eng->layer_plans) {
        if (!allow_moe_sparse) {
            continue;
        }
        uint32_t exp_count = 0;
        bool valid_moe = false;
        for (const auto & tb : plan.tensors) {
            if (is_streamable_moe_expert_tensor(tb.name, tb.tensor)) {
                const uint32_t ne2 = static_cast<uint32_t>(tb.tensor->ne[2]);
                if (!valid_moe) {
                    exp_count = ne2;
                    valid_moe = true;
                } else if (ne2 != exp_count) {
                    valid_moe = false;
                    break;
                }
            }
        }
        if (valid_moe && exp_count > 1) {
            plan.is_moe_sparse = true;
            plan.n_experts = exp_count;
            std::vector<TensorBinding> exps;
            std::vector<TensorBinding> non_exps;
            for (auto & tb : plan.tensors) {
                if (is_streamable_moe_expert_tensor(tb.name, tb.tensor)) {
                    exps.push_back(std::move(tb));
                } else {
                    non_exps.push_back(std::move(tb));
                }
            }
            plan.tensors = std::move(exps);
            plan.moe_non_expert_tensors = std::move(non_exps);
            eng->moe_layers += 1;
        }
    }

    // Compute per-layer aligned byte sizes (for MoE layers, this is the expert-only size).
    uint64_t max_dense_layer_bytes = 0;
    uint64_t max_moe_layer_bytes = 0;
    for (auto & plan : eng->layer_plans) {
        uint64_t off = 0;
        for (auto & tb : plan.tensors) {
            off = align_up(off, buft_align);
            tb.offset = off;
            const size_t alloc_sz = std::max<size_t>(
                ggml_backend_buft_get_alloc_size(target_buft, tb.tensor),
                static_cast<size_t>(tb.nbytes)
            );
            tb.alloc_size = static_cast<uint64_t>(alloc_sz);
            off += alloc_sz;
        }
        plan.total_bytes = align_up(off, 4096);
        if (plan.is_moe_sparse) {
            max_moe_layer_bytes = std::max(max_moe_layer_bytes, plan.total_bytes);
        } else {
            max_dense_layer_bytes = std::max(max_dense_layer_bytes, plan.total_bytes);
        }
    }

    // Compute GPU base resident size (GPU non-layer tensors + all MoE non-expert tensors).
    uint64_t base_resident_bytes = 0;
    auto add_base_res = [&](const TensorBinding & tb) {
        base_resident_bytes = align_up(base_resident_bytes, buft_align);
        const size_t alloc_sz = std::max<size_t>(
            ggml_backend_buft_get_alloc_size(target_buft, tb.tensor),
            static_cast<size_t>(tb.nbytes)
        );
        base_resident_bytes += alloc_sz;
    };
    for (const auto & tb : gpu_non_layer_tensors) {
        add_base_res(tb);
    }
    for (const auto & plan : eng->layer_plans) {
        for (const auto & tb : plan.moe_non_expert_tensors) {
            add_base_res(tb);
        }
    }

    // 3. Decide hybrid partial-residency split R = resident_layers in [0..n_layers].
    uint32_t R = 0;
    uint64_t effective_budget = cfg->vram_budget_auto ? 0 : cfg->vram_budget_bytes;
    bool probed_safe_cap_valid = false;
    uint64_t probed_safe_cap = 0;
    if (eng->using_gpu) {
        uint64_t free_vram = 0;
        uint64_t total_vram = 0;
        const bool probed = ls_probe_gpu_memory(eng->using_gpu, &free_vram, &total_vram);
        if (probed && total_vram > 0) {
            const uint64_t headroom = eng->is_unified_memory
                ? std::max<uint64_t>(1024ULL * 1024 * 1024, total_vram / 7)
                : std::max<uint64_t>(512ULL * 1024 * 1024, total_vram / 10);
            const uint64_t safe_cap = (free_vram > headroom)
                ? std::min<uint64_t>(free_vram - headroom, (free_vram * 85ULL) / 100ULL)
                : 0;
            probed_safe_cap_valid = true;
            probed_safe_cap = safe_cap;

            if (cfg->resident_layers < 0) {
                if (cfg->vram_budget_auto) {
                    // When auto=true, cfg->vram_budget_bytes carries optional draft reserve.
                    effective_budget = (safe_cap > cfg->vram_budget_bytes)
                        ? (safe_cap - cfg->vram_budget_bytes)
                        : 0;
                    std::fprintf(
                        stderr,
                        "[layerstream-guard] auto VRAM budget: %llu MiB (probed %llu / %llu MiB free, unified=%s)\n",
                        static_cast<unsigned long long>(effective_budget / (1024 * 1024)),
                        static_cast<unsigned long long>(free_vram / (1024 * 1024)),
                        static_cast<unsigned long long>(total_vram / (1024 * 1024)),
                        eng->is_unified_memory ? "true" : "false"
                    );
                } else if (effective_budget > safe_cap) {
                    std::fprintf(
                        stderr,
                        "[layerstream-guard] requested --vram-budget-mb (%llu MiB) exceeds safe free GPU memory "
                        "(%llu MiB free / %llu MiB total); clamping to %llu MiB to prevent OOM/swap thrashing\n",
                        static_cast<unsigned long long>(effective_budget / (1024 * 1024)),
                        static_cast<unsigned long long>(free_vram / (1024 * 1024)),
                        static_cast<unsigned long long>(total_vram / (1024 * 1024)),
                        static_cast<unsigned long long>(safe_cap / (1024 * 1024))
                    );
                    effective_budget = safe_cap;
                }
            }
        } else if (cfg->resident_layers < 0 && cfg->vram_budget_auto) {
            std::fprintf(
                stderr,
                "[layerstream-guard] warning: GPU memory probe failed in auto VRAM mode; defaulting to 0 resident layers\n"
            );
            effective_budget = 0;
        }
    }

    const uint64_t kv_bytes = ls_llama::estimate_kv_cache_bytes(
        eng->model, eng->n_layers, n_ctx, cfg->kv_cache_type
    );
    const uint64_t compute_scratch = std::max<uint64_t>(
        128ULL * 1024 * 1024,
        (static_cast<uint64_t>(n_batch) * 384ULL * 1024 * 1024) / 512ULL
    );
    const uint64_t dense_slots_gpu = (max_dense_layer_bytes > 0) ? (2 * max_dense_layer_bytes) : 0;
    const uint64_t moe_slot_gpu = (max_moe_layer_bytes > 0) ? max_moe_layer_bytes : 0;
    const uint64_t pin_reserve = eng->is_uma_zero_copy
        ? 0
        : std::max(max_dense_layer_bytes, max_moe_layer_bytes);
    const uint64_t slot_reserve = dense_slots_gpu + moe_slot_gpu + pin_reserve;
    const uint64_t driver_and_io_reserve = eng->using_gpu
        ? (448ULL * 1024 * 1024)
        : (64ULL * 1024 * 1024);
    const uint64_t reserve = slot_reserve + kv_bytes + compute_scratch + driver_and_io_reserve;

    if (cfg->resident_layers >= 0) {
        R = std::min<uint32_t>(static_cast<uint32_t>(cfg->resident_layers), eng->n_layers);
        if (probed_safe_cap_valid && R > 0) {
            // Clamp explicit resident_layers if it would exceed safe GPU memory capacity.
            uint64_t max_allowed = probed_safe_cap;
            uint64_t total_all_layers = 0;
            for (uint32_t il = 0; il < eng->n_layers; ++il) {
                total_all_layers += eng->layer_plans[il].total_bytes;
            }
            const bool fits_full = (R == eng->n_layers) &&
                (base_resident_bytes + total_all_layers + kv_bytes + compute_scratch + driver_and_io_reserve <= max_allowed);
            if (!fits_full) {
                uint32_t clamped_r = 0;
                uint64_t used = base_resident_bytes + reserve;
                while (clamped_r < R && used + eng->layer_plans[clamped_r].total_bytes <= max_allowed) {
                    used += eng->layer_plans[clamped_r].total_bytes;
                    ++clamped_r;
                }
                if (clamped_r < R) {
                    std::fprintf(
                        stderr,
                        "[layerstream-guard] requested resident_layers=%u exceeds safe GPU cap (%llu MiB); clamping to %u\n",
                        R,
                        static_cast<unsigned long long>(max_allowed / (1024 * 1024)),
                        clamped_r
                    );
                    R = clamped_r;
                }
            }
        }
    } else if (effective_budget > 0) {
        if (kv_bytes * 2 > effective_budget && cfg->kv_cache_type == 0) {
            std::fprintf(
                stderr,
                "[layerstream-guard] warning: F16 KV cache (~%llu MiB at ctx=%u) exceeds 50%% of VRAM budget (%llu MiB); "
                "consider --kv-cache-type q8_0 or q4_0\n",
                static_cast<unsigned long long>(kv_bytes / (1024 * 1024)),
                n_ctx,
                static_cast<unsigned long long>(effective_budget / (1024 * 1024))
            );
        }
        uint64_t used = base_resident_bytes + reserve;
        // If ALL layers fit cleanly inside `effective_budget` WITHOUT streaming slots,
        // promote to full residency (R = n_layers) so small models never needlessly stream!
        uint64_t total_all_layers = 0;
        for (uint32_t il = 0; il < eng->n_layers; ++il) {
            total_all_layers += eng->layer_plans[il].total_bytes;
        }
        if (base_resident_bytes + total_all_layers + kv_bytes + compute_scratch + driver_and_io_reserve <= effective_budget) {
            R = eng->n_layers;
        } else {
            while (R < eng->n_layers && used + eng->layer_plans[R].total_bytes <= effective_budget) {
                used += eng->layer_plans[R].total_bytes;
                ++R;
            }
        }
    }
    eng->resident_layers = R;
    eng->streamed_layers = eng->n_layers - R;

    // Count streamed dense vs streamed MoE layers for R..n_layers.
    uint64_t streamed_dense_max = 0;
    uint64_t streamed_moe_max = 0;
    for (uint32_t l = R; l < eng->n_layers; ++l) {
        if (eng->layer_plans[l].is_moe_sparse) {
            eng->streamed_moe_layers += 1;
            streamed_moe_max = std::max(streamed_moe_max, eng->layer_plans[l].total_bytes);
        } else {
            eng->streamed_dense_layers += 1;
            streamed_dense_max = std::max(streamed_dense_max, eng->layer_plans[l].total_bytes);
        }
    }
    eng->moe_active_streaming = (eng->streamed_moe_layers > 0);
    eng->stream_end_layer = eng->n_layers;

    // Resolve `--stream-head` candidate: stream through ping-pong GPU slots when it fits
    // within `streamed_dense_max` (0 extra VRAM!), otherwise fall back cleanly to CPU RAM
    // (on GPU) or normal resident buffer (on CPU).
    if (stream_head_candidate.has_value()) {
        TensorBinding head_tb = std::move(*stream_head_candidate);
        const size_t head_alloc = align_up(
            std::max<size_t>(
                ggml_backend_buft_get_alloc_size(target_buft, head_tb.tensor),
                static_cast<size_t>(head_tb.nbytes)
            ),
            4096
        );
        if (eng->streamed_dense_layers > 0 && head_alloc <= streamed_dense_max) {
            LayerPlan head_plan;
            head_plan.layer_idx = static_cast<int32_t>(eng->n_layers);
            head_tb.offset = 0;
            head_tb.alloc_size = static_cast<uint64_t>(head_alloc);
            head_plan.total_bytes = head_alloc;
            head_plan.tensors.push_back(std::move(head_tb));
            eng->layer_plans.push_back(std::move(head_plan));
            eng->head_streamed = true;
            eng->streamed_dense_layers += 1;
            eng->stream_end_layer = eng->n_layers + 1;
        } else if (allow_cpu_offload) {
            cpu_non_layer_tensors.push_back(std::move(head_tb));
        } else {
            gpu_non_layer_tensors.push_back(std::move(head_tb));
        }
    }

    // 4. Build combined GPU resident plan (GPU non-layer + all MoE non-expert + blk.0..R-1).
    uint64_t res_off = 0;
    auto push_resident = [&](TensorBinding tb) {
        res_off = align_up(res_off, buft_align);
        tb.offset = res_off;
        const size_t alloc_sz = std::max<size_t>(
            ggml_backend_buft_get_alloc_size(target_buft, tb.tensor),
            static_cast<size_t>(tb.nbytes)
        );
        res_off += alloc_sz;
        eng->resident_plan.tensors.push_back(std::move(tb));
    };

    for (auto & tb : gpu_non_layer_tensors) {
        push_resident(std::move(tb));
    }
    for (uint32_t l = 0; l < eng->n_layers; ++l) {
        for (auto & tb : eng->layer_plans[l].moe_non_expert_tensors) {
            push_resident(std::move(tb));
        }
        eng->layer_plans[l].moe_non_expert_tensors.clear();
    }
    for (uint32_t l = 0; l < R; ++l) {
        for (auto & tb : eng->layer_plans[l].tensors) {
            push_resident(std::move(tb));
        }
        eng->layer_plans[l].tensors.clear();
        eng->layer_plans[l].total_bytes = 0;
    }
    eng->resident_bytes = align_up(std::max<uint64_t>(res_off, 4096), 4096);
    eng->resident_plan.total_bytes = eng->resident_bytes;
    eng->resident_plan.finalize_arrays();

    // 4b. Build CPU resident plan (if --cpu-embeddings / --cpu-head / fallback --stream-head).
    if (!cpu_non_layer_tensors.empty()) {
        uint64_t cpu_off = 0;
        for (auto & tb : cpu_non_layer_tensors) {
            cpu_off = align_up(cpu_off, cpu_align);
            tb.offset = cpu_off;
            const size_t alloc_sz = std::max<size_t>(
                ggml_backend_buft_get_alloc_size(cpu_buft, tb.tensor),
                static_cast<size_t>(tb.nbytes)
            );
            cpu_off += alloc_sz;
            eng->cpu_resident_plan.tensors.push_back(std::move(tb));
        }
        eng->cpu_resident_bytes = align_up(std::max<uint64_t>(cpu_off, 4096), 4096);
        eng->cpu_resident_plan.total_bytes = eng->cpu_resident_bytes;
        eng->cpu_resident_plan.finalize_arrays();
    }

    // 5. Allocate real backend buffers: GPU Resident (+ optional CPU Resident).
    eng->resident_buf = ggml_backend_buft_alloc_buffer(target_buft, eng->resident_bytes);
    if (!eng->resident_buf) {
        ls_internal::set_error(err_buf, err_cap, "failed to allocate resident backend buffer");
        ls_engine_free(eng);
        return nullptr;
    }
    ggml_backend_buffer_set_usage(eng->resident_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    auto * res_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(eng->resident_buf));
    for (const auto & tb : eng->resident_plan.tensors) {
        tb.tensor->buffer = eng->resident_buf;
        tb.tensor->data = res_base + tb.offset;
        ggml_backend_buffer_init_tensor(eng->resident_buf, tb.tensor);
    }

    if (!eng->cpu_resident_plan.tensors.empty()) {
        eng->cpu_resident_buf = ggml_backend_buft_alloc_buffer(cpu_buft, eng->cpu_resident_bytes);
        if (!eng->cpu_resident_buf) {
            ls_internal::set_error(err_buf, err_cap, "failed to allocate CPU resident backend buffer");
            ls_engine_free(eng);
            return nullptr;
        }
        ggml_backend_buffer_set_usage(eng->cpu_resident_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto * cpu_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(eng->cpu_resident_buf));
        for (const auto & tb : eng->cpu_resident_plan.tensors) {
            tb.tensor->buffer = eng->cpu_resident_buf;
            tb.tensor->data = cpu_base + tb.offset;
            ggml_backend_buffer_init_tensor(eng->cpu_resident_buf, tb.tensor);
        }
        int rc = eng->load_tensors_cb(
            eng->io_user_data,
            -1,
            eng->cpu_resident_plan.name_ptrs.data(),
            eng->cpu_resident_plan.offsets.data(),
            eng->cpu_resident_plan.sizes.data(),
            eng->cpu_resident_plan.tensors.size(),
            cpu_base,
            eng->cpu_resident_bytes
        );
        if (rc != 0) {
            ls_internal::set_error(err_buf, err_cap, "failed to load CPU resident tensors");
            ls_engine_free(eng);
            return nullptr;
        }
    }

    // Populate GPU resident tensors once from GGUF.
    if (!eng->resident_plan.tensors.empty()) {
        if (eng->is_uma_zero_copy) {
            int rc = eng->load_tensors_cb(
                eng->io_user_data,
                -1,
                eng->resident_plan.name_ptrs.data(),
                eng->resident_plan.offsets.data(),
                eng->resident_plan.sizes.data(),
                eng->resident_plan.tensors.size(),
                res_base,
                eng->resident_bytes
            );
            if (rc != 0) {
                ls_internal::set_error(err_buf, err_cap, "failed to load resident tensors (UMA/host)");
                ls_engine_free(eng);
                return nullptr;
            }
        } else {
            // Stream resident tensors in bounded chunks (<= max single tensor) so loading
            // multi-GiB hybrid resident layers never doubles peak host RAM or triggers swap!
            uint64_t max_res_tensor = 4096;
            for (const auto & tb : eng->resident_plan.tensors) {
                max_res_tensor = std::max(max_res_tensor, tb.nbytes);
            }
            const size_t chunk_cap = align_up(
                std::max<size_t>(static_cast<size_t>(max_res_tensor), 64ULL * 1024 * 1024),
                4096
            );
            uint8_t * staging = alloc_aligned_4k(chunk_cap);
            if (!staging) {
                ls_internal::set_error(err_buf, err_cap, "failed to allocate resident staging buffer");
                ls_engine_free(eng);
                return nullptr;
            }

            int rc = 0;
            size_t idx = 0;
            const auto & all_res = eng->resident_plan.tensors;
            while (idx < all_res.size() && rc == 0) {
                std::vector<const char *> c_names;
                std::vector<uint64_t> c_offs;
                std::vector<uint64_t> c_sizes;
                std::vector<size_t> c_indices;
                size_t used = 0;

                while (idx < all_res.size()) {
                    const size_t sz = static_cast<size_t>(all_res[idx].nbytes);
                    if (!c_indices.empty() && used + sz > chunk_cap) {
                        break;
                    }
                    c_names.push_back(all_res[idx].name.c_str());
                    c_offs.push_back(static_cast<uint64_t>(used));
                    c_sizes.push_back(all_res[idx].nbytes);
                    c_indices.push_back(idx);
                    used += sz;
                    ++idx;
                }

                rc = eng->load_tensors_cb(
                    eng->io_user_data,
                    -1,
                    c_names.data(),
                    c_offs.data(),
                    c_sizes.data(),
                    c_names.size(),
                    staging,
                    chunk_cap
                );
                if (rc == 0) {
                    for (size_t k = 0; k < c_indices.size(); ++k) {
                        const auto & tb = all_res[c_indices[k]];
                        ggml_backend_tensor_set(tb.tensor, staging + c_offs[k], 0, tb.nbytes);
                    }
                }
            }
            free_aligned_4k(staging);
            if (rc != 0) {
                ls_internal::set_error(err_buf, err_cap, "failed to load resident tensors (discrete)");
                ls_engine_free(eng);
                return nullptr;
            }
        }
    }

    // 6. Allocate streaming buffers:
    //    - 2 ping-pong GPU slots for streamed dense layers (and optional streamed LM head)
    //    - 1 shared GPU expert slot for streamed sparse MoE layers
    if (eng->streamed_dense_layers > 0) {
        eng->slot_bytes = align_up(std::max<uint64_t>(streamed_dense_max, 4096), 4096);
        for (int s = 0; s < 2; ++s) {
            eng->slot_buf[s] = ggml_backend_buft_alloc_buffer(target_buft, eng->slot_bytes);
            if (!eng->slot_buf[s]) {
                ls_internal::set_error(err_buf, err_cap, "failed to allocate ping-pong GPU slot buffer");
                ls_engine_free(eng);
                return nullptr;
            }
            ggml_backend_buffer_set_usage(eng->slot_buf[s], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        eng->next_dense_layer.assign(eng->stream_end_layer, -1);
        int32_t prev_dense = -1;
        int dense_rank = 0;
        for (uint32_t l = R; l < eng->stream_end_layer; ++l) {
            auto & plan = eng->layer_plans[l];
            if (plan.is_moe_sparse || plan.tensors.empty()) {
                continue;
            }
            plan.finalize_arrays();
            plan.slot_idx = dense_rank & 1;
            ++dense_rank;

            if (eng->first_dense_stream_layer < 0) {
                eng->first_dense_stream_layer = static_cast<int32_t>(l);
            }
            if (prev_dense >= 0) {
                eng->next_dense_layer[prev_dense] = static_cast<int32_t>(l);
            }
            prev_dense = static_cast<int32_t>(l);

            auto * slot_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(eng->slot_buf[plan.slot_idx]));
            for (const auto & tb : plan.tensors) {
                tb.tensor->buffer = eng->slot_buf[plan.slot_idx];
                tb.tensor->data = slot_base + tb.offset;
                ggml_backend_buffer_init_tensor(eng->slot_buf[plan.slot_idx], tb.tensor);
                eng->tensor_to_layer[tb.tensor] = static_cast<int32_t>(l);
            }
        }
    }

    if (eng->streamed_moe_layers > 0) {
        eng->expert_slot_bytes = align_up(std::max<uint64_t>(streamed_moe_max, 4096), 4096);
        eng->expert_slot_buf = ggml_backend_buft_alloc_buffer(target_buft, eng->expert_slot_bytes);
        if (!eng->expert_slot_buf) {
            ls_internal::set_error(err_buf, err_cap, "failed to allocate MoE expert GPU slot buffer");
            ls_engine_free(eng);
            return nullptr;
        }
        ggml_backend_buffer_set_usage(eng->expert_slot_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        auto * expert_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(eng->expert_slot_buf));
        for (uint32_t l = R; l < eng->n_layers; ++l) {
            auto & plan = eng->layer_plans[l];
            if (!plan.is_moe_sparse) {
                continue;
            }
            plan.finalize_arrays();
            for (const auto & tb : plan.tensors) {
                tb.tensor->buffer = eng->expert_slot_buf;
                tb.tensor->data = expert_base + tb.offset;
                ggml_backend_buffer_init_tensor(eng->expert_slot_buf, tb.tensor);
                eng->tensor_to_moe_layer[tb.tensor] = static_cast<int32_t>(l);
            }
        }
    }

    if (!eng->is_uma_zero_copy && (eng->streamed_dense_layers > 0 || eng->streamed_moe_layers > 0)) {
        // Add +2 MiB headroom for 4K-aligned O_DIRECT span padding around a single layer.
        const uint64_t max_stream_buf = std::max(eng->slot_bytes, eng->expert_slot_bytes);
        const size_t pin_bytes = align_up(max_stream_buf + (2ULL * 1024 * 1024), 4096);
#ifdef LAYERSTREAM_HAS_CUDA_RT
        if (eng->is_cuda_backend && ls_cuda::ensure_loaded() &&
            ls_cuda::stream_create_with_flags(&eng->copy_stream, ls_cuda::kStreamNonBlocking) == ls_cuda::kSuccess) {
            eng->pin_is_cuda_host = true;
            if (ls_cuda::host_alloc(reinterpret_cast<void **>(&eng->pin_buf), pin_bytes, ls_cuda::kHostAllocDefault) != ls_cuda::kSuccess) {
                ls_internal::set_error(err_buf, err_cap, "failed to allocate CUDA pinned staging buffer");
                ls_engine_free(eng);
                return nullptr;
            }
            std::memset(eng->pin_buf, 0, pin_bytes);
            eng->pin_cap = pin_bytes;
            for (int s = 0; s < 2; ++s) {
                if (ls_cuda::event_create_with_flags(&eng->slot_event[s], ls_cuda::kEventDisableTiming) != ls_cuda::kSuccess) {
                    ls_internal::set_error(err_buf, err_cap, "failed to create CUDA slot event");
                    ls_engine_free(eng);
                    return nullptr;
                }
            }
        } else
#endif
        {
            eng->pin_buf = alloc_aligned_4k(pin_bytes);
            eng->pin_cap = pin_bytes;
            if (!eng->pin_buf) {
                ls_internal::set_error(err_buf, err_cap, "failed to allocate 4K-aligned host staging buffer");
                ls_engine_free(eng);
                return nullptr;
            }
            std::memset(eng->pin_buf, 0, pin_bytes);
        }
    }

    if (eng->streamed_dense_layers > 0 || eng->streamed_moe_layers > 0) {
        try {
            eng->worker_thread = std::thread([eng]() { eng->worker_loop(); });
        } catch (const std::exception & e) {
            ls_internal::set_error(err_buf, err_cap, std::string("failed to start I/O worker thread: ") + e.what());
            ls_engine_free(eng);
            return nullptr;
        } catch (...) {
            ls_internal::set_error(err_buf, err_cap, "failed to start I/O worker thread");
            ls_engine_free(eng);
            return nullptr;
        }
    }

    ls_llama::clear_model_no_alloc(eng->model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_batch;
    cparams.n_ubatch = n_batch;
    cparams.n_threads = static_cast<int32_t>(n_threads);
    cparams.n_threads_batch = static_cast<int32_t>(n_threads);
    if (!eng->using_gpu) {
        cparams.offload_kqv = false;
        cparams.op_offload = false;
    }
    if (cfg->kv_cache_type == 1) {
        cparams.type_k = GGML_TYPE_Q8_0;
        cparams.type_v = GGML_TYPE_Q8_0;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    } else if (cfg->kv_cache_type == 2) {
        cparams.type_k = GGML_TYPE_Q4_0;
        cparams.type_v = GGML_TYPE_Q4_0;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    } else {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    }
    // Use abort_callback for cooperative cancellation on every model, and only install cb_eval
    // when layers are actually streamed so 100% resident models stay on llama.cpp's fast path.
    cparams.abort_callback = &ls_engine::abort_cb;
    cparams.abort_callback_data = eng;
    if (eng->streamed_dense_layers > 0 || eng->streamed_moe_layers > 0) {
        cparams.cb_eval = &ls_engine::cb_eval;
        cparams.cb_eval_user_data = eng;
    }

    eng->ctx = llama_init_from_model(eng->model, cparams);
    if (!eng->ctx) {
        ls_internal::set_error(err_buf, err_cap, "llama_init_from_model failed");
        ls_engine_free(eng);
        return nullptr;
    }

    llama_memory_t mem = llama_get_memory(eng->ctx);
    eng->can_speculate = mem != nullptr && llama_memory_can_shift(mem) && llama_memory_seq_rm(mem, 0, 1, -1);
    return eng;
}

void ls_engine_free(ls_engine * eng) {
    if (!eng) {
        return;
    }

    if (eng->profile_enabled && eng->total_layer_loads.load(std::memory_order_relaxed) > 0) {
        const double loads = static_cast<double>(eng->total_layer_loads.load(std::memory_order_relaxed));
        const double passes = std::max(1.0, static_cast<double>(eng->prof_pass_count.load(std::memory_order_relaxed)));
        const double io_ms = static_cast<double>(eng->prof_io_read_us.load(std::memory_order_relaxed)) / 1000.0;
        const double h2d_ms = static_cast<double>(eng->prof_h2d_sync_us.load(std::memory_order_relaxed)) / 1000.0;
        const double wait_ms = static_cast<double>(eng->prof_cb_wait_us.load(std::memory_order_relaxed)) / 1000.0;
        const double dec_ms = static_cast<double>(eng->prof_decode_total_us.load(std::memory_order_relaxed)) / 1000.0;
        std::fprintf(
            stderr,
            "[layerstream-profile] passes=%.0f layer_loads=%.0f io_read_total=%.1fms (%.2fms/layer) "
            "h2d_prev_wait=%.1fms (%.2fms/layer) cb_wait_total=%.1fms (%.2fms/layer) "
            "decode_total=%.1fms (%.1fms/pass) compute_unoverlapped=%.1fms\n",
            passes, loads, io_ms, io_ms / loads, h2d_ms, h2d_ms / loads,
            wait_ms, wait_ms / loads, dec_ms, dec_ms / passes, std::max(0.0, dec_ms - wait_ms)
        );
    }

    if (eng->worker_thread.joinable()) {
        {
            std::lock_guard<std::mutex> lk(eng->worker_mu);
            eng->stop_worker = true;
        }
        eng->worker_cv.notify_all();
        eng->worker_thread.join();
    }

    if (eng->ctx) {
        llama_free(eng->ctx);
        eng->ctx = nullptr;
    }
    if (eng->model) {
        llama_model_free(eng->model);
        eng->model = nullptr;
    }

#ifdef LAYERSTREAM_HAS_CUDA_RT
    for (int s = 0; s < 2; ++s) {
        if (eng->slot_event[s]) {
            ls_cuda::event_destroy(eng->slot_event[s]);
            eng->slot_event[s] = nullptr;
        }
    }
    if (eng->pin_buf) {
        if (eng->pin_is_cuda_host) {
            ls_cuda::free_host(eng->pin_buf);
        } else {
            free_aligned_4k(eng->pin_buf);
        }
        eng->pin_buf = nullptr;
    }
    if (eng->copy_stream) {
        ls_cuda::stream_destroy(eng->copy_stream);
        eng->copy_stream = nullptr;
    }
#else
    if (eng->pin_buf) {
        free_aligned_4k(eng->pin_buf);
        eng->pin_buf = nullptr;
    }
#endif

    if (eng->expert_slot_buf) {
        ggml_backend_buffer_free(eng->expert_slot_buf);
        eng->expert_slot_buf = nullptr;
    }
    for (int s = 0; s < 2; ++s) {
        if (eng->slot_buf[s]) {
            ggml_backend_buffer_free(eng->slot_buf[s]);
            eng->slot_buf[s] = nullptr;
        }
    }
    if (eng->cpu_resident_buf) {
        ggml_backend_buffer_free(eng->cpu_resident_buf);
        eng->cpu_resident_buf = nullptr;
    }
    if (eng->resident_buf) {
        ggml_backend_buffer_free(eng->resident_buf);
        eng->resident_buf = nullptr;
    }

    delete eng;

#ifdef __GLIBC__
    // Return freed host allocator arenas to the OS when running in-process.
    malloc_trim(0);
#endif
}

void ls_engine_get_stats(const ls_engine * eng, ls_engine_stats * out) {
    if (!eng || !out) {
        return;
    }
    out->n_layers = eng->n_layers;
    out->resident_layers = eng->resident_layers;
    out->streamed_layers = eng->streamed_layers;
    out->resident_bytes = eng->resident_bytes;
    out->cpu_resident_bytes = eng->cpu_resident_bytes;
    out->slot_bytes = eng->slot_bytes;
    out->expert_slot_bytes = eng->expert_slot_bytes;
    out->total_layer_loads = eng->total_layer_loads.load(std::memory_order_relaxed);
    out->moe_layers = eng->moe_layers;
    out->moe_experts_loaded = eng->moe_experts_loaded.load(std::memory_order_relaxed);
    out->moe_experts_total = eng->moe_experts_total.load(std::memory_order_relaxed);
    out->head_streamed = eng->head_streamed;
    out->moe_active_streaming = eng->moe_active_streaming;
    out->is_uma_zero_copy = eng->is_uma_zero_copy;
    out->is_unified_memory = eng->is_unified_memory;
    out->using_gpu = eng->using_gpu;
    out->can_speculate = eng->can_speculate;
    std::memset(out->backend_name, 0, sizeof(out->backend_name));
    const size_t bn_len = std::min(sizeof(out->backend_name) - 1, eng->backend_name.size());
    std::memcpy(out->backend_name, eng->backend_name.data(), bn_len);
}

int32_t ls_engine_vocab_size(const ls_engine * eng) {
    if (!eng || !eng->vocab) {
        return 0;
    }
    return llama_vocab_n_tokens(eng->vocab);
}

uint32_t ls_engine_n_ctx(const ls_engine * eng) {
    if (!eng || !eng->ctx) {
        return 0;
    }
    return llama_n_ctx(eng->ctx);
}

void ls_engine_set_cancel(ls_engine * eng, bool cancel) {
    if (!eng) {
        return;
    }
    eng->cancel_requested.store(cancel, std::memory_order_relaxed);
}

void ls_engine_set_cancel_cb(ls_engine * eng, ls_cancel_check_cb cb, const void * user_data) {
    if (!eng) {
        return;
    }
    eng->cancel_cb = cb;
    eng->cancel_cb_user_data = user_data;
}

int32_t ls_engine_apply_chat_template(
    const ls_engine * eng,
    const ls_chat_message * messages,
    size_t n_messages,
    bool add_assistant_prompt,
    char * out_buf,
    int32_t out_cap
) {
    if (!eng || !eng->model || (!messages && n_messages > 0)) {
        return -1;
    }
    const char * tmpl = llama_model_chat_template(eng->model, nullptr);
    bool has_tool_role = false;
    for (size_t i = 0; i < n_messages; ++i) {
        if (messages[i].role && std::strcmp(messages[i].role, "tool") == 0) {
            has_tool_role = true;
            break;
        }
    }

    // Check whether the model's embedded template natively preserves `role == "tool"`;
    // if a non-tool template silently drops `tool` turns, fall back to `"user"` with an
    // explicit `<tool_response>` wrapper so tool results are never lost or confused with user text.
    bool map_tool_to_tagged_user = false;
    if (has_tool_role && tmpl) {
        const char * marker = "__ls_tool_probe_7f3a__";
        llama_chat_message probe_msg{"tool", marker};
        char probe_buf[512];
        int32_t pn = llama_chat_apply_template(tmpl, &probe_msg, 1, false, probe_buf, sizeof(probe_buf));
        if (pn >= 0 && pn < static_cast<int32_t>(sizeof(probe_buf))) {
            std::string rendered_probe(probe_buf, static_cast<size_t>(pn));
            if (rendered_probe.find(marker) == std::string::npos) {
                map_tool_to_tagged_user = true;
            }
        }
    }

    std::vector<std::string> wrapped_contents;
    std::vector<llama_chat_message> msgs;
    msgs.reserve(n_messages);
    if (map_tool_to_tagged_user) {
        wrapped_contents.reserve(n_messages);
    }
    for (size_t i = 0; i < n_messages; ++i) {
        const char * role = messages[i].role ? messages[i].role : "user";
        const char * content = messages[i].content ? messages[i].content : "";
        if (map_tool_to_tagged_user && std::strcmp(role, "tool") == 0) {
            wrapped_contents.push_back(std::string("<tool_response>\n") + content + "\n</tool_response>");
            msgs.push_back(llama_chat_message{"user", wrapped_contents.back().c_str()});
        } else {
            msgs.push_back(llama_chat_message{role, content});
        }
    }
    int32_t needed = llama_chat_apply_template(
        tmpl,
        msgs.data(),
        msgs.size(),
        add_assistant_prompt,
        out_buf,
        out_cap
    );
    if (needed < 0) {
        // Fallback to ChatML (which natively formats `<|im_start|>tool\n...`) if unrecognized.
        std::vector<llama_chat_message> raw_msgs;
        raw_msgs.reserve(n_messages);
        for (size_t i = 0; i < n_messages; ++i) {
            raw_msgs.push_back(llama_chat_message{
                messages[i].role ? messages[i].role : "user",
                messages[i].content ? messages[i].content : "",
            });
        }
        needed = llama_chat_apply_template(
            "chatml",
            raw_msgs.data(),
            raw_msgs.size(),
            add_assistant_prompt,
            out_buf,
            out_cap
        );
    }
    return needed;
}

int32_t ls_engine_tokenize(
    const ls_engine * eng,
    const char * text,
    size_t text_len,
    int32_t * out_tokens,
    int32_t max_tokens,
    bool add_special,
    bool parse_special
) {
    if (!eng || !eng->vocab || !text) {
        return -1;
    }
    return llama_tokenize(
        eng->vocab,
        text,
        static_cast<int32_t>(text_len),
        out_tokens,
        max_tokens,
        add_special,
        parse_special
    );
}

int32_t ls_engine_token_to_piece(
    const ls_engine * eng,
    int32_t token,
    char * out_buf,
    int32_t buf_cap
) {
    if (!eng || !eng->vocab || !out_buf) {
        return -1;
    }
    return llama_token_to_piece(eng->vocab, token, out_buf, buf_cap, 0, false);
}

bool ls_engine_token_is_eog(const ls_engine * eng, int32_t token) {
    if (!eng || !eng->vocab) {
        return true;
    }
    return llama_vocab_is_eog(eng->vocab, token);
}

void ls_engine_kv_clear(ls_engine * eng) {
    if (!eng || !eng->ctx) {
        return;
    }
    llama_memory_clear(llama_get_memory(eng->ctx), true);
}

bool ls_engine_kv_trim(ls_engine * eng, int32_t pos_min) {
    if (!eng || !eng->ctx) {
        return false;
    }
    return llama_memory_seq_rm(llama_get_memory(eng->ctx), 0, pos_min, -1);
}

int ls_engine_save_state(
    ls_engine * eng,
    const char * path,
    const int32_t * tokens,
    size_t n_tokens,
    char * err_buf,
    size_t err_cap
) {
    if (!eng || !eng->ctx || !path || path[0] == '\0' || (n_tokens > 0 && !tokens)) {
        ls_internal::set_error(err_buf, err_cap, "invalid arguments to ls_engine_save_state");
        return -1;
    }
    if (!llama_state_save_file(eng->ctx, path, tokens, n_tokens)) {
        ls_internal::set_error(
            err_buf,
            err_cap,
            std::string("llama_state_save_file failed for '") + path + "'"
        );
        return -2;
    }
    return 0;
}

int ls_engine_load_state(
    ls_engine * eng,
    const char * path,
    int32_t * out_tokens,
    size_t cap_tokens,
    size_t * out_n_tokens,
    char * err_buf,
    size_t err_cap
) {
    if (!eng || !eng->ctx || !path || path[0] == '\0' || (!out_tokens && cap_tokens > 0) || !out_n_tokens) {
        ls_internal::set_error(err_buf, err_cap, "invalid arguments to ls_engine_load_state");
        return -1;
    }
    *out_n_tokens = 0;
    llama_memory_clear(llama_get_memory(eng->ctx), true);
    size_t loaded_n = 0;
    if (!llama_state_load_file(eng->ctx, path, out_tokens, cap_tokens, &loaded_n)) {
        llama_memory_clear(llama_get_memory(eng->ctx), true);
        ls_internal::set_error(
            err_buf,
            err_cap,
            std::string("llama_state_load_file failed for '") + path +
                "' (verify file exists, matches this GGUF model & kv_cache_type, and fits within ctx_size)"
        );
        return -2;
    }
    *out_n_tokens = loaded_n;
    return 0;
}


namespace ls_internal {

llama_context * engine_ctx(const ls_engine * eng) {
    return eng ? eng->ctx : nullptr;
}

const llama_vocab * engine_vocab(const ls_engine * eng) {
    return eng ? eng->vocab : nullptr;
}

void engine_prepare_decode_pass(ls_engine * eng) {
    eng->prepare_decode_pass();
}

bool engine_is_cancelled(const ls_engine * eng) {
    return eng->is_cancelled();
}

bool engine_eval_failed(const ls_engine * eng) {
    return eng->eval_failed;
}

std::string engine_eval_error_msg(const ls_engine * eng) {
    return eng->eval_error_msg;
}

bool engine_profile_enabled(const ls_engine * eng) {
    return eng->profile_enabled;
}

void engine_record_decode_pass_us(ls_engine * eng, uint64_t elapsed_us) {
    eng->prof_pass_count.fetch_add(1, std::memory_order_relaxed);
    eng->prof_decode_total_us.fetch_add(elapsed_us, std::memory_order_relaxed);
}

}  // namespace ls_internal

