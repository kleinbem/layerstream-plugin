#include "stream_engine.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama-model.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef LAYERSTREAM_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace {

static std::once_flag g_llama_init_once;

static void ensure_llama_initialized() {
    std::call_once(g_llama_init_once, []() {
        // Prevent ggml-cuda from capturing 80 separate per-layer CUDA graphs
        // when cb_eval slices the compute graph at layer boundaries.
        setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 0);
        llama_backend_init();
    });
}

static void set_error(char * buf, size_t cap, const std::string & msg) {
    if (!buf || cap == 0) {
        return;
    }
    const size_t n = std::min(cap - 1, msg.size());
    std::memcpy(buf, msg.data(), n);
    buf[n] = '\0';
}

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

struct TensorBinding {
    std::string name;
    ggml_tensor * tensor = nullptr;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
};

struct LayerPlan {
    int32_t layer_idx = -1;
    uint64_t total_bytes = 0;
    std::vector<TensorBinding> tensors;
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

} // namespace

struct ls_engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;

    bool full_resident_reference = false;
    bool using_gpu = false;
    bool is_uma_zero_copy = true;
    bool can_speculate = false;

    uint32_t n_layers = 0;
    uint32_t resident_layers = 0;
    uint32_t streamed_layers = 0;
    uint64_t resident_bytes = 0;
    uint64_t slot_bytes = 0;
    std::atomic<uint64_t> total_layer_loads{0};

    // Real backend buffers owned by us (never freed by llama_model_free because no_alloc=true).
    ggml_backend_buffer_t resident_buf = nullptr;
    ggml_backend_buffer_t slot_buf[2] = {nullptr, nullptr};

    // Host staging buffers for discrete GPUs (page-aligned / cudaHostAlloc).
    uint8_t * pin_buf[2] = {nullptr, nullptr};
    size_t pin_cap[2] = {0, 0};
#ifdef LAYERSTREAM_USE_CUDA
    bool pin_is_cuda_host = false;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t slot_event[2] = {nullptr, nullptr};
#endif

    // Layer plans indexed by layer index [0..n_layers).
    std::vector<LayerPlan> layer_plans;
    LayerPlan resident_plan;

    // O(1) map from any streamed ggml_tensor* to its layer index.
    std::unordered_map<const ggml_tensor *, int32_t> tensor_to_layer;

    // IO callbacks into Rust.
    void * io_user_data = nullptr;
    ls_load_tensors_cb load_tensors_cb = nullptr;

    // Per-slot state for async background worker.
    std::mutex slot_mu;
    std::condition_variable slot_cv;
    int32_t slot_loaded_layer[2] = {-1, -1};
    int32_t slot_inflight_layer[2] = {-1, -1};
    int slot_error[2] = {0, 0};

    // Single-slot prefetch queue serviced by background thread.
    std::mutex worker_mu;
    std::condition_variable worker_cv;
    std::vector<PrefetchJob> pending_jobs;
    bool stop_worker = false;
    std::thread worker_thread;

    // Active decode pass tracking (accessed only from eval thread during llama_decode).
    int32_t active_stream_layer = -1;
    int32_t pending_release_layer = -1;
    int32_t highest_enqueued_layer = -1;
    bool eval_failed = false;

    int load_layer_into_slot(int32_t layer_idx, int slot) {
        const auto & plan = layer_plans[layer_idx];
        if (plan.tensors.empty()) {
            return 0;
        }

        if (is_uma_zero_copy) {
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
        } else {
            // Discrete GPU: O_DIRECT read into pinned staging buffer, then H2D copy.
            int rc = load_tensors_cb(
                io_user_data,
                layer_idx,
                plan.name_ptrs.data(),
                plan.offsets.data(),
                plan.sizes.data(),
                plan.tensors.size(),
                pin_buf[slot],
                pin_cap[slot]
            );
            if (rc != 0) {
                return rc;
            }

#ifdef LAYERSTREAM_USE_CUDA
            if (copy_stream != nullptr) {
                auto * gpu_dst = static_cast<uint8_t *>(ggml_backend_buffer_get_base(slot_buf[slot]));
                cudaError_t ce = cudaMemcpyAsync(
                    gpu_dst,
                    pin_buf[slot],
                    plan.total_bytes,
                    cudaMemcpyHostToDevice,
                    copy_stream
                );
                if (ce != cudaSuccess) {
                    return -10;
                }
                ce = cudaEventRecord(slot_event[slot], copy_stream);
                if (ce != cudaSuccess) {
                    return -11;
                }
            } else
#endif
            {
                // Generic backend fallback (e.g., discrete Vulkan): copy each tensor into GPU slot.
                for (const auto & tb : plan.tensors) {
                    ggml_backend_tensor_set(tb.tensor, pin_buf[slot] + tb.offset, 0, tb.nbytes);
                }
            }
        }

        total_layer_loads.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    void worker_loop() {
        while (true) {
            PrefetchJob job;
            {
                std::unique_lock<std::mutex> lk(worker_mu);
                worker_cv.wait(lk, [&]() { return stop_worker || !pending_jobs.empty(); });
                if (stop_worker) {
                    return;
                }
                job = pending_jobs.front();
                pending_jobs.erase(pending_jobs.begin());
            }

            int rc = load_layer_into_slot(job.layer_idx, job.slot);

            {
                std::lock_guard<std::mutex> lk(slot_mu);
                slot_inflight_layer[job.slot] = -1;
                if (rc == 0) {
                    slot_loaded_layer[job.slot] = job.layer_idx;
                    slot_error[job.slot] = 0;
                } else {
                    slot_loaded_layer[job.slot] = -1;
                    slot_error[job.slot] = rc;
                }
            }
            slot_cv.notify_all();
        }
    }

    void enqueue_prefetch(int32_t layer_idx) {
        if (layer_idx < static_cast<int32_t>(resident_layers) ||
            layer_idx >= static_cast<int32_t>(n_layers)) {
            return;
        }
        const int slot = (layer_idx - static_cast<int32_t>(resident_layers)) & 1;

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
        if (layer_idx < static_cast<int32_t>(resident_layers) ||
            layer_idx >= static_cast<int32_t>(n_layers)) {
            return true;
        }
        const int slot = (layer_idx - static_cast<int32_t>(resident_layers)) & 1;

        // If neither loaded nor in flight (e.g. first layer or skipped layer), enqueue now.
        enqueue_prefetch(layer_idx);

        int err = 0;
        {
            std::unique_lock<std::mutex> lk(slot_mu);
            slot_cv.wait(lk, [&]() {
                return slot_inflight_layer[slot] != layer_idx &&
                       (slot_loaded_layer[slot] == layer_idx || slot_error[slot] != 0);
            });
            err = slot_error[slot];
        }
        if (err != 0) {
            return false;
        }

#ifdef LAYERSTREAM_USE_CUDA
        if (!is_uma_zero_copy && copy_stream != nullptr && slot_event[slot] != nullptr) {
            if (cudaEventSynchronize(slot_event[slot]) != cudaSuccess) {
                return false;
            }
        }
#endif
        return true;
    }

    int32_t inspect_node_stream_layer(const ggml_tensor * node) const {
        if (!node) {
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

    void prepare_decode_pass() {
        eval_failed = false;
        if (streamed_layers == 0) {
            return;
        }
        // Drain any stale inflight jobs from a prior cancelled/aborted pass.
        {
            std::unique_lock<std::mutex> lk(slot_mu);
            slot_cv.wait(lk, [&]() {
                return slot_inflight_layer[0] == -1 && slot_inflight_layer[1] == -1;
            });
        }
        const int32_t first_stream = static_cast<int32_t>(resident_layers);
        if (!wait_slot_ready(first_stream)) {
            eval_failed = true;
            return;
        }
        active_stream_layer = first_stream;
        pending_release_layer = -1;
        highest_enqueued_layer = first_stream;

        if (first_stream + 1 < static_cast<int32_t>(n_layers)) {
            enqueue_prefetch(first_stream + 1);
            highest_enqueued_layer = first_stream + 1;
        }
    }

    static bool cb_eval(ggml_tensor * t, bool ask, void * user_data) {
        auto * self = static_cast<ls_engine *>(user_data);
        if (self->streamed_layers == 0 || self->eval_failed) {
            return false;
        }

        if (ask) {
            const int32_t op_layer = self->inspect_node_stream_layer(t);
            if (op_layer < 0 || op_layer == self->active_stream_layer) {
                return false;
            }

            // Transitioning to a new streamed layer: wait on its slot BEFORE
            // ggml_backend_sched executes gv = [j0 .. t]. Both the outgoing
            // layer's slot and incoming layer's slot remain valid during gv.
            if (!self->wait_slot_ready(op_layer)) {
                self->eval_failed = true;
                return false;
            }
            self->pending_release_layer = self->active_stream_layer;
            self->active_stream_layer = op_layer;
            return true;
        } else {
            // ggml_backend_synchronize(split_backend) has just completed.
            // `pending_release_layer`'s slot is now completely free; start
            // prefetching `active_stream_layer + 1` into that freed slot.
            const int32_t next_layer = self->active_stream_layer + 1;
            if (next_layer < static_cast<int32_t>(self->n_layers) &&
                next_layer > self->highest_enqueued_layer) {
                self->enqueue_prefetch(next_layer);
                self->highest_enqueued_layer = next_layer;
            }
            return true;
        }
    }
};

static bool select_target_buft(
    bool use_gpu,
    ggml_backend_buffer_type_t & out_buft,
    bool & out_using_gpu,
    bool & out_is_uma
) {
    out_using_gpu = false;
    out_is_uma = true;

    if (use_gpu) {
        const size_t n_dev = ggml_backend_dev_count();
        ggml_backend_dev_t chosen = nullptr;
        for (size_t i = 0; i < n_dev; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const auto kind = ggml_backend_dev_type(dev);
            if (kind == GGML_BACKEND_DEVICE_TYPE_GPU || kind == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                chosen = dev;
                break;
            }
        }
        if (chosen != nullptr) {
            out_using_gpu = true;
            out_buft = ggml_backend_dev_buffer_type(chosen);
            out_is_uma = ggml_backend_buft_is_host(out_buft);

#ifdef LAYERSTREAM_USE_CUDA
            // On integrated CUDA GPUs (e.g., Jetson Orin Nano), use mapped
            // cuda_host buffer type so O_DIRECT reads directly into GPU-visible
            // memory with zero H2D copy and half the RAM footprint.
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.integrated != 0) {
                ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(chosen);
                if (host_buft != nullptr && ggml_backend_dev_supports_buft(chosen, host_buft)) {
                    out_buft = host_buft;
                    out_is_uma = true;
                }
            }
#endif
            return true;
        }
    }

    out_buft = ggml_backend_cpu_buffer_type();
    out_using_gpu = false;
    out_is_uma = true;
    return out_buft != nullptr;
}

static uint8_t * alloc_aligned_4k(size_t size) {
    const size_t aligned = align_up(std::max<size_t>(size, 4096), 4096);
    void * ptr = nullptr;
    if (posix_memalign(&ptr, 4096, aligned) != 0) {
        return nullptr;
    }
    return static_cast<uint8_t *>(ptr);
}

ls_engine * ls_engine_create(const ls_engine_config * cfg, char * err_buf, size_t err_cap) {
    if (!cfg || !cfg->model_path) {
        set_error(err_buf, err_cap, "invalid ls_engine_config: model_path is null");
        return nullptr;
    }

    ensure_llama_initialized();

    auto * eng = new ls_engine();
    eng->full_resident_reference = cfg->full_resident_reference;
    eng->io_user_data = cfg->io_user_data;
    eng->load_tensors_cb = cfg->load_tensors_cb;

    const uint32_t n_ctx = cfg->n_ctx > 0 ? cfg->n_ctx : 2048;
    const uint32_t n_batch = cfg->n_batch > 0 ? cfg->n_batch : 512;
    const uint32_t n_threads = cfg->n_threads > 0 ? cfg->n_threads : 4;

    if (cfg->full_resident_reference) {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = cfg->use_gpu ? 999 : 0;
        mparams.no_alloc = false;

        eng->model = llama_model_load_from_file(cfg->model_path, mparams);
        if (!eng->model) {
            set_error(err_buf, err_cap, "llama_model_load_from_file failed (reference mode)");
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
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

        eng->ctx = llama_init_from_model(eng->model, cparams);
        if (!eng->ctx) {
            set_error(err_buf, err_cap, "llama_init_from_model failed (reference mode)");
            ls_engine_free(eng);
            return nullptr;
        }
        eng->can_speculate = llama_memory_can_shift(llama_get_memory(eng->ctx));
        return eng;
    }

    if (!cfg->load_tensors_cb) {
        set_error(err_buf, err_cap, "load_tensors_cb must be set in streaming mode");
        ls_engine_free(eng);
        return nullptr;
    }

    ggml_backend_buffer_type_t target_buft = nullptr;
    if (!select_target_buft(cfg->use_gpu, target_buft, eng->using_gpu, eng->is_uma_zero_copy)) {
        set_error(err_buf, err_cap, "failed to select target ggml backend buffer type");
        ls_engine_free(eng);
        return nullptr;
    }

    // 1. Load metadata and tensor graph with zero memory allocation.
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = eng->using_gpu ? 999 : 0;
    mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    mparams.no_alloc = true;

    eng->model = llama_model_load_from_file(cfg->model_path, mparams);
    if (!eng->model) {
        set_error(err_buf, err_cap, "llama_model_load_from_file(no_alloc=true) failed");
        ls_engine_free(eng);
        return nullptr;
    }
    eng->vocab = llama_model_get_vocab(eng->model);
    eng->n_layers = static_cast<uint32_t>(llama_model_n_layer(eng->model));
    eng->layer_plans.resize(eng->n_layers);
    for (uint32_t i = 0; i < eng->n_layers; ++i) {
        eng->layer_plans[i].layer_idx = static_cast<int32_t>(i);
    }

    // 2. Partition all model tensors into non-layer resident tensors vs blk.{L}.* layers.
    const size_t buft_align = std::max<size_t>(ggml_backend_buft_get_alignment(target_buft), 256);
    std::unordered_set<const ggml_tensor *> seen_ptrs;

    std::vector<TensorBinding> non_layer_tensors;
    for (const auto & kv : eng->model->tensors_by_name) {
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
            non_layer_tensors.push_back(std::move(tb));
        }
    }

    // Compute per-layer aligned byte sizes.
    uint64_t max_layer_bytes = 0;
    for (auto & plan : eng->layer_plans) {
        uint64_t off = 0;
        for (auto & tb : plan.tensors) {
            off = align_up(off, buft_align);
            tb.offset = off;
            const size_t alloc_sz = std::max<size_t>(
                ggml_backend_buft_get_alloc_size(target_buft, tb.tensor),
                static_cast<size_t>(tb.nbytes)
            );
            off += alloc_sz;
        }
        plan.total_bytes = align_up(off, 4096);
        max_layer_bytes = std::max(max_layer_bytes, plan.total_bytes);
    }

    // Compute non-layer resident size.
    uint64_t base_resident_bytes = 0;
    for (const auto & tb : non_layer_tensors) {
        base_resident_bytes = align_up(base_resident_bytes, buft_align);
        const size_t alloc_sz = std::max<size_t>(
            ggml_backend_buft_get_alloc_size(target_buft, tb.tensor),
            static_cast<size_t>(tb.nbytes)
        );
        base_resident_bytes += alloc_sz;
    }

    // 3. Decide hybrid partial-residency split R = resident_layers in [0..n_layers].
    uint32_t R = 0;
    if (cfg->resident_layers >= 0) {
        R = std::min<uint32_t>(static_cast<uint32_t>(cfg->resident_layers), eng->n_layers);
    } else if (cfg->vram_budget_bytes > 0) {
        // Reserve 2 ping-pong slots plus conservative 512 MiB headroom for KV + compute graph.
        const uint64_t reserve = 2 * max_layer_bytes + (512ULL * 1024 * 1024);
        uint64_t used = base_resident_bytes + reserve;
        while (R < eng->n_layers && used + eng->layer_plans[R].total_bytes <= cfg->vram_budget_bytes) {
            used += eng->layer_plans[R].total_bytes;
            ++R;
        }
    }
    eng->resident_layers = R;
    eng->streamed_layers = eng->n_layers - R;

    // 4. Build combined resident plan (non-layer tensors + blk.0..R-1).
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

    for (auto & tb : non_layer_tensors) {
        push_resident(std::move(tb));
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

    // 5. Allocate real backend buffers: Resident + (if streaming) Slot A & Slot B.
    eng->resident_buf = ggml_backend_buft_alloc_buffer(target_buft, eng->resident_bytes);
    if (!eng->resident_buf) {
        set_error(err_buf, err_cap, "failed to allocate resident backend buffer");
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

    // Populate resident tensors once from GGUF.
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
                set_error(err_buf, err_cap, "failed to load resident tensors (UMA/host)");
                ls_engine_free(eng);
                return nullptr;
            }
        } else {
            uint8_t * staging = alloc_aligned_4k(eng->resident_bytes);
            if (!staging) {
                set_error(err_buf, err_cap, "failed to allocate resident staging buffer");
                ls_engine_free(eng);
                return nullptr;
            }
            int rc = eng->load_tensors_cb(
                eng->io_user_data,
                -1,
                eng->resident_plan.name_ptrs.data(),
                eng->resident_plan.offsets.data(),
                eng->resident_plan.sizes.data(),
                eng->resident_plan.tensors.size(),
                staging,
                eng->resident_bytes
            );
            if (rc == 0) {
                for (const auto & tb : eng->resident_plan.tensors) {
                    ggml_backend_tensor_set(tb.tensor, staging + tb.offset, 0, tb.nbytes);
                }
            }
            std::free(staging);
            if (rc != 0) {
                set_error(err_buf, err_cap, "failed to load resident tensors (discrete)");
                ls_engine_free(eng);
                return nullptr;
            }
        }
    }

    // 6. Allocate 2 real ping-pong slot buffers for streamed layers [R..n_layers).
    if (eng->streamed_layers > 0) {
        uint64_t streamed_max = 0;
        for (uint32_t l = R; l < eng->n_layers; ++l) {
            streamed_max = std::max(streamed_max, eng->layer_plans[l].total_bytes);
            eng->layer_plans[l].finalize_arrays();
        }
        eng->slot_bytes = align_up(std::max<uint64_t>(streamed_max, 4096), 4096);

        for (int s = 0; s < 2; ++s) {
            eng->slot_buf[s] = ggml_backend_buft_alloc_buffer(target_buft, eng->slot_bytes);
            if (!eng->slot_buf[s]) {
                set_error(err_buf, err_cap, "failed to allocate ping-pong GPU slot buffer");
                ls_engine_free(eng);
                return nullptr;
            }
            ggml_backend_buffer_set_usage(eng->slot_buf[s], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        if (!eng->is_uma_zero_copy) {
#ifdef LAYERSTREAM_USE_CUDA
            if (cudaStreamCreateWithFlags(&eng->copy_stream, cudaStreamNonBlocking) == cudaSuccess) {
                eng->pin_is_cuda_host = true;
                for (int s = 0; s < 2; ++s) {
                    if (cudaHostAlloc(reinterpret_cast<void **>(&eng->pin_buf[s]), eng->slot_bytes, cudaHostAllocDefault) != cudaSuccess ||
                        cudaEventCreateWithFlags(&eng->slot_event[s], cudaEventDisableTiming) != cudaSuccess) {
                        set_error(err_buf, err_cap, "failed to allocate CUDA pinned staging buffer");
                        ls_engine_free(eng);
                        return nullptr;
                    }
                    eng->pin_cap[s] = eng->slot_bytes;
                }
            } else
#endif
            {
                for (int s = 0; s < 2; ++s) {
                    eng->pin_buf[s] = alloc_aligned_4k(eng->slot_bytes);
                    eng->pin_cap[s] = eng->slot_bytes;
                    if (!eng->pin_buf[s]) {
                        set_error(err_buf, err_cap, "failed to allocate 4K-aligned host staging buffer");
                        ls_engine_free(eng);
                        return nullptr;
                    }
                }
            }
        }

        // Alias every streamed layer tensor into Slot A (even) or Slot B (odd).
        for (uint32_t l = R; l < eng->n_layers; ++l) {
            const int slot = static_cast<int>((l - R) & 1);
            auto * slot_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(eng->slot_buf[slot]));
            for (const auto & tb : eng->layer_plans[l].tensors) {
                tb.tensor->buffer = eng->slot_buf[slot];
                tb.tensor->data = slot_base + tb.offset;
                ggml_backend_buffer_init_tensor(eng->slot_buf[slot], tb.tensor);
                eng->tensor_to_layer[tb.tensor] = static_cast<int32_t>(l);
            }
        }

        eng->worker_thread = std::thread([eng]() { eng->worker_loop(); });
    }

    // 7. Critical: clear hparams.no_alloc BEFORE llama_init_from_model so
    // llama_context allocates real KV cache + compute scheduler buffers!
    eng->model->hparams.no_alloc = false;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_batch;
    cparams.n_ubatch = n_batch;
    cparams.n_threads = static_cast<int32_t>(n_threads);
    cparams.n_threads_batch = static_cast<int32_t>(n_threads);
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (eng->streamed_layers > 0) {
        cparams.cb_eval = &ls_engine::cb_eval;
        cparams.cb_eval_user_data = eng;
    }

    eng->ctx = llama_init_from_model(eng->model, cparams);
    if (!eng->ctx) {
        set_error(err_buf, err_cap, "llama_init_from_model failed");
        ls_engine_free(eng);
        return nullptr;
    }

    eng->can_speculate = llama_memory_can_shift(llama_get_memory(eng->ctx));
    return eng;
}

void ls_engine_free(ls_engine * eng) {
    if (!eng) {
        return;
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

#ifdef LAYERSTREAM_USE_CUDA
    for (int s = 0; s < 2; ++s) {
        if (eng->slot_event[s]) {
            cudaEventDestroy(eng->slot_event[s]);
            eng->slot_event[s] = nullptr;
        }
        if (eng->pin_buf[s]) {
            if (eng->pin_is_cuda_host) {
                cudaFreeHost(eng->pin_buf[s]);
            } else {
                std::free(eng->pin_buf[s]);
            }
            eng->pin_buf[s] = nullptr;
        }
    }
    if (eng->copy_stream) {
        cudaStreamDestroy(eng->copy_stream);
        eng->copy_stream = nullptr;
    }
#else
    for (int s = 0; s < 2; ++s) {
        if (eng->pin_buf[s]) {
            std::free(eng->pin_buf[s]);
            eng->pin_buf[s] = nullptr;
        }
    }
#endif

    for (int s = 0; s < 2; ++s) {
        if (eng->slot_buf[s]) {
            ggml_backend_buffer_free(eng->slot_buf[s]);
            eng->slot_buf[s] = nullptr;
        }
    }
    if (eng->resident_buf) {
        ggml_backend_buffer_free(eng->resident_buf);
        eng->resident_buf = nullptr;
    }

    delete eng;
}

void ls_engine_get_stats(const ls_engine * eng, ls_engine_stats * out) {
    if (!eng || !out) {
        return;
    }
    out->n_layers = eng->n_layers;
    out->resident_layers = eng->resident_layers;
    out->streamed_layers = eng->streamed_layers;
    out->resident_bytes = eng->resident_bytes;
    out->slot_bytes = eng->slot_bytes;
    out->total_layer_loads = eng->total_layer_loads.load(std::memory_order_relaxed);
    out->is_uma_zero_copy = eng->is_uma_zero_copy;
    out->using_gpu = eng->using_gpu;
    out->can_speculate = eng->can_speculate;
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
    std::vector<llama_chat_message> msgs;
    msgs.reserve(n_messages);
    for (size_t i = 0; i < n_messages; ++i) {
        msgs.push_back(llama_chat_message{
            messages[i].role ? messages[i].role : "user",
            messages[i].content ? messages[i].content : "",
        });
    }
    const char * tmpl = llama_model_chat_template(eng->model, nullptr);
    int32_t needed = llama_chat_apply_template(
        tmpl,
        msgs.data(),
        msgs.size(),
        add_assistant_prompt,
        out_buf,
        out_cap
    );
    if (needed < 0) {
        // Fallback to ChatML if model uses an unrecognized custom Jinja template.
        needed = llama_chat_apply_template(
            "chatml",
            msgs.data(),
            msgs.size(),
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
    return llama_token_to_piece(eng->vocab, token, out_buf, buf_cap, 0, true);
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

static int32_t argmax_logits(const float * logits, int32_t n_vocab) {
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

int ls_engine_eval_greedy(
    ls_engine * eng,
    const int32_t * tokens,
    int32_t n_tokens,
    int32_t pos_start,
    int32_t * out_next_token,
    char * err_buf,
    size_t err_cap
) {
    if (!eng || !eng->ctx || !tokens || n_tokens <= 0 || !out_next_token) {
        set_error(err_buf, err_cap, "invalid arguments to ls_engine_eval_greedy");
        return -1;
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos_start + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }

    eng->prepare_decode_pass();
    if (eng->eval_failed) {
        llama_batch_free(batch);
        set_error(err_buf, err_cap, "layer prefetch failed before llama_decode");
        return -2;
    }

    const int rc = llama_decode(eng->ctx, batch);
    llama_batch_free(batch);

    if (rc != 0 || eng->eval_failed) {
        set_error(err_buf, err_cap, "llama_decode failed");
        return rc != 0 ? rc : -3;
    }

    const float * logits = llama_get_logits_ith(eng->ctx, n_tokens - 1);
    if (!logits) {
        set_error(err_buf, err_cap, "llama_get_logits_ith returned null");
        return -4;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(eng->vocab);
    *out_next_token = argmax_logits(logits, n_vocab);
    return 0;
}

int ls_engine_eval_sample(
    ls_engine * eng,
    const int32_t * tokens,
    int32_t n_tokens,
    int32_t pos_start,
    const ls_sample_params * sample_params,
    int32_t * out_next_token,
    char * err_buf,
    size_t err_cap
) {
    if (!sample_params || sample_params->temperature <= 0.0f) {
        return ls_engine_eval_greedy(eng, tokens, n_tokens, pos_start, out_next_token, err_buf, err_cap);
    }
    if (!eng || !eng->ctx || !tokens || n_tokens <= 0 || !out_next_token) {
        set_error(err_buf, err_cap, "invalid arguments to ls_engine_eval_sample");
        return -1;
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos_start + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }

    eng->prepare_decode_pass();
    if (eng->eval_failed) {
        llama_batch_free(batch);
        set_error(err_buf, err_cap, "layer prefetch failed before sampled llama_decode");
        return -2;
    }

    const int rc = llama_decode(eng->ctx, batch);
    llama_batch_free(batch);

    if (rc != 0 || eng->eval_failed) {
        set_error(err_buf, err_cap, "sampled llama_decode failed");
        return rc != 0 ? rc : -3;
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * chain = llama_sampler_chain_init(sparams);
    if (sample_params->top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(sample_params->top_k));
    }
    if (sample_params->top_p > 0.0f && sample_params->top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(sample_params->top_p, 1));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_temp(sample_params->temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(sample_params->seed));

    *out_next_token = llama_sampler_sample(chain, eng->ctx, n_tokens - 1);
    llama_sampler_free(chain);
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
    if (!eng || !eng->ctx || !tokens || n_tokens <= 0 || !out_argmax_tokens) {
        set_error(err_buf, err_cap, "invalid arguments to ls_engine_eval_verify_greedy");
        return -1;
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos_start + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1; // request logits for every verified position
    }

    eng->prepare_decode_pass();
    if (eng->eval_failed) {
        llama_batch_free(batch);
        set_error(err_buf, err_cap, "layer prefetch failed before verify llama_decode");
        return -2;
    }

    const int rc = llama_decode(eng->ctx, batch);
    llama_batch_free(batch);

    if (rc != 0 || eng->eval_failed) {
        set_error(err_buf, err_cap, "verify llama_decode failed");
        return rc != 0 ? rc : -3;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(eng->vocab);
    for (int32_t i = 0; i < n_tokens; ++i) {
        const float * logits = llama_get_logits_ith(eng->ctx, i);
        if (!logits) {
            set_error(err_buf, err_cap, "llama_get_logits_ith returned null during verify");
            return -4;
        }
        out_argmax_tokens[i] = argmax_logits(logits, n_vocab);
    }
    return 0;
}
