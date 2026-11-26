#include "stream_engine_state.hpp"

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
        mparams.load_mtp = true;
        if (!cfg->use_gpu && g_cpu_only_devs[0] != nullptr) {
            mparams.devices = g_cpu_only_devs;
        } else if (cfg->use_gpu && ref_gpu_dev != nullptr) {
            // Pass only the GPU device: llama.cpp automatically adds CPU as a fallback
            // and treats every entry in mparams.devices as a layer-split compute target.
            active_devs[0] = ref_gpu_dev;
            active_devs[1] = nullptr;
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
        cparams.n_rs_seq = 16;
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
        eng->rs_rollback_max = llama_n_rs_seq(eng->ctx);
        const bool can_kv_trim =
            mem != nullptr && llama_memory_can_shift(mem) && llama_memory_seq_rm(mem, 0, 1, -1);
        eng->can_speculate = can_kv_trim || eng->rs_rollback_max > 0;
        eng->init_mtp_context(n_ctx, n_batch, n_threads, cfg->kv_cache_type);
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
    mparams.load_mtp = true;
    if (!eng->using_gpu && g_cpu_only_devs[0] != nullptr) {
        mparams.devices = g_cpu_only_devs;
    } else if (eng->using_gpu && chosen_gpu_dev != nullptr) {
        // Pass only the GPU device: llama.cpp automatically adds CPU as a fallback
        // and treats every entry in mparams.devices as a layer-split compute target.
        active_devs[0] = chosen_gpu_dev;
        active_devs[1] = nullptr;
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

            constexpr uint64_t kMiB = 1024ULL * 1024ULL;
            if (cfg->resident_layers < 0) {
                if (cfg->vram_budget_auto) {
                    // When auto=true, cfg->vram_budget_bytes carries optional draft reserve.
                    effective_budget = (safe_cap > cfg->vram_budget_bytes)
                        ? (safe_cap - cfg->vram_budget_bytes)
                        : 0;
                    std::fprintf(
                        stderr,
                        "[layerstream-guard] auto VRAM budget: %llu MiB (probed %llu / %llu MiB free, unified=%s)\n",
                        static_cast<unsigned long long>(effective_budget / kMiB),
                        static_cast<unsigned long long>(free_vram / kMiB),
                        static_cast<unsigned long long>(total_vram / kMiB),
                        eng->is_unified_memory ? "true" : "false"
                    );
                } else if (effective_budget > safe_cap) {
                    std::fprintf(
                        stderr,
                        "[layerstream-guard] requested --vram-budget-mb (%llu MiB) exceeds safe free GPU memory "
                        "(%llu MiB free / %llu MiB total); clamping to %llu MiB to prevent OOM/swap thrashing\n",
                        static_cast<unsigned long long>(effective_budget / kMiB),
                        static_cast<unsigned long long>(free_vram / kMiB),
                        static_cast<unsigned long long>(total_vram / kMiB),
                        static_cast<unsigned long long>(safe_cap / kMiB)
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
                    constexpr uint64_t kMiB = 1024ULL * 1024ULL;
                    std::fprintf(
                        stderr,
                        "[layerstream-guard] requested resident_layers=%u exceeds safe GPU cap (%llu MiB); clamping to %u\n",
                        R,
                        static_cast<unsigned long long>(max_allowed / kMiB),
                        clamped_r
                    );
                    R = clamped_r;
                }
            }
        }
    } else if (effective_budget > 0) {
        if (kv_bytes * 2 > effective_budget && cfg->kv_cache_type == 0) {
            constexpr uint64_t kMiB = 1024ULL * 1024ULL;
            std::fprintf(
                stderr,
                "[layerstream-guard] warning: F16 KV cache (~%llu MiB at ctx=%u) exceeds 50%% of VRAM budget (%llu MiB); "
                "consider --kv-cache-type q8_0 or q4_0\n",
                static_cast<unsigned long long>(kv_bytes / kMiB),
                n_ctx,
                static_cast<unsigned long long>(effective_budget / kMiB)
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
        const int n_pin_bufs = (eng->streamed_dense_layers > 0) ? 2 : 1;
#ifdef LAYERSTREAM_HAS_CUDA_RT
        if (eng->is_cuda_backend && ls_cuda::ensure_loaded() &&
            ls_cuda::stream_create_with_flags(&eng->copy_stream, ls_cuda::kStreamNonBlocking) == ls_cuda::kSuccess) {
            eng->pin_is_cuda_host = true;
            eng->pin_cap = pin_bytes;
            for (int s = 0; s < n_pin_bufs; ++s) {
                if (ls_cuda::host_alloc(reinterpret_cast<void **>(&eng->pin_buf[s]), pin_bytes, ls_cuda::kHostAllocDefault) != ls_cuda::kSuccess) {
                    ls_internal::set_error(err_buf, err_cap, "failed to allocate CUDA pinned staging buffer");
                    ls_engine_free(eng);
                    return nullptr;
                }
                std::memset(eng->pin_buf[s], 0, pin_bytes);
            }
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
            eng->pin_cap = pin_bytes;
            for (int s = 0; s < n_pin_bufs; ++s) {
                eng->pin_buf[s] = alloc_aligned_4k(pin_bytes);
                if (!eng->pin_buf[s]) {
                    ls_internal::set_error(err_buf, err_cap, "failed to allocate 4K-aligned host staging buffer");
                    ls_engine_free(eng);
                    return nullptr;
                }
                std::memset(eng->pin_buf[s], 0, pin_bytes);
            }
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
    cparams.n_rs_seq = 16;
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
    eng->rs_rollback_max = llama_n_rs_seq(eng->ctx);
    const bool can_kv_trim =
        mem != nullptr && llama_memory_can_shift(mem) && llama_memory_seq_rm(mem, 0, 1, -1);
    eng->can_speculate = can_kv_trim || eng->rs_rollback_max > 0;
    eng->init_mtp_context(n_ctx, n_batch, n_threads, cfg->kv_cache_type);
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

    if (eng->ctx_mtp) {
        llama_free(eng->ctx_mtp);
        eng->ctx_mtp = nullptr;
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
    for (int s = 0; s < 2; ++s) {
        if (eng->pin_buf[s]) {
            if (eng->pin_is_cuda_host) {
                ls_cuda::free_host(eng->pin_buf[s]);
            } else {
                free_aligned_4k(eng->pin_buf[s]);
            }
            eng->pin_buf[s] = nullptr;
        }
    }
    if (eng->copy_stream) {
        ls_cuda::stream_destroy(eng->copy_stream);
        eng->copy_stream = nullptr;
    }
#else
    for (int s = 0; s < 2; ++s) {
        if (eng->pin_buf[s]) {
            free_aligned_4k(eng->pin_buf[s]);
            eng->pin_buf[s] = nullptr;
        }
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
    out->rs_rollback_max = eng->rs_rollback_max;
    out->mtp_layers = eng->mtp_layers;
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
    eng->mtp_reset_state();
}

bool ls_engine_kv_trim(ls_engine * eng, int32_t pos_min) {
    if (!eng || !eng->ctx) {
        return false;
    }
    const bool ok = llama_memory_seq_rm(llama_get_memory(eng->ctx), 0, pos_min, -1);
    if (ok && eng->ctx_mtp && !eng->mtp_is_mem_shared) {
        llama_memory_seq_rm(llama_get_memory(eng->ctx_mtp), 0, pos_min, -1);
    }
    return ok;
}

void ls_engine_mtp_accept(ls_engine * eng, int32_t n_accepted) {
    if (!eng || !eng->ctx_mtp || eng->mtp_verify_rows <= 0 || eng->mtp_n_embd <= 0) {
        return;
    }
    const int32_t clamped = std::max<int32_t>(0, std::min<int32_t>(n_accepted, eng->mtp_verify_rows - 1));
    const size_t n_embd_sz = static_cast<size_t>(eng->mtp_n_embd);
    std::memcpy(
        eng->mtp_pending_h.data(),
        eng->mtp_verify_h.data() + static_cast<size_t>(clamped) * n_embd_sz,
        n_embd_sz * sizeof(float)
    );
}

int32_t ls_engine_mtp_draft(
    ls_engine * eng,
    int32_t last_token,
    int32_t pos0,
    int32_t max_draft,
    int32_t * out_tokens,
    char * err_buf,
    size_t err_cap
) {
    if (!eng || !out_tokens || max_draft <= 0) {
        ls_internal::set_error(err_buf, err_cap, "invalid arguments to ls_engine_mtp_draft");
        return -1;
    }
    if (!eng->ctx_mtp || !eng->vocab || eng->mtp_n_embd <= 0 || eng->mtp_pending_h.empty()) {
        return 0;
    }
    const int32_t vocab_n = llama_vocab_n_tokens(eng->vocab);
    if (vocab_n <= 0) {
        return 0;
    }
    const int32_t eff_max = eng->mtp_chain_heads
        ? std::min<int32_t>(max_draft, static_cast<int32_t>(eng->mtp_layers))
        : max_draft;
    const size_t n_embd_sz = static_cast<size_t>(eng->mtp_n_embd);

    llama_batch_ext * mb = llama_batch_ext_init(eng->ctx_mtp);
    if (!mb) {
        ls_internal::set_error(err_buf, err_cap, "llama_batch_ext_init failed for MTP context");
        return -2;
    }

    std::vector<float> chain_h;
    if (eng->mtp_chain_heads) {
        chain_h = eng->mtp_pending_h;
    }

    int32_t last_idx = llama_batch_ext_add_token(mb, 0, last_token);
    if (last_idx < 0) {
        llama_batch_ext_free(mb);
        return 0;
    }
    const llama_pos p0 = pos0;
    llama_batch_ext_set_pos(mb, last_idx, &p0);
    llama_batch_ext_set_output_logits(mb, last_idx, true);
    llama_batch_ext_set_embd_state(mb, last_idx, llama_embd{eng->mtp_pending_h.data(), 1, n_embd_sz});

    int32_t drafted = 0;
    llama_memory_t mem_mtp = llama_get_memory(eng->ctx_mtp);

    for (int32_t step = 0; step < eff_max; ++step) {
        if (eng->is_cancelled()) {
            break;
        }
        if (eng->mtp_chain_heads) {
            llama_memory_seq_rm(mem_mtp, 0, pos0, -1);
            llama_set_nextn_layer_offset(eng->ctx_mtp, step);
        }
        if (llama_process(eng->ctx_mtp, LLAMA_PROCESS_TYPE_DECODE, mb) != 0) {
            break;
        }

        const float * logits = llama_get_logits_ith(eng->ctx_mtp, last_idx);
        if (!logits) {
            break;
        }
        int32_t best_id = 0;
        float best_val = logits[0];
        for (int32_t v = 1; v < vocab_n; ++v) {
            if (logits[v] > best_val) {
                best_val = logits[v];
                best_id = v;
            }
        }
        if (llama_vocab_is_eog(eng->vocab, best_id)) {
            break;
        }
        out_tokens[drafted++] = best_id;
        if (drafted >= eff_max) {
            break;
        }

        const float * h_row = llama_get_embeddings_nextn_ith(eng->ctx_mtp, last_idx);
        if (!h_row) {
            break;
        }
        llama_batch_ext_clear(mb);
        if (eng->mtp_chain_heads) {
            chain_h.insert(chain_h.end(), h_row, h_row + n_embd_sz);
            const int32_t n_rows = drafted + 1;
            bool batch_err = false;
            for (int32_t t = 0; t < n_rows; ++t) {
                const int32_t tok = (t == 0) ? last_token : out_tokens[t - 1];
                const int32_t idx = llama_batch_ext_add_token(mb, 0, tok);
                if (idx < 0) {
                    batch_err = true;
                    break;
                }
                const llama_pos pt = pos0 + t;
                llama_batch_ext_set_pos(mb, idx, &pt);
                llama_batch_ext_set_output_logits(mb, idx, t == n_rows - 1);
                llama_batch_ext_set_embd_state(
                    mb, idx, llama_embd{chain_h.data() + static_cast<size_t>(t) * n_embd_sz, 1, n_embd_sz}
                );
                last_idx = idx;
            }
            if (batch_err) {
                break;
            }
        } else {
            last_idx = llama_batch_ext_add_token(mb, 0, best_id);
            if (last_idx < 0) {
                break;
            }
            const llama_pos pn = eng->mtp_is_mem_shared ? pos0 : (pos0 + step + 1);
            llama_batch_ext_set_pos(mb, last_idx, &pn);
            llama_batch_ext_set_output_logits(mb, last_idx, true);
            llama_batch_ext_set_embd_state(mb, last_idx, llama_embd{h_row, 1, n_embd_sz});
        }
    }

    if (eng->mtp_chain_heads) {
        llama_set_nextn_layer_offset(eng->ctx_mtp, 0);
    }
    if (!eng->mtp_is_mem_shared) {
        llama_memory_seq_rm(mem_mtp, 0, pos0, -1);
    }
    llama_batch_ext_free(mb);
    return drafted;
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


extern "C" {

int ls_engine_run_decode_batch(
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
    if (!eng || !eng->ctx || !tokens || n_tokens <= 0) {
        ls_internal::set_error(err_buf, err_cap, "invalid arguments to ls_engine_run_decode_batch");
        return -1;
    }

    auto report_failure = [&](int rc, const char * fallback_msg) -> int {
        if (eng->is_cancelled()) {
            ls_internal::set_error(err_buf, err_cap, "request cancelled");
            return -100;
        }
        if (!eng->eval_error_msg.empty()) {
            ls_internal::set_error(err_buf, err_cap, eng->eval_error_msg);
        } else {
            ls_internal::set_error(err_buf, err_cap, fallback_msg);
        }
        return rc != 0 ? rc : -3;
    };

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
    } guard(eng->ctx);

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
    const auto t0 = eng->profile_enabled ? Clock::now() : Clock::time_point{};
    eng->prepare_decode_pass();
    if (eng->eval_failed) {
        return report_failure(-2, prefetch_err_msg);
    }

    const int rc = llama_process(eng->ctx, LLAMA_PROCESS_TYPE_DECODE, guard.batch);
    if (eng->profile_enabled) {
        const auto dt = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
        eng->prof_pass_count.fetch_add(1, std::memory_order_relaxed);
        eng->prof_decode_total_us.fetch_add(static_cast<uint64_t>(dt), std::memory_order_relaxed);
    }
    if (rc != 0 || eng->eval_failed) {
        return report_failure(rc, decode_err_msg);
    }
    eng->mtp_on_target_decode(tokens, n_tokens, pos_start);
    return 0;
}

const float * ls_engine_get_logits_ith(const ls_engine * eng, int32_t idx) {
    if (!eng || !eng->ctx) {
        return nullptr;
    }
    return llama_get_logits_ith(eng->ctx, idx);
}

const llama_vocab * ls_engine_vocab_ptr(const ls_engine * eng) {
    return eng ? eng->vocab : nullptr;
}

llama_sampler * ls_llama_grammar_init_nothrow(
    const llama_vocab * vocab,
    const char * grammar_str,
    const char * root,
    char * err_buf,
    size_t err_cap
) {
    try {
        llama_sampler * smpl = llama_sampler_init_grammar(vocab, grammar_str, root);
        if (!smpl) {
            ls_internal::set_error(err_buf, err_cap, "failed to parse GBNF grammar");
        }
        return smpl;
    } catch (const std::exception & e) {
        ls_internal::set_error(err_buf, err_cap, std::string("GBNF grammar init exception: ") + e.what());
        return nullptr;
    } catch (...) {
        ls_internal::set_error(err_buf, err_cap, "unknown GBNF grammar init exception");
        return nullptr;
    }
}

int ls_llama_grammar_accept_nothrow(
    llama_sampler * smpl,
    int32_t token,
    char * err_buf,
    size_t err_cap
) {
    if (!smpl) {
        return 0;
    }
    try {
        llama_sampler_accept(smpl, token);
        return 0;
    } catch (const std::exception & e) {
        ls_internal::set_error(err_buf, err_cap, std::string("grammar accept failed: ") + e.what());
        return -1;
    } catch (...) {
        ls_internal::set_error(err_buf, err_cap, "unknown grammar accept exception");
        return -1;
    }
}

int ls_llama_sampler_apply_nothrow(
    llama_sampler * smpl,
    llama_token_data_array * cur_p,
    char * err_buf,
    size_t err_cap
) {
    if (!smpl || !cur_p) {
        return 0;
    }
    try {
        llama_sampler_apply(smpl, cur_p);
        return 0;
    } catch (const std::exception & e) {
        ls_internal::set_error(err_buf, err_cap, std::string("sampler apply failed: ") + e.what());
        return -6;
    } catch (...) {
        ls_internal::set_error(err_buf, err_cap, "unknown sampler apply exception");
        return -6;
    }
}

}  // extern "C"

