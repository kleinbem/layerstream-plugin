# Upstream `llama.cpp` Tracking

`layerstream-plugin` builds against unmodified upstream [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) (`master`, pinned in [`.github/workflows/maintain.yml`](../.github/workflows/maintain.yml) and [`crates/layerstream-core/build.rs`](../crates/layerstream-core/build.rs)).

## Patch Policy

- **Zero local patches by default.** Peer projects that carry large `llama.cpp` patch queues (`llamafile` with 49 patches, `llama-box` with 26 patches, `mesh-llm` with ~1.86 MB of pipeline-stage patches) pay a continuous rebase tax. `ollama` recently moved to upstream `llama-server` with a single temporary compatibility hook patch.
- **Isolate internal header access.** Every direct `llama-model.h` touchpoint lives in [`crates/layerstream-core/cpp/llama_compat.hpp`](../crates/layerstream-core/cpp/llama_compat.hpp) (currently 5 helpers: `tensors_by_name`, `is_tok_embd_tensor`, `is_output_head_tensor`, `estimate_kv_cache_bytes`, and `clear_model_no_alloc`).
- **Upstream missing primitives instead of patching locally.** Keep temporary workarounds in `llama_compat.hpp` linked to their upstream issue/PR, and drop them once upstream ships the public API.

## Our Upstream PR Queue

| Priority | Item | Status | Impact on `layerstream-plugin` |
|---|---|---|---|
| 1 | [`ggml-org/llama.cpp#32893`](https://github.com/ggml-org/llama.cpp/pull/32893) — `llama : allocate context memory when no_alloc model has bound weights` | Open (CI green, awaiting review) | Removes `clear_model_no_alloc()` protected-member workaround in `llama_compat.hpp`. |
| 2 | Public tensor-enumeration & per-layer KV metadata C API in `llama.h` | Issue-first draft next | Drops `#include "llama-model.h"` completely so `layerstream-plugin` builds 100% against public `llama.h` + `ggml-backend.h`. |
| 3 | Per-context / runtime CUDA Graph toggle (replacing process-wide `static const` `GGML_CUDA_DISABLE_GRAPHS` in `graph.cuh`) | Planned (after `#27330` / `#29768`) | Enables CUDA Graphs for 100%-resident models while keeping them disabled for streamed models in the same process. |

## Tracked Upstream PRs & Issues

| Ref | Type | Area | Why it matters to `layerstream-plugin` | Action |
|---|---|---|---|---|
| [`ggml-org/llama.cpp#32893`](https://github.com/ggml-org/llama.cpp/pull/32893) | PR (ours) | Core / `no_alloc` | Allocates KV/compute buffers when `no_alloc=true` models have externally bound weights. | Follow review, drop `clear_model_no_alloc()` once merged. |
| [`ggml-org/llama.cpp#32076`](https://github.com/ggml-org/llama.cpp/pull/32076) | PR | MoE / Sync | `llama_moe_cache: run everything without syncs` (`GGML_OP_CUSTOM` GPU↔host signalling). Directly overlaps with out-of-core MoE streaming and Vulkan/Windows sync cost. | Benchmark on `orin-nano` (CUDA) and `nixos-nvme` (Intel Vulkan); share numbers. |
| [`ggml-org/llama.cpp#30685`](https://github.com/ggml-org/llama.cpp/pull/30685) | PR | `ggml` / Async | `ggml: allow input async copies` (`set_tensor_async` + single sync). Foundation for overlapping H2D transfers. | Watch & test async H2D overlap. |
| [`ggml-org/llama.cpp#32516`](https://github.com/ggml-org/llama.cpp/pull/32516) | PR | Vulkan / Intel | Fixes i915/Mesa cancelling Vulkan submissions after 20s (`CONFIG_DRM_I915_REQUEST_TIMEOUT`). Affects long prefills on Intel iGPUs (`nixos-nvme`). | Test on Intel RPL-U Vulkan and report results if helpful. |
| [`ggml-org/llama.cpp#28044`](https://github.com/ggml-org/llama.cpp/pull/28044) | PR | CUDA / OOM | Returns `GGML_STATUS_ALLOC_FAILED` on CUDA pool allocation failure instead of aborting the process. | Watch; improves worker resilience under tight VRAM budgets. |
| [`ggml-org/llama.cpp#29768`](https://github.com/ggml-org/llama.cpp/pull/29768) | PR | CUDA Graphs | Avoids repeated warmup after stable graph replay. | Watch for 100%-resident CUDA throughput. |
| [`ggml-org/llama.cpp#27330`](https://github.com/ggml-org/llama.cpp/issues/27330) | Issue | CUDA Graphs | CUDA graphs hang on `sm_120` (RTX 5090 Laptop); keeps `GGML_CUDA_GRAPHS=OFF` in `build.rs` until fixed. | Re-evaluate `GGML_CUDA_GRAPHS` once resolved. |
| [`ggml-org/llama.cpp#31970`](https://github.com/ggml-org/llama.cpp/pull/31970) | PR | `ggml` / Sched | Syncs splits when no inputs are copied (defensive hybrid CPU/GPU split correctness). | Watch. |
| [`ggml-org/llama.cpp#32822`](https://github.com/ggml-org/llama.cpp/pull/32822) | PR | GGUF | Rejects GGUF tensor data size exceeding file size (fixes `#32821`); mirrors our `gguf_index.rs` bounds check. | Watch. |
| [`ggml-org/llama.cpp#30888`](https://github.com/ggml-org/llama.cpp/issues/30888) | Issue | Vulkan / Windows | `DEVICE_LOCAL\|HOST_VISIBLE` exhausting WDDM shared budget (`ErrorOutOfDeviceMemory`). | Watch for Windows Vulkan users. |
| [`ggml-org/llama.cpp#31052`](https://github.com/ggml-org/llama.cpp/issues/31052) | Issue | Vulkan / Intel | Intel Iris Xe Windows Vulkan `ErrorOutOfDeviceMemory` despite available memory. | Watch for Windows Intel Vulkan users. |

## Maintenance Workflow

- `.github/workflows/maintain.yml` opens weekly fast-forward PRs tracking `ggml-org/llama.cpp` `master`.
- When bumping the pin, re-check this list and remove any workarounds in `llama_compat.hpp` whose upstream PR has landed.
