# `layerstream-plugin` — Architecture, Benchmarks & Engineering Notes

This document records the engineering design of `layerstream-plugin`, complete hardware benchmark data collected on an **8 GB NVIDIA Jetson Orin Nano** (comparing **`layerstream-plugin`**, **stock `llama.cpp`** — representing the underlying GGUF runtime of Ollama and LM Studio — **Hugging Face `accelerate`**, and **AirLLM 4.0.0**), honest limitations, and **opportunities outside the plugin** (OS/storage, `llama.cpp`, `mesh-llm`, and future model paradigms like Next-Concept Modeling).

---

## 1. Zero-Patch `llama.cpp` Architecture

Both `ggml-cuda` and `ggml-vulkan` validate backend buffer ownership by checking function-pointer identity inside `supports_buft()`. Custom `ggml_backend_buffer_type` wrappers are therefore rejected by stock `llama.cpp` GPU schedulers.

`layerstream-plugin` solves this with **zero patches to upstream `llama.cpp`**:

1. **Metadata-Only Discovery (0 MB GPU memory):** `crates/layerstream-core/src/gguf_index.rs` parses GGUF v2/v3 headers (single-file or multi-part `*-00001-of-0000N.gguf`) in pure Rust (~20–50 ms, `0 MB` GPU memory) so `/v1/models` advertises all configured models immediately.
2. **Deferred Allocation (`no_alloc = true` + `LLAMA_LOAD_MODE_NONE`):** On first request, `stream_engine.cpp` loads only the model graph metadata via `llama_model_load_from_file`.
3. **Real Backend Buffers (`Resident` + `Slot A` + `Slot B`):**
   - Non-layer tensors (`token_embd.weight`, `output.weight`, norms) plus early resident layers `blk.0..R-1` are packed into one real `ggml_backend_buffer_t` (`resident_buf`) loaded in bounded host chunks (`<= max_single_tensor`) to prevent 2× RAM spikes during load.
   - Streamed layers `blk.R..L-1` are aliased into two real ping-pong backend buffers (`slot_buf[0]` for even, `slot_buf[1]` for odd) sized to the single largest streamed layer.
   - `model->hparams.no_alloc` is cleared immediately before `llama_init_from_model()` so `llama_context` allocates real Flash Attention KV cache (`Q8_0` default, or `Q4_0`/`F16`) and compute scheduler buffers (upstream: `ggml-org/llama.cpp#31141`, already merged upstream; remove once pinned `llama.cpp` is bumped past `#31141`).
4. **Race-Free Tensor-Pointer `cb_eval` Hook & Single-Owner I/O Worker:**
   - Every streamed `ggml_tensor*` is mapped O(1) to its layer index (`tensor_to_layer`).
   - A single I/O worker thread exclusively owns `pin_buf`, the dedicated `cudaStreamNonBlocking` copy stream, and `LayerIoPool` scratch buffers, handling both background dense prefetches and priority MoE expert slice jobs cleanly without data races. (On Vulkan, `ggml_backend_tensor_set` runs on the eval thread inside `cb_eval(ask=false)` while the worker handles `pread` only.)
   - Waiting on the incoming layer's slot & event inside `cb_eval(t, true)` and enqueueing `layer + 1` (or wraparound layer `R` after the final layer) inside `cb_eval(t, false)` is architecture-agnostic across all 70+ `llama.cpp` model families.
5. **Hardware-Aware `O_DIRECT` Streaming & Unified-Memory Budgeting:**
   - **Discrete GPUs & Jetson CUDA:** Full-layer 4K-aligned `O_DIRECT` `pread` straight into a single shared `cudaHostAlloc` pinned staging buffer (`slot_bytes + 2 MiB`) → per-tensor `cudaMemcpyAsync` on a dedicated non-blocking copy stream overlapped with GPU compute, zeroing quantized row-padding tails between heterogeneous layer overwrites. *(On Jetson Orin CUDA, upstream `llama.cpp` currently disables Tegra `CUDA_Host` unified memory — `upstream: ggml-org/llama.cpp#15034` — so `uma_zero_copy` is `false`; `layerstream-plugin` explicitly separates `is_unified_memory` from `is_uma_zero_copy` so UMA RAM headroom reserves `~1,068 MiB` and prevents swap/OOM.)*
   - **Host-Mapped UMA Backends (Apple Silicon Metal / Mesa Vulkan iGPUs):** When `ggml_backend_buft_is_host(target_buft)` is true (`uma_zero_copy == true`), `O_DIRECT` / `F_NOCACHE` reads directly into GPU-visible slot memory.
6. **Sparse MoE Active-Expert Streaming & TinyLFU Hot-Expert RAM Cache:**
   - For sparse MoE layers (`*_exps.weight`), `cb_eval` reads the router top-K expert indices produced by `ffn_moe_topk` / `ffn_moe_logits`, looks up active experts in a scan-resistant **TinyLFU-LRU RAM cache**, and fetches only missing experts from NVMe.
   - When >25% of a layer's experts are missing in a single pass (e.g., during multi-token prompt prefill), `stream_engine.cpp` automatically switches to a single sequential full-layer `O_DIRECT` read — exploiting the 3–4× sequential-vs-random NVMe bandwidth advantage while keeping 1-token decode sparse.
7. **Speculative Verification (Greedy & `temperature > 0`):**
   - **Greedy (`temperature == 0`):** Exact multi-token batch verification (`ls_engine_eval_verify_greedy`). Matching batch-size streaming (`R=0`, `R=2`, sparse MoE, synthetic mixed Dense+MoE) is **100% bit-identical** to stock `llama.cpp` across CPU, Intel Vulkan0, and Jetson CUDA `sm_87`. *(Honest caveat: batched `(K+1)` verification uses GEMM kernels whereas 1-token decode uses GEMV kernels, so rare FP tie-break tokens can differ between unspeculated 1-token decode and `(K+1)` speculative verification.)*
   - **Sampled (`temperature > 0`):** Exact Leviathan et al. speculative rejection sampling (`ls_engine_eval_verify_sample`) over the filtered `top_k → top_p → temp` distribution (verified against `eval_sample` via synthetic distribution tests).
   - **Proposer Cascade:** Zero-VRAM n-gram lookup (`min_n=2` standalone; raised to `min_n=4` verbatim match when paired with `--draft-model`), adaptive draft length (`6..20`), draft `n_batch` capped at `64`, padded-vocab draft compatibility (`<=256` trailing pad tokens), and automatic disablement on non-shiftable recurrent/hybrid models (`!partial_seq_rm`).
8. **Idle Memory Eviction (0 MB Model/KV Weights Idle):**
   - `Residency<LoadedPair>` guards a single global GPU lease with `--idle-timeout-secs` (`0` = immediate eviction after every request), dropping `StreamEngine` on `spawn_blocking`.
   - On Jetson Orin Nano, live `/proc/meminfo` measurements showed `MemAvailable`: `5,140 MiB` before start → `5,046 MiB` after metadata index → `2,087 MiB` while serving 14B → `4,814 MiB` after idle eviction (`~2,727 MiB` of model/KV/slot memory freed; `~230–325 MiB` remains held by CUDA's initialized primary driver context/cuBLAS).

---

## 2. Bare-Metal Jetson Orin Nano Benchmarks (4-Engine Comparison)

**Hardware:** NVIDIA Jetson Orin Nano (`aarch64`, 6× Cortex-A78AE @ `1.728 GHz` `performance`, 7.3 GiB usable unified LPDDR5, Ampere `sm_87` CUDA 12.6, Crucial T500 PCIe NVMe SSD behind **LUKS2 `dm-crypt` AES-XTS**, INA3221 hardware `VDD_IN` power rail measuring `~7.7–8.1 W` idle).

Full normalized dataset (**130 records** across 4 engines): [benchmarks.json](../benchmarks/hf-dataset/data/benchmarks.json), [benchmarks.jsonl](../benchmarks/hf-dataset/data/benchmarks.jsonl), and [train-00000-of-00001.parquet](../benchmarks/hf-dataset/data/train-00000-of-00001.parquet).

### A. `Qwen2.5-14B-Instruct` (8.37 GiB GGUF vs. HF 4-bit) — Round 8 (`v0.1.0`) & Competitors

*Round 8 runs use `--warmup --reps 3` (median with min–max range recorded in dataset), `pread_mib_s`, live `swap_delta_mib`, and gross + net `J/tok` (subtracting measured board idle watts).*

| Engine & Configuration | Workload | `R` | Load | TTFT | Decode (`tok/s`) | `pread` BW | Peak RSS | Swap $\Delta$ | Power (Idle) | Gross / Net `J/tok` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Stock `llama.cpp`** (`-ngl 12`/`18`, `-b 512`)* | all 3 | — | — | OOM | `error_rc_1` | — | 5,978 MiB | +3,711 MiB | 9.9 W (8.0 W) | — |
| **Stock `llama.cpp`** (`-ngl 0 -dev none -b 512`)* | `open_ended` | — | — | >300 s | `timeout` | — | 6,110 MiB | +363 MiB | 10.6 W (7.9 W) | — |
| **HF `accelerate`** (`4-bit NF4`, disk offload)† | `open_ended` | — | 144.0 s | >156 s | `timeout_300s` | — | — | — | — | — |
| **AirLLM 4.0.0** (`4-bit`, cold first-run)‡ | `prompt_heavy` | — | 365.53 s | 32.09 s | 0.032 *(32 tok)* | — | 5,391 MiB | — | — | — |
| **AirLLM 4.0.0** (`4-bit`, warm shards)‡ | `code_edit` | — | 2.55 s | 33.37 s | 0.033 *(8 tok)* | — | 3,536 MiB | — | 9.30 W | 288.2 / — |
| **AirLLM 4.0.0** (`4-bit`, warm shards)‡ | `open_ended` | — | 2.57 s | 32.86 s | 0.033 *(8 tok)* | — | 3,531 MiB | — | 9.33 W | 287.1 / — |
| **AirLLM 4.0.0** (`BF16`, `prefetching=True`)‡ | `open_ended` | — | 81.81 s | 45.41 s | 0.023 *(4 tok)* | — | 5,218 MiB | — | 9.31 W | 412.4 / — |
| **`layerstream` R8** (`R=0`, `reps=2`) | `open_ended` | 0/48 | 2.99 s | 5.97 s | **0.174** *(16 tok)* | 1,427 MiB/s | **2,458 MiB** | **-3.0 MiB** | 10.37 W (7.71 W) | 59.75 / 15.33 |
| **`layerstream` R8** (`auto` + `0.5B`, `reps=3`) | `prompt_heavy` | 4/48 | 4.94 s | 4.47 s | **0.760** *(0.755–0.760)* | 2,796 MiB/s | 3,709 MiB | **-0.3 MiB** | 13.21 W (7.77 W) | 18.69 / 7.69 |
| **`layerstream` R8** (`auto` + `0.5B`, `reps=3`) | `open_ended` | 4/48 | 4.94 s | 2.95 s | **1.578** *(1.564–1.582)* | 2,794 MiB/s | 3,716 MiB | **0.0 MiB** | 12.84 W (7.77 W) | 8.75 / 3.45 |
| **`layerstream` R8** (`auto` + `0.5B`, `reps=3`) | `code_edit` | 4/48 | 4.94 s | 3.26 s | **3.158** *(2.244–3.159)* | 2,800 MiB/s | 3,714 MiB | **-0.3 MiB** | 12.88 W (7.77 W) | 4.87 / 1.93 |
| **`layerstream` R8** (`b5400` + `0.5B`, `reps=3`) | `prompt_heavy` | 7/48 | 4.58 s | 4.24 s | **0.824** *(0.817–0.827)* | **2,924 MiB/s** | 4,171 MiB | **0.0 MiB** | 13.29 W (7.70 W) | 17.51 / 7.37 |
| **`layerstream` R8** (`b5400` + `0.5B`, `reps=3`) | `open_ended` | 7/48 | 4.58 s | **2.74 s** | **1.699** *(1.685–1.713)* | **2,932 MiB/s** | 4,178 MiB | **0.0 MiB** | 13.29 W (7.70 W) | 8.42 / 3.54 |
| **`layerstream` R8** (`b5400` + `0.5B`, `reps=3`) | `code_edit` | 7/48 | 4.58 s | 3.01 s | **3.405** *(3.284–3.414)* | **2,961 MiB/s** | 4,176 MiB | **0.0 MiB** | 13.16 W (7.70 W) | **4.61 / 1.91** |
| **`layerstream` R8** (`b5400` + `0.5B`, `temp=0.7`)§ | `code_edit` | 6/48 | 4.33 s | 7.12 s | 1.370 *(1.369–1.372)* | 1,026 MiB/s§ | 4,028 MiB | **0.0 MiB** | 9.92 W (7.72 W) | 8.56 / 1.90 |
| **`layerstream` R6** (`R=20` + `0.5B`, max `R`) | `code_edit` | 20/48 | 6.75 s | 2.31 s | **5.974** *(open: 2.109)* | — | 6,153 MiB | — | 12.89 W | 2.73 / — |

- \*Stock `llama.cpp` (pinned `0693dbfb`, CUDA `sm_87`) also represents the underlying GGUF runtime used by Ollama and LM Studio.
- †HF `accelerate` (`transformers 5.23.0 + bnb 0.50.2`) and ‡AirLLM 4.0.0 ran in the same `torch 2.14.1+cu126` venv (which emits a Jetson `CC 8.7` unsupported-build warning). AirLLM warm runs generated 8 tokens (`BF16`: 4 tokens) vs. 48 tokens for `layerstream-plugin`, and AirLLM requires `36.8 GiB` on disk (`29 GiB` HF cache + `7.8 GiB` sharded layers) vs. `8.37 GiB` for the single `.gguf`.
- §During the `temp=0.7` and `32B` R8 runs, `orin_crypt` (`dm-crypt` AES-XTS) dropped from `~2.9 GiB/s` to `~1.03–1.34 GiB/s` under host `zram` pressure (`avg_watts` also dropped from `13.2 W` → `9.9 W`); notice net energy per token (`1.90 J/tok` on `code_edit`) stayed identical because `pread_mib_s` scaled linearly.
- **Multi-turn KV prefix reuse (`r8_14b_auto_draft05b_multiturn`):** Turn 1 (`1,377` prompt tokens) TTFT `14.877 s` → Turn 2 (`1,414` prompt tokens) TTFT **`2.993 s`** (**`4.97×` faster**).

### B. `Qwen2.5-32B-Instruct-Q4_K_M` (18.49 GiB = 2.53× Total RAM)

| Engine & Configuration | Workload | `R` | Load | TTFT | Decode (`tok/s`) | `pread` BW | Peak RSS | Swap $\Delta$ | Gross / Net `J/tok` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Stock `llama.cpp`** (`-ngl 10`) | `open_ended` | — | — | >300 s | `timeout` | — | 5,715 MiB | **+17,243 MiB** | — |
| **`layerstream` R8** (`b5400` + `0.5B`, `reps=2`) | `prompt_heavy` | 2/64 | 3.76 s | 18.01 s | **0.171** *(0.156–0.185)* | 1,229 MiB/s | 4,109 MiB | **0.0 MiB** | 64.13 / 14.21 |
| **`layerstream` R8** (`b5400` + `0.5B`, `reps=2`) | `open_ended` | 2/64 | 3.76 s | 16.56 s | **0.304** *(0.304–0.304)* | 1,311 MiB/s | 4,115 MiB | **0.0 MiB** | 36.61 / 8.30 |
| **`layerstream` R8** (`b5400` + `0.5B`, `reps=2`) | `code_edit` | 2/64 | 3.76 s | 14.70 s | **0.840** *(0.838–0.841)* | 1,338 MiB/s | 4,113 MiB | **-0.5 MiB** | 15.51 / 3.82 |
| **`layerstream` R6** (`R=0`, pure stream) | `prompt_heavy` | 0/64 | 2.14 s | 14.48 s | **0.094** *(559.7 GiB read)* | — | **2,957 MiB** | — | 119.1 / — |
| **`layerstream` R6** (`b5600` + `0.5B`, clean RAM) | `code_edit` | 6/64 | 6.84 s | 8.42 s | **0.944** *(open: 0.506)* | — | 5,294 MiB | — | 14.3 / — |

### C. Sparse MoE Models (`OLMoE-1B-7B` & `Qwen3-30B-A3B`)

| Model & Engine | Workload | `R` | Load | TTFT | Decode (`tok/s`) | Cache Hit | `pread` BW | Peak RSS | Swap $\Delta$ | Gross / Net `J/tok` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **OLMoE-1B-7B** — `layerstream` full-layer | `open_ended` | 0/16 | 0.51 s | 1.45 s | 0.763 *(0.763–0.763)* | — | 1,943 MiB/s | 1,312 MiB | -2.0 MiB | 14.34 / 4.01 |
| **OLMoE-1B-7B** — `layerstream` **sparse + TinyLFU** | `prompt_heavy` | 0/16 | 0.68 s | 2.08 s | **13.667** *(13.65–13.92)* | **100%** | 1,405 MiB/s | 2,921 MiB | **0.0 MiB** | **1.67 / 0.57** |
| **OLMoE-1B-7B** — `layerstream` **sparse + TinyLFU** | `code_edit` | 0/16 | 0.68 s | 1.87 s | **13.054** *(12.45–13.24)* | **99.3%** | 1,269 MiB/s | 2,953 MiB | **0.0 MiB** | **1.44 / 0.51** |
| **OLMoE-1B-7B** — `layerstream` **sparse + TinyLFU** | `open_ended` | 0/16 | 0.68 s | 1.72 s | **12.047** *(11.45–12.51)* | **98.0%** | 1,168 MiB/s | 2,971 MiB | **0.0 MiB** | **1.40 / 0.45** |
| **Qwen3-30B-A3B** — Stock `llama.cpp` GPU MoE (`-ot exps=CPU`) | `open_ended` | — | — | OOM | `error_rc_1` | — | — | 6,014 MiB | +12,342 MiB | — |
| **Qwen3-30B-A3B** — Stock `llama.cpp` **CPU `mmap` `-b 512`** | `open_ended` | — | 25.34 s | 17.37 s | **1.270** *(23 tok)* | OS cache | — | 6,071 MiB | +248 MiB | **15.26** / — |
| **Qwen3-30B-A3B** — `layerstream` sparse + TinyLFU + `0.6B` | `open_ended` | 0/48 | **3.26 s** | **11.24 s** | 0.317 *(0.233–0.400)* | 23.3% | 708 MiB/s | **3,969 MiB** | **0.0 MiB** | 32.98 / 3.21 |
| **Qwen3-30B-A3B** — `layerstream` sparse + TinyLFU + `0.6B` | `code_edit` | 0/48 | 33.06 s* | 13.60 s | 0.113 *(0.066–0.159)* | 16.0% | 267 MiB/s* | **3,974 MiB** | **-2.0 MiB** | 89.98 / 5.21 |

- **Why `OLMoE` jumps `15.8×–17.9×` while `Qwen3-30B-A3B` trails stock `llama.cpp` CPU `mmap` on `orin-nano`:** `OLMoE` (`Q2_K`, 2.5 GiB total) fits its hot experts inside `layerstream`'s TinyLFU RAM cache (`98–100%` hit rate), yielding `12–13.7 tok/s` at `R=0` and `0 MiB` swap. On `Qwen3-30B-A3B` (`17.3 GiB`, 128 experts/layer), the 1 GiB TinyLFU cache captures `16–23%` of expert slices, while the remaining random small-slice `O_DIRECT` bursts trigger a Jetson Tegra PCIe/NVMe kernel quirk (`nvme nvme1: I/O tag ... QID 5/6 timeout, completion polled` — 30-second IRQ loss stalls, visible in the `33.06 s` load and `267 MiB/s` `pread` on `code_edit`). Meanwhile, stock `llama.cpp` CPU `mmap` lets the Linux page cache occupy ~6 GiB of RAM to absorb random expert accesses at `1.27 tok/s`.

---

## 3. Hardware & Upstream Findings Uncovered During Testing

1. **Jetson L4T R39 (CUDA 13 driver) vs. Mismatched `cuBLAS 12.6` Toolkit:**
   - During early testing with `cudaPackages_12_6` against `orin-nano`'s JetPack 7 / L4T R39 (CUDA 13) host driver, `cublasSgemm` (`F32`) silently returned `CUBLAS_STATUS_SUCCESS` without writing `ffn_moe_logits-0` (`[2048, 128] × [2048, N]`) on `Qwen3-30B-A3B`.
   - **Root fix in `flake.nix` (no runtime workaround needed):** Switching `flake.nix` from `pkgs.cudaPackages_12_6` to nixpkgs' default `pkgs.cudaPackages` (`CUDA 13.4` / `libcublas 13.8.2.12`, matching the JetPack 7 driver) resolved the `cublasSgemm` fault natively with standard `F32` router GEMMs, allowing us to remove our temporary `bf16` env workaround completely.
2. **Jetson `orin-nano` `nvme1` PCIe MSI-X IRQ Loss Under Random `O_DIRECT` Bursts:**
   - Sequential full-layer `O_DIRECT` reads (`156–275 MiB`) sustain `2.8–2.96 GiB/s` cleanly with zero timeouts. Heavy bursts of tiny random `O_DIRECT` slices (e.g., 128-expert `Qwen3-30B-A3B`) occasionally hit 30-second Tegra NVMe completion-polling timeouts in `dmesg`. Lowering our sequential full-layer crossover from `>=75%` to `>25%` of active experts cut `Qwen3-30B-A3B` TTFT **4.3×** (`59.2 s → 13.8 s`) and total time **6.4×** (`560 s → 88 s`).
3. **Key Negative & Trade-Off Ablations (Rounds 2–6):**
   - **Multi-threaded `O_DIRECT` (`io_threads=1` vs `2` vs `4`) regressed up to 57% on LUKS2 `dm-crypt`:** `io_threads=1` achieved `3.427 / 1.691 tok/s` vs. `2.127 / 0.719 tok/s` at `io_threads=4`. A single full-layer `pread` already fans 128 KB BIOs across all 6 `kcryptd` workers without breaking sequential NVMe locality.
   - **Draft model Pareto on 8 GB edge hardware:** `Qwen2.5-0.5B` (469 MiB) beat `Qwen2.5-1.5B` (1.04 GiB) by **2.5×–3.0×** because spending ~580 MiB on extra resident target layers beat the negligible acceptance gain.

---

## 4. Opportunities Outside `layerstream-plugin` & Future Tailwinds

### A. NixOS / Host Storage (`nix-config` on `orin-nano`)
1. **Unencrypted `/mnt/models` LV (or `dm-crypt` `--perf-no_read_workqueue` / `--perf-same_cpu_crypt`):** Public open-weight GGUF files contain no secrets; bypassing CPU AES-XTS decryption on `vg_orin-models` would lift Crucial T500 sustained reads from `~2.4–2.96 GiB/s` (and `1.03 GiB/s` under contention) toward the ~3.3–3.5 GiB/s PCIe Gen3 x4 ceiling while saving ~2–3 W.
2. **Block device `max_sectors_kb` (`128 KB → 1024 KB`):** Reduces BIO/IRQ count per 156 MiB layer read on `nvme1n1` / `dm-*`.
3. **AMD APU / Unified-Memory Hosts (e.g., Ryzen 8700G + 64 GB DDR5):** On Linux `amdgpu`, raising `ttm.pages_limit` / `ttm.page_pool_size` above the default 50% RAM cap lets Vulkan iGPUs utilize ~48–54 GB of unified DDR5 for huge resident + TinyLFU expert caches on 123B–235B+ models.

### B. Upstream `ggml-org/llama.cpp`
1. **Jetson Tegra Unified Memory (`upstream: ggml-org/llama.cpp#15034`):** When upstream resolves Tegra UMA coherency and re-enables `CUDA_Host` on integrated Jetson GPUs, `layerstream-plugin`'s `uma_zero_copy` path will activate automatically, freeing the `166–277 MiB` `pin_buf` and eliminating the H2D copy step.
2. **GPUDirect Storage (`cuFile` / KvikIO) on desktop Linux NVIDIA GPUs:** Direct NVMe-to-VRAM transfers bypassing host bounce buffers (desktop-only; unsupported on Jetson L4T today).

### C. `Mesh-LLM` Ecosystem (`mesh-llm`)
1. **VRAM handoff / auto-evict between Skippy and on-demand plugins:** Currently `mesh-llm` keeps its default Skippy model resident while routing a request to an `on_demand` plugin, forcing `--vram-budget-mb auto` to share RAM. Adding a lightweight pre-acquire eviction hook in `mesh-llm` would let `layerstream-plugin` claim the full host RAM budget on demand and reload Skippy in ~2–5 s afterward.
2. **Long-TTFT proxy timeouts & client-disconnect cancellation propagation:** Ensuring `mesh-llm`'s HTTP proxy tolerates 10–30 s TTFTs on oversized models and immediately cancels the upstream plugin request when the caller disconnects.
3. **Multi-backend Linux release packaging (`GGML_BACKEND_DL=ON`):** Shipping a single Linux archive with `libggml-cpu.so` + `libggml-vulkan.so` + `libggml-cuda.so` so `mesh-llm plugins install` auto-selects CUDA or Vulkan by target triple alone.

### D. Future Model & Plugin Roadmap
1. **Non-autoregressive coarse-to-fine / parallel decoding models — e.g., Next-Concept Modeling (NCM, `arXiv:2610.14162`) and block-diffusion LLMs *(future tailwind)*:**
   - Standard autoregressive LLMs require 1 NVMe sweep per generated token (or ~1 sweep per 2–6 tokens with speculation). Coarse-to-fine parallel architectures like NCM generate entire blocks in **~9–12 parallel level passes** (analogous to prefill batches).
   - Too early to integrate today (published NCM checkpoints top out at ≤725M and require a multiscale VQ-VAE tokenizer/loop not yet in `llama.cpp`/GGUF), but if 14B–70B+ NCM or block-diffusion checkpoints land in `llama.cpp`, `layerstream-plugin`'s zero-patch `cb_eval` layer streamer would immediately turn hundreds of NVMe sweeps into ~10 wide batched sweeps.
2. **Disk-offloaded / paged KV cache** for 32K–128K context windows on 4–8 GB GPUs.
3. **Adaptive TinyLFU expert cache sizing** scaled to live `MemAvailable` (so 64 GB hosts automatically cache 20–40 GB of hot experts for 100B–400B MoEs).
