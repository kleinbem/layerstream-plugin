# layerstream-plugin

Out-of-core GGUF layer-streaming inference engine and [Mesh-LLM](https://github.com/Mesh-LLM/mesh-llm) v3 plugin.

Run oversized dense and sparse MoE GGUF models (**14B–32B+** on 8 GB edge GPUs, or **100B–235B+** on 64 GB unified-memory mini-PCs) using **stock, unpatched `llama.cpp`**, page-aligned `O_DIRECT` async ping-pong GPU streaming, auto-sized hybrid partial-residency (`blk.0..R-1` pinned in VRAM), sparse MoE active-expert streaming with a scan-resistant TinyLFU hot-expert RAM cache, speculative verification (greedy and `temperature > 0`), multi-turn KV prefix reuse, and isolated on-demand GPU workers for true **0 MB VRAM/UMA & 0 MB GPU driver context while idle**.

## Benchmark Highlights (`orin-nano`, 7.3 GiB RAM, CUDA `sm_87`, LUKS2 NVMe)

Measured bare-metal on an 8 GB NVIDIA Jetson Orin Nano with **INA3221 hardware power telemetry** (idle board `~7.7–8.1 W`; both gross and net `J/tok` reported), `pread_mib_s` kernel I/O telemetry, and live `swap_delta_mib`. Full **130-run dataset** and methodology: [docs/DETAILS.md](docs/DETAILS.md) and [benchmarks/hf-dataset/README.md](benchmarks/hf-dataset/README.md).

### 1. Oversized Dense Models (`Qwen2.5-14B` 8.37 GiB & `Qwen2.5-32B` 18.49 GiB = 2.53× RAM)

| Model & Engine | Configuration | Load | TTFT | `prompt_heavy` | `open_ended` | `code_edit` | Peak RSS | Swap $\Delta$ | Energy (`code_edit`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **14B** — Stock `llama.cpp` (`0693dbfb`)* | `-ngl 12`/`18`, `-b 512` | — | OOM | OOM (`rc=1`) | OOM (`rc=1`) | OOM (`rc=1`) | 5,978 MiB | +3,711 MiB | — |
| **14B** — Stock `llama.cpp` (`0693dbfb`)* | CPU `mmap` (`-ngl 0 -dev none`) | — | >300 s | — | Timeout (300 s) | — | 6,110 MiB | +363 MiB | — |
| **14B** — HF `accelerate` (`4-bit NF4`)† | `device_map="auto"` + disk offload | 144.0 s | >156 s | — | Timeout (300 s) | — | — | — | — |
| **14B** — AirLLM 4.0.0 (`4-bit`)‡ | Warm shards (cold: `365.5 s`) | 2.55 s | 32.9–33.4 s | 0.032 tok/s *(cold)* | 0.033 tok/s *(8 tok)* | 0.033 tok/s *(8 tok)* | 3,536 MiB | — | 288.2 J/tok gross |
| **14B** — `layerstream-plugin` (`v0.1.0`) | `R=0` pure stream (`reps=2`) | 2.99 s | 5.97 s | — | **0.174 tok/s** | — | **2,458 MiB** | **0 MiB** | 59.8 J/tok *(15.3 net)* |
| **14B** — `layerstream-plugin` (`v0.1.0`) | `--vram-budget-mb auto` + `0.5B` (`R=4`, `reps=3`) | 4.94 s | 2.95–4.47 s | **0.760 tok/s** | **1.578 tok/s** | **3.158 tok/s** | 3,716 MiB | **0 MiB** | 4.87 J/tok *(1.93 net)* |
| **14B** — `layerstream-plugin` (`v0.1.0`) | `--vram-budget-mb 5400` + `0.5B` (`R=7`, `reps=3`) | 4.58 s | **2.74–4.24 s** | **0.824 tok/s** | **1.699 tok/s** | **3.405 tok/s** | 4,178 MiB | **0 MiB** | **4.61 J/tok** *(1.91 net)* |
| **14B** — `layerstream-plugin` (R6 max `R`) | `R=20` + `0.5B` (6.1 GiB RSS, clean RAM) | 6.75 s | 2.12–4.08 s | **1.033 tok/s** | **2.109 tok/s** | **5.974 tok/s** | 6,153 MiB | — | **2.73 J/tok** gross |
| **32B** — Stock `llama.cpp` (`0693dbfb`)* | `-ngl 10` | — | >300 s | — | Timeout (300 s) | — | 5,715 MiB | **+17,243 MiB** | — |
| **32B** — `layerstream-plugin` (`v0.1.0`) | `--vram-budget-mb 5400` + `0.5B` (`R=2`, `reps=2`) | 3.76 s | 14.7–18.0 s | **0.171 tok/s** | **0.304 tok/s** | **0.840 tok/s** | 4,115 MiB | **0 MiB** | 15.5 J/tok *(3.82 net)* |
| **32B** — `layerstream-plugin` (R6, `R=6`) | `--vram-budget-mb 5600` + `0.5B` (clean RAM) | 6.84 s | 7.92–11.69 s | **0.336 tok/s** | **0.506 tok/s** | **0.944 tok/s** | 5,294 MiB | — | **14.3 J/tok** gross |

### 2. Sparse MoE Models (`OLMoE-1B-7B` & `Qwen3-30B-A3B`)

| Model & Engine | Configuration | Load | TTFT | Decode (`open_ended`) | Decode (`code_edit`) | Cache Hit | Peak RSS | Swap $\Delta$ | Energy (`open_ended`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **OLMoE-1B-7B** (`Q2_K`, 64 experts) | `layerstream` `R=0` full-layer (`reps=2`) | 0.51 s | 1.45 s | 0.763 tok/s | — | — | 1,312 MiB | 0 MiB | 14.34 J/tok *(4.01 net)* |
| **OLMoE-1B-7B** (`Q2_K`, 64 experts) | `layerstream` `R=0` **sparse + TinyLFU** (`v0.2.1`, `reps=3`) | 0.80 s | 2.07 s | **12.54 tok/s** | **12.84 tok/s** (`13.62` `prompt_heavy`) | **98–100%** | 3,097 MiB | **0 MiB** | **1.44 J/tok** *(0.49 net)* |
| **Qwen3-30B-A3B** (`Q4_K_M`, 17.3 GiB) | Stock `llama.cpp` GPU MoE (`-ot exps=CPU`) | — | OOM | OOM (`rc=1`) | — | — | 6,014 MiB | +12,342 MiB | — |
| **Qwen3-30B-A3B** (`Q4_K_M`, 17.3 GiB) | Stock `llama.cpp` **CPU `mmap`** (`-ngl 0 -dev none -b 512`) | 25.34 s | 17.37 s | 1.270 tok/s | — | OS page cache | 6,071 MiB | +248 MiB | 15.26 J/tok gross |
| **Qwen3-30B-A3B** (`Q4_K_M`, 17.3 GiB) | `layerstream` `R=0` sparse + TinyLFU + `0.6B` (`v0.1.0`, `reps=2`) | 3.26 s | 11.24 s | 0.317 tok/s | 0.113 tok/s *(NVMe stall)* | 16–23% | 3,969 MiB | 0 MiB | 32.98 J/tok *(3.21 net)* |
| **Qwen3-30B-A3B** (`Q4_K_M`, 17.3 GiB) | `layerstream` `R=0` **sparse + TinyLFU + `0.6B` (`v0.2.1`, `reps=2`)** | **3.59 s** | **12.13 s** | **1.446 tok/s** *(wins!)* | **0.948 tok/s** *(8.4× vs v0.1.0)* | 6–9% *(cold)* | **4,045 MiB** | **+4.4 MiB** | **11.55 J/tok** *(3.29 net)* |

- **Sustained I/O & Multi-Turn Reuse:** Measured **`2,799–2,961 MiB/s`** `pread_mib_s` on `14B` through LUKS2 `dm-crypt` (`0 MiB` swap), **`1,266 MiB/s`** on scattered `800 KiB` `Qwen3-30B-A3B` MoE slices (`0` NVMe stalls, `-7.6 MiB` page cache delta), and **`4.97×` faster Turn 2 TTFT** (`14.88 s → 2.99 s`) on a 1,377-token multi-turn prompt via KV prefix reuse.
- **Honest caveats:** \*Stock `llama.cpp` also represents the underlying GGUF runtime used by Ollama and LM Studio. †HF `accelerate` and ‡AirLLM ran in the same `torch 2.14.1+cu126` venv (which warns Jetson `CC 8.7` is unsupported); AirLLM warm runs generated 8 tokens (`bnb 4-bit`) vs. 48 tokens (`GGUF Q4_K_M`) for `layerstream-plugin`. On `Qwen3-30B-A3B` (`128` experts), `v0.1.0` lost to stock `llama.cpp` CPU `mmap` because small random `O_DIRECT` slices triggered 30 s Tegra `nvme1` IRQ stalls (`0.317` / `0.113 tok/s`); `v0.2.1` eliminated those stalls via parallel `POSIX_FADV_RANDOM` + `POSIX_FADV_DONTNEED` sparse reads and a 50% full-layer threshold, reaching **`1.446 tok/s`** (`4,045 MiB` RSS vs `6,071 MiB`, `11.55` vs `15.26 J/tok` — measured even while Frigate NVR was active on `orin-nano`).

## Why `layerstream-plugin`?

* **True 0 MB GPU & driver memory while idle (isolated on-demand worker):** Startup parses only GGUF metadata headers (~20–50 ms, `0 MB` GPU memory) so `/v1/models` advertises all configured models immediately. GPU inference runs inside a lazily spawned isolated worker subprocess (`layerstream _worker`, opt-out via `--in-process`). When `--idle-timeout-secs` elapses (or `POST /v1/unload` is called), the worker exits and the OS kernel reclaims **100%** of VRAM/UMA, CUDA/Vulkan/Metal driver contexts (`~300–524 MiB` on Jetson CUDA), and host heap arenas. With `--idle-timeout-secs 0`, the worker is synchronously reaped *before* the HTTP response / SSE stream finishes — preventing race conditions with immediate native model reloads on tight unified-memory GPUs (tracking: `Mesh-LLM/mesh-llm#3174`).
* **Zero `llama.cpp` patches:** Loads models with `llama_model_params.no_alloc = true`, binds layers to real backend `ggml_backend_buffer_t` slots (`Slot A` / `Slot B`), and synchronizes via a race-free tensor-pointer `cb_eval` hook. Inherits all 70+ `llama.cpp` architectures, Flash Attention, quantized KV cache (`Q8_0` default), and chat templates automatically.
* **Single-owner async `O_DIRECT` I/O:**
  * **Discrete & Jetson CUDA:** Full-layer 4K-aligned `O_DIRECT` `pread` straight into a single shared `cudaHostAlloc` pinned staging buffer → per-tensor `cudaMemcpyAsync` on a dedicated non-blocking copy stream overlapped with GPU compute. *(On Jetson Orin CUDA, upstream `llama.cpp` currently disables Tegra `CUDA_Host` unified memory — `upstream: ggml-org/llama.cpp#15034` — so `layerstream-plugin` uses this discrete pinned-staging path while still sizing headroom for unified physical RAM.)*
  * **Host-mapped UMA GPUs (Apple Silicon Metal / Mesa Vulkan iGPUs):** When the backend reports host-accessible device buffers (`uma_zero_copy=true`), `O_DIRECT`/`F_NOCACHE` reads directly into GPU-visible slot memory.
  * **Auto-sized hybrid partial-residency (`--vram-budget-mb auto`):** Probes live GPU/UMA memory, reserves OS headroom + KV cache (`Q8_0`) + ping-pong slots + draft model footprint, and pins early layers `blk.0..R-1` without triggering swap or OOM.
* **Sparse MoE streaming + TinyLFU hot-expert cache:** Inspects MoE router indices per layer, caches hot experts in a scan-resistant TinyLFU-LRU RAM cache (**15.8×–17.9× speedup** on `OLMoE-1B-7B`), reads scattered expert misses in parallel on Linux via `POSIX_FADV_RANDOM` + immediate `POSIX_FADV_DONTNEED` (`1,266 MiB/s`, `~0 MiB` page-cache growth, zero Tegra NVMe IRQ stalls), and switches to a single sequential full-layer `O_DIRECT` read when >50% of experts are active (e.g., during prompt prefill).
* **Speculative verification (greedy & sampled):** Zero-VRAM n-gram speculation is built in, and `--draft-model` pairs a tiny resident draft GGUF (`0.5B`–`0.6B` on 8 GB GPUs) with adaptive draft depth (`6..20`) and Leviathan rejection sampling for `temperature > 0`.
* **Fail-closed preflight advisor (`check`):** Run `layerstream-plugin check` to validate GPU availability, `O_DIRECT`, NVMe bandwidth, draft tokenizer compatibility, and expected throughput before serving.

## Usage with Mesh-LLM

Install from GitHub releases (or point `command` to your built binary):

```bash
mesh-llm plugins install kleinbem/layerstream-plugin
```

### Recommended: Host-managed `[[models]]` (`model_assets.v1`, shared Hugging Face cache)

On Mesh-LLM `>= 0.80.0`, configure `[[models]]` with `provider = "layerstream"` so `layerstream-plugin` shares `mesh-llm`'s Hugging Face cache, credentials, `HF_HUB_OFFLINE` policy, and `[models.source]` overrides (`repo`, `revision`, or `local_dir`) automatically — no duplicate GGUF download directory required:

```toml
[[plugin]]
name = "layerstream"
args = [
  "--vram-budget-mb", "auto",
  "--idle-timeout-secs", "60",
]

[[models]]
model = "unsloth/Qwen3-30B-A3B-GGUF"
provider = "layerstream"
variant = "Qwen3-30B-A3B-Q4_K_M.gguf"

# Optional overrides:
# [models.options]
# file = "Q4_K_M/model-00001-of-00003.gguf" # auto-expands multi-part shards
# alias = "qwen3-30b"                       # override advertised /v1/models ID
# draft_file = "Qwen3-0.6B-Q4_K_M.gguf"     # draft GGUF from the same repo/local_dir
# draft_model = "/models/qwen3-0.6b.gguf"   # or external local path / hf:// URI
#
# [models.source]
# repo = "unsloth/Qwen3-30B-A3B-GGUF"       # optional if model is a short alias
# revision = "main"                         # or local_dir = "/models/qwen3"
#
# [models.lifecycle]
# load = "on_demand"
# idle_timeout_secs = 60
```

### Direct CLI Arguments (older Mesh-LLM hosts or explicit local paths)

You can also pass local paths or `hf://` URIs directly via plugin `args` (both modes can be combined):

```toml
[[plugin]]
name = "layerstream"
args = [
  "--model", "/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
  "--draft-model", "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
  "--vram-budget-mb", "auto",
  "--idle-timeout-secs", "60",
]
```

*(Note: when passing `hf://` URIs via `--model`, pre-download with `layerstream-plugin pull` or let background pull finish before sending the first request.)*

## Standalone CLI Mode (No Mesh-LLM Required)

```bash
# 1. Discover & pick the best GGUF quant on Hugging Face for your GPU + NVMe:
layerstream-plugin recommend Qwen/Qwen2.5-14B-Instruct-GGUF

# 2. Pull the recommended GGUF from Hugging Face (ETag + SHA-256 verified resume):
layerstream-plugin pull hf://Qwen/Qwen2.5-14B-Instruct-GGUF:qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf

# 3. Preflight check (GPU, O_DIRECT, NVMe speed, draft tokenizer, estimated tok/s):
layerstream-plugin check \
  --model /models/Qwen2.5-14B-Instruct-Q4_K_M.gguf \
  --draft-model /models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --vram-budget-mb auto

# 4. Serve an OpenAI-compatible HTTP API:
layerstream-plugin serve \
  --model /models/Qwen2.5-14B-Instruct-Q4_K_M.gguf \
  --draft-model /models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --vram-budget-mb auto \
  --idle-timeout-secs 60 \
  --port 8080
```

Endpoints:
* `GET /health`
* `GET /v1/models`
* `POST /v1/unload` (immediately terminates and reaps any idle/warm GPU worker, returning 100% of VRAM/UMA and driver contexts to the OS)
* `POST /v1/chat/completions` (SSE streaming, stateful `<think>` → `reasoning_content` and `<tool_call>` → `tool_calls`, stop-sequence holdback, client-disconnect cancellation, `503` backpressure when queue is full)
* `POST /v1/completions`

## Python API (AirLLM-style, Native Rust/C++)

Standalone native Python bindings (`PyO3` + `maturin`) live in [bindings/python/](bindings/python/README.md) (excluded from the default Cargo workspace so `layerstream-plugin` stays Python-free):

```python
from layerstream import AutoModel

with AutoModel.from_pretrained("Qwen/Qwen2.5-14B-Instruct") as model:
    for chunk in model.generate("Explain NVMe O_DIRECT streaming.", stream=True):
        print(chunk, end="", flush=True)
    print("\n", model.stats())
```

Accepts local `.gguf` paths, `hf://org/repo:file.gguf` URIs, HF `-GGUF` repos (hardware-aware auto-quant selection), and official base HF repos (`{org}/{name}-GGUF`), with full GIL release (`py.detach`), `isolated_worker=True` for **true 0 MB VRAM / CUDA context handoff back to PyTorch** on `.unload()`, drop-in in-process `layerstream.OpenAI` / `AsyncOpenAI` clients, native GBNF / JSON Schema / Pydantic structured decoding (`response_format` / `json_schema`), `<think>` & `<tool_call>` parsing, `recommend()` / `check()` / `inspect_gguf()`, and `.tokenize()` / `.kv_trim()`. Build locally via `nix develop .#python` (see [bindings/python/README.md](bindings/python/README.md)).

## Build & Test

```bash
nix develop
cargo test --locked --workspace

# Run full real-model parity tests (R=0, R=2 hybrid, n-gram, multi-turn KV reuse, temp=0.7 speculation, synthetic mixed Dense+MoE, HTTP + 0-weight eviction):
LAYERSTREAM_TEST_GGUF=/path/to/model.gguf cargo test --locked --workspace -- --nocapture

# With Vulkan GPU:
LAYERSTREAM_TEST_GGUF=/path/to/model.gguf LAYERSTREAM_TEST_GPU=1 \
  cargo test --locked -p layerstream-core --features vulkan --test parity_test -- --nocapture

# With CUDA (x86_64 or Jetson Orin Nano aarch64 — `dynamic-backends` keeps idle coordinator RSS ~10–15 MiB with zero CUDA DT_NEEDED):
nix develop .#cuda
cargo build --release -p layerstream-plugin --features cuda,dynamic-backends
```

## Documentation & Reproducibility

* [docs/DETAILS.md](docs/DETAILS.md) — Zero-patch architecture, full `orin-nano` 4-engine benchmark tables, tuning ablations, honest limitations, Jetson platform findings, and future opportunities (including `mesh-llm` and Next-Concept Modeling).
* [Interactive Benchmark Explorer & Hardware Estimator (HF Space)](https://huggingface.co/spaces/kleinbem/layerstream-benchmarks) ([source](benchmarks/hf-space/)) — 4-engine charts, Round 2 → Round 8 tuning story, browser hardware estimator, and searchable 130-run table.
* [Hugging Face Benchmark Dataset](https://huggingface.co/datasets/kleinbem/layerstream-orin-nano-benchmarks) ([HF Collection](https://huggingface.co/collections/kleinbem/layerstream-edge-out-of-core-llm-benchmarks-6b00a3f994c2e1c552883a46) · [local bundle](benchmarks/hf-dataset/)) — Published **130-run** normalized dataset (`train-00000-of-00001.parquet`, `benchmarks.json`, `benchmarks.jsonl`, `scripts/`, and Dataset Card).
