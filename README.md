# layerstream-plugin

Out-of-core GGUF layer-streaming inference engine and [Mesh-LLM](https://github.com/Mesh-LLM/mesh-llm) plugin.

Run oversized dense GGUF models on low-VRAM GPUs (4–8 GB VRAM, including NVIDIA Jetson Orin Nano and consumer RTX / Vulkan GPUs) using **stock, unpatched `llama.cpp`**, async ping-pong GPU buffers, coalesced 4K-aligned `O_DIRECT` NVMe reads, hybrid partial-residency (`blk.0..R-1` pinned in VRAM), zero-VRAM n-gram + optional `--draft-model` speculative verification, and lazy **0 MB idle VRAM** eviction.

## Why `layerstream-plugin`?

* **0 MB VRAM while idle:** Startup parses only GGUF metadata headers (~20–50 ms) so `/v1/models` advertises all configured models without touching GPU memory. Buffers are freed back to 0 MB after `--idle-timeout-secs` (`0` = immediate eviction after each request).
* **Zero `llama.cpp` patches:** Loads models with `llama_model_params.no_alloc = true`, binds layers to two **real** backend `ggml_backend_buffer_t` slots (`Slot A` / `Slot B`), and uses a race-free tensor-pointer `cb_eval` hook. Inherits all 70+ `llama.cpp` architectures, Flash Attention, KV cache, and chat templates automatically.
* **Hardware-aware I/O:**
  * **Discrete GPUs (CUDA / Vulkan):** Coalesced 4K-aligned `O_DIRECT` `pread` $\rightarrow$ pinned host staging buffers $\rightarrow$ dedicated non-blocking copy stream overlapped with GPU compute.
  * **Integrated / UMA GPUs (Jetson Orin Nano, Intel/AMD iGPU):** Zero-copy mapped slots — `O_DIRECT` reads directly into GPU-visible memory with half the RAM footprint.
  * **Hybrid partial-residency:** Pass `--vram-budget-mb` (or `--resident-layers N`) to keep early layers permanently resident in VRAM and stream only the overflow layers off NVMe.
* **Speculative verification:** Zero-VRAM n-gram speculation is enabled by default (and `--draft-model` supports a tiny resident draft GGUF) to verify multiple tokens per single NVMe sweep.

## Usage with Mesh-LLM

Install from GitHub releases (or point `command` to your built binary):

```bash
mesh-llm plugins install kleinbem/layerstream-plugin
```

Add to `~/.mesh-llm/config.toml` (pair with `--max-vram` headroom or `mode = "on_demand"` so Mesh's resident Skippy runtime and `layerstream` coexist cleanly):

```toml
[[plugin]]
name = "layerstream"
args = [
  "--model", "/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
  "--vram-budget-mb", "2048",
  "--idle-timeout-secs", "60",
]
```

Then run `mesh-llm serve` normally — the model appears on `http://127.0.0.1:9337/v1/models` with 0 MB VRAM idle and routes across your entire mesh.

## Standalone CLI Mode (No Mesh-LLM Required)

```bash
layerstream-plugin serve \
  --model /models/Llama-3.3-70B-Instruct-Q4_K_M.gguf \
  --draft-model /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --vram-budget-mb 4096 \
  --port 8080
```

Endpoints:
* `GET /health`
* `GET /v1/models`
* `POST /v1/chat/completions` (SSE streaming, `<think>` $\rightarrow$ `reasoning_content`, `<tool_call>` $\rightarrow$ `tool_calls`, client-disconnect cancellation)
* `POST /v1/completions`

## Build & Test

```bash
nix develop
cargo test --workspace

# Run full real-model parity tests (R=0, R=2 hybrid, n-gram speculation, HTTP chat + 0-VRAM eviction):
LAYERSTREAM_TEST_GGUF=/path/to/model.gguf cargo test --workspace -- --nocapture

# With Vulkan GPU:
LAYERSTREAM_TEST_GGUF=/path/to/model.gguf LAYERSTREAM_TEST_GPU=1 \
  cargo test -p layerstream-core --features vulkan --test parity_test -- --nocapture

# With CUDA (x86_64 or Jetson Orin Nano aarch64):
cargo build --release -p layerstream-plugin --features cuda
```
