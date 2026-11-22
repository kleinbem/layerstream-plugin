# layerstream (Python bindings)

Native Python bindings (`PyO3` + `maturin`) for [`layerstream-core`](../../crates/layerstream-core): an out-of-core GGUF layer-streaming inference engine backed by unpatched `llama.cpp`, `O_DIRECT` / `F_NOCACHE` async ping-pong GPU streaming, auto-sized hybrid residency, sparse MoE TinyLFU expert caching, and speculative decoding.

## Quickstart

```python
from layerstream import AutoModel

with AutoModel.from_pretrained("Qwen/Qwen2.5-14B-Instruct") as model:
    print(model.generate("Explain NVMe O_DIRECT streaming.", max_tokens=128))
```

### Streaming & Multi-Turn Chat

```python
from layerstream import AutoModel

with AutoModel.from_pretrained(
    "hf://Qwen/Qwen2.5-14B-Instruct-GGUF:qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
    draft_model="hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf",
    vram_budget_mb="auto",
    kv_cache_type="q8_0",
) as model:
    for chunk in model.chat(
        [{"role": "user", "content": "Compare O_DIRECT vs mmap for 14B models."}],
        stream=True,
    ):
        print(chunk, end="", flush=True)
    print("\n", model.stats())
```

### Supported `from_pretrained(model, ...)` Inputs

1. **Local GGUF path:** `"/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf"`
2. **Explicit Hugging Face URI:** `"hf://org/repo:file.gguf"` (auto-expands multi-part shards and verifies ETag/SHA-256)
3. **HF GGUF repo ID:** `"Qwen/Qwen2.5-14B-Instruct-GGUF"` (auto-selects the best quant for probed GPU/NVMe, or pass `quant="Q4_K_M"`)
4. **Base HF repo ID:** `"Qwen/Qwen2.5-14B-Instruct"` (resolves the official `{org}/{name}-GGUF` sibling repository)

## Building & Trying the Demos Locally

```bash
nix develop .#python
cd bindings/python
maturin build --locked --release --features vulkan --out dist   # or omit --features for CPU
python3 -m zipfile -e dist/*.whl .wheel-local

# 3-line Hugging Face quickstart (or pass --model /path/to/model.gguf):
PYTHONPATH=.wheel-local python3 examples/quickstart.py

# Multi-turn KV reuse & MoE/streaming telemetry demo:
PYTHONPATH=.wheel-local python3 examples/streaming_benchmark.py /path/to/model.gguf --vram-budget-mb 2048

# Unit & integration tests:
PYTHONPATH=.wheel-local pytest -v
```

See the root [README.md](../../README.md) for bare-metal benchmarks (vs AirLLM, HF `accelerate`, and stock `llama.cpp`) and Mesh-LLM plugin usage.
