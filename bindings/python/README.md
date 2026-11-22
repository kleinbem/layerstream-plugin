# layerstream (Python SDK)

Native Python SDK (`PyO3` + `maturin`) for [`layerstream-core`](../../crates/layerstream-core): an out-of-core GGUF layer-streaming inference engine backed by unpatched `llama.cpp`, `O_DIRECT` / `F_NOCACHE` async ping-pong GPU streaming, auto-sized hybrid residency, sparse MoE TinyLFU expert caching, and speculative decoding.

## Quickstart

```python
from layerstream import AutoModel

with AutoModel.from_pretrained("Qwen/Qwen2.5-14B-Instruct") as model:
    print(model.generate("Explain NVMe O_DIRECT streaming.", max_tokens=128))
```

### True 0 MB VRAM Handoff (`isolated_worker=True`)

When mixing `layerstream` with PyTorch / Whisper / Diffusers on 8–16 GB GPUs, pass `isolated_worker=True` so GPU inference runs in an isolated child worker process. Calling `.unload()` (or exiting the `with` block) terminates and reaps the worker so **100% of VRAM and CUDA/Vulkan/Metal driver contexts (~300–524 MiB)** return to `torch.cuda`:

```python
from layerstream import AutoModel

with AutoModel.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    draft_model="Qwen/Qwen2.5-0.5B-Instruct",
    isolated_worker=True,
) as model:
    for chunk in model.chat(
        [{"role": "user", "content": "Compare O_DIRECT vs mmap for 14B models."}],
        stream=True,
    ):
        print(chunk, end="", flush=True)
# <- Worker reaped; 0 B GPU / CUDA driver context remains in Python!
```

### Drop-In In-Process `OpenAI` / `AsyncOpenAI` & Structured Outputs

Use `layerstream.OpenAI` or `layerstream.AsyncOpenAI` in-process (no daemon or port required) with native GBNF / JSON Schema / Pydantic constrained decoding, `<think>` → `reasoning_content`, and `<tool_call>` → `tool_calls`:

```python
from pydantic import BaseModel
from layerstream import OpenAI

class HardwareVerdict(BaseModel):
    quant: str
    fits_edge_gpu: bool

with OpenAI("Qwen/Qwen2.5-14B-Instruct", isolated_worker=True) as client:
    resp = client.chat.completions.parse(
        messages=[{"role": "user", "content": "Recommend Q4_K_M for an 8 GB Orin Nano."}],
        response_format=HardwareVerdict,
    )
    verdict: HardwareVerdict = resp.choices[0].message.parsed
    print(verdict)
```

### Hardware Advisor, Preflight & Low-Level Controls

```python
import layerstream

# Rank GGUF quantizations in any HF repo (or base repo with -GGUF sibling) against GPU + NVMe:
recs = layerstream.recommend("Qwen/Qwen3-30B-A3B-GGUF")

# Header-only (~20 ms, 0 MB VRAM) local GGUF metadata inspection:
info = layerstream.inspect_gguf("/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf")

# Preflight check (resolves/downloads uncached HF specs, probes GPU VRAM, and benchmarks storage I/O):
report = layerstream.check("hf://Qwen/Qwen2.5-14B-Instruct-GGUF:qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf")
```

`AutoModel` also exposes `.tokenize(text)`, `.detokenize(ids)`, `.apply_chat_template(messages)`, `.kv_clear()`, `.kv_trim(pos_min)`, `grammar=...`, `json_schema=...`, `tools=[...]`, `parse_output=True`, and `async for` on `TokenStream`.

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
