# layerstream (Python SDK)

Native Python SDK (`PyO3` + `maturin`) for [`layerstream-core`](https://github.com/kleinbem/layerstream-plugin/tree/main/crates/layerstream-core): an out-of-core GGUF layer-streaming inference engine backed by unpatched `llama.cpp`, `O_DIRECT` / `F_NOCACHE` async ping-pong GPU streaming, auto-sized hybrid residency, sparse MoE TinyLFU expert caching, and speculative decoding.

## Installation (Python 3.14+, Zero Compile Required)

```bash
# Apple Silicon (Metal), AMD, Intel, NVIDIA (Vulkan), or CPU fallback:
pip install layerstream

# Linux x86_64 & aarch64 with NVIDIA Ampere+ GPU (CUDA 13.5+ driver >= 595; adds cuBLAS/cudart from PyPI, no nvcc needed):
pip install "layerstream[cuda]"
```

## Quickstart

```python
from layerstream import AutoModel

with AutoModel.from_pretrained("Qwen/Qwen2.5-14B-Instruct", draft_model="auto") as model:
    print(model.generate("Explain NVMe O_DIRECT streaming.", max_tokens=128, min_p=0.05))
```

## Built-In `layerstream` CLI

```bash
# Interactive terminal chat with live tok/s, VRAM residency, and NVMe telemetry:
layerstream chat Qwen/Qwen2.5-14B-Instruct --draft-model auto

# Rank Hugging Face GGUF quantizations for your GPU + NVMe:
layerstream recommend Qwen/Qwen3-30B-A3B-GGUF

# Fail-closed hardware & storage preflight check:
layerstream check Qwen/Qwen2.5-14B-Instruct

# Native Rust `axum` + `tokio` OpenAI HTTP + SSE server (/v1/chat/completions, /v1/models, /v1/unload):
layerstream serve Qwen/Qwen2.5-14B-Instruct --port 8080
```

## Instant NVMe KV-Cache Snapshots (`save_state` / `load_state`)

Prefill a large system prompt or repository context once, save the quantized (`q8_0`) KV cache to NVMe, and restore it in ~20 ms on future runs without re-streaming model weights:

```python
from layerstream import AutoModel

with AutoModel.from_pretrained("Qwen/Qwen2.5-14B-Instruct") as model:
    model.chat([{"role": "system", "content": repo_context}, {"role": "user", "content": "Ready?"}], max_tokens=8)
    model.save_state("repo_context.kv")

with AutoModel.from_pretrained("Qwen/Qwen2.5-14B-Instruct") as model:
    model.load_state("repo_context.kv")
    print(model.chat([{"role": "system", "content": repo_context}, {"role": "user", "content": "Ready?"}, {"role": "assistant", "content": "Yes."}, {"role": "user", "content": "Summarize module A."}]))
```

## True 0 MB VRAM Handoff (`isolated_worker=True`)

When mixing `layerstream` with PyTorch / Whisper / Diffusers on 8–16 GB GPUs, pass `isolated_worker=True` so GPU inference runs in an isolated child worker process. Calling `.unload()` (or exiting the `with` block) terminates and reaps the worker so **100% of VRAM and CUDA/Vulkan/Metal driver contexts (~300–524 MiB)** return to `torch.cuda`:

```python
from layerstream import AutoModel

with AutoModel.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    draft_model="auto",
    isolated_worker=True,
) as model:
    for chunk in model.chat(
        [{"role": "user", "content": "Compare O_DIRECT vs mmap for 14B models."}],
        stream=True,
    ):
        print(chunk, end="", flush=True)
# <- Worker reaped; 0 B GPU / CUDA driver context remains in Python!
```

## Drop-In In-Process `OpenAI` / `AsyncOpenAI` & Structured Outputs

Use `layerstream.OpenAI` or `layerstream.AsyncOpenAI` in-process (no daemon or port required) with native GBNF / JSON Schema / Pydantic / `dataclass` constrained decoding, `<think>` -> `reasoning_content`, and `<tool_call>` -> `tool_calls`:

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

## Building Locally from Source

```bash
nix develop .#python
cd bindings/python
maturin build --locked --release --features vulkan,dynamic-backends --out dist
python3 -m zipfile -e dist/*.whl .wheel-local
PYTHONPATH=.wheel-local pytest -v
```

See the root [README.md](https://github.com/kleinbem/layerstream-plugin/blob/main/README.md) for bare-metal benchmarks (vs AirLLM, HF `accelerate`, and stock `llama.cpp`) and Mesh-LLM plugin usage.
