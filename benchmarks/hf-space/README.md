---
title: LayerStream Edge LLM Benchmarks & GGUF Fit MCP
emoji: ⚡
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "5.49.1"
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
datasets:
  - kleinbem/layerstream-orin-nano-benchmarks
tags:
  - mcp-server-track
  - benchmarks
  - gguf
  - edge-ai
  - jetson
  - llama-cpp
  - layer-streaming
  - moe
short_description: 5-engine NVMe LLM benchmark explorer, GGUF fit & MCP server
---

# LayerStream Edge LLM Benchmarks & Live GGUF Fit Estimator (MCP Server)

Interactive **Gradio 5 + native MCP Server** for **[`layerstream-plugin`](https://github.com/kleinbem/layerstream-plugin)** (`v0.6.3`), backed by the bare-metal NVIDIA Jetson Orin Nano dataset **[`kleinbem/layerstream-orin-nano-benchmarks`](https://huggingface.co/datasets/kleinbem/layerstream-orin-nano-benchmarks)** (**185 runs × 81 columns** across `layerstream-plugin`, stock `llama.cpp`, `colibri v2.2.0`, Hugging Face `accelerate`, and `AirLLM 4.0.0`).

### Connect as an MCP Server (Claude Code, Cursor, VS Code, Zed)

```json
{
  "mcpServers": {
    "layerstream-benchmarks": {
      "url": "https://kleinbem-layerstream-benchmarks.hf.space/gradio_api/mcp/sse"
    }
  }
}
```

Exposed MCP tools:
- `estimate_hf_gguf_fit` — Live-inspects any public Hugging Face GGUF repository (HTTP Range header probe, zero multi-GB download), ranks quants for your VRAM/NVMe budget, and predicts LayerStream throughput.
- `recommend_edge_model` — Recommends the best GGUF + quant + draft model for a target edge GPU backed by bare-metal Round 12 measurements.
- `query_benchmarks` — Filters all 185 bare-metal benchmark runs by model, engine, workload, and round.
- `compare_round_progression` — Shows round-over-round throughput & energy progression (`R2` → `R12 v0.6.3`).
