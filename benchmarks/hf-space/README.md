---
title: LayerStream Edge LLM Benchmarks
emoji: ⚡
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
datasets:
  - kleinbem/layerstream-orin-nano-benchmarks
tags:
  - benchmarks
  - gguf
  - edge-ai
  - jetson
  - llama-cpp
  - layer-streaming
  - moe
short_description: 5-engine NVMe LLM benchmark explorer & live GGUF estimator
---

# LayerStream Edge LLM Benchmarks & Live GGUF Fit Estimator

Interactive static dashboard (always running, zero sleep) for **[`layerstream-plugin`](https://github.com/kleinbem/layerstream-plugin)** (`v0.6.3`), backed by the bare-metal NVIDIA Jetson Orin Nano dataset **[`kleinbem/layerstream-orin-nano-benchmarks`](https://huggingface.co/datasets/kleinbem/layerstream-orin-nano-benchmarks)** (**185 runs × 81 columns** across `layerstream-plugin`, stock `llama.cpp`, `colibri v2.2.0`, Hugging Face `accelerate`, and `AirLLM 4.0.0`).
