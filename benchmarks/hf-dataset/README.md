---
pretty_name: "Out-of-Core NVMe Layer Streaming on 8 GB Edge GPU: layerstream-plugin vs. llama.cpp, HF accelerate & AirLLM"
language:
  - en
license: apache-2.0
task_categories:
  - text-generation
tags:
  - llm-inference
  - edge-ai
  - jetson
  - llama-cpp
  - airllm
  - accelerate
  - nvme-streaming
  - moe
  - speculative-decoding
  - benchmark
  - power-efficiency
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-00000-of-00001.parquet
---

# Out-of-Core NVMe Layer Streaming on an 8 GB Jetson Orin Nano: 4-Engine Hardware Benchmark Dataset

**Source code & `v0.1.0` release:** [github.com/kleinbem/layerstream-plugin](https://github.com/kleinbem/layerstream-plugin) ([`v0.1.0` releases](https://github.com/kleinbem/layerstream-plugin/releases/tag/v0.1.0) · [engineering details](https://github.com/kleinbem/layerstream-plugin/blob/main/docs/DETAILS.md)) · **Interactive Explorer:** [kleinbem/layerstream-benchmarks Space](https://huggingface.co/spaces/kleinbem/layerstream-benchmarks)

Reproducible bare-metal benchmark dataset (**130 normalized runs, 76 columns** in Parquet + JSON + JSONL, plus reproduction scripts in `scripts/`) comparing four inference approaches on an **8 GB NVIDIA Jetson Orin Nano** running oversized dense and sparse MoE models (**7B MoE, 14B at 8.37 GiB, 30B-A3B MoE at 17.3 GiB, and 32B at 18.49 GiB — 2.53× total system RAM**):

1. **`layerstream-plugin v0.1.0`** (`113` rows across iterative tuning rounds R2–R8) — Native Rust + stock unpatched `llama.cpp` (`0693dbfb`) with page-aligned `O_DIRECT → cudaHostAlloc` async ping-pong GPU streaming, auto-sized hybrid partial-residency, sparse MoE active-expert streaming + scan-resistant TinyLFU hot-expert RAM cache, and speculative verification (greedy & `temperature > 0`).
2. **Stock `llama.cpp`** (`0693dbfb`, CUDA `sm_87`, `12` rows) — Standard `mmap` + `-ngl` / `-ot exps=CPU` partial offload (also representing the underlying GGUF runtime used by Ollama and LM Studio).
3. **Hugging Face `accelerate`** (`transformers 5.23.0 + bitsandbytes 0.50.2`, `1` row) — `device_map="auto"` + `offload_folder` + 4-bit NF4 disk offloading.
4. **`AirLLM 4.0.0`** (`torch 2.14.1+cu126`, `bitsandbytes 0.50.2`, `transformers 5.23.0`, `4` rows) — Python layer-by-layer sharded streaming (`4bit` & `BF16`).

Every row was measured directly on bare metal with `VmHWM` peak RSS, `/proc/meminfo` page-cache & swap deltas, kernel `pread_mib_s` I/O telemetry (Round 8), and **Jetson INA3221 hardware power-rail telemetry (`idle_watts`, `avg_watts`, `net_watts`, gross & net `J/tok`)**. Both wins and honest losses/limitations are reported transparently.

---

## Hardware & Environment

| Property | Value |
|---|---|
| **Device** | NVIDIA Jetson Orin Nano (`aarch64`, 6× Cortex-A78AE @ 1.728 GHz `performance`) |
| **Memory** | 7.3 GiB usable unified LPDDR5 (68 GB/s), Ampere 1024-core GPU (`sm_87` @ 1,020 MHz) |
| **Storage** | Crucial T500 PCIe NVMe SSD behind **LUKS2 `dm-crypt` (AES-XTS)** (`ext4` on LVM) |
| **OS / Stack** | NixOS 26.11 (`aarch64-linux`), JetPack 6.2 / CUDA 12.6, INA3221 `hwmon` `VDD_IN` rail (`~7.7–8.1 W` idle) |
| **Models** | `Qwen2.5-14B-Instruct` (`Q4_K_M` 8.37 GiB GGUF vs. HF safetensors `4bit`/`BF16`), `Qwen2.5-32B-Instruct-Q4_K_M` (18.49 GiB GGUF), `OLMoE-1B-7B-0125-Instruct-Q2_K` (2.5 GiB, 64 experts), `Qwen3-30B-A3B-Q4_K_M` (17.3 GiB, 128 experts), plus `Qwen2.5-0.5B` / `1.5B` & `Qwen3-0.6B` draft models |

### Standardized Workloads (Exact Same Prompts Across Engines)
1. **`prompt_heavy`** (`340` prompt tokens → `32` generated tokens): Systems-architecture summarization.
2. **`code_edit`** (`106–113` prompt tokens → `48` generated tokens): Verbatim Rust struct edit (high structural overlap).
3. **`open_ended`** (`31` prompt tokens → `48` generated tokens): Technical explanation from scratch (unconstrained prose).
4. **`multi_turn_prefix_reuse`** (`1,377` Turn-1 tokens → `1,414` Turn-2 tokens): Multi-turn KV prefix reuse latency test.

---

## 1. Headline 4-Engine Comparison (`Qwen2.5-14B-Instruct`, 8.37 GiB on 7.3 GiB Orin Nano)

*Round 8 (`v0.1.0`) rows report median across `--reps 3` with `--warmup`, exact `pread_mib_s`, live `swap_delta_mib`, and gross / net `J/tok` (subtracting measured idle board watts).*

| Engine & Configuration | Workload | `R` | Load | TTFT | Decode (`tok/s`) | `pread` BW | Peak RSS | Swap $\Delta$ | Power (Idle) | Gross / Net `J/tok` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Stock `llama.cpp`** (`-ngl 12`/`18`, `-b 512`)* | all 3 | — | — | OOM | `error_rc_1` | — | 5,978 MiB | +3,711 MiB | 9.9 W (8.0 W) | — |
| **Stock `llama.cpp`** (`-ngl 0 -dev none -b 512`)* | `open_ended` | — | — | >300 s | `timeout` | — | 6,110 MiB | +363 MiB | 10.6 W (7.9 W) | — |
| **HF `accelerate`** (`4-bit NF4` disk offload)† | `open_ended` | — | 144.0 s | >156 s | `timeout_300s` | — | — | — | — | — |
| **AirLLM 4.0.0** (`4bit`, cold first-run)‡ | `prompt_heavy` | — | 365.53 s | 32.09 s | 0.032 *(32 tok)* | — | 5,391 MiB | — | — | — |
| **AirLLM 4.0.0** (`4bit`, warm shards)‡ | `code_edit` | — | 2.55 s | 33.37 s | 0.033 *(8 tok)* | — | 3,536 MiB | — | 9.30 W | 288.2 / — |
| **AirLLM 4.0.0** (`4bit`, warm shards)‡ | `open_ended` | — | 2.57 s | 32.86 s | 0.033 *(8 tok)* | — | 3,531 MiB | — | 9.33 W | 287.1 / — |
| **AirLLM 4.0.0** (`BF16`, `prefetching=True`)‡ | `open_ended` | — | 81.81 s | 45.41 s | 0.023 *(4 tok)* | — | 5,218 MiB | — | 9.31 W | 412.4 / — |
| **`layerstream` R8** (`R=0` pure stream, `reps=2`) | `open_ended` | 0/48 | 2.99 s | 5.97 s | **0.174** *(16 tok)* | 1,427 MiB/s | **2,458 MiB** | **-3.0 MiB** | 10.37 W (7.71 W) | 59.75 / 15.33 |
| **`layerstream` R8** (`auto` + `0.5B`, `reps=3`) | `prompt_heavy` | 4/48 | 4.94 s | 4.47 s | **0.760** *(0.755–0.760)* | 2,796 MiB/s | 3,709 MiB | **-0.3 MiB** | 13.21 W (7.77 W) | 18.69 / 7.69 |
| **`layerstream` R8** (`auto` + `0.5B`, `reps=3`) | `open_ended` | 4/48 | 4.94 s | 2.95 s | **1.578** *(1.564–1.582)* | 2,794 MiB/s | 3,716 MiB | **0.0 MiB** | 12.84 W (7.77 W) | 8.75 / 3.45 |
| **`layerstream` R8** (`auto` + `0.5B`, `reps=3`) | `code_edit` | 4/48 | 4.94 s | 3.26 s | **3.158** *(2.244–3.159)* | 2,800 MiB/s | 3,714 MiB | **-0.3 MiB** | 12.88 W (7.77 W) | 4.87 / 1.93 |
| **`layerstream` R8** (`b5400` + `0.5B`, `reps=3`) | `prompt_heavy` | 7/48 | 4.58 s | 4.24 s | **0.824** *(0.817–0.827)* | **2,924 MiB/s** | 4,171 MiB | **0.0 MiB** | 13.29 W (7.70 W) | 17.51 / 7.37 |
| **`layerstream` R8** (`b5400` + `0.5B`, `reps=3`) | `open_ended` | 7/48 | 4.58 s | **2.74 s** | **1.699** *(1.685–1.713)* | **2,932 MiB/s** | 4,178 MiB | **0.0 MiB** | 13.29 W (7.70 W) | 8.42 / 3.54 |
| **`layerstream` R8** (`b5400` + `0.5B`, `reps=3`) | `code_edit` | 7/48 | 4.58 s | 3.01 s | **3.405** *(3.284–3.414)* | **2,961 MiB/s** | 4,176 MiB | **0.0 MiB** | 13.16 W (7.70 W) | **4.61 / 1.91** |
| **`layerstream` R6** (`R=20` + `0.5B`, max `R`) | `code_edit` | 20/48 | 6.75 s | 2.31 s | **5.974** *(open: 2.109)* | — | 6,153 MiB | — | 12.89 W | 2.73 / — |

- \*Stock `llama.cpp` also represents the underlying GGUF runtime used by Ollama and LM Studio.
- †HF `accelerate` and ‡AirLLM 4.0.0 ran in the same `torch 2.14.1+cu126` Python venv (which emits a Jetson `CC 8.7` unsupported-build warning). AirLLM warm runs generated `8` tokens (`BF16`: `4` tokens) vs. `48` tokens for `layerstream-plugin`, and AirLLM uses `bnb 4-bit` (`36.8 GiB` total disk footprint including HF cache + shards) vs. `GGUF Q4_K_M` (`8.37 GiB` single file, zero conversion).
- **Multi-turn KV prefix reuse (`r8_14b_auto_draft05b_multiturn`):** Turn 1 (`1,377` tokens) `14.877 s` → Turn 2 (`1,414` tokens) **`2.993 s`** (**`4.97×` faster**).

---

## 2. Running a 32B Model (`18.49 GiB`, 2.53× Total System RAM) on 7.3 GiB Orin Nano

| Engine & Mode (`Qwen2.5-32B-Instruct-Q4_K_M`) | Workload | `R` | Load | TTFT | Decode (`tok/s`) | `pread` BW | Peak RSS | Swap $\Delta$ | Gross / Net `J/tok` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Stock `llama.cpp`** (`-ngl 10`) | `open_ended` | — | — | >300 s | `timeout` | — | 5,715 MiB | **+17,243 MiB** | — |
| **`layerstream` R6** (`R=0`, 0 draft) | `prompt_heavy` | 0/64 | **2.14 s** | 14.48 s | **0.094** *(559.7 GiB read)* | — | **2,957 MiB** | — | 119.1 / — |
| **`layerstream` R8** (`b5400` + `0.5B`, `reps=2`) | `prompt_heavy` | 2/64 | 3.76 s | 18.01 s | **0.171** *(0.156–0.185)* | 1,229 MiB/s | 4,109 MiB | **0.0 MiB** | 64.13 / 14.21 |
| **`layerstream` R8** (`b5400` + `0.5B`, `reps=2`) | `open_ended` | 2/64 | 3.76 s | 16.56 s | **0.304** *(0.304–0.304)* | 1,311 MiB/s | 4,115 MiB | **0.0 MiB** | 36.61 / 8.30 |
| **`layerstream` R8** (`b5400` + `0.5B`, `reps=2`) | `code_edit` | 2/64 | 3.76 s | 14.70 s | **0.840** *(0.838–0.841)* | 1,338 MiB/s | 4,113 MiB | **-0.5 MiB** | 15.51 / 3.82 |
| **`layerstream` R6** (`b5600` + `0.5B`, clean RAM) | `code_edit` | 6/64 | 6.84 s | 8.42 s | **0.944** *(open: 0.506)* | — | 5,294 MiB | — | 14.3 / — |

*(Note: `573,082 MiB` read in `r6_32b_stream_r0` equals **`559.7 GiB`**.)*

---

## 3. Sparse MoE Benchmarks (`OLMoE-1B-7B` & `Qwen3-30B-A3B`)

| Model & Engine | Workload | `R` | Load | TTFT | Decode (`tok/s`) | Cache Hit | `pread` BW | Peak RSS | Swap $\Delta$ | Gross / Net `J/tok` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **OLMoE-1B-7B** (`Q2_K`) — `layerstream` full-layer | `open_ended` | 0/16 | 0.51 s | 1.45 s | 0.763 | — | 1,943 MiB/s | 1,312 MiB | -2.0 MiB | 14.34 / 4.01 |
| **OLMoE-1B-7B** (`Q2_K`) — `layerstream` **sparse + TinyLFU** | `prompt_heavy` | 0/16 | 0.68 s | 2.08 s | **13.667** *(13.65–13.92)* | **100%** | 1,405 MiB/s | 2,921 MiB | **0.0 MiB** | **1.67 / 0.57** |
| **OLMoE-1B-7B** (`Q2_K`) — `layerstream` **sparse + TinyLFU** | `code_edit` | 0/16 | 0.68 s | 1.87 s | **13.054** *(12.45–13.24)* | **99.3%** | 1,269 MiB/s | 2,953 MiB | **0.0 MiB** | **1.44 / 0.51** |
| **OLMoE-1B-7B** (`Q2_K`) — `layerstream` **sparse + TinyLFU** | `open_ended` | 0/16 | 0.68 s | 1.72 s | **12.047** *(11.45–12.51)* | **98.0%** | 1,168 MiB/s | 2,971 MiB | **0.0 MiB** | **1.40 / 0.45** |
| **Qwen3-30B-A3B** (`17.3 GiB`) — Stock `llama.cpp` GPU MoE | `open_ended` | — | — | OOM | `error_rc_1` | — | — | 6,014 MiB | +12,342 MiB | — |
| **Qwen3-30B-A3B** (`17.3 GiB`) — Stock `llama.cpp` **CPU `mmap`** | `open_ended` | — | 25.34 s | 17.37 s | **1.270** *(wins!)* | OS cache | — | 6,071 MiB | +248 MiB | **15.26** / — |
| **Qwen3-30B-A3B** (`17.3 GiB`) — `layerstream` sparse + TinyLFU + `0.6B` | `open_ended` | 0/48 | **3.26 s** | **11.24 s** | 0.317 *(0.233–0.400)* | 23.3% | 708 MiB/s | **3,969 MiB** | **0.0 MiB** | 32.98 / 3.21 |
| **Qwen3-30B-A3B** (`17.3 GiB`) — `layerstream` sparse + TinyLFU + `0.6B` | `code_edit` | 0/48 | 33.06 s | 13.60 s | 0.113 *(0.066–0.159)* | 16.0% | 267 MiB/s | **3,974 MiB** | **-2.0 MiB** | 89.98 / 5.21 |

---

## 4. Honest Trade-Offs & Limitations

- **Where stock `llama.cpp` CPU `mmap` beats `layerstream-plugin`:** On `Qwen3-30B-A3B` (`128` experts/layer) on `orin-nano`, stock `llama.cpp` CPU `mmap` (`-ngl 0 -dev none -b 512`) reached **`1.270 tok/s`** (`6,071 MiB` RSS, `+248 MiB` swap) because the Linux page cache absorbs random hot expert slices in RAM, whereas `layerstream-plugin`'s small random `O_DIRECT` bursts trigger 30-second Tegra PCIe/NVMe IRQ stalls (`nvme1: I/O tag ... timeout, completion polled`) on `orin-nano`.
- **Jetson CUDA `uma_zero_copy=false`:** Upstream `llama.cpp` currently disables `CUDA_Host` unified memory on Jetson (`upstream: ggml-org/llama.cpp#15034`). All Jetson CUDA rows in this dataset therefore have `uma_zero_copy: false` and `is_unified_memory: true`, using pinned `cudaHostAlloc` + `cudaMemcpyAsync` instead of host-mapped device memory.
- **Post-eviction CUDA driver context:** Cold startup uses `0 MB` GPU memory; after serving and idle eviction, all model/KV/slot buffers are freed, leaving `~230–325 MiB` held by CUDA's primary runtime context.
- **Host `dm-crypt` bandwidth variance:** On `orin-nano`, LUKS2 `dm-crypt` sequential `pread` measured `2,794–2,961 MiB/s` under clean memory conditions and dropped to `1,025–1,338 MiB/s` under `zram` pressure (`r8_14b_*_temp07`, `r8_32b_*`); every Round 8 row records exact `pread_mib_s` alongside throughput so bandwidth effects are never hidden.
- **Single-sweep latency floor on dense `R=0`:** Without hybrid residency or speculation, dense NVMe streaming is strictly bounded by `model_size / nvme_bw` (`~0.17–0.19 tok/s` on 14B, `~0.094 tok/s` on 32B).

---

## 5. Dataset Schema (76 Columns) & Reproduction

Each of the **130 rows** in `data/train-00000-of-00001.parquet` (`benchmarks.json`, `benchmarks.jsonl`) includes:
`round`, `source_file`, `hardware`, `hardware_id`, `engine`, `engine_version`, `git_rev`, `mode`, `status`, `model`, `draft_model`, `workload`, `chat_template_applied`, `warmup`, `reps`, `temperature`, `seed`, `threads`, `kv_cache_type`, `vram_budget_auto`, `vram_budget_mb`, `prompt_tokens`, `completion_tokens`, `load_time_s`, `ttft_s`, `ttft_s_min`, `ttft_s_max`, `decode_time_s`, `total_time_s`, `decode_tok_per_s`, `decode_tok_per_s_min`, `decode_tok_per_s_max`, `total_tok_per_s`, `total_tok_per_s_min`, `total_tok_per_s_max`, `sweeps`, `drafted_tokens`, `accepted_tokens`, `resident_layers`, `streamed_layers`, `n_layers`, `resident_mib`, `slot_mib`, `expert_slot_mib`, `io_mib_read`, `pread_mib_s`, `effective_io_mib_s`, `expert_cache_hits`, `expert_cache_misses`, `expert_cache_hit_rate`, `expert_cache_mib_saved`, `moe_layers`, `moe_active_streaming`, `uma_zero_copy`, `is_unified_memory`, `direct_io_active`, `peak_rss_mib`, `current_rss_mib`, `mem_avail_before_mib`, `mem_avail_after_mib`, `swap_delta_mib`, `cuda_max_allocated_mib`, `page_cache_delta_mib`, `page_cache_delta_since_start_mib`, `idle_watts`, `avg_watts`, `net_watts`, `joules_per_tok`, `net_joules_per_tok`, `turn1_prompt_tokens`, `turn1_ttft_s`, `turn2_prompt_tokens`, `turn2_ttft_s`, `ttft_speedup_x`, `notes`, `output_preview`.

```bash
# Reproduce layerstream-plugin (CUDA sm_87)
cargo run --release -p layerstream-core --example bench --features cuda -- \
  --model Qwen2.5-14B-Instruct-Q4_K_M.gguf \
  --draft-model qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --vram-budget-mb auto --kv-cache-type q8_0 --threads 4 --warmup --reps 3 --multi-turn

# Reproduce stock llama.cpp competitor baseline
python3 benchmarks/bench_llama_cpp.py \
  --bin ./llama-completion --model Qwen2.5-14B-Instruct-Q4_K_M.gguf \
  --ngl 12 --extra-arg -b --extra-arg 512 --extra-arg -ub --extra-arg 512

# Reproduce HF accelerate baseline
python3 benchmarks/bench_accelerate.py \
  --model Qwen/Qwen2.5-14B-Instruct --offload-dir ./accel-offload \
  --gpu-max-mem 3GiB --cpu-max-mem 1GiB

# Reproduce AirLLM 4.0.0 baseline
python3 benchmarks/bench_airllm.py \
  --model Qwen/Qwen2.5-14B-Instruct --compression 4bit \
  --layer-shards-dir ./airllm-shards

# Regenerate normalized JSON / JSONL / Parquet dataset
python3 benchmarks/export_hf_dataset.py \
  --results-dir benchmarks/results --out-dir benchmarks/hf-dataset
```
