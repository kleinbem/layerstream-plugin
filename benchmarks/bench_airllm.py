#!/usr/bin/env python3
"""Reproducible AirLLM benchmark matching crates/layerstream-core/examples/bench.rs."""

import argparse
import json
import os
import time
from pathlib import Path

WORKLOADS = [
    (
        "prompt_heavy",
        "You are reviewing a systems architecture document. Read the following technical specification carefully and summarize the core memory-management mechanism in one concise sentence.\n\n"
        "Specification:\n"
        "Modern large language models are structured as sequential stacks of transformer blocks where intermediate hidden states are several orders of magnitude smaller than layer weight matrices. "
        "On memory-constrained edge accelerators such as unified-memory ARM SoCs with 8 GiB LPDDR5 and PCIe NVMe storage, a 14-billion-parameter 4-bit quantized checkpoint requires roughly 9 GiB of weights plus key-value cache and compute workspace, exceeding total physical memory. "
        "Standard inference engines either fail during tensor allocation or rely on operating system virtual-memory paging via mmap, which pollutes the Linux page cache, triggers erratic swap thrashing, and serializes I/O behind synchronous page faults. "
        "Layer-streaming inference avoids full residency by allocating a small fixed pool of reusable compute buffers sized to the largest single transformer block, keeping embeddings, output projections, and optionally early hot layers resident, and asynchronously streaming remaining blocks from NVMe using page-aligned O_DIRECT reads overlapped with GPU compute. "
        "On unified-memory architectures where the CPU and GPU share physical DRAM, zero-copy mapped host buffers eliminate redundant host-to-device DMA copies and halve peak staging memory. "
        "Prompt prefill runs layer-major over multi-hundred-token batches so an entire prompt requires only one sequential storage sweep. "
        "Autoregressive decode remains storage-bandwidth-bound at one token per full sweep unless paired with speculative verification, where a zero-VRAM n-gram lookup proposer or a tiny resident draft model proposes multiple candidate tokens that are verified in parallel within a single storage pass, multiplying effective decode throughput without altering greedy target output distribution.\n\n"
        "Summary:",
        32,
    ),
    (
        "code_edit",
        "Repeat the following Rust struct verbatim and add a single field `pub timeout_secs: u64,` right before the closing brace:\n\n"
        "```rust\n"
        "#[derive(Debug, Clone)]\n"
        "pub struct StreamConfig {\n"
        "    pub model_path: PathBuf,\n"
        "    pub n_ctx: u32,\n"
        "    pub n_batch: u32,\n"
        "    pub resident_layers: Option<u32>,\n"
        "    pub vram_budget_bytes: u64,\n"
        "    pub use_gpu: bool,\n"
        "    pub prefer_direct_io: bool,\n"
        "}\n"
        "```\n\n"
        "Updated struct:\n```rust\n",
        48,
    ),
    (
        "open_ended",
        "Explain clearly how page-aligned O_DIRECT NVMe reads and asynchronous ping-pong GPU buffers interact when streaming transformer layers on an 8 GB edge device:",
        48,
    ),
]


def read_proc_kb(path: str, prefix: str) -> int:
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(prefix):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def page_cache_mib() -> float:
    return read_proc_kb("/proc/meminfo", "Cached:") / 1024.0


def peak_rss_mib() -> float:
    return read_proc_kb("/proc/self/status", "VmHWM:") / 1024.0


def current_rss_mib() -> float:
    return read_proc_kb("/proc/self/status", "VmRSS:") / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF repo id or local model dir")
    parser.add_argument("--compression", default="4bit", choices=["none", "4bit", "8bit"])
    parser.add_argument("--layer-shards-dir", default=None)
    parser.add_argument("--workload", default=None)
    args = parser.parse_args()

    import airllm
    import torch
    from airllm import AutoModel

    cache_before_load = page_cache_mib()
    t_load = time.perf_counter()

    kwargs = {"prefetching": True}
    if args.compression != "none":
        kwargs["compression"] = args.compression
    if args.layer_shards_dir:
        kwargs["layer_shards_saving_path"] = args.layer_shards_dir

    model = AutoModel.from_pretrained(args.model, **kwargs)
    load_time_s = time.perf_counter() - t_load

    results = []
    for wid, prompt, max_tokens in WORKLOADS:
        if args.workload and wid != args.workload:
            continue

        cache_before = page_cache_mib()
        input_tokens = model.tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=1024,
            padding=False,
        )
        prompt_tokens = int(input_tokens["input_ids"].shape[1])

        t0 = time.perf_counter()
        gen_out = model.generate(
            input_tokens["input_ids"].cuda(),
            max_new_tokens=max_tokens,
            use_cache=True,
            do_sample=False,
            return_dict_in_generate=True,
        )
        total_s = time.perf_counter() - t0
        out_ids = gen_out.sequences[0][prompt_tokens:]
        completion_tokens = int(out_ids.shape[0])
        text = model.tokenizer.decode(out_ids, skip_special_tokens=True)
        cache_after = page_cache_mib()

        results.append(
            {
                "engine": "airllm",
                "engine_version": getattr(airllm, "__version__", "unknown"),
                "mode": f"airllm_{args.compression}_prefetch",
                "model": args.model,
                "workload": wid,
                "load_time_s": round(load_time_s, 3),
                "total_time_s": round(total_s, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tok_per_s": round(completion_tokens / max(total_s, 1e-6), 3),
                "peak_rss_mib": round(peak_rss_mib(), 1),
                "current_rss_mib": round(current_rss_mib(), 1),
                "cuda_max_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1),
                "page_cache_delta_mib": round(cache_after - cache_before, 1),
                "page_cache_delta_since_start_mib": round(cache_after - cache_before_load, 1),
                "output_preview": text[:160],
            }
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
