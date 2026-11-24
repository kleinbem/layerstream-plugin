#!/usr/bin/env python3
"""Reproducible Hugging Face transformers + accelerate disk-offload competitor baseline.

Uses `device_map="auto"` + `offload_folder` + bitsandbytes 4-bit (NF4) with bounded
GPU/CPU memory budgets (`max_memory`) so oversized models (e.g. Qwen2.5-14B on an 8 GB
Jetson Orin Nano) offload cleanly to NVMe rather than OOM-crashing.
"""

import argparse
import json
import threading
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


def mem_snapshot() -> dict[str, float]:
    cached = read_proc_kb("/proc/meminfo", "Cached:") / 1024.0
    avail = read_proc_kb("/proc/meminfo", "MemAvailable:") / 1024.0
    swap_total = read_proc_kb("/proc/meminfo", "SwapTotal:") / 1024.0
    swap_free = read_proc_kb("/proc/meminfo", "SwapFree:") / 1024.0
    return {
        "cached_mib": cached,
        "mem_avail_mib": avail,
        "swap_used_mib": max(0.0, swap_total - swap_free),
        "peak_rss_mib": read_proc_kb("/proc/self/status", "VmHWM:") / 1024.0,
        "current_rss_mib": read_proc_kb("/proc/self/status", "VmRSS:") / 1024.0,
    }


def read_ina3221_watts() -> float | None:
    for hw in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (hw / "name").read_text().strip() != "ina3221":
                continue
        except OSError:
            continue
        p_path = hw / "power1_input"
        if p_path.exists():
            try:
                return float(p_path.read_text().strip()) / 1_000_000.0
            except OSError, ValueError:
                pass
        try:
            mv = float((hw / "in1_input").read_text().strip())
            ma = float((hw / "curr1_input").read_text().strip())
            return (mv * ma) / 1_000_000.0
        except OSError, ValueError:
            continue
    return None


class PowerSampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._samples: list[float] = []
        self._thread: threading.Thread | None = None
        if read_ina3221_watts() is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            w = read_ina3221_watts()
            if w is not None:
                self._samples.append(w)
            self._stop.wait(0.25)

    def finish(self) -> float | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return (sum(self._samples) / len(self._samples)) if self._samples else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local HF snapshot dir")
    ap.add_argument("--offload-dir", required=True)
    ap.add_argument("--mode", default="hf_accelerate_4bit_disk")
    ap.add_argument("--gpu-max-mem", default="3GiB")
    ap.add_argument("--cpu-max-mem", default="1GiB")
    ap.add_argument("--workload", default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    Path(args.offload_dir).mkdir(parents=True, exist_ok=True)
    start_snap = mem_snapshot()
    t_load = time.monotonic()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kwargs: dict = {
        "device_map": "auto",
        "max_memory": {0: args.gpu_max_mem, "cpu": args.cpu_max_mem},
        "offload_folder": args.offload_dir,
        "offload_state_dict": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float16,
    }
    if not args.no_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    load_time_s = time.monotonic() - t_load

    results = []
    for wl_id, prompt, default_max_tok in WORKLOADS:
        if args.workload and wl_id != args.workload:
            continue
        max_tok = args.max_tokens or default_max_tok
        before = mem_snapshot()
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(0)
        attention_mask = inputs["attention_mask"].to(0)
        prompt_tokens = int(input_ids.shape[1])

        ps = PowerSampler()
        t0 = time.monotonic()
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tok,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        total_s = time.monotonic() - t0
        avg_w = ps.finish()
        after = mem_snapshot()

        gen_ids = out_ids[0][prompt_tokens:]
        comp_toks = int(gen_ids.shape[0])
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        j_per_tok = (
            round((avg_w * total_s) / max(comp_toks, 1), 2)
            if (avg_w is not None and comp_toks > 0)
            else None
        )

        results.append(
            {
                "engine": "hf-accelerate",
                "mode": args.mode,
                "model": args.model,
                "gpu_max_mem": args.gpu_max_mem,
                "cpu_max_mem": args.cpu_max_mem,
                "workload": wl_id,
                "load_time_s": round(load_time_s, 3),
                "total_time_s": round(total_s, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp_toks,
                "total_tok_per_s": round(comp_toks / max(total_s, 1e-6), 4),
                "peak_rss_mib": round(after["peak_rss_mib"], 1),
                "current_rss_mib": round(after["current_rss_mib"], 1),
                "page_cache_delta_mib": round(after["cached_mib"] - before["cached_mib"], 1),
                "page_cache_delta_since_start_mib": round(
                    after["cached_mib"] - start_snap["cached_mib"], 1
                ),
                "swap_delta_mib": round(after["swap_used_mib"] - before["swap_used_mib"], 1),
                "avg_watts": round(avg_w, 2) if avg_w is not None else None,
                "joules_per_tok": j_per_tok,
                "output_preview": text[:160],
            }
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
