#!/usr/bin/env python3
"""Reproducible stock llama.cpp (mmap + -ngl / MoE CPU offload) competitor baseline.

Matches the exact prompts, token counts, INA3221 power sampling, and memory/swap
telemetry of `crates/layerstream-core/examples/bench.rs`.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import threading
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


def mem_snapshot() -> dict[str, float]:
    cached = read_proc_kb("/proc/meminfo", "Cached:") / 1024.0
    avail = read_proc_kb("/proc/meminfo", "MemAvailable:") / 1024.0
    swap_total = read_proc_kb("/proc/meminfo", "SwapTotal:") / 1024.0
    swap_free = read_proc_kb("/proc/meminfo", "SwapFree:") / 1024.0
    return {
        "cached_mib": cached,
        "mem_avail_mib": avail,
        "swap_used_mib": max(0.0, swap_total - swap_free),
    }


def read_ina3221_watts() -> float | None:
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            if Path(hw, "name").read_text().strip() != "ina3221":
                continue
        except OSError:
            continue
        p_path = Path(hw, "power1_input")
        if p_path.exists():
            try:
                return float(p_path.read_text().strip()) / 1_000_000.0
            except (OSError, ValueError):
                pass
        try:
            mv = float(Path(hw, "in1_input").read_text().strip())
            ma = float(Path(hw, "curr1_input").read_text().strip())
            return (mv * ma) / 1_000_000.0
        except (OSError, ValueError):
            continue
    return None


def sample_idle_watts(duration_s: float = 1.5) -> float | None:
    samples = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        w = read_ina3221_watts()
        if w is not None:
            samples.append(w)
        time.sleep(0.25)
    return sum(samples) / len(samples) if samples else None


class ChildMonitor:
    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._stop = threading.Event()
        self.watts: list[float] = []
        self.peak_rss_mib = 0.0
        self.peak_swap_mib = 0.0
        self.min_avail_mib = 1e9
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            w = read_ina3221_watts()
            if w is not None:
                self.watts.append(w)
            hwm = read_proc_kb(f"/proc/{self._pid}/status", "VmHWM:") / 1024.0
            rss = read_proc_kb(f"/proc/{self._pid}/status", "VmRSS:") / 1024.0
            self.peak_rss_mib = max(self.peak_rss_mib, hwm, rss)
            snap = mem_snapshot()
            self.peak_swap_mib = max(self.peak_swap_mib, snap["swap_used_mib"])
            self.min_avail_mib = min(self.min_avail_mib, snap["mem_avail_mib"])
            self._stop.wait(0.2)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def parse_llama_timings(stderr: str) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {
        "load_time_s": None,
        "prompt_tokens": None,
        "prompt_eval_s": None,
        "completion_tokens": None,
        "eval_s": None,
        "decode_tok_per_s": None,
    }
    m = re.search(r"load time\s*=\s*([\d.]+)\s*ms", stderr)
    if m:
        out["load_time_s"] = float(m.group(1)) / 1000.0
    m = re.search(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", stderr
    )
    if m:
        out["prompt_eval_s"] = float(m.group(1)) / 1000.0
        out["prompt_tokens"] = int(m.group(2))
    m = re.search(
        r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([\d.]+)\s*tokens per second",
        stderr,
    )
    if m:
        out["eval_s"] = float(m.group(1)) / 1000.0
        out["completion_tokens"] = int(m.group(2))
        out["decode_tok_per_s"] = float(m.group(3))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="Path to llama-completion or llama-cli")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="llama_cpp_mmap")
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--kv-cache-type", default="q8_0")
    ap.add_argument("--extra-arg", action="append", default=[])
    ap.add_argument("--workload", default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--timeout-s", type=int, default=420)
    args = ap.parse_args()

    idle_w = sample_idle_watts()
    results = []

    for wl_id, prompt, default_max_tok in WORKLOADS:
        if args.workload and wl_id != args.workload:
            continue
        max_tok = args.max_tokens or default_max_tok
        before = mem_snapshot()

        cmd = [
            args.bin,
            "-m",
            args.model,
            "-ngl",
            str(args.ngl),
            "-c",
            str(args.ctx),
            "-t",
            str(args.threads),
            "-n",
            str(max_tok),
            "--temp",
            "0",
            "--no-conversation",
            "--no-display-prompt",
            "-p",
            prompt,
        ]
        if args.kv_cache_type != "f16":
            cmd += ["-ctk", args.kv_cache_type, "-ctv", args.kv_cache_type, "-fa", "on"]
        cmd += args.extra_arg

        t0 = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        mon = ChildMonitor(proc.pid)
        status = "ok"
        try:
            stdout, stderr = proc.communicate(timeout=args.timeout_s)
            if proc.returncode != 0:
                status = f"error_rc_{proc.returncode}"
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            status = "timeout"
        finally:
            mon.stop()

        wall_s = time.monotonic() - t0
        after = mem_snapshot()
        timings = parse_llama_timings(stderr)
        avg_w = (sum(mon.watts) / len(mon.watts)) if mon.watts else None
        toks = timings["completion_tokens"] or 0
        eval_s = timings["eval_s"] or wall_s
        prompt_s = timings["prompt_eval_s"] or 0.0
        # Compute gross energy over `wall_s` — the exact interval during which `mon.watts` was sampled.
        j_per_tok = (
            round((avg_w * wall_s) / max(toks, 1), 2)
            if (avg_w is not None and toks > 0)
            else None
        )

        results.append(
            {
                "engine": "llama.cpp",
                "mode": args.mode,
                "status": status,
                "model": Path(args.model).name,
                "ngl": args.ngl,
                "kv_cache_type": args.kv_cache_type,
                "extra_args": args.extra_arg,
                "workload": wl_id,
                "load_time_s": round(timings["load_time_s"], 3)
                if timings["load_time_s"] is not None
                else None,
                "ttft_s": round(prompt_s, 3) if timings["prompt_eval_s"] is not None else None,
                "decode_time_s": round(eval_s, 3),
                "total_time_s": round(wall_s, 3),
                "prompt_tokens": timings["prompt_tokens"],
                "completion_tokens": toks,
                "decode_tok_per_s": round(timings["decode_tok_per_s"], 3)
                if timings["decode_tok_per_s"] is not None
                else 0.0,
                "total_tok_per_s": round(toks / max(infer_s, 1e-6), 3) if toks > 0 else 0.0,
                "peak_rss_mib": round(mon.peak_rss_mib, 1),
                "mem_avail_before_mib": round(before["mem_avail_mib"], 1),
                "mem_avail_min_mib": round(mon.min_avail_mib, 1)
                if mon.min_avail_mib < 1e8
                else None,
                "page_cache_delta_mib": round(after["cached_mib"] - before["cached_mib"], 1),
                "swap_delta_mib": round(mon.peak_swap_mib - before["swap_used_mib"], 1),
                "idle_watts": round(idle_w, 2) if idle_w is not None else None,
                "avg_watts": round(avg_w, 2) if avg_w is not None else None,
                "joules_per_tok": j_per_tok,
                "output_preview": stdout.strip()[:160],
                "stderr_tail": stderr.strip()[-300:] if status != "ok" else "",
            }
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
