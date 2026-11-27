#!/usr/bin/env python3
"""Reproducible JustVugg/colibri competitor baseline.

Matches the exact prompts, token counts, INA3221 power sampling, and memory/swap
telemetry of `crates/layerstream-core/examples/bench.rs` and `benchmarks/bench_llama_cpp.py`,
so `colibri` can be compared apples-to-apples against `layerstream-plugin` in the
Hugging Face dataset (`kleinbem/layerstream-orin-nano-benchmarks`) and Space.

Usage:
    python3 benchmarks/bench_colibri.py \
        --colibri-bin /path/to/colibri/c/coli \
        --model /mnt/models/bench/OLMoE-1B-7B-0125-Instruct-Q2_K.gguf \
        --mode colibri_olmoe_cuda \
        --out benchmarks/results/competitors/colibri_olmoe.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

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

TOK_S_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*tok/s", re.IGNORECASE)


def read_proc_kb(path: str, prefix: str) -> int:
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
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
    for hw in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (hw / "name").read_text(encoding="utf-8").strip() != "ina3221":
                continue
        except OSError:
            continue
        p_path = hw / "power1_input"
        if p_path.exists():
            try:
                return float(p_path.read_text(encoding="utf-8").strip()) / 1_000_000.0
            except OSError, ValueError:
                pass
    return None


def sample_idle_watts(duration_s: float = 1.0) -> float | None:
    if read_ina3221_watts() is None:
        return None
    samples: list[float] = []
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        w = read_ina3221_watts()
        if w is not None:
            samples.append(w)
        time.sleep(0.2)
    return sum(samples) / len(samples) if samples else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark JustVugg/colibri on LayerStream workloads")
    ap.add_argument("--colibri-bin", type=Path, required=True, help="Path to colibri binary")
    ap.add_argument("--model", type=Path, required=True, help="Path to GGUF model checkpoint")
    ap.add_argument("--mode", required=True, help="Benchmark mode label, e.g. colibri_olmoe_cuda")
    ap.add_argument("--engine-version", default="colibri", help="Colibri release tag or commit")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--extra-arg", action="append", default=[], help="Extra args passed to colibri")
    ap.add_argument("--out", type=Path, default=None, help="Optional output JSON path")
    args = ap.parse_args()

    idle_w = sample_idle_watts(1.0)
    records: list[dict[str, Any]] = []

    for wl_id, prompt, max_toks in WORKLOADS:
        before = mem_snapshot()
        stop_evt = threading.Event()
        watts: list[float] = []
        peak_rss_mib = 0.0

        def _poll_power(evt: threading.Event = stop_evt, samples: list[float] = watts) -> None:
            while not evt.is_set():
                w = read_ina3221_watts()
                if w is not None:
                    samples.append(w)
                time.sleep(0.25)

        t = threading.Thread(target=_poll_power, daemon=True)
        t.start()
        t0 = time.monotonic()
        status = "ok"
        stdout_txt = ""
        stderr_txt = ""
        try:
            proc = subprocess.Popen(  # noqa: S603 — explicit argv list invoking colibri binary
                [
                    str(args.colibri_bin),
                    "--model",
                    str(args.model),
                    "--prompt",
                    prompt,
                    "--max-tokens",
                    str(max_toks),
                    *args.extra_arg,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            while proc.poll() is None:
                rss = read_proc_kb(f"/proc/{proc.pid}/status", "VmHWM:") / 1024.0
                peak_rss_mib = max(peak_rss_mib, rss)
                if time.monotonic() - t0 > args.timeout:
                    proc.kill()
                    status = f"timeout_{args.timeout}s"
                    break
                time.sleep(0.2)
            out_s, err_s = proc.communicate(timeout=5)
            stdout_txt = out_s or ""
            stderr_txt = err_s or ""
            if status == "ok" and proc.returncode != 0:
                status = f"error_rc_{proc.returncode}"
        except OSError as exc:
            status = f"error_{type(exc).__name__}"
        finally:
            stop_evt.set()
            t.join(timeout=2)

        elapsed = time.monotonic() - t0
        after = mem_snapshot()
        avg_w = (sum(watts) / len(watts)) if watts else None
        net_w = (max(0.0, avg_w - idle_w)) if (avg_w is not None and idle_w is not None) else None

        decode_tps = None
        m = TOK_S_RE.search(stderr_txt + "\n" + stdout_txt)
        if m:
            decode_tps = round(float(m.group(1)), 3)

        j_per_tok = (
            round((avg_w * elapsed) / max_toks, 2)
            if (avg_w is not None and status == "ok")
            else None
        )
        net_j_per_tok = (
            round((net_w * elapsed) / max_toks, 2)
            if (net_w is not None and status == "ok")
            else None
        )

        records.append(
            {
                "engine": "colibri",
                "engine_version": args.engine_version,
                "mode": args.mode,
                "status": status,
                "model": args.model.name,
                "workload": wl_id,
                "completion_tokens": max_toks if status == "ok" else 0,
                "total_time_s": round(elapsed, 3),
                "decode_tok_per_s": decode_tps,
                "peak_rss_mib": round(peak_rss_mib, 1),
                "mem_avail_before_mib": round(before["mem_avail_mib"], 1),
                "mem_avail_after_mib": round(after["mem_avail_mib"], 1),
                "swap_delta_mib": round(after["swap_used_mib"] - before["swap_used_mib"], 1),
                "page_cache_delta_mib": round(after["cached_mib"] - before["cached_mib"], 1),
                "idle_watts": round(idle_w, 2) if idle_w is not None else None,
                "avg_watts": round(avg_w, 2) if avg_w is not None else None,
                "net_watts": round(net_w, 2) if net_w is not None else None,
                "joules_per_tok": j_per_tok,
                "net_joules_per_tok": net_j_per_tok,
                "output_preview": stdout_txt.strip()[:160],
            }
        )

    rendered = json.dumps(records, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
