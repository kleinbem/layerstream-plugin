#!/usr/bin/env python3
"""One-command reproducible benchmark round runner for bare-metal LayerStream hosts
(e.g., 8 GB Jetson Orin Nano, Linux/CUDA, macOS/Metal).

Runs the standardized 6-config benchmark matrix (plus optional MTP and hybrid
SSM/DeltaNet models), checks free memory beforehand, logs full stderr for
CPU-fallback / graph-split / NVMe-timeout auditing, and automatically runs
`benchmarks/check_regressions.py` against a baseline round.

Usage:
    python3 benchmarks/run_round.py \
        --round r12 \
        --bench target-cuda-dyn/release/examples/bench \
        --models-dir /mnt/models/bench \
        --out-dir /mnt/models/bench/results/round12_v062 \
        --baseline benchmarks/results/round11_v061
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def read_mem_available_mib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) / 1024.0
    return None


def run_case(
    bench_bin: Path,
    out_dir: Path,
    name: str,
    extra_args: list[str],
    cooldown_s: int,
) -> None:
    json_path = out_dir / f"{name}.json"
    log_path = out_dir / f"{name}.log"
    cmd = [str(bench_bin), *extra_args]
    print(f"==> Running {name} ...", flush=True)
    with json_path.open("w", encoding="utf-8") as jf, log_path.open("w", encoding="utf-8") as lf:
        subprocess.run(  # noqa: S603 — explicit argv list invoking compiled bench binary
            cmd,
            stdout=jf,
            stderr=lf,
            check=True,
        )
    if cooldown_s > 0:
        time.sleep(cooldown_s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a standardized LayerStream benchmark round")
    ap.add_argument("--round", required=True, help="Short round prefix, e.g. r12")
    ap.add_argument("--bench", type=Path, required=True, help="Path to compiled bench binary")
    ap.add_argument("--models-dir", type=Path, required=True, help="Directory with GGUF checkpoints")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for JSON + logs")
    ap.add_argument("--baseline", type=Path, default=None, help="Baseline round dir for regression check")
    ap.add_argument("--mtp-model", type=Path, default=None, help="Optional MTP-enabled GGUF to benchmark")
    ap.add_argument("--hybrid-model", type=Path, default=None, help="Optional hybrid SSM/DeltaNet GGUF")
    ap.add_argument("--cooldown", type=int, default=3, help="Cooldown seconds between runs")
    ap.add_argument("--min-mem-mib", type=float, default=3800.0, help="Warn if MemAvailable is below this")
    args = ap.parse_args()

    mem_avail = read_mem_available_mib()
    if mem_avail is not None:
        print(f"Host MemAvailable: {mem_avail:.0f} MiB", flush=True)
        if mem_avail < args.min_mem_mib:
            print(
                f"WARNING: MemAvailable ({mem_avail:.0f} MiB) < {args.min_mem_mib:.0f} MiB — "
                "stop heavy background services (e.g. mesh-llm, frigate) for clean numbers!",
                file=sys.stderr,
                flush=True,
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    m = args.models_dir
    r = args.round

    cases: list[tuple[str, Path, list[str]]] = [
        (
            f"{r}_olmoe_r0_cached",
            m / "OLMoE-1B-7B-0125-Instruct-Q2_K.gguf",
            [
                "--mode", f"{r}_olmoe_r0_sparse_cached",
                "--resident-layers", "0",
                "--kv-cache-type", "q8_0",
                "--keep-expert-cache", "--warmup", "--reps", "3",
            ],
        ),
        (
            f"{r}_14b_b5400_draft05b",
            m / "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
            [
                "--draft-model", str(m / "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
                "--mode", f"{r}_14b_b5400_draft05b",
                "--vram-budget-mb", "5400",
                "--kv-cache-type", "q8_0",
                "--ngram", "--warmup", "--reps", "3",
            ],
        ),
        (
            f"{r}_14b_auto_draft05b",
            m / "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
            [
                "--draft-model", str(m / "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
                "--mode", f"{r}_14b_auto_draft05b",
                "--vram-budget-mb", "auto",
                "--kv-cache-type", "q8_0",
                "--ngram", "--multi-turn", "--warmup", "--reps", "3",
            ],
        ),
        (
            f"{r}_qwen3_30b_a3b_draft06b_open",
            m / "Qwen3-30B-A3B-Q4_K_M.gguf",
            [
                "--draft-model", str(m / "Qwen3-0.6B-Q4_K_M.gguf"),
                "--mode", f"{r}_qwen3_30b_a3b_r0_draft06b",
                "--resident-layers", "0",
                "--kv-cache-type", "q8_0",
                "--ngram", "--chat", "--workload", "open_ended", "--max-tokens", "32", "--reps", "2",
            ],
        ),
        (
            f"{r}_qwen3_30b_a3b_draft06b_code",
            m / "Qwen3-30B-A3B-Q4_K_M.gguf",
            [
                "--draft-model", str(m / "Qwen3-0.6B-Q4_K_M.gguf"),
                "--mode", f"{r}_qwen3_30b_a3b_r0_draft06b",
                "--resident-layers", "0",
                "--kv-cache-type", "q8_0",
                "--ngram", "--chat", "--workload", "code_edit", "--max-tokens", "32", "--reps", "2",
            ],
        ),
        (
            f"{r}_32b_b5400_draft05b",
            m / "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
            [
                "--draft-model", str(m / "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
                "--mode", f"{r}_32b_b5400_draft05b",
                "--vram-budget-mb", "5400",
                "--kv-cache-type", "q8_0",
                "--ngram", "--reps", "2",
            ],
        ),
    ]

    if args.mtp_model is not None:
        cases.append((
            f"{r}_mtp_self_spec",
            args.mtp_model,
            [
                "--mode", f"{r}_mtp_self_spec",
                "--vram-budget-mb", "auto",
                "--kv-cache-type", "q8_0",
                "--chat", "--warmup", "--reps", "2",
            ],
        ))

    if args.hybrid_model is not None:
        cases.append((
            f"{r}_hybrid_rs_rollback",
            args.hybrid_model,
            [
                "--mode", f"{r}_hybrid_rs_rollback",
                "--vram-budget-mb", "auto",
                "--kv-cache-type", "q8_0",
                "--ngram", "--chat", "--warmup", "--reps", "2",
            ],
        ))

    ran = 0
    for name, model_path, extra in cases:
        if not model_path.is_file():
            print(f"Skipping {name}: missing {model_path}", file=sys.stderr)
            continue
        run_case(args.bench, args.out_dir, name, ["--model", str(model_path), *extra], args.cooldown)
        ran += 1

    print(f"\nCompleted {ran} benchmark case(s) -> {args.out_dir}", flush=True)

    if args.baseline is not None and ran > 0:
        checker = Path(__file__).resolve().parent / "check_regressions.py"
        res = subprocess.run(  # noqa: S603 — explicit argv list invoking repo-local checker
            [
                sys.executable,
                str(checker),
                "--baseline",
                str(args.baseline),
                "--candidate",
                str(args.out_dir),
                "--logs-dir",
                str(args.out_dir),
            ],
            check=False,
        )
        return res.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
