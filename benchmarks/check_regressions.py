#!/usr/bin/env python3
"""Compare a candidate LayerStream benchmark round against a baseline round and
scan companion `.log` files for silent GPU/NVMe regressions.

Catches issues that CPU-only CI cannot see:
- Silent CPU layer fallback (`assigned to device CPU` when GPU was requested)
- Excessive ggml graph splits (`splits > 4`)
- NVMe/PCIe timeouts (`nvme.*timeout`, `I/O tag.*timeout`)
- Decode throughput drops, TTFT spikes, Peak RSS balloons, and swap thrashing

Usage:
    python3 benchmarks/check_regressions.py \
        --baseline benchmarks/results/round11_v061 \
        --candidate benchmarks/results/round12_v062 \
        --logs-dir /mnt/models/bench/results/round12_v062
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROUND_PREFIX_RE = re.compile(r"^r\d+_")
SPLITS_RE = re.compile(r"sched_reserve:\s+graph:.*?\bsplits\s*=\s*(\d+)")
NVME_TIMEOUT_RE = re.compile(r"(nvme\d+.*timeout|I/O tag.*timeout)", re.IGNORECASE)


def canonical_mode(mode: str) -> str:
    return ROUND_PREFIX_RE.sub("", mode)


def load_round_records(round_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(round_dir.glob("*.json")):
        if path.name == "hardware.json" or path.stat().st_size == 0:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for rec in data:
            if not isinstance(rec, dict):
                continue
            model = str(rec.get("model") or "unknown")
            mode = canonical_mode(str(rec.get("mode") or ""))
            workload = str(rec.get("workload") or "")
            records[(model, mode, workload)] = rec
    return records


def scan_logs(logs_dir: Path, max_splits: int) -> list[str]:
    failures: list[str] = []
    for log_path in sorted(logs_dir.glob("*.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        cpu_lines = [line for line in text.splitlines() if "assigned to device CPU" in line]
        if cpu_lines:
            failures.append(
                f"FAIL [log:{log_path.name}] {len(cpu_lines)} layer(s) silently assigned to CPU "
                f"(first: {cpu_lines[0].strip()})"
            )
        for match in SPLITS_RE.finditer(text):
            splits = int(match.group(1))
            if splits > max_splits:
                failures.append(
                    f"FAIL [log:{log_path.name}] ggml graph splits={splits} exceeds max {max_splits}"
                )
                break
        if NVME_TIMEOUT_RE.search(text):
            failures.append(f"FAIL [log:{log_path.name}] NVMe/PCIe I/O timeout detected in log")
    return failures


def check_pair(
    key: tuple[str, str, str],
    base: dict[str, Any] | None,
    cand: dict[str, Any],
    max_tps_drop_pct: float,
    max_ttft_rise_pct: float,
    max_rss_jump_mib: float,
    max_swap_mib: float,
) -> tuple[str, list[str], list[str]]:
    model, mode, workload = key
    warns: list[str] = []
    fails: list[str] = []

    if workload == "multi_turn_prefix_reuse":
        b_t2 = base.get("turn2_ttft_s") if base else None
        c_t2 = cand.get("turn2_ttft_s")
        spd = cand.get("ttft_speedup_x")
        row = (
            f"| `{model}` | `{mode}` | `multi_turn` | "
            f"{b_t2 if b_t2 is not None else '—'} s | {c_t2} s | {spd}× TTFT | — | — |"
        )
        return row, warns, fails

    b_tps = base.get("decode_tok_per_s") if base else None
    c_tps = cand.get("decode_tok_per_s")
    b_ttft = base.get("ttft_s") if base else None
    c_ttft = cand.get("ttft_s")
    b_rss = base.get("peak_rss_mib") if base else None
    c_rss = cand.get("peak_rss_mib")
    c_swap = cand.get("swap_delta_mib") or 0.0
    c_j = cand.get("joules_per_tok")

    delta_str = "new"
    if isinstance(b_tps, (int, float)) and isinstance(c_tps, (int, float)) and b_tps > 0:
        pct = (c_tps / b_tps - 1.0) * 100.0
        delta_str = f"{pct:+.1f}%"
        if pct < -max_tps_drop_pct:
            warns.append(
                f"WARN [{model} {workload}] decode tok/s dropped {pct:+.1f}% ({b_tps:.3f} -> {c_tps:.3f})"
            )

    if isinstance(b_ttft, (int, float)) and isinstance(c_ttft, (int, float)) and b_ttft > 0:
        ttft_pct = (c_ttft / b_ttft - 1.0) * 100.0
        if ttft_pct > max_ttft_rise_pct:
            warns.append(
                f"WARN [{model} {workload}] TTFT rose {ttft_pct:+.1f}% ({b_ttft:.2f}s -> {c_ttft:.2f}s)"
            )

    if isinstance(b_rss, (int, float)) and isinstance(c_rss, (int, float)):
        rss_delta = c_rss - b_rss
        if rss_delta > max_rss_jump_mib:
            fails.append(
                f"FAIL [{model} {workload}] peak RSS jumped +{rss_delta:.0f} MiB "
                f"({b_rss:.0f} -> {c_rss:.0f} MiB)"
            )

    if isinstance(c_swap, (int, float)) and c_swap > max_swap_mib:
        fails.append(f"FAIL [{model} {workload}] swap delta +{c_swap:.1f} MiB exceeds {max_swap_mib} MiB")

    if cand.get("using_gpu") is True and str(cand.get("backend_name") or "").upper() == "CPU":
        fails.append(f"FAIL [{model} {workload}] GPU requested but backend_name is CPU")

    row = (
        f"| `{model}` | `{mode}` | `{workload}` | "
        f"{f'{b_tps:.3f}' if isinstance(b_tps, (int, float)) else '—'} | "
        f"{f'{c_tps:.3f}' if isinstance(c_tps, (int, float)) else '—'} | "
        f"{delta_str} | "
        f"{f'{c_rss:.0f} MiB' if isinstance(c_rss, (int, float)) else '—'} | "
        f"{f'{c_j:.2f}' if isinstance(c_j, (int, float)) else '—'} |"
    )
    return row, warns, fails


def main() -> int:
    ap = argparse.ArgumentParser(description="Check LayerStream benchmark round for regressions")
    ap.add_argument("--baseline", type=Path, required=True, help="Baseline round directory")
    ap.add_argument("--candidate", type=Path, required=True, help="Candidate round directory")
    ap.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Optional directory containing companion *.log files (defaults to --candidate)",
    )
    ap.add_argument("--max-tps-drop-pct", type=float, default=10.0)
    ap.add_argument("--max-ttft-rise-pct", type=float, default=15.0)
    ap.add_argument("--max-rss-jump-mib", type=float, default=350.0)
    ap.add_argument("--max-swap-mib", type=float, default=32.0)
    ap.add_argument("--max-splits", type=int, default=4)
    ap.add_argument("--strict", action="store_true", help="Exit non-zero on WARN as well as FAIL")
    args = ap.parse_args()

    base_map = load_round_records(args.baseline)
    cand_map = load_round_records(args.candidate)
    if not cand_map:
        print(f"No benchmark JSON records found in {args.candidate}", file=sys.stderr)
        return 2

    print("| Model | Mode | Workload | Baseline `tok/s` | Candidate `tok/s` | Delta | Peak RSS | `J/tok` |")
    print("|---|---|---|---:|---:|---:|---:|---:|")

    all_warns: list[str] = []
    all_fails: list[str] = []
    for key, cand in sorted(cand_map.items()):
        row, warns, fails = check_pair(
            key,
            base_map.get(key),
            cand,
            args.max_tps_drop_pct,
            args.max_ttft_rise_pct,
            args.max_rss_jump_mib,
            args.max_swap_mib,
        )
        print(row)
        all_warns.extend(warns)
        all_fails.extend(fails)

    logs_dir = args.logs_dir or args.candidate
    if logs_dir.is_dir():
        all_fails.extend(scan_logs(logs_dir, args.max_splits))

    print()
    if all_warns:
        print("### Warnings")
        for w in all_warns:
            print(f"- {w}")
    if all_fails:
        print("### Failures")
        for f in all_fails:
            print(f"- {f}")

    if all_fails:
        return 1
    if args.strict and all_warns:
        return 1
    print(f"PASS: {len(cand_map)} records verified against {args.baseline.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
