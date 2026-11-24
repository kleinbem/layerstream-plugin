#!/usr/bin/env python3
"""Generate verified Markdown benchmark tables from normalized benchmarks.json."""

import argparse
import json
from pathlib import Path
from typing import Any


def fmt(val: Any, suffix: str = "", digits: int = 2) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{digits}f}{suffix}"
    return f"{val}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = json.loads(args.input.read_text(encoding="utf-8"))
    print(f"Loaded {len(rows)} benchmark records.\n")

    print("## Full Normalized Table\n")
    print(
        "| Round | Engine | Mode | Model | Workload | R | Load (s) | TTFT (s) | "
        "Decode (tok/s) | Total (tok/s) | Sweeps | Acc/Draft | Peak RSS (MiB) | "
        "Cache Δ (MiB) | Watts | J/tok |"
    )
    print("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        acc_draft = (
            f"{r['accepted_tokens']}/{r['drafted_tokens']}"
            if r.get("drafted_tokens") is not None
            else "—"
        )
        print(
            f"| {r.get('round', '—')} "
            f"| {r.get('engine', '—')} "
            f"| `{r.get('mode', '—')}` "
            f"| `{r.get('model', '—')}` "
            f"| `{r.get('workload', '—')}` "
            f"| {fmt(r.get('resident_layers'), digits=0)} "
            f"| {fmt(r.get('load_time_s'), digits=2)} "
            f"| {fmt(r.get('ttft_s'), digits=2)} "
            f"| **{fmt(r.get('decode_tok_per_s'), digits=3)}** "
            f"| {fmt(r.get('total_tok_per_s'), digits=3)} "
            f"| {fmt(r.get('sweeps'), digits=0)} "
            f"| {acc_draft} "
            f"| {fmt(r.get('peak_rss_mib'), digits=0)} "
            f"| {fmt(r.get('page_cache_delta_mib'), digits=1)} "
            f"| {fmt(r.get('avg_watts'), digits=2)} "
            f"| {fmt(r.get('joules_per_tok'), digits=2)} |"
        )


if __name__ == "__main__":
    main()
