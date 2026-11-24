#!/usr/bin/env python3
"""Aggregate raw layerstream-plugin, stock llama.cpp, Hugging Face accelerate, and AirLLM
JSON benchmark outputs into a normalized JSON + JSONL + Parquet Hugging Face dataset bundle.

Usage:
    python3 benchmarks/export_hf_dataset.py \
        --results-dir benchmarks/results \
        --out-dir benchmarks/hf-dataset
"""

import argparse
import json
from pathlib import Path
from typing import Any


def extract_json_array(raw: str) -> list[dict[str, Any]]:
    """Extract the trailing JSON array from stdout that may include bracketed library banners."""
    pos = len(raw)
    while True:
        idx = raw.rfind("[", 0, pos)
        if idx == -1:
            return []
        candidate = raw[idx:].strip()
        try:
            data = json.loads(candidate)
            if isinstance(data, list) and (not data or isinstance(data[0], dict)):
                return data
        except json.JSONDecodeError:
            pass
        pos = idx


DEFAULT_HARDWARE = "NVIDIA Jetson Orin Nano (8 GiB unified LPDDR5, CUDA sm_87, dm-crypt NVMe)"
DEFAULT_HARDWARE_ID = "orin-nano-8gb-cuda"


def load_dir_hardware(round_dir: Path) -> tuple[str, str]:
    meta_path = round_dir / "hardware.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                hw = meta.get("hardware") or DEFAULT_HARDWARE
                hw_id = meta.get("hardware_id") or DEFAULT_HARDWARE_ID
                return str(hw), str(hw_id)
        except OSError, json.JSONDecodeError:
            pass
    return DEFAULT_HARDWARE, DEFAULT_HARDWARE_ID


def normalize_record(
    rec: dict[str, Any],
    source_file: str,
    round_name: str,
    default_hw: str = DEFAULT_HARDWARE,
    default_hw_id: str = DEFAULT_HARDWARE_ID,
) -> dict[str, Any]:
    engine = rec.get("engine")
    default_ver = "0.1.0" if engine == "layerstream-plugin" else None
    return {
        "round": round_name,
        "source_file": source_file,
        "hardware": rec.get("hardware") or default_hw,
        "hardware_id": rec.get("hardware_id") or default_hw_id,
        "engine": engine,
        "engine_version": rec.get("engine_version", default_ver),
        "git_rev": rec.get("git_rev"),
        "mode": rec.get("mode"),
        "status": rec.get("status", "ok"),
        "model": rec.get("model"),
        "draft_model": rec.get("draft_model"),
        "workload": rec.get("workload"),
        "chat_template_applied": rec.get("chat_template_applied"),
        "warmup": rec.get("warmup"),
        "reps": rec.get("reps"),
        "temperature": rec.get("temperature"),
        "seed": rec.get("seed"),
        "threads": rec.get("threads"),
        "kv_cache_type": rec.get("kv_cache_type"),
        "vram_budget_auto": rec.get("vram_budget_auto"),
        "vram_budget_mb": rec.get("vram_budget_mb"),
        "prompt_tokens": rec.get("prompt_tokens"),
        "completion_tokens": rec.get("completion_tokens"),
        "load_time_s": rec.get("load_time_s"),
        "ttft_s": rec.get("ttft_s"),
        "ttft_s_min": rec.get("ttft_s_min"),
        "ttft_s_max": rec.get("ttft_s_max"),
        "decode_time_s": rec.get("decode_time_s"),
        "total_time_s": rec.get("total_time_s"),
        "decode_tok_per_s": rec.get("decode_tok_per_s"),
        "decode_tok_per_s_min": rec.get("decode_tok_per_s_min"),
        "decode_tok_per_s_max": rec.get("decode_tok_per_s_max"),
        "total_tok_per_s": rec.get("total_tok_per_s"),
        "total_tok_per_s_min": rec.get("total_tok_per_s_min"),
        "total_tok_per_s_max": rec.get("total_tok_per_s_max"),
        "sweeps": rec.get("sweeps"),
        "drafted_tokens": rec.get("drafted_tokens"),
        "accepted_tokens": rec.get("accepted_tokens"),
        "resident_layers": rec.get("resident_layers"),
        "streamed_layers": rec.get("streamed_layers"),
        "n_layers": rec.get("n_layers"),
        "resident_mib": rec.get("resident_mib"),
        "slot_mib": rec.get("slot_mib"),
        "expert_slot_mib": rec.get("expert_slot_mib"),
        "io_mib_read": rec.get("io_mib_read"),
        "pread_mib_s": rec.get("pread_mib_s"),
        "effective_io_mib_s": rec.get("effective_io_mib_s"),
        "expert_cache_hits": rec.get("expert_cache_hits"),
        "expert_cache_misses": rec.get("expert_cache_misses"),
        "expert_cache_hit_rate": rec.get("expert_cache_hit_rate"),
        "expert_cache_mib_saved": rec.get("expert_cache_mib_saved"),
        "moe_layers": rec.get("moe_layers"),
        "moe_active_streaming": rec.get("moe_active_streaming"),
        "uma_zero_copy": rec.get("uma_zero_copy"),
        "is_unified_memory": rec.get("is_unified_memory"),
        "direct_io_active": rec.get("direct_io_active"),
        "peak_rss_mib": rec.get("peak_rss_mib"),
        "current_rss_mib": rec.get("current_rss_mib"),
        "mem_avail_before_mib": rec.get("mem_avail_before_mib"),
        "mem_avail_after_mib": rec.get("mem_avail_after_mib"),
        "swap_delta_mib": rec.get("swap_delta_mib"),
        "cuda_max_allocated_mib": rec.get("cuda_max_allocated_mib"),
        "page_cache_delta_mib": rec.get("page_cache_delta_mib"),
        "page_cache_delta_since_start_mib": rec.get("page_cache_delta_since_start_mib"),
        "idle_watts": rec.get("idle_watts"),
        "avg_watts": rec.get("avg_watts"),
        "net_watts": rec.get("net_watts"),
        "joules_per_tok": rec.get("joules_per_tok"),
        "net_joules_per_tok": rec.get("net_joules_per_tok"),
        "turn1_prompt_tokens": rec.get("turn1_prompt_tokens"),
        "turn1_ttft_s": rec.get("turn1_ttft_s"),
        "turn2_prompt_tokens": rec.get("turn2_prompt_tokens"),
        "turn2_ttft_s": rec.get("turn2_ttft_s"),
        "ttft_speedup_x": rec.get("ttft_speedup_x"),
        "notes": rec.get("notes"),
        "output_preview": rec.get("output_preview") or rec.get("turn2_output_preview"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for path in sorted(args.results_dir.rglob("*.json")):
        if path.name == "hardware.json" or path.stat().st_size == 0:
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            items = extract_json_array(raw)
        except json.JSONDecodeError as exc:
            print(f"Skipping malformed JSON {path}: {exc}")
            continue
        round_name = path.parent.name
        dir_hw, dir_hw_id = load_dir_hardware(path.parent)
        records.extend(
            normalize_record(item, path.name, round_name, dir_hw, dir_hw_id) for item in items
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    json_path = data_dir / "benchmarks.json"
    json_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} normalized records -> {json_path}")

    jsonl_path = data_dir / "benchmarks.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote JSONL -> {jsonl_path}")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(records)
        parquet_path = data_dir / "train-00000-of-00001.parquet"
        pq.write_table(table, parquet_path)
        print(f"Wrote Parquet -> {parquet_path}")
    except ImportError:
        print("pyarrow not installed; skipped Parquet export (JSON/JSONL written).")


if __name__ == "__main__":
    main()
