"""LayerStream Edge LLM Benchmarks & Live Hugging Face GGUF Fit Estimator.

Interactive Gradio 5 dashboard + native Model Context Protocol (MCP) server
backed by the bare-metal Jetson Orin Nano benchmark dataset
(`kleinbem/layerstream-orin-nano-benchmarks`).
"""

from __future__ import annotations

import json
import re
import struct
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATASET_REPO = "kleinbem/layerstream-orin-nano-benchmarks"
LOCAL_BENCHMARKS = Path(__file__).resolve().parent / "benchmarks.json"

KNOWN_QUANTS = (
    "UD-IQ1_S",
    "UD-IQ1_M",
    "UD-IQ2_XXS",
    "UD-IQ2_XS",
    "UD-IQ2_M",
    "UD-IQ3_XXS",
    "UD-Q2_K_XL",
    "UD-Q3_K_XL",
    "UD-Q4_K_XL",
    "UD-Q5_K_XL",
    "UD-Q6_K_XL",
    "UD-Q8_K_XL",
    "IQ1_S",
    "IQ1_M",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ2_M",
    "IQ3_XXS",
    "IQ3_XS",
    "IQ3_S",
    "IQ3_M",
    "IQ4_XS",
    "IQ4_NL",
    "Q2_K_XL",
    "Q2_K_L",
    "Q2_K_S",
    "Q2_K",
    "Q3_K_XL",
    "Q3_K_L",
    "Q3_K_M",
    "Q3_K_S",
    "Q4_K_XL",
    "Q4_K_L",
    "Q4_K_M",
    "Q4_K_S",
    "Q4_1",
    "Q4_0",
    "Q5_K_XL",
    "Q5_K_L",
    "Q5_K_M",
    "Q5_K_S",
    "Q5_1",
    "Q5_0",
    "Q6_K_XL",
    "Q6_K",
    "Q8_0",
    "BF16",
    "F16",
    "FP16",
)

SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})$")
ROUND_NUM_RE = re.compile(r"round(\d+)")


def _load_records() -> list[dict[str, Any]]:
    if LOCAL_BENCHMARKS.is_file():
        return json.loads(LOCAL_BENCHMARKS.read_text(encoding="utf-8"))
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename="data/benchmarks.json",
            repo_type="dataset",
        )
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except ImportError:
        return []
    except OSError:
        return []
    except ValueError:
        return []


RECORDS: list[dict[str, Any]] = _load_records()
DF = pd.DataFrame(RECORDS) if RECORDS else pd.DataFrame()
if not DF.empty and "round" in DF.columns:
    DF["round_num"] = (
        DF["round"]
        .fillna("")
        .astype(str)
        .map(lambda s: int(m.group(1)) if (m := ROUND_NUM_RE.search(s)) else 0)
    )


def _extract_quant(path: str) -> str:
    fname = path.rsplit("/", 1)[-1]
    stem = fname.removesuffix(".gguf")
    m = SHARD_RE.match(stem)
    clean = m.group(1) if m else stem
    upper = clean.upper()
    for q in KNOWN_QUANTS:
        if q in upper:
            return q
    return "GGUF"


def _probe_remote_gguf_header(repo_id: str, filename: str) -> dict[str, Any]:
    """Fetch only the first 256 KiB of a remote GGUF via HTTP Range and parse key metadata."""
    url = f"https://huggingface.co/{repo_id}/resolve/main/{urllib.parse.quote(filename, safe='/')}"
    req = urllib.request.Request(
        url,
        headers={
            "Range": "bytes=0-262143",
            "User-Agent": "layerstream-hf-space/0.6.3",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 — hardcoded https://huggingface.co URL
            buf = resp.read()
    except urllib.error.URLError:
        return {}
    except TimeoutError:
        return {}
    except OSError:
        return {}
    return _parse_gguf_header(buf)


def _parse_gguf_header(buf: bytes) -> dict[str, Any]:
    if len(buf) < 24 or buf[:4] != b"GGUF":
        return {}
    version, tensor_count, kv_count = struct.unpack_from("<IQQ", buf, 4)
    pos = 24
    meta: dict[str, Any] = {
        "gguf_version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
    }

    def read_u32() -> int | None:
        nonlocal pos
        if pos + 4 > len(buf):
            return None
        val = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        return int(val)

    def read_u64() -> int | None:
        nonlocal pos
        if pos + 8 > len(buf):
            return None
        val = struct.unpack_from("<Q", buf, pos)[0]
        pos += 8
        return int(val)

    def read_str() -> str | None:
        nonlocal pos
        n = read_u64()
        if n is None or n > 65536 or pos + n > len(buf):
            return None
        s = buf[pos : pos + n].decode("utf-8", errors="replace")
        pos += n
        return s

    scalar_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip_val(vtype: int) -> Any:
        nonlocal pos
        if vtype == 8:
            return read_str()
        if vtype == 4:
            return read_u32()
        if vtype == 10:
            return read_u64()
        if vtype in scalar_sizes:
            sz = scalar_sizes[vtype]
            if pos + sz > len(buf):
                return None
            pos += sz
            return True
        if vtype == 9:
            elem_type = read_u32()
            count = read_u64()
            if elem_type is None or count is None:
                return None
            if elem_type in scalar_sizes:
                total = scalar_sizes[elem_type] * count
                if pos + total > len(buf):
                    return None
                pos += total
                return True
            if count > 4096:
                return None
            for _ in range(count):
                if skip_val(elem_type) is None:
                    return None
            return True
        return None

    for _ in range(min(kv_count, 512)):
        key = read_str()
        if key is None:
            break
        vtype = read_u32()
        if vtype is None:
            break
        val = skip_val(vtype)
        if val is None:
            break
        if key == "general.architecture" and isinstance(val, str):
            meta["architecture"] = val
        elif key == "general.name" and isinstance(val, str):
            meta["name"] = val
        elif key.endswith(".block_count") and isinstance(val, int):
            meta["block_count"] = val
        elif key.endswith(".context_length") and isinstance(val, int):
            meta["context_length"] = val
        elif key.endswith(".expert_count") and isinstance(val, int):
            meta["expert_count"] = val
        elif key.endswith(".expert_used_count") and isinstance(val, int):
            meta["expert_used_count"] = val
        elif key.endswith(".nextn_predict_layers") and isinstance(val, int):
            meta["mtp_layers"] = val

    return meta


def estimate_hf_gguf_fit(
    repo_id: str,
    vram_gb: float = 8.0,
    nvme_mib_s: float = 2800.0,
    unified_memory: bool = True,
    quant_filter: str = "",
) -> str:
    """Inspect a public Hugging Face GGUF repo live and estimate LayerStream hardware fit.

    Fetches the repository file tree and probes the leading 256 KiB GGUF header over
    HTTP Range (zero multi-GB download) to detect architecture, layer count, MoE
    experts, and built-in MTP heads, then ranks every quant for the target GPU/NVMe.

    Args:
        repo_id: Hugging Face repository ID (e.g. 'bartowski/Qwen2.5-14B-Instruct-GGUF').
        vram_gb: Available GPU VRAM or unified RAM in GiB (e.g. 8.0 for Jetson Orin Nano).
        nvme_mib_s: Sustained NVMe sequential read throughput in MiB/s (e.g. 2800.0).
        unified_memory: True for unified-memory SoCs (Jetson, Apple Silicon, Strix Halo).
        quant_filter: Optional case-insensitive substring filter (e.g. 'Q4_K_M' or 'Q2_K').

    Returns:
        Markdown report with GGUF metadata, quant fit table, predicted tok/s, and CLI snippet.
    """
    clean_repo = (
        repo_id.strip()
        .removeprefix("hf://")
        .removeprefix("https://huggingface.co/")
        .removeprefix("http://huggingface.co/")
        .strip("/")
    )
    if clean_repo.count("/") != 1:
        return (
            "Please provide a valid Hugging Face repo ID in `org/repo` format "
            "(for example `bartowski/Qwen2.5-14B-Instruct-GGUF`)."
        )

    tree_url = f"https://huggingface.co/api/models/{clean_repo}/tree/main?recursive=true"
    req = urllib.request.Request(tree_url, headers={"User-Agent": "layerstream-hf-space/0.6.3"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:  # noqa: S310 — hardcoded https://huggingface.co API
            entries = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return f"Hugging Face API returned HTTP {exc.code} for `{clean_repo}`."
    except urllib.error.URLError as exc:
        return f"Failed to reach Hugging Face API for `{clean_repo}`: {exc.reason}"
    except TimeoutError:
        return f"Timed out querying Hugging Face API for `{clean_repo}`."
    except ValueError as exc:
        return f"Invalid JSON from Hugging Face API for `{clean_repo}`: {exc}"

    grouped: dict[str, dict[str, Any]] = {}
    for item in entries:
        if item.get("type") != "file":
            continue
        path = str(item.get("path", ""))
        if not path.lower().endswith(".gguf"):
            continue
        size = int((item.get("lfs") or {}).get("size") or item.get("size") or 0)
        stem = path[:-5]
        m = SHARD_RE.match(stem)
        if m:
            key = m.group(1)
            shard_idx = int(m.group(2))
            shard_total = int(m.group(3))
            g = grouped.setdefault(
                key,
                {
                    "primary": path,
                    "first_idx": shard_idx,
                    "bytes": 0,
                    "shards": shard_total,
                    "quant": _extract_quant(path),
                },
            )
            g["bytes"] += size
            if shard_idx < g["first_idx"]:
                g["first_idx"] = shard_idx
                g["primary"] = path
        else:
            grouped[path] = {
                "primary": path,
                "first_idx": 1,
                "bytes": size,
                "shards": 1,
                "quant": _extract_quant(path),
            }

    if not grouped:
        return f"No `.gguf` files found in `{clean_repo}`."

    qf = quant_filter.strip().upper()
    candidates = [
        c
        for c in grouped.values()
        if not qf or qf in c["quant"].upper() or qf in c["primary"].upper()
    ]
    if not candidates:
        return f"No `.gguf` files in `{clean_repo}` matched filter `{quant_filter}`."

    candidates.sort(key=lambda c: c["bytes"])
    usable_gib = max(1.0, vram_gb - (2.2 if unified_memory else 1.2))
    nvme_gib_s = max(0.2, nvme_mib_s / 1024.0)

    # Probe the first Q4_K_M candidate (or smallest candidate) via HTTP Range.
    probe_cand = next(
        (c for c in candidates if "Q4_K_M" in c["quant"]),
        candidates[0],
    )
    header = _probe_remote_gguf_header(clean_repo, probe_cand["primary"])
    is_moe = int(header.get("expert_count") or 0) > 0
    experts_total = int(header.get("expert_count") or 0)
    experts_used = int(header.get("expert_used_count") or 0)
    mtp_layers = int(header.get("mtp_layers") or 0)

    lines = [f"### Live GGUF Analysis: `{clean_repo}`"]
    if header:
        meta_bits = []
        if "architecture" in header:
            meta_bits.append(f"arch=`{header['architecture']}`")
        if "block_count" in header:
            meta_bits.append(f"layers=`{header['block_count']}`")
        if "context_length" in header:
            meta_bits.append(f"ctx=`{header['context_length']}`")
        if is_moe:
            meta_bits.append(f"MoE=`{experts_total}x` (top-{experts_used or '?'})")
        if mtp_layers > 0:
            meta_bits.append(f"MTP=`{mtp_layers}` (built-in self-speculation)")
        lines.append(
            f"**Header probe (`{probe_cand['primary']}`, 256 KiB Range):** " + " · ".join(meta_bits)
        )

    lines.append(
        f"**Target hardware:** `{vram_gb:.1f} GiB` "
        f"({'unified' if unified_memory else 'discrete'}) · "
        f"`~{usable_gib:.1f} GiB` usable layer budget · `{nvme_mib_s:.0f} MiB/s` NVMe\n"
    )
    lines.append(
        "| Quant | Size (GiB) | Shards | Fit Mode | Resident % | Est. Decode (`tok/s`) | Recommended |"
    )
    lines.append("|---|---:|---:|---|---:|---:|:---:|")

    best: dict[str, Any] | None = None
    best_score = -1e9
    rows_meta: list[tuple[dict[str, Any], str, float, float, float]] = []

    for c in candidates:
        size_gib = c["bytes"] / (1024**3)
        if size_gib <= usable_gib * 0.92:
            fit = "Full GPU Resident"
            res_pct = 100.0
            tps_low = max(12.0, (40.0 if unified_memory else 80.0) / max(size_gib, 0.5))
            tps_high = tps_low * 1.35
        else:
            res_pct = min(90.0, max(0.0, (usable_gib / max(size_gib, 0.1)) * 75.0))
            stream_gib = size_gib * (1.0 - res_pct / 100.0)
            if is_moe and experts_total > 0 and experts_used > 0:
                fit = "Sparse MoE + TinyLFU"
                active_frac = min(1.0, (experts_used / experts_total) * 1.35)
                eff_stream = max(0.08, stream_gib * active_frac * 0.45)
                tps_low = min(18.0, nvme_gib_s / eff_stream * 0.65)
                tps_high = min(22.0, nvme_gib_s / eff_stream * 1.15)
            else:
                fit = "Hybrid NVMe Stream + Draft"
                base_tps = nvme_gib_s / max(stream_gib, 0.15)
                tps_low = base_tps * 1.4
                tps_high = base_tps * 3.8

        # Score favors Q4_K_M / Q4_K_S / Q3_K_M quality while rewarding usable throughput.
        q_bonus = 25.0 if "Q4_K_M" in c["quant"] else (18.0 if "Q4_" in c["quant"] else 10.0)
        if "Q2_K" in c["quant"] and is_moe:
            q_bonus = 22.0
        score = q_bonus + min(tps_high, 15.0) * 1.5 - max(0.0, size_gib - usable_gib * 3.0) * 4.0
        rows_meta.append((c, fit, res_pct, tps_low, tps_high))
        if score > best_score:
            best_score = score
            best = c

    for c, fit, res_pct, tps_low, tps_high in rows_meta:
        size_gib = c["bytes"] / (1024**3)
        star = "⭐" if c is best else ""
        lines.append(
            f"| `{c['quant']}` | {size_gib:.2f} | {c['shards']} | {fit} | "
            f"{res_pct:.0f}% | `{tps_low:.2f} - {tps_high:.2f}` | {star} |"
        )

    if best is not None:
        hf_uri = f"hf://{clean_repo}:{best['primary']}"
        lines.append("\n#### Quick Start (`layerstream-plugin v0.6.3`)")
        lines.append("```bash")
        lines.append("pip install layerstream layerstream-cuda13")
        lines.append(f"layerstream serve --model {hf_uri} --vram-budget-mb auto")
        lines.append("```")
        lines.append("```python")
        lines.append("from layerstream import AutoModel")
        lines.append(f'with AutoModel.from_pretrained("{hf_uri}") as model:')
        lines.append('    print(model.chat([{"role": "user", "content": "Hello!"}]))')
        lines.append("```")

    return "\n".join(lines)


def recommend_edge_model(vram_gb: float = 8.0, use_case: str = "coding") -> str:
    """Recommend the best GGUF model, quant, and draft speculator for an edge GPU.

    Backed by bare-metal Jetson Orin Nano Round 12 (`v0.6.3`) measurements.

    Args:
        vram_gb: Total GPU VRAM or unified memory in GiB (e.g. 4, 8, 12, 16, 24).
        use_case: One of 'coding', 'chat', 'reasoning', or 'low_power'.

    Returns:
        Markdown recommendation with measured Round 12 Orin Nano baselines and CLI command.
    """
    uc = use_case.strip().lower()
    if vram_gb <= 6.0 or uc == "low_power":
        rec_model = "allenai/OLMoE-1B-7B-0125-Instruct-GGUF:OLMoE-1B-7B-0125-Instruct-Q2_K.gguf"
        draft = None
        why = (
            "64-expert sparse MoE (`2.5 GiB`) with TinyLFU hot-expert cache (`100%` hit rate). "
            "Measured on 8 GB Orin Nano (R12 `v0.6.3`): **`15.03-15.08 tok/s`** at **`1.04-1.33 J/tok`**, "
            "`3.1-3.5 GiB` Peak RSS, `0 MiB` swap."
        )
    elif vram_gb <= 10.0 and uc in {"coding", "chat"}:
        rec_model = "Qwen/Qwen2.5-14B-Instruct-GGUF:qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
        draft = "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf"
        why = (
            "8.37 GiB dense 14B (`Q4_K_M`) streamed over NVMe with `0.5B` speculative draft. "
            "Measured on 8 GB Orin Nano (R12 `v0.6.3`, `b5400`): **`4.873 tok/s`** `code_edit` (`3.49 J/tok`), "
            "**`1.702 tok/s`** `open_ended`, **`0.987 tok/s`** `prompt_heavy`, `0 MiB` swap."
        )
    elif vram_gb <= 10.0 and uc == "reasoning":
        rec_model = "unsloth/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf"
        draft = "Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf"
        why = (
            "17.3 GiB 128-expert MoE (`Q4_K_M`) with parallel sparse expert streaming + 2 GiB Edge expert cache. "
            "Measured on 8 GB Orin Nano (R12 `v0.6.3`): **`1.662 tok/s`** `open_ended` (`9.09 J/tok`), "
            "**`0.864 tok/s`** `code_edit`, `0 MiB` swap."
        )
    else:
        rec_model = "Qwen/Qwen2.5-32B-Instruct-GGUF:qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf"
        draft = "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf"
        why = (
            "18.49 GiB dense 32B (`Q4_K_M`) — runs even on 8 GB Orin Nano at **`1.588 tok/s`** `code_edit` "
            "(R12 `v0.6.3`) and scales linearly with higher VRAM residency on 12-24 GB GPUs."
        )

    cmd = f"layerstream serve --model hf://{rec_model} --vram-budget-mb auto"
    if draft:
        cmd += f" --draft-model hf://{draft}"

    return (
        f"### Recommended Setup (`{vram_gb:.0f} GiB` · `{use_case}`)\n\n"
        f"- **Target model:** `hf://{rec_model}`\n"
        + (f"- **Draft model:** `hf://{draft}`\n" if draft else "- **Draft model:** None needed\n")
        + f"- **Why:** {why}\n\n"
        f"```bash\n{cmd}\n```"
    )


def query_benchmarks(
    model_contains: str = "",
    engine: str = "all",
    workload: str = "all",
    round_name: str = "all",
    limit: int = 25,
) -> str:
    """Query the 185-run bare-metal Jetson Orin Nano benchmark dataset.

    Args:
        model_contains: Case-insensitive substring match on model filename (e.g. '14B', 'OLMoE', '32B').
        engine: Engine filter ('all', 'layerstream-plugin', 'llama.cpp', 'colibri', 'airllm', 'accelerate').
        workload: Workload filter ('all', 'code_edit', 'open_ended', 'prompt_heavy', 'multi_turn_prefix_reuse').
        round_name: Round filter ('all' or e.g. 'round12_v063', 'round11_v061').
        limit: Maximum rows to return (1..100).

    Returns:
        Markdown table of matching benchmark rows sorted by decode_tok_per_s descending.
    """
    if DF.empty:
        return "Benchmark dataset is empty."

    sub = _filter_df(model_contains, engine, workload, round_name)
    if sub.empty:
        return "No benchmark rows matched the requested filters."

    sub = sub.sort_values(by="decode_tok_per_s", ascending=False, na_position="last").head(
        max(1, min(int(limit), 100))
    )
    lines = [
        f"Found **{len(sub)}** matching rows (showing top {len(sub)} by decode `tok/s`):\n",
        "| Round | Engine | Model | Workload | Decode (`tok/s`) | TTFT (s) | Peak RSS (MiB) | Swap Δ | `J/tok` |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in sub.iterrows():
        dec = f"{r['decode_tok_per_s']:.3f}" if pd.notna(r.get("decode_tok_per_s")) else "—"
        ttft = f"{r['ttft_s']:.2f}" if pd.notna(r.get("ttft_s")) else "—"
        rss = f"{r['peak_rss_mib']:.0f}" if pd.notna(r.get("peak_rss_mib")) else "—"
        swp = f"{r['swap_delta_mib']:+.1f}" if pd.notna(r.get("swap_delta_mib")) else "—"
        jpt = f"{r['joules_per_tok']:.2f}" if pd.notna(r.get("joules_per_tok")) else "—"
        lines.append(
            f"| `{r.get('round', '')}` | `{r.get('engine', '')}` | `{r.get('model', '')}` | "
            f"`{r.get('workload', '')}` | **{dec}** | {ttft} | {rss} | {swp} | {jpt} |"
        )
    return "\n".join(lines)


def compare_round_progression(
    model_contains: str = "Qwen2.5-14B",
    workload: str = "code_edit",
) -> str:
    """Compare best layerstream-plugin decode throughput and energy across rounds R2..R12.

    Args:
        model_contains: Substring matching the model (e.g. 'Qwen2.5-14B', 'OLMoE', 'Qwen2.5-32B', 'Qwen3-30B').
        workload: Workload name ('code_edit', 'open_ended', or 'prompt_heavy').

    Returns:
        Markdown table showing peak throughput per round from earliest to Round 12 (v0.6.3).
    """
    if DF.empty:
        return "Benchmark dataset is empty."
    sub = _filter_df(model_contains, "layerstream-plugin", workload, "all")
    sub = sub[sub["decode_tok_per_s"].notna()]
    if sub.empty:
        return (
            f"No `layerstream-plugin` rows for model `{model_contains}` and workload `{workload}`."
        )

    best_per_round = (
        sub.sort_values("decode_tok_per_s", ascending=False)
        .groupby("round", as_index=False)
        .first()
        .sort_values("round_num")
    )
    lines = [
        f"### Round-over-Round Progression: `{model_contains}` (`{workload}`)\n",
        "| Round | Version | Mode | Decode (`tok/s`) | Peak RSS (MiB) | `J/tok` |",
        "|---|---|---|---:|---:|---:|",
    ]
    for _, r in best_per_round.iterrows():
        rss = f"{r['peak_rss_mib']:.0f}" if pd.notna(r.get("peak_rss_mib")) else "—"
        jpt = f"{r['joules_per_tok']:.2f}" if pd.notna(r.get("joules_per_tok")) else "—"
        ver = r.get("engine_version") or "—"
        lines.append(
            f"| `{r['round']}` | `{ver}` | `{r.get('mode', '')}` | "
            f"**{r['decode_tok_per_s']:.3f}** | {rss} | {jpt} |"
        )
    return "\n".join(lines)


def _filter_df(
    model_contains: str = "",
    engine: str = "all",
    workload: str = "all",
    round_name: str = "all",
) -> pd.DataFrame:
    if DF.empty:
        return DF
    sub = DF
    if model_contains.strip():
        needle = model_contains.strip().lower()
        sub = sub[sub["model"].fillna("").astype(str).str.lower().str.contains(needle, regex=False)]
    if engine and engine != "all":
        sub = sub[sub["engine"].fillna("").astype(str) == engine]
    if workload and workload != "all":
        sub = sub[sub["workload"].fillna("").astype(str) == workload]
    if round_name and round_name != "all":
        sub = sub[sub["round"].fillna("").astype(str) == round_name]
    return sub


def _build_pareto_charts(
    model_contains: str, engine: str, workload: str, round_name: str
) -> tuple[go.Figure, go.Figure, go.Figure]:
    sub = _filter_df(model_contains, engine, workload, round_name)
    valid = sub[sub["decode_tok_per_s"].notna()] if not sub.empty else sub

    if valid.empty:
        empty = go.Figure()
        empty.update_layout(title="No matching numeric benchmark rows")
        return empty, empty, empty

    fig_rss = px.scatter(
        valid,
        x="peak_rss_mib",
        y="decode_tok_per_s",
        color="engine",
        symbol="workload",
        hover_data=["round", "model", "mode", "ttft_s", "swap_delta_mib", "joules_per_tok"],
        title="Throughput (tok/s) vs. Peak RSS (MiB)",
        labels={"peak_rss_mib": "Peak RSS (MiB)", "decode_tok_per_s": "Decode (tok/s)"},
    )
    fig_rss.update_traces(marker={"size": 10})

    energy_df = valid[valid["joules_per_tok"].notna()]
    if energy_df.empty:
        fig_energy = go.Figure()
        fig_energy.update_layout(title="No INA3221 energy rows in current filter")
    else:
        fig_energy = px.scatter(
            energy_df,
            x="joules_per_tok",
            y="decode_tok_per_s",
            color="model",
            symbol="engine",
            hover_data=["round", "mode", "workload", "avg_watts", "peak_rss_mib"],
            title="Energy Efficiency: Throughput (tok/s) vs. Gross Joules/Token (lower J/tok is better)",
            labels={"joules_per_tok": "Gross Energy (J/tok)", "decode_tok_per_s": "Decode (tok/s)"},
        )
        fig_energy.update_traces(marker={"size": 10})

    ls_df = valid[valid["engine"] == "layerstream-plugin"]
    if ls_df.empty:
        fig_prog = go.Figure()
        fig_prog.update_layout(title="Select layerstream-plugin to view round progression")
    else:
        prog = (
            ls_df.sort_values("decode_tok_per_s", ascending=False)
            .groupby(["round_num", "round", "model"], as_index=False)
            .first()
            .sort_values("round_num")
        )
        fig_prog = px.line(
            prog,
            x="round",
            y="decode_tok_per_s",
            color="model",
            markers=True,
            hover_data=["mode", "engine_version", "peak_rss_mib", "joules_per_tok"],
            title="Round-over-Round Peak Decode Throughput (R2 → R12 v0.6.3)",
            labels={"round": "Benchmark Round", "decode_tok_per_s": "Peak Decode (tok/s)"},
        )

    return fig_rss, fig_energy, fig_prog


def _explorer_table(
    model_contains: str, engine: str, workload: str, round_name: str
) -> pd.DataFrame:
    sub = _filter_df(model_contains, engine, workload, round_name)
    cols = [
        "round",
        "engine",
        "engine_version",
        "model",
        "workload",
        "mode",
        "decode_tok_per_s",
        "ttft_s",
        "peak_rss_mib",
        "swap_delta_mib",
        "joules_per_tok",
    ]
    present = [c for c in cols if c in sub.columns]
    if sub.empty:
        return pd.DataFrame(columns=present)
    return sub[present].sort_values(by="decode_tok_per_s", ascending=False, na_position="last")


def build_app() -> gr.Blocks:
    engines = ["all"] + (
        sorted(DF["engine"].dropna().astype(str).unique().tolist()) if not DF.empty else []
    )
    workloads = ["all"] + (
        sorted(DF["workload"].dropna().astype(str).unique().tolist()) if not DF.empty else []
    )
    rounds = ["all"] + (
        sorted(DF["round"].dropna().astype(str).unique().tolist(), reverse=True)
        if not DF.empty
        else []
    )

    with gr.Blocks(
        title="LayerStream Edge LLM Benchmarks & GGUF Fit MCP", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            f"# ⚡ LayerStream Edge LLM Benchmarks & Live HF GGUF Fit Estimator\n"
            f"Backed by **{len(DF)} bare-metal runs** on an **8 GB NVIDIA Jetson Orin Nano** "
            f"([`{DATASET_REPO}`](https://huggingface.co/datasets/{DATASET_REPO})) · "
            f"**Round 12 (`v0.6.3`) Records:** `OLMoE-1B-7B` **15.08 tok/s** (`1.04 J/tok`), "
            f"`Qwen2.5-14B` **4.87 tok/s** (`3.49 J/tok`), `Qwen3-30B-A3B` **1.66 tok/s**, "
            f"`Qwen2.5-32B` **1.59 tok/s** (`0 MiB` swap) · **Native MCP Server enabled**."
        )

        with gr.Tabs():
            with gr.Tab("1. Pareto & Round Progression"):
                with gr.Row():
                    f_model = gr.Textbox(
                        label="Model Filter", placeholder="e.g. 14B, OLMoE, 32B, Qwen3"
                    )
                    f_engine = gr.Dropdown(choices=engines, value="all", label="Engine")
                    f_workload = gr.Dropdown(choices=workloads, value="all", label="Workload")
                    f_round = gr.Dropdown(choices=rounds, value="all", label="Round")
                p_rss = gr.Plot(label="Throughput vs. Peak RSS")
                with gr.Row():
                    p_energy = gr.Plot(label="Energy Efficiency (J/tok)")
                    p_prog = gr.Plot(label="Round Progression (R2 → R12)")
                for ctrl in (f_model, f_engine, f_workload, f_round):
                    ctrl.change(
                        _build_pareto_charts,
                        inputs=[f_model, f_engine, f_workload, f_round],
                        outputs=[p_rss, p_energy, p_prog],
                    )
                demo.load(
                    _build_pareto_charts,
                    inputs=[f_model, f_engine, f_workload, f_round],
                    outputs=[p_rss, p_energy, p_prog],
                )

            with gr.Tab("2. Live HF GGUF Fit Estimator (MCP Tool)"):
                gr.Markdown(
                    "Paste **any public Hugging Face GGUF repository** — inspects the Hub tree and "
                    "probes the first 256 KiB GGUF header via HTTP Range (zero multi-GB download)."
                )
                with gr.Row():
                    g_repo = gr.Textbox(
                        label="Hugging Face GGUF Repo",
                        value="Qwen/Qwen2.5-14B-Instruct-GGUF",
                    )
                    g_vram = gr.Slider(4, 64, value=8, step=1, label="GPU VRAM / Unified RAM (GiB)")
                    g_nvme = gr.Slider(
                        500, 7000, value=2800, step=100, label="NVMe Read BW (MiB/s)"
                    )
                with gr.Row():
                    g_unified = gr.Checkbox(
                        value=True, label="Unified Memory (Jetson / Apple Silicon)"
                    )
                    g_qfilter = gr.Textbox(
                        label="Quant Filter (optional)", placeholder="e.g. Q4_K_M"
                    )
                    g_btn = gr.Button("Inspect & Estimate", variant="primary")
                gr.Examples(
                    examples=[
                        ["Qwen/Qwen2.5-14B-Instruct-GGUF", 8, 2800, True, ""],
                        ["allenai/OLMoE-1B-7B-0125-Instruct-GGUF", 8, 2800, True, ""],
                        ["unsloth/Qwen3-30B-A3B-GGUF", 8, 2800, True, "Q4_K"],
                        ["Qwen/Qwen2.5-32B-Instruct-GGUF", 12, 3500, False, "Q4_K"],
                    ],
                    inputs=[g_repo, g_vram, g_nvme, g_unified, g_qfilter],
                )
                g_out = gr.Markdown()
                g_btn.click(
                    estimate_hf_gguf_fit,
                    inputs=[g_repo, g_vram, g_nvme, g_unified, g_qfilter],
                    outputs=g_out,
                    api_name="estimate_hf_gguf_fit",
                )

            with gr.Tab("3. Edge Model Recommender (MCP Tool)"):
                with gr.Row():
                    r_vram = gr.Slider(
                        4, 32, value=8, step=2, label="Target VRAM / Unified RAM (GiB)"
                    )
                    r_use = gr.Radio(
                        choices=["coding", "chat", "reasoning", "low_power"],
                        value="coding",
                        label="Primary Use Case",
                    )
                    r_btn = gr.Button("Recommend", variant="primary")
                r_out = gr.Markdown(value=recommend_edge_model(8.0, "coding"))
                r_btn.click(
                    recommend_edge_model,
                    inputs=[r_vram, r_use],
                    outputs=r_out,
                    api_name="recommend_edge_model",
                )

            with gr.Tab("4. Data Explorer, Data Studio SQL & MCP Setup"):
                with gr.Row():
                    q_model = gr.Textbox(label="Model Filter", placeholder="e.g. 14B")
                    q_engine = gr.Dropdown(choices=engines, value="all", label="Engine")
                    q_work = gr.Dropdown(choices=workloads, value="all", label="Workload")
                    q_round = gr.Dropdown(choices=rounds, value="all", label="Round")
                    q_limit = gr.Slider(5, 100, value=25, step=5, label="Limit")
                q_btn = gr.Button("Run Query (MCP Tool)", variant="primary")
                q_md = gr.Markdown()
                q_btn.click(
                    query_benchmarks,
                    inputs=[q_model, q_engine, q_work, q_round, q_limit],
                    outputs=q_md,
                    api_name="query_benchmarks",
                )

                with gr.Row():
                    p_m = gr.Textbox(label="Progression Model", value="Qwen2.5-14B")
                    p_w = gr.Dropdown(
                        choices=["code_edit", "open_ended", "prompt_heavy"],
                        value="code_edit",
                        label="Workload",
                    )
                    p_btn = gr.Button("Compare Rounds (MCP Tool)")
                p_md = gr.Markdown()
                p_btn.click(
                    compare_round_progression,
                    inputs=[p_m, p_w],
                    outputs=p_md,
                    api_name="compare_round_progression",
                )

                gr.DataFrame(
                    value=_explorer_table("", "all", "all", "round12_v063"),
                    label="Round 12 (v0.6.3) Snapshot",
                )
                gr.Markdown(
                    """### Connect Your AI Agent via MCP
Add this to Claude Code, Cursor, VS Code, or Zed:
```json
{
  "mcpServers": {
    "layerstream-benchmarks": {
      "url": "https://kleinbem-layerstream-benchmarks.hf.space/gradio_api/mcp/sse"
    }
  }
}
```

### Run Live SQL in Hugging Face Data Studio / DuckDB
```sql
SELECT round, engine, model, workload,
       ROUND(decode_tok_per_s, 3) AS tok_s,
       ROUND(peak_rss_mib, 0) AS rss_mib,
       ROUND(joules_per_tok, 2) AS j_per_tok
FROM 'hf://datasets/kleinbem/layerstream-orin-nano-benchmarks/data/train-00000-of-00001.parquet'
WHERE round = 'round12_v063'
ORDER BY decode_tok_per_s DESC;
```"""
                )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(mcp_server=True)
