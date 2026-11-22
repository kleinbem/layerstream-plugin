#!/usr/bin/env python3
"""Minimal streaming chat demo for layerstream's native Python bindings.

Examples:
    # 3-line Hugging Face base-repo auto-resolution:
    python3 examples/quickstart.py

    # Local GGUF with forced out-of-core streaming + optional draft speculator:
    python3 examples/quickstart.py \
        --model /path/to/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
        --draft-model /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
        --vram-budget-mb 2048
"""

from __future__ import annotations

import argparse
import time

from layerstream import AutoModel, probe_gpu_memory


def parse_budget(raw: str) -> str | int:
    if raw.strip().lower() == "auto":
        return "auto"
    return int(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="layerstream Python quickstart demo")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Local .gguf path, hf://org/repo:file.gguf URI, HF GGUF repo, or base HF repo",
    )
    parser.add_argument("--draft-model", default=None, help="Optional draft GGUF for speculation")
    parser.add_argument("--quant", default=None, help="Optional quant override (e.g. Q4_K_M)")
    parser.add_argument("--cache-dir", default=None, help="Optional model cache directory")
    parser.add_argument(
        "--vram-budget-mb",
        default="auto",
        help="'auto', 0 (pure R=0 streaming), or integer MiB budget",
    )
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--verbose", action="store_true", help="Show llama.cpp loader logs")
    parser.add_argument(
        "--prompt",
        default="In 3 concise bullet points, explain why O_DIRECT helps NVMe layer streaming for LLMs.",
    )
    args = parser.parse_args()

    mem = probe_gpu_memory(not args.cpu)
    if mem is not None:
        print(f"GPU memory: {mem[0] / (1024**3):.1f} GiB free / {mem[1] / (1024**3):.1f} GiB total")

    t0 = time.perf_counter()
    with AutoModel.from_pretrained(
        args.model,
        draft_model=args.draft_model,
        quant=args.quant,
        cache_dir=args.cache_dir,
        vram_budget_mb=parse_budget(args.vram_budget_mb),
        ctx_size=args.ctx_size,
        use_gpu=not args.cpu,
        verbose=args.verbose,
    ) as model:
        load_s = time.perf_counter() - t0
        st0 = model.stats()
        print(
            f"Loaded in {load_s:.2f}s | {st0['resident_layers']}/{st0['n_layers']} resident, "
            f"{st0['streamed_layers']} streamed, gpu={st0['using_gpu']}, "
            f"moe_layers={st0['moe_layers']}\n"
        )

        t_gen = time.perf_counter()
        ttft = None
        for chunk in model.chat(
            [{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=True,
        ):
            if ttft is None:
                ttft = time.perf_counter() - t_gen
            print(chunk, end="", flush=True)
        total_s = time.perf_counter() - t_gen
        print()

        summary = model.last_summary or {}
        st = model.stats()
        io = st["io"]
        toks = summary.get("completion_tokens", 0)
        mib_s = (io["bytes_read"] / (1024 * 1024)) / max(io["pread_us"] / 1e6, 1e-6)
        spec = summary.get("speculation", {})
        print(
            f"\nStats: {toks} tok | TTFT={(ttft or 0.0):.2f}s | {toks / max(total_s, 1e-6):.2f} tok/s | "
            f"sweeps={spec.get('sweeps', 0)} (accepted={spec.get('accepted_tokens', 0)}/{spec.get('drafted_tokens', 0)})"
        )
        print(
            f"I/O:   {io['bytes_read'] / (1024**3):.2f} GiB @ {mib_s:.0f} MiB/s "
            f"(direct_io={io['direct_io_active']}, cache_hits={io['cache_hits']}, misses={io['cache_misses']})"
        )


if __name__ == "__main__":
    main()
