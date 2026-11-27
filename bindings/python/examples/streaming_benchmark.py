#!/usr/bin/env python3
"""Multi-turn KV prefix reuse & sparse MoE / streaming telemetry demo.

Demonstrates:
  1. Turn 1 cold prefill + streaming decode
  2. Turn 2 multi-turn continuation reusing cached KV prefix (faster TTFT)
  3. Residency, speculation, sparse MoE active-expert ratio, and O_DIRECT I/O counters
"""

import argparse
import time

from layerstream import AutoModel


def main() -> None:
    parser = argparse.ArgumentParser(description="layerstream multi-turn & telemetry demo")
    parser.add_argument("model", help="Local .gguf path, hf:// URI, or HF repo ID")
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--vram-budget-mb", default="auto")
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    budget: str | int = (
        "auto" if args.vram_budget_mb.strip().lower() == "auto" else int(args.vram_budget_mb)
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a concise systems engineering assistant."},
        {
            "role": "user",
            "content": "Briefly explain what a Mixture-of-Experts transformer is in 2 sentences.",
        },
    ]

    with AutoModel.from_pretrained(
        args.model,
        draft_model=args.draft_model,
        vram_budget_mb=budget,
        ctx_size=args.ctx_size,
        use_gpu=not args.cpu,
        verbose=args.verbose,
    ) as model:
        for turn, user_msg in enumerate(
            [
                None,
                "Now give one concrete advantage and one disadvantage of that architecture.",
            ],
            start=1,
        ):
            if user_msg is not None:
                messages.append({"role": "user", "content": user_msg})

            t0 = time.perf_counter()
            ttft = None
            reply_parts: list[str] = []
            print(f"\n=== Turn {turn} ===")
            for chunk in model.chat(messages, max_tokens=args.max_tokens, stream=True):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                reply_parts.append(chunk)
                print(chunk, end="", flush=True)
            elapsed = time.perf_counter() - t0
            reply = "".join(reply_parts)
            messages.append({"role": "assistant", "content": reply})

            summary = model.last_summary or {}
            toks = summary.get("completion_tokens", 0)
            print(
                f"\n[Turn {turn}: {toks} tok | TTFT={(ttft or 0.0):.2f}s | "
                f"{toks / max(elapsed, 1e-6):.2f} tok/s | sweeps={summary.get('speculation', {}).get('sweeps', 0)}]"
            )

        st = model.stats()
        io = st["io"]
        print("\n=== Final Engine Stats ===")
        print(
            f"Layers: {st['resident_layers']}/{st['n_layers']} resident, "
            f"{st['streamed_layers']} streamed, total_layer_loads={st['total_layer_loads']}"
        )
        if st["moe_layers"] > 0:
            pct = 100.0 * st["moe_experts_loaded"] / max(st["moe_experts_total"], 1)
            print(
                f"MoE:    {st['moe_layers']} layers, active_streaming={st['moe_active_streaming']}, "
                f"experts_loaded={st['moe_experts_loaded']}/{st['moe_experts_total']} ({pct:.1f}%)"
            )
        print(
            f"I/O:    {io['bytes_read'] / (1024**3):.2f} GiB read, direct_io={io['direct_io_active']}, "
            f"cache_hits={io['cache_hits']}, cache_misses={io['cache_misses']}"
        )


if __name__ == "__main__":
    main()
