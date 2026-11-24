"""Command-line entrypoint (`layerstream` / `python -m layerstream`).

Delegates HTTP/SSE serving directly to `layerstream-server` (native Rust `axum` + `tokio`),
and model diagnostics / downloads / inference to `layerstream-core`.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any, Literal

from . import (
    AutoModel,
    __version__,
    check,
    inspect_gguf,
    probe_gpu_device,
    pull,
    recommend,
    start_server,
)


def _parse_vram_budget(raw: str | None) -> int | Literal["auto"] | None:
    if raw is None:
        return "auto"
    s = raw.strip()
    if not s or s.lower() == "none":
        return None
    if s.lower() == "auto":
        return "auto"
    return int(s)


def _fmt_gib(n_bytes: int | float | None) -> str:
    if n_bytes is None:
        return "-"
    return f"{float(n_bytes) / (1024.0**3):.2f} GiB"


def _cmd_inspect(args: argparse.Namespace) -> int:
    info = inspect_gguf(args.path)
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    print(f"File:           {args.path}")
    print(f"Name:           {info.get('name') or '-'}")
    print(f"Architecture:   {info['architecture']}")
    print(f"Layers:         {info['n_layers']}")
    print(f"Context Length: {info.get('context_length') or '-'}")
    print(f"Total Size:     {_fmt_gib(info['total_file_bytes'])}")
    print(f"Shards:         {info['shard_count']}")
    print(f"Tensors:        {info['tensor_count']}")
    return 0


def _cmd_recommend(args: argparse.Namespace) -> int:
    recs = recommend(
        args.repo,
        revision=args.revision,
        free_ram_gb=args.free_ram_gb,
        nvme_mib_s=args.nvme_mib_s,
        has_draft=not args.no_draft,
        cache_dir=args.cache_dir,
        use_gpu=not args.cpu,
    )
    if args.json:
        print(json.dumps(recs, indent=2))
        return 0
    if not recs:
        print("No GGUF candidates found.", file=sys.stderr)
        return 1
    print(f"Repository: {recs[0]['resolved_repo']}")
    print(
        f"{'Rec':<4} {'Quant':<12} {'Size':>10} {'Verdict':<14} {'Pinned':>7} {'Est tok/s':>14}  URI"
    )
    print("-" * 96)
    for r in recs:
        mark = "*" if r["recommended"] else " "
        tps = f"{r['est_tps_low']:.1f}-{r['est_tps_high']:.1f}"
        print(
            f" {mark}   {r['quant']:<12} {_fmt_gib(r['total_bytes']):>10} "
            f"{r['verdict']:<14} {r['pinned_pct']:>6}% {tps:>14}  {r['hf_uri']}"
        )
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    try:
        report = check(
            args.model,
            draft_model=args.draft_model,
            quant=args.quant,
            revision=args.revision,
            cache_dir=args.cache_dir,
            use_gpu=not args.cpu,
            allow_slow=args.allow_slow,
        )
    except Exception as exc:  # noqa: BLE001 — top-level CLI preflight boundary
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            print(f"Preflight FAILED: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"Verdict:        {report['verdict']} ({report['summary']})")
    print(f"Model:          {report['model_path']} ({_fmt_gib(report['total_file_bytes'])})")
    if report.get("draft_model_path"):
        print(f"Draft Model:    {report['draft_model_path']}")
    print(
        f"Architecture:   {report['architecture']} ({report['n_layers']} layers, ctx={report.get('context_length') or '-'})"
    )
    print(
        f"Backend / GPU:  {report.get('gpu_backend') or 'CPU'} "
        f"(free={_fmt_gib(report.get('gpu_free_bytes'))}, total={_fmt_gib(report.get('gpu_total_bytes'))})"
    )
    print(f"Storage I/O:    {report['storage_mib_s']:.0f} MiB/s (direct_io={report['direct_io']})")
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    local_path = pull(
        args.spec,
        cache_dir=args.cache_dir,
        quant=args.quant,
        revision=args.revision,
    )
    print(local_path)
    return 0


def _format_turn_telemetry(
    elapsed_s: float, summary: Mapping[str, Any] | None, st: Mapping[str, Any]
) -> str:
    comp = int((summary or {}).get("completion_tokens", 0))
    tps = comp / max(elapsed_s, 1e-4)
    spec: Mapping[str, Any] = (summary or {}).get("speculation") or {}
    drafted = int(spec.get("drafted_tokens", 0))
    accepted = int(spec.get("accepted_tokens", 0))
    backend = st.get("backend_name") or ("GPU" if st.get("using_gpu") else "CPU")
    res_l = st.get("resident_layers", 0)
    tot_l = st.get("n_layers", 0)
    io_stats: Mapping[str, Any] = st.get("io") or {}
    io_gib = float(io_stats.get("bytes_read", 0)) / (1024.0**3)
    parts = [
        f"{comp} tok in {elapsed_s:.2f}s ({tps:.2f} tok/s)",
        f"backend={backend}",
        f"resident={res_l}/{tot_l}",
        f"kv={st.get('cached_kv_tokens', 0)}",
        f"nvme_read={io_gib:.2f} GiB",
    ]
    if drafted > 0:
        parts.append(f"spec={accepted}/{drafted}")
    return " | ".join(parts)


def _resolve_use_gpu(cpu_flag: bool, *, isolated: bool = False) -> bool:
    if cpu_flag:
        return False
    if isolated:
        # Probe GPU availability in a short-lived child process so the parent coordinator
        # never initializes CUDA/Vulkan/Metal driver contexts.
        res = subprocess.run(
            [
                sys.executable,
                "-c",
                "import layerstream; raise SystemExit(0 if layerstream.probe_gpu_device(True) is not None else 1)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        )
        if res.returncode == 0:
            return True
    elif probe_gpu_device(True) is not None:
        return True
    print("[layerstream] no GPU device detected; falling back to CPU mode", file=sys.stderr)
    return False


def _cmd_chat(args: argparse.Namespace) -> int:
    vram_budget = _parse_vram_budget(args.vram_budget_mb)
    use_gpu = _resolve_use_gpu(args.cpu, isolated=args.isolated_worker)

    with AutoModel.from_pretrained(
        args.model,
        draft_model=args.draft_model,
        quant=args.quant,
        revision=args.revision,
        cache_dir=args.cache_dir,
        vram_budget_mb=vram_budget,
        resident_layers=args.resident_layers,
        ctx_size=args.ctx_size,
        batch_size=args.batch_size,
        threads=args.threads,
        kv_cache_type=args.kv_cache_type,
        use_gpu=use_gpu,
        prefer_direct_io=not args.no_direct_io,
        isolated_worker=args.isolated_worker,
        verbose=args.verbose,
    ) as model:
        if args.load_state:
            restored = model.load_state(args.load_state)
            print(
                f"[layerstream] restored {restored} KV tokens from {args.load_state}",
                file=sys.stderr,
            )

        messages: list[dict[str, str]] = (
            [{"role": "system", "content": args.system}] if args.system else []
        )
        gen_kwargs: dict[str, Any] = {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "min_p": args.min_p,
            "top_k": args.top_k,
            "repeat_penalty": args.repeat_penalty,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
            "seed": args.seed,
            "ngram_speculation": not args.no_ngram,
        }

        if args.prompt is not None:
            messages.append({"role": "user", "content": args.prompt})
            t0 = time.perf_counter()
            summary = None
            stream = model.chat(messages, stream=True, **gen_kwargs)
            if isinstance(stream, (str, dict)):
                raise TypeError("expected streaming TokenStream")
            with stream as s:
                for piece in s:
                    sys.stdout.write(piece)
                    sys.stdout.flush()
                summary = s.summary
            sys.stdout.write("\n")
            sys.stdout.flush()
            elapsed = time.perf_counter() - t0
            if args.save_state:
                saved = model.save_state(args.save_state)
                print(
                    f"[layerstream] saved {saved} KV tokens to {args.save_state}",
                    file=sys.stderr,
                )
            if not args.quiet_stats:
                print(
                    f"[layerstream] {_format_turn_telemetry(elapsed, summary, model.stats())}",
                    file=sys.stderr,
                )
            return 0

        st0 = model.stats()
        print(
            f"[layerstream {__version__}] loaded {st0.get('model_path')} "
            f"(backend={st0.get('backend_name')}, resident={st0.get('resident_layers')}/{st0.get('n_layers')}). "
            "Commands: /stats, /clear, /save <path>, /load <path>, /quit",
            file=sys.stderr,
        )

        while True:
            try:
                user_line = input("\n> ").strip()
            except EOFError, KeyboardInterrupt:
                print(file=sys.stderr)
                break
            if not user_line:
                continue
            match user_line:
                case "/quit" | "/exit":
                    break
                case "/clear":
                    model.kv_clear()
                    messages = [{"role": "system", "content": args.system}] if args.system else []
                    print(
                        "[layerstream] cleared KV cache and conversation history", file=sys.stderr
                    )
                    continue
                case "/stats":
                    print(json.dumps(model.stats(), indent=2), file=sys.stderr)
                    continue
                case _:
                    pass
            if user_line.startswith("/save "):
                target = user_line[6:].strip()
                try:
                    n_tok = model.save_state(target)
                    print(f"[layerstream] saved {n_tok} KV tokens -> {target}", file=sys.stderr)
                except (OSError, RuntimeError, ValueError) as exc:
                    print(f"[layerstream] save failed: {exc}", file=sys.stderr)
                continue
            if user_line.startswith("/load "):
                target = user_line[6:].strip()
                try:
                    n_tok = model.load_state(target)
                    print(f"[layerstream] restored {n_tok} KV tokens <- {target}", file=sys.stderr)
                except (OSError, RuntimeError, ValueError) as exc:
                    print(f"[layerstream] load failed: {exc}", file=sys.stderr)
                continue

            messages.append({"role": "user", "content": user_line})
            reply_parts: list[str] = []
            t0 = time.perf_counter()
            summary = None
            try:
                stream = model.chat(messages, stream=True, **gen_kwargs)
                if isinstance(stream, (str, dict)):
                    raise TypeError("expected streaming TokenStream")
                with stream as s:
                    for piece in s:
                        sys.stdout.write(piece)
                        sys.stdout.flush()
                        reply_parts.append(piece)
                    summary = s.summary
            except KeyboardInterrupt:
                sys.stdout.write("\n[cancelled]\n")
                sys.stdout.flush()
                messages.pop()
                continue

            sys.stdout.write("\n")
            sys.stdout.flush()
            messages.append({"role": "assistant", "content": "".join(reply_parts)})
            elapsed = time.perf_counter() - t0
            if not args.quiet_stats:
                print(
                    f"[{_format_turn_telemetry(elapsed, summary, model.stats())}]",
                    file=sys.stderr,
                )

        if args.save_state:
            saved = model.save_state(args.save_state)
            print(
                f"[layerstream] saved {saved} KV tokens to {args.save_state}",
                file=sys.stderr,
            )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    vram_budget = _parse_vram_budget(args.vram_budget_mb)
    use_gpu = _resolve_use_gpu(args.cpu, isolated=True)
    api_key = args.api_key if args.api_key is not None else os.environ.get("LAYERSTREAM_API_KEY")

    with start_server(
        args.model,
        host=args.host,
        port=args.port,
        api_key=api_key,
        draft_model=args.draft_model,
        quant=args.quant,
        revision=args.revision,
        cache_dir=args.cache_dir,
        vram_budget_mb=vram_budget,
        resident_layers=args.resident_layers,
        ctx_size=args.ctx_size,
        batch_size=args.batch_size,
        threads=args.threads,
        kv_cache_type=args.kv_cache_type,
        use_gpu=use_gpu,
        prefer_direct_io=not args.no_direct_io,
        idle_timeout_secs=args.idle_timeout_secs,
        verbose=args.verbose,
    ) as srv:
        print(
            f"[layerstream {__version__}] serving OpenAI API on {srv.base_url} "
            f"(model={args.model}, use_gpu={use_gpu})",
            file=sys.stderr,
            flush=True,
        )
        try:
            srv.wait()
        except KeyboardInterrupt:
            print("\n[layerstream] shutting down HTTP server", file=sys.stderr)
    return 0


def _add_engine_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="Local .gguf path, hf:// URI, or Hugging Face repo ID")
    p.add_argument(
        "--draft-model",
        default=None,
        help="Optional resident draft speculator ('auto', local .gguf, hf:// URI, or HF repo)",
    )
    p.add_argument("--quant", default=None, help="Preferred GGUF quantization (e.g. Q4_K_M)")
    p.add_argument("--revision", default=None, help="Hugging Face revision/branch")
    p.add_argument("--cache-dir", default=None, help="Model download cache directory")
    p.add_argument(
        "--vram-budget-mb",
        default="auto",
        help="VRAM residency budget in MiB or 'auto' (default: auto)",
    )
    p.add_argument("--resident-layers", type=int, default=None, help="Override pinned layer count")
    p.add_argument("--ctx-size", type=int, default=4096, help="Context window tokens")
    p.add_argument("--batch-size", type=int, default=512, help="Prefill chunk batch size")
    p.add_argument("--threads", type=int, default=None, help="CPU thread count")
    p.add_argument(
        "--kv-cache-type",
        default="q8_0",
        choices=["f16", "q8_0", "q4_0"],
        help="KV cache quantization (default: q8_0)",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU-only execution")
    p.add_argument("--no-direct-io", action="store_true", help="Disable O_DIRECT / F_NOCACHE")
    p.add_argument("--verbose", action="store_true", help="Enable llama.cpp / ggml debug logs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="layerstream",
        description="Out-of-core GGUF layer-streaming LLM engine & CLI",
    )
    parser.add_argument("--version", action="version", version=f"layerstream {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_chat = sub.add_parser("chat", help="Interactive or single-turn streaming chat")
    _add_engine_flags(p_chat)
    p_chat.add_argument(
        "--isolated-worker",
        action="store_true",
        help="Run inference in an isolated child worker process",
    )
    p_chat.add_argument("-p", "--prompt", default=None, help="Run a single non-interactive prompt")
    p_chat.add_argument("-s", "--system", default=None, help="Optional system prompt")
    p_chat.add_argument("--max-tokens", type=int, default=512)
    p_chat.add_argument("--temperature", type=float, default=0.0)
    p_chat.add_argument("--top-p", type=float, default=0.95)
    p_chat.add_argument("--min-p", type=float, default=0.0)
    p_chat.add_argument("--top-k", type=int, default=40)
    p_chat.add_argument("--repeat-penalty", type=float, default=1.0)
    p_chat.add_argument("--frequency-penalty", type=float, default=0.0)
    p_chat.add_argument("--presence-penalty", type=float, default=0.0)
    p_chat.add_argument("--seed", type=int, default=42)
    p_chat.add_argument("--no-ngram", action="store_true", help="Disable n-gram speculation")
    p_chat.add_argument("--load-state", default=None, help="Restore KV cache snapshot before chat")
    p_chat.add_argument("--save-state", default=None, help="Save KV cache snapshot on exit")
    p_chat.add_argument("--quiet-stats", action="store_true", help="Suppress per-turn telemetry")
    p_chat.set_defaults(func=_cmd_chat)

    p_serve = sub.add_parser("serve", help="Serve an OpenAI-compatible HTTP + SSE API (Rust axum)")
    _add_engine_flags(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    p_serve.add_argument(
        "--idle-timeout-secs",
        type=int,
        default=60,
        help="Idle seconds before evicting model/KV memory to 0 MB (default: 60; 0 = immediate)",
    )
    p_serve.add_argument(
        "--api-key",
        default=None,
        help="Optional Bearer API key (defaults to LAYERSTREAM_API_KEY if set)",
    )
    p_serve.set_defaults(func=_cmd_serve)

    p_rec = sub.add_parser("recommend", help="Rank Hugging Face GGUF quants for your GPU + NVMe")
    p_rec.add_argument("repo", help="Hugging Face repo ID (e.g. Qwen/Qwen3-30B-A3B-GGUF)")
    p_rec.add_argument("--revision", default=None)
    p_rec.add_argument("--free-ram-gb", type=float, default=None)
    p_rec.add_argument("--nvme-mib-s", type=float, default=None)
    p_rec.add_argument("--no-draft", action="store_true")
    p_rec.add_argument("--cache-dir", default=None)
    p_rec.add_argument("--cpu", action="store_true")
    p_rec.add_argument("--json", action="store_true")
    p_rec.set_defaults(func=_cmd_recommend)

    p_chk = sub.add_parser("check", help="Run fail-closed hardware & storage preflight check")
    p_chk.add_argument("model", help="Local .gguf path, hf:// URI, or HF repo")
    p_chk.add_argument("--draft-model", default=None)
    p_chk.add_argument("--quant", default=None)
    p_chk.add_argument("--revision", default=None)
    p_chk.add_argument("--cache-dir", default=None)
    p_chk.add_argument("--cpu", action="store_true")
    p_chk.add_argument("--allow-slow", action="store_true")
    p_chk.add_argument("--json", action="store_true")
    p_chk.set_defaults(func=_cmd_check)

    p_pull = sub.add_parser("pull", help="Download a GGUF checkpoint into the local cache")
    p_pull.add_argument("spec", help="hf://org/repo:file.gguf URI or HF repo ID")
    p_pull.add_argument("--quant", default=None)
    p_pull.add_argument("--revision", default=None)
    p_pull.add_argument("--cache-dir", default=None)
    p_pull.set_defaults(func=_cmd_pull)

    p_insp = sub.add_parser("inspect", help="Inspect GGUF v2/v3 metadata headers (0 MB VRAM)")
    p_insp.add_argument("path", help="Path to local .gguf file")
    p_insp.add_argument("--json", action="store_true")
    p_insp.set_defaults(func=_cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
