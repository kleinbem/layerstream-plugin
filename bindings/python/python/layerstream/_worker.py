"""Isolated GPU worker subprocess wrapper & entrypoint (`python -m layerstream._worker`).

All IPC framing, loopback TCP auth, cancellation, and OS process termination/reap
live in `layerstream-core::worker_ipc` (Rust), exposed via `NativeWorkerClient`,
`NativeWorkerStream`, and `run_worker_server`.
"""

import ctypes
import json
import math
import os
import sys
import typing
from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from typing import Any

from ._layerstream import (
    NativeWorkerClient,
    NativeWorkerStream,
    json_schema_to_gbnf,
    run_worker_server,
)

WorkerTokenStream = NativeWorkerStream


def _to_worker_open_payload(open_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    vram_raw = open_kwargs.get("vram_budget_mb", "auto")
    vram_budget_auto = False
    vram_budget_bytes = 0
    if vram_raw is None:
        vram_budget_auto = False
        vram_budget_bytes = 0
    elif isinstance(vram_raw, str):
        raw = vram_raw.strip()
        if raw.lower() == "auto":
            vram_budget_auto = True
        else:
            vram_budget_bytes = int(raw) * 1024 * 1024
    else:
        vram_budget_bytes = int(vram_raw) * 1024 * 1024

    return {
        "model": str(open_kwargs["model"]),
        "draft_model": open_kwargs.get("draft_model"),
        "quant": open_kwargs.get("quant"),
        "revision": open_kwargs.get("revision"),
        "cache_dir": open_kwargs.get("cache_dir"),
        "vram_budget_auto": vram_budget_auto,
        "vram_budget_bytes": vram_budget_bytes,
        "resident_layers": open_kwargs.get("resident_layers"),
        "ctx_size": int(open_kwargs.get("ctx_size", 4096)),
        "batch_size": int(open_kwargs.get("batch_size", 512)),
        "threads": open_kwargs.get("threads"),
        "kv_cache_type": str(open_kwargs.get("kv_cache_type", "f16")).strip().lower(),
        "use_gpu": bool(open_kwargs.get("use_gpu", True)),
        "prefer_direct_io": bool(open_kwargs.get("prefer_direct_io", True)),
        "cpu_embeddings": bool(open_kwargs.get("cpu_embeddings", False)),
        "cpu_head": bool(open_kwargs.get("cpu_head", False)),
        "stream_head": bool(open_kwargs.get("stream_head", False)),
        "moe_active_streaming": bool(open_kwargs.get("moe_active_streaming", True)),
        "max_draft_tokens": int(open_kwargs.get("max_draft_tokens", 5)),
        "verbose": bool(open_kwargs.get("verbose", False)),
    }


def _gen_params_json(params: Mapping[str, Any]) -> str:
    stop = params.get("stop")
    stop_list: list[str]
    if stop is None:
        stop_list = []
    elif isinstance(stop, str):
        stop_list = [stop]
    else:
        stop_list = list(stop)
    payload: dict[str, Any] = {
        "max_tokens": int(params.get("max_tokens", 256)),
        "temperature": float(params.get("temperature", 0.0)),
        "top_p": float(params.get("top_p", 0.95)),
        "min_p": float(params.get("min_p", 0.0)),
        "top_k": int(params.get("top_k", 40)),
        "repeat_penalty": float(params.get("repeat_penalty", 1.0)),
        "frequency_penalty": float(params.get("frequency_penalty", 0.0)),
        "presence_penalty": float(params.get("presence_penalty", 0.0)),
        "penalty_last_n": int(params.get("penalty_last_n", 64)),
        "seed": int(params.get("seed", 42)),
        "stop": stop_list,
        "add_generation_prompt": bool(params.get("add_generation_prompt", True)),
        "ngram_speculation": bool(params.get("ngram_speculation", True)),
        "max_draft_tokens": params.get("max_draft_tokens"),
        "tools": params.get("tools"),
        "grammar": params.get("grammar"),
        "parse_output": bool(params.get("parse_output", False)),
    }
    return json.dumps(payload)


class IsolatedWorkerClient:
    """Thin Python adapter around Rust `NativeWorkerClient`."""

    def __init__(
        self,
        open_kwargs: dict[str, Any],
        *,
        on_progress: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        open_json = json.dumps(_to_worker_open_payload(open_kwargs))
        self._native = NativeWorkerClient.spawn(
            sys.executable,
            ["-c", "from layerstream._worker import _main; _main()"],
            open_json,
            on_progress=on_progress,
        )

    def rpc(self, op: str, params: dict[str, Any]) -> dict[str, Any]:
        res, _ = self.rpc_with_summary(op, params)
        return res

    def rpc_with_summary(
        self, op: str, params: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        match op:
            case "generate":
                text, summary = self._native.generate(
                    str(params["prompt"]), _gen_params_json(params)
                )
                return {"text": text}, summary
            case "chat":
                text, parsed, summary = self._native.chat(
                    json.dumps(params["messages"]), _gen_params_json(params)
                )
                return {"text": text, "parsed": parsed}, summary
            case "tokenize":
                tokens = self._native.tokenize(
                    str(params["text"]),
                    bool(params.get("add_special", False)),
                    bool(params.get("parse_special", False)),
                )
                return {"tokens": tokens}, None
            case "detokenize":
                text = self._native.detokenize([int(t) for t in params["tokens"]])
                return {"text": text}, None
            case "apply_chat_template":
                tools = params.get("tools")
                prompt = self._native.apply_chat_template(
                    json.dumps(params["messages"]),
                    bool(params.get("add_generation_prompt", True)),
                    json.dumps(tools) if tools is not None else None,
                )
                return {"prompt": prompt}, None
            case "kv_clear":
                self._native.kv_clear()
                return {}, None
            case "kv_trim":
                return {"trimmed": self._native.kv_trim(int(params["pos_min"]))}, None
            case "save_state":
                return {"tokens": self._native.save_state(str(params["path"]))}, None
            case "load_state":
                return {"tokens": self._native.load_state(str(params["path"]))}, None
            case other:
                raise ValueError(f"unknown worker op '{other}'")

    def start_stream(self, op: str, params: dict[str, Any]) -> NativeWorkerStream:
        if op == "generate_stream":
            return self._native.generate_stream(str(params["prompt"]), _gen_params_json(params))
        if op == "chat_stream":
            return self._native.chat_stream(
                json.dumps(params["messages"]), _gen_params_json(params)
            )
        raise ValueError(f"unknown worker stream op '{op}'")

    @property
    def is_loaded(self) -> bool:
        return self._native.is_loaded

    @property
    def worker_pid(self) -> int | None:
        return self._native.worker_pid

    @property
    def model_path(self) -> str | None:
        return self._native.model_path

    @property
    def last_summary(self) -> dict[str, Any] | None:
        return self._native.last_summary

    def stats(self) -> dict[str, Any]:
        return self._native.stats()

    def unload(self) -> None:
        self._native.unload()


def normalize_open_kwargs(
    model: str,
    *,
    draft_model: str | None,
    quant: str | None,
    revision: str | None,
    cache_dir: str | PathLike[str] | None,
    vram_budget_mb: int | str | None,
    resident_layers: int | None,
    ctx_size: int,
    batch_size: int,
    threads: int | None,
    kv_cache_type: str,
    use_gpu: bool,
    prefer_direct_io: bool,
    cpu_embeddings: bool,
    cpu_head: bool,
    stream_head: bool,
    moe_active_streaming: bool,
    max_draft_tokens: int,
    verbose: bool,
) -> dict[str, Any]:
    if isinstance(vram_budget_mb, bool):
        raise TypeError(
            "vram_budget_mb must be 'auto', None, or a non-negative integer MiB (not bool)"
        )
    if isinstance(vram_budget_mb, int) and vram_budget_mb < 0:
        raise ValueError("vram_budget_mb must be non-negative")
    if isinstance(vram_budget_mb, str):
        raw = vram_budget_mb.strip()
        if raw.lower() != "auto":
            try:
                parsed_mb = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"invalid vram_budget_mb '{raw}' (expected 'auto' or non-negative integer MiB)"
                ) from exc
            if parsed_mb < 0:
                raise ValueError("vram_budget_mb must be non-negative")
    if kv_cache_type.strip().lower() not in ("f16", "q8_0", "q4_0"):
        raise ValueError(f"invalid kv_cache_type '{kv_cache_type}' (expected f16, q8_0, or q4_0)")

    return {
        "model": model,
        "draft_model": draft_model,
        "quant": quant,
        "revision": revision,
        "cache_dir": os.fspath(cache_dir) if cache_dir is not None else None,
        "vram_budget_mb": vram_budget_mb,
        "resident_layers": resident_layers,
        "ctx_size": ctx_size,
        "batch_size": batch_size,
        "threads": threads,
        "kv_cache_type": kv_cache_type,
        "use_gpu": use_gpu,
        "prefer_direct_io": prefer_direct_io,
        "cpu_embeddings": cpu_embeddings,
        "cpu_head": cpu_head,
        "stream_head": stream_head,
        "moe_active_streaming": moe_active_streaming,
        "max_draft_tokens": max_draft_tokens,
        "verbose": verbose,
    }


def serialize_schema_arg(json_schema: Any) -> str | None:
    """Compile `json_schema` (dict, JSON str, or Pydantic class) to GBNF ahead of IPC."""
    if json_schema is None:
        return None
    return json_schema_to_gbnf(json_schema)


def _as_f32(v: float) -> float:
    return ctypes.c_float(v).value if math.isfinite(v) else v


def _validate_sample_args(
    *,
    temperature: float,
    top_p: float,
    min_p: float,
    repeat_penalty: float,
    frequency_penalty: float,
    presence_penalty: float,
    penalty_last_n: int = 64,
) -> None:
    temperature = _as_f32(temperature)
    top_p = _as_f32(top_p)
    min_p = _as_f32(min_p)
    repeat_penalty = _as_f32(repeat_penalty)
    frequency_penalty = _as_f32(frequency_penalty)
    presence_penalty = _as_f32(presence_penalty)
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be a finite non-negative number")
    if not math.isfinite(top_p) or top_p <= 0.0 or top_p > 1.0:
        raise ValueError("top_p must be in (0.0, 1.0]")
    if not math.isfinite(min_p) or min_p < 0.0 or min_p > 1.0:
        raise ValueError("min_p must be in [0.0, 1.0]")
    if not math.isfinite(repeat_penalty) or repeat_penalty <= 0.0:
        raise ValueError("repeat_penalty must be a finite positive number")
    if not math.isfinite(frequency_penalty) or frequency_penalty < -2.0 or frequency_penalty > 2.0:
        raise ValueError("frequency_penalty must be in [-2.0, 2.0]")
    if not math.isfinite(presence_penalty) or presence_penalty < -2.0 or presence_penalty > 2.0:
        raise ValueError("presence_penalty must be in [-2.0, 2.0]")
    if penalty_last_n < -1:
        raise ValueError("penalty_last_n must be >= -1")


def _normalize_stop_and_grammar(
    stop: str | Sequence[str] | None,
    grammar: str | None,
    json_schema: Any,
) -> tuple[str | list[str] | None, str | None]:
    if grammar is not None and json_schema is not None:
        raise ValueError("pass at most one of 'grammar' or 'json_schema', not both")
    if stop is not None and not isinstance(stop, str):
        raw_stop = typing.cast(object, stop)
        if not isinstance(raw_stop, Sequence):
            raise TypeError("stop must be None, a string, or a sequence of strings")
        stop_items: list[str] = []
        for s in typing.cast(Sequence[object], raw_stop):
            if not isinstance(s, str):
                raise TypeError("stop must be None, a string, or a sequence of strings")
            stop_items.append(s)
        stop_val: str | list[str] | None = stop_items
    else:
        stop_val = stop

    eff_grammar = grammar if grammar is not None else serialize_schema_arg(json_schema)
    if eff_grammar is not None and not eff_grammar.strip():
        raise ValueError("grammar must not be empty")
    return stop_val, eff_grammar


def prepare_generate_params(
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.95,
    min_p: float = 0.0,
    top_k: int = 40,
    repeat_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    penalty_last_n: int = 64,
    seed: int = 42,
    stop: str | Sequence[str] | None = None,
    ngram_speculation: bool = True,
    max_draft_tokens: int | None = None,
    grammar: str | None = None,
    json_schema: Any = None,
) -> dict[str, Any]:
    _validate_sample_args(
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        penalty_last_n=penalty_last_n,
    )
    stop_val, eff_grammar = _normalize_stop_and_grammar(stop, grammar, json_schema)
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "penalty_last_n": penalty_last_n,
        "seed": seed,
        "stop": stop_val,
        "ngram_speculation": ngram_speculation,
        "max_draft_tokens": max_draft_tokens,
        "grammar": eff_grammar,
    }


def prepare_chat_params(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.95,
    min_p: float = 0.0,
    top_k: int = 40,
    repeat_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    penalty_last_n: int = 64,
    seed: int = 42,
    stop: str | Sequence[str] | None = None,
    add_generation_prompt: bool = True,
    ngram_speculation: bool = True,
    max_draft_tokens: int | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    grammar: str | None = None,
    json_schema: Any = None,
) -> dict[str, Any]:
    _validate_sample_args(
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        penalty_last_n=penalty_last_n,
    )
    stop_val, eff_grammar = _normalize_stop_and_grammar(stop, grammar, json_schema)
    clean_msgs = serialize_messages_arg(messages)
    clean_tools = validate_tools_arg(tools)
    return {
        "messages": clean_msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "penalty_last_n": penalty_last_n,
        "seed": seed,
        "stop": stop_val,
        "add_generation_prompt": add_generation_prompt,
        "ngram_speculation": ngram_speculation,
        "max_draft_tokens": max_draft_tokens,
        "tools": clean_tools,
        "grammar": eff_grammar,
    }


def validate_tools_arg(tools: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    raw_tools = typing.cast(object, tools)
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes)):
        raise TypeError("tools must be a sequence of tool dicts")
    out: list[dict[str, Any]] = []
    for item in typing.cast(Sequence[object], raw_tools):
        if not isinstance(item, Mapping):
            raise TypeError("each entry in 'tools' must be a dict/mapping")
        out.append({str(k): v for k, v in typing.cast(Mapping[object, Any], item).items()})
    return out


def serialize_messages_arg(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    raw_messages = typing.cast(object, messages)
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
        raise TypeError("messages must be an iterable of {'role': str, 'content': str} dicts")
    out: list[dict[str, Any]] = []
    for raw_item in typing.cast(Sequence[object], raw_messages):
        if not isinstance(raw_item, Mapping):
            raise TypeError("each message must be a mapping")
        item = typing.cast(Mapping[str, Any], raw_item)
        if "role" not in item:
            raise ValueError("each message must contain a 'role' string")
        role = item["role"]
        if not isinstance(role, str):
            raise TypeError("message 'role' must be a string")
        if "content" not in item and "tool_calls" not in item:
            raise ValueError("each message must contain a 'content' string")
        d: dict[str, Any] = {"role": role, "content": item.get("content")}
        if "tool_calls" in item and item["tool_calls"] is not None:
            d["tool_calls"] = list(item["tool_calls"])
        if "tool_call_id" in item and item["tool_call_id"] is not None:
            d["tool_call_id"] = str(item["tool_call_id"])
        if "name" in item and item["name"] is not None:
            d["name"] = str(item["name"])
        out.append(d)
    return out


def _main() -> None:
    if "--worker-port" not in sys.argv:
        sys.exit(2)
    idx = sys.argv.index("--worker-port")
    if idx + 1 >= len(sys.argv):
        sys.exit(2)
    run_worker_server(int(sys.argv[idx + 1]))


if __name__ == "__main__":
    _main()
