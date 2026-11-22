"""Native Python SDK for `layerstream-core` (out-of-core GGUF layer streaming)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from os import PathLike
from types import TracebackType
from typing import Any

from ._layerstream import (
    NativeAutoModel as _NativeAutoModel,
    StreamTagParser,
    TokenStream as _NativeTokenStream,
    __version__,
    check,
    format_tools_system_prompt,
    inspect_gguf,
    json_object_gbnf,
    json_schema_to_gbnf,
    parse_assistant_output,
    probe_gpu_memory,
    pull,
    recommend,
)
from ._worker import (
    WorkerTokenStream as _WorkerTokenStream,
    _IsolatedWorkerClient,
    normalize_open_kwargs as _normalize_open_kwargs,
    serialize_messages_arg as _serialize_messages_arg,
    serialize_schema_arg as _serialize_schema_arg,
)
from .openai import AsyncOpenAI, OpenAI


class TokenStream(Iterator[str], AsyncIterator[str]):
    """Synchronous (`for chunk in s`) and asynchronous (`async for chunk in s`) token stream."""

    def __init__(self, inner: _NativeTokenStream | _WorkerTokenStream) -> None:
        self._inner = inner

    def __iter__(self) -> TokenStream:
        return self

    def __next__(self) -> str:
        return next(self._inner)

    def __enter__(self) -> TokenStream:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> bool:
        self.close()
        return False

    def close(self) -> None:
        self._inner.close()

    def __aiter__(self) -> TokenStream:
        return self

    async def __anext__(self) -> str:
        def _step() -> tuple[bool, str | None]:
            try:
                return True, next(self._inner)
            except StopIteration:
                return False, None

        has_item, chunk = await asyncio.to_thread(_step)
        if not has_item or chunk is None:
            raise StopAsyncIteration
        return chunk

    async def __aenter__(self) -> TokenStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> bool:
        await self.aclose()
        return False

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)


class AutoModel:
    """Out-of-core GGUF layer-streaming model (in-process or `isolated_worker=True`)."""

    def __init__(
        self,
        backend: _NativeAutoModel | _IsolatedWorkerClient,
        *,
        isolated_worker: bool,
    ) -> None:
        self._backend = backend
        self._isolated = isolated_worker

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        draft_model: str | None = None,
        quant: str | None = None,
        revision: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        vram_budget_mb: int | str | None = "auto",
        resident_layers: int | None = None,
        ctx_size: int = 4096,
        batch_size: int = 512,
        threads: int | None = None,
        kv_cache_type: str = "q8_0",
        use_gpu: bool = True,
        prefer_direct_io: bool = True,
        cpu_embeddings: bool = False,
        cpu_head: bool = False,
        stream_head: bool = False,
        moe_active_streaming: bool = True,
        max_draft_tokens: int = 5,
        verbose: bool = False,
        isolated_worker: bool = False,
    ) -> AutoModel:
        if not isolated_worker:
            native = _NativeAutoModel.from_pretrained(
                model,
                draft_model=draft_model,
                quant=quant,
                revision=revision,
                cache_dir=cache_dir,
                vram_budget_mb=vram_budget_mb,
                resident_layers=resident_layers,
                ctx_size=ctx_size,
                batch_size=batch_size,
                threads=threads,
                kv_cache_type=kv_cache_type,
                use_gpu=use_gpu,
                prefer_direct_io=prefer_direct_io,
                cpu_embeddings=cpu_embeddings,
                cpu_head=cpu_head,
                stream_head=stream_head,
                moe_active_streaming=moe_active_streaming,
                max_draft_tokens=max_draft_tokens,
                verbose=verbose,
            )
            return cls(native, isolated_worker=False)

        open_kwargs = _normalize_open_kwargs(
            model,
            draft_model=draft_model,
            quant=quant,
            revision=revision,
            cache_dir=cache_dir,
            vram_budget_mb=vram_budget_mb,
            resident_layers=resident_layers,
            ctx_size=ctx_size,
            batch_size=batch_size,
            threads=threads,
            kv_cache_type=kv_cache_type,
            use_gpu=use_gpu,
            prefer_direct_io=prefer_direct_io,
            cpu_embeddings=cpu_embeddings,
            cpu_head=cpu_head,
            stream_head=stream_head,
            moe_active_streaming=moe_active_streaming,
            max_draft_tokens=max_draft_tokens,
            verbose=verbose,
        )
        worker = _IsolatedWorkerClient(open_kwargs)
        return cls(worker, isolated_worker=True)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: bool = False,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
    ) -> str | TokenStream:
        if not self._isolated:
            assert isinstance(self._backend, _NativeAutoModel)
            res = self._backend.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                stop=stop,
                stream=stream,
                ngram_speculation=ngram_speculation,
                max_draft_tokens=max_draft_tokens,
                grammar=grammar,
                json_schema=json_schema,
            )
            return TokenStream(res) if stream else res

        assert isinstance(self._backend, _IsolatedWorkerClient)
        if grammar is not None and json_schema is not None:
            raise ValueError("pass at most one of 'grammar' or 'json_schema', not both")
        if stop is not None and not isinstance(stop, str):
            if not isinstance(stop, Sequence) or not all(isinstance(s, str) for s in stop):
                raise TypeError("stop must be None, a string, or a sequence of strings")
            stop_val: str | list[str] | None = list(stop)
        else:
            stop_val = stop

        eff_grammar = grammar if grammar is not None else _serialize_schema_arg(json_schema)
        if eff_grammar is not None and not eff_grammar.strip():
            raise ValueError("grammar must not be empty")

        params: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "stop": stop_val,
            "ngram_speculation": ngram_speculation,
            "max_draft_tokens": max_draft_tokens,
            "grammar": eff_grammar,
        }
        if stream:
            return TokenStream(self._backend.start_stream("generate_stream", params))
        out = self._backend._rpc("generate", params)
        return str(out["text"])

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: bool = False,
        add_generation_prompt: bool = True,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        parse_output: bool = False,
    ) -> str | dict[str, Any] | TokenStream:
        if not self._isolated:
            assert isinstance(self._backend, _NativeAutoModel)
            res = self._backend.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                stop=stop,
                stream=stream,
                add_generation_prompt=add_generation_prompt,
                ngram_speculation=ngram_speculation,
                max_draft_tokens=max_draft_tokens,
                tools=tools,
                grammar=grammar,
                json_schema=json_schema,
                parse_output=parse_output,
            )
            return TokenStream(res) if stream else res

        assert isinstance(self._backend, _IsolatedWorkerClient)
        if stream and parse_output:
            raise ValueError(
                "parse_output=True cannot be combined with stream=True "
                "(use StreamTagParser or layerstream.OpenAI)"
            )
        if grammar is not None and json_schema is not None:
            raise ValueError("pass at most one of 'grammar' or 'json_schema', not both")
        if stop is not None and not isinstance(stop, str):
            if not isinstance(stop, Sequence) or not all(isinstance(s, str) for s in stop):
                raise TypeError("stop must be None, a string, or a sequence of strings")
            stop_val: str | list[str] | None = list(stop)
        else:
            stop_val = stop

        clean_msgs = _serialize_messages_arg(messages)
        eff_grammar = grammar if grammar is not None else _serialize_schema_arg(json_schema)
        if eff_grammar is not None and not eff_grammar.strip():
            raise ValueError("grammar must not be empty")

        params: dict[str, Any] = {
            "messages": clean_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "stop": stop_val,
            "add_generation_prompt": add_generation_prompt,
            "ngram_speculation": ngram_speculation,
            "max_draft_tokens": max_draft_tokens,
            "tools": list(tools) if tools is not None else None,
            "grammar": eff_grammar,
        }
        if stream:
            return TokenStream(self._backend.start_stream("chat_stream", params))
        params["parse_output"] = parse_output
        out = self._backend._rpc("chat", params)
        return out["parsed"] if parse_output else str(out["text"])

    def tokenize(
        self,
        text: str,
        *,
        add_special: bool = False,
        parse_special: bool = False,
    ) -> list[int]:
        if not self._isolated:
            assert isinstance(self._backend, _NativeAutoModel)
            return self._backend.tokenize(
                text, add_special=add_special, parse_special=parse_special
            )
        assert isinstance(self._backend, _IsolatedWorkerClient)
        res = self._backend._rpc(
            "tokenize",
            {"text": text, "add_special": add_special, "parse_special": parse_special},
        )
        return list(res["tokens"])

    def detokenize(self, tokens: Sequence[int]) -> str:
        if not self._isolated:
            assert isinstance(self._backend, _NativeAutoModel)
            return self._backend.detokenize(list(tokens))
        assert isinstance(self._backend, _IsolatedWorkerClient)
        res = self._backend._rpc("detokenize", {"tokens": [int(t) for t in tokens]})
        return str(res["text"])

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        if not self._isolated:
            assert isinstance(self._backend, _NativeAutoModel)
            return self._backend.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt, tools=tools
            )
        assert isinstance(self._backend, _IsolatedWorkerClient)
        clean_msgs = _serialize_messages_arg(messages)
        res = self._backend._rpc(
            "apply_chat_template",
            {
                "messages": clean_msgs,
                "add_generation_prompt": add_generation_prompt,
                "tools": list(tools) if tools is not None else None,
            },
        )
        return str(res["prompt"])

    def kv_clear(self) -> None:
        if not self._isolated:
            assert isinstance(self._backend, _NativeAutoModel)
            self._backend.kv_clear()
            return
        assert isinstance(self._backend, _IsolatedWorkerClient)
        self._backend._rpc("kv_clear", {})

    def kv_trim(self, pos_min: int) -> bool:
        if pos_min < 0:
            raise ValueError("pos_min must be non-negative")
        if not self._isolated:
            assert isinstance(self._backend, _NativeAutoModel)
            return self._backend.kv_trim(pos_min)
        assert isinstance(self._backend, _IsolatedWorkerClient)
        res = self._backend._rpc("kv_trim", {"pos_min": int(pos_min)})
        return bool(res["trimmed"])

    def stats(self) -> dict[str, Any]:
        return self._backend.stats()

    @property
    def last_summary(self) -> dict[str, Any] | None:
        return self._backend.last_summary

    @property
    def is_loaded(self) -> bool:
        return self._backend.is_loaded

    @property
    def isolated_worker(self) -> bool:
        return self._isolated

    @property
    def worker_pid(self) -> int | None:
        if self._isolated:
            assert isinstance(self._backend, _IsolatedWorkerClient)
            return self._backend.worker_pid
        return None

    @property
    def model_path(self) -> str | None:
        return self._backend.model_path

    def unload(self) -> None:
        self._backend.unload()

    def __enter__(self) -> AutoModel:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> bool:
        self.unload()
        return False

    async def __aenter__(self) -> AutoModel:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> bool:
        await asyncio.to_thread(self.unload)
        return False

    def __repr__(self) -> str:
        p = self.model_path
        if p is not None:
            return (
                f"AutoModel(loaded=True, isolated_worker={self._isolated}, model_path={p!r})"
            )
        return "AutoModel(loaded=False)"


__all__ = [
    "AsyncOpenAI",
    "AutoModel",
    "OpenAI",
    "StreamTagParser",
    "TokenStream",
    "__version__",
    "check",
    "format_tools_system_prompt",
    "inspect_gguf",
    "json_object_gbnf",
    "json_schema_to_gbnf",
    "parse_assistant_output",
    "probe_gpu_memory",
    "pull",
    "recommend",
]
