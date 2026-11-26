"""In-process OpenAI-compatible sync (`OpenAI`) & async (`AsyncOpenAI`) clients (Python 3.15+)."""

import threading
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Literal, Unpack, overload
lazy import asyncio
lazy import time
lazy import uuid

from ._layerstream import build_chat_completion_dict
from ._openai_types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkChoice,
    Completion,
    CompletionChoice,
    CompletionUsage,
    FunctionCall,
    ModelObject,
    SyncPage,
)
from ._schema import compile_response_format, parse_structured_text
from ._streams import (
    AsyncStreamMixin,
    ChatCompletionStream,
    CompletionStream,
    usage_from_summary,
)
from ._types import JsonDict, SamplingOptions


def _merge_extra_body(
    extra_body: Mapping[str, Any] | None,
    *,
    min_p: float,
    top_k: int,
    repeat_penalty: float,
    penalty_last_n: int,
    ngram_speculation: bool,
    max_draft_tokens: int | None,
) -> tuple[float, int, float, int, bool, int | None]:
    if not extra_body:
        return min_p, top_k, repeat_penalty, penalty_last_n, ngram_speculation, max_draft_tokens
    if "min_p" in extra_body:
        min_p = float(extra_body["min_p"])
    if "top_k" in extra_body:
        top_k = int(extra_body["top_k"])
    if "repeat_penalty" in extra_body:
        repeat_penalty = float(extra_body["repeat_penalty"])
    if "penalty_last_n" in extra_body:
        penalty_last_n = int(extra_body["penalty_last_n"])
    if "ngram_speculation" in extra_body:
        ngram_speculation = bool(extra_body["ngram_speculation"])
    if "max_draft_tokens" in extra_body and extra_body["max_draft_tokens"] is not None:
        max_draft_tokens = int(extra_body["max_draft_tokens"])
    return min_p, top_k, repeat_penalty, penalty_last_n, ngram_speculation, max_draft_tokens


class _ModelManager:
    def __init__(
        self,
        model: Any = None,
        *,
        isolated_worker: bool = False,
        model_kwargs: JsonDict,
    ) -> None:
        from . import AutoModel

        self._auto_cls = AutoModel
        self._lock = threading.Lock()
        self._isolated_worker = isolated_worker
        self._model_kwargs = dict(model_kwargs)
        self._models: JsonDict = {}
        self._owned: set[str] = set()
        self._default_id: str | None = None

        if model is not None:
            if isinstance(model, str):
                loaded = AutoModel.from_pretrained(
                    model, isolated_worker=isolated_worker, **self._model_kwargs
                )
                self._models[model] = loaded
                self._owned.add(model)
                self._default_id = model
            elif hasattr(model, "generate") and hasattr(model, "chat"):
                mid = str(getattr(model, "model_path", None) or "default")
                self._models[mid] = model
                self._default_id = mid
            else:
                raise TypeError("model must be None, a model spec string, or an AutoModel instance")

    def resolve(self, requested_model: str | None) -> tuple[str, Any]:
        with self._lock:
            key = requested_model or self._default_id
            if not key:
                raise ValueError(
                    "no model specified; pass `model=...` to OpenAI(...) or `create(model=...)`"
                )
            existing = self._models.get(key)
            if existing is not None and existing.is_loaded:
                return key, existing
            if existing is None and len(self._models) == 1 and self._default_id is not None:
                default_inst = self._models[self._default_id]
                if default_inst.is_loaded and (
                    requested_model is None or requested_model == default_inst.model_path
                ):
                    return self._default_id, default_inst

            loaded = self._auto_cls.from_pretrained(
                key,
                isolated_worker=self._isolated_worker,
                **self._model_kwargs,
            )
            self._models[key] = loaded
            self._owned.add(key)
            if self._default_id is None:
                self._default_id = key
            return key, loaded

    def list_models(self) -> SyncPage:
        with self._lock:
            items = [ModelObject(id=mid) for mid, inst in self._models.items() if inst.is_loaded]
        return SyncPage(data=items)

    def unload_all(self) -> None:
        with self._lock:
            owned_models = [self._models[k] for k in self._owned if k in self._models]
            for k in list(self._owned):
                self._models.pop(k, None)
            if self._default_id in self._owned:
                self._default_id = None
            self._owned.clear()
        for inst in owned_models:
            inst.unload()


class _ChatCompletionsResource:
    def __init__(self, mgr: _ModelManager) -> None:
        self._mgr = mgr

    @overload
    def create(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        stream: Literal[False] = False,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Any = None,
        extra_body: Mapping[str, Any] | None = None,
        **sampling: Unpack[SamplingOptions],
    ) -> ChatCompletion[Any]: ...

    @overload
    def create(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        stream: Literal[True],
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Any = None,
        extra_body: Mapping[str, Any] | None = None,
        **sampling: Unpack[SamplingOptions],
    ) -> ChatCompletionStream: ...

    def create(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        stream: bool = False,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Any = None,
        extra_body: Mapping[str, Any] | None = None,
        **sampling: Unpack[SamplingOptions],
    ) -> ChatCompletion[Any] | ChatCompletionStream:
        return self._create_impl(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            stream=stream,
            tools=tools,
            response_format=response_format,
            extra_body=extra_body,
            **sampling,
        )

    def _create_impl(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
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
        stream: bool = False,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Any = None,
        compiled_grammar: str | None = None,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> ChatCompletion[Any] | ChatCompletionStream:
        (
            min_p,
            top_k,
            repeat_penalty,
            penalty_last_n,
            ngram_speculation,
            max_draft_tokens,
        ) = _merge_extra_body(
            extra_body,
            min_p=min_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            penalty_last_n=penalty_last_n,
            ngram_speculation=ngram_speculation,
            max_draft_tokens=max_draft_tokens,
        )

        model_id, eng = self._mgr.resolve(model)
        eff_max = (
            max_completion_tokens
            if max_completion_tokens is not None
            else (max_tokens if max_tokens is not None else 512)
        )
        grammar = (
            compiled_grammar
            if compiled_grammar is not None
            else compile_response_format(response_format)[0]
        )
        req_id = f"chatcmpl-ls-{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        sample_kw: JsonDict = {
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "penalty_last_n": penalty_last_n,
            "seed": seed,
        }

        if stream:
            tok_stream = eng.chat(
                messages,
                max_tokens=eff_max,
                stop=stop,
                stream=True,
                tools=tools,
                grammar=grammar,
                ngram_speculation=ngram_speculation,
                max_draft_tokens=max_draft_tokens,
                **sample_kw,
            )
            return ChatCompletionStream(
                tok_stream,
                req_id,
                created,
                model_id,
                structured_json=grammar is not None,
            )

        if hasattr(eng, "_chat_with_summary"):
            raw, summary = eng._chat_with_summary(
                messages,
                max_tokens=eff_max,
                stop=stop,
                tools=tools,
                grammar=grammar,
                ngram_speculation=ngram_speculation,
                max_draft_tokens=max_draft_tokens,
                **sample_kw,
            )
        else:
            raw = eng.chat(
                messages,
                max_tokens=eff_max,
                stop=stop,
                stream=False,
                tools=tools,
                grammar=grammar,
                ngram_speculation=ngram_speculation,
                max_draft_tokens=max_draft_tokens,
                **sample_kw,
            )
            summary = getattr(eng, "last_summary", None)

        if not isinstance(raw, str):
            raise TypeError("expected string chat completion")
        built = build_chat_completion_dict(
            req_id, created, model_id, raw, grammar is not None, summary
        )
        choice0 = built["choices"][0]
        msg_d = choice0["message"]
        raw_tcs = msg_d["tool_calls"]
        tcs = (
            [
                ChatCompletionMessageToolCall(
                    id=tc["id"],
                    type=tc["type"],
                    function=FunctionCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in raw_tcs
            ]
            if raw_tcs
            else None
        )
        msg: ChatCompletionMessage[Any] = ChatCompletionMessage(
            role="assistant",
            content=msg_d["content"],
            reasoning_content=msg_d["reasoning_content"],
            tool_calls=tcs,
        )
        return ChatCompletion(
            id=req_id,
            created=created,
            model=model_id,
            choices=[Choice(index=0, message=msg, finish_reason=choice0["finish_reason"])],
            usage=usage_from_summary(summary),
        )

    @overload
    def parse[T](
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        response_format: type[T],
        **kwargs: Any,
    ) -> ChatCompletion[T]: ...

    @overload
    def parse(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        response_format: Mapping[str, Any] | None,
        **kwargs: Any,
    ) -> ChatCompletion[Any]: ...

    def parse(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        response_format: Any,
        **kwargs: Any,
    ) -> ChatCompletion[Any]:
        grammar, parser_target = compile_response_format(response_format)
        resp = self._create_impl(
            messages=messages,
            response_format=None,
            compiled_grammar=grammar,
            stream=False,
            **kwargs,
        )
        if not isinstance(resp, ChatCompletion):
            raise TypeError("expected non-streaming ChatCompletion")
        content = resp.choices[0].message.content or ""
        resp.choices[0].message.parsed = parse_structured_text(content, parser_target)
        return resp


class _ChatNamespace:
    def __init__(self, mgr: _ModelManager) -> None:
        self.completions = _ChatCompletionsResource(mgr)


class _CompletionsResource:
    def __init__(self, mgr: _ModelManager) -> None:
        self._mgr = mgr

    @overload
    def create(
        self,
        *,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
        stream: Literal[False] = False,
        grammar: str | None = None,
        json_schema: Any = None,
        extra_body: Mapping[str, Any] | None = None,
        **sampling: Unpack[SamplingOptions],
    ) -> Completion: ...

    @overload
    def create(
        self,
        *,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
        stream: Literal[True],
        grammar: str | None = None,
        json_schema: Any = None,
        extra_body: Mapping[str, Any] | None = None,
        **sampling: Unpack[SamplingOptions],
    ) -> CompletionStream: ...

    def create(
        self,
        *,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
        stream: bool = False,
        grammar: str | None = None,
        json_schema: Any = None,
        extra_body: Mapping[str, Any] | None = None,
        **sampling: Unpack[SamplingOptions],
    ) -> Completion | CompletionStream:
        (
            min_p,
            top_k,
            repeat_penalty,
            penalty_last_n,
            ngram_speculation,
            max_draft_tokens,
        ) = _merge_extra_body(
            extra_body,
            min_p=sampling.get("min_p", 0.0),
            top_k=sampling.get("top_k", 40),
            repeat_penalty=sampling.get("repeat_penalty", 1.0),
            penalty_last_n=sampling.get("penalty_last_n", 64),
            ngram_speculation=sampling.get("ngram_speculation", True),
            max_draft_tokens=sampling.get("max_draft_tokens"),
        )
        stop = sampling.get("stop")
        model_id, eng = self._mgr.resolve(model)
        req_id = f"cmpl-ls-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        sample_kw: JsonDict = {
            "temperature": sampling.get("temperature", 0.0),
            "top_p": sampling.get("top_p", 0.95),
            "min_p": min_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "frequency_penalty": sampling.get("frequency_penalty", 0.0),
            "presence_penalty": sampling.get("presence_penalty", 0.0),
            "penalty_last_n": penalty_last_n,
            "seed": sampling.get("seed", 42),
        }

        if stream:
            tok_stream = eng.generate(
                prompt,
                max_tokens=max_tokens,
                stop=stop,
                stream=True,
                grammar=grammar,
                json_schema=json_schema,
                ngram_speculation=ngram_speculation,
                max_draft_tokens=max_draft_tokens,
                **sample_kw,
            )
            return CompletionStream(tok_stream, req_id, created, model_id)

        if hasattr(eng, "_generate_with_summary"):
            text, summary = eng._generate_with_summary(
                prompt,
                max_tokens=max_tokens,
                stop=stop,
                grammar=grammar,
                json_schema=json_schema,
                ngram_speculation=ngram_speculation,
                max_draft_tokens=max_draft_tokens,
                **sample_kw,
            )
        else:
            text = eng.generate(
                prompt,
                max_tokens=max_tokens,
                stop=stop,
                stream=False,
                grammar=grammar,
                json_schema=json_schema,
                ngram_speculation=ngram_speculation,
                max_draft_tokens=max_draft_tokens,
                **sample_kw,
            )
            summary = getattr(eng, "last_summary", None)

        if not isinstance(text, str):
            raise TypeError("expected string completion")
        reason = summary["finish_reason"] if summary else "stop"
        return Completion(
            id=req_id,
            created=created,
            model=model_id,
            choices=[CompletionChoice(text=text, index=0, finish_reason=reason)],
            usage=usage_from_summary(summary),
        )


class _ModelsResource:
    def __init__(self, mgr: _ModelManager) -> None:
        self._mgr = mgr

    def list(self) -> SyncPage:
        return self._mgr.list_models()


class OpenAI:
    """In-process OpenAI-compatible client backed directly by `layerstream.AutoModel`."""

    def __init__(
        self,
        model: Any = None,
        *,
        isolated_worker: bool = False,
        **model_kwargs: Any,
    ) -> None:
        model_kwargs.pop("api_key", None)
        model_kwargs.pop("base_url", None)
        self._mgr = _ModelManager(
            model,
            isolated_worker=isolated_worker,
            model_kwargs=model_kwargs,
        )
        self.chat = _ChatNamespace(self._mgr)
        self.completions = _CompletionsResource(self._mgr)
        self.models = _ModelsResource(self._mgr)

    def unload(self) -> None:
        self._mgr.unload_all()

    close = unload

    def __enter__(self) -> OpenAI:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None:
        self.close()


class _AsyncChatCompletionsResource:
    def __init__(self, sync_res: _ChatCompletionsResource) -> None:
        self._sync = sync_res

    @overload
    async def create(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        stream: Literal[False] = False,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Any = None,
        extra_body: Mapping[str, Any] | None = None,
        **sampling: Unpack[SamplingOptions],
    ) -> ChatCompletion[Any]: ...

    @overload
    async def create(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        stream: Literal[True],
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Any = None,
        extra_body: Mapping[str, Any] | None = None,
        **sampling: Unpack[SamplingOptions],
    ) -> ChatCompletionStream: ...

    async def create(self, **kwargs: Any) -> ChatCompletion[Any] | ChatCompletionStream:
        return await asyncio.to_thread(self._sync.create, **kwargs)

    @overload
    async def parse[T](
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        response_format: type[T],
        **kwargs: Any,
    ) -> ChatCompletion[T]: ...

    @overload
    async def parse(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        response_format: Mapping[str, Any] | None,
        **kwargs: Any,
    ) -> ChatCompletion[Any]: ...

    async def parse(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        response_format: Any,
        **kwargs: Any,
    ) -> ChatCompletion[Any]:
        return await asyncio.to_thread(
            self._sync.parse,
            messages=messages,
            response_format=response_format,
            **kwargs,
        )


class _AsyncChatNamespace:
    def __init__(self, sync_res: _ChatCompletionsResource) -> None:
        self.completions = _AsyncChatCompletionsResource(sync_res)


class _AsyncCompletionsResource:
    def __init__(self, sync_res: _CompletionsResource) -> None:
        self._sync = sync_res

    @overload
    async def create(
        self,
        *,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
        stream: Literal[False] = False,
        grammar: str | None = None,
        json_schema: Any = None,
        extra_body: Mapping[str, Any] | None = None,
        **sampling: Unpack[SamplingOptions],
    ) -> Completion: ...

    @overload
    async def create(
        self,
        *,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
        stream: Literal[True],
        grammar: str | None = None,
        json_schema: Any = None,
        extra_body: Mapping[str, Any] | None = None,
        **sampling: Unpack[SamplingOptions],
    ) -> CompletionStream: ...

    async def create(self, **kwargs: Any) -> Completion | CompletionStream:
        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncModelsResource:
    def __init__(self, sync_res: _ModelsResource) -> None:
        self._sync = sync_res

    async def list(self) -> SyncPage:
        return await asyncio.to_thread(self._sync.list)


class AsyncOpenAI:
    """Async in-process OpenAI-compatible client (`await client.chat.completions.create(...)`)."""

    def __init__(
        self,
        model: Any = None,
        *,
        isolated_worker: bool = False,
        **model_kwargs: Any,
    ) -> None:
        self._sync_client = OpenAI(
            model,
            isolated_worker=isolated_worker,
            **model_kwargs,
        )
        self.chat = _AsyncChatNamespace(self._sync_client.chat.completions)
        self.completions = _AsyncCompletionsResource(self._sync_client.completions)
        self.models = _AsyncModelsResource(self._sync_client.models)

    async def unload(self) -> None:
        await asyncio.to_thread(self._sync_client.unload)

    aclose = unload
    close = unload

    async def __aenter__(self) -> AsyncOpenAI:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None:
        await self.unload()


__all__ = [
    "AsyncOpenAI",
    "AsyncStreamMixin",
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionMessage",
    "ChatCompletionMessageToolCall",
    "ChatCompletionStream",
    "Choice",
    "ChoiceDelta",
    "ChoiceDeltaToolCall",
    "ChoiceDeltaToolCallFunction",
    "ChunkChoice",
    "Completion",
    "CompletionChoice",
    "CompletionStream",
    "CompletionUsage",
    "FunctionCall",
    "ModelObject",
    "OpenAI",
    "SyncPage",
]
