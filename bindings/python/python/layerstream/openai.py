"""In-process OpenAI-compatible sync (`OpenAI`) & async (`AsyncOpenAI`) clients.

Wraps `layerstream.AutoModel` directly in-process (or with `isolated_worker=True`)
with zero HTTP daemon or port setup, providing drop-in compatibility with OpenAI SDK
callers:
- `client.chat.completions.create(..., stream=False | True, tools=..., response_format=...)`
- `client.chat.completions.parse(..., response_format=MyPydanticModel)`
- `client.completions.create(..., stream=False | True)`
- `client.models.list()`
- Stateful `<think>...</think>` -> `reasoning_content` and `<tool_call>...</tool_call>` -> `tool_calls`
- Native GBNF / JSON-Object / JSON-Schema / Pydantic constrained decoding.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from types import TracebackType
from typing import Any

from ._layerstream import (
    StreamTagParser,
    json_object_gbnf,
    json_schema_to_gbnf,
    parse_assistant_output,
)


def _dump_dataclass(obj: Any) -> dict[str, Any]:
    return asdict(obj)


@dataclass
class FunctionCall:
    name: str
    arguments: str

    def model_dump(self) -> dict[str, Any]:
        return _dump_dataclass(self)

    to_dict = model_dump


@dataclass
class ChatCompletionMessageToolCall:
    id: str
    function: FunctionCall
    type: str = "function"

    def model_dump(self) -> dict[str, Any]:
        return _dump_dataclass(self)

    to_dict = model_dump


@dataclass
class ChoiceDeltaToolCallFunction:
    name: str | None = None
    arguments: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return _dump_dataclass(self)

    to_dict = model_dump


@dataclass
class ChoiceDeltaToolCall:
    index: int
    id: str | None = None
    type: str = "function"
    function: ChoiceDeltaToolCallFunction | None = None

    def model_dump(self) -> dict[str, Any]:
        return _dump_dataclass(self)

    to_dict = model_dump


@dataclass
class ChatCompletionMessage:
    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    parsed: Any = None

    def model_dump(self) -> dict[str, Any]:
        return _dump_dataclass(self)

    to_dict = model_dump


@dataclass
class Choice:
    index: int
    message: ChatCompletionMessage
    finish_reason: str | None = "stop"
    logprobs: None = None

    def model_dump(self) -> dict[str, Any]:
        return _dump_dataclass(self)

    to_dict = model_dump


@dataclass
class CompletionUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def model_dump(self) -> dict[str, Any]:
        return _dump_dataclass(self)

    to_dict = model_dump


@dataclass
class ChatCompletion:
    id: str
    choices: list[Choice]
    created: int
    model: str
    object: str = "chat.completion"
    usage: CompletionUsage | None = None

    def model_dump(self) -> dict[str, Any]:
        return _dump_dataclass(self)

    to_dict = model_dump


@dataclass
class ChoiceDelta:
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChoiceDeltaToolCall] | None = None

    def model_dump(self) -> dict[str, Any]:
        return _dump_dataclass(self)

    to_dict = model_dump


@dataclass
class ChunkChoice:
    index: int
    delta: ChoiceDelta
    finish_reason: str | None = None
    logprobs: None = None

    def model_dump(self) -> dict[str, Any]:
        return _dump_dataclass(self)

    to_dict = model_dump


@dataclass
class ChatCompletionChunk:
    id: str
    choices: list[ChunkChoice]
    created: int
    model: str
    object: str = "chat.completion.chunk"
    usage: CompletionUsage | None = None

    def model_dump(self) -> dict[str, Any]:
        return _dump_dataclass(self)

    to_dict = model_dump


@dataclass
class CompletionChoice:
    text: str
    index: int = 0
    finish_reason: str | None = "stop"
    logprobs: None = None

    def model_dump(self) -> dict[str, Any]:
        return _dump_dataclass(self)

    to_dict = model_dump


@dataclass
class Completion:
    id: str
    choices: list[CompletionChoice]
    created: int
    model: str
    object: str = "text_completion"
    usage: CompletionUsage | None = None

    def model_dump(self) -> dict[str, Any]:
        return _dump_dataclass(self)

    to_dict = model_dump


@dataclass
class ModelObject:
    id: str
    created: int = 0
    object: str = "model"
    owned_by: str = "layerstream"

    def model_dump(self) -> dict[str, Any]:
        return _dump_dataclass(self)

    to_dict = model_dump


@dataclass
class SyncPage:
    data: list[ModelObject] = field(default_factory=list)
    object: str = "list"

    def __iter__(self) -> Iterator[ModelObject]:
        return iter(self.data)

    def model_dump(self) -> dict[str, Any]:
        return _dump_dataclass(self)

    to_dict = model_dump


def _dataclass_to_json_schema(dc_cls: type[Any]) -> dict[str, Any]:
    type_map: dict[Any, dict[str, Any]] = {
        str: {"type": "string"},
        "str": {"type": "string"},
        int: {"type": "integer"},
        "int": {"type": "integer"},
        float: {"type": "number"},
        "float": {"type": "number"},
        bool: {"type": "boolean"},
        "bool": {"type": "boolean"},
    }
    props: dict[str, Any] = {}
    required: list[str] = []
    for f in dataclasses.fields(dc_cls):
        py_t = f.type
        props[f.name] = type_map.get(py_t, {"type": "string"})
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            required.append(f.name)
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def _compile_response_format(response_format: Any) -> tuple[str | None, Any]:
    """Return `(gbnf_grammar_or_none, parser_target_or_none)`."""
    if response_format is None:
        return None, None

    # Direct Pydantic v2 / v1 BaseModel class or Python dataclass.
    if isinstance(response_format, type):
        if hasattr(response_format, "model_json_schema") or hasattr(response_format, "schema"):
            return json_schema_to_gbnf(response_format), response_format
        if dataclasses.is_dataclass(response_format):
            schema = _dataclass_to_json_schema(response_format)
            return json_schema_to_gbnf(schema), response_format
        raise TypeError(f"unsupported response_format class '{response_format.__name__}'")

    if isinstance(response_format, Mapping):
        rf_type = response_format.get("type", "text")
        if rf_type == "text":
            return None, None
        if rf_type == "json_object":
            return json_object_gbnf(), dict
        if rf_type == "json_schema":
            inner = response_format.get("json_schema", response_format)
            schema = inner.get("schema", inner) if isinstance(inner, Mapping) else inner
            if schema is None:
                raise ValueError("response_format 'json_schema' requires a 'schema' mapping")
            return json_schema_to_gbnf(schema), dict
        if rf_type == "gbnf":
            grammar = response_format.get("grammar")
            if not isinstance(grammar, str) or not grammar.strip():
                raise ValueError("response_format 'gbnf' requires a non-empty 'grammar' string")
            return grammar, None
        raise ValueError(
            f"unsupported response_format type '{rf_type}' "
            "(expected 'text', 'json_object', 'json_schema', or 'gbnf')"
        )

    raise TypeError(
        "response_format must be None, a mapping ({'type': ...}), or a Pydantic/dataclass type"
    )


def _parse_structured_text(raw_text: str, parser_target: Any) -> Any:
    if parser_target is None:
        return None
    if hasattr(parser_target, "model_validate_json"):
        return parser_target.model_validate_json(raw_text)
    if hasattr(parser_target, "parse_raw"):
        return parser_target.parse_raw(raw_text)
    data = json.loads(raw_text)
    if isinstance(parser_target, type) and dataclasses.is_dataclass(parser_target):
        if not isinstance(data, Mapping):
            raise ValueError("expected JSON object for dataclass response_format")
        return parser_target(**data)
    return data


def _usage_from_summary(summary: Mapping[str, Any] | None) -> CompletionUsage | None:
    if summary is None:
        return None
    p = int(summary.get("prompt_tokens", 0))
    c = int(summary.get("completion_tokens", 0))
    return CompletionUsage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


class ChatCompletionStream(Iterator[ChatCompletionChunk]):
    """Synchronous & asynchronous iterator yielding OpenAI `ChatCompletionChunk` events."""

    def __init__(
        self,
        token_stream: Any,
        owner_model: Any,
        req_id: str,
        created: int,
        model_id: str,
        structured_json: bool,
    ) -> None:
        self._stream = token_stream
        self._owner_model = owner_model
        self._id = req_id
        self._created = created
        self._model_id = model_id
        self._structured_json = structured_json
        self._tag_parser = None if structured_json else StreamTagParser()
        self._pending: list[ChatCompletionChunk] = [
            ChatCompletionChunk(
                id=req_id,
                created=created,
                model=model_id,
                choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant"))],
            )
        ]
        self._finished = False

    def __iter__(self) -> ChatCompletionStream:
        return self

    def __enter__(self) -> ChatCompletionStream:
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
        self._finished = True
        self._stream.close()

    def _enqueue_deltas(self, deltas: Sequence[Mapping[str, Any]]) -> None:
        for ev in deltas:
            kind = ev["type"]
            if kind == "content":
                self._pending.append(
                    ChatCompletionChunk(
                        id=self._id,
                        created=self._created,
                        model=self._model_id,
                        choices=[
                            ChunkChoice(
                                index=0,
                                delta=ChoiceDelta(content=str(ev["text"])),
                            )
                        ],
                    )
                )
            elif kind == "reasoning":
                self._pending.append(
                    ChatCompletionChunk(
                        id=self._id,
                        created=self._created,
                        model=self._model_id,
                        choices=[
                            ChunkChoice(
                                index=0,
                                delta=ChoiceDelta(reasoning_content=str(ev["text"])),
                            )
                        ],
                    )
                )
            elif kind == "tool_call":
                self._pending.append(
                    ChatCompletionChunk(
                        id=self._id,
                        created=self._created,
                        model=self._model_id,
                        choices=[
                            ChunkChoice(
                                index=0,
                                delta=ChoiceDelta(
                                    tool_calls=[
                                        ChoiceDeltaToolCall(
                                            index=int(ev["index"]),
                                            id=str(ev["id"]),
                                            type="function",
                                            function=ChoiceDeltaToolCallFunction(
                                                name=str(ev["name"]),
                                                arguments=str(ev["arguments"]),
                                            ),
                                        )
                                    ]
                                ),
                            )
                        ],
                    )
                )

    def __next__(self) -> ChatCompletionChunk:
        while not self._pending:
            if self._finished:
                raise StopIteration
            try:
                piece = next(self._stream)
            except StopIteration:
                self._finished = True
                if self._tag_parser is not None:
                    self._enqueue_deltas(self._tag_parser.flush())
                summary = self._owner_model.last_summary
                raw_reason = summary["finish_reason"] if summary else "stop"
                if self._tag_parser is not None and self._tag_parser.has_tool_calls:
                    finish_reason = "tool_calls"
                elif raw_reason == "length":
                    finish_reason = "length"
                else:
                    finish_reason = "stop"
                self._pending.append(
                    ChatCompletionChunk(
                        id=self._id,
                        created=self._created,
                        model=self._model_id,
                        choices=[
                            ChunkChoice(
                                index=0,
                                delta=ChoiceDelta(),
                                finish_reason=finish_reason,
                            )
                        ],
                        usage=_usage_from_summary(summary),
                    )
                )
                break

            if self._tag_parser is None:
                if piece:
                    self._pending.append(
                        ChatCompletionChunk(
                            id=self._id,
                            created=self._created,
                            model=self._model_id,
                            choices=[
                                ChunkChoice(index=0, delta=ChoiceDelta(content=piece))
                            ],
                        )
                    )
            else:
                self._enqueue_deltas(self._tag_parser.push(piece))

        if not self._pending:
            raise StopIteration
        return self._pending.pop(0)

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        def _step() -> tuple[bool, ChatCompletionChunk | None]:
            try:
                return True, self.__next__()
            except StopIteration:
                return False, None

        has_item, chunk = await asyncio.to_thread(_step)
        if not has_item or chunk is None:
            raise StopAsyncIteration
        return chunk

    async def __aenter__(self) -> ChatCompletionStream:
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


class CompletionStream(Iterator[Completion]):
    """Synchronous & asynchronous iterator yielding OpenAI `Completion` text chunks."""

    def __init__(
        self,
        token_stream: Any,
        owner_model: Any,
        req_id: str,
        created: int,
        model_id: str,
    ) -> None:
        self._stream = token_stream
        self._owner_model = owner_model
        self._id = req_id
        self._created = created
        self._model_id = model_id
        self._finished = False

    def __iter__(self) -> CompletionStream:
        return self

    def __enter__(self) -> CompletionStream:
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
        self._finished = True
        self._stream.close()

    def __next__(self) -> Completion:
        if self._finished:
            raise StopIteration
        try:
            piece = next(self._stream)
            return Completion(
                id=self._id,
                created=self._created,
                model=self._model_id,
                choices=[CompletionChoice(text=piece, index=0, finish_reason=None)],
            )
        except StopIteration:
            self._finished = True
            summary = self._owner_model.last_summary
            reason = summary["finish_reason"] if summary else "stop"
            return Completion(
                id=self._id,
                created=self._created,
                model=self._model_id,
                choices=[CompletionChoice(text="", index=0, finish_reason=reason)],
                usage=_usage_from_summary(summary),
            )

    def __aiter__(self) -> AsyncIterator[Completion]:
        return self

    async def __anext__(self) -> Completion:
        def _step() -> tuple[bool, Completion | None]:
            try:
                return True, self.__next__()
            except StopIteration:
                return False, None

        has_item, item = await asyncio.to_thread(_step)
        if not has_item or item is None:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)


class _ModelManager:
    def __init__(
        self,
        model: Any = None,
        *,
        isolated_worker: bool = False,
        model_kwargs: dict[str, Any],
    ) -> None:
        from . import AutoModel

        self._auto_cls = AutoModel
        self._lock = threading.Lock()
        self._isolated_worker = isolated_worker
        self._model_kwargs = dict(model_kwargs)
        self._models: dict[str, Any] = {}
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
                mid = getattr(model, "model_path", None) or "default"
                self._models[str(mid)] = model
                self._default_id = str(mid)
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
            items = [
                ModelObject(id=mid)
                for mid, inst in self._models.items()
                if inst.is_loaded
            ]
        return SyncPage(data=items)

    def unload_all(self) -> None:
        with self._lock:
            owned_models = [self._models[k] for k in self._owned if k in self._models]
            for k in list(self._owned):
                self._models.pop(k, None)
            self._owned.clear()
        for inst in owned_models:
            inst.unload()


class _ChatCompletionsResource:
    def __init__(self, mgr: _ModelManager) -> None:
        self._mgr = mgr

    def create(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: bool = False,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Any = None,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
    ) -> ChatCompletion | ChatCompletionStream:
        model_id, eng = self._mgr.resolve(model)
        eff_max = max_completion_tokens if max_completion_tokens is not None else (max_tokens or 512)
        grammar, _parser_target = _compile_response_format(response_format)
        req_id = f"chatcmpl-ls-{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        if stream:
            tok_stream = eng.chat(
                messages,
                max_tokens=eff_max,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                stop=stop,
                stream=True,
                tools=tools,
                grammar=grammar,
                ngram_speculation=ngram_speculation,
                max_draft_tokens=max_draft_tokens,
            )
            return ChatCompletionStream(
                tok_stream,
                eng,
                req_id,
                created,
                model_id,
                structured_json=grammar is not None,
            )

        raw = eng.chat(
            messages,
            max_tokens=eff_max,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop=stop,
            stream=False,
            tools=tools,
            grammar=grammar,
            ngram_speculation=ngram_speculation,
            max_draft_tokens=max_draft_tokens,
        )
        assert isinstance(raw, str)
        summary = eng.last_summary
        raw_reason = summary["finish_reason"] if summary else "stop"

        if grammar is not None:
            msg = ChatCompletionMessage(role="assistant", content=raw)
            finish_reason = "length" if raw_reason == "length" else "stop"
        else:
            parsed_dict = parse_assistant_output(raw)
            tcs = [
                ChatCompletionMessageToolCall(
                    id=tc["id"],
                    type=tc["type"],
                    function=FunctionCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in parsed_dict["tool_calls"]
            ]
            msg = ChatCompletionMessage(
                role="assistant",
                content=parsed_dict["content"] if parsed_dict["content"] or not tcs else None,
                reasoning_content=parsed_dict["reasoning_content"],
                tool_calls=tcs if tcs else None,
            )
            if tcs:
                finish_reason = "tool_calls"
            elif raw_reason == "length":
                finish_reason = "length"
            else:
                finish_reason = "stop"

        return ChatCompletion(
            id=req_id,
            created=created,
            model=model_id,
            choices=[Choice(index=0, message=msg, finish_reason=finish_reason)],
            usage=_usage_from_summary(summary),
        )

    def parse(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        response_format: Any,
        model: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
    ) -> ChatCompletion:
        _grammar, parser_target = _compile_response_format(response_format)
        resp = self.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop=stop,
            stream=False,
            tools=tools,
            response_format=response_format,
            ngram_speculation=ngram_speculation,
            max_draft_tokens=max_draft_tokens,
        )
        assert isinstance(resp, ChatCompletion)
        content = resp.choices[0].message.content or ""
        resp.choices[0].message.parsed = _parse_structured_text(content, parser_target)
        return resp


class _ChatNamespace:
    def __init__(self, mgr: _ModelManager) -> None:
        self.completions = _ChatCompletionsResource(mgr)


class _CompletionsResource:
    def __init__(self, mgr: _ModelManager) -> None:
        self._mgr = mgr

    def create(
        self,
        *,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: bool = False,
        grammar: str | None = None,
        json_schema: Any = None,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
    ) -> Completion | CompletionStream:
        model_id, eng = self._mgr.resolve(model)
        req_id = f"cmpl-ls-{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        if stream:
            tok_stream = eng.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                stop=stop,
                stream=True,
                grammar=grammar,
                json_schema=json_schema,
                ngram_speculation=ngram_speculation,
                max_draft_tokens=max_draft_tokens,
            )
            return CompletionStream(tok_stream, eng, req_id, created, model_id)

        text = eng.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop=stop,
            stream=False,
            grammar=grammar,
            json_schema=json_schema,
            ngram_speculation=ngram_speculation,
            max_draft_tokens=max_draft_tokens,
        )
        assert isinstance(text, str)
        summary = eng.last_summary
        reason = summary["finish_reason"] if summary else "stop"
        return Completion(
            id=req_id,
            created=created,
            model=model_id,
            choices=[CompletionChoice(text=text, index=0, finish_reason=reason)],
            usage=_usage_from_summary(summary),
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
        # Ignore harmless OpenAI SDK kwargs (`api_key`, `base_url`) if caller passes them.
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
    ) -> bool:
        self.close()
        return False


class _AsyncChatCompletionsResource:
    def __init__(self, sync_res: _ChatCompletionsResource) -> None:
        self._sync = sync_res

    async def create(self, **kwargs: Any) -> ChatCompletion | ChatCompletionStream:
        return await asyncio.to_thread(self._sync.create, **kwargs)

    async def parse(self, **kwargs: Any) -> ChatCompletion:
        return await asyncio.to_thread(self._sync.parse, **kwargs)


class _AsyncChatNamespace:
    def __init__(self, sync_res: _ChatCompletionsResource) -> None:
        self.completions = _AsyncChatCompletionsResource(sync_res)


class _AsyncCompletionsResource:
    def __init__(self, sync_res: _CompletionsResource) -> None:
        self._sync = sync_res

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
    ) -> bool:
        await self.aclose()
        return False
