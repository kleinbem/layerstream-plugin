"""In-process OpenAI-compatible sync (`OpenAI`) & async (`AsyncOpenAI`) clients (Python 3.14+)."""

import asyncio
import dataclasses
import enum
import json
import sys
import threading
import time
import types
import typing
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from types import TracebackType
from typing import Any, Literal, Union, get_args, get_origin, overload, override

from ._layerstream import (
    StreamTagParser,
    json_object_gbnf,
    json_schema_to_gbnf,
    parse_assistant_output,
)

type JsonDict = dict[str, Any]


class _OpenAIModel:
    """Shared serialization base for OpenAI response dataclasses."""

    __slots__ = ()

    def model_dump(self) -> JsonDict:
        return asdict(self)  # type: ignore[call-overload]

    to_dict = model_dump


@dataclass(slots=True)
class FunctionCall(_OpenAIModel):
    name: str
    arguments: str


@dataclass(slots=True)
class ChatCompletionMessageToolCall(_OpenAIModel):
    id: str
    function: FunctionCall
    type: str = "function"


@dataclass(slots=True)
class ChoiceDeltaToolCallFunction(_OpenAIModel):
    name: str | None = None
    arguments: str | None = None


@dataclass(slots=True)
class ChoiceDeltaToolCall(_OpenAIModel):
    index: int
    id: str | None = None
    type: str = "function"
    function: ChoiceDeltaToolCallFunction | None = None


@dataclass(slots=True)
class ChatCompletionMessage(_OpenAIModel):
    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    parsed: Any = None


@dataclass(slots=True)
class Choice(_OpenAIModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str | None = "stop"
    logprobs: None = None


@dataclass(slots=True)
class CompletionUsage(_OpenAIModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(slots=True)
class ChatCompletion(_OpenAIModel):
    id: str
    choices: list[Choice]
    created: int
    model: str
    object: str = "chat.completion"
    usage: CompletionUsage | None = None


@dataclass(slots=True)
class ChoiceDelta(_OpenAIModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChoiceDeltaToolCall] | None = None


@dataclass(slots=True)
class ChunkChoice(_OpenAIModel):
    index: int
    delta: ChoiceDelta
    finish_reason: str | None = None
    logprobs: None = None


@dataclass(slots=True)
class ChatCompletionChunk(_OpenAIModel):
    id: str
    choices: list[ChunkChoice]
    created: int
    model: str
    object: str = "chat.completion.chunk"
    usage: CompletionUsage | None = None


@dataclass(slots=True)
class CompletionChoice(_OpenAIModel):
    text: str
    index: int = 0
    finish_reason: str | None = "stop"
    logprobs: None = None


@dataclass(slots=True)
class Completion(_OpenAIModel):
    id: str
    choices: list[CompletionChoice]
    created: int
    model: str
    object: str = "text_completion"
    usage: CompletionUsage | None = None


@dataclass(slots=True)
class ModelObject(_OpenAIModel):
    id: str
    created: int = 0
    object: str = "model"
    owned_by: str = "layerstream"


def _empty_model_list() -> list[ModelObject]:
    return []


@dataclass(slots=True)
class SyncPage(_OpenAIModel):
    data: list[ModelObject] = field(default_factory=_empty_model_list)
    object: str = "list"

    def __iter__(self) -> Iterator[ModelObject]:
        return iter(self.data)


class AsyncStreamMixin[T](Iterator[T], AsyncIterator[T]):
    """Generic PEP 695 sync + async context-managed stream base."""

    def close(self) -> None:
        raise NotImplementedError

    @override
    def __iter__(self) -> Iterator[T]:
        return self

    def __enter__(self) -> typing.Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None:
        self.close()

    @override
    def __aiter__(self) -> AsyncIterator[T]:
        return self

    @override
    async def __anext__(self) -> T:
        def _step() -> tuple[bool, T | None]:
            try:
                return True, next(self)
            except StopIteration:
                return False, None

        try:
            has_item, chunk = await asyncio.to_thread(_step)
        except asyncio.CancelledError:
            try:
                await asyncio.to_thread(self.close)
            except asyncio.CancelledError:
                self.close()
            raise
        if not has_item or chunk is None:
            raise StopAsyncIteration
        return chunk

    async def __aenter__(self) -> typing.Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)


def _annotation_to_json_schema(tp: Any, field_label: str) -> JsonDict:
    if tp is Any:
        return {}
    if tp is None or tp is types.NoneType:
        return {"type": "null"}
    if tp is str:
        return {"type": "string"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is list:
        return {"type": "array"}
    if tp is dict:
        return {"type": "object"}

    if isinstance(tp, type):
        cls = typing.cast(type[Any], tp)
        if issubclass(cls, enum.Enum):
            return {"enum": [member.value for member in cls]}
        if dataclasses.is_dataclass(cls):
            return _dataclass_to_json_schema(cls)
        if hasattr(cls, "model_json_schema"):
            schema_fn: Any = cls.model_json_schema
            return dict(typing.cast(Mapping[str, Any], schema_fn()))
        if hasattr(cls, "schema"):
            schema_fn_v1: Any = cls.schema
            return dict(typing.cast(Mapping[str, Any], schema_fn_v1()))

    origin = get_origin(tp)
    args = get_args(tp)
    if origin in (Union, types.UnionType):
        return {"anyOf": [_annotation_to_json_schema(a, field_label) for a in args]}
    if origin is Literal:
        for a in args:
            if not isinstance(a, (str, int, float, bool, type(None))):
                raise TypeError(f"unsupported Literal value {a!r} on '{field_label}'")
        return {"enum": list(args)}
    if origin in (list, Sequence):
        return (
            {"type": "array", "items": _annotation_to_json_schema(args[0], field_label)}
            if args
            else {"type": "array"}
        )
    if origin is tuple:
        if not args:
            return {"type": "array"}
        if len(args) == 1 and args[0] == ():
            return {"type": "array", "maxItems": 0}
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _annotation_to_json_schema(args[0], field_label)}
        pfx = [_annotation_to_json_schema(a, field_label) for a in args]
        return {
            "type": "array",
            "prefixItems": pfx,
            "minItems": len(pfx),
            "maxItems": len(pfx),
        }
    if origin in (dict, Mapping):
        if not args:
            return {"type": "object"}
        if args[0] is not str:
            raise TypeError(f"JSON object keys in '{field_label}' must be 'str', got {args[0]!r}")
        return {
            "type": "object",
            "additionalProperties": _annotation_to_json_schema(args[1], field_label),
        }

    raise TypeError(f"unsupported dataclass field annotation {tp!r} on '{field_label}'")


def _resolve_dataclass_field_types(dc_cls: type[Any]) -> dict[str, Any]:
    """Resolve PEP 649 deferred annotations in Python 3.14+ without `eval()`."""
    mod_ns = getattr(sys.modules.get(dc_cls.__module__), "__dict__", None)
    return typing.get_type_hints(dc_cls, globalns=mod_ns)


def _dataclass_to_json_schema(dc_cls: type[Any]) -> JsonDict:
    resolved_types = _resolve_dataclass_field_types(dc_cls)
    props: JsonDict = {}
    required: list[str] = []
    for f in dataclasses.fields(dc_cls):
        py_t = resolved_types.get(f.name, f.type)
        props[f.name] = _annotation_to_json_schema(py_t, f"{dc_cls.__name__}.{f.name}")
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            required.append(f.name)
    try:
        import importlib

        pydantic: Any = importlib.import_module("pydantic")
        return dict(typing.cast(Mapping[str, Any], pydantic.TypeAdapter(dc_cls).json_schema()))
    except ImportError:
        pass
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def _coerce_value_for_type(val: Any, tp: Any) -> Any:
    if val is None or isinstance(tp, str):
        return val
    if isinstance(tp, type):
        cls = typing.cast(type[Any], tp)
        if issubclass(cls, enum.Enum):
            return cls(val)
        if dataclasses.is_dataclass(cls) and isinstance(val, Mapping):
            return _instantiate_dataclass(cls, typing.cast(Mapping[str, Any], val))
        return val
    origin = get_origin(tp)
    args = get_args(tp)
    if origin in (Union, types.UnionType):
        for branch in args:
            if branch is types.NoneType and val is None:
                return None
            try:
                return _coerce_value_for_type(val, branch)
            except TypeError, ValueError, KeyError:
                continue
        return val
    if origin in (list, Sequence) and args and isinstance(val, list):
        items = typing.cast(list[Any], val)
        return [_coerce_value_for_type(v, args[0]) for v in items]
    if origin is tuple and isinstance(val, list):
        items = typing.cast(list[Any], val)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_value_for_type(v, args[0]) for v in items)
        if args and len(args) == len(items):
            return tuple(_coerce_value_for_type(v, a) for v, a in zip(items, args, strict=True))
        return tuple(items)
    if origin in (dict, Mapping) and len(args) == 2 and isinstance(val, Mapping):
        mapping = typing.cast(Mapping[Any, Any], val)
        return {k: _coerce_value_for_type(v, args[1]) for k, v in mapping.items()}
    return val


def _instantiate_dataclass(dc_cls: type[Any], data: Mapping[str, Any]) -> Any:
    try:
        import importlib

        pydantic: Any = importlib.import_module("pydantic")
        return pydantic.TypeAdapter(dc_cls).validate_python(data)
    except ImportError:
        pass

    resolved_types = _resolve_dataclass_field_types(dc_cls)
    kwargs = {
        k: _coerce_value_for_type(v, resolved_types[k]) if k in resolved_types else v
        for k, v in data.items()
    }
    return dc_cls(**kwargs)


def _compile_response_format(response_format: Any) -> tuple[str | None, Any]:
    if response_format is None:
        return None, None

    if isinstance(response_format, type):
        cls = typing.cast(type[Any], response_format)
        if hasattr(cls, "model_json_schema") or hasattr(cls, "schema"):
            return json_schema_to_gbnf(cls), cls
        if dataclasses.is_dataclass(cls):
            schema = _dataclass_to_json_schema(cls)
            return json_schema_to_gbnf(schema), cls
        raise TypeError(f"unsupported response_format class '{cls.__name__}'")

    if isinstance(response_format, Mapping):
        rf = typing.cast(Mapping[str, Any], response_format)
        match rf.get("type", "text"):
            case "text":
                return None, None
            case "json_object":
                return json_object_gbnf(), dict
            case "json_schema":
                inner: Any = rf.get("json_schema", rf)
                schema: Any = (
                    typing.cast(Mapping[str, Any], inner).get("schema", inner)
                    if isinstance(inner, Mapping)
                    else inner
                )
                if schema is None:
                    raise ValueError("response_format 'json_schema' requires a 'schema' mapping")
                return json_schema_to_gbnf(schema), dict
            case "gbnf":
                grammar: Any = rf.get("grammar")
                if not isinstance(grammar, str) or not grammar.strip():
                    raise ValueError("response_format 'gbnf' requires a non-empty 'grammar' string")
                return grammar, None
            case other:
                raise ValueError(
                    f"unsupported response_format type '{other}' "
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
        return _instantiate_dataclass(
            typing.cast(type[Any], parser_target), typing.cast(Mapping[str, Any], data)
        )
    return data


def _usage_from_summary(summary: Mapping[str, Any] | None) -> CompletionUsage | None:
    if summary is None:
        return None
    p = int(summary.get("prompt_tokens", 0))
    c = int(summary.get("completion_tokens", 0))
    return CompletionUsage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


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


class ChatCompletionStream(AsyncStreamMixin[ChatCompletionChunk]):
    """Synchronous & asynchronous iterator yielding OpenAI `ChatCompletionChunk` events."""

    def __init__(
        self,
        token_stream: Any,
        req_id: str,
        created: int,
        model_id: str,
        structured_json: bool,
    ) -> None:
        self._stream = token_stream
        self._id = req_id
        self._created = created
        self._model_id = model_id
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

    @override
    def close(self) -> None:
        self._finished = True
        self._stream.close()

    def _enqueue_deltas(self, deltas: Sequence[Mapping[str, Any]]) -> None:
        for ev in deltas:
            match ev["type"]:
                case "content":
                    delta = ChoiceDelta(content=str(ev["text"]))
                case "reasoning":
                    delta = ChoiceDelta(reasoning_content=str(ev["text"]))
                case "tool_call":
                    delta = ChoiceDelta(
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
                    )
                case _:
                    continue
            self._pending.append(
                ChatCompletionChunk(
                    id=self._id,
                    created=self._created,
                    model=self._model_id,
                    choices=[ChunkChoice(index=0, delta=delta)],
                )
            )

    @override
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
                summary = getattr(self._stream, "summary", None)
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
                            ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason=finish_reason)
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
                            choices=[ChunkChoice(index=0, delta=ChoiceDelta(content=piece))],
                        )
                    )
            else:
                self._enqueue_deltas(self._tag_parser.push(piece))

        if not self._pending:
            raise StopIteration
        return self._pending.pop(0)


class CompletionStream(AsyncStreamMixin[Completion]):
    """Synchronous & asynchronous iterator yielding OpenAI `Completion` text chunks."""

    def __init__(
        self,
        token_stream: Any,
        req_id: str,
        created: int,
        model_id: str,
    ) -> None:
        self._stream = token_stream
        self._id = req_id
        self._created = created
        self._model_id = model_id
        self._finished = False

    @override
    def close(self) -> None:
        self._finished = True
        self._stream.close()

    @override
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
            summary = getattr(self._stream, "summary", None)
            reason = summary["finish_reason"] if summary else "stop"
            return Completion(
                id=self._id,
                created=self._created,
                model=self._model_id,
                choices=[CompletionChoice(text="", index=0, finish_reason=reason)],
                usage=_usage_from_summary(summary),
            )


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
        stream: Literal[False] = False,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Any = None,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> ChatCompletion: ...

    @overload
    def create(
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
        stream: Literal[True],
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Any = None,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> ChatCompletionStream: ...

    def create(
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
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> ChatCompletion | ChatCompletionStream:
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
            max_completion_tokens if max_completion_tokens is not None else (max_tokens or 512)
        )
        grammar, _parser_target = _compile_response_format(response_format)
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
                tool_calls=tcs or None,
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
        **kwargs: Any,
    ) -> ChatCompletion:
        _grammar, parser_target = _compile_response_format(response_format)
        resp = self.create(
            messages=messages,
            response_format=response_format,
            stream=False,
            **kwargs,
        )
        content = resp.choices[0].message.content or ""
        resp.choices[0].message.parsed = _parse_structured_text(content, parser_target)
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
        stream: Literal[False] = False,
        grammar: str | None = None,
        json_schema: Any = None,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> Completion: ...

    @overload
    def create(
        self,
        *,
        prompt: str,
        model: str | None = None,
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
        stream: Literal[True],
        grammar: str | None = None,
        json_schema: Any = None,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> CompletionStream: ...

    def create(
        self,
        *,
        prompt: str,
        model: str | None = None,
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
        stream: bool = False,
        grammar: str | None = None,
        json_schema: Any = None,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
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
            min_p=min_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            penalty_last_n=penalty_last_n,
            ngram_speculation=ngram_speculation,
            max_draft_tokens=max_draft_tokens,
        )

        model_id, eng = self._mgr.resolve(model)
        req_id = f"cmpl-ls-{uuid.uuid4().hex[:16]}"
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
        stream: Literal[False] = False,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Any = None,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> ChatCompletion: ...

    @overload
    async def create(
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
        stream: Literal[True],
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Any = None,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> ChatCompletionStream: ...

    async def create(self, **kwargs: Any) -> ChatCompletion | ChatCompletionStream:
        return await asyncio.to_thread(self._sync.create, **kwargs)

    async def parse(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        response_format: Any,
        **kwargs: Any,
    ) -> ChatCompletion:
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
        stream: Literal[False] = False,
        grammar: str | None = None,
        json_schema: Any = None,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> Completion: ...

    @overload
    async def create(
        self,
        *,
        prompt: str,
        model: str | None = None,
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
        stream: Literal[True],
        grammar: str | None = None,
        json_schema: Any = None,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
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
