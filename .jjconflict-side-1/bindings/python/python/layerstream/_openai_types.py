"""OpenAI-compatible response dataclasses with PEP 695 generic defaults (Python 3.15+)."""

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from ._types import JsonDict


class _OpenAIModel:
    """Shared serialization base for OpenAI response dataclasses."""

    __slots__ = ()

    def model_dump(self) -> JsonDict:
        return asdict(self)  # type: ignore[call-overload]

    to_dict = model_dump


@dataclass(frozen=True, slots=True)
class FunctionCall(_OpenAIModel):
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ChatCompletionMessageToolCall(_OpenAIModel):
    id: str
    function: FunctionCall
    type: str = "function"


@dataclass(frozen=True, slots=True)
class ChoiceDeltaToolCallFunction(_OpenAIModel):
    name: str | None = None
    arguments: str | None = None


@dataclass(frozen=True, slots=True)
class ChoiceDeltaToolCall(_OpenAIModel):
    index: int
    id: str | None = None
    type: str = "function"
    function: ChoiceDeltaToolCallFunction | None = None


@dataclass(slots=True)
class ChatCompletionMessage[T = Any](_OpenAIModel):
    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    parsed: T | None = None


@dataclass(frozen=True, slots=True)
class Choice[T = Any](_OpenAIModel):
    index: int
    message: ChatCompletionMessage[T]
    finish_reason: str | None = "stop"
    logprobs: None = None


@dataclass(frozen=True, slots=True)
class CompletionUsage(_OpenAIModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class ChatCompletion[T = Any](_OpenAIModel):
    id: str
    choices: list[Choice[T]]
    created: int
    model: str
    object: str = "chat.completion"
    usage: CompletionUsage | None = None


@dataclass(frozen=True, slots=True)
class ChoiceDelta(_OpenAIModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChoiceDeltaToolCall] | None = None


@dataclass(frozen=True, slots=True)
class ChunkChoice(_OpenAIModel):
    index: int
    delta: ChoiceDelta
    finish_reason: str | None = None
    logprobs: None = None


@dataclass(frozen=True, slots=True)
class ChatCompletionChunk(_OpenAIModel):
    id: str
    choices: list[ChunkChoice]
    created: int
    model: str
    object: str = "chat.completion.chunk"
    usage: CompletionUsage | None = None


@dataclass(frozen=True, slots=True)
class CompletionChoice(_OpenAIModel):
    text: str
    index: int = 0
    finish_reason: str | None = "stop"
    logprobs: None = None


@dataclass(frozen=True, slots=True)
class Completion(_OpenAIModel):
    id: str
    choices: list[CompletionChoice]
    created: int
    model: str
    object: str = "text_completion"
    usage: CompletionUsage | None = None


@dataclass(frozen=True, slots=True)
class ModelObject(_OpenAIModel):
    id: str
    created: int = 0
    object: str = "model"
    owned_by: str = "layerstream"


def _empty_model_list() -> list[ModelObject]:
    return []


@dataclass(frozen=True, slots=True)
class SyncPage(_OpenAIModel):
    data: list[ModelObject] = field(default_factory=_empty_model_list)
    object: str = "list"

    def __iter__(self) -> Iterator[ModelObject]:
        return iter(self.data)
