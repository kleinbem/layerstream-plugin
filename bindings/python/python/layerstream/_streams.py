"""Sync + async context-managed token and OpenAI completion streams (Python 3.15+)."""

import typing
from collections import deque
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from types import TracebackType
from typing import Any, override
lazy import asyncio

from ._layerstream import StreamTagParser, normalize_finish_reason
from ._openai_types import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkChoice,
    Completion,
    CompletionChoice,
    CompletionUsage,
)


def usage_from_summary(summary: Mapping[str, Any] | None) -> CompletionUsage | None:
    if summary is None:
        return None
    p = int(summary.get("prompt_tokens", 0))
    c = int(summary.get("completion_tokens", 0))
    return CompletionUsage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


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
        self._pending: deque[ChatCompletionChunk] = deque(
            [
                ChatCompletionChunk(
                    id=req_id,
                    created=created,
                    model=model_id,
                    choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant"))],
                )
            ]
        )
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
                finish_reason = normalize_finish_reason(
                    raw_reason,
                    bool(self._tag_parser is not None and self._tag_parser.has_tool_calls),
                )
                self._pending.append(
                    ChatCompletionChunk(
                        id=self._id,
                        created=self._created,
                        model=self._model_id,
                        choices=[
                            ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason=finish_reason)
                        ],
                        usage=usage_from_summary(summary),
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

        return self._pending.popleft()


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
                usage=usage_from_summary(summary),
            )
