"""Property-based (`hypothesis`) and native `pytest-asyncio` tests for Python 3.14+ SDK."""

import asyncio
import enum
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from hypothesis import given, settings, strategies as st

from layerstream import (
    AsyncOpenAI,
    AutoModel,
    StreamTagParser,
    json_schema_to_gbnf,
    parse_assistant_output,
)

# Match Rust `str::trim()` (`char::is_whitespace` / Unicode `White_Space`), which excludes
# ASCII file/group/record/unit separators `\x1c`..=`\x1f` that Python `str.strip()` removes.
_RUST_TRIM_RE = re.compile(
    r"^[\t\n\v\f\r \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+"
    r"|[\t\n\v\f\r \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+$"
)


def _rust_trim(s: str) -> str:
    return _RUST_TRIM_RE.sub("", s)


def _split_at_offsets(text: str, cuts: list[int]) -> list[str]:
    if not text:
        return [""]
    pts = sorted({0, len(text), *(c % (len(text) + 1) for c in cuts)})
    return [text[pts[i] : pts[i + 1]] for i in range(len(pts) - 1)]


@given(
    prefix=st.text(max_size=40),
    thought=st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=40),
    middle=st.text(max_size=40),
    arg_val=st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=40),
    suffix=st.text(max_size=40),
    cuts=st.lists(st.integers(min_value=0, max_value=256), max_size=24),
)
@settings(max_examples=120)
def test_stream_tag_parser_chunk_split_equivalence(
    prefix: str,
    thought: str,
    middle: str,
    arg_val: str,
    suffix: str,
    cuts: list[int],
) -> None:
    """Streaming `StreamTagParser` across arbitrary UTF-8 chunk boundaries must match
    one-shot `parse_assistant_output` on visible content, reasoning, and tool calls."""
    safe_prefix = prefix.replace("<think>", "").replace("<tool_call>", "")
    safe_thought = thought.replace("</think>", "")
    safe_middle = middle.replace("<think>", "").replace("<tool_call>", "")
    safe_suffix = suffix.replace("<think>", "").replace("<tool_call>", "")
    tool_json = json.dumps({"name": "run_op", "arguments": {"text": arg_val}})

    full = (
        f"{safe_prefix}<think>{safe_thought}</think>"
        f"{safe_middle}<tool_call>{tool_json}</tool_call>{safe_suffix}"
    )
    oneshot = parse_assistant_output(full)

    parser = StreamTagParser()
    events: list[dict[str, object]] = []
    for piece in _split_at_offsets(full, cuts):
        events.extend(parser.push(piece))
    events.extend(parser.flush())

    stream_content = _rust_trim("".join(str(e["text"]) for e in events if e["type"] == "content"))
    stream_reasoning_parts = [str(e["text"]) for e in events if e["type"] == "reasoning"]
    joined_reasoning = _rust_trim("".join(stream_reasoning_parts)) or None
    stream_calls = [
        (str(e["name"]), str(e["arguments"])) for e in events if e["type"] == "tool_call"
    ]
    oneshot_calls = [
        (tc["function"]["name"], tc["function"]["arguments"]) for tc in oneshot["tool_calls"]
    ]

    assert stream_content == oneshot["content"]
    assert joined_reasoning == oneshot["reasoning_content"]
    assert stream_calls == oneshot_calls


@given(
    raw=st.text(max_size=120),
    cuts=st.lists(st.integers(min_value=0, max_value=140), max_size=20),
)
@settings(max_examples=100)
def test_stream_tag_parser_never_panics_on_arbitrary_unicode(raw: str, cuts: list[int]) -> None:
    """Fuzz `StreamTagParser` & `parse_assistant_output` with arbitrary Unicode & partial tags."""
    oneshot = parse_assistant_output(raw)
    assert isinstance(oneshot["content"], str)

    parser = StreamTagParser()
    out = []
    for chunk in _split_at_offsets(raw, cuts):
        out.extend(parser.push(chunk))
    out.extend(parser.flush())
    assert all(e["type"] in ("content", "reasoning", "tool_call") for e in out)


@given(
    req_count=st.integers(min_value=1, max_value=4),
    str_max=st.integers(min_value=1, max_value=12),
    arr_max=st.integers(min_value=1, max_value=5),
    nullable=st.booleans(),
)
@settings(max_examples=50)
def test_json_schema_to_gbnf_properties(
    req_count: int,
    str_max: int,
    arr_max: int,
    nullable: bool,
) -> None:
    """Generated bounded JSON Schemas always compile into valid `root ::= ws ...` GBNF."""
    props: dict[str, object] = {}
    required: list[str] = []
    for idx in range(req_count):
        key = f"field_{idx}"
        base: dict[str, object]
        if idx % 3 == 0:
            base = {"type": "string", "minLength": 0, "maxLength": str_max}
        elif idx % 3 == 1:
            base = {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 0,
                "maxItems": arr_max,
            }
        else:
            base = {"enum": ["ok", "skip", 1, False]}
        props[key] = {"anyOf": [base, {"type": "null"}]} if nullable else base
        required.append(key)

    schema = {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }
    gbnf = json_schema_to_gbnf(schema)
    assert gbnf.startswith("root ::= ws ")
    for key in required:
        assert key in gbnf


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": math.nan}, "temperature"),
        ({"temperature": math.inf}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.01}, "top_p"),
        ({"min_p": -0.01}, "min_p"),
        ({"min_p": 1.01}, "min_p"),
        ({"repeat_penalty": 0.0}, "repeat_penalty"),
        ({"repeat_penalty": -1.0}, "repeat_penalty"),
        ({"frequency_penalty": math.inf}, "frequency_penalty"),
        ({"presence_penalty": math.nan}, "presence_penalty"),
        ({"penalty_last_n": -2}, "penalty_last_n"),
    ],
)
def test_sampler_validation_boundaries_inproc_and_isolated(
    synthetic_gguf: Path,
    kwargs: dict[str, float],
    match: str,
) -> None:
    """Invalid sampler ranges must raise `ValueError` identically in both modes."""
    for isolated in (False, True):
        with (
            AutoModel.from_pretrained(
                str(synthetic_gguf),
                use_gpu=False,
                ctx_size=128,
                batch_size=32,
                vram_budget_mb=0,
                prefer_direct_io=False,
                isolated_worker=isolated,
            ) as model,
            pytest.raises(ValueError, match=match),
        ):
            model.generate("a b", max_tokens=2, **kwargs)  # type: ignore[arg-type]


async def test_native_pytest_asyncio_cancellation_and_pep649_dataclasses(
    synthetic_json_gguf: Path,
) -> None:
    """Native `pytest-asyncio` coroutine verifying PEP 649 deferred dataclass annotations
    (nested dataclass, StrEnum, union `A | None`, tuple, Literal) and mid-stream
    `asyncio.CancelledError` cleanup."""

    class Mode(enum.StrEnum):
        FAST = "fast"
        SAFE = "safe"

    @dataclass
    class SubItem:
        active: bool

    @dataclass
    class NestedReport:
        mode: Mode
        tag: Literal["alpha", "beta"] | None
        pair: tuple[int, bool]
        sub: SubItem
        kind: Literal["v1", "v2"] = "v1"

    async with AsyncOpenAI(
        str(synthetic_json_gguf),
        use_gpu=False,
        ctx_size=512,
        batch_size=64,
        vram_budget_mb=0,
        prefer_direct_io=False,
    ) as client:
        parsed_resp = await client.chat.completions.parse(
            messages=[{"role": "user", "content": "a b"}],
            response_format=NestedReport,
            max_tokens=128,
            extra_body={"min_p": 0.05, "repeat_penalty": 1.1, "penalty_last_n": 32},
        )
        obj = parsed_resp.choices[0].message.parsed
        assert isinstance(obj, NestedReport)
        assert obj.mode in (Mode.FAST, Mode.SAFE)
        assert obj.tag in ("alpha", "beta", None)
        assert isinstance(obj.pair, tuple) and len(obj.pair) == 2
        assert isinstance(obj.sub, SubItem) and isinstance(obj.sub.active, bool)
        assert obj.kind in ("v1", "v2")

        # Task cancellation during async streaming must cleanly close the stream without hanging.
        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": "a b c"}],
            max_tokens=64,
            stream=True,
        )

        async def _consume_one_and_cancel() -> None:
            async with stream as s:
                first = await anext(s)
                assert first.object == "chat.completion.chunk"
                raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await _consume_one_and_cancel()

        # Follow-up request succeeds immediately (lock was released cleanly on cancellation).
        follow = await client.completions.create(prompt="a b", max_tokens=4)
        assert len(follow.choices[0].text) > 0
