"""Focused unit tests covering edge cases across `layerstream` Python SDK modules.

Exercises backend discovery, CLI subcommands/REPL, isolated worker validation helpers,
and in-process OpenAI/AsyncOpenAI schema/streaming/fallback paths without external downloads.
"""

from __future__ import annotations

import asyncio
import ctypes
import dataclasses
import enum
import json
import os
import sys
import types
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pytest

import layerstream
from layerstream import (
    AutoModel,
    LayerStreamError,
    TokenStream,
    __main__ as cli,
    _schema,
    _streams,
    _worker,
    openai as ls_openai,
)

_LS: Any = layerstream

# ---------------------------------------------------------------------------
# Helpers / lightweight fakes
# ---------------------------------------------------------------------------


class _FakeTokenStream:
    def __init__(
        self,
        pieces: Sequence[str],
        *,
        summary: dict[str, Any] | None = None,
        raise_on_next: BaseException | None = None,
    ) -> None:
        self._it = iter(pieces)
        self.summary = summary
        self.closed = False
        self._raise_on_next = raise_on_next

    def __iter__(self) -> _FakeTokenStream:
        return self

    def __next__(self) -> str:
        if self._raise_on_next is not None:
            exc = self._raise_on_next
            self._raise_on_next = None
            raise exc
        return next(self._it)

    def close(self) -> None:
        self.closed = True


class _FakeBackend:
    def __init__(
        self,
        *,
        model_path: str | None = "/fake/model.gguf",
        chat_reply: str | _FakeTokenStream = "hello",
        gen_reply: str | _FakeTokenStream = "world",
        summary: dict[str, Any] | None = None,
    ) -> None:
        self._loaded = True
        self._model_path = model_path
        self._chat_reply = chat_reply
        self._gen_reply = gen_reply
        self._summary = summary or {
            "prompt_tokens": 4,
            "completion_tokens": 3,
            "finish_reason": "stop",
        }
        self.unloaded = False
        self.kv_cleared = False
        self.saved_paths: list[str] = []
        self.loaded_paths: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_path(self) -> str | None:
        return self._model_path if self._loaded else None

    @property
    def last_summary(self) -> dict[str, Any] | None:
        return self._summary

    def stats(self) -> dict[str, Any]:
        return {
            "loaded": self._loaded,
            "model_path": self._model_path,
            "backend_name": "CPU",
            "using_gpu": False,
            "resident_layers": 2,
            "n_layers": 6,
            "cached_kv_tokens": 7,
            "io": {"bytes_read": 1024**3},
        }

    def unload(self) -> None:
        self._loaded = False
        self.unloaded = True

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        if kwargs.get("return_summary"):
            return self._gen_reply, self._summary
        return self._gen_reply

    def chat(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Any:
        if kwargs.get("return_summary"):
            return self._chat_reply, self._summary
        return self._chat_reply

    def tokenize(
        self, text: str, *, add_special: bool = False, parse_special: bool = False
    ) -> list[int]:
        return [1, 2]

    def detokenize(self, tokens: Sequence[int]) -> str:
        return "tok"

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        return "prompt"

    def kv_clear(self) -> None:
        self.kv_cleared = True

    def kv_trim(self, pos_min: int) -> bool:
        return pos_min >= 0

    def save_state(self, path: str) -> int:
        self.saved_paths.append(path)
        return 11

    def load_state(self, path: str) -> int:
        self.loaded_paths.append(path)
        return 11


# ---------------------------------------------------------------------------
# layerstream/__init__.py
# ---------------------------------------------------------------------------


def test_configure_native_backends_env_cuda_plugin_local_and_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pkg = tmp_path / "pkg"
    local_backends = fake_pkg / "_backends"
    local_backends.mkdir(parents=True)
    bad_so = local_backends / "libggml-base.so.0"
    ok_so = local_backends / "libggml-base.so.1"
    ggml_so = local_backends / "libggml.so.0"
    for p in (bad_so, ok_so, ggml_so):
        p.write_bytes(b"\x7fELF")

    env_dir1 = tmp_path / "env1"
    env_dir2 = tmp_path / "env2"
    cuda_dir = tmp_path / "cuda_backends"
    nix_ld_dir = tmp_path / "nix_ld"
    for d in (env_dir1, env_dir2, cuda_dir, nix_ld_dir):
        d.mkdir()
    (nix_ld_dir / "libstdc++.so.6").write_bytes(b"\x7fELF")

    fake_cuda = types.ModuleType("layerstream_cuda13")
    preloaded: list[bool] = []
    fake_cuda.preload_nvidia_libraries = lambda: preloaded.append(True)  # type: ignore[attr-defined]
    fake_cuda.backend_dir = lambda: str(cuda_dir)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "layerstream_cuda13", fake_cuda)

    monkeypatch.setattr(layerstream, "__file__", str(fake_pkg / "__init__.py"))
    monkeypatch.setenv("LAYERSTREAM_BACKEND_DIRS", f"{env_dir1}{os.pathsep}{env_dir1}")
    monkeypatch.setenv("LAYERSTREAM_BACKEND_DIR", str(env_dir2))
    monkeypatch.delenv("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", raising=False)
    monkeypatch.setenv("NIX_LD_LIBRARY_PATH", str(nix_ld_dir))

    cdll_calls: list[str] = []

    def fake_cdll(path: str, mode: int = 0) -> object:
        cdll_calls.append(path)
        if path.endswith("libggml-base.so.0"):
            raise OSError("bad so")
        return object()

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)

    dirs = _LS._configure_native_backends()
    assert preloaded == [True]
    assert str(env_dir1) in dirs
    assert str(env_dir2) in dirs
    assert str(cuda_dir) in dirs
    assert str(local_backends) in dirs
    assert str(ok_so) in cdll_calls
    assert str(ggml_so) in cdll_calls
    assert layerstream.configured_backend_dirs() == dirs
    monkeypatch.delenv("LAYERSTREAM_BACKEND_DIRS", raising=False)
    assert isinstance(layerstream.configured_backend_dirs(), list)


async def test_init_serve_cleanup_automodel_guards_and_async_cm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waited: list[bool] = []

    class _FakeServerHandle:
        def __enter__(self) -> _FakeServerHandle:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def wait(self) -> None:
            waited.append(True)

    monkeypatch.setattr(layerstream, "start_server", lambda *a, **kw: _FakeServerHandle())
    layerstream.serve("fake.gguf", port=9001, use_gpu=False)
    assert waited == [True]

    with pytest.raises(TypeError, match="on_progress"):
        AutoModel.from_pretrained("fake.gguf", on_progress="not-callable")  # type: ignore[arg-type]

    backend = _FakeBackend()
    model: AutoModel = _LS.AutoModel(backend, isolated_worker=False)
    assert model.worker_pid is None
    assert "loaded=True" in repr(model)
    with pytest.raises(ValueError, match="non-negative"):
        model.kv_trim(-1)

    async with model as m:
        assert m is model
    assert backend.unloaded is True
    assert repr(model) == "AutoModel(loaded=False)"

    # Exercise _cleanup_live_resources with both a live stream and live model (including failing close/unload).
    class _BrokenStream(_FakeTokenStream):
        def close(self) -> None:
            super().close()
            raise RuntimeError("close boom")

    class _BrokenBackend(_FakeBackend):
        def unload(self) -> None:
            super().unload()
            raise RuntimeError("unload boom")

    bs = _BrokenStream([])
    ts = TokenStream(bs)  # type: ignore[arg-type]
    bb = _BrokenBackend()
    bm: AutoModel = _LS.AutoModel(bb, isolated_worker=False)
    _LS._cleanup_live_resources()
    assert bs.closed is True
    assert bb.unloaded is True
    del ts, bm


# ---------------------------------------------------------------------------
# layerstream/__main__.py (CLI)
# ---------------------------------------------------------------------------


def test_cli_rust_entrypoint_help_version_and_errors() -> None:
    assert cli.main(["--version"]) == 0
    assert cli.main(["--help"]) == 0
    assert cli.main([]) == 2
    assert cli.main(["nonexistent-command"]) == 2
    assert cli.main(["check", "/definitely/missing/model.gguf", "--cpu"]) == 1
    assert cli.main(["check", "/definitely/missing/model.gguf", "--cpu", "--json"]) == 1


# ---------------------------------------------------------------------------
# layerstream/_worker.py
# ---------------------------------------------------------------------------


def test_worker_main_argument_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["_worker.py"])
    with pytest.raises(SystemExit) as e1:
        _worker._main()
    assert e1.value.code == 2

    monkeypatch.setattr(sys, "argv", ["_worker.py", "--worker-port"])
    with pytest.raises(SystemExit) as e2:
        _worker._main()
    assert e2.value.code == 2

    ports: list[int] = []
    monkeypatch.setattr(_worker, "run_worker_server", lambda port: ports.append(port))
    monkeypatch.setattr(sys, "argv", ["_worker.py", "--worker-port", "55123"])
    _worker._main()
    assert ports == [55123]


# ---------------------------------------------------------------------------
# layerstream/openai.py
# ---------------------------------------------------------------------------


def test_openai_schema_coercion_response_format_and_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert issubclass(LayerStreamError, RuntimeError)
    # SyncPage iteration and default factory
    empty_page = ls_openai.SyncPage()
    assert list(empty_page) == []

    # AsyncStreamMixin base close() raises NotImplementedError
    class _DummyAsyncStream(ls_openai.AsyncStreamMixin[str]):
        def __next__(self) -> str:
            raise StopIteration

    with pytest.raises(NotImplementedError):
        _DummyAsyncStream().close()

    # _annotation_to_json_schema edge cases
    class _V2Model:
        @classmethod
        def model_json_schema(cls) -> dict[str, Any]:
            return {"type": "object"}

        @classmethod
        def model_validate_json(cls, raw: str) -> dict[str, Any]:
            return {"v2": json.loads(raw)}

    assert _schema._annotation_to_json_schema(Any, "f") == {}
    assert _schema._annotation_to_json_schema(float, "f") == {"type": "number"}
    assert _schema._annotation_to_json_schema(list, "f") == {"type": "array"}
    assert _schema._annotation_to_json_schema(dict, "f") == {"type": "object"}
    assert _schema._annotation_to_json_schema(_V2Model, "f") == {"type": "object"}
    assert _schema._annotation_to_json_schema(tuple, "f") == {"type": "array"}
    assert _schema._annotation_to_json_schema(typing.Tuple, "f") == {"type": "array"}  # noqa: UP006
    assert _schema._annotation_to_json_schema(tuple[()], "f") == {"type": "array", "maxItems": 0}
    assert _schema._annotation_to_json_schema(tuple[int, ...], "f") == {
        "type": "array",
        "items": {"type": "integer"},
    }
    assert _schema._annotation_to_json_schema(dict[str, int], "f") == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }
    assert _schema._annotation_to_json_schema(Mapping, "f") == {"type": "object"}
    assert _schema._annotation_to_json_schema(typing.Dict, "f") == {"type": "object"}  # noqa: UP006
    with pytest.raises(TypeError, match="unsupported Literal"):
        _schema._annotation_to_json_schema(Literal[b"bytes"], "f")
    with pytest.raises(TypeError, match="must be 'str'"):
        _schema._annotation_to_json_schema(dict[int, str], "f")
    with pytest.raises(TypeError, match="unsupported dataclass field annotation"):
        _schema._annotation_to_json_schema(set[int], "f")

    # _coerce_value_for_type edge cases
    class _Color(enum.StrEnum):
        RED = "red"

    class _Size(enum.StrEnum):
        SMALL = "small"

    assert _schema._coerce_value_for_type(None, int) is None
    assert _schema._coerce_value_for_type("raw", "ForwardRef") == "raw"
    assert _schema._coerce_value_for_type("not-a-color", _Color | _Size) == "not-a-color"
    assert _schema._coerce_value_for_type([1, 2], tuple[int, ...]) == (1, 2)
    assert _schema._coerce_value_for_type([1, 2, 3], tuple[int, int]) == (1, 2, 3)
    assert _schema._coerce_value_for_type({"k": "red"}, dict[str, _Color]) == {"k": _Color.RED}

    # Fake pydantic path in _dataclass_to_json_schema and _instantiate_dataclass
    @dataclasses.dataclass
    class _SimpleDC:
        x: int

    class _FakeAdapter:
        def __init__(self, cls: type[Any]) -> None:
            self._cls = cls

        def json_schema(self) -> dict[str, Any]:
            return {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}

        def validate_python(self, data: Mapping[str, Any]) -> Any:
            return self._cls(**data)

    fake_pyd = types.SimpleNamespace(TypeAdapter=_FakeAdapter)
    monkeypatch.setattr(_schema, "_get_pydantic", lambda: fake_pyd)
    assert _schema._dataclass_to_json_schema(_SimpleDC)["type"] == "object"
    assert _schema._instantiate_dataclass(_SimpleDC, {"x": 7}) == _SimpleDC(x=7)
    monkeypatch.setattr(_schema, "_get_pydantic", lambda: None)

    # compile_response_format & parse_structured_text & usage_from_summary
    with pytest.raises(TypeError, match="unsupported response_format class"):
        _schema.compile_response_format(int)
    with pytest.raises(TypeError, match="response_format must be"):
        _schema.compile_response_format(123)
    assert _schema.parse_structured_text("{}", None) is None
    assert _schema.parse_structured_text('{"a": 1}', _V2Model) == {"v2": {"a": 1}}
    assert _schema.parse_structured_text('{"a": 1}', dict) == {"a": 1}
    with pytest.raises(ValueError, match="expected JSON object"):
        _schema.parse_structured_text("[1, 2]", _SimpleDC)
    assert _streams.usage_from_summary(None) is None

    # _merge_extra_body remaining keys
    merged = ls_openai._merge_extra_body(
        {"top_k": 12, "ngram_speculation": False, "max_draft_tokens": 3},
        min_p=0.0,
        top_k=40,
        repeat_penalty=1.0,
        penalty_last_n=64,
        ngram_speculation=True,
        max_draft_tokens=None,
    )
    assert merged == (0.0, 12, 1.0, 64, False, 3)


async def test_openai_streams_model_manager_and_fallback_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ChatCompletionStream: reasoning + tool_call + unknown delta + finish_reason="tool_calls"
    raw_stream = _FakeTokenStream(
        [
            "<think>plan</think>",
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Berlin"}}</tool_call>',
        ],
        summary={"prompt_tokens": 5, "completion_tokens": 9, "finish_reason": "stop"},
    )
    ccs = ls_openai.ChatCompletionStream(raw_stream, "id1", 100, "m", structured_json=False)
    ccs._enqueue_deltas([{"type": "unknown_event"}])
    chunks = list(ccs)
    assert any(c.choices[0].delta.reasoning_content == "plan" for c in chunks)
    assert any(c.choices[0].delta.tool_calls for c in chunks)
    assert chunks[-1].choices[0].finish_reason == "tool_calls"
    with pytest.raises(StopIteration):
        next(ccs)

    # ChatCompletionStream: structured_json=True + stop finish_reason
    json_stream = _FakeTokenStream(['{"ok": true}'], summary=None)
    ccs2 = ls_openai.ChatCompletionStream(json_stream, "id2", 100, "m", structured_json=True)
    chunks2 = list(ccs2)
    assert any(c.choices[0].delta.content == '{"ok": true}' for c in chunks2)
    assert chunks2[-1].choices[0].finish_reason == "stop"

    # AsyncStreamMixin.__anext__ cancellation closes stream (including double-CancelledError during close)
    cancel_inner = _FakeTokenStream(["x"])
    cancel_stream = ls_openai.CompletionStream(cancel_inner, "id3", 100, "m")

    async def _cancelled_to_thread(fn: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(fn, "__name__", "") == "_step":
            raise asyncio.CancelledError
        return fn(*args, **kwargs)

    monkeypatch.setattr(_streams.asyncio, "to_thread", _cancelled_to_thread)
    with pytest.raises(asyncio.CancelledError):
        await anext(cancel_stream)
    assert cancel_inner.closed is True

    cancel_inner2 = _FakeTokenStream(["x"])
    cancel_stream2 = ls_openai.CompletionStream(cancel_inner2, "id3b", 100, "m")

    async def _always_cancelled_to_thread(fn: Any, *args: Any, **kwargs: Any) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr(_streams.asyncio, "to_thread", _always_cancelled_to_thread)
    with pytest.raises(asyncio.CancelledError):
        await anext(cancel_stream2)
    assert cancel_inner2.closed is True
    monkeypatch.undo()

    # _ModelManager: invalid model type, empty resolve, existing AutoModel instance, lazy load
    with pytest.raises(TypeError, match="model must be None"):
        ls_openai.OpenAI(model=12345)

    empty_client = ls_openai.OpenAI()
    with pytest.raises(ValueError, match="no model specified"):
        empty_client.completions.create(prompt="hi")

    # Plain engine without _chat_with_summary / _generate_with_summary
    class _PlainEngine:
        def __init__(self, chat_out: Any, gen_out: Any) -> None:
            self.is_loaded = True
            self.model_path = "/plain/m.gguf"
            self.last_summary = {
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "finish_reason": "stop",
            }
            self._chat_out = chat_out
            self._gen_out = gen_out

        def chat(self, *a: object, **kw: object) -> Any:
            return self._chat_out

        def generate(self, *a: object, **kw: object) -> Any:
            return self._gen_out

        def unload(self) -> None:
            self.is_loaded = False

    plain = _PlainEngine(
        '<tool_call>{"name": "ping", "arguments": {}}</tool_call>',
        "plain-gen",
    )
    client = ls_openai.OpenAI(model=plain)
    assert client.models.list().data[0].id == "/plain/m.gguf"
    chat_resp = client.chat.completions.create(
        model="/plain/m.gguf",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert chat_resp.choices[0].finish_reason == "tool_calls"
    comp_resp = client.completions.create(prompt="hi")
    assert comp_resp.choices[0].text == "plain-gen"

    # Non-string guard branches in _ChatCompletionsResource & _CompletionsResource
    bad_eng = _PlainEngine({"not": "str"}, 999)
    bad_client = ls_openai.OpenAI(model=bad_eng)
    with pytest.raises(TypeError, match="expected string chat completion"):
        bad_client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(TypeError, match="expected string completion"):
        bad_client.completions.create(prompt="hi")
    monkeypatch.setattr(bad_client.chat.completions, "_create_impl", lambda **kw: "not-a-chat-cmpl")
    with pytest.raises(TypeError, match="expected non-streaming ChatCompletion"):
        bad_client.chat.completions.parse(
            messages=[{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )

    # Lazy resolve path + single-default-model alias by model_path + AsyncOpenAI.models.list()
    lazy_backend = _FakeBackend(model_path="/lazy/m.gguf")
    monkeypatch.setattr(
        AutoModel,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: _LS.AutoModel(lazy_backend, isolated_worker=False)),
    )
    async with ls_openai.AsyncOpenAI() as aclient:
        c_out = await aclient.chat.completions.create(
            model="lazy-spec",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert c_out.choices[0].finish_reason == "stop"
        alias_out = await aclient.completions.create(model="/lazy/m.gguf", prompt="hi")
        assert alias_out.model == "lazy-spec"
        models_page = await aclient.models.list()
        assert [m.id for m in models_page] == ["lazy-spec"]
    assert lazy_backend.unloaded is True
