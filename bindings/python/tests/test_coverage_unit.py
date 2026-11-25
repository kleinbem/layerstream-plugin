"""Focused unit tests covering edge cases across `layerstream` Python SDK modules.

Exercises backend discovery, CLI subcommands/REPL, isolated worker validation helpers,
and in-process OpenAI/AsyncOpenAI schema/streaming/fallback paths without external downloads.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import dataclasses
import enum
import json
import os
import subprocess
import sys
import types
import typing
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pytest

import layerstream
from layerstream import AutoModel, TokenStream, __main__ as cli, _worker, openai as ls_openai

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
    for d in (env_dir1, env_dir2, cuda_dir):
        d.mkdir()

    fake_cuda = types.ModuleType("layerstream_cuda13")
    preloaded: list[bool] = []
    fake_cuda.preload_nvidia_libraries = lambda: preloaded.append(True)  # type: ignore[attr-defined]
    fake_cuda.backend_dir = lambda: str(cuda_dir)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "layerstream_cuda13", fake_cuda)

    monkeypatch.setattr(layerstream, "__file__", str(fake_pkg / "__init__.py"))
    monkeypatch.setenv("LAYERSTREAM_BACKEND_DIRS", f"{env_dir1}{os.pathsep}{env_dir1}")
    monkeypatch.setenv("LAYERSTREAM_BACKEND_DIR", str(env_dir2))
    monkeypatch.delenv("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", raising=False)
    monkeypatch.delenv("NIX_LD_LIBRARY_PATH", raising=False)

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


def test_cli_helpers_recommend_check_pull_and_serve(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli._parse_vram_budget(None) == "auto"
    assert cli._parse_vram_budget("") is None
    assert cli._parse_vram_budget("none") is None
    assert cli._parse_vram_budget("AUTO") == "auto"
    assert cli._parse_vram_budget("4096") == 4096
    assert cli._fmt_gib(None) == "-"

    # _cmd_recommend: JSON, empty, and formatted table
    rec_item = {
        "resolved_repo": "Org/Model-GGUF",
        "recommended": True,
        "quant": "Q4_K_M",
        "total_bytes": 4 * 1024**3,
        "verdict": "fast",
        "pinned_pct": 100,
        "est_tps_low": 20.0,
        "est_tps_high": 30.0,
        "hf_uri": "hf://Org/Model-GGUF:m.gguf",
    }
    monkeypatch.setattr(cli, "recommend", lambda *a, **kw: [rec_item])
    assert cli.main(["recommend", "Org/Model-GGUF", "--json"]) == 0
    assert "Q4_K_M" in capsys.readouterr().out

    assert cli.main(["recommend", "Org/Model-GGUF"]) == 0
    table_out = capsys.readouterr().out
    assert "Repository: Org/Model-GGUF" in table_out
    assert "Q4_K_M" in table_out

    monkeypatch.setattr(cli, "recommend", lambda *a, **kw: [])
    assert cli.main(["recommend", "Org/Empty-GGUF"]) == 1
    assert "No GGUF candidates found" in capsys.readouterr().err

    # _cmd_check: non-JSON failure + success with draft_model_path
    def _fail_check(*a: object, **kw: object) -> dict[str, Any]:
        raise RuntimeError("disk too slow")

    monkeypatch.setattr(cli, "check", _fail_check)
    assert cli.main(["check", "fake.gguf"]) == 1
    assert "Preflight FAILED: disk too slow" in capsys.readouterr().err

    monkeypatch.setattr(
        cli,
        "check",
        lambda *a, **kw: {
            "verdict": "ok",
            "summary": "good",
            "model_path": "/m.gguf",
            "draft_model_path": "/draft.gguf",
            "total_file_bytes": 1024**3,
            "architecture": "llama",
            "n_layers": 4,
            "context_length": 2048,
            "gpu_backend": "Vulkan",
            "gpu_free_bytes": 8 * 1024**3,
            "gpu_total_bytes": 16 * 1024**3,
            "storage_mib_s": 3500.0,
            "direct_io": True,
        },
    )
    assert cli.main(["check", "fake.gguf"]) == 0
    chk_out = capsys.readouterr().out
    assert "Draft Model:    /draft.gguf" in chk_out

    # _cmd_pull
    monkeypatch.setattr(cli, "pull", lambda *a, **kw: "/cache/model.gguf")
    assert cli.main(["pull", "Org/Model-GGUF"]) == 0
    assert capsys.readouterr().out.strip() == "/cache/model.gguf"

    # _resolve_use_gpu (isolated rc=0, isolated rc=1, inproc GPU found, inproc fallback)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess([], 0))
    assert cli._resolve_use_gpu(False, isolated=True) is True
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess([], 1))
    assert cli._resolve_use_gpu(False, isolated=True) is False
    monkeypatch.setattr(cli, "probe_gpu_device", lambda _: {"name": "FakeGPU"})
    assert cli._resolve_use_gpu(False, isolated=False) is True
    monkeypatch.setattr(cli, "probe_gpu_device", lambda _: None)
    assert cli._resolve_use_gpu(False, isolated=False) is False

    # _cmd_serve (normal + KeyboardInterrupt)
    class _FakeServeHandle:
        def __init__(self, *, interrupt: bool) -> None:
            self.base_url = "http://127.0.0.1:8080"
            self._interrupt = interrupt

        def __enter__(self) -> _FakeServeHandle:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def wait(self) -> None:
            if self._interrupt:
                raise KeyboardInterrupt

    monkeypatch.setattr(cli, "start_server", lambda *a, **kw: _FakeServeHandle(interrupt=False))
    assert cli.main(["serve", "fake.gguf", "--cpu"]) == 0
    assert "serving OpenAI API" in capsys.readouterr().err

    monkeypatch.setattr(cli, "start_server", lambda *a, **kw: _FakeServeHandle(interrupt=True))
    assert cli.main(["serve", "fake.gguf", "--cpu"]) == 0
    assert "shutting down HTTP server" in capsys.readouterr().err


def test_cli_chat_repl_edge_cases_and_non_stream_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _ReplModel:
        def __init__(self, *, stream_factory: Any) -> None:
            self._stream_factory = stream_factory
            self.saved: list[str] = []

        def __enter__(self) -> _ReplModel:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def stats(self) -> dict[str, Any]:
            return {
                "model_path": "fake.gguf",
                "backend_name": "CPU",
                "resident_layers": 1,
                "n_layers": 2,
                "cached_kv_tokens": 3,
                "io": {"bytes_read": 0},
            }

        def kv_clear(self) -> None:
            pass

        def save_state(self, path: str) -> int:
            if "fail" in path:
                raise OSError("cannot write")
            self.saved.append(path)
            return 5

        def load_state(self, path: str) -> int:
            if "fail" in path:
                raise ValueError("bad state")
            return 5

        def chat(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Any:
            return self._stream_factory()

    # Non-stream return guard in single-prompt mode
    monkeypatch.setattr(
        cli.AutoModel,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: _ReplModel(stream_factory=lambda: "not-a-stream")),
    )
    with pytest.raises(TypeError, match="expected streaming TokenStream"):
        cli.main(["chat", "fake.gguf", "--cpu", "-p", "hi"])

    # Interactive REPL: blank line, /save error, /load error, KeyboardInterrupt mid-stream, EOFError, --save-state
    inputs = iter(["", "/save fail.bin", "/load fail.bin", "trigger cancel"])

    def _fake_input(_prompt: str = "") -> str:
        try:
            return next(inputs)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr("builtins.input", _fake_input)

    def _cancel_stream() -> contextlib.AbstractContextManager[Any]:
        class _Ctx:
            summary = None

            def __enter__(self) -> _Ctx:
                return self

            def __exit__(self, *args: object) -> None:
                pass

            def __iter__(self) -> Iterator[str]:
                raise KeyboardInterrupt

        return _Ctx()

    save_target = str(tmp_path / "exit.kv")
    repl_model = _ReplModel(stream_factory=_cancel_stream)
    monkeypatch.setattr(
        cli.AutoModel,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: repl_model),
    )
    assert cli.main(["chat", "fake.gguf", "--cpu", "--save-state", save_target]) == 0
    captured = capsys.readouterr()
    assert "[cancelled]" in captured.out
    assert "save failed" in captured.err
    assert "load failed" in captured.err
    assert save_target in repl_model.saved

    # Non-stream guard in interactive REPL
    inputs2 = iter(["hello", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(inputs2))
    monkeypatch.setattr(
        cli.AutoModel,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: _ReplModel(stream_factory=lambda: {"bad": 1})),
    )
    with pytest.raises(TypeError, match="expected streaming TokenStream"):
        cli.main(["chat", "fake.gguf", "--cpu"])


# ---------------------------------------------------------------------------
# layerstream/_worker.py
# ---------------------------------------------------------------------------


def test_worker_validation_helpers_client_adapter_and_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # _gen_params_json stop forms
    p_str = json.loads(_worker._gen_params_json({"stop": "END"}))
    assert p_str["stop"] == ["END"]
    p_list = json.loads(_worker._gen_params_json({"stop": ("A", "B")}))
    assert p_list["stop"] == ["A", "B"]

    # serialize_schema_arg & _normalize_stop_and_grammar
    gbnf = _worker.serialize_schema_arg({"type": "object", "properties": {"a": {"type": "string"}}})
    assert gbnf is not None and "root ::=" in gbnf

    with pytest.raises(ValueError, match="at most one"):
        _worker._normalize_stop_and_grammar(None, 'root ::= "a"', {"type": "string"})
    with pytest.raises(TypeError, match="stop must be"):
        _worker._normalize_stop_and_grammar(123, None, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="stop must be"):
        _worker._normalize_stop_and_grammar(["ok", 42], None, None)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="grammar must not be empty"):
        _worker._normalize_stop_and_grammar(["a"], "   ", None)
    assert _worker._normalize_stop_and_grammar(("x", "y"), None, None) == (["x", "y"], None)

    # validate_tools_arg
    with pytest.raises(TypeError, match="tools must be a sequence"):
        _worker.validate_tools_arg("not-a-list")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="each entry in 'tools'"):
        _worker.validate_tools_arg(["bad"])  # type: ignore[list-item]
    assert _worker.validate_tools_arg([{"type": "function"}]) == [{"type": "function"}]

    # serialize_messages_arg
    with pytest.raises(TypeError, match="messages must be an iterable"):
        _worker.serialize_messages_arg("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="each message must be a mapping"):
        _worker.serialize_messages_arg(["bad"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="contain a 'role'"):
        _worker.serialize_messages_arg([{"content": "hi"}])
    with pytest.raises(TypeError, match="'role' must be a string"):
        _worker.serialize_messages_arg([{"role": 1, "content": "hi"}])
    with pytest.raises(ValueError, match="contain a 'content'"):
        _worker.serialize_messages_arg([{"role": "user"}])
    msgs = _worker.serialize_messages_arg(
        [{"role": "assistant", "tool_calls": [{"id": "c1"}], "tool_call_id": "c1", "name": "fn"}]
    )
    assert msgs == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1"}],
            "tool_call_id": "c1",
            "name": "fn",
        }
    ]

    # IsolatedWorkerClient adapter paths via fake NativeWorkerClient
    class _FakeNativeWorker:
        def __init__(self) -> None:
            self.is_loaded = True
            self.worker_pid = 4242
            self.model_path = "/fake/w.gguf"
            self.last_summary = {"finish_reason": "stop"}

        def generate(self, prompt: str, params_json: str) -> tuple[str, dict[str, Any]]:
            return "gen-text", {"finish_reason": "stop"}

        def chat(
            self, messages_json: str, params_json: str
        ) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
            return "chat-text", {"content": "chat-text"}, {"finish_reason": "stop"}

        def chat_stream(self, messages_json: str, params_json: str) -> _FakeTokenStream:
            return _FakeTokenStream(["a"])

    client = object.__new__(_worker.IsolatedWorkerClient)
    client._native = _FakeNativeWorker()  # type: ignore[assignment]
    assert client.model_path == "/fake/w.gguf"
    assert client.generate("hi", return_summary=True) == ("gen-text", {"finish_reason": "stop"})
    with pytest.raises(ValueError, match="cannot be combined"):
        client.chat([{"role": "user", "content": "hi"}], stream=True, parse_output=True)
    assert isinstance(
        client.chat([{"role": "user", "content": "hi"}], stream=True), _FakeTokenStream
    )
    assert client.chat([{"role": "user", "content": "hi"}], return_summary=True) == (
        "chat-text",
        {"finish_reason": "stop"},
    )

    # _worker._main() argument handling
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

    class _V1Model:
        @classmethod
        def schema(cls) -> dict[str, Any]:
            return {"type": "object"}

        @classmethod
        def parse_raw(cls, raw: str) -> dict[str, Any]:
            return {"v1": json.loads(raw)}

    assert ls_openai._annotation_to_json_schema(Any, "f") == {}
    assert ls_openai._annotation_to_json_schema(float, "f") == {"type": "number"}
    assert ls_openai._annotation_to_json_schema(list, "f") == {"type": "array"}
    assert ls_openai._annotation_to_json_schema(dict, "f") == {"type": "object"}
    assert ls_openai._annotation_to_json_schema(_V2Model, "f") == {"type": "object"}
    assert ls_openai._annotation_to_json_schema(_V1Model, "f") == {"type": "object"}
    assert ls_openai._annotation_to_json_schema(tuple, "f") == {"type": "array"}
    assert ls_openai._annotation_to_json_schema(typing.Tuple, "f") == {"type": "array"}  # noqa: UP006
    assert ls_openai._annotation_to_json_schema(tuple[()], "f") == {"type": "array", "maxItems": 0}
    assert ls_openai._annotation_to_json_schema(tuple[int, ...], "f") == {
        "type": "array",
        "items": {"type": "integer"},
    }
    assert ls_openai._annotation_to_json_schema(dict[str, int], "f") == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }
    assert ls_openai._annotation_to_json_schema(Mapping, "f") == {"type": "object"}
    assert ls_openai._annotation_to_json_schema(typing.Dict, "f") == {"type": "object"}  # noqa: UP006
    with pytest.raises(TypeError, match="unsupported Literal"):
        ls_openai._annotation_to_json_schema(Literal[b"bytes"], "f")
    with pytest.raises(TypeError, match="must be 'str'"):
        ls_openai._annotation_to_json_schema(dict[int, str], "f")
    with pytest.raises(TypeError, match="unsupported dataclass field annotation"):
        ls_openai._annotation_to_json_schema(set[int], "f")

    # _coerce_value_for_type edge cases
    class _Color(enum.StrEnum):
        RED = "red"

    assert ls_openai._coerce_value_for_type(None, int) is None
    assert ls_openai._coerce_value_for_type("raw", "ForwardRef") == "raw"
    assert ls_openai._coerce_value_for_type("not-a-color", _Color | int) == "not-a-color"
    assert ls_openai._coerce_value_for_type([1, 2], tuple[int, ...]) == (1, 2)
    assert ls_openai._coerce_value_for_type([1, 2, 3], tuple[int, int]) == (1, 2, 3)
    assert ls_openai._coerce_value_for_type({"k": "red"}, dict[str, _Color]) == {"k": _Color.RED}

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
    monkeypatch.setattr(ls_openai, "_get_pydantic", lambda: fake_pyd)
    assert ls_openai._dataclass_to_json_schema(_SimpleDC)["type"] == "object"
    assert ls_openai._instantiate_dataclass(_SimpleDC, {"x": 7}) == _SimpleDC(x=7)
    monkeypatch.setattr(ls_openai, "_get_pydantic", lambda: None)

    # _compile_response_format & _parse_structured_text & _usage_from_summary
    with pytest.raises(TypeError, match="unsupported response_format class"):
        ls_openai._compile_response_format(int)
    with pytest.raises(TypeError, match="response_format must be"):
        ls_openai._compile_response_format(123)
    assert ls_openai._parse_structured_text("{}", None) is None
    assert ls_openai._parse_structured_text('{"a": 1}', _V1Model) == {"v1": {"a": 1}}
    assert ls_openai._parse_structured_text('{"a": 1}', dict) == {"a": 1}
    with pytest.raises(ValueError, match="expected JSON object"):
        ls_openai._parse_structured_text("[1, 2]", _SimpleDC)
    assert ls_openai._usage_from_summary(None) is None

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

    # AsyncStreamMixin.__anext__ cancellation closes stream
    cancel_inner = _FakeTokenStream(["x"])
    cancel_stream = ls_openai.CompletionStream(cancel_inner, "id3", 100, "m")

    async def _cancelled_to_thread(fn: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(fn, "__name__", "") == "_step":
            raise asyncio.CancelledError
        return fn(*args, **kwargs)

    monkeypatch.setattr(ls_openai.asyncio, "to_thread", _cancelled_to_thread)
    with pytest.raises(asyncio.CancelledError):
        await anext(cancel_stream)
    assert cancel_inner.closed is True
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

    # Lazy resolve path + AsyncOpenAI.models.list()
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
        models_page = await aclient.models.list()
        assert [m.id for m in models_page] == ["lazy-spec"]
    assert lazy_backend.unloaded is True
