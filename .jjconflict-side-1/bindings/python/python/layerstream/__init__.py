"""Native Python 3.15+ SDK for `layerstream-core` and `layerstream-server`."""

import atexit
import contextlib
import ctypes
import os
import sys
import weakref
from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Protocol, Unpack, cast, overload, override
lazy import asyncio

_DLL_DIR_HANDLES: list[Any] = []


def _configure_native_backends() -> list[str]:
    """Preload companion CUDA 13 runtimes (if installed) and configure dynamic ggml backend paths."""
    rtld_global = getattr(os, "RTLD_GLOBAL", ctypes.RTLD_GLOBAL)
    pkg_dir = Path(__file__).resolve().parent
    local_backends = pkg_dir / "_backends"

    # On Windows, register `_backends` with the DLL loader so `llama.dll` / `ggml*.dll` resolve.
    # coverage: Windows-only DLL loader API; POSIX `pathlib.Path` raises UnsupportedOperation if `os.name == "nt"`.
    if (
        os.name == "nt" and local_backends.is_dir() and hasattr(os, "add_dll_directory")
    ):  # pragma: no cover
        with contextlib.suppress(OSError):
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(local_backends)))

    # On NixOS, store-built `python3` ignores `NIX_LD_LIBRARY_PATH` and `/run/opengl-driver/lib`
    # when `dlopen`ing portable manylinux wheels; preload host C++/Vulkan/CUDA driver libraries if present.
    nix_host_dirs = [
        p
        for raw_dir in os.environ.get("NIX_LD_LIBRARY_PATH", "").split(os.pathsep)
        if (stripped := raw_dir.strip()) and (p := Path(stripped)).is_dir()
    ]
    opengl_dir = Path("/run/opengl-driver/lib")
    # coverage: NixOS-only host driver path; absent on standard Linux/macOS/Windows runners.
    if opengl_dir.is_dir():  # pragma: no cover
        nix_host_dirs.append(opengl_dir)
    for lib_name in ("libstdc++.so.6", "libvulkan.so.1", "libcuda.so.1"):
        for host_dir in nix_host_dirs:
            candidate = host_dir / lib_name
            if candidate.is_file():
                with contextlib.suppress(OSError):
                    ctypes.CDLL(str(candidate), mode=rtld_global)
                break

    # Preload bundled `libggml-base` with RTLD_GLOBAL so companion `layerstream-cuda13`
    # resolves `libggml-base.so.0` even when installed in a separate site-packages tree.
    if local_backends.is_dir():
        for pattern in ("libggml-base.so*", "libggml.so*"):
            for match in sorted(local_backends.glob(pattern)):
                if match.is_file():
                    try:
                        ctypes.CDLL(str(match), mode=rtld_global)
                        break
                    except OSError:
                        continue

    dirs: list[str] = []

    def _add(p: str | Path | None) -> None:
        if p and Path(p).is_dir() and (s := str(p)) not in dirs:
            dirs.append(s)

    existing_list = os.environ.get("LAYERSTREAM_BACKEND_DIRS", "").strip()
    if existing_list:
        for entry in existing_list.split(os.pathsep):
            _add(entry.strip())

    _add(os.environ.get("LAYERSTREAM_BACKEND_DIR", "").strip() or None)
    _add(os.environ.get("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", "").strip() or None)

    with contextlib.suppress(ImportError, OSError):
        import importlib

        cuda_plugin: Any = importlib.import_module("layerstream_cuda13")
        cuda_plugin.preload_nvidia_libraries()
        _add(str(cuda_plugin.backend_dir()))

    _add(local_backends)
    if dirs:
        os.environ["LAYERSTREAM_BACKEND_DIRS"] = os.pathsep.join(dirs)
    return dirs


_DISCOVERED_BACKEND_DIRS: list[str] = _configure_native_backends()


def configured_backend_dirs() -> list[str]:
    """Return the priority-ordered native ggml backend plugin directories active in this process."""
    raw = os.environ.get("LAYERSTREAM_BACKEND_DIRS", "").strip()
    if raw:
        return [p for p in raw.split(os.pathsep) if p]
    return list(_DISCOVERED_BACKEND_DIRS)


from ._layerstream import (  # noqa: E402
    NativeAutoModel as _NativeAutoModel,
    NativeWorkerClient as _NativeWorkerClient,
    NativeWorkerStream as _WorkerTokenStream,
    ServerHandle,
    StreamTagParser,
    TokenStream as _NativeTokenStream,
    __version__,
    check,
    format_tools_system_prompt,
    inspect_gguf,
    json_object_gbnf,
    json_schema_to_gbnf,
    parse_assistant_output,
    probe_gpu_device,
    probe_gpu_memory,
    pull,
    recommend,
    start_server,
)
from ._streams import AsyncStreamMixin as _AsyncStreamMixin  # noqa: E402
from ._types import (  # noqa: E402
    CheckReportDict,
    EngineStatsDict,
    GenerateSummaryDict,
    GgufInfoDict,
    IoStatsDict,
    LayerStreamError,
    ModelLoadOptions,
    ParsedAssistantOutputDict,
    ProgressEventDict,
    QuantRecommendationDict,
    SamplingOptions,
    SpeculationStatsDict,
    ToolCallDict,
    ToolCallFunctionDict,
)
from .openai import AsyncOpenAI, OpenAI  # noqa: E402


def serve(
    model: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    api_key: str | None = None,
    idle_timeout_secs: int = 60,
    **options: Unpack[ModelLoadOptions],
) -> None:
    """Run `layerstream-server` (native Rust `axum` + `tokio` OpenAI HTTP/SSE server) until interrupted."""
    with start_server(
        model,
        host=host,
        port=port,
        api_key=api_key,
        idle_timeout_secs=idle_timeout_secs,
        **options,
    ) as handle:
        handle.wait()


_LIVE_STREAMS: weakref.WeakSet[TokenStream] = weakref.WeakSet()
_LIVE_MODELS: weakref.WeakSet[AutoModel] = weakref.WeakSet()


@atexit.register
def _cleanup_live_resources() -> None:
    """Close active streams and unload live models during `Py_FinalizeEx()` before C `exit()`
    runs native C++ static destructors (e.g. `ggml-metal` residency sets on macOS)."""
    for s in list(_LIVE_STREAMS):
        with contextlib.suppress(Exception):
            s.close()
    for m in list(_LIVE_MODELS):
        with contextlib.suppress(Exception):
            m.unload()


class TokenStream(_AsyncStreamMixin[str]):
    """Synchronous (`for chunk in s`) and asynchronous (`async for chunk in s`) token stream."""

    def __init__(self, inner: _NativeTokenStream | _WorkerTokenStream) -> None:
        self._inner = inner
        _LIVE_STREAMS.add(self)

    @override
    def __next__(self) -> str:
        return next(self._inner)

    @override
    def close(self) -> None:
        self._inner.close()

    @property
    def summary(self) -> GenerateSummaryDict | None:
        return cast(GenerateSummaryDict | None, self._inner.summary)


class _ModelBackend(Protocol):
    """Structural contract shared by `_NativeAutoModel` and `_IsolatedWorkerClient`."""

    @property
    def is_loaded(self) -> bool: ...
    @property
    def model_path(self) -> str | None: ...
    @property
    def last_summary(self) -> dict[str, Any] | None: ...
    def stats(self) -> dict[str, Any]: ...
    def unload(self) -> None: ...
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        stream: bool = False,
        grammar: str | None = None,
        json_schema: Any = None,
        return_summary: bool = False,
        **sampling: Unpack[SamplingOptions],
    ) -> Any: ...
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 256,
        stream: bool = False,
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        parse_output: bool = False,
        return_summary: bool = False,
        **sampling: Unpack[SamplingOptions],
    ) -> Any: ...
    def tokenize(
        self, text: str, *, add_special: bool = False, parse_special: bool = False
    ) -> list[int]: ...
    def detokenize(self, tokens: Sequence[int]) -> str: ...
    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> str: ...
    def kv_clear(self) -> None: ...
    def kv_trim(self, pos_min: int) -> bool: ...
    def save_state(self, path: str) -> int: ...
    def load_state(self, path: str) -> int: ...


class AutoModel:
    """Out-of-core GGUF layer-streaming model (in-process or `isolated_worker=True`)."""

    def __init__(
        self,
        backend: _ModelBackend,
        *,
        isolated_worker: bool,
    ) -> None:
        self._backend: _ModelBackend = backend
        self._isolated = isolated_worker
        _LIVE_MODELS.add(self)

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        draft_model: str | Literal["auto"] | None = None,
        quant: str | None = None,
        revision: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        vram_budget_mb: int | str | None = "auto",
        resident_layers: int | None = None,
        ctx_size: int = 4096,
        batch_size: int = 512,
        threads: int | None = None,
        kv_cache_type: Literal["f16", "q8_0", "q4_0"] = "q8_0",
        use_gpu: bool = True,
        prefer_direct_io: bool = True,
        cpu_embeddings: bool = False,
        cpu_head: bool = False,
        stream_head: bool = False,
        moe_active_streaming: bool = True,
        max_draft_tokens: int = 5,
        verbose: bool = False,
        isolated_worker: bool = False,
        on_progress: Callable[[dict[str, Any]], Any] | None = None,
    ) -> AutoModel:
        if on_progress is not None and not callable(on_progress):
            raise TypeError("on_progress must be None or a callable accepting a progress dict")
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
                on_progress=on_progress,
            )
            return cls(native, isolated_worker=False)

        worker = _NativeWorkerClient.spawn(
            sys.executable,
            ["-c", "from layerstream._worker import _main; _main()"],
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
            on_progress=on_progress,
        )
        return cls(worker, isolated_worker=True)

    @overload
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        stream: Literal[False] = False,
        grammar: str | None = None,
        json_schema: Any = None,
        **sampling: Unpack[SamplingOptions],
    ) -> str: ...

    @overload
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        stream: Literal[True],
        grammar: str | None = None,
        json_schema: Any = None,
        **sampling: Unpack[SamplingOptions],
    ) -> TokenStream: ...

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        stream: bool = False,
        grammar: str | None = None,
        json_schema: Any = None,
        **sampling: Unpack[SamplingOptions],
    ) -> str | TokenStream:
        res = self._backend.generate(
            prompt,
            max_tokens=max_tokens,
            stream=stream,
            grammar=grammar,
            json_schema=json_schema,
            **sampling,
        )
        if isinstance(res, (_NativeTokenStream, _WorkerTokenStream)):
            return TokenStream(res)
        return str(res)

    def _generate_with_summary(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any] | None]:
        text, summary = self._backend.generate(prompt, stream=False, return_summary=True, **kwargs)
        return str(text), summary

    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 256,
        stream: Literal[False] = False,
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        parse_output: Literal[False] = False,
        **sampling: Unpack[SamplingOptions],
    ) -> str: ...

    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 256,
        stream: Literal[False] = False,
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        parse_output: Literal[True],
        **sampling: Unpack[SamplingOptions],
    ) -> ParsedAssistantOutputDict: ...

    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 256,
        stream: Literal[True],
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        parse_output: Literal[False] = False,
        **sampling: Unpack[SamplingOptions],
    ) -> TokenStream: ...

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 256,
        stream: bool = False,
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        parse_output: bool = False,
        **sampling: Unpack[SamplingOptions],
    ) -> str | ParsedAssistantOutputDict | TokenStream:
        res = self._backend.chat(
            messages,
            max_tokens=max_tokens,
            stream=stream,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            grammar=grammar,
            json_schema=json_schema,
            parse_output=parse_output,
            **sampling,
        )
        if isinstance(res, (_NativeTokenStream, _WorkerTokenStream)):
            return TokenStream(res)
        return cast(str | ParsedAssistantOutputDict, res)

    def _chat_with_summary(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        parse_output: bool = False,
        **kwargs: Any,
    ) -> tuple[str | dict[str, Any], dict[str, Any] | None]:
        payload, summary = self._backend.chat(
            messages,
            stream=False,
            parse_output=parse_output,
            return_summary=True,
            **kwargs,
        )
        return payload, summary

    def tokenize(
        self,
        text: str,
        *,
        add_special: bool = False,
        parse_special: bool = False,
    ) -> list[int]:
        return self._backend.tokenize(text, add_special=add_special, parse_special=parse_special)

    def detokenize(self, tokens: Sequence[int]) -> str:
        return self._backend.detokenize(list(tokens))

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        return self._backend.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tools=tools
        )

    def kv_clear(self) -> None:
        self._backend.kv_clear()

    def kv_trim(self, pos_min: int) -> bool:
        if pos_min < 0:
            raise ValueError("pos_min must be non-negative")
        return self._backend.kv_trim(pos_min)

    def save_state(self, path: str | PathLike[str]) -> int:
        """Persist KV cache memory and committed token sequence to `path` on disk."""
        raw_path = str(Path(path).resolve())
        return int(self._backend.save_state(raw_path))

    def load_state(self, path: str | PathLike[str]) -> int:
        """Restore a KV cache snapshot and committed token sequence from `path`."""
        resolved = Path(path).resolve()
        raw_path = str(resolved)
        if not resolved.is_file():
            raise FileNotFoundError(f"KV state file not found: '{raw_path}'")
        return int(self._backend.load_state(raw_path))

    def stats(self) -> EngineStatsDict:
        st = self._backend.stats()
        if st.get("loaded"):
            st.setdefault("isolated_worker", False)
            st.setdefault("worker_pid", None)
        return cast(EngineStatsDict, st)

    @property
    def last_summary(self) -> GenerateSummaryDict | None:
        return cast(GenerateSummaryDict | None, self._backend.last_summary)

    @property
    def is_loaded(self) -> bool:
        return self._backend.is_loaded

    @property
    def isolated_worker(self) -> bool:
        return self._isolated

    @property
    def worker_pid(self) -> int | None:
        if isinstance(self._backend, _NativeWorkerClient):
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
    ) -> None:
        self.unload()

    async def __aenter__(self) -> AutoModel:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None:
        await asyncio.to_thread(self.unload)

    @override
    def __repr__(self) -> str:
        p = self.model_path
        if p is not None:
            return f"AutoModel(loaded=True, isolated_worker={self._isolated}, model_path={p!r})"
        return "AutoModel(loaded=False)"


__all__ = [
    "AsyncOpenAI",
    "AutoModel",
    "CheckReportDict",
    "EngineStatsDict",
    "GenerateSummaryDict",
    "GgufInfoDict",
    "IoStatsDict",
    "LayerStreamError",
    "ModelLoadOptions",
    "OpenAI",
    "ParsedAssistantOutputDict",
    "ProgressEventDict",
    "QuantRecommendationDict",
    "SamplingOptions",
    "ServerHandle",
    "SpeculationStatsDict",
    "StreamTagParser",
    "TokenStream",
    "ToolCallDict",
    "ToolCallFunctionDict",
    "__version__",
    "check",
    "configured_backend_dirs",
    "format_tools_system_prompt",
    "inspect_gguf",
    "json_object_gbnf",
    "json_schema_to_gbnf",
    "parse_assistant_output",
    "probe_gpu_device",
    "probe_gpu_memory",
    "pull",
    "recommend",
    "serve",
    "start_server",
]
