"""Isolated GPU worker subprocess client & entrypoint (`python -m layerstream._worker`).

Running `StreamEngine` inside a dedicated child process guarantees the parent Python
process (e.g., PyTorch, Jupyter, FastAPI) never initializes CUDA/Vulkan/Metal drivers.
Calling `.unload()` (or exiting `with AutoModel.from_pretrained(..., isolated_worker=True):`,
or discarding the `AutoModel` and its streams) terminates and reaps the worker subprocess
so the OS kernel immediately reclaims 100% of VRAM/UMA, CUDA primary driver contexts
(~300-524 MiB), and C++ heap arenas.
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import socket
import subprocess
import sys
import threading
import weakref
from collections.abc import Iterator, Mapping, Sequence
from os import PathLike
from types import TracebackType
from typing import Any

from ._layerstream import (
    NativeAutoModel,
    json_schema_to_gbnf,
    parse_assistant_output,
)


def _send_json_line(sock: socket.socket, lock: threading.Lock, obj: dict[str, Any]) -> None:
    raw = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
    with lock:
        sock.sendall(raw)


def _file_not_found_or_runtime(err_type: str, msg: str) -> BaseException:
    if err_type == "FileNotFoundError":
        return FileNotFoundError(msg)
    if err_type == "ValueError":
        return ValueError(msg)
    if err_type == "TypeError":
        return TypeError(msg)
    return RuntimeError(msg)


def _reap_proc(proc: subprocess.Popen[bytes]) -> None:
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=3.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=2.0)
        return
    except Exception:
        pass
    try:
        proc.kill()
        proc.wait(timeout=2.0)
    except Exception:
        pass


class _WorkerState:
    """Low-level IPC & subprocess state owned via `weakref.finalize` on `_IsolatedWorkerClient`."""

    def __init__(
        self,
        proc: subprocess.Popen[bytes],
        sock: socket.socket,
    ) -> None:
        self.proc: subprocess.Popen[bytes] | None = proc
        self.sock: socket.socket | None = sock
        self.send_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.next_id = 1
        self.pending_rpc: dict[int, queue.Queue[dict[str, Any]]] = {}
        self.active_streams: dict[int, queue.Queue[dict[str, Any]]] = {}
        self.loaded = False
        self.model_path: str | None = None
        self.last_summary: dict[str, Any] | None = None


def _shutdown_worker_state(state: _WorkerState) -> None:
    with state.state_lock:
        state.loaded = False
        state.model_path = None
        sock = state.sock
        state.sock = None
        proc = state.proc
        state.proc = None

    if sock is not None:
        try:
            _send_json_line(sock, state.send_lock, {"id": 0, "op": "shutdown", "params": {}})
        except Exception:
            pass
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass

    if proc is not None:
        _reap_proc(proc)


def _reader_loop(state: _WorkerState, reader_file: Any) -> None:
    try:
        for line in reader_file:
            if not line:
                break
            msg = json.loads(line)
            req_id = int(msg["id"])
            with state.state_lock:
                stream_q = state.active_streams.get(req_id)
                rpc_q = state.pending_rpc.get(req_id)
            if stream_q is not None:
                stream_q.put(msg)
            elif rpc_q is not None:
                rpc_q.put(msg)
    except Exception:
        pass
    finally:
        try:
            reader_file.close()
        except Exception:
            pass
        with state.state_lock:
            state.loaded = False
            eof_msg = {
                "ok": False,
                "event": "error",
                "err_type": "RuntimeError",
                "message": "isolated worker exited",
            }
            for q in state.pending_rpc.values():
                q.put(eof_msg)
            for q in state.active_streams.values():
                q.put(eof_msg)
            state.pending_rpc.clear()
            state.active_streams.clear()


class WorkerTokenStream(Iterator[str]):
    """Streaming token iterator backed by an isolated worker subprocess."""

    def __init__(
        self,
        client: _IsolatedWorkerClient,
        req_id: int,
        ev_queue: queue.Queue[dict[str, Any]],
    ) -> None:
        self._client: _IsolatedWorkerClient | None = client
        self._req_id = req_id
        self._queue = ev_queue
        self._done = False
        self._summary: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def __iter__(self) -> WorkerTokenStream:
        return self

    def __enter__(self) -> WorkerTokenStream:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> bool:
        self.close()
        return False

    @property
    def summary(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._summary) if self._summary is not None else None

    def close(self) -> None:
        with self._lock:
            if self._done:
                client = self._client
                self._client = None
                return
            self._done = True
            client = self._client
            self._client = None
        if client is not None:
            sum_out = client._cancel_stream(self._req_id, self._queue)
            if sum_out is not None:
                with self._lock:
                    self._summary = sum_out

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __next__(self) -> str:
        while True:
            with self._lock:
                if self._done:
                    raise StopIteration
                client = self._client
            if client is None:
                raise StopIteration
            try:
                msg = self._queue.get(timeout=0.05)
            except queue.Empty:
                if not client.is_loaded:
                    with self._lock:
                        self._done = True
                        self._client = None
                    raise StopIteration from None
                continue
            except KeyboardInterrupt:
                self.close()
                raise

            kind = msg.get("event")
            if kind == "chunk":
                return str(msg["text"])
            if kind == "done":
                summary = msg.get("summary")
                with self._lock:
                    self._done = True
                    self._summary = dict(summary) if isinstance(summary, Mapping) else None
                    self._client = None
                client._finish_stream(self._req_id, summary)
                raise StopIteration
            if kind == "error":
                with self._lock:
                    self._done = True
                    self._client = None
                client._finish_stream(self._req_id, None)
                raise _file_not_found_or_runtime(
                    str(msg.get("err_type", "RuntimeError")),
                    str(msg.get("message", "worker generation failed")),
                )


class _IsolatedWorkerClient:
    """Parent-side IPC client managing one isolated child `python -m layerstream._worker`."""

    def __init__(self, open_kwargs: dict[str, Any]) -> None:
        token = secrets.token_hex(32)
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        srv.settimeout(30.0)
        port = srv.getsockname()[1]

        env = os.environ.copy()
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{pkg_root}{os.pathsep}{existing_pp}" if existing_pp else pkg_root
        proc: subprocess.Popen[bytes] | None = None
        conn: socket.socket | None = None
        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import sys; from layerstream._worker import _run_worker_main; sys.exit(_run_worker_main(int(sys.argv[1])))",
                    str(port),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=None,
                env=env,
            )
            assert proc.stdin is not None
            proc.stdin.write((token + "\n").encode("ascii"))
            proc.stdin.flush()

            conn, _ = srv.accept()
            conn.settimeout(None)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception as exc:
            if conn is not None:
                conn.close()
            if proc is not None:
                _reap_proc(proc)
            raise RuntimeError(f"failed to start isolated layerstream worker: {exc}") from exc
        finally:
            srv.close()

        state = _WorkerState(proc, conn)
        self._state = state
        # Register weakref finalizer BEFORE starting the reader thread (which only captures
        # `state`, never `self`), so discarding `AutoModel` and its `WorkerTokenStream`s
        # deterministically shuts down IPC and reaps the child process even without `.unload()`.
        self._finalizer = weakref.finalize(self, _shutdown_worker_state, state)

        reader_file = conn.makefile("r", encoding="utf-8")
        hello_line = reader_file.readline()
        if not hello_line:
            self.unload()
            raise RuntimeError("isolated layerstream worker exited before handshake")
        hello = json.loads(hello_line)
        if hello.get("token") != token:
            self.unload()
            raise RuntimeError("isolated layerstream worker failed authentication handshake")

        self._reader_thread = threading.Thread(
            target=_reader_loop,
            args=(state, reader_file),
            name="layerstream-worker-ipc",
            daemon=True,
        )
        self._reader_thread.start()

        try:
            resp, _ = self._rpc_with_summary("open", open_kwargs)
            with state.state_lock:
                state.loaded = True
                state.model_path = resp.get("model_path")
        except BaseException:
            self.unload()
            raise

    def _alloc_id(self) -> int:
        with self._state.state_lock:
            rid = self._state.next_id
            self._state.next_id += 1
            return rid

    def _rpc(self, op: str, params: dict[str, Any]) -> dict[str, Any]:
        res, _ = self._rpc_with_summary(op, params)
        return res

    def _rpc_with_summary(
        self, op: str, params: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        state = self._state
        with state.state_lock:
            sock = state.sock
            if sock is None or (op != "open" and not state.loaded):
                raise RuntimeError("model has been unloaded")
            req_id = state.next_id
            state.next_id += 1
            q: queue.Queue[dict[str, Any]] = queue.Queue()
            state.pending_rpc[req_id] = q

        try:
            _send_json_line(sock, state.send_lock, {"id": req_id, "op": op, "params": params})
            while True:
                try:
                    resp = q.get(timeout=0.05)
                    break
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            try:
                _send_json_line(
                    sock,
                    state.send_lock,
                    {"id": req_id, "op": "cancel", "params": {"target_id": req_id}},
                )
                try:
                    q.get(timeout=2.0)
                except queue.Empty:
                    pass
            except Exception:
                pass
            raise
        finally:
            with state.state_lock:
                state.pending_rpc.pop(req_id, None)

        if not resp.get("ok", False):
            raise _file_not_found_or_runtime(
                str(resp.get("err_type", "RuntimeError")),
                str(resp.get("message", "worker request failed")),
            )
        raw_sum = resp.get("summary")
        sum_copy = dict(raw_sum) if isinstance(raw_sum, Mapping) else None
        if sum_copy is not None:
            with state.state_lock:
                state.last_summary = dict(sum_copy)
        return resp["result"], sum_copy

    def start_stream(self, op: str, params: dict[str, Any]) -> WorkerTokenStream:
        state = self._state
        with state.state_lock:
            sock = state.sock
            if sock is None or not state.loaded:
                raise RuntimeError("model has been unloaded")
            req_id = state.next_id
            state.next_id += 1
            ev_q: queue.Queue[dict[str, Any]] = queue.Queue()
            state.active_streams[req_id] = ev_q

        try:
            _send_json_line(sock, state.send_lock, {"id": req_id, "op": op, "params": params})
        except Exception:
            with state.state_lock:
                state.active_streams.pop(req_id, None)
            raise
        return WorkerTokenStream(self, req_id, ev_q)

    def _cancel_stream(
        self, req_id: int, ev_q: queue.Queue[dict[str, Any]]
    ) -> dict[str, Any] | None:
        state = self._state
        with state.state_lock:
            sock = state.sock
            still_active = req_id in state.active_streams
        if not still_active or sock is None:
            return None
        captured_summary: dict[str, Any] | None = None
        try:
            _send_json_line(
                sock,
                state.send_lock,
                {"id": self._alloc_id(), "op": "cancel", "params": {"target_id": req_id}},
            )
            while True:
                try:
                    msg = ev_q.get(timeout=2.0)
                except queue.Empty:
                    break
                if msg.get("event") in ("done", "error"):
                    raw_sum = msg.get("summary")
                    captured_summary = dict(raw_sum) if isinstance(raw_sum, Mapping) else None
                    self._finish_stream(req_id, captured_summary)
                    break
        except Exception:
            pass
        finally:
            with state.state_lock:
                state.active_streams.pop(req_id, None)
        return captured_summary

    def _finish_stream(self, req_id: int, summary: Mapping[str, Any] | None) -> None:
        with self._state.state_lock:
            self._state.active_streams.pop(req_id, None)
            if summary is not None:
                self._state.last_summary = dict(summary)

    @property
    def is_loaded(self) -> bool:
        with self._state.state_lock:
            return (
                self._state.loaded
                and self._state.proc is not None
                and self._state.proc.poll() is None
            )

    @property
    def worker_pid(self) -> int | None:
        with self._state.state_lock:
            if self._state.proc is not None and self._state.proc.poll() is None:
                return self._state.proc.pid
            return None

    @property
    def model_path(self) -> str | None:
        with self._state.state_lock:
            return self._state.model_path if self._state.loaded else None

    @property
    def last_summary(self) -> dict[str, Any] | None:
        with self._state.state_lock:
            return dict(self._state.last_summary) if self._state.last_summary is not None else None

    def stats(self) -> dict[str, Any]:
        if not self.is_loaded:
            return {"loaded": False}
        st = self._rpc("stats", {})
        st["isolated_worker"] = True
        st["worker_pid"] = self.worker_pid
        return st

    def unload(self) -> None:
        self._finalizer()


def normalize_open_kwargs(
    model: str,
    *,
    draft_model: str | None,
    quant: str | None,
    revision: str | None,
    cache_dir: str | PathLike[str] | None,
    vram_budget_mb: int | str | None,
    resident_layers: int | None,
    ctx_size: int,
    batch_size: int,
    threads: int | None,
    kv_cache_type: str,
    use_gpu: bool,
    prefer_direct_io: bool,
    cpu_embeddings: bool,
    cpu_head: bool,
    stream_head: bool,
    moe_active_streaming: bool,
    max_draft_tokens: int,
    verbose: bool,
) -> dict[str, Any]:
    if isinstance(vram_budget_mb, bool):
        raise TypeError(
            "vram_budget_mb must be 'auto', None, or a non-negative integer MiB (not bool)"
        )
    if isinstance(vram_budget_mb, int) and vram_budget_mb < 0:
        raise ValueError("vram_budget_mb must be non-negative")
    if isinstance(vram_budget_mb, str):
        raw = vram_budget_mb.strip()
        if raw.lower() != "auto":
            try:
                parsed_mb = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"invalid vram_budget_mb '{raw}' (expected 'auto' or non-negative integer MiB)"
                ) from exc
            if parsed_mb < 0:
                raise ValueError("vram_budget_mb must be non-negative")
    if kv_cache_type.strip().lower() not in ("f16", "q8_0", "q4_0"):
        raise ValueError(
            f"invalid kv_cache_type '{kv_cache_type}' (expected f16, q8_0, or q4_0)"
        )

    return {
        "model": model,
        "draft_model": draft_model,
        "quant": quant,
        "revision": revision,
        "cache_dir": os.fspath(cache_dir) if cache_dir is not None else None,
        "vram_budget_mb": vram_budget_mb,
        "resident_layers": resident_layers,
        "ctx_size": ctx_size,
        "batch_size": batch_size,
        "threads": threads,
        "kv_cache_type": kv_cache_type,
        "use_gpu": use_gpu,
        "prefer_direct_io": prefer_direct_io,
        "cpu_embeddings": cpu_embeddings,
        "cpu_head": cpu_head,
        "stream_head": stream_head,
        "moe_active_streaming": moe_active_streaming,
        "max_draft_tokens": max_draft_tokens,
        "verbose": verbose,
    }


def serialize_schema_arg(json_schema: Any) -> str | None:
    """Compile `json_schema` (dict, JSON str, or Pydantic class) to GBNF ahead of IPC."""
    if json_schema is None:
        return None
    return json_schema_to_gbnf(json_schema)


def validate_tools_arg(tools: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        raise TypeError("tools must be a sequence of tool dicts")
    out: list[dict[str, Any]] = []
    for item in tools:
        if not isinstance(item, Mapping):
            raise TypeError("each entry in 'tools' must be a dict/mapping")
        out.append(dict(item))
    return out


def serialize_messages_arg(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise TypeError("messages must be an iterable of {'role': str, 'content': str} dicts")
    out: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, Mapping):
            raise TypeError("each message must be a mapping")
        if "role" not in item:
            raise ValueError("each message must contain a 'role' string")
        role = item["role"]
        if not isinstance(role, str):
            raise TypeError("message 'role' must be a string")
        if "content" not in item and "tool_calls" not in item:
            raise ValueError("each message must contain a 'content' string")
        d: dict[str, Any] = {"role": role, "content": item.get("content")}
        if "tool_calls" in item and item["tool_calls"] is not None:
            d["tool_calls"] = list(item["tool_calls"])
        if "name" in item and item["name"] is not None:
            d["name"] = str(item["name"])
        if "tool_call_id" in item and item["tool_call_id"] is not None:
            d["tool_call_id"] = str(item["tool_call_id"])
        out.append(d)
    if not out:
        raise ValueError("messages must not be empty")
    return out


def _run_worker_main(port: int) -> int:
    token = sys.stdin.readline().strip()
    if not token:
        return 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect(("127.0.0.1", port))
    send_lock = threading.Lock()
    _send_json_line(sock, send_lock, {"token": token})

    cancel_lock = threading.Lock()
    cancelled_ids: set[int] = set()
    active_stream_holder: dict[int, Any] = {}
    work_q: queue.Queue[dict[str, Any] | None] = queue.Queue()

    # Exit immediately if parent process dies and closes stdin pipe.
    def watch_parent_stdin() -> None:
        try:
            while sys.stdin.read(64):
                pass
        except Exception:
            pass
        os._exit(0)

    threading.Thread(target=watch_parent_stdin, daemon=True).start()

    def engine_loop() -> None:
        model: NativeAutoModel | None = None
        while True:
            cmd = work_q.get()
            if cmd is None:
                if model is not None:
                    try:
                        model.unload()
                    except Exception:
                        pass
                break

            req_id = int(cmd["id"])
            op = str(cmd["op"])
            params: dict[str, Any] = cmd.get("params") or {}

            with cancel_lock:
                if req_id in cancelled_ids:
                    cancelled_ids.discard(req_id)
                    if op in ("generate_stream", "chat_stream"):
                        _send_json_line(
                            sock,
                            send_lock,
                            {"id": req_id, "event": "done", "summary": None},
                        )
                    else:
                        _send_json_line(
                            sock,
                            send_lock,
                            {"id": req_id, "ok": True, "result": {}, "summary": None},
                        )
                    continue

            try:
                if op == "open":
                    model = NativeAutoModel.from_pretrained(**params)
                    _send_json_line(
                        sock,
                        send_lock,
                        {
                            "id": req_id,
                            "ok": True,
                            "result": {"model_path": model.model_path},
                        },
                    )
                elif model is None:
                    raise RuntimeError("model has been unloaded")
                elif op == "stats":
                    _send_json_line(
                        sock,
                        send_lock,
                        {"id": req_id, "ok": True, "result": model.stats()},
                    )
                elif op == "tokenize":
                    ids = model.tokenize(
                        params["text"],
                        add_special=bool(params.get("add_special", False)),
                        parse_special=bool(params.get("parse_special", False)),
                    )
                    _send_json_line(
                        sock,
                        send_lock,
                        {"id": req_id, "ok": True, "result": {"tokens": ids}},
                    )
                elif op == "detokenize":
                    text = model.detokenize(list(params["tokens"]))
                    _send_json_line(
                        sock,
                        send_lock,
                        {"id": req_id, "ok": True, "result": {"text": text}},
                    )
                elif op == "apply_chat_template":
                    rendered = model.apply_chat_template(
                        params["messages"],
                        add_generation_prompt=bool(params.get("add_generation_prompt", True)),
                        tools=params.get("tools"),
                    )
                    _send_json_line(
                        sock,
                        send_lock,
                        {"id": req_id, "ok": True, "result": {"prompt": rendered}},
                    )
                elif op == "kv_clear":
                    model.kv_clear()
                    _send_json_line(sock, send_lock, {"id": req_id, "ok": True, "result": {}})
                elif op == "kv_trim":
                    ok = model.kv_trim(int(params["pos_min"]))
                    _send_json_line(
                        sock,
                        send_lock,
                        {"id": req_id, "ok": True, "result": {"trimmed": ok}},
                    )
                elif op in ("generate", "chat"):
                    parse_out = bool(params.pop("parse_output", False))
                    # Run via NativeTokenStream so mid-request cancel aborts C++ immediately.
                    if op == "generate":
                        stream_iter = model.generate(
                            params.pop("prompt"), stream=True, **params
                        )
                    else:
                        stream_iter = model.chat(
                            params.pop("messages"), stream=True, **params
                        )
                    with cancel_lock:
                        active_stream_holder[req_id] = stream_iter
                        cancelled_before_reg = req_id in cancelled_ids
                    if cancelled_before_reg:
                        stream_iter.close()
                    chunks: list[str] = []
                    try:
                        for piece in stream_iter:
                            with cancel_lock:
                                if req_id in cancelled_ids:
                                    stream_iter.close()
                                    break
                            chunks.append(piece)
                    finally:
                        with cancel_lock:
                            active_stream_holder.pop(req_id, None)
                            cancelled_ids.discard(req_id)
                        stream_iter.close()
                    text_out = "".join(chunks)
                    result_payload: dict[str, Any]
                    if parse_out:
                        result_payload = {"parsed": parse_assistant_output(text_out)}
                    else:
                        result_payload = {"text": text_out}
                    req_summary = stream_iter.summary or model.last_summary
                    _send_json_line(
                        sock,
                        send_lock,
                        {
                            "id": req_id,
                            "ok": True,
                            "result": result_payload,
                            "summary": req_summary,
                        },
                    )
                elif op in ("generate_stream", "chat_stream"):
                    if op == "generate_stream":
                        stream_iter = model.generate(
                            params.pop("prompt"), stream=True, **params
                        )
                    else:
                        stream_iter = model.chat(
                            params.pop("messages"), stream=True, **params
                        )
                    with cancel_lock:
                        active_stream_holder[req_id] = stream_iter
                        cancelled_before_reg = req_id in cancelled_ids
                    if cancelled_before_reg:
                        stream_iter.close()
                    try:
                        for piece in stream_iter:
                            with cancel_lock:
                                if req_id in cancelled_ids:
                                    stream_iter.close()
                                    break
                            _send_json_line(
                                sock,
                                send_lock,
                                {"id": req_id, "event": "chunk", "text": piece},
                            )
                    finally:
                        with cancel_lock:
                            active_stream_holder.pop(req_id, None)
                            cancelled_ids.discard(req_id)
                        stream_iter.close()
                    req_summary = stream_iter.summary or model.last_summary
                    _send_json_line(
                        sock,
                        send_lock,
                        {"id": req_id, "event": "done", "summary": req_summary},
                    )
                else:
                    raise ValueError(f"unknown worker op '{op}'")
            except Exception as exc:
                err_name = type(exc).__name__
                if op in ("generate_stream", "chat_stream"):
                    _send_json_line(
                        sock,
                        send_lock,
                        {
                            "id": req_id,
                            "event": "error",
                            "err_type": err_name,
                            "message": str(exc),
                        },
                    )
                else:
                    _send_json_line(
                        sock,
                        send_lock,
                        {
                            "id": req_id,
                            "ok": False,
                            "err_type": err_name,
                            "message": str(exc),
                        },
                    )

    worker_thread = threading.Thread(target=engine_loop, name="layerstream-engine")
    worker_thread.start()

    reader = sock.makefile("r", encoding="utf-8")
    try:
        for line in reader:
            if not line:
                break
            msg = json.loads(line)
            op = msg.get("op")
            if op == "shutdown":
                break
            if op == "cancel":
                target_id = int(msg["params"]["target_id"])
                with cancel_lock:
                    cancelled_ids.add(target_id)
                    live = active_stream_holder.get(target_id)
                if live is not None:
                    try:
                        live.close()
                    except Exception:
                        pass
                continue
            work_q.put(msg)
    except Exception:
        pass
    finally:
        with cancel_lock:
            for live in list(active_stream_holder.values()):
                try:
                    live.close()
                except Exception:
                    pass
        work_q.put(None)
        worker_thread.join(timeout=3.0)
        try:
            reader.close()
            sock.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(2)
    sys.exit(_run_worker_main(int(sys.argv[1])))
