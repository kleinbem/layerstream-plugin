from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from layerstream import AutoModel, TokenStream, __version__, probe_gpu_memory


def test_version_and_probe() -> None:
    assert isinstance(__version__, str) and len(__version__) > 0
    mem = probe_gpu_memory(use_gpu=False)
    assert mem is None or (isinstance(mem, tuple) and len(mem) == 2)


def test_load_generate_stream_chat_stats_and_unload(synthetic_gguf: Path) -> None:
    with AutoModel.from_pretrained(
        str(synthetic_gguf),
        use_gpu=False,
        ctx_size=128,
        batch_size=64,
        vram_budget_mb=0,
        kv_cache_type="q8_0",
        prefer_direct_io=False,
    ) as model:
        assert model.is_loaded
        assert model.model_path == str(synthetic_gguf)

        st = model.stats()
        assert st["loaded"] is True
        assert st["n_layers"] == 6
        assert st["moe_layers"] == 2
        assert st["moe_active_streaming"] is True
        assert st["resident_layers"] == 0
        assert st["streamed_layers"] == 6
        assert st["using_gpu"] is False
        assert "io" in st and isinstance(st["io"]["bytes_read"], int)

        # Deterministic greedy generation (non-streaming)
        out1 = model.generate("a b c d e f", max_tokens=10, temperature=0.0)
        assert isinstance(out1, str) and len(out1) > 0
        summary = model.last_summary
        assert summary is not None
        assert summary["completion_tokens"] == 10
        assert summary["finish_reason"] in ("length", "stop")

        # Streaming must match greedy output identically
        stream = model.generate(
            "a b c d e f", max_tokens=10, temperature=0.0, stream=True
        )
        assert isinstance(stream, TokenStream)
        chunks = list(stream)
        assert "".join(chunks) == out1

        # Early stream break cancels cleanly and releases the model lock
        early_stream = model.generate(
            "a b c d e f", max_tokens=20, temperature=0.0, stream=True
        )
        with early_stream as s:
            first = next(s)
            assert isinstance(first, str)

        # Chat template + multi-turn generation works right after stream cancellation
        reply = model.chat(
            [{"role": "user", "content": "a b c"}],
            max_tokens=8,
            temperature=0.0,
        )
        assert isinstance(reply, str) and len(reply) > 0

        # Stats record actual layer streaming I/O
        st_after = model.stats()
        assert st_after["total_layer_loads"] > 0
        assert st_after["io"]["bytes_read"] > 0

    # Context manager exited -> model deterministically unloaded
    assert not model.is_loaded
    assert model.stats() == {"loaded": False}
    with pytest.raises(RuntimeError, match="unloaded"):
        model.generate("a b c", max_tokens=4)


def test_gil_released_during_generation(synthetic_gguf: Path) -> None:
    """Verify a background Python thread makes progress concurrently while Rust/C++ generates."""
    with AutoModel.from_pretrained(
        str(synthetic_gguf),
        use_gpu=False,
        ctx_size=128,
        batch_size=64,
        vram_budget_mb=0,
        prefer_direct_io=False,
    ) as model:
        done = threading.Event()
        ticks = 0

        def ticker() -> None:
            nonlocal ticks
            while not done.is_set():
                ticks += 1
                time.sleep(0.001)

        worker = threading.Thread(target=ticker)
        worker.start()
        try:
            for _ in range(4):
                out = model.generate(
                    "a b c d e f a b c d",
                    max_tokens=24,
                    temperature=0.0,
                    ngram_speculation=False,
                )
                assert len(out) > 0
        finally:
            done.set()
            worker.join(timeout=5.0)

        assert ticks > 0, "background Python thread made zero progress (GIL was held)"


def test_from_pretrained_hf_uri_and_base_repo_resolution(
    synthetic_gguf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = synthetic_gguf.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    tree_gguf = json.dumps(
        [
            {
                "type": "file",
                "path": "Synthetic-Q4_K_M.gguf",
                "size": len(payload),
                "lfs": {"size": len(payload)},
            }
        ]
    ).encode("utf-8")
    tree_base = json.dumps(
        [{"type": "file", "path": "model.safetensors", "size": 128}]
    ).encode("utf-8")

    class MockHfHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/models/MockOrg/TinyBase/tree/main?recursive=true":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(tree_base)))
                self.end_headers()
                self.wfile.write(tree_base)
            elif (
                self.path
                == "/api/models/MockOrg/TinyBase-GGUF/tree/main?recursive=true"
            ):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(tree_gguf)))
                self.end_headers()
                self.wfile.write(tree_gguf)
            elif (
                self.path
                == "/MockOrg/TinyBase-GGUF/resolve/main/Synthetic-Q4_K_M.gguf"
            ):
                self.send_response(200)
                self.send_header("ETag", f'"{sha}"')
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif self.path == "/api/models/MockOrg/Missing/tree/main?recursive=true":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(tree_base)))
                self.end_headers()
                self.wfile.write(tree_base)
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

    server = HTTPServer(("127.0.0.1", 0), MockHfHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{port}")
        cache_dir = tmp_path / "hf-cache"

        # 1. Base HF repo resolves to official `-GGUF` sibling, downloads, and generates!
        with AutoModel.from_pretrained(
            "MockOrg/TinyBase",
            cache_dir=cache_dir,
            use_gpu=False,
            ctx_size=128,
            batch_size=64,
            vram_budget_mb=0,
            prefer_direct_io=False,
        ) as model:
            st = model.stats()
            assert st["resolved_repo"] == "MockOrg/TinyBase-GGUF"
            assert st["selected_quant"] == "Q4_K_M"
            out = model.generate("a b c", max_tokens=6)
            assert len(out) > 0

        # 2. Explicit hf:// URI hits local cache and opens cleanly
        with AutoModel.from_pretrained(
            "hf://MockOrg/TinyBase-GGUF:Synthetic-Q4_K_M.gguf",
            cache_dir=cache_dir,
            use_gpu=False,
            ctx_size=128,
            batch_size=64,
            vram_budget_mb=0,
            prefer_direct_io=False,
        ) as model2:
            assert model2.is_loaded

        # 3. Missing -GGUF sibling produces actionable guidance
        with pytest.raises(RuntimeError, match="MockOrg/Missing-GGUF"):
            AutoModel.from_pretrained(
                "MockOrg/Missing",
                cache_dir=cache_dir,
                use_gpu=False,
            )

        # 4. Missing local .gguf raises FileNotFoundError immediately
        with pytest.raises(FileNotFoundError):
            AutoModel.from_pretrained(str(tmp_path / "nonexistent.gguf"), use_gpu=False)
    finally:
        server.shutdown()
        server.server_close()
