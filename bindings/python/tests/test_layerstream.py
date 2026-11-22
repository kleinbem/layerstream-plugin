from __future__ import annotations

import gc
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

        # 2b. Explicit quant override works, and unknown quant lists available quants
        with AutoModel.from_pretrained(
            "MockOrg/TinyBase-GGUF",
            quant="Q4_K_M",
            cache_dir=cache_dir,
            use_gpu=False,
            ctx_size=128,
            batch_size=64,
            vram_budget_mb=0,
            prefer_direct_io=False,
        ) as model3:
            assert model3.stats()["selected_quant"] == "Q4_K_M"

        with pytest.raises(RuntimeError, match="Q4_K_M"):
            AutoModel.from_pretrained(
                "MockOrg/TinyBase-GGUF",
                quant="Q2_K",
                cache_dir=cache_dir,
                use_gpu=False,
            )

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


def test_draft_model_speculation_parity_and_incompatible_rejection(
    synthetic_gguf: Path,
    incompatible_vocab_gguf: Path,
) -> None:
    prompt = "a b c d e f a b c"
    with AutoModel.from_pretrained(
        str(synthetic_gguf),
        use_gpu=False,
        ctx_size=128,
        batch_size=64,
        vram_budget_mb=0,
        prefer_direct_io=False,
    ) as baseline:
        expected = baseline.generate(
            prompt, max_tokens=12, temperature=0.0, ngram_speculation=False
        )

    # Identical draft model must preserve exact greedy output and accept drafted tokens.
    with AutoModel.from_pretrained(
        str(synthetic_gguf),
        draft_model=str(synthetic_gguf),
        max_draft_tokens=4,
        use_gpu=False,
        ctx_size=128,
        batch_size=64,
        vram_budget_mb=0,
        prefer_direct_io=False,
    ) as spec_model:
        st = spec_model.stats()
        assert st["draft_model_path"] == str(synthetic_gguf)
        actual = spec_model.generate(
            prompt, max_tokens=12, temperature=0.0, ngram_speculation=False
        )
        assert actual == expected
        summary = spec_model.last_summary
        assert summary is not None
        assert summary["speculation"]["drafted_tokens"] > 0
        assert summary["speculation"]["accepted_tokens"] > 0

    # Incompatible draft vocabulary (512 vs 64) must fail closed on open.
    with pytest.raises(RuntimeError, match="incompatible draft_model tokenizer"):
        AutoModel.from_pretrained(
            str(synthetic_gguf),
            draft_model=str(incompatible_vocab_gguf),
            use_gpu=False,
            ctx_size=128,
            batch_size=64,
            vram_budget_mb=0,
            prefer_direct_io=False,
        )


def test_stop_sequences_and_sampling_seed_determinism(synthetic_gguf: Path) -> None:
    with AutoModel.from_pretrained(
        str(synthetic_gguf),
        use_gpu=False,
        ctx_size=128,
        batch_size=64,
        vram_budget_mb=0,
        prefer_direct_io=False,
    ) as model:
        full = model.generate("a b c d e f", max_tokens=16, temperature=0.0)
        assert len(full) >= 6
        needle = full[3:5]
        expected_prefix = full[: full.index(needle)]

        # Single-string stop sequence (non-streaming)
        stopped_str = model.generate(
            "a b c d e f", max_tokens=16, temperature=0.0, stop=needle
        )
        assert stopped_str == expected_prefix
        assert needle not in stopped_str
        assert model.last_summary is not None
        assert model.last_summary["finish_reason"] == "stop"

        # Sequence stop (streaming) must also hold back and never leak the stop string
        stopped_stream = "".join(
            model.generate(
                "a b c d e f",
                max_tokens=16,
                temperature=0.0,
                stop=[needle],
                stream=True,
            )
        )
        assert stopped_stream == expected_prefix
        assert needle not in stopped_stream

        # Sampled generation (temperature > 0) must be deterministic for a fixed seed
        s1 = model.generate(
            "a b c d", max_tokens=10, temperature=0.8, top_p=0.9, top_k=20, seed=12345
        )
        s2 = model.generate(
            "a b c d", max_tokens=10, temperature=0.8, top_p=0.9, top_k=20, seed=12345
        )
        assert s1 == s2 and len(s1) > 0


def test_implicit_stream_drop_cancellation_and_concurrent_threads(
    synthetic_gguf: Path,
) -> None:
    with AutoModel.from_pretrained(
        str(synthetic_gguf),
        use_gpu=False,
        ctx_size=128,
        batch_size=64,
        vram_budget_mb=0,
        prefer_direct_io=False,
    ) as model:
        expected = model.generate("a b c d", max_tokens=8, temperature=0.0)

        # Break out of a stream without `with` / `.close()` -> Drop cancels background worker
        for chunk in model.generate(
            "a b c d", max_tokens=24, temperature=0.0, stream=True
        ):
            assert isinstance(chunk, str)
            break
        gc.collect()

        # Subsequent call acquires the engine lock cleanly without deadlock
        after = model.generate("a b c d", max_tokens=8, temperature=0.0)
        assert after == expected

        # Two threads calling `.generate()` on the same AutoModel serialize cleanly
        def run_once() -> str:
            return model.generate("a b c d", max_tokens=8, temperature=0.0)

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(run_once)
            f2 = pool.submit(run_once)
            assert f1.result(timeout=30.0) == expected
            assert f2.result(timeout=30.0) == expected


def test_input_validation_and_unload_idempotency(synthetic_gguf: Path) -> None:
    with pytest.raises(TypeError):
        AutoModel.from_pretrained(
            str(synthetic_gguf), vram_budget_mb=True, use_gpu=False  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        AutoModel.from_pretrained(str(synthetic_gguf), vram_budget_mb=-1, use_gpu=False)
    with pytest.raises(ValueError):
        AutoModel.from_pretrained(
            str(synthetic_gguf), vram_budget_mb="not-a-number", use_gpu=False  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        AutoModel.from_pretrained(
            str(synthetic_gguf), kv_cache_type="bogus", use_gpu=False  # type: ignore[arg-type]
        )

    # Explicit vram_budget_mb=None means pure streaming (resident_layers == 0)
    model = AutoModel.from_pretrained(
        str(synthetic_gguf),
        use_gpu=False,
        ctx_size=128,
        batch_size=64,
        vram_budget_mb=None,
        prefer_direct_io=False,
    )
    try:
        assert model.stats()["resident_layers"] == 0

        with pytest.raises(ValueError, match="empty"):
            model.chat([])
        with pytest.raises(ValueError, match="content"):
            model.chat([{"role": "user"}])
        with pytest.raises(TypeError, match="role"):
            model.chat([{"role": 123, "content": "hi"}])  # type: ignore[list-item]
        with pytest.raises(TypeError, match="stop"):
            model.generate("a b", max_tokens=4, stop=123)  # type: ignore[arg-type]
    finally:
        model.unload()
        model.unload()  # idempotent

    assert not model.is_loaded
    with pytest.raises(RuntimeError, match="unloaded"):
        model.generate("a b", max_tokens=4, stream=True)
    with pytest.raises(RuntimeError, match="unloaded"):
        model.chat([{"role": "user", "content": "a b"}], max_tokens=4)


def test_sigint_cancels_streaming_and_non_streaming_generation(
    synthetic_gguf: Path,
) -> None:
    """SIGINT must interrupt both stream=False and stream=True and leave the model usable."""
    with AutoModel.from_pretrained(
        str(synthetic_gguf),
        use_gpu=False,
        ctx_size=256,
        batch_size=64,
        vram_budget_mb=0,
        prefer_direct_io=False,
    ) as model:
        # 1. Non-streaming generation must honor SIGINT via py.check_signals().
        timer = threading.Timer(0.02, signal.raise_signal, args=(signal.SIGINT,))
        timer.start()
        try:
            with pytest.raises(KeyboardInterrupt):
                for _ in range(64):
                    model.generate(
                        "a b c d e f",
                        max_tokens=192,
                        temperature=0.0,
                        ngram_speculation=False,
                    )
        finally:
            timer.cancel()

        # Model lock must be cleanly released right after KeyboardInterrupt.
        recovered = model.generate("a b c d", max_tokens=4, temperature=0.0)
        assert isinstance(recovered, str) and len(recovered) > 0

        # 2. Streaming generation must honor SIGINT and join its worker cleanly.
        timer = threading.Timer(0.02, signal.raise_signal, args=(signal.SIGINT,))
        timer.start()
        try:
            with pytest.raises(KeyboardInterrupt):
                for _ in range(64):
                    for _ in model.generate(
                        "a b c d e f",
                        max_tokens=192,
                        temperature=0.0,
                        stream=True,
                        ngram_speculation=False,
                    ):
                        pass
        finally:
            timer.cancel()

        recovered2 = model.generate("a b c d", max_tokens=4, temperature=0.0)
        assert recovered2 == recovered

        # 3. SIGINT must promptly interrupt both stream=False and stream=True while another
        # retained stream holds the engine mutex (neither call can complete until SIGINT cancels).
        held_stream = model.generate(
            "a b c d e f",
            max_tokens=192,
            temperature=0.0,
            stream=True,
            ngram_speculation=False,
        )
        try:
            assert isinstance(next(held_stream), str)
            # Let the retained stream fill its 64-slot channel and hold the engine mutex.
            time.sleep(0.2)

            timer = threading.Timer(0.05, signal.raise_signal, args=(signal.SIGINT,))
            started = time.monotonic()
            timer.start()
            try:
                with pytest.raises(KeyboardInterrupt):
                    model.generate(
                        "a b c d e f",
                        max_tokens=64,
                        temperature=0.0,
                        ngram_speculation=False,
                    )
            finally:
                timer.cancel()
            assert time.monotonic() - started < 2.0

            blocked_stream = model.generate(
                "a b c d e f",
                max_tokens=64,
                temperature=0.0,
                stream=True,
                ngram_speculation=False,
            )
            timer = threading.Timer(0.05, signal.raise_signal, args=(signal.SIGINT,))
            started = time.monotonic()
            timer.start()
            try:
                with pytest.raises(KeyboardInterrupt):
                    next(blocked_stream)
            finally:
                timer.cancel()
                blocked_stream.close()
            assert time.monotonic() - started < 2.0
        finally:
            held_stream.close()

        recovered3 = model.generate("a b c d", max_tokens=4, temperature=0.0)
        assert recovered3 == recovered


def test_interpreter_exit_with_live_model_and_stream(synthetic_gguf: Path) -> None:
    """Exiting Python without calling .close() or .unload() must not crash or hang."""
    script = (
        "import time\n"
        "from layerstream import AutoModel\n"
        f"m = AutoModel.from_pretrained({str(synthetic_gguf)!r}, use_gpu=False, "
        "ctx_size=256, batch_size=64, vram_budget_mb=0, prefer_direct_io=False)\n"
        "s = m.generate('a b c d', max_tokens=192, temperature=0.0, stream=True, "
        "ngram_speculation=False)\n"
        "first = next(s)\n"
        "assert isinstance(first, str)\n"
        "# Give the worker time to fill the 64-slot channel and block on send.\n"
        "time.sleep(0.25)\n"
        "# Intentionally exit without closing s or unloading m.\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, (
        f"subprocess exited with {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_repeated_load_unload_releases_file_descriptors(synthetic_gguf: Path) -> None:
    """Repeated load/generate/unload cycles must not leak OS file descriptors."""
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.is_dir():
        pytest.skip("/proc/self/fd not available on this platform")

    def open_fd_count() -> int:
        return len(os.listdir(fd_dir))

    # Warm up once (e.g., backend init / dynamic library handles), then measure delta.
    with AutoModel.from_pretrained(
        str(synthetic_gguf),
        use_gpu=False,
        ctx_size=128,
        batch_size=64,
        threads=1,
        vram_budget_mb=0,
        prefer_direct_io=False,
    ) as warm:
        warm.generate("a b c", max_tokens=4, temperature=0.0)
    gc.collect()
    baseline_fds = open_fd_count()

    for _ in range(8):
        with AutoModel.from_pretrained(
            str(synthetic_gguf),
            use_gpu=False,
            ctx_size=128,
            batch_size=64,
            threads=1,
            vram_budget_mb=0,
            prefer_direct_io=False,
        ) as model:
            out = model.generate("a b c", max_tokens=4, temperature=0.0)
            assert len(out) > 0
    gc.collect()
    after_fds = open_fd_count()
    assert after_fds <= baseline_fds + 1, (
        f"file descriptor leak detected: baseline={baseline_fds}, after 8 cycles={after_fds}"
    )
