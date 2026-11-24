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
        stream = model.generate("a b c d e f", max_tokens=10, temperature=0.0, stream=True)
        assert isinstance(stream, TokenStream)
        chunks = list(stream)
        assert "".join(chunks) == out1

        # Early stream break cancels cleanly and releases the model lock
        early_stream = model.generate("a b c d e f", max_tokens=20, temperature=0.0, stream=True)
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
    tree_base = json.dumps([{"type": "file", "path": "model.safetensors", "size": 128}]).encode(
        "utf-8"
    )

    class MockHfHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.path == "/api/models/MockOrg/TinyBase/tree/main?recursive=true":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(tree_base)))
                self.end_headers()
                self.wfile.write(tree_base)
            elif self.path == "/api/models/MockOrg/TinyBase-GGUF/tree/main?recursive=true":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(tree_gguf)))
                self.end_headers()
                self.wfile.write(tree_gguf)
            elif self.path == "/MockOrg/TinyBase-GGUF/resolve/main/Synthetic-Q4_K_M.gguf":
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
    with pytest.raises(RuntimeError, match="incompatible draft model tokenizer"):
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
        stopped_str = model.generate("a b c d e f", max_tokens=16, temperature=0.0, stop=needle)
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
        for chunk in model.generate("a b c d", max_tokens=24, temperature=0.0, stream=True):
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
            str(synthetic_gguf),
            vram_budget_mb=True,
            use_gpu=False,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        AutoModel.from_pretrained(str(synthetic_gguf), vram_budget_mb=-1, use_gpu=False)
    with pytest.raises(ValueError):
        AutoModel.from_pretrained(
            str(synthetic_gguf),
            vram_budget_mb="not-a-number",  # type: ignore[arg-type]
            use_gpu=False,
        )
    with pytest.raises(ValueError):
        AutoModel.from_pretrained(
            str(synthetic_gguf),
            kv_cache_type="bogus",  # type: ignore[arg-type]
            use_gpu=False,
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
        timeout=90,
        check=False,
    )
    assert proc.returncode == 0, (
        f"subprocess exited with {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_repeated_load_unload_releases_file_descriptors(synthetic_gguf: Path) -> None:
    """Repeated load/generate/unload cycles must not leak OS file descriptors."""
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.is_dir():
        pytest.skip("/proc/self/fd not available on this platform")

    def open_fd_count() -> int:
        return sum(1 for _ in fd_dir.iterdir())

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


def test_isolated_worker_parity_and_process_reap(synthetic_gguf: Path) -> None:
    """Verify isolated_worker=True runs in a separate OS process, matches in-process output
    bit-for-bit, supports tokenize/detokenize/kv_trim, and reaps the worker on unload."""

    with AutoModel.from_pretrained(
        str(synthetic_gguf),
        use_gpu=False,
        ctx_size=128,
        batch_size=64,
        vram_budget_mb=0,
        prefer_direct_io=False,
    ) as inproc:
        expected_gen = inproc.generate("a b c d e f", max_tokens=10, temperature=0.0)
        expected_chat = inproc.chat(
            [{"role": "user", "content": "a b c"}], max_tokens=8, temperature=0.0
        )
        expected_toks = inproc.tokenize("a b c")
        expected_detok = inproc.detokenize(expected_toks)

    with AutoModel.from_pretrained(
        str(synthetic_gguf),
        use_gpu=False,
        ctx_size=128,
        batch_size=64,
        vram_budget_mb=0,
        prefer_direct_io=False,
        isolated_worker=True,
    ) as iso:
        assert iso.is_loaded
        assert iso.isolated_worker is True
        pid = iso.worker_pid
        assert isinstance(pid, int) and pid > 0 and pid != os.getpid()

        st = iso.stats()
        assert st["loaded"] is True
        assert st["isolated_worker"] is True
        assert st["worker_pid"] == pid

        assert iso.tokenize("a b c") == expected_toks
        assert iso.detokenize(expected_toks) == expected_detok
        assert iso.generate("a b c d e f", max_tokens=10, temperature=0.0) == expected_gen

        streamed = "".join(iso.generate("a b c d e f", max_tokens=10, temperature=0.0, stream=True))
        assert streamed == expected_gen
        assert iso.last_summary is not None
        assert iso.last_summary["completion_tokens"] == 10

        # Early break cancels worker stream cleanly without wedging the engine
        for chunk in iso.generate("a b c d e f", max_tokens=24, temperature=0.0, stream=True):
            assert isinstance(chunk, str)
            break

        assert (
            iso.chat([{"role": "user", "content": "a b c"}], max_tokens=8, temperature=0.0)
            == expected_chat
        )
        assert iso.kv_trim(2) is True
        iso.kv_clear()

    # Worker process must be completely terminated and reaped after context exit!
    assert not iso.is_loaded
    assert iso.worker_pid is None
    with pytest.raises(OSError) as exc_info:
        os.kill(pid, 0)
    assert isinstance(exc_info.value, ProcessLookupError) or (
        os.name == "nt" and getattr(exc_info.value, "winerror", None) == 87
    )


def test_grammar_and_json_schema_constrained_decoding(synthetic_json_gguf: Path) -> None:
    """Verify GBNF & JSON Schema constrained decoding (greedy, sampled, streaming, and
    with draft_model speculative verification) always produces schema-valid JSON."""
    from layerstream import json_object_gbnf, json_schema_to_gbnf

    schema = {
        "$defs": {
            "Meta": {
                "type": "object",
                "properties": {
                    "ok": {"type": "boolean"},
                },
                "required": ["ok"],
                "additionalProperties": False,
            }
        },
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["fast", "safe"]},
            "n": {"type": "integer"},
            "meta": {"$ref": "#/$defs/Meta"},
        },
        "required": ["mode", "n", "meta"],
        "additionalProperties": False,
    }
    compiled = json_schema_to_gbnf(schema)
    assert compiled.startswith("root ::= ws ")
    assert json_object_gbnf().startswith("root ::= ws json-object")

    with AutoModel.from_pretrained(
        str(synthetic_json_gguf),
        draft_model=str(synthetic_json_gguf),
        max_draft_tokens=4,
        use_gpu=False,
        ctx_size=256,
        batch_size=64,
        vram_budget_mb=0,
        prefer_direct_io=False,
    ) as model:
        # 1. Greedy JSON schema constraint (with draft speculation active)
        raw_greedy = model.generate(
            "a b c",
            max_tokens=96,
            temperature=0.0,
            json_schema=schema,
        )
        parsed = json.loads(raw_greedy)
        assert set(parsed.keys()) == {"mode", "n", "meta"}
        assert parsed["mode"] in ("fast", "safe")
        assert type(parsed["n"]) is int
        assert isinstance(parsed["meta"], dict)
        assert set(parsed["meta"].keys()) == {"ok"}
        assert type(parsed["meta"]["ok"]) is bool

        # 2. Sampled (temperature=0.8) + streaming JSON schema constraint
        raw_sampled = "".join(
            model.generate(
                "a b c",
                max_tokens=96,
                temperature=0.8,
                seed=777,
                json_schema=schema,
                stream=True,
            )
        )
        parsed_sampled = json.loads(raw_sampled)
        assert set(parsed_sampled.keys()) == {"mode", "n", "meta"}
        assert parsed_sampled["mode"] in ("fast", "safe")
        assert type(parsed_sampled["n"]) is int
        assert isinstance(parsed_sampled["meta"], dict)
        assert set(parsed_sampled["meta"].keys()) == {"ok"}
        assert type(parsed_sampled["meta"]["ok"]) is bool

        # 3. Verify llama.cpp GBNF parser accepts additionalProperties key-exclusion & oneOf(number, integer)
        addl_schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}, "opt": {"type": "boolean"}},
            "additionalProperties": {"type": "boolean"},
        }
        raw_addl = model.generate("a b", max_tokens=64, temperature=0.0, json_schema=addl_schema)
        parsed_addl = json.loads(raw_addl)
        assert isinstance(parsed_addl, dict)
        if "count" in parsed_addl:
            assert type(parsed_addl["count"]) is int

        oneof_num_schema = {
            "type": "object",
            "properties": {"v": {"oneOf": [{"type": "number"}, {"type": "integer"}]}},
            "required": ["v"],
            "additionalProperties": False,
        }
        raw_oneof = model.generate(
            "a b", max_tokens=64, temperature=0.0, json_schema=oneof_num_schema
        )
        parsed_oneof = json.loads(raw_oneof)
        assert isinstance(parsed_oneof["v"], float)
        assert parsed_oneof["v"] % 1.0 != 0.0

        # 4. Invalid grammar fails closed with RuntimeError
        with pytest.raises(RuntimeError, match="grammar"):
            model.generate("a b", max_tokens=8, grammar="not valid gbnf ===")


def test_in_process_openai_and_async_openai_clients(synthetic_json_gguf: Path) -> None:
    """Verify drop-in OpenAI / AsyncOpenAI clients, .parse(), tools, and tag parsing."""
    import asyncio
    from dataclasses import dataclass

    from layerstream import (
        AsyncOpenAI,
        OpenAI,
        StreamTagParser,
        format_tools_system_prompt,
        parse_assistant_output,
    )

    @dataclass
    class VerdictSchema:
        ok: bool
        counts: list[int]

    @dataclass
    class UnsupportedSchema:
        bad: set[int]

    class MockPydanticV2Model:
        def __init__(self, mode: str, flag: bool) -> None:
            self.mode = mode
            self.flag = flag

        @classmethod
        def model_json_schema(cls) -> dict[str, object]:
            return {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["alpha", "beta"]},
                    "flag": {"type": "boolean"},
                },
                "required": ["mode", "flag"],
                "additionalProperties": False,
            }

        @classmethod
        def model_validate_json(cls, raw: str) -> MockPydanticV2Model:
            d = json.loads(raw)
            return cls(mode=d["mode"], flag=d["flag"])

    # Verify Rust StreamTagParser & parse_assistant_output bindings directly
    parsed_tags = parse_assistant_output(
        '<think>step 1</think>Result <tool_call>{"name":"ping","arguments":{"x":1}}</tool_call>'
    )
    assert parsed_tags["reasoning_content"] == "step 1"
    assert parsed_tags["content"] == "Result"
    assert len(parsed_tags["tool_calls"]) == 1
    assert parsed_tags["tool_calls"][0]["function"]["name"] == "ping"

    parser = StreamTagParser()
    evs: list[dict[str, object]] = []
    for part in [
        "<thi",
        "nk>plan</think>ok <tool_",
        'call>{"name":"f","arguments":{}}</tool_call>',
    ]:
        evs.extend(parser.push(part))
    evs.extend(parser.flush())
    assert parser.has_tool_calls is True
    assert any(e["type"] == "reasoning" and e["text"] == "plan" for e in evs)
    assert any(e["type"] == "tool_call" and e["name"] == "f" for e in evs)

    tools_spec = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    assert "lookup" in format_tools_system_prompt(tools_spec)
    with pytest.raises(TypeError, match="tools"):
        format_tools_system_prompt([1])  # type: ignore[list-item]

    with OpenAI(
        str(synthetic_json_gguf),
        use_gpu=False,
        ctx_size=1024,
        batch_size=64,
        vram_budget_mb=0,
        prefer_direct_io=False,
    ) as client:
        models_page = client.models.list()
        assert len(models_page.data) == 1

        # Standard chat completion with tools
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "a b c"}],
            tools=tools_spec,
            max_tokens=8,
            temperature=0.0,
        )
        assert resp.object == "chat.completion"
        assert resp.usage is not None and resp.usage.total_tokens > 0
        assert isinstance(resp.model_dump(), dict)

        # Structured `.parse(...)` with Pydantic-v2-style model class
        parsed_resp = client.chat.completions.parse(
            messages=[{"role": "user", "content": "a b"}],
            response_format=MockPydanticV2Model,
            max_tokens=96,
            temperature=0.0,
        )
        inst = parsed_resp.choices[0].message.parsed
        assert isinstance(inst, MockPydanticV2Model)
        assert inst.mode in ("alpha", "beta")
        assert isinstance(inst.flag, bool)

        # Structured `.parse(...)` with dataclass (including list[int] collection)
        dc_resp = client.chat.completions.parse(
            messages=[{"role": "user", "content": "a b"}],
            response_format=VerdictSchema,
            max_tokens=96,
            temperature=0.0,
        )
        assert isinstance(dc_resp.choices[0].message.parsed, VerdictSchema)
        assert type(dc_resp.choices[0].message.parsed.ok) is bool
        assert isinstance(dc_resp.choices[0].message.parsed.counts, list)
        assert all(type(x) is int for x in dc_resp.choices[0].message.parsed.counts)

        # Unsupported dataclass field annotations fail closed with TypeError
        with pytest.raises(TypeError, match="unsupported dataclass field annotation"):
            client.chat.completions.parse(
                messages=[{"role": "user", "content": "a b"}],
                response_format=UnsupportedSchema,
                max_tokens=16,
            )

        # Streaming chat completion chunks
        stream_chunks = list(
            client.chat.completions.create(
                messages=[{"role": "user", "content": "a b"}],
                max_tokens=6,
                stream=True,
            )
        )
        assert stream_chunks[0].choices[0].delta.role == "assistant"
        assert stream_chunks[-1].choices[0].finish_reason in ("stop", "length", "tool_calls")

    # AsyncOpenAI + async TokenStream
    async def _run_async() -> None:
        async with AsyncOpenAI(
            str(synthetic_json_gguf),
            use_gpu=False,
            ctx_size=512,
            batch_size=64,
            vram_budget_mb=0,
            prefer_direct_io=False,
        ) as aclient:
            aresp = await aclient.chat.completions.create(
                messages=[{"role": "user", "content": "a b"}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "tag": {"type": "string", "minLength": 2, "maxLength": 4},
                                "ok": {"type": "boolean"},
                            },
                            "required": ["tag", "ok"],
                            "additionalProperties": False,
                        }
                    },
                },
                max_tokens=64,
            )
            decoded = json.loads(aresp.choices[0].message.content or "")
            assert decoded["ok"] in (True, False)
            assert 2 <= len(decoded["tag"]) <= 4

            astream = await aclient.chat.completions.create(
                messages=[{"role": "user", "content": "a b"}],
                max_tokens=6,
                stream=True,
            )
            async with astream as s:
                collected = [
                    ch.choices[0].delta.content async for ch in s if ch.choices[0].delta.content
                ]
            assert len(collected) > 0

            cstream = await aclient.completions.create(
                prompt="a b",
                max_tokens=4,
                stream=True,
            )
            async with cstream as cs:
                comp_parts = [item.choices[0].text async for item in cs]
            assert len(comp_parts) > 0

    asyncio.run(_run_async())

    # Verify isolated worker is reaped when AutoModel is garbage-collected without `.unload()`
    import gc
    import time

    from layerstream import AutoModel

    leaked = AutoModel.from_pretrained(
        str(synthetic_json_gguf),
        use_gpu=False,
        ctx_size=256,
        batch_size=32,
        vram_budget_mb=0,
        prefer_direct_io=False,
        isolated_worker=True,
    )
    tmpl_rendered = leaked.apply_chat_template(
        [
            {"role": "user", "content": "run"},
            {"role": "tool", "name": "lookup", "tool_call_id": "call_1", "content": "42"},
        ]
    )
    assert "tool" in tmpl_rendered
    assert "42" in tmpl_rendered
    pid = leaked.worker_pid
    assert pid is not None
    del leaked
    gc.collect()
    deadline = time.monotonic() + 5.0
    still_alive = True
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            still_alive = False
            break
        except OSError as exc:
            if os.name == "nt" and getattr(exc, "winerror", None) == 87:
                still_alive = False
                break
            raise
        time.sleep(0.05)
    assert not still_alive, f"isolated worker {pid} still alive after GC"


def test_advisor_check_recommend_pull_and_inspect(
    synthetic_gguf: Path,
    incompatible_vocab_gguf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify inspect_gguf, check, recommend, and pull bindings."""
    from layerstream import check, inspect_gguf, pull, recommend

    info = inspect_gguf(synthetic_gguf)
    assert info["architecture"] == "ernie4_5-moe"
    assert info["n_layers"] == 6
    assert info["shard_count"] == 1
    assert info["tensor_count"] > 0

    report = check(
        str(synthetic_gguf),
        draft_model=str(synthetic_gguf),
        use_gpu=False,
    )
    assert report["ok"] is True
    assert report["verdict"] in ("GOOD", "OK")
    assert report["n_layers"] == 6

    # Incompatible draft must fail closed in preflight check() without loading full weights
    with pytest.raises(RuntimeError, match="incompatible draft model tokenizer"):
        check(
            str(synthetic_gguf),
            draft_model=str(incompatible_vocab_gguf),
            use_gpu=False,
        )

    # Mock HF server for recommend() + pull()
    payload = synthetic_gguf.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    gib = 1024 * 1024 * 1024
    tree_gguf = json.dumps(
        [
            {
                "type": "file",
                "path": "Model-Q4_K_M.gguf",
                "size": 8 * gib,
                "lfs": {"size": 8 * gib},
            },
            {
                "type": "file",
                "path": "Model-Q8_0.gguf",
                "size": 16 * gib,
                "lfs": {"size": 16 * gib},
            },
            {
                "type": "file",
                "path": "Tiny.gguf",
                "size": len(payload),
                "lfs": {"size": len(payload)},
            },
        ]
    ).encode("utf-8")
    tree_base = json.dumps([{"type": "file", "path": "model.safetensors", "size": 64}]).encode(
        "utf-8"
    )

    class AdvisorHfHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.path == "/api/models/AdvisorOrg/Model/tree/main?recursive=true":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(tree_base)))
                self.end_headers()
                self.wfile.write(tree_base)
            elif self.path == "/api/models/AdvisorOrg/Model-GGUF/tree/main?recursive=true":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(tree_gguf)))
                self.end_headers()
                self.wfile.write(tree_gguf)
            elif self.path == "/AdvisorOrg/Model-GGUF/resolve/main/Tiny.gguf":
                self.send_response(200)
                self.send_header("ETag", f'"{sha}"')
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

    server = HTTPServer(("127.0.0.1", 0), AdvisorHfHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{port}")
        cache_dir = tmp_path / "advisor-cache"
        recs = recommend(
            "AdvisorOrg/Model",
            free_ram_gb=5.2,
            nvme_mib_s=2800.0,
            has_draft=True,
            cache_dir=cache_dir,
            use_gpu=False,
        )
        assert len(recs) == 3
        assert recs[0]["resolved_repo"] == "AdvisorOrg/Model-GGUF"
        assert any(r["recommended"] for r in recs)

        pulled_path = pull("hf://AdvisorOrg/Model-GGUF:Tiny.gguf", cache_dir=cache_dir)
        assert Path(pulled_path).read_bytes() == payload
    finally:
        server.shutdown()
        server.server_close()


def test_kv_state_save_load_and_modern_samplers(
    synthetic_gguf: Path,
    tmp_path: Path,
) -> None:
    """Verify save_state/load_state token-exact resume (in-process & isolated_worker) and new samplers."""
    snap_path = tmp_path / "snapshots" / "prefix.kv"
    base_prompt = "a b c d e f a b c"
    continuation_prompt = "a b c d e f a b c d e"

    with AutoModel.from_pretrained(
        str(synthetic_gguf),
        draft_model="auto",  # gracefully skips on tiny <2 GiB synthetic GGUF
        use_gpu=False,
        ctx_size=128,
        batch_size=64,
        vram_budget_mb=0,
        kv_cache_type="q8_0",
        prefer_direct_io=False,
    ) as model:
        st0 = model.stats()
        assert st0["backend_name"] == "CPU"
        assert st0["draft_model_path"] is None

        # Prefill base_prompt + 1 token, then snapshot KV state to disk.
        _ = model.generate(base_prompt, max_tokens=1, temperature=0.0, ngram_speculation=False)
        saved_tokens = model.save_state(snap_path)
        assert saved_tokens > 0
        assert snap_path.is_file() and snap_path.stat().st_size > 0

        # Reference continuation from scratch on a clean KV cache.
        model.kv_clear()
        expected_cont = model.generate(
            continuation_prompt,
            max_tokens=10,
            temperature=0.0,
            ngram_speculation=False,
        )

        # Modern samplers: min_p + repeat/frequency/presence penalties (both greedy & sampled).
        penalized_greedy = model.generate(
            "a a a a a b c",
            max_tokens=8,
            temperature=0.0,
            repeat_penalty=1.35,
            frequency_penalty=0.4,
            presence_penalty=0.2,
            penalty_last_n=16,
        )
        assert isinstance(penalized_greedy, str) and len(penalized_greedy) > 0

        sampled_min_p = model.generate(
            "a b c d",
            max_tokens=8,
            temperature=0.75,
            min_p=0.1,
            top_p=0.9,
            repeat_penalty=1.15,
            seed=123,
        )
        assert isinstance(sampled_min_p, str) and len(sampled_min_p) > 0

        # Invalid sampler arguments raise ValueError cleanly.
        with pytest.raises(ValueError, match="min_p"):
            model.generate("a b", min_p=1.5)
        with pytest.raises(ValueError, match="repeat_penalty"):
            model.generate("a b", repeat_penalty=0.0)

        with pytest.raises(FileNotFoundError):
            model.load_state(tmp_path / "missing.kv")

    # Restore KV snapshot in a fresh in-process AutoModel AND in an isolated_worker AutoModel
    # and verify continuation output matches fresh prefill bit-for-bit!
    for use_isolated in (False, True):
        with AutoModel.from_pretrained(
            str(synthetic_gguf),
            use_gpu=False,
            ctx_size=128,
            batch_size=64,
            vram_budget_mb=0,
            kv_cache_type="q8_0",
            prefer_direct_io=False,
            isolated_worker=use_isolated,
        ) as restored_model:
            restored_n = restored_model.load_state(snap_path)
            assert restored_n == saved_tokens
            assert restored_model.stats()["cached_kv_tokens"] == saved_tokens
            got_cont = restored_model.generate(
                continuation_prompt,
                max_tokens=10,
                temperature=0.0,
                ngram_speculation=False,
            )
            assert got_cont == expected_cont

            if use_isolated:
                iso_snap = tmp_path / "snapshots" / "isolated.kv"
                assert restored_model.save_state(iso_snap) > 0
                assert iso_snap.is_file()
                with pytest.raises(FileNotFoundError):
                    restored_model.load_state(tmp_path / "does_not_exist.kv")


def test_progress_callbacks_and_backend_discovery(
    synthetic_gguf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify on_progress callbacks in pull() and AutoModel.from_pretrained() (in-process & isolated)."""
    from layerstream import configured_backend_dirs, probe_gpu_device, pull

    assert isinstance(configured_backend_dirs(), list)
    assert probe_gpu_device(use_gpu=False) is None

    payload = synthetic_gguf.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()

    class ProgHfHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.path.startswith("/ProgOrg/Repo/resolve/main/"):
                self.send_response(200)
                self.send_header("ETag", f'"{sha}"')
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

    server = HTTPServer(("127.0.0.1", 0), ProgHfHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{port}")
        cache_dir = tmp_path / "prog-cache"

        events: list[dict[str, object]] = []
        pulled = pull(
            "hf://ProgOrg/Repo:one.gguf",
            cache_dir=cache_dir,
            on_progress=lambda ev: events.append(dict(ev)),
        )
        assert Path(pulled).is_file()
        assert len(events) >= 2
        assert events[-1]["done"] is True
        assert events[-1]["downloaded_bytes"] == len(payload)

        # Isolated worker forwards live on_progress events over IPC during open!
        iso_events: list[dict[str, object]] = []
        with AutoModel.from_pretrained(
            "hf://ProgOrg/Repo:two.gguf",
            cache_dir=cache_dir,
            use_gpu=False,
            ctx_size=128,
            batch_size=32,
            vram_budget_mb=0,
            prefer_direct_io=False,
            isolated_worker=True,
            on_progress=lambda ev: iso_events.append(dict(ev)),
        ) as m:
            assert m.is_loaded
        assert len(iso_events) >= 2
        assert iso_events[-1]["done"] is True

        # Exception raised inside on_progress aborts download and propagates exact Python error.
        class CustomAbortError(RuntimeError):
            pass

        def _abort_cb(_ev: dict[str, object]) -> None:
            raise CustomAbortError("stop right here")

        with pytest.raises(CustomAbortError, match="stop right here"):
            pull("hf://ProgOrg/Repo:three.gguf", cache_dir=cache_dir, on_progress=_abort_cb)
    finally:
        server.shutdown()
        server.server_close()


def test_cli_inspect_check_chat_and_rust_axum_server(
    synthetic_gguf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `python -m layerstream` CLI (single-turn + interactive REPL) and Rust `start_server` HTTP + SSE + `/v1/unload`."""
    import io
    import urllib.error
    import urllib.request

    from layerstream import start_server
    from layerstream.__main__ import main as cli_main

    # 1. inspect (human & --json), check (human & --json & failure)
    assert cli_main(["inspect", str(synthetic_gguf)]) == 0
    assert cli_main(["inspect", str(synthetic_gguf), "--json"]) == 0
    assert cli_main(["check", str(synthetic_gguf), "--cpu"]) == 0
    assert cli_main(["check", str(synthetic_gguf), "--cpu", "--json"]) == 0
    assert cli_main(["check", str(tmp_path / "missing.gguf"), "--cpu", "--json"]) == 1

    # 2. Single-turn `chat --prompt` with `--save-state` and `--load-state`
    cli_snap = tmp_path / "cli_chat.kv"
    assert (
        cli_main(
            [
                "chat",
                str(synthetic_gguf),
                "--cpu",
                "--no-direct-io",
                "--ctx-size",
                "128",
                "--batch-size",
                "32",
                "--max-tokens",
                "6",
                "--prompt",
                "a b c",
                "--save-state",
                str(cli_snap),
            ]
        )
        == 0
    )
    assert cli_snap.is_file()

    assert (
        cli_main(
            [
                "chat",
                str(synthetic_gguf),
                "--cpu",
                "--no-direct-io",
                "--ctx-size",
                "128",
                "--batch-size",
                "32",
                "--max-tokens",
                "6",
                "--load-state",
                str(cli_snap),
                "--prompt",
                "a b c d",
                "--quiet-stats",
            ]
        )
        == 0
    )

    # 3. Interactive REPL commands: `/stats`, `a b`, `/save`, `/clear`, `/load`, `/quit`
    repl_snap = tmp_path / "repl.kv"
    repl_inputs = iter(
        [
            "/stats",
            "a b c",
            f"/save {repl_snap}",
            "/clear",
            f"/load {repl_snap}",
            "/quit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(repl_inputs))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert (
        cli_main(
            [
                "chat",
                str(synthetic_gguf),
                "--cpu",
                "--no-direct-io",
                "--ctx-size",
                "128",
                "--batch-size",
                "32",
                "--max-tokens",
                "4",
                "--system",
                "be concise",
            ]
        )
        == 0
    )
    assert repl_snap.is_file()

    # 4. Native Rust `layerstream-server` (`start_server`) HTTP + SSE + Bearer auth + `/v1/unload`
    with start_server(
        str(synthetic_gguf),
        host="127.0.0.1",
        port=0,
        api_key="secret-test-key",
        use_gpu=False,
        ctx_size=128,
        batch_size=32,
        vram_budget_mb=0,
        prefer_direct_io=False,
    ) as srv:
        assert srv.port > 0
        base = f"http://{srv.host}:{srv.port}"
        auth_hdr = {
            "Authorization": "Bearer secret-test-key",
            "Content-Type": "application/json",
        }

        # `/health` is unauthenticated and reports 0-VRAM lazy idle state (`loaded: False`).
        with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
            health = json.loads(resp.read().decode("utf-8"))
            assert health["status"] == "ok"
            assert health["loaded"] is False

        # `/v1/models` rejects missing/wrong auth with 401 (`authentication_error`).
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base}/v1/models", timeout=5)
        assert exc_info.value.code == 401

        # Authenticated `/v1/models`.
        req_models = urllib.request.Request(f"{base}/v1/models", headers=auth_hdr)
        with urllib.request.urlopen(req_models, timeout=5) as resp:
            models_json = json.loads(resp.read().decode("utf-8"))
            assert len(models_json["data"]) == 1

        # Invalid sampler parameter (`min_p > 1`) returns 400 Bad Request.
        bad_body = json.dumps(
            {"messages": [{"role": "user", "content": "a b"}], "min_p": 1.5}
        ).encode("utf-8")
        req_bad = urllib.request.Request(
            f"{base}/v1/chat/completions", data=bad_body, headers=auth_hdr, method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as bad_exc:
            urllib.request.urlopen(req_bad, timeout=5)
        assert bad_exc.value.code == 400

        # Non-streaming `/v1/chat/completions` with modern samplers (`min_p`, `repeat_penalty`).
        # Keep greedy `temperature: 0.0` so the tiny random-weights synthetic GGUF does not
        # occasionally sample EOS on token 1.
        chat_body = json.dumps(
            {
                "messages": [{"role": "user", "content": "a b c"}],
                "max_tokens": 6,
                "temperature": 0.0,
                "min_p": 0.05,
                "repeat_penalty": 1.15,
            }
        ).encode("utf-8")
        req_chat = urllib.request.Request(
            f"{base}/v1/chat/completions", data=chat_body, headers=auth_hdr, method="POST"
        )
        with urllib.request.urlopen(req_chat, timeout=15) as resp:
            chat_resp = json.loads(resp.read().decode("utf-8"))
            assert chat_resp["object"] == "chat.completion"
            assert len(chat_resp["choices"][0]["message"]["content"]) > 0

        # Streaming SSE `/v1/chat/completions`.
        sse_body = json.dumps(
            {"messages": [{"role": "user", "content": "a b c"}], "max_tokens": 6, "stream": True}
        ).encode("utf-8")
        req_sse = urllib.request.Request(
            f"{base}/v1/chat/completions", data=sse_body, headers=auth_hdr, method="POST"
        )
        sse_lines: list[str] = []
        with urllib.request.urlopen(req_sse, timeout=15) as resp:
            for raw_line in resp:
                decoded = raw_line.decode("utf-8").strip()
                if decoded.startswith("data:"):
                    sse_lines.append(decoded[5:].strip())
        assert sse_lines[-1] == "[DONE]"
        parsed_chunks = [json.loads(x) for x in sse_lines[:-1]]
        assert any(c["choices"][0]["delta"].get("content") for c in parsed_chunks)

        # Explicit `/v1/unload` evicts active model back to `loaded: False`.
        req_unload = urllib.request.Request(
            f"{base}/v1/unload", data=b"{}", headers=auth_hdr, method="POST"
        )
        with urllib.request.urlopen(req_unload, timeout=10) as resp:
            unloaded = json.loads(resp.read().decode("utf-8"))
            assert unloaded["unloaded"] is True
            assert unloaded["loaded"] is False
