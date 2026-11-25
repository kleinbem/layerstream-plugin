# layerstream-plugin

Out-of-core GGUF layer-streaming LLM inference engine, Axum OpenAI HTTP/SSE server, Mesh-LLM v3 plugin, C ABI (`layerstream-ffi`), and native Python 3.15+ / Node.js 24+ SDKs.

## Layout

| Path | What lives here |
|---|---|
| `crates/layerstream-core/` | Rust core (`O_DIRECT` + `io_uring` I/O, GGUF parser, HF cache, GBNF grammar, residency, Rust sampler) + zero-patch C++20 `llama.cpp` shim (`cpp/`, with all `llama-model.h` access isolated in `llama_compat.hpp`). |
| `crates/layerstream-ffi/` | C ABI shared/static library (`cdylib` + `staticlib`) and C11/C++ header (`include/layerstream.h`) for Go, C/C++, Swift, C#, and Zig. |
| `crates/layerstream-server/` | Axum OpenAI-compatible HTTP/SSE server (`/health`, `/v1/models`, `/v1/unload`, `/v1/chat/completions`, `/v1/completions`) and constant-time Bearer/worker auth. |
| `crates/layerstream-plugin/` | Standalone CLI (`serve`, `plugin`, `check`, `pull`, `recommend`) and `WorkerCoordinator` process-isolated GPU worker for 0 MB idle VRAM. |
| `bindings/python/` | `layerstream` PyPI package (`PyO3` `abi3-py315`, `maturin`, bundled Vulkan + all CPU SIMD variants on Linux `manylinux_2_34`, Metal on macOS 15+). |
| `bindings/python-cuda13/` | `layerstream-cuda13` companion PyPI wheel (`libggml-cuda.so` for CUDA 13.5+, `sm_80+`, `manylinux_2_34` `x86_64` & `aarch64`). |
| `bindings/node/` | `layerstream` Node-API (`napi9`, Node.js `>=24`, `ES2024`) SDK and TypeScript definitions. |
| `xtask/` | Zero-dependency pure-Rust task runner (`cargo xtask ...`). |
| `flake.nix` | Nix devShells (`default`, `python`, `node`, `cuda`) pinning `llama-cpp-src`. |

## Conventions & Floors

- **Newest-only stack:** Rust Edition 2024 (`rust-version = "1.89"`), C++20, Python `>=3.15` (`abi3-py315`, `pathlib`, `pyright strict`, Ruff), Linux `manylinux_2_34` (`glibc >= 2.34`), CUDA `13.5+` (`driver >= 595`, `sm_80+`), macOS `15.0+` Apple Silicon only, Node.js `>=24` (pure ESM).
- **Zero-patch `llama.cpp`:** Never fork or patch `llama.cpp` directly. Keep all integration inside `crates/layerstream-core/cpp/` (isolating internal `llama-model.h` access in `llama_compat.hpp`), track upstream PRs/issues in `docs/UPSTREAM_LLAMA_CPP.md`, and keep `flake.nix` and `.github/workflows/*.yml` pinned to the same `llama.cpp` `master` commit SHA.
- **Rust safety & lints:** Enforced via `[workspace.lints.clippy]` (`unwrap_used`, `undocumented_unsafe_blocks`, `dbg_macro`, `todo`) with `-D warnings` — no `.unwrap()` in production code and every `unsafe` block/impl requires a `// SAFETY:` comment.
- **7 synced version manifests:** `Cargo.toml`, `plugin.toml`, `bindings/python/Cargo.toml`, `bindings/python/pyproject.toml`, `bindings/python-cuda13/pyproject.toml`, `bindings/node/Cargo.toml`, and `bindings/node/package.json` must match (`cargo xtask check-versions` / `cargo xtask bump X.Y.Z`).
- **Workarounds need links:** Every workaround comment must include an open `upstream: Owner/repo#123` or `tracking: Owner/repo#123` link (`nix-check-workarounds .`). Never use closing keywords (`fix:`, `closes:`) before external repo links.
- **Max 2,000 LoC per source file:** Keep every tracked source file (`.rs`, `.cpp`, `.hpp`, `.py`, `.js`, `.ts`) at or below 2,000 lines (`cargo xtask check-loc`, enforced in CI). Split larger modules into submodules before adding new code.
- **Local builds:** Keep `CARGO_TARGET_DIR` under `~/.cache/` (never `/tmp`), run heavy builds at idle priority (`chrt --idle 0 ionice -c3 ... -j 4`), and commit in a dedicated jj workspace (`just jj::ws-new layerstream-plugin <name>`).

## Common Commands

- `cargo xtask preflight` — version sync, Rust/Python/Nix formatting, `ruff`, and workaround link check.
- `cargo xtask test-rust` — workspace `clippy -D warnings` + unit/protocol/lifecycle tests.
- `cargo xtask test-python-fast` — fast pure-Python unit tests (`layerstream-cuda13`).
- `cargo xtask audit-wheel-elf <wheel.whl>` — strict Linux wheel ELF / RUNPATH / PyPI 100 MiB auditor.
- `cargo xtask gpu-smoke vulkan` / `cargo xtask gpu-smoke cuda` — real GPU `parity_test` on local hardware (`nixos-nvme` / `orin-nano`).
