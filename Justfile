set shell := ["bash", "-euo", "pipefail", "-c"]

# List available recipes
default:
    @just --list

# Verify all 7 workspace, plugin, Python, and Node manifest versions match
check-versions:
    @cargo_ver=$(awk -F'"' '/^version =/{print $2; exit}' Cargo.toml); \
    plugin_ver=$(awk -F'"' '/^version =/{print $2; exit}' plugin.toml); \
    py_cargo_ver=$(awk -F'"' '/^version =/{print $2; exit}' bindings/python/Cargo.toml); \
    py_proj_ver=$(awk -F'"' '/^version =/{print $2; exit}' bindings/python/pyproject.toml); \
    py_cuda_ver=$(awk -F'"' '/^version =/{print $2; exit}' bindings/python-cuda13/pyproject.toml); \
    node_cargo_ver=$(awk -F'"' '/^version =/{print $2; exit}' bindings/node/Cargo.toml); \
    node_pkg_ver=$(awk -F'"' '/"version"[[:space:]]*:/{print $4; exit}' bindings/node/package.json); \
    echo "Cargo=${cargo_ver} plugin=${plugin_ver} py=${py_proj_ver} py_cuda=${py_cuda_ver} node=${node_pkg_ver}"; \
    test "$cargo_ver" = "$plugin_ver"; \
    test "$cargo_ver" = "$py_cargo_ver"; \
    test "$cargo_ver" = "$py_proj_ver"; \
    test "$cargo_ver" = "$py_cuda_ver"; \
    test "$cargo_ver" = "$node_cargo_ver"; \
    test "$cargo_ver" = "$node_pkg_ver"

# Run fast local preflight checks (versions, formatting, lint, flake eval, workarounds)
preflight: check-versions
    cargo fmt --all -- --check
    (cd bindings/python && cargo fmt -- --check)
    (cd bindings/node && cargo fmt -- --check)
    nix develop .#python -c bash -c 'cd bindings/python && ruff check python tests scripts ../python-cuda13/python ../python-cuda13/scripts ../python-cuda13/setup.py && ruff format --check python tests scripts ../python-cuda13/python ../python-cuda13/scripts ../python-cuda13/setup.py'
    nix fmt -- --ci
    command -v nix-check-workarounds >/dev/null && nix-check-workarounds . || true

# Format Rust, Python, and Nix files
fmt:
    cargo fmt --all
    (cd bindings/python && cargo fmt)
    (cd bindings/node && cargo fmt)
    nix develop .#python -c bash -c 'cd bindings/python && ruff format python tests scripts ../python-cuda13/python ../python-cuda13/scripts ../python-cuda13/setup.py'
    nix fmt

# Run Rust workspace clippy + tests at idle priority using a cache target dir
test-rust:
    CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/layerstream-target}" \
      chrt --idle 0 ionice -c3 cargo clippy --locked --workspace --all-targets -j 4 -- -D warnings
    CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/layerstream-target}" \
      chrt --idle 0 ionice -c3 cargo test --locked --workspace -j 4 -- --test-threads 1

# Run Python unit tests (fast pure-Python tests without rebuilding native wheel)
test-python-fast:
    nix develop .#python -c bash -c 'cd bindings/python && pytest -v tests/test_audit_wheel_elf.py tests/test_cuda13_plugin.py'

# Run real GPU parity smoke test (backend: vulkan or cuda) on local hardware (nixos-nvme / orin-nano)
gpu-smoke backend="vulkan":
    @if [ "{{backend}}" = "cuda" ]; then \
      nix develop .#cuda -c bash -c 'CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/layerstream-gpu-smoke}" chrt --idle 0 ionice -c3 cargo test --locked -p layerstream-core --features cuda --test parity_test -j 4 -- --test-threads 1'; \
    elif [ "{{backend}}" = "vulkan" ]; then \
      nix develop -c bash -c 'CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/layerstream-gpu-smoke}" chrt --idle 0 ionice -c3 cargo test --locked -p layerstream-core --features vulkan --test parity_test -j 4 -- --test-threads 1'; \
    else \
      echo "Unsupported backend '{{backend}}' (use 'vulkan' or 'cuda')" >&2; exit 1; \
    fi

# Bump version across all workspace, plugin, Python, and Node manifests + lockfiles
bump version:
    #!/usr/bin/env python3
    import pathlib, re, subprocess, sys

    ver = "{{version}}"
    if not re.fullmatch(r"\d+\.\d+\.\d+", ver):
        raise SystemExit(f"invalid semver: {ver}")

    edits = [
        ("Cargo.toml", r'(?m)^version = "[^"]+"', f'version = "{ver}"'),
        ("plugin.toml", r'(?m)^version = "[^"]+"', f'version = "{ver}"'),
        ("bindings/python/Cargo.toml", r'(?m)^version = "[^"]+"', f'version = "{ver}"'),
        ("bindings/python/pyproject.toml", r'(?m)^version = "[^"]+"', f'version = "{ver}"'),
        ("bindings/python/pyproject.toml", r"layerstream-cuda13==[0-9.]+", f"layerstream-cuda13=={ver}"),
        ("bindings/python-cuda13/pyproject.toml", r'(?m)^version = "[^"]+"', f'version = "{ver}"'),
        ("bindings/python-cuda13/python/layerstream_cuda13/__init__.py", r'(?m)^__version__ = "[^"]+"', f'__version__ = "{ver}"'),
        ("bindings/node/Cargo.toml", r'(?m)^version = "[^"]+"', f'version = "{ver}"'),
        ("bindings/node/package.json", r'("version"\s*:\s*")[^"]+(")', rf"\g<1>{ver}\g<2>"),
    ]
    for rel, pat, repl in edits:
        p = pathlib.Path(rel)
        text = p.read_text()
        new_text, n = re.subn(pat, repl, text)
        if n == 0:
            raise SystemExit(f"pattern not found in {rel}")
        p.write_text(new_text)

    for cwd in (".", "bindings/python", "bindings/node"):
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--offline"],
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            check=True,
        )
    subprocess.run(["just", "check-versions"], check=True)
