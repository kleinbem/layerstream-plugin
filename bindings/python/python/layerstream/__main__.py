"""Command-line entrypoint (`layerstream` / `python -m layerstream`).

All CLI argument parsing, diagnostics, chat REPL, and HTTP/SSE serving live in Rust
(`bindings/python/src/cli.rs` + `layerstream-core` + `layerstream-server`).
"""

import builtins
import sys

from ._layerstream import run_cli as _run_cli


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return int(_run_cli(args, builtins.input))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
