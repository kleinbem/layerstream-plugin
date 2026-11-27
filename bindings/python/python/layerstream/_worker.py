"""Child-process entrypoint for `isolated_worker=True` (`python -m layerstream._worker`).

All IPC framing, loopback TCP auth, parameter validation, cancellation, and OS process
termination/reap live in Rust (`layerstream-core::worker_ipc` + `bindings/python/src/worker.rs`).
"""

import sys

from ._layerstream import run_worker_server


def _main() -> None:
    if "--worker-port" not in sys.argv:
        sys.exit(2)
    idx = sys.argv.index("--worker-port")
    if idx + 1 >= len(sys.argv):
        sys.exit(2)
    run_worker_server(int(sys.argv[idx + 1]))


if __name__ == "__main__":
    _main()
