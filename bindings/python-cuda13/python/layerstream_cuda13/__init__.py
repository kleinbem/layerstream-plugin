"""CUDA 13 dynamic backend plugin package for `layerstream`."""

import ctypes
import importlib.util
import os
from pathlib import Path

__version__ = "0.6.3"


def backend_dir() -> str:
    """Return the directory containing `libggml-cuda.so`."""
    return str(Path(__file__).resolve().parent / "_backends")


# Workaround: PyPI `nvidia-cuda-runtime` and `nvidia-cublas` (CUDA 13) install shared libraries
# under `site-packages/nvidia/cu13/lib` outside the dynamic linker search path
# (upstream: abetlen/llama-cpp-python#1460).
def preload_nvidia_libraries() -> list[str]:
    """Preload `libcudart` and `libcublas` from PyPI `nvidia.cu13` packages with RTLD_GLOBAL."""
    loaded: list[str] = []
    rtld_global = getattr(os, "RTLD_GLOBAL", ctypes.RTLD_GLOBAL)
    candidates = [
        ("nvidia.cu13", "libcudart.so*"),
        ("nvidia.cu13", "libcublasLt.so*"),
        ("nvidia.cu13", "libcublas.so*"),
    ]
    for mod_name, pattern in candidates:
        try:
            spec = importlib.util.find_spec(mod_name)
        except ImportError, ValueError:
            spec = None
        if spec is None or not spec.submodule_search_locations:
            continue
        for root in spec.submodule_search_locations:
            lib_dir = Path(root) / "lib"
            if not lib_dir.is_dir():
                continue
            for match in sorted(lib_dir.glob(pattern)):
                if match.is_file():
                    try:
                        ctypes.CDLL(str(match), mode=rtld_global)
                        loaded.append(str(match))
                        break
                    except OSError:
                        continue
    return loaded


__all__ = ["__version__", "backend_dir", "preload_nvidia_libraries"]
