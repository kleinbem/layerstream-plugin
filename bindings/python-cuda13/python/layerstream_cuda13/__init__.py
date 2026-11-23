"""CUDA 13 dynamic backend plugin package for `layerstream`."""

import ctypes
import glob
import importlib.util
import os

__version__ = "0.4.0"


def backend_dir() -> str:
    """Return the directory containing `libggml-cuda.so`."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_backends")


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
            lib_dir = os.path.join(root, "lib")
            for match in sorted(glob.glob(os.path.join(lib_dir, pattern))):
                if os.path.isfile(match):
                    try:
                        ctypes.CDLL(match, mode=rtld_global)
                        loaded.append(match)
                        break
                    except OSError:
                        continue
    return loaded


__all__ = ["__version__", "backend_dir", "preload_nvidia_libraries"]
