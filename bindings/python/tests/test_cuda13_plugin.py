"""Unit tests for the `layerstream-cuda13` companion backend plugin loader."""

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

_CUDA13_INIT = (
    Path(__file__).resolve().parents[2]
    / "python-cuda13"
    / "python"
    / "layerstream_cuda13"
    / "__init__.py"
)


def _load_cuda13_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("layerstream_cuda13_testmod", _CUDA13_INIT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["layerstream_cuda13_testmod"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_backend_dir_points_to_package_backends() -> None:
    mod = _load_cuda13_module()
    expected = str(_CUDA13_INIT.parent / "_backends")
    assert mod.backend_dir() == expected


def test_preload_nvidia_libraries_returns_empty_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_cuda13_module()
    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda _name: None)
    assert mod.preload_nvidia_libraries() == []


def test_preload_nvidia_libraries_loads_cu13_with_rtld_global_and_skips_broken(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_cuda13_module()
    lib_dir = tmp_path / "nvidia" / "cu13" / "lib"
    lib_dir.mkdir(parents=True)
    cudart = lib_dir / "libcudart.so.13"
    cublas_lt = lib_dir / "libcublasLt.so.13"
    cublas_bad = lib_dir / "libcublas.so.13.0"
    cublas_ok = lib_dir / "libcublas.so.13.8"
    for p in (cudart, cublas_lt, cublas_bad, cublas_ok):
        p.write_bytes(b"\x7fELF")

    fake_spec = importlib.machinery.ModuleSpec("nvidia.cu13", loader=None, is_package=True)
    fake_spec.submodule_search_locations = [str(tmp_path / "nvidia" / "cu13")]
    monkeypatch.setattr(
        mod.importlib.util,
        "find_spec",
        lambda name: fake_spec if name == "nvidia.cu13" else None,
    )

    loaded_calls: list[tuple[str, int]] = []

    def fake_cdll(path: str, mode: int = 0) -> object:
        if path.endswith("libcublas.so.13.0"):
            raise OSError("simulated corrupt shared object")
        loaded_calls.append((path, mode))
        return object()

    monkeypatch.setattr(mod.ctypes, "CDLL", fake_cdll)
    loaded = mod.preload_nvidia_libraries()

    expected_mode = getattr(os, "RTLD_GLOBAL", mod.ctypes.RTLD_GLOBAL)
    assert loaded == [str(cudart), str(cublas_lt), str(cublas_ok)]
    assert loaded_calls == [
        (str(cudart), expected_mode),
        (str(cublas_lt), expected_mode),
        (str(cublas_ok), expected_mode),
    ]
