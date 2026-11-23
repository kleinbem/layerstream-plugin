#!/usr/bin/env python3
"""Build `libggml-cuda.so` from pinned `llama.cpp` and package `layerstream-cuda13` wheel."""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_llama_dir(pkg_root: Path) -> Path:
    env_dir = os.environ.get("LLAMA_CPP_DIR", "").strip()
    if env_dir:
        p = Path(env_dir)
        if (p / "CMakeLists.txt").is_file():
            return p.resolve()
    vendor = (pkg_root / "../../third_party/llama.cpp").resolve()
    if (vendor / "CMakeLists.txt").is_file():
        return vendor
    raise SystemExit(
        "ERROR: could not find llama.cpp source tree (set LLAMA_CPP_DIR or checkout third_party/llama.cpp)"
    )


def default_cuda_archs() -> str:
    arch = platform.machine().lower()
    if arch in ("aarch64", "arm64"):
        # Native Jetson Orin (sm_87) SASS plus Ampere/Hopper/Blackwell CUDA 13.0 PTX.
        return "80-virtual;87-real;90-virtual"
    # Native Ada (sm_89) SASS plus Turing/Ampere/Hopper/Blackwell CUDA 13.0 PTX keeps the wheel well under PyPI's 100 MiB cap.
    return "75-virtual;80-virtual;89-real;90-virtual"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dist", help="Output wheel directory")
    parser.add_argument(
        "--skip-cmake",
        action="store_true",
        help="Package an already-staged python/layerstream_cuda13/_backends/libggml-cuda.so",
    )
    args = parser.parse_args()

    pkg_root = Path(__file__).resolve().parent.parent
    stage_dir = pkg_root / "python" / "layerstream_cuda13" / "_backends"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_so = stage_dir / "libggml-cuda.so"

    if not args.skip_cmake:
        llama_dir = resolve_llama_dir(pkg_root)
        build_dir = pkg_root / "build" / "cmake-cuda"
        build_dir.mkdir(parents=True, exist_ok=True)
        archs = os.environ.get("CMAKE_CUDA_ARCHITECTURES", "").strip() or default_cuda_archs()

        cmake_cmd = [
            "cmake",
            "-S",
            str(llama_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
            "-DGGML_BACKEND_DL=ON",
            "-DGGML_CUDA=ON",
            # Workaround: ggml-cuda holds a single graph slot invalidated by per-layer cb_eval splits
            # (tracking: kleinbem/layerstream-plugin#5, upstream: ggml-org/llama.cpp#27330).
            "-DGGML_CUDA_GRAPHS=OFF",
            "-DGGML_CUDA_COMPRESSION_MODE=size",
            "-DGGML_CUDA_CCCL_VERSION=v3.4.3",
            "-DGGML_CUDA_NCCL=OFF",
            "-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16",
            f"-DCMAKE_CUDA_ARCHITECTURES={archs}",
            "-DGGML_CPU=OFF",
            "-DGGML_VULKAN=OFF",
            "-DGGML_METAL=OFF",
            "-DGGML_OPENMP=OFF",
            "-DGGML_NATIVE=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_TOOLS=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",
            "-DLLAMA_BUILD_COMMON=OFF",
            "-DLLAMA_BUILD_APP=OFF",
            "-DLLAMA_CURL=OFF",
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
            "-DCMAKE_BUILD_RPATH=$ORIGIN",
            "-DCMAKE_INSTALL_RPATH=$ORIGIN",
            "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath,$ORIGIN",
            "-DCMAKE_MODULE_LINKER_FLAGS=-Wl,-rpath,$ORIGIN",
        ]
        if shutil.which("ninja"):
            cmake_cmd.extend(["-G", "Ninja"])

        subprocess.run(cmake_cmd, check=True)
        jobs = str(min(os.cpu_count() or 4, 4))
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "ggml-cuda", "-j", jobs],
            check=True,
        )

        candidates = list(build_dir.rglob("libggml-cuda.so"))
        if not candidates:
            raise SystemExit(f"ERROR: libggml-cuda.so was not produced under {build_dir}")

        for old in stage_dir.glob("libggml*"):
            old.unlink()
        shutil.copyfile(candidates[0], staged_so)
        if shutil.which("strip"):
            subprocess.run(["strip", "--strip-unneeded", str(staged_so)], check=True)

    if not staged_so.is_file():
        raise SystemExit(f"ERROR: staged backend missing at {staged_so}")

    out_dir = (pkg_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_whl in out_dir.glob("layerstream_cuda13-*.whl"):
        old_whl.unlink()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(out_dir),
            str(pkg_root),
        ],
        check=True,
    )

    built = sorted(out_dir.glob("layerstream_cuda13-*.whl"))
    if not built:
        raise SystemExit(f"ERROR: no wheel built in {out_dir}")
    print(f"Built CUDA plugin wheel: {built[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
