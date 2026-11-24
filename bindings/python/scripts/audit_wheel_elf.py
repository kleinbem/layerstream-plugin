#!/usr/bin/env python3
"""Strict ELF & PyPI compliance auditor for Linux `layerstream` and `layerstream-cuda13` wheels.

Workaround: `auditwheel repair` hashes/renames runtime-`dlopen`ed `libggml-*.so` backend plugins
(upstream: pypa/auditwheel#594), so Linux wheels build with `--auditwheel skip` and enforce
manylinux_2_34 symbol/RUNPATH bounds here.
"""

import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

WARN_WHEEL_BYTES = 90 * 1024 * 1024  # Warn when <10 MiB headroom remains below PyPI 100 MiB cap
MAX_WHEEL_BYTES = 98 * 1024 * 1024  # PyPI hard cap is 100 MiB
MAX_GLIBC = (2, 34)
MAX_GLIBCXX = (3, 4, 29)
MAX_CXXABI = (1, 3, 13)
FORBIDDEN_RPATH_SUBSTRINGS = (
    "/opt/",
    "/nix/",
    "/home/",
    "/tmp/",
    "/root/",
    "/github/",
    "/project/",
    "stubs",
)


def _parse_ver(raw: str) -> tuple[int, ...]:
    return tuple(int(p) for p in raw.split(".") if p.isdigit())


def audit_wheel(wheel_path: Path) -> None:
    if not wheel_path.is_file():
        raise SystemExit(f"ERROR: wheel not found: {wheel_path}")

    size = wheel_path.stat().st_size
    size_mib = size / (1024 * 1024)
    headroom_mib = 100.0 - size_mib
    print(
        f"[audit] {wheel_path.name}: {size_mib:.2f} MiB ({headroom_mib:.2f} MiB under PyPI 100 MiB cap)"
    )
    if size > MAX_WHEEL_BYTES:
        raise SystemExit(
            f"ERROR: {wheel_path.name} is {size_mib:.2f} MiB, exceeding PyPI 100 MiB safety limit"
        )
    if size > WARN_WHEEL_BYTES:
        print(
            f"::warning::{wheel_path.name} is {size_mib:.2f} MiB "
            f"({headroom_mib:.2f} MiB headroom below PyPI's 100 MiB cap)"
        )

    if "-linux_" in wheel_path.name:
        raise SystemExit(
            f"ERROR: un-tagged linux_* wheel will be rejected by PyPI: {wheel_path.name}"
        )

    is_base_wheel = wheel_path.name.startswith("layerstream-")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(wheel_path) as zf:
            so_members = [m for m in zf.namelist() if m.endswith(".so") or ".so." in Path(m).name]
            if not so_members:
                raise SystemExit(f"ERROR: no shared objects found in {wheel_path.name}")

            for member in so_members:
                extracted = Path(zf.extract(member, tmp_dir))
                dyn = subprocess.check_output(["readelf", "-d", str(extracted)], text=True)
                ver = subprocess.check_output(["readelf", "-V", str(extracted)], text=True)

                needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", dyn)
                rpaths = re.findall(
                    r"\((?:RPATH|RUNPATH)\)\s+Library r(?:un)?path: \[([^\]]+)\]", dyn
                )
                print(f"[audit]   {member}: NEEDED={needed} RUNPATH={rpaths}")

                for rp in rpaths:
                    if "" in rp.split(":"):
                        raise SystemExit(f"ERROR: empty CWD segment in RUNPATH of {member}: {rp!r}")
                    for bad in FORBIDDEN_RPATH_SUBSTRINGS:
                        if bad in rp:
                            raise SystemExit(
                                f"ERROR: forbidden build path '{bad}' in RUNPATH of {member}: {rp}"
                            )

                if is_base_wheel and "_layerstream" in Path(member).name:
                    for lib in needed:
                        if any(
                            k in lib for k in ("libvulkan", "libcudart", "libcuda.", "libcublas")
                        ):
                            raise SystemExit(
                                f"ERROR: base extension {member} directly links GPU driver/runtime {lib}"
                            )

                for m in re.finditer(r"GLIBC_([0-9]+\.[0-9]+(?:\.[0-9]+)?)", ver):
                    v = _parse_ver(m.group(1))
                    if v[:2] > MAX_GLIBC:
                        raise SystemExit(
                            f"ERROR: {member} requires GLIBC_{m.group(1)} > {MAX_GLIBC}"
                        )

                for m in re.finditer(r"GLIBCXX_([0-9]+\.[0-9]+\.[0-9]+)", ver):
                    v = _parse_ver(m.group(1))
                    if v > MAX_GLIBCXX:
                        raise SystemExit(
                            f"ERROR: {member} requires GLIBCXX_{m.group(1)} > {MAX_GLIBCXX}"
                        )

                for m in re.finditer(r"CXXABI_([0-9]+\.[0-9]+\.[0-9]+)", ver):
                    v = _parse_ver(m.group(1))
                    if v > MAX_CXXABI:
                        raise SystemExit(
                            f"ERROR: {member} requires CXXABI_{m.group(1)} > {MAX_CXXABI}"
                        )

    print(f"[audit] PASS: {wheel_path.name}")


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("Usage: audit_wheel_elf.py <wheel.whl> [...]")
    for arg in argv:
        audit_wheel(Path(arg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
