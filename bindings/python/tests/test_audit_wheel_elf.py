"""Unit tests for `scripts/audit_wheel_elf.py` PyPI size-cap and ELF/RUNPATH guards."""

import importlib.util
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

_AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_wheel_elf.py"


def _load_auditor() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_wheel_elf", _AUDIT_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_wheel_elf"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_dummy_wheel(
    path: Path, member: str = "layerstream_cuda13/_backends/libggml-cuda.so"
) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, b"\x7fELF")
    return path


def test_audit_wheel_rejects_oversized_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = _load_auditor()
    whl = _make_dummy_wheel(
        tmp_path / "layerstream_cuda13-0.4.0-py3-none-manylinux_2_34_x86_64.whl"
    )
    monkeypatch.setattr(auditor, "MAX_WHEEL_BYTES", 10)
    with pytest.raises(SystemExit, match="exceeding PyPI 100 MiB safety limit"):
        auditor.audit_wheel(whl)


def test_audit_wheel_warns_when_headroom_low(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    auditor = _load_auditor()
    whl = _make_dummy_wheel(
        tmp_path / "layerstream_cuda13-0.4.0-py3-none-manylinux_2_34_x86_64.whl"
    )
    monkeypatch.setattr(auditor, "WARN_WHEEL_BYTES", 1)
    monkeypatch.setattr(auditor, "MAX_WHEEL_BYTES", 10 * 1024 * 1024)
    monkeypatch.setattr(
        auditor.subprocess,
        "check_output",
        lambda cmd, text=True: (
            " 0x000000000000001d (RUNPATH) Library runpath: [$ORIGIN]\n"
            if "-d" in cmd
            else "GLIBC_2.34\nGLIBCXX_3.4.29\nCXXABI_1.3.13\n"
        ),
    )
    auditor.audit_wheel(whl)
    out = capsys.readouterr().out
    assert "::warning::" in out
    assert "headroom below PyPI's 100 MiB cap" in out


def test_audit_wheel_rejects_untagged_linux_and_empty_runpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = _load_auditor()
    untagged = _make_dummy_wheel(tmp_path / "layerstream-0.4.0-cp314-abi3-linux_x86_64.whl")
    with pytest.raises(SystemExit, match="un-tagged linux_\\* wheel"):
        auditor.audit_wheel(untagged)

    bad_rpath = _make_dummy_wheel(
        tmp_path / "layerstream-0.4.0-cp314-abi3-manylinux_2_34_x86_64.whl",
        member="layerstream/_backends/libggml.so",
    )
    monkeypatch.setattr(
        auditor.subprocess,
        "check_output",
        lambda cmd, text=True: (
            " 0x000000000000001d (RUNPATH) Library runpath: [:$ORIGIN]\n"
            if "-d" in cmd
            else "GLIBC_2.34\n"
        ),
    )
    with pytest.raises(SystemExit, match="empty CWD segment in RUNPATH"):
        auditor.audit_wheel(bad_rpath)
