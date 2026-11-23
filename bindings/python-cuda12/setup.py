import os
import platform
from setuptools import Distribution, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return True


class PlatformWheel(_bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        plat = os.environ.get("LAYERSTREAM_CUDA_WHEEL_PLAT", "").strip()
        if not plat:
            arch = platform.machine().lower()
            if arch in ("aarch64", "arm64"):
                plat = "manylinux_2_31_aarch64"
            else:
                plat = f"manylinux_2_31_{arch}"
        return ("py3", "none", plat)


setup(
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": PlatformWheel},
)
