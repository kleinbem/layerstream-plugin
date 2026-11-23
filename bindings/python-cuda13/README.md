# layerstream-cuda13

Companion CUDA 13.5 backend plugin package (`libggml-cuda.so`) for [`layerstream`](https://pypi.org/project/layerstream/) on Linux `x86_64` and `aarch64` (`manylinux_2_34`), targeting Ampere (`sm_80`) through Blackwell (`sm_120a` / `sm_121a`, including NVIDIA Jetson Orin on JetPack 7 and DGX Spark GB10).

Requires an NVIDIA CUDA 13.5+ host driver (`>= 595`; older GPUs or drivers use the Vulkan backend bundled in `layerstream`). Install via the `cuda` extra (no local CUDA Toolkit or `nvcc` required):

```bash
pip install "layerstream[cuda]"
```

When installed alongside `layerstream`, `layerstream` automatically preloads `nvidia-cuda-runtime` and `nvidia-cublas` (`nvidia.cu13`) from your Python environment, registers `CUDA0` ahead of the bundled Vulkan fallback, and activates `cudaHostAlloc` + `cudaMemcpyAsync` overlapped NVMe-to-VRAM layer streaming.
