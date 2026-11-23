# layerstream-cuda13

Companion CUDA 13 backend plugin package (`libggml-cuda.so`) for [`layerstream`](https://pypi.org/project/layerstream/) on Linux `x86_64` and `aarch64` (including NVIDIA Jetson Orin on JetPack 7).

Requires an NVIDIA CUDA 13 host driver (`>= 580.65`; older drivers automatically fall back to the Vulkan backend bundled in `layerstream`). Install via the `cuda` extra (no local CUDA Toolkit or `nvcc` required):

```bash
pip install "layerstream[cuda]"
```

When installed alongside `layerstream`, `layerstream` automatically preloads `nvidia-cuda-runtime` and `nvidia-cublas` (`nvidia.cu13`) from your Python environment, registers `CUDA0` ahead of the bundled Vulkan fallback, and activates `cudaHostAlloc` + `cudaMemcpyAsync` overlapped NVMe-to-VRAM layer streaming.
