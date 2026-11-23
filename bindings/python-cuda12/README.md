# layerstream-cuda12

Companion CUDA 12 backend plugin package (`libggml-cuda.so`) for [`layerstream`](https://pypi.org/project/layerstream/).

Install via the `cuda` extra on Linux with an NVIDIA GPU (no local CUDA Toolkit or `nvcc` required):

```bash
pip install "layerstream[cuda]"
```

When installed alongside `layerstream`, `layerstream` automatically preloads `nvidia-cuda-runtime-cu12` and `nvidia-cublas-cu12` from your Python environment, registers `CUDA0` ahead of the bundled Vulkan fallback, and activates `cudaHostAlloc` + `cudaMemcpyAsync` overlapped NVMe-to-VRAM layer streaming.
