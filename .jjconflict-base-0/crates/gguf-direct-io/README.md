# gguf-direct-io

Pure-Rust GGUF v2/v3 header parser, 4K-aligned coalesced span planner, and `O_DIRECT` / `F_NOCACHE` / `FILE_FLAG_NO_BUFFERING` pread reader.

Extracted from [`layerstream-plugin`](https://github.com/kleinbem/layerstream-plugin) with zero C++ or `llama.cpp` build dependencies.

## Features

- **Fast header-only GGUF v2/v3 parser (`GgufIndex`)**: discovers multi-part `*-00001-of-0000N.gguf` shards, extracts metadata and tensor byte offsets in milliseconds with 0 MB VRAM, and validates draft/target vocabulary compatibility (`check_gguf_draft_compatible`).
- **4K-aligned coalesced span planner**: merges nearby tensor or sub-tensor (e.g. active MoE expert) reads within a 64 KiB gap into minimal aligned disk spans.
- **Cross-platform direct I/O pool (`LayerIoPool`)**: uses `O_DIRECT` + `POSIX_FADV_RANDOM`/`DONTNEED` on Linux, `F_NOCACHE` + `F_RDAHEAD=0` on macOS, and `FILE_FLAG_NO_BUFFERING` on Windows, backed by a persistent scoped worker pool and a bounded scan-resistant S3-FIFO expert slice cache.
- **Storage bandwidth probes**: `probe_storage_mib_per_s` and `probe_dir_storage_mib_per_s` for fast startup preflight diagnostics.

## Usage

```rust,no_run
use gguf_direct_io::{GgufIndex, LayerIoPool};

let index = GgufIndex::open("model-00001-of-00002.gguf")?;
println!(
    "arch={} layers={} tensors={}",
    index.meta.architecture,
    index.meta.n_layers,
    index.tensors.len()
);

let pool = LayerIoPool::new(index, true)?;
let stats = pool.stats();
println!("direct_io_active={}", stats.direct_io_active);
# Ok::<(), anyhow::Error>(())
```

## License

Apache-2.0
