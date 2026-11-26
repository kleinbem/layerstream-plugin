# hf-gguf

Pure-Rust Hugging Face GGUF model discovery, hardware-aware quant recommendation, and resumable multi-shard downloader.

Extracted from [`layerstream-plugin`](https://github.com/kleinbem/layerstream-plugin) with zero C++ or `llama.cpp` build dependencies.

## Features

- **`hf://` URI parser (`HfSpec`)**: supports `hf://org/repo:file.gguf[@rev]`, `hf://org/repo/file.gguf`, and `alias=hf://...` specs with strict path-traversal validation.
- **Resumable multi-shard downloads**: HTTP `Range` resume via atomic `.part` files, automatic multi-part GGUF shard expansion (`*-00001-of-0000N.gguf`), progress callbacks, and `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` / `~/.cache/huggingface/token` authentication.
- **Repository discovery & quant ranking**: lists GGUF candidates on the Hugging Face Hub, resolves base repos to `-GGUF` siblings, ranks quants against available VRAM and storage bandwidth (`recommend_candidates`), and auto-selects compatible tiny draft models (`resolve_auto_draft_model`).

## Usage

```rust,no_run
use hf_gguf::{HfSpec, default_cache_dir, ensure_hf_downloaded};

let spec = HfSpec::parse("hf://Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf")?;
println!("url={}", spec.download_url());

let local_path = ensure_hf_downloaded(
    "hf://Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf",
    &default_cache_dir(),
)?;
println!("cached at {}", local_path.display());
# Ok::<(), anyhow::Error>(())
```

## License

Apache-2.0
