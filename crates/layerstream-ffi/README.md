# `layerstream-ffi` — C ABI for Go, C/C++, Swift, C#, and Zig

`layerstream-ffi` builds `liblayerstream_ffi.{so,dylib,dll}` (`cdylib`) and `liblayerstream_ffi.a` (`staticlib`) alongside the standalone C11/C++ header [`include/layerstream.h`](include/layerstream.h).

## Build

```bash
cargo build --release -p layerstream-ffi
# With GPU backend:
cargo build --release -p layerstream-ffi --features vulkan
cargo build --release -p layerstream-ffi --features metal
cargo build --release -p layerstream-ffi --features cuda
```

## Example (C11)

See [`examples/smoke.c`](examples/smoke.c):

```bash
cc -std=c11 -Icrates/layerstream-ffi/include \
  crates/layerstream-ffi/examples/smoke.c \
  -Ltarget/release -llayerstream_ffi -Wl,-rpath,target/release -o smoke
./smoke /path/to/model.gguf
```

## Using from Go (`cgo`)

```go
/*
#cgo CFLAGS: -I${SRCDIR}/include
#cgo LDFLAGS: -L${SRCDIR}/target/release -llayerstream_ffi
#include "layerstream.h"
*/
import "C"
```

## Memory & Thread-Safety Rules

- Every `char *` returned by `layerstream_*` (except `layerstream_version()` and `layerstream_last_error()`) is heap-allocated and must be freed with `layerstream_string_free()`.
- Errors are reported via thread-local `layerstream_last_error()` (valid until the next call on the same thread; do not free).
- `LayerStreamEngine *` is internally mutex-synchronized and safe to share across threads; free with `layerstream_engine_free()`.
- `LayerStreamCancelToken *` is lock-free and safe to cancel from any thread via `layerstream_cancel_token_cancel()`; free with `layerstream_cancel_token_free()`.
