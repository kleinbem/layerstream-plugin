//! Pure-Rust GGUF v2/v3 header parser, 4K-aligned coalesced span planner,
//! and cross-platform `O_DIRECT` pread reader.

pub mod gguf_index;
pub mod io_reader;

pub use gguf_index::{
    CoalescedSpan, DIRECT_IO_ALIGN, GgufIndex, MAX_DRAFT_VOCAB_PAD_DELTA, ModelMeta,
    ResolvedTensorLoc, SpanSlice, TensorLocation, align_down, align_up,
    check_gguf_draft_compatible, check_vocab_lists_compatible,
};
pub use io_reader::{
    AlignedBuf, DirectSpanSlice, IoStats, LayerIoPool, probe_dir_storage_mib_per_s,
    probe_storage_mib_per_s,
};
