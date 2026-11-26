//! Pure-Rust GGUF v2/v3 header parser, 4K-aligned coalesced span planner,
//! and cross-platform `O_DIRECT` pread reader.

pub mod expert_profile;
pub mod gguf_index;
pub mod io_reader;
#[cfg(target_os = "linux")]
mod uring;

/// Error returned by GGUF indexing and direct I/O operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GgufIoError {
    /// Underlying filesystem or OS I/O error with operation context.
    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
    /// Invalid or unsupported GGUF binary header, metadata, or shard layout.
    #[error("{0}")]
    InvalidFormat(String),
    /// Requested tensor was not found in the GGUF index.
    #[error("tensor '{0}' missing from GGUF index")]
    MissingTensor(String),
    /// Slice offset or range exceeds bounds or overflows arithmetic.
    #[error("{0}")]
    OutOfBounds(String),
    /// Aligned I/O buffer allocation failed.
    #[error("{0}")]
    AllocationFailed(String),
    /// Target and draft GGUF models have incompatible vocabularies.
    #[error("{0}")]
    IncompatibleDraft(String),
}

impl GgufIoError {
    pub(crate) fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }
}

impl From<std::io::Error> for GgufIoError {
    fn from(source: std::io::Error) -> Self {
        Self::Io {
            context: "I/O error".to_string(),
            source,
        }
    }
}

/// Convenience `Result` alias for `gguf-direct-io`.
pub type Result<T> = std::result::Result<T, GgufIoError>;

pub use gguf_index::{
    CoalescedSpan, DIRECT_IO_ALIGN, GgufIndex, MAX_DRAFT_VOCAB_PAD_DELTA, ModelMeta,
    ResolvedTensorLoc, SpanSlice, TensorLocation, align_down, align_up,
    check_gguf_draft_compatible, check_vocab_lists_compatible,
};
pub use io_reader::{
    AlignedBuf, DirectSpanSlice, IoStats, LayerIoPool, probe_dir_storage_mib_per_s,
    probe_storage_mib_per_s,
};
