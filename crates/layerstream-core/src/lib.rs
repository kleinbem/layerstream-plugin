pub mod chat_parse;
pub mod engine;
pub mod ffi;
pub use gguf_direct_io::gguf_index;
pub use json_schema_gbnf as grammar;
#[cfg(feature = "hf")]
pub mod hf;
pub use gguf_direct_io::io_reader;
pub mod openai;
pub mod profile;
pub mod residency;
pub mod sampler;
#[cfg(feature = "hf")]
pub mod sdk;
pub mod speculator;
pub mod tuning;
#[cfg(feature = "hf")]
pub mod worker_ipc;

pub use chat_parse::{
    IncomingChatMessage, OpenAiFunctionCall, OpenAiToolCall, ParsedAssistantOutput, StreamDelta,
    StreamTagParser, format_tools_system_prompt, inject_tools_into_messages,
    parse_assistant_output,
};
pub use engine::{
    CancelledError, ChatMessage, EngineConfig, GenerateOptions, GenerateSummary, KvCacheType,
    PromptTooLongError, StopFilter, StreamEngine, effective_mtp_draft_cap,
};
pub use ffi::{LsEngineStats, LsSampleParams};
pub use gguf_index::{
    CoalescedSpan, GgufIndex, MAX_DRAFT_VOCAB_PAD_DELTA, ModelMeta, TensorLocation,
    check_gguf_draft_compatible, check_vocab_lists_compatible,
};
pub use grammar::{json_object_gbnf, json_schema_to_gbnf};
pub use io_reader::{IoStats, LayerIoPool, probe_dir_storage_mib_per_s, probe_storage_mib_per_s};
pub use openai::{
    CompiledResponseFormat, OpenAiUsage, compile_response_format, normalize_finish_reason,
    normalize_kv_cache_type, normalize_vram_budget_mb, validate_sample_params,
};
pub use profile::{HardwareProfile, MemoryTier, MemoryTopology, ProfileOverride, StorageStack};
pub use residency::{AdmissionPermit, EngineLease, Residency};
pub use speculator::{DraftModelProposer, NgramProposer, SpeculationStats};
pub use tuning::{
    TUNING_SCHEMA_VERSION, TuneCandidateResult, TunedProfile, default_tuning_dir,
    hardware_fingerprint, run_dir_storage_sweep, run_storage_sweep, select_best_candidate,
};
