pub mod chat_parse;
pub mod engine;
pub mod ffi;
pub mod gguf_index;
pub mod grammar;
#[cfg(feature = "hf")]
pub mod hf;
pub mod io_reader;
pub mod residency;
#[cfg(feature = "hf")]
pub mod sdk;
pub mod speculator;
#[cfg(feature = "hf")]
pub mod worker_ipc;

pub use chat_parse::{
    IncomingChatMessage, OpenAiFunctionCall, OpenAiToolCall, ParsedAssistantOutput, StreamDelta,
    StreamTagParser, format_tools_system_prompt, inject_tools_into_messages,
    parse_assistant_output,
};
pub use engine::{
    CancelledError, ChatMessage, EngineConfig, GenerateOptions, GenerateSummary, KvCacheType,
    PromptTooLongError, StopFilter, StreamEngine,
};
pub use ffi::{LsEngineStats, LsSampleParams};
pub use gguf_index::{
    CoalescedSpan, GgufIndex, MAX_DRAFT_VOCAB_PAD_DELTA, ModelMeta, TensorLocation,
    check_gguf_draft_compatible, check_vocab_lists_compatible,
};
pub use grammar::{json_object_gbnf, json_schema_to_gbnf};
pub use io_reader::{IoStats, LayerIoPool, probe_dir_storage_mib_per_s, probe_storage_mib_per_s};
pub use residency::{AdmissionPermit, EngineLease, Residency};
pub use speculator::{DraftModelProposer, NgramProposer, SpeculationStats};
