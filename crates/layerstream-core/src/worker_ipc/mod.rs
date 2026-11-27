//! Unified isolated-worker IPC client and server shared across the Python and
//! Node.js bindings.
//!
//! Running `LoadedSession` inside a dedicated child process guarantees the
//! parent process (e.g., PyTorch, Jupyter, FastAPI, Node) never initializes
//! CUDA/Vulkan/Metal drivers. Calling `WorkerClient::unload()` (or dropping the
//! client) terminates and reaps the child so the OS kernel immediately reclaims
//! 100% of VRAM/UMA, CUDA driver contexts, and C++ heap arenas.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::chat_parse::{
    IncomingChatMessage, ParsedAssistantOutput, inject_tools_into_messages, parse_assistant_output,
};
use crate::engine::{GenerateOptions, GenerateSummary, KvCacheType, PromptTooLongError};
use crate::ffi::LsSampleParams;
use crate::hf::{DownloadProgress, ProgressCallback};
use crate::sdk::{EngineStatsSnapshot, LoadedSession, OpenSessionOptions};

/// Error returned across the worker IPC boundary with a structured error kind
/// so Python and Node bindings can map it to native exception types.
#[derive(Debug, Clone)]
pub struct WorkerError {
    pub kind: String,
    pub message: String,
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for WorkerError {}

impl WorkerError {
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            message: message.into(),
        }
    }
}

/// Serializable model-open parameters sent from parent to isolated worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerOpenParams {
    pub model: String,
    #[serde(default)]
    pub draft_model: Option<String>,
    #[serde(default)]
    pub quant: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
    #[serde(default = "default_true")]
    pub vram_budget_auto: bool,
    #[serde(default)]
    pub vram_budget_bytes: u64,
    #[serde(default)]
    pub resident_layers: Option<u32>,
    #[serde(default = "default_ctx_size")]
    pub ctx_size: u32,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default)]
    pub threads: Option<u32>,
    #[serde(default)]
    pub kv_cache_type: KvCacheType,
    #[serde(default = "default_true")]
    pub use_gpu: bool,
    #[serde(default = "default_true")]
    pub prefer_direct_io: bool,
    #[serde(default)]
    pub cpu_embeddings: bool,
    #[serde(default)]
    pub cpu_head: bool,
    #[serde(default)]
    pub stream_head: bool,
    #[serde(default = "default_true")]
    pub moe_active_streaming: bool,
    #[serde(default = "default_max_draft_tokens")]
    pub max_draft_tokens: usize,
    #[serde(default)]
    pub verbose: bool,
    #[serde(default)]
    pub stream_progress: bool,
}

fn default_true() -> bool {
    true
}
fn default_ctx_size() -> u32 {
    4096
}
fn default_batch_size() -> u32 {
    512
}
fn default_max_draft_tokens() -> usize {
    5
}
fn default_max_tokens() -> usize {
    256
}
fn default_top_p() -> f32 {
    0.95
}
fn default_top_k() -> i32 {
    40
}
fn default_repeat_penalty() -> f32 {
    1.0
}
fn default_penalty_last_n() -> i32 {
    64
}
fn default_seed() -> u32 {
    42
}

/// Serializable generation / chat parameters sent from parent to isolated worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerGenParams {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub min_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default = "default_penalty_last_n")]
    pub penalty_last_n: i32,
    #[serde(default = "default_seed")]
    pub seed: u32,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default = "default_true")]
    pub add_generation_prompt: bool,
    #[serde(default = "default_true")]
    pub ngram_speculation: bool,
    #[serde(default)]
    pub max_draft_tokens: Option<usize>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub grammar: Option<String>,
    #[serde(default)]
    pub parse_output: bool,
}

impl Default for WorkerGenParams {
    fn default() -> Self {
        Self {
            max_tokens: default_max_tokens(),
            temperature: 0.0,
            top_p: default_top_p(),
            min_p: 0.0,
            top_k: default_top_k(),
            repeat_penalty: default_repeat_penalty(),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            penalty_last_n: default_penalty_last_n(),
            seed: default_seed(),
            stop: Vec::new(),
            add_generation_prompt: true,
            ngram_speculation: true,
            max_draft_tokens: None,
            tools: None,
            grammar: None,
            parse_output: false,
        }
    }
}

impl WorkerGenParams {
    fn to_generate_options(&self, default_max_draft: usize) -> Result<GenerateOptions> {
        let sample = LsSampleParams {
            temperature: self.temperature,
            top_p: self.top_p,
            min_p: self.min_p,
            top_k: self.top_k,
            repeat_penalty: self.repeat_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            penalty_last_n: self.penalty_last_n,
            seed: self.seed,
        };
        sample.validate().map_err(|msg| anyhow!("{msg}"))?;
        Ok(GenerateOptions {
            max_tokens: self.max_tokens,
            sample,
            stop: self.stop.clone(),
            enable_ngram_speculation: self.ngram_speculation,
            max_draft_tokens: self.max_draft_tokens.unwrap_or(default_max_draft).max(1),
            grammar: self.grammar.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub enum WorkerStreamEvent {
    Chunk(String),
    Done(Option<GenerateSummary>),
    Error(WorkerError),
}

fn send_json_line(stream: &mut TcpStream, msg: &Value) -> std::io::Result<()> {
    let mut line = serde_json::to_vec(msg).map_err(std::io::Error::other)?;
    line.push(b'\n');
    stream.write_all(&line)?;
    stream.flush()
}

fn generate_auth_token() -> Result<String> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).map_err(|e| anyhow!("getrandom failed: {e}"))?;
    let mut hex = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{b:02x}");
    }
    Ok(hex)
}

fn classify_error(err: &anyhow::Error) -> (&'static str, String) {
    let msg = format!("{err:#}");
    if err.downcast_ref::<PromptTooLongError>().is_some() {
        return ("ValueError", msg);
    }
    for cause in err.chain() {
        if let Some(io_err) = cause.downcast_ref::<std::io::Error>()
            && io_err.kind() == std::io::ErrorKind::NotFound
        {
            return ("FileNotFoundError", msg);
        }
    }
    if msg.contains("not found") || msg.contains("No such file") {
        ("FileNotFoundError", msg)
    } else if msg.contains("invalid ") || msg.contains("must be ") || msg.contains("unknown ") {
        ("ValueError", msg)
    } else {
        ("RuntimeError", msg)
    }
}

mod client;
mod server;

pub use client::{WorkerCancelHandle, WorkerClient, WorkerStream};
#[cfg(test)]
use client::{WorkerState, reader_loop};
pub use server::run_worker_server;
