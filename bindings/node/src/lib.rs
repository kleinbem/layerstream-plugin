use std::collections::VecDeque;
use std::future::poll_fn;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::task::{Poll, Waker};
use std::time::Duration;

use layerstream_core::sdk::{
    CheckOptions, EngineStatsSnapshot, LoadedSession, OpenSessionOptions, RecommendOptions,
    check_model, inspect_gguf_file, pull_model, recommend_repo,
};
use layerstream_core::worker_ipc::{
    WorkerClient, WorkerError, WorkerGenParams, WorkerOpenParams, WorkerStream,
    run_worker_server as core_run_worker_server,
};
use layerstream_core::{
    ChatMessage, GenerateOptions, GenerateSummary, IncomingChatMessage, KvCacheType,
    LsSampleParams, ParsedAssistantOutput, StreamDelta, StreamEngine,
    StreamTagParser as CoreStreamTagParser,
    compile_response_format as core_compile_response_format,
    format_tools_system_prompt as core_format_tools,
    inject_tools_into_messages as core_inject_tools, json_object_gbnf as core_json_object_gbnf,
    json_schema_to_gbnf as core_json_schema_to_gbnf,
    normalize_kv_cache_type as core_normalize_kv_cache_type,
    normalize_vram_budget_mb as core_normalize_vram_budget_mb,
    parse_assistant_output as core_parse_assistant_output,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::Deserialize;
use serde_json::Value as JsonValue;

fn invalid_arg(msg: impl Into<String>) -> Error {
    Error::new(Status::InvalidArg, msg.into())
}

fn generic_err(msg: impl Into<String>) -> Error {
    Error::new(Status::GenericFailure, msg.into())
}

fn map_anyhow_err(err: anyhow::Error) -> Error {
    let msg = format!("{err:#}");
    if msg.contains("local GGUF model not found") || msg.contains("KV state file not found") {
        Error::new(Status::GenericFailure, format!("ENOENT: {msg}"))
    } else {
        Error::new(Status::GenericFailure, msg)
    }
}

async fn run_blocking<T, F>(f: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| generic_err(format!("background worker panicked: {e}")))?
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct SpeculationStatsJs {
    pub sweeps: u32,
    pub drafted_tokens: u32,
    pub accepted_tokens: u32,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct GenerateSummaryJs {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub finish_reason: String,
    pub speculation: SpeculationStatsJs,
}

fn summary_to_js(s: &GenerateSummary) -> GenerateSummaryJs {
    GenerateSummaryJs {
        prompt_tokens: s.prompt_tokens.min(u32::MAX as usize) as u32,
        completion_tokens: s.completion_tokens.min(u32::MAX as usize) as u32,
        finish_reason: s.finish_reason.clone(),
        speculation: SpeculationStatsJs {
            sweeps: s.speculation.sweeps.min(u32::MAX as usize) as u32,
            drafted_tokens: s.speculation.drafted_tokens.min(u32::MAX as usize) as u32,
            accepted_tokens: s.speculation.accepted_tokens.min(u32::MAX as usize) as u32,
        },
    }
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct IoStatsJs {
    pub bytes_read: f64,
    pub span_reads: f64,
    pub pread_us: f64,
    pub cache_hits: f64,
    pub cache_misses: f64,
    pub cache_bytes_served: f64,
    pub direct_io_active: bool,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct EngineStatsJs {
    pub loaded: bool,
    pub isolated_worker: bool,
    pub worker_pid: Option<u32>,
    pub backend_name: Option<String>,
    pub cached_kv_tokens: Option<u32>,
    pub model_path: Option<String>,
    pub draft_model_path: Option<String>,
    pub source_uri: Option<String>,
    pub resolved_repo: Option<String>,
    pub selected_quant: Option<String>,
    pub n_ctx: Option<u32>,
    pub vocab_size: Option<i32>,
    pub n_layers: Option<u32>,
    pub resident_layers: Option<u32>,
    pub streamed_layers: Option<u32>,
    pub resident_bytes: Option<f64>,
    pub cpu_resident_bytes: Option<f64>,
    pub slot_bytes: Option<f64>,
    pub expert_slot_bytes: Option<f64>,
    pub total_layer_loads: Option<f64>,
    pub moe_layers: Option<u32>,
    pub moe_experts_loaded: Option<f64>,
    pub moe_experts_total: Option<f64>,
    pub head_streamed: Option<bool>,
    pub moe_active_streaming: Option<bool>,
    pub is_uma_zero_copy: Option<bool>,
    pub is_unified_memory: Option<bool>,
    pub using_gpu: Option<bool>,
    pub can_speculate: Option<bool>,
    pub io: Option<IoStatsJs>,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct GpuMemoryJs {
    pub free_bytes: f64,
    pub total_bytes: f64,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct GgufInfoJs {
    pub architecture: String,
    pub name: Option<String>,
    pub n_layers: u32,
    pub context_length: Option<u32>,
    pub total_file_bytes: f64,
    pub shard_count: u32,
    pub tensor_count: u32,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct QuantRecommendationJs {
    pub resolved_repo: String,
    pub quant: String,
    pub primary_file: String,
    pub total_bytes: f64,
    pub shard_count: u32,
    pub hf_uri: String,
    pub verdict: String,
    pub pinned_pct: u32,
    pub est_load_s: f64,
    pub est_tps_low: f64,
    pub est_tps_high: f64,
    pub recommended: bool,
    pub score: i32,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct CheckReportJs {
    pub ok: bool,
    pub verdict: String,
    pub summary: String,
    pub model_path: String,
    pub draft_model_path: Option<String>,
    pub architecture: String,
    pub n_layers: u32,
    pub context_length: Option<u32>,
    pub total_file_bytes: f64,
    pub storage_mib_s: f64,
    pub direct_io: bool,
    pub gpu_free_bytes: Option<f64>,
    pub gpu_total_bytes: Option<f64>,
    pub fits_in_vram: bool,
    pub est_tps_low: f64,
    pub est_tps_high: f64,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct ToolCallFunctionJs {
    pub name: String,
    pub arguments: String,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct ToolCallJs {
    pub id: String,
    #[napi(js_name = "type")]
    pub kind: String,
    pub function: ToolCallFunctionJs,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct ParsedAssistantOutputJs {
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCallJs>,
}

fn parsed_to_js(p: ParsedAssistantOutput) -> ParsedAssistantOutputJs {
    ParsedAssistantOutputJs {
        content: p.content,
        reasoning_content: p.reasoning_content,
        tool_calls: p
            .tool_calls
            .into_iter()
            .map(|tc| ToolCallJs {
                id: tc.id,
                kind: tc.kind,
                function: ToolCallFunctionJs {
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                },
            })
            .collect(),
    }
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct StreamDeltaJs {
    #[napi(js_name = "type")]
    pub kind: String,
    pub text: Option<String>,
    pub index: Option<u32>,
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

fn deltas_to_js(deltas: Vec<StreamDelta>) -> Vec<StreamDeltaJs> {
    deltas
        .into_iter()
        .map(|d| match d {
            StreamDelta::Content(text) => StreamDeltaJs {
                kind: "content".to_string(),
                text: Some(text),
                index: None,
                id: None,
                name: None,
                arguments: None,
            },
            StreamDelta::Reasoning(text) => StreamDeltaJs {
                kind: "reasoning".to_string(),
                text: Some(text),
                index: None,
                id: None,
                name: None,
                arguments: None,
            },
            StreamDelta::ToolCall {
                index,
                id,
                name,
                arguments,
            } => StreamDeltaJs {
                kind: "tool_call".to_string(),
                text: None,
                index: Some(index.min(u32::MAX as usize) as u32),
                id: Some(id),
                name: Some(name),
                arguments: Some(arguments),
            },
        })
        .collect()
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct GenerateResultJs {
    pub text: String,
    pub summary: GenerateSummaryJs,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct ChatResultJs {
    pub text: String,
    pub parsed: Option<ParsedAssistantOutputJs>,
    pub summary: GenerateSummaryJs,
}

fn push_waker(list: &mut Vec<Waker>, waker: &Waker) {
    if !list.iter().any(|w| w.will_wake(waker)) {
        list.push(waker.clone());
    }
}

struct CancelState {
    flag: AtomicBool,
    wakers: Mutex<Vec<Waker>>,
}

impl CancelState {
    fn new() -> Self {
        Self {
            flag: AtomicBool::new(false),
            wakers: Mutex::new(Vec::new()),
        }
    }

    fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
        if let Ok(mut list) = self.wakers.lock() {
            let drained = std::mem::take(&mut *list);
            drop(list);
            for w in drained {
                w.wake();
            }
        }
    }

    fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    fn register_waker(&self, waker: &Waker) {
        if let Ok(mut list) = self.wakers.lock() {
            push_waker(&mut list, waker);
        }
    }
}

/// Cooperative cancellation handle flipped synchronously from JS `AbortSignal` listeners.
#[napi]
pub struct NativeCancelToken {
    state: Arc<CancelState>,
}

#[napi]
impl NativeCancelToken {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            state: Arc::new(CancelState::new()),
        }
    }

    #[napi]
    pub fn cancel(&self) {
        self.state.cancel();
    }

    #[napi(getter)]
    pub fn is_cancelled(&self) -> bool {
        self.state.is_cancelled()
    }
}

impl Default for NativeCancelToken {
    fn default() -> Self {
        Self::new()
    }
}

#[napi]
pub struct StreamTagParser {
    inner: Mutex<CoreStreamTagParser>,
}

#[napi]
impl StreamTagParser {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(CoreStreamTagParser::new()),
        }
    }

    #[napi(getter)]
    pub fn has_tool_calls(&self) -> Result<bool> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| generic_err("StreamTagParser lock poisoned"))?;
        Ok(guard.has_tool_calls())
    }

    #[napi]
    pub fn push(&self, piece: String) -> Result<Vec<StreamDeltaJs>> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| generic_err("StreamTagParser lock poisoned"))?;
        Ok(deltas_to_js(guard.push(&piece)))
    }

    #[napi]
    pub fn flush(&self) -> Result<Vec<StreamDeltaJs>> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| generic_err("StreamTagParser lock poisoned"))?;
        Ok(deltas_to_js(guard.flush()))
    }
}

impl Default for StreamTagParser {
    fn default() -> Self {
        Self::new()
    }
}

const STREAM_BUFFER_CAPACITY: usize = 64;

struct StreamInner {
    queue: VecDeque<String>,
    closed_by_consumer: bool,
    producer_finished: bool,
    worker_exited: bool,
    error: Option<String>,
    summary: Option<GenerateSummaryJs>,
    read_wakers: Vec<Waker>,
    exit_wakers: Vec<Waker>,
}

struct StreamShared {
    inner: Mutex<StreamInner>,
    not_full_cv: Condvar,
    cancel: Arc<CancelState>,
}

impl StreamShared {
    fn new(cancel: Arc<CancelState>) -> Self {
        Self {
            inner: Mutex::new(StreamInner {
                queue: VecDeque::with_capacity(STREAM_BUFFER_CAPACITY),
                closed_by_consumer: false,
                producer_finished: false,
                worker_exited: false,
                error: None,
                summary: None,
                read_wakers: Vec::new(),
                exit_wakers: Vec::new(),
            }),
            not_full_cv: Condvar::new(),
            cancel,
        }
    }

    fn send_chunk(&self, text: String) -> bool {
        let Ok(mut st) = self.inner.lock() else {
            return false;
        };
        while !st.closed_by_consumer
            && !self.cancel.is_cancelled()
            && st.queue.len() >= STREAM_BUFFER_CAPACITY
        {
            st = match self.not_full_cv.wait(st) {
                Ok(g) => g,
                Err(_) => return false,
            };
        }
        if st.closed_by_consumer || self.cancel.is_cancelled() {
            return false;
        }
        st.queue.push_back(text);
        let wakers = std::mem::take(&mut st.read_wakers);
        drop(st);
        for w in wakers {
            w.wake();
        }
        true
    }

    fn finish_producer(
        &self,
        res: anyhow::Result<Option<GenerateSummary>>,
        model_last_summary: &Mutex<Option<GenerateSummaryJs>>,
    ) {
        let Ok(mut st) = self.inner.lock() else {
            return;
        };
        match res {
            Ok(Some(sum)) => {
                let js_sum = summary_to_js(&sum);
                if let Ok(mut mls) = model_last_summary.lock() {
                    *mls = Some(js_sum.clone());
                }
                st.summary = Some(js_sum);
            }
            Ok(None) => {}
            Err(e) => {
                if !st.closed_by_consumer && !self.cancel.is_cancelled() {
                    st.error = Some(format!("{e:#}"));
                }
            }
        }
        st.producer_finished = true;
        let wakers = std::mem::take(&mut st.read_wakers);
        drop(st);
        for w in wakers {
            w.wake();
        }
    }

    fn mark_worker_exited(&self) {
        if let Ok(mut st) = self.inner.lock() {
            st.producer_finished = true;
            st.worker_exited = true;
            let read_wakers = std::mem::take(&mut st.read_wakers);
            let exit_wakers = std::mem::take(&mut st.exit_wakers);
            drop(st);
            for w in read_wakers {
                w.wake();
            }
            for w in exit_wakers {
                w.wake();
            }
        }
    }

    fn cancel_and_disconnect(&self) {
        self.cancel.cancel();
        if let Ok(mut st) = self.inner.lock() {
            st.closed_by_consumer = true;
            st.queue.clear();
            let wakers = std::mem::take(&mut st.read_wakers);
            drop(st);
            self.not_full_cv.notify_all();
            for w in wakers {
                w.wake();
            }
        } else {
            self.not_full_cv.notify_all();
        }
    }

    async fn next_chunk(&self) -> Result<Option<String>> {
        poll_fn(|cx| {
            if self.cancel.is_cancelled() {
                return Poll::Ready(Ok(None));
            }
            self.cancel.register_waker(cx.waker());
            if self.cancel.is_cancelled() {
                return Poll::Ready(Ok(None));
            }
            let mut st = match self.inner.lock() {
                Ok(g) => g,
                Err(_) => return Poll::Ready(Err(generic_err("stream lock poisoned"))),
            };
            if st.closed_by_consumer {
                return Poll::Ready(Ok(None));
            }
            if let Some(chunk) = st.queue.pop_front() {
                drop(st);
                self.not_full_cv.notify_one();
                return Poll::Ready(Ok(Some(chunk)));
            }
            if let Some(err) = st.error.take() {
                return Poll::Ready(Err(generic_err(err)));
            }
            if st.producer_finished || st.worker_exited {
                return Poll::Ready(Ok(None));
            }
            push_waker(&mut st.read_wakers, cx.waker());
            Poll::Pending
        })
        .await
    }

    async fn wait_worker_exited(&self) {
        poll_fn(|cx| {
            let mut st = match self.inner.lock() {
                Ok(g) => g,
                Err(_) => return Poll::Ready(()),
            };
            if st.worker_exited {
                return Poll::Ready(());
            }
            push_waker(&mut st.exit_wakers, cx.waker());
            Poll::Pending
        })
        .await;
    }
}

struct WorkerExitGuard {
    shared: Arc<StreamShared>,
}

impl Drop for WorkerExitGuard {
    fn drop(&mut self) {
        self.shared.mark_worker_exited();
    }
}

#[napi]
pub struct NativeTokenStream {
    shared: Arc<StreamShared>,
}

impl Drop for NativeTokenStream {
    fn drop(&mut self) {
        self.shared.cancel_and_disconnect();
    }
}

#[napi]
impl NativeTokenStream {
    /// Synchronously signals the underlying C++ / NVMe engine to abort and wakes any full-buffer send.
    #[napi]
    pub fn cancel(&self) {
        self.shared.cancel_and_disconnect();
    }

    #[napi(getter)]
    pub fn summary(&self) -> Result<Option<GenerateSummaryJs>> {
        let st = self
            .shared
            .inner
            .lock()
            .map_err(|_| generic_err("stream lock poisoned"))?;
        Ok(st.summary.clone())
    }

    #[napi]
    pub async fn close(&self) -> Result<()> {
        self.shared.cancel_and_disconnect();
        self.shared.wait_worker_exited().await;
        Ok(())
    }

    #[napi]
    pub async fn next_chunk(&self) -> Result<Option<String>> {
        self.shared.next_chunk().await
    }
}

struct GateInner {
    busy: bool,
    wakers: Vec<Waker>,
}

struct EngineGate {
    inner: Mutex<GateInner>,
    cv: Condvar,
}

struct EnginePermit {
    gate: Arc<EngineGate>,
}

impl Drop for EnginePermit {
    fn drop(&mut self) {
        if let Ok(mut g) = self.gate.inner.lock() {
            g.busy = false;
            let wakers = std::mem::take(&mut g.wakers);
            drop(g);
            self.gate.cv.notify_one();
            for w in wakers {
                w.wake();
            }
        }
    }
}

impl EngineGate {
    fn new() -> Self {
        Self {
            inner: Mutex::new(GateInner {
                busy: false,
                wakers: Vec::new(),
            }),
            cv: Condvar::new(),
        }
    }

    async fn acquire_async(
        self: &Arc<Self>,
        cancel: Option<&Arc<CancelState>>,
    ) -> Result<EnginePermit> {
        poll_fn(|cx| {
            if let Some(c) = cancel {
                if c.is_cancelled() {
                    return Poll::Ready(Err(generic_err("ABORT_ERR: operation aborted")));
                }
                c.register_waker(cx.waker());
                if c.is_cancelled() {
                    return Poll::Ready(Err(generic_err("ABORT_ERR: operation aborted")));
                }
            }
            let mut g = match self.inner.lock() {
                Ok(g) => g,
                Err(_) => return Poll::Ready(Err(generic_err("model gate poisoned"))),
            };
            if !g.busy {
                g.busy = true;
                return Poll::Ready(Ok(EnginePermit {
                    gate: Arc::clone(self),
                }));
            }
            push_waker(&mut g.wakers, cx.waker());
            Poll::Pending
        })
        .await
    }

    fn acquire_sync(
        self: &Arc<Self>,
        cancel: &Arc<CancelState>,
    ) -> anyhow::Result<Option<EnginePermit>> {
        let mut g = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("model gate poisoned"))?;
        while g.busy {
            if cancel.is_cancelled() {
                return Ok(None);
            }
            let (next_g, _) = self
                .cv
                .wait_timeout(g, Duration::from_millis(5))
                .map_err(|_| anyhow::anyhow!("model gate poisoned"))?;
            g = next_g;
        }
        if cancel.is_cancelled() {
            return Ok(None);
        }
        g.busy = true;
        Ok(Some(EnginePermit {
            gate: Arc::clone(self),
        }))
    }
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct OpenOptionsWire {
    draft_model: Option<String>,
    quant: Option<String>,
    revision: Option<String>,
    cache_dir: Option<String>,
    vram_budget_auto: Option<bool>,
    vram_budget_bytes: Option<u64>,
    resident_layers: Option<u32>,
    ctx_size: Option<u32>,
    batch_size: Option<u32>,
    threads: Option<u32>,
    kv_cache_type: Option<String>,
    use_gpu: Option<bool>,
    prefer_direct_io: Option<bool>,
    cpu_embeddings: Option<bool>,
    cpu_head: Option<bool>,
    stream_head: Option<bool>,
    moe_active_streaming: Option<bool>,
    max_draft_tokens: Option<usize>,
    verbose: Option<bool>,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct GenerateOptionsWire {
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    top_k: Option<i32>,
    repeat_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    penalty_last_n: Option<i32>,
    seed: Option<u32>,
    stop: Option<Vec<String>>,
    ngram_speculation: Option<bool>,
    max_draft_tokens: Option<usize>,
    grammar: Option<String>,
    add_generation_prompt: Option<bool>,
    tools: Option<Vec<JsonValue>>,
    parse_output: Option<bool>,
}

fn parse_wire_messages(raw_json: &str) -> Result<Vec<ChatMessage>> {
    let items: Vec<IncomingChatMessage> = serde_json::from_str(raw_json)
        .map_err(|e| invalid_arg(format!("invalid chat messages payload: {e}")))?;
    if items.is_empty() {
        return Err(invalid_arg("messages must not be empty"));
    }
    items
        .iter()
        .map(|m| m.to_chat_message().map_err(invalid_arg))
        .collect()
}

fn inject_tools_into_messages(messages: &mut Vec<ChatMessage>, tools: &[JsonValue]) -> Result<()> {
    for t in tools {
        if !t.is_object() {
            return Err(invalid_arg("each entry in 'tools' must be an object"));
        }
    }
    core_inject_tools(messages, tools);
    Ok(())
}

fn build_generate_options(wire: &GenerateOptionsWire) -> Result<GenerateOptions> {
    let grammar = match wire.grammar.as_deref() {
        None => None,
        Some(raw) => {
            if raw.trim().is_empty() {
                return Err(invalid_arg("grammar must not be empty"));
            }
            Some(raw.to_string())
        }
    };
    let sample = LsSampleParams {
        temperature: wire.temperature.unwrap_or(0.0),
        top_p: wire.top_p.unwrap_or(0.95),
        min_p: wire.min_p.unwrap_or(0.0),
        top_k: wire.top_k.unwrap_or(40),
        repeat_penalty: wire.repeat_penalty.unwrap_or(1.0),
        frequency_penalty: wire.frequency_penalty.unwrap_or(0.0),
        presence_penalty: wire.presence_penalty.unwrap_or(0.0),
        penalty_last_n: wire.penalty_last_n.unwrap_or(64),
        seed: wire.seed.unwrap_or(42),
    };
    sample.validate().map_err(invalid_arg)?;
    Ok(GenerateOptions {
        max_tokens: wire.max_tokens.unwrap_or(256),
        sample,
        stop: wire.stop.clone().unwrap_or_default(),
        enable_ngram_speculation: wire.ngram_speculation.unwrap_or(true),
        max_draft_tokens: wire.max_draft_tokens.unwrap_or(0),
        grammar,
    })
}

#[napi]
pub struct NativeAutoModel {
    inner: Arc<Mutex<Option<LoadedSession>>>,
    gate: Arc<EngineGate>,
    loaded: Arc<AtomicBool>,
    cached_model_path: Arc<Mutex<Option<String>>>,
    last_summary: Arc<Mutex<Option<GenerateSummaryJs>>>,
    active_streams: Arc<Mutex<Vec<Weak<StreamShared>>>>,
}

#[napi]
impl NativeAutoModel {
    #[napi(factory)]
    pub async fn open(model: String, opts_json: String) -> Result<Self> {
        let opts: OpenOptionsWire = serde_json::from_str(&opts_json)
            .map_err(|e| invalid_arg(format!("invalid open options: {e}")))?;

        StreamEngine::set_log_verbose(opts.verbose.unwrap_or(false));
        let kv_type: KvCacheType = opts
            .kv_cache_type
            .as_deref()
            .unwrap_or("q8_0")
            .parse()
            .map_err(|e: anyhow::Error| invalid_arg(e.to_string()))?;

        let open_opts = OpenSessionOptions {
            draft_model: opts.draft_model,
            quant: opts.quant,
            revision: opts.revision,
            cache_dir: opts.cache_dir.map(PathBuf::from),
            vram_budget_auto: opts.vram_budget_auto.unwrap_or(true),
            vram_budget_bytes: opts.vram_budget_bytes.unwrap_or(0),
            resident_layers: opts.resident_layers,
            ctx_size: opts.ctx_size.unwrap_or(4096),
            batch_size: opts.batch_size.unwrap_or(512),
            threads: opts.threads,
            kv_cache_type: kv_type,
            use_gpu: opts.use_gpu.unwrap_or(true),
            prefer_direct_io: opts.prefer_direct_io.unwrap_or(true),
            cpu_embeddings: opts.cpu_embeddings.unwrap_or(false),
            cpu_head: opts.cpu_head.unwrap_or(false),
            stream_head: opts.stream_head.unwrap_or(false),
            moe_active_streaming: opts.moe_active_streaming.unwrap_or(true),
            max_draft_tokens: opts.max_draft_tokens.unwrap_or(5).max(1),
            on_progress: None,
        };

        let loaded = run_blocking(move || {
            LoadedSession::open(&model, open_opts).map_err(|e| {
                let msg = format!("{e:#}").replace("draft_model", "draftModel");
                if msg.contains("local GGUF model not found") {
                    Error::new(Status::GenericFailure, format!("ENOENT: {msg}"))
                } else {
                    Error::new(Status::GenericFailure, msg)
                }
            })
        })
        .await?;

        let path_str = loaded.model_path.to_string_lossy().into_owned();
        Ok(Self {
            inner: Arc::new(Mutex::new(Some(loaded))),
            gate: Arc::new(EngineGate::new()),
            loaded: Arc::new(AtomicBool::new(true)),
            cached_model_path: Arc::new(Mutex::new(Some(path_str))),
            last_summary: Arc::new(Mutex::new(None)),
            active_streams: Arc::new(Mutex::new(Vec::new())),
        })
    }

    #[napi]
    pub async fn generate_with_summary(
        &self,
        prompt: String,
        opts_json: String,
        cancel_token: Option<&NativeCancelToken>,
    ) -> Result<GenerateResultJs> {
        let wire: GenerateOptionsWire = serde_json::from_str(&opts_json)
            .map_err(|e| invalid_arg(format!("invalid generate options: {e}")))?;
        let mut opts = build_generate_options(&wire)?;
        let cancel = cancel_token
            .map(|t| t.state.clone())
            .unwrap_or_else(|| Arc::new(CancelState::new()));

        let permit = self.gate.acquire_async(Some(&cancel)).await?;
        let inner = self.inner.clone();
        let last_summary = self.last_summary.clone();

        run_blocking(move || {
            let _permit = permit;
            let (text, summary) = (|| -> anyhow::Result<(String, GenerateSummary)> {
                let mut guard = inner
                    .lock()
                    .map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
                let loaded = guard
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("model has been unloaded"))?;
                if opts.max_draft_tokens == 0 {
                    opts.max_draft_tokens = loaded.default_max_draft_tokens;
                }
                let mut out = String::new();
                let summary = loaded.engine.generate_stream_with_cancel(
                    &prompt,
                    &opts,
                    loaded.draft.as_mut(),
                    Some(&cancel.flag),
                    |_tok, piece| {
                        if cancel.is_cancelled() {
                            return false;
                        }
                        out.push_str(piece);
                        true
                    },
                )?;
                Ok((out, summary))
            })()
            .map_err(map_anyhow_err)?;

            let js_summary = summary_to_js(&summary);
            if let Ok(mut ls) = last_summary.lock() {
                *ls = Some(js_summary.clone());
            }
            Ok(GenerateResultJs {
                text,
                summary: js_summary,
            })
        })
        .await
    }

    #[napi]
    pub fn generate_stream(&self, prompt: String, opts_json: String) -> Result<NativeTokenStream> {
        if !self.loaded.load(Ordering::Acquire) {
            return Err(generic_err("model has been unloaded"));
        }
        let wire: GenerateOptionsWire = serde_json::from_str(&opts_json)
            .map_err(|e| invalid_arg(format!("invalid generate options: {e}")))?;
        let opts = build_generate_options(&wire)?;
        self.spawn_token_stream(prompt, opts)
    }

    #[napi]
    pub async fn chat_with_summary(
        &self,
        messages_json: String,
        opts_json: String,
        cancel_token: Option<&NativeCancelToken>,
    ) -> Result<ChatResultJs> {
        let mut messages = parse_wire_messages(&messages_json)?;
        let wire: GenerateOptionsWire = serde_json::from_str(&opts_json)
            .map_err(|e| invalid_arg(format!("invalid chat options: {e}")))?;
        if let Some(ref tools) = wire.tools {
            inject_tools_into_messages(&mut messages, tools)?;
        }
        let add_gen = wire.add_generation_prompt.unwrap_or(true);
        let parse_out = wire.parse_output.unwrap_or(false);
        let mut opts = build_generate_options(&wire)?;

        let cancel = cancel_token
            .map(|t| t.state.clone())
            .unwrap_or_else(|| Arc::new(CancelState::new()));
        let permit = self.gate.acquire_async(Some(&cancel)).await?;
        let inner = self.inner.clone();
        let last_summary = self.last_summary.clone();

        run_blocking(move || {
            let _permit = permit;
            let (text, summary) = (|| -> anyhow::Result<(String, GenerateSummary)> {
                let mut guard = inner
                    .lock()
                    .map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
                let loaded = guard
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("model has been unloaded"))?;
                let prompt = loaded.engine.apply_chat_template(&messages, add_gen)?;
                if opts.max_draft_tokens == 0 {
                    opts.max_draft_tokens = loaded.default_max_draft_tokens;
                }
                let mut out = String::new();
                let summary = loaded.engine.generate_stream_with_cancel(
                    &prompt,
                    &opts,
                    loaded.draft.as_mut(),
                    Some(&cancel.flag),
                    |_tok, piece| {
                        if cancel.is_cancelled() {
                            return false;
                        }
                        out.push_str(piece);
                        true
                    },
                )?;
                Ok((out, summary))
            })()
            .map_err(map_anyhow_err)?;

            let js_summary = summary_to_js(&summary);
            if let Ok(mut ls) = last_summary.lock() {
                *ls = Some(js_summary.clone());
            }
            let parsed = if parse_out {
                Some(parsed_to_js(core_parse_assistant_output(&text)))
            } else {
                None
            };
            Ok(ChatResultJs {
                text,
                parsed,
                summary: js_summary,
            })
        })
        .await
    }

    #[napi]
    pub async fn chat_stream(
        &self,
        messages_json: String,
        opts_json: String,
        cancel_token: Option<&NativeCancelToken>,
    ) -> Result<NativeTokenStream> {
        if !self.loaded.load(Ordering::Acquire) {
            return Err(generic_err("model has been unloaded"));
        }
        let mut messages = parse_wire_messages(&messages_json)?;
        let wire: GenerateOptionsWire = serde_json::from_str(&opts_json)
            .map_err(|e| invalid_arg(format!("invalid chat options: {e}")))?;
        if wire.parse_output.unwrap_or(false) {
            return Err(invalid_arg(
                "parseOutput=true cannot be combined with stream=true (use StreamTagParser or OpenAI)",
            ));
        }
        if let Some(ref tools) = wire.tools {
            inject_tools_into_messages(&mut messages, tools)?;
        }
        let add_gen = wire.add_generation_prompt.unwrap_or(true);
        let opts = build_generate_options(&wire)?;

        let cancel = cancel_token.map(|t| t.state.clone());
        let permit = self.gate.acquire_async(cancel.as_ref()).await?;
        let inner = self.inner.clone();
        let prompt = run_blocking(move || {
            let _permit = permit;
            let guard = inner
                .lock()
                .map_err(|_| generic_err("model lock poisoned"))?;
            let loaded = guard
                .as_ref()
                .ok_or_else(|| generic_err("model has been unloaded"))?;
            loaded
                .engine
                .apply_chat_template(&messages, add_gen)
                .map_err(map_anyhow_err)
        })
        .await?;

        if let Some(ref c) = cancel
            && c.is_cancelled()
        {
            return Err(generic_err("ABORT_ERR: operation aborted"));
        }
        self.spawn_token_stream(prompt, opts)
    }

    fn spawn_token_stream(
        &self,
        prompt: String,
        mut opts: GenerateOptions,
    ) -> Result<NativeTokenStream> {
        let cancel = Arc::new(CancelState::new());
        let shared = Arc::new(StreamShared::new(cancel.clone()));
        {
            let mut active = self
                .active_streams
                .lock()
                .map_err(|_| generic_err("active streams lock poisoned"))?;
            if !self.loaded.load(Ordering::Acquire) {
                return Err(generic_err("model has been unloaded"));
            }
            active.retain(|w| w.strong_count() > 0);
            active.push(Arc::downgrade(&shared));
        }

        let worker_shared = Arc::clone(&shared);
        let gate = Arc::clone(&self.gate);
        let inner = Arc::clone(&self.inner);
        let last_summary = Arc::clone(&self.last_summary);

        std::thread::Builder::new()
            .name("layerstream-node-stream".to_string())
            .spawn(move || {
                let _exit_guard = WorkerExitGuard {
                    shared: Arc::clone(&worker_shared),
                };
                let res = (|| -> anyhow::Result<Option<GenerateSummary>> {
                    let Some(_permit) = gate.acquire_sync(&cancel)? else {
                        return Ok(None);
                    };
                    let mut guard = inner
                        .lock()
                        .map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
                    let loaded = guard
                        .as_mut()
                        .ok_or_else(|| anyhow::anyhow!("model has been unloaded"))?;
                    if opts.max_draft_tokens == 0 {
                        opts.max_draft_tokens = loaded.default_max_draft_tokens;
                    }
                    let summary = loaded.engine.generate_stream_with_cancel(
                        &prompt,
                        &opts,
                        loaded.draft.as_mut(),
                        Some(&cancel.flag),
                        |_tok, piece| {
                            if cancel.is_cancelled() {
                                return false;
                            }
                            if piece.is_empty() {
                                return true;
                            }
                            worker_shared.send_chunk(piece.to_string())
                        },
                    )?;
                    Ok(Some(summary))
                })();

                worker_shared.finish_producer(res, &last_summary);
            })
            .map_err(|e| generic_err(format!("failed to start stream worker thread: {e}")))?;

        Ok(NativeTokenStream { shared })
    }

    #[napi]
    pub async fn tokenize(
        &self,
        text: String,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>> {
        let permit = self.gate.acquire_async(None).await?;
        let inner = self.inner.clone();
        run_blocking(move || {
            let _permit = permit;
            let guard = inner
                .lock()
                .map_err(|_| generic_err("model lock poisoned"))?;
            let loaded = guard
                .as_ref()
                .ok_or_else(|| generic_err("model has been unloaded"))?;
            loaded
                .engine
                .tokenize(&text, add_special, parse_special)
                .map_err(map_anyhow_err)
        })
        .await
    }

    #[napi]
    pub async fn detokenize(&self, tokens: Vec<i32>) -> Result<String> {
        let permit = self.gate.acquire_async(None).await?;
        let inner = self.inner.clone();
        run_blocking(move || {
            let _permit = permit;
            let guard = inner
                .lock()
                .map_err(|_| generic_err("model lock poisoned"))?;
            let loaded = guard
                .as_ref()
                .ok_or_else(|| generic_err("model has been unloaded"))?;
            loaded.engine.detokenize(&tokens).map_err(map_anyhow_err)
        })
        .await
    }

    #[napi]
    pub async fn apply_chat_template(
        &self,
        messages_json: String,
        add_generation_prompt: bool,
        tools_json: Option<String>,
    ) -> Result<String> {
        let mut messages = parse_wire_messages(&messages_json)?;
        if let Some(raw_tools) = tools_json {
            let tools: Vec<JsonValue> = serde_json::from_str(&raw_tools)
                .map_err(|e| invalid_arg(format!("invalid tools JSON: {e}")))?;
            inject_tools_into_messages(&mut messages, &tools)?;
        }
        let permit = self.gate.acquire_async(None).await?;
        let inner = self.inner.clone();
        run_blocking(move || {
            let _permit = permit;
            let guard = inner
                .lock()
                .map_err(|_| generic_err("model lock poisoned"))?;
            let loaded = guard
                .as_ref()
                .ok_or_else(|| generic_err("model has been unloaded"))?;
            loaded
                .engine
                .apply_chat_template(&messages, add_generation_prompt)
                .map_err(map_anyhow_err)
        })
        .await
    }

    #[napi]
    pub async fn kv_clear(&self) -> Result<()> {
        let permit = self.gate.acquire_async(None).await?;
        let inner = self.inner.clone();
        run_blocking(move || {
            let _permit = permit;
            let mut guard = inner
                .lock()
                .map_err(|_| generic_err("model lock poisoned"))?;
            let loaded = guard
                .as_mut()
                .ok_or_else(|| generic_err("model has been unloaded"))?;
            loaded.engine.kv_clear();
            if let Some(ref mut d) = loaded.draft {
                d.reset();
            }
            Ok(())
        })
        .await
    }

    #[napi]
    pub async fn kv_trim(&self, pos_min: i32) -> Result<bool> {
        if pos_min < 0 {
            return Err(invalid_arg("posMin must be non-negative"));
        }
        let permit = self.gate.acquire_async(None).await?;
        let inner = self.inner.clone();
        run_blocking(move || {
            let _permit = permit;
            let mut guard = inner
                .lock()
                .map_err(|_| generic_err("model lock poisoned"))?;
            let loaded = guard
                .as_mut()
                .ok_or_else(|| generic_err("model has been unloaded"))?;
            let ok = loaded.engine.kv_trim(pos_min);
            if ok && let Some(ref mut d) = loaded.draft {
                d.trim_to(pos_min as usize);
            }
            Ok(ok)
        })
        .await
    }

    #[napi]
    pub async fn save_state(&self, path: String) -> Result<u32> {
        if path.trim().is_empty() {
            return Err(invalid_arg("path must be a non-empty string"));
        }
        let permit = self.gate.acquire_async(None).await?;
        let inner = self.inner.clone();
        run_blocking(move || {
            let _permit = permit;
            let mut guard = inner
                .lock()
                .map_err(|_| generic_err("model lock poisoned"))?;
            let loaded = guard
                .as_mut()
                .ok_or_else(|| generic_err("model has been unloaded"))?;
            let n = loaded.engine.save_state(&path).map_err(map_anyhow_err)?;
            Ok(n.min(u32::MAX as usize) as u32)
        })
        .await
    }

    #[napi]
    pub async fn load_state(&self, path: String) -> Result<u32> {
        if path.trim().is_empty() {
            return Err(invalid_arg("path must be a non-empty string"));
        }
        let p = PathBuf::from(&path);
        if !p.is_file() {
            return Err(Error::new(
                Status::GenericFailure,
                format!("ENOENT: KV state file not found: '{path}'"),
            ));
        }
        let permit = self.gate.acquire_async(None).await?;
        let inner = self.inner.clone();
        run_blocking(move || {
            let _permit = permit;
            let mut guard = inner
                .lock()
                .map_err(|_| generic_err("model lock poisoned"))?;
            let loaded = guard
                .as_mut()
                .ok_or_else(|| generic_err("model has been unloaded"))?;
            let n = loaded.engine.load_state(&p).map_err(map_anyhow_err)?;
            if let Some(ref mut d) = loaded.draft {
                d.reset();
            }
            Ok(n.min(u32::MAX as usize) as u32)
        })
        .await
    }

    #[napi]
    pub async fn stats(&self) -> Result<EngineStatsJs> {
        let permit = self.gate.acquire_async(None).await?;
        let inner = self.inner.clone();
        run_blocking(move || {
            let _permit = permit;
            let guard = inner
                .lock()
                .map_err(|_| generic_err("model lock poisoned"))?;
            let Some(loaded) = guard.as_ref() else {
                return Ok(EngineStatsJs {
                    loaded: false,
                    isolated_worker: false,
                    worker_pid: None,
                    backend_name: None,
                    cached_kv_tokens: None,
                    model_path: None,
                    draft_model_path: None,
                    source_uri: None,
                    resolved_repo: None,
                    selected_quant: None,
                    n_ctx: None,
                    vocab_size: None,
                    n_layers: None,
                    resident_layers: None,
                    streamed_layers: None,
                    resident_bytes: None,
                    cpu_resident_bytes: None,
                    slot_bytes: None,
                    expert_slot_bytes: None,
                    total_layer_loads: None,
                    moe_layers: None,
                    moe_experts_loaded: None,
                    moe_experts_total: None,
                    head_streamed: None,
                    moe_active_streaming: None,
                    is_uma_zero_copy: None,
                    is_unified_memory: None,
                    using_gpu: None,
                    can_speculate: None,
                    io: None,
                });
            };
            let st = loaded.engine.stats();
            let io = loaded.engine.io_stats();
            Ok(EngineStatsJs {
                loaded: true,
                isolated_worker: false,
                worker_pid: None,
                backend_name: Some(st.backend_name_str()),
                cached_kv_tokens: Some(loaded.engine.cached_kv_len().min(u32::MAX as usize) as u32),
                model_path: Some(loaded.model_path.to_string_lossy().into_owned()),
                draft_model_path: loaded
                    .draft_model_path
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                source_uri: loaded.source_uri.clone(),
                resolved_repo: loaded.resolved_repo.clone(),
                selected_quant: loaded.selected_quant.clone(),
                n_ctx: Some(loaded.engine.n_ctx()),
                vocab_size: Some(loaded.engine.vocab_size()),
                n_layers: Some(st.n_layers),
                resident_layers: Some(st.resident_layers),
                streamed_layers: Some(st.streamed_layers),
                resident_bytes: Some(st.resident_bytes as f64),
                cpu_resident_bytes: Some(st.cpu_resident_bytes as f64),
                slot_bytes: Some(st.slot_bytes as f64),
                expert_slot_bytes: Some(st.expert_slot_bytes as f64),
                total_layer_loads: Some(st.total_layer_loads as f64),
                moe_layers: Some(st.moe_layers),
                moe_experts_loaded: Some(st.moe_experts_loaded as f64),
                moe_experts_total: Some(st.moe_experts_total as f64),
                head_streamed: Some(st.head_streamed),
                moe_active_streaming: Some(st.moe_active_streaming),
                is_uma_zero_copy: Some(st.is_uma_zero_copy),
                is_unified_memory: Some(st.is_unified_memory),
                using_gpu: Some(st.using_gpu),
                can_speculate: Some(st.can_speculate),
                io: Some(IoStatsJs {
                    bytes_read: io.bytes_read as f64,
                    span_reads: io.span_reads as f64,
                    pread_us: io.pread_us as f64,
                    cache_hits: io.cache_hits as f64,
                    cache_misses: io.cache_misses as f64,
                    cache_bytes_served: io.cache_bytes_served as f64,
                    direct_io_active: io.direct_io_active,
                }),
            })
        })
        .await
    }

    #[napi(getter)]
    pub fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::Acquire)
    }

    #[napi(getter)]
    pub fn model_path(&self) -> Option<String> {
        if !self.loaded.load(Ordering::Acquire) {
            return None;
        }
        self.cached_model_path.lock().ok().and_then(|g| g.clone())
    }

    #[napi(getter)]
    pub fn last_summary(&self) -> Option<GenerateSummaryJs> {
        self.last_summary.lock().ok().and_then(|g| g.clone())
    }

    #[napi]
    pub async fn unload(&self) -> Result<()> {
        let streams_to_stop: Vec<Arc<StreamShared>> = {
            let mut active = self
                .active_streams
                .lock()
                .map_err(|_| generic_err("active streams lock poisoned"))?;
            self.loaded.store(false, Ordering::Release);
            active.drain(..).filter_map(|weak| weak.upgrade()).collect()
        };
        for st in &streams_to_stop {
            st.cancel_and_disconnect();
        }
        for st in &streams_to_stop {
            st.wait_worker_exited().await;
        }

        let permit = self.gate.acquire_async(None).await?;
        let inner = self.inner.clone();
        let cached_path = self.cached_model_path.clone();
        let last_summary = self.last_summary.clone();
        run_blocking(move || {
            let _permit = permit;
            let dropped = {
                let mut guard = inner
                    .lock()
                    .map_err(|_| generic_err("model lock poisoned"))?;
                if let Ok(mut cp) = cached_path.lock() {
                    *cp = None;
                }
                if let Ok(mut ls) = last_summary.lock() {
                    *ls = None;
                }
                guard.take()
            };
            drop(dropped);
            Ok(())
        })
        .await
    }
}

#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[napi]
pub async fn probe_gpu_memory(use_gpu: Option<bool>) -> Result<Option<GpuMemoryJs>> {
    let flag = use_gpu.unwrap_or(true);
    run_blocking(move || {
        Ok(
            StreamEngine::probe_gpu_memory(flag).map(|(free_b, total_b)| GpuMemoryJs {
                free_bytes: free_b as f64,
                total_bytes: total_b as f64,
            }),
        )
    })
    .await
}

#[napi]
pub async fn inspect_gguf(path: String) -> Result<GgufInfoJs> {
    run_blocking(move || {
        let pb = PathBuf::from(&path);
        let info = inspect_gguf_file(&pb).map_err(map_anyhow_err)?;
        Ok(GgufInfoJs {
            architecture: info.architecture,
            name: info.name,
            n_layers: info.n_layers,
            context_length: info.context_length,
            total_file_bytes: info.total_file_bytes as f64,
            shard_count: info.shard_count as u32,
            tensor_count: info.tensor_count as u32,
        })
    })
    .await
}

#[napi]
pub async fn pull(spec: String, cache_dir: Option<String>) -> Result<String> {
    run_blocking(move || {
        let local = pull_model(&spec, cache_dir.map(PathBuf::from), None, None, None)
            .map_err(map_anyhow_err)?;
        Ok(local.to_string_lossy().into_owned())
    })
    .await
}

#[napi(object)]
#[derive(Clone, Debug, Default)]
pub struct RecommendOptionsJs {
    pub revision: Option<String>,
    pub free_ram_gb: Option<f64>,
    pub nvme_mib_s: Option<f64>,
    pub has_draft: Option<bool>,
    pub cache_dir: Option<String>,
    pub use_gpu: Option<bool>,
}

#[napi]
pub async fn recommend(
    repo: String,
    opts: Option<RecommendOptionsJs>,
) -> Result<Vec<QuantRecommendationJs>> {
    let o = opts.unwrap_or_default();
    if let Some(gb) = o.free_ram_gb
        && (!gb.is_finite() || gb <= 0.0)
    {
        return Err(invalid_arg("freeRamGb must be positive"));
    }
    if let Some(bw) = o.nvme_mib_s
        && (!bw.is_finite() || bw <= 0.0)
    {
        return Err(invalid_arg("nvmeMibS must be positive"));
    }
    let rec_opts = RecommendOptions {
        revision: o.revision,
        free_ram_gb: o.free_ram_gb,
        nvme_mib_s: o.nvme_mib_s,
        has_draft: o.has_draft.unwrap_or(true),
        cache_dir: o.cache_dir.map(PathBuf::from),
        use_gpu: o.use_gpu.unwrap_or(true),
    };
    run_blocking(move || {
        let (_effective_repo, recs) = recommend_repo(&repo, rec_opts).map_err(map_anyhow_err)?;
        Ok(recs
            .into_iter()
            .map(|r| QuantRecommendationJs {
                resolved_repo: r.resolved_repo,
                quant: r.quant,
                primary_file: r.primary_file,
                total_bytes: r.total_bytes as f64,
                shard_count: r.shard_count,
                hf_uri: r.hf_uri,
                verdict: r.verdict,
                pinned_pct: r.pinned_pct as u32,
                est_load_s: r.est_load_s,
                est_tps_low: r.est_tps_low,
                est_tps_high: r.est_tps_high,
                recommended: r.recommended,
                score: r.score,
            })
            .collect())
    })
    .await
}

#[napi(object)]
#[derive(Clone, Debug, Default)]
pub struct CheckOptionsJs {
    pub draft_model: Option<String>,
    pub quant: Option<String>,
    pub revision: Option<String>,
    pub cache_dir: Option<String>,
    pub use_gpu: Option<bool>,
    pub allow_slow: Option<bool>,
}

#[napi]
pub async fn check(model: String, opts: Option<CheckOptionsJs>) -> Result<CheckReportJs> {
    let o = opts.unwrap_or_default();
    let chk_opts = CheckOptions {
        draft_model: o.draft_model,
        quant: o.quant,
        revision: o.revision,
        cache_dir: o.cache_dir.map(PathBuf::from),
        use_gpu: o.use_gpu.unwrap_or(true),
        allow_slow: o.allow_slow.unwrap_or(false),
    };
    run_blocking(move || {
        let rep = check_model(&model, chk_opts).map_err(|e| {
            let msg = format!("{e:#}").replace("draft_model", "draftModel");
            if msg.contains("local GGUF model not found") {
                Error::new(Status::GenericFailure, format!("ENOENT: {msg}"))
            } else {
                Error::new(Status::GenericFailure, msg)
            }
        })?;
        Ok(CheckReportJs {
            ok: rep.ok,
            verdict: rep.verdict,
            summary: rep.summary,
            model_path: rep.model_path,
            draft_model_path: rep.draft_model_path,
            architecture: rep.architecture,
            n_layers: rep.n_layers,
            context_length: rep.context_length,
            total_file_bytes: rep.total_file_bytes as f64,
            storage_mib_s: rep.storage_mib_s,
            direct_io: rep.direct_io,
            gpu_free_bytes: rep.gpu_free_bytes.map(|b| b as f64),
            gpu_total_bytes: rep.gpu_total_bytes.map(|b| b as f64),
            fits_in_vram: rep.fits_in_vram,
            est_tps_low: rep.est_tps_low,
            est_tps_high: rep.est_tps_high,
        })
    })
    .await
}

#[napi]
pub fn json_object_gbnf() -> String {
    core_json_object_gbnf()
}

#[napi]
pub fn json_schema_to_gbnf(schema_json: String) -> Result<String> {
    let val: JsonValue = serde_json::from_str(&schema_json)
        .map_err(|e| invalid_arg(format!("invalid JSON schema string: {e}")))?;
    core_json_schema_to_gbnf(&val).map_err(|e| invalid_arg(format!("invalid jsonSchema: {e:#}")))
}

#[napi]
pub fn parse_assistant_output(raw: String) -> ParsedAssistantOutputJs {
    parsed_to_js(core_parse_assistant_output(&raw))
}

fn map_worker_err(err: anyhow::Error) -> Error {
    if let Some(we) = err.downcast_ref::<WorkerError>() {
        return match we.kind.as_str() {
            "FileNotFoundError" => {
                Error::new(Status::GenericFailure, format!("ENOENT: {}", we.message))
            }
            "ValueError" => invalid_arg(we.message.clone()),
            _ => generic_err(we.message.clone()),
        };
    }
    map_anyhow_err(err)
}

fn stats_snapshot_to_js(s: EngineStatsSnapshot, worker_pid: Option<u32>) -> EngineStatsJs {
    EngineStatsJs {
        loaded: s.loaded,
        isolated_worker: s.isolated_worker,
        worker_pid,
        backend_name: Some(s.backend_name),
        cached_kv_tokens: Some(s.cached_kv_tokens as u32),
        model_path: Some(s.model_path),
        draft_model_path: s.draft_model_path,
        source_uri: s.source_uri,
        resolved_repo: s.resolved_repo,
        selected_quant: s.selected_quant,
        n_ctx: Some(s.n_ctx),
        vocab_size: Some(s.vocab_size),
        n_layers: Some(s.n_layers),
        resident_layers: Some(s.resident_layers),
        streamed_layers: Some(s.streamed_layers),
        resident_bytes: Some(s.resident_bytes as f64),
        cpu_resident_bytes: Some(s.cpu_resident_bytes as f64),
        slot_bytes: Some(s.slot_bytes as f64),
        expert_slot_bytes: Some(s.expert_slot_bytes as f64),
        total_layer_loads: Some(s.total_layer_loads as f64),
        moe_layers: Some(s.moe_layers),
        moe_experts_loaded: Some(s.moe_experts_loaded as f64),
        moe_experts_total: Some(s.moe_experts_total as f64),
        head_streamed: Some(s.head_streamed),
        moe_active_streaming: Some(s.moe_active_streaming),
        is_uma_zero_copy: Some(s.is_uma_zero_copy),
        is_unified_memory: Some(s.is_unified_memory),
        using_gpu: Some(s.using_gpu),
        can_speculate: Some(s.can_speculate),
        io: Some(IoStatsJs {
            bytes_read: s.io.bytes_read as f64,
            span_reads: s.io.span_reads as f64,
            pread_us: s.io.pread_us as f64,
            cache_hits: s.io.cache_hits as f64,
            cache_misses: s.io.cache_misses as f64,
            cache_bytes_served: s.io.cache_bytes_served as f64,
            direct_io_active: s.io.direct_io_active,
        }),
    }
}

fn wire_to_worker_gen_params(wire_json: &str) -> Result<WorkerGenParams> {
    let w: GenerateOptionsWire = serde_json::from_str(wire_json)
        .map_err(|e| invalid_arg(format!("invalid generation options JSON: {e}")))?;
    Ok(WorkerGenParams {
        max_tokens: w.max_tokens.unwrap_or(256),
        temperature: w.temperature.unwrap_or(0.0),
        top_p: w.top_p.unwrap_or(0.95),
        min_p: w.min_p.unwrap_or(0.0),
        top_k: w.top_k.unwrap_or(40),
        repeat_penalty: w.repeat_penalty.unwrap_or(1.0),
        frequency_penalty: w.frequency_penalty.unwrap_or(0.0),
        presence_penalty: w.presence_penalty.unwrap_or(0.0),
        penalty_last_n: w.penalty_last_n.unwrap_or(64),
        seed: w.seed.unwrap_or(42),
        stop: w.stop.unwrap_or_default(),
        add_generation_prompt: w.add_generation_prompt.unwrap_or(true),
        ngram_speculation: w.ngram_speculation.unwrap_or(true),
        max_draft_tokens: w.max_draft_tokens,
        tools: w.tools,
        grammar: w.grammar,
        parse_output: w.parse_output.unwrap_or(false),
    })
}

#[napi]
pub struct NativeWorkerStream {
    inner: Arc<Mutex<Option<WorkerStream>>>,
    last_summary: Arc<Mutex<Option<GenerateSummary>>>,
    cancel_handle: layerstream_core::worker_ipc::WorkerCancelHandle,
    closed: Arc<AtomicBool>,
}

#[napi]
impl NativeWorkerStream {
    #[napi(getter)]
    pub fn summary(&self) -> Option<GenerateSummaryJs> {
        let guard = self.last_summary.lock().ok()?;
        guard.as_ref().map(summary_to_js)
    }

    #[napi]
    pub fn cancel(&self) {
        self.closed.store(true, Ordering::SeqCst);
        let _ = self.cancel_handle.cancel();
        if let Ok(mut guard) = self.inner.lock()
            && let Some(stream) = guard.as_mut()
        {
            stream.close();
            if let Some(s) = stream.summary()
                && let Ok(mut sg) = self.last_summary.lock()
            {
                *sg = Some(s);
            }
        }
    }

    #[napi]
    pub async fn next_chunk(&self) -> Result<Option<String>> {
        if self.closed.load(Ordering::SeqCst) {
            return Ok(None);
        }
        let inner = Arc::clone(&self.inner);
        let last_summary = Arc::clone(&self.last_summary);
        let closed = Arc::clone(&self.closed);
        run_blocking(move || {
            let mut stream = {
                let mut guard = inner
                    .lock()
                    .map_err(|_| generic_err("worker stream lock poisoned"))?;
                let Some(s) = guard.take() else {
                    return Ok(None);
                };
                s
            };
            let res = stream.next_chunk();
            if closed.load(Ordering::SeqCst) {
                stream.close();
            }
            if let Some(s) = stream.summary()
                && let Ok(mut sg) = last_summary.lock()
            {
                *sg = Some(s);
            }
            if let Ok(mut guard) = inner.lock() {
                *guard = Some(stream);
            }
            if closed.load(Ordering::SeqCst) {
                return Ok(None);
            }
            res.map_err(map_worker_err)
        })
        .await
    }

    #[napi]
    pub async fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::SeqCst);
        let _ = self.cancel_handle.cancel();
        let inner = Arc::clone(&self.inner);
        let last_summary = Arc::clone(&self.last_summary);
        run_blocking(move || {
            let mut stream = {
                let mut guard = inner
                    .lock()
                    .map_err(|_| generic_err("worker stream lock poisoned"))?;
                let Some(s) = guard.take() else {
                    return Ok(());
                };
                s
            };
            stream.close();
            if let Some(s) = stream.summary()
                && let Ok(mut sg) = last_summary.lock()
            {
                *sg = Some(s);
            }
            if let Ok(mut guard) = inner.lock() {
                *guard = Some(stream);
            }
            Ok(())
        })
        .await
    }
}

#[napi]
pub struct NativeWorkerClient {
    inner: Arc<WorkerClient>,
}

#[napi]
impl NativeWorkerClient {
    #[napi(factory)]
    pub async fn spawn(
        program: String,
        args: Vec<String>,
        model: String,
        opts_json: String,
    ) -> Result<Self> {
        let wire: OpenOptionsWire = serde_json::from_str(&opts_json)
            .map_err(|e| invalid_arg(format!("invalid open options JSON: {e}")))?;
        let kv_cache_type: KvCacheType = wire
            .kv_cache_type
            .as_deref()
            .unwrap_or("f16")
            .parse()
            .map_err(|e: anyhow::Error| invalid_arg(format!("{e:#}")))?;
        let open = WorkerOpenParams {
            model,
            draft_model: wire.draft_model,
            quant: wire.quant,
            revision: wire.revision,
            cache_dir: wire.cache_dir.map(PathBuf::from),
            vram_budget_auto: wire.vram_budget_auto.unwrap_or(true),
            vram_budget_bytes: wire.vram_budget_bytes.unwrap_or(0),
            resident_layers: wire.resident_layers,
            ctx_size: wire.ctx_size.unwrap_or(4096),
            batch_size: wire.batch_size.unwrap_or(512),
            threads: wire.threads,
            kv_cache_type,
            use_gpu: wire.use_gpu.unwrap_or(true),
            prefer_direct_io: wire.prefer_direct_io.unwrap_or(true),
            cpu_embeddings: wire.cpu_embeddings.unwrap_or(false),
            cpu_head: wire.cpu_head.unwrap_or(false),
            stream_head: wire.stream_head.unwrap_or(false),
            moe_active_streaming: wire.moe_active_streaming.unwrap_or(true),
            max_draft_tokens: wire.max_draft_tokens.unwrap_or(5),
            verbose: wire.verbose.unwrap_or(false),
            stream_progress: false,
        };
        let client = run_blocking(move || {
            WorkerClient::spawn(&program, &args, open, None).map_err(map_worker_err)
        })
        .await?;
        Ok(Self {
            inner: Arc::new(client),
        })
    }

    #[napi(getter)]
    pub fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }

    #[napi(getter)]
    pub fn worker_pid(&self) -> Option<u32> {
        self.inner.worker_pid()
    }

    #[napi(getter)]
    pub fn model_path(&self) -> Option<String> {
        self.inner.model_path()
    }

    #[napi(getter)]
    pub fn last_summary(&self) -> Option<GenerateSummaryJs> {
        self.inner.last_summary().as_ref().map(summary_to_js)
    }

    #[napi]
    pub async fn stats(&self) -> Result<EngineStatsJs> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            if !client.is_loaded() {
                return Ok(EngineStatsJs {
                    loaded: false,
                    isolated_worker: true,
                    worker_pid: None,
                    backend_name: None,
                    cached_kv_tokens: None,
                    model_path: None,
                    draft_model_path: None,
                    source_uri: None,
                    resolved_repo: None,
                    selected_quant: None,
                    n_ctx: None,
                    vocab_size: None,
                    n_layers: None,
                    resident_layers: None,
                    streamed_layers: None,
                    resident_bytes: None,
                    cpu_resident_bytes: None,
                    slot_bytes: None,
                    expert_slot_bytes: None,
                    total_layer_loads: None,
                    moe_layers: None,
                    moe_experts_loaded: None,
                    moe_experts_total: None,
                    head_streamed: None,
                    moe_active_streaming: None,
                    is_uma_zero_copy: None,
                    is_unified_memory: None,
                    using_gpu: None,
                    can_speculate: None,
                    io: None,
                });
            }
            let pid = client.worker_pid();
            let snap = client.stats().map_err(map_worker_err)?;
            Ok(stats_snapshot_to_js(snap, pid))
        })
        .await
    }

    #[napi]
    pub async fn tokenize(
        &self,
        text: String,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            client
                .tokenize(&text, add_special, parse_special)
                .map_err(map_worker_err)
        })
        .await
    }

    #[napi]
    pub async fn detokenize(&self, tokens: Vec<i32>) -> Result<String> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || client.detokenize(&tokens).map_err(map_worker_err)).await
    }

    #[napi]
    pub async fn apply_chat_template(
        &self,
        messages_json: String,
        add_generation_prompt: bool,
        tools_json: Option<String>,
    ) -> Result<String> {
        let msgs: Vec<IncomingChatMessage> = serde_json::from_str(&messages_json)
            .map_err(|e| invalid_arg(format!("invalid chat messages payload: {e}")))?;
        let tools: Option<Vec<JsonValue>> = tools_json
            .map(|raw| serde_json::from_str(&raw))
            .transpose()
            .map_err(|e| invalid_arg(format!("invalid tools JSON: {e}")))?;
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            client
                .apply_chat_template(&msgs, add_generation_prompt, tools.as_deref())
                .map_err(map_worker_err)
        })
        .await
    }

    #[napi]
    pub async fn kv_clear(&self) -> Result<()> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || client.kv_clear().map_err(map_worker_err)).await
    }

    #[napi]
    pub async fn kv_trim(&self, pos_min: u32) -> Result<bool> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || client.kv_trim(pos_min as usize).map_err(map_worker_err)).await
    }

    #[napi]
    pub async fn save_state(&self, path: String) -> Result<u32> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            client
                .save_state(&path)
                .map(|n| n as u32)
                .map_err(map_worker_err)
        })
        .await
    }

    #[napi]
    pub async fn load_state(&self, path: String) -> Result<u32> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            client
                .load_state(&path)
                .map(|n| n as u32)
                .map_err(map_worker_err)
        })
        .await
    }

    #[napi]
    pub async fn generate_with_summary(
        &self,
        prompt: String,
        wire_json: String,
    ) -> Result<GenerateResultJs> {
        let params = wire_to_worker_gen_params(&wire_json)?;
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            let (text, summary) = client.generate(&prompt, &params).map_err(map_worker_err)?;
            Ok(GenerateResultJs {
                text,
                summary: summary.as_ref().map(summary_to_js).unwrap_or_else(|| {
                    summary_to_js(&GenerateSummary {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        finish_reason: "stop".to_string(),
                        speculation: Default::default(),
                    })
                }),
            })
        })
        .await
    }

    #[napi]
    pub async fn chat_with_summary(
        &self,
        messages_json: String,
        wire_json: String,
    ) -> Result<ChatResultJs> {
        let msgs: Vec<IncomingChatMessage> = serde_json::from_str(&messages_json)
            .map_err(|e| invalid_arg(format!("invalid chat messages payload: {e}")))?;
        let params = wire_to_worker_gen_params(&wire_json)?;
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            let (text, parsed, summary) = client.chat(&msgs, &params).map_err(map_worker_err)?;
            Ok(ChatResultJs {
                text,
                parsed: parsed.map(parsed_to_js),
                summary: summary.as_ref().map(summary_to_js).unwrap_or_else(|| {
                    summary_to_js(&GenerateSummary {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        finish_reason: "stop".to_string(),
                        speculation: Default::default(),
                    })
                }),
            })
        })
        .await
    }

    #[napi]
    pub fn generate_stream(&self, prompt: String, wire_json: String) -> Result<NativeWorkerStream> {
        let params = wire_to_worker_gen_params(&wire_json)?;
        let stream = self
            .inner
            .generate_stream(&prompt, &params)
            .map_err(map_worker_err)?;
        let cancel_handle = stream.cancel_handle();
        Ok(NativeWorkerStream {
            inner: Arc::new(Mutex::new(Some(stream))),
            last_summary: Arc::new(Mutex::new(None)),
            cancel_handle,
            closed: Arc::new(AtomicBool::new(false)),
        })
    }

    #[napi]
    pub fn chat_stream(
        &self,
        messages_json: String,
        wire_json: String,
    ) -> Result<NativeWorkerStream> {
        let msgs: Vec<IncomingChatMessage> = serde_json::from_str(&messages_json)
            .map_err(|e| invalid_arg(format!("invalid chat messages payload: {e}")))?;
        let params = wire_to_worker_gen_params(&wire_json)?;
        let stream = self
            .inner
            .chat_stream(&msgs, &params)
            .map_err(map_worker_err)?;
        let cancel_handle = stream.cancel_handle();
        Ok(NativeWorkerStream {
            inner: Arc::new(Mutex::new(Some(stream))),
            last_summary: Arc::new(Mutex::new(None)),
            cancel_handle,
            closed: Arc::new(AtomicBool::new(false)),
        })
    }

    #[napi]
    pub async fn unload(&self) -> Result<()> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            client.unload();
            Ok(())
        })
        .await
    }
}

#[napi(object)]
pub struct CompiledResponseFormatJs {
    pub grammar: Option<String>,
    pub expects_json: bool,
}

#[napi(object)]
pub struct NormalizedBudgetAndKvJs {
    pub vram_budget_auto: bool,
    pub vram_budget_bytes: f64,
    pub kv_cache_type: String,
}

#[napi]
pub fn normalize_open_budget_and_kv(
    vram_budget_json: String,
    kv_cache_type: String,
) -> Result<NormalizedBudgetAndKvJs> {
    let val: JsonValue = serde_json::from_str(&vram_budget_json)
        .map_err(|e| invalid_arg(format!("invalid vramBudgetMb JSON: {e}")))?;
    let (vram_budget_auto, bytes) =
        core_normalize_vram_budget_mb(&val).map_err(|e| invalid_arg(format!("{e:#}")))?;
    let kv =
        core_normalize_kv_cache_type(&kv_cache_type).map_err(|e| invalid_arg(format!("{e:#}")))?;
    Ok(NormalizedBudgetAndKvJs {
        vram_budget_auto,
        vram_budget_bytes: bytes as f64,
        kv_cache_type: kv.to_string(),
    })
}

#[napi]
pub fn compile_response_format(
    rf_json: String,
    allow_raw_schema: Option<bool>,
) -> Result<CompiledResponseFormatJs> {
    let val: JsonValue = serde_json::from_str(&rf_json)
        .map_err(|e| invalid_arg(format!("invalid response_format JSON: {e}")))?;
    let compiled = core_compile_response_format(&val, allow_raw_schema.unwrap_or(true))
        .map_err(|e| invalid_arg(format!("{e:#}")))?;
    Ok(CompiledResponseFormatJs {
        grammar: compiled.grammar,
        expects_json: compiled.expects_json,
    })
}

#[napi]
pub fn run_worker_server(port: u32) -> Result<()> {
    let port_u16 = u16::try_from(port).map_err(|_| invalid_arg("invalid worker port"))?;
    core_run_worker_server(port_u16).map_err(map_worker_err)
}

#[napi]
pub fn format_tools_system_prompt(tools_json: String) -> Result<String> {
    let val: JsonValue = serde_json::from_str(&tools_json)
        .map_err(|e| invalid_arg(format!("invalid tools JSON: {e}")))?;
    let JsonValue::Array(items) = val else {
        return Err(invalid_arg("tools must be an array of tool objects"));
    };
    for item in &items {
        if !item.is_object() {
            return Err(invalid_arg("each entry in 'tools' must be an object"));
        }
    }
    Ok(core_format_tools(&items))
}
