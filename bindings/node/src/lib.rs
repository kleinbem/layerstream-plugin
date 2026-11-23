use std::collections::VecDeque;
use std::future::poll_fn;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::task::{Poll, Waker};
use std::time::Duration;

use layerstream_core::hf::{self, PretrainedResolveOptions, resolve_pretrained_model};
use layerstream_core::{
    ChatMessage, DraftModelProposer, EngineConfig, GenerateOptions, GenerateSummary, GgufIndex,
    KvCacheType, LsSampleParams, ParsedAssistantOutput, StreamDelta, StreamEngine,
    StreamTagParser as CoreStreamTagParser, check_gguf_draft_compatible,
    format_tools_system_prompt as core_format_tools, json_object_gbnf as core_json_object_gbnf,
    json_schema_to_gbnf as core_json_schema_to_gbnf,
    parse_assistant_output as core_parse_assistant_output, probe_dir_storage_mib_per_s,
    probe_storage_mib_per_s,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::Deserialize;
use serde_json::Value as JsonValue;

const MIN_STREAMING_STORAGE_MIB_S: f64 = 150.0;
const OVERSIZE_GUARD_BYTES: u64 = 1024 * 1024 * 1024;

fn invalid_arg(msg: impl Into<String>) -> Error {
    Error::new(Status::InvalidArg, msg.into())
}

fn generic_err(msg: impl Into<String>) -> Error {
    Error::new(Status::GenericFailure, msg.into())
}

fn map_anyhow_err(err: anyhow::Error) -> Error {
    let msg = format!("{err:#}");
    if msg.contains("local GGUF model not found") {
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

struct LoadedEngine {
    engine: StreamEngine,
    draft: Option<DraftModelProposer>,
    model_path: PathBuf,
    draft_model_path: Option<PathBuf>,
    source_uri: Option<String>,
    resolved_repo: Option<String>,
    selected_quant: Option<String>,
    default_max_draft_tokens: usize,
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
    top_k: Option<i32>,
    seed: Option<u32>,
    stop: Option<Vec<String>>,
    ngram_speculation: Option<bool>,
    max_draft_tokens: Option<usize>,
    grammar: Option<String>,
    add_generation_prompt: Option<bool>,
    tools: Option<Vec<JsonValue>>,
    parse_output: Option<bool>,
}

#[derive(Deserialize)]
struct IncomingChatMsgWire {
    role: String,
    #[serde(default)]
    content: Option<JsonValue>,
    #[serde(default)]
    tool_calls: Option<Vec<JsonValue>>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

fn parse_wire_messages(raw_json: &str) -> Result<Vec<ChatMessage>> {
    let items: Vec<IncomingChatMsgWire> = serde_json::from_str(raw_json)
        .map_err(|e| invalid_arg(format!("invalid chat messages payload: {e}")))?;
    if items.is_empty() {
        return Err(invalid_arg("messages must not be empty"));
    }
    let mut out = Vec::with_capacity(items.len());
    for m in items {
        let mut content = match m.content {
            None | Some(JsonValue::Null) => String::new(),
            Some(JsonValue::String(s)) => s,
            Some(JsonValue::Array(parts)) => {
                let mut joined = Vec::with_capacity(parts.len());
                for p in parts {
                    match p {
                        JsonValue::String(s) => joined.push(s),
                        JsonValue::Object(map) => {
                            let Some(t) = map.get("text").and_then(JsonValue::as_str) else {
                                return Err(invalid_arg(
                                    "content part object must contain a string 'text' field",
                                ));
                            };
                            joined.push(t.to_string());
                        }
                        _ => {
                            return Err(invalid_arg(
                                "message 'content' array parts must be strings or {text: string} objects",
                            ));
                        }
                    }
                }
                joined.join("")
            }
            Some(_) => {
                return Err(invalid_arg(
                    "message 'content' must be a string, null, or array of text parts",
                ));
            }
        };

        if m.role == "assistant"
            && let Some(calls) = m.tool_calls
        {
            for tc in calls {
                let func = tc.get("function").unwrap_or(&tc);
                if let Some(name) = func.get("name").and_then(JsonValue::as_str)
                    && !name.is_empty()
                {
                    let raw_args = func.get("arguments").cloned().unwrap_or(JsonValue::Null);
                    let args_val = match raw_args {
                        JsonValue::String(ref s) => serde_json::from_str::<JsonValue>(s)
                            .unwrap_or(JsonValue::String(s.clone())),
                        other => other,
                    };
                    let call_json = serde_json::json!({
                        "name": name,
                        "arguments": args_val,
                    });
                    if !content.is_empty() && !content.ends_with('\n') {
                        content.push('\n');
                    }
                    content.push_str(&format!("<tool_call>{call_json}</tool_call>"));
                }
            }
        }

        if m.role == "tool" {
            let mut prefix = String::new();
            if let Some(name) = m.name
                && !name.is_empty()
            {
                prefix.push_str(&format!("tool={name} "));
            }
            if let Some(id) = m.tool_call_id
                && !id.is_empty()
            {
                prefix.push_str(&format!("call_id={id}: "));
            }
            if !prefix.is_empty() {
                content = format!("{prefix}{content}");
            }
        }

        out.push(ChatMessage {
            role: m.role,
            content,
        });
    }
    Ok(out)
}

fn inject_tools_into_messages(messages: &mut Vec<ChatMessage>, tools: &[JsonValue]) -> Result<()> {
    if tools.is_empty() {
        return Ok(());
    }
    for t in tools {
        if !t.is_object() {
            return Err(invalid_arg("each entry in 'tools' must be an object"));
        }
    }
    let tool_prompt = core_format_tools(tools);
    if let Some(first) = messages.first_mut()
        && first.role == "system"
    {
        if !first.content.is_empty() {
            first.content.push_str("\n\n");
        }
        first.content.push_str(&tool_prompt);
    } else {
        messages.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: tool_prompt,
            },
        );
    }
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
    Ok(GenerateOptions {
        max_tokens: wire.max_tokens.unwrap_or(256),
        sample: LsSampleParams {
            temperature: wire.temperature.unwrap_or(0.0),
            top_p: wire.top_p.unwrap_or(0.95),
            top_k: wire.top_k.unwrap_or(40),
            seed: wire.seed.unwrap_or(42),
        },
        stop: wire.stop.clone().unwrap_or_default(),
        enable_ngram_speculation: wire.ngram_speculation.unwrap_or(true),
        max_draft_tokens: wire.max_draft_tokens.unwrap_or(0),
        grammar,
    })
}

#[napi]
pub struct NativeAutoModel {
    inner: Arc<Mutex<Option<LoadedEngine>>>,
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

        let verbose = opts.verbose.unwrap_or(false);
        StreamEngine::set_log_verbose(verbose);

        let kv_str = opts.kv_cache_type.as_deref().unwrap_or("q8_0");
        let kv_type: KvCacheType = kv_str
            .parse()
            .map_err(|e: anyhow::Error| invalid_arg(e.to_string()))?;

        let use_gpu = opts.use_gpu.unwrap_or(true);
        let ctx_size = opts.ctx_size.unwrap_or(4096);
        let batch_size = opts.batch_size.unwrap_or(512);
        let prefer_direct_io = opts.prefer_direct_io.unwrap_or(true);
        let max_draft_tokens = opts.max_draft_tokens.unwrap_or(5).max(1);
        let vram_budget_auto = opts.vram_budget_auto.unwrap_or(true);
        let vram_budget_bytes = opts.vram_budget_bytes.unwrap_or(0);
        let resident_layers = opts.resident_layers;
        let cpu_embeddings = opts.cpu_embeddings.unwrap_or(false);
        let cpu_head = opts.cpu_head.unwrap_or(false);
        let stream_head = opts.stream_head.unwrap_or(false);
        let moe_active_streaming = opts.moe_active_streaming.unwrap_or(true);

        let n_threads = opts.threads.unwrap_or_else(|| {
            if use_gpu {
                2
            } else {
                std::thread::available_parallelism()
                    .map(|n| n.get().min(8) as u32)
                    .unwrap_or(4)
            }
        });

        let cache_dir = opts.cache_dir.map(PathBuf::from);
        let draft_owned = opts.draft_model;
        let quant_owned = opts.quant;
        let rev_owned = opts.revision;

        let loaded = run_blocking(move || {
            let res = (|| -> anyhow::Result<LoadedEngine> {
                let target_resolve = PretrainedResolveOptions {
                    revision: rev_owned.clone(),
                    quant: quant_owned,
                    cache_dir: cache_dir.clone(),
                    vram_budget_bytes: if vram_budget_bytes > 0 {
                        Some(vram_budget_bytes)
                    } else {
                        None
                    },
                    use_gpu,
                    has_draft: draft_owned.is_some(),
                };
                let resolved_target = resolve_pretrained_model(&model, &target_resolve)?;

                let (draft_proposer, draft_path) = if let Some(draft_spec) = draft_owned {
                    let draft_resolve = PretrainedResolveOptions {
                        revision: rev_owned,
                        quant: None,
                        cache_dir,
                        vram_budget_bytes: None,
                        use_gpu,
                        has_draft: false,
                    };
                    let resolved_draft = resolve_pretrained_model(&draft_spec, &draft_resolve)?;
                    let mut draft_cfg = EngineConfig::reference(&resolved_draft.local_path);
                    draft_cfg.n_ctx = ctx_size;
                    draft_cfg.n_batch = batch_size.clamp(16, 64);
                    draft_cfg.n_threads = n_threads;
                    draft_cfg.kv_cache_type = kv_type;
                    draft_cfg.use_gpu = use_gpu;
                    draft_cfg.prefer_direct_io = prefer_direct_io;
                    let proposer = DraftModelProposer::open(draft_cfg, max_draft_tokens)?;
                    (Some(proposer), Some(resolved_draft.local_path))
                } else {
                    (None, None)
                };

                let mut cfg = EngineConfig::streaming(&resolved_target.local_path);
                cfg.n_ctx = ctx_size;
                cfg.n_batch = batch_size.max(1);
                cfg.n_threads = n_threads.max(1);
                cfg.resident_layers = resident_layers;
                cfg.vram_budget_bytes = vram_budget_bytes;
                cfg.vram_budget_auto = vram_budget_auto && resident_layers.is_none();
                cfg.kv_cache_type = kv_type;
                cfg.use_gpu = use_gpu;
                cfg.prefer_direct_io = prefer_direct_io;
                cfg.cpu_embeddings = cpu_embeddings;
                cfg.cpu_head = cpu_head;
                cfg.stream_head = stream_head;
                cfg.moe_active_streaming = moe_active_streaming;

                let engine = StreamEngine::open(&cfg)?;
                if let Some(ref proposer) = draft_proposer {
                    engine
                        .check_draft_compatible(proposer.engine())
                        .map_err(|reason| {
                            anyhow::anyhow!("incompatible draftModel tokenizer: {reason}")
                        })?;
                }

                Ok(LoadedEngine {
                    engine,
                    draft: draft_proposer,
                    model_path: resolved_target.local_path,
                    draft_model_path: draft_path,
                    source_uri: resolved_target.source_uri,
                    resolved_repo: resolved_target.resolved_repo,
                    selected_quant: resolved_target.selected_quant,
                    default_max_draft_tokens: max_draft_tokens,
                })
            })();
            res.map_err(map_anyhow_err)
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
        if !pb.is_file() {
            return Err(generic_err(format!(
                "ENOENT: local GGUF model not found: '{}'",
                pb.display()
            )));
        }
        let idx = GgufIndex::open(&pb).map_err(map_anyhow_err)?;
        Ok(GgufInfoJs {
            architecture: idx.meta.architecture,
            name: idx.meta.name,
            n_layers: idx.meta.n_layers,
            context_length: idx.meta.context_length,
            total_file_bytes: idx.meta.total_file_bytes as f64,
            shard_count: idx.shards.len() as u32,
            tensor_count: idx.tensors.len() as u32,
        })
    })
    .await
}

#[napi]
pub async fn pull(spec: String, cache_dir: Option<String>) -> Result<String> {
    run_blocking(move || {
        let root = cache_dir
            .map(PathBuf::from)
            .unwrap_or_else(hf::default_cache_dir);
        let local = hf::ensure_hf_downloaded(&spec, &root).map_err(map_anyhow_err)?;
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
    let has_draft = o.has_draft.unwrap_or(true);
    let use_gpu = o.use_gpu.unwrap_or(true);
    let rev_owned = o.revision;
    let cache_dir = o.cache_dir.map(PathBuf::from);
    let free_ram_gb = o.free_ram_gb;
    let nvme_mib_s = o.nvme_mib_s;

    run_blocking(move || {
        let (effective_repo, recs) =
            (|| -> anyhow::Result<(String, Vec<hf::QuantRecommendation>)> {
                let (repo_id, rev) = hf::parse_repo_and_revision(&repo, rev_owned.as_deref())?;
                let mut effective_repo = repo_id.clone();
                let mut candidates = hf::list_hf_gguf_candidates(&repo_id, &rev)?;

                if candidates.is_empty() && !repo_id.to_ascii_lowercase().ends_with("-gguf") {
                    let sibling = format!("{repo_id}-GGUF");
                    if let Ok(sib_cands) = hf::list_hf_gguf_candidates(&sibling, &rev)
                        && !sib_cands.is_empty()
                    {
                        effective_repo = sibling;
                        candidates = sib_cands;
                    }
                }
                if candidates.is_empty() {
                    anyhow::bail!(
                        "no .gguf files found in Hugging Face repository '{repo_id}@{rev}'"
                    );
                }

                let free_vram_bytes = if let Some(gb) = free_ram_gb {
                    (gb.max(0.5) * 1024.0 * 1024.0 * 1024.0) as u64
                } else if use_gpu && let Some((free, _)) = StreamEngine::probe_gpu_memory(true) {
                    free
                } else {
                    4 * 1024 * 1024 * 1024
                };

                let storage_bw = if let Some(bw) = nvme_mib_s {
                    bw.max(10.0)
                } else {
                    let probe_dir = cache_dir.unwrap_or_else(hf::default_cache_dir);
                    let _ = std::fs::create_dir_all(&probe_dir);
                    probe_dir_storage_mib_per_s(&probe_dir).unwrap_or(1500.0)
                };

                let recs = hf::recommend_candidates(
                    &effective_repo,
                    &rev,
                    &candidates,
                    free_vram_bytes,
                    storage_bw,
                    has_draft,
                );
                Ok((effective_repo, recs))
            })()
            .map_err(map_anyhow_err)?;

        Ok(recs
            .into_iter()
            .map(|r| QuantRecommendationJs {
                resolved_repo: effective_repo.clone(),
                quant: r.candidate.quant,
                primary_file: r.candidate.primary_file,
                total_bytes: r.candidate.total_bytes as f64,
                shard_count: r.candidate.shard_count,
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
    let use_gpu = o.use_gpu.unwrap_or(true);
    let allow_slow = o.allow_slow.unwrap_or(false);
    let draft_owned = o.draft_model;
    let quant_owned = o.quant;
    let rev_owned = o.revision;
    let cache_dir = o.cache_dir.map(PathBuf::from);

    run_blocking(move || {
        let res = (|| -> anyhow::Result<CheckReportJs> {
            let resolve_opts = PretrainedResolveOptions {
                revision: rev_owned.clone(),
                quant: quant_owned,
                cache_dir: cache_dir.clone(),
                vram_budget_bytes: None,
                use_gpu,
                has_draft: draft_owned.is_some(),
            };
            let resolved_target = resolve_pretrained_model(&model, &resolve_opts)?;
            let idx = GgufIndex::open(&resolved_target.local_path)?;

            let (draft_path, draft_reserve) = if let Some(d_spec) = draft_owned {
                let d_opts = PretrainedResolveOptions {
                    revision: rev_owned,
                    quant: None,
                    cache_dir,
                    vram_budget_bytes: None,
                    use_gpu,
                    has_draft: false,
                };
                let resolved_draft = resolve_pretrained_model(&d_spec, &d_opts)?;
                check_gguf_draft_compatible(&resolved_target.local_path, &resolved_draft.local_path)
                    .map_err(|e| anyhow::anyhow!("incompatible draftModel tokenizer: {e}"))?;
                let draft_idx = GgufIndex::open(&resolved_draft.local_path)?;
                let reserve = draft_idx
                    .meta
                    .total_file_bytes
                    .saturating_add(64 * 1024 * 1024);
                (Some(resolved_draft.local_path), reserve)
            } else {
                (None, 0u64)
            };

            let gpu_mem = if use_gpu {
                let probed = StreamEngine::probe_gpu_memory(true);
                if probed.is_none() {
                    anyhow::bail!(
                        "no usable GPU device found (CUDA/Vulkan/Metal); pass useGpu: false for CPU mode"
                    );
                }
                probed
            } else {
                None
            };

            let size_bytes = idx.meta.total_file_bytes;
            let size_gib = size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let fits_in_vram = gpu_mem
                .map(|(free_b, _)| {
                    size_bytes
                        .saturating_add(draft_reserve)
                        .saturating_add(768 * 1024 * 1024)
                        < free_b
                })
                .unwrap_or(false);

            let probe_res = probe_storage_mib_per_s(&resolved_target.local_path, true).ok();
            let (mib_s, direct_ok) = probe_res.unwrap_or((0.0, false));
            let is_oversize = size_bytes >= OVERSIZE_GUARD_BYTES && !fits_in_vram;

            if is_oversize && use_gpu && !allow_slow {
                if probe_res.is_none() || !mib_s.is_finite() || mib_s <= 0.0 {
                    anyhow::bail!(
                        "could not measure storage throughput for '{}' ({size_gib:.1} GiB) \
                         required for out-of-core GPU streaming; pass allowSlow: true to bypass preflight.",
                        resolved_target.local_path.display()
                    );
                }
                if !direct_ok {
                    anyhow::bail!(
                        "storage throughput for '{}' ({size_gib:.1} GiB) is unverified because \
                         direct I/O (O_DIRECT/F_NOCACHE) was unavailable (measured {mib_s:.0} MiB/s from page cache). \
                         Move the GGUF to a direct-I/O filesystem or pass allowSlow: true.",
                        resolved_target.local_path.display()
                    );
                }
                if mib_s < MIN_STREAMING_STORAGE_MIB_S {
                    anyhow::bail!(
                        "storage for '{}' ({size_gib:.1} GiB) measured only {mib_s:.0} MiB/s \
                         (minimum recommended {MIN_STREAMING_STORAGE_MIB_S:.0} MiB/s NVMe). \
                         Move the GGUF to local NVMe or pass allowSlow: true.",
                        resolved_target.local_path.display()
                    );
                }
            }

            let pinned_mib = gpu_mem.map_or(0.0, |(f, _)| {
                (f.saturating_sub(draft_reserve) as f64 * 0.45) / (1024.0 * 1024.0)
            });
            let est_stream_mib = ((size_bytes as f64 / (1024.0 * 1024.0)) - pinned_mib).max(0.0);
            let base_tps = if est_stream_mib > 1.0 && mib_s > 1.0 {
                mib_s / est_stream_mib
            } else {
                0.0
            };

            let (verdict, est_tps_low, est_tps_high, summary) = if fits_in_vram {
                (
                    "GOOD",
                    15.0,
                    50.0,
                    format!("GOOD (100% resident in VRAM, {size_gib:.2} GiB)"),
                )
            } else if draft_path.is_some() {
                let lo = (base_tps * 2.5 * 100.0).round() / 100.0;
                let hi = (base_tps * 5.0 * 100.0).round() / 100.0;
                (
                    "GOOD",
                    lo,
                    hi,
                    format!(
                        "GOOD ({size_gib:.2} GiB streamed + draft speculator, storage={mib_s:.0} MiB/s, direct_io={direct_ok}, est ~{lo:.2}-{hi:.2} tok/s)"
                    ),
                )
            } else if size_bytes >= 4 * 1024 * 1024 * 1024 {
                let tps = (base_tps * 100.0).round() / 100.0;
                (
                    "SLOW",
                    tps,
                    tps,
                    format!(
                        "SLOW ({size_gib:.2} GiB streamed without draftModel, storage={mib_s:.0} MiB/s, est ~{tps:.2} tok/s)"
                    ),
                )
            } else {
                let tps = (base_tps * 100.0).round() / 100.0;
                (
                    "OK",
                    tps,
                    tps,
                    format!(
                        "OK ({size_gib:.2} GiB, storage={mib_s:.0} MiB/s, direct_io={direct_ok})"
                    ),
                )
            };

            Ok(CheckReportJs {
                ok: true,
                verdict: verdict.to_string(),
                summary,
                model_path: resolved_target.local_path.to_string_lossy().into_owned(),
                draft_model_path: draft_path.map(|p| p.to_string_lossy().into_owned()),
                architecture: idx.meta.architecture,
                n_layers: idx.meta.n_layers,
                context_length: idx.meta.context_length,
                total_file_bytes: size_bytes as f64,
                storage_mib_s: mib_s,
                direct_io: direct_ok,
                gpu_free_bytes: gpu_mem.map(|(f, _)| f as f64),
                gpu_total_bytes: gpu_mem.map(|(_, t)| t as f64),
                fits_in_vram,
                est_tps_low,
                est_tps_high,
            })
        })();
        res.map_err(map_anyhow_err)
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
