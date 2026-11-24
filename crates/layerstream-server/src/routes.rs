//! OpenAI-compatible HTTP routes (`/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/health`)
//! with SSE streaming, tool-call & thinking extraction, client-disconnect cancellation, and
//! lazy 0-VRAM idle residency.

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use layerstream_core::{
    CancelledError, ChatMessage, DraftModelProposer, EngineConfig, GenerateOptions, KvCacheType,
    LsSampleParams, PromptTooLongError, Residency, StreamEngine,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::parse::{
    IncomingChatMessage, StreamDelta, StreamTagParser, inject_tools_into_messages,
    parse_assistant_output,
};
use crate::registry::{ModelEntry, ModelRegistry};

static SEED_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_random_seed() -> u32 {
    let n = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut x = (unix_now_nanos() as u64)
        ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (std::process::id() as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    // SplitMix64 finalizer for well-distributed per-request seeds.
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x as u32) ^ ((x >> 32) as u32)
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub registry: ModelRegistry,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub resident_layers: Option<u32>,
    pub vram_budget_bytes: u64,
    pub vram_budget_auto: bool,
    pub kv_cache_type: KvCacheType,
    pub use_gpu: bool,
    pub prefer_direct_io: bool,
    pub cpu_embeddings: bool,
    pub cpu_head: bool,
    pub stream_head: bool,
    pub moe_active_streaming: bool,
    pub idle_timeout: Duration,
    pub enable_ngram_speculation: bool,
    pub max_draft_tokens: usize,
    pub max_queued_requests: usize,
}

pub struct LoadedPair {
    pub target: StreamEngine,
    pub draft: Option<DraftModelProposer>,
}

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ServerConfig>,
    pub registry: Arc<RwLock<ModelRegistry>>,
    pub loading_status: Arc<RwLock<Option<String>>>,
    pub loading_error: Arc<RwLock<Option<String>>>,
    pub residency: Arc<Residency<LoadedPair>>,
    pub shutdown: Arc<AtomicBool>,
    active_cancel: Arc<StdMutex<Option<Arc<AtomicBool>>>>,
}

impl AppState {
    pub fn new(config: ServerConfig) -> Self {
        let registry = Arc::new(RwLock::new(config.registry.clone()));
        let residency = Arc::new(Residency::new(
            config.idle_timeout,
            config.max_queued_requests.max(1),
        ));
        Self {
            config: Arc::new(config),
            registry,
            loading_status: Arc::new(RwLock::new(None)),
            loading_error: Arc::new(RwLock::new(None)),
            residency,
            shutdown: Arc::new(AtomicBool::new(false)),
            active_cancel: Arc::new(StdMutex::new(None)),
        }
    }

    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Ok(guard) = self.active_cancel.lock()
            && let Some(flag) = guard.as_ref()
        {
            flag.store(true, Ordering::SeqCst);
        }
    }

    pub fn model_count(&self) -> usize {
        self.registry.read().map(|r| r.len()).unwrap_or(0)
    }

    pub fn registry_snapshot(&self) -> ModelRegistry {
        self.registry.read().map(|r| r.clone()).unwrap_or_default()
    }

    pub fn set_registry(&self, new_registry: ModelRegistry) {
        if let Ok(mut guard) = self.registry.write() {
            *guard = new_registry;
        }
    }

    pub fn set_loading_status(&self, status: Option<String>) {
        if let Ok(mut guard) = self.loading_status.write() {
            *guard = status;
        }
    }

    pub fn set_loading_error(&self, err: Option<String>) {
        if let Ok(mut guard) = self.loading_error.write() {
            *guard = err;
        }
    }

    /// Atomically publishes a background pull failure before clearing `loading_status`,
    /// avoiding a transient window where neither status nor error is visible.
    pub fn set_loading_failed(&self, err: String) {
        self.set_loading_error(Some(err));
        self.set_loading_status(None);
    }

    pub fn loading_status(&self) -> Option<String> {
        self.loading_status.read().ok().and_then(|g| g.clone())
    }

    pub fn loading_error(&self) -> Option<String> {
        self.loading_error.read().ok().and_then(|g| g.clone())
    }

    fn register_active_cancel(&self, flag: Arc<AtomicBool>) -> ActiveCancelGuard {
        if let Ok(mut guard) = self.active_cancel.lock() {
            if self.shutdown.load(Ordering::SeqCst) {
                flag.store(true, Ordering::SeqCst);
            }
            *guard = Some(flag.clone());
        } else if self.shutdown.load(Ordering::SeqCst) {
            flag.store(true, Ordering::SeqCst);
        }
        ActiveCancelGuard {
            slot: self.active_cancel.clone(),
            flag,
        }
    }

    pub fn spawn_reaper(&self) -> Option<tokio::task::JoinHandle<()>> {
        if self.config.idle_timeout.is_zero() {
            return None;
        }
        let residency = self.residency.clone();
        let shutdown = self.shutdown.clone();
        Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            loop {
                interval.tick().await;
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
                if residency.evict_idle(Instant::now()) {
                    tracing::info!(
                        "evicted idle layerstream model (0 MB model/KV weights; driver context only)"
                    );
                }
            }
        }))
    }

    fn load_pair(&self, entry: &ModelEntry) -> Result<LoadedPair> {
        let mut cfg = EngineConfig::streaming(&entry.path);
        cfg.n_ctx = self.config.n_ctx;
        cfg.n_batch = self.config.n_batch;
        cfg.n_threads = self.config.n_threads;
        cfg.resident_layers = self.config.resident_layers;
        cfg.vram_budget_auto = self.config.vram_budget_auto;
        let draft_reserve = entry.draft_path.as_ref().map_or(0, |draft_path| {
            let draft_bytes = layerstream_core::GgufIndex::open(draft_path)
                .map(|idx| idx.meta.total_file_bytes)
                .unwrap_or_else(|_| std::fs::metadata(draft_path).map(|m| m.len()).unwrap_or(0));
            // Account for draft weights + compute buffer (~64 MiB) + draft KV cache scaled by ctx & kv type.
            let kv_scale_num: u64 = match self.config.kv_cache_type {
                KvCacheType::F16 => 4,
                KvCacheType::Q8_0 => 2,
                KvCacheType::Q4_0 => 1,
            };
            // ~16 KiB/token at Q4_0 -> ~32 KiB/tok at Q8_0 -> ~64 KiB/tok at F16 (conservative for 0.5B-1.5B).
            let draft_kv_bytes = (self.config.n_ctx as u64)
                .saturating_mul(16 * 1024)
                .saturating_mul(kv_scale_num);
            draft_bytes
                .saturating_add(64 * 1024 * 1024)
                .saturating_add(draft_kv_bytes)
        });
        if let Some(draft_path) = &entry.draft_path
            && let Some((free_b, _)) = StreamEngine::probe_gpu_memory(self.config.use_gpu)
            && draft_reserve * 10 > free_b * 3
        {
            tracing::warn!(
                draft = %draft_path.display(),
                draft_mib = draft_reserve / (1024 * 1024),
                free_vram_mib = free_b / (1024 * 1024),
                "draft model uses >30% of available GPU memory; consider a smaller ~0.5B draft GGUF so more target layers stay resident"
            );
        }
        cfg.vram_budget_bytes = if self.config.vram_budget_auto {
            // In auto mode, pass draft_reserve so C++ subtracts it from the probed safe budget.
            draft_reserve
        } else if self.config.vram_budget_bytes > 0 {
            self.config.vram_budget_bytes.saturating_sub(draft_reserve)
        } else {
            0
        };
        cfg.kv_cache_type = self.config.kv_cache_type;
        cfg.use_gpu = self.config.use_gpu;
        cfg.prefer_direct_io = self.config.prefer_direct_io;
        cfg.cpu_embeddings = self.config.cpu_embeddings;
        cfg.cpu_head = self.config.cpu_head;
        cfg.stream_head = self.config.stream_head;
        cfg.moe_active_streaming = self.config.moe_active_streaming;

        let target = StreamEngine::open(&cfg)?;
        let draft = match &entry.draft_path {
            Some(draft_path) => {
                let mut dcfg = EngineConfig::reference(draft_path);
                dcfg.n_ctx = self.config.n_ctx;
                dcfg.n_batch = self.config.n_batch;
                dcfg.n_threads = self.config.n_threads;
                dcfg.kv_cache_type = self.config.kv_cache_type;
                dcfg.use_gpu = self.config.use_gpu;
                let proposer = DraftModelProposer::open(dcfg, self.config.max_draft_tokens)?;
                if let Err(reason) = target.check_draft_compatible(proposer.engine()) {
                    bail!(
                        "incompatible --draft-model {} for {}: {reason}",
                        draft_path.display(),
                        entry.id
                    );
                }
                Some(proposer)
            }
            None => None,
        };

        let stats = target.stats();
        tracing::info!(
            model = %entry.id,
            n_layers = stats.n_layers,
            resident_layers = stats.resident_layers,
            streamed_layers = stats.streamed_layers,
            resident_mb = stats.resident_bytes / (1024 * 1024),
            cpu_resident_mb = stats.cpu_resident_bytes / (1024 * 1024),
            slot_mb = stats.slot_bytes / (1024 * 1024),
            expert_slot_mb = stats.expert_slot_bytes / (1024 * 1024),
            head_streamed = stats.head_streamed,
            moe_layers = stats.moe_layers,
            moe_active_streaming = stats.moe_active_streaming,
            uma_zero_copy = stats.is_uma_zero_copy,
            unified_memory = stats.is_unified_memory,
            gpu = stats.using_gpu,
            can_speculate = stats.can_speculate,
            "loaded layerstream model"
        );
        Ok(LoadedPair { target, draft })
    }
}

struct ActiveCancelGuard {
    slot: Arc<StdMutex<Option<Arc<AtomicBool>>>>,
    flag: Arc<AtomicBool>,
}

impl Drop for ActiveCancelGuard {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.slot.lock()
            && guard.as_ref().is_some_and(|f| Arc::ptr_eq(f, &self.flag))
        {
            *guard = None;
        }
    }
}

pub const WORKER_TOKEN_HEADER: &str = "x-layerstream-worker-token";
pub const MAX_REQUEST_BODY_BYTES: usize = 16 * 1024 * 1024;

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/unload", post(unload_model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(text_completions))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .with_state(state)
}

/// Wraps an Axum router with optional `Authorization: Bearer <api_key>` enforcement on all routes
/// except `/health`. An explicitly provided empty/whitespace `Some("")` fails closed (`401`).
pub fn with_api_key_auth(router: Router, api_key: Option<String>) -> Router {
    let Some(raw_key) = api_key else {
        return router;
    };
    let trimmed = raw_key.trim();
    if trimmed.is_empty() {
        return router.layer(axum::middleware::from_fn(
            move |req: axum::extract::Request, next: axum::middleware::Next| async move {
                if req.uri().path() == "/health" {
                    return next.run(req).await;
                }
                api_error(
                    StatusCode::UNAUTHORIZED,
                    "Unauthorized (empty API key configured)",
                )
            },
        ));
    }
    let expected = Arc::new(format!("Bearer {trimmed}").into_bytes());
    router.layer(axum::middleware::from_fn(
        move |req: axum::extract::Request, next: axum::middleware::Next| {
            let expected = expected.clone();
            async move {
                if req.uri().path() == "/health" {
                    return next.run(req).await;
                }
                let authorized = req
                    .headers()
                    .get(axum::http::header::AUTHORIZATION)
                    .map(|v| ct_eq(v.as_bytes(), &expected))
                    .unwrap_or(false);
                if !authorized {
                    return api_error(StatusCode::UNAUTHORIZED, "Unauthorized");
                }
                next.run(req).await
            }
        },
    ))
}

/// Builds the standalone HTTP router enforcing an optional `Authorization: Bearer <api_key>`
/// on `/v1/*` routes while keeping `/health` unauthenticated for liveness probes.
/// An explicitly provided empty/whitespace `Some("")` fails closed (`401`) rather than disabling auth.
pub fn build_router_with_api_key(state: AppState, api_key: Option<String>) -> Router {
    with_api_key_auth(build_router(state), api_key)
}

/// Builds the worker router requiring a matching `x-layerstream-worker-token` header on every request.
pub fn build_router_with_token(state: AppState, expected_token: String) -> Router {
    let expected = Arc::new(expected_token.into_bytes());
    build_router(state).layer(axum::middleware::from_fn(
        move |req: axum::extract::Request, next: axum::middleware::Next| {
            let expected = expected.clone();
            async move {
                let authorized = req
                    .headers()
                    .get(WORKER_TOKEN_HEADER)
                    .map(|v| ct_eq(v.as_bytes(), &expected))
                    .unwrap_or(false);
                if !authorized {
                    return api_error(StatusCode::FORBIDDEN, "missing or invalid worker token");
                }
                next.run(req).await
            }
        },
    ))
}

async fn unload_model(State(state): State<AppState>) -> impl IntoResponse {
    let unloaded = state.residency.evict_now().await;
    if unloaded {
        tracing::info!("unloaded layerstream model via /v1/unload");
    }
    Json(json!({
        "unloaded": unloaded,
        "loaded": state.residency.is_loaded(),
        "worker_isolated": false,
    }))
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    if let Some(err) = state.loading_error() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "status": "error",
                "error": err,
                "models": state.model_count(),
                "loaded": state.residency.is_loaded(),
            })),
        )
            .into_response();
    }
    if let Some(status) = state.loading_status() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "loading",
                "detail": status,
                "models": state.model_count(),
                "loaded": false,
            })),
        )
            .into_response();
    }
    Json(json!({
        "status": "ok",
        "models": state.model_count(),
        "loaded": state.residency.is_loaded(),
    }))
    .into_response()
}

pub fn models_list_json(registry: &ModelRegistry) -> Value {
    let now = unix_now();
    let data: Vec<Value> = registry
        .list()
        .into_iter()
        .map(|entry| {
            json!({
                "id": entry.id,
                "object": "model",
                "created": now,
                "owned_by": "layerstream",
                "workload_class": "causal_generation",
                "availability": "loadable",
                "meta": {
                    "architecture": entry.meta.architecture,
                    "n_layers": entry.meta.n_layers,
                    "context_length": entry.meta.context_length,
                    "file_bytes": entry.meta.total_file_bytes,
                }
            })
        })
        .collect();

    json!({
        "object": "list",
        "data": data,
    })
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let payload = state
        .registry
        .read()
        .map(|r| models_list_json(&r))
        .unwrap_or_else(|_| json!({ "object": "list", "data": [] }));
    Json(payload)
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<IncomingChatMessage>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    penalty_last_n: Option<i32>,
    #[serde(default)]
    seed: Option<u32>,
    #[serde(default)]
    stop: Option<StopSpec>,
    #[serde(default)]
    response_format: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopSpec {
    Single(String),
    Many(Vec<String>),
}

impl StopSpec {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

#[derive(Debug, Deserialize)]
struct TextCompletionRequest {
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    penalty_last_n: Option<i32>,
    #[serde(default)]
    seed: Option<u32>,
    #[serde(default)]
    stop: Option<StopSpec>,
    #[serde(default)]
    response_format: Option<Value>,
}

#[derive(Debug, Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    kind: String,
}

pub fn api_error(status: StatusCode, msg: impl Into<String>) -> Response {
    let kind = if status == StatusCode::UNAUTHORIZED {
        "authentication_error"
    } else if status == StatusCode::BAD_REQUEST
        || status == StatusCode::NOT_FOUND
        || status == StatusCode::FORBIDDEN
    {
        "invalid_request_error"
    } else if status == StatusCode::SERVICE_UNAVAILABLE {
        "service_unavailable"
    } else {
        "server_error"
    };
    (
        status,
        Json(ErrorEnvelope {
            error: ErrorBody {
                message: msg.into(),
                kind: kind.to_string(),
            },
        }),
    )
        .into_response()
}

struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn emit_stream_deltas(
    deltas: Vec<StreamDelta>,
    req_id: &str,
    created: u64,
    model: &str,
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
) -> bool {
    for d in deltas {
        let delta_json = match d {
            StreamDelta::Content(s) if !s.is_empty() => json!({ "content": s }),
            StreamDelta::Reasoning(s) if !s.is_empty() => json!({ "reasoning_content": s }),
            StreamDelta::ToolCall {
                index,
                id,
                name,
                arguments,
            } => json!({
                "tool_calls": [{
                    "index": index,
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    }
                }]
            }),
            _ => continue,
        };
        let chunk = json!({
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta_json,
                "finish_reason": null,
            }]
        });
        if tx
            .send(Ok(Event::default().data(chunk.to_string())))
            .is_err()
        {
            return false;
        }
    }
    true
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let entry = match resolve_model(&state, req.model.as_deref()) {
        Ok(e) => e,
        Err(resp) => return *resp,
    };

    let allow_tool_calls = match &req.tool_choice {
        None => true,
        Some(Value::String(s)) if s == "auto" => true,
        Some(Value::String(s)) if s == "none" => false,
        Some(other) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "unsupported tool_choice {other}; only \"auto\" and \"none\" are supported"
                ),
            );
        }
    };

    let sample = LsSampleParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(0.95),
        min_p: req.min_p.unwrap_or(0.0),
        top_k: req.top_k.unwrap_or(40),
        repeat_penalty: req.repeat_penalty.unwrap_or(1.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        penalty_last_n: req.penalty_last_n.unwrap_or(64),
        seed: req.seed.unwrap_or_else(next_random_seed),
    };
    if let Err(msg) = sample.validate() {
        return api_error(StatusCode::BAD_REQUEST, msg);
    }
    // Acquire queue admission permit SYNCHRONOUSLY before returning 200 OK SSE headers so
    // queue saturation always returns HTTP 503 Service Unavailable (never 200 OK + [DONE]).
    let permit = match state.residency.try_acquire_admission() {
        Ok(p) => p,
        Err(err) => return api_error(StatusCode::SERVICE_UNAVAILABLE, err.to_string()),
    };

    let mut messages: Vec<ChatMessage> = match req
        .messages
        .iter()
        .map(IncomingChatMessage::to_chat_message)
        .collect::<std::result::Result<Vec<_>, _>>()
    {
        Ok(msgs) => msgs,
        Err(msg) => return api_error(StatusCode::BAD_REQUEST, msg),
    };

    if allow_tool_calls && let Some(tools) = &req.tools {
        inject_tools_into_messages(&mut messages, tools);
    }

    let max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(512)
        .clamp(1, state.config.n_ctx.max(1) as usize);
    let compiled_rf = match req.response_format.as_ref() {
        Some(rf) => match layerstream_core::compile_response_format(rf, false) {
            Ok(c) => Some(c),
            Err(e) => return api_error(StatusCode::BAD_REQUEST, format!("{e:#}")),
        },
        None => None,
    };
    let opts = GenerateOptions {
        max_tokens,
        sample,
        stop: req.stop.map(StopSpec::into_vec).unwrap_or_default(),
        enable_ngram_speculation: state.config.enable_ngram_speculation,
        max_draft_tokens: state.config.max_draft_tokens,
        grammar: compiled_rf.and_then(|c| c.grammar),
    };

    let req_id = format!("chatcmpl-ls-{}", unix_now_nanos());
    let created = unix_now();
    let model_id = entry.id.clone();

    if req.stream {
        let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let cancelled = Arc::new(AtomicBool::new(state.shutdown.load(Ordering::SeqCst)));
        let cancel_worker = cancelled.clone();
        let state_bg = state.clone();

        tokio::spawn(async move {
            let mut lease = tokio::select! {
                biased;
                _ = tx.closed() => return,
                lease = state_bg.residency.lock_with_permit(permit) => lease,
            };
            if cancel_worker.load(Ordering::Relaxed)
                || state_bg.shutdown.load(Ordering::Relaxed)
                || tx.is_closed()
            {
                return;
            }
            let _active_guard = state_bg.register_active_cancel(cancel_worker.clone());

            let state_load = state_bg.clone();
            let entry_bg = entry.clone();
            let model_bg = model_id.clone();
            let req_id_bg = req_id.clone();
            let tx_stream = tx.clone();

            let _ = tokio::task::spawn_blocking(move || {
                if cancel_worker.load(Ordering::Relaxed) {
                    return;
                }
                let pair = match lease.get_or_load(&entry_bg.id, || state_load.load_pair(&entry_bg))
                {
                    Ok(p) => p,
                    Err(err) => {
                        tracing::error!("model load error: {err:#}");
                        let err_chunk =
                            json!({"error": {"message": format!("{err:#}"), "type": "server_error"}});
                        let _ = tx_stream.send(Ok(Event::default().data(err_chunk.to_string())));
                        return;
                    }
                };

                let prompt = match pair.target.apply_chat_template(&messages, true) {
                    Ok(p) => p,
                    Err(err) => {
                        tracing::error!("chat template error: {err:#}");
                        let err_chunk =
                            json!({"error": {"message": format!("{err:#}"), "type": "server_error"}});
                        let _ = tx_stream.send(Ok(Event::default().data(err_chunk.to_string())));
                        return;
                    }
                };

                // Initial role delta
                let role_chunk = json!({
                    "id": req_id_bg,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_bg,
                    "choices": [{
                        "index": 0,
                        "delta": { "role": "assistant" },
                        "finish_reason": null,
                    }]
                });
                if tx_stream
                    .send(Ok(Event::default().data(role_chunk.to_string())))
                    .is_err()
                {
                    return;
                }

                let mut tag_parser = StreamTagParser::new();
                let res = pair.target.generate_stream_with_cancel(
                    &prompt,
                    &opts,
                    pair.draft.as_mut(),
                    Some(&cancel_worker),
                    |_tok, piece| {
                        if cancel_worker.load(Ordering::Relaxed) {
                            return false;
                        }
                        if allow_tool_calls {
                            let deltas = tag_parser.push(piece);
                            emit_stream_deltas(deltas, &req_id_bg, created, &model_bg, &tx_stream)
                        } else {
                            let deltas: Vec<_> = tag_parser
                                .push(piece)
                                .into_iter()
                                .filter(|d| !matches!(d, StreamDelta::ToolCall { .. }))
                                .collect();
                            emit_stream_deltas(deltas, &req_id_bg, created, &model_bg, &tx_stream)
                        }
                    },
                );

                let finish = match res {
                    Ok(summary) => {
                        if summary.finish_reason == "cancelled" {
                            false
                        } else {
                            let flushed: Vec<_> = if allow_tool_calls {
                                tag_parser.flush()
                            } else {
                                tag_parser
                                    .flush()
                                    .into_iter()
                                    .filter(|d| !matches!(d, StreamDelta::ToolCall { .. }))
                                    .collect()
                            };
                            let _ = emit_stream_deltas(
                                flushed, &req_id_bg, created, &model_bg, &tx_stream,
                            );
                            let finish_reason = if allow_tool_calls && tag_parser.has_tool_calls() {
                                "tool_calls".to_string()
                            } else {
                                summary.finish_reason
                            };
                            let final_chunk = json!({
                                "id": req_id_bg,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_bg,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason,
                                }],
                                "usage": {
                                    "prompt_tokens": summary.prompt_tokens,
                                    "completion_tokens": summary.completion_tokens,
                                    "total_tokens": summary.prompt_tokens + summary.completion_tokens,
                                }
                            });
                            let _ =
                                tx_stream.send(Ok(Event::default().data(final_chunk.to_string())));
                            true
                        }
                    }
                    Err(err) if err.downcast_ref::<CancelledError>().is_some() => false,
                    Err(err) => {
                        tracing::error!("stream generation failed: {err:#}");
                        let kind = if err.downcast_ref::<PromptTooLongError>().is_some() {
                            "invalid_request_error"
                        } else {
                            "server_error"
                        };
                        let err_chunk = json!({"error": {"message": format!("{err:#}"), "type": kind}});
                        let _ = tx_stream.send(Ok(Event::default().data(err_chunk.to_string())));
                        false
                    }
                };
                if finish {
                    let _ = tx_stream.send(Ok(Event::default().data("[DONE]")));
                }
            })
            .await;
        });

        let guard = Arc::new(CancelOnDrop(cancelled));
        let stream = tokio_stream::StreamExt::map(UnboundedReceiverStream::new(rx), move |ev| {
            let _keep_guard = &guard;
            ev
        });
        return Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    // Non-streaming JSON response (with CancelOnDrop so client TCP disconnect aborts prefill/decode!).
    let cancelled = Arc::new(AtomicBool::new(state.shutdown.load(Ordering::SeqCst)));
    let _cancel_guard = CancelOnDrop(cancelled.clone());
    let mut lease = state.residency.lock_with_permit(permit).await;
    if cancelled.load(Ordering::Relaxed) || state.shutdown.load(Ordering::Relaxed) {
        return api_error(StatusCode::SERVICE_UNAVAILABLE, "request cancelled");
    }
    let _active_guard = state.register_active_cancel(cancelled.clone());

    let state_bg = state.clone();
    let cancel_worker = cancelled.clone();
    let result = tokio::task::spawn_blocking(
        move || -> Result<(String, layerstream_core::GenerateSummary)> {
            if cancel_worker.load(Ordering::Relaxed) {
                return Err(CancelledError.into());
            }
            let pair = lease.get_or_load(&entry.id, || state_bg.load_pair(&entry))?;
            let prompt = pair.target.apply_chat_template(&messages, true)?;
            let mut raw_output = String::new();
            let summary = pair.target.generate_stream_with_cancel(
                &prompt,
                &opts,
                pair.draft.as_mut(),
                Some(&cancel_worker),
                |_tok, piece| {
                    if cancel_worker.load(Ordering::Relaxed) {
                        return false;
                    }
                    raw_output.push_str(piece);
                    true
                },
            )?;
            Ok((raw_output, summary))
        },
    )
    .await;

    match result {
        Ok(Ok((raw_output, summary))) => {
            if summary.finish_reason == "cancelled" {
                return api_error(StatusCode::SERVICE_UNAVAILABLE, "request cancelled");
            }
            let parsed = parse_assistant_output(&raw_output);
            let tool_calls = if allow_tool_calls {
                parsed.tool_calls
            } else {
                Vec::new()
            };
            let finish_reason = if !tool_calls.is_empty() {
                "tool_calls".to_string()
            } else {
                summary.finish_reason
            };
            let mut msg = json!({
                "role": "assistant",
                "content": if parsed.content.is_empty() && !tool_calls.is_empty() {
                    Value::Null
                } else {
                    json!(parsed.content)
                },
            });
            if let Some(rc) = parsed.reasoning_content {
                msg["reasoning_content"] = json!(rc);
            }
            if !tool_calls.is_empty() {
                msg["tool_calls"] = json!(tool_calls);
            }

            Json(json!({
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": msg,
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": summary.prompt_tokens,
                    "completion_tokens": summary.completion_tokens,
                    "total_tokens": summary.prompt_tokens + summary.completion_tokens,
                }
            }))
            .into_response()
        }
        Ok(Err(err)) if err.downcast_ref::<PromptTooLongError>().is_some() => {
            api_error(StatusCode::BAD_REQUEST, format!("{err:#}"))
        }
        Ok(Err(err)) if err.downcast_ref::<CancelledError>().is_some() => {
            api_error(StatusCode::SERVICE_UNAVAILABLE, "request cancelled")
        }
        Ok(Err(err)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("{err:#}")),
        Err(err) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("worker panicked: {err}"),
        ),
    }
}

async fn text_completions(
    State(state): State<AppState>,
    Json(req): Json<TextCompletionRequest>,
) -> Response {
    let entry = match resolve_model(&state, req.model.as_deref()) {
        Ok(e) => e,
        Err(resp) => return *resp,
    };

    let sample = LsSampleParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(0.95),
        min_p: req.min_p.unwrap_or(0.0),
        top_k: req.top_k.unwrap_or(40),
        repeat_penalty: req.repeat_penalty.unwrap_or(1.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        penalty_last_n: req.penalty_last_n.unwrap_or(64),
        seed: req.seed.unwrap_or_else(next_random_seed),
    };
    if let Err(msg) = sample.validate() {
        return api_error(StatusCode::BAD_REQUEST, msg);
    }

    let permit = match state.residency.try_acquire_admission() {
        Ok(p) => p,
        Err(err) => return api_error(StatusCode::SERVICE_UNAVAILABLE, err.to_string()),
    };

    let compiled_rf = match req.response_format.as_ref() {
        Some(rf) => match layerstream_core::compile_response_format(rf, false) {
            Ok(c) => Some(c),
            Err(e) => return api_error(StatusCode::BAD_REQUEST, format!("{e:#}")),
        },
        None => None,
    };
    let opts = GenerateOptions {
        max_tokens: req
            .max_tokens
            .unwrap_or(256)
            .clamp(1, state.config.n_ctx.max(1) as usize),
        sample,
        stop: req.stop.map(StopSpec::into_vec).unwrap_or_default(),
        enable_ngram_speculation: state.config.enable_ngram_speculation,
        max_draft_tokens: state.config.max_draft_tokens,
        grammar: compiled_rf.and_then(|c| c.grammar),
    };

    let req_id = format!("cmpl-ls-{}", unix_now_nanos());
    let created = unix_now();
    let model_id = entry.id.clone();

    if req.stream {
        let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let cancelled = Arc::new(AtomicBool::new(state.shutdown.load(Ordering::SeqCst)));
        let cancel_worker = cancelled.clone();
        let state_bg = state.clone();
        let prompt = req.prompt;

        tokio::spawn(async move {
            let mut lease = tokio::select! {
                biased;
                _ = tx.closed() => return,
                lease = state_bg.residency.lock_with_permit(permit) => lease,
            };
            if cancel_worker.load(Ordering::Relaxed)
                || state_bg.shutdown.load(Ordering::Relaxed)
                || tx.is_closed()
            {
                return;
            }
            let _active_guard = state_bg.register_active_cancel(cancel_worker.clone());

            let state_load = state_bg.clone();
            let entry_bg = entry.clone();
            let model_bg = model_id.clone();
            let req_id_bg = req_id.clone();
            let tx_stream = tx.clone();

            let _ = tokio::task::spawn_blocking(move || {
                if cancel_worker.load(Ordering::Relaxed) {
                    return;
                }
                let pair = match lease.get_or_load(&entry_bg.id, || state_load.load_pair(&entry_bg))
                {
                    Ok(p) => p,
                    Err(err) => {
                        tracing::error!("model load error: {err:#}");
                        let err_chunk =
                            json!({"error": {"message": format!("{err:#}"), "type": "server_error"}});
                        let _ = tx_stream.send(Ok(Event::default().data(err_chunk.to_string())));
                        return;
                    }
                };
                match pair.target.generate_stream_with_cancel(
                    &prompt,
                    &opts,
                    pair.draft.as_mut(),
                    Some(&cancel_worker),
                    |_tok, piece| {
                        if cancel_worker.load(Ordering::Relaxed) {
                            return false;
                        }
                        let chunk = json!({
                            "id": req_id_bg,
                            "object": "text_completion",
                            "created": created,
                            "model": model_bg,
                            "choices": [{
                                "text": piece,
                                "index": 0,
                                "finish_reason": null,
                            }]
                        });
                        tx_stream
                            .send(Ok(Event::default().data(chunk.to_string())))
                            .is_ok()
                    },
                ) {
                    Ok(summary) if summary.finish_reason != "cancelled" => {
                        let final_chunk = json!({
                            "id": req_id_bg,
                            "object": "text_completion",
                            "created": created,
                            "model": model_bg,
                            "choices": [{
                                "text": "",
                                "index": 0,
                                "finish_reason": summary.finish_reason,
                            }]
                        });
                        let _ = tx_stream.send(Ok(Event::default().data(final_chunk.to_string())));
                        let _ = tx_stream.send(Ok(Event::default().data("[DONE]")));
                    }
                    Ok(_) => {}
                    Err(err) if err.downcast_ref::<CancelledError>().is_some() => {}
                    Err(err) => {
                        tracing::error!("text stream generation failed: {err:#}");
                        let kind = if err.downcast_ref::<PromptTooLongError>().is_some() {
                            "invalid_request_error"
                        } else {
                            "server_error"
                        };
                        let err_chunk = json!({"error": {"message": format!("{err:#}"), "type": kind}});
                        let _ = tx_stream.send(Ok(Event::default().data(err_chunk.to_string())));
                    }
                }
            })
            .await;
        });

        let guard = Arc::new(CancelOnDrop(cancelled));
        let stream = tokio_stream::StreamExt::map(UnboundedReceiverStream::new(rx), move |ev| {
            let _keep_guard = &guard;
            ev
        });
        return Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    let cancelled = Arc::new(AtomicBool::new(state.shutdown.load(Ordering::SeqCst)));
    let _cancel_guard = CancelOnDrop(cancelled.clone());
    let mut lease = state.residency.lock_with_permit(permit).await;
    if cancelled.load(Ordering::Relaxed) || state.shutdown.load(Ordering::Relaxed) {
        return api_error(StatusCode::SERVICE_UNAVAILABLE, "request cancelled");
    }
    let _active_guard = state.register_active_cancel(cancelled.clone());

    let state_bg = state.clone();
    let cancel_worker = cancelled.clone();
    let prompt = req.prompt;

    let result = tokio::task::spawn_blocking(
        move || -> Result<(String, layerstream_core::GenerateSummary)> {
            if cancel_worker.load(Ordering::Relaxed) {
                return Err(CancelledError.into());
            }
            let pair = lease.get_or_load(&entry.id, || state_bg.load_pair(&entry))?;
            let mut text = String::new();
            let summary = pair.target.generate_stream_with_cancel(
                &prompt,
                &opts,
                pair.draft.as_mut(),
                Some(&cancel_worker),
                |_tok, piece| {
                    if cancel_worker.load(Ordering::Relaxed) {
                        return false;
                    }
                    text.push_str(piece);
                    true
                },
            )?;
            Ok((text, summary))
        },
    )
    .await;

    match result {
        Ok(Ok((text, summary))) => {
            if summary.finish_reason == "cancelled" {
                return api_error(StatusCode::SERVICE_UNAVAILABLE, "request cancelled");
            }
            Json(json!({
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "text": text,
                    "index": 0,
                    "finish_reason": summary.finish_reason,
                }],
                "usage": {
                    "prompt_tokens": summary.prompt_tokens,
                    "completion_tokens": summary.completion_tokens,
                    "total_tokens": summary.prompt_tokens + summary.completion_tokens,
                }
            }))
            .into_response()
        }
        Ok(Err(err)) if err.downcast_ref::<PromptTooLongError>().is_some() => {
            api_error(StatusCode::BAD_REQUEST, format!("{err:#}"))
        }
        Ok(Err(err)) if err.downcast_ref::<CancelledError>().is_some() => {
            api_error(StatusCode::SERVICE_UNAVAILABLE, "request cancelled")
        }
        Ok(Err(err)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("{err:#}")),
        Err(err) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("worker panicked: {err}"),
        ),
    }
}

fn resolve_model(
    state: &AppState,
    requested: Option<&str>,
) -> std::result::Result<ModelEntry, Box<Response>> {
    if let Some(err) = state.loading_error() {
        return Err(Box::new(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("background model download failed: {err}"),
        )));
    }
    if let Some(status) = state.loading_status() {
        return Err(Box::new(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("model is still downloading: {status}"),
        )));
    }
    let Ok(registry) = state.registry.read() else {
        return Err(Box::new(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "model registry lock poisoned",
        )));
    };
    if let Some(id) = requested {
        if let Some(entry) = registry.get(id) {
            return Ok(entry.clone());
        }
        return Err(Box::new(api_error(
            StatusCode::NOT_FOUND,
            format!("model '{id}' not found in layerstream registry"),
        )));
    }
    registry.first().cloned().ok_or_else(|| {
        Box::new(api_error(
            StatusCode::NOT_FOUND,
            "no models configured in layerstream",
        ))
    })
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unix_now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_accepts_text_parts_and_rejects_multimodal_parts() {
        let ok_msg = IncomingChatMessage {
            role: "user".to_string(),
            content: json!([
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"}
            ]),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let converted = ok_msg.to_chat_message().unwrap();
        assert_eq!(converted.content, "hello\nworld");

        let bad_msg = IncomingChatMessage {
            role: "user".to_string(),
            content: json!([
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
            ]),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let err = bad_msg.to_chat_message().unwrap_err();
        assert!(err.contains("image_url"), "unexpected error: {err}");
    }
}
