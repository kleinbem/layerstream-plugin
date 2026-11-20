//! OpenAI-compatible HTTP routes (`/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/health`)
//! with SSE streaming, tool-call & thinking extraction, client-disconnect cancellation, and
//! lazy 0-VRAM idle residency.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use layerstream_core::{
    ChatMessage, DraftModelProposer, EngineConfig, GenerateOptions, KvCacheType, LsSampleParams,
    Residency, StreamEngine,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::parse::{format_tools_system_prompt, parse_assistant_output};
use crate::registry::{ModelEntry, ModelRegistry};

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub registry: ModelRegistry,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub resident_layers: Option<u32>,
    pub vram_budget_bytes: u64,
    pub kv_cache_type: KvCacheType,
    pub use_gpu: bool,
    pub prefer_direct_io: bool,
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
    pub residency: Arc<Residency<LoadedPair>>,
}

impl AppState {
    pub fn new(config: ServerConfig) -> Self {
        let residency = Arc::new(Residency::new(
            config.idle_timeout,
            config.max_queued_requests.max(1),
        ));
        Self {
            config: Arc::new(config),
            residency,
        }
    }

    pub fn spawn_reaper(&self) {
        if self.config.idle_timeout.is_zero() {
            return;
        }
        let residency = self.residency.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            loop {
                interval.tick().await;
                if residency.evict_idle(Instant::now()) {
                    tracing::info!("evicted idle layerstream model (VRAM returned to 0 MB)");
                }
            }
        });
    }

    fn load_pair(&self, entry: &ModelEntry) -> Result<LoadedPair> {
        let mut cfg = EngineConfig::streaming(&entry.path);
        cfg.n_ctx = self.config.n_ctx;
        cfg.n_batch = self.config.n_batch;
        cfg.n_threads = self.config.n_threads;
        cfg.resident_layers = self.config.resident_layers;
        cfg.vram_budget_bytes = self.config.vram_budget_bytes;
        cfg.kv_cache_type = self.config.kv_cache_type;
        cfg.use_gpu = self.config.use_gpu;
        cfg.prefer_direct_io = self.config.prefer_direct_io;

        let target = StreamEngine::open(&cfg)?;
        let draft = match &entry.draft_path {
            Some(draft_path) => {
                let mut dcfg = EngineConfig::reference(draft_path);
                dcfg.n_ctx = self.config.n_ctx;
                dcfg.n_batch = self.config.n_batch;
                dcfg.n_threads = self.config.n_threads;
                dcfg.use_gpu = self.config.use_gpu;
                Some(DraftModelProposer::open(
                    dcfg,
                    self.config.max_draft_tokens,
                )?)
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
            slot_mb = stats.slot_bytes / (1024 * 1024),
            uma_zero_copy = stats.is_uma_zero_copy,
            gpu = stats.using_gpu,
            can_speculate = stats.can_speculate,
            "loaded layerstream model"
        );
        Ok(LoadedPair { target, draft })
    }
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(text_completions))
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "models": state.config.registry.len(),
        "loaded": state.residency.is_loaded(),
    }))
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let now = unix_now();
    let data: Vec<Value> = state
        .config
        .registry
        .list()
        .into_iter()
        .map(|entry| {
            json!({
                "id": entry.id,
                "object": "model",
                "created": now,
                "owned_by": "layerstream",
                "meta": {
                    "architecture": entry.meta.architecture,
                    "n_layers": entry.meta.n_layers,
                    "context_length": entry.meta.context_length,
                    "file_bytes": entry.meta.total_file_bytes,
                }
            })
        })
        .collect();

    Json(json!({
        "object": "list",
        "data": data,
    }))
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<IncomingMessage>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
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
    top_k: Option<i32>,
    #[serde(default)]
    seed: Option<u32>,
    #[serde(default)]
    stop: Option<StopSpec>,
}

#[derive(Debug, Deserialize)]
struct IncomingMessage {
    role: String,
    #[serde(default)]
    content: Value,
}

impl IncomingMessage {
    fn content_text(&self) -> String {
        match &self.content {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
            Value::Null => String::new(),
            other => other.to_string(),
        }
    }
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
    top_k: Option<i32>,
    #[serde(default)]
    seed: Option<u32>,
    #[serde(default)]
    stop: Option<StopSpec>,
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

fn api_error(status: StatusCode, msg: impl Into<String>) -> Response {
    (
        status,
        Json(ErrorEnvelope {
            error: ErrorBody {
                message: msg.into(),
                kind: "invalid_request_error".to_string(),
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

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let entry = match resolve_model(&state.config.registry, req.model.as_deref()) {
        Ok(e) => e.clone(),
        Err(resp) => return *resp,
    };

    let mut messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content_text(),
        })
        .collect();

    if let Some(tools) = &req.tools
        && !tools.is_empty()
    {
        let tool_prompt = format_tools_system_prompt(tools);
        if let Some(first) = messages.first_mut()
            && first.role == "system"
        {
            first.content.push_str("\n\n");
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
    }

    let max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(512)
        .max(1);
    let opts = GenerateOptions {
        max_tokens,
        sample: LsSampleParams {
            temperature: req.temperature.unwrap_or(0.0),
            top_p: req.top_p.unwrap_or(0.95),
            top_k: req.top_k.unwrap_or(40),
            repeat_penalty: 1.0,
            seed: req.seed.unwrap_or(42),
        },
        stop: req.stop.map(StopSpec::into_vec).unwrap_or_default(),
        enable_ngram_speculation: state.config.enable_ngram_speculation,
        max_draft_tokens: state.config.max_draft_tokens,
    };

    let req_id = format!("chatcmpl-ls-{}", unix_now_nanos());
    let created = unix_now();
    let model_id = entry.id.clone();

    if req.stream {
        let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancel_worker = cancelled.clone();
        let state_bg = state.clone();

        tokio::spawn(async move {
            let Ok(mut lease) = state_bg.residency.acquire().await else {
                let _ = tx.send(Ok(Event::default().data("[DONE]")));
                return;
            };

            let state_load = state_bg.clone();
            let entry_bg = entry.clone();
            let model_bg = model_id.clone();
            let req_id_bg = req_id.clone();
            let tx_stream = tx.clone();

            let _ = tokio::task::spawn_blocking(move || {
                let pair = match lease.get_or_load(&entry_bg.id, || state_load.load_pair(&entry_bg))
                {
                    Ok(p) => p,
                    Err(err) => {
                        tracing::error!("model load error: {err:#}");
                        return;
                    }
                };

                let prompt = match pair.target.apply_chat_template(&messages, true) {
                    Ok(p) => p,
                    Err(err) => {
                        tracing::error!("chat template error: {err:#}");
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
                let _ = tx_stream.send(Ok(Event::default().data(role_chunk.to_string())));

                let mut in_think = false;
                let res = pair.target.generate_stream(
                    &prompt,
                    &opts,
                    pair.draft.as_mut(),
                    |_tok, piece| {
                        if cancel_worker.load(Ordering::Relaxed) {
                            return false;
                        }
                        let mut delta = serde_json::Map::new();
                        if piece.contains("<think>") {
                            in_think = true;
                        }
                        let cleaned = piece.replace("<think>", "").replace("</think>", "");
                        if !cleaned.is_empty() {
                            if in_think {
                                delta.insert("reasoning_content".to_string(), json!(cleaned));
                            } else {
                                delta.insert("content".to_string(), json!(cleaned));
                            }
                        }
                        if piece.contains("</think>") {
                            in_think = false;
                        }
                        if !delta.is_empty() {
                            let chunk = json!({
                                "id": req_id_bg,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_bg,
                                "choices": [{
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": null,
                                }]
                            });
                            if tx_stream
                                .send(Ok(Event::default().data(chunk.to_string())))
                                .is_err()
                            {
                                return false;
                            }
                        }
                        true
                    },
                );

                let finish = match res {
                    Ok(summary) => {
                        let final_chunk = json!({
                            "id": req_id_bg,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_bg,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": summary.finish_reason,
                            }],
                            "usage": {
                                "prompt_tokens": summary.prompt_tokens,
                                "completion_tokens": summary.completion_tokens,
                                "total_tokens": summary.prompt_tokens + summary.completion_tokens,
                            }
                        });
                        let _ = tx_stream.send(Ok(Event::default().data(final_chunk.to_string())));
                        true
                    }
                    Err(err) => {
                        tracing::error!("stream generation failed: {err:#}");
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

    // Non-streaming JSON response
    let mut lease = match state.residency.acquire().await {
        Ok(l) => l,
        Err(err) => return api_error(StatusCode::SERVICE_UNAVAILABLE, err.to_string()),
    };

    let state_bg = state.clone();
    let result = tokio::task::spawn_blocking(
        move || -> Result<(String, layerstream_core::GenerateSummary)> {
            let pair = lease.get_or_load(&entry.id, || state_bg.load_pair(&entry))?;
            let prompt = pair.target.apply_chat_template(&messages, true)?;
            let mut raw_output = String::new();
            let summary = pair.target.generate_stream(
                &prompt,
                &opts,
                pair.draft.as_mut(),
                |_tok, piece| {
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
            let parsed = parse_assistant_output(&raw_output);
            let finish_reason = if !parsed.tool_calls.is_empty() {
                "tool_calls".to_string()
            } else {
                summary.finish_reason
            };
            let mut msg = json!({
                "role": "assistant",
                "content": if parsed.content.is_empty() && !parsed.tool_calls.is_empty() {
                    Value::Null
                } else {
                    json!(parsed.content)
                },
            });
            if let Some(rc) = parsed.reasoning_content {
                msg["reasoning_content"] = json!(rc);
            }
            if !parsed.tool_calls.is_empty() {
                msg["tool_calls"] = json!(parsed.tool_calls);
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
    let entry = match resolve_model(&state.config.registry, req.model.as_deref()) {
        Ok(e) => e.clone(),
        Err(resp) => return *resp,
    };

    let opts = GenerateOptions {
        max_tokens: req.max_tokens.unwrap_or(256).max(1),
        sample: LsSampleParams {
            temperature: req.temperature.unwrap_or(0.0),
            top_p: req.top_p.unwrap_or(0.95),
            top_k: req.top_k.unwrap_or(40),
            repeat_penalty: 1.0,
            seed: req.seed.unwrap_or(42),
        },
        stop: req.stop.map(StopSpec::into_vec).unwrap_or_default(),
        enable_ngram_speculation: state.config.enable_ngram_speculation,
        max_draft_tokens: state.config.max_draft_tokens,
    };

    let req_id = format!("cmpl-ls-{}", unix_now_nanos());
    let created = unix_now();
    let model_id = entry.id.clone();

    if req.stream {
        let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancel_worker = cancelled.clone();
        let state_bg = state.clone();
        let prompt = req.prompt;

        tokio::spawn(async move {
            let Ok(mut lease) = state_bg.residency.acquire().await else {
                let _ = tx.send(Ok(Event::default().data("[DONE]")));
                return;
            };
            let state_load = state_bg.clone();
            let entry_bg = entry.clone();
            let model_bg = model_id.clone();
            let req_id_bg = req_id.clone();
            let tx_stream = tx.clone();

            let _ = tokio::task::spawn_blocking(move || {
                let Ok(pair) = lease.get_or_load(&entry_bg.id, || state_load.load_pair(&entry_bg))
                else {
                    return;
                };
                if let Ok(summary) = pair.target.generate_stream(
                    &prompt,
                    &opts,
                    pair.draft.as_mut(),
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

    let mut lease = match state.residency.acquire().await {
        Ok(l) => l,
        Err(err) => return api_error(StatusCode::SERVICE_UNAVAILABLE, err.to_string()),
    };
    let state_bg = state.clone();
    let prompt = req.prompt;

    let result = tokio::task::spawn_blocking(
        move || -> Result<(String, layerstream_core::GenerateSummary)> {
            let pair = lease.get_or_load(&entry.id, || state_bg.load_pair(&entry))?;
            let mut text = String::new();
            let summary = pair.target.generate_stream(
                &prompt,
                &opts,
                pair.draft.as_mut(),
                |_tok, piece| {
                    text.push_str(piece);
                    true
                },
            )?;
            Ok((text, summary))
        },
    )
    .await;

    match result {
        Ok(Ok((text, summary))) => Json(json!({
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
        .into_response(),
        Ok(Err(err)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("{err:#}")),
        Err(err) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("worker panicked: {err}"),
        ),
    }
}

fn resolve_model<'a>(
    registry: &'a ModelRegistry,
    requested: Option<&str>,
) -> std::result::Result<&'a ModelEntry, Box<Response>> {
    if let Some(id) = requested {
        if let Some(entry) = registry.get(id) {
            return Ok(entry);
        }
        return Err(Box::new(api_error(
            StatusCode::NOT_FOUND,
            format!("model '{id}' not found in layerstream registry"),
        )));
    }
    registry.first().ok_or_else(|| {
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
