use std::path::PathBuf;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use layerstream_core::KvCacheType;
use layerstream_server::{AppState, ModelRegistry, ServerConfig, build_router};
use serde_json::{Value, json};
use tower::ServiceExt;

#[tokio::test]
async fn health_and_empty_models_work_with_zero_vram() {
    let cfg = ServerConfig {
        registry: ModelRegistry::default(),
        n_ctx: 512,
        n_batch: 128,
        n_threads: 2,
        resident_layers: Some(0),
        vram_budget_bytes: 0,
        vram_budget_auto: false,
        kv_cache_type: KvCacheType::F16,
        use_gpu: false,
        prefer_direct_io: true,
        cpu_embeddings: false,
        cpu_head: false,
        stream_head: false,
        moe_active_streaming: true,
        idle_timeout: Duration::ZERO,
        enable_ngram_speculation: true,
        max_draft_tokens: 4,
        max_queued_requests: 4,
    };
    let state = AppState::new(cfg);
    assert!(!state.residency.is_loaded());

    let app = build_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 64)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["loaded"], false);
}

#[tokio::test]
async fn unknown_model_returns_404_json_error() {
    let cfg = ServerConfig {
        registry: ModelRegistry::default(),
        n_ctx: 512,
        n_batch: 128,
        n_threads: 2,
        resident_layers: Some(0),
        vram_budget_bytes: 0,
        vram_budget_auto: false,
        kv_cache_type: KvCacheType::F16,
        use_gpu: false,
        prefer_direct_io: true,
        cpu_embeddings: false,
        cpu_head: false,
        stream_head: false,
        moe_active_streaming: true,
        idle_timeout: Duration::ZERO,
        enable_ngram_speculation: true,
        max_draft_tokens: 4,
        max_queued_requests: 4,
    };
    let app = build_router(AppState::new(cfg));
    let body = json!({
        "model": "does-not-exist",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("not found")
    );
}

#[tokio::test]
async fn chat_completions_and_immediate_eviction_on_real_gguf() {
    let Ok(model_path) = std::env::var("LAYERSTREAM_TEST_GGUF") else {
        eprintln!("skipping real HTTP chat test (set LAYERSTREAM_TEST_GGUF)");
        return;
    };
    let registry =
        ModelRegistry::from_sources(&[format!("test-model={model_path}")], None, None).unwrap();
    let cfg = ServerConfig {
        registry,
        n_ctx: 512,
        n_batch: 128,
        n_threads: 4,
        resident_layers: Some(0),
        vram_budget_bytes: 0,
        vram_budget_auto: false,
        kv_cache_type: KvCacheType::F16,
        use_gpu: false,
        prefer_direct_io: true,
        cpu_embeddings: false,
        cpu_head: false,
        stream_head: false,
        moe_active_streaming: true,
        idle_timeout: Duration::ZERO,
        enable_ngram_speculation: true,
        max_draft_tokens: 4,
        max_queued_requests: 4,
    };
    let state = AppState::new(cfg);
    // Before request: 0 MB VRAM allocated.
    assert!(!state.residency.is_loaded());

    let app = build_router(state.clone());

    // 1. /v1/models lists model without loading weights.
    let models_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(models_resp.status(), StatusCode::OK);
    assert!(!state.residency.is_loaded());
    let mbytes = axum::body::to_bytes(models_resp.into_body(), 1024 * 64)
        .await
        .unwrap();
    let mv: Value = serde_json::from_slice(&mbytes).unwrap();
    assert_eq!(mv["data"][0]["id"], "test-model");

    // 2. Non-streaming chat completion.
    let body = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Reply with one word: Paris."}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": false
    });
    let chat_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(chat_resp.status(), StatusCode::OK);
    let cbytes = axum::body::to_bytes(chat_resp.into_body(), 1024 * 256)
        .await
        .unwrap();
    let cv: Value = serde_json::from_slice(&cbytes).unwrap();
    let content = cv["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(!content.is_empty(), "expected non-empty assistant response");
    // Because idle_timeout == 0, GPU lease was immediately evicted back to 0 MB!
    assert!(!state.residency.is_loaded());

    // 3. SSE streaming chat completion emits OpenAI chunks and terminates with [DONE].
    let stream_body = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Reply with one word: Paris."}],
        "max_tokens": 6,
        "temperature": 0.0,
        "stream": true
    });
    let sse_resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(stream_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(sse_resp.status(), StatusCode::OK);
    let sse_bytes = axum::body::to_bytes(sse_resp.into_body(), 1024 * 256)
        .await
        .unwrap();
    let sse_text = String::from_utf8_lossy(&sse_bytes);
    assert!(sse_text.contains("chat.completion.chunk"));
    assert!(sse_text.contains("[DONE]"));

    let _ = PathBuf::from(model_path);
}
