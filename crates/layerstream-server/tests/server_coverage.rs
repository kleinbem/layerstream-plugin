#![allow(clippy::unwrap_used)]

//! Comprehensive HTTP server, ModelRegistry, AppState, and `serve_with_state_until`
//! coverage tests using a tiny deterministic synthetic GGUF (0 external files).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use layerstream_core::KvCacheType;
use layerstream_server::{
    AppState, ModelRegistry, ServerConfig, api_error, build_router, serve_with_state_until,
};
use serde_json::{Value, json};
use tower::ServiceExt;

struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "layerstream-server-cov-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_tiny_llama_gguf(path: &Path, n_vocab: u64, alt_vocab: bool) {
    const N_EMBD: u64 = 64;
    const N_HEAD: u32 = 2;
    const N_FF: u64 = 128;
    const N_LAYERS: u32 = 2;
    const ALIGN: usize = 32;

    let write_str = |buf: &mut Vec<u8>, s: &str| {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    };
    let kv_str = |buf: &mut Vec<u8>, k: &str, v: &str| {
        write_str(buf, k);
        buf.extend_from_slice(&8u32.to_le_bytes());
        write_str(buf, v);
    };
    let kv_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
        write_str(buf, k);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    };
    let kv_f32 = |buf: &mut Vec<u8>, k: &str, v: f32| {
        write_str(buf, k);
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    };
    let kv_bool = |buf: &mut Vec<u8>, k: &str, v: bool| {
        write_str(buf, k);
        buf.extend_from_slice(&7u32.to_le_bytes());
        buf.push(u8::from(v));
    };

    let mut seed: u64 = 0xDEAD_BEEF_1337_0001;
    let mut next_f32 = |scale: f32| -> f32 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let u = ((seed >> 40) as f32) / ((1u32 << 24) as f32);
        (u * 2.0 - 1.0) * scale
    };

    struct TSpec {
        name: String,
        dims: Vec<u64>,
        bytes: Vec<u8>,
        rel_offset: u64,
    }
    let mut tensors: Vec<TSpec> = Vec::new();
    let mut push_t = |name: String, dims: Vec<u64>, is_norm: bool| {
        let nelems: usize = dims.iter().map(|&d| d as usize).product();
        let mut bytes = Vec::with_capacity(nelems * 4);
        for _ in 0..nelems {
            let v = if is_norm {
                1.0 + next_f32(0.05)
            } else {
                next_f32(0.15)
            };
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        tensors.push(TSpec {
            name,
            dims,
            bytes,
            rel_offset: 0,
        });
    };

    push_t("token_embd.weight".into(), vec![N_EMBD, n_vocab], false);
    push_t("output_norm.weight".into(), vec![N_EMBD], true);
    push_t("output.weight".into(), vec![N_EMBD, n_vocab], false);
    for il in 0..N_LAYERS {
        push_t(format!("blk.{il}.attn_norm.weight"), vec![N_EMBD], true);
        push_t(
            format!("blk.{il}.attn_q.weight"),
            vec![N_EMBD, N_EMBD],
            false,
        );
        push_t(
            format!("blk.{il}.attn_k.weight"),
            vec![N_EMBD, N_EMBD],
            false,
        );
        push_t(
            format!("blk.{il}.attn_v.weight"),
            vec![N_EMBD, N_EMBD],
            false,
        );
        push_t(
            format!("blk.{il}.attn_output.weight"),
            vec![N_EMBD, N_EMBD],
            false,
        );
        push_t(format!("blk.{il}.ffn_norm.weight"), vec![N_EMBD], true);
        push_t(
            format!("blk.{il}.ffn_gate.weight"),
            vec![N_EMBD, N_FF],
            false,
        );
        push_t(format!("blk.{il}.ffn_up.weight"), vec![N_EMBD, N_FF], false);
        push_t(
            format!("blk.{il}.ffn_down.weight"),
            vec![N_FF, N_EMBD],
            false,
        );
    }

    let mut cur_off = 0usize;
    for t in &mut tensors {
        cur_off = (cur_off + ALIGN - 1) & !(ALIGN - 1);
        t.rel_offset = cur_off as u64;
        cur_off += t.bytes.len();
    }

    let mut kv = Vec::new();
    let mut kv_n = 0u64;
    kv_str(&mut kv, "general.architecture", "llama");
    kv_n += 1;
    kv_str(&mut kv, "general.name", "synthetic-server-test");
    kv_n += 1;
    kv_u32(&mut kv, "llama.block_count", N_LAYERS);
    kv_n += 1;
    kv_u32(&mut kv, "llama.context_length", 128);
    kv_n += 1;
    kv_u32(&mut kv, "llama.embedding_length", N_EMBD as u32);
    kv_n += 1;
    kv_u32(&mut kv, "llama.feed_forward_length", N_FF as u32);
    kv_n += 1;
    kv_u32(&mut kv, "llama.attention.head_count", N_HEAD);
    kv_n += 1;
    kv_u32(&mut kv, "llama.attention.head_count_kv", N_HEAD);
    kv_n += 1;
    kv_f32(&mut kv, "llama.attention.layer_norm_rms_epsilon", 1e-5);
    kv_n += 1;
    kv_f32(&mut kv, "llama.rope.freq_base", 10_000.0);
    kv_n += 1;
    kv_str(&mut kv, "tokenizer.ggml.model", "gpt2");
    kv_n += 1;
    kv_str(&mut kv, "tokenizer.ggml.pre", "default");
    kv_n += 1;
    write_str(&mut kv, "tokenizer.ggml.merges");
    kv.extend_from_slice(&9u32.to_le_bytes());
    kv.extend_from_slice(&8u32.to_le_bytes());
    kv.extend_from_slice(&0u64.to_le_bytes());
    kv_n += 1;
    write_str(&mut kv, "tokenizer.ggml.tokens");
    kv.extend_from_slice(&9u32.to_le_bytes());
    kv.extend_from_slice(&8u32.to_le_bytes());
    kv.extend_from_slice(&n_vocab.to_le_bytes());
    for i in 0..n_vocab {
        let tok = match i {
            0 => "<bos>".to_string(),
            1 => "<eos>".to_string(),
            2 => " ".to_string(),
            _ => {
                let ch = (b'a' + ((i - 3) % 26) as u8) as char;
                if alt_vocab {
                    format!("ALT_{ch}_{i}")
                } else if i < 29 {
                    ch.to_string()
                } else {
                    format!("{ch}{i}")
                }
            }
        };
        write_str(&mut kv, &tok);
    }
    kv_n += 1;
    write_str(&mut kv, "tokenizer.ggml.token_type");
    kv.extend_from_slice(&9u32.to_le_bytes());
    kv.extend_from_slice(&5u32.to_le_bytes());
    kv.extend_from_slice(&n_vocab.to_le_bytes());
    for i in 0..n_vocab {
        let tt: i32 = if i < 2 { 3 } else { 1 };
        kv.extend_from_slice(&tt.to_le_bytes());
    }
    kv_n += 1;
    kv_u32(&mut kv, "tokenizer.ggml.bos_token_id", 0);
    kv_n += 1;
    kv_u32(&mut kv, "tokenizer.ggml.eos_token_id", 1);
    kv_n += 1;
    kv_bool(&mut kv, "tokenizer.ggml.add_bos_token", true);
    kv_n += 1;

    let mut hdr = Vec::new();
    hdr.extend_from_slice(b"GGUF");
    hdr.extend_from_slice(&3u32.to_le_bytes());
    hdr.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    hdr.extend_from_slice(&kv_n.to_le_bytes());
    hdr.extend_from_slice(&kv);
    for t in &tensors {
        write_str(&mut hdr, &t.name);
        hdr.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for &d in &t.dims {
            hdr.extend_from_slice(&d.to_le_bytes());
        }
        hdr.extend_from_slice(&0u32.to_le_bytes());
        hdr.extend_from_slice(&t.rel_offset.to_le_bytes());
    }
    let aligned = (hdr.len() + ALIGN - 1) & !(ALIGN - 1);
    hdr.resize(aligned, 0);

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&hdr).unwrap();
    let mut written = 0usize;
    for t in &tensors {
        let pad = (t.rel_offset as usize).saturating_sub(written);
        if pad > 0 {
            f.write_all(&vec![0u8; pad]).unwrap();
            written += pad;
        }
        f.write_all(&t.bytes).unwrap();
        written += t.bytes.len();
    }
    f.flush().unwrap();
}

fn make_config(registry: ModelRegistry, idle_timeout: Duration) -> ServerConfig {
    ServerConfig {
        registry,
        n_ctx: 128,
        n_batch: 32,
        n_threads: 2,
        resident_layers: Some(0),
        vram_budget_bytes: 0,
        vram_budget_auto: false,
        kv_cache_type: KvCacheType::F16,
        use_gpu: false,
        prefer_direct_io: false,
        cpu_embeddings: false,
        cpu_head: false,
        stream_head: false,
        moe_active_streaming: true,
        idle_timeout,
        enable_ngram_speculation: true,
        max_draft_tokens: 3,
        max_queued_requests: 4,
    }
}

#[test]
fn model_registry_from_sources_shards_aliases_and_draft_validation() {
    let dir = TempDirGuard::new("registry");
    let shard1 = dir.path().join("demo-00001-of-00002.gguf");
    let shard2 = dir.path().join("demo-00002-of-00002.gguf");
    let draft = dir.path().join("draft.gguf");
    let incompat = dir.path().join("incompat.gguf");
    let corrupt = dir.path().join("corrupt.gguf");

    write_tiny_llama_gguf(&shard1, 64, false);
    write_tiny_llama_gguf(&shard2, 64, false);
    write_tiny_llama_gguf(&draft, 64, false);
    write_tiny_llama_gguf(&incompat, 64, true);
    std::fs::write(&corrupt, b"NOT_GGUF").unwrap();

    // Missing models_dir fails cleanly.
    assert!(ModelRegistry::from_sources(&[], Some(&dir.path().join("missing")), None).is_err());

    // Incompatible draft fails closed at startup.
    assert!(
        ModelRegistry::from_sources(
            &[format!("m={}", shard1.display())],
            None,
            Some(incompat.clone()),
        )
        .is_err()
    );

    // Explicit alias + bare path + directory discovery (skips shard2, corrupt, duplicate ID).
    let reg = ModelRegistry::from_sources(
        &[
            format!("demo={}", shard1.display()),
            draft.to_string_lossy().into_owned(),
        ],
        Some(dir.path()),
        Some(draft.clone()),
    )
    .unwrap();
    assert!(!reg.is_empty());
    assert!(reg.get("demo").is_some());
    assert!(reg.get("draft").is_some());
    assert!(reg.first().is_some());
    // shard2 must NOT be registered as a separate model.
    assert!(reg.get("demo-00002-of-00002").is_none());
}

#[tokio::test]
async fn server_chat_text_streaming_unload_errors_and_graceful_serve() {
    let dir = TempDirGuard::new("routes");
    let target = dir.path().join("tiny.gguf");
    let draft = dir.path().join("draft.gguf");
    write_tiny_llama_gguf(&target, 64, false);
    write_tiny_llama_gguf(&draft, 64, false);

    let reg =
        ModelRegistry::from_sources(&[format!("tiny={}", target.display())], None, Some(draft))
            .unwrap();
    // Non-zero idle_timeout so the model stays loaded for `/v1/unload` verification.
    let state = AppState::new(make_config(reg.clone(), Duration::from_secs(60)));

    // Exercise AppState loading status / error transitions on /health.
    state.set_loading_status(Some("downloading tiny.gguf".to_string()));
    assert_eq!(
        state.loading_status().as_deref(),
        Some("downloading tiny.gguf")
    );
    let app = build_router(state.clone());

    let loading_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(loading_resp.status(), StatusCode::SERVICE_UNAVAILABLE);

    state.set_loading_failed("network timeout".to_string());
    assert!(state.loading_status().is_none());
    assert_eq!(state.loading_error().as_deref(), Some("network timeout"));
    let err_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(err_resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

    // Clear error and swap registry via snapshot/set_registry.
    state.set_loading_error(None);
    let snap = state.registry_snapshot();
    state.set_registry(snap);

    // /v1/models lists registered model without loading.
    let m_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(m_resp.status(), StatusCode::OK);
    let m_val: Value = serde_json::from_slice(
        &axum::body::to_bytes(m_resp.into_body(), 64 * 1024)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(m_val["data"][0]["id"], "tiny");

    // 1. Non-streaming /v1/chat/completions (with tools + stop + seed + draft speculator).
    let chat_body = json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": "abc"}],
        "tools": [{"type": "function", "function": {"name": "ping", "description": "p", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
        "max_tokens": 4,
        "temperature": 0.0,
        "stop": "ZZZ",
        "stream": false
    });
    let chat_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(chat_resp.status(), StatusCode::OK);
    let cv: Value = serde_json::from_slice(
        &axum::body::to_bytes(chat_resp.into_body(), 128 * 1024)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(cv["object"], "chat.completion");
    assert!(cv["usage"]["completion_tokens"].as_u64().unwrap() > 0);
    assert!(state.residency.is_loaded());

    // 2. Streaming /v1/chat/completions (SSE + tool_choice: "none" + Many stop).
    let chat_sse_body = json!({
        "messages": [{"role": "user", "content": "abc"}],
        "tool_choice": "none",
        "max_completion_tokens": 4,
        "stop": ["ZZZ", "YYY"],
        "stream": true
    });
    let sse_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_sse_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(sse_resp.status(), StatusCode::OK);
    let sse_text = String::from_utf8(
        axum::body::to_bytes(sse_resp.into_body(), 128 * 1024)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(sse_text.contains("chat.completion.chunk"));
    assert!(sse_text.contains("[DONE]"));

    // 3. Non-streaming /v1/completions + Streaming /v1/completions.
    let text_body = json!({
        "model": "tiny",
        "prompt": "abc",
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": false
    });
    let text_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(text_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(text_resp.status(), StatusCode::OK);
    let tv: Value = serde_json::from_slice(
        &axum::body::to_bytes(text_resp.into_body(), 128 * 1024)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(tv["object"], "text_completion");

    let text_sse_body = json!({
        "prompt": "abc",
        "max_tokens": 4,
        "stream": true
    });
    let text_sse = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(text_sse_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(text_sse.status(), StatusCode::OK);
    let text_sse_str = String::from_utf8(
        axum::body::to_bytes(text_sse.into_body(), 128 * 1024)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(text_sse_str.contains("text_completion"));
    assert!(text_sse_str.contains("[DONE]"));

    // 4. Error branches: unsupported tool_choice, invalid temperature, invalid response_format.
    for bad in [
        json!({"model": "tiny", "messages": [{"role": "user", "content": "a"}], "tool_choice": "required"}),
        json!({"model": "tiny", "messages": [{"role": "user", "content": "a"}], "temperature": -1.0}),
        json!({"model": "tiny", "messages": [{"role": "user", "content": "a"}], "response_format": {"type": "bogus"}}),
    ] {
        let r = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(bad.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    }

    // 5. /v1/unload evicts the loaded model back to 0 VRAM.
    let unload_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/unload")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unload_resp.status(), StatusCode::OK);
    let uv: Value = serde_json::from_slice(
        &axum::body::to_bytes(unload_resp.into_body(), 64 * 1024)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(uv["unloaded"], true);
    assert_eq!(uv["loaded"], false);

    // 6. api_error 500 branch & serve_* entrypoints (graceful shutdown + spawn/abort).
    let err500 = api_error(StatusCode::INTERNAL_SERVER_ERROR, "boom");
    assert_eq!(err500.status(), StatusCode::INTERNAL_SERVER_ERROR);

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let srv_state = AppState::new(make_config(reg.clone(), Duration::from_secs(60)));
    let srv_state_shut = srv_state.clone();
    let task = tokio::spawn(async move {
        serve_with_state_until(listener, srv_state, None, async move {
            let _ = rx.await;
        })
        .await
        .unwrap();
    });
    tokio::time::sleep(Duration::from_millis(30)).await;
    srv_state_shut.request_shutdown();
    let _ = tx.send(());
    task.await.unwrap();

    for mode in 0..4 {
        let l = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let cfg = make_config(reg.clone(), Duration::from_secs(60));
        let h = tokio::spawn(async move {
            let _ = match mode {
                0 => layerstream_server::serve(l, cfg).await,
                1 => layerstream_server::serve_with_api_key(l, cfg, Some("k".into())).await,
                2 => layerstream_server::serve_with_state(l, AppState::new(cfg)).await,
                _ => {
                    layerstream_server::serve_with_state_and_token(
                        l,
                        AppState::new(cfg),
                        "tok".into(),
                    )
                    .await
                }
            };
        });
        tokio::time::sleep(Duration::from_millis(15)).await;
        h.abort();
        let _ = h.await;
    }
}
