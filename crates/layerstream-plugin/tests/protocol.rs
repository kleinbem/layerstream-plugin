//! End-to-end Mesh-LLM plugin IPC protocol tests against the real `layerstream-plugin` binary
//! (modeled on official Mesh-LLM plugin protocol/lifecycle suites).

#![cfg(unix)]

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use mesh_llm_plugin::models::{ASSET_CAPABILITY, AssetPlan, CONFIG_ENV, PreparedModelAssets};
use mesh_llm_plugin::{LocalStream, PROTOCOL_VERSION, proto, read_envelope, write_envelope};
use serde_json::{Value, json};
use tokio::net::UnixListener;
use tokio::process::Command;
use tokio::time::timeout;

async fn exchange(
    stream: &mut LocalStream,
    id: u64,
    payload: proto::envelope::Payload,
) -> Result<proto::envelope::Payload> {
    write_envelope(
        stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "layerstream-plugin".into(),
            request_id: id,
            payload: Some(payload),
        },
    )
    .await?;
    let response = timeout(Duration::from_secs(10), read_envelope(stream)).await??;
    assert_eq!(response.request_id, id);
    response.payload.context("missing response payload")
}

fn plugin_command(socket: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_layerstream-plugin"));
    cmd.env("MESH_LLM_PLUGIN_ENDPOINT", socket)
        .env("MESH_LLM_PLUGIN_TRANSPORT", "unix")
        .env("MESH_LLM_PLUGIN_NAME", "layerstream-plugin")
        .env_remove(CONFIG_ENV)
        .stdin(Stdio::null())
        .kill_on_drop(true);
    cmd
}

/// Writes a minimal valid GGUF v3 header parseable by `GgufIndex::open` (0 tensors, 2 KV pairs).
fn write_minimal_gguf(path: &Path) -> Result<()> {
    fn push_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes()); // version 3
    bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    bytes.extend_from_slice(&2u64.to_le_bytes()); // kv_count = 2

    // KV 1: general.architecture = "llama" (GGUF_TYPE_STRING = 8)
    push_str(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8u32.to_le_bytes());
    push_str(&mut bytes, "llama");

    // KV 2: llama.block_count = 4 (GGUF_TYPE_UINT32 = 4)
    push_str(&mut bytes, "llama.block_count");
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&4u32.to_le_bytes());

    while bytes.len() % 32 != 0 {
        bytes.push(0);
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

#[tokio::test]
async fn executable_handshake_health_and_clean_shutdown() -> Result<()> {
    let dir = tempfile::tempdir_in("/tmp")?;
    let socket = dir.path().join("host.sock");
    let models_dir = dir.path().join("models");
    std::fs::create_dir_all(&models_dir)?;
    let listener = UnixListener::bind(&socket)?;

    let mut child = plugin_command(&socket)
        .args([
            "--no-gpu",
            "--models-dir",
            &models_dir.display().to_string(),
        ])
        .spawn()?;
    let (stream, _) = timeout(Duration::from_secs(10), listener.accept()).await??;
    let mut stream = LocalStream::Unix(stream);

    // 1. Initialize handshake.
    let init = exchange(
        &mut stream,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION,
            host_version: "layerstream-protocol-test".into(),
            ..Default::default()
        }),
    )
    .await?;
    let proto::envelope::Payload::InitializeResponse(init_resp) = init else {
        bail!("unexpected initialize response: {init:?}");
    };
    assert_eq!(init_resp.plugin_protocol_version, PROTOCOL_VERSION);
    let manifest = init_resp.manifest.context("missing manifest")?;
    assert_eq!(manifest.endpoints.len(), 1);
    assert!(manifest.endpoints[0].managed_by_plugin);
    assert_eq!(
        manifest.endpoints[0].protocol.as_deref(),
        Some("openai_compatible")
    );
    let endpoint_url = manifest.endpoints[0]
        .address
        .clone()
        .context("missing endpoint address")?;
    assert!(endpoint_url.starts_with("http://127.0.0.1:"));
    assert!(endpoint_url.ends_with("/v1"));

    // 2. Health check reports idle 0 MB VRAM.
    let health = exchange(
        &mut stream,
        2,
        proto::envelope::Payload::HealthRequest(proto::HealthRequest {}),
    )
    .await?;
    let proto::envelope::Payload::HealthResponse(health_resp) = health else {
        bail!("unexpected health response: {health:?}");
    };
    assert!(health_resp.detail.contains("idle_0mb_vram"));

    // 3. Clean shutdown tears down the plugin cleanly.
    let shutdown = exchange(
        &mut stream,
        3,
        proto::envelope::Payload::ShutdownRequest(proto::ShutdownRequest {
            reason: "protocol test complete".into(),
        }),
    )
    .await?;
    assert!(matches!(
        shutdown,
        proto::envelope::Payload::ShutdownResponse(_)
    ));
    drop(stream);

    let status = timeout(Duration::from_secs(10), child.wait()).await??;
    assert!(status.success(), "plugin exited with {status}");
    Ok(())
}

#[tokio::test]
async fn managed_models_prepare_assets_and_register_via_model_assets_v1() -> Result<()> {
    let dir = tempfile::tempdir_in("/tmp")?;
    let socket = dir.path().join("host.sock");
    let snapshot_dir = dir.path().join("hf-snapshot");
    std::fs::create_dir_all(&snapshot_dir)?;
    let shard1 = snapshot_dir.join("demo-00001-of-00002.gguf");
    let shard2 = snapshot_dir.join("demo-00002-of-00002.gguf");
    write_minimal_gguf(&shard1)?;
    write_minimal_gguf(&shard2)?;

    let local_gguf = dir.path().join("cli-local.gguf");
    write_minimal_gguf(&local_gguf)?;

    let listener = UnixListener::bind(&socket)?;
    let managed_env = serde_json::to_string(&json!([
        {
            "model": "org/demo-gguf@v1",
            "provider": "layerstream-plugin",
            "variant": "demo-00001-of-00002.gguf",
            "lifecycle": {
                "load": "on_demand",
                "idle_timeout_secs": 30,
                "prefetch": true
            }
        }
    ]))?;

    let mut child = plugin_command(&socket)
        .env(CONFIG_ENV, managed_env)
        .args([
            "--no-gpu",
            "--model",
            &format!("cli-model={}", local_gguf.display()),
        ])
        .spawn()?;
    let (stream, _) = timeout(Duration::from_secs(10), listener.accept()).await??;
    let mut stream = LocalStream::Unix(stream);

    // 1. Initialize handshake advertising model_assets.v1.
    let init = exchange(
        &mut stream,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION,
            host_version: "layerstream-protocol-test".into(),
            host_capabilities: vec![ASSET_CAPABILITY.into()],
            ..Default::default()
        }),
    )
    .await?;
    let proto::envelope::Payload::InitializeResponse(init_resp) = init else {
        bail!("unexpected initialize response: {init:?}");
    };
    let endpoint_url = init_resp
        .manifest
        .context("missing manifest")?
        .endpoints
        .into_iter()
        .next()
        .and_then(|e| e.address)
        .context("missing endpoint address")?;

    // 2. Receive plugin-originated PrepareModelAssetsRequest and verify shard expansion.
    let asset_req = timeout(Duration::from_secs(10), read_envelope(&mut stream)).await??;
    let Some(proto::envelope::Payload::PrepareModelAssetsRequest(body)) = asset_req.payload else {
        bail!("expected PrepareModelAssetsRequest, got: {asset_req:?}");
    };
    let plan: AssetPlan = serde_json::from_str(&body.plan_json)?;
    assert_eq!(plan.model, "org/demo-gguf@v1");
    assert_eq!(plan.repo, "org/demo-gguf");
    assert_eq!(plan.revision, "v1");
    assert_eq!(
        plan.files,
        vec![
            "demo-00001-of-00002.gguf".to_string(),
            "demo-00002-of-00002.gguf".to_string(),
        ]
    );

    // 3. Respond with PreparedModelAssets pointing to the local synthetic GGUF shards.
    let assets = PreparedModelAssets {
        directory: snapshot_dir.clone(),
        files: BTreeMap::from([
            ("demo-00001-of-00002.gguf".into(), shard1.clone()),
            ("demo-00002-of-00002.gguf".into(), shard2.clone()),
        ]),
        revision: "0123456789abcdef0123456789abcdef01234567".into(),
    };
    write_envelope(
        &mut stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "layerstream-plugin".into(),
            request_id: asset_req.request_id,
            payload: Some(proto::envelope::Payload::PrepareModelAssetsResponse(
                proto::PrepareModelAssetsResponse {
                    assets_json: serde_json::to_string(&assets)?,
                },
            )),
        },
    )
    .await?;

    // 4. Poll HealthRequest until both CLI and managed models are registered at 0 MB VRAM.
    let mut ready_detail = String::new();
    for attempt in 0..40 {
        let health = exchange(
            &mut stream,
            10 + attempt,
            proto::envelope::Payload::HealthRequest(proto::HealthRequest {}),
        )
        .await?;
        let proto::envelope::Payload::HealthResponse(resp) = health else {
            bail!("unexpected health response: {health:?}");
        };
        if resp.detail.contains("models=2") && resp.detail.contains("state=idle_0mb_vram") {
            ready_detail = resp.detail;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        ready_detail.contains("models=2") && ready_detail.contains("state=idle_0mb_vram"),
        "merged CLI + managed models never reached idle_0mb_vram: {ready_detail:?}"
    );

    // 5. Verify GET /v1/models lists both CLI and managed models with Mesh-LLM inventory fields.
    let models_url = format!("{endpoint_url}/models");
    let models_json: Value = tokio::task::spawn_blocking(move || -> Result<Value> {
        let mut resp = ureq::get(&models_url).call()?;
        let body = resp.body_mut().read_to_string()?;
        Ok(serde_json::from_str(&body)?)
    })
    .await??;
    let data = models_json["data"]
        .as_array()
        .context("missing data array")?;
    assert_eq!(data.len(), 2);
    let managed_entry = data
        .iter()
        .find(|entry| entry["id"] == "org/demo-gguf@v1")
        .context("missing managed model in /v1/models")?;
    assert_eq!(managed_entry["workload_class"], "causal_generation");
    assert_eq!(managed_entry["availability"], "loadable");
    assert_eq!(managed_entry["meta"]["architecture"], "llama");
    assert_eq!(managed_entry["meta"]["n_layers"], 4);
    assert!(
        data.iter().any(|entry| entry["id"] == "cli-model"),
        "CLI --model entry should be preserved alongside managed models: {data:?}"
    );

    // 6. Clean shutdown.
    let shutdown = exchange(
        &mut stream,
        99,
        proto::envelope::Payload::ShutdownRequest(proto::ShutdownRequest {
            reason: "managed model test complete".into(),
        }),
    )
    .await?;
    assert!(matches!(
        shutdown,
        proto::envelope::Payload::ShutdownResponse(_)
    ));
    drop(stream);

    let status = timeout(Duration::from_secs(10), child.wait()).await??;
    assert!(status.success(), "plugin exited with {status}");
    Ok(())
}

#[tokio::test]
async fn managed_models_without_host_capability_returns_startup_disabled() -> Result<()> {
    let dir = tempfile::tempdir_in("/tmp")?;
    let socket = dir.path().join("host.sock");
    let listener = UnixListener::bind(&socket)?;
    let managed_env = serde_json::to_string(&json!([
        {
            "model": "org/demo-gguf",
            "provider": "layerstream-plugin",
            "variant": "demo.gguf"
        }
    ]))?;

    let mut child = plugin_command(&socket)
        .env(CONFIG_ENV, managed_env)
        .args(["--no-gpu"])
        .spawn()?;
    let (stream, _) = timeout(Duration::from_secs(10), listener.accept()).await??;
    let mut stream = LocalStream::Unix(stream);

    let resp = exchange(
        &mut stream,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION,
            host_version: "older-host".into(),
            host_capabilities: vec![],
            ..Default::default()
        }),
    )
    .await?;
    let proto::envelope::Payload::ErrorResponse(err) = resp else {
        bail!("expected ErrorResponse when host lacks model_assets.v1, got: {resp:?}");
    };
    assert_eq!(
        err.code,
        mesh_llm_plugin::PluginError::startup_disabled("").code
    );
    assert!(err.message.contains(ASSET_CAPABILITY), "{}", err.message);
    drop(stream);
    let _ = timeout(Duration::from_secs(5), child.wait()).await;
    Ok(())
}

#[tokio::test]
async fn invalid_managed_model_config_returns_startup_disabled() -> Result<()> {
    let dir = tempfile::tempdir_in("/tmp")?;
    let socket = dir.path().join("host.sock");
    let listener = UnixListener::bind(&socket)?;
    let managed_env = serde_json::to_string(&json!([
        {
            "model": "org/demo-gguf",
            "provider": "layerstream-plugin"
        }
    ]))?;

    let mut child = plugin_command(&socket)
        .env(CONFIG_ENV, managed_env)
        .args(["--no-gpu"])
        .spawn()?;
    let (stream, _) = timeout(Duration::from_secs(10), listener.accept()).await??;
    let mut stream = LocalStream::Unix(stream);

    let resp = exchange(
        &mut stream,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION,
            host_version: "layerstream-protocol-test".into(),
            host_capabilities: vec![ASSET_CAPABILITY.into()],
            ..Default::default()
        }),
    )
    .await?;
    let proto::envelope::Payload::ErrorResponse(err) = resp else {
        bail!("expected ErrorResponse for invalid managed config, got: {resp:?}");
    };
    assert_eq!(
        err.code,
        mesh_llm_plugin::PluginError::startup_disabled("").code
    );
    assert!(
        err.message.contains("invalid managed model configuration"),
        "{}",
        err.message
    );
    drop(stream);
    let _ = timeout(Duration::from_secs(5), child.wait()).await;
    Ok(())
}

#[tokio::test]
async fn managed_model_host_error_surfaces_in_health() -> Result<()> {
    let dir = tempfile::tempdir_in("/tmp")?;
    let socket = dir.path().join("host.sock");
    let listener = UnixListener::bind(&socket)?;
    let managed_env = serde_json::to_string(&json!([
        {
            "model": "org/demo-gguf",
            "provider": "layerstream-plugin",
            "variant": "demo.gguf"
        }
    ]))?;

    let mut child = plugin_command(&socket)
        .env(CONFIG_ENV, managed_env)
        .args(["--no-gpu"])
        .spawn()?;
    let (stream, _) = timeout(Duration::from_secs(10), listener.accept()).await??;
    let mut stream = LocalStream::Unix(stream);

    let init = exchange(
        &mut stream,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION,
            host_version: "layerstream-protocol-test".into(),
            host_capabilities: vec![ASSET_CAPABILITY.into()],
            ..Default::default()
        }),
    )
    .await?;
    assert!(matches!(
        init,
        proto::envelope::Payload::InitializeResponse(_)
    ));

    let asset_req = timeout(Duration::from_secs(10), read_envelope(&mut stream)).await??;
    assert!(matches!(
        asset_req.payload,
        Some(proto::envelope::Payload::PrepareModelAssetsRequest(_))
    ));
    write_envelope(
        &mut stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "layerstream-plugin".into(),
            request_id: asset_req.request_id,
            payload: Some(proto::envelope::Payload::ErrorResponse(
                proto::ErrorResponse {
                    code: -32602,
                    message: "model assets: offline cache miss".into(),
                    data_json: String::new(),
                },
            )),
        },
    )
    .await?;

    let mut saw_error = None;
    for attempt in 0..40 {
        let health = exchange(
            &mut stream,
            20 + attempt,
            proto::envelope::Payload::HealthRequest(proto::HealthRequest {}),
        )
        .await?;
        if let proto::envelope::Payload::ErrorResponse(err) = health {
            saw_error = Some(err.message);
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let msg = saw_error.context("expected health check to report asset preparation failure")?;
    assert!(msg.contains("offline cache miss"), "{msg}");

    drop(stream);
    let _ = timeout(Duration::from_secs(5), child.wait()).await;
    Ok(())
}

#[tokio::test]
async fn zero_models_returns_startup_disabled() -> Result<()> {
    let dir = tempfile::tempdir_in("/tmp")?;
    let socket = dir.path().join("host.sock");
    let listener = UnixListener::bind(&socket)?;

    let mut child = plugin_command(&socket).args(["--no-gpu"]).spawn()?;
    let (stream, _) = timeout(Duration::from_secs(10), listener.accept()).await??;
    let mut stream = LocalStream::Unix(stream);

    let resp = exchange(
        &mut stream,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION,
            host_version: "layerstream-protocol-test".into(),
            ..Default::default()
        }),
    )
    .await?;
    let proto::envelope::Payload::ErrorResponse(err) = resp else {
        bail!("expected ErrorResponse for 0 configured models, got: {resp:?}");
    };
    assert_eq!(
        err.code,
        mesh_llm_plugin::PluginError::startup_disabled("").code,
        "0 models must return startup_disabled (not crash-loop), got: {err:?}"
    );
    drop(stream);
    let _ = timeout(Duration::from_secs(5), child.wait()).await;
    Ok(())
}

#[tokio::test]
async fn incompatible_host_protocol_is_rejected() -> Result<()> {
    let dir = tempfile::tempdir_in("/tmp")?;
    let socket = dir.path().join("host.sock");
    let listener = UnixListener::bind(&socket)?;

    let mut child = plugin_command(&socket).args(["--no-gpu"]).spawn()?;
    let (stream, _) = timeout(Duration::from_secs(10), listener.accept()).await??;
    let mut stream = LocalStream::Unix(stream);

    let resp = exchange(
        &mut stream,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION + 99,
            host_version: "incompatible-host".into(),
            ..Default::default()
        }),
    )
    .await?;
    let proto::envelope::Payload::ErrorResponse(err) = resp else {
        bail!("expected ErrorResponse for incompatible host protocol, got: {resp:?}");
    };
    assert_eq!(
        err.code,
        mesh_llm_plugin::PluginError::startup_disabled("").code,
        "protocol mismatch must return startup_disabled (not crash-loop), got: {err:?}"
    );
    drop(stream);
    let _ = timeout(Duration::from_secs(5), child.wait()).await;
    Ok(())
}
