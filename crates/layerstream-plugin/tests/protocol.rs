//! End-to-end Mesh-LLM plugin IPC protocol tests against the real `layerstream-plugin` binary
//! (modeled on official Mesh-LLM plugin protocol/lifecycle suites).

#![cfg(unix)]

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use mesh_llm_plugin::{LocalStream, PROTOCOL_VERSION, proto, read_envelope, write_envelope};
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
        .stdin(Stdio::null())
        .kill_on_drop(true);
    cmd
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
