//! Real-binary integration tests for `layerstream-plugin _worker` and isolated coordinator mode.

#![cfg(unix)]

use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::process::Command;
use tokio::time::timeout;

async fn http_req(
    port: u16,
    method: &str,
    path: &str,
    extra_header: Option<(&str, &str)>,
    body: &str,
) -> Result<String> {
    let mut s = TcpStream::connect(("127.0.0.1", port)).await?;
    let mut req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    if let Some((k, v)) = extra_header {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    req.push_str(body);
    s.write_all(req.as_bytes()).await?;
    let mut out = String::new();
    timeout(Duration::from_secs(10), s.read_to_string(&mut out)).await??;
    Ok(out)
}

#[tokio::test]
async fn real_worker_requires_token_and_exits_on_stdin_eof() -> Result<()> {
    let token = "deadbeef0123456789abcdef01234567";
    let mut child = Command::new(env!("CARGO_BIN_EXE_layerstream-plugin"))
        .arg("_worker")
        .env("RUST_LOG", "warn")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;

    let mut stdin = child.stdin.take().context("worker stdin")?;
    let stdout = child.stdout.take().context("worker stdout")?;

    let spec = serde_json::json!({
        "models": [],
        "n_ctx": 512,
        "n_batch": 128,
        "n_threads": 2,
        "resident_layers": 0,
        "vram_budget_bytes": 0,
        "vram_budget_auto": false,
        "kv_cache_type": "q8_0",
        "use_gpu": false,
        "prefer_direct_io": true,
        "cpu_embeddings": false,
        "cpu_head": false,
        "stream_head": false,
        "moe_active_streaming": true,
        "enable_ngram_speculation": true,
        "max_draft_tokens": 4,
        "max_queued_requests": 4,
        "worker_token": token,
    });
    stdin
        .write_all(format!("{}\n", serde_json::to_string(&spec)?).as_bytes())
        .await?;
    stdin.flush().await?;

    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    let n = timeout(Duration::from_secs(10), reader.read_line(&mut line)).await??;
    if n == 0 {
        bail!("worker exited before printing ready line");
    }
    let url = line
        .trim()
        .strip_prefix("LAYERSTREAM_WORKER_READY ")
        .context("missing LAYERSTREAM_WORKER_READY prefix")?;
    let port: u16 = url
        .strip_prefix("http://127.0.0.1:")
        .context("unexpected ready url")?
        .parse()?;

    // 1. Missing token -> 403 Forbidden.
    let unauth = http_req(port, "GET", "/health", None, "").await?;
    assert!(unauth.starts_with("HTTP/1.1 403"), "{unauth}");

    // 2. Wrong token -> 403 Forbidden.
    let wrong = http_req(
        port,
        "GET",
        "/health",
        Some(("x-layerstream-worker-token", "wrong")),
        "",
    )
    .await?;
    assert!(wrong.starts_with("HTTP/1.1 403"), "{wrong}");

    // 3. Valid token -> 200 OK.
    let ok = http_req(
        port,
        "GET",
        "/health",
        Some(("x-layerstream-worker-token", token)),
        "",
    )
    .await?;
    assert!(ok.starts_with("HTTP/1.1 200"), "{ok}");
    let body = ok.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("");
    let v: Value = serde_json::from_str(body)?;
    assert_eq!(v["status"], "ok");
    assert_eq!(v["loaded"], false);

    // 4. Closing stdin triggers immediate clean worker exit.
    drop(stdin);
    let status = timeout(Duration::from_secs(5), child.wait()).await??;
    assert!(status.success(), "worker exited with {status}");
    Ok(())
}

#[tokio::test]
async fn real_serve_isolated_mode_keeps_zero_workers_when_idle() -> Result<()> {
    let dir = tempfile::tempdir_in("/tmp")?;
    let models_dir = dir.path().join("models");
    std::fs::create_dir_all(&models_dir)?;

    // Grab an unused ephemeral port.
    let probe = std::net::TcpListener::bind("127.0.0.1:0")?;
    let port = probe.local_addr()?.port();
    drop(probe);

    let mut child = Command::new(env!("CARGO_BIN_EXE_layerstream-plugin"))
        .args([
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--no-gpu",
            "--models-dir",
            &models_dir.display().to_string(),
        ])
        .env("RUST_LOG", "warn")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;

    // Poll /health until ready.
    let mut health_raw = String::new();
    for _ in 0..40 {
        if let Ok(resp) = http_req(port, "GET", "/health", None, "").await {
            health_raw = resp;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        health_raw.starts_with("HTTP/1.1 200"),
        "serve did not become healthy: {health_raw}"
    );
    let hv: Value = serde_json::from_str(
        health_raw
            .split_once("\r\n\r\n")
            .map(|(_, b)| b)
            .unwrap_or(""),
    )?;
    assert_eq!(hv["worker_isolated"], true);
    assert_eq!(hv["loaded"], false);

    // Unknown model returns 404 without spawning any worker.
    let nf = http_req(
        port,
        "POST",
        "/v1/chat/completions",
        None,
        r#"{"model":"missing","messages":[{"role":"user","content":"hi"}]}"#,
    )
    .await?;
    assert!(nf.starts_with("HTTP/1.1 404"), "{nf}");

    // /v1/unload reports unloaded=false and worker_isolated=true.
    let ul = http_req(port, "POST", "/v1/unload", None, "").await?;
    assert!(ul.starts_with("HTTP/1.1 200"), "{ul}");
    let uv: Value = serde_json::from_str(ul.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or(""))?;
    assert_eq!(uv["unloaded"], false);
    assert_eq!(uv["worker_isolated"], true);

    // SIGTERM exits cleanly.
    let pid = child.id().context("serve pid")?;
    std::process::Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()?;
    let status = timeout(Duration::from_secs(5), child.wait()).await??;
    assert!(status.success(), "serve exited with {status}");
    Ok(())
}
