//! Lifecycle, cancellation, queue-admission, and non-blocking `hf://` pull integration tests
//! against the real `layerstream-plugin` binary with a mock Mesh-LLM host.
//!
//! Model-backed tests need `LAYERSTREAM_TEST_MODEL=/path/to/small.gguf`
//! (e.g. `qwen2.5-0.5b-instruct-q4_k_m.gguf`) and run on CPU (`--no-gpu`).
//! Run: `cargo test --release -p layerstream-plugin --test lifecycle -- --nocapture --test-threads 1`

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use mesh_llm_plugin::{LocalStream, PROTOCOL_VERSION, proto, read_envelope, write_envelope};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, UnixListener};
use tokio::process::{Child, Command};
use tokio::time::{sleep, timeout};

/// Mesh-LLM host default (`DEFAULT_PLUGIN_CONNECT_TIMEOUT_SECS`).
const HOST_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

fn test_model() -> Option<PathBuf> {
    std::env::var_os("LAYERSTREAM_TEST_MODEL")
        .map(PathBuf::from)
        .filter(|p| p.is_file())
}

fn plugin_command(socket: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_layerstream-plugin"));
    cmd.env("MESH_LLM_PLUGIN_ENDPOINT", socket)
        .env("MESH_LLM_PLUGIN_TRANSPORT", "unix")
        .env("MESH_LLM_PLUGIN_NAME", "layerstream-plugin")
        .env("RUST_LOG", "warn")
        .stdin(Stdio::null())
        .kill_on_drop(true);
    cmd
}

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
    response.payload.context("missing response payload")
}

struct Running {
    child: Child,
    ipc: Option<LocalStream>,
    port: u16,
    _dir: tempfile::TempDir,
}

async fn start_plugin(model: &Path, extra: &[&str]) -> Result<Running> {
    let dir = tempfile::tempdir_in("/tmp")?;
    let socket = dir.path().join("host.sock");
    let listener = UnixListener::bind(&socket)?;
    let mut args: Vec<String> = vec![
        "--model".into(),
        model.display().to_string(),
        "--no-gpu".into(),
        "--vram-budget-mb".into(),
        "0".into(),
        "--ctx-size".into(),
        "4096".into(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let child = plugin_command(&socket).args(&args).spawn()?;
    let (stream, _) = timeout(HOST_CONNECT_TIMEOUT, listener.accept()).await??;
    let mut stream = LocalStream::Unix(stream);
    let init = exchange(
        &mut stream,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION,
            host_version: "layerstream-lifecycle-test".into(),
            ..Default::default()
        }),
    )
    .await?;
    let proto::envelope::Payload::InitializeResponse(resp) = init else {
        bail!("unexpected initialize response: {init:?}");
    };
    let url = resp.manifest.context("manifest")?.endpoints[0]
        .address
        .clone()
        .context("endpoint address")?;
    let port: u16 = url
        .trim_start_matches("http://127.0.0.1:")
        .trim_end_matches("/v1")
        .parse()?;
    Ok(Running {
        child,
        ipc: Some(stream),
        port,
        _dir: dir,
    })
}

fn chat_body(max_tokens: usize, stream: bool) -> String {
    format!(
        r#"{{"messages":[{{"role":"user","content":"Write a detailed chronological history of European maritime navigation from antiquity to the twentieth century."}}],"max_tokens":{max_tokens},"stream":{stream},"temperature":0}}"#
    )
}

async fn post(port: u16, path: &str, body: &str) -> Result<TcpStream> {
    let mut s = TcpStream::connect(("127.0.0.1", port)).await?;
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await?;
    Ok(s)
}

async fn read_some(s: &mut TcpStream, min_bytes: usize, limit: Duration) -> Result<String> {
    let mut buf = Vec::new();
    let deadline = Instant::now() + limit;
    let mut chunk = [0u8; 4096];
    while buf.len() < min_bytes {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        match timeout(left, s.read(&mut chunk)).await {
            Ok(Ok(0)) => break,
            Ok(Ok(n)) => buf.extend_from_slice(&chunk[..n]),
            _ => break,
        }
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

async fn read_to_end(s: &mut TcpStream, limit: Duration) -> Result<String> {
    let mut buf = Vec::new();
    timeout(limit, s.read_to_end(&mut buf)).await??;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

async fn wait_exit(child: &mut Child, limit: Duration) -> Option<Duration> {
    let t0 = Instant::now();
    match timeout(limit, child.wait()).await {
        Ok(_) => Some(t0.elapsed()),
        Err(_) => None,
    }
}

#[tokio::test]
async fn host_eof_during_generation_exits_promptly() -> Result<()> {
    let Some(model) = test_model() else {
        eprintln!("skipped: LAYERSTREAM_TEST_MODEL not set");
        return Ok(());
    };
    let mut p = start_plugin(&model, &[]).await?;
    let mut req = post(p.port, "/v1/chat/completions", &chat_body(900, false)).await?;
    sleep(Duration::from_secs(3)).await;
    drop(p.ipc.take());
    let exited = wait_exit(&mut p.child, Duration::from_secs(15)).await;
    let _ = read_some(&mut req, 1, Duration::from_millis(10)).await;
    match exited {
        Some(d) => {
            assert!(
                d < Duration::from_secs(5),
                "plugin lingered {:.1}s after host EOF",
                d.as_secs_f64()
            );
        }
        None => {
            let _ = p.child.kill().await;
            panic!("plugin still running 15s after host EOF");
        }
    }
    Ok(())
}

#[tokio::test]
async fn sigterm_during_generation_exits_promptly() -> Result<()> {
    let Some(model) = test_model() else {
        eprintln!("skipped: LAYERSTREAM_TEST_MODEL not set");
        return Ok(());
    };
    let mut p = start_plugin(&model, &[]).await?;
    let _req = post(p.port, "/v1/chat/completions", &chat_body(900, false)).await?;
    sleep(Duration::from_secs(3)).await;
    let pid = p.child.id().context("pid")?;
    std::process::Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()?;
    let exited = wait_exit(&mut p.child, Duration::from_secs(15)).await;
    match exited {
        Some(d) => {
            assert!(
                d < Duration::from_secs(5),
                "plugin lingered {:.1}s after SIGTERM",
                d.as_secs_f64()
            );
        }
        None => {
            let _ = p.child.kill().await;
            panic!("plugin still running 15s after SIGTERM");
        }
    }
    Ok(())
}

#[tokio::test]
async fn non_streaming_client_disconnect_cancels_generation() -> Result<()> {
    let Some(model) = test_model() else {
        eprintln!("skipped: LAYERSTREAM_TEST_MODEL not set");
        return Ok(());
    };
    let mut p = start_plugin(&model, &[]).await?;
    let req = post(p.port, "/v1/chat/completions", &chat_body(900, false)).await?;
    sleep(Duration::from_secs(3)).await;
    drop(req);
    let t0 = Instant::now();
    let mut probe = post(p.port, "/v1/chat/completions", &chat_body(1, false)).await?;
    let resp = read_to_end(&mut probe, Duration::from_secs(30)).await?;
    let waited = t0.elapsed();
    let _ = p.child.kill().await;
    assert!(
        resp.starts_with("HTTP/1.1 200"),
        "expected 200 OK on follow-up request, got: {}",
        resp.lines().next().unwrap_or("")
    );
    assert!(
        waited < Duration::from_secs(10),
        "abandoned non-streaming request kept engine busy for {:.1}s",
        waited.as_secs_f64()
    );
    Ok(())
}

#[tokio::test]
async fn streaming_request_with_full_queue_gets_503() -> Result<()> {
    let Some(model) = test_model() else {
        eprintln!("skipped: LAYERSTREAM_TEST_MODEL not set");
        return Ok(());
    };
    let mut p = start_plugin(&model, &[]).await?;
    let mut holders = Vec::new();
    for _ in 0..8 {
        let mut s = post(p.port, "/v1/chat/completions", &chat_body(2000, true)).await?;
        let _ = read_some(&mut s, 1, Duration::from_millis(250)).await;
        holders.push(s);
    }
    sleep(Duration::from_secs(1)).await;
    let mut ninth = post(p.port, "/v1/chat/completions", &chat_body(16, true)).await?;
    let resp = read_to_end(&mut ninth, Duration::from_secs(15)).await?;
    let status = resp.lines().next().unwrap_or("").to_string();
    drop(holders);
    let _ = p.child.kill().await;
    assert!(
        status.contains(" 503 "),
        "queue-full streaming request must return 503 Service Unavailable, got: {status:?}"
    );
    Ok(())
}

#[tokio::test]
async fn hf_model_download_does_not_block_host_connect() -> Result<()> {
    let mock = TcpListener::bind("127.0.0.1:0").await?;
    let mock_addr = mock.local_addr()?;
    let server = tokio::spawn(async move {
        while let Ok((mut sock, _)) = mock.accept().await {
            tokio::spawn(async move {
                let mut req = [0u8; 4096];
                let _ = sock.read(&mut req).await;
                let header =
                    "HTTP/1.1 200 OK\r\nContent-Length: 1073741824\r\nConnection: close\r\n\r\n";
                if sock.write_all(header.as_bytes()).await.is_err() {
                    return;
                }
                let chunk = vec![0u8; 64 * 1024];
                loop {
                    if sock.write_all(&chunk).await.is_err() {
                        return;
                    }
                    sleep(Duration::from_secs(1)).await;
                }
            });
        }
    });

    let dir = tempfile::tempdir_in("/tmp")?;
    let socket = dir.path().join("host.sock");
    let listener = UnixListener::bind(&socket)?;
    let cache = dir.path().join("cache");
    let mut child = plugin_command(&socket)
        .env("HF_ENDPOINT", format!("http://{mock_addr}"))
        .env("LAYERSTREAM_CACHE_DIR", &cache)
        .env_remove("HF_TOKEN")
        .args(["--no-gpu", "--model", "hf://testorg/testrepo:model.gguf"])
        .spawn()?;
    let connected = timeout(HOST_CONNECT_TIMEOUT, listener.accept()).await;
    let _ = child.kill().await;
    server.abort();
    assert!(
        connected.is_ok(),
        "plugin did not connect to host within {HOST_CONNECT_TIMEOUT:?} while hf:// pull was pending"
    );
    Ok(())
}
