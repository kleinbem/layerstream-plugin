//! Embedded `layerstream-server` HTTP/SSE server lifecycle bindings for Python.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;

use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use layerstream_core::{KvCacheType, StreamEngine, hf};
use layerstream_server::{AppState, ModelRegistry, ServerConfig, serve_with_state_until};

use crate::{VramBudgetArg, build_progress_callback, map_open_err};

struct ServerThreadState {
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    state: Option<AppState>,
    worker: Option<JoinHandle<()>>,
    finished: Arc<AtomicBool>,
}

/// Running handle to `layerstream-server`'s native Rust `axum` + `tokio` OpenAI HTTP/SSE server.
#[pyclass]
pub struct ServerHandle {
    host: String,
    port: u16,
    inner: Mutex<ServerThreadState>,
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        let mut st = self.inner.lock();
        if let Some(app_state) = st.state.take() {
            app_state.request_shutdown();
        }
        if let Some(tx) = st.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(h) = st.worker.take() {
            let _ = h.join();
        }
    }
}

#[pymethods]
impl ServerHandle {
    #[getter]
    fn host(&self) -> &str {
        &self.host
    }

    #[getter]
    fn port(&self) -> u16 {
        self.port
    }

    #[getter]
    fn base_url(&self) -> String {
        if self.host.contains(':') && !self.host.starts_with('[') {
            format!("http://[{}]:{}/v1", self.host, self.port)
        } else {
            format!("http://{}:{}/v1", self.host, self.port)
        }
    }

    fn stop(&self, py: Python<'_>) -> PyResult<()> {
        let (app_state, tx, worker) = {
            let mut guard = self.inner.lock();
            (
                guard.state.take(),
                guard.shutdown_tx.take(),
                guard.worker.take(),
            )
        };
        if let Some(s) = app_state {
            s.request_shutdown();
        }
        if let Some(t) = tx {
            let _ = t.send(());
        }
        if let Some(h) = worker {
            py.detach(move || {
                let _ = h.join();
            });
        }
        Ok(())
    }

    fn wait(&self, py: Python<'_>) -> PyResult<()> {
        loop {
            if let Err(err) = py.check_signals() {
                let _ = self.stop(py);
                return Err(err);
            }
            let finished = py.detach(|| {
                let guard = self.inner.lock();
                if guard.finished.load(Ordering::Acquire) {
                    return true;
                }
                guard.worker.as_ref().is_some_and(|w| w.is_finished())
            });
            if finished {
                self.stop(py)?;
                return Ok(());
            }
            py.detach(|| std::thread::sleep(Duration::from_millis(50)));
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.stop(py)?;
        Ok(false)
    }
}

/// Start `layerstream-server` (native Rust `axum` + `tokio` OpenAI HTTP/SSE server) on a background
/// runtime thread and return a `ServerHandle` immediately once the TCP socket is bound.
#[pyfunction]
#[pyo3(signature = (
    model,
    *,
    host="127.0.0.1",
    port=8080,
    api_key=None,
    draft_model=None,
    quant=None,
    revision=None,
    cache_dir=None,
    vram_budget_mb=VramBudgetArg::Auto,
    resident_layers=None,
    ctx_size=4096,
    batch_size=512,
    threads=None,
    kv_cache_type="q8_0",
    use_gpu=true,
    prefer_direct_io=true,
    cpu_embeddings=false,
    cpu_head=false,
    stream_head=false,
    moe_active_streaming=true,
    idle_timeout_secs=60,
    max_draft_tokens=5,
    verbose=false,
    on_progress=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn start_server(
    py: Python<'_>,
    model: &str,
    host: &str,
    port: u16,
    api_key: Option<String>,
    draft_model: Option<&str>,
    quant: Option<&str>,
    revision: Option<&str>,
    cache_dir: Option<PathBuf>,
    vram_budget_mb: VramBudgetArg,
    resident_layers: Option<u32>,
    ctx_size: u32,
    batch_size: u32,
    threads: Option<u32>,
    kv_cache_type: &str,
    use_gpu: bool,
    prefer_direct_io: bool,
    cpu_embeddings: bool,
    cpu_head: bool,
    stream_head: bool,
    moe_active_streaming: bool,
    idle_timeout_secs: u64,
    max_draft_tokens: usize,
    verbose: bool,
    on_progress: Option<&Bound<'_, PyAny>>,
) -> PyResult<ServerHandle> {
    StreamEngine::set_log_verbose(verbose);
    if let Some(ref k) = api_key
        && k.trim().is_empty()
    {
        return Err(PyValueError::new_err(
            "api_key must not be empty; pass None to run unauthenticated",
        ));
    }
    let kv_type: KvCacheType = kv_cache_type
        .parse()
        .map_err(|e: anyhow::Error| PyValueError::new_err(e.to_string()))?;
    let (progress_cb, cb_err_slot) = build_progress_callback(on_progress)?;
    let (vram_budget_auto, vram_budget_bytes) = match vram_budget_mb {
        VramBudgetArg::Auto => (true, 0u64),
        VramBudgetArg::Bytes(b) => (false, b),
    };

    let model_owned = model.to_string();
    let host_owned = host.to_string();
    let draft_owned = draft_model.map(str::to_string);
    let quant_owned = quant.map(str::to_string);
    let rev_owned = revision.map(str::to_string);

    let cfg_res = py.detach(move || -> anyhow::Result<(ServerConfig, String)> {
        let vram_opt = (vram_budget_bytes > 0).then_some(vram_budget_bytes);
        let target_opts = hf::PretrainedResolveOptions {
            revision: rev_owned.clone(),
            quant: quant_owned,
            cache_dir: cache_dir.clone(),
            vram_budget_bytes: vram_opt,
            use_gpu,
            has_draft: draft_owned.is_some(),
            on_progress: progress_cb.clone(),
        };
        let resolved_target = hf::resolve_pretrained_model(&model_owned, &target_opts)?;

        let resolved_draft_path = if let Some(d_spec) = draft_owned {
            if d_spec.trim().eq_ignore_ascii_case("auto") {
                hf::resolve_auto_draft_model(
                    &resolved_target,
                    cache_dir,
                    vram_opt,
                    use_gpu,
                    progress_cb,
                )?
                .map(|r| r.local_path)
            } else {
                let d_opts = hf::PretrainedResolveOptions {
                    revision: rev_owned,
                    quant: None,
                    cache_dir,
                    vram_budget_bytes: None,
                    use_gpu,
                    has_draft: false,
                    on_progress: progress_cb,
                };
                Some(hf::resolve_pretrained_model(&d_spec, &d_opts)?.local_path)
            }
        } else {
            None
        };

        let mut registry = ModelRegistry::default();
        let alias = {
            let t = model_owned.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        };
        registry.register_file(&resolved_target.local_path, alias, resolved_draft_path)?;
        let n_threads = threads.unwrap_or_else(|| {
            if use_gpu {
                2
            } else {
                std::thread::available_parallelism()
                    .map(|n| n.get().min(8) as u32)
                    .unwrap_or(4)
            }
        });
        let cfg = ServerConfig {
            registry,
            n_ctx: ctx_size.max(1),
            n_batch: batch_size.max(1),
            n_threads: n_threads.max(1),
            resident_layers,
            vram_budget_bytes,
            vram_budget_auto: vram_budget_auto && resident_layers.is_none(),
            kv_cache_type: kv_type,
            use_gpu,
            prefer_direct_io,
            cpu_embeddings,
            cpu_head,
            stream_head,
            moe_active_streaming,
            idle_timeout: Duration::from_secs(idle_timeout_secs),
            enable_ngram_speculation: true,
            max_draft_tokens: max_draft_tokens.max(1),
            max_queued_requests: 8,
        };
        Ok((cfg, host_owned))
    });

    if let Some(py_err) = cb_err_slot.lock().take() {
        return Err(py_err);
    }
    let (server_cfg, bind_host) = cfg_res.map_err(map_open_err)?;

    let (ready_tx, ready_rx) = mpsc::sync_channel::<anyhow::Result<(String, u16, AppState)>>(1);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let finished = Arc::new(AtomicBool::new(false));
    let worker_finished = finished.clone();

    let worker = std::thread::spawn(move || {
        struct MarkDone(Arc<AtomicBool>);
        impl Drop for MarkDone {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let _done = MarkDone(worker_finished);
        let rt = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                let _ = ready_tx.send(Err(anyhow::anyhow!("failed to build tokio runtime: {e}")));
                return;
            }
        };
        rt.block_on(async move {
            let bind_addr = format!("{bind_host}:{port}");
            let listener = match tokio::net::TcpListener::bind(&bind_addr).await {
                Ok(l) => l,
                Err(e) => {
                    let _ = ready_tx.send(Err(anyhow::anyhow!("failed to bind {bind_addr}: {e}")));
                    return;
                }
            };
            let local_addr = match listener.local_addr() {
                Ok(a) => a,
                Err(e) => {
                    let _ = ready_tx.send(Err(anyhow::anyhow!("local_addr failed: {e}")));
                    return;
                }
            };
            let state = AppState::new(server_cfg);
            let run_state = state.clone();
            let _ = ready_tx.send(Ok((local_addr.ip().to_string(), local_addr.port(), state)));

            let _ = serve_with_state_until(listener, run_state, api_key, async move {
                let _ = shutdown_rx.await;
            })
            .await;
        });
        rt.shutdown_timeout(Duration::from_secs(2));
    });

    let (bound_host, bound_port, app_state) = py
        .detach(move || ready_rx.recv())
        .map_err(|_| PyRuntimeError::new_err("server startup thread exited unexpectedly"))?
        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;

    Ok(ServerHandle {
        host: bound_host,
        port: bound_port,
        inner: Mutex::new(ServerThreadState {
            shutdown_tx: Some(shutdown_tx),
            state: Some(app_state),
            worker: Some(worker),
            finished,
        }),
    })
}
