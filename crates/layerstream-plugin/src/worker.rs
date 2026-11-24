//! On-demand isolated GPU worker subprocess coordinator & worker entrypoint.
//!
//! Keeping GPU inference inside a lazily spawned child process guarantees that
//! the coordinator process (`layerstream plugin` / `layerstream serve`) never
//! initializes CUDA/Vulkan/Metal drivers or loads `libggml-cuda.so` /
//! `libcublasLt.so`. When the worker exits on idle timeout (or synchronously
//! before HTTP response EOF when `--idle-timeout-secs 0`), the OS kernel
//! immediately reclaims 100% of VRAM/UMA, GPU driver contexts, and host heap
//! arenas — dropping idle footprint from ~524 MiB down to ~10–15 MiB.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderValue, Request, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::client::conn::http1;
use hyper_util::rt::TokioIo;
use layerstream_core::KvCacheType;
use layerstream_server::{
    AppState, MAX_REQUEST_BODY_BYTES, ModelRegistry, ServerConfig, WORKER_TOKEN_HEADER, api_error,
    build_router_with_token, models_list_json, with_api_key_auth,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{
    Mutex as AsyncMutex, OwnedMutexGuard, OwnedSemaphorePermit, Semaphore, mpsc, oneshot, watch,
};
use tokio_stream::wrappers::ReceiverStream;

const WORKER_READY_PREFIX: &str = "LAYERSTREAM_WORKER_READY ";
const WORKER_STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const WORKER_STOP_GRACE: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerModelSpec {
    id: String,
    path: PathBuf,
    draft_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerSpec {
    models: Vec<WorkerModelSpec>,
    n_ctx: u32,
    n_batch: u32,
    n_threads: u32,
    resident_layers: Option<u32>,
    vram_budget_bytes: u64,
    vram_budget_auto: bool,
    kv_cache_type: String,
    use_gpu: bool,
    prefer_direct_io: bool,
    cpu_embeddings: bool,
    cpu_head: bool,
    stream_head: bool,
    moe_active_streaming: bool,
    enable_ngram_speculation: bool,
    max_draft_tokens: usize,
    max_queued_requests: usize,
    worker_token: String,
}

fn kv_cache_type_to_str(kv: KvCacheType) -> &'static str {
    match kv {
        KvCacheType::F16 => "f16",
        KvCacheType::Q8_0 => "q8_0",
        KvCacheType::Q4_0 => "q4_0",
    }
}

fn parse_kv_cache_type(s: &str) -> Result<KvCacheType> {
    match s {
        "f16" => Ok(KvCacheType::F16),
        "q8_0" => Ok(KvCacheType::Q8_0),
        "q4_0" => Ok(KvCacheType::Q4_0),
        other => bail!("invalid worker kv_cache_type '{other}'"),
    }
}

fn generate_worker_token() -> Result<String> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes)
        .map_err(|e| anyhow!("failed to generate worker auth token: {e}"))?;
    let mut hex = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut hex, "{b:02x}");
    }
    Ok(hex)
}

/// Entrypoint for the hidden `layerstream _worker` subprocess.
pub async fn run_worker_process() -> Result<()> {
    let mut stdin_reader = BufReader::new(tokio::io::stdin());
    let mut spec_line = String::new();
    let n = tokio::time::timeout(
        Duration::from_secs(10),
        stdin_reader.read_line(&mut spec_line),
    )
    .await
    .context("timed out waiting for worker spec on stdin")?
    .context("read worker spec from stdin")?;
    if n == 0 {
        bail!("worker stdin closed before receiving spec");
    }
    let spec: WorkerSpec =
        serde_json::from_str(spec_line.trim()).context("parse worker spec JSON")?;

    let mut registry = ModelRegistry::default();
    for m in &spec.models {
        registry.register_file(&m.path, Some(m.id.clone()), m.draft_path.clone())?;
    }

    let cfg = ServerConfig {
        registry,
        n_ctx: spec.n_ctx,
        n_batch: spec.n_batch,
        n_threads: spec.n_threads,
        resident_layers: spec.resident_layers,
        vram_budget_bytes: spec.vram_budget_bytes,
        vram_budget_auto: spec.vram_budget_auto,
        kv_cache_type: parse_kv_cache_type(&spec.kv_cache_type)?,
        use_gpu: spec.use_gpu,
        prefer_direct_io: spec.prefer_direct_io,
        cpu_embeddings: spec.cpu_embeddings,
        cpu_head: spec.cpu_head,
        stream_head: spec.stream_head,
        moe_active_streaming: spec.moe_active_streaming,
        // Parent supervisor owns idle eviction and terminates the entire worker process.
        idle_timeout: Duration::from_secs(86_400),
        enable_ngram_speculation: spec.enable_ngram_speculation,
        max_draft_tokens: spec.max_draft_tokens,
        max_queued_requests: spec.max_queued_requests.max(1),
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind ephemeral loopback port for layerstream worker")?;
    let local_addr = listener.local_addr()?;

    let state = AppState::new(cfg);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let shutdown_tx = Arc::new(StdMutex::new(Some(shutdown_tx)));

    let trigger_shutdown = {
        let state = state.clone();
        let shutdown_tx = shutdown_tx.clone();
        move || {
            state.request_shutdown();
            if let Ok(mut guard) = shutdown_tx.lock()
                && let Some(tx) = guard.take()
            {
                let _ = tx.send(());
            }
        }
    };

    // Parent death / graceful stop watcher: closing parent's stdin pipe triggers immediate EOF.
    let eof_trigger = trigger_shutdown.clone();
    tokio::spawn(async move {
        let mut buf = [0u8; 64];
        loop {
            match stdin_reader.read(&mut buf).await {
                Ok(0) | Err(_) => {
                    eof_trigger();
                    break;
                }
                Ok(_) => {}
            }
        }
    });

    // OS signal watcher (SIGTERM / Ctrl-C).
    let sig_trigger = trigger_shutdown.clone();
    tokio::spawn(async move {
        worker_shutdown_signal().await;
        sig_trigger();
    });

    // Signal readiness to parent coordinator on stdout.
    {
        let mut stdout = tokio::io::stdout();
        stdout
            .write_all(format!("{WORKER_READY_PREFIX}http://{local_addr}\n").as_bytes())
            .await
            .context("write worker readiness line")?;
        stdout
            .flush()
            .await
            .context("flush worker readiness line")?;
    }

    let app = build_router_with_token(state.clone(), spec.worker_token);
    let server = axum::serve(listener, app).with_graceful_shutdown(async move {
        let _ = shutdown_rx.await;
    });

    // Run server; if graceful shutdown is triggered while a request is winding down,
    // allow a brief grace window so cancellation completes cleanly and exit promptly.
    let res = server.await;
    state.request_shutdown();
    res.context("layerstream worker HTTP server error")
}

async fn worker_shutdown_signal() {
    #[cfg(unix)]
    {
        if let Ok(mut term) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            tokio::select! {
                _ = term.recv() => {},
                _ = tokio::signal::ctrl_c() => {},
            }
            return;
        }
    }
    let _ = tokio::signal::ctrl_c().await;
}

struct RunningWorker {
    child: Child,
    stdin: Option<ChildStdin>,
    addr: SocketAddr,
    token: String,
    registry_generation: u64,
}

impl RunningWorker {
    fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    fn pid(&self) -> Option<u32> {
        self.child.id()
    }

    async fn stop(mut self) {
        let pid = self.pid();
        // Close stdin first to trigger graceful EOF shutdown inside the worker.
        drop(self.stdin.take());
        match tokio::time::timeout(WORKER_STOP_GRACE, self.child.wait()).await {
            Ok(Ok(_)) => {}
            Ok(Err(err)) => {
                tracing::debug!(?pid, "worker wait returned error: {err}");
            }
            Err(_) => {
                tracing::warn!(
                    ?pid,
                    "layerstream worker did not exit within grace window; sending kill"
                );
                let _ = self.child.kill().await;
                let _ = tokio::time::timeout(Duration::from_secs(2), self.child.wait()).await;
            }
        }
    }
}

struct CoordinatorActivity {
    in_flight: AtomicUsize,
    last_finished: StdMutex<Instant>,
}

struct CoordinatorPermit {
    activity: Arc<CoordinatorActivity>,
    _permit: OwnedSemaphorePermit,
}

impl Drop for CoordinatorPermit {
    fn drop(&mut self) {
        if let Ok(mut ts) = self.activity.last_finished.lock() {
            *ts = Instant::now();
        }
        self.activity.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

struct DropAbort(Option<tokio::task::JoinHandle<()>>);

impl Drop for DropAbort {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            handle.abort();
        }
    }
}

/// Parent HTTP coordinator that serves `/health`, `/v1/models`, and `/v1/unload`
/// with 0 MB GPU memory and lazily spawns an isolated `_worker` subprocess on
/// inference requests.
#[derive(Clone)]
pub struct WorkerCoordinator {
    config: Arc<ServerConfig>,
    registry: Arc<RwLock<ModelRegistry>>,
    registry_generation: Arc<AtomicU64>,
    loading_status: Arc<RwLock<Option<String>>>,
    loading_error: Arc<RwLock<Option<String>>>,
    shutdown: Arc<AtomicBool>,
    shutdown_tx: watch::Sender<bool>,
    worker: Arc<AsyncMutex<Option<RunningWorker>>>,
    loaded: Arc<AtomicBool>,
    admission: Arc<Semaphore>,
    activity: Arc<CoordinatorActivity>,
    worker_exe: Arc<PathBuf>,
    #[cfg(test)]
    worker_args: Arc<Vec<String>>,
    #[cfg(test)]
    worker_env: Arc<Vec<(String, String)>>,
}

impl WorkerCoordinator {
    pub fn new(config: ServerConfig) -> Result<Self> {
        let worker_exe = std::env::current_exe()
            .context("resolve current executable path for isolated worker")?;
        Ok(Self::with_worker_exe(config, worker_exe))
    }

    pub fn with_worker_exe(config: ServerConfig, worker_exe: PathBuf) -> Self {
        let registry = Arc::new(RwLock::new(config.registry.clone()));
        let max_queued = config.max_queued_requests.max(1);
        let (shutdown_tx, _) = watch::channel(false);
        Self {
            config: Arc::new(config),
            registry,
            registry_generation: Arc::new(AtomicU64::new(0)),
            loading_status: Arc::new(RwLock::new(None)),
            loading_error: Arc::new(RwLock::new(None)),
            shutdown: Arc::new(AtomicBool::new(false)),
            shutdown_tx,
            worker: Arc::new(AsyncMutex::new(None)),
            loaded: Arc::new(AtomicBool::new(false)),
            admission: Arc::new(Semaphore::new(max_queued)),
            activity: Arc::new(CoordinatorActivity {
                in_flight: AtomicUsize::new(0),
                last_finished: StdMutex::new(Instant::now()),
            }),
            worker_exe: Arc::new(worker_exe),
            #[cfg(test)]
            worker_args: Arc::new(vec!["_worker".to_string()]),
            #[cfg(test)]
            worker_env: Arc::new(Vec::new()),
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
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
            self.registry_generation.fetch_add(1, Ordering::SeqCst);
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

    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.shutdown_tx.send(true);
    }

    async fn wait_shutdown(&self) {
        if self.shutdown.load(Ordering::SeqCst) {
            return;
        }
        let mut rx = self.shutdown_tx.subscribe();
        let _ = rx.wait_for(|&shutting_down| shutting_down).await;
    }

    /// Cancels any active proxy request and terminates + reaps the worker subprocess.
    pub async fn shutdown_cleanly(&self) {
        self.request_shutdown();
        if let Ok(mut guard) =
            tokio::time::timeout(Duration::from_secs(2), self.worker.clone().lock_owned()).await
        {
            stop_worker_locked(&mut guard, &self.loaded).await;
        }
    }

    /// Explicitly evicts any running worker subprocess once exclusive access is acquired.
    pub async fn evict_now(&self) -> bool {
        let mut guard = self.worker.clone().lock_owned().await;
        stop_worker_locked(&mut guard, &self.loaded).await
    }

    fn try_acquire_admission(&self) -> Result<CoordinatorPermit> {
        let permit = self
            .admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| anyhow!("layerstream request queue is full"))?;
        self.activity.in_flight.fetch_add(1, Ordering::SeqCst);
        Ok(CoordinatorPermit {
            activity: self.activity.clone(),
            _permit: permit,
        })
    }

    pub fn spawn_reaper(&self) -> Option<tokio::task::JoinHandle<()>> {
        if self.config.idle_timeout.is_zero() {
            return None;
        }
        let state = self.clone();
        let tick = (self.config.idle_timeout / 2)
            .clamp(Duration::from_millis(200), Duration::from_secs(5));
        Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(tick);
            loop {
                interval.tick().await;
                if state.shutdown.load(Ordering::Relaxed) {
                    break;
                }
                if state.evict_idle_if_due(Instant::now()).await {
                    tracing::info!(
                        "terminated idle layerstream worker (0 MB VRAM/UMA and 0 MB GPU driver context)"
                    );
                }
            }
        }))
    }

    async fn evict_idle_if_due(&self, now: Instant) -> bool {
        let Ok(mut guard) = self.worker.clone().try_lock_owned() else {
            return false;
        };
        if guard.is_none() || self.activity.in_flight.load(Ordering::SeqCst) != 0 {
            return false;
        }
        let expired = self
            .activity
            .last_finished
            .lock()
            .map(|ts| now.saturating_duration_since(*ts) >= self.config.idle_timeout)
            .unwrap_or(false);
        if !expired {
            return false;
        }
        stop_worker_locked(&mut guard, &self.loaded).await
    }

    fn build_worker_spec(&self, token: String) -> Result<(WorkerSpec, u64)> {
        let generation = self.registry_generation.load(Ordering::SeqCst);
        let models: Vec<WorkerModelSpec> = self
            .registry
            .read()
            .map_err(|_| anyhow!("model registry lock poisoned"))?
            .list()
            .into_iter()
            .map(|entry| WorkerModelSpec {
                id: entry.id.clone(),
                path: entry.path.clone(),
                draft_path: entry.draft_path.clone(),
            })
            .collect();
        let spec = WorkerSpec {
            models,
            n_ctx: self.config.n_ctx,
            n_batch: self.config.n_batch,
            n_threads: self.config.n_threads,
            resident_layers: self.config.resident_layers,
            vram_budget_bytes: self.config.vram_budget_bytes,
            vram_budget_auto: self.config.vram_budget_auto,
            kv_cache_type: kv_cache_type_to_str(self.config.kv_cache_type).to_string(),
            use_gpu: self.config.use_gpu,
            prefer_direct_io: self.config.prefer_direct_io,
            cpu_embeddings: self.config.cpu_embeddings,
            cpu_head: self.config.cpu_head,
            stream_head: self.config.stream_head,
            moe_active_streaming: self.config.moe_active_streaming,
            enable_ngram_speculation: self.config.enable_ngram_speculation,
            max_draft_tokens: self.config.max_draft_tokens,
            max_queued_requests: self.config.max_queued_requests,
            worker_token: token,
        };
        Ok((spec, generation))
    }

    async fn ensure_worker(
        &self,
        guard: &mut OwnedMutexGuard<Option<RunningWorker>>,
    ) -> Result<(SocketAddr, String)> {
        let current_gen = self.registry_generation.load(Ordering::SeqCst);
        let need_spawn = match guard.as_mut() {
            Some(existing) => {
                if existing.is_alive() && existing.registry_generation == current_gen {
                    false
                } else {
                    stop_worker_locked(guard, &self.loaded).await;
                    true
                }
            }
            None => true,
        };

        if need_spawn {
            let worker = self.spawn_worker().await?;
            self.loaded.store(true, Ordering::SeqCst);
            **guard = Some(worker);
        }

        let running = guard.as_ref().expect("worker just initialized");
        Ok((running.addr, running.token.clone()))
    }

    async fn spawn_worker(&self) -> Result<RunningWorker> {
        let token = generate_worker_token()?;
        let (spec, generation) = self.build_worker_spec(token.clone())?;
        let mut spec_json = serde_json::to_string(&spec).context("serialize worker spec")?;
        spec_json.push('\n');

        let mut cmd = Command::new(self.worker_exe.as_ref());
        #[cfg(not(test))]
        cmd.arg("_worker");
        #[cfg(test)]
        {
            cmd.args(self.worker_args.as_ref());
            for (k, v) in self.worker_env.as_ref() {
                cmd.env(k, v);
            }
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);

        let mut child = cmd.spawn().with_context(|| {
            format!(
                "spawn isolated layerstream worker ({})",
                self.worker_exe.display()
            )
        })?;
        let mut stdin = child.stdin.take().context("capture worker stdin")?;
        let stdout = child.stdout.take().context("capture worker stdout")?;

        stdin
            .write_all(spec_json.as_bytes())
            .await
            .context("write spec to worker stdin")?;
        stdin.flush().await.context("flush worker stdin")?;

        let mut reader = BufReader::new(stdout);
        let mut ready_line = String::new();
        let read_res = tokio::time::timeout(WORKER_STARTUP_TIMEOUT, async {
            loop {
                ready_line.clear();
                let n = reader.read_line(&mut ready_line).await?;
                if n == 0 {
                    bail!("layerstream worker exited before becoming ready");
                }
                if let Some(url) = ready_line.trim().strip_prefix(WORKER_READY_PREFIX) {
                    let addr_str = url
                        .strip_prefix("http://")
                        .ok_or_else(|| anyhow!("unexpected worker ready URL: {url}"))?;
                    let addr: SocketAddr = addr_str
                        .parse()
                        .with_context(|| format!("parse worker ready socket addr '{addr_str}'"))?;
                    return Ok(addr);
                }
            }
        })
        .await;

        let addr = match read_res {
            Ok(Ok(addr)) => addr,
            Ok(Err(err)) => {
                drop(stdin);
                let _ = child.kill().await;
                let _ = child.wait().await;
                return Err(err);
            }
            Err(_) => {
                drop(stdin);
                let _ = child.kill().await;
                let _ = child.wait().await;
                bail!(
                    "layerstream worker did not become ready within {}s",
                    WORKER_STARTUP_TIMEOUT.as_secs()
                );
            }
        };

        // Keep draining worker stdout in the background so stray library prints never trigger SIGPIPE.
        tokio::spawn(async move {
            let mut line = String::new();
            while let Ok(n) = reader.read_line(&mut line).await {
                if n == 0 {
                    break;
                }
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    tracing::debug!(worker_stdout = %trimmed, "layerstream worker stdout");
                }
                line.clear();
            }
        });

        tracing::info!(
            pid = ?child.id(),
            addr = %addr,
            "spawned isolated layerstream GPU worker"
        );

        Ok(RunningWorker {
            child,
            stdin: Some(stdin),
            addr,
            token,
            registry_generation: generation,
        })
    }
}

async fn stop_worker_locked(
    guard: &mut OwnedMutexGuard<Option<RunningWorker>>,
    loaded: &AtomicBool,
) -> bool {
    let Some(worker) = guard.take() else {
        loaded.store(false, Ordering::SeqCst);
        return false;
    };
    let pid = worker.pid();
    worker.stop().await;
    loaded.store(false, Ordering::SeqCst);
    tracing::debug!(?pid, "reaped isolated layerstream GPU worker");
    true
}

pub fn build_coordinator_router(state: WorkerCoordinator) -> Router {
    Router::new()
        .route("/health", get(coord_health))
        .route("/v1/models", get(coord_list_models))
        .route("/v1/unload", post(coord_unload))
        .route("/v1/chat/completions", post(coord_proxy_inference))
        .route("/v1/completions", post(coord_proxy_inference))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .with_state(state)
}

pub fn build_coordinator_router_with_api_key(
    state: WorkerCoordinator,
    api_key: Option<String>,
) -> Router {
    with_api_key_auth(build_coordinator_router(state), api_key)
}

pub async fn serve_with_coordinator(
    listener: TcpListener,
    coordinator: WorkerCoordinator,
) -> Result<()> {
    serve_with_coordinator_and_api_key(listener, coordinator, None).await
}

pub async fn serve_with_coordinator_and_api_key(
    listener: TcpListener,
    coordinator: WorkerCoordinator,
    api_key: Option<String>,
) -> Result<()> {
    let _reaper = DropAbort(coordinator.spawn_reaper());
    let app = build_coordinator_router_with_api_key(coordinator, api_key);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn coord_health(State(state): State<WorkerCoordinator>) -> impl IntoResponse {
    if let Some(err) = state.loading_error() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "status": "error",
                "error": err,
                "models": state.model_count(),
                "loaded": state.is_loaded(),
                "worker_isolated": true,
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
                "worker_isolated": true,
            })),
        )
            .into_response();
    }
    Json(json!({
        "status": "ok",
        "models": state.model_count(),
        "loaded": state.is_loaded(),
        "worker_isolated": true,
    }))
    .into_response()
}

async fn coord_list_models(State(state): State<WorkerCoordinator>) -> impl IntoResponse {
    let payload = state
        .registry
        .read()
        .map(|r| models_list_json(&r))
        .unwrap_or_else(|_| json!({ "object": "list", "data": [] }));
    Json(payload)
}

async fn coord_unload(State(state): State<WorkerCoordinator>) -> impl IntoResponse {
    let unloaded = state.evict_now().await;
    if unloaded {
        tracing::info!(
            "unloaded isolated layerstream worker via /v1/unload (0 MB VRAM/UMA and 0 MB GPU driver context)"
        );
    }
    Json(json!({
        "unloaded": unloaded,
        "loaded": state.is_loaded(),
        "worker_isolated": true,
    }))
}

/// Guard that ensures an abnormally cancelled non-streaming request still reaps the
/// worker when `idle_timeout == 0` instead of leaving a warm GPU worker behind.
struct EvictOnAbnormalDrop {
    guard: Option<OwnedMutexGuard<Option<RunningWorker>>>,
    loaded: Arc<AtomicBool>,
    activity: Arc<CoordinatorActivity>,
    evict_immediately: bool,
}

impl EvictOnAbnormalDrop {
    fn take(mut self) -> OwnedMutexGuard<Option<RunningWorker>> {
        self.guard.take().expect("guard present")
    }
}

impl Drop for EvictOnAbnormalDrop {
    fn drop(&mut self) {
        if let Some(mut guard) = self.guard.take()
            && self.evict_immediately
            && self.activity.in_flight.load(Ordering::SeqCst) <= 1
        {
            let loaded = self.loaded.clone();
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    stop_worker_locked(&mut guard, &loaded).await;
                });
            }
        }
    }
}

async fn coord_proxy_inference(
    State(state): State<WorkerCoordinator>,
    req: Request<Body>,
) -> Response {
    if let Some(err) = state.loading_error() {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("background model download failed: {err}"),
        );
    }
    if let Some(status) = state.loading_status() {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("model is still downloading: {status}"),
        );
    }

    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| "/v1/chat/completions".to_string());
    let content_type = req
        .headers()
        .get(header::CONTENT_TYPE)
        .cloned()
        .unwrap_or_else(|| HeaderValue::from_static("application/json"));

    let body_bytes = match axum::body::to_bytes(req.into_body(), MAX_REQUEST_BODY_BYTES).await {
        Ok(b) => b,
        Err(err) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("failed to read request body: {err}"),
            );
        }
    };

    // Validate JSON & model existence in the parent before waking any GPU worker.
    let parsed: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(err) => {
            return api_error(StatusCode::BAD_REQUEST, format!("invalid JSON body: {err}"));
        }
    };
    let requested_model = parsed.get("model").and_then(Value::as_str);
    {
        let Ok(registry) = state.registry.read() else {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "model registry lock poisoned",
            );
        };
        if let Some(id) = requested_model {
            if registry.get(id).is_none() {
                return api_error(
                    StatusCode::NOT_FOUND,
                    format!("model '{id}' not found in layerstream registry"),
                );
            }
        } else if registry.first().is_none() {
            return api_error(StatusCode::NOT_FOUND, "no models configured in layerstream");
        }
    }

    // Acquire queue admission synchronously before waiting on worker execution so queue
    // saturation always returns 503 Service Unavailable immediately.
    let permit = match state.try_acquire_admission() {
        Ok(p) => p,
        Err(err) => return api_error(StatusCode::SERVICE_UNAVAILABLE, err.to_string()),
    };

    let worker_guard = tokio::select! {
        biased;
        _ = state.wait_shutdown() => {
            return api_error(StatusCode::SERVICE_UNAVAILABLE, "layerstream shutting down");
        }
        guard = state.worker.clone().lock_owned() => guard,
    };

    if state.shutdown.load(Ordering::SeqCst) {
        return api_error(StatusCode::SERVICE_UNAVAILABLE, "layerstream shutting down");
    }

    let evict_immediately = state.config.idle_timeout.is_zero();
    let mut guard_holder = EvictOnAbnormalDrop {
        guard: Some(worker_guard),
        loaded: state.loaded.clone(),
        activity: state.activity.clone(),
        evict_immediately,
    };

    let (worker_addr, worker_token) = tokio::select! {
        biased;
        _ = state.wait_shutdown() => {
            return api_error(StatusCode::SERVICE_UNAVAILABLE, "layerstream shutting down");
        }
        res = state.ensure_worker(guard_holder.guard.as_mut().expect("guard present")) => match res {
            Ok(v) => v,
            Err(err) => {
                tracing::error!("failed to start layerstream worker: {err:#}");
                return api_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("failed to start layerstream worker: {err:#}"),
                );
            }
        }
    };

    let tcp = match TcpStream::connect(worker_addr).await {
        Ok(s) => s,
        Err(err) => {
            tracing::error!(addr = %worker_addr, "failed to connect to layerstream worker: {err}");
            stop_worker_locked(
                guard_holder.guard.as_mut().expect("guard present"),
                &state.loaded,
            )
            .await;
            return api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("failed to connect to layerstream worker: {err}"),
            );
        }
    };

    let (mut sender, conn) = match http1::handshake(TokioIo::new(tcp)).await {
        Ok(pair) => pair,
        Err(err) => {
            return api_error(
                StatusCode::BAD_GATEWAY,
                format!("worker HTTP handshake failed: {err}"),
            );
        }
    };

    let conn_driver = DropAbort(Some(tokio::spawn(async move {
        let _ = conn.await;
    })));

    let upstream_req = match http::Request::builder()
        .method(http::Method::POST)
        .uri(&path_and_query)
        .header(header::HOST, format!("127.0.0.1:{}", worker_addr.port()))
        .header(header::CONTENT_TYPE, content_type)
        .header(WORKER_TOKEN_HEADER, worker_token)
        .body(Full::new(body_bytes))
    {
        Ok(r) => r,
        Err(err) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to build worker request: {err}"),
            );
        }
    };

    let upstream_resp = tokio::select! {
        biased;
        _ = state.wait_shutdown() => {
            return api_error(StatusCode::SERVICE_UNAVAILABLE, "layerstream shutting down");
        }
        res = sender.send_request(upstream_req) => match res {
            Ok(r) => r,
            Err(err) => {
                tracing::error!("worker request failed: {err}");
                stop_worker_locked(
                    guard_holder.guard.as_mut().expect("guard present"),
                    &state.loaded,
                )
                .await;
                return api_error(
                    StatusCode::BAD_GATEWAY,
                    format!("worker request failed: {err}"),
                );
            }
        }
    };

    let status = upstream_resp.status();
    let resp_ct = upstream_resp
        .headers()
        .get(header::CONTENT_TYPE)
        .cloned()
        .unwrap_or_else(|| HeaderValue::from_static("application/json"));
    let is_sse = resp_ct
        .to_str()
        .map(|v| v.to_ascii_lowercase().starts_with("text/event-stream"))
        .unwrap_or(false);

    if is_sse {
        let mut worker_guard = guard_holder.take();
        let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(32);
        let state_bg = state.clone();
        tokio::spawn(async move {
            let _permit = permit;
            let _driver = conn_driver;
            let mut body = upstream_resp.into_body();
            let mut client_disconnected = false;
            let mut shutdown_hit = false;

            loop {
                tokio::select! {
                    biased;
                    _ = state_bg.wait_shutdown() => {
                        shutdown_hit = true;
                        break;
                    }
                    _ = tx.closed() => {
                        client_disconnected = true;
                        break;
                    }
                    next = body.frame() => {
                        match next {
                            Some(Ok(frame)) => {
                                if let Ok(data) = frame.into_data()
                                    && !data.is_empty()
                                {
                                    let send_res = tokio::select! {
                                        biased;
                                        _ = state_bg.wait_shutdown() => {
                                            shutdown_hit = true;
                                            break;
                                        }
                                        res = tx.send(Ok(data)) => res,
                                    };
                                    if send_res.is_err() {
                                        client_disconnected = true;
                                        break;
                                    }
                                }
                            }
                            Some(Err(err)) => {
                                let _ = tx.send(Err(std::io::Error::other(err.to_string()))).await;
                                break;
                            }
                            None => break,
                        }
                    }
                }
            }

            // Close upstream connection before tearing down worker.
            drop(body);
            drop(_driver);

            // For `--idle-timeout-secs 0` (when no other request is queued behind us),
            // synchronously reap the worker BEFORE dropping `tx` (HTTP chunked EOF) so
            // callers like Mesh-LLM `exclusive_gpu` (tracking: Mesh-LLM/mesh-llm#3174,
            // upstream: Mesh-LLM/mesh-llm#3195) see 0 MB VRAM the instant the stream ends.
            if (evict_immediately && state_bg.activity.in_flight.load(Ordering::SeqCst) <= 1)
                || shutdown_hit
            {
                stop_worker_locked(&mut worker_guard, &state_bg.loaded).await;
            } else if client_disconnected {
                tracing::debug!("SSE client disconnected; worker generation cancelled");
            }
            drop(worker_guard);
            drop(_permit);
            drop(tx);
        });

        let stream = ReceiverStream::new(rx);
        return Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, resp_ct)
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from_stream(stream))
            .unwrap_or_else(|_| {
                api_error(StatusCode::INTERNAL_SERVER_ERROR, "response build error")
            });
    }

    // Non-streaming response: buffer full response, reap worker if idle_timeout == 0,
    // then return the HTTP response so VRAM is guaranteed free before client receives it.
    let collected = tokio::select! {
        biased;
        _ = state.wait_shutdown() => {
            return api_error(StatusCode::SERVICE_UNAVAILABLE, "layerstream shutting down");
        }
        res = upstream_resp.into_body().collect() => match res {
            Ok(c) => c.to_bytes(),
            Err(err) => {
                stop_worker_locked(
                    guard_holder.guard.as_mut().expect("guard present"),
                    &state.loaded,
                )
                .await;
                return api_error(
                    StatusCode::BAD_GATEWAY,
                    format!("failed reading worker response: {err}"),
                );
            }
        }
    };

    drop(conn_driver);
    let mut worker_guard = guard_holder.take();
    if evict_immediately && state.activity.in_flight.load(Ordering::SeqCst) <= 1 {
        stop_worker_locked(&mut worker_guard, &state.loaded).await;
    }
    drop(worker_guard);
    drop(permit);

    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, resp_ct)
        .body(Body::from(collected))
        .unwrap_or_else(|_| api_error(StatusCode::INTERNAL_SERVER_ERROR, "response build error"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use layerstream_core::ModelMeta;
    use layerstream_server::ModelEntry;
    use std::convert::Infallible;

    const MOCK_WORKER_ENV: &str = "LAYERSTREAM_MOCK_WORKER_CHILD";

    fn mock_registry() -> ModelRegistry {
        let mut reg = ModelRegistry::default();
        reg.register_entry(ModelEntry {
            id: "mock-14b".to_string(),
            path: PathBuf::from("/nonexistent/mock-14b.gguf"),
            draft_path: None,
            meta: ModelMeta {
                architecture: "qwen2".to_string(),
                name: Some("mock-14b".to_string()),
                n_layers: 48,
                context_length: Some(32768),
                total_file_bytes: 8 * 1024 * 1024 * 1024,
            },
        });
        reg
    }

    fn mock_coordinator(idle_timeout: Duration, max_queued: usize) -> WorkerCoordinator {
        let cfg = ServerConfig {
            registry: mock_registry(),
            n_ctx: 512,
            n_batch: 128,
            n_threads: 2,
            resident_layers: Some(0),
            vram_budget_bytes: 0,
            vram_budget_auto: false,
            kv_cache_type: KvCacheType::Q8_0,
            use_gpu: false,
            prefer_direct_io: true,
            cpu_embeddings: false,
            cpu_head: false,
            stream_head: false,
            moe_active_streaming: true,
            idle_timeout,
            enable_ngram_speculation: true,
            max_draft_tokens: 4,
            max_queued_requests: max_queued,
        };
        let exe = std::env::current_exe().expect("test exe");
        let mut coord = WorkerCoordinator::with_worker_exe(cfg, exe);
        coord.worker_args = Arc::new(vec![
            "--exact".to_string(),
            "worker::tests::mock_worker_child_entry".to_string(),
            "--nocapture".to_string(),
        ]);
        coord.worker_env = Arc::new(vec![(MOCK_WORKER_ENV.to_string(), "1".to_string())]);
        coord
    }

    #[cfg(unix)]
    fn pid_is_alive(pid: u32) -> bool {
        // SAFETY: `kill(pid, 0)` only probes process existence without sending a signal.
        unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
    }

    async fn http_post_raw(addr: SocketAddr, path: &str, body: &str) -> Result<String> {
        let mut s = TcpStream::connect(addr).await?;
        let req = format!(
            "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        s.write_all(req.as_bytes()).await?;
        let mut out = String::new();
        s.read_to_string(&mut out).await?;
        Ok(out)
    }

    async fn http_get_raw(addr: SocketAddr, path: &str) -> Result<String> {
        let mut s = TcpStream::connect(addr).await?;
        let req = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
        s.write_all(req.as_bytes()).await?;
        let mut out = String::new();
        s.read_to_string(&mut out).await?;
        Ok(out)
    }

    fn split_http_body(raw: &str) -> &str {
        raw.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("")
    }

    /// Dual-purpose test function: returns immediately during normal `cargo test`,
    /// and acts as a real subprocess mock worker when `LAYERSTREAM_MOCK_WORKER_CHILD=1`.
    #[test]
    fn mock_worker_child_entry() {
        if std::env::var(MOCK_WORKER_ENV).as_deref() != Ok("1") {
            return;
        }
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("mock worker rt");
        rt.block_on(async {
            let mut stdin_reader = BufReader::new(tokio::io::stdin());
            let mut line = String::new();
            let n = stdin_reader.read_line(&mut line).await.expect("read spec");
            assert!(n > 0, "empty spec");
            let spec: WorkerSpec = serde_json::from_str(line.trim()).expect("parse spec");
            let expected = Arc::new(spec.worker_token.into_bytes());
            let pid = std::process::id();

            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
            let addr = listener.local_addr().expect("addr");

            let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
            let shutdown_tx = Arc::new(StdMutex::new(Some(shutdown_tx)));
            let tx_eof = shutdown_tx.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 32];
                loop {
                    match stdin_reader.read(&mut buf).await {
                        Ok(0) | Err(_) => {
                            if let Ok(mut g) = tx_eof.lock()
                                && let Some(tx) = g.take()
                            {
                                let _ = tx.send(());
                            }
                            break;
                        }
                        Ok(_) => {}
                    }
                }
            });

            let app = Router::new()
                .route(
                    "/v1/chat/completions",
                    post(move |Json(body): Json<Value>| async move {
                        let delay_ms = body.get("delay_ms").and_then(Value::as_u64).unwrap_or(0);
                        if delay_ms > 0 {
                            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                        }
                        if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
                            let (tx, rx) = mpsc::unbounded_channel::<
                                std::result::Result<axum::response::sse::Event, Infallible>,
                            >();
                            tokio::spawn(async move {
                                let chunk = json!({
                                    "worker_pid": pid,
                                    "choices": [{"delta": {"content": "hello"}}]
                                });
                                let _ = tx
                                    .send(Ok(axum::response::sse::Event::default()
                                        .data(chunk.to_string())));
                                let _ = tx
                                    .send(Ok(axum::response::sse::Event::default().data("[DONE]")));
                            });
                            return axum::response::Sse::new(
                                tokio_stream::wrappers::UnboundedReceiverStream::new(rx),
                            )
                            .into_response();
                        }
                        Json(json!({
                            "ok": true,
                            "worker_pid": pid,
                            "model": body.get("model").cloned().unwrap_or(Value::Null),
                        }))
                        .into_response()
                    }),
                )
                .layer(axum::middleware::from_fn(
                    move |req: axum::extract::Request, next: axum::middleware::Next| {
                        let expected = expected.clone();
                        async move {
                            let ok = req
                                .headers()
                                .get(WORKER_TOKEN_HEADER)
                                .map(|v| v.as_bytes() == expected.as_slice())
                                .unwrap_or(false);
                            if !ok {
                                return api_error(StatusCode::FORBIDDEN, "bad token");
                            }
                            next.run(req).await
                        }
                    },
                ));

            let mut stdout = tokio::io::stdout();
            stdout
                .write_all(format!("{WORKER_READY_PREFIX}http://{addr}\n").as_bytes())
                .await
                .expect("write ready");
            stdout.flush().await.expect("flush ready");

            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await;
        });
        std::process::exit(0);
    }

    async fn spawn_coord_server(
        coord: WorkerCoordinator,
    ) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let _ = serve_with_coordinator(listener, coord).await;
        });
        (addr, handle)
    }

    #[tokio::test]
    async fn immediate_evict_reaps_worker_before_json_response_completes() -> Result<()> {
        let coord = mock_coordinator(Duration::ZERO, 4);
        let (addr, server) = spawn_coord_server(coord.clone()).await;

        assert!(!coord.is_loaded());
        let resp = http_post_raw(
            addr,
            "/v1/chat/completions",
            r#"{"model":"mock-14b","messages":[{"role":"user","content":"hi"}]}"#,
        )
        .await?;
        assert!(resp.starts_with("HTTP/1.1 200"), "{resp}");
        let body: Value = serde_json::from_str(split_http_body(&resp))?;
        let pid = body["worker_pid"].as_u64().context("worker_pid")? as u32;

        // By the time the HTTP 200 response completes, the worker MUST ALREADY be reaped.
        assert!(
            !coord.is_loaded(),
            "worker still marked loaded after JSON response"
        );
        #[cfg(unix)]
        assert!(
            !pid_is_alive(pid),
            "worker pid {pid} still alive after idle_timeout=0 JSON response"
        );

        coord.shutdown_cleanly().await;
        server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn immediate_evict_reaps_worker_before_sse_eof() -> Result<()> {
        let coord = mock_coordinator(Duration::ZERO, 4);
        let (addr, server) = spawn_coord_server(coord.clone()).await;

        let resp = http_post_raw(
            addr,
            "/v1/chat/completions",
            r#"{"model":"mock-14b","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .await?;
        assert!(resp.starts_with("HTTP/1.1 200"), "{resp}");
        assert!(resp.contains("text/event-stream"), "{resp}");
        assert!(resp.contains("data: [DONE]"), "{resp}");

        let pid = resp
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter_map(|d| serde_json::from_str::<Value>(d).ok())
            .find_map(|v| v["worker_pid"].as_u64())
            .context("worker_pid in SSE")? as u32;

        // By the time HTTP chunked EOF arrives, the worker MUST ALREADY be dead & reaped.
        assert!(
            !coord.is_loaded(),
            "worker still marked loaded after SSE EOF"
        );
        #[cfg(unix)]
        assert!(
            !pid_is_alive(pid),
            "worker pid {pid} still alive after idle_timeout=0 SSE EOF"
        );

        coord.shutdown_cleanly().await;
        server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn warm_worker_reused_then_reaped_by_idle_reaper() -> Result<()> {
        let coord = mock_coordinator(Duration::from_millis(600), 4);
        let (addr, server) = spawn_coord_server(coord.clone()).await;

        // Back-to-back requests reuse the same warm worker PID.
        let r1 = http_post_raw(
            addr,
            "/v1/chat/completions",
            r#"{"model":"mock-14b","messages":[{"role":"user","content":"1"}]}"#,
        )
        .await?;
        let pid1 = serde_json::from_str::<Value>(split_http_body(&r1))?["worker_pid"]
            .as_u64()
            .context("pid1")? as u32;
        assert!(coord.is_loaded());

        let r2 = http_post_raw(
            addr,
            "/v1/chat/completions",
            r#"{"model":"mock-14b","messages":[{"role":"user","content":"2"}]}"#,
        )
        .await?;
        let pid2 = serde_json::from_str::<Value>(split_http_body(&r2))?["worker_pid"]
            .as_u64()
            .context("pid2")? as u32;
        assert_eq!(
            pid1, pid2,
            "warm worker should be reused within idle_timeout"
        );

        // Wait for idle reaper to terminate & reap the worker.
        tokio::time::sleep(Duration::from_millis(1400)).await;
        assert!(!coord.is_loaded(), "idle reaper did not evict worker");
        #[cfg(unix)]
        assert!(!pid_is_alive(pid1), "reaped worker pid {pid1} still alive");

        // Next request cleanly spawns a fresh worker PID.
        let r3 = http_post_raw(
            addr,
            "/v1/chat/completions",
            r#"{"model":"mock-14b","messages":[{"role":"user","content":"3"}]}"#,
        )
        .await?;
        let pid3 = serde_json::from_str::<Value>(split_http_body(&r3))?["worker_pid"]
            .as_u64()
            .context("pid3")? as u32;
        assert_ne!(pid1, pid3, "fresh worker should spawn after idle eviction");

        coord.shutdown_cleanly().await;
        server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn explicit_unload_endpoint_terminates_warm_worker() -> Result<()> {
        let coord = mock_coordinator(Duration::from_secs(60), 4);
        let (addr, server) = spawn_coord_server(coord.clone()).await;

        // Idle /v1/unload reports unloaded=false without spawning anything.
        let u0 = http_post_raw(addr, "/v1/unload", "").await?;
        assert!(u0.starts_with("HTTP/1.1 200"), "{u0}");
        let v0: Value = serde_json::from_str(split_http_body(&u0))?;
        assert_eq!(v0["unloaded"], false);
        assert_eq!(v0["worker_isolated"], true);

        // Inference spawns worker.
        let r1 = http_post_raw(
            addr,
            "/v1/chat/completions",
            r#"{"model":"mock-14b","messages":[{"role":"user","content":"hi"}]}"#,
        )
        .await?;
        let pid = serde_json::from_str::<Value>(split_http_body(&r1))?["worker_pid"]
            .as_u64()
            .context("pid")? as u32;
        let h1 = http_get_raw(addr, "/health").await?;
        let hv1: Value = serde_json::from_str(split_http_body(&h1))?;
        assert_eq!(hv1["loaded"], true);

        // Explicit /v1/unload kills & reaps worker immediately.
        let u1 = http_post_raw(addr, "/v1/unload", "").await?;
        let v1: Value = serde_json::from_str(split_http_body(&u1))?;
        assert_eq!(v1["unloaded"], true);
        assert_eq!(v1["loaded"], false);
        #[cfg(unix)]
        assert!(
            !pid_is_alive(pid),
            "worker pid {pid} still alive after /v1/unload"
        );

        coord.shutdown_cleanly().await;
        server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn queue_full_returns_503_synchronously() -> Result<()> {
        let coord = mock_coordinator(Duration::from_secs(60), 2);
        let (addr, server) = spawn_coord_server(coord.clone()).await;

        let slow_body =
            r#"{"model":"mock-14b","delay_ms":800,"messages":[{"role":"user","content":"slow"}]}"#;
        let t1 = tokio::spawn(http_post_raw(addr, "/v1/chat/completions", slow_body));
        let t2 = tokio::spawn(http_post_raw(addr, "/v1/chat/completions", slow_body));
        tokio::time::sleep(Duration::from_millis(200)).await;

        let r3 = http_post_raw(
            addr,
            "/v1/chat/completions",
            r#"{"model":"mock-14b","messages":[{"role":"user","content":"overflow"}]}"#,
        )
        .await?;
        assert!(
            r3.starts_with("HTTP/1.1 503"),
            "expected 503 when queue is full, got: {}",
            r3.lines().next().unwrap_or("")
        );

        let _ = t1.await;
        let _ = t2.await;
        coord.shutdown_cleanly().await;
        server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn coordinator_api_key_enforces_bearer_auth_while_keeping_worker_isolated() -> Result<()>
    {
        let coord = mock_coordinator(Duration::from_secs(60), 2);
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let server_coord = coord.clone();
        let server = tokio::spawn(async move {
            let _ = serve_with_coordinator_and_api_key(
                listener,
                server_coord,
                Some("secret-token".to_string()),
            )
            .await;
        });

        async fn send_get(addr: SocketAddr, path: &str, auth: Option<&str>) -> Result<String> {
            let mut stream = TcpStream::connect(addr).await?;
            let auth_hdr = auth
                .map(|v| format!("Authorization: {v}\r\n"))
                .unwrap_or_default();
            let req = format!(
                "GET {path} HTTP/1.1\r\nHost: localhost\r\n{auth_hdr}Connection: close\r\n\r\n"
            );
            stream.write_all(req.as_bytes()).await?;
            let mut buf = Vec::new();
            stream.read_to_end(&mut buf).await?;
            Ok(String::from_utf8_lossy(&buf).into_owned())
        }

        let health = send_get(addr, "/health", None).await?;
        assert!(health.starts_with("HTTP/1.1 200"), "{health}");
        let health_json: Value = serde_json::from_str(split_http_body(&health))?;
        assert_eq!(health_json["worker_isolated"], true);

        let unauth = send_get(addr, "/v1/models", None).await?;
        assert!(unauth.starts_with("HTTP/1.1 401"), "{unauth}");

        let wrong = send_get(addr, "/v1/models", Some("Bearer wrong-token")).await?;
        assert!(wrong.starts_with("HTTP/1.1 401"), "{wrong}");

        let ok = send_get(addr, "/v1/models", Some("Bearer secret-token")).await?;
        assert!(ok.starts_with("HTTP/1.1 200"), "{ok}");

        coord.shutdown_cleanly().await;
        server.abort();
        Ok(())
    }
}
