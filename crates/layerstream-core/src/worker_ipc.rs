//! Unified isolated-worker IPC client and server shared across the Python and
//! Node.js bindings.
//!
//! Running `LoadedSession` inside a dedicated child process guarantees the
//! parent process (e.g., PyTorch, Jupyter, FastAPI, Node) never initializes
//! CUDA/Vulkan/Metal drivers. Calling `WorkerClient::unload()` (or dropping the
//! client) terminates and reaps the child so the OS kernel immediately reclaims
//! 100% of VRAM/UMA, CUDA driver contexts, and C++ heap arenas.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::chat_parse::{
    IncomingChatMessage, ParsedAssistantOutput, inject_tools_into_messages, parse_assistant_output,
};
use crate::engine::{GenerateOptions, GenerateSummary, KvCacheType, PromptTooLongError};
use crate::ffi::LsSampleParams;
use crate::hf::{DownloadProgress, ProgressCallback};
use crate::sdk::{EngineStatsSnapshot, LoadedSession, OpenSessionOptions};

/// Error returned across the worker IPC boundary with a structured error kind
/// so Python and Node bindings can map it to native exception types.
#[derive(Debug, Clone)]
pub struct WorkerError {
    pub kind: String,
    pub message: String,
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for WorkerError {}

impl WorkerError {
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            message: message.into(),
        }
    }
}

/// Serializable model-open parameters sent from parent to isolated worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerOpenParams {
    pub model: String,
    #[serde(default)]
    pub draft_model: Option<String>,
    #[serde(default)]
    pub quant: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
    #[serde(default = "default_true")]
    pub vram_budget_auto: bool,
    #[serde(default)]
    pub vram_budget_bytes: u64,
    #[serde(default)]
    pub resident_layers: Option<u32>,
    #[serde(default = "default_ctx_size")]
    pub ctx_size: u32,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default)]
    pub threads: Option<u32>,
    #[serde(default)]
    pub kv_cache_type: KvCacheType,
    #[serde(default = "default_true")]
    pub use_gpu: bool,
    #[serde(default = "default_true")]
    pub prefer_direct_io: bool,
    #[serde(default)]
    pub cpu_embeddings: bool,
    #[serde(default)]
    pub cpu_head: bool,
    #[serde(default)]
    pub stream_head: bool,
    #[serde(default = "default_true")]
    pub moe_active_streaming: bool,
    #[serde(default = "default_max_draft_tokens")]
    pub max_draft_tokens: usize,
    #[serde(default)]
    pub verbose: bool,
    #[serde(default)]
    pub stream_progress: bool,
}

fn default_true() -> bool {
    true
}
fn default_ctx_size() -> u32 {
    4096
}
fn default_batch_size() -> u32 {
    512
}
fn default_max_draft_tokens() -> usize {
    5
}
fn default_max_tokens() -> usize {
    256
}
fn default_top_p() -> f32 {
    0.95
}
fn default_top_k() -> i32 {
    40
}
fn default_repeat_penalty() -> f32 {
    1.0
}
fn default_penalty_last_n() -> i32 {
    64
}
fn default_seed() -> u32 {
    42
}

/// Serializable generation / chat parameters sent from parent to isolated worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerGenParams {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub min_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default = "default_penalty_last_n")]
    pub penalty_last_n: i32,
    #[serde(default = "default_seed")]
    pub seed: u32,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default = "default_true")]
    pub add_generation_prompt: bool,
    #[serde(default = "default_true")]
    pub ngram_speculation: bool,
    #[serde(default)]
    pub max_draft_tokens: Option<usize>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub grammar: Option<String>,
    #[serde(default)]
    pub parse_output: bool,
}

impl Default for WorkerGenParams {
    fn default() -> Self {
        Self {
            max_tokens: default_max_tokens(),
            temperature: 0.0,
            top_p: default_top_p(),
            min_p: 0.0,
            top_k: default_top_k(),
            repeat_penalty: default_repeat_penalty(),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            penalty_last_n: default_penalty_last_n(),
            seed: default_seed(),
            stop: Vec::new(),
            add_generation_prompt: true,
            ngram_speculation: true,
            max_draft_tokens: None,
            tools: None,
            grammar: None,
            parse_output: false,
        }
    }
}

impl WorkerGenParams {
    fn to_generate_options(&self, default_max_draft: usize) -> GenerateOptions {
        GenerateOptions {
            max_tokens: self.max_tokens,
            sample: LsSampleParams {
                temperature: self.temperature,
                top_p: self.top_p,
                min_p: self.min_p,
                top_k: self.top_k,
                repeat_penalty: self.repeat_penalty,
                frequency_penalty: self.frequency_penalty,
                presence_penalty: self.presence_penalty,
                penalty_last_n: self.penalty_last_n,
                seed: self.seed,
            },
            stop: self.stop.clone(),
            enable_ngram_speculation: self.ngram_speculation,
            max_draft_tokens: self.max_draft_tokens.unwrap_or(default_max_draft).max(1),
            grammar: self.grammar.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum WorkerStreamEvent {
    Chunk(String),
    Done(Option<GenerateSummary>),
    Error(WorkerError),
}

fn send_json_line(stream: &mut TcpStream, msg: &Value) -> std::io::Result<()> {
    let mut line = serde_json::to_vec(msg).map_err(std::io::Error::other)?;
    line.push(b'\n');
    stream.write_all(&line)?;
    stream.flush()
}

fn generate_auth_token() -> Result<String> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).map_err(|e| anyhow!("getrandom failed: {e}"))?;
    let mut hex = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{b:02x}");
    }
    Ok(hex)
}

fn classify_error(err: &anyhow::Error) -> (&'static str, String) {
    let msg = format!("{err:#}");
    if err.downcast_ref::<PromptTooLongError>().is_some() {
        return ("ValueError", msg);
    }
    for cause in err.chain() {
        if let Some(io_err) = cause.downcast_ref::<std::io::Error>()
            && io_err.kind() == std::io::ErrorKind::NotFound
        {
            return ("FileNotFoundError", msg);
        }
    }
    if msg.contains("not found") || msg.contains("No such file") {
        ("FileNotFoundError", msg)
    } else if msg.contains("invalid ") || msg.contains("must be ") || msg.contains("unknown ") {
        ("ValueError", msg)
    } else {
        ("RuntimeError", msg)
    }
}

struct WorkerState {
    child: Option<Child>,
    child_stdin: Option<ChildStdin>,
    sock_writer: Option<TcpStream>,
    reader_thread: Option<JoinHandle<()>>,
    loaded: bool,
    model_path: Option<String>,
    last_summary: Option<GenerateSummary>,
    next_id: u64,
    pending_rpc: HashMap<u64, mpsc::Sender<Value>>,
    active_streams: HashMap<u64, mpsc::Sender<WorkerStreamEvent>>,
}

impl WorkerState {
    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn is_child_alive(&mut self) -> bool {
        let Some(child) = self.child.as_mut() else {
            return false;
        };
        matches!(child.try_wait(), Ok(None))
    }

    fn child_pid(&mut self) -> Option<u32> {
        if self.is_child_alive() {
            self.child.as_ref().map(Child::id)
        } else {
            None
        }
    }
}

/// Parent-side handle to an isolated worker subprocess.
pub struct WorkerClient {
    state: Arc<Mutex<WorkerState>>,
    send_lock: Arc<Mutex<()>>,
}

impl WorkerClient {
    /// Spawn a worker subprocess running `program args... --worker-port <port>`,
    /// perform the loopback TCP token handshake, and open the model.
    pub fn spawn(
        program: &str,
        args: &[String],
        open: WorkerOpenParams,
        on_progress: Option<ProgressCallback>,
    ) -> Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .context("failed to bind loopback TCP listener for isolated worker")?;
        let port = listener.local_addr()?.port();
        listener
            .set_nonblocking(true)
            .context("failed to set worker listener nonblocking")?;

        let token = generate_auth_token()?;

        let mut cmd = Command::new(program);
        cmd.args(args)
            .arg("--worker-port")
            .arg(port.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to spawn isolated worker '{program}'"))?;

        let mut child_stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("isolated worker stdin pipe missing"))?;
        writeln!(child_stdin, "{token}").context("failed to write auth token to worker stdin")?;
        child_stdin
            .flush()
            .context("failed to flush auth token to worker stdin")?;

        // Accept TCP connection with a 30s deadline while watching for early child exit.
        let deadline = Instant::now() + Duration::from_secs(30);
        let stream = loop {
            match listener.accept() {
                Ok((s, _addr)) => break s,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if let Ok(Some(status)) = child.try_wait() {
                        bail!("isolated worker exited during startup (status: {status})");
                    }
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        bail!(
                            "timed out waiting for isolated worker to connect on 127.0.0.1:{port}"
                        );
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(e).context("accepting isolated worker connection failed");
                }
            }
        };
        drop(listener);

        stream.set_nonblocking(false)?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(Duration::from_secs(30)))?;

        let mut reader = BufReader::new(stream.try_clone()?);
        let mut hello_line = String::new();
        reader
            .read_line(&mut hello_line)
            .context("failed to read worker handshake line")?;
        let hello: Value = serde_json::from_str(&hello_line)
            .context("invalid JSON in isolated worker handshake")?;
        if hello.get("token").and_then(Value::as_str) != Some(token.as_str()) {
            let _ = child.kill();
            let _ = child.wait();
            bail!("isolated worker auth token mismatch");
        }
        reader.get_ref().set_read_timeout(None)?;

        let state = Arc::new(Mutex::new(WorkerState {
            child: Some(child),
            child_stdin: Some(child_stdin),
            sock_writer: Some(stream),
            reader_thread: None,
            loaded: true,
            model_path: None,
            last_summary: None,
            next_id: 1,
            pending_rpc: HashMap::new(),
            active_streams: HashMap::new(),
        }));

        let reader_state = Arc::clone(&state);
        let reader_thread = thread::Builder::new()
            .name("layerstream-worker-reader".to_string())
            .spawn(move || reader_loop(reader, reader_state))?;

        {
            let mut st = state.lock().expect("worker state lock poisoned");
            st.reader_thread = Some(reader_thread);
        }

        let client = Self {
            state,
            send_lock: Arc::new(Mutex::new(())),
        };

        let mut open_params = open;
        open_params.stream_progress = on_progress.is_some();
        let open_val = serde_json::to_value(&open_params)?;

        match client.rpc_internal("open", open_val, on_progress) {
            Ok((res, _)) => {
                let model_path = res
                    .get("model_path")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
                {
                    let mut st = client.state.lock().expect("worker state lock poisoned");
                    st.model_path = model_path;
                }
                Ok(client)
            }
            Err(err) => {
                client.unload();
                Err(err)
            }
        }
    }

    pub fn is_loaded(&self) -> bool {
        let mut st = self.state.lock().expect("worker state lock poisoned");
        st.loaded && st.is_child_alive()
    }

    pub fn worker_pid(&self) -> Option<u32> {
        let mut st = self.state.lock().expect("worker state lock poisoned");
        if !st.loaded {
            return None;
        }
        st.child_pid()
    }

    pub fn model_path(&self) -> Option<String> {
        let st = self.state.lock().expect("worker state lock poisoned");
        if st.loaded {
            st.model_path.clone()
        } else {
            None
        }
    }

    pub fn last_summary(&self) -> Option<GenerateSummary> {
        let st = self.state.lock().expect("worker state lock poisoned");
        st.last_summary.clone()
    }

    fn send_cmd(&self, msg: &Value) -> Result<()> {
        let _send_guard = self.send_lock.lock().expect("worker send lock poisoned");
        let mut st = self.state.lock().expect("worker state lock poisoned");
        if !st.loaded {
            return Err(anyhow!(WorkerError::new(
                "RuntimeError",
                "model has been unloaded"
            )));
        }
        let sock = st
            .sock_writer
            .as_mut()
            .ok_or_else(|| WorkerError::new("RuntimeError", "model has been unloaded"))?;
        send_json_line(sock, msg).map_err(|e| {
            anyhow!(WorkerError::new(
                "RuntimeError",
                format!("worker IPC send failed: {e}")
            ))
        })
    }

    fn rpc_internal(
        &self,
        op: &str,
        params: Value,
        on_progress: Option<ProgressCallback>,
    ) -> Result<(Value, Option<GenerateSummary>)> {
        let (req_id, rx) = {
            let mut st = self.state.lock().expect("worker state lock poisoned");
            if !st.loaded || st.sock_writer.is_none() {
                return Err(anyhow!(WorkerError::new(
                    "RuntimeError",
                    "model has been unloaded"
                )));
            }
            let req_id = st.alloc_id();
            let (tx, rx) = mpsc::channel();
            st.pending_rpc.insert(req_id, tx);
            (req_id, rx)
        };

        let send_res = self.send_cmd(&json!({"id": req_id, "op": op, "params": params}));
        if let Err(err) = send_res {
            let mut st = self.state.lock().expect("worker state lock poisoned");
            st.pending_rpc.remove(&req_id);
            return Err(err);
        }

        let resp = loop {
            match rx.recv() {
                Ok(msg) => {
                    if msg.get("event").and_then(Value::as_str) == Some("progress") {
                        if let Some(ref cb) = on_progress
                            && let Some(p_val) = msg.get("progress")
                            && let Ok(p) = serde_json::from_value::<DownloadProgress>(p_val.clone())
                        {
                            let _ = cb(&p);
                        }
                        continue;
                    }
                    break msg;
                }
                Err(_) => {
                    let mut st = self.state.lock().expect("worker state lock poisoned");
                    st.pending_rpc.remove(&req_id);
                    return Err(anyhow!(WorkerError::new(
                        "RuntimeError",
                        "isolated worker process terminated unexpectedly"
                    )));
                }
            }
        };

        {
            let mut st = self.state.lock().expect("worker state lock poisoned");
            st.pending_rpc.remove(&req_id);
        }

        if !resp.get("ok").and_then(Value::as_bool).unwrap_or(false) {
            let kind = resp
                .get("err_type")
                .and_then(Value::as_str)
                .unwrap_or("RuntimeError");
            let msg = resp
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("worker request failed");
            return Err(anyhow!(WorkerError::new(kind, msg)));
        }

        let summary: Option<GenerateSummary> = resp
            .get("summary")
            .filter(|v| !v.is_null())
            .and_then(|v| serde_json::from_value(v.clone()).ok());
        if let Some(ref sum) = summary {
            let mut st = self.state.lock().expect("worker state lock poisoned");
            st.last_summary = Some(sum.clone());
        }
        Ok((resp.get("result").cloned().unwrap_or(Value::Null), summary))
    }

    pub fn stats(&self) -> Result<EngineStatsSnapshot> {
        let (res, _) = self.rpc_internal("stats", json!({}), None)?;
        let mut snap: EngineStatsSnapshot = serde_json::from_value(res)?;
        snap.isolated_worker = true;
        Ok(snap)
    }

    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
        let (res, _) = self.rpc_internal(
            "tokenize",
            json!({
                "text": text,
                "add_special": add_special,
                "parse_special": parse_special,
            }),
            None,
        )?;
        Ok(serde_json::from_value(
            res.get("tokens").cloned().unwrap_or_else(|| json!([])),
        )?)
    }

    pub fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        let (res, _) = self.rpc_internal("detokenize", json!({"tokens": tokens}), None)?;
        Ok(res
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string())
    }

    pub fn apply_chat_template(
        &self,
        messages: &[IncomingChatMessage],
        add_generation_prompt: bool,
        tools: Option<&[Value]>,
    ) -> Result<String> {
        let (res, _) = self.rpc_internal(
            "apply_chat_template",
            json!({
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "tools": tools,
            }),
            None,
        )?;
        Ok(res
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string())
    }

    pub fn kv_clear(&self) -> Result<()> {
        let _ = self.rpc_internal("kv_clear", json!({}), None)?;
        Ok(())
    }

    pub fn kv_trim(&self, pos_min: usize) -> Result<bool> {
        let (res, _) = self.rpc_internal("kv_trim", json!({"pos_min": pos_min}), None)?;
        Ok(res.get("trimmed").and_then(Value::as_bool).unwrap_or(false))
    }

    pub fn save_state(&self, path: &str) -> Result<usize> {
        let (res, _) = self.rpc_internal("save_state", json!({"path": path}), None)?;
        Ok(res.get("tokens").and_then(Value::as_u64).unwrap_or(0) as usize)
    }

    pub fn load_state(&self, path: &str) -> Result<usize> {
        let (res, _) = self.rpc_internal("load_state", json!({"path": path}), None)?;
        Ok(res.get("tokens").and_then(Value::as_u64).unwrap_or(0) as usize)
    }

    pub fn generate(
        &self,
        prompt: &str,
        params: &WorkerGenParams,
    ) -> Result<(String, Option<GenerateSummary>)> {
        let (res, summary) = self.rpc_internal(
            "generate",
            json!({"prompt": prompt, "params": params}),
            None,
        )?;
        let text = res
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        Ok((text, summary))
    }

    pub fn chat(
        &self,
        messages: &[IncomingChatMessage],
        params: &WorkerGenParams,
    ) -> Result<(
        String,
        Option<ParsedAssistantOutput>,
        Option<GenerateSummary>,
    )> {
        let (res, summary) = self.rpc_internal(
            "chat",
            json!({"messages": messages, "params": params}),
            None,
        )?;
        let text = res
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let parsed: Option<ParsedAssistantOutput> = res
            .get("parsed")
            .filter(|v| !v.is_null())
            .and_then(|v| serde_json::from_value(v.clone()).ok());
        Ok((text, parsed, summary))
    }

    pub fn generate_stream(&self, prompt: &str, params: &WorkerGenParams) -> Result<WorkerStream> {
        self.start_stream(
            "generate_stream",
            json!({"prompt": prompt, "params": params}),
        )
    }

    pub fn chat_stream(
        &self,
        messages: &[IncomingChatMessage],
        params: &WorkerGenParams,
    ) -> Result<WorkerStream> {
        self.start_stream(
            "chat_stream",
            json!({"messages": messages, "params": params}),
        )
    }

    fn start_stream(&self, op: &str, params: Value) -> Result<WorkerStream> {
        let (req_id, rx) = {
            let mut st = self.state.lock().expect("worker state lock poisoned");
            if !st.loaded || st.sock_writer.is_none() {
                return Err(anyhow!(WorkerError::new(
                    "RuntimeError",
                    "model has been unloaded"
                )));
            }
            let req_id = st.alloc_id();
            let (tx, rx) = mpsc::channel();
            st.active_streams.insert(req_id, tx);
            (req_id, rx)
        };

        if let Err(err) = self.send_cmd(&json!({"id": req_id, "op": op, "params": params})) {
            let mut st = self.state.lock().expect("worker state lock poisoned");
            st.active_streams.remove(&req_id);
            return Err(err);
        }

        Ok(WorkerStream {
            state: Arc::clone(&self.state),
            send_lock: Arc::clone(&self.send_lock),
            req_id,
            rx,
            done: false,
            summary: None,
        })
    }

    /// Gracefully shut down, kill if necessary, and reap the isolated worker process.
    pub fn unload(&self) {
        let (child_opt, reader_opt) = {
            let _send_guard = self.send_lock.lock().expect("worker send lock poisoned");
            let mut st = self.state.lock().expect("worker state lock poisoned");
            st.loaded = false;
            if let Some(mut sock) = st.sock_writer.take() {
                let _ =
                    send_json_line(&mut sock, &json!({"id": 0, "op": "shutdown", "params": {}}));
                let _ = sock.shutdown(std::net::Shutdown::Both);
            }
            // Drop stdin pipe so child's stdin-EOF watchdog fires immediately.
            st.child_stdin.take();
            st.pending_rpc.clear();
            st.active_streams.clear();
            (st.child.take(), st.reader_thread.take())
        };

        if let Some(mut child) = child_opt {
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) if Instant::now() < deadline => {
                        thread::sleep(Duration::from_millis(15));
                    }
                    _ => {
                        let _ = child.kill();
                        let _ = child.wait();
                        break;
                    }
                }
            }
        }

        if let Some(handle) = reader_opt {
            let _ = handle.join();
        }
    }
}

impl Drop for WorkerClient {
    fn drop(&mut self) {
        self.unload();
    }
}

/// Active streaming token iterator backed by an isolated worker subprocess.
pub struct WorkerStream {
    state: Arc<Mutex<WorkerState>>,
    send_lock: Arc<Mutex<()>>,
    req_id: u64,
    rx: mpsc::Receiver<WorkerStreamEvent>,
    done: bool,
    summary: Option<GenerateSummary>,
}

impl WorkerStream {
    pub fn summary(&self) -> Option<GenerateSummary> {
        self.summary.clone()
    }

    pub fn next_chunk(&mut self) -> Result<Option<String>> {
        if self.done {
            return Ok(None);
        }
        match self.rx.recv() {
            Ok(WorkerStreamEvent::Chunk(text)) => Ok(Some(text)),
            Ok(WorkerStreamEvent::Done(summary)) => {
                self.done = true;
                self.summary = summary.clone();
                let mut st = self.state.lock().expect("worker state lock poisoned");
                st.active_streams.remove(&self.req_id);
                if let Some(sum) = summary {
                    st.last_summary = Some(sum);
                }
                Ok(None)
            }
            Ok(WorkerStreamEvent::Error(err)) => {
                self.done = true;
                let mut st = self.state.lock().expect("worker state lock poisoned");
                st.active_streams.remove(&self.req_id);
                Err(anyhow!(err))
            }
            Err(_) => {
                self.done = true;
                let mut st = self.state.lock().expect("worker state lock poisoned");
                st.active_streams.remove(&self.req_id);
                Ok(None)
            }
        }
    }

    pub fn close(&mut self) {
        if self.done {
            return;
        }
        self.done = true;

        let cancel_sent = {
            let _send_guard = self.send_lock.lock().expect("worker send lock poisoned");
            let mut st = self.state.lock().expect("worker state lock poisoned");
            let cancel_id = st.alloc_id();
            if let Some(sock) = st.sock_writer.as_mut() {
                send_json_line(
                    sock,
                    &json!({"id": cancel_id, "op": "cancel", "params": {"target_id": self.req_id}}),
                )
                .is_ok()
            } else {
                false
            }
        };

        if cancel_sent {
            let deadline = Instant::now() + Duration::from_secs(2);
            while Instant::now() < deadline {
                match self.rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(WorkerStreamEvent::Done(summary)) => {
                        self.summary = summary.clone();
                        if let Some(sum) = summary {
                            let mut st = self.state.lock().expect("worker state lock poisoned");
                            st.last_summary = Some(sum);
                        }
                        break;
                    }
                    Ok(WorkerStreamEvent::Error(_)) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                        break;
                    }
                    Ok(WorkerStreamEvent::Chunk(_)) | Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
            }
        }

        let mut st = self.state.lock().expect("worker state lock poisoned");
        st.active_streams.remove(&self.req_id);
    }
}

impl Drop for WorkerStream {
    fn drop(&mut self) {
        self.close();
    }
}

fn reader_loop(mut reader: BufReader<TcpStream>, state: Arc<Mutex<WorkerState>>) {
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(req_id) = msg.get("id").and_then(Value::as_u64) else {
            continue;
        };

        let st = state.lock().expect("worker state lock poisoned");
        if let Some(tx) = st.pending_rpc.get(&req_id).cloned() {
            drop(st);
            let _ = tx.send(msg);
            continue;
        }
        if let Some(tx) = st.active_streams.get(&req_id).cloned() {
            drop(st);
            let ev = match msg.get("event").and_then(Value::as_str) {
                Some("chunk") => WorkerStreamEvent::Chunk(
                    msg.get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                ),
                Some("done") => {
                    let sum = msg
                        .get("summary")
                        .filter(|v| !v.is_null())
                        .and_then(|v| serde_json::from_value(v.clone()).ok());
                    WorkerStreamEvent::Done(sum)
                }
                Some("error") => WorkerStreamEvent::Error(WorkerError::new(
                    msg.get("err_type")
                        .and_then(Value::as_str)
                        .unwrap_or("RuntimeError"),
                    msg.get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("worker stream failed"),
                )),
                _ => continue,
            };
            let _ = tx.send(ev);
        }
    }

    // Wake any blocked callers if the worker disconnected unexpectedly.
    let mut st = state.lock().expect("worker state lock poisoned");
    st.loaded = false;
    for (_, tx) in st.pending_rpc.drain() {
        let _ = tx.send(json!({
            "ok": false,
            "err_type": "RuntimeError",
            "message": "isolated worker process terminated unexpectedly",
        }));
    }
    for (_, tx) in st.active_streams.drain() {
        let _ = tx.send(WorkerStreamEvent::Done(None));
    }
}

type CancelState = Arc<Mutex<(HashSet<u64>, HashMap<u64, Arc<AtomicBool>>)>>;

/// Child-side entrypoint for an isolated worker subprocess.
///
/// Reads the one-time auth token from stdin, monitors stdin EOF to self-exit if
/// the parent dies, connects to `127.0.0.1:<port>`, and serves `LoadedSession`
/// requests until `shutdown` or disconnect.
pub fn run_worker_server(port: u16) -> Result<()> {
    let mut stdin = std::io::stdin();
    let mut token_line = String::new();
    {
        let mut reader = BufReader::new(&mut stdin);
        reader.read_line(&mut token_line)?;
    }
    let token = token_line.trim().to_string();
    if token.is_empty() {
        bail!("missing worker auth token on stdin");
    }

    // Exit immediately if parent dies and closes stdin pipe.
    thread::Builder::new()
        .name("layerstream-parent-watchdog".to_string())
        .spawn(|| {
            let mut buf = [0u8; 64];
            let mut stdin = std::io::stdin();
            loop {
                match stdin.read(&mut buf) {
                    Ok(0) | Err(_) => std::process::exit(0),
                    Ok(_) => {}
                }
            }
        })?;

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let stream = TcpStream::connect(addr)
        .with_context(|| format!("failed to connect to parent worker socket on {addr}"))?;
    stream.set_nodelay(true)?;

    let writer = Arc::new(Mutex::new(stream.try_clone()?));
    {
        let mut w = writer.lock().expect("worker writer lock poisoned");
        send_json_line(&mut w, &json!({"token": token}))?;
    }

    let cancel_state: CancelState = Arc::new(Mutex::new((HashSet::new(), HashMap::new())));

    let (work_tx, work_rx) = mpsc::channel::<Option<Value>>();

    let loop_writer = Arc::clone(&writer);
    let loop_cancel = Arc::clone(&cancel_state);
    let engine_thread = thread::Builder::new()
        .name("layerstream-worker-engine".to_string())
        .spawn(move || worker_engine_loop(work_rx, loop_writer, loop_cancel))?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let op = msg.get("op").and_then(Value::as_str).unwrap_or("");
        if op == "shutdown" {
            break;
        }
        if op == "cancel" {
            if let Some(target_id) = msg
                .get("params")
                .and_then(|p| p.get("target_id"))
                .and_then(Value::as_u64)
            {
                let mut cs = cancel_state.lock().expect("cancel state lock poisoned");
                cs.0.insert(target_id);
                if let Some(flag) = cs.1.get(&target_id) {
                    flag.store(true, Ordering::Relaxed);
                }
            }
            continue;
        }
        if work_tx.send(Some(msg)).is_err() {
            break;
        }
    }

    {
        let cs = cancel_state.lock().expect("cancel state lock poisoned");
        for flag in cs.1.values() {
            flag.store(true, Ordering::Relaxed);
        }
    }
    let _ = work_tx.send(None);
    let _ = engine_thread.join();
    Ok(())
}

fn worker_engine_loop(
    work_rx: mpsc::Receiver<Option<Value>>,
    writer: Arc<Mutex<TcpStream>>,
    cancel_state: CancelState,
) {
    let mut session: Option<LoadedSession> = None;

    let send = |msg: &Value| {
        if let Ok(mut w) = writer.lock() {
            let _ = send_json_line(&mut w, msg);
        }
    };

    while let Ok(Some(cmd)) = work_rx.recv() {
        let req_id = cmd.get("id").and_then(Value::as_u64).unwrap_or(0);
        let op = cmd
            .get("op")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let params = cmd.get("params").cloned().unwrap_or_else(|| json!({}));

        let already_cancelled = {
            let mut cs = cancel_state.lock().expect("cancel state lock poisoned");
            cs.0.remove(&req_id)
        };
        if already_cancelled {
            if op == "generate_stream" || op == "chat_stream" {
                send(&json!({"id": req_id, "event": "done", "summary": Value::Null}));
            } else {
                send(&json!({"id": req_id, "ok": true, "result": {}, "summary": Value::Null}));
            }
            continue;
        }

        let res: Result<()> = (|| {
            match op.as_str() {
                "open" => {
                    let open: WorkerOpenParams = serde_json::from_value(params)?;
                    crate::engine::StreamEngine::set_log_verbose(open.verbose);
                    let on_progress: Option<ProgressCallback> = if open.stream_progress {
                        let pw = Arc::clone(&writer);
                        Some(Arc::new(move |ev: &DownloadProgress| {
                            if let Ok(mut w) = pw.lock() {
                                let _ = send_json_line(
                                    &mut w,
                                    &json!({"id": req_id, "event": "progress", "progress": ev}),
                                );
                            }
                            Ok(())
                        }))
                    } else {
                        None
                    };
                    let opts = OpenSessionOptions {
                        draft_model: open.draft_model,
                        quant: open.quant,
                        revision: open.revision,
                        cache_dir: open.cache_dir,
                        vram_budget_auto: open.vram_budget_auto,
                        vram_budget_bytes: open.vram_budget_bytes,
                        resident_layers: open.resident_layers,
                        ctx_size: open.ctx_size,
                        batch_size: open.batch_size,
                        threads: open.threads,
                        kv_cache_type: open.kv_cache_type,
                        use_gpu: open.use_gpu,
                        prefer_direct_io: open.prefer_direct_io,
                        cpu_embeddings: open.cpu_embeddings,
                        cpu_head: open.cpu_head,
                        stream_head: open.stream_head,
                        moe_active_streaming: open.moe_active_streaming,
                        max_draft_tokens: open.max_draft_tokens,
                        on_progress,
                    };
                    let loaded = LoadedSession::open(&open.model, opts)?;
                    let model_path = loaded.model_path.to_string_lossy().into_owned();
                    session = Some(loaded);
                    send(&json!({"id": req_id, "ok": true, "result": {"model_path": model_path}}));
                }
                _ => {
                    let Some(sess) = session.as_mut() else {
                        bail!("model has been unloaded");
                    };
                    match op.as_str() {
                        "stats" => {
                            let snap = sess.stats_snapshot();
                            send(&json!({"id": req_id, "ok": true, "result": snap}));
                        }
                        "tokenize" => {
                            let text = params
                                .get("text")
                                .and_then(Value::as_str)
                                .ok_or_else(|| anyhow!("invalid tokenize 'text'"))?;
                            let add_special = params
                                .get("add_special")
                                .and_then(Value::as_bool)
                                .unwrap_or(false);
                            let parse_special = params
                                .get("parse_special")
                                .and_then(Value::as_bool)
                                .unwrap_or(false);
                            let tokens = sess.engine.tokenize(text, add_special, parse_special)?;
                            send(&json!({"id": req_id, "ok": true, "result": {"tokens": tokens}}));
                        }
                        "detokenize" => {
                            let tokens: Vec<i32> = serde_json::from_value(
                                params.get("tokens").cloned().unwrap_or_else(|| json!([])),
                            )?;
                            let text = sess.engine.detokenize(&tokens)?;
                            send(&json!({"id": req_id, "ok": true, "result": {"text": text}}));
                        }
                        "apply_chat_template" => {
                            let msgs: Vec<IncomingChatMessage> = serde_json::from_value(
                                params.get("messages").cloned().unwrap_or_else(|| json!([])),
                            )?;
                            let add_gen = params
                                .get("add_generation_prompt")
                                .and_then(Value::as_bool)
                                .unwrap_or(true);
                            let tools: Option<Vec<Value>> = params
                                .get("tools")
                                .filter(|v| !v.is_null())
                                .map(|v| serde_json::from_value(v.clone()))
                                .transpose()?;
                            let chat_msgs = build_chat_messages(&msgs, tools.as_deref())?;
                            let prompt = sess.engine.apply_chat_template(&chat_msgs, add_gen)?;
                            send(&json!({"id": req_id, "ok": true, "result": {"prompt": prompt}}));
                        }
                        "kv_clear" => {
                            sess.engine.kv_clear();
                            send(&json!({"id": req_id, "ok": true, "result": {}}));
                        }
                        "kv_trim" => {
                            let pos_min = params
                                .get("pos_min")
                                .and_then(Value::as_u64)
                                .ok_or_else(|| anyhow!("invalid kv_trim 'pos_min'"))?
                                as i32;
                            let trimmed = sess.engine.kv_trim(pos_min);
                            send(
                                &json!({"id": req_id, "ok": true, "result": {"trimmed": trimmed}}),
                            );
                        }
                        "save_state" => {
                            let path = params
                                .get("path")
                                .and_then(Value::as_str)
                                .ok_or_else(|| anyhow!("invalid save_state 'path'"))?;
                            let tokens = sess.engine.save_state(path)?;
                            send(&json!({"id": req_id, "ok": true, "result": {"tokens": tokens}}));
                        }
                        "load_state" => {
                            let path = params
                                .get("path")
                                .and_then(Value::as_str)
                                .ok_or_else(|| anyhow!("invalid load_state 'path'"))?;
                            let tokens = sess.engine.load_state(path)?;
                            send(&json!({"id": req_id, "ok": true, "result": {"tokens": tokens}}));
                        }
                        "generate" | "chat" | "generate_stream" | "chat_stream" => {
                            let is_chat = op == "chat" || op == "chat_stream";
                            let is_stream = op == "generate_stream" || op == "chat_stream";
                            let gp: WorkerGenParams = serde_json::from_value(
                                params.get("params").cloned().unwrap_or_else(|| json!({})),
                            )?;
                            let prompt = if is_chat {
                                let msgs: Vec<IncomingChatMessage> = serde_json::from_value(
                                    params.get("messages").cloned().unwrap_or_else(|| json!([])),
                                )?;
                                let chat_msgs = build_chat_messages(&msgs, gp.tools.as_deref())?;
                                sess.engine
                                    .apply_chat_template(&chat_msgs, gp.add_generation_prompt)?
                            } else {
                                params
                                    .get("prompt")
                                    .and_then(Value::as_str)
                                    .ok_or_else(|| anyhow!("invalid generate 'prompt'"))?
                                    .to_string()
                            };

                            let cancel_flag = Arc::new(AtomicBool::new(false));
                            {
                                let mut cs =
                                    cancel_state.lock().expect("cancel state lock poisoned");
                                if cs.0.remove(&req_id) {
                                    cancel_flag.store(true, Ordering::Relaxed);
                                }
                                cs.1.insert(req_id, Arc::clone(&cancel_flag));
                            }

                            let gen_opts = gp.to_generate_options(sess.default_max_draft_tokens);
                            let mut collected = String::new();
                            let gen_res = sess.engine.generate_stream_with_cancel(
                                &prompt,
                                &gen_opts,
                                sess.draft.as_mut(),
                                Some(&cancel_flag),
                                |_tok, piece| {
                                    if cancel_flag.load(Ordering::Relaxed) {
                                        return false;
                                    }
                                    if is_stream {
                                        send(
                                            &json!({"id": req_id, "event": "chunk", "text": piece}),
                                        );
                                    } else {
                                        collected.push_str(piece);
                                    }
                                    true
                                },
                            );

                            {
                                let mut cs =
                                    cancel_state.lock().expect("cancel state lock poisoned");
                                cs.1.remove(&req_id);
                                cs.0.remove(&req_id);
                            }

                            let summary = gen_res?;
                            sess.last_summary = Some(summary.clone());

                            if is_stream {
                                send(&json!({"id": req_id, "event": "done", "summary": summary}));
                            } else {
                                let parsed = if is_chat && gp.parse_output {
                                    Some(parse_assistant_output(&collected))
                                } else {
                                    None
                                };
                                send(&json!({
                                    "id": req_id,
                                    "ok": true,
                                    "result": {"text": collected, "parsed": parsed},
                                    "summary": summary,
                                }));
                            }
                        }
                        other => bail!("unknown worker op '{other}'"),
                    }
                }
            }
            Ok(())
        })();

        if let Err(err) = res {
            let (err_type, message) = classify_error(&err);
            if op == "generate_stream" || op == "chat_stream" {
                send(&json!({
                    "id": req_id,
                    "event": "error",
                    "err_type": err_type,
                    "message": message,
                }));
            } else {
                send(&json!({
                    "id": req_id,
                    "ok": false,
                    "err_type": err_type,
                    "message": message,
                }));
            }
        }
    }

    drop(session);
}

fn build_chat_messages(
    incoming: &[IncomingChatMessage],
    tools: Option<&[Value]>,
) -> Result<Vec<crate::engine::ChatMessage>> {
    let mut out = Vec::with_capacity(incoming.len());
    for msg in incoming {
        out.push(
            msg.to_chat_message()
                .map_err(|e| anyhow!("invalid chat message: {e}"))?,
        );
    }
    if let Some(t) = tools
        && !t.is_empty()
    {
        inject_tools_into_messages(&mut out, t);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_token_is_64_lowercase_hex_chars_and_unique() {
        let a = generate_auth_token().expect("token a");
        let b = generate_auth_token().expect("token b");
        assert_eq!(a.len(), 64);
        assert!(a.bytes().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }

    #[test]
    fn worker_gen_params_roundtrip_defaults() {
        let p = WorkerGenParams::default();
        let v = serde_json::to_value(&p).expect("serialize");
        let p2: WorkerGenParams = serde_json::from_value(v).expect("deserialize");
        assert_eq!(p2.max_tokens, 256);
        assert_eq!(p2.top_k, 40);
        assert_eq!(p2.seed, 42);
        assert!(p2.add_generation_prompt);
        assert!(p2.ngram_speculation);
    }

    #[test]
    fn classify_error_maps_not_found_and_value_errors() {
        let nf = anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "missing file",
        ));
        assert_eq!(classify_error(&nf).0, "FileNotFoundError");

        let ptl = anyhow::Error::new(PromptTooLongError {
            prompt_tokens: 5000,
            n_ctx: 4096,
        });
        assert_eq!(classify_error(&ptl).0, "ValueError");

        let unk = anyhow!("unknown worker op 'bogus'");
        assert_eq!(classify_error(&unk).0, "ValueError");

        let rt = anyhow!("engine exploded");
        assert_eq!(classify_error(&rt).0, "RuntimeError");
    }
}
