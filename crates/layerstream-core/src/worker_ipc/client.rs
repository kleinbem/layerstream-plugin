use super::*;

pub(super) struct WorkerState {
    pub(super) child: Option<Child>,
    pub(super) child_stdin: Option<ChildStdin>,
    pub(super) sock_writer: Option<TcpStream>,
    pub(super) reader_thread: Option<JoinHandle<()>>,
    pub(super) loaded: bool,
    pub(super) model_path: Option<String>,
    pub(super) last_summary: Option<GenerateSummary>,
    pub(super) next_id: u64,
    pub(super) pending_rpc: HashMap<u64, mpsc::Sender<Value>>,
    pub(super) active_streams: HashMap<u64, mpsc::Sender<WorkerStreamEvent>>,
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
    pub(super) state: Arc<Mutex<WorkerState>>,
    pub(super) send_lock: Arc<Mutex<()>>,
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
            let mut st = state.lock();
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
                    let mut st = client.state.lock();
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
        let mut st = self.state.lock();
        st.loaded && st.is_child_alive()
    }

    pub fn worker_pid(&self) -> Option<u32> {
        let mut st = self.state.lock();
        if !st.loaded {
            return None;
        }
        st.child_pid()
    }

    pub fn model_path(&self) -> Option<String> {
        let st = self.state.lock();
        if st.loaded {
            st.model_path.clone()
        } else {
            None
        }
    }

    pub fn last_summary(&self) -> Option<GenerateSummary> {
        let st = self.state.lock();
        st.last_summary.clone()
    }

    fn send_cmd(&self, msg: &Value) -> Result<()> {
        let _send_guard = self.send_lock.lock();
        let mut st = self.state.lock();
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

    pub(super) fn rpc_internal(
        &self,
        op: &str,
        params: Value,
        on_progress: Option<ProgressCallback>,
    ) -> Result<(Value, Option<GenerateSummary>)> {
        let (req_id, rx) = {
            let mut st = self.state.lock();
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
            let mut st = self.state.lock();
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
                    let mut st = self.state.lock();
                    st.pending_rpc.remove(&req_id);
                    return Err(anyhow!(WorkerError::new(
                        "RuntimeError",
                        "isolated worker process terminated unexpectedly"
                    )));
                }
            }
        };

        {
            let mut st = self.state.lock();
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
            let mut st = self.state.lock();
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
            let mut st = self.state.lock();
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
            let mut st = self.state.lock();
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
        let (child_opt, stdin_opt, reader_opt) = {
            let _send_guard = self.send_lock.lock();
            let mut st = self.state.lock();
            st.loaded = false;
            if let Some(mut sock) = st.sock_writer.take() {
                let _ =
                    send_json_line(&mut sock, &json!({"id": 0, "op": "shutdown", "params": {}}));
                let _ = sock.shutdown(std::net::Shutdown::Both);
            }
            st.pending_rpc.clear();
            st.active_streams.clear();
            // Keep `child_stdin` open while waiting for graceful exit so the child's
            // parent-death watchdog does not race `std::process::exit(0)` against
            // native GPU/Vulkan teardown on the engine thread.
            (
                st.child.take(),
                st.child_stdin.take(),
                st.reader_thread.take(),
            )
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
        drop(stdin_opt);

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

/// Thread-safe handle that cancels an active `WorkerStream` even while another thread
/// holds the stream inside `next_chunk()`.
#[derive(Clone)]
pub struct WorkerCancelHandle {
    state: Arc<Mutex<WorkerState>>,
    send_lock: Arc<Mutex<()>>,
    req_id: u64,
}

impl WorkerCancelHandle {
    /// Sends a non-blocking cancel RPC to the isolated worker for this stream's request ID.
    pub fn cancel(&self) -> bool {
        let _send_guard = self.send_lock.lock();
        let mut st = self.state.lock();
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
    }
}

impl WorkerStream {
    /// Returns a lock-free-of-stream handle that can cancel this stream during a pending read.
    pub fn cancel_handle(&self) -> WorkerCancelHandle {
        WorkerCancelHandle {
            state: Arc::clone(&self.state),
            send_lock: Arc::clone(&self.send_lock),
            req_id: self.req_id,
        }
    }

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
                let mut st = self.state.lock();
                st.active_streams.remove(&self.req_id);
                if let Some(sum) = summary {
                    st.last_summary = Some(sum);
                }
                Ok(None)
            }
            Ok(WorkerStreamEvent::Error(err)) => {
                self.done = true;
                let mut st = self.state.lock();
                st.active_streams.remove(&self.req_id);
                Err(anyhow!(err))
            }
            Err(_) => {
                self.done = true;
                let mut st = self.state.lock();
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
            let _send_guard = self.send_lock.lock();
            let mut st = self.state.lock();
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
                            let mut st = self.state.lock();
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

        let mut st = self.state.lock();
        st.active_streams.remove(&self.req_id);
    }
}

impl Drop for WorkerStream {
    fn drop(&mut self) {
        self.close();
    }
}

pub(super) fn reader_loop(mut reader: BufReader<TcpStream>, state: Arc<Mutex<WorkerState>>) {
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

        let st = state.lock();
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
    let mut st = state.lock();
    st.loaded = false;
    for (_, tx) in st.pending_rpc.drain() {
        let _ = tx.send(json!({
            "ok": false,
            "err_type": "RuntimeError",
            "message": "isolated worker process terminated unexpectedly",
        }));
    }
    for (_, tx) in st.active_streams.drain() {
        let _ = tx.send(WorkerStreamEvent::Error(WorkerError::new(
            "RuntimeError",
            "isolated worker process terminated unexpectedly",
        )));
    }
}
