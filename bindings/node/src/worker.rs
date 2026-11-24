use super::*;

pub(crate) fn map_worker_err(err: anyhow::Error) -> Error {
    if let Some(we) = err.downcast_ref::<WorkerError>() {
        return match we.kind.as_str() {
            "FileNotFoundError" => {
                Error::new(Status::GenericFailure, format!("ENOENT: {}", we.message))
            }
            "ValueError" => invalid_arg(we.message.clone()),
            _ => generic_err(we.message.clone()),
        };
    }
    map_anyhow_err(err)
}

fn stats_snapshot_to_js(s: EngineStatsSnapshot, worker_pid: Option<u32>) -> EngineStatsJs {
    EngineStatsJs {
        loaded: s.loaded,
        isolated_worker: s.isolated_worker,
        worker_pid,
        backend_name: Some(s.backend_name),
        cached_kv_tokens: Some(s.cached_kv_tokens as u32),
        model_path: Some(s.model_path),
        draft_model_path: s.draft_model_path,
        source_uri: s.source_uri,
        resolved_repo: s.resolved_repo,
        selected_quant: s.selected_quant,
        n_ctx: Some(s.n_ctx),
        vocab_size: Some(s.vocab_size),
        n_layers: Some(s.n_layers),
        resident_layers: Some(s.resident_layers),
        streamed_layers: Some(s.streamed_layers),
        resident_bytes: Some(s.resident_bytes as f64),
        cpu_resident_bytes: Some(s.cpu_resident_bytes as f64),
        slot_bytes: Some(s.slot_bytes as f64),
        expert_slot_bytes: Some(s.expert_slot_bytes as f64),
        total_layer_loads: Some(s.total_layer_loads as f64),
        moe_layers: Some(s.moe_layers),
        moe_experts_loaded: Some(s.moe_experts_loaded as f64),
        moe_experts_total: Some(s.moe_experts_total as f64),
        head_streamed: Some(s.head_streamed),
        moe_active_streaming: Some(s.moe_active_streaming),
        is_uma_zero_copy: Some(s.is_uma_zero_copy),
        is_unified_memory: Some(s.is_unified_memory),
        using_gpu: Some(s.using_gpu),
        can_speculate: Some(s.can_speculate),
        io: Some(IoStatsJs {
            bytes_read: s.io.bytes_read as f64,
            span_reads: s.io.span_reads as f64,
            pread_us: s.io.pread_us as f64,
            cache_hits: s.io.cache_hits as f64,
            cache_misses: s.io.cache_misses as f64,
            cache_bytes_served: s.io.cache_bytes_served as f64,
            direct_io_active: s.io.direct_io_active,
        }),
    }
}

fn wire_to_worker_gen_params(wire_json: &str) -> Result<WorkerGenParams> {
    let w: GenerateOptionsWire = serde_json::from_str(wire_json)
        .map_err(|e| invalid_arg(format!("invalid generation options JSON: {e}")))?;
    Ok(WorkerGenParams {
        max_tokens: w.max_tokens.unwrap_or(256),
        temperature: w.temperature.unwrap_or(0.0),
        top_p: w.top_p.unwrap_or(0.95),
        min_p: w.min_p.unwrap_or(0.0),
        top_k: w.top_k.unwrap_or(40),
        repeat_penalty: w.repeat_penalty.unwrap_or(1.0),
        frequency_penalty: w.frequency_penalty.unwrap_or(0.0),
        presence_penalty: w.presence_penalty.unwrap_or(0.0),
        penalty_last_n: w.penalty_last_n.unwrap_or(64),
        seed: w.seed.unwrap_or(42),
        stop: w.stop.unwrap_or_default(),
        add_generation_prompt: w.add_generation_prompt.unwrap_or(true),
        ngram_speculation: w.ngram_speculation.unwrap_or(true),
        max_draft_tokens: w.max_draft_tokens,
        tools: w.tools,
        grammar: w.grammar,
        parse_output: w.parse_output.unwrap_or(false),
    })
}

#[napi]
pub struct NativeWorkerStream {
    inner: Arc<Mutex<Option<WorkerStream>>>,
    last_summary: Arc<Mutex<Option<GenerateSummary>>>,
    cancel_handle: layerstream_core::worker_ipc::WorkerCancelHandle,
    closed: Arc<AtomicBool>,
}

#[napi]
impl NativeWorkerStream {
    #[napi(getter)]
    pub fn summary(&self) -> Option<GenerateSummaryJs> {
        let guard = self.last_summary.lock().ok()?;
        guard.as_ref().map(summary_to_js)
    }

    #[napi]
    pub fn cancel(&self) {
        self.closed.store(true, Ordering::SeqCst);
        let _ = self.cancel_handle.cancel();
        if let Ok(mut guard) = self.inner.lock()
            && let Some(stream) = guard.as_mut()
        {
            stream.close();
            if let Some(s) = stream.summary()
                && let Ok(mut sg) = self.last_summary.lock()
            {
                *sg = Some(s);
            }
        }
    }

    #[napi]
    pub async fn next_chunk(&self) -> Result<Option<String>> {
        if self.closed.load(Ordering::SeqCst) {
            return Ok(None);
        }
        let inner = Arc::clone(&self.inner);
        let last_summary = Arc::clone(&self.last_summary);
        let closed = Arc::clone(&self.closed);
        run_blocking(move || {
            let mut stream = {
                let mut guard = inner
                    .lock()
                    .map_err(|_| generic_err("worker stream lock poisoned"))?;
                let Some(s) = guard.take() else {
                    return Ok(None);
                };
                s
            };
            let res = stream.next_chunk();
            if closed.load(Ordering::SeqCst) {
                stream.close();
            }
            if let Some(s) = stream.summary()
                && let Ok(mut sg) = last_summary.lock()
            {
                *sg = Some(s);
            }
            if let Ok(mut guard) = inner.lock() {
                *guard = Some(stream);
            }
            if closed.load(Ordering::SeqCst) {
                return Ok(None);
            }
            res.map_err(map_worker_err)
        })
        .await
    }

    #[napi]
    pub async fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::SeqCst);
        let _ = self.cancel_handle.cancel();
        let inner = Arc::clone(&self.inner);
        let last_summary = Arc::clone(&self.last_summary);
        run_blocking(move || {
            let mut stream = {
                let mut guard = inner
                    .lock()
                    .map_err(|_| generic_err("worker stream lock poisoned"))?;
                let Some(s) = guard.take() else {
                    return Ok(());
                };
                s
            };
            stream.close();
            if let Some(s) = stream.summary()
                && let Ok(mut sg) = last_summary.lock()
            {
                *sg = Some(s);
            }
            if let Ok(mut guard) = inner.lock() {
                *guard = Some(stream);
            }
            Ok(())
        })
        .await
    }
}

#[napi]
pub struct NativeWorkerClient {
    inner: Arc<WorkerClient>,
}

#[napi]
impl NativeWorkerClient {
    #[napi(factory)]
    pub async fn spawn(
        program: String,
        args: Vec<String>,
        model: String,
        opts_json: String,
    ) -> Result<Self> {
        let wire: OpenOptionsWire = serde_json::from_str(&opts_json)
            .map_err(|e| invalid_arg(format!("invalid open options JSON: {e}")))?;
        let kv_cache_type: KvCacheType = wire
            .kv_cache_type
            .as_deref()
            .unwrap_or("f16")
            .parse()
            .map_err(|e: anyhow::Error| invalid_arg(format!("{e:#}")))?;
        let open = WorkerOpenParams {
            model,
            draft_model: wire.draft_model,
            quant: wire.quant,
            revision: wire.revision,
            cache_dir: wire.cache_dir.map(PathBuf::from),
            vram_budget_auto: wire.vram_budget_auto.unwrap_or(true),
            vram_budget_bytes: wire.vram_budget_bytes.unwrap_or(0),
            resident_layers: wire.resident_layers,
            ctx_size: wire.ctx_size.unwrap_or(4096),
            batch_size: wire.batch_size.unwrap_or(512),
            threads: wire.threads,
            kv_cache_type,
            use_gpu: wire.use_gpu.unwrap_or(true),
            prefer_direct_io: wire.prefer_direct_io.unwrap_or(true),
            cpu_embeddings: wire.cpu_embeddings.unwrap_or(false),
            cpu_head: wire.cpu_head.unwrap_or(false),
            stream_head: wire.stream_head.unwrap_or(false),
            moe_active_streaming: wire.moe_active_streaming.unwrap_or(true),
            max_draft_tokens: wire.max_draft_tokens.unwrap_or(5),
            verbose: wire.verbose.unwrap_or(false),
            stream_progress: false,
        };
        let client = run_blocking(move || {
            WorkerClient::spawn(&program, &args, open, None).map_err(map_worker_err)
        })
        .await?;
        Ok(Self {
            inner: Arc::new(client),
        })
    }

    #[napi(getter)]
    pub fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }

    #[napi(getter)]
    pub fn worker_pid(&self) -> Option<u32> {
        self.inner.worker_pid()
    }

    #[napi(getter)]
    pub fn model_path(&self) -> Option<String> {
        self.inner.model_path()
    }

    #[napi(getter)]
    pub fn last_summary(&self) -> Option<GenerateSummaryJs> {
        self.inner.last_summary().as_ref().map(summary_to_js)
    }

    #[napi]
    pub async fn stats(&self) -> Result<EngineStatsJs> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            if !client.is_loaded() {
                return Ok(EngineStatsJs {
                    loaded: false,
                    isolated_worker: true,
                    worker_pid: None,
                    backend_name: None,
                    cached_kv_tokens: None,
                    model_path: None,
                    draft_model_path: None,
                    source_uri: None,
                    resolved_repo: None,
                    selected_quant: None,
                    n_ctx: None,
                    vocab_size: None,
                    n_layers: None,
                    resident_layers: None,
                    streamed_layers: None,
                    resident_bytes: None,
                    cpu_resident_bytes: None,
                    slot_bytes: None,
                    expert_slot_bytes: None,
                    total_layer_loads: None,
                    moe_layers: None,
                    moe_experts_loaded: None,
                    moe_experts_total: None,
                    head_streamed: None,
                    moe_active_streaming: None,
                    is_uma_zero_copy: None,
                    is_unified_memory: None,
                    using_gpu: None,
                    can_speculate: None,
                    io: None,
                });
            }
            let pid = client.worker_pid();
            let snap = client.stats().map_err(map_worker_err)?;
            Ok(stats_snapshot_to_js(snap, pid))
        })
        .await
    }

    #[napi]
    pub async fn tokenize(
        &self,
        text: String,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            client
                .tokenize(&text, add_special, parse_special)
                .map_err(map_worker_err)
        })
        .await
    }

    #[napi]
    pub async fn detokenize(&self, tokens: Vec<i32>) -> Result<String> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || client.detokenize(&tokens).map_err(map_worker_err)).await
    }

    #[napi]
    pub async fn apply_chat_template(
        &self,
        messages_json: String,
        add_generation_prompt: bool,
        tools_json: Option<String>,
    ) -> Result<String> {
        let msgs: Vec<IncomingChatMessage> = serde_json::from_str(&messages_json)
            .map_err(|e| invalid_arg(format!("invalid chat messages payload: {e}")))?;
        let tools: Option<Vec<JsonValue>> = tools_json
            .map(|raw| serde_json::from_str(&raw))
            .transpose()
            .map_err(|e| invalid_arg(format!("invalid tools JSON: {e}")))?;
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            client
                .apply_chat_template(&msgs, add_generation_prompt, tools.as_deref())
                .map_err(map_worker_err)
        })
        .await
    }

    #[napi]
    pub async fn kv_clear(&self) -> Result<()> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || client.kv_clear().map_err(map_worker_err)).await
    }

    #[napi]
    pub async fn kv_trim(&self, pos_min: u32) -> Result<bool> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || client.kv_trim(pos_min as usize).map_err(map_worker_err)).await
    }

    #[napi]
    pub async fn save_state(&self, path: String) -> Result<u32> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            client
                .save_state(&path)
                .map(|n| n as u32)
                .map_err(map_worker_err)
        })
        .await
    }

    #[napi]
    pub async fn load_state(&self, path: String) -> Result<u32> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            client
                .load_state(&path)
                .map(|n| n as u32)
                .map_err(map_worker_err)
        })
        .await
    }

    #[napi]
    pub async fn generate_with_summary(
        &self,
        prompt: String,
        wire_json: String,
    ) -> Result<GenerateResultJs> {
        let params = wire_to_worker_gen_params(&wire_json)?;
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            let (text, summary) = client.generate(&prompt, &params).map_err(map_worker_err)?;
            Ok(GenerateResultJs {
                text,
                summary: summary.as_ref().map(summary_to_js).unwrap_or_else(|| {
                    summary_to_js(&GenerateSummary {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        finish_reason: "stop".to_string(),
                        speculation: Default::default(),
                    })
                }),
            })
        })
        .await
    }

    #[napi]
    pub async fn chat_with_summary(
        &self,
        messages_json: String,
        wire_json: String,
    ) -> Result<ChatResultJs> {
        let msgs: Vec<IncomingChatMessage> = serde_json::from_str(&messages_json)
            .map_err(|e| invalid_arg(format!("invalid chat messages payload: {e}")))?;
        let params = wire_to_worker_gen_params(&wire_json)?;
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            let (text, parsed, summary) = client.chat(&msgs, &params).map_err(map_worker_err)?;
            Ok(ChatResultJs {
                text,
                parsed: parsed.map(parsed_to_js),
                summary: summary.as_ref().map(summary_to_js).unwrap_or_else(|| {
                    summary_to_js(&GenerateSummary {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        finish_reason: "stop".to_string(),
                        speculation: Default::default(),
                    })
                }),
            })
        })
        .await
    }

    #[napi]
    pub fn generate_stream(&self, prompt: String, wire_json: String) -> Result<NativeWorkerStream> {
        let params = wire_to_worker_gen_params(&wire_json)?;
        let stream = self
            .inner
            .generate_stream(&prompt, &params)
            .map_err(map_worker_err)?;
        let cancel_handle = stream.cancel_handle();
        Ok(NativeWorkerStream {
            inner: Arc::new(Mutex::new(Some(stream))),
            last_summary: Arc::new(Mutex::new(None)),
            cancel_handle,
            closed: Arc::new(AtomicBool::new(false)),
        })
    }

    #[napi]
    pub fn chat_stream(
        &self,
        messages_json: String,
        wire_json: String,
    ) -> Result<NativeWorkerStream> {
        let msgs: Vec<IncomingChatMessage> = serde_json::from_str(&messages_json)
            .map_err(|e| invalid_arg(format!("invalid chat messages payload: {e}")))?;
        let params = wire_to_worker_gen_params(&wire_json)?;
        let stream = self
            .inner
            .chat_stream(&msgs, &params)
            .map_err(map_worker_err)?;
        let cancel_handle = stream.cancel_handle();
        Ok(NativeWorkerStream {
            inner: Arc::new(Mutex::new(Some(stream))),
            last_summary: Arc::new(Mutex::new(None)),
            cancel_handle,
            closed: Arc::new(AtomicBool::new(false)),
        })
    }

    #[napi]
    pub async fn unload(&self) -> Result<()> {
        let client = Arc::clone(&self.inner);
        run_blocking(move || {
            client.unload();
            Ok(())
        })
        .await
    }
}
