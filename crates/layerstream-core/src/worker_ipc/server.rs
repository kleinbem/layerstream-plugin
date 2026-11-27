use super::*;

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

    // Exit if parent dies unexpectedly and closes stdin pipe, without racing
    // `std::process::exit(0)` against normal engine/GPU teardown on clean shutdown.
    let clean_shutdown = Arc::new(AtomicBool::new(false));
    let watchdog_shutdown = Arc::clone(&clean_shutdown);
    thread::Builder::new()
        .name("layerstream-parent-watchdog".to_string())
        .spawn(move || {
            let mut buf = [0u8; 64];
            let mut stdin = std::io::stdin();
            loop {
                match stdin.read(&mut buf) {
                    Ok(0) | Err(_) => {
                        // Brief grace window in case stdin closed right before TCP shutdown/EOF.
                        for _ in 0..10 {
                            if watchdog_shutdown.load(Ordering::Acquire) {
                                return;
                            }
                            thread::sleep(Duration::from_millis(25));
                        }
                        if watchdog_shutdown.load(Ordering::Acquire) {
                            return;
                        }
                        std::process::exit(0);
                    }
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
        let mut w = writer.lock();
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
                let mut cs = cancel_state.lock();
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

    clean_shutdown.store(true, Ordering::Release);
    {
        let cs = cancel_state.lock();
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
        let mut w = writer.lock();
        let _ = send_json_line(&mut w, msg);
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
            let mut cs = cancel_state.lock();
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
                            let mut w = pw.lock();
                            let _ = send_json_line(
                                &mut w,
                                &json!({"id": req_id, "event": "progress", "progress": ev}),
                            );
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
                                let mut cs = cancel_state.lock();
                                if cs.0.remove(&req_id) {
                                    cancel_flag.store(true, Ordering::Relaxed);
                                }
                                cs.1.insert(req_id, Arc::clone(&cancel_flag));
                            }

                            let gen_opts = gp.to_generate_options(sess.default_max_draft_tokens)?;
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
                                let mut cs = cancel_state.lock();
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

    fn write_tiny_llama_gguf(path: &std::path::Path) {
        const N_VOCAB: u64 = 64;
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

        let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
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

        push_t("token_embd.weight".into(), vec![N_EMBD, N_VOCAB], false);
        push_t("output_norm.weight".into(), vec![N_EMBD], true);
        push_t("output.weight".into(), vec![N_EMBD, N_VOCAB], false);
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
        kv_str(&mut kv, "general.name", "tiny-ipc-test");
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
        kv.extend_from_slice(&N_VOCAB.to_le_bytes());
        for i in 0..N_VOCAB {
            let tok = match i {
                0 => "<bos>".to_string(),
                1 => "<eos>".to_string(),
                2 => " ".to_string(),
                _ => {
                    let ch = (b'a' + ((i - 3) % 26) as u8) as char;
                    if i < 29 {
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
        kv.extend_from_slice(&N_VOCAB.to_le_bytes());
        for i in 0..N_VOCAB {
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

        let mut f = std::fs::File::create(path).expect("create temp gguf");
        f.write_all(&hdr).expect("write hdr");
        let mut written = 0usize;
        for t in &tensors {
            let pad = (t.rel_offset as usize).saturating_sub(written);
            if pad > 0 {
                f.write_all(&vec![0u8; pad]).expect("write pad");
                written += pad;
            }
            f.write_all(&t.bytes).expect("write tensor");
            written += t.bytes.len();
        }
        f.flush().expect("flush");
    }

    fn spawn_in_process_worker_pair() -> (WorkerClient, JoinHandle<()>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback");
        let addr = listener.local_addr().expect("local_addr");
        let srv_handle = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept");
            let writer = Arc::new(Mutex::new(stream.try_clone().expect("clone")));
            let cancel_state: CancelState = Arc::new(Mutex::new((HashSet::new(), HashMap::new())));
            let (work_tx, work_rx) = mpsc::channel::<Option<Value>>();
            let eng_w = Arc::clone(&writer);
            let eng_c = Arc::clone(&cancel_state);
            let eng = thread::spawn(move || worker_engine_loop(work_rx, eng_w, eng_c));
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
                    if let Some(tid) = msg
                        .get("params")
                        .and_then(|p| p.get("target_id"))
                        .and_then(Value::as_u64)
                    {
                        let mut cs = cancel_state.lock();
                        cs.0.insert(tid);
                        if let Some(flag) = cs.1.get(&tid) {
                            flag.store(true, Ordering::Relaxed);
                        }
                    }
                    continue;
                }
                if work_tx.send(Some(msg)).is_err() {
                    break;
                }
            }
            let _ = work_tx.send(None);
            let _ = eng.join();
        });

        let stream = TcpStream::connect(addr).expect("connect");
        let reader = BufReader::new(stream.try_clone().expect("clone"));
        let state = Arc::new(Mutex::new(WorkerState {
            child: None,
            child_stdin: None,
            sock_writer: Some(stream),
            reader_thread: None,
            loaded: true,
            model_path: None,
            last_summary: None,
            next_id: 1,
            pending_rpc: HashMap::new(),
            active_streams: HashMap::new(),
        }));
        let rs = Arc::clone(&state);
        let rt = thread::spawn(move || reader_loop(reader, rs));
        state.lock().reader_thread = Some(rt);
        let client = WorkerClient {
            state,
            send_lock: Arc::new(Mutex::new(())),
        };
        (client, srv_handle)
    }

    #[test]
    fn worker_ipc_in_process_full_rpc_and_streaming_lifecycle() {
        let dir = std::env::temp_dir().join(format!("ls-worker-ipc-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let gguf = dir.join("tiny.gguf");
        let kv_state = dir.join("kv.state");
        write_tiny_llama_gguf(&gguf);

        let (client, srv) = spawn_in_process_worker_pair();
        assert!(!client.is_loaded()); // no Child handle in in-process harness
        assert!(client.worker_pid().is_none());
        assert!(client.model_path().is_none());
        assert!(client.last_summary().is_none());

        // Calling stats before open returns an error from worker_engine_loop.
        let before_open = client.stats();
        assert!(before_open.is_err());

        // Open with missing file -> FileNotFoundError.
        let bad_open = serde_json::to_value(WorkerOpenParams {
            model: dir
                .join("does-not-exist.gguf")
                .to_string_lossy()
                .into_owned(),
            draft_model: None,
            quant: None,
            revision: None,
            cache_dir: None,
            vram_budget_auto: false,
            vram_budget_bytes: 0,
            resident_layers: Some(0),
            ctx_size: 128,
            batch_size: 32,
            threads: Some(2),
            kv_cache_type: KvCacheType::F16,
            use_gpu: false,
            prefer_direct_io: false,
            cpu_embeddings: false,
            cpu_head: false,
            stream_head: false,
            moe_active_streaming: true,
            max_draft_tokens: 2,
            verbose: false,
            stream_progress: true,
        })
        .expect("serialize bad open");
        let err = client.rpc_internal("open", bad_open, None).unwrap_err();
        let we = err.downcast_ref::<WorkerError>().expect("WorkerError");
        assert_eq!(we.kind, "FileNotFoundError");
        assert!(!format!("{we}").is_empty());

        // Open valid synthetic GGUF.
        let good_open = serde_json::to_value(WorkerOpenParams {
            model: gguf.to_string_lossy().into_owned(),
            draft_model: None,
            quant: None,
            revision: None,
            cache_dir: None,
            vram_budget_auto: false,
            vram_budget_bytes: 0,
            resident_layers: Some(0),
            ctx_size: 128,
            batch_size: 32,
            threads: Some(2),
            kv_cache_type: KvCacheType::F16,
            use_gpu: false,
            prefer_direct_io: false,
            cpu_embeddings: false,
            cpu_head: false,
            stream_head: false,
            moe_active_streaming: true,
            max_draft_tokens: 2,
            verbose: false,
            stream_progress: false,
        })
        .expect("serialize good open");
        let (open_res, _) = client.rpc_internal("open", good_open, None).expect("open");
        assert!(open_res["model_path"].as_str().is_some());

        // Stats, tokenize, detokenize, apply_chat_template.
        let snap = client.stats().expect("stats");
        assert!(snap.isolated_worker);
        assert_eq!(snap.n_layers, 2);
        let toks = client.tokenize("abc", true, false).expect("tokenize");
        assert!(!toks.is_empty());
        let detok = client.detokenize(&toks).expect("detokenize");
        assert!(!detok.is_empty());

        let msgs = vec![IncomingChatMessage {
            role: "user".to_string(),
            content: json!("abc"),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let tools = vec![json!({
            "type": "function",
            "function": {"name": "get_weather", "description": "desc", "parameters": {"type": "object"}}
        })];
        let tmpl = client
            .apply_chat_template(&msgs, true, Some(&tools))
            .expect("apply_chat_template");
        assert!(!tmpl.is_empty());

        // Generate + chat (with parse_output) + invalid sampling params error.
        let gp = WorkerGenParams {
            max_tokens: 4,
            temperature: 0.0,
            parse_output: true,
            tools: Some(tools.clone()),
            ..WorkerGenParams::default()
        };
        let (gen_txt, gen_sum) = client.generate("abc", &gp).expect("generate");
        assert!(!gen_txt.is_empty());
        assert!(gen_sum.is_some());
        assert!(client.last_summary().is_some());

        let (chat_txt, parsed, chat_sum) = client.chat(&msgs, &gp).expect("chat");
        assert!(!chat_txt.is_empty());
        assert!(parsed.is_some());
        assert!(chat_sum.is_some());

        let bad_gp = WorkerGenParams {
            temperature: -1.0,
            ..WorkerGenParams::default()
        };
        assert!(client.generate("abc", &bad_gp).is_err());

        // Streaming generate + streaming chat with cancel & close.
        let mut gs = client.generate_stream("abc", &gp).expect("generate_stream");
        let mut chunks = Vec::new();
        while let Some(c) = gs.next_chunk().expect("next_chunk") {
            chunks.push(c);
        }
        assert!(!chunks.is_empty());
        assert!(gs.summary().is_some());
        assert!(gs.next_chunk().expect("done next").is_none());

        let long_gp = WorkerGenParams {
            max_tokens: 16,
            ..gp.clone()
        };
        let mut cs = client.chat_stream(&msgs, &long_gp).expect("chat_stream");
        let cancel_h = cs.cancel_handle();
        let _ = cs.next_chunk();
        assert!(cancel_h.cancel());
        cs.close();

        // Repopulate KV after cancelled stream, then exercise save_state / kv_trim / kv_clear / load_state.
        let _ = client.generate("abc", &gp).expect("repopulate kv");
        let saved = client
            .save_state(&kv_state.to_string_lossy())
            .expect("save_state");
        assert!(saved > 0);
        let _ = client.kv_trim(1).expect("kv_trim");
        client.kv_clear().expect("kv_clear");
        let loaded = client
            .load_state(&kv_state.to_string_lossy())
            .expect("load_state");
        assert_eq!(loaded, saved);
        assert!(client.rpc_internal("bogus_op", json!({}), None).is_err());

        // Unload cleanly and verify post-unload calls fail.
        client.unload();
        assert!(client.stats().is_err());
        assert!(client.generate_stream("abc", &gp).is_err());
        let _ = srv.join();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
