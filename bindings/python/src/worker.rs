use super::*;

pub(crate) fn map_worker_err(err: anyhow::Error) -> PyErr {
    if let Some(we) = err.downcast_ref::<WorkerError>() {
        return match we.kind.as_str() {
            "FileNotFoundError" => PyFileNotFoundError::new_err(we.message.clone()),
            "ValueError" => PyValueError::new_err(we.message.clone()),
            "TypeError" => PyTypeError::new_err(we.message.clone()),
            _ => PyRuntimeError::new_err(we.message.clone()),
        };
    }
    map_open_err(err)
}

#[pyclass(name = "NativeWorkerStream")]
pub(crate) struct PyWorkerStream {
    inner: Mutex<Option<WorkerStream>>,
    cancel_handle: layerstream_core::worker_ipc::WorkerCancelHandle,
    closed: AtomicBool,
}

#[pymethods]
impl PyWorkerStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<String>> {
        if self.closed.load(Ordering::SeqCst) {
            return Ok(None);
        }
        let Some(mut stream) = self.inner.lock().take() else {
            return Ok(None);
        };

        let res = py.detach(move || {
            let next = stream.next_chunk();
            (stream, next)
        });
        let (mut stream, next) = res;
        let was_closed = self.closed.load(Ordering::SeqCst);
        if was_closed {
            stream = py.detach(move || {
                stream.close();
                stream
            });
        }
        *self.inner.lock() = Some(stream);
        if was_closed {
            return Ok(None);
        }
        next.map_err(map_worker_err)
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.closed.store(true, Ordering::SeqCst);
        let cancel_handle = self.cancel_handle.clone();
        py.detach(move || {
            let _ = cancel_handle.cancel();
        });
        let Some(mut stream) = self.inner.lock().take() else {
            return Ok(());
        };

        let stream = py.detach(move || {
            stream.close();
            stream
        });
        *self.inner.lock() = Some(stream);
        Ok(())
    }

    #[getter]
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let guard = self.inner.lock();
        let Some(sum) = guard.as_ref().and_then(WorkerStream::summary) else {
            return Ok(None);
        };
        Ok(Some(serialize_to_pydict(py, &sum)?))
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
        self.close(py)?;
        Ok(false)
    }
}

#[pyclass(name = "NativeWorkerClient")]
pub(crate) struct PyWorkerClient {
    inner: Arc<WorkerClient>,
}

#[pymethods]
impl PyWorkerClient {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        program,
        args,
        model,
        *,
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
        max_draft_tokens=5,
        verbose=false,
        on_progress=None,
    ))]
    fn spawn(
        py: Python<'_>,
        program: String,
        args: Vec<String>,
        model: String,
        draft_model: Option<String>,
        quant: Option<String>,
        revision: Option<String>,
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
        max_draft_tokens: usize,
        verbose: bool,
        on_progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let kv_type = kv_cache_type
            .parse::<KvCacheType>()
            .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
        let (vram_budget_auto, vram_budget_bytes) = match vram_budget_mb {
            VramBudgetArg::Auto => (true, 0),
            VramBudgetArg::Bytes(b) => (false, b),
        };
        let (progress_cb, err_slot) = build_progress_callback(on_progress)?;
        let open = WorkerOpenParams {
            model,
            draft_model,
            quant,
            revision,
            cache_dir,
            vram_budget_auto,
            vram_budget_bytes,
            resident_layers,
            ctx_size,
            batch_size,
            threads,
            kv_cache_type: kv_type,
            use_gpu,
            prefer_direct_io,
            cpu_embeddings,
            cpu_head,
            stream_head,
            moe_active_streaming,
            max_draft_tokens,
            verbose,
            stream_progress: progress_cb.is_some(),
        };
        let res = py.detach(move || WorkerClient::spawn(&program, &args, open, progress_cb));
        if let Some(py_err) = err_slot.lock().take() {
            return Err(py_err);
        }
        let client = res.map_err(map_worker_err)?;
        Ok(Self {
            inner: Arc::new(client),
        })
    }

    #[getter]
    fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }

    #[getter]
    fn worker_pid(&self) -> Option<u32> {
        self.inner.worker_pid()
    }

    #[getter]
    fn model_path(&self) -> Option<String> {
        self.inner.model_path()
    }

    #[getter]
    fn last_summary<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(sum) = self.inner.last_summary() else {
            return Ok(None);
        };
        Ok(Some(serialize_to_pydict(py, &sum)?))
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if !self.inner.is_loaded() {
            let d = PyDict::new(py);
            d.set_item("loaded", false)?;
            d.set_item("isolated_worker", true)?;
            d.set_item("worker_pid", py.None())?;
            return Ok(d);
        }
        let client = Arc::clone(&self.inner);
        let snap = py.detach(move || client.stats()).map_err(map_worker_err)?;
        let d = serialize_to_pydict(py, &snap)?;
        d.set_item("isolated_worker", true)?;
        d.set_item("worker_pid", self.inner.worker_pid())?;
        Ok(d)
    }

    #[pyo3(signature = (text, *, add_special=false, parse_special=false))]
    fn tokenize(
        &self,
        py: Python<'_>,
        text: String,
        add_special: bool,
        parse_special: bool,
    ) -> PyResult<Vec<i32>> {
        let client = Arc::clone(&self.inner);
        py.detach(move || client.tokenize(&text, add_special, parse_special))
            .map_err(map_worker_err)
    }

    fn detokenize(&self, py: Python<'_>, tokens: Vec<i32>) -> PyResult<String> {
        let client = Arc::clone(&self.inner);
        py.detach(move || client.detokenize(&tokens))
            .map_err(map_worker_err)
    }

    #[pyo3(signature = (messages, *, add_generation_prompt=true, tools=None))]
    fn apply_chat_template(
        &self,
        py: Python<'_>,
        messages: &Bound<'_, PyAny>,
        add_generation_prompt: bool,
        tools: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let msgs = parse_incoming_chat_messages(messages)?;
        let tools_vec = parse_tools_arg(tools)?;
        let tools_opt = (!tools_vec.is_empty()).then_some(tools_vec);
        let client = Arc::clone(&self.inner);
        py.detach(move || {
            client.apply_chat_template(&msgs, add_generation_prompt, tools_opt.as_deref())
        })
        .map_err(map_worker_err)
    }

    fn kv_clear(&self, py: Python<'_>) -> PyResult<()> {
        let client = Arc::clone(&self.inner);
        py.detach(move || client.kv_clear()).map_err(map_worker_err)
    }

    fn kv_trim(&self, py: Python<'_>, pos_min: usize) -> PyResult<bool> {
        let client = Arc::clone(&self.inner);
        py.detach(move || client.kv_trim(pos_min))
            .map_err(map_worker_err)
    }

    fn save_state(&self, py: Python<'_>, path: String) -> PyResult<usize> {
        let client = Arc::clone(&self.inner);
        py.detach(move || client.save_state(&path))
            .map_err(map_worker_err)
    }

    fn load_state(&self, py: Python<'_>, path: String) -> PyResult<usize> {
        let client = Arc::clone(&self.inner);
        py.detach(move || client.load_state(&path))
            .map_err(map_worker_err)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        prompt,
        *,
        max_tokens=256,
        temperature=0.0,
        top_p=0.95,
        min_p=0.0,
        top_k=40,
        repeat_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        penalty_last_n=64,
        seed=42,
        stop=None,
        stream=false,
        ngram_speculation=true,
        max_draft_tokens=None,
        grammar=None,
        json_schema=None,
        return_summary=false,
    ))]
    fn generate(
        &self,
        py: Python<'_>,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        min_p: f32,
        top_k: i32,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        penalty_last_n: i32,
        seed: u32,
        stop: Option<&Bound<'_, PyAny>>,
        stream: bool,
        ngram_speculation: bool,
        max_draft_tokens: Option<usize>,
        grammar: Option<&str>,
        json_schema: Option<&Bound<'_, PyAny>>,
        return_summary: bool,
    ) -> PyResult<Py<PyAny>> {
        let sample = build_sample_params(
            temperature,
            top_p,
            min_p,
            top_k,
            repeat_penalty,
            frequency_penalty,
            presence_penalty,
            penalty_last_n,
            seed,
        )?;
        let stop_vec = parse_stop_sequences(stop)?;
        let eff_grammar = resolve_grammar_arg(grammar, json_schema)?;
        let params = WorkerGenParams {
            max_tokens,
            temperature: sample.temperature,
            top_p: sample.top_p,
            min_p: sample.min_p,
            top_k: sample.top_k,
            repeat_penalty: sample.repeat_penalty,
            frequency_penalty: sample.frequency_penalty,
            presence_penalty: sample.presence_penalty,
            penalty_last_n: sample.penalty_last_n,
            seed: sample.seed,
            stop: stop_vec,
            add_generation_prompt: true,
            ngram_speculation,
            max_draft_tokens,
            tools: None,
            grammar: eff_grammar,
            parse_output: false,
        };
        let client = Arc::clone(&self.inner);
        if stream {
            let ws = py
                .detach(move || client.generate_stream(&prompt, &params))
                .map_err(map_worker_err)?;
            let cancel_handle = ws.cancel_handle();
            let py_stream = Py::new(
                py,
                PyWorkerStream {
                    inner: Mutex::new(Some(ws)),
                    cancel_handle,
                    closed: AtomicBool::new(false),
                },
            )?;
            return Ok(py_stream.into_any());
        }
        let (text, sum) = py
            .detach(move || client.generate(&prompt, &params))
            .map_err(map_worker_err)?;
        if return_summary {
            let py_sum = sum.map(|s| serialize_to_pydict(py, &s)).transpose()?;
            return Ok((text, py_sum).into_pyobject(py)?.into_any().unbind());
        }
        Ok(text.into_pyobject(py)?.into_any().unbind())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        messages,
        *,
        max_tokens=256,
        temperature=0.0,
        top_p=0.95,
        min_p=0.0,
        top_k=40,
        repeat_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        penalty_last_n=64,
        seed=42,
        stop=None,
        stream=false,
        add_generation_prompt=true,
        ngram_speculation=true,
        max_draft_tokens=None,
        tools=None,
        grammar=None,
        json_schema=None,
        parse_output=false,
        return_summary=false,
    ))]
    fn chat(
        &self,
        py: Python<'_>,
        messages: &Bound<'_, PyAny>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        min_p: f32,
        top_k: i32,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        penalty_last_n: i32,
        seed: u32,
        stop: Option<&Bound<'_, PyAny>>,
        stream: bool,
        add_generation_prompt: bool,
        ngram_speculation: bool,
        max_draft_tokens: Option<usize>,
        tools: Option<&Bound<'_, PyAny>>,
        grammar: Option<&str>,
        json_schema: Option<&Bound<'_, PyAny>>,
        parse_output: bool,
        return_summary: bool,
    ) -> PyResult<Py<PyAny>> {
        if stream && parse_output {
            return Err(PyValueError::new_err(
                "parse_output=True cannot be combined with stream=True (use StreamTagParser or layerstream.OpenAI)",
            ));
        }
        let sample = build_sample_params(
            temperature,
            top_p,
            min_p,
            top_k,
            repeat_penalty,
            frequency_penalty,
            presence_penalty,
            penalty_last_n,
            seed,
        )?;
        let stop_vec = parse_stop_sequences(stop)?;
        let eff_grammar = resolve_grammar_arg(grammar, json_schema)?;
        let msgs = parse_incoming_chat_messages(messages)?;
        let tools_vec = parse_tools_arg(tools)?;
        let params = WorkerGenParams {
            max_tokens,
            temperature: sample.temperature,
            top_p: sample.top_p,
            min_p: sample.min_p,
            top_k: sample.top_k,
            repeat_penalty: sample.repeat_penalty,
            frequency_penalty: sample.frequency_penalty,
            presence_penalty: sample.presence_penalty,
            penalty_last_n: sample.penalty_last_n,
            seed: sample.seed,
            stop: stop_vec,
            add_generation_prompt,
            ngram_speculation,
            max_draft_tokens,
            tools: (!tools_vec.is_empty()).then_some(tools_vec),
            grammar: eff_grammar,
            parse_output,
        };
        let client = Arc::clone(&self.inner);
        if stream {
            let ws = py
                .detach(move || client.chat_stream(&msgs, &params))
                .map_err(map_worker_err)?;
            let cancel_handle = ws.cancel_handle();
            let py_stream = Py::new(
                py,
                PyWorkerStream {
                    inner: Mutex::new(Some(ws)),
                    cancel_handle,
                    closed: AtomicBool::new(false),
                },
            )?;
            return Ok(py_stream.into_any());
        }
        let (text, parsed, sum) = py
            .detach(move || client.chat(&msgs, &params))
            .map_err(map_worker_err)?;
        let payload: Py<PyAny> = if parse_output {
            match parsed {
                Some(p) => serialize_to_pydict(py, &p)?.into_any().unbind(),
                None => py.None(),
            }
        } else {
            text.into_pyobject(py)?.into_any().unbind()
        };
        if return_summary {
            let py_sum = sum.map(|s| serialize_to_pydict(py, &s)).transpose()?;
            return Ok((payload, py_sum).into_pyobject(py)?.into_any().unbind());
        }
        Ok(payload)
    }

    fn unload(&self, py: Python<'_>) {
        let client = Arc::clone(&self.inner);
        py.detach(move || client.unload());
    }
}
