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

type WorkerChatResult<'py> = (
    String,
    Option<Bound<'py, PyDict>>,
    Option<Bound<'py, PyDict>>,
);

#[pyclass(name = "NativeWorkerClient")]
pub(crate) struct PyWorkerClient {
    inner: Arc<WorkerClient>,
}

#[pymethods]
impl PyWorkerClient {
    #[staticmethod]
    #[pyo3(signature = (program, args, open_json, on_progress=None))]
    fn spawn(
        py: Python<'_>,
        program: String,
        args: Vec<String>,
        open_json: String,
        on_progress: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let open: WorkerOpenParams = serde_json::from_str(&open_json)
            .map_err(|e| PyValueError::new_err(format!("invalid worker open params: {e}")))?;
        let progress_cb: Option<ProgressCallback> = on_progress.map(|cb| {
            let cb = Arc::new(cb);
            let f: ProgressCallback = Arc::new(move |ev: &DownloadProgress| {
                Python::attach(|py| {
                    let dict = serialize_to_pydict(py, ev)?;
                    cb.bind(py).call1((dict,))?;
                    Ok::<(), PyErr>(())
                })
                .map_err(|e| layerstream_core::hf::HfError::Cancelled(e.to_string()))
            });
            f
        });
        let client = py
            .detach(move || WorkerClient::spawn(&program, &args, open, progress_cb))
            .map_err(map_worker_err)?;
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

    #[pyo3(signature = (messages_json, add_generation_prompt=true, tools_json=None))]
    fn apply_chat_template(
        &self,
        py: Python<'_>,
        messages_json: String,
        add_generation_prompt: bool,
        tools_json: Option<String>,
    ) -> PyResult<String> {
        let msgs: Vec<IncomingChatMessage> = serde_json::from_str(&messages_json)
            .map_err(|e| PyValueError::new_err(format!("invalid messages JSON: {e}")))?;
        let tools: Option<Vec<JsonValue>> = tools_json
            .map(|raw| serde_json::from_str(&raw))
            .transpose()
            .map_err(|e| PyValueError::new_err(format!("invalid tools JSON: {e}")))?;
        let client = Arc::clone(&self.inner);
        py.detach(move || {
            client.apply_chat_template(&msgs, add_generation_prompt, tools.as_deref())
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

    fn generate<'py>(
        &self,
        py: Python<'py>,
        prompt: String,
        params_json: String,
    ) -> PyResult<(String, Option<Bound<'py, PyDict>>)> {
        let params: WorkerGenParams = serde_json::from_str(&params_json)
            .map_err(|e| PyValueError::new_err(format!("invalid generate params: {e}")))?;
        let client = Arc::clone(&self.inner);
        let (text, sum) = py
            .detach(move || client.generate(&prompt, &params))
            .map_err(map_worker_err)?;
        let py_sum = sum.map(|s| serialize_to_pydict(py, &s)).transpose()?;
        Ok((text, py_sum))
    }

    fn chat<'py>(
        &self,
        py: Python<'py>,
        messages_json: String,
        params_json: String,
    ) -> PyResult<WorkerChatResult<'py>> {
        let msgs: Vec<IncomingChatMessage> = serde_json::from_str(&messages_json)
            .map_err(|e| PyValueError::new_err(format!("invalid messages JSON: {e}")))?;
        let params: WorkerGenParams = serde_json::from_str(&params_json)
            .map_err(|e| PyValueError::new_err(format!("invalid chat params: {e}")))?;
        let client = Arc::clone(&self.inner);
        let (text, parsed, sum) = py
            .detach(move || client.chat(&msgs, &params))
            .map_err(map_worker_err)?;
        let py_parsed = parsed.map(|p| serialize_to_pydict(py, &p)).transpose()?;
        let py_sum = sum.map(|s| serialize_to_pydict(py, &s)).transpose()?;
        Ok((text, py_parsed, py_sum))
    }

    fn generate_stream(
        &self,
        py: Python<'_>,
        prompt: String,
        params_json: String,
    ) -> PyResult<PyWorkerStream> {
        let params: WorkerGenParams = serde_json::from_str(&params_json)
            .map_err(|e| PyValueError::new_err(format!("invalid generate params: {e}")))?;
        let client = Arc::clone(&self.inner);
        let stream = py
            .detach(move || client.generate_stream(&prompt, &params))
            .map_err(map_worker_err)?;
        let cancel_handle = stream.cancel_handle();
        Ok(PyWorkerStream {
            inner: Mutex::new(Some(stream)),
            cancel_handle,
            closed: AtomicBool::new(false),
        })
    }

    fn chat_stream(
        &self,
        py: Python<'_>,
        messages_json: String,
        params_json: String,
    ) -> PyResult<PyWorkerStream> {
        let msgs: Vec<IncomingChatMessage> = serde_json::from_str(&messages_json)
            .map_err(|e| PyValueError::new_err(format!("invalid messages JSON: {e}")))?;
        let params: WorkerGenParams = serde_json::from_str(&params_json)
            .map_err(|e| PyValueError::new_err(format!("invalid chat params: {e}")))?;
        let client = Arc::clone(&self.inner);
        let stream = py
            .detach(move || client.chat_stream(&msgs, &params))
            .map_err(map_worker_err)?;
        let cancel_handle = stream.cancel_handle();
        Ok(PyWorkerStream {
            inner: Mutex::new(Some(stream)),
            cancel_handle,
            closed: AtomicBool::new(false),
        })
    }

    fn unload(&self, py: Python<'_>) {
        let client = Arc::clone(&self.inner);
        py.detach(move || client.unload());
    }
}
