use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use layerstream_core::hf::{PretrainedResolveOptions, resolve_pretrained_model};
use layerstream_core::{
    ChatMessage, DraftModelProposer, EngineConfig, GenerateOptions, GenerateSummary, KvCacheType,
    LsSampleParams, StreamEngine,
};
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyString, PyType};

struct LoadedEngine {
    engine: StreamEngine,
    draft: Option<DraftModelProposer>,
    model_path: PathBuf,
    draft_model_path: Option<PathBuf>,
    source_uri: Option<String>,
    resolved_repo: Option<String>,
    selected_quant: Option<String>,
    default_max_draft_tokens: usize,
    last_summary: Option<GenerateSummary>,
}

enum StreamEvent {
    Chunk(String),
    Done,
    Err(String),
}

struct StreamState {
    rx: Option<mpsc::Receiver<StreamEvent>>,
    worker: Option<JoinHandle<()>>,
    done: bool,
}

/// Streaming token iterator returned by `AutoModel.generate(..., stream=True)` and
/// `AutoModel.chat(..., stream=True)`. Releases the Python GIL while waiting for each
/// token and cancels underlying C++/NVMe execution immediately on `.close()` or drop.
#[pyclass]
pub struct TokenStream {
    state: Mutex<StreamState>,
    cancel: Arc<AtomicBool>,
}

impl Drop for TokenStream {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        let handle = if let Ok(mut st) = self.state.lock() {
            st.rx = None;
            st.done = true;
            st.worker.take()
        } else {
            None
        };
        if let Some(h) = handle {
            let _ = h.join();
        }
    }
}

#[pymethods]
impl TokenStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
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

    /// Cancels generation and waits (with the GIL released) for the background worker to exit.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.cancel.store(true, Ordering::Relaxed);
        let handle = {
            let mut st = self
                .state
                .lock()
                .map_err(|_| PyRuntimeError::new_err("stream lock poisoned"))?;
            st.rx = None;
            st.done = true;
            st.worker.take()
        };
        if let Some(h) = handle {
            py.detach(move || {
                let _ = h.join();
            });
        }
        Ok(())
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<String>> {
        loop {
            if let Err(err) = py.check_signals() {
                let _ = self.close(py);
                return Err(err);
            }

            let recv_result = py.detach(|| {
                let st = self.state.lock().ok()?;
                if st.done {
                    return Some(Ok(StreamEvent::Done));
                }
                let rx = st.rx.as_ref()?;
                match rx.recv_timeout(Duration::from_millis(25)) {
                    Ok(ev) => Some(Ok(ev)),
                    Err(RecvTimeoutError::Timeout) => Some(Err(true)),
                    Err(RecvTimeoutError::Disconnected) => Some(Ok(StreamEvent::Done)),
                }
            });

            match recv_result {
                None | Some(Ok(StreamEvent::Done)) => {
                    self.finish_stream(py);
                    return Ok(None);
                }
                Some(Ok(StreamEvent::Chunk(text))) => return Ok(Some(text)),
                Some(Ok(StreamEvent::Err(msg))) => {
                    self.finish_stream(py);
                    return Err(PyRuntimeError::new_err(msg));
                }
                Some(Err(_timeout)) => continue,
            }
        }
    }
}

impl TokenStream {
    fn finish_stream(&self, py: Python<'_>) {
        let handle = if let Ok(mut st) = self.state.lock() {
            st.done = true;
            st.rx = None;
            st.worker.take()
        } else {
            None
        };
        if let Some(h) = handle {
            py.detach(move || {
                let _ = h.join();
            });
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum VramBudgetArg {
    Auto,
    Bytes(u64),
}

impl<'a, 'py> FromPyObject<'a, 'py> for VramBudgetArg {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if obj.is_none() {
            return Ok(Self::Bytes(0));
        }
        if obj.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "vram_budget_mb must be 'auto', None, or a non-negative integer MiB (not bool)",
            ));
        }
        if let Ok(s) = obj.cast::<PyString>() {
            let owned = s.extract::<String>()?;
            let raw = owned.trim();
            if raw.eq_ignore_ascii_case("auto") {
                return Ok(Self::Auto);
            }
            let mb: u64 = raw.parse().map_err(|_| {
                PyValueError::new_err(format!(
                    "invalid vram_budget_mb '{raw}' (expected 'auto' or non-negative integer MiB)"
                ))
            })?;
            return Ok(Self::Bytes(mb.saturating_mul(1024 * 1024)));
        }
        if let Ok(mb) = obj.extract::<i64>() {
            if mb < 0 {
                return Err(PyValueError::new_err("vram_budget_mb must be non-negative"));
            }
            return Ok(Self::Bytes((mb as u64).saturating_mul(1024 * 1024)));
        }
        Err(PyTypeError::new_err(
            "vram_budget_mb must be 'auto', None, or a non-negative integer MiB",
        ))
    }
}

fn parse_stop_sequences(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    let Some(val) = obj else {
        return Ok(Vec::new());
    };
    if val.is_none() {
        return Ok(Vec::new());
    }
    if let Ok(s) = val.cast::<PyString>() {
        return Ok(vec![s.extract::<String>()?]);
    }
    val.extract::<Vec<String>>()
        .map_err(|_| PyTypeError::new_err("stop must be None, a string, or a sequence of strings"))
}

fn parse_chat_messages(messages: &Bound<'_, PyAny>) -> PyResult<Vec<ChatMessage>> {
    let iter = messages.try_iter().map_err(|_| {
        PyTypeError::new_err("messages must be an iterable of {'role': str, 'content': str} dicts")
    })?;
    let mut out = Vec::new();
    for item in iter {
        let item = item?;
        let role: String = item
            .get_item("role")
            .map_err(|_| PyValueError::new_err("each message must contain a 'role' string"))?
            .extract()
            .map_err(|_| PyTypeError::new_err("message 'role' must be a string"))?;
        let content: String = item
            .get_item("content")
            .map_err(|_| PyValueError::new_err("each message must contain a 'content' string"))?
            .extract()
            .map_err(|_| PyTypeError::new_err("message 'content' must be a string"))?;
        out.push(ChatMessage { role, content });
    }
    if out.is_empty() {
        return Err(PyValueError::new_err("messages must not be empty"));
    }
    Ok(out)
}

fn summary_to_pydict<'py>(
    py: Python<'py>,
    summary: &GenerateSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("prompt_tokens", summary.prompt_tokens)?;
    d.set_item("completion_tokens", summary.completion_tokens)?;
    d.set_item("finish_reason", &summary.finish_reason)?;
    let spec = PyDict::new(py);
    spec.set_item("sweeps", summary.speculation.sweeps)?;
    spec.set_item("drafted_tokens", summary.speculation.drafted_tokens)?;
    spec.set_item("accepted_tokens", summary.speculation.accepted_tokens)?;
    d.set_item("speculation", spec)?;
    Ok(d)
}

fn map_open_err(err: anyhow::Error) -> PyErr {
    let msg = format!("{err:#}");
    if msg.contains("local GGUF model not found") {
        PyFileNotFoundError::new_err(msg)
    } else {
        PyRuntimeError::new_err(msg)
    }
}

/// Drop-in AirLLM-style interface to `layerstream-core`'s out-of-core GGUF streaming engine.
#[pyclass]
pub struct AutoModel {
    inner: Arc<Mutex<Option<LoadedEngine>>>,
    loaded: Arc<AtomicBool>,
}

#[pymethods]
impl AutoModel {
    /// Load a GGUF model from a local path, `hf://org/repo:file.gguf` URI, HF GGUF repo,
    /// or official base HF repo (auto-resolving `{org}/{name}-GGUF`).
    #[classmethod]
    #[pyo3(signature = (
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
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_pretrained(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        model: &str,
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
        max_draft_tokens: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        StreamEngine::set_log_verbose(verbose);
        let kv_type: KvCacheType = kv_cache_type
            .parse()
            .map_err(|e: anyhow::Error| PyValueError::new_err(e.to_string()))?;

        let (vram_budget_auto, vram_budget_bytes) = match vram_budget_mb {
            VramBudgetArg::Auto => (true, 0u64),
            VramBudgetArg::Bytes(b) => (false, b),
        };

        let n_threads = threads.unwrap_or_else(|| {
            if use_gpu {
                2
            } else {
                std::thread::available_parallelism()
                    .map(|n| n.get().min(8) as u32)
                    .unwrap_or(4)
            }
        });

        let model_owned = model.to_string();
        let draft_owned = draft_model.map(str::to_string);
        let quant_owned = quant.map(str::to_string);
        let rev_owned = revision.map(str::to_string);

        let loaded = py
            .detach(move || -> anyhow::Result<LoadedEngine> {
                let target_resolve = PretrainedResolveOptions {
                    revision: rev_owned.clone(),
                    quant: quant_owned,
                    cache_dir: cache_dir.clone(),
                    vram_budget_bytes: if vram_budget_bytes > 0 {
                        Some(vram_budget_bytes)
                    } else {
                        None
                    },
                    use_gpu,
                    has_draft: draft_owned.is_some(),
                };
                let resolved_target = resolve_pretrained_model(&model_owned, &target_resolve)?;

                let (draft_proposer, draft_path) = if let Some(draft_spec) = draft_owned {
                    let draft_resolve = PretrainedResolveOptions {
                        revision: rev_owned,
                        quant: None,
                        cache_dir,
                        vram_budget_bytes: None,
                        use_gpu,
                        has_draft: false,
                    };
                    let resolved_draft = resolve_pretrained_model(&draft_spec, &draft_resolve)?;
                    let mut draft_cfg = EngineConfig::reference(&resolved_draft.local_path);
                    draft_cfg.n_ctx = ctx_size;
                    draft_cfg.n_batch = batch_size.clamp(16, 64);
                    draft_cfg.n_threads = n_threads;
                    draft_cfg.kv_cache_type = kv_type;
                    draft_cfg.use_gpu = use_gpu;
                    draft_cfg.prefer_direct_io = prefer_direct_io;
                    let proposer = DraftModelProposer::open(draft_cfg, max_draft_tokens.max(1))?;
                    (Some(proposer), Some(resolved_draft.local_path))
                } else {
                    (None, None)
                };

                let mut cfg = EngineConfig::streaming(&resolved_target.local_path);
                cfg.n_ctx = ctx_size;
                cfg.n_batch = batch_size.max(1);
                cfg.n_threads = n_threads.max(1);
                cfg.resident_layers = resident_layers;
                cfg.vram_budget_bytes = vram_budget_bytes;
                cfg.vram_budget_auto = vram_budget_auto && resident_layers.is_none();
                cfg.kv_cache_type = kv_type;
                cfg.use_gpu = use_gpu;
                cfg.prefer_direct_io = prefer_direct_io;
                cfg.cpu_embeddings = cpu_embeddings;
                cfg.cpu_head = cpu_head;
                cfg.stream_head = stream_head;
                cfg.moe_active_streaming = moe_active_streaming;

                let engine = StreamEngine::open(&cfg)?;

                if let Some(ref proposer) = draft_proposer {
                    engine
                        .check_draft_compatible(proposer.engine())
                        .map_err(|reason| {
                            anyhow::anyhow!("incompatible draft_model tokenizer: {reason}")
                        })?;
                }

                Ok(LoadedEngine {
                    engine,
                    draft: draft_proposer,
                    model_path: resolved_target.local_path,
                    draft_model_path: draft_path,
                    source_uri: resolved_target.source_uri,
                    resolved_repo: resolved_target.resolved_repo,
                    selected_quant: resolved_target.selected_quant,
                    default_max_draft_tokens: max_draft_tokens.max(1),
                    last_summary: None,
                })
            })
            .map_err(map_open_err)?;

        Ok(Self {
            inner: Arc::new(Mutex::new(Some(loaded))),
            loaded: Arc::new(AtomicBool::new(true)),
        })
    }

    /// Generate text from a raw prompt string. Returns a `str` when `stream=False`,
    /// or a `TokenStream` iterator when `stream=True`.
    #[pyo3(signature = (
        prompt,
        *,
        max_tokens=256,
        temperature=0.0,
        top_p=0.95,
        top_k=40,
        seed=42,
        stop=None,
        stream=false,
        ngram_speculation=true,
        max_draft_tokens=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        seed: u32,
        stop: Option<&Bound<'_, PyAny>>,
        stream: bool,
        ngram_speculation: bool,
        max_draft_tokens: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let stop_seqs = parse_stop_sequences(stop)?;
        let opts = GenerateOptions {
            max_tokens,
            sample: LsSampleParams {
                temperature,
                top_p,
                top_k,
                seed,
            },
            stop: stop_seqs,
            enable_ngram_speculation: ngram_speculation,
            max_draft_tokens: max_draft_tokens.unwrap_or(0),
        };
        self.run_generation(py, prompt.to_string(), opts, stream)
    }

    /// Apply the model's embedded chat template to `messages` (`[{"role": ..., "content": ...}]`)
    /// and generate the assistant reply. Returns a `str` or `TokenStream`.
    #[pyo3(signature = (
        messages,
        *,
        max_tokens=256,
        temperature=0.0,
        top_p=0.95,
        top_k=40,
        seed=42,
        stop=None,
        stream=false,
        add_generation_prompt=true,
        ngram_speculation=true,
        max_draft_tokens=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn chat(
        &self,
        py: Python<'_>,
        messages: &Bound<'_, PyAny>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        seed: u32,
        stop: Option<&Bound<'_, PyAny>>,
        stream: bool,
        add_generation_prompt: bool,
        ngram_speculation: bool,
        max_draft_tokens: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let parsed_messages = parse_chat_messages(messages)?;
        let stop_seqs = parse_stop_sequences(stop)?;

        let prompt = with_engine_checking_signals(py, &self.inner, |guard| {
            let loaded = guard
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("model has been unloaded"))?;
            loaded
                .engine
                .apply_chat_template(&parsed_messages, add_generation_prompt)
        })?;

        let opts = GenerateOptions {
            max_tokens,
            sample: LsSampleParams {
                temperature,
                top_p,
                top_k,
                seed,
            },
            stop: stop_seqs,
            enable_ngram_speculation: ngram_speculation,
            max_draft_tokens: max_draft_tokens.unwrap_or(0),
        };
        self.run_generation(py, prompt, opts, stream)
    }

    /// Return current engine residency, MoE, UMA, and O_DIRECT I/O telemetry.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self.inner.clone();
        let snapshot = py.detach(move || {
            let guard = inner.lock().ok()?;
            let loaded = guard.as_ref()?;
            Some((
                loaded.engine.stats(),
                loaded.engine.io_stats(),
                loaded.engine.n_ctx(),
                loaded.engine.vocab_size(),
                loaded.model_path.to_string_lossy().into_owned(),
                loaded
                    .draft_model_path
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
                loaded.source_uri.clone(),
                loaded.resolved_repo.clone(),
                loaded.selected_quant.clone(),
            ))
        });

        let dict = PyDict::new(py);
        let Some((
            st,
            io,
            n_ctx,
            vocab_size,
            model_path,
            draft_model_path,
            source_uri,
            resolved_repo,
            selected_quant,
        )) = snapshot
        else {
            dict.set_item("loaded", false)?;
            return Ok(dict);
        };

        dict.set_item("loaded", true)?;
        dict.set_item("model_path", model_path)?;
        dict.set_item("draft_model_path", draft_model_path)?;
        dict.set_item("source_uri", source_uri)?;
        dict.set_item("resolved_repo", resolved_repo)?;
        dict.set_item("selected_quant", selected_quant)?;
        dict.set_item("n_ctx", n_ctx)?;
        dict.set_item("vocab_size", vocab_size)?;
        dict.set_item("n_layers", st.n_layers)?;
        dict.set_item("resident_layers", st.resident_layers)?;
        dict.set_item("streamed_layers", st.streamed_layers)?;
        dict.set_item("resident_bytes", st.resident_bytes)?;
        dict.set_item("cpu_resident_bytes", st.cpu_resident_bytes)?;
        dict.set_item("slot_bytes", st.slot_bytes)?;
        dict.set_item("expert_slot_bytes", st.expert_slot_bytes)?;
        dict.set_item("total_layer_loads", st.total_layer_loads)?;
        dict.set_item("moe_layers", st.moe_layers)?;
        dict.set_item("moe_experts_loaded", st.moe_experts_loaded)?;
        dict.set_item("moe_experts_total", st.moe_experts_total)?;
        dict.set_item("head_streamed", st.head_streamed)?;
        dict.set_item("moe_active_streaming", st.moe_active_streaming)?;
        dict.set_item("is_uma_zero_copy", st.is_uma_zero_copy)?;
        dict.set_item("is_unified_memory", st.is_unified_memory)?;
        dict.set_item("using_gpu", st.using_gpu)?;
        dict.set_item("can_speculate", st.can_speculate)?;

        let io_dict = PyDict::new(py);
        io_dict.set_item("bytes_read", io.bytes_read)?;
        io_dict.set_item("span_reads", io.span_reads)?;
        io_dict.set_item("pread_us", io.pread_us)?;
        io_dict.set_item("cache_hits", io.cache_hits)?;
        io_dict.set_item("cache_misses", io.cache_misses)?;
        io_dict.set_item("cache_bytes_served", io.cache_bytes_served)?;
        io_dict.set_item("direct_io_active", io.direct_io_active)?;
        dict.set_item("io", io_dict)?;
        Ok(dict)
    }

    /// Summary of the most recent completed `.generate()` or `.chat()` call.
    #[getter]
    fn last_summary<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let inner = self.inner.clone();
        let summary = py.detach(move || {
            inner
                .lock()
                .ok()
                .and_then(|g| g.as_ref().and_then(|l| l.last_summary.clone()))
        });
        match summary {
            Some(s) => Ok(Some(summary_to_pydict(py, &s)?)),
            None => Ok(None),
        }
    }

    #[getter]
    fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::Acquire)
    }

    #[getter]
    fn model_path(&self, py: Python<'_>) -> Option<String> {
        let inner = self.inner.clone();
        py.detach(move || {
            inner.lock().ok().and_then(|g| {
                g.as_ref()
                    .map(|l| l.model_path.to_string_lossy().into_owned())
            })
        })
    }

    /// Deterministically drop model weights, KV cache, ping-pong buffers, and draft model.
    fn unload(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        let loaded = self.loaded.clone();
        py.detach(move || -> anyhow::Result<()> {
            let dropped = {
                let mut guard = inner
                    .lock()
                    .map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
                loaded.store(false, Ordering::Release);
                guard.take()
            };
            drop(dropped);
            Ok(())
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))
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
        self.unload(py)?;
        Ok(false)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        match self.model_path(py) {
            Some(p) => format!("AutoModel(loaded=True, model_path={p:?})"),
            None => "AutoModel(loaded=False)".to_string(),
        }
    }
}

fn with_engine_checking_signals<T, F>(
    py: Python<'_>,
    inner: &Arc<Mutex<Option<LoadedEngine>>>,
    mut f: F,
) -> PyResult<T>
where
    T: Send,
    F: FnMut(&mut Option<LoadedEngine>) -> anyhow::Result<T> + Send,
{
    enum Step<T> {
        Ready(anyhow::Result<T>),
        Busy,
    }

    loop {
        py.check_signals()?;
        let step = py.detach(|| match inner.try_lock() {
            Ok(mut guard) => Step::Ready(f(&mut guard)),
            Err(std::sync::TryLockError::WouldBlock) => {
                std::thread::sleep(Duration::from_millis(5));
                Step::Busy
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                Step::Ready(Err(anyhow::anyhow!("model lock poisoned")))
            }
        });
        match step {
            Step::Ready(res) => return res.map_err(|e| PyRuntimeError::new_err(format!("{e:#}"))),
            Step::Busy => continue,
        }
    }
}

fn lock_engine_with_cancel<'a>(
    inner: &'a Mutex<Option<LoadedEngine>>,
    cancel: &AtomicBool,
) -> anyhow::Result<Option<std::sync::MutexGuard<'a, Option<LoadedEngine>>>> {
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }
        match inner.try_lock() {
            Ok(guard) => return Ok(Some(guard)),
            Err(std::sync::TryLockError::WouldBlock) => {
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                anyhow::bail!("model lock poisoned");
            }
        }
    }
}

impl AutoModel {
    fn run_generation(
        &self,
        py: Python<'_>,
        prompt: String,
        mut opts: GenerateOptions,
        stream: bool,
    ) -> PyResult<Py<PyAny>> {
        if !stream {
            let inner = self.inner.clone();
            let cancel = Arc::new(AtomicBool::new(false));
            let worker_cancel = cancel.clone();
            let (tx, rx) = mpsc::sync_channel::<anyhow::Result<String>>(1);
            let rx = Mutex::new(rx);

            let worker = std::thread::spawn(move || {
                let res = (|| -> anyhow::Result<String> {
                    let Some(mut guard) = lock_engine_with_cancel(&inner, &worker_cancel)? else {
                        return Ok(String::new());
                    };
                    let loaded = guard
                        .as_mut()
                        .ok_or_else(|| anyhow::anyhow!("model has been unloaded"))?;
                    if opts.max_draft_tokens == 0 {
                        opts.max_draft_tokens = loaded.default_max_draft_tokens;
                    }
                    let mut out = String::new();
                    let summary = loaded.engine.generate_stream_with_cancel(
                        &prompt,
                        &opts,
                        loaded.draft.as_mut(),
                        Some(&worker_cancel),
                        |_tok, piece| {
                            if worker_cancel.load(Ordering::Relaxed) {
                                return false;
                            }
                            out.push_str(piece);
                            true
                        },
                    )?;
                    loaded.last_summary = Some(summary);
                    Ok(out)
                })();
                let _ = tx.send(res);
            });

            let result = loop {
                if let Err(err) = py.check_signals() {
                    cancel.store(true, Ordering::Relaxed);
                    py.detach(move || {
                        let _ = worker.join();
                    });
                    return Err(err);
                }
                match py.detach(|| {
                    rx.lock()
                        .map_err(|_| RecvTimeoutError::Disconnected)
                        .and_then(|r| r.recv_timeout(Duration::from_millis(25)))
                }) {
                    Ok(res) => {
                        py.detach(move || {
                            let _ = worker.join();
                        });
                        break res;
                    }
                    Err(RecvTimeoutError::Timeout) => continue,
                    Err(RecvTimeoutError::Disconnected) => {
                        py.detach(move || {
                            let _ = worker.join();
                        });
                        break Err(anyhow::anyhow!("generation worker exited unexpectedly"));
                    }
                }
            };
            let output = result.map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
            return Ok(PyString::new(py, &output).into_any().unbind());
        }

        // Fail fast if already unloaded before spawning a stream worker thread.
        if !self.loaded.load(Ordering::Acquire) {
            return Err(PyRuntimeError::new_err("model has been unloaded"));
        }

        let (tx, rx) = mpsc::sync_channel::<StreamEvent>(64);
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let inner = self.inner.clone();

        let worker = std::thread::spawn(move || {
            let res = (|| -> anyhow::Result<()> {
                let Some(mut guard) = lock_engine_with_cancel(&inner, &worker_cancel)? else {
                    return Ok(());
                };
                let loaded = guard
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("model has been unloaded"))?;
                if opts.max_draft_tokens == 0 {
                    opts.max_draft_tokens = loaded.default_max_draft_tokens;
                }
                let summary = loaded.engine.generate_stream_with_cancel(
                    &prompt,
                    &opts,
                    loaded.draft.as_mut(),
                    Some(&worker_cancel),
                    |_tok, piece| {
                        if worker_cancel.load(Ordering::Relaxed) {
                            return false;
                        }
                        if piece.is_empty() {
                            return true;
                        }
                        tx.send(StreamEvent::Chunk(piece.to_string())).is_ok()
                    },
                )?;
                loaded.last_summary = Some(summary);
                Ok(())
            })();

            match res {
                Ok(()) => {
                    let _ = tx.send(StreamEvent::Done);
                }
                Err(e) => {
                    let _ = tx.send(StreamEvent::Err(format!("{e:#}")));
                }
            }
        });

        let stream_obj = TokenStream {
            state: Mutex::new(StreamState {
                rx: Some(rx),
                worker: Some(worker),
                done: false,
            }),
            cancel,
        };
        Ok(Py::new(py, stream_obj)?.into_any())
    }
}

/// Probe `(free_bytes, total_bytes)` on the primary GPU backend, or `None` if unavailable.
#[pyfunction]
#[pyo3(signature = (use_gpu=true))]
fn probe_gpu_memory(py: Python<'_>, use_gpu: bool) -> Option<(u64, u64)> {
    py.detach(|| StreamEngine::probe_gpu_memory(use_gpu))
}

#[pymodule]
fn _layerstream(m: &Bound<'_, PyModule>) -> PyResult<()> {
    StreamEngine::set_log_verbose(false);
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<AutoModel>()?;
    m.add_class::<TokenStream>()?;
    m.add_function(wrap_pyfunction!(probe_gpu_memory, m)?)?;
    Ok(())
}
