use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use layerstream_core::hf::{self, DownloadProgress, ProgressCallback};
use layerstream_core::sdk::{
    CheckOptions, LoadedSession, OpenSessionOptions, RecommendOptions, check_model,
    inspect_gguf_file, pull_model, recommend_repo,
};
use layerstream_core::worker_ipc::{
    WorkerClient, WorkerError, WorkerGenParams, WorkerOpenParams, WorkerStream, run_worker_server,
};
use layerstream_core::{
    ChatMessage, GenerateOptions, GenerateSummary, IncomingChatMessage, KvCacheType,
    LsSampleParams, StreamDelta, StreamEngine, StreamTagParser, compile_response_format,
    format_tools_system_prompt, inject_tools_into_messages, json_object_gbnf, json_schema_to_gbnf,
    normalize_kv_cache_type, normalize_vram_budget_mb, parse_assistant_output,
    validate_sample_params,
};
use layerstream_server::{AppState, ModelRegistry, ServerConfig, serve_with_state_until};
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyDict, PyFloat, PyInt, PyList, PyMapping, PySequence, PyString, PyType,
};
use serde::Serialize;
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};

enum StreamEvent {
    Chunk(String),
    Done(GenerateSummary),
    Err(String),
}

struct StreamState {
    rx: Option<mpsc::Receiver<StreamEvent>>,
    worker: Option<JoinHandle<()>>,
    done: bool,
    summary: Option<GenerateSummary>,
}

fn json_to_py<'py>(py: Python<'py>, val: &JsonValue) -> PyResult<Bound<'py, PyAny>> {
    match val {
        JsonValue::Null => Ok(py.None().into_bound(py)),
        JsonValue::Bool(b) => Ok((*b).into_pyobject(py)?.to_owned().into_any()),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any())
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_pyobject(py)?.into_any())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any())
            } else {
                Ok(py.None().into_bound(py))
            }
        }
        JsonValue::String(s) => Ok(PyString::new(py, s).into_any()),
        JsonValue::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_any())
        }
        JsonValue::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into_any())
        }
    }
}

fn serialize_to_py<'py, T: Serialize>(py: Python<'py>, value: &T) -> PyResult<Bound<'py, PyAny>> {
    let val = serde_json::to_value(value)
        .map_err(|e| PyRuntimeError::new_err(format!("JSON serialization failed: {e}")))?;
    json_to_py(py, &val)
}

fn serialize_to_pydict<'py, T: Serialize>(
    py: Python<'py>,
    value: &T,
) -> PyResult<Bound<'py, PyDict>> {
    serialize_to_py(py, value)?
        .cast_into::<PyDict>()
        .map_err(|_| PyRuntimeError::new_err("expected serialized object dict"))
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

    #[getter]
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let sum = self
            .state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("stream lock poisoned"))?
            .summary
            .clone();
        match sum {
            Some(s) => Ok(Some(serialize_to_pydict(py, &s)?)),
            None => Ok(None),
        }
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
                    return None;
                }
                let rx = st.rx.as_ref()?;
                match rx.recv_timeout(Duration::from_millis(25)) {
                    Ok(ev) => Some(Ok(ev)),
                    Err(RecvTimeoutError::Timeout) => Some(Err(true)),
                    Err(RecvTimeoutError::Disconnected) => None,
                }
            });

            match recv_result {
                None => {
                    self.finish_stream(py);
                    return Ok(None);
                }
                Some(Ok(StreamEvent::Done(summary))) => {
                    if let Ok(mut st) = self.state.lock() {
                        st.summary = Some(summary);
                    }
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

/// Stateful incremental parser splitting `<think>...</think>` reasoning blocks and
/// `<tool_call>...</tool_call>` JSON payloads across arbitrary token boundaries.
#[pyclass(name = "StreamTagParser")]
pub struct PyStreamTagParser {
    inner: Mutex<StreamTagParser>,
}

#[pymethods]
impl PyStreamTagParser {
    #[new]
    fn new() -> Self {
        Self {
            inner: Mutex::new(StreamTagParser::new()),
        }
    }

    #[getter]
    fn has_tool_calls(&self) -> PyResult<bool> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("StreamTagParser lock poisoned"))?;
        Ok(guard.has_tool_calls())
    }

    fn push<'py>(&self, py: Python<'py>, piece: &str) -> PyResult<Bound<'py, PyList>> {
        let deltas = {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("StreamTagParser lock poisoned"))?;
            guard.push(piece)
        };
        deltas_to_pylist(py, &deltas)
    }

    fn flush<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let deltas = {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("StreamTagParser lock poisoned"))?;
            guard.flush()
        };
        deltas_to_pylist(py, &deltas)
    }
}

fn deltas_to_pylist<'py>(py: Python<'py>, deltas: &[StreamDelta]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for d in deltas {
        let item = PyDict::new(py);
        match d {
            StreamDelta::Content(text) => {
                item.set_item("type", "content")?;
                item.set_item("text", text)?;
            }
            StreamDelta::Reasoning(text) => {
                item.set_item("type", "reasoning")?;
                item.set_item("text", text)?;
            }
            StreamDelta::ToolCall {
                index,
                id,
                name,
                arguments,
            } => {
                item.set_item("type", "tool_call")?;
                item.set_item("index", *index)?;
                item.set_item("id", id)?;
                item.set_item("name", name)?;
                item.set_item("arguments", arguments)?;
            }
        }
        list.append(item)?;
    }
    Ok(list)
}

fn pyany_to_json_value(obj: &Bound<'_, PyAny>, depth: usize) -> PyResult<JsonValue> {
    if depth > 64 {
        return Err(PyValueError::new_err("JSON value nesting depth exceeds 64"));
    }
    if obj.is_none() {
        return Ok(JsonValue::Null);
    }
    if obj.is_instance_of::<PyBool>() {
        return Ok(JsonValue::Bool(obj.extract::<bool>()?));
    }
    if obj.is_instance_of::<PyInt>() {
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(JsonValue::Number(i.into()));
        }
        let u = obj.extract::<u64>()?;
        return Ok(JsonValue::Number(u.into()));
    }
    if obj.is_instance_of::<PyFloat>() {
        let f = obj.extract::<f64>()?;
        let num = JsonNumber::from_f64(f)
            .ok_or_else(|| PyValueError::new_err("non-finite float is not valid JSON"))?;
        return Ok(JsonValue::Number(num));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(JsonValue::String(s.extract::<String>()?));
    }
    // Support Pydantic v2 (`model_json_schema`) and v1 (`schema`) model classes.
    if obj.is_instance_of::<PyType>() {
        if obj.hasattr("model_json_schema")? {
            let schema_dict = obj.call_method0("model_json_schema")?;
            return pyany_to_json_value(&schema_dict, depth + 1);
        }
        if obj.hasattr("schema")? {
            let schema_dict = obj.call_method0("schema")?;
            return pyany_to_json_value(&schema_dict, depth + 1);
        }
    }
    if let Ok(map) = obj.cast::<PyMapping>() {
        let keys = map.keys()?;
        let mut out = JsonMap::with_capacity(keys.len());
        for k in keys.iter() {
            let key_str: String = k
                .extract()
                .map_err(|_| PyTypeError::new_err("JSON object keys must be strings"))?;
            let val = map.get_item(&k)?;
            out.insert(key_str, pyany_to_json_value(&val, depth + 1)?);
        }
        return Ok(JsonValue::Object(out));
    }
    if let Ok(seq) = obj.cast::<PySequence>() {
        let len = seq.len()?;
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let item = seq.get_item(i)?;
            out.push(pyany_to_json_value(&item, depth + 1)?);
        }
        return Ok(JsonValue::Array(out));
    }
    Err(PyTypeError::new_err(format!(
        "cannot convert Python type '{}' to JSON",
        obj.get_type().name()?
    )))
}

fn extract_json_schema_value(obj: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    if let Ok(s) = obj.cast::<PyString>() {
        let raw = s.extract::<String>()?;
        return serde_json::from_str(&raw)
            .map_err(|e| PyValueError::new_err(format!("invalid JSON schema string: {e}")));
    }
    pyany_to_json_value(obj, 0)
}

fn resolve_grammar_arg(
    grammar: Option<&str>,
    json_schema: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    if grammar.is_some() && json_schema.is_some() {
        return Err(PyValueError::new_err(
            "pass at most one of 'grammar' or 'json_schema', not both",
        ));
    }
    if let Some(g) = grammar {
        if g.trim().is_empty() {
            return Err(PyValueError::new_err("grammar must not be empty"));
        }
        return Ok(Some(g.to_string()));
    }
    if let Some(schema_obj) = json_schema {
        if schema_obj.is_none() {
            return Ok(None);
        }
        let schema_val = extract_json_schema_value(schema_obj)?;
        let compiled = json_schema_to_gbnf(&schema_val)
            .map_err(|e| PyValueError::new_err(format!("invalid json_schema: {e:#}")))?;
        return Ok(Some(compiled));
    }
    Ok(None)
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

#[allow(clippy::too_many_arguments)]
fn build_sample_params(
    temperature: f32,
    top_p: f32,
    min_p: f32,
    top_k: i32,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    penalty_last_n: i32,
    seed: u32,
) -> PyResult<LsSampleParams> {
    let sample = LsSampleParams {
        temperature,
        top_p,
        min_p,
        top_k,
        repeat_penalty,
        frequency_penalty,
        presence_penalty,
        penalty_last_n,
        seed,
    };
    sample.validate().map_err(PyValueError::new_err)?;
    Ok(sample)
}

type PyErrSlot = Arc<Mutex<Option<PyErr>>>;

fn build_progress_callback(
    on_progress: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Option<ProgressCallback>, PyErrSlot)> {
    let err_slot: PyErrSlot = Arc::new(Mutex::new(None));
    let Some(cb_bound) = on_progress else {
        return Ok((None, err_slot));
    };
    if cb_bound.is_none() {
        return Ok((None, err_slot));
    }
    if !cb_bound.is_callable() {
        return Err(PyTypeError::new_err(
            "on_progress must be None or a callable accepting a progress dict",
        ));
    }
    let py_cb: Py<PyAny> = cb_bound.clone().unbind();
    let slot_clone = err_slot.clone();
    let cb: ProgressCallback = Arc::new(move |ev: &DownloadProgress| {
        Python::attach(|py| -> anyhow::Result<()> {
            let call_res: PyResult<()> = (|| {
                let d = serialize_to_pydict(py, ev)?;
                py_cb.bind(py).call1((d,))?;
                Ok(())
            })();
            if let Err(py_err) = call_res {
                if let Ok(mut guard) = slot_clone.lock() {
                    *guard = Some(py_err);
                }
                anyhow::bail!("on_progress callback raised Python exception");
            }
            Ok(())
        })
    });
    Ok((Some(cb), err_slot))
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

        let content_val = match item.get_item("content") {
            Ok(c) => pyany_to_json_value(&c, 0)?,
            Err(_) if item.get_item("tool_calls").is_ok() => JsonValue::Null,
            Err(_) => {
                return Err(PyValueError::new_err(
                    "each message must contain a 'content' string",
                ));
            }
        };
        let tool_calls = match item.get_item("tool_calls") {
            Ok(tc) if !tc.is_none() => match pyany_to_json_value(&tc, 0)? {
                JsonValue::Array(arr) => Some(arr),
                _ => return Err(PyTypeError::new_err("tool_calls must be a sequence")),
            },
            _ => None,
        };
        let name = match item.get_item("name") {
            Ok(n) if !n.is_none() => Some(
                n.extract::<String>()
                    .map_err(|_| PyTypeError::new_err("message 'name' must be a string"))?,
            ),
            _ => None,
        };
        let tool_call_id = match item.get_item("tool_call_id") {
            Ok(id) if !id.is_none() => Some(
                id.extract::<String>()
                    .map_err(|_| PyTypeError::new_err("message 'tool_call_id' must be a string"))?,
            ),
            _ => None,
        };

        let incoming = IncomingChatMessage {
            role,
            content: content_val,
            tool_calls,
            tool_call_id,
            name,
        };
        out.push(incoming.to_chat_message().map_err(PyValueError::new_err)?);
    }
    if out.is_empty() {
        return Err(PyValueError::new_err("messages must not be empty"));
    }
    Ok(out)
}

fn parse_tools_arg(tools: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<JsonValue>> {
    let Some(t_obj) = tools else {
        return Ok(Vec::new());
    };
    if t_obj.is_none() {
        return Ok(Vec::new());
    }
    let val = pyany_to_json_value(t_obj, 0)?;
    match val {
        JsonValue::Array(items) => {
            for item in &items {
                if !item.is_object() {
                    return Err(PyTypeError::new_err(
                        "each entry in 'tools' must be a dict/mapping",
                    ));
                }
            }
            Ok(items)
        }
        _ => Err(PyTypeError::new_err(
            "tools must be a sequence of tool dicts",
        )),
    }
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
#[pyclass(name = "NativeAutoModel")]
pub struct AutoModel {
    inner: Arc<Mutex<Option<LoadedSession>>>,
    loaded: Arc<AtomicBool>,
}

#[pymethods]
impl AutoModel {
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
        on_progress=None,
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
        on_progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        StreamEngine::set_log_verbose(verbose);
        let kv_type: KvCacheType = kv_cache_type
            .parse()
            .map_err(|e: anyhow::Error| PyValueError::new_err(e.to_string()))?;
        let (progress_cb, cb_err_slot) = build_progress_callback(on_progress)?;
        let (vram_budget_auto, vram_budget_bytes) = match vram_budget_mb {
            VramBudgetArg::Auto => (true, 0u64),
            VramBudgetArg::Bytes(b) => (false, b),
        };

        let model_owned = model.to_string();
        let open_opts = OpenSessionOptions {
            draft_model: draft_model.map(str::to_string),
            quant: quant.map(str::to_string),
            revision: revision.map(str::to_string),
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
            on_progress: progress_cb,
        };

        let loaded_res = py.detach(move || LoadedSession::open(&model_owned, open_opts));
        if let Ok(mut guard) = cb_err_slot.lock()
            && let Some(py_err) = guard.take()
        {
            return Err(py_err);
        }
        let loaded = loaded_res.map_err(map_open_err)?;

        Ok(Self {
            inner: Arc::new(Mutex::new(Some(loaded))),
            loaded: Arc::new(AtomicBool::new(true)),
        })
    }

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
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &self,
        py: Python<'_>,
        prompt: &str,
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
        let stop_seqs = parse_stop_sequences(stop)?;
        let resolved_grammar = resolve_grammar_arg(grammar, json_schema)?;
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
        let opts = GenerateOptions {
            max_tokens,
            sample,
            stop: stop_seqs,
            enable_ngram_speculation: ngram_speculation,
            max_draft_tokens: max_draft_tokens.unwrap_or(0),
            grammar: resolved_grammar,
        };
        self.run_generation(py, prompt.to_string(), opts, stream, false, return_summary)
    }

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
    #[allow(clippy::too_many_arguments)]
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
        let mut parsed_messages = parse_chat_messages(messages)?;
        let parsed_tools = parse_tools_arg(tools)?;
        inject_tools_into_messages(&mut parsed_messages, &parsed_tools);

        let stop_seqs = parse_stop_sequences(stop)?;
        let resolved_grammar = resolve_grammar_arg(grammar, json_schema)?;

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
            sample,
            stop: stop_seqs,
            enable_ngram_speculation: ngram_speculation,
            max_draft_tokens: max_draft_tokens.unwrap_or(0),
            grammar: resolved_grammar,
        };
        self.run_generation(py, prompt, opts, stream, parse_output, return_summary)
    }

    #[pyo3(signature = (text, *, add_special=false, parse_special=false))]
    fn tokenize(
        &self,
        py: Python<'_>,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> PyResult<Vec<i32>> {
        let text_owned = text.to_string();
        with_engine_checking_signals(py, &self.inner, |guard| {
            let loaded = guard
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("model has been unloaded"))?;
            loaded
                .engine
                .tokenize(&text_owned, add_special, parse_special)
        })
    }

    fn detokenize(&self, py: Python<'_>, tokens: Vec<i32>) -> PyResult<String> {
        with_engine_checking_signals(py, &self.inner, |guard| {
            let loaded = guard
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("model has been unloaded"))?;
            loaded.engine.detokenize(&tokens)
        })
    }

    #[pyo3(signature = (messages, *, add_generation_prompt=true, tools=None))]
    fn apply_chat_template(
        &self,
        py: Python<'_>,
        messages: &Bound<'_, PyAny>,
        add_generation_prompt: bool,
        tools: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let mut parsed_messages = parse_chat_messages(messages)?;
        let parsed_tools = parse_tools_arg(tools)?;
        inject_tools_into_messages(&mut parsed_messages, &parsed_tools);
        with_engine_checking_signals(py, &self.inner, |guard| {
            let loaded = guard
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("model has been unloaded"))?;
            loaded
                .engine
                .apply_chat_template(&parsed_messages, add_generation_prompt)
        })
    }

    fn kv_clear(&self, py: Python<'_>) -> PyResult<()> {
        with_engine_checking_signals(py, &self.inner, |guard| {
            let loaded = guard
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("model has been unloaded"))?;
            loaded.engine.kv_clear();
            if let Some(ref mut d) = loaded.draft {
                d.reset();
            }
            Ok(())
        })
    }

    fn kv_trim(&self, py: Python<'_>, pos_min: i32) -> PyResult<bool> {
        if pos_min < 0 {
            return Err(PyValueError::new_err("pos_min must be non-negative"));
        }
        with_engine_checking_signals(py, &self.inner, |guard| {
            let loaded = guard
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("model has been unloaded"))?;
            let ok = loaded.engine.kv_trim(pos_min);
            if ok && let Some(ref mut d) = loaded.draft {
                d.trim_to(pos_min as usize);
            }
            Ok(ok)
        })
    }

    fn save_state(&self, py: Python<'_>, path: PathBuf) -> PyResult<usize> {
        with_engine_checking_signals(py, &self.inner, |guard| {
            let loaded = guard
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("model has been unloaded"))?;
            loaded.engine.save_state(&path)
        })
    }

    fn load_state(&self, py: Python<'_>, path: PathBuf) -> PyResult<usize> {
        if !path.is_file() {
            return Err(PyFileNotFoundError::new_err(format!(
                "KV state file not found: '{}'",
                path.display()
            )));
        }
        with_engine_checking_signals(py, &self.inner, |guard| {
            let loaded = guard
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("model has been unloaded"))?;
            let n = loaded.engine.load_state(&path)?;
            if let Some(ref mut d) = loaded.draft {
                d.reset();
            }
            Ok(n)
        })
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self.inner.clone();
        let snapshot = py.detach(move || {
            let guard = inner.lock().ok()?;
            let loaded = guard.as_ref()?;
            Some(loaded.stats_snapshot())
        });
        match snapshot {
            Some(snap) => serialize_to_pydict(py, &snap),
            None => {
                let dict = PyDict::new(py);
                dict.set_item("loaded", false)?;
                Ok(dict)
            }
        }
    }

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
            Some(s) => Ok(Some(serialize_to_pydict(py, &s)?)),
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
            Some(p) => format!("AutoModel(loaded=True, isolated_worker=False, model_path={p:?})"),
            None => "AutoModel(loaded=False)".to_string(),
        }
    }
}

fn with_engine_checking_signals<T, F>(
    py: Python<'_>,
    inner: &Arc<Mutex<Option<LoadedSession>>>,
    mut f: F,
) -> PyResult<T>
where
    T: Send,
    F: FnMut(&mut Option<LoadedSession>) -> anyhow::Result<T> + Send,
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
    inner: &'a Mutex<Option<LoadedSession>>,
    cancel: &AtomicBool,
) -> anyhow::Result<Option<std::sync::MutexGuard<'a, Option<LoadedSession>>>> {
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
        parse_output: bool,
        return_summary: bool,
    ) -> PyResult<Py<PyAny>> {
        if !stream {
            let inner = self.inner.clone();
            let cancel = Arc::new(AtomicBool::new(false));
            let worker_cancel = cancel.clone();
            let (tx, rx) = mpsc::sync_channel::<anyhow::Result<(String, GenerateSummary)>>(1);
            let rx = Mutex::new(rx);

            let worker = std::thread::spawn(move || {
                let res = (|| -> anyhow::Result<(String, GenerateSummary)> {
                    let Some(mut guard) = lock_engine_with_cancel(&inner, &worker_cancel)? else {
                        return Ok((
                            String::new(),
                            GenerateSummary {
                                prompt_tokens: 0,
                                completion_tokens: 0,
                                finish_reason: "cancelled".to_string(),
                                speculation: Default::default(),
                            },
                        ));
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
                    loaded.last_summary = Some(summary.clone());
                    Ok((out, summary))
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
            let (output, summary) =
                result.map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
            let payload_obj = if parse_output {
                let parsed = parse_assistant_output(&output);
                serialize_to_py(py, &parsed)?
            } else {
                PyString::new(py, &output).into_any()
            };
            if return_summary {
                let sum_dict = serialize_to_pydict(py, &summary)?;
                return Ok((payload_obj, sum_dict)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind());
            }
            return Ok(payload_obj.unbind());
        }

        if !self.loaded.load(Ordering::Acquire) {
            return Err(PyRuntimeError::new_err("model has been unloaded"));
        }

        let (tx, rx) = mpsc::sync_channel::<StreamEvent>(64);
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let inner = self.inner.clone();

        let worker = std::thread::spawn(move || {
            let res = (|| -> anyhow::Result<Option<GenerateSummary>> {
                let Some(mut guard) = lock_engine_with_cancel(&inner, &worker_cancel)? else {
                    return Ok(None);
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
                loaded.last_summary = Some(summary.clone());
                Ok(Some(summary))
            })();

            match res {
                Ok(Some(summary)) => {
                    let _ = tx.send(StreamEvent::Done(summary));
                }
                Ok(None) => {}
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
                summary: None,
            }),
            cancel,
        };
        Ok(Py::new(py, stream_obj)?.into_any())
    }
}

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
        if let Ok(mut st) = self.inner.lock() {
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
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("server handle lock poisoned"))?;
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
                let guard = self.inner.lock().ok()?;
                if guard.finished.load(Ordering::Acquire) {
                    return Some(true);
                }
                Some(guard.worker.as_ref().is_some_and(|w| w.is_finished()))
            });
            if finished.unwrap_or(true) {
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
fn start_server(
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

    if let Ok(mut guard) = cb_err_slot.lock()
        && let Some(py_err) = guard.take()
    {
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

#[pyfunction]
#[pyo3(signature = (use_gpu=true))]
fn probe_gpu_memory(py: Python<'_>, use_gpu: bool) -> Option<(u64, u64)> {
    py.detach(|| StreamEngine::probe_gpu_memory(use_gpu))
}

#[pyfunction]
#[pyo3(signature = (use_gpu=true))]
fn probe_gpu_device(py: Python<'_>, use_gpu: bool) -> Option<(u64, u64, String)> {
    py.detach(|| StreamEngine::probe_gpu_device(use_gpu))
}

#[pyfunction]
fn inspect_gguf<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    if !path.is_file() {
        return Err(PyFileNotFoundError::new_err(format!(
            "local GGUF model not found: '{}'",
            path.display()
        )));
    }
    let info = py
        .detach(move || inspect_gguf_file(&path))
        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
    serialize_to_pydict(py, &info)
}

#[pyfunction]
#[pyo3(signature = (spec, *, cache_dir=None, quant=None, revision=None, on_progress=None))]
fn pull(
    py: Python<'_>,
    spec: &str,
    cache_dir: Option<PathBuf>,
    quant: Option<&str>,
    revision: Option<&str>,
    on_progress: Option<&Bound<'_, PyAny>>,
) -> PyResult<String> {
    let (progress_cb, cb_err_slot) = build_progress_callback(on_progress)?;
    let spec_owned = spec.to_string();
    let quant_owned = quant.map(str::to_string);
    let rev_owned = revision.map(str::to_string);
    let local_res =
        py.detach(move || pull_model(&spec_owned, cache_dir, quant_owned, rev_owned, progress_cb));
    if let Ok(mut guard) = cb_err_slot.lock()
        && let Some(py_err) = guard.take()
    {
        return Err(py_err);
    }
    let local = local_res.map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
    Ok(local.to_string_lossy().into_owned())
}

#[pyfunction]
#[pyo3(signature = (
    repo,
    *,
    revision=None,
    free_ram_gb=None,
    nvme_mib_s=None,
    has_draft=true,
    cache_dir=None,
    use_gpu=true,
))]
#[allow(clippy::too_many_arguments)]
fn recommend<'py>(
    py: Python<'py>,
    repo: &str,
    revision: Option<&str>,
    free_ram_gb: Option<f64>,
    nvme_mib_s: Option<f64>,
    has_draft: bool,
    cache_dir: Option<PathBuf>,
    use_gpu: bool,
) -> PyResult<Bound<'py, PyList>> {
    if let Some(gb) = free_ram_gb
        && (!gb.is_finite() || gb <= 0.0)
    {
        return Err(PyValueError::new_err("free_ram_gb must be positive"));
    }
    if let Some(bw) = nvme_mib_s
        && (!bw.is_finite() || bw <= 0.0)
    {
        return Err(PyValueError::new_err("nvme_mib_s must be positive"));
    }
    let repo_owned = repo.to_string();
    let opts = RecommendOptions {
        revision: revision.map(str::to_string),
        free_ram_gb,
        nvme_mib_s,
        has_draft,
        cache_dir,
        use_gpu,
    };
    let (_resolved_repo, recs) = py
        .detach(move || recommend_repo(&repo_owned, opts))
        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
    serialize_to_py(py, &recs)?
        .cast_into::<PyList>()
        .map_err(|_| PyRuntimeError::new_err("expected recommendation list"))
}

#[pyfunction]
#[pyo3(signature = (
    model,
    *,
    draft_model=None,
    quant=None,
    revision=None,
    cache_dir=None,
    use_gpu=true,
    allow_slow=false,
))]
#[allow(clippy::too_many_arguments)]
fn check<'py>(
    py: Python<'py>,
    model: &str,
    draft_model: Option<&str>,
    quant: Option<&str>,
    revision: Option<&str>,
    cache_dir: Option<PathBuf>,
    use_gpu: bool,
    allow_slow: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let model_owned = model.to_string();
    let opts = CheckOptions {
        draft_model: draft_model.map(str::to_string),
        quant: quant.map(str::to_string),
        revision: revision.map(str::to_string),
        cache_dir,
        use_gpu,
        allow_slow,
    };
    let report = py
        .detach(move || check_model(&model_owned, opts))
        .map_err(map_open_err)?;
    serialize_to_pydict(py, &report)
}

#[pyfunction(name = "json_schema_to_gbnf")]
fn py_json_schema_to_gbnf(schema: &Bound<'_, PyAny>) -> PyResult<String> {
    let val = extract_json_schema_value(schema)?;
    json_schema_to_gbnf(&val).map_err(|e| PyValueError::new_err(format!("{e:#}")))
}

#[pyfunction(name = "json_object_gbnf")]
fn py_json_object_gbnf() -> String {
    json_object_gbnf()
}

#[pyfunction(name = "parse_assistant_output")]
fn py_parse_assistant_output<'py>(py: Python<'py>, raw: &str) -> PyResult<Bound<'py, PyDict>> {
    let parsed = parse_assistant_output(raw);
    serialize_to_pydict(py, &parsed)
}

#[pyfunction(name = "format_tools_system_prompt")]
fn py_format_tools_system_prompt(tools: &Bound<'_, PyAny>) -> PyResult<String> {
    let parsed = parse_tools_arg(Some(tools))?;
    Ok(format_tools_system_prompt(&parsed))
}

fn map_worker_err(err: anyhow::Error) -> PyErr {
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
struct PyWorkerStream {
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
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("worker stream lock poisoned"))?;
        let Some(mut stream) = guard.take() else {
            return Ok(None);
        };
        drop(guard);

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
        if let Ok(mut guard) = self.inner.lock() {
            *guard = Some(stream);
        }
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
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("worker stream lock poisoned"))?;
        let Some(mut stream) = guard.take() else {
            return Ok(());
        };
        drop(guard);

        let stream = py.detach(move || {
            stream.close();
            stream
        });
        if let Ok(mut guard) = self.inner.lock() {
            *guard = Some(stream);
        }
        Ok(())
    }

    #[getter]
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("worker stream lock poisoned"))?;
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
struct PyWorkerClient {
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
                .map_err(|e| anyhow::anyhow!("{e}"))
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

#[pyfunction(name = "compile_response_format")]
#[pyo3(signature = (response_format, allow_raw_schema=false))]
fn py_compile_response_format(
    response_format: &Bound<'_, PyAny>,
    allow_raw_schema: bool,
) -> PyResult<(Option<String>, bool)> {
    let val = pyany_to_json_value(response_format, 0)?;
    let compiled = compile_response_format(&val, allow_raw_schema)
        .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
    Ok((compiled.grammar, compiled.expects_json))
}

#[pyfunction(name = "validate_sample_params")]
#[pyo3(signature = (temperature, top_p, min_p, repeat_penalty, frequency_penalty, presence_penalty, penalty_last_n=64))]
fn py_validate_sample_params(
    temperature: f64,
    top_p: f64,
    min_p: f64,
    repeat_penalty: f64,
    frequency_penalty: f64,
    presence_penalty: f64,
    penalty_last_n: i32,
) -> PyResult<()> {
    validate_sample_params(
        temperature,
        top_p,
        min_p,
        repeat_penalty,
        frequency_penalty,
        presence_penalty,
        penalty_last_n,
    )
    .map_err(|e| PyValueError::new_err(format!("{e:#}")))
}

#[pyfunction(name = "normalize_open_budget_and_kv")]
fn py_normalize_open_budget_and_kv(
    vram_budget_mb: &Bound<'_, PyAny>,
    kv_cache_type: &str,
) -> PyResult<(bool, u64, &'static str)> {
    if vram_budget_mb.is_instance_of::<pyo3::types::PyBool>() {
        return Err(PyTypeError::new_err(
            "vram_budget_mb must be 'auto', None, or a non-negative integer MiB (not bool)",
        ));
    }
    let val = pyany_to_json_value(vram_budget_mb, 0)?;
    let (auto, bytes) =
        normalize_vram_budget_mb(&val).map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
    let kv = normalize_kv_cache_type(kv_cache_type)
        .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
    Ok((auto, bytes, kv))
}

#[pyfunction(name = "run_worker_server")]
fn py_run_worker_server(py: Python<'_>, port: u16) -> PyResult<()> {
    py.detach(move || run_worker_server(port))
        .map_err(map_worker_err)
}

#[pymodule]
fn _layerstream(m: &Bound<'_, PyModule>) -> PyResult<()> {
    StreamEngine::set_log_verbose(false);
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<AutoModel>()?;
    m.add_class::<TokenStream>()?;
    m.add_class::<PyWorkerClient>()?;
    m.add_class::<PyWorkerStream>()?;
    m.add_class::<PyStreamTagParser>()?;
    m.add_class::<ServerHandle>()?;
    m.add_function(wrap_pyfunction!(py_run_worker_server, m)?)?;
    m.add_function(wrap_pyfunction!(start_server, m)?)?;
    m.add_function(wrap_pyfunction!(probe_gpu_memory, m)?)?;
    m.add_function(wrap_pyfunction!(probe_gpu_device, m)?)?;
    m.add_function(wrap_pyfunction!(inspect_gguf, m)?)?;
    m.add_function(wrap_pyfunction!(pull, m)?)?;
    m.add_function(wrap_pyfunction!(recommend, m)?)?;
    m.add_function(wrap_pyfunction!(check, m)?)?;
    m.add_function(wrap_pyfunction!(py_json_schema_to_gbnf, m)?)?;
    m.add_function(wrap_pyfunction!(py_json_object_gbnf, m)?)?;
    m.add_function(wrap_pyfunction!(py_compile_response_format, m)?)?;
    m.add_function(wrap_pyfunction!(py_validate_sample_params, m)?)?;
    m.add_function(wrap_pyfunction!(py_normalize_open_budget_and_kv, m)?)?;
    m.add_function(wrap_pyfunction!(py_parse_assistant_output, m)?)?;
    m.add_function(wrap_pyfunction!(py_format_tools_system_prompt, m)?)?;
    Ok(())
}
