use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::Duration;

use layerstream_core::hf::{DownloadProgress, HfError, ProgressCallback};
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
    parse_assistant_output,
};
use parking_lot::{Mutex, MutexGuard};
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
        let handle = {
            let mut st = self.state.lock();
            st.rx = None;
            st.done = true;
            st.worker.take()
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
            let mut st = self.state.lock();
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
        let sum = self.state.lock().summary.clone();
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
                let st = self.state.lock();
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
                    self.state.lock().summary = Some(summary);
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
        let handle = {
            let mut st = self.state.lock();
            st.done = true;
            st.rx = None;
            st.worker.take()
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
        Ok(self.inner.lock().has_tool_calls())
    }

    fn push<'py>(&self, py: Python<'py>, piece: &str) -> PyResult<Bound<'py, PyList>> {
        let deltas = self.inner.lock().push(piece);
        deltas_to_pylist(py, &deltas)
    }

    fn flush<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let deltas = self.inner.lock().flush();
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
pub(crate) enum VramBudgetArg {
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

pub(crate) fn build_progress_callback(
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
        Python::attach(|py| -> layerstream_core::hf::Result<()> {
            let call_res: PyResult<()> = (|| {
                let d = serialize_to_pydict(py, ev)?;
                py_cb.bind(py).call1((d,))?;
                Ok(())
            })();
            if let Err(py_err) = call_res {
                *slot_clone.lock() = Some(py_err);
                return Err(HfError::Cancelled(
                    "on_progress callback raised Python exception".into(),
                ));
            }
            Ok(())
        })
    });
    Ok((Some(cb), err_slot))
}

fn parse_incoming_chat_messages(messages: &Bound<'_, PyAny>) -> PyResult<Vec<IncomingChatMessage>> {
    if messages.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "messages must be an iterable of {'role': str, 'content': str} dicts",
        ));
    }
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

        out.push(IncomingChatMessage {
            role,
            content: content_val,
            tool_calls,
            tool_call_id,
            name,
        });
    }
    if out.is_empty() {
        return Err(PyValueError::new_err("messages must not be empty"));
    }
    Ok(out)
}

fn parse_chat_messages(messages: &Bound<'_, PyAny>) -> PyResult<Vec<ChatMessage>> {
    parse_incoming_chat_messages(messages)?
        .into_iter()
        .map(|incoming| incoming.to_chat_message().map_err(PyValueError::new_err))
        .collect()
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

pub(crate) fn map_open_err(err: anyhow::Error) -> PyErr {
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
        if let Some(py_err) = cb_err_slot.lock().take() {
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
            let guard = inner.lock();
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
        let summary = py.detach(move || inner.lock().as_ref().and_then(|l| l.last_summary.clone()));
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
            inner
                .lock()
                .as_ref()
                .map(|l| l.model_path.to_string_lossy().into_owned())
        })
    }

    fn unload(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        let loaded = self.loaded.clone();
        py.detach(move || {
            let dropped = {
                let mut guard = inner.lock();
                loaded.store(false, Ordering::Release);
                guard.take()
            };
            drop(dropped);
        });
        Ok(())
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
            Some(mut guard) => Step::Ready(f(&mut guard)),
            None => {
                std::thread::sleep(Duration::from_millis(5));
                Step::Busy
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
) -> Option<MutexGuard<'a, Option<LoadedSession>>> {
    loop {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        if let Some(guard) = inner.try_lock() {
            return Some(guard);
        }
        std::thread::sleep(Duration::from_millis(5));
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
                    let Some(mut guard) = lock_engine_with_cancel(&inner, &worker_cancel) else {
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
                match py.detach(|| rx.lock().recv_timeout(Duration::from_millis(25))) {
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
                let Some(mut guard) = lock_engine_with_cancel(&inner, &worker_cancel) else {
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

mod server;
use server::{ServerHandle, start_server};

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
    if let Some(py_err) = cb_err_slot.lock().take() {
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

mod cli;
mod worker;
use worker::{PyWorkerClient, PyWorkerStream, map_worker_err};

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
    m.add_function(wrap_pyfunction!(cli::run_cli, m)?)?;
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
    m.add_function(wrap_pyfunction!(py_parse_assistant_output, m)?)?;
    m.add_function(wrap_pyfunction!(py_format_tools_system_prompt, m)?)?;
    Ok(())
}
