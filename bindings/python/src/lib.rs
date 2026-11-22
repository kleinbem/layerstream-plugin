use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use layerstream_core::hf::{self, PretrainedResolveOptions, resolve_pretrained_model};
use layerstream_core::{
    ChatMessage, DraftModelProposer, EngineConfig, GenerateOptions, GenerateSummary, GgufIndex,
    KvCacheType, LsSampleParams, ParsedAssistantOutput, StreamDelta, StreamEngine, StreamTagParser,
    check_gguf_draft_compatible, format_tools_system_prompt, json_object_gbnf, json_schema_to_gbnf,
    parse_assistant_output, probe_dir_storage_mib_per_s, probe_storage_mib_per_s,
};
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyDict, PyFloat, PyInt, PyList, PyMapping, PySequence, PyString, PyType,
};
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};

const MIN_STREAMING_STORAGE_MIB_S: f64 = 150.0;
const OVERSIZE_GUARD_BYTES: u64 = 1024 * 1024 * 1024;

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
    Done(GenerateSummary),
    Err(String),
}

struct StreamState {
    rx: Option<mpsc::Receiver<StreamEvent>>,
    worker: Option<JoinHandle<()>>,
    done: bool,
    summary: Option<GenerateSummary>,
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
            Some(s) => Ok(Some(summary_to_pydict(py, &s)?)),
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

fn parsed_output_to_pydict<'py>(
    py: Python<'py>,
    parsed: &ParsedAssistantOutput,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("content", &parsed.content)?;
    d.set_item("reasoning_content", parsed.reasoning_content.as_deref())?;
    let calls = PyList::empty(py);
    for tc in &parsed.tool_calls {
        let c = PyDict::new(py);
        c.set_item("id", &tc.id)?;
        c.set_item("type", &tc.kind)?;
        let f = PyDict::new(py);
        f.set_item("name", &tc.function.name)?;
        f.set_item("arguments", &tc.function.arguments)?;
        c.set_item("function", f)?;
        calls.append(c)?;
    }
    d.set_item("tool_calls", calls)?;
    Ok(d)
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
    // Support Pydantic v2 (`model_json_schema`) and v1 (`schema`) model classes or instances.
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
        let trimmed = g.trim();
        if trimmed.is_empty() {
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

        // Support OpenAI assistant tool-call / tool-result messages where `content` may be None
        // or a list of `{"type": "text", "text": ...}` content parts.
        let content_obj = match item.get_item("content") {
            Ok(c) => Some(c),
            Err(_) if item.get_item("tool_calls").is_ok() => None,
            Err(_) => {
                return Err(PyValueError::new_err(
                    "each message must contain a 'content' string",
                ));
            }
        };
        let mut content = match content_obj {
            None => String::new(),
            Some(c) if c.is_none() => String::new(),
            Some(c) => {
                if let Ok(s) = c.cast::<PyString>() {
                    s.extract::<String>()?
                } else if let Ok(seq) = c.cast::<PySequence>() {
                    let mut parts = Vec::with_capacity(seq.len()?);
                    for idx in 0..seq.len()? {
                        let part = seq.get_item(idx)?;
                        if let Ok(ps) = part.cast::<PyString>() {
                            parts.push(ps.extract::<String>()?);
                        } else if let Ok(text_field) = part.get_item("text") {
                            parts.push(text_field.extract::<String>().map_err(|_| {
                                PyTypeError::new_err("content part 'text' must be a string")
                            })?);
                        }
                    }
                    parts.join("")
                } else {
                    return Err(PyTypeError::new_err(
                        "message 'content' must be a string, None, or list of text parts",
                    ));
                }
            }
        };

        // Re-serialize assistant `tool_calls` into `<tool_call>` blocks for multi-turn history.
        if role == "assistant"
            && let Ok(tc_obj) = item.get_item("tool_calls")
            && !tc_obj.is_none()
            && let Ok(tc_seq) = tc_obj.cast::<PySequence>()
        {
            for idx in 0..tc_seq.len()? {
                let tc = tc_seq.get_item(idx)?;
                if let Ok(func) = tc.get_item("function") {
                    let name: String = func.get_item("name")?.extract()?;
                    let args_any = func.get_item("arguments")?;
                    let args_val = if let Ok(s) = args_any.cast::<PyString>() {
                        let raw = s.extract::<String>()?;
                        serde_json::from_str::<JsonValue>(&raw).unwrap_or(JsonValue::String(raw))
                    } else {
                        pyany_to_json_value(&args_any, 0)?
                    };
                    let call_json = serde_json::json!({
                        "name": name,
                        "arguments": args_val,
                    });
                    if !content.is_empty() && !content.ends_with('\n') {
                        content.push('\n');
                    }
                    content.push_str(&format!("<tool_call>{call_json}</tool_call>"));
                }
            }
        }

        if role == "tool" {
            let mut prefix = String::new();
            if let Ok(name_obj) = item.get_item("name")
                && !name_obj.is_none()
                && let Ok(name_str) = name_obj.extract::<String>()
                && !name_str.is_empty()
            {
                prefix.push_str(&format!("tool={name_str} "));
            }
            if let Ok(id_obj) = item.get_item("tool_call_id")
                && !id_obj.is_none()
                && let Ok(id_str) = id_obj.extract::<String>()
                && !id_str.is_empty()
            {
                prefix.push_str(&format!("call_id={id_str}: "));
            }
            if !prefix.is_empty() {
                content = format!("{prefix}{content}");
            }
        }
        out.push(ChatMessage { role, content });
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

fn inject_tools_into_messages(messages: &mut Vec<ChatMessage>, tools: &[JsonValue]) {
    if tools.is_empty() {
        return;
    }
    let tool_prompt = format_tools_system_prompt(tools);
    if let Some(first) = messages.first_mut()
        && first.role == "system"
    {
        if !first.content.is_empty() {
            first.content.push_str("\n\n");
        }
        first.content.push_str(&tool_prompt);
    } else {
        messages.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: tool_prompt,
            },
        );
    }
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
#[pyclass(name = "NativeAutoModel")]
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
        grammar=None,
        json_schema=None,
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
        grammar: Option<&str>,
        json_schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let stop_seqs = parse_stop_sequences(stop)?;
        let resolved_grammar = resolve_grammar_arg(grammar, json_schema)?;
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
            grammar: resolved_grammar,
        };
        self.run_generation(py, prompt.to_string(), opts, stream, false, false)
    }

    #[pyo3(signature = (
        prompt,
        *,
        max_tokens=256,
        temperature=0.0,
        top_p=0.95,
        top_k=40,
        seed=42,
        stop=None,
        ngram_speculation=true,
        max_draft_tokens=None,
        grammar=None,
        json_schema=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn generate_with_summary(
        &self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        seed: u32,
        stop: Option<&Bound<'_, PyAny>>,
        ngram_speculation: bool,
        max_draft_tokens: Option<usize>,
        grammar: Option<&str>,
        json_schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let stop_seqs = parse_stop_sequences(stop)?;
        let resolved_grammar = resolve_grammar_arg(grammar, json_schema)?;
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
            grammar: resolved_grammar,
        };
        self.run_generation(py, prompt.to_string(), opts, false, false, true)
    }

    /// Apply the model's embedded chat template to `messages` (`[{"role": ..., "content": ...}]`)
    /// and generate the assistant reply. Returns a `str`, parsed `dict` (`parse_output=True`),
    /// or `TokenStream` (`stream=True`).
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
        tools=None,
        grammar=None,
        json_schema=None,
        parse_output=false,
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
        tools: Option<&Bound<'_, PyAny>>,
        grammar: Option<&str>,
        json_schema: Option<&Bound<'_, PyAny>>,
        parse_output: bool,
    ) -> PyResult<Py<PyAny>> {
        if stream && parse_output {
            return Err(PyValueError::new_err(
                "parse_output=True cannot be combined with stream=True (use StreamTagParser or layerstream.OpenAI)",
            ));
        }
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
            sample: LsSampleParams {
                temperature,
                top_p,
                top_k,
                seed,
            },
            stop: stop_seqs,
            enable_ngram_speculation: ngram_speculation,
            max_draft_tokens: max_draft_tokens.unwrap_or(0),
            grammar: resolved_grammar,
        };
        self.run_generation(py, prompt, opts, stream, parse_output, false)
    }

    #[pyo3(signature = (
        messages,
        *,
        max_tokens=256,
        temperature=0.0,
        top_p=0.95,
        top_k=40,
        seed=42,
        stop=None,
        add_generation_prompt=true,
        ngram_speculation=true,
        max_draft_tokens=None,
        tools=None,
        grammar=None,
        json_schema=None,
        parse_output=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn chat_with_summary(
        &self,
        py: Python<'_>,
        messages: &Bound<'_, PyAny>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        seed: u32,
        stop: Option<&Bound<'_, PyAny>>,
        add_generation_prompt: bool,
        ngram_speculation: bool,
        max_draft_tokens: Option<usize>,
        tools: Option<&Bound<'_, PyAny>>,
        grammar: Option<&str>,
        json_schema: Option<&Bound<'_, PyAny>>,
        parse_output: bool,
    ) -> PyResult<Py<PyAny>> {
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
            sample: LsSampleParams {
                temperature,
                top_p,
                top_k,
                seed,
            },
            stop: stop_seqs,
            enable_ngram_speculation: ngram_speculation,
            max_draft_tokens: max_draft_tokens.unwrap_or(0),
            grammar: resolved_grammar,
        };
        self.run_generation(py, prompt, opts, false, parse_output, true)
    }

    /// Tokenize `text` into vocabulary token IDs using the loaded model's tokenizer.
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

    /// Convert a sequence of token IDs back into a UTF-8 string.
    fn detokenize(&self, py: Python<'_>, tokens: Vec<i32>) -> PyResult<String> {
        with_engine_checking_signals(py, &self.inner, |guard| {
            let loaded = guard
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("model has been unloaded"))?;
            loaded.engine.detokenize(&tokens)
        })
    }

    /// Format `messages` with the model's embedded chat template (or ChatML fallback).
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

    /// Clear the target (and draft) KV prefix cache.
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

    /// Roll back KV cache positions `>= pos_min` (for multi-branch prefix sharing).
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
        dict.set_item("isolated_worker", false)?;
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
            Some(p) => format!("AutoModel(loaded=True, isolated_worker=False, model_path={p:?})"),
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
                parsed_output_to_pydict(py, &parsed)?.into_any()
            } else {
                PyString::new(py, &output).into_any()
            };
            if return_summary {
                let sum_dict = summary_to_pydict(py, &summary)?;
                return Ok((payload_obj, sum_dict)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind());
            }
            return Ok(payload_obj.unbind());
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

/// Probe `(free_bytes, total_bytes)` on the primary GPU backend, or `None` if unavailable.
#[pyfunction]
#[pyo3(signature = (use_gpu=true))]
fn probe_gpu_memory(py: Python<'_>, use_gpu: bool) -> Option<(u64, u64)> {
    py.detach(|| StreamEngine::probe_gpu_memory(use_gpu))
}

/// Parse GGUF v2/v3 metadata headers in pure Rust (0 MB GPU memory, ~milliseconds).
#[pyfunction]
fn inspect_gguf<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    if !path.is_file() {
        return Err(PyFileNotFoundError::new_err(format!(
            "local GGUF model not found: '{}'",
            path.display()
        )));
    }
    let idx = py
        .detach(move || GgufIndex::open(&path))
        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;

    let d = PyDict::new(py);
    d.set_item("architecture", &idx.meta.architecture)?;
    d.set_item("name", idx.meta.name.as_deref())?;
    d.set_item("n_layers", idx.meta.n_layers)?;
    d.set_item("context_length", idx.meta.context_length)?;
    d.set_item("total_file_bytes", idx.meta.total_file_bytes)?;
    d.set_item("shard_count", idx.shards.len())?;
    d.set_item("tensor_count", idx.tensors.len())?;
    Ok(d)
}

/// Download an `hf://org/repo:file.gguf` URI (including multi-part shards) into the local cache.
#[pyfunction]
#[pyo3(signature = (spec, *, cache_dir=None))]
fn pull(py: Python<'_>, spec: &str, cache_dir: Option<PathBuf>) -> PyResult<String> {
    let spec_owned = spec.to_string();
    let local = py
        .detach(move || -> anyhow::Result<PathBuf> {
            let root = cache_dir.unwrap_or_else(hf::default_cache_dir);
            hf::ensure_hf_downloaded(&spec_owned, &root)
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
    Ok(local.to_string_lossy().into_owned())
}

/// Query a Hugging Face repository (or base repo with an official `-GGUF` sibling) and rank
/// GGUF quantizations against probed (or overridden) GPU/UMA memory and NVMe bandwidth.
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
    let rev_owned = revision.map(str::to_string);

    let (effective_repo, recs) = py
        .detach(
            move || -> anyhow::Result<(String, Vec<hf::QuantRecommendation>)> {
                let (repo_id, rev) =
                    hf::parse_repo_and_revision(&repo_owned, rev_owned.as_deref())?;
                let mut effective_repo = repo_id.clone();
                let mut candidates = hf::list_hf_gguf_candidates(&repo_id, &rev)?;

                if candidates.is_empty() && !repo_id.to_ascii_lowercase().ends_with("-gguf") {
                    let sibling = format!("{repo_id}-GGUF");
                    if let Ok(sib_cands) = hf::list_hf_gguf_candidates(&sibling, &rev)
                        && !sib_cands.is_empty()
                    {
                        effective_repo = sibling;
                        candidates = sib_cands;
                    }
                }
                if candidates.is_empty() {
                    anyhow::bail!(
                        "no .gguf files found in Hugging Face repository '{repo_id}@{rev}'"
                    );
                }

                let free_vram_bytes = if let Some(gb) = free_ram_gb {
                    (gb.max(0.5) * 1024.0 * 1024.0 * 1024.0) as u64
                } else if use_gpu && let Some((free, _)) = StreamEngine::probe_gpu_memory(true) {
                    free
                } else {
                    4 * 1024 * 1024 * 1024
                };

                let storage_bw = if let Some(bw) = nvme_mib_s {
                    bw.max(10.0)
                } else {
                    let probe_dir = cache_dir.unwrap_or_else(hf::default_cache_dir);
                    let _ = std::fs::create_dir_all(&probe_dir);
                    probe_dir_storage_mib_per_s(&probe_dir).unwrap_or(1500.0)
                };

                let recs = hf::recommend_candidates(
                    &effective_repo,
                    &rev,
                    &candidates,
                    free_vram_bytes,
                    storage_bw,
                    has_draft,
                );
                Ok((effective_repo, recs))
            },
        )
        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;

    let out = PyList::empty(py);
    for r in &recs {
        let d = PyDict::new(py);
        d.set_item("resolved_repo", &effective_repo)?;
        d.set_item("quant", &r.candidate.quant)?;
        d.set_item("primary_file", &r.candidate.primary_file)?;
        d.set_item("total_bytes", r.candidate.total_bytes)?;
        d.set_item("shard_count", r.candidate.shard_count)?;
        d.set_item("hf_uri", &r.hf_uri)?;
        d.set_item("verdict", &r.verdict)?;
        d.set_item("pinned_pct", r.pinned_pct)?;
        d.set_item("est_load_s", r.est_load_s)?;
        d.set_item("est_tps_low", r.est_tps_low)?;
        d.set_item("est_tps_high", r.est_tps_high)?;
        d.set_item("recommended", r.recommended)?;
        d.set_item("score", r.score)?;
        out.append(d)?;
    }
    Ok(out)
}

/// Run fail-closed preflight diagnostics (GGUF metadata, GPU VRAM, NVMe O_DIRECT throughput,
/// draft tokenizer compatibility, and estimated decode throughput) without loading weights.
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
    struct CheckReport {
        model_path: PathBuf,
        draft_path: Option<PathBuf>,
        idx: GgufIndex,
        gpu_mem: Option<(u64, u64)>,
        mib_s: f64,
        direct_ok: bool,
        fits_in_vram: bool,
        verdict: &'static str,
        summary: String,
        est_tps_low: f64,
        est_tps_high: f64,
    }

    let model_owned = model.to_string();
    let draft_owned = draft_model.map(str::to_string);
    let quant_owned = quant.map(str::to_string);
    let rev_owned = revision.map(str::to_string);

    let report = py
        .detach(move || -> anyhow::Result<CheckReport> {
            let resolve_opts = PretrainedResolveOptions {
                revision: rev_owned.clone(),
                quant: quant_owned,
                cache_dir: cache_dir.clone(),
                vram_budget_bytes: None,
                use_gpu,
                has_draft: draft_owned.is_some(),
            };
            let resolved_target = resolve_pretrained_model(&model_owned, &resolve_opts)?;
            let idx = GgufIndex::open(&resolved_target.local_path)?;

            let (draft_path, draft_reserve) = if let Some(d_spec) = draft_owned {
                let d_opts = PretrainedResolveOptions {
                    revision: rev_owned,
                    quant: None,
                    cache_dir,
                    vram_budget_bytes: None,
                    use_gpu,
                    has_draft: false,
                };
                let resolved_draft = resolve_pretrained_model(&d_spec, &d_opts)?;
                check_gguf_draft_compatible(&resolved_target.local_path, &resolved_draft.local_path)
                    .map_err(|e| anyhow::anyhow!("incompatible draft_model tokenizer: {e}"))?;
                let draft_idx = GgufIndex::open(&resolved_draft.local_path)?;
                let reserve = draft_idx
                    .meta
                    .total_file_bytes
                    .saturating_add(64 * 1024 * 1024);
                (Some(resolved_draft.local_path), reserve)
            } else {
                (None, 0u64)
            };

            let gpu_mem = if use_gpu {
                let probed = StreamEngine::probe_gpu_memory(true);
                if probed.is_none() {
                    anyhow::bail!("no usable GPU device found (CUDA/Vulkan/Metal); pass use_gpu=False for CPU mode");
                }
                probed
            } else {
                None
            };

            let size_bytes = idx.meta.total_file_bytes;
            let size_gib = size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let fits_in_vram = gpu_mem
                .map(|(free_b, _)| {
                    size_bytes
                        .saturating_add(draft_reserve)
                        .saturating_add(768 * 1024 * 1024)
                        < free_b
                })
                .unwrap_or(false);

            let probe_res = probe_storage_mib_per_s(&resolved_target.local_path, true).ok();
            let (mib_s, direct_ok) = probe_res.unwrap_or((0.0, false));
            let is_oversize = size_bytes >= OVERSIZE_GUARD_BYTES && !fits_in_vram;

            if is_oversize && use_gpu && !allow_slow {
                if probe_res.is_none() || !mib_s.is_finite() || mib_s <= 0.0 {
                    anyhow::bail!(
                        "could not measure storage throughput for '{}' ({size_gib:.1} GiB) \
                         required for out-of-core GPU streaming; pass allow_slow=True to bypass preflight.",
                        resolved_target.local_path.display()
                    );
                }
                if !direct_ok {
                    anyhow::bail!(
                        "storage throughput for '{}' ({size_gib:.1} GiB) is unverified because \
                         direct I/O (O_DIRECT/F_NOCACHE) was unavailable (measured {mib_s:.0} MiB/s from page cache). \
                         Move the GGUF to a direct-I/O filesystem or pass allow_slow=True.",
                        resolved_target.local_path.display()
                    );
                }
                if mib_s < MIN_STREAMING_STORAGE_MIB_S {
                    anyhow::bail!(
                        "storage for '{}' ({size_gib:.1} GiB) measured only {mib_s:.0} MiB/s \
                         (minimum recommended {MIN_STREAMING_STORAGE_MIB_S:.0} MiB/s NVMe). \
                         Move the GGUF to local NVMe or pass allow_slow=True.",
                        resolved_target.local_path.display()
                    );
                }
            }

            let pinned_mib = gpu_mem.map_or(0.0, |(f, _)| {
                (f.saturating_sub(draft_reserve) as f64 * 0.45) / (1024.0 * 1024.0)
            });
            let est_stream_mib = ((size_bytes as f64 / (1024.0 * 1024.0)) - pinned_mib).max(0.0);
            let base_tps = if est_stream_mib > 1.0 && mib_s > 1.0 {
                mib_s / est_stream_mib
            } else {
                0.0
            };

            let (verdict, est_tps_low, est_tps_high, summary) = if fits_in_vram {
                (
                    "GOOD",
                    15.0,
                    50.0,
                    format!("GOOD (100% resident in VRAM, {size_gib:.2} GiB)"),
                )
            } else if draft_path.is_some() {
                let lo = (base_tps * 2.5 * 100.0).round() / 100.0;
                let hi = (base_tps * 5.0 * 100.0).round() / 100.0;
                (
                    "GOOD",
                    lo,
                    hi,
                    format!(
                        "GOOD ({size_gib:.2} GiB streamed + draft speculator, storage={mib_s:.0} MiB/s, direct_io={direct_ok}, est ~{lo:.2}-{hi:.2} tok/s)"
                    ),
                )
            } else if size_bytes >= 4 * 1024 * 1024 * 1024 {
                let tps = (base_tps * 100.0).round() / 100.0;
                (
                    "SLOW",
                    tps,
                    tps,
                    format!(
                        "SLOW ({size_gib:.2} GiB streamed without draft_model, storage={mib_s:.0} MiB/s, est ~{tps:.2} tok/s)"
                    ),
                )
            } else {
                let tps = (base_tps * 100.0).round() / 100.0;
                (
                    "OK",
                    tps,
                    tps,
                    format!("OK ({size_gib:.2} GiB, storage={mib_s:.0} MiB/s, direct_io={direct_ok})"),
                )
            };

            Ok(CheckReport {
                model_path: resolved_target.local_path,
                draft_path,
                idx,
                gpu_mem,
                mib_s,
                direct_ok,
                fits_in_vram,
                verdict,
                summary,
                est_tps_low,
                est_tps_high,
            })
        })
        .map_err(map_open_err)?;

    let d = PyDict::new(py);
    d.set_item("ok", true)?;
    d.set_item("verdict", report.verdict)?;
    d.set_item("summary", &report.summary)?;
    d.set_item(
        "model_path",
        report.model_path.to_string_lossy().into_owned(),
    )?;
    d.set_item(
        "draft_model_path",
        report
            .draft_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()),
    )?;
    d.set_item("architecture", &report.idx.meta.architecture)?;
    d.set_item("n_layers", report.idx.meta.n_layers)?;
    d.set_item("context_length", report.idx.meta.context_length)?;
    d.set_item("total_file_bytes", report.idx.meta.total_file_bytes)?;
    d.set_item("storage_mib_s", report.mib_s)?;
    d.set_item("direct_io", report.direct_ok)?;
    d.set_item("gpu_free_bytes", report.gpu_mem.map(|(f, _)| f))?;
    d.set_item("gpu_total_bytes", report.gpu_mem.map(|(_, t)| t))?;
    d.set_item("fits_in_vram", report.fits_in_vram)?;
    d.set_item("est_tps_low", report.est_tps_low)?;
    d.set_item("est_tps_high", report.est_tps_high)?;
    Ok(d)
}

/// Compile a JSON Schema (`dict`, JSON `str`, or Pydantic `BaseModel` subclass) into a GBNF grammar.
#[pyfunction(name = "json_schema_to_gbnf")]
fn py_json_schema_to_gbnf(schema: &Bound<'_, PyAny>) -> PyResult<String> {
    let val = extract_json_schema_value(schema)?;
    json_schema_to_gbnf(&val).map_err(|e| PyValueError::new_err(format!("{e:#}")))
}

/// Return a built-in GBNF grammar constraining output to any valid RFC 8259 JSON object (`{...}`).
#[pyfunction(name = "json_object_gbnf")]
fn py_json_object_gbnf() -> String {
    json_object_gbnf()
}

/// Parse `<think>...</think>` reasoning blocks and `<tool_call>...</tool_call>` JSON payloads
/// from raw assistant output text.
#[pyfunction(name = "parse_assistant_output")]
fn py_parse_assistant_output<'py>(py: Python<'py>, raw: &str) -> PyResult<Bound<'py, PyDict>> {
    let parsed = parse_assistant_output(raw);
    parsed_output_to_pydict(py, &parsed)
}

/// Format OpenAI `tools` definitions into a `<tool_call>` system prompt block.
#[pyfunction(name = "format_tools_system_prompt")]
fn py_format_tools_system_prompt(tools: &Bound<'_, PyAny>) -> PyResult<String> {
    let parsed = parse_tools_arg(Some(tools))?;
    Ok(format_tools_system_prompt(&parsed))
}

#[pymodule]
fn _layerstream(m: &Bound<'_, PyModule>) -> PyResult<()> {
    StreamEngine::set_log_verbose(false);
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<AutoModel>()?;
    m.add_class::<TokenStream>()?;
    m.add_class::<PyStreamTagParser>()?;
    m.add_function(wrap_pyfunction!(probe_gpu_memory, m)?)?;
    m.add_function(wrap_pyfunction!(inspect_gguf, m)?)?;
    m.add_function(wrap_pyfunction!(pull, m)?)?;
    m.add_function(wrap_pyfunction!(recommend, m)?)?;
    m.add_function(wrap_pyfunction!(check, m)?)?;
    m.add_function(wrap_pyfunction!(py_json_schema_to_gbnf, m)?)?;
    m.add_function(wrap_pyfunction!(py_json_object_gbnf, m)?)?;
    m.add_function(wrap_pyfunction!(py_parse_assistant_output, m)?)?;
    m.add_function(wrap_pyfunction!(py_format_tools_system_prompt, m)?)?;
    Ok(())
}
