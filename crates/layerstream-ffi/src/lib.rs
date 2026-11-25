//! C ABI shared/static library for `layerstream` (Go, C/C++, Swift, C#, Zig).
//!
//! All public `extern "C"` functions catch Rust panics, report errors through the
//! thread-local [`layerstream_last_error`] slot, and return heap strings that the
//! caller frees with [`layerstream_string_free`].

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use layerstream_core::sdk::{LoadedSession, OpenSessionOptions, inspect_gguf_file};
use layerstream_core::{
    GenerateOptions, IncomingChatMessage, KvCacheType, LsSampleParams, StreamEngine,
    inject_tools_into_messages, json_object_gbnf, json_schema_to_gbnf, normalize_kv_cache_type,
    normalize_vram_budget_mb, parse_assistant_output, validate_sample_params,
};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};

const VERSION_CSTR: &CStr =
    match CStr::from_bytes_with_nul(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes()) {
        Ok(s) => s,
        Err(_) => panic!("CARGO_PKG_VERSION contains interior NUL"),
    };

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

fn set_last_error(msg: impl Into<String>) {
    let raw = msg.into().replace('\0', " ");
    let cstr = CString::new(raw).unwrap_or_default();
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(cstr);
    });
}

fn into_c_string(s: String) -> *mut c_char {
    let sanitized = if s.contains('\0') {
        s.replace('\0', " ")
    } else {
        s
    };
    match CString::new(sanitized) {
        Ok(c) => c.into_raw(),
        Err(err) => {
            set_last_error(format!("failed to allocate C string: {err}"));
            ptr::null_mut()
        }
    }
}

/// # Safety
/// `ptr` must be either null or a valid NUL-terminated C string for the duration of the call.
unsafe fn read_required_str<'a>(ptr: *const c_char, name: &str) -> Result<&'a str> {
    if ptr.is_null() {
        bail!("{name} must not be NULL");
    }
    // SAFETY: Caller guarantees `ptr` points to a valid NUL-terminated C string.
    let cstr = unsafe { CStr::from_ptr(ptr) };
    cstr.to_str()
        .with_context(|| format!("{name} is not valid UTF-8"))
}

/// # Safety
/// `ptr` must be either null or a valid NUL-terminated C string for the duration of the call.
unsafe fn read_optional_str<'a>(ptr: *const c_char, name: &str) -> Result<Option<&'a str>> {
    if ptr.is_null() {
        return Ok(None);
    }
    // SAFETY: Caller guarantees `ptr` points to a valid NUL-terminated C string.
    let cstr = unsafe { CStr::from_ptr(ptr) };
    let s = cstr
        .to_str()
        .with_context(|| format!("{name} is not valid UTF-8"))?;
    if s.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(s))
    }
}

fn run_ptr<T, F>(f: F) -> *mut T
where
    F: FnOnce() -> Result<*mut T>,
{
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(err)) => {
            set_last_error(format!("{err:#}"));
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("layerstream FFI call panicked");
            ptr::null_mut()
        }
    }
}

fn run_int<F>(f: F) -> c_int
where
    F: FnOnce() -> Result<c_int>,
{
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(val)) => val,
        Ok(Err(err)) => {
            set_last_error(format!("{err:#}"));
            -1
        }
        Err(_) => {
            set_last_error("layerstream FFI call panicked");
            -1
        }
    }
}

fn run_i64<F>(f: F) -> i64
where
    F: FnOnce() -> Result<i64>,
{
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(val)) => val,
        Ok(Err(err)) => {
            set_last_error(format!("{err:#}"));
            -1
        }
        Err(_) => {
            set_last_error("layerstream FFI call panicked");
            -1
        }
    }
}

/// Opaque handle to a loaded LayerStream inference session.
pub struct LayerStreamEngine {
    session: Mutex<LoadedSession>,
}

/// Thread-safe cooperative cancellation token.
pub struct LayerStreamCancelToken {
    flag: Arc<AtomicBool>,
}

/// Streaming token callback invoked for each UTF-8 token piece.
pub type LayerStreamTokenCb = Option<
    extern "C" fn(chunk_utf8: *const c_char, chunk_len: usize, user_data: *mut c_void) -> bool,
>;

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct OpenOptsWire {
    #[serde(alias = "draftModel")]
    draft_model: Option<String>,
    quant: Option<String>,
    revision: Option<String>,
    #[serde(alias = "cacheDir")]
    cache_dir: Option<String>,
    #[serde(alias = "vramBudgetMb")]
    vram_budget_mb: Option<JsonValue>,
    #[serde(alias = "residentLayers")]
    resident_layers: Option<u32>,
    #[serde(alias = "ctxSize")]
    ctx_size: Option<u32>,
    #[serde(alias = "batchSize")]
    batch_size: Option<u32>,
    threads: Option<u32>,
    #[serde(alias = "kvCacheType")]
    kv_cache_type: Option<String>,
    #[serde(alias = "useGpu")]
    use_gpu: Option<bool>,
    #[serde(alias = "preferDirectIo")]
    prefer_direct_io: Option<bool>,
    #[serde(alias = "cpuEmbeddings")]
    cpu_embeddings: Option<bool>,
    #[serde(alias = "cpuHead")]
    cpu_head: Option<bool>,
    #[serde(alias = "streamHead")]
    stream_head: Option<bool>,
    #[serde(alias = "moeActiveStreaming")]
    moe_active_streaming: Option<bool>,
    #[serde(alias = "maxDraftTokens")]
    max_draft_tokens: Option<usize>,
    verbose: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct GenOptsWire {
    #[serde(alias = "maxTokens")]
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    #[serde(alias = "topP")]
    top_p: Option<f32>,
    #[serde(alias = "minP")]
    min_p: Option<f32>,
    #[serde(alias = "topK")]
    top_k: Option<i32>,
    #[serde(alias = "repeatPenalty")]
    repeat_penalty: Option<f32>,
    #[serde(alias = "frequencyPenalty")]
    frequency_penalty: Option<f32>,
    #[serde(alias = "presencePenalty")]
    presence_penalty: Option<f32>,
    #[serde(alias = "penaltyLastN")]
    penalty_last_n: Option<i32>,
    seed: Option<u32>,
    stop: Option<Vec<String>>,
    #[serde(alias = "ngramSpeculation")]
    ngram_speculation: Option<bool>,
    #[serde(alias = "maxDraftTokens")]
    max_draft_tokens: Option<usize>,
    grammar: Option<String>,
    #[serde(alias = "jsonSchema")]
    json_schema: Option<JsonValue>,
    #[serde(alias = "addGenerationPrompt")]
    add_generation_prompt: Option<bool>,
    tools: Option<Vec<JsonValue>>,
    #[serde(alias = "parseOutput")]
    parse_output: Option<bool>,
}

fn parse_open_options(opts_json: Option<&str>) -> Result<(OpenSessionOptions, bool)> {
    let wire: OpenOptsWire = match opts_json {
        Some(s) => serde_json::from_str(s).context("invalid open_opts_json")?,
        None => OpenOptsWire::default(),
    };

    let vram_val = wire
        .vram_budget_mb
        .unwrap_or_else(|| JsonValue::String("auto".to_string()));
    let (vram_budget_auto, vram_budget_bytes) = normalize_vram_budget_mb(&vram_val)?;
    let kv_cache_type = match wire.kv_cache_type.as_deref() {
        Some(kv) => match normalize_kv_cache_type(kv)? {
            "f16" => KvCacheType::F16,
            "q4_0" => KvCacheType::Q4_0,
            _ => KvCacheType::Q8_0,
        },
        None => KvCacheType::Q8_0,
    };

    let opts = OpenSessionOptions {
        draft_model: wire.draft_model,
        quant: wire.quant,
        revision: wire.revision,
        cache_dir: wire.cache_dir.map(PathBuf::from),
        vram_budget_auto,
        vram_budget_bytes,
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
        on_progress: None,
    };
    Ok((opts, wire.verbose.unwrap_or(false)))
}

fn parse_gen_options(
    opts_json: Option<&str>,
    default_max_draft: usize,
) -> Result<(GenerateOptions, bool, Option<Vec<JsonValue>>, bool)> {
    let wire: GenOptsWire = match opts_json {
        Some(s) => serde_json::from_str(s).context("invalid gen_opts_json")?,
        None => GenOptsWire::default(),
    };
    if wire.grammar.is_some() && wire.json_schema.is_some() {
        bail!("pass at most one of 'grammar' or 'json_schema', not both");
    }
    let grammar = if let Some(g) = wire.grammar {
        if g.trim().is_empty() {
            bail!("grammar must not be empty");
        }
        Some(g)
    } else if let Some(schema) = wire.json_schema {
        Some(json_schema_to_gbnf(&schema).map_err(|e| anyhow::anyhow!("{e}"))?)
    } else {
        None
    };

    let sample = LsSampleParams {
        temperature: wire.temperature.unwrap_or(0.0),
        top_p: wire.top_p.unwrap_or(0.95),
        min_p: wire.min_p.unwrap_or(0.0),
        top_k: wire.top_k.unwrap_or(40),
        repeat_penalty: wire.repeat_penalty.unwrap_or(1.0),
        frequency_penalty: wire.frequency_penalty.unwrap_or(0.0),
        presence_penalty: wire.presence_penalty.unwrap_or(0.0),
        penalty_last_n: wire.penalty_last_n.unwrap_or(64),
        seed: wire.seed.unwrap_or(42),
    };
    validate_sample_params(
        f64::from(sample.temperature),
        f64::from(sample.top_p),
        f64::from(sample.min_p),
        f64::from(sample.repeat_penalty),
        f64::from(sample.frequency_penalty),
        f64::from(sample.presence_penalty),
        sample.penalty_last_n,
    )?;

    let opts = GenerateOptions {
        max_tokens: wire.max_tokens.unwrap_or(256),
        sample,
        stop: wire.stop.unwrap_or_default(),
        enable_ngram_speculation: wire.ngram_speculation.unwrap_or(true),
        max_draft_tokens: wire.max_draft_tokens.unwrap_or(default_max_draft).max(1),
        grammar,
    };
    Ok((
        opts,
        wire.add_generation_prompt.unwrap_or(true),
        wire.tools,
        wire.parse_output.unwrap_or(false),
    ))
}

fn format_chat_prompt(
    session: &LoadedSession,
    messages_json: &str,
    add_generation_prompt: bool,
    tools: Option<&[JsonValue]>,
) -> Result<String> {
    let incoming: Vec<IncomingChatMessage> =
        serde_json::from_str(messages_json).context("invalid messages_json")?;
    if incoming.is_empty() {
        bail!("messages must not be empty");
    }
    let mut msgs = Vec::with_capacity(incoming.len());
    for msg in &incoming {
        msgs.push(msg.to_chat_message().map_err(|e| anyhow::anyhow!("{e}"))?);
    }
    if let Some(t) = tools {
        inject_tools_into_messages(&mut msgs, t);
    }
    session
        .engine
        .apply_chat_template(&msgs, add_generation_prompt)
}

#[unsafe(no_mangle)]
pub extern "C" fn layerstream_version() -> *const c_char {
    VERSION_CSTR.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn layerstream_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(cstr) => cstr.as_ptr(),
        None => ptr::null(),
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn layerstream_clear_error() {
    clear_last_error();
}

/// # Safety
/// `ptr` must be either null or a pointer previously returned by a `layerstream_*` function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_string_free(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: Caller guarantees `ptr` was created via `CString::into_raw` by this library.
    let _ = unsafe { CString::from_raw(ptr) };
}

#[unsafe(no_mangle)]
pub extern "C" fn layerstream_probe_gpu_json(use_gpu: bool) -> *mut c_char {
    run_ptr(|| {
        let payload = match StreamEngine::probe_gpu_device(use_gpu) {
            Some((free_bytes, total_bytes, name)) => json!({
                "available": true,
                "free_bytes": free_bytes,
                "total_bytes": total_bytes,
                "device_name": name,
            }),
            None => json!({
                "available": false,
                "free_bytes": 0,
                "total_bytes": 0,
                "device_name": JsonValue::Null,
            }),
        };
        Ok(into_c_string(payload.to_string()))
    })
}

/// # Safety
/// `path` must be a valid NUL-terminated UTF-8 C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_inspect_gguf_json(path: *const c_char) -> *mut c_char {
    run_ptr(|| {
        // SAFETY: Caller guarantees `path` is null or a valid NUL-terminated C string.
        let path_str = unsafe { read_required_str(path, "path") }?;
        let info = inspect_gguf_file(Path::new(path_str))?;
        Ok(into_c_string(serde_json::to_string(&info)?))
    })
}

/// # Safety
/// `schema_json` must be a valid NUL-terminated UTF-8 C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_json_schema_to_gbnf(
    schema_json: *const c_char,
) -> *mut c_char {
    run_ptr(|| {
        // SAFETY: Caller guarantees `schema_json` is null or a valid NUL-terminated C string.
        let raw = unsafe { read_required_str(schema_json, "schema_json") }?;
        let schema: JsonValue = serde_json::from_str(raw).context("invalid schema_json")?;
        let gbnf = json_schema_to_gbnf(&schema).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(into_c_string(gbnf))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn layerstream_json_object_gbnf() -> *mut c_char {
    run_ptr(|| Ok(into_c_string(json_object_gbnf())))
}

/// # Safety
/// `raw_text` must be a valid NUL-terminated UTF-8 C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_parse_assistant_output_json(
    raw_text: *const c_char,
) -> *mut c_char {
    run_ptr(|| {
        // SAFETY: Caller guarantees `raw_text` is null or a valid NUL-terminated C string.
        let raw = unsafe { read_required_str(raw_text, "raw_text") }?;
        let parsed = parse_assistant_output(raw);
        let val = json!({
            "content": parsed.content,
            "reasoning_content": parsed.reasoning_content,
            "tool_calls": parsed.tool_calls.into_iter().map(|tc| json!({
                "id": tc.id,
                "type": tc.kind,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            })).collect::<Vec<_>>(),
        });
        Ok(into_c_string(val.to_string()))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn layerstream_cancel_token_new() -> *mut LayerStreamCancelToken {
    run_ptr(|| {
        Ok(Box::into_raw(Box::new(LayerStreamCancelToken {
            flag: Arc::new(AtomicBool::new(false)),
        })))
    })
}

/// # Safety
/// `tok` must be null or a valid pointer returned by `layerstream_cancel_token_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_cancel_token_cancel(tok: *const LayerStreamCancelToken) {
    if tok.is_null() {
        return;
    }
    // SAFETY: Caller guarantees `tok` points to a live `LayerStreamCancelToken`.
    unsafe { &*tok }.flag.store(true, Ordering::Relaxed);
}

/// # Safety
/// `tok` must be null or a valid pointer returned by `layerstream_cancel_token_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_cancel_token_free(tok: *mut LayerStreamCancelToken) {
    if tok.is_null() {
        return;
    }
    // SAFETY: Caller transfers ownership of `tok` created by `Box::into_raw`.
    let _ = unsafe { Box::from_raw(tok) };
}

/// # Safety
/// `model_spec` must be a valid NUL-terminated UTF-8 C string; `open_opts_json` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_engine_open(
    model_spec: *const c_char,
    open_opts_json: *const c_char,
) -> *mut LayerStreamEngine {
    run_ptr(|| {
        // SAFETY: Caller guarantees valid NUL-terminated C strings (or null for `open_opts_json`).
        let spec = unsafe { read_required_str(model_spec, "model_spec") }?;
        // SAFETY: Same as above.
        let opts_str = unsafe { read_optional_str(open_opts_json, "open_opts_json") }?;
        let (opts, verbose) = parse_open_options(opts_str)?;
        StreamEngine::set_log_verbose(verbose);
        let session = LoadedSession::open(spec, opts)?;
        Ok(Box::into_raw(Box::new(LayerStreamEngine {
            session: Mutex::new(session),
        })))
    })
}

/// # Safety
/// `engine` must be null or a valid pointer returned by `layerstream_engine_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_engine_free(engine: *mut LayerStreamEngine) {
    if engine.is_null() {
        return;
    }
    // SAFETY: Caller transfers ownership of `engine` created by `Box::into_raw`.
    let _ = unsafe { Box::from_raw(engine) };
}

/// # Safety
/// `engine` must be a valid pointer returned by `layerstream_engine_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_engine_stats_json(
    engine: *const LayerStreamEngine,
) -> *mut c_char {
    run_ptr(|| {
        if engine.is_null() {
            bail!("engine must not be NULL");
        }
        // SAFETY: Caller guarantees `engine` points to a live `LayerStreamEngine`.
        let guard = unsafe { &*engine }.session.lock();
        let snap = guard.stats_snapshot();
        Ok(into_c_string(serde_json::to_string(&snap)?))
    })
}

/// # Safety
/// `engine` and `prompt` must be valid; `gen_opts_json` and `cancel_tok` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_engine_generate(
    engine: *mut LayerStreamEngine,
    prompt: *const c_char,
    gen_opts_json: *const c_char,
    cancel_tok: *const LayerStreamCancelToken,
) -> *mut c_char {
    run_ptr(|| {
        if engine.is_null() {
            bail!("engine must not be NULL");
        }
        // SAFETY: Caller guarantees valid NUL-terminated C strings.
        let prompt_str = unsafe { read_required_str(prompt, "prompt") }?;
        // SAFETY: Same as above.
        let opts_str = unsafe { read_optional_str(gen_opts_json, "gen_opts_json") }?;
        let cancel_flag = if cancel_tok.is_null() {
            None
        } else {
            // SAFETY: Caller guarantees `cancel_tok` is valid for the duration of the call.
            Some(unsafe { &*cancel_tok }.flag.as_ref())
        };

        // SAFETY: Caller guarantees `engine` points to a live `LayerStreamEngine`.
        let mut session = unsafe { &*engine }.session.lock();
        let (opts, _, _, _) = parse_gen_options(opts_str, session.default_max_draft_tokens)?;
        let mut text = String::new();
        let LoadedSession {
            ref mut engine,
            ref mut draft,
            ref mut last_summary,
            ..
        } = *session;
        let summary = engine.generate_stream_with_cancel(
            prompt_str,
            &opts,
            draft.as_mut(),
            cancel_flag,
            |_tok, piece| {
                text.push_str(piece);
                true
            },
        )?;
        *last_summary = Some(summary.clone());
        Ok(into_c_string(
            json!({ "text": text, "summary": summary }).to_string(),
        ))
    })
}

/// # Safety
/// `engine` and `prompt` must be valid; `gen_opts_json`, `cancel_tok`, `cb`, and `user_data` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_engine_generate_stream(
    engine: *mut LayerStreamEngine,
    prompt: *const c_char,
    gen_opts_json: *const c_char,
    cancel_tok: *const LayerStreamCancelToken,
    cb: LayerStreamTokenCb,
    user_data: *mut c_void,
) -> *mut c_char {
    run_ptr(|| {
        if engine.is_null() {
            bail!("engine must not be NULL");
        }
        // SAFETY: Caller guarantees valid NUL-terminated C strings.
        let prompt_str = unsafe { read_required_str(prompt, "prompt") }?;
        // SAFETY: Same as above.
        let opts_str = unsafe { read_optional_str(gen_opts_json, "gen_opts_json") }?;
        let cancel_flag = if cancel_tok.is_null() {
            None
        } else {
            // SAFETY: Caller guarantees `cancel_tok` is valid for the duration of the call.
            Some(unsafe { &*cancel_tok }.flag.as_ref())
        };

        // SAFETY: Caller guarantees `engine` points to a live `LayerStreamEngine`.
        let mut session = unsafe { &*engine }.session.lock();
        let (opts, _, _, _) = parse_gen_options(opts_str, session.default_max_draft_tokens)?;
        let LoadedSession {
            ref mut engine,
            ref mut draft,
            ref mut last_summary,
            ..
        } = *session;
        let summary = engine.generate_stream_with_cancel(
            prompt_str,
            &opts,
            draft.as_mut(),
            cancel_flag,
            |_tok, piece| {
                if let Some(callback) = cb {
                    callback(piece.as_ptr().cast::<c_char>(), piece.len(), user_data)
                } else {
                    true
                }
            },
        )?;
        *last_summary = Some(summary.clone());
        Ok(into_c_string(serde_json::to_string(&summary)?))
    })
}

/// # Safety
/// `engine` and `messages_json` must be valid; `gen_opts_json` and `cancel_tok` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_engine_chat(
    engine: *mut LayerStreamEngine,
    messages_json: *const c_char,
    gen_opts_json: *const c_char,
    cancel_tok: *const LayerStreamCancelToken,
) -> *mut c_char {
    run_ptr(|| {
        if engine.is_null() {
            bail!("engine must not be NULL");
        }
        // SAFETY: Caller guarantees valid NUL-terminated C strings.
        let msgs_str = unsafe { read_required_str(messages_json, "messages_json") }?;
        // SAFETY: Same as above.
        let opts_str = unsafe { read_optional_str(gen_opts_json, "gen_opts_json") }?;
        let cancel_flag = if cancel_tok.is_null() {
            None
        } else {
            // SAFETY: Caller guarantees `cancel_tok` is valid for the duration of the call.
            Some(unsafe { &*cancel_tok }.flag.as_ref())
        };

        // SAFETY: Caller guarantees `engine` points to a live `LayerStreamEngine`.
        let mut session = unsafe { &*engine }.session.lock();
        let (opts, add_gen_prompt, tools, parse_out) =
            parse_gen_options(opts_str, session.default_max_draft_tokens)?;
        let prompt = format_chat_prompt(&session, msgs_str, add_gen_prompt, tools.as_deref())?;
        let mut text = String::new();
        let LoadedSession {
            ref mut engine,
            ref mut draft,
            ref mut last_summary,
            ..
        } = *session;
        let summary = engine.generate_stream_with_cancel(
            &prompt,
            &opts,
            draft.as_mut(),
            cancel_flag,
            |_tok, piece| {
                text.push_str(piece);
                true
            },
        )?;
        *last_summary = Some(summary.clone());
        let parsed = if parse_out {
            let p = parse_assistant_output(&text);
            json!({
                "content": p.content,
                "reasoning_content": p.reasoning_content,
                "tool_calls": p.tool_calls.into_iter().map(|tc| json!({
                    "id": tc.id,
                    "type": tc.kind,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                })).collect::<Vec<_>>(),
            })
        } else {
            JsonValue::Null
        };
        Ok(into_c_string(
            json!({ "text": text, "parsed": parsed, "summary": summary }).to_string(),
        ))
    })
}

/// # Safety
/// `engine` and `messages_json` must be valid; `gen_opts_json`, `cancel_tok`, `cb`, and `user_data` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_engine_chat_stream(
    engine: *mut LayerStreamEngine,
    messages_json: *const c_char,
    gen_opts_json: *const c_char,
    cancel_tok: *const LayerStreamCancelToken,
    cb: LayerStreamTokenCb,
    user_data: *mut c_void,
) -> *mut c_char {
    run_ptr(|| {
        if engine.is_null() {
            bail!("engine must not be NULL");
        }
        // SAFETY: Caller guarantees valid NUL-terminated C strings.
        let msgs_str = unsafe { read_required_str(messages_json, "messages_json") }?;
        // SAFETY: Same as above.
        let opts_str = unsafe { read_optional_str(gen_opts_json, "gen_opts_json") }?;
        let cancel_flag = if cancel_tok.is_null() {
            None
        } else {
            // SAFETY: Caller guarantees `cancel_tok` is valid for the duration of the call.
            Some(unsafe { &*cancel_tok }.flag.as_ref())
        };

        // SAFETY: Caller guarantees `engine` points to a live `LayerStreamEngine`.
        let mut session = unsafe { &*engine }.session.lock();
        let (opts, add_gen_prompt, tools, _) =
            parse_gen_options(opts_str, session.default_max_draft_tokens)?;
        let prompt = format_chat_prompt(&session, msgs_str, add_gen_prompt, tools.as_deref())?;
        let LoadedSession {
            ref mut engine,
            ref mut draft,
            ref mut last_summary,
            ..
        } = *session;
        let summary = engine.generate_stream_with_cancel(
            &prompt,
            &opts,
            draft.as_mut(),
            cancel_flag,
            |_tok, piece| {
                if let Some(callback) = cb {
                    callback(piece.as_ptr().cast::<c_char>(), piece.len(), user_data)
                } else {
                    true
                }
            },
        )?;
        *last_summary = Some(summary.clone());
        Ok(into_c_string(serde_json::to_string(&summary)?))
    })
}

/// # Safety
/// `engine` must be a valid pointer returned by `layerstream_engine_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_engine_kv_clear(engine: *mut LayerStreamEngine) -> c_int {
    run_int(|| {
        if engine.is_null() {
            bail!("engine must not be NULL");
        }
        // SAFETY: Caller guarantees `engine` points to a live `LayerStreamEngine`.
        unsafe { &*engine }.session.lock().engine.kv_clear();
        Ok(0)
    })
}

/// # Safety
/// `engine` must be a valid pointer returned by `layerstream_engine_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_engine_kv_trim(
    engine: *mut LayerStreamEngine,
    pos_min: i32,
) -> c_int {
    run_int(|| {
        if engine.is_null() {
            bail!("engine must not be NULL");
        }
        if pos_min < 0 {
            bail!("pos_min must be non-negative");
        }
        // SAFETY: Caller guarantees `engine` points to a live `LayerStreamEngine`.
        let trimmed = unsafe { &*engine }.session.lock().engine.kv_trim(pos_min);
        Ok(c_int::from(trimmed))
    })
}

/// # Safety
/// `engine` and `path` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_engine_save_state(
    engine: *mut LayerStreamEngine,
    path: *const c_char,
) -> i64 {
    run_i64(|| {
        if engine.is_null() {
            bail!("engine must not be NULL");
        }
        // SAFETY: Caller guarantees `path` is null or a valid NUL-terminated C string.
        let path_str = unsafe { read_required_str(path, "path") }?;
        // SAFETY: Caller guarantees `engine` points to a live `LayerStreamEngine`.
        let n = unsafe { &*engine }
            .session
            .lock()
            .engine
            .save_state(path_str)?;
        i64::try_from(n).context("saved state size exceeds i64::MAX")
    })
}

/// # Safety
/// `engine` and `path` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn layerstream_engine_load_state(
    engine: *mut LayerStreamEngine,
    path: *const c_char,
) -> i64 {
    run_i64(|| {
        if engine.is_null() {
            bail!("engine must not be NULL");
        }
        // SAFETY: Caller guarantees `path` is null or a valid NUL-terminated C string.
        let path_str = unsafe { read_required_str(path, "path") }?;
        // SAFETY: Caller guarantees `engine` points to a live `LayerStreamEngine`.
        let n = unsafe { &*engine }
            .session
            .lock()
            .engine
            .load_state(path_str)?;
        i64::try_from(n).context("loaded state size exceeds i64::MAX")
    })
}
