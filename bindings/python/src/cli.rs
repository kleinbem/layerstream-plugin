//! Native Rust CLI implementation for `layerstream` / `python -m layerstream`.
//!
//! Replaces the 500-line Python `argparse` CLI previously in `__main__.py`, keeping
//! all argument parsing, formatting, chat REPL, and server orchestration in Rust.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use layerstream_core::sdk::{
    CheckOptions, EngineStatsSnapshot, LoadedSession, OpenSessionOptions, RecommendOptions,
    check_model, inspect_gguf_file, pull_model, recommend_repo,
};
use layerstream_core::{
    ChatMessage, GenerateOptions, GenerateSummary, KvCacheType, LsSampleParams, StreamEngine,
    normalize_kv_cache_type, normalize_vram_budget_mb, validate_sample_params,
};
use pyo3::exceptions::{PyEOFError, PyKeyboardInterrupt, PyValueError};
use pyo3::prelude::*;
use serde_json::Value as JsonValue;

use crate::VramBudgetArg;
use crate::server::start_server;

fn fmt_gib(n_bytes: Option<u64>) -> String {
    match n_bytes {
        Some(b) => format!("{:.2} GiB", (b as f64) / (1024.0 * 1024.0 * 1024.0)),
        None => "-".to_string(),
    }
}

fn parse_vram_budget_str(raw: Option<&str>) -> Result<(bool, u64), String> {
    let val = match raw {
        None => JsonValue::String("auto".to_string()),
        Some(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
                JsonValue::Null
            } else if trimmed.eq_ignore_ascii_case("auto") {
                JsonValue::String("auto".to_string())
            } else {
                let n: i64 = trimmed
                    .parse()
                    .map_err(|_| format!("invalid --vram-budget-mb '{trimmed}'"))?;
                JsonValue::Number(n.into())
            }
        }
    };
    normalize_vram_budget_mb(&val).map_err(|e| format!("{e:#}"))
}

fn parse_vram_arg(raw: Option<&str>) -> Result<VramBudgetArg, String> {
    let (auto, bytes) = parse_vram_budget_str(raw)?;
    Ok(if auto {
        VramBudgetArg::Auto
    } else {
        VramBudgetArg::Bytes(bytes)
    })
}

fn format_turn_telemetry(
    elapsed_s: f64,
    summary: Option<&GenerateSummary>,
    st: &EngineStatsSnapshot,
) -> String {
    let comp = summary.map_or(0, |s| s.completion_tokens);
    let tps = (comp as f64) / elapsed_s.max(1e-4);
    let (drafted, accepted) = summary.map_or((0, 0), |s| {
        (s.speculation.drafted_tokens, s.speculation.accepted_tokens)
    });
    let backend = if !st.backend_name.is_empty() {
        st.backend_name.as_str()
    } else if st.using_gpu {
        "GPU"
    } else {
        "CPU"
    };
    let io_gib = (st.io.bytes_read as f64) / (1024.0 * 1024.0 * 1024.0);
    let mut parts = vec![
        format!("{comp} tok in {elapsed_s:.2}s ({tps:.2} tok/s)"),
        format!("backend={backend}"),
        format!("resident={}/{}", st.resident_layers, st.n_layers),
        format!("kv={}", st.cached_kv_tokens),
        format!("nvme_read={io_gib:.2} GiB"),
    ];
    if drafted > 0 {
        parts.push(format!("spec={accepted}/{drafted}"));
    }
    parts.join(" | ")
}

fn resolve_use_gpu(cpu_flag: bool) -> bool {
    if cpu_flag {
        return false;
    }
    if StreamEngine::probe_gpu_device(true).is_some() {
        return true;
    }
    eprintln!("[layerstream] no GPU device detected; falling back to CPU mode");
    false
}

fn next_val<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse_kv(raw: &str) -> Result<KvCacheType, String> {
    match normalize_kv_cache_type(raw).map_err(|e| format!("{e:#}"))? {
        "f16" => Ok(KvCacheType::F16),
        "q4_0" => Ok(KvCacheType::Q4_0),
        _ => Ok(KvCacheType::Q8_0),
    }
}

fn print_help() {
    println!(
        "usage: layerstream [--version] <command> [options]\n\n\
         commands:\n  \
           chat       Interactive or single-turn streaming chat\n  \
           serve      Serve an OpenAI-compatible HTTP + SSE API (Rust axum)\n  \
           recommend  Rank Hugging Face GGUF quants for your GPU + NVMe\n  \
           check      Run fail-closed hardware & storage preflight check\n  \
           pull       Download a GGUF checkpoint into the local cache\n  \
           inspect    Inspect GGUF v2/v3 metadata headers (0 MB VRAM)"
    );
}

fn cmd_inspect(py: Python<'_>, args: &[String]) -> PyResult<i32> {
    let mut path: Option<String> = None;
    let mut as_json = false;
    for arg in args {
        match arg.as_str() {
            "--json" => as_json = true,
            "-h" | "--help" => {
                println!("usage: layerstream inspect <path> [--json]");
                return Ok(0);
            }
            s if s.starts_with('-') => {
                return Err(PyValueError::new_err(format!("unknown flag: {s}")));
            }
            s => path = Some(s.to_string()),
        }
    }
    let Some(p) = path else {
        return Err(PyValueError::new_err("inspect requires <path>"));
    };
    let pb = PathBuf::from(&p);
    let info = py
        .detach(move || inspect_gguf_file(&pb))
        .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
    if as_json {
        if let Ok(text) = serde_json::to_string_pretty(&info) {
            println!("{text}");
        }
        return Ok(0);
    }
    println!("File:           {p}");
    println!("Name:           {}", info.name.as_deref().unwrap_or("-"));
    println!("Architecture:   {}", info.architecture);
    println!("Layers:         {}", info.n_layers);
    println!(
        "Context Length: {}",
        info.context_length
            .map_or_else(|| "-".to_string(), |c| c.to_string())
    );
    println!("Total Size:     {}", fmt_gib(Some(info.total_file_bytes)));
    println!("Shards:         {}", info.shard_count);
    println!("Tensors:        {}", info.tensor_count);
    Ok(0)
}

fn cmd_recommend(py: Python<'_>, args: &[String]) -> PyResult<i32> {
    let mut repo: Option<String> = None;
    let mut revision: Option<String> = None;
    let mut free_ram_gb: Option<f64> = None;
    let mut nvme_mib_s: Option<f64> = None;
    let mut no_draft = false;
    let mut cache_dir: Option<PathBuf> = None;
    let mut cpu = false;
    let mut as_json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--revision" => {
                revision = Some(
                    next_val(args, &mut i, "--revision")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--free-ram-gb" => {
                let s = next_val(args, &mut i, "--free-ram-gb").map_err(PyValueError::new_err)?;
                free_ram_gb =
                    Some(s.parse().map_err(|_| {
                        PyValueError::new_err(format!("invalid --free-ram-gb '{s}'"))
                    })?);
            }
            "--nvme-mib-s" => {
                let s = next_val(args, &mut i, "--nvme-mib-s").map_err(PyValueError::new_err)?;
                nvme_mib_s =
                    Some(s.parse().map_err(|_| {
                        PyValueError::new_err(format!("invalid --nvme-mib-s '{s}'"))
                    })?);
            }
            "--no-draft" => no_draft = true,
            "--cache-dir" => {
                cache_dir = Some(PathBuf::from(
                    next_val(args, &mut i, "--cache-dir").map_err(PyValueError::new_err)?,
                ));
            }
            "--cpu" => cpu = true,
            "--json" => as_json = true,
            "-h" | "--help" => {
                println!("usage: layerstream recommend <repo> [--json] [--cpu] ...");
                return Ok(0);
            }
            s if s.starts_with('-') => {
                return Err(PyValueError::new_err(format!("unknown flag: {s}")));
            }
            s => repo = Some(s.to_string()),
        }
        i += 1;
    }
    let Some(repo_id) = repo else {
        return Err(PyValueError::new_err("recommend requires <repo>"));
    };
    let opts = RecommendOptions {
        revision,
        free_ram_gb,
        nvme_mib_s,
        has_draft: !no_draft,
        cache_dir,
        use_gpu: !cpu,
    };
    let (resolved_repo, recs) = py
        .detach(move || recommend_repo(&repo_id, opts))
        .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
    if as_json {
        if let Ok(text) = serde_json::to_string_pretty(&recs) {
            println!("{text}");
        }
        return Ok(0);
    }
    if recs.is_empty() {
        eprintln!("No GGUF candidates found.");
        return Ok(1);
    }
    println!("Repository: {resolved_repo}");
    println!(
        "{:<4} {:<12} {:>10} {:<14} {:>7} {:>14}  URI",
        "Rec", "Quant", "Size", "Verdict", "Pinned", "Est tok/s"
    );
    println!("{}", "-".repeat(96));
    for r in &recs {
        let mark = if r.recommended { "*" } else { " " };
        let tps = format!("{:.1}-{:.1}", r.est_tps_low, r.est_tps_high);
        println!(
            " {mark}   {:<12} {:>10} {:<14} {:>6}% {:>14}  {}",
            r.quant,
            fmt_gib(Some(r.total_bytes)),
            r.verdict,
            r.pinned_pct,
            tps,
            r.hf_uri
        );
    }
    Ok(0)
}

fn cmd_check(py: Python<'_>, args: &[String]) -> PyResult<i32> {
    let mut model: Option<String> = None;
    let mut draft_model: Option<String> = None;
    let mut quant: Option<String> = None;
    let mut revision: Option<String> = None;
    let mut cache_dir: Option<PathBuf> = None;
    let mut cpu = false;
    let mut allow_slow = false;
    let mut as_json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--draft-model" => {
                draft_model = Some(
                    next_val(args, &mut i, "--draft-model")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--quant" => {
                quant = Some(
                    next_val(args, &mut i, "--quant")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--revision" => {
                revision = Some(
                    next_val(args, &mut i, "--revision")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--cache-dir" => {
                cache_dir = Some(PathBuf::from(
                    next_val(args, &mut i, "--cache-dir").map_err(PyValueError::new_err)?,
                ));
            }
            "--cpu" => cpu = true,
            "--allow-slow" => allow_slow = true,
            "--json" => as_json = true,
            "-h" | "--help" => {
                println!("usage: layerstream check <model> [--cpu] [--json] ...");
                return Ok(0);
            }
            s if s.starts_with('-') => {
                return Err(PyValueError::new_err(format!("unknown flag: {s}")));
            }
            s => model = Some(s.to_string()),
        }
        i += 1;
    }
    let Some(model_spec) = model else {
        return Err(PyValueError::new_err("check requires <model>"));
    };
    let opts = CheckOptions {
        draft_model,
        quant,
        revision,
        cache_dir,
        use_gpu: !cpu,
        allow_slow,
    };
    let report = match py.detach(move || check_model(&model_spec, opts)) {
        Ok(r) => r,
        Err(err) => {
            if as_json {
                println!(
                    "{}",
                    serde_json::json!({ "ok": false, "error": format!("{err:#}") })
                );
            } else {
                eprintln!("Preflight FAILED: {err:#}");
            }
            return Ok(1);
        }
    };
    if as_json {
        if let Ok(text) = serde_json::to_string_pretty(&report) {
            println!("{text}");
        }
        return Ok(0);
    }
    println!("Verdict:        {} ({})", report.verdict, report.summary);
    println!(
        "Model:          {} ({})",
        report.model_path,
        fmt_gib(Some(report.total_file_bytes))
    );
    if let Some(ref dp) = report.draft_model_path {
        println!("Draft Model:    {dp}");
    }
    println!(
        "Architecture:   {} ({} layers, ctx={})",
        report.architecture,
        report.n_layers,
        report
            .context_length
            .map_or_else(|| "-".to_string(), |c| c.to_string())
    );
    println!(
        "Backend / GPU:  {} (free={}, total={})",
        report.gpu_backend.as_deref().unwrap_or("CPU"),
        fmt_gib(report.gpu_free_bytes),
        fmt_gib(report.gpu_total_bytes),
    );
    println!(
        "Storage I/O:    {:.0} MiB/s (direct_io={})",
        report.storage_mib_s, report.direct_io
    );
    Ok(0)
}

fn cmd_pull(py: Python<'_>, args: &[String]) -> PyResult<i32> {
    let mut spec: Option<String> = None;
    let mut quant: Option<String> = None;
    let mut revision: Option<String> = None;
    let mut cache_dir: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--quant" => {
                quant = Some(
                    next_val(args, &mut i, "--quant")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--revision" => {
                revision = Some(
                    next_val(args, &mut i, "--revision")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--cache-dir" => {
                cache_dir = Some(PathBuf::from(
                    next_val(args, &mut i, "--cache-dir").map_err(PyValueError::new_err)?,
                ));
            }
            "-h" | "--help" => {
                println!("usage: layerstream pull <spec> [--quant ...] [--cache-dir ...]");
                return Ok(0);
            }
            s if s.starts_with('-') => {
                return Err(PyValueError::new_err(format!("unknown flag: {s}")));
            }
            s => spec = Some(s.to_string()),
        }
        i += 1;
    }
    let Some(spec_str) = spec else {
        return Err(PyValueError::new_err("pull requires <spec>"));
    };
    let local_path = py
        .detach(move || pull_model(&spec_str, cache_dir, quant, revision, None))
        .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
    println!("{}", local_path.display());
    Ok(0)
}

fn cmd_chat(py: Python<'_>, args: &[String], input_fn: Option<&Bound<'_, PyAny>>) -> PyResult<i32> {
    let mut model: Option<String> = None;
    let mut draft_model: Option<String> = None;
    let mut quant: Option<String> = None;
    let mut revision: Option<String> = None;
    let mut cache_dir: Option<PathBuf> = None;
    let mut vram_raw: Option<String> = None;
    let mut resident_layers: Option<u32> = None;
    let mut ctx_size: u32 = 4096;
    let mut batch_size: u32 = 512;
    let mut threads: Option<u32> = None;
    let mut kv_raw = "q8_0".to_string();
    let mut cpu = false;
    let mut no_direct_io = false;
    let mut verbose = false;
    let mut prompt: Option<String> = None;
    let mut system: Option<String> = None;
    let mut max_tokens: usize = 512;
    let mut temperature: f32 = 0.0;
    let mut top_p: f32 = 0.95;
    let mut min_p: f32 = 0.0;
    let mut top_k: i32 = 40;
    let mut repeat_penalty: f32 = 1.0;
    let mut frequency_penalty: f32 = 0.0;
    let mut presence_penalty: f32 = 0.0;
    let mut seed: u32 = 42;
    let mut no_ngram = false;
    let mut load_state: Option<String> = None;
    let mut save_state: Option<String> = None;
    let mut quiet_stats = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--draft-model" => {
                draft_model = Some(
                    next_val(args, &mut i, "--draft-model")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--quant" => {
                quant = Some(
                    next_val(args, &mut i, "--quant")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--revision" => {
                revision = Some(
                    next_val(args, &mut i, "--revision")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--cache-dir" => {
                cache_dir = Some(PathBuf::from(
                    next_val(args, &mut i, "--cache-dir").map_err(PyValueError::new_err)?,
                ));
            }
            "--vram-budget-mb" => {
                vram_raw = Some(
                    next_val(args, &mut i, "--vram-budget-mb")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--resident-layers" => {
                let s =
                    next_val(args, &mut i, "--resident-layers").map_err(PyValueError::new_err)?;
                resident_layers = Some(s.parse().map_err(|_| {
                    PyValueError::new_err(format!("invalid --resident-layers '{s}'"))
                })?);
            }
            "--ctx-size" => {
                let s = next_val(args, &mut i, "--ctx-size").map_err(PyValueError::new_err)?;
                ctx_size = s
                    .parse()
                    .map_err(|_| PyValueError::new_err(format!("invalid --ctx-size '{s}'")))?;
            }
            "--batch-size" => {
                let s = next_val(args, &mut i, "--batch-size").map_err(PyValueError::new_err)?;
                batch_size = s
                    .parse()
                    .map_err(|_| PyValueError::new_err(format!("invalid --batch-size '{s}'")))?;
            }
            "--threads" => {
                let s = next_val(args, &mut i, "--threads").map_err(PyValueError::new_err)?;
                threads = Some(
                    s.parse()
                        .map_err(|_| PyValueError::new_err(format!("invalid --threads '{s}'")))?,
                );
            }
            "--kv-cache-type" => {
                kv_raw = next_val(args, &mut i, "--kv-cache-type")
                    .map_err(PyValueError::new_err)?
                    .to_string();
            }
            "--cpu" => cpu = true,
            "--no-direct-io" => no_direct_io = true,
            "--verbose" => verbose = true,
            "--isolated-worker" => {}
            "-p" | "--prompt" => {
                prompt = Some(
                    next_val(args, &mut i, "--prompt")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "-s" | "--system" => {
                system = Some(
                    next_val(args, &mut i, "--system")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--max-tokens" => {
                let s = next_val(args, &mut i, "--max-tokens").map_err(PyValueError::new_err)?;
                max_tokens = s
                    .parse()
                    .map_err(|_| PyValueError::new_err(format!("invalid --max-tokens '{s}'")))?;
            }
            "--temperature" => {
                let s = next_val(args, &mut i, "--temperature").map_err(PyValueError::new_err)?;
                temperature = s
                    .parse()
                    .map_err(|_| PyValueError::new_err(format!("invalid --temperature '{s}'")))?;
            }
            "--top-p" => {
                let s = next_val(args, &mut i, "--top-p").map_err(PyValueError::new_err)?;
                top_p = s
                    .parse()
                    .map_err(|_| PyValueError::new_err(format!("invalid --top-p '{s}'")))?;
            }
            "--min-p" => {
                let s = next_val(args, &mut i, "--min-p").map_err(PyValueError::new_err)?;
                min_p = s
                    .parse()
                    .map_err(|_| PyValueError::new_err(format!("invalid --min-p '{s}'")))?;
            }
            "--top-k" => {
                let s = next_val(args, &mut i, "--top-k").map_err(PyValueError::new_err)?;
                top_k = s
                    .parse()
                    .map_err(|_| PyValueError::new_err(format!("invalid --top-k '{s}'")))?;
            }
            "--repeat-penalty" => {
                let s =
                    next_val(args, &mut i, "--repeat-penalty").map_err(PyValueError::new_err)?;
                repeat_penalty = s.parse().map_err(|_| {
                    PyValueError::new_err(format!("invalid --repeat-penalty '{s}'"))
                })?;
            }
            "--frequency-penalty" => {
                let s =
                    next_val(args, &mut i, "--frequency-penalty").map_err(PyValueError::new_err)?;
                frequency_penalty = s.parse().map_err(|_| {
                    PyValueError::new_err(format!("invalid --frequency-penalty '{s}'"))
                })?;
            }
            "--presence-penalty" => {
                let s =
                    next_val(args, &mut i, "--presence-penalty").map_err(PyValueError::new_err)?;
                presence_penalty = s.parse().map_err(|_| {
                    PyValueError::new_err(format!("invalid --presence-penalty '{s}'"))
                })?;
            }
            "--seed" => {
                let s = next_val(args, &mut i, "--seed").map_err(PyValueError::new_err)?;
                seed = s
                    .parse()
                    .map_err(|_| PyValueError::new_err(format!("invalid --seed '{s}'")))?;
            }
            "--no-ngram" => no_ngram = true,
            "--load-state" => {
                load_state = Some(
                    next_val(args, &mut i, "--load-state")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--save-state" => {
                save_state = Some(
                    next_val(args, &mut i, "--save-state")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--quiet-stats" => quiet_stats = true,
            "-h" | "--help" => {
                println!("usage: layerstream chat <model> [-p PROMPT] [--cpu] ...");
                return Ok(0);
            }
            s if s.starts_with('-') => {
                return Err(PyValueError::new_err(format!("unknown flag: {s}")));
            }
            s => model = Some(s.to_string()),
        }
        i += 1;
    }

    let Some(model_spec) = model else {
        return Err(PyValueError::new_err("chat requires <model>"));
    };
    let (vram_budget_auto, vram_budget_bytes) =
        parse_vram_budget_str(vram_raw.as_deref()).map_err(PyValueError::new_err)?;
    let kv_cache_type = parse_kv(&kv_raw).map_err(PyValueError::new_err)?;
    let sample = LsSampleParams {
        temperature,
        top_p,
        min_p,
        top_k,
        repeat_penalty,
        frequency_penalty,
        presence_penalty,
        penalty_last_n: 64,
        seed,
    };
    validate_sample_params(
        f64::from(sample.temperature),
        f64::from(sample.top_p),
        f64::from(sample.min_p),
        f64::from(sample.repeat_penalty),
        f64::from(sample.frequency_penalty),
        f64::from(sample.presence_penalty),
        sample.penalty_last_n,
    )
    .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;

    let use_gpu = resolve_use_gpu(cpu);
    StreamEngine::set_log_verbose(verbose);
    let open_opts = OpenSessionOptions {
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
        kv_cache_type,
        use_gpu,
        prefer_direct_io: !no_direct_io,
        cpu_embeddings: false,
        cpu_head: false,
        stream_head: false,
        moe_active_streaming: true,
        max_draft_tokens: 5,
        on_progress: None,
    };

    let mut session = py
        .detach(move || LoadedSession::open(&model_spec, open_opts))
        .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;

    if let Some(ref lp) = load_state {
        let restored = session
            .engine
            .load_state(lp)
            .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
        eprintln!("[layerstream] restored {restored} KV tokens from {lp}");
    }

    let gen_opts = GenerateOptions {
        max_tokens,
        sample,
        stop: Vec::new(),
        enable_ngram_speculation: !no_ngram,
        max_draft_tokens: session.default_max_draft_tokens.max(1),
        grammar: None,
    };

    let mut messages: Vec<ChatMessage> = system
        .as_ref()
        .map(|s| {
            vec![ChatMessage {
                role: "system".to_string(),
                content: s.clone(),
            }]
        })
        .unwrap_or_default();

    if let Some(single_prompt) = prompt {
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: single_prompt,
        });
        let formatted = session
            .engine
            .apply_chat_template(&messages, true)
            .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
        let t0 = Instant::now();
        let cancel = Arc::new(AtomicBool::new(false));
        let summary = {
            let LoadedSession {
                ref mut engine,
                ref mut draft,
                ..
            } = session;
            engine
                .generate_stream_with_cancel(
                    &formatted,
                    &gen_opts,
                    draft.as_mut(),
                    Some(cancel.as_ref()),
                    |_tok, piece| {
                        let mut out = std::io::stdout().lock();
                        let _ = out.write_all(piece.as_bytes());
                        let _ = out.flush();
                        true
                    },
                )
                .map_err(|e| PyValueError::new_err(format!("{e:#}")))?
        };
        println!();
        let elapsed = t0.elapsed().as_secs_f64();
        if let Some(ref sp) = save_state {
            let saved = session
                .engine
                .save_state(sp)
                .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
            eprintln!("[layerstream] saved {saved} KV tokens to {sp}");
        }
        if !quiet_stats {
            let snap = session.stats_snapshot();
            eprintln!(
                "[layerstream] {}",
                format_turn_telemetry(elapsed, Some(&summary), &snap)
            );
        }
        return Ok(0);
    }

    let st0 = session.stats_snapshot();
    eprintln!(
        "[layerstream {}] loaded {} (backend={}, resident={}/{}). Commands: /stats, /clear, /save <path>, /load <path>, /quit",
        env!("CARGO_PKG_VERSION"),
        st0.model_path,
        st0.backend_name,
        st0.resident_layers,
        st0.n_layers
    );

    loop {
        let user_line = match input_fn {
            Some(func) => match func.call1(("\n> ",)) {
                Ok(val) => val.extract::<String>()?.trim().to_string(),
                Err(err)
                    if err.is_instance_of::<PyEOFError>(py)
                        || err.is_instance_of::<PyKeyboardInterrupt>(py) =>
                {
                    eprintln!();
                    break;
                }
                Err(err) => return Err(err),
            },
            None => {
                let mut line = String::new();
                print!("\n> ");
                let _ = std::io::stdout().flush();
                match std::io::stdin().read_line(&mut line) {
                    Ok(0) => {
                        eprintln!();
                        break;
                    }
                    Ok(_) => line.trim().to_string(),
                    Err(_) => break,
                }
            }
        };
        if user_line.is_empty() {
            continue;
        }
        match user_line.as_str() {
            "/quit" | "/exit" => break,
            "/clear" => {
                session.engine.kv_clear();
                messages = system
                    .as_ref()
                    .map(|s| {
                        vec![ChatMessage {
                            role: "system".to_string(),
                            content: s.clone(),
                        }]
                    })
                    .unwrap_or_default();
                eprintln!("[layerstream] cleared KV cache and conversation history");
                continue;
            }
            "/stats" => {
                if let Ok(text) = serde_json::to_string_pretty(&session.stats_snapshot()) {
                    eprintln!("{text}");
                }
                continue;
            }
            _ => {}
        }
        if let Some(target) = user_line.strip_prefix("/save ") {
            let target = target.trim();
            match session.engine.save_state(target) {
                Ok(n_tok) => eprintln!("[layerstream] saved {n_tok} KV tokens -> {target}"),
                Err(err) => eprintln!("[layerstream] save failed: {err:#}"),
            }
            continue;
        }
        if let Some(target) = user_line.strip_prefix("/load ") {
            let target = target.trim();
            match session.engine.load_state(target) {
                Ok(n_tok) => eprintln!("[layerstream] restored {n_tok} KV tokens <- {target}"),
                Err(err) => eprintln!("[layerstream] load failed: {err:#}"),
            }
            continue;
        }

        messages.push(ChatMessage {
            role: "user".to_string(),
            content: user_line,
        });
        let formatted = session
            .engine
            .apply_chat_template(&messages, true)
            .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
        let t0 = Instant::now();
        let mut reply = String::new();
        let cancel = Arc::new(AtomicBool::new(false));
        let summary = {
            let LoadedSession {
                ref mut engine,
                ref mut draft,
                ..
            } = session;
            engine
                .generate_stream_with_cancel(
                    &formatted,
                    &gen_opts,
                    draft.as_mut(),
                    Some(cancel.as_ref()),
                    |_tok, piece| {
                        let mut out = std::io::stdout().lock();
                        let _ = out.write_all(piece.as_bytes());
                        let _ = out.flush();
                        reply.push_str(piece);
                        true
                    },
                )
                .map_err(|e| PyValueError::new_err(format!("{e:#}")))?
        };
        println!();
        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: reply,
        });
        let elapsed = t0.elapsed().as_secs_f64();
        if !quiet_stats {
            let snap = session.stats_snapshot();
            eprintln!(
                "[{}]",
                format_turn_telemetry(elapsed, Some(&summary), &snap)
            );
        }
    }

    if let Some(ref sp) = save_state {
        let saved = session
            .engine
            .save_state(sp)
            .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
        eprintln!("[layerstream] saved {saved} KV tokens to {sp}");
    }
    Ok(0)
}

fn cmd_serve(py: Python<'_>, args: &[String]) -> PyResult<i32> {
    let mut model: Option<String> = None;
    let mut draft_model: Option<String> = None;
    let mut quant: Option<String> = None;
    let mut revision: Option<String> = None;
    let mut cache_dir: Option<PathBuf> = None;
    let mut vram_raw: Option<String> = None;
    let mut resident_layers: Option<u32> = None;
    let mut ctx_size: u32 = 4096;
    let mut batch_size: u32 = 512;
    let mut threads: Option<u32> = None;
    let mut kv_cache_type = "q8_0".to_string();
    let mut cpu = false;
    let mut no_direct_io = false;
    let mut verbose = false;
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 8080;
    let mut idle_timeout_secs: u64 = 60;
    let mut api_key: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--draft-model" => {
                draft_model = Some(
                    next_val(args, &mut i, "--draft-model")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--quant" => {
                quant = Some(
                    next_val(args, &mut i, "--quant")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--revision" => {
                revision = Some(
                    next_val(args, &mut i, "--revision")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--cache-dir" => {
                cache_dir = Some(PathBuf::from(
                    next_val(args, &mut i, "--cache-dir").map_err(PyValueError::new_err)?,
                ));
            }
            "--vram-budget-mb" => {
                vram_raw = Some(
                    next_val(args, &mut i, "--vram-budget-mb")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "--resident-layers" => {
                let s =
                    next_val(args, &mut i, "--resident-layers").map_err(PyValueError::new_err)?;
                resident_layers = Some(s.parse().map_err(|_| {
                    PyValueError::new_err(format!("invalid --resident-layers '{s}'"))
                })?);
            }
            "--ctx-size" => {
                let s = next_val(args, &mut i, "--ctx-size").map_err(PyValueError::new_err)?;
                ctx_size = s
                    .parse()
                    .map_err(|_| PyValueError::new_err(format!("invalid --ctx-size '{s}'")))?;
            }
            "--batch-size" => {
                let s = next_val(args, &mut i, "--batch-size").map_err(PyValueError::new_err)?;
                batch_size = s
                    .parse()
                    .map_err(|_| PyValueError::new_err(format!("invalid --batch-size '{s}'")))?;
            }
            "--threads" => {
                let s = next_val(args, &mut i, "--threads").map_err(PyValueError::new_err)?;
                threads = Some(
                    s.parse()
                        .map_err(|_| PyValueError::new_err(format!("invalid --threads '{s}'")))?,
                );
            }
            "--kv-cache-type" => {
                kv_cache_type = next_val(args, &mut i, "--kv-cache-type")
                    .map_err(PyValueError::new_err)?
                    .to_string();
            }
            "--cpu" => cpu = true,
            "--no-direct-io" => no_direct_io = true,
            "--verbose" => verbose = true,
            "--host" => {
                host = next_val(args, &mut i, "--host")
                    .map_err(PyValueError::new_err)?
                    .to_string();
            }
            "--port" => {
                let s = next_val(args, &mut i, "--port").map_err(PyValueError::new_err)?;
                port = s
                    .parse()
                    .map_err(|_| PyValueError::new_err(format!("invalid --port '{s}'")))?;
            }
            "--idle-timeout-secs" => {
                let s =
                    next_val(args, &mut i, "--idle-timeout-secs").map_err(PyValueError::new_err)?;
                idle_timeout_secs = s.parse().map_err(|_| {
                    PyValueError::new_err(format!("invalid --idle-timeout-secs '{s}'"))
                })?;
            }
            "--api-key" => {
                api_key = Some(
                    next_val(args, &mut i, "--api-key")
                        .map_err(PyValueError::new_err)?
                        .to_string(),
                );
            }
            "-h" | "--help" => {
                println!("usage: layerstream serve <model> [--host 127.0.0.1] [--port 8080] ...");
                return Ok(0);
            }
            s if s.starts_with('-') => {
                return Err(PyValueError::new_err(format!("unknown flag: {s}")));
            }
            s => model = Some(s.to_string()),
        }
        i += 1;
    }

    let Some(model_spec) = model else {
        return Err(PyValueError::new_err("serve requires <model>"));
    };
    let vram_arg = parse_vram_arg(vram_raw.as_deref()).map_err(PyValueError::new_err)?;
    let use_gpu = resolve_use_gpu(cpu);
    let eff_key = api_key.or_else(|| std::env::var("LAYERSTREAM_API_KEY").ok());

    let handle = start_server(
        py,
        &model_spec,
        &host,
        port,
        eff_key,
        draft_model.as_deref(),
        quant.as_deref(),
        revision.as_deref(),
        cache_dir,
        vram_arg,
        resident_layers,
        ctx_size,
        batch_size,
        threads,
        &kv_cache_type,
        use_gpu,
        !no_direct_io,
        false,
        false,
        false,
        true,
        idle_timeout_secs,
        5,
        verbose,
        None,
    )?;

    eprintln!(
        "[layerstream {}] serving OpenAI API on {} (model={}, use_gpu={})",
        env!("CARGO_PKG_VERSION"),
        handle.base_url_str(),
        model_spec,
        use_gpu
    );
    if let Err(err) = handle.wait_py(py) {
        if err.is_instance_of::<PyKeyboardInterrupt>(py) {
            eprintln!("\n[layerstream] shutting down HTTP server");
        } else {
            let _ = handle.stop_py(py);
            return Err(err);
        }
    }
    let _ = handle.stop_py(py);
    Ok(0)
}

#[pyfunction]
#[pyo3(signature = (argv, input_fn=None))]
pub(crate) fn run_cli(
    py: Python<'_>,
    argv: Vec<String>,
    input_fn: Option<&Bound<'_, PyAny>>,
) -> PyResult<i32> {
    let Some(first) = argv.first().map(String::as_str) else {
        print_help();
        return Ok(2);
    };
    match first {
        "--version" | "-V" => {
            println!("layerstream {}", env!("CARGO_PKG_VERSION"));
            Ok(0)
        }
        "-h" | "--help" => {
            print_help();
            Ok(0)
        }
        "inspect" => cmd_inspect(py, &argv[1..]),
        "recommend" => cmd_recommend(py, &argv[1..]),
        "check" => cmd_check(py, &argv[1..]),
        "pull" => cmd_pull(py, &argv[1..]),
        "chat" => cmd_chat(py, &argv[1..], input_fn),
        "serve" => cmd_serve(py, &argv[1..]),
        other => {
            eprintln!("layerstream: unknown command '{other}' (run `layerstream --help`)");
            Ok(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vram_and_gib_formatting_work() {
        assert_eq!(fmt_gib(None), "-");
        assert_eq!(fmt_gib(Some(1024 * 1024 * 1024)), "1.00 GiB");
        assert_eq!(parse_vram_budget_str(None).ok(), Some((true, 0)));
        assert_eq!(parse_vram_budget_str(Some("AUTO")).ok(), Some((true, 0)));
        assert_eq!(parse_vram_budget_str(Some("none")).ok(), Some((false, 0)));
        assert_eq!(parse_vram_budget_str(Some("")).ok(), Some((false, 0)));
        assert_eq!(
            parse_vram_budget_str(Some("4096")).ok(),
            Some((false, 4096 * 1024 * 1024))
        );
        assert!(parse_vram_budget_str(Some("bogus")).is_err());
    }
}
