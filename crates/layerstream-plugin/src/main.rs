mod config;
mod managed_models;
mod mesh_adapter;
mod worker;

use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use layerstream_core::{hf, probe_dir_storage_mib_per_s};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

use crate::config::{EngineArgs, run_preflight_checks};

#[derive(Debug, Parser)]
#[command(
    name = "layerstream-plugin",
    version,
    about = "Out-of-core GGUF layer-streaming inference engine and Mesh-LLM plugin"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Engine arguments used when launched directly by mesh-llm in plugin mode.
    #[command(flatten)]
    plugin_args: EngineArgs,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run as a standalone OpenAI-compatible HTTP server (no mesh-llm required).
    Serve {
        /// Listen host address.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Listen TCP port.
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Optional Bearer API key for `/v1/*` routes (`Authorization: Bearer <KEY>`).
        #[arg(long, env = "LAYERSTREAM_API_KEY")]
        api_key: Option<String>,
        #[command(flatten)]
        engine: EngineArgs,
    },
    /// Run as a Mesh-LLM v3 IPC plugin (`MESH_LLM_PLUGIN_ENDPOINT`; default when no subcommand is passed).
    Plugin {
        #[command(flatten)]
        engine: EngineArgs,
    },
    /// Validate GPU drivers, storage throughput, draft compatibility, and estimated decode speed.
    Check {
        #[command(flatten)]
        engine: EngineArgs,
    },
    /// Download a Hugging Face GGUF (`hf://org/repo:file.gguf`) into the local cache and exit.
    Pull {
        /// Hugging Face URI (`hf://org/repo:file.gguf`).
        spec: String,
        /// Optional cache directory (defaults to `$XDG_CACHE_HOME/layerstream/models`).
        #[arg(long, env = "LAYERSTREAM_CACHE_DIR")]
        cache_dir: Option<std::path::PathBuf>,
    },
    /// Inspect a local or Hugging Face GGUF header (~256 KiB HTTP Range) for architecture, MoE, hybrid-SSM, and built-in MTP heads without downloading weights.
    Inspect {
        /// Local `.gguf` path or Hugging Face URI (`hf://org/repo:file.gguf`).
        spec: String,
        /// Emit machine-readable JSON output.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Query a Hugging Face GGUF repo and recommend the best quant for your probed GPU & NVMe budget.
    Recommend {
        /// Hugging Face repository (`org/repo`, `hf://org/repo@rev`, or URL).
        repo: String,
        /// Optional branch/tag/commit revision (defaults to `main`).
        #[arg(long)]
        revision: Option<String>,
        /// Override free GPU/UMA memory in GiB (defaults to live GPU probe).
        #[arg(long)]
        free_ram_gb: Option<f64>,
        /// Override sustained NVMe O_DIRECT bandwidth in MiB/s (defaults to live probe).
        #[arg(long)]
        nvme_mib_s: Option<f64>,
        /// Assume no draft speculator (show conservative 1-token/sweep floor).
        #[arg(long, default_value_t = false)]
        no_draft: bool,
        /// Cache directory used for live NVMe bandwidth probe.
        #[arg(long, env = "LAYERSTREAM_CACHE_DIR")]
        cache_dir: Option<std::path::PathBuf>,
        /// Emit machine-readable JSON output.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Benchmark local storage & I/O scheduling knobs (quality-preserving) and persist the best profile.
    Tune {
        /// Optional local `.gguf` model file to sweep against (defaults to a temporary 32 MiB probe file in --cache-dir).
        #[arg(long)]
        model: Option<std::path::PathBuf>,
        /// Probe directory used when --model is omitted.
        #[arg(long, env = "LAYERSTREAM_CACHE_DIR")]
        cache_dir: Option<std::path::PathBuf>,
        /// Hardware/OS profile override: auto, edge, balanced, or performance.
        #[arg(long, env = "LAYERSTREAM_HARDWARE_PROFILE", default_value = "auto")]
        profile: String,
        /// Directory where tuned profiles are persisted (defaults to `$XDG_CONFIG_HOME/layerstream/tuning`).
        #[arg(long)]
        tuning_dir: Option<std::path::PathBuf>,
        /// Do not persist the tuned profile to disk.
        #[arg(long, default_value_t = false)]
        no_save: bool,
        /// Disable GPU probing during tuning.
        #[arg(long, default_value_t = false)]
        no_gpu: bool,
        /// Emit machine-readable JSON output.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Internal on-demand isolated GPU worker subprocess (spawned automatically by parent coordinator).
    #[command(name = "_worker", hide = true)]
    Worker,
}

fn validate_cli(cli: &Cli) -> Result<()> {
    if cli.command.is_some()
        && (!cli.plugin_args.models.is_empty()
            || cli.plugin_args.in_process
            || cli.plugin_args.no_gpu
            || cli.plugin_args.allow_slow
            || cli.plugin_args.low_vram
            || cli.plugin_args.idle_timeout_secs.is_some()
            || cli.plugin_args.threads.is_some())
    {
        anyhow::bail!(
            "engine flags (such as --model, --in-process, --idle-timeout-secs) must be placed after the subcommand (for example: 'layerstream-plugin serve --model ...')"
        );
    }
    Ok(())
}

fn run_pull_command(
    spec: &str,
    cache_dir: Option<std::path::PathBuf>,
) -> Result<std::path::PathBuf> {
    let root = cache_dir.unwrap_or_else(hf::default_cache_dir);
    let local = hf::ensure_hf_downloaded(spec, &root)?;
    println!("{}", local.display());
    Ok(local)
}

fn run_inspect_command(spec: &str, json: bool) -> Result<String> {
    let meta = if spec.starts_with("hf://") {
        let parsed = hf::HfSpec::parse(spec)?;
        hf::inspect_remote_gguf(&parsed)?
    } else {
        hf::inspect_local_gguf(std::path::Path::new(spec))?
    };
    let output = if json {
        serde_json::to_string_pretty(&meta)?
    } else {
        meta.summary_line()
    };
    println!("{output}");
    Ok(output)
}

fn format_recommend_report(
    recs: &[hf::QuantRecommendation],
    free_vram_bytes: u64,
    storage_bw: f64,
    no_draft: bool,
    header_peek: Option<&str>,
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Hardware profile: {:.1} GiB free GPU/UMA | {:.0} MiB/s NVMe | draft={}\n",
        free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        storage_bw,
        if no_draft { "no" } else { "yes" }
    ));
    out.push_str(&format!(
        "{:<3} {:<11} {:>8} {:>6} {:<12} {:>6} {:>13}  FILE\n",
        "", "QUANT", "SIZE", "SHARDS", "VERDICT", "PIN%", "EST TOK/S"
    ));
    for r in recs {
        let mark = if r.recommended { "★" } else { " " };
        let gib = r.candidate.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        out.push_str(&format!(
            "{:<3} {:<11} {:>6.2} G {:>6} {:<12} {:>5}% {:>5.2}-{:<6.2}  {}\n",
            mark,
            r.candidate.quant,
            gib,
            r.candidate.shard_count,
            r.verdict,
            r.pinned_pct,
            r.est_tps_low,
            r.est_tps_high,
            r.candidate.primary_file
        ));
    }
    if let Some(best) = recs.iter().find(|r| r.recommended) {
        if let Some(summary) = header_peek {
            out.push_str(&format!(
                "\nHeader peek ({}):\n",
                best.candidate.primary_file
            ));
            out.push_str(&format!("  {summary}\n"));
        }
        out.push_str("\nRecommended pull command:\n");
        out.push_str(&format!("  layerstream-plugin pull {}\n", best.hf_uri));
    }
    out
}

fn run_recommend_command(
    repo: &str,
    revision: Option<&str>,
    free_ram_gb: Option<f64>,
    nvme_mib_s: Option<f64>,
    no_draft: bool,
    cache_dir: Option<std::path::PathBuf>,
    json: bool,
) -> Result<String> {
    let (repo_id, rev) = hf::parse_repo_and_revision(repo, revision)?;
    let free_vram_bytes = if let Some(gb) = free_ram_gb {
        (gb.max(0.5) * 1024.0 * 1024.0 * 1024.0) as u64
    } else if let Some((free, _total)) = layerstream_core::StreamEngine::probe_gpu_memory(true) {
        free
    } else {
        anyhow::bail!(
            "no usable GPU found for live probe; pass --free-ram-gb <GiB> to evaluate target hardware"
        );
    };

    let storage_bw = if let Some(bw) = nvme_mib_s {
        bw.max(10.0)
    } else {
        let probe_dir = cache_dir.unwrap_or_else(hf::default_cache_dir);
        let _ = std::fs::create_dir_all(&probe_dir);
        probe_dir_storage_mib_per_s(&probe_dir).unwrap_or(1500.0)
    };

    let candidates = hf::list_hf_gguf_candidates(&repo_id, &rev)?;
    if candidates.is_empty() {
        anyhow::bail!("no GGUF files found in {repo_id}@{rev}");
    }
    let recs = hf::recommend_candidates(
        &repo_id,
        &rev,
        &candidates,
        free_vram_bytes,
        storage_bw,
        !no_draft,
    );

    let output = if json {
        serde_json::to_string_pretty(&recs)?
    } else {
        let header_peek = recs.iter().find(|r| r.recommended).and_then(|best| {
            let parsed = hf::HfSpec::parse(&best.hf_uri).ok()?;
            hf::inspect_remote_gguf(&parsed)
                .ok()
                .map(|meta| meta.summary_line())
        });
        format_recommend_report(
            &recs,
            free_vram_bytes,
            storage_bw,
            no_draft,
            header_peek.as_deref(),
        )
    };
    print!("{output}");
    if json {
        println!();
    }
    Ok(output)
}

fn run_check_command(engine: EngineArgs) -> Result<()> {
    let use_gpu = !engine.no_gpu;
    let allow_slow = engine.allow_slow;
    let server_cfg = engine.into_server_config()?;
    run_preflight_checks(&server_cfg.registry, use_gpu, allow_slow, true)
}

fn run_tune_command(
    model: Option<std::path::PathBuf>,
    cache_dir: Option<std::path::PathBuf>,
    profile_arg: &str,
    tuning_dir: Option<std::path::PathBuf>,
    no_save: bool,
    no_gpu: bool,
    json: bool,
) -> Result<String> {
    let profile_override: layerstream_core::ProfileOverride = profile_arg
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid --profile: {e}"))?;
    let probe_dir = cache_dir.unwrap_or_else(hf::default_cache_dir);
    let probe_path = model.as_deref().unwrap_or(probe_dir.as_path());
    let hw_profile =
        layerstream_core::HardwareProfile::detect(probe_path, !no_gpu, false, profile_override);

    let tuned = if let Some(ref model_path) = model {
        layerstream_core::run_storage_sweep(model_path, &hw_profile)?
    } else {
        layerstream_core::run_dir_storage_sweep(&probe_dir, &hw_profile)?
    };

    let saved_path = if no_save {
        None
    } else {
        let dir = tuning_dir.unwrap_or_else(layerstream_core::default_tuning_dir);
        Some(tuned.save(&dir)?)
    };

    let output = if json {
        serde_json::to_string_pretty(&tuned)?
    } else {
        tuned.format_report(saved_path.as_deref())
    };
    print!("{output}");
    if json {
        println!();
    }
    Ok(output)
}

fn validate_api_key(api_key: Option<&str>) -> Result<()> {
    if let Some(k) = api_key
        && k.trim().is_empty()
    {
        anyhow::bail!(
            "--api-key / LAYERSTREAM_API_KEY must not be empty; unset it to run unauthenticated"
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    // Log exclusively to stderr so stdout stays clean for CLI pipes and Mesh-LLM plugin host logs.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    validate_cli(&cli)?;

    if let Some(Command::Pull { spec, cache_dir }) = cli.command {
        let _ = run_pull_command(&spec, cache_dir)?;
        return Ok(());
    }

    if let Some(Command::Inspect { spec, json }) = cli.command {
        let _ = run_inspect_command(&spec, json)?;
        return Ok(());
    }

    if let Some(Command::Recommend {
        repo,
        revision,
        free_ram_gb,
        nvme_mib_s,
        no_draft,
        cache_dir,
        json,
    }) = cli.command
    {
        let _ = run_recommend_command(
            &repo,
            revision.as_deref(),
            free_ram_gb,
            nvme_mib_s,
            no_draft,
            cache_dir,
            json,
        )?;
        return Ok(());
    }

    if let Some(Command::Check { engine }) = cli.command {
        run_check_command(engine)?;
        return Ok(());
    }

    if let Some(Command::Tune {
        model,
        cache_dir,
        profile,
        tuning_dir,
        no_save,
        no_gpu,
        json,
    }) = cli.command
    {
        let _ = run_tune_command(
            model, cache_dir, &profile, tuning_dir, no_save, no_gpu, json,
        )?;
        return Ok(());
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("build tokio runtime")?;

    let result = rt.block_on(async move {
        match cli.command {
            Some(Command::Worker) => worker::run_worker_process().await,
            Some(Command::Serve {
                host,
                port,
                api_key,
                engine,
            }) => {
                validate_api_key(api_key.as_deref())?;
                let in_process = engine.in_process;
                let server_cfg = engine.into_server_config()?;
                let addr = format!("{host}:{port}");
                let listener = TcpListener::bind(&addr)
                    .await
                    .with_context(|| format!("bind standalone HTTP listener on {addr}"))?;
                let local_is_loopback = listener
                    .local_addr()
                    .map(|a| a.ip().is_loopback())
                    .unwrap_or(false);
                if !local_is_loopback && api_key.is_none() {
                    tracing::warn!(
                        addr = %addr,
                        "listening on non-loopback address without --api-key; /v1/* endpoints including /v1/unload are exposed unauthenticated"
                    );
                }
                tracing::info!(
                    addr = %addr,
                    models = server_cfg.registry.len(),
                    worker_isolated = !in_process,
                    authenticated = api_key.is_some(),
                    "starting standalone layerstream OpenAI server (0 MB GPU memory idle)"
                );
                if in_process {
                    let state = layerstream_server::AppState::new(server_cfg);
                    let shutdown_state = state.clone();
                    layerstream_server::serve_with_state_until(
                        listener,
                        state,
                        api_key,
                        async move {
                            mesh_adapter::shutdown_signal().await;
                            shutdown_state.request_shutdown();
                        },
                    )
                    .await
                } else {
                    let coordinator = worker::WorkerCoordinator::new(server_cfg)?;
                    let coord_shutdown = coordinator.clone();
                    let serve_res = tokio::select! {
                        res = worker::serve_with_coordinator_and_api_key(listener, coordinator, api_key) => res,
                        _ = mesh_adapter::shutdown_signal() => Ok(()),
                    };
                    coord_shutdown.shutdown_cleanly().await;
                    serve_res
                }
            }
            Some(Command::Plugin { engine }) => run_plugin_mode(engine).await,
            None => run_plugin_mode(cli.plugin_args).await,
            Some(
                Command::Pull { .. }
                | Command::Inspect { .. }
                | Command::Check { .. }
                | Command::Recommend { .. }
                | Command::Tune { .. },
            ) => {
                unreachable!()
            }
        }
    });

    // Bound runtime teardown so host EOF / SIGTERM always exits promptly instead of waiting
    // indefinitely on any still-unwinding blocking worker.
    rt.shutdown_timeout(Duration::from_secs(2));
    result
}

async fn run_plugin_mode(engine: EngineArgs) -> Result<()> {
    let in_process = engine.in_process;
    let has_cli_sources = engine.has_model_sources();
    let default_draft_model = engine.draft_model.clone();
    let cache_root = engine
        .cache_dir
        .clone()
        .unwrap_or_else(hf::default_cache_dir);
    let use_gpu = !engine.no_gpu;
    let allow_slow = engine.allow_slow;

    let (managed_plans, config_error) = match managed_models::load_managed_model_plans() {
        Ok(plans) => (plans, None),
        Err(err) => (Vec::new(), Some(format!("{err:#}"))),
    };
    let managed_idle_timeout_secs =
        managed_models::effective_managed_idle_timeout_secs(&managed_plans);
    let managed_startup = mesh_adapter::ManagedStartupConfig {
        plans: managed_plans,
        config_error,
        default_draft_model,
        cache_root,
        use_gpu,
        allow_slow,
    };

    let (server_cfg, deferred) =
        engine.into_server_config_with_deferred(true, managed_idle_timeout_secs)?;
    mesh_adapter::run_mesh_plugin(
        server_cfg,
        in_process,
        has_cli_sources,
        deferred,
        managed_startup,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_minimal_gguf(path: &std::path::Path) {
        let mut gguf = Vec::with_capacity(8192);
        gguf.extend_from_slice(b"GGUF");
        gguf.extend_from_slice(&3u32.to_le_bytes()); // version
        gguf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        gguf.extend_from_slice(&2u64.to_le_bytes()); // n_kv
        // KV 1: general.architecture = "llama" (type 8 = string)
        let k1 = b"general.architecture";
        gguf.extend_from_slice(&(k1.len() as u64).to_le_bytes());
        gguf.extend_from_slice(k1);
        gguf.extend_from_slice(&8u32.to_le_bytes());
        let v1 = b"llama";
        gguf.extend_from_slice(&(v1.len() as u64).to_le_bytes());
        gguf.extend_from_slice(v1);
        // KV 2: llama.block_count = 4u32 (type 4 = u32)
        let k2 = b"llama.block_count";
        gguf.extend_from_slice(&(k2.len() as u64).to_le_bytes());
        gguf.extend_from_slice(k2);
        gguf.extend_from_slice(&4u32.to_le_bytes());
        gguf.extend_from_slice(&4u32.to_le_bytes());
        gguf.resize(8192, 0);
        std::fs::write(path, &gguf).expect("write minimal gguf");
    }

    #[test]
    fn validate_cli_rejects_misplaced_engine_flags_and_accepts_subcommands() {
        let ok = Cli::try_parse_from(["layerstream-plugin", "check", "--no-gpu"])
            .expect("parse valid check");
        validate_cli(&ok).expect("valid subcommand passes");

        let misplaced = Cli::try_parse_from(["layerstream-plugin", "--no-gpu", "check"])
            .expect("parse misplaced flag");
        assert!(validate_cli(&misplaced).is_err());

        let misplaced_model =
            Cli::try_parse_from(["layerstream-plugin", "--model", "foo.gguf", "serve"])
                .expect("parse misplaced model");
        assert!(validate_cli(&misplaced_model).is_err());

        validate_api_key(None).expect("none key ok");
        validate_api_key(Some("secret-token")).expect("non-empty key ok");
        assert!(validate_api_key(Some("")).is_err());
        assert!(validate_api_key(Some("   ")).is_err());
    }

    #[test]
    fn inspect_check_pull_and_recommend_formatting_work() {
        let dir = std::env::temp_dir().join(format!("ls-main-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let gguf_path = dir.join("model.gguf");
        write_minimal_gguf(&gguf_path);

        let text_out =
            run_inspect_command(&gguf_path.display().to_string(), false).expect("inspect text");
        assert!(text_out.contains("unknown") || !text_out.is_empty());

        let json_out =
            run_inspect_command(&gguf_path.display().to_string(), true).expect("inspect json");
        assert!(json_out.contains("\"architecture\""));

        // Invalid hf:// spec errors cleanly
        assert!(run_inspect_command("hf://bad-spec-without-file", false).is_err());
        assert!(run_pull_command("hf://bad-spec", Some(dir.clone())).is_err());

        // Check subcommand in --no-gpu mode
        let cli = Cli::try_parse_from([
            "layerstream-plugin",
            "check",
            "--no-gpu",
            "--model",
            &gguf_path.display().to_string(),
        ])
        .expect("parse check");
        let Some(Command::Check { engine }) = cli.command else {
            panic!("expected Check command");
        };
        run_check_command(engine).expect("check --no-gpu succeeds");

        // Recommend report formatting
        let candidates = vec![
            hf::HfGgufCandidate {
                quant: "Q4_K_M".to_string(),
                primary_file: "model-Q4_K_M.gguf".to_string(),
                shard_count: 1,
                total_bytes: 8 * 1024 * 1024 * 1024,
            },
            hf::HfGgufCandidate {
                quant: "Q8_0".to_string(),
                primary_file: "model-Q8_0.gguf".to_string(),
                shard_count: 2,
                total_bytes: 16 * 1024 * 1024 * 1024,
            },
        ];
        let recs = hf::recommend_candidates(
            "org/model",
            "main",
            &candidates,
            6 * 1024 * 1024 * 1024,
            2400.0,
            true,
        );
        let report = format_recommend_report(
            &recs,
            6 * 1024 * 1024 * 1024,
            2400.0,
            false,
            Some("llama | 32L | ctx=4096"),
        );
        assert!(report.contains("Hardware profile: 6.0 GiB"));
        assert!(report.contains("Q4_K_M"));
        assert!(report.contains("Header peek (model-Q4_K_M.gguf):"));
        assert!(report.contains("Recommended pull command:"));

        // Recommend error path for invalid repo
        assert!(
            run_recommend_command(
                "not-a-valid-repo",
                None,
                Some(6.0),
                Some(2000.0),
                false,
                Some(dir.clone()),
                false
            )
            .is_err()
        );

        // End-to-end offline `run_recommend_command` (text + JSON) against a local mock HF server.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let server = std::thread::spawn(move || {
            use std::io::{BufRead, BufReader, Write};
            // Handles 3 requests: JSON mode tree API, text mode tree API, text mode header peek (404).
            for _ in 0..3 {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let mut reader = BufReader::new(stream.try_clone().expect("clone"));
                let mut req_line = String::new();
                let _ = reader.read_line(&mut req_line);
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap_or(0) > 0 && line != "\r\n" {
                    line.clear();
                }
                if req_line.contains("/api/models/") {
                    let body = r#"[{"type":"file","path":"demo-Q4_K_M.gguf","size":4294967296}]"#;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(resp.as_bytes());
                } else {
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                }
            }
        });
        let prev_ep = std::env::var_os("HF_ENDPOINT");
        // SAFETY: test binary runs single-threaded for this scoped mock request.
        unsafe { std::env::set_var("HF_ENDPOINT", format!("http://{addr}")) };
        let json_out = run_recommend_command(
            "org/demo",
            Some("main"),
            Some(8.0),
            Some(2000.0),
            false,
            Some(dir.clone()),
            true,
        )
        .expect("recommend json");
        assert!(json_out.contains("demo-Q4_K_M.gguf"));
        let text_out = run_recommend_command(
            "org/demo",
            Some("main"),
            Some(8.0),
            Some(2000.0),
            true,
            Some(dir.clone()),
            false,
        )
        .expect("recommend text");
        assert!(text_out.contains("Hardware profile: 8.0 GiB"));
        assert!(text_out.contains("Recommended pull command:"));
        match prev_ep {
            Some(v) => {
                // SAFETY: restores saved HF_ENDPOINT.
                unsafe { std::env::set_var("HF_ENDPOINT", v) }
            }
            None => {
                // SAFETY: removes test-only HF_ENDPOINT override.
                unsafe { std::env::remove_var("HF_ENDPOINT") }
            }
        }
        server.join().expect("join mock server");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tune_command_sweeps_local_file_formats_text_and_json_and_saves() {
        let dir = std::env::temp_dir().join(format!("ls-tune-cli-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let probe = dir.join("probe.gguf");
        std::fs::write(&probe, vec![0x42u8; 4 * 1024 * 1024]).expect("write 4 MiB probe");
        let tune_dir = dir.join("tuning");

        let text = run_tune_command(
            Some(probe.clone()),
            None,
            "auto",
            Some(tune_dir.clone()),
            false,
            true,
            false,
        )
        .expect("tune text + save");
        assert!(text.contains("Hardware profile:"));
        assert!(text.contains("Saved tuned profile:"));

        let json = run_tune_command(Some(probe), None, "edge", None, true, true, true)
            .expect("tune json + no_save");
        assert!(json.contains("\"schema_version\": 1"));
        assert!(json.contains("\"best_n_batch\": 256"));

        assert!(
            run_tune_command(None, Some(dir.clone()), "bogus", None, true, true, false).is_err()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
