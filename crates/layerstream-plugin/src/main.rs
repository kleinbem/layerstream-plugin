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
    /// Internal on-demand isolated GPU worker subprocess (spawned automatically by parent coordinator).
    #[command(name = "_worker", hide = true)]
    Worker,
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

    if let Some(Command::Pull { spec, cache_dir }) = cli.command {
        let root = cache_dir.unwrap_or_else(hf::default_cache_dir);
        let local = hf::ensure_hf_downloaded(&spec, &root)?;
        println!("{}", local.display());
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
        let (repo_id, rev) = hf::parse_repo_and_revision(&repo, revision.as_deref())?;
        let free_vram_bytes = if let Some(gb) = free_ram_gb {
            (gb.max(0.5) * 1024.0 * 1024.0 * 1024.0) as u64
        } else if let Some((free, _total)) = layerstream_core::StreamEngine::probe_gpu_memory(true)
        {
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

        if json {
            println!("{}", serde_json::to_string_pretty(&recs)?);
            return Ok(());
        }

        println!(
            "Hardware profile: {:.1} GiB free GPU/UMA | {:.0} MiB/s NVMe | draft={}",
            free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            storage_bw,
            if no_draft { "no" } else { "yes" }
        );
        println!(
            "{:<3} {:<11} {:>8} {:>6} {:<12} {:>6} {:>13}  FILE",
            "", "QUANT", "SIZE", "SHARDS", "VERDICT", "PIN%", "EST TOK/S"
        );
        for r in &recs {
            let mark = if r.recommended { "★" } else { " " };
            let gib = r.candidate.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            println!(
                "{:<3} {:<11} {:>6.2} G {:>6} {:<12} {:>5}% {:>5.2}-{:<6.2}  {}",
                mark,
                r.candidate.quant,
                gib,
                r.candidate.shard_count,
                r.verdict,
                r.pinned_pct,
                r.est_tps_low,
                r.est_tps_high,
                r.candidate.primary_file
            );
        }
        if let Some(best) = recs.iter().find(|r| r.recommended) {
            println!("\nRecommended pull command:");
            println!("  layerstream-plugin pull {}", best.hf_uri);
        }
        return Ok(());
    }

    if let Some(Command::Check { engine }) = cli.command {
        let use_gpu = !engine.no_gpu;
        let allow_slow = engine.allow_slow;
        let server_cfg = engine.into_server_config()?;
        run_preflight_checks(&server_cfg.registry, use_gpu, allow_slow, true)?;
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
                if let Some(ref k) = api_key
                    && k.trim().is_empty()
                {
                    anyhow::bail!(
                        "--api-key / LAYERSTREAM_API_KEY must not be empty; unset it to run unauthenticated"
                    );
                }
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
            Some(Command::Pull { .. } | Command::Check { .. } | Command::Recommend { .. }) => {
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
