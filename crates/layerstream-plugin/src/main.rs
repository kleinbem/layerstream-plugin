mod config;
mod hf;
mod mesh_adapter;

use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
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

    if let Some(Command::Pull { spec, cache_dir }) = cli.command {
        let root = cache_dir.unwrap_or_else(hf::default_cache_dir);
        let local = hf::ensure_hf_downloaded(&spec, &root)?;
        println!("{}", local.display());
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
            Some(Command::Serve { host, port, engine }) => {
                let server_cfg = engine.into_server_config()?;
                let addr = format!("{host}:{port}");
                let listener = TcpListener::bind(&addr)
                    .await
                    .with_context(|| format!("bind standalone HTTP listener on {addr}"))?;
                tracing::info!(
                    addr = %addr,
                    models = server_cfg.registry.len(),
                    "starting standalone layerstream OpenAI server (0 MB model/KV weights idle)"
                );
                layerstream_server::serve(listener, server_cfg).await
            }
            Some(Command::Plugin { engine }) => {
                let has_sources = engine.has_model_sources();
                let (server_cfg, deferred) = engine.into_server_config_with_deferred(true)?;
                mesh_adapter::run_mesh_plugin(server_cfg, has_sources, deferred).await
            }
            None => {
                let has_sources = cli.plugin_args.has_model_sources();
                let (server_cfg, deferred) =
                    cli.plugin_args.into_server_config_with_deferred(true)?;
                mesh_adapter::run_mesh_plugin(server_cfg, has_sources, deferred).await
            }
            Some(Command::Pull { .. } | Command::Check { .. }) => unreachable!(),
        }
    });

    // Bound runtime teardown so host EOF / SIGTERM always exits promptly instead of waiting
    // indefinitely on any still-unwinding blocking worker.
    rt.shutdown_timeout(Duration::from_secs(2));
    result
}
