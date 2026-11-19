mod config;
mod mesh_adapter;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

use crate::config::EngineArgs;

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
    /// Run as a Mesh-LLM stdio plugin (also the default when no subcommand is passed).
    Plugin {
        #[command(flatten)]
        engine: EngineArgs,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Log exclusively to stderr so Mesh-LLM's stdio protobuf channel stays clean.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

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
                "starting standalone layerstream OpenAI server (0 MB VRAM idle)"
            );
            layerstream_server::serve(listener, server_cfg).await
        }
        Some(Command::Plugin { engine }) => {
            let server_cfg = engine.into_server_config()?;
            mesh_adapter::run_mesh_plugin(server_cfg).await
        }
        None => {
            let server_cfg = cli.plugin_args.into_server_config()?;
            mesh_adapter::run_mesh_plugin(server_cfg).await
        }
    }
}
