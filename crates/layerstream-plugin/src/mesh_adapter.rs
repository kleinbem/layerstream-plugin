//! Native Mesh-LLM plugin adapter hosting `layerstream-server` in-process on an
//! ephemeral loopback port (0 MB VRAM idle, zero child-process orchestration).

use std::sync::Arc;

use anyhow::{Context, Result};
use layerstream_server::{AppState, ServerConfig, serve_with_state};
use mesh_llm_plugin::{
    PluginMetadata, PluginRuntime, PluginStartupPolicy, capability, plugin_server_info,
};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const PLUGIN_NAME: &str = "layerstream";
const ENDPOINT_ID: &str = "layerstream";

pub async fn run_mesh_plugin(server_config: ServerConfig) -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind ephemeral loopback port for layerstream plugin")?;
    let port = listener.local_addr()?.port();
    let endpoint_base = format!("http://127.0.0.1:{port}/v1");

    let state = AppState::new(server_config);
    let listener_slot = Arc::new(Mutex::new(Some(listener)));
    let startup_state = state.clone();
    let health_state = state.clone();

    let plugin = mesh_llm_plugin::plugin! {
        metadata: PluginMetadata::new(
            PLUGIN_NAME,
            VERSION,
            plugin_server_info(
                "layerstream-plugin",
                VERSION,
                "Layerstream GGUF NVMe Streaming Provider",
                "Runs oversized GGUF models via ping-pong GPU layer streaming with 0 MB idle VRAM.",
                Some(
                    "Configure [[plugin]] name = \"layerstream\" with args such as \
                     [\"--model\", \"/models/llama-70b.gguf\", \"--vram-budget-mb\", \"2048\"].",
                ),
            ),
        ),
        startup_policy: PluginStartupPolicy::Any,
        provides: [
            capability("endpoint:inference"),
            capability("endpoint:inference/openai_compatible"),
            capability("backend:layerstream"),
            capability("backend:ssd-layer-streaming"),
        ],
        inference: [
            mesh_llm_plugin::inference::provider(ENDPOINT_ID, endpoint_base.clone()),
        ],
        health: move |_context| {
            let health_state = health_state.clone();
            let endpoint_base = endpoint_base.clone();
            Box::pin(async move {
                let loaded = if health_state.residency.is_loaded() {
                    "loaded"
                } else {
                    "idle_0mb_vram"
                };
                Ok(format!(
                    "endpoint={endpoint_base} models={} state={loaded}",
                    health_state.config.registry.len()
                ))
            })
        },
        on_initialized: move |_context| {
            let startup_state = startup_state.clone();
            let listener_slot = listener_slot.clone();
            Box::pin(async move {
                let Some(listener) = listener_slot.lock().await.take() else {
                    return Ok(());
                };
                tokio::spawn(async move {
                    if let Err(err) = serve_with_state(listener, startup_state).await {
                        tracing::error!("layerstream HTTP server exited with error: {err:#}");
                    }
                });
                Ok(())
            })
        },
    };

    PluginRuntime::run(plugin).await
}
