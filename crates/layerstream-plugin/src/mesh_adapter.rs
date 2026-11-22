//! Native Mesh-LLM plugin adapter hosting `layerstream-server` on an ephemeral
//! loopback port (0 MB VRAM/UMA and 0 MB GPU driver context while idle via
//! on-demand worker isolation, or opt-in `--in-process` mode).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use layerstream_server::{AppState, ModelRegistry, ServerConfig, serve_with_state};
use mesh_llm_plugin::models::ASSET_CAPABILITY;
use mesh_llm_plugin::{
    PROTOCOL_VERSION, PluginError, PluginMetadata, PluginRuntime, PluginStartupPolicy,
    SimplePlugin, capability, plugin_server_info,
};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, oneshot};
use tokio::task::JoinHandle;

use crate::config::DeferredPullPlan;
use crate::managed_models::{ManagedModelPlan, prepare_managed_models};
use crate::worker::{WorkerCoordinator, serve_with_coordinator};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_PLUGIN_NAME: &str = "layerstream";
const ENDPOINT_ID: &str = "layerstream";

#[derive(Debug, Clone)]
pub struct ManagedStartupConfig {
    pub plans: Vec<ManagedModelPlan>,
    pub config_error: Option<String>,
    pub default_draft_model: Option<String>,
    pub cache_root: PathBuf,
    pub use_gpu: bool,
    pub allow_slow: bool,
}

impl Default for ManagedStartupConfig {
    fn default() -> Self {
        Self {
            plans: Vec::new(),
            config_error: None,
            default_draft_model: None,
            cache_root: crate::hf::default_cache_dir(),
            use_gpu: false,
            allow_slow: false,
        }
    }
}

#[derive(Clone)]
pub enum BackendHandle {
    Isolated(WorkerCoordinator),
    InProcess(AppState),
}

impl BackendHandle {
    fn model_count(&self) -> usize {
        match self {
            Self::Isolated(c) => c.model_count(),
            Self::InProcess(s) => s.model_count(),
        }
    }

    fn is_loaded(&self) -> bool {
        match self {
            Self::Isolated(c) => c.is_loaded(),
            Self::InProcess(s) => s.residency.is_loaded(),
        }
    }

    fn registry_snapshot(&self) -> ModelRegistry {
        match self {
            Self::Isolated(c) => c.registry_snapshot(),
            Self::InProcess(s) => s.registry_snapshot(),
        }
    }

    fn set_registry(&self, registry: ModelRegistry) {
        match self {
            Self::Isolated(c) => c.set_registry(registry),
            Self::InProcess(s) => s.set_registry(registry),
        }
    }

    fn set_loading_status(&self, status: Option<String>) {
        match self {
            Self::Isolated(c) => c.set_loading_status(status),
            Self::InProcess(s) => s.set_loading_status(status),
        }
    }

    fn set_loading_failed(&self, err: String) {
        match self {
            Self::Isolated(c) => c.set_loading_failed(err),
            Self::InProcess(s) => s.set_loading_failed(err),
        }
    }

    fn loading_status(&self) -> Option<String> {
        match self {
            Self::Isolated(c) => c.loading_status(),
            Self::InProcess(s) => s.loading_status(),
        }
    }

    fn loading_error(&self) -> Option<String> {
        match self {
            Self::Isolated(c) => c.loading_error(),
            Self::InProcess(s) => s.loading_error(),
        }
    }

    async fn shutdown_cleanly(&self) {
        match self {
            Self::Isolated(c) => c.shutdown_cleanly().await,
            Self::InProcess(s) => s.request_shutdown(),
        }
    }

    async fn serve(self, listener: TcpListener) -> Result<()> {
        match self {
            Self::Isolated(c) => serve_with_coordinator(listener, c).await,
            Self::InProcess(s) => serve_with_state(listener, s).await,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_plugin(
    endpoint_base: String,
    state: BackendHandle,
    has_cli_model_sources: bool,
    listener_slot: Arc<Mutex<Option<TcpListener>>>,
    shutdown_rx_slot: Arc<Mutex<Option<oneshot::Receiver<()>>>>,
    server_handle_slot: Arc<Mutex<Option<JoinHandle<()>>>>,
    prep_handle_slot: Arc<Mutex<Option<JoinHandle<()>>>>,
    deferred_pull: Option<DeferredPullPlan>,
    managed_startup: ManagedStartupConfig,
) -> SimplePlugin {
    let plugin_name =
        std::env::var("MESH_LLM_PLUGIN_NAME").unwrap_or_else(|_| DEFAULT_PLUGIN_NAME.to_string());
    let startup_state = state.clone();
    let health_state = state.clone();
    let health_handle = server_handle_slot.clone();
    let deferred_slot = Arc::new(Mutex::new(deferred_pull));
    let has_managed_models = !managed_startup.plans.is_empty();
    let managed_config_error = managed_startup.config_error.clone();
    let managed_slot = Arc::new(Mutex::new(Some(managed_startup)));

    mesh_llm_plugin::plugin! {
        metadata: PluginMetadata::new(
            plugin_name.clone(),
            VERSION,
            plugin_server_info(
                plugin_name,
                VERSION,
                "Layerstream GGUF NVMe Streaming Provider",
                "Runs oversized dense & MoE GGUF models via GPU layer streaming with 0 MB idle model weights.",
                Some(
                    "Configure [[models]] with provider = \"layerstream\" (uses mesh-llm's shared \
                     Hugging Face cache via model_assets.v1), or pass args such as \
                     [\"--model\", \"hf://org/repo:model.gguf\"].",
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
            let health_handle = health_handle.clone();
            let endpoint_base = endpoint_base.clone();
            Box::pin(async move {
                if let Some(handle) = health_handle.lock().await.as_ref()
                    && handle.is_finished()
                {
                    anyhow::bail!("layerstream HTTP server task exited unexpectedly");
                }
                if let Some(err) = health_state.loading_error() {
                    anyhow::bail!("background model preparation failed: {err}");
                }
                if let Some(status) = health_state.loading_status() {
                    return Ok(format!(
                        "endpoint={endpoint_base} models={} state=downloading detail={status}",
                        health_state.model_count()
                    ));
                }
                let loaded = if health_state.is_loaded() {
                    "loaded"
                } else {
                    "idle_0mb_vram"
                };
                Ok(format!(
                    "endpoint={endpoint_base} models={} state={loaded}",
                    health_state.model_count()
                ))
            })
        },
        on_initialized: move |context| {
            let startup_state = startup_state.clone();
            let listener_slot = listener_slot.clone();
            let shutdown_rx_slot = shutdown_rx_slot.clone();
            let server_handle_slot = server_handle_slot.clone();
            let prep_handle_slot = prep_handle_slot.clone();
            let deferred_slot = deferred_slot.clone();
            let managed_slot = managed_slot.clone();
            let owned_ctx = context.owned();
            Box::pin(async move {
                let Some(listener) = listener_slot.lock().await.take() else {
                    return Ok(());
                };
                let deferred_plan = deferred_slot.lock().await.take();
                let managed_cfg = managed_slot
                    .lock()
                    .await
                    .take()
                    .filter(|cfg| !cfg.plans.is_empty());

                if deferred_plan.is_some() || managed_cfg.is_some() {
                    let initial_status = match (&deferred_plan, &managed_cfg) {
                        (Some(cli_plan), Some(managed)) => format!(
                            "preparing {} CLI and {} host-managed model(s)",
                            cli_plan.models.len(),
                            managed.plans.len()
                        ),
                        (None, Some(managed)) => format!(
                            "preparing {} host-managed model(s) via {ASSET_CAPABILITY}",
                            managed.plans.len()
                        ),
                        (Some(cli_plan), None) => format!(
                            "pulling {} model(s) from Hugging Face",
                            cli_plan.models.len()
                        ),
                        (None, None) => unreachable!(),
                    };
                    startup_state.set_loading_status(Some(initial_status));
                }

                let shutdown_rx = shutdown_rx_slot.lock().await.take();
                let serve_state = startup_state.clone();
                let handle = tokio::spawn(async move {
                    let serve_fut = serve_state.serve(listener);
                    tokio::pin!(serve_fut);
                    if let Some(rx) = shutdown_rx {
                        tokio::select! {
                            res = &mut serve_fut => {
                                if let Err(err) = res {
                                    tracing::error!("layerstream HTTP server exited with error: {err:#}");
                                }
                            }
                            _ = rx => {
                                tracing::info!("layerstream HTTP server shutting down cleanly");
                            }
                        }
                    } else if let Err(err) = serve_fut.await {
                        tracing::error!("layerstream HTTP server exited with error: {err:#}");
                    }
                });
                *server_handle_slot.lock().await = Some(handle);

                if deferred_plan.is_some() || managed_cfg.is_some() {
                    let pull_state = startup_state;
                    let prep_handle = tokio::spawn(async move {
                        let result: Result<ModelRegistry> = async {
                            let base_registry = if let Some(plan) = deferred_plan {
                                tokio::task::spawn_blocking(move || plan.execute())
                                    .await
                                    .context("join background model pull task")??
                            } else {
                                pull_state.registry_snapshot()
                            };

                            if let Some(managed) = managed_cfg {
                                let status_state = pull_state.clone();
                                prepare_managed_models(
                                    &owned_ctx,
                                    &managed.plans,
                                    base_registry,
                                    managed.default_draft_model,
                                    managed.cache_root,
                                    managed.use_gpu,
                                    managed.allow_slow,
                                    move |msg| status_state.set_loading_status(Some(msg)),
                                )
                                .await
                            } else {
                                Ok(base_registry)
                            }
                        }
                        .await;

                        match result {
                            Ok(registry) => {
                                let count = registry.len();
                                pull_state.set_registry(registry);
                                pull_state.set_loading_status(None);
                                tracing::info!(
                                    models = count,
                                    "background model preparation complete; ready to serve"
                                );
                            }
                            Err(err) => {
                                tracing::error!("background model preparation failed: {err:#}");
                                pull_state.set_loading_failed(format!("{err:#}"));
                            }
                        }
                    });
                    *prep_handle_slot.lock().await = Some(prep_handle);
                }
                Ok(())
            })
        },
    }
    .on_initialize(move |request, _| {
        let managed_config_error = managed_config_error.clone();
        Box::pin(async move {
            if request.host_protocol_version != PROTOCOL_VERSION {
                return Err(PluginError::startup_disabled(format!(
                    "incompatible Mesh plugin protocol: host={}, plugin={PROTOCOL_VERSION}",
                    request.host_protocol_version
                )));
            }
            if let Some(err) = managed_config_error {
                return Err(PluginError::startup_disabled(format!(
                    "invalid managed model configuration: {err}"
                )));
            }
            if !has_cli_model_sources && !has_managed_models {
                return Err(PluginError::startup_disabled(
                    "no models configured; configure [[models]] with provider = \"layerstream\", \
                     or pass --model <GGUF_OR_HF> / --models-dir <DIR>",
                ));
            }
            if has_managed_models
                && !request
                    .host_capabilities
                    .iter()
                    .any(|cap| cap == ASSET_CAPABILITY)
            {
                return Err(PluginError::startup_disabled(format!(
                    "managed models require host capability {ASSET_CAPABILITY}; \
                     upgrade mesh-llm or pass --model/--models-dir"
                )));
            }
            Ok(())
        })
    })
}

pub async fn run_mesh_plugin(
    server_config: ServerConfig,
    in_process: bool,
    has_cli_model_sources: bool,
    deferred_pull: Option<DeferredPullPlan>,
    managed_startup: ManagedStartupConfig,
) -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind ephemeral loopback port for layerstream plugin")?;
    let port = listener.local_addr()?.port();
    let endpoint_base = format!("http://127.0.0.1:{port}/v1");

    let state = if in_process {
        BackendHandle::InProcess(AppState::new(server_config))
    } else {
        BackendHandle::Isolated(WorkerCoordinator::new(server_config)?)
    };
    let listener_slot = Arc::new(Mutex::new(Some(listener)));
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let shutdown_rx_slot = Arc::new(Mutex::new(Some(shutdown_rx)));
    let server_handle_slot = Arc::new(Mutex::new(None));
    let prep_handle_slot = Arc::new(Mutex::new(None));

    let plugin = build_plugin(
        endpoint_base,
        state.clone(),
        has_cli_model_sources,
        listener_slot,
        shutdown_rx_slot,
        server_handle_slot.clone(),
        prep_handle_slot.clone(),
        deferred_pull,
        managed_startup,
    );

    let result = tokio::select! {
        res = PluginRuntime::run(plugin) => res,
        _ = shutdown_signal() => Ok(()),
    };

    if let Some(prep_handle) = prep_handle_slot.lock().await.take() {
        prep_handle.abort();
    }

    // Immediately cancel any in-flight prefill/decode and reap any isolated GPU worker
    // so GPU/RAM buffers and driver contexts are released promptly on host EOF,
    // ShutdownRequest, or SIGTERM.
    state.shutdown_cleanly().await;
    let _ = shutdown_tx.send(());
    if let Some(handle) = server_handle_slot.lock().await.take() {
        let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
    }
    result
}

pub(crate) async fn shutdown_signal() {
    #[cfg(unix)]
    {
        if let Ok(mut term) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            tokio::select! {
                _ = term.recv() => {},
                _ = tokio::signal::ctrl_c() => {},
            }
            return;
        }
    }
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use layerstream_core::KvCacheType;
    use layerstream_server::ModelRegistry;
    use mesh_llm_plugin::Plugin;

    use super::*;

    #[test]
    fn manifest_registers_managed_openai_compatible_provider() {
        let cfg = ServerConfig {
            registry: ModelRegistry::default(),
            n_ctx: 512,
            n_batch: 128,
            n_threads: 2,
            resident_layers: Some(0),
            vram_budget_bytes: 0,
            vram_budget_auto: false,
            kv_cache_type: KvCacheType::Q8_0,
            use_gpu: false,
            prefer_direct_io: true,
            cpu_embeddings: false,
            cpu_head: false,
            stream_head: false,
            moe_active_streaming: true,
            idle_timeout: Duration::ZERO,
            enable_ngram_speculation: true,
            max_draft_tokens: 8,
            max_queued_requests: 4,
        };
        let plugin = build_plugin(
            "http://127.0.0.1:1/v1".to_string(),
            BackendHandle::InProcess(AppState::new(cfg)),
            true,
            Arc::new(Mutex::new(None)),
            Arc::new(Mutex::new(None)),
            Arc::new(Mutex::new(None)),
            Arc::new(Mutex::new(None)),
            None,
            ManagedStartupConfig::default(),
        );
        let manifest = plugin.manifest().unwrap();
        assert_eq!(manifest.endpoints.len(), 1);
        assert!(manifest.endpoints[0].managed_by_plugin);
        assert_eq!(
            manifest.endpoints[0].protocol.as_deref(),
            Some("openai_compatible")
        );
    }
}
