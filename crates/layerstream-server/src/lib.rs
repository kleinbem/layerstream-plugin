pub mod parse;
pub mod registry;
pub mod routes;

use anyhow::Result;
use tokio::net::TcpListener;

pub use registry::{ModelEntry, ModelRegistry};
pub use routes::{
    AppState, ServerConfig, WORKER_TOKEN_HEADER, api_error, build_router,
    build_router_with_api_key, build_router_with_token, models_list_json,
};

pub async fn serve(listener: TcpListener, config: ServerConfig) -> Result<()> {
    serve_with_api_key(listener, config, None).await
}

pub async fn serve_with_api_key(
    listener: TcpListener,
    config: ServerConfig,
    api_key: Option<String>,
) -> Result<()> {
    let state = AppState::new(config);
    serve_with_state_and_api_key(listener, state, api_key).await
}

struct AbortOnDrop(Option<tokio::task::JoinHandle<()>>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            handle.abort();
        }
    }
}

pub async fn serve_with_state(listener: TcpListener, state: AppState) -> Result<()> {
    serve_with_state_and_api_key(listener, state, None).await
}

pub async fn serve_with_state_and_api_key(
    listener: TcpListener,
    state: AppState,
    api_key: Option<String>,
) -> Result<()> {
    let _reaper = AbortOnDrop(state.spawn_reaper());
    let app = build_router_with_api_key(state, api_key);
    axum::serve(listener, app).await?;
    Ok(())
}

pub async fn serve_with_state_until<F>(
    listener: TcpListener,
    state: AppState,
    api_key: Option<String>,
    shutdown: F,
) -> Result<()>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let _reaper = AbortOnDrop(state.spawn_reaper());
    let evict_state = state.clone();
    let app = build_router_with_api_key(state, api_key);
    let res = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await;
    evict_state.residency.evict_now().await;
    res?;
    Ok(())
}

pub async fn serve_with_state_and_token(
    listener: TcpListener,
    state: AppState,
    token: String,
) -> Result<()> {
    let _reaper = AbortOnDrop(state.spawn_reaper());
    let app = build_router_with_token(state, token);
    axum::serve(listener, app).await?;
    Ok(())
}
