pub mod parse;
pub mod registry;
pub mod routes;

use anyhow::Result;
use tokio::net::TcpListener;

pub use registry::{ModelEntry, ModelRegistry};
pub use routes::{
    AppState, ServerConfig, WORKER_TOKEN_HEADER, api_error, build_router, build_router_with_token,
    models_list_json,
};

pub async fn serve(listener: TcpListener, config: ServerConfig) -> Result<()> {
    let state = AppState::new(config);
    serve_with_state(listener, state).await
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
    let _reaper = AbortOnDrop(state.spawn_reaper());
    let app = build_router(state);
    axum::serve(listener, app).await?;
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
