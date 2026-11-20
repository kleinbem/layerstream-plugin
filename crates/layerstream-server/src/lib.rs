pub mod parse;
pub mod registry;
pub mod routes;

use anyhow::Result;
use tokio::net::TcpListener;

pub use registry::{ModelEntry, ModelRegistry};
pub use routes::{AppState, ServerConfig, build_router};

pub async fn serve(listener: TcpListener, config: ServerConfig) -> Result<()> {
    let state = AppState::new(config);
    serve_with_state(listener, state).await
}

pub async fn serve_with_state(listener: TcpListener, state: AppState) -> Result<()> {
    state.spawn_reaper();
    let app = build_router(state);
    axum::serve(listener, app).await?;
    Ok(())
}
