use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context;
use siglip_onnx_server::config::ServiceConfig;
use siglip_onnx_server::server::{OnnxTextEmbedder, build_router};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("reading SigLIP 2 ONNX service configuration");
    let service_config = ServiceConfig::from_environment()?;
    info!("loading local SigLIP 2 ONNX text artifacts");
    let text_embedder = Arc::new(OnnxTextEmbedder::load(service_config.clone())?);
    let socket_address = SocketAddr::from(([0, 0, 0, 0], service_config.port()));
    info!(%socket_address, "binding SigLIP 2 ONNX HTTP listener");
    let listener = tokio::net::TcpListener::bind(socket_address)
        .await
        .with_context(|| format!("binding SigLIP 2 ONNX service on {socket_address}"))?;

    info!(
        model_id = service_config.model_id(),
        revision = service_config.model_revision(),
        dimension = service_config.embedding_dimension(),
        "starting CPU-only SigLIP 2 ONNX text embedding service"
    );
    axum::serve(listener, build_router(text_embedder)).await?;
    Ok(())
}
