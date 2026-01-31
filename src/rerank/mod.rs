//! Reranking support for search results
//!
//! Provides reranking capabilities using:
//! - HTTP backend (for custom servers)
//! - Xinference backend (zero-config, automatic)

mod http_backend;

pub use http_backend::*;

use crate::config::RerankerConfig;
use crate::embedding_backend::EmbeddingBackendKind;
use crate::error::{Error, Result};
use crate::models::reranker_model_spec;
use crate::xinference::{
    ensure_xinference_ready, get_or_init_xinference_manager, normalize_xinference_base_url,
    xinference_auth_token, SharedXinferenceManager, XinferenceReranker,
};
use async_trait::async_trait;
use std::path::Path;
use tracing::info;

#[derive(Debug, Clone)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
}

#[async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(&self, query: &str, documents: Vec<String>) -> Result<Vec<RerankResult>>;
    fn model_name(&self) -> &str;
}

/// Create a reranker (sync, for HTTP backend only)
pub fn create_reranker(config: &RerankerConfig, backend_url: &str) -> Result<Box<dyn Reranker>> {
    if reranker_model_spec(&config.model).is_none() {
        return Err(Error::Embedding(format!(
            "Reranker model '{}' is not allowlisted",
            config.model
        )));
    }

    let reranker = HttpReranker::new(config, backend_url)?;
    Ok(Box::new(reranker))
}

/// Create a reranker automatically based on backend kind
///
/// For xinference backend: automatically ensures xinference is installed, server is running,
/// and model is launched before creating the reranker.
///
/// For http backend: creates HttpReranker directly.
pub async fn create_reranker_auto(
    config: &RerankerConfig,
    base_dir: &Path,
) -> Result<Box<dyn Reranker>> {
    let backend_kind: EmbeddingBackendKind = config
        .backend
        .parse()
        .unwrap_or(EmbeddingBackendKind::Xinference);

    match backend_kind {
        EmbeddingBackendKind::Xinference => {
            info!("Using Xinference reranker backend, ensuring dependencies...");

            // Ensure xinference is installed and ready (sync check)
            ensure_xinference_ready(base_dir)?;

            let base_url = normalize_xinference_base_url(&config.url)?;
            let auth_token = xinference_auth_token();

            // Get or initialize the global manager
            let manager_lock = get_or_init_xinference_manager(&base_url, auth_token).await?;

            // Ensure server is running
            {
                let mut guard = manager_lock.lock().await;
                if let Some(ref mut manager) = *guard {
                    manager.ensure_running().await?;
                }
            }

            // Create reranker (this will launch model if needed)
            let reranker =
                XinferenceReranker::from_global_manager(manager_lock.clone(), &config.model)
                    .await?;
            Ok(Box::new(reranker) as Box<dyn Reranker>)
        }
        EmbeddingBackendKind::Http => {
            if reranker_model_spec(&config.model).is_none() {
                return Err(Error::Embedding(format!(
                    "Reranker model '{}' is not allowlisted",
                    config.model
                )));
            }
            let reranker = HttpReranker::new(config, &config.url)?;
            Ok(Box::new(reranker) as Box<dyn Reranker>)
        }
    }
}

/// Create a reranker with optional Xinference manager support
pub async fn create_reranker_with_xinference(
    config: &RerankerConfig,
    backend_kind: EmbeddingBackendKind,
    backend_url: &str,
    xinf_manager: Option<SharedXinferenceManager>,
) -> Result<Box<dyn Reranker>> {
    match backend_kind {
        EmbeddingBackendKind::Xinference => {
            let manager = xinf_manager.ok_or_else(|| {
                Error::Config("Xinference manager required for xinference backend".into())
            })?;
            let reranker = XinferenceReranker::new(manager, &config.model).await?;
            Ok(Box::new(reranker) as Box<dyn Reranker>)
        }
        EmbeddingBackendKind::Http => {
            if reranker_model_spec(&config.model).is_none() {
                return Err(Error::Embedding(format!(
                    "Reranker model '{}' is not allowlisted",
                    config.model
                )));
            }
            let reranker = HttpReranker::new(config, backend_url)?;
            Ok(Box::new(reranker) as Box<dyn Reranker>)
        }
    }
}
