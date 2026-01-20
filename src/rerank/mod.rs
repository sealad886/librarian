//! Reranking support for search results

mod http_backend;

pub use http_backend::*;

use crate::config::RerankerConfig;
use crate::error::{Error, Result};
use crate::models::reranker_model_spec;
use async_trait::async_trait;

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
