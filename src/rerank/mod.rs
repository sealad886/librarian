//! Reranking support for search results

#[cfg(feature = "local-embed")]
mod fastembed_impl;

#[cfg(feature = "local-embed")]
pub use fastembed_impl::*;

use crate::config::RerankerConfig;
use crate::error::Result;
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

pub fn create_reranker(config: &RerankerConfig) -> Result<Box<dyn Reranker>> {
    #[cfg(feature = "local-embed")]
    {
        let reranker = FastEmbedReranker::new(config)?;
        Ok(Box::new(reranker))
    }

    #[cfg(not(feature = "local-embed"))]
    {
        Err(crate::error::Error::Embedding(
            "No reranker backend available. Enable 'local-embed' feature.".to_string(),
        ))
    }
}
