//! Embedding generation
//!
//! This module provides an abstraction over embedding models with:
//! - A trait for different embedding backends
//! - Local embedding support via fastembed
//! - Batch processing for efficiency

#[cfg(feature = "local-embed")]
mod fastembed_impl;

#[cfg(feature = "local-embed")]
pub use fastembed_impl::*;

use crate::config::EmbeddingConfig;
use crate::error::Result;
use async_trait::async_trait;

/// Trait for embedding providers
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;

    /// Get the embedding dimension
    fn dimension(&self) -> usize;

    /// Get the model name
    fn model_name(&self) -> &str;
}

/// Create an embedder based on configuration
pub fn create_embedder(config: &EmbeddingConfig) -> Result<Box<dyn Embedder>> {
    #[cfg(feature = "local-embed")]
    {
        let embedder = FastEmbedder::new(config)?;
        Ok(Box::new(embedder))
    }

    #[cfg(not(feature = "local-embed"))]
    {
        Err(Error::Embedding(
            "No embedding backend available. Enable 'local-embed' feature.".to_string(),
        ))
    }
}

/// Helper to embed in batches with progress
pub async fn embed_in_batches(
    embedder: &dyn Embedder,
    texts: Vec<String>,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut all_embeddings = Vec::with_capacity(texts.len());

    for chunk in texts.chunks(batch_size) {
        let batch_texts: Vec<String> = chunk.to_vec();
        let embeddings = embedder.embed(batch_texts).await?;
        all_embeddings.extend(embeddings);
    }

    Ok(all_embeddings)
}

#[cfg(test)]
mod tests {
    // Note: Real embedding tests require model download
    // These are basic structural tests

    #[test]
    fn test_batch_splitting() {
        let texts: Vec<String> = (0..10).map(|i| format!("text {}", i)).collect();
        let chunks: Vec<_> = texts.chunks(3).collect();

        assert_eq!(chunks.len(), 4); // 3 + 3 + 3 + 1
        assert_eq!(chunks[0].len(), 3);
        assert_eq!(chunks[3].len(), 1);
    }
}
