//! FastEmbed implementation for local embeddings

use super::Embedder;
use crate::config::EmbeddingConfig;
use crate::error::{Error, Result};
use async_trait::async_trait;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

/// FastEmbed-based embedder
pub struct FastEmbedder {
    model: Arc<Mutex<TextEmbedding>>,
    model_name: String,
    dimension: usize,
}

impl FastEmbedder {
    /// Create a new FastEmbed embedder
    pub fn new(config: &EmbeddingConfig) -> Result<Self> {
        info!("Initializing FastEmbed with model: {}", config.model);
        
        // Map model name to fastembed model enum
        let model_enum = match config.model.as_str() {
            "BAAI/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
            "BAAI/bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
            "BAAI/bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15,
            "sentence-transformers/all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2,
            _ => {
                // Try to use default model
                debug!("Unknown model '{}', using default BGESmallENV15", config.model);
                EmbeddingModel::BGESmallENV15
            }
        };

        let options = InitOptions::new(model_enum)
            .with_show_download_progress(true);

        let model = TextEmbedding::try_new(options)
            .map_err(|e| Error::Embedding(format!("Failed to initialize model: {}", e)))?;

        info!("FastEmbed model loaded successfully");

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            model_name: config.model.clone(),
            dimension: config.dimension,
        })
    }
}

#[async_trait]
impl Embedder for FastEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Embedding {} texts", texts.len());

        // FastEmbed is synchronous, so we wrap in blocking task
        let model = self.model.clone();
        let embeddings = tokio::task::spawn_blocking(move || {
            let model = model.blocking_lock();
            model.embed(texts, None)
        })
        .await
        .map_err(|e| Error::Embedding(format!("Task join error: {}", e)))?
        .map_err(|e| Error::Embedding(format!("Embedding failed: {}", e)))?;

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Get the expected dimension for a known model
pub fn get_model_dimension(model_name: &str) -> Option<usize> {
    match model_name {
        "BAAI/bge-small-en-v1.5" => Some(384),
        "BAAI/bge-base-en-v1.5" => Some(768),
        "BAAI/bge-large-en-v1.5" => Some(1024),
        "sentence-transformers/all-MiniLM-L6-v2" => Some(384),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_dimensions() {
        assert_eq!(get_model_dimension("BAAI/bge-small-en-v1.5"), Some(384));
        assert_eq!(get_model_dimension("BAAI/bge-base-en-v1.5"), Some(768));
        assert_eq!(get_model_dimension("unknown-model"), None);
    }

    // Integration test - requires model download
    #[tokio::test]
    #[ignore]  // Run manually with: cargo test -- --ignored
    async fn test_fastembed_integration() {
        let config = EmbeddingConfig {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            dimension: 384,
            batch_size: 32,
        };

        let embedder = FastEmbedder::new(&config).unwrap();
        let texts = vec!["Hello world".to_string(), "Test embedding".to_string()];
        
        let embeddings = embedder.embed(texts).await.unwrap();
        
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
    }
}
