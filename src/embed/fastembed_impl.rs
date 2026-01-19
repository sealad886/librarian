//! FastEmbed implementation for local embeddings

use super::Embedder;
use crate::config::EmbeddingConfig;
use crate::error::{Error, Result};
use async_trait::async_trait;
use fastembed::{EmbeddingModel, ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, InitOptions, TextEmbedding};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

/// FastEmbed-based embedder
pub struct FastEmbedder {
    model: Arc<Mutex<TextEmbedding>>,
    image_model: Option<Arc<Mutex<ImageEmbedding>>>,
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
                debug!(
                    "Unknown model '{}', using default BGESmallENV15",
                    config.model
                );
                EmbeddingModel::BGESmallENV15
            }
        };

        let options = InitOptions::new(model_enum).with_show_download_progress(true);

        let model = TextEmbedding::try_new(options)
            .map_err(|e| Error::Embedding(format!("Failed to initialize model: {}", e)))?;

        info!("FastEmbed model loaded successfully");

        let image_model = if config.supports_multimodal {
            info!("Initializing FastEmbed image model for multimodal embeddings");
            let image_options =
                ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32).with_show_download_progress(true);
            let image_model = ImageEmbedding::try_new(image_options)
                .map_err(|e| Error::Embedding(format!("Failed to initialize image model: {}", e)))?;
            Some(Arc::new(Mutex::new(image_model)))
        } else {
            None
        };

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            image_model,
            model_name: config.model.clone(),
            dimension: config.resolved_dimension(),
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

    async fn embed_images(&self, images: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let model = self.image_model.clone().ok_or_else(|| {
            Error::Embedding("Image embedding is not available for this model".to_string())
        })?;

        debug!("Embedding {} images", images.len());

        let embeddings = tokio::task::spawn_blocking(move || {
            let model = model.blocking_lock();
            model.embed(images, None)
        })
        .await
        .map_err(|e| Error::Embedding(format!("Task join error: {}", e)))?
        .map_err(|e| Error::Embedding(format!("Image embedding failed: {}", e)))?;

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::embedding_dimension_for_model;

    #[test]
    fn test_model_dimensions() {
        assert_eq!(
            embedding_dimension_for_model("BAAI/bge-small-en-v1.5"),
            Some(384)
        );
        assert_eq!(
            embedding_dimension_for_model("BAAI/bge-base-en-v1.5"),
            Some(768)
        );
        assert_eq!(embedding_dimension_for_model("unknown-model"), None);
    }

    // Integration test - requires model download
    #[tokio::test]
    #[ignore] // Run manually with: cargo test -- --ignored
    async fn test_fastembed_integration() {
        let config = EmbeddingConfig {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            dimension: 384,
            batch_size: 32,
            supports_multimodal: false,
        };

        let embedder = FastEmbedder::new(&config).unwrap();
        let texts = vec!["Hello world".to_string(), "Test embedding".to_string()];

        let embeddings = embedder.embed(texts).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
    }
}
