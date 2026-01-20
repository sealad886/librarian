//! Embedding generation
//!
//! This module provides an abstraction over embedding models with:
//! - A trait for different embedding backends
//! - HTTP embedding backend
//! - Batch processing for efficiency

mod http_backend;

pub use http_backend::*;

use crate::config::ResolvedEmbeddingConfig;
use crate::error::{Error, Result};
use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct ImageEmbedInput {
    pub image_path: String,
    pub text: Option<String>,
}

pub fn normalize_embedding(vector: &[f32]) -> Vec<f32> {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        return vector.to_vec();
    }
    vector.iter().map(|v| v / norm).collect()
}

pub fn fuse_embeddings(image: &[f32], text: &[f32]) -> Vec<f32> {
    let image_norm = normalize_embedding(image);
    let text_norm = normalize_embedding(text);
    let mut combined = Vec::with_capacity(image.len());
    for (i, t) in image_norm.iter().zip(text_norm.iter()) {
        combined.push((i + t) / 2.0);
    }
    normalize_embedding(&combined)
}

/// Trait for embedding providers
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;

    /// Embed a batch of images (file paths)
    async fn embed_images(&self, _images: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Err(Error::Embedding(
            "Image embedding is not supported by this backend".to_string(),
        ))
    }

    /// Embed a batch of image + optional text inputs (joint models)
    async fn embed_image_text(&self, _inputs: Vec<ImageEmbedInput>) -> Result<Vec<Vec<f32>>> {
        Err(Error::Embedding(
            "Image+text embedding is not supported by this backend".to_string(),
        ))
    }

    /// Get the embedding dimension
    fn dimension(&self) -> usize;

    /// Get the model name
    fn model_name(&self) -> &str;
}

/// Create an embedder based on configuration
pub fn create_embedder(config: &ResolvedEmbeddingConfig) -> Result<Box<dyn Embedder>> {
    let embedder = HttpEmbedder::new(config)?;
    Ok(Box::new(embedder))
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

/// Helper to embed images in batches with progress
pub async fn embed_images_in_batches(
    embedder: &dyn Embedder,
    images: Vec<String>,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut all_embeddings = Vec::with_capacity(images.len());

    for chunk in images.chunks(batch_size) {
        let batch_images: Vec<String> = chunk.to_vec();
        let embeddings = embedder.embed_images(batch_images).await?;
        all_embeddings.extend(embeddings);
    }

    Ok(all_embeddings)
}

/// Helper to embed image+text inputs in batches
pub async fn embed_image_text_in_batches(
    embedder: &dyn Embedder,
    inputs: Vec<ImageEmbedInput>,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut all_embeddings = Vec::with_capacity(inputs.len());

    for chunk in inputs.chunks(batch_size) {
        let batch_inputs: Vec<ImageEmbedInput> = chunk.to_vec();
        let embeddings = embedder.embed_image_text(batch_inputs).await?;
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
