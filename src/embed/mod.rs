//! Embedding generation
//!
//! This module provides an abstraction over embedding models with:
//! - A trait for different embedding backends
//! - HTTP embedding backend (for custom servers)
//! - Xinference embedding backend (zero-config, automatic)
//! - Batch processing for efficiency
//! - Multimodal support (text, image, audio, video)

mod http_backend;

pub use http_backend::*;

use crate::config::ResolvedEmbeddingConfig;
use crate::embedding_backend::EmbeddingBackendKind;
use crate::error::{Error, Result};
use crate::xinference::{
    ensure_xinference_ready, get_or_init_xinference_manager, XinferenceEmbedder,
    XinferenceManager,
};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

/// Supported media modalities for multimodal embedding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaModality {
    Text,
    Image,
    Audio,
    Video,
}

/// Input for image embedding (with optional text for joint models)
#[derive(Debug, Clone)]
pub struct ImageEmbedInput {
    pub image_path: String,
    pub text: Option<String>,
}

/// Input for audio embedding (with optional text for joint models)
#[derive(Debug, Clone)]
pub struct AudioEmbedInput {
    pub audio_path: String,
    pub text: Option<String>,
}

/// Input for video embedding (with optional text for joint models)
#[derive(Debug, Clone)]
pub struct VideoEmbedInput {
    pub video_path: String,
    pub text: Option<String>,
}

/// Unified multimodal embedding input supporting all modalities
#[derive(Debug, Clone)]
pub enum MultimodalInput {
    /// Text-only input
    Text(String),
    /// Image input with optional text
    Image(ImageEmbedInput),
    /// Audio input with optional text
    Audio(AudioEmbedInput),
    /// Video input with optional text
    Video(VideoEmbedInput),
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
    async fn embed_multimode(&self, _inputs: Vec<ImageEmbedInput>) -> Result<Vec<Vec<f32>>> {
        Err(Error::Embedding(
            "Image+text embedding is not supported by this backend".to_string(),
        ))
    }

    /// Embed a batch of audio files (file paths)
    async fn embed_audio(&self, _audios: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Err(Error::Embedding(
            "Audio embedding is not supported by this backend".to_string(),
        ))
    }

    /// Embed a batch of audio + optional text inputs (joint models)
    async fn embed_audio_multimode(&self, _inputs: Vec<AudioEmbedInput>) -> Result<Vec<Vec<f32>>> {
        Err(Error::Embedding(
            "Audio+text embedding is not supported by this backend".to_string(),
        ))
    }

    /// Embed a batch of video files (file paths)
    async fn embed_video(&self, _videos: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Err(Error::Embedding(
            "Video embedding is not supported by this backend".to_string(),
        ))
    }

    /// Embed a batch of video + optional text inputs (joint models)
    async fn embed_video_multimode(&self, _inputs: Vec<VideoEmbedInput>) -> Result<Vec<Vec<f32>>> {
        Err(Error::Embedding(
            "Video+text embedding is not supported by this backend".to_string(),
        ))
    }

    /// Embed unified multimodal inputs (auto-dispatches to appropriate method)
    async fn embed_unified(&self, inputs: Vec<MultimodalInput>) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }

        // Group inputs by modality for efficient batching
        let mut texts = Vec::new();
        let mut text_indices = Vec::new();
        let mut images = Vec::new();
        let mut image_indices = Vec::new();
        let mut audios = Vec::new();
        let mut audio_indices = Vec::new();
        let mut videos = Vec::new();
        let mut video_indices = Vec::new();

        for (idx, input) in inputs.iter().enumerate() {
            match input {
                MultimodalInput::Text(t) => {
                    text_indices.push(idx);
                    texts.push(t.clone());
                }
                MultimodalInput::Image(i) => {
                    image_indices.push(idx);
                    images.push(i.clone());
                }
                MultimodalInput::Audio(a) => {
                    audio_indices.push(idx);
                    audios.push(a.clone());
                }
                MultimodalInput::Video(v) => {
                    video_indices.push(idx);
                    videos.push(v.clone());
                }
            }
        }

        // Process each modality
        let mut results = vec![vec![]; inputs.len()];

        if !texts.is_empty() {
            let text_embeddings = self.embed(texts).await?;
            for (i, emb) in text_indices.into_iter().zip(text_embeddings) {
                results[i] = emb;
            }
        }

        if !images.is_empty() {
            let image_embeddings = self.embed_multimode(images).await?;
            for (i, emb) in image_indices.into_iter().zip(image_embeddings) {
                results[i] = emb;
            }
        }

        if !audios.is_empty() {
            let audio_embeddings = self.embed_audio_multimode(audios).await?;
            for (i, emb) in audio_indices.into_iter().zip(audio_embeddings) {
                results[i] = emb;
            }
        }

        if !videos.is_empty() {
            let video_embeddings = self.embed_video_multimode(videos).await?;
            for (i, emb) in video_indices.into_iter().zip(video_embeddings) {
                results[i] = emb;
            }
        }

        Ok(results)
    }

    /// Get the embedding dimension
    fn dimension(&self) -> usize;

    /// Get the model name
    fn model_name(&self) -> &str;

    /// Get supported modalities for this embedder
    fn supported_modalities(&self) -> Vec<MediaModality> {
        vec![MediaModality::Text] // Default to text-only
    }
}

/// Create an embedder based on configuration (sync, for HTTP backend only)
pub fn create_embedder(config: &ResolvedEmbeddingConfig) -> Result<Box<dyn Embedder>> {
    let embedder = HttpEmbedder::new(config)?;
    Ok(Box::new(embedder))
}

/// Create an embedder automatically based on backend kind
///
/// For xinference backend: automatically ensures xinference is installed, server is running,
/// and model is launched before creating the embedder.
///
/// For http backend: creates HttpEmbedder directly.
pub async fn create_embedder_auto(config: &ResolvedEmbeddingConfig) -> Result<Box<dyn Embedder>> {
    match config.backend.kind {
        EmbeddingBackendKind::Xinference => {
            info!("Using Xinference backend, ensuring dependencies...");

            // Ensure xinference is installed and ready (sync check)
            ensure_xinference_ready()?;

            // Extract port from URL
            let port = extract_port_from_url(&config.backend.url)?;

            // Get or initialize the global manager
            let manager_lock = get_or_init_xinference_manager(port).await?;

            // Ensure server is running
            {
                let mut guard = manager_lock.lock().await;
                if let Some(ref mut manager) = *guard {
                    manager.ensure_running().await?;
                }
            }

            // Create embedder (this will launch model if needed)
            let embedder = XinferenceEmbedder::from_global_manager(&config.model_id).await?;
            Ok(Box::new(embedder))
        }
        EmbeddingBackendKind::Http => {
            let embedder = HttpEmbedder::new(config)?;
            Ok(Box::new(embedder))
        }
    }
}

/// Extract port number from a URL string
fn extract_port_from_url(url: &str) -> Result<u16> {
    url::Url::parse(url)
        .map_err(|e| Error::Config(format!("Invalid backend URL: {}", e)))?
        .port()
        .unwrap_or(9997) // Default xinference port
        .try_into()
        .map_err(|_| Error::Config("Port out of range".into()))
}

/// Create an embedder with optional Xinference manager support
pub async fn create_embedder_with_xinference(
    config: &ResolvedEmbeddingConfig,
    xinf_manager: Option<Arc<Mutex<XinferenceManager>>>,
) -> Result<Box<dyn Embedder>> {
    match config.backend.kind {
        EmbeddingBackendKind::Xinference => {
            let manager = xinf_manager.ok_or_else(|| {
                Error::Config("Xinference manager required for xinference backend".into())
            })?;
            let embedder = XinferenceEmbedder::new(manager, &config.model_id).await?;
            Ok(Box::new(embedder))
        }
        EmbeddingBackendKind::Http => {
            let embedder = HttpEmbedder::new(config)?;
            Ok(Box::new(embedder))
        }
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
pub async fn embed_multimode_in_batches(
    embedder: &dyn Embedder,
    inputs: Vec<ImageEmbedInput>,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut all_embeddings = Vec::with_capacity(inputs.len());

    for chunk in inputs.chunks(batch_size) {
        let batch_inputs: Vec<ImageEmbedInput> = chunk.to_vec();
        let embeddings = embedder.embed_multimode(batch_inputs).await?;
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
