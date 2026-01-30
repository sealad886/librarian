use super::{AudioEmbedInput, Embedder, ImageEmbedInput, MediaModality, VideoEmbedInput};
use crate::config::{EmbeddingDimensionSource, ResolvedEmbeddingConfig};
use crate::embedding_backend::{
    AudioTextInput, EmbeddingBackendClient, ImageTextInput, VideoTextInput,
};
use crate::error::{Error, Result};
use crate::models::{embedding_model_capabilities, EmbeddingModelCapabilities};
use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use std::fs;

pub struct HttpEmbedder {
    client: EmbeddingBackendClient,
    model_id: String,
    family: String,
    dimension: usize,
    dimension_source: EmbeddingDimensionSource,
    capabilities: Option<EmbeddingModelCapabilities>,
}

impl HttpEmbedder {
    pub fn new(config: &ResolvedEmbeddingConfig) -> Result<Self> {
        let client = EmbeddingBackendClient::new(&config.backend.url)?;
        let capabilities = embedding_model_capabilities(&config.model_id);
        Ok(Self {
            client,
            model_id: config.model_id.clone(),
            family: config.family.clone(),
            dimension: config.dimension,
            dimension_source: config.dimension_source,
            capabilities,
        })
    }

    fn validate_dimensions(&self, embeddings: &[Vec<f32>]) -> Result<()> {
        if let Some(mismatch) = embeddings.iter().find(|vec| vec.len() != self.dimension) {
            return Err(Error::Embedding(format!(
                "Embedding dimension mismatch for model '{}' (family '{}', source {}): expected {}, got {}",
                self.model_id,
                self.family,
                self.dimension_source,
                self.dimension,
                mismatch.len()
            )));
        }
        Ok(())
    }

    fn encode_file_base64(path: &str) -> Result<String> {
        let bytes = fs::read(path)
            .map_err(|e| Error::Embedding(format!("Failed to read file '{}': {}", path, e)))?;
        Ok(STANDARD.encode(bytes))
    }

    fn supports_image(&self) -> bool {
        self.capabilities.map(|c| c.supports_image).unwrap_or(false)
    }

    fn supports_audio(&self) -> bool {
        self.capabilities.map(|c| c.supports_audio).unwrap_or(false)
    }

    fn supports_video(&self) -> bool {
        self.capabilities.map(|c| c.supports_video).unwrap_or(false)
    }

    fn detect_mime_type(path: &str) -> Option<String> {
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());

        match ext.as_deref() {
            // Image types
            Some("jpg") | Some("jpeg") => Some("image/jpeg".to_string()),
            Some("png") => Some("image/png".to_string()),
            Some("gif") => Some("image/gif".to_string()),
            Some("webp") => Some("image/webp".to_string()),
            Some("bmp") => Some("image/bmp".to_string()),
            // Audio types
            Some("mp3") => Some("audio/mpeg".to_string()),
            Some("wav") => Some("audio/wav".to_string()),
            Some("flac") => Some("audio/flac".to_string()),
            Some("ogg") => Some("audio/ogg".to_string()),
            Some("m4a") => Some("audio/mp4".to_string()),
            Some("aac") => Some("audio/aac".to_string()),
            Some("wma") => Some("audio/x-ms-wma".to_string()),
            // Video types
            Some("mp4") => Some("video/mp4".to_string()),
            Some("webm") => Some("video/webm".to_string()),
            Some("avi") => Some("video/x-msvideo".to_string()),
            Some("mov") => Some("video/quicktime".to_string()),
            Some("mkv") => Some("video/x-matroska".to_string()),
            Some("wmv") => Some("video/x-ms-wmv".to_string()),
            _ => None,
        }
    }
}

#[async_trait]
impl Embedder for HttpEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let embeddings = self.client.embed_text(&self.model_id, texts).await?;
        self.validate_dimensions(&embeddings)?;
        Ok(embeddings)
    }

    async fn embed_images(&self, images: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        if !self.supports_image() {
            return Err(Error::Embedding(format!(
                "Model '{}' does not support image embeddings",
                self.model_id
            )));
        }

        let inputs = images
            .into_iter()
            .map(|path| {
                let mime = Self::detect_mime_type(&path);
                let base64 = Self::encode_file_base64(&path)?;
                Ok(ImageTextInput {
                    image_base64: base64,
                    image_mime: mime,
                    text: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let embeddings = self.client.embed_image(&self.model_id, inputs).await?;
        self.validate_dimensions(&embeddings)?;
        Ok(embeddings)
    }

    async fn embed_multimode(&self, inputs: Vec<ImageEmbedInput>) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        if !self.supports_image() {
            return Err(Error::Embedding(format!(
                "Model '{}' does not support image inputs",
                self.model_id
            )));
        }

        let request_inputs = inputs
            .into_iter()
            .map(|input| {
                let mime = Self::detect_mime_type(&input.image_path);
                let base64 = Self::encode_file_base64(&input.image_path)?;
                Ok(ImageTextInput {
                    image_base64: base64,
                    image_mime: mime,
                    text: input.text,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let embeddings = self
            .client
            .embed_image(&self.model_id, request_inputs)
            .await?;
        self.validate_dimensions(&embeddings)?;
        Ok(embeddings)
    }

    async fn embed_audio(&self, audios: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if audios.is_empty() {
            return Ok(Vec::new());
        }

        if !self.supports_audio() {
            return Err(Error::Embedding(format!(
                "Model '{}' does not support audio embeddings",
                self.model_id
            )));
        }

        let inputs = audios
            .into_iter()
            .map(|path| {
                let mime = Self::detect_mime_type(&path);
                let base64 = Self::encode_file_base64(&path)?;
                Ok(AudioTextInput {
                    audio_base64: base64,
                    audio_mime: mime,
                    text: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let embeddings = self.client.embed_audio(&self.model_id, inputs).await?;
        self.validate_dimensions(&embeddings)?;
        Ok(embeddings)
    }

    async fn embed_audio_multimode(&self, inputs: Vec<AudioEmbedInput>) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        if !self.supports_audio() {
            return Err(Error::Embedding(format!(
                "Model '{}' does not support audio inputs",
                self.model_id
            )));
        }

        let request_inputs = inputs
            .into_iter()
            .map(|input| {
                let mime = Self::detect_mime_type(&input.audio_path);
                let base64 = Self::encode_file_base64(&input.audio_path)?;
                Ok(AudioTextInput {
                    audio_base64: base64,
                    audio_mime: mime,
                    text: input.text,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let embeddings = self
            .client
            .embed_audio(&self.model_id, request_inputs)
            .await?;
        self.validate_dimensions(&embeddings)?;
        Ok(embeddings)
    }

    async fn embed_video(&self, videos: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if videos.is_empty() {
            return Ok(Vec::new());
        }

        if !self.supports_video() {
            return Err(Error::Embedding(format!(
                "Model '{}' does not support video embeddings",
                self.model_id
            )));
        }

        let inputs = videos
            .into_iter()
            .map(|path| {
                let mime = Self::detect_mime_type(&path);
                let base64 = Self::encode_file_base64(&path)?;
                Ok(VideoTextInput {
                    video_base64: base64,
                    video_mime: mime,
                    text: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let embeddings = self.client.embed_video(&self.model_id, inputs).await?;
        self.validate_dimensions(&embeddings)?;
        Ok(embeddings)
    }

    async fn embed_video_multimode(&self, inputs: Vec<VideoEmbedInput>) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        if !self.supports_video() {
            return Err(Error::Embedding(format!(
                "Model '{}' does not support video inputs",
                self.model_id
            )));
        }

        let request_inputs = inputs
            .into_iter()
            .map(|input| {
                let mime = Self::detect_mime_type(&input.video_path);
                let base64 = Self::encode_file_base64(&input.video_path)?;
                Ok(VideoTextInput {
                    video_base64: base64,
                    video_mime: mime,
                    text: input.text,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let embeddings = self
            .client
            .embed_video(&self.model_id, request_inputs)
            .await?;
        self.validate_dimensions(&embeddings)?;
        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_id
    }

    fn supported_modalities(&self) -> Vec<MediaModality> {
        let mut modalities = vec![MediaModality::Text];
        if self.supports_image() {
            modalities.push(MediaModality::Image);
        }
        if self.supports_audio() {
            modalities.push(MediaModality::Audio);
        }
        if self.supports_video() {
            modalities.push(MediaModality::Video);
        }
        modalities
    }
}
