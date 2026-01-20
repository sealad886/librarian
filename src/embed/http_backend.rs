use super::{Embedder, ImageEmbedInput};
use crate::config::{EmbeddingDimensionSource, ResolvedEmbeddingConfig};
use crate::embedding_backend::{EmbeddingBackendClient, ImageTextInput};
use crate::error::{Error, Result};
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use std::fs;

pub struct HttpEmbedder {
    client: EmbeddingBackendClient,
    model_id: String,
    family: String,
    dimension: usize,
    dimension_source: EmbeddingDimensionSource,
    supports_image_inputs: bool,
}

impl HttpEmbedder {
    pub fn new(config: &ResolvedEmbeddingConfig) -> Result<Self> {
        let client = EmbeddingBackendClient::new(&config.backend.url)?;
        Ok(Self {
            client,
            model_id: config.model_id.clone(),
            family: config.family.clone(),
            dimension: config.dimension,
            dimension_source: config.dimension_source,
            supports_image_inputs: config.supports_image_inputs(),
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

    fn encode_image_base64(path: &str) -> Result<String> {
        let bytes = fs::read(path).map_err(|e| {
            Error::Embedding(format!("Failed to read image '{}': {}", path, e))
        })?;
        Ok(STANDARD.encode(bytes))
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

        if !self.supports_image_inputs {
            return Err(Error::Embedding(format!(
                "Model '{}' does not support image embeddings",
                self.model_id
            )));
        }

        let inputs = images
            .into_iter()
            .map(|path| {
                let base64 = Self::encode_image_base64(&path)?;
                Ok(ImageTextInput {
                    image_base64: base64,
                    image_mime: None,
                    text: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let embeddings = self.client.embed_image_text(&self.model_id, inputs).await?;
        self.validate_dimensions(&embeddings)?;
        Ok(embeddings)
    }

    async fn embed_image_text(&self, inputs: Vec<ImageEmbedInput>) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        if !self.supports_image_inputs {
            return Err(Error::Embedding(format!(
                "Model '{}' does not support image inputs",
                self.model_id
            )));
        }

        let request_inputs = inputs
            .into_iter()
            .map(|input| {
                let base64 = Self::encode_image_base64(&input.image_path)?;
                Ok(ImageTextInput {
                    image_base64: base64,
                    image_mime: None,
                    text: input.text,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let embeddings = self
            .client
            .embed_image_text(&self.model_id, request_inputs)
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
}
