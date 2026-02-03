//! Xinference embedding backend
//!
//! Implements the Embedder trait using Xinference's OpenAI-compatible API.

use crate::config::ResolvedEmbeddingConfig;
use crate::embed::{Embedder, ImageEmbedInput, MediaModality};
use crate::error::{Error, Result};
use crate::xinference::{hf_to_xinference_name, xinference_request_json, SharedXinferenceManager};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info};
use url::Url;

/// OpenAI-compatible embedding request
#[derive(Debug, Serialize)]
struct OpenAIEmbedRequest {
    model: String,
    input: Vec<String>,
}

/// OpenAI-compatible embedding response
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIEmbedResponse {
    data: Vec<EmbeddingData>,
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
}

/// Individual embedding data
#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

/// Xinference embedder implementation
#[allow(dead_code)]
pub struct XinferenceEmbedder {
    manager: SharedXinferenceManager,
    model_name: String,
    dimension: usize,
    allow_custom: bool,
    client: Client,
}

impl XinferenceEmbedder {
    /// Create a new XinferenceEmbedder
    ///
    /// This will:
    /// 1. Ensure the Xinference server is running
    /// 2. Launch the embedding model if not already running
    /// 3. Return an embedder ready for use
    pub async fn new(
        manager: SharedXinferenceManager,
        config: &ResolvedEmbeddingConfig,
    ) -> Result<Self> {
        let model = config.model_id.as_str();
        let xinf_name = hf_to_xinference_name(model);
        let expected_dimension = config.dimension;
        let allow_custom = config.allow_custom;
        let mut dimension = expected_dimension;

        {
            let mut guard = manager.lock().await;
            let mgr = guard
                .as_mut()
                .ok_or_else(|| Error::Config("Xinference manager not initialized".into()))?;

            // Ensure server is running
            mgr.ensure_running().await?;

            if allow_custom {
                info!(
                    model = %model,
                    "Using custom Xinference embedding model; skipping registry lookup"
                );
            } else {
                let registration = mgr
                    .fetch_model_registration("embedding", &xinf_name)
                    .await?;
                if let Some(dim) = registration.dimension {
                    if dim != expected_dimension {
                        return Err(Error::Config(format!(
                            "Xinference model '{}' reports dimension {}, but config expects {}. Update embedding.dimension or reset the collection.",
                            xinf_name, dim, expected_dimension
                        )));
                    }
                    dimension = dim;
                }
            }

            let model_uid = mgr
                .ensure_model_launched(model, "embedding", !allow_custom)
                .await?;
            debug!(
                model_name = %xinf_name,
                model_uid = %model_uid,
                "Xinference embedding model ready"
            );
        }

        Ok(Self {
            manager,
            model_name: model.to_string(),
            dimension,
            allow_custom,
            client: build_xinference_client()?,
        })
    }

    /// Create a new XinferenceEmbedder using the global manager
    ///
    /// This is used by the automatic embedder creation flow.
    pub async fn from_global_manager(
        manager: SharedXinferenceManager,
        config: &ResolvedEmbeddingConfig,
    ) -> Result<Self> {
        Self::new(manager, config).await
    }

    /// Internal embed implementation
    async fn embed_internal(
        &self,
        base_url: Url,
        auth_token: Option<String>,
        model_uid: String,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let url = base_url
            .join("/v1/embeddings")
            .map_err(|e| Error::Config(format!("Invalid URL: {}", e)))?;

        let request = OpenAIEmbedRequest {
            model: model_uid,
            input: texts,
        };

        debug!(
            "Sending embedding request for {} texts",
            request.input.len()
        );

        let body = serde_json::to_value(&request).map_err(|e| {
            Error::Embedding(format!("Failed to serialize embedding request: {}", e))
        })?;
        let embed_response: OpenAIEmbedResponse = xinference_request_json(
            &self.client,
            reqwest::Method::POST,
            url,
            auth_token.as_deref(),
            Some(body),
            "embeddings",
        )
        .await?;

        // Validate dimensions
        for data in &embed_response.data {
            if data.embedding.len() != self.dimension {
                return Err(Error::Embedding(format!(
                    "Dimension mismatch: expected {}, got {}",
                    self.dimension,
                    data.embedding.len()
                )));
            }
        }

        // Sort by index and extract embeddings
        let mut data = embed_response.data;
        data.sort_by_key(|d| d.index);
        Ok(data.into_iter().map(|d| d.embedding).collect())
    }
}

#[async_trait]
impl Embedder for XinferenceEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        // Ensure server and model are still running before embedding
        let (base_url, auth_token, model_uid) = {
            let mut mgr = self.manager.lock().await;
            let manager = mgr
                .as_mut()
                .ok_or_else(|| Error::Config("Xinference manager not initialized".into()))?;
            manager.ensure_running().await?;
            let uid = manager
                .ensure_model_launched(&self.model_name, "embedding", !self.allow_custom)
                .await?;
            (
                manager.base_url().clone(),
                manager.auth_token().map(|t| t.to_string()),
                uid,
            )
        };

        self.embed_internal(base_url, auth_token, model_uid, texts)
            .await
    }

    async fn embed_images(&self, _images: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Err(Error::Embedding(
            "Image embedding via Xinference is not yet supported. Use a multimodal-capable backend for image embeddings.".to_string(),
        ))
    }

    async fn embed_multimode(&self, _inputs: Vec<ImageEmbedInput>) -> Result<Vec<Vec<f32>>> {
        Err(Error::Embedding(
            "Multimodal embedding via Xinference is not yet supported. Use a multimodal-capable backend.".to_string(),
        ))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn supported_modalities(&self) -> Vec<MediaModality> {
        // Xinference only supports text embeddings via OpenAI-compatible API
        vec![MediaModality::Text]
    }
}

fn build_xinference_client() -> Result<Client> {
    Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(|e| Error::Embedding(format!("Failed to create HTTP client: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_embed_request_serialization() {
        let request = OpenAIEmbedRequest {
            model: "model_uid".to_string(),
            input: vec!["Hello world".to_string()],
        };

        let json = serde_json::to_value(&request).unwrap();
        let expected = serde_json::json!({
            "model": "model_uid",
            "input": ["Hello world"]
        });
        assert_eq!(json, expected);
    }

    #[test]
    fn test_openai_embed_response_deserialization() {
        let json = r#"{
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1}
            ],
            "model": "bge-small-en-v1.5"
        }"#;

        let response: OpenAIEmbedResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].embedding.len(), 3);
        assert_eq!(response.data[0].index, 0);
    }
}
