//! Xinference embedding backend
//!
//! Implements the Embedder trait using Xinference's OpenAI-compatible API.

use crate::embed::{Embedder, ImageEmbedInput};
use crate::error::{Error, Result};
use crate::xinference::{get_xinference_model_spec, hf_to_xinference_name, XinferenceManager};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::debug;
use url::Url;

/// OpenAI-compatible embedding request
#[derive(Debug, Serialize)]
struct OpenAIEmbedRequest {
    model: String,
    input: Vec<String>,
}

/// OpenAI-compatible embedding response
#[derive(Debug, Deserialize)]
struct OpenAIEmbedResponse {
    data: Vec<EmbeddingData>,
}

/// Individual embedding data
#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

/// Xinference embedder implementation
pub struct XinferenceEmbedder {
    manager: Arc<Mutex<XinferenceManager>>,
    model_name: String,
    xinf_model_name: String,
    dimension: usize,
    client: Client,
    base_url: Url,
}

impl XinferenceEmbedder {
    /// Create a new XinferenceEmbedder
    ///
    /// This will:
    /// 1. Ensure the Xinference server is running
    /// 2. Launch the embedding model if not already running
    /// 3. Return an embedder ready for use
    pub async fn new(manager: Arc<Mutex<XinferenceManager>>, model: &str) -> Result<Self> {
        let xinf_name = hf_to_xinference_name(model);

        // Get model spec for dimension
        let spec = get_xinference_model_spec(model).ok_or_else(|| {
            Error::Config(format!(
                "Unknown Xinference embedding model: '{}'. Supported models include: bge-small-en-v1.5, bge-base-en-v1.5, bge-large-en-v1.5, all-MiniLM-L6-v2",
                model
            ))
        })?;

        let (base_url, _model_uid) = {
            let mut mgr = manager.lock().await;

            // Ensure server is running
            mgr.ensure_running().await?;

            // Ensure model is launched
            let uid = mgr.ensure_model_launched(model, "embedding").await?;

            (mgr.base_url().clone(), uid)
        };

        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| Error::Embedding(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            manager,
            model_name: model.to_string(),
            xinf_model_name: xinf_name,
            dimension: spec.dimension,
            client,
            base_url,
        })
    }

    /// Create a new XinferenceEmbedder using the global manager
    ///
    /// This is used by the automatic embedder creation flow.
    pub async fn from_global_manager(model: &str) -> Result<Self> {
        use crate::xinference::get_or_init_xinference_manager;

        let xinf_name = hf_to_xinference_name(model);

        // Get model spec for dimension
        let spec = get_xinference_model_spec(model).ok_or_else(|| {
            Error::Config(format!(
                "Unknown Xinference embedding model: '{}'. Supported models include: bge-small-en-v1.5, bge-base-en-v1.5, bge-large-en-v1.5, all-MiniLM-L6-v2",
                model
            ))
        })?;

        // Get the global manager lock
        let manager_lock = get_or_init_xinference_manager(9997).await?;
        let (base_url, _model_uid) = {
            let mut guard = manager_lock.lock().await;
            let mgr = guard
                .as_mut()
                .ok_or_else(|| Error::Config("Xinference manager not initialized".into()))?;

            // Ensure model is launched
            let uid = mgr.ensure_model_launched(model, "embedding").await?;
            (mgr.base_url().clone(), uid)
        };

        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| Error::Embedding(format!("Failed to create HTTP client: {}", e)))?;

        // Create a dummy Arc<Mutex<XinferenceManager>> since we use the global one
        // This is a temporary workaround; ideally we'd refactor to not need this field
        let dummy_manager = Arc::new(Mutex::new(crate::xinference::XinferenceManager::new(9997)?));

        Ok(Self {
            manager: dummy_manager,
            model_name: model.to_string(),
            xinf_model_name: xinf_name,
            dimension: spec.dimension,
            client,
            base_url,
        })
    }

    /// Internal embed implementation
    async fn embed_internal(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let url = self
            .base_url
            .join("/v1/embeddings")
            .map_err(|e| Error::Config(format!("Invalid URL: {}", e)))?;

        let request = OpenAIEmbedRequest {
            model: self.xinf_model_name.clone(),
            input: texts,
        };

        debug!(
            "Sending embedding request for {} texts",
            request.input.len()
        );

        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::Embedding(format!("Embedding request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Embedding(format!(
                "Embedding request failed: HTTP {} - {}",
                status, body
            )));
        }

        let embed_response: OpenAIEmbedResponse = response
            .json()
            .await
            .map_err(|e| Error::Embedding(format!("Failed to parse embedding response: {}", e)))?;

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
        {
            let mut mgr = self.manager.lock().await;
            mgr.ensure_running().await?;
            if !mgr.is_model_launched(&self.xinf_model_name) {
                mgr.ensure_model_launched(&self.model_name, "embedding")
                    .await?;
            }
        }

        self.embed_internal(texts).await
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_embed_request_serialization() {
        let request = OpenAIEmbedRequest {
            model: "bge-small-en-v1.5".to_string(),
            input: vec!["Hello world".to_string(), "Test".to_string()],
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("bge-small-en-v1.5"));
        assert!(json.contains("Hello world"));
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
