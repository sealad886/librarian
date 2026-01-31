//! Xinference reranker backend
//!
//! Implements the Reranker trait using Xinference's rerank API.

use crate::error::{Error, Result};
use crate::rerank::{RerankResult, Reranker};
use crate::xinference::{hf_to_xinference_name, xinference_reranker_models, XinferenceManager};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::debug;
use url::Url;

/// Xinference rerank request
#[derive(Debug, Serialize)]
struct RerankRequest {
    model: String,
    query: String,
    documents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_n: Option<usize>,
}

/// Xinference rerank response
#[derive(Debug, Deserialize)]
struct RerankResponse {
    results: Vec<RerankResultData>,
}

/// Individual rerank result data
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct RerankResultData {
    index: usize,
    relevance_score: f32,
    #[serde(default)]
    #[allow(dead_code)]
    document: Option<String>,
}

/// Xinference reranker implementation
pub struct XinferenceReranker {
    manager: Arc<Mutex<XinferenceManager>>,
    model_name: String,
    xinf_model_name: String,
    client: Client,
    base_url: Url,
}

impl XinferenceReranker {
    /// Create a new XinferenceReranker
    ///
    /// This will:
    /// 1. Ensure the Xinference server is running
    /// 2. Launch the reranker model if not already running
    /// 3. Return a reranker ready for use
    pub async fn new(manager: Arc<Mutex<XinferenceManager>>, model: &str) -> Result<Self> {
        let xinf_name = hf_to_xinference_name(model);

        // Verify model is a known reranker model
        if !xinference_reranker_models().contains(xinf_name.as_str()) {
            let allowed = crate::models::allowlisted_reranker_models().join(", ");
            return Err(Error::Config(format!(
                "Unknown Xinference reranker model: '{}'. Supported models: {}",
                model, allowed
            )));
        }

        let base_url = {
            let mut mgr = manager.lock().await;

            // Ensure server is running
            mgr.ensure_running().await?;

            // Ensure reranker model is launched
            mgr.ensure_model_launched(model, "rerank").await?;

            mgr.base_url().clone()
        };

        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| Error::Embedding(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            manager,
            model_name: model.to_string(),
            xinf_model_name: xinf_name,
            client,
            base_url,
        })
    }

    /// Create a new XinferenceReranker using the global manager
    ///
    /// This is used by the automatic reranker creation flow.
    pub async fn from_global_manager(model: &str) -> Result<Self> {
        use crate::xinference::get_or_init_xinference_manager;

        let xinf_name = hf_to_xinference_name(model);

        // Verify model is a known reranker model
        if !xinference_reranker_models().contains(xinf_name.as_str()) {
            let allowed = crate::models::allowlisted_reranker_models().join(", ");
            return Err(Error::Config(format!(
                "Unknown Xinference reranker model: '{}'. Supported models: {}",
                model, allowed
            )));
        }

        // Get the global manager lock
        let manager_lock = get_or_init_xinference_manager(9997).await?;
        let base_url = {
            let mut guard = manager_lock.lock().await;
            let mgr = guard
                .as_mut()
                .ok_or_else(|| Error::Config("Xinference manager not initialized".into()))?;

            // Ensure model is launched
            mgr.ensure_model_launched(model, "rerank").await?;
            mgr.base_url().clone()
        };

        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| Error::Embedding(format!("Failed to create HTTP client: {}", e)))?;

        // Create a dummy Arc<Mutex<XinferenceManager>> since we use the global one
        let dummy_manager = Arc::new(Mutex::new(crate::xinference::XinferenceManager::new(9997)?));

        Ok(Self {
            manager: dummy_manager,
            model_name: model.to_string(),
            xinf_model_name: xinf_name,
            client,
            base_url,
        })
    }

    /// Internal rerank implementation
    async fn rerank_internal(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let url = self
            .base_url
            .join("/v1/rerank")
            .map_err(|e| Error::Config(format!("Invalid URL: {}", e)))?;

        let request = RerankRequest {
            model: self.xinf_model_name.clone(),
            query: query.to_string(),
            documents,
            top_n: None, // Return all results
        };

        debug!(
            "Sending rerank request for {} documents",
            request.documents.len()
        );

        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::Embedding(format!("Rerank request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Embedding(format!(
                "Rerank request failed: HTTP {} - {}",
                status, body
            )));
        }

        let rerank_response: RerankResponse = response
            .json()
            .await
            .map_err(|e| Error::Embedding(format!("Failed to parse rerank response: {}", e)))?;

        Ok(rerank_response
            .results
            .into_iter()
            .map(|r| RerankResult {
                index: r.index,
                score: r.relevance_score,
            })
            .collect())
    }
}

#[async_trait]
impl Reranker for XinferenceReranker {
    async fn rerank(&self, query: &str, documents: Vec<String>) -> Result<Vec<RerankResult>> {
        // Ensure server and model are still running
        {
            let mut mgr = self.manager.lock().await;
            mgr.ensure_running().await?;
            if !mgr.is_model_launched(&self.xinf_model_name) {
                mgr.ensure_model_launched(&self.model_name, "rerank")
                    .await?;
            }
        }

        self.rerank_internal(query, documents).await
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rerank_request_serialization() {
        let request = RerankRequest {
            model: "bge-reranker-base".to_string(),
            query: "What is Rust?".to_string(),
            documents: vec![
                "Rust is a programming language".to_string(),
                "Python is a scripting language".to_string(),
            ],
            top_n: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("bge-reranker-base"));
        assert!(json.contains("What is Rust?"));
        assert!(!json.contains("top_n")); // Should be skipped when None
    }

    #[test]
    fn test_rerank_response_deserialization() {
        let json = r#"{
            "results": [
                {"index": 0, "relevance_score": 0.95, "document": "Rust is a programming language"},
                {"index": 1, "relevance_score": 0.45, "document": "Python is a scripting language"}
            ]
        }"#;

        let response: RerankResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.results.len(), 2);
        assert_eq!(response.results[0].index, 0);
        assert!((response.results[0].relevance_score - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_rerank_response_without_document() {
        let json = r#"{
            "results": [
                {"index": 0, "relevance_score": 0.95}
            ]
        }"#;

        let response: RerankResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.results.len(), 1);
        // Document field removed per Issue #11 - field was never read
    }
}
