//! Xinference reranker backend
//!
//! Implements the Reranker trait using Xinference's rerank API.

use crate::error::{Error, Result};
use crate::rerank::{RerankResult, Reranker};
use crate::xinference::{
    build_xinference_http_client, hf_to_xinference_name, xinference_request_json,
    SharedXinferenceManager,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::debug;

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
    manager: SharedXinferenceManager,
    model_name: String,
    client: Client,
}

impl XinferenceReranker {
    /// Create a new XinferenceReranker
    ///
    /// This will:
    /// 1. Ensure the Xinference server is running
    /// 2. Launch the reranker model if not already running
    /// 3. Return a reranker ready for use
    pub async fn new(manager: SharedXinferenceManager, model: &str) -> Result<Self> {
        let xinf_name = hf_to_xinference_name(model);

        {
            let mut guard = manager.lock().await;
            let mgr = guard
                .as_mut()
                .ok_or_else(|| Error::Config("Xinference manager not initialized".into()))?;

            // Ensure server is running
            mgr.ensure_running().await?;

            // Ensure model is registered
            let _ = mgr.fetch_model_registration("rerank", &xinf_name).await?;

            // Ensure reranker model is launched
            let model_uid = mgr.ensure_model_launched(model, "rerank", true).await?;
            debug!(
                model_name = %xinf_name,
                model_uid = %model_uid,
                "Xinference reranker model ready"
            );
        }

        Ok(Self {
            manager,
            model_name: model.to_string(),
            client: build_xinference_http_client()?,
        })
    }

    /// Create a new XinferenceReranker using the global manager
    ///
    /// This is used by the automatic reranker creation flow.
    pub async fn from_global_manager(
        manager: SharedXinferenceManager,
        model: &str,
    ) -> Result<Self> {
        Self::new(manager, model).await
    }

    /// Internal rerank implementation
    async fn rerank_internal(
        &self,
        base_url: url::Url,
        auth_token: Option<String>,
        model_uid: String,
        query: &str,
        documents: Vec<String>,
    ) -> Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let url = base_url
            .join("/v1/rerank")
            .map_err(|e| Error::Config(format!("Invalid URL: {}", e)))?;

        let request = RerankRequest {
            model: model_uid,
            query: query.to_string(),
            documents,
            top_n: None, // Return all results
        };

        debug!(
            "Sending rerank request for {} documents",
            request.documents.len()
        );

        let body = serde_json::to_value(&request)
            .map_err(|e| Error::Embedding(format!("Failed to serialize rerank request: {}", e)))?;
        let rerank_response: RerankResponse = xinference_request_json(
            &self.client,
            reqwest::Method::POST,
            url,
            auth_token.as_deref(),
            Some(body),
            "rerank",
        )
        .await?;

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
        let (base_url, auth_token, model_uid) = {
            let mut mgr = self.manager.lock().await;
            let manager = mgr
                .as_mut()
                .ok_or_else(|| Error::Config("Xinference manager not initialized".into()))?;
            manager.ensure_running().await?;
            let uid = manager
                .ensure_model_launched(&self.model_name, "rerank", true)
                .await?;
            (
                manager.base_url().clone(),
                manager.auth_token().map(|t| t.to_string()),
                uid,
            )
        };

        self.rerank_internal(base_url, auth_token, model_uid, query, documents)
            .await
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
            model: "model_uid".to_string(),
            query: "What is Rust?".to_string(),
            documents: vec![
                "Rust is a programming language".to_string(),
                "Python is a scripting language".to_string(),
            ],
            top_n: None,
        };

        let json = serde_json::to_value(&request).unwrap();
        let expected = serde_json::json!({
            "model": "model_uid",
            "query": "What is Rust?",
            "documents": [
                "Rust is a programming language",
                "Python is a scripting language"
            ]
        });
        assert_eq!(json, expected);
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
