use super::{RerankResult, Reranker};
use crate::config::RerankerConfig;
use crate::error::{Error, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use url::Url;

#[derive(Debug, Clone, Serialize)]
struct RerankRequest {
    model: String,
    query: String,
    documents: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RerankResponse {
    results: Vec<RerankItem>,
}

#[derive(Debug, Clone, Deserialize)]
struct RerankItem {
    index: usize,
    score: f32,
}

pub struct HttpReranker {
    client: Client,
    base_url: Url,
    model_id: String,
}

impl HttpReranker {
    pub fn new(config: &RerankerConfig, backend_url: &str) -> Result<Self> {
        let base_url = Url::parse(backend_url)
            .map_err(|e| Error::Config(format!("Invalid reranker backend URL: {}", e)))?;
        Ok(Self {
            client: Client::new(),
            base_url,
            model_id: config.model.clone(),
        })
    }

    fn endpoint(&self, path: &str) -> Result<Url> {
        self.base_url
            .join(path)
            .map_err(|e| Error::Config(format!("Invalid reranker backend URL: {}", e)))
    }
}

#[async_trait]
impl Reranker for HttpReranker {
    async fn rerank(&self, query: &str, documents: Vec<String>) -> Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let url = self.endpoint("/v1/rerank")?;
        let request = RerankRequest {
            model: self.model_id.clone(),
            query: query.to_string(),
            documents,
        };

        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await?
            .error_for_status()?;

        let parsed = response.json::<RerankResponse>().await?;
        Ok(parsed
            .results
            .into_iter()
            .map(|item| RerankResult {
                index: item.index,
                score: item.score,
            })
            .collect())
    }

    fn model_name(&self) -> &str {
        &self.model_id
    }
}
