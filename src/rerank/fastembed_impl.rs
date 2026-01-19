//! FastEmbed reranker implementation

use super::{RerankResult, Reranker};
use crate::config::RerankerConfig;
use crate::error::{Error, Result};
use async_trait::async_trait;
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

pub struct FastEmbedReranker {
    model: Arc<Mutex<TextRerank>>,
    model_name: String,
}

impl FastEmbedReranker {
    pub fn new(config: &RerankerConfig) -> Result<Self> {
        info!("Initializing FastEmbed reranker with model: {}", config.model);

        let model_enum = match config.model.as_str() {
            "BAAI/bge-reranker-base" => RerankerModel::BGERerankerBase,
            _ => {
                warn!(
                    "Unknown reranker model '{}', defaulting to BGE reranker base",
                    config.model
                );
                RerankerModel::BGERerankerBase
            }
        };

        let options = RerankInitOptions::new(model_enum).with_show_download_progress(true);

        let model = TextRerank::try_new(options)
            .map_err(|e| Error::Embedding(format!("Failed to initialize reranker: {}", e)))?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            model_name: config.model.clone(),
        })
    }
}

#[async_trait]
impl Reranker for FastEmbedReranker {
    async fn rerank(&self, query: &str, documents: Vec<String>) -> Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Reranking {} documents", documents.len());

        let model = self.model.clone();
        let query = query.to_string();
        let results = tokio::task::spawn_blocking(move || {
            let model = model.blocking_lock();
            let doc_refs: Vec<&String> = documents.iter().collect();
            model.rerank(&query, doc_refs, false, None)
        })
        .await
        .map_err(|e| Error::Embedding(format!("Task join error: {}", e)))?
        .map_err(|e| Error::Embedding(format!("Rerank failed: {}", e)))?;

        let mapped = results
            .into_iter()
            .map(|r| RerankResult {
                index: r.index,
                score: r.score,
            })
            .collect();

        Ok(mapped)
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}
