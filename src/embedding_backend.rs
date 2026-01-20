use crate::error::{Error, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::time::Duration;
use url::Url;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingBackendKind {
    Http,
}

impl FromStr for EmbeddingBackendKind {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_lowercase().as_str() {
            "http" | "https" | "python-sidecar" => Ok(Self::Http),
            "openai-compatible" => Err(Error::Config(
                "Embedding backend 'openai-compatible' is not implemented".to_string(),
            )),
            _ => Err(Error::Config(format!(
                "Unsupported embedding backend '{}'; only 'http' is supported",
                value
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingBackendConfig {
    pub kind: EmbeddingBackendKind,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    #[serde(default)]
    pub backend_version: Option<String>,
    #[serde(default)]
    pub models: Vec<BackendModelCapabilities>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendModelCapabilities {
    #[serde(alias = "model")]
    pub id: String,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub modalities: Vec<String>,
    #[serde(default, alias = "embedding_dim", alias = "dimension")]
    pub embedding_dim: Option<usize>,
    #[serde(default)]
    pub multivector: Option<bool>,
    #[serde(default)]
    pub supports_mrl: Option<bool>,
    #[serde(default)]
    pub max_batch: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendProbeResponse {
    #[serde(alias = "model", alias = "model_id")]
    pub id: String,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub modalities: Vec<String>,
    #[serde(default, alias = "embedding_dim", alias = "dimension")]
    pub embedding_dim: Option<usize>,
    #[serde(default)]
    pub multivector: Option<bool>,
    #[serde(default)]
    pub supports_mrl: Option<bool>,
    #[serde(default)]
    pub max_batch: Option<usize>,
    #[serde(default)]
    pub text_embeddings: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    pub image_embeddings: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    pub joint_embeddings: Option<Vec<Vec<f32>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageTextInput {
    pub image_base64: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_mime: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct EmbedTextRequest {
    model: String,
    inputs: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct EmbedImageTextRequest {
    model: String,
    inputs: Vec<ImageTextInput>,
}

#[derive(Debug, Clone, Serialize)]
struct ProbeRequest {
    model: String,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_mime: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum EmbeddingResponse {
    Embeddings { embeddings: Vec<Vec<f32>> },
    Vectors { vectors: Vec<Vec<f32>> },
    Data { data: Vec<EmbeddingData> },
}

#[derive(Debug, Clone, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl EmbeddingResponse {
    fn into_embeddings(self) -> Vec<Vec<f32>> {
        match self {
            EmbeddingResponse::Embeddings { embeddings } => embeddings,
            EmbeddingResponse::Vectors { vectors } => vectors,
            EmbeddingResponse::Data { data } => data.into_iter().map(|d| d.embedding).collect(),
        }
    }
}

pub struct EmbeddingBackendClient {
    client: Client,
    base_url: Url,
    retries: usize,
}

impl EmbeddingBackendClient {
    pub fn new(base_url: &str) -> Result<Self> {
        let base_url = Url::parse(base_url)?;
        let timeout = Duration::from_secs(30);
        let client = Client::builder().timeout(timeout).build()?;
        Ok(Self {
            client,
            base_url,
            retries: 2,
        })
    }

    fn endpoint(&self, path: &str) -> Result<Url> {
        self.base_url
            .join(path)
            .map_err(|e| Error::Config(format!("Invalid embedding backend URL: {}", e)))
    }

    async fn send_with_retry<T: for<'de> Deserialize<'de>>(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<T> {
        let mut last_err: Option<Error> = None;
        for attempt in 0..=self.retries {
            let req = request.try_clone().ok_or_else(|| {
                Error::Embedding("Failed to clone backend request".to_string())
            })?;
            match req.send().await {
                Ok(response) => match response.error_for_status() {
                    Ok(ok) => return Ok(ok.json::<T>().await?),
                    Err(e) => last_err = Some(Error::Embedding(e.to_string())),
                },
                Err(e) => last_err = Some(Error::Embedding(e.to_string())),
            }

            if attempt < self.retries {
                tokio::time::sleep(Duration::from_millis(200 * (attempt + 1) as u64)).await;
            }
        }

        Err(last_err.unwrap_or_else(|| {
            Error::Embedding("Embedding backend request failed".to_string())
        }))
    }

    pub async fn capabilities(&self) -> Result<BackendCapabilities> {
        let url = self.endpoint("/capabilities")?;
        let request = self.client.get(url);
        self.send_with_retry(request).await
    }

    pub async fn probe(
        &self,
        model: &str,
        text: &str,
        image_base64: Option<String>,
        image_mime: Option<String>,
    ) -> Result<BackendProbeResponse> {
        let url = self.endpoint("/probe")?;
        let request = ProbeRequest {
            model: model.to_string(),
            text: text.to_string(),
            image_base64,
            image_mime,
        };
        let request = self.client.post(url).json(&request);
        self.send_with_retry(request).await
    }

    pub async fn embed_text(&self, model: &str, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let url = self.endpoint("/v1/embed/text")?;
        let request = EmbedTextRequest {
            model: model.to_string(),
            inputs,
        };
        let parsed: EmbeddingResponse = self
            .send_with_retry(self.client.post(url).json(&request))
            .await?;
        Ok(parsed.into_embeddings())
    }

    pub async fn embed_image_text(
        &self,
        model: &str,
        inputs: Vec<ImageTextInput>,
    ) -> Result<Vec<Vec<f32>>> {
        let url = self.endpoint("/v1/embed/image_text")?;
        let request = EmbedImageTextRequest {
            model: model.to_string(),
            inputs,
        };
        let parsed: EmbeddingResponse = self
            .send_with_retry(self.client.post(url).json(&request))
            .await?;
        Ok(parsed.into_embeddings())
    }
}
