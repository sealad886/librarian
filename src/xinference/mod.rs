//! Xinference backend integration
//!
//! This module provides automatic Xinference server lifecycle management
//! for embedding and reranking operations. Key features:
//! - Zero-config experience: server starts automatically on first use
//! - Model auto-launch: models are downloaded and started on demand
//! - Graceful shutdown: clean process termination on exit

mod deps;
mod embedder;
mod registry;
mod registry_sync;
mod reranker;

pub use deps::*;
pub use embedder::*;
pub use registry::*;
pub use registry_sync::*;
pub use reranker::*;

use crate::error::{Error, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};
use url::Url;

/// Default Xinference server port
pub const DEFAULT_XINFERENCE_PORT: u16 = 9997;

/// Xinference model specification for embedding models
#[derive(Debug, Clone)]
pub struct XinferenceModelSpec {
    pub dimension: usize,
    pub model_type: String,
}

static XINFERENCE_EMBEDDING_MODELS: OnceLock<HashMap<String, XinferenceModelSpec>> =
    OnceLock::new();
static XINFERENCE_RERANK_MODELS: OnceLock<HashSet<String>> = OnceLock::new();

/// Registry of known Xinference embedding models and their dimensions from snapshots.
pub fn xinference_embedding_models() -> &'static HashMap<String, XinferenceModelSpec> {
    XINFERENCE_EMBEDDING_MODELS.get_or_init(|| {
        let mut models = HashMap::new();
        if let Ok(snapshot) = registry_snapshot(RegistryType::Embedding) {
            for entry in &snapshot.registrations {
                if let Some(dimension) = entry.dimension {
                    let name = entry.model_name.clone();
                    models.insert(
                        name,
                        XinferenceModelSpec {
                            dimension,
                            model_type: entry.model_type.clone(),
                        },
                    );
                }
            }
        }
        models
    })
}

/// Registry of known Xinference reranker models from snapshots.
pub fn xinference_reranker_models() -> &'static HashSet<String> {
    XINFERENCE_RERANK_MODELS.get_or_init(|| {
        let mut models = HashSet::new();
        if let Ok(snapshot) = registry_snapshot(RegistryType::Rerank) {
            for entry in &snapshot.registrations {
                models.insert(entry.model_name.clone());
                models.insert(hf_to_xinference_name(&entry.model_id));
            }
        }
        models
    })
}

/// Map HuggingFace model ID to Xinference model name.
/// Removes the organization prefix (e.g., "BAAI/bge-small-en-v1.5" -> "bge-small-en-v1.5")
pub fn hf_to_xinference_name(hf_model: &str) -> String {
    hf_model
        .split('/')
        .next_back()
        .unwrap_or(hf_model)
        .to_string()
}

/// Get the Xinference model spec for a given model name
pub fn get_xinference_model_spec(model: &str) -> Option<XinferenceModelSpec> {
    let xinf_name = hf_to_xinference_name(model);
    xinference_embedding_models().get(&xinf_name).cloned()
}

/// Running model information from Xinference
#[derive(Debug, Clone, Deserialize)]
pub struct RunningModel {
    #[serde(alias = "id", alias = "model_uid")]
    pub model_uid: String,
    #[serde(default, alias = "model_name", alias = "model")]
    pub model_name: String,
    #[serde(default, alias = "model_type")]
    pub model_type: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ModelListResponse {
    Array(Vec<RunningModel>),
    Data { data: Vec<RunningModel> },
    Models { models: Vec<RunningModel> },
}

/// Model launch request
#[derive(Debug, Serialize)]
struct ModelLaunchRequest {
    model_name: String,
    model_type: String,
}

/// Model launch response
#[derive(Debug, Deserialize)]
struct ModelLaunchResponse {
    model_uid: String,
}

/// Manages the Xinference server lifecycle and model operations
pub struct XinferenceManager {
    port: u16,
    process: Option<Child>,
    client: Client,
    base_url: Url,
    launched_models: HashMap<String, String>, // model_name -> model_uid
}

impl XinferenceManager {
    /// Create a new XinferenceManager with the given port
    pub fn new(port: u16) -> Result<Self> {
        let base_url = Url::parse(&format!("http://127.0.0.1:{}", port))
            .map_err(|e| Error::Config(format!("Invalid Xinference URL: {}", e)))?;

        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| Error::Embedding(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            port,
            process: None,
            client,
            base_url,
            launched_models: HashMap::new(),
        })
    }

    /// Get the base URL for API calls
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    /// Get the port number
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Check if the xinference server process is running (as our child)
    pub fn is_process_running(&self) -> bool {
        self.process.is_some()
    }

    /// Start the Xinference server as a subprocess
    pub async fn start(&mut self) -> Result<()> {
        // Check if already running (as our child process)
        if self.process.is_some() {
            debug!("Xinference server already running as child process");
            return Ok(());
        }

        // Check if xinference is already running externally on this port
        if self.health_check().await? {
            info!("Xinference server already running on port {}", self.port);
            return Ok(());
        }

        info!("Starting Xinference server on port {}...", self.port);

        // Spawn xinference-local as subprocess
        let child = Command::new("xinference-local")
            .args(["--host", "0.0.0.0", "--port", &self.port.to_string()])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                Error::Embedding(format!(
                    "Failed to start xinference-local: {}. Is xinference installed?",
                    e
                ))
            })?;

        self.process = Some(child);

        // Wait for server to be ready
        self.wait_for_ready().await?;

        info!("Xinference server started successfully");
        Ok(())
    }

    /// Stop the Xinference server
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(mut child) = self.process.take() {
            info!("Stopping Xinference server...");

            // Send kill signal
            if let Err(e) = child.kill() {
                warn!("Failed to kill xinference process: {}", e);
            }

            // Wait for process to exit
            if let Err(e) = child.wait() {
                warn!("Failed to wait for xinference process: {}", e);
            }

            info!("Xinference server stopped");
        }

        self.launched_models.clear();
        Ok(())
    }

    /// Check if the server is healthy by querying the models endpoint
    pub async fn health_check(&self) -> Result<bool> {
        let url = self
            .base_url
            .join("/v1/models")
            .map_err(|e| Error::Config(format!("Invalid URL: {}", e)))?;

        match self.client.get(url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Wait for the server to become ready
    async fn wait_for_ready(&self) -> Result<()> {
        let max_attempts = 60; // 30 seconds total
        let delay = Duration::from_millis(500);

        for attempt in 0..max_attempts {
            if self.health_check().await? {
                debug!("Xinference server ready after {} attempts", attempt + 1);
                return Ok(());
            }
            tokio::time::sleep(delay).await;
        }

        Err(Error::Embedding(
            "Xinference server failed to start within 30 seconds".to_string(),
        ))
    }

    /// Ensure the server is running, starting it if needed
    pub async fn ensure_running(&mut self) -> Result<()> {
        if !self.health_check().await? {
            self.start().await?;
        }
        Ok(())
    }

    /// List all running models
    pub async fn list_running_models(&self) -> Result<Vec<RunningModel>> {
        let url = self
            .base_url
            .join("/v1/models")
            .map_err(|e| Error::Config(format!("Invalid URL: {}", e)))?;

        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| Error::Embedding(format!("Failed to list models: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::Embedding(format!(
                "Failed to list models: HTTP {}",
                response.status()
            )));
        }

        let body = response
            .text()
            .await
            .map_err(|e| Error::Embedding(format!("Failed to read models response: {}", e)))?;

        let parsed: ModelListResponse = serde_json::from_str(&body).map_err(|e| {
            Error::Embedding(format!(
                "Failed to parse models response: {} (body: {})",
                e,
                truncate_body(&body, 1000)
            ))
        })?;

        let mut models = match parsed {
            ModelListResponse::Array(models) => models,
            ModelListResponse::Data { data } => data,
            ModelListResponse::Models { models } => models,
        };

        for model in &mut models {
            if model.model_name.is_empty() {
                model.model_name = model.model_uid.clone();
            }
        }

        Ok(models)
    }

    /// Query if a specific model is running
    pub async fn query_running_model(&self, model_name: &str) -> Result<Option<String>> {
        let models = self.list_running_models().await?;
        Ok(models
            .iter()
            .find(|m| m.model_name == model_name)
            .map(|m| m.model_uid.clone()))
    }

    /// Launch a model in Xinference
    pub async fn launch_model(&mut self, model_name: &str, model_type: &str) -> Result<String> {
        info!("Launching {} model: {}", model_type, model_name);

        let mut attempted_fallback = false;
        loop {
            let url = self
                .base_url
                .join("/v1/models")
                .map_err(|e| Error::Config(format!("Invalid URL: {}", e)))?;

            let request = ModelLaunchRequest {
                model_name: model_name.to_string(),
                model_type: model_type.to_string(),
            };

            let response = self
                .client
                .post(url)
                .json(&request)
                .send()
                .await
                .map_err(|e| Error::Embedding(format!("Failed to launch model: {}", e)))?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                if !attempted_fallback && is_xoscar_start_method_error(&body) {
                    attempted_fallback = true;
                    warn!(
                        "Detected incompatible xinference server on port {} (xoscar start_method mismatch). Starting managed server on a new port...",
                        self.port
                    );
                    self.restart_on_fallback_port().await?;
                    continue;
                }
                return Err(Error::Embedding(format!(
                    "Failed to launch model '{}': HTTP {} - {}",
                    model_name, status, body
                )));
            }

            let launch_response: ModelLaunchResponse = response
                .json()
                .await
                .map_err(|e| Error::Embedding(format!("Failed to parse launch response: {}", e)))?;

            self.launched_models
                .insert(model_name.to_string(), launch_response.model_uid.clone());

            info!(
                "Model {} launched with UID: {}",
                model_name, launch_response.model_uid
            );

            return Ok(launch_response.model_uid);
        }
    }

    async fn restart_on_fallback_port(&mut self) -> Result<()> {
        let new_port = find_available_port()?;
        self.stop().await?;
        self.set_port(new_port)?;
        self.start().await?;
        Ok(())
    }

    fn set_port(&mut self, port: u16) -> Result<()> {
        let base_url = Url::parse(&format!("http://127.0.0.1:{}", port))
            .map_err(|e| Error::Config(format!("Invalid Xinference URL: {}", e)))?;
        self.port = port;
        self.base_url = base_url;
        Ok(())
    }

    /// Check if a model is already launched (in our tracking)
    pub fn is_model_launched(&self, model_name: &str) -> bool {
        self.launched_models.contains_key(model_name)
    }

    /// Get the UID for a launched model
    pub fn get_model_uid(&self, model_name: &str) -> Option<&String> {
        self.launched_models.get(model_name)
    }

    /// Ensure a model is launched, launching it if needed
    pub async fn ensure_model_launched(
        &mut self,
        hf_model: &str,
        model_type: &str,
    ) -> Result<String> {
        let xinf_name = hf_to_xinference_name(hf_model);

        // Check our local tracking
        if let Some(uid) = self.launched_models.get(&xinf_name) {
            return Ok(uid.clone());
        }

        // Check if already running in Xinference
        if let Some(uid) = self.query_running_model(&xinf_name).await? {
            self.launched_models.insert(xinf_name.clone(), uid.clone());
            return Ok(uid);
        }

        // Launch the model
        self.launch_model(&xinf_name, model_type).await
    }
}

impl Drop for XinferenceManager {
    fn drop(&mut self) {
        if let Some(mut child) = self.process.take() {
            debug!("Cleaning up Xinference process on drop");
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

// Global manager for singleton pattern
static GLOBAL_XINFERENCE_MANAGER: OnceLock<Mutex<Option<XinferenceManager>>> = OnceLock::new();

/// Get or initialize the global Xinference manager
pub async fn get_or_init_xinference_manager(
    port: u16,
) -> Result<&'static Mutex<Option<XinferenceManager>>> {
    let manager_lock = GLOBAL_XINFERENCE_MANAGER.get_or_init(|| Mutex::new(None));

    {
        let mut guard = manager_lock.lock().await;
        if guard.is_none() {
            let manager = XinferenceManager::new(port)?;
            *guard = Some(manager);
        }
    }

    Ok(manager_lock)
}

/// Shutdown the global Xinference manager if running
pub async fn shutdown_global_xinference() {
    if let Some(manager_lock) = GLOBAL_XINFERENCE_MANAGER.get() {
        let mut guard = manager_lock.lock().await;
        if let Some(ref mut manager) = *guard {
            if let Err(e) = manager.stop().await {
                warn!("Error stopping Xinference manager: {}", e);
            }
        }
        *guard = None;
    }
}

fn truncate_body(body: &str, limit: usize) -> String {
    let trimmed: String = body.chars().take(limit).collect();
    if body.chars().count() > limit {
        format!("{}…", trimmed)
    } else {
        trimmed
    }
}

fn is_xoscar_start_method_error(body: &str) -> bool {
    body.contains("append_sub_pool() got an unexpected keyword argument 'start_method'")
}

fn find_available_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| Error::Embedding(format!("Failed to bind a local port: {}", e)))?;
    let port = listener
        .local_addr()
        .map_err(|e| Error::Embedding(format!("Failed to read local port: {}", e)))?
        .port();
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_to_xinference_name() {
        assert_eq!(
            hf_to_xinference_name("BAAI/bge-small-en-v1.5"),
            "bge-small-en-v1.5"
        );
        assert_eq!(
            hf_to_xinference_name("sentence-transformers/all-MiniLM-L6-v2"),
            "all-MiniLM-L6-v2"
        );
        assert_eq!(
            hf_to_xinference_name("bge-small-en-v1.5"),
            "bge-small-en-v1.5"
        );
        assert_eq!(
            hf_to_xinference_name("BAAI/bge-reranker-base"),
            "bge-reranker-base"
        );
    }

    #[test]
    fn test_xinference_model_registry() {
        let models = xinference_embedding_models();
        assert!(models.contains_key("bge-small-en-v1.5"));
        if let Some(spec) = models.get("bge-small-en-v1.5") {
            assert!(spec.dimension > 0);
        }
    }

    #[test]
    fn test_get_xinference_model_spec() {
        // HuggingFace format
        let spec = get_xinference_model_spec("BAAI/bge-small-en-v1.5");
        assert!(spec.is_some());
        assert_eq!(spec.unwrap().dimension, 384);

        // Direct name
        let spec = get_xinference_model_spec("bge-base-en-v1.5");
        assert!(spec.is_some());
        assert_eq!(spec.unwrap().dimension, 768);

        // Unknown model
        let spec = get_xinference_model_spec("unknown-model");
        assert!(spec.is_none());
    }

    #[test]
    fn test_xinference_reranker_models() {
        let models = xinference_reranker_models();
        assert!(models.contains("bge-reranker-base"));
    }
}
