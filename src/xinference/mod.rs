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
use reqwest::{Client, Method, StatusCode};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::time::sleep;
use tracing::{debug, info, warn};
use url::Url;

/// Default Xinference server port
pub const DEFAULT_XINFERENCE_PORT: u16 = 9997;

pub type SharedXinferenceManager = Arc<Mutex<Option<XinferenceManager>>>;

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

#[derive(Debug, Deserialize)]
struct ModelRegistration {
    model_name: String,
    #[serde(default)]
    pub(crate) dimensions: Option<usize>,
    #[serde(default)]
    max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ModelRegistrationItem {
    model_name: String,
    #[serde(default)]
    is_builtin: bool,
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
    process: Option<Child>,
    client: Client,
    base_url: Url,
    auth_token: Option<String>,
    launched_models: HashMap<String, String>, // model_name -> model_uid
}

impl XinferenceManager {
    /// Create a new XinferenceManager with the given base URL.
    pub fn new(base_url: Url, auth_token: Option<String>) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| Error::Embedding(format!("Failed to create HTTP client: {}", e)))?;

        debug!(base_url = %base_url, "Initialized Xinference manager");

        Ok(Self {
            process: None,
            client,
            base_url,
            auth_token,
            launched_models: HashMap::new(),
        })
    }

    /// Get the base URL for API calls
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    /// Get the port number
    pub fn port(&self) -> u16 {
        self.base_url
            .port_or_known_default()
            .unwrap_or(DEFAULT_XINFERENCE_PORT)
    }

    pub fn auth_token(&self) -> Option<&str> {
        self.auth_token.as_deref()
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
            info!("Xinference server already running at {}", self.base_url);
            return Ok(());
        }

        if !is_localhost(&self.base_url) {
            return Err(Error::Embedding(format!(
                "Xinference server at {} is not reachable and auto-start is only supported for localhost URLs. Start Xinference manually or update embedding.url.",
                self.base_url
            )));
        }

        let host = self
            .base_url
            .host_str()
            .unwrap_or("127.0.0.1")
            .to_string();
        let port = self.port();

        info!("Starting Xinference server at {}...", self.base_url);

        // Spawn xinference-local as subprocess
        let child = Command::new("xinference-local")
            .args(["--host", &host, "--port", &port.to_string()])
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

    /// Check if the server is healthy by querying the cluster version endpoint
    pub async fn health_check(&self) -> Result<bool> {
        let url = self
            .base_url
            .join("/v1/cluster/version")
            .map_err(|e| Error::Config(format!("Invalid URL: {}", e)))?;

        match xinference_request(
            &self.client,
            Method::GET,
            url,
            self.auth_token.as_deref(),
            None,
            "health_check",
        )
        .await
        {
            Ok((status, _)) => {
                if status.is_success() {
                    Ok(true)
                } else if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
                    Err(Error::Embedding(
                        "Xinference authentication required. Set LIBRARIAN_XINFERENCE_API_KEY or XINFERENCE_API_KEY."
                            .to_string(),
                    ))
                } else {
                    Ok(false)
                }
            }
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
        let healthy = self.health_check().await?;
        if healthy {
            return Ok(());
        }

        if !is_localhost(&self.base_url) {
            return Err(Error::Embedding(format!(
                "Xinference server at {} is not reachable. Start the server or update embedding.url/reranker.url.",
                self.base_url
            )));
        }

        self.start().await?;
        Ok(())
    }

    /// Fetch a model registration record for a given type/name.
    pub async fn fetch_model_registration(
        &self,
        model_type: &str,
        model_name: &str,
    ) -> Result<ModelRegistration> {
        let url = self
            .base_url
            .join(&format!(
                "/v1/model_registrations/{}/{}",
                model_type, model_name
            ))
            .map_err(|e| Error::Config(format!("Invalid Xinference URL: {}", e)))?;

        let (status, body_text) = xinference_request(
            &self.client,
            Method::GET,
            url,
            self.auth_token.as_deref(),
            None,
            "model_registration",
        )
        .await?;

        if status.is_success() {
            let registration: ModelRegistration = serde_json::from_str(&body_text).map_err(|e| {
                Error::Embedding(format!(
                    "Failed to parse Xinference model registration: {} (body: {})",
                    e,
                    truncate_body(&body_text, 1000)
                ))
            })?;

            debug!(
                model_name = %registration.model_name,
                dimensions = registration.dimensions,
                max_tokens = registration.max_tokens,
                "Resolved Xinference model registration"
            );

            return Ok(registration);
        }

        if status == StatusCode::NOT_FOUND {
            let available = self
                .list_model_registrations(model_type)
                .await
                .unwrap_or_default();
            let suggestions = available
                .iter()
                .take(10)
                .map(|item| item.model_name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(Error::Config(format!(
                "Xinference model '{}' (type '{}') not found. Available models include: {}",
                model_name, model_type, suggestions
            )));
        }

        if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
            return Err(Error::Embedding(
                "Xinference authentication required. Set LIBRARIAN_XINFERENCE_API_KEY or XINFERENCE_API_KEY."
                    .to_string(),
            ));
        }

        Err(Error::Embedding(format!(
            "Xinference model registration lookup failed: HTTP {} - {}",
            status,
            truncate_body(&body_text, 1000)
        )))
    }

    /// List available model registrations for a given type.
    pub async fn list_model_registrations(
        &self,
        model_type: &str,
    ) -> Result<Vec<ModelRegistrationItem>> {
        let url = self
            .base_url
            .join(&format!("/v1/model_registrations/{}", model_type))
            .map_err(|e| Error::Config(format!("Invalid Xinference URL: {}", e)))?;

        let registrations: Vec<ModelRegistrationItem> = xinference_request_json(
            &self.client,
            Method::GET,
            url,
            self.auth_token.as_deref(),
            None,
            "list_model_registrations",
        )
        .await?;

        Ok(registrations)
    }

    /// List all running models
    pub async fn list_running_models(&self) -> Result<Vec<RunningModel>> {
        let url = self
            .base_url
            .join("/v1/models")
            .map_err(|e| Error::Config(format!("Invalid URL: {}", e)))?;
        let parsed: ModelListResponse = xinference_request_json(
            &self.client,
            Method::GET,
            url,
            self.auth_token.as_deref(),
            None,
            "list_models",
        )
        .await?;

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

        let mut url = self
            .base_url
            .join("/v1/models")
            .map_err(|e| Error::Config(format!("Invalid URL: {}", e)))?;
        url.query_pairs_mut()
            .append_pair("wait_ready", "true");

        let request = ModelLaunchRequest {
            model_name: model_name.to_string(),
            model_type: model_type.to_string(),
        };
        let body = serde_json::to_value(request)
            .map_err(|e| Error::Embedding(format!("Failed to serialize launch request: {}", e)))?;

        let launch_response: ModelLaunchResponse = xinference_request_json(
            &self.client,
            Method::POST,
            url,
            self.auth_token.as_deref(),
            Some(body),
            "launch_model",
        )
        .await?;

        self.launched_models
            .insert(model_name.to_string(), launch_response.model_uid.clone());

        info!(
            model_name = %model_name,
            model_type = %model_type,
            model_uid = %launch_response.model_uid,
            "Xinference model launched"
        );

        Ok(launch_response.model_uid)
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
        require_registry: bool,
    ) -> Result<String> {
        let xinf_name = hf_to_xinference_name(hf_model);
        debug!(
            model_id = %hf_model,
            model_name = %xinf_name,
            model_type = %model_type,
            "Resolving Xinference model"
        );

        // Ensure model exists in registry for deterministic selection
        if require_registry {
            let _ = self.fetch_model_registration(model_type, &xinf_name).await?;
        }

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
static GLOBAL_XINFERENCE_MANAGER: OnceLock<Arc<Mutex<Option<XinferenceManager>>>> = OnceLock::new();

/// Get or initialize the global Xinference manager
pub async fn get_or_init_xinference_manager(
    base_url: &Url,
    auth_token: Option<String>,
) -> Result<SharedXinferenceManager> {
    let manager_lock = GLOBAL_XINFERENCE_MANAGER
        .get_or_init(|| Arc::new(Mutex::new(None)))
        .clone();

    {
        let mut guard = manager_lock.lock().await;
        if let Some(ref manager) = *guard {
            if manager.base_url() != base_url {
                return Err(Error::Config(format!(
                    "Xinference manager already initialized for {}, but requested {}. Use a single Xinference endpoint or restart.",
                    manager.base_url(),
                    base_url
                )));
            }
        } else {
            let manager = XinferenceManager::new(base_url.clone(), auth_token)?;
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

fn is_localhost(url: &Url) -> bool {
    matches!(
        url.host_str(),
        Some("127.0.0.1") | Some("localhost") | Some("0.0.0.0") | Some("::1")
    )
}

fn format_error_chain(err: &dyn std::error::Error) -> String {
    let mut parts = vec![err.to_string()];
    let mut current = err.source();
    while let Some(source) = current {
        parts.push(source.to_string());
        current = source.source();
    }
    parts.join(" -> ")
}

pub(crate) fn xinference_auth_token() -> Option<String> {
    std::env::var("LIBRARIAN_XINFERENCE_API_KEY")
        .ok()
        .or_else(|| std::env::var("XINFERENCE_API_KEY").ok())
        .or_else(|| std::env::var("XINFERENCE_AUTH_TOKEN").ok())
}

pub(crate) fn normalize_xinference_base_url(raw: &str) -> Result<Url> {
    let mut url =
        Url::parse(raw).map_err(|e| Error::Config(format!("Invalid Xinference URL '{}': {}", raw, e)))?;

    match url.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(Error::Config(format!(
                "Unsupported Xinference URL scheme '{}'; use http or https",
                scheme
            )));
        }
    }

    if url.port().is_none() {
        url.set_port(Some(DEFAULT_XINFERENCE_PORT))
            .map_err(|_| Error::Config("Failed to apply default Xinference port".to_string()))?;
    }

    let path = url.path().trim_end_matches('/');
    if !path.is_empty() && path != "/" {
        if path == "/v1" {
            url.set_path("/");
        } else {
            return Err(Error::Config(format!(
                "Xinference base URL must not include a path (got '{}'). Use http://host:port",
                url.path()
            )));
        }
    } else {
        url.set_path("/");
    }

    url.set_query(None);
    url.set_fragment(None);

    Ok(url)
}

pub(crate) async fn xinference_request(
    client: &Client,
    method: Method,
    url: Url,
    auth_token: Option<&str>,
    body: Option<serde_json::Value>,
    context: &str,
) -> Result<(StatusCode, String)> {
    let mut delay = Duration::from_millis(200);
    let max_retries = 3;

    for attempt in 0..=max_retries {
        debug!(
            method = %method,
            url = %url,
            attempt,
            "Xinference request"
        );

        let start = Instant::now();
        let mut builder = client.request(method.clone(), url.clone());
        if let Some(token) = auth_token {
            builder = builder.bearer_auth(token);
        }
        if let Some(ref payload) = body {
            builder = builder.json(payload);
        }

        let response = builder.send().await;
        match response {
            Ok(resp) => {
                let status = resp.status();
                let latency_ms = start.elapsed().as_millis();
                let body_text = resp.text().await.unwrap_or_default();
                debug!(
                    method = %method,
                    url = %url,
                    status = %status,
                    latency_ms,
                    "Xinference response"
                );

                if status.is_server_error() && attempt < max_retries {
                    debug!(
                        method = %method,
                        url = %url,
                        status = %status,
                        attempt,
                        "Retrying Xinference request after server error"
                    );
                    sleep(delay).await;
                    delay = delay.saturating_mul(2);
                    continue;
                }

                return Ok((status, body_text));
            }
            Err(e) => {
                let latency_ms = start.elapsed().as_millis();
                debug!(
                    method = %method,
                    url = %url,
                    latency_ms,
                    error_chain = %format_error_chain(&e),
                    "Xinference request error"
                );

                let retryable = e.is_timeout() || e.is_connect();
                if retryable && attempt < max_retries {
                    sleep(delay).await;
                    delay = delay.saturating_mul(2);
                    continue;
                }

                return Err(Error::Embedding(format!(
                    "Xinference request failed ({}) after {} attempt(s): {}",
                    context,
                    attempt + 1,
                    e
                )));
            }
        }
    }

    Err(Error::Embedding(format!(
        "Xinference request failed ({}) after retries",
        context
    )))
}

pub(crate) async fn xinference_request_json<T: DeserializeOwned>(
    client: &Client,
    method: Method,
    url: Url,
    auth_token: Option<&str>,
    body: Option<serde_json::Value>,
    context: &str,
) -> Result<T> {
    let (status, body_text) =
        xinference_request(client, method, url, auth_token, body, context).await?;

    if status.is_success() {
        return serde_json::from_str(&body_text).map_err(|e| {
            Error::Embedding(format!(
                "Failed to parse Xinference {} response: {} (body: {})",
                context,
                e,
                truncate_body(&body_text, 1000)
            ))
        });
    }

    if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
        return Err(Error::Embedding(format!(
            "Xinference {} failed: HTTP {} - {}. Set LIBRARIAN_XINFERENCE_API_KEY or XINFERENCE_API_KEY if auth is enabled.",
            context,
            status,
            truncate_body(&body_text, 1000)
        )));
    }

    if status.is_client_error() {
        return Err(Error::Embedding(format!(
            "Xinference {} failed: HTTP {} - {}",
            context,
            status,
            truncate_body(&body_text, 1000)
        )));
    }

    Err(Error::Embedding(format!(
        "Xinference {} failed after retries: HTTP {} - {}",
        context,
        status,
        truncate_body(&body_text, 1000)
    )))
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
