//! Configuration management for librarian
//!
//! Handles loading, saving, and validating configuration from TOML files.

mod defaults;

pub use defaults::*;

use crate::embedding_backend::{
    BackendCapabilities, EmbeddingBackendClient, EmbeddingBackendConfig, EmbeddingBackendKind,
};
use crate::error::{Error, Result};
use std::str::FromStr;
use crate::models::{
    allowlisted_embedding_models, allowlisted_reranker_models, embedding_model_capabilities,
    embedding_model_spec, reranker_model_spec, supported_multimodal_embedding_models,
    MultimodalStrategy,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

const PROBE_TEXT: &str = "ping";
const PROBE_IMAGE_PNG_BASE64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=";

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Qdrant connection URL
    #[serde(default = "default_qdrant_url")]
    pub qdrant_url: String,

    /// Environment variable name for Qdrant API key
    #[serde(default = "default_qdrant_api_key_env")]
    pub qdrant_api_key_env: String,

    /// Qdrant collection name
    #[serde(default = "default_collection_name")]
    pub collection_name: String,

    /// Embedding model configuration
    #[serde(default)]
    pub embedding: EmbeddingConfig,

    /// Chunking configuration
    #[serde(default)]
    pub chunk: ChunkConfig,

    /// Web crawling configuration
    #[serde(default)]
    pub crawl: CrawlConfig,

    /// Query configuration
    #[serde(default)]
    pub query: QueryConfig,

    /// Reranker configuration
    #[serde(default)]
    pub reranker: RerankerConfig,

    /// Paths configuration (internal, not user-editable)
    #[serde(skip)]
    pub paths: PathsConfig,
}

/// Custom embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomEmbeddingConfig {
    /// Custom model identifier
    #[serde(default)]
    pub id: String,

    /// Backend kind (http)
    #[serde(default = "default_embedding_custom_backend")]
    pub backend: String,

    /// Backend base URL
    #[serde(default = "default_embedding_custom_url")]
    pub url: String,

    /// Optional model family label
    #[serde(default)]
    pub family: Option<String>,

    /// Supported modalities (text, image, image_text)
    #[serde(default)]
    pub modalities: Vec<String>,

    /// Optional embedding dimension for custom model
    #[serde(default)]
    pub dimension: Option<usize>,

    /// Whether the model emits multi-vector representations
    #[serde(default)]
    pub multivector: Option<bool>,

    /// Whether the model supports MRL
    #[serde(default)]
    pub supports_mrl: Option<bool>,

    /// Maximum batch size supported by the backend
    #[serde(default)]
    pub max_batch: Option<usize>,
}

/// Embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model name/identifier
    #[serde(default = "default_embedding_model")]
    pub model: String,

    /// Embedding backend kind (http)
    #[serde(default = "default_embedding_backend")]
    pub backend: String,

    /// Embedding backend URL
    #[serde(default = "default_embedding_backend_url")]
    pub url: String,

    /// Allow custom embedding models
    #[serde(default = "default_embedding_allow_custom")]
    pub allow_custom: bool,

    /// Enable multimodal embedding endpoints
    #[serde(default = "default_embedding_multimodal")]
    pub multimodal: bool,

    /// Optional embedding dimension override
    #[serde(default)]
    pub dimension: Option<usize>,

    /// Batch size for embedding
    #[serde(default = "default_embedding_batch_size")]
    pub batch_size: usize,

    /// Custom model and backend metadata
    #[serde(default)]
    pub custom: CustomEmbeddingConfig,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingDimensionSource {
    Config,
    Probe,
    Registry,
    Custom,
}

impl EmbeddingDimensionSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            EmbeddingDimensionSource::Config => "config",
            EmbeddingDimensionSource::Probe => "probe",
            EmbeddingDimensionSource::Registry => "registry",
            EmbeddingDimensionSource::Custom => "custom",
        }
    }
}

impl fmt::Display for EmbeddingDimensionSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedEmbeddingConfig {
    pub model_id: String,
    pub family: String,
    pub modalities: Vec<String>,
    pub dimension: usize,
    pub dimension_source: EmbeddingDimensionSource,
    pub backend: EmbeddingBackendConfig,
    pub strategy: MultimodalStrategy,
    pub supports_text: bool,
    pub supports_image: bool,
    pub supports_joint_inputs: bool,
    pub supports_multi_vector: bool,
    pub supports_mrl: bool,
    pub max_batch: usize,
}

impl ResolvedEmbeddingConfig {
    pub fn supports_image_inputs(&self) -> bool {
        self.supports_image || self.supports_joint_inputs
    }

    pub fn effective_batch_size(&self, configured: usize) -> usize {
        if configured == 0 {
            return self.max_batch.max(1);
        }
        configured.min(self.max_batch.max(1))
    }
}

/// Chunking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Maximum characters per chunk
    #[serde(default = "default_chunk_max_chars")]
    pub max_chars: usize,

    /// Overlap characters between chunks
    #[serde(default = "default_chunk_overlap")]
    pub overlap_chars: usize,

    /// Prefer breaking at heading boundaries
    #[serde(default = "default_prefer_heading_boundaries")]
    pub prefer_heading_boundaries: bool,

    /// Minimum chunk size (don't create tiny chunks)
    #[serde(default = "default_chunk_min_chars")]
    pub min_chars: usize,
}

/// Web crawling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlConfig {
    /// Maximum crawl depth from seed URL
    #[serde(default = "default_crawl_max_depth")]
    pub max_depth: u32,

    /// Maximum pages to crawl per source
    #[serde(default = "default_crawl_max_pages")]
    pub max_pages: u32,

    /// Allowed domains (empty = same domain as seed only)
    #[serde(default)]
    pub allowed_domains: Vec<String>,

    /// Path prefix to restrict crawling to (e.g., /docs/)
    /// Empty means no path restriction (entire domain allowed)
    #[serde(default)]
    pub path_prefix: Option<String>,

    /// Requests per second per host
    #[serde(default = "default_crawl_rate_limit")]
    pub rate_limit_per_host: f64,

    /// User agent string
    #[serde(default = "default_crawl_user_agent")]
    pub user_agent: String,

    /// Request timeout in seconds
    #[serde(default = "default_crawl_timeout")]
    pub timeout_secs: u64,

    /// Whether to respect robots.txt
    #[serde(default = "default_respect_robots")]
    pub respect_robots_txt: bool,

    /// Auto-detect SPAs and use JavaScript rendering when needed
    #[serde(default = "default_auto_js_rendering")]
    pub auto_js_rendering: bool,

    /// Time to wait for page load when JS rendering (milliseconds)
    #[serde(default = "default_js_page_load_timeout")]
    pub js_page_load_timeout_ms: u64,

    /// Time to wait after load for dynamic content (milliseconds)
    #[serde(default = "default_js_render_wait")]
    pub js_render_wait_ms: u64,

    /// Disable browser sandbox (required in some Docker/CI environments)
    #[serde(default)]
    pub js_no_sandbox: bool,

    /// Multimodal crawling configuration
    #[serde(default)]
    pub multimodal: MultimodalCrawlConfig,
}

/// Query configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfig {
    /// Default number of results
    #[serde(default = "default_query_k")]
    pub default_k: usize,

    /// Maximum results allowed
    #[serde(default = "default_query_max_results")]
    pub max_results: usize,

    /// Minimum similarity score (0.0 - 1.0)
    #[serde(default = "default_query_min_score")]
    pub min_score: f32,

    /// Enable hybrid BM25 + vector search
    #[serde(default)]
    pub hybrid_search: bool,

    /// BM25 weight when hybrid is enabled (0.0 - 1.0)
    #[serde(default = "default_bm25_weight")]
    pub bm25_weight: f32,
}

/// Reranker configuration (cross-encoder model for result reranking)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerConfig {
    /// Enable reranking (all retrieved results are reranked)
    #[serde(default = "default_reranker_enabled")]
    pub enabled: bool,

    /// Model name/identifier for cross-encoder reranker
    #[serde(default = "default_reranker_model")]
    pub model: String,

    /// Number of results to return after reranking
    #[serde(default = "default_reranker_top_k")]
    pub top_k: usize,
}

/// Multimodal crawling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalCrawlConfig {
    /// Enable multimodal asset discovery and indexing
    #[serde(default = "default_multimodal_enabled")]
    pub enabled: bool,

    /// Include images
    #[serde(default = "default_multimodal_include_images")]
    pub include_images: bool,

    /// Include audio (not yet supported)
    #[serde(default = "default_multimodal_include_audio")]
    pub include_audio: bool,

    /// Include video (not yet supported)
    #[serde(default = "default_multimodal_include_video")]
    pub include_video: bool,

    /// Maximum allowed asset size in bytes
    #[serde(default = "default_multimodal_max_asset_bytes")]
    pub max_asset_bytes: usize,

    /// Minimum allowed asset size in bytes
    #[serde(default = "default_multimodal_min_asset_bytes")]
    pub min_asset_bytes: usize,

    /// Maximum assets per page
    #[serde(default = "default_multimodal_max_assets_per_page")]
    pub max_assets_per_page: usize,

    /// Allowed MIME type prefixes (e.g., ["image/"])
    #[serde(default = "default_multimodal_allowed_mime_prefixes")]
    pub allowed_mime_prefixes: Vec<String>,

    /// Minimum relevance score (0.0 - 1.0)
    #[serde(default = "default_multimodal_min_relevance_score")]
    pub min_relevance_score: f32,

    /// Include CSS background images if detected
    #[serde(default = "default_multimodal_include_css_background_images")]
    pub include_css_background_images: bool,
}

/// Internal paths configuration
#[derive(Debug, Clone, Default)]
pub struct PathsConfig {
    /// Base directory for librarian data
    pub base_dir: PathBuf,

    /// Path to config file
    pub config_file: PathBuf,

    /// Path to SQLite database
    pub db_file: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            qdrant_url: default_qdrant_url(),
            qdrant_api_key_env: default_qdrant_api_key_env(),
            collection_name: default_collection_name(),
            embedding: EmbeddingConfig::default(),
            chunk: ChunkConfig::default(),
            crawl: CrawlConfig::default(),
            query: QueryConfig::default(),
            reranker: RerankerConfig::default(),
            paths: PathsConfig::default(),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: default_embedding_model(),
            backend: default_embedding_backend(),
            url: default_embedding_backend_url(),
            allow_custom: default_embedding_allow_custom(),
            multimodal: default_embedding_multimodal(),
            dimension: None,
            batch_size: default_embedding_batch_size(),
            custom: CustomEmbeddingConfig::default(),
        }
    }
}

impl Default for CustomEmbeddingConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            backend: default_embedding_custom_backend(),
            url: default_embedding_custom_url(),
            family: None,
            modalities: Vec::new(),
            dimension: None,
            multivector: None,
            supports_mrl: None,
            max_batch: None,
        }
    }
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_chars: default_chunk_max_chars(),
            overlap_chars: default_chunk_overlap(),
            prefer_heading_boundaries: default_prefer_heading_boundaries(),
            min_chars: default_chunk_min_chars(),
        }
    }
}

impl Default for CrawlConfig {
    fn default() -> Self {
        Self {
            max_depth: default_crawl_max_depth(),
            max_pages: default_crawl_max_pages(),
            allowed_domains: Vec::new(),
            path_prefix: None,
            rate_limit_per_host: default_crawl_rate_limit(),
            user_agent: default_crawl_user_agent(),
            timeout_secs: default_crawl_timeout(),
            respect_robots_txt: default_respect_robots(),
            auto_js_rendering: default_auto_js_rendering(),
            js_page_load_timeout_ms: default_js_page_load_timeout(),
            js_render_wait_ms: default_js_render_wait(),
            js_no_sandbox: false,
            multimodal: MultimodalCrawlConfig::default(),
        }
    }
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            default_k: default_query_k(),
            max_results: default_query_max_results(),
            min_score: default_query_min_score(),
            hybrid_search: false,
            bm25_weight: default_bm25_weight(),
        }
    }
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            enabled: default_reranker_enabled(),
            model: default_reranker_model(),
            top_k: default_reranker_top_k(),
        }
    }
}

impl Default for MultimodalCrawlConfig {
    fn default() -> Self {
        Self {
            enabled: default_multimodal_enabled(),
            include_images: default_multimodal_include_images(),
            include_audio: default_multimodal_include_audio(),
            include_video: default_multimodal_include_video(),
            max_asset_bytes: default_multimodal_max_asset_bytes(),
            min_asset_bytes: default_multimodal_min_asset_bytes(),
            max_assets_per_page: default_multimodal_max_assets_per_page(),
            allowed_mime_prefixes: default_multimodal_allowed_mime_prefixes(),
            min_relevance_score: default_multimodal_min_relevance_score(),
            include_css_background_images: default_multimodal_include_css_background_images(),
        }
    }
}

impl Config {
    /// Get the default base directory for librarian (~/.librarian)
    pub fn default_base_dir() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".librarian")
    }

    /// Get the default config file path
    pub fn default_config_path() -> PathBuf {
        Self::default_base_dir().join("config.toml")
    }

    /// Initialize paths configuration
    fn init_paths(&mut self, base_dir: Option<PathBuf>) {
        let base = base_dir.unwrap_or_else(Self::default_base_dir);
        self.paths = PathsConfig {
            config_file: base.join("config.toml"),
            db_file: base.join("metadata.db"),
            base_dir: base,
        };
    }

    /// Load configuration from a specific file path
    pub fn load(config_path: &Path) -> Result<Self> {
        debug!("Loading config from {:?}", config_path);

        if !config_path.exists() {
            return Err(Error::Config(format!(
                "Config file not found: {}",
                config_path.display()
            )));
        }

        let content = std::fs::read_to_string(config_path)?;
        let mut config: Config = toml::from_str(&content)?;

        // Set up paths based on config file location
        let base = config_path.parent().unwrap_or(Path::new(".")).to_path_buf();
        config.paths = PathsConfig {
            config_file: config_path.to_path_buf(),
            db_file: base.join("metadata.db"),
            base_dir: base,
        };

        config.validate()?;
        Ok(config)
    }

    /// Load configuration from default location
    pub fn load_default() -> Result<Self> {
        Self::load(&Self::default_config_path())
    }

    /// Load configuration from a specific base directory
    pub fn load_from(base_dir: Option<PathBuf>) -> Result<Self> {
        let mut config = Config::default();
        config.init_paths(base_dir);

        if config.paths.config_file.exists() {
            debug!("Loading config from {:?}", config.paths.config_file);
            let content = std::fs::read_to_string(&config.paths.config_file)?;
            let mut loaded: Config = toml::from_str(&content)?;
            loaded.paths = config.paths;
            config = loaded;
        } else {
            debug!("No config file found, using defaults");
        }

        Ok(config)
    }

    /// Save configuration to file
    pub fn save(&self) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.paths.config_file.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self)?;
        std::fs::write(&self.paths.config_file, content)?;
        info!("Saved config to {:?}", self.paths.config_file);
        Ok(())
    }

    /// Get the Qdrant API key from environment
    pub fn qdrant_api_key(&self) -> Option<String> {
        std::env::var(&self.qdrant_api_key_env).ok()
    }

    /// Check if librarian is initialized (config and DB exist)
    pub fn is_initialized(&self) -> bool {
        self.paths.config_file.exists() && self.paths.db_file.exists()
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.chunk.max_chars < self.chunk.min_chars {
            return Err(Error::Config(
                "chunk.max_chars must be >= chunk.min_chars".to_string(),
            ));
        }

        if self.chunk.overlap_chars >= self.chunk.max_chars {
            return Err(Error::Config(
                "chunk.overlap_chars must be < chunk.max_chars".to_string(),
            ));
        }

        if self.query.min_score < 0.0 || self.query.min_score > 1.0 {
            return Err(Error::Config(
                "query.min_score must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.query.bm25_weight < 0.0 || self.query.bm25_weight > 1.0 {
            return Err(Error::Config(
                "query.bm25_weight must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.crawl.rate_limit_per_host <= 0.0 {
            return Err(Error::Config(
                "crawl.rate_limit_per_host must be positive".to_string(),
            ));
        }

        if self.embedding.batch_size == 0 {
            return Err(Error::Config(
                "embedding.batch_size must be > 0".to_string(),
            ));
        }

        if self.embedding.backend.trim().is_empty() {
            return Err(Error::Config(
                "embedding.backend must not be empty".to_string(),
            ));
        }

        if self.embedding.url.trim().is_empty() {
            return Err(Error::Config(
                "embedding.url must not be empty".to_string(),
            ));
        }

        if let Some(dimension) = self.embedding.dimension {
            if dimension == 0 {
                return Err(Error::Config(
                    "embedding.dimension must be > 0".to_string(),
                ));
            }
        }

        if !self.embedding.allow_custom
            && (self.embedding.model.trim() == "custom"
                || embedding_model_spec(&self.embedding.model).is_none())
        {
            let allowed = allowlisted_embedding_models().join(", ");
            return Err(Error::Config(format!(
                "Embedding model '{}' is not allowlisted. Set embedding.allow_custom = true to use custom models. Allowed: {}",
                self.embedding.model, allowed
            )));
        }

        if self.embedding.multimodal {
            if let Some(caps) = embedding_model_capabilities(&self.embedding.model) {
                if !caps.supports_image {
                    let supported = supported_multimodal_embedding_models().join(", ");
                    return Err(Error::Config(format!(
                        "embedding.multimodal requires a multimodal embedding model. '{}' is not supported. Allowed: {}",
                        self.embedding.model, supported
                    )));
                }

                if caps.supports_multi_vector {
                    return Err(Error::Config(format!(
                        "Late-interaction embedding models like '{}' are not yet supported",
                        self.embedding.model
                    )));
                }
            } else if !self.embedding.allow_custom {
                return Err(Error::Config(
                    "embedding.multimodal is enabled for a non-allowlisted model. Set embedding.allow_custom = true and configure embedding.custom.*"
                        .to_string(),
                ));
            }
        }

        if self.reranker.enabled && reranker_model_spec(&self.reranker.model).is_none() {
            let allowed = allowlisted_reranker_models().join(", ");
            return Err(Error::Config(format!(
                "Reranker model '{}' is not allowlisted. Allowed: {}",
                self.reranker.model, allowed
            )));
        }

        // Multimodal validation
        if self.crawl.multimodal.enabled {
            if !self.embedding.multimodal {
                return Err(Error::Config(
                    "crawl.multimodal.enabled requires embedding.multimodal = true".to_string(),
                ));
            }

            if let Some(caps) = embedding_model_capabilities(&self.embedding.model) {
                if !caps.supports_image {
                    let supported = supported_multimodal_embedding_models().join(", ");
                    return Err(Error::Config(format!(
                        "crawl.multimodal.enabled requires a multimodal embedding model. '{}' is not supported. Allowed: {}",
                        self.embedding.model, supported
                    )));
                }

                if caps.strategy == MultimodalStrategy::LateInteraction {
                    return Err(Error::Config(format!(
                        "Late-interaction embedding models like '{}' are not yet supported for multimodal crawling",
                        self.embedding.model
                    )));
                }
            } else if !self.embedding.allow_custom {
                return Err(Error::Config(
                    "crawl.multimodal.enabled requires embedding.allow_custom = true for non-allowlisted models"
                        .to_string(),
                ));
            }

            if self.crawl.multimodal.include_audio || self.crawl.multimodal.include_video {
                return Err(Error::Config(
                    "Audio/video ingestion not supported yet. Disable include_audio/include_video"
                        .to_string(),
                ));
            }

            if self.crawl.multimodal.min_relevance_score < 0.0
                || self.crawl.multimodal.min_relevance_score > 1.0
            {
                return Err(Error::Config(
                    "crawl.multimodal.min_relevance_score must be between 0.0 and 1.0"
                        .to_string(),
                ));
            }

            if self.crawl.multimodal.min_asset_bytes > self.crawl.multimodal.max_asset_bytes {
                return Err(Error::Config(
                    "crawl.multimodal.min_asset_bytes must be <= max_asset_bytes".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Resolve embedding configuration against allowlist and backend probe
    pub async fn resolve_embedding_config(&self) -> Result<ResolvedEmbeddingConfig> {
        let raw_model = self.embedding.model.trim();
        if raw_model.is_empty() {
            return Err(Error::Config("embedding.model must not be empty".to_string()));
        }

        let allowlisted = embedding_model_spec(raw_model);
        let is_custom_model = raw_model == "custom" || allowlisted.is_none();

        if is_custom_model && !self.embedding.allow_custom {
            let allowed = allowlisted_embedding_models().join(", ");
            return Err(Error::Config(format!(
                "Embedding model '{}' is not allowlisted. Set embedding.allow_custom = true to use custom models. Allowed: {}",
                raw_model, allowed
            )));
        }

        let model_id = if raw_model == "custom" {
            let custom_id = self.embedding.custom.id.trim();
            if custom_id.is_empty() {
                return Err(Error::Config(
                    "embedding.custom.id must be set when embedding.model = 'custom'".to_string(),
                ));
            }
            custom_id
        } else {
            raw_model
        };

        if is_custom_model && self.embedding.custom.id.trim().is_empty() {
            return Err(Error::Config(
                "embedding.custom.id must be set when using a custom model".to_string(),
            ));
        }

        if is_custom_model && self.embedding.custom.id.trim() != model_id {
            return Err(Error::Config(format!(
                "embedding.custom.id '{}' must match resolved model id '{}'",
                self.embedding.custom.id, model_id
            )));
        }

        let (backend_kind, backend_url) = if is_custom_model {
            (
                EmbeddingBackendKind::from_str(&self.embedding.custom.backend)?,
                self.embedding.custom.url.trim(),
            )
        } else {
            (
                EmbeddingBackendKind::from_str(&self.embedding.backend)?,
                self.embedding.url.trim(),
            )
        };

        if backend_url.is_empty() {
            return Err(Error::Config(
                "embedding backend url must be set".to_string(),
            ));
        }

        let backend = EmbeddingBackendConfig {
            kind: backend_kind,
            url: backend_url.to_string(),
        };

        let client = EmbeddingBackendClient::new(&backend.url)?;

        if is_custom_model {
            let capabilities: BackendCapabilities = client.capabilities().await?;
            if !capabilities.models.is_empty()
                && !capabilities
                    .models
                    .iter()
                    .any(|model| model.id == model_id)
            {
                return Err(Error::Embedding(format!(
                    "Embedding backend does not advertise model '{}' in /capabilities",
                    model_id
                )));
            }
        }

        let wants_image_probe = self.embedding.multimodal
            || allowlisted.map(|spec| spec.capabilities.supports_image).unwrap_or(false)
            || (is_custom_model
                && self
                    .embedding
                    .custom
                    .modalities
                    .iter()
                    .any(|m| m == "image" || m == "image_text"));

        let probe = client
            .probe(
                model_id,
                PROBE_TEXT,
                wants_image_probe.then(|| PROBE_IMAGE_PNG_BASE64.to_string()),
                wants_image_probe.then(|| "image/png".to_string()),
            )
            .await?;

        if probe.id != model_id {
            return Err(Error::Embedding(format!(
                "Embedding backend probe returned model '{}' but expected '{}'",
                probe.id, model_id
            )));
        }

        let probe_family = probe.family.clone();
        let family = probe_family
            .or_else(|| allowlisted.map(|spec| spec.family.to_string()))
            .or_else(|| self.embedding.custom.family.clone())
            .unwrap_or_else(|| "custom".to_string());

        let probe_modalities = probe.modalities.clone();
        let modalities = if !probe_modalities.is_empty() {
            probe_modalities
        } else if let Some(spec) = allowlisted {
            spec.modalities.iter().map(|m| (*m).to_string()).collect()
        } else {
            self.embedding.custom.modalities.clone()
        };

        if modalities.is_empty() {
            return Err(Error::Config(
                "Embedding modalities could not be determined; set embedding.custom.modalities or ensure /probe returns modalities"
                    .to_string(),
            ));
        }

        let (supports_joint_inputs, supports_image, supports_text, strategy) = if let Some(spec) = allowlisted {
            (
                spec.capabilities.supports_joint_inputs,
                spec.capabilities.supports_image,
                spec.capabilities.supports_text,
                spec.capabilities.strategy,
            )
        } else {
            let supports_joint_inputs = modalities.iter().any(|m| m == "image_text");
            let supports_image = supports_joint_inputs || modalities.iter().any(|m| m == "image");
            let supports_text = supports_joint_inputs || modalities.iter().any(|m| m == "text");
            let strategy = if supports_joint_inputs {
                MultimodalStrategy::VlEmbedding
            } else if supports_image {
                MultimodalStrategy::DualEncoder
            } else {
                MultimodalStrategy::DualEncoder
            };
            (supports_joint_inputs, supports_image, supports_text, strategy)
        };

        if wants_image_probe {
            let has_joint = probe
                .joint_embeddings
                .as_ref()
                .map(|v| !v.is_empty())
                .unwrap_or(false);
            let has_image = probe
                .image_embeddings
                .as_ref()
                .map(|v| !v.is_empty())
                .unwrap_or(false);
            if supports_joint_inputs && !has_joint {
                return Err(Error::Embedding(format!(
                    "Embedding backend probe for '{}' did not return joint embeddings",
                    model_id
                )));
            }
            if !supports_joint_inputs && supports_image && !has_image {
                return Err(Error::Embedding(format!(
                    "Embedding backend probe for '{}' did not return image embeddings",
                    model_id
                )));
            }
        }

        let has_text = probe
            .text_embeddings
            .as_ref()
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        if !has_text && !supports_joint_inputs {
            return Err(Error::Embedding(format!(
                "Embedding backend probe for '{}' did not return text embeddings",
                model_id
            )));
        }

        let probe_dimension = probe.embedding_dim.or_else(|| {
            let mut dim: Option<usize> = None;
            for embeddings in [
                probe.text_embeddings.as_ref(),
                probe.image_embeddings.as_ref(),
                probe.joint_embeddings.as_ref(),
            ] {
                if let Some(values) = embeddings {
                    if let Some(first) = values.first() {
                        let candidate = first.len();
                        if values.iter().any(|vec| vec.len() != candidate) {
                            return None;
                        }
                        if let Some(existing) = dim {
                            if existing != candidate {
                                return None;
                            }
                        } else {
                            dim = Some(candidate);
                        }
                    }
                }
            }
            dim
        });

        if let Some(probe_dim) = probe.embedding_dim {
            for (label, embeddings) in [
                ("text", probe.text_embeddings.as_ref()),
                ("image", probe.image_embeddings.as_ref()),
                ("joint", probe.joint_embeddings.as_ref()),
            ] {
                if let Some(values) = embeddings {
                    if values.iter().any(|vec| vec.len() != probe_dim) {
                        return Err(Error::Embedding(format!(
                            "Embedding backend probe returned {} embeddings with inconsistent dimension for model '{}' (expected {})",
                            label, model_id, probe_dim
                        )));
                    }
                }
            }
        }

        if probe_dimension.is_none() && (has_text
            || probe
                .image_embeddings
                .as_ref()
                .map(|v| !v.is_empty())
                .unwrap_or(false)
            || probe
                .joint_embeddings
                .as_ref()
                .map(|v| !v.is_empty())
                .unwrap_or(false))
        {
            return Err(Error::Embedding(format!(
                "Embedding backend probe returned embeddings for '{}' but dimension could not be determined",
                model_id
            )));
        }
        let registry_dimension = allowlisted.and_then(|spec| spec.default_dimension);
        let custom_dimension = self.embedding.custom.dimension;

        if let (Some(config_dimension), Some(probe_dimension)) =
            (self.embedding.dimension, probe_dimension)
        {
            if config_dimension != probe_dimension {
                return Err(Error::Embedding(format!(
                    "Embedding dimension mismatch for model '{}' (family '{}', source config): config {} != probe {}",
                    model_id, family, config_dimension, probe_dimension
                )));
            }
        }

        let (dimension, dimension_source) = if let Some(config_dimension) = self.embedding.dimension {
            (config_dimension, EmbeddingDimensionSource::Config)
        } else if let Some(probe_dimension) = probe_dimension {
            (probe_dimension, EmbeddingDimensionSource::Probe)
        } else if let Some(registry_dimension) = registry_dimension {
            (registry_dimension, EmbeddingDimensionSource::Registry)
        } else if let Some(custom_dimension) = custom_dimension {
            (custom_dimension, EmbeddingDimensionSource::Custom)
        } else {
            return Err(Error::Config(
                "Embedding dimension could not be resolved; set embedding.dimension or ensure /probe returns a dimension"
                    .to_string(),
            ));
        };

        if let (Some(config_dimension), Some(registry_dimension)) =
            (self.embedding.dimension, registry_dimension)
        {
            if config_dimension != registry_dimension {
                warn!(
                    "Embedding dimension {} does not match registry default {} for model '{}'",
                    config_dimension, registry_dimension, model_id
                );
            }
        }

        let supports_multi_vector = probe
            .multivector
            .or_else(|| allowlisted.map(|spec| spec.capabilities.supports_multi_vector))
            .or(self.embedding.custom.multivector)
            .unwrap_or(false);

        let strategy = if supports_multi_vector {
            MultimodalStrategy::LateInteraction
        } else {
            strategy
        };

        let supports_mrl = probe
            .supports_mrl
            .or_else(|| allowlisted.map(|spec| spec.supports_mrl))
            .or(self.embedding.custom.supports_mrl)
            .unwrap_or(false);

        let max_batch = probe
            .max_batch
            .or_else(|| allowlisted.map(|spec| spec.max_batch))
            .or(self.embedding.custom.max_batch)
            .unwrap_or(self.embedding.batch_size.max(1));

        if supports_multi_vector || strategy == MultimodalStrategy::LateInteraction {
            return Err(Error::Embedding(format!(
                "Late-interaction embedding models like '{}' are not supported (family '{}')",
                model_id, family
            )));
        }

        if self.embedding.multimodal && !(supports_image || supports_joint_inputs) {
            return Err(Error::Embedding(format!(
                "embedding.multimodal is enabled, but model '{}' (family '{}') does not support image inputs",
                model_id, family
            )));
        }

        Ok(ResolvedEmbeddingConfig {
            model_id: model_id.to_string(),
            family,
            modalities,
            dimension,
            dimension_source,
            backend,
            strategy,
            supports_text,
            supports_image,
            supports_joint_inputs,
            supports_multi_vector: supports_multi_vector,
            supports_mrl,
            max_batch,
        })
    }
}

/// Get the database URL for sqlx
pub fn database_url(config: &Config) -> String {
    format!("sqlite://{}?mode=rwc", config.paths.db_file.display())
}

/// Render config TOML with defaults and irrelevant fields commented out.
pub fn render_config_toml(
    config: &Config,
    defaults: &Config,
    irrelevant: &HashSet<String>,
) -> String {
    let mut lines = Vec::new();

    lines.push("# librarian configuration".to_string());
    lines.push("".to_string());

    push_kv(
        &mut lines,
        "qdrant_url",
        toml_string(&config.qdrant_url),
        config.qdrant_url == defaults.qdrant_url,
        irrelevant.contains("qdrant_url"),
    );
    push_kv(
        &mut lines,
        "qdrant_api_key_env",
        toml_string(&config.qdrant_api_key_env),
        config.qdrant_api_key_env == defaults.qdrant_api_key_env,
        irrelevant.contains("qdrant_api_key_env"),
    );
    push_kv(
        &mut lines,
        "collection_name",
        toml_string(&config.collection_name),
        config.collection_name == defaults.collection_name,
        irrelevant.contains("collection_name"),
    );

    lines.push("".to_string());
    lines.push("[embedding]".to_string());
    push_kv(
        &mut lines,
        "model",
        toml_string(&config.embedding.model),
        config.embedding.model == defaults.embedding.model,
        irrelevant.contains("embedding.model"),
    );
    push_kv(
        &mut lines,
        "backend",
        toml_string(&config.embedding.backend),
        config.embedding.backend == defaults.embedding.backend,
        irrelevant.contains("embedding.backend"),
    );
    push_kv(
        &mut lines,
        "url",
        toml_string(&config.embedding.url),
        config.embedding.url == defaults.embedding.url,
        irrelevant.contains("embedding.url"),
    );
    push_kv(
        &mut lines,
        "allow_custom",
        toml_bool(config.embedding.allow_custom),
        config.embedding.allow_custom == defaults.embedding.allow_custom,
        irrelevant.contains("embedding.allow_custom"),
    );
    push_kv(
        &mut lines,
        "multimodal",
        toml_bool(config.embedding.multimodal),
        config.embedding.multimodal == defaults.embedding.multimodal,
        irrelevant.contains("embedding.multimodal"),
    );

    if let Some(dimension) = config.embedding.dimension.or_else(|| {
        embedding_model_spec(&config.embedding.model).and_then(|spec| spec.default_dimension)
    }) {
        let defaults_dimension = defaults.embedding.dimension.or_else(|| {
            embedding_model_spec(&defaults.embedding.model).and_then(|spec| spec.default_dimension)
        });
        push_kv(
            &mut lines,
            "dimension",
            toml_integer(dimension as i64),
            config.embedding.dimension == defaults.embedding.dimension
                && Some(dimension) == defaults_dimension,
            irrelevant.contains("embedding.dimension"),
        );
    } else {
        lines.push("# dimension = 0".to_string());
    }
    push_kv(
        &mut lines,
        "batch_size",
        toml_integer(config.embedding.batch_size as i64),
        config.embedding.batch_size == defaults.embedding.batch_size,
        irrelevant.contains("embedding.batch_size"),
    );

    lines.push("".to_string());
    lines.push("[embedding.custom]".to_string());
    push_kv(
        &mut lines,
        "id",
        toml_string(&config.embedding.custom.id),
        config.embedding.custom.id == defaults.embedding.custom.id,
        irrelevant.contains("embedding.custom.id"),
    );
    push_kv(
        &mut lines,
        "backend",
        toml_string(&config.embedding.custom.backend),
        config.embedding.custom.backend == defaults.embedding.custom.backend,
        irrelevant.contains("embedding.custom.backend"),
    );
    push_kv(
        &mut lines,
        "url",
        toml_string(&config.embedding.custom.url),
        config.embedding.custom.url == defaults.embedding.custom.url,
        irrelevant.contains("embedding.custom.url"),
    );
    push_kv(
        &mut lines,
        "family",
        toml_string(config.embedding.custom.family.as_deref().unwrap_or("")),
        config.embedding.custom.family == defaults.embedding.custom.family,
        irrelevant.contains("embedding.custom.family"),
    );
    push_kv(
        &mut lines,
        "modalities",
        toml_array(&config.embedding.custom.modalities),
        config.embedding.custom.modalities == defaults.embedding.custom.modalities,
        irrelevant.contains("embedding.custom.modalities"),
    );

    if let Some(custom_dimension) = config.embedding.custom.dimension {
        let default_custom_dimension = defaults.embedding.custom.dimension;
        push_kv(
            &mut lines,
            "dimension",
            toml_integer(custom_dimension as i64),
            config.embedding.custom.dimension == defaults.embedding.custom.dimension
                && Some(custom_dimension) == default_custom_dimension,
            irrelevant.contains("embedding.custom.dimension"),
        );
    } else {
        lines.push("# dimension = 0".to_string());
    }
    push_kv(
        &mut lines,
        "multivector",
        toml_bool(config.embedding.custom.multivector.unwrap_or(false)),
        config.embedding.custom.multivector == defaults.embedding.custom.multivector,
        irrelevant.contains("embedding.custom.multivector"),
    );
    push_kv(
        &mut lines,
        "supports_mrl",
        toml_bool(config.embedding.custom.supports_mrl.unwrap_or(false)),
        config.embedding.custom.supports_mrl == defaults.embedding.custom.supports_mrl,
        irrelevant.contains("embedding.custom.supports_mrl"),
    );
    let custom_max_batch_display = config
        .embedding
        .custom
        .max_batch
        .unwrap_or(config.embedding.batch_size);
    let default_custom_max_batch_display = defaults
        .embedding
        .custom
        .max_batch
        .unwrap_or(defaults.embedding.batch_size);
    push_kv(
        &mut lines,
        "max_batch",
        toml_integer(custom_max_batch_display as i64),
        config.embedding.custom.max_batch == defaults.embedding.custom.max_batch
            && custom_max_batch_display == default_custom_max_batch_display,
        irrelevant.contains("embedding.custom.max_batch"),
    );

    lines.push("".to_string());
    lines.push("[chunk]".to_string());
    push_kv(
        &mut lines,
        "max_chars",
        toml_integer(config.chunk.max_chars as i64),
        config.chunk.max_chars == defaults.chunk.max_chars,
        irrelevant.contains("chunk.max_chars"),
    );
    push_kv(
        &mut lines,
        "min_chars",
        toml_integer(config.chunk.min_chars as i64),
        config.chunk.min_chars == defaults.chunk.min_chars,
        irrelevant.contains("chunk.min_chars"),
    );
    push_kv(
        &mut lines,
        "overlap_chars",
        toml_integer(config.chunk.overlap_chars as i64),
        config.chunk.overlap_chars == defaults.chunk.overlap_chars,
        irrelevant.contains("chunk.overlap_chars"),
    );
    push_kv(
        &mut lines,
        "prefer_heading_boundaries",
        toml_bool(config.chunk.prefer_heading_boundaries),
        config.chunk.prefer_heading_boundaries == defaults.chunk.prefer_heading_boundaries,
        irrelevant.contains("chunk.prefer_heading_boundaries"),
    );

    lines.push("".to_string());
    lines.push("[query]".to_string());
    push_kv(
        &mut lines,
        "default_k",
        toml_integer(config.query.default_k as i64),
        config.query.default_k == defaults.query.default_k,
        irrelevant.contains("query.default_k"),
    );
    push_kv(
        &mut lines,
        "max_results",
        toml_integer(config.query.max_results as i64),
        config.query.max_results == defaults.query.max_results,
        irrelevant.contains("query.max_results"),
    );
    push_kv(
        &mut lines,
        "min_score",
        toml_float(config.query.min_score as f64),
        config.query.min_score == defaults.query.min_score,
        irrelevant.contains("query.min_score"),
    );
    push_kv(
        &mut lines,
        "hybrid_search",
        toml_bool(config.query.hybrid_search),
        config.query.hybrid_search == defaults.query.hybrid_search,
        irrelevant.contains("query.hybrid_search"),
    );
    push_kv(
        &mut lines,
        "bm25_weight",
        toml_float(config.query.bm25_weight as f64),
        config.query.bm25_weight == defaults.query.bm25_weight,
        irrelevant.contains("query.bm25_weight"),
    );

    lines.push("".to_string());
    lines.push("[reranker]".to_string());
    push_kv(
        &mut lines,
        "enabled",
        toml_bool(config.reranker.enabled),
        config.reranker.enabled == defaults.reranker.enabled,
        irrelevant.contains("reranker.enabled"),
    );
    push_kv(
        &mut lines,
        "model",
        toml_string(&config.reranker.model),
        config.reranker.model == defaults.reranker.model,
        irrelevant.contains("reranker.model"),
    );
    push_kv(
        &mut lines,
        "top_k",
        toml_integer(config.reranker.top_k as i64),
        config.reranker.top_k == defaults.reranker.top_k,
        irrelevant.contains("reranker.top_k"),
    );

    lines.push("".to_string());
    lines.push("[crawl]".to_string());
    push_kv(
        &mut lines,
        "max_depth",
        toml_integer(config.crawl.max_depth as i64),
        config.crawl.max_depth == defaults.crawl.max_depth,
        irrelevant.contains("crawl.max_depth"),
    );
    push_kv(
        &mut lines,
        "max_pages",
        toml_integer(config.crawl.max_pages as i64),
        config.crawl.max_pages == defaults.crawl.max_pages,
        irrelevant.contains("crawl.max_pages"),
    );
    push_kv(
        &mut lines,
        "allowed_domains",
        toml_array(&config.crawl.allowed_domains),
        config.crawl.allowed_domains == defaults.crawl.allowed_domains,
        irrelevant.contains("crawl.allowed_domains"),
    );
    push_kv(
        &mut lines,
        "path_prefix",
        toml_string(config.crawl.path_prefix.as_deref().unwrap_or("")),
        config.crawl.path_prefix == defaults.crawl.path_prefix,
        irrelevant.contains("crawl.path_prefix"),
    );
    push_kv(
        &mut lines,
        "rate_limit_per_host",
        toml_float(config.crawl.rate_limit_per_host),
        config.crawl.rate_limit_per_host == defaults.crawl.rate_limit_per_host,
        irrelevant.contains("crawl.rate_limit_per_host"),
    );
    push_kv(
        &mut lines,
        "user_agent",
        toml_string(&config.crawl.user_agent),
        config.crawl.user_agent == defaults.crawl.user_agent,
        irrelevant.contains("crawl.user_agent"),
    );
    push_kv(
        &mut lines,
        "timeout_secs",
        toml_integer(config.crawl.timeout_secs as i64),
        config.crawl.timeout_secs == defaults.crawl.timeout_secs,
        irrelevant.contains("crawl.timeout_secs"),
    );
    push_kv(
        &mut lines,
        "respect_robots_txt",
        toml_bool(config.crawl.respect_robots_txt),
        config.crawl.respect_robots_txt == defaults.crawl.respect_robots_txt,
        irrelevant.contains("crawl.respect_robots_txt"),
    );
    push_kv(
        &mut lines,
        "auto_js_rendering",
        toml_bool(config.crawl.auto_js_rendering),
        config.crawl.auto_js_rendering == defaults.crawl.auto_js_rendering,
        irrelevant.contains("crawl.auto_js_rendering"),
    );
    push_kv(
        &mut lines,
        "js_page_load_timeout_ms",
        toml_integer(config.crawl.js_page_load_timeout_ms as i64),
        config.crawl.js_page_load_timeout_ms == defaults.crawl.js_page_load_timeout_ms,
        irrelevant.contains("crawl.js_page_load_timeout_ms"),
    );
    push_kv(
        &mut lines,
        "js_render_wait_ms",
        toml_integer(config.crawl.js_render_wait_ms as i64),
        config.crawl.js_render_wait_ms == defaults.crawl.js_render_wait_ms,
        irrelevant.contains("crawl.js_render_wait_ms"),
    );
    push_kv(
        &mut lines,
        "js_no_sandbox",
        toml_bool(config.crawl.js_no_sandbox),
        config.crawl.js_no_sandbox == defaults.crawl.js_no_sandbox,
        irrelevant.contains("crawl.js_no_sandbox"),
    );

    lines.push("".to_string());
    lines.push("[crawl.multimodal]".to_string());
    push_kv(
        &mut lines,
        "enabled",
        toml_bool(config.crawl.multimodal.enabled),
        config.crawl.multimodal.enabled == defaults.crawl.multimodal.enabled,
        irrelevant.contains("crawl.multimodal.enabled"),
    );
    push_kv(
        &mut lines,
        "include_images",
        toml_bool(config.crawl.multimodal.include_images),
        config.crawl.multimodal.include_images == defaults.crawl.multimodal.include_images,
        irrelevant.contains("crawl.multimodal.include_images"),
    );
    push_kv(
        &mut lines,
        "include_audio",
        toml_bool(config.crawl.multimodal.include_audio),
        config.crawl.multimodal.include_audio == defaults.crawl.multimodal.include_audio,
        irrelevant.contains("crawl.multimodal.include_audio"),
    );
    push_kv(
        &mut lines,
        "include_video",
        toml_bool(config.crawl.multimodal.include_video),
        config.crawl.multimodal.include_video == defaults.crawl.multimodal.include_video,
        irrelevant.contains("crawl.multimodal.include_video"),
    );
    push_kv(
        &mut lines,
        "max_asset_bytes",
        toml_integer(config.crawl.multimodal.max_asset_bytes as i64),
        config.crawl.multimodal.max_asset_bytes == defaults.crawl.multimodal.max_asset_bytes,
        irrelevant.contains("crawl.multimodal.max_asset_bytes"),
    );
    push_kv(
        &mut lines,
        "min_asset_bytes",
        toml_integer(config.crawl.multimodal.min_asset_bytes as i64),
        config.crawl.multimodal.min_asset_bytes == defaults.crawl.multimodal.min_asset_bytes,
        irrelevant.contains("crawl.multimodal.min_asset_bytes"),
    );
    push_kv(
        &mut lines,
        "max_assets_per_page",
        toml_integer(config.crawl.multimodal.max_assets_per_page as i64),
        config.crawl.multimodal.max_assets_per_page
            == defaults.crawl.multimodal.max_assets_per_page,
        irrelevant.contains("crawl.multimodal.max_assets_per_page"),
    );
    push_kv(
        &mut lines,
        "allowed_mime_prefixes",
        toml_array(&config.crawl.multimodal.allowed_mime_prefixes),
        config.crawl.multimodal.allowed_mime_prefixes
            == defaults.crawl.multimodal.allowed_mime_prefixes,
        irrelevant.contains("crawl.multimodal.allowed_mime_prefixes"),
    );
    push_kv(
        &mut lines,
        "min_relevance_score",
        toml_float(config.crawl.multimodal.min_relevance_score as f64),
        config.crawl.multimodal.min_relevance_score
            == defaults.crawl.multimodal.min_relevance_score,
        irrelevant.contains("crawl.multimodal.min_relevance_score"),
    );
    push_kv(
        &mut lines,
        "include_css_background_images",
        toml_bool(config.crawl.multimodal.include_css_background_images),
        config.crawl.multimodal.include_css_background_images
            == defaults.crawl.multimodal.include_css_background_images,
        irrelevant.contains("crawl.multimodal.include_css_background_images"),
    );

    lines.join("\n") + "\n"
}

fn push_kv(lines: &mut Vec<String>, key: &str, value: String, is_default: bool, is_irrelevant: bool) {
    let prefix = if is_default || is_irrelevant { "# " } else { "" };
    lines.push(format!("{}{} = {}", prefix, key, value));
}

fn toml_string(value: &str) -> String {
    toml::Value::String(value.to_string()).to_string()
}

fn toml_integer(value: i64) -> String {
    toml::Value::Integer(value).to_string()
}

fn toml_float(value: f64) -> String {
    toml::Value::Float(value).to_string()
}

fn toml_bool(value: bool) -> String {
    toml::Value::Boolean(value).to_string()
}

fn toml_array(values: &[String]) -> String {
    let items = values
        .iter()
        .cloned()
        .map(toml::Value::String)
        .collect::<Vec<_>>();
    toml::Value::Array(items).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use tempfile::TempDir;
    use wiremock::{Mock, MockServer, ResponseTemplate};
    use wiremock::matchers::{method, path};
    use serde_json::json;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.qdrant_url, "http://127.0.0.1:6334");
        assert_eq!(config.collection_name, "librarian_docs");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_save_load() {
        let tmp = TempDir::new().unwrap();
        let mut config = Config::default();
        config.init_paths(Some(tmp.path().to_path_buf()));
        config.collection_name = "test_collection".to_string();

        config.save().unwrap();
        assert!(config.paths.config_file.exists());

        let loaded = Config::load_from(Some(tmp.path().to_path_buf())).unwrap();
        assert_eq!(loaded.collection_name, "test_collection");
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();

        // Invalid: overlap >= max
        config.chunk.overlap_chars = config.chunk.max_chars;
        assert!(config.validate().is_err());

        // Fix it
        config.chunk.overlap_chars = 100;
        assert!(config.validate().is_ok());

        // Invalid: min > max
        config.chunk.min_chars = config.chunk.max_chars + 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_allowlist_rejects_unknown_without_custom() {
        let mut config = Config::default();
        config.embedding.model = "custom-model".to_string();
        config.embedding.allow_custom = false;

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_allowlist_allows_unknown_with_custom_enabled() {
        let mut config = Config::default();
        config.embedding.model = "custom-model".to_string();
        config.embedding.allow_custom = true;

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_multimodal_validation_requires_embedding_support() {
        let mut config = Config::default();
        config.crawl.multimodal.enabled = true;
        config.crawl.multimodal.include_images = true;
        assert!(config.validate().is_err());

        // Enable model support, now validation should pass
        config.embedding.model = "jinaai/jina-clip-v2".to_string();
        config.embedding.multimodal = true;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_multimodal_validation_rejects_late_interaction() {
        let mut config = Config::default();
        config.crawl.multimodal.enabled = true;
        config.crawl.multimodal.include_images = true;
        config.embedding.multimodal = true;
        config.embedding.model = "vidore/colpali".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_multimodal_min_asset_bytes_validation() {
        let mut config = Config::default();
        config.crawl.multimodal.enabled = true;
        config.crawl.multimodal.include_images = true;
        config.embedding.model = "jinaai/jina-clip-v2".to_string();
        config.crawl.multimodal.min_asset_bytes = config.crawl.multimodal.max_asset_bytes + 1;

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_render_config_comments_defaults() {
        let config = Config::default();
        let defaults = Config::default();
        let irrelevant = HashSet::new();
        let rendered = render_config_toml(&config, &defaults, &irrelevant);

        assert!(rendered.contains("# qdrant_url ="));
        assert!(rendered.contains("[embedding]"));
        assert!(rendered.contains("# model = \"BAAI/bge-small-en-v1.5\""));
    }

    #[test]
    fn test_render_config_uncomments_custom_values() {
        let mut config = Config::default();
        config.collection_name = "custom_collection".to_string();
        let defaults = Config::default();
        let irrelevant = HashSet::new();
        let rendered = render_config_toml(&config, &defaults, &irrelevant);

        assert!(rendered.contains("collection_name = \"custom_collection\""));
        assert!(!rendered.contains("# collection_name = \"custom_collection\""));
    }

    #[test]
    fn test_render_config_comments_irrelevant_fields() {
        let config = Config::default();
        let defaults = Config::default();
        let mut irrelevant = HashSet::new();
        irrelevant.insert("reranker.model".to_string());
        let rendered = render_config_toml(&config, &defaults, &irrelevant);

        let reranker_section = rendered
            .split("[reranker]")
            .nth(1)
            .unwrap_or("");
        assert!(reranker_section.contains("# model = \"BAAI/bge-reranker-base\""));
    }

    #[tokio::test]
    async fn test_resolve_embedding_config_rejects_unadvertised_custom_model() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/capabilities"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"id": "other-model"}]
            })))
            .mount(&mock_server)
            .await;

        let mut config = Config::default();
        config.embedding.allow_custom = true;
        config.embedding.model = "custom".to_string();
        config.embedding.custom.id = "custom-model".to_string();
        config.embedding.custom.url = mock_server.uri();
        config.embedding.custom.modalities = vec!["text".to_string()];

        let err = config.resolve_embedding_config().await.unwrap_err();
        assert!(err
            .to_string()
            .contains("does not advertise model 'custom-model'"));
    }

    #[tokio::test]
    async fn test_resolve_embedding_config_rejects_dimension_mismatch() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/probe"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "BAAI/bge-small-en-v1.5",
                "embedding_dim": 768,
                "text_embeddings": [vec![0.0_f32; 768]]
            })))
            .mount(&mock_server)
            .await;

        let mut config = Config::default();
        config.embedding.url = mock_server.uri();
        config.embedding.dimension = Some(384);

        let err = config.resolve_embedding_config().await.unwrap_err();
        assert!(err
            .to_string()
            .contains("config 384 != probe 768"));
    }

    #[tokio::test]
    async fn test_resolve_embedding_config_requires_image_embeddings_for_custom_multimodal() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/capabilities"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"id": "custom-mm"}]
            })))
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/probe"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "custom-mm",
                "modalities": ["text", "image"],
                "embedding_dim": 64,
                "text_embeddings": [vec![0.0_f32; 64]]
            })))
            .mount(&mock_server)
            .await;

        let mut config = Config::default();
        config.embedding.allow_custom = true;
        config.embedding.model = "custom".to_string();
        config.embedding.multimodal = true;
        config.embedding.custom.id = "custom-mm".to_string();
        config.embedding.custom.url = mock_server.uri();
        config.embedding.custom.modalities = vec!["text".to_string(), "image".to_string()];

        let err = config.resolve_embedding_config().await.unwrap_err();
        assert!(err
            .to_string()
            .contains("did not return image embeddings"));
    }
}
