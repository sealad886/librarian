//! Configuration management for librarian
//!
//! Handles loading, saving, and validating configuration from TOML files.

mod defaults;

pub use defaults::*;

use crate::error::{Error, Result};
use crate::models::{
    embedding_model_capabilities, is_multimodal_embedding_model,
    supported_multimodal_embedding_models, MultimodalStrategy,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

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

/// Embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model name/identifier
    #[serde(default = "default_embedding_model")]
    pub model: String,

    /// Embedding dimension (must match model)
    #[serde(default = "default_embedding_dimension")]
    pub dimension: usize,

    /// Batch size for embedding
    #[serde(default = "default_embedding_batch_size")]
    pub batch_size: usize,
}

/// Lookup the expected embedding dimension for a known model
/// TODO: include all models that are recommended during init
pub fn embedding_dimension_for_model(model: &str) -> Option<usize> {
    match model {
        "BAAI/bge-small-en-v1.5" => Some(384),
        "BAAI/bge-base-en-v1.5" => Some(768),
        "BAAI/bge-large-en-v1.5" => Some(1024),
        "sentence-transformers/all-MiniLM-L6-v2" => Some(384),
        _ => None,
    }
}

impl EmbeddingConfig {
    /// Resolve the effective embedding dimension based on the configured model
    pub fn resolved_dimension(&self) -> usize {
        let configured = self.dimension;

        if let Some(expected) = embedding_dimension_for_model(&self.model) {
            if expected != configured {
                warn!(
                    "Embedding dimension {} does not match model '{}' ({}); using configured dimension {}",
                    configured, self.model, expected, configured
                );
            }
            return configured;
        }

        if cfg!(feature = "local-embed") {
            let fallback = embedding_dimension_for_model("BAAI/bge-small-en-v1.5").unwrap_or(384);
            if configured != fallback {
                warn!(
                    "Embedding model '{}' is not supported by the local embed backend; configured dimension {} will be used (fallback model uses {})",
                    self.model, configured, fallback
                );
            }
        }

        configured
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
            dimension: default_embedding_dimension(),
            batch_size: default_embedding_batch_size(),
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

        // Multimodal validation
        if self.crawl.multimodal.enabled {
            // Check if the embedding model supports multimodal via the model registry
            if !is_multimodal_embedding_model(&self.embedding.model) {
                let supported = supported_multimodal_embedding_models().join(", ");
                return Err(Error::Config(format!(
                    "crawl.multimodal.enabled requires a multimodal embedding model. '{}' is not supported. Allowed: {}",
                    self.embedding.model, supported
                )));
            }

            // Reject late-interaction strategy for multimodal (not yet supported)
            if let Some(caps) = embedding_model_capabilities(&self.embedding.model) {
                if caps.strategy == MultimodalStrategy::LateInteraction {
                    return Err(Error::Config(format!(
                        "Late-interaction embedding models like '{}' are not yet supported for multimodal crawling",
                        self.embedding.model
                    )));
                }
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
        "dimension",
        toml_integer(config.embedding.dimension as i64),
        config.embedding.dimension == defaults.embedding.dimension,
        irrelevant.contains("embedding.dimension"),
    );
    push_kv(
        &mut lines,
        "batch_size",
        toml_integer(config.embedding.batch_size as i64),
        config.embedding.batch_size == defaults.embedding.batch_size,
        irrelevant.contains("embedding.batch_size"),
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
    use tempfile::TempDir;
    use std::collections::HashSet;

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
    fn test_resolved_dimension_keeps_configured_value() {
        let mut config = Config::default();
        config.embedding.model = "BAAI/bge-base-en-v1.5".to_string();
        // Intentionally mismatched dimension to ensure configured value is preserved
        config.embedding.dimension = 384;

        assert_eq!(config.embedding.resolved_dimension(), 384);
    }

    #[test]
    fn test_resolved_dimension_unknown_model_keeps_configured_value() {
        let mut config = Config::default();
        config.embedding.model = "custom-model".to_string();
        config.embedding.dimension = 512;

        assert_eq!(config.embedding.resolved_dimension(), 512);
    }

    #[test]
    fn test_multimodal_validation_requires_embedding_support() {
        let mut config = Config::default();
        config.crawl.multimodal.enabled = true;
        config.crawl.multimodal.include_images = true;
        assert!(config.validate().is_err());

        // Enable model support, now validation should pass
        config.embedding.model = "jinaai/jina-clip-v2".to_string();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_multimodal_validation_rejects_late_interaction() {
        let mut config = Config::default();
        config.crawl.multimodal.enabled = true;
        config.crawl.multimodal.include_images = true;
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
}
