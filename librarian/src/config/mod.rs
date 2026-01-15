//! Configuration management for librarian
//!
//! Handles loading, saving, and validating configuration from TOML files.

mod defaults;

pub use defaults::*;

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

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
        let base = config_path
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf();
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

        Ok(())
    }
}

/// Get the database URL for sqlx
pub fn database_url(config: &Config) -> String {
    format!("sqlite://{}?mode=rwc", config.paths.db_file.display())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

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
}
