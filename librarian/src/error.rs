//! Custom error types for librarian

use thiserror::Error;

/// Main error type for librarian operations
#[derive(Error, Debug)]
pub enum Error {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Qdrant error: {0}")]
    Qdrant(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Crawl error: {0}")]
    Crawl(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("URL parse error: {0}")]
    UrlParse(#[from] url::ParseError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("TOML parse error: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("TOML serialize error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    #[error("Source not found: {0}")]
    SourceNotFound(String),

    #[error("Document not found: {0}")]
    DocumentNotFound(String),

    #[error("Not initialized: run 'librarian init' first")]
    NotInitialized,

    #[error("Already initialized at {0}")]
    AlreadyInitialized(String),

    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Rate limited: {0}")]
    RateLimited(String),

    #[error("Robots.txt disallowed: {0}")]
    RobotsDisallowed(String),

    #[error("Max depth exceeded")]
    MaxDepthExceeded,

    #[error("Max pages exceeded")]
    MaxPagesExceeded,

    #[error("Unsupported content type: {0}")]
    UnsupportedContentType(String),

    #[error("MCP protocol error: {0}")]
    McpProtocol(String),

    #[error("{0}")]
    Other(String),
}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Error::Other(err.to_string())
    }
}

/// Result type alias for librarian
pub type Result<T> = std::result::Result<T, Error>;

/// Convert qdrant errors
impl From<qdrant_client::QdrantError> for Error {
    fn from(err: qdrant_client::QdrantError) -> Self {
        Error::Qdrant(err.to_string())
    }
}
