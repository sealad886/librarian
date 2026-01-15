//! Default values for configuration

/// Default Qdrant URL for local development
pub fn default_qdrant_url() -> String {
    std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6333".to_string())
}

/// Default environment variable name for Qdrant API key
pub fn default_qdrant_api_key_env() -> String {
    "QDRANT_API_KEY".to_string()
}

/// Default collection name
pub fn default_collection_name() -> String {
    "librarian_docs".to_string()
}

/// Default embedding model (BAAI/bge-small-en-v1.5)
pub fn default_embedding_model() -> String {
    "BAAI/bge-small-en-v1.5".to_string()
}

/// Default embedding dimension for bge-small-en-v1.5
pub fn default_embedding_dimension() -> usize {
    384
}

/// Default batch size for embedding
pub fn default_embedding_batch_size() -> usize {
    32
}

/// Default maximum characters per chunk
pub fn default_chunk_max_chars() -> usize {
    1500
}

/// Default minimum characters per chunk
pub fn default_chunk_min_chars() -> usize {
    100
}

/// Default overlap characters between chunks
pub fn default_chunk_overlap() -> usize {
    200
}

/// Default: prefer heading boundaries
pub fn default_prefer_heading_boundaries() -> bool {
    true
}

/// Default maximum crawl depth
pub fn default_crawl_max_depth() -> u32 {
    3
}

/// Default maximum pages per source
pub fn default_crawl_max_pages() -> u32 {
    1000
}

/// Default rate limit (requests per second per host)
pub fn default_crawl_rate_limit() -> f64 {
    2.0
}

/// Default user agent
pub fn default_crawl_user_agent() -> String {
    format!("librarian/{} (Documentation Indexer)", env!("CARGO_PKG_VERSION"))
}

/// Default request timeout in seconds
pub fn default_crawl_timeout() -> u64 {
    30
}

/// Default: respect robots.txt
pub fn default_respect_robots() -> bool {
    true
}

/// Default number of query results
pub fn default_query_k() -> usize {
    10
}

/// Default maximum query results
pub fn default_query_max_results() -> usize {
    100
}

/// Default minimum similarity score
pub fn default_query_min_score() -> f32 {
    0.0
}

/// Default BM25 weight for hybrid search
pub fn default_bm25_weight() -> f32 {
    0.3
}

/// Default reranker model (cross-encoder)
pub fn default_reranker_model() -> String {
    "BAAI/bge-reranker-base".to_string()
}

/// Default: reranker disabled
pub fn default_reranker_enabled() -> bool {
    false
}

/// Default number of results to return after reranking
pub fn default_reranker_top_k() -> usize {
    10
}
