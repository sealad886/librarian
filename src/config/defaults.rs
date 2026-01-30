//! Default values for configuration

/// Default Qdrant gRPC URL for local development (port 6334, not 6333 REST)
pub fn default_qdrant_url() -> String {
    std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6334".to_string())
}

/// Default environment variable name for Qdrant API key
pub fn default_qdrant_api_key_env() -> String {
    "".to_string()
}

/// Default collection name
pub fn default_collection_name() -> String {
    "librarian_docs".to_string()
}

/// Default embedding model (BAAI/bge-small-en-v1.5)
pub fn default_embedding_model() -> String {
    "BAAI/bge-small-en-v1.5".to_string()
}

/// Allow custom embedding models
pub fn default_embedding_allow_custom() -> bool {
    false
}

/// Enable multimodal embeddings
pub fn default_embedding_multimodal() -> bool {
    false
}

/// Default embedding backend kind (xinference for zero-config experience)
pub fn default_embedding_backend() -> String {
    "xinference".to_string()
}

/// Default embedding backend URL (used for xinference and http backends)
pub fn default_embedding_backend_url() -> String {
    std::env::var("LIBRARIAN_EMBEDDING_BACKEND_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:9997".to_string())
}

/// Default custom embedding backend kind
pub fn default_embedding_custom_backend() -> String {
    "http".to_string()
}

/// Default custom embedding backend URL
pub fn default_embedding_custom_url() -> String {
    std::env::var("LIBRARIAN_CUSTOM_EMBEDDING_BACKEND_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:7997".to_string())
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
    format!(
        "librarian/{} (Documentation Indexer)",
        env!("CARGO_PKG_VERSION")
    )
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

/// Default reranker backend kind (xinference for auto-managed)
pub fn default_reranker_backend() -> String {
    "xinference".to_string()
}

/// Default reranker backend URL
pub fn default_reranker_url() -> String {
    "http://localhost:9997".to_string()
}

/// Default number of results to return after reranking
pub fn default_reranker_top_k() -> usize {
    10
}

/// Default: enable automatic JS rendering for SPAs
pub fn default_auto_js_rendering() -> bool {
    true
}

/// Default page load timeout for JS rendering (30 seconds)
pub fn default_js_page_load_timeout() -> u64 {
    30000
}

/// Default wait time after page load for dynamic content (2 seconds)
pub fn default_js_render_wait() -> u64 {
    2000
}

/// Default: multimodal crawling disabled
pub fn default_multimodal_enabled() -> bool {
    false
}

/// Default: include images
pub fn default_multimodal_include_images() -> bool {
    true
}

/// Default: include audio disabled
pub fn default_multimodal_include_audio() -> bool {
    false
}

/// Default: include video disabled
pub fn default_multimodal_include_video() -> bool {
    false
}

/// Default: maximum asset bytes (5 MB)
pub fn default_multimodal_max_asset_bytes() -> usize {
    5_000_000
}

/// Default: minimum asset bytes (4 KB)
pub fn default_multimodal_min_asset_bytes() -> usize {
    4_096
}

/// Default: maximum assets per page
pub fn default_multimodal_max_assets_per_page() -> usize {
    10
}

/// Default: allowed MIME prefixes (images only)
pub fn default_multimodal_allowed_mime_prefixes() -> Vec<String> {
    vec!["image/".to_string()]
}

/// Default: minimum relevance score threshold
pub fn default_multimodal_min_relevance_score() -> f32 {
    0.6
}

/// Default: include CSS background images disabled
pub fn default_multimodal_include_css_background_images() -> bool {
    false
}

// =============================================================================
// Audio Processing Defaults
// =============================================================================

/// Default: maximum audio duration in seconds (10 minutes)
pub fn default_audio_max_duration_secs() -> u64 {
    600
}

/// Default: allowed audio MIME types
pub fn default_audio_allowed_mime_types() -> Vec<String> {
    vec![
        "audio/mpeg".to_string(),
        "audio/mp3".to_string(),
        "audio/wav".to_string(),
        "audio/x-wav".to_string(),
        "audio/ogg".to_string(),
        "audio/flac".to_string(),
        "audio/aac".to_string(),
        "audio/mp4".to_string(),
        "audio/webm".to_string(),
    ]
}

/// Default: transcription enabled
pub fn default_audio_transcription_enabled() -> bool {
    true
}

/// Default: transcription URL (local Whisper endpoint)
pub fn default_audio_transcription_url() -> String {
    "http://127.0.0.1:8000/v1/audio/transcriptions".to_string()
}

// =============================================================================
// Video Processing Defaults
// =============================================================================

/// Default: maximum video duration in seconds (5 minutes)
pub fn default_video_max_duration_secs() -> u64 {
    300
}

/// Default: keyframe extraction interval (every 10 seconds)
pub fn default_video_keyframe_interval_secs() -> f64 {
    10.0
}

/// Default: maximum keyframes per video
pub fn default_video_max_keyframes() -> usize {
    30
}

/// Default: allowed video MIME types
pub fn default_video_allowed_mime_types() -> Vec<String> {
    vec![
        "video/mp4".to_string(),
        "video/webm".to_string(),
        "video/ogg".to_string(),
        "video/quicktime".to_string(),
        "video/x-msvideo".to_string(),
        "video/x-matroska".to_string(),
    ]
}

/// Default: extract audio track from videos
pub fn default_video_extract_audio() -> bool {
    true
}
