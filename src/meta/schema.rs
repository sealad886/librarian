//! SQLite schema definition

/// Current schema version - increment when adding migrations
pub const CURRENT_SCHEMA_VERSION: i32 = 2;

/// SQL schema for the metadata database
pub const SCHEMA_SQL: &str = r#"
-- Sources: registered ingestion sources
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    uri TEXT NOT NULL UNIQUE,
    name TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    config_json TEXT
);

-- Documents: individual files or pages
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id),
    uri TEXT NOT NULL,
    title TEXT,
    content_hash TEXT NOT NULL,
    content_type TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(source_id, uri)
);

-- Chunks: individual chunks with embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    headings_json TEXT,
    qdrant_point_id TEXT NOT NULL,
    modality TEXT NOT NULL DEFAULT 'text',
    media_url TEXT,
    media_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(doc_id, chunk_index)
);

-- Ingestion runs: tracking history
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id),
    operation TEXT NOT NULL DEFAULT 'ingest',
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    docs_processed INTEGER DEFAULT 0,
    chunks_created INTEGER DEFAULT 0,
    chunks_updated INTEGER DEFAULT 0,
    chunks_deleted INTEGER DEFAULT 0,
    errors_json TEXT
);

-- Schema version tracking for migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- Track collection configuration for consistency checks
CREATE TABLE IF NOT EXISTS collection_config (
    collection_name TEXT PRIMARY KEY,
    vector_dimension INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_family TEXT NOT NULL,
    distance_metric TEXT NOT NULL DEFAULT 'cosine',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    verified_at TEXT
);

-- Document links: cross-references between documents
CREATE TABLE IF NOT EXISTS document_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    to_uri TEXT NOT NULL,
    to_doc_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
    link_text TEXT,
    link_type TEXT NOT NULL DEFAULT 'href',
    created_at TEXT NOT NULL
);

-- Query log: analytics for search queries
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    intent TEXT,
    source_filter TEXT,
    result_count INTEGER NOT NULL DEFAULT 0,
    top_score REAL,
    latency_ms INTEGER,
    created_at TEXT NOT NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(chunk_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_point ON chunks(qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_runs_source ON ingestion_runs(source_id);
CREATE INDEX IF NOT EXISTS idx_links_from ON document_links(from_doc_id);
CREATE INDEX IF NOT EXISTS idx_links_to_doc ON document_links(to_doc_id);
CREATE INDEX IF NOT EXISTS idx_links_to_uri ON document_links(to_uri);
CREATE INDEX IF NOT EXISTS idx_query_log_created ON query_log(created_at);
"#;
