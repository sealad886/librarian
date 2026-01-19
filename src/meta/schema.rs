//! SQLite schema definition

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

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(chunk_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_point ON chunks(qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_runs_source ON ingestion_runs(source_id);
"#;
