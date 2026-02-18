//! Metadata storage using SQLite
//!
//! This module handles all local metadata storage including:
//! - Sources (registered ingestion sources)
//! - Documents (individual files/pages)
//! - Chunks (embedded text chunks)
//! - Ingestion runs (history and stats)

mod schema;

pub use schema::*;

use crate::config::Config;
use crate::error::{Error, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use sqlx::FromRow;
use std::str::FromStr;
use tracing::debug;
use uuid::Uuid;

/// Source types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    Dir,
    Url,
    Sitemap,
}

impl std::fmt::Display for SourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceType::Dir => write!(f, "dir"),
            SourceType::Url => write!(f, "url"),
            SourceType::Sitemap => write!(f, "sitemap"),
        }
    }
}

impl FromStr for SourceType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "dir" => Ok(SourceType::Dir),
            "url" => Ok(SourceType::Url),
            "sitemap" => Ok(SourceType::Sitemap),
            _ => Err(Error::Config(format!("Unknown source type: {}", s))),
        }
    }
}

/// Ingestion run status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunOperation {
    Ingest,
    Update,
    Reindex,
}

impl std::fmt::Display for RunStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunStatus::Running => write!(f, "running"),
            RunStatus::Completed => write!(f, "completed"),
            RunStatus::Failed => write!(f, "failed"),
        }
    }
}

impl std::fmt::Display for RunOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunOperation::Ingest => write!(f, "ingest"),
            RunOperation::Update => write!(f, "update"),
            RunOperation::Reindex => write!(f, "reindex"),
        }
    }
}

impl FromStr for RunStatus {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "running" => Ok(RunStatus::Running),
            "completed" => Ok(RunStatus::Completed),
            "failed" => Ok(RunStatus::Failed),
            _ => Err(Error::Config(format!("Unknown run status: {}", s))),
        }
    }
}

impl FromStr for RunOperation {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "ingest" => Ok(RunOperation::Ingest),
            "update" => Ok(RunOperation::Update),
            "reindex" => Ok(RunOperation::Reindex),
            _ => Err(Error::Config(format!("Unknown run operation: {}", s))),
        }
    }
}

/// A registered ingestion source
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct Source {
    pub id: String,
    pub source_type: String,
    pub uri: String,
    pub name: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub config_json: Option<String>,
}

impl Source {
    pub fn new(source_type: SourceType, uri: String, name: Option<String>) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            id: Uuid::new_v4().to_string(),
            source_type: source_type.to_string(),
            uri,
            name,
            created_at: now.clone(),
            updated_at: now,
            config_json: None,
        }
    }

    pub fn get_type(&self) -> Result<SourceType> {
        self.source_type.parse()
    }
}

/// A document (file or web page)
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub source_id: String,
    pub uri: String,
    pub title: Option<String>,
    pub content_hash: String,
    pub content_type: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

impl Document {
    pub fn new(source_id: String, uri: String, content_hash: String) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            id: Uuid::new_v4().to_string(),
            source_id,
            uri,
            title: None,
            content_hash,
            content_type: None,
            created_at: now.clone(),
            updated_at: now,
        }
    }
}

/// A text chunk
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub doc_id: String,
    pub chunk_index: i32,
    pub chunk_hash: String,
    pub chunk_text: String,
    pub char_start: i32,
    pub char_end: i32,
    pub headings_json: Option<String>,
    pub qdrant_point_id: String,
    pub modality: String,
    pub media_url: Option<String>,
    pub media_hash: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

impl Chunk {
    pub fn new(
        doc_id: String,
        chunk_index: i32,
        chunk_hash: String,
        chunk_text: String,
        char_start: i32,
        char_end: i32,
        headings: Option<Vec<String>>,
    ) -> Self {
        let now = Utc::now().to_rfc3339();
        // Use chunk_hash to derive stable Qdrant point ID
        let point_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, chunk_hash.as_bytes()).to_string();

        Self {
            id: Uuid::new_v4().to_string(),
            doc_id,
            chunk_index,
            chunk_hash,
            chunk_text,
            char_start,
            char_end,
            headings_json: headings.map(|h| serde_json::to_string(&h).unwrap_or_default()),
            qdrant_point_id: point_id,
            modality: "text".to_string(),
            media_url: None,
            media_hash: None,
            created_at: now.clone(),
            updated_at: now,
        }
    }

    pub fn new_media(
        doc_id: String,
        chunk_index: i32,
        chunk_hash: String,
        chunk_text: String,
        media_url: String,
        media_hash: Option<String>,
    ) -> Self {
        let now = Utc::now().to_rfc3339();
        let point_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, chunk_hash.as_bytes()).to_string();

        let char_end = chunk_text.len() as i32;

        Self {
            id: Uuid::new_v4().to_string(),
            doc_id,
            chunk_index,
            chunk_hash,
            chunk_text,
            char_start: 0,
            char_end,
            headings_json: None,
            qdrant_point_id: point_id,
            modality: "image".to_string(),
            media_url: Some(media_url),
            media_hash,
            created_at: now.clone(),
            updated_at: now,
        }
    }

    /// Create a new chunk with a specific modality (audio, video, etc.)
    pub fn new_with_modality(
        doc_id: String,
        chunk_index: i32,
        chunk_hash: String,
        chunk_text: String,
        modality: &str,
        media_url: Option<String>,
        media_hash: Option<String>,
    ) -> Self {
        let now = Utc::now().to_rfc3339();
        let point_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, chunk_hash.as_bytes()).to_string();

        let char_end = chunk_text.len() as i32;

        Self {
            id: Uuid::new_v4().to_string(),
            doc_id,
            chunk_index,
            chunk_hash,
            chunk_text,
            char_start: 0,
            char_end,
            headings_json: None,
            qdrant_point_id: point_id,
            modality: modality.to_string(),
            media_url,
            media_hash,
            created_at: now.clone(),
            updated_at: now,
        }
    }

    pub fn headings(&self) -> Vec<String> {
        self.headings_json
            .as_ref()
            .and_then(|j| serde_json::from_str(j).ok())
            .unwrap_or_default()
    }

    /// Return the canonical Qdrant point UUID for this chunk.
    pub fn point_uuid(&self) -> Uuid {
        Uuid::try_parse(&self.qdrant_point_id)
            .unwrap_or_else(|_| Uuid::new_v5(&Uuid::NAMESPACE_OID, self.qdrant_point_id.as_bytes()))
    }
}

/// An ingestion run record
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct IngestionRun {
    pub id: String,
    pub source_id: String,
    pub operation: String,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub status: String,
    pub docs_processed: i32,
    pub chunks_created: i32,
    pub chunks_updated: i32,
    pub chunks_deleted: i32,
    pub errors_json: Option<String>,
}

impl IngestionRun {
    pub fn new(source_id: String, operation: RunOperation) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_id,
            operation: operation.to_string(),
            started_at: Utc::now().to_rfc3339(),
            completed_at: None,
            status: RunStatus::Running.to_string(),
            docs_processed: 0,
            chunks_created: 0,
            chunks_updated: 0,
            chunks_deleted: 0,
            errors_json: None,
        }
    }
}

/// Collection configuration record for tracking embedding model/dimension consistency
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct CollectionConfigRecord {
    pub collection_name: String,
    pub vector_dimension: i32,
    pub embedding_model: String,
    pub embedding_family: String,
    pub distance_metric: String,
    pub created_at: String,
    pub updated_at: String,
    pub verified_at: Option<String>,
}

/// A cross-reference link between documents discovered during parsing.
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct DocumentLink {
    pub id: i64,
    pub from_doc_id: String,
    pub to_uri: String,
    pub to_doc_id: Option<String>,
    pub link_text: Option<String>,
    pub link_type: String,
    pub created_at: String,
}

/// A query log entry for analytics.
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct QueryLogEntry {
    pub id: i64,
    pub query_text: String,
    pub intent: Option<String>,
    pub source_filter: Option<String>,
    pub result_count: i32,
    pub top_score: Option<f64>,
    pub latency_ms: Option<i64>,
    pub created_at: String,
}

/// Aggregate query analytics for a time period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnalytics {
    pub total_queries: i64,
    pub avg_top_score: Option<f64>,
    pub avg_result_count: Option<f64>,
    pub avg_latency_ms: Option<f64>,
    pub zero_result_queries: i64,
    pub period_days: i32,
}

/// Metadata database handle
#[derive(Clone)]
pub struct MetaDb {
    pool: SqlitePool,
}

impl MetaDb {
    /// Connect to the metadata database
    pub async fn connect(config: &Config) -> Result<Self> {
        let db_path = &config.paths.db_file;

        // Create parent directory if needed
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let options = SqliteConnectOptions::new()
            .filename(db_path)
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
            .synchronous(sqlx::sqlite::SqliteSynchronous::Normal);

        debug!("Connecting to SQLite database at {:?}", db_path);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;

        Ok(Self { pool })
    }

    /// Initialize the database schema
    pub async fn init_schema(&self) -> Result<()> {
        debug!("Initializing database schema");
        sqlx::query(SCHEMA_SQL).execute(&self.pool).await?;

        // Backfill optional columns for existing installations
        // Add ingestion_runs.operation if missing
        let has_operation: Option<(i32,)> = sqlx::query_as(
            "SELECT 1 FROM pragma_table_info('ingestion_runs') WHERE name='operation'",
        )
        .fetch_optional(&self.pool)
        .await?;

        if has_operation.is_none() {
            sqlx::query(
                "ALTER TABLE ingestion_runs ADD COLUMN operation TEXT NOT NULL DEFAULT 'ingest'",
            )
            .execute(&self.pool)
            .await?;
        }

        let has_modality: Option<(i32,)> =
            sqlx::query_as("SELECT 1 FROM pragma_table_info('chunks') WHERE name='modality'")
                .fetch_optional(&self.pool)
                .await?;

        if has_modality.is_none() {
            sqlx::query("ALTER TABLE chunks ADD COLUMN modality TEXT NOT NULL DEFAULT 'text'")
                .execute(&self.pool)
                .await?;
        }

        let has_media_url: Option<(i32,)> =
            sqlx::query_as("SELECT 1 FROM pragma_table_info('chunks') WHERE name='media_url'")
                .fetch_optional(&self.pool)
                .await?;

        if has_media_url.is_none() {
            sqlx::query("ALTER TABLE chunks ADD COLUMN media_url TEXT")
                .execute(&self.pool)
                .await?;
        }

        let has_media_hash: Option<(i32,)> =
            sqlx::query_as("SELECT 1 FROM pragma_table_info('chunks') WHERE name='media_hash'")
                .fetch_optional(&self.pool)
                .await?;

        if has_media_hash.is_none() {
            sqlx::query("ALTER TABLE chunks ADD COLUMN media_hash TEXT")
                .execute(&self.pool)
                .await?;
        }

        // Run migrations for new schema features
        self.run_migrations().await?;

        Ok(())
    }

    // ===== Schema Migration Infrastructure =====

    /// Get current schema version, returns 0 if schema_version table doesn't exist
    pub async fn get_schema_version(&self) -> Result<i32> {
        // Check if schema_version table exists
        let exists: Option<(i32,)> = sqlx::query_as(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'",
        )
        .fetch_optional(&self.pool)
        .await?;

        if exists.is_none() {
            return Ok(0);
        }

        let version: Option<(i32,)> = sqlx::query_as("SELECT MAX(version) FROM schema_version")
            .fetch_optional(&self.pool)
            .await?;

        Ok(version.map(|v| v.0).unwrap_or(0))
    }

    /// Run all pending migrations
    pub async fn run_migrations(&self) -> Result<()> {
        let current = self.get_schema_version().await?;

        if current < 1 {
            self.apply_migration_v1().await?;
        }

        if current < 2 {
            self.apply_migration_v2().await?;
        }

        Ok(())
    }

    /// Migration v1: Add schema_version and collection_config tables
    async fn apply_migration_v1(&self) -> Result<()> {
        debug!("Applying migration v1: collection config tracking");

        // Create schema_version if not exists
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        // Create collection_config if not exists
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS collection_config (
                collection_name TEXT PRIMARY KEY,
                vector_dimension INTEGER NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_family TEXT NOT NULL,
                distance_metric TEXT NOT NULL DEFAULT 'cosine',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                verified_at TEXT
            )",
        )
        .execute(&self.pool)
        .await?;

        // Record migration
        let now = Utc::now().to_rfc3339();
        sqlx::query("INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (1, ?)")
            .bind(&now)
            .execute(&self.pool)
            .await?;

        debug!("Migration v1 applied successfully");
        Ok(())
    }

    /// Migration v2: Add document_links and query_log tables
    async fn apply_migration_v2(&self) -> Result<()> {
        debug!("Applying migration v2: document links and query analytics");

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS document_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                to_uri TEXT NOT NULL,
                to_doc_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
                link_text TEXT,
                link_type TEXT NOT NULL DEFAULT 'href',
                created_at TEXT NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_links_from ON document_links(from_doc_id)")
            .execute(&self.pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_links_to_doc ON document_links(to_doc_id)")
            .execute(&self.pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_links_to_uri ON document_links(to_uri)")
            .execute(&self.pool)
            .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                intent TEXT,
                source_filter TEXT,
                result_count INTEGER NOT NULL DEFAULT 0,
                top_score REAL,
                latency_ms INTEGER,
                created_at TEXT NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_query_log_created ON query_log(created_at)")
            .execute(&self.pool)
            .await?;

        let now = Utc::now().to_rfc3339();
        sqlx::query("INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (2, ?)")
            .bind(&now)
            .execute(&self.pool)
            .await?;

        debug!("Migration v2 applied successfully");
        Ok(())
    }

    // ===== Collection Config Operations =====

    /// Get collection configuration by name
    pub async fn get_collection_config(
        &self,
        collection_name: &str,
    ) -> Result<Option<CollectionConfigRecord>> {
        let record = sqlx::query_as::<_, CollectionConfigRecord>(
            "SELECT * FROM collection_config WHERE collection_name = ?",
        )
        .bind(collection_name)
        .fetch_optional(&self.pool)
        .await?;
        Ok(record)
    }

    /// Upsert collection configuration
    pub async fn upsert_collection_config(&self, config: &CollectionConfigRecord) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        debug!(
            "Upserting collection config for {} at {}",
            config.collection_name, now
        );
        sqlx::query(
            "INSERT INTO collection_config 
             (collection_name, vector_dimension, embedding_model, embedding_family, distance_metric, created_at, updated_at, verified_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(collection_name) DO UPDATE SET
             vector_dimension = excluded.vector_dimension,
             embedding_model = excluded.embedding_model,
             embedding_family = excluded.embedding_family,
             distance_metric = excluded.distance_metric,
             updated_at = excluded.updated_at,
             verified_at = excluded.verified_at",
        )
        .bind(&config.collection_name)
        .bind(config.vector_dimension)
        .bind(&config.embedding_model)
        .bind(&config.embedding_family)
        .bind(&config.distance_metric)
        .bind(&config.created_at)
        .bind(&now)
        .bind(&config.verified_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Update verification timestamp for a collection
    pub async fn update_collection_verified_at(&self, collection_name: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query("UPDATE collection_config SET verified_at = ?, updated_at = ? WHERE collection_name = ?")
            .bind(&now)
            .bind(&now)
            .bind(collection_name)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    // ===== Document Link Operations =====

    /// Store document links extracted during parsing.
    ///
    /// Replaces all existing links for the given document (delete + insert).
    ///
    /// # Arguments
    /// * `doc_id` - The source document ID
    /// * `links` - Extracted links from parsing
    ///
    /// # Returns
    /// Number of links stored (excludes empty URLs)
    ///
    /// # Errors
    /// Returns error if database operations fail
    pub async fn store_document_links(
        &self,
        doc_id: &str,
        links: &[crate::parse::ExtractedLink],
    ) -> Result<usize> {
        sqlx::query("DELETE FROM document_links WHERE from_doc_id = ?")
            .bind(doc_id)
            .execute(&self.pool)
            .await?;

        let now = Utc::now().to_rfc3339();
        let mut count = 0;
        for link in links {
            if link.url.is_empty() {
                continue;
            }
            let link_type = if link.is_internal {
                "internal"
            } else {
                "external"
            };
            sqlx::query(
                "INSERT INTO document_links (from_doc_id, to_uri, link_text, link_type, created_at)
                 VALUES (?, ?, ?, ?, ?)",
            )
            .bind(doc_id)
            .bind(&link.url)
            .bind(link.text.as_deref())
            .bind(link_type)
            .bind(&now)
            .execute(&self.pool)
            .await?;
            count += 1;
        }
        Ok(count)
    }

    /// Resolve document links by matching `to_uri` against known document URIs.
    ///
    /// Called after ingestion to populate the `to_doc_id` column for links
    /// whose target URI matches an existing document.
    ///
    /// # Arguments
    /// * `source_id` - Restrict resolution to documents within this source
    ///
    /// # Returns
    /// Number of links resolved
    ///
    /// # Errors
    /// Returns error if the update query fails
    pub async fn resolve_document_links(&self, source_id: &str) -> Result<u64> {
        let result = sqlx::query(
            "UPDATE document_links SET to_doc_id = d.id
             FROM documents d
             WHERE document_links.to_uri = d.uri
             AND document_links.to_doc_id IS NULL
             AND d.source_id = ?",
        )
        .bind(source_id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    /// Get outgoing links from a document.
    ///
    /// # Arguments
    /// * `doc_id` - The document to get links from
    ///
    /// # Returns
    /// All links originating from this document
    ///
    /// # Errors
    /// Returns error if the query fails
    pub async fn get_outgoing_links(&self, doc_id: &str) -> Result<Vec<DocumentLink>> {
        let links = sqlx::query_as::<_, DocumentLink>(
            "SELECT * FROM document_links WHERE from_doc_id = ? ORDER BY id",
        )
        .bind(doc_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(links)
    }

    /// Get incoming links to a document (documents that link to this one).
    ///
    /// # Arguments
    /// * `doc_id` - The target document
    ///
    /// # Returns
    /// All links pointing to this document
    ///
    /// # Errors
    /// Returns error if the query fails
    pub async fn get_incoming_links(&self, doc_id: &str) -> Result<Vec<DocumentLink>> {
        let links = sqlx::query_as::<_, DocumentLink>(
            "SELECT * FROM document_links WHERE to_doc_id = ? ORDER BY id",
        )
        .bind(doc_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(links)
    }

    /// Get related documents (linked from or to this document).
    ///
    /// # Arguments
    /// * `doc_id` - The document to find relations for
    ///
    /// # Returns
    /// Distinct documents linked to or from this document
    ///
    /// # Errors
    /// Returns error if the query fails
    pub async fn get_related_documents(&self, doc_id: &str) -> Result<Vec<Document>> {
        let docs = sqlx::query_as::<_, Document>(
            "SELECT DISTINCT d.* FROM documents d
             WHERE d.id IN (
                 SELECT to_doc_id FROM document_links WHERE from_doc_id = ? AND to_doc_id IS NOT NULL
                 UNION
                 SELECT from_doc_id FROM document_links WHERE to_doc_id = ?
             )",
        )
        .bind(doc_id)
        .bind(doc_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(docs)
    }

    // ===== Query Log Operations =====

    /// Log a search query for analytics.
    ///
    /// # Arguments
    /// * `query_text` - The search query string
    /// * `intent` - Optional detected intent (e.g. "factual", "comparative")
    /// * `source_filter` - Optional source filter applied
    /// * `result_count` - Number of results returned
    /// * `top_score` - Score of the top result, if any
    /// * `latency_ms` - Query latency in milliseconds
    ///
    /// # Errors
    /// Returns error if the insert fails
    pub async fn log_query(
        &self,
        query_text: &str,
        intent: Option<&str>,
        source_filter: Option<&str>,
        result_count: i32,
        top_score: Option<f32>,
        latency_ms: Option<i64>,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO query_log (query_text, intent, source_filter, result_count, top_score, latency_ms, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(query_text)
        .bind(intent)
        .bind(source_filter)
        .bind(result_count)
        .bind(top_score.map(|s| s as f64))
        .bind(latency_ms)
        .bind(&now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Get recent query log entries.
    ///
    /// # Arguments
    /// * `limit` - Maximum number of entries to return
    ///
    /// # Returns
    /// Most recent query log entries, ordered by creation time descending
    ///
    /// # Errors
    /// Returns error if the query fails
    pub async fn get_recent_queries(&self, limit: i64) -> Result<Vec<QueryLogEntry>> {
        let entries = sqlx::query_as::<_, QueryLogEntry>(
            "SELECT * FROM query_log ORDER BY created_at DESC LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        Ok(entries)
    }

    /// Get query analytics summary for a time period.
    ///
    /// # Arguments
    /// * `days` - Number of days to look back
    ///
    /// # Returns
    /// Aggregate statistics including total queries, averages, and zero-result count
    ///
    /// # Errors
    /// Returns error if any aggregate query fails
    pub async fn get_query_analytics(&self, days: i32) -> Result<QueryAnalytics> {
        let since = (Utc::now() - chrono::Duration::days(days as i64)).to_rfc3339();

        let total: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM query_log WHERE created_at >= ?")
            .bind(&since)
            .fetch_one(&self.pool)
            .await?;

        let avg_score: (Option<f64>,) = sqlx::query_as(
            "SELECT AVG(top_score) FROM query_log WHERE created_at >= ? AND top_score IS NOT NULL",
        )
        .bind(&since)
        .fetch_one(&self.pool)
        .await?;

        let avg_results: (Option<f64>,) =
            sqlx::query_as("SELECT AVG(result_count) FROM query_log WHERE created_at >= ?")
                .bind(&since)
                .fetch_one(&self.pool)
                .await?;

        let avg_latency: (Option<f64>,) = sqlx::query_as(
            "SELECT AVG(latency_ms) FROM query_log WHERE created_at >= ? AND latency_ms IS NOT NULL",
        )
        .bind(&since)
        .fetch_one(&self.pool)
        .await?;

        let zero_result: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM query_log WHERE created_at >= ? AND result_count = 0",
        )
        .bind(&since)
        .fetch_one(&self.pool)
        .await?;

        Ok(QueryAnalytics {
            total_queries: total.0,
            avg_top_score: avg_score.0,
            avg_result_count: avg_results.0,
            avg_latency_ms: avg_latency.0,
            zero_result_queries: zero_result.0,
            period_days: days,
        })
    }

    /// Check if database is initialized
    pub async fn is_initialized(&self) -> Result<bool> {
        let result: Option<(i32,)> =
            sqlx::query_as("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sources'")
                .fetch_optional(&self.pool)
                .await?;
        Ok(result.is_some())
    }

    // ===== Source Operations =====

    /// Insert a new source
    pub async fn insert_source(&self, source: &Source) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO sources (id, source_type, uri, name, created_at, updated_at, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&source.id)
        .bind(&source.source_type)
        .bind(&source.uri)
        .bind(&source.name)
        .bind(&source.created_at)
        .bind(&source.updated_at)
        .bind(&source.config_json)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Get source by ID
    pub async fn get_source(&self, id: &str) -> Result<Option<Source>> {
        let source = sqlx::query_as::<_, Source>("SELECT * FROM sources WHERE id = ?")
            .bind(id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(source)
    }

    /// Get source by URI
    pub async fn get_source_by_uri(&self, uri: &str) -> Result<Option<Source>> {
        let source = sqlx::query_as::<_, Source>("SELECT * FROM sources WHERE uri = ?")
            .bind(uri)
            .fetch_optional(&self.pool)
            .await?;
        Ok(source)
    }

    /// Get source by name (case-sensitive match)
    pub async fn get_source_by_name(&self, name: &str) -> Result<Option<Source>> {
        let source = sqlx::query_as::<_, Source>("SELECT * FROM sources WHERE name = ?")
            .bind(name)
            .fetch_optional(&self.pool)
            .await?;
        Ok(source)
    }

    /// List all sources
    pub async fn list_sources(&self) -> Result<Vec<Source>> {
        let sources = sqlx::query_as::<_, Source>("SELECT * FROM sources ORDER BY created_at DESC")
            .fetch_all(&self.pool)
            .await?;
        Ok(sources)
    }

    /// Resolve sources by optional source ID list.
    ///
    /// When `source_ids` is `None`, returns all sources.
    /// Missing IDs are ignored.
    pub async fn resolve_sources(&self, source_ids: Option<&[String]>) -> Result<Vec<Source>> {
        match source_ids {
            Some(ids) => {
                let mut sources = Vec::new();
                for id in ids {
                    if let Some(source) = self.get_source(id).await? {
                        sources.push(source);
                    }
                }
                Ok(sources)
            }
            None => self.list_sources().await,
        }
    }

    /// Delete a source and all its documents/chunks
    pub async fn delete_source(&self, id: &str) -> Result<()> {
        // Delete chunks first (cascade)
        sqlx::query(
            "DELETE FROM chunks WHERE doc_id IN (SELECT id FROM documents WHERE source_id = ?)",
        )
        .bind(id)
        .execute(&self.pool)
        .await?;

        // Delete documents
        sqlx::query("DELETE FROM documents WHERE source_id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;

        // Delete ingestion runs
        sqlx::query("DELETE FROM ingestion_runs WHERE source_id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;

        // Delete source
        sqlx::query("DELETE FROM sources WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Update source name
    pub async fn update_source_name(&self, id: &str, new_name: Option<String>) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE sources SET name = ?, updated_at = ? WHERE id = ?
            "#,
        )
        .bind(new_name)
        .bind(Utc::now().to_rfc3339())
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ===== Document Operations =====

    /// Insert or update a document, returning the stored document with canonical ID.
    ///
    /// On conflict (same source_id + uri), updates the existing row but preserves
    /// the original document ID. Callers MUST use the returned document's ID
    /// for any subsequent operations (e.g., chunk writes) to avoid FK violations.
    pub async fn upsert_document(&self, doc: &Document) -> Result<Document> {
        sqlx::query(
            r#"
            INSERT INTO documents (id, source_id, uri, title, content_hash, content_type, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, uri) DO UPDATE SET
                title = excluded.title,
                content_hash = excluded.content_hash,
                content_type = excluded.content_type,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(&doc.id)
        .bind(&doc.source_id)
        .bind(&doc.uri)
        .bind(&doc.title)
        .bind(&doc.content_hash)
        .bind(&doc.content_type)
        .bind(&doc.created_at)
        .bind(&doc.updated_at)
        .execute(&self.pool)
        .await?;

        let stored = self
            .get_document_by_uri(&doc.source_id, &doc.uri)
            .await?
            .ok_or_else(|| Error::DocumentNotFound(doc.uri.clone()))?;

        debug!(
            doc_id = %stored.id,
            source_id = %stored.source_id,
            uri = %stored.uri,
            "Persisted document after upsert"
        );

        Ok(stored)
    }

    /// Get document by ID
    pub async fn get_document(&self, id: &str) -> Result<Option<Document>> {
        let doc = sqlx::query_as::<_, Document>("SELECT * FROM documents WHERE id = ?")
            .bind(id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(doc)
    }

    /// Get document by source and URI
    pub async fn get_document_by_uri(
        &self,
        source_id: &str,
        uri: &str,
    ) -> Result<Option<Document>> {
        let doc = sqlx::query_as::<_, Document>(
            "SELECT * FROM documents WHERE source_id = ? AND uri = ?",
        )
        .bind(source_id)
        .bind(uri)
        .fetch_optional(&self.pool)
        .await?;
        Ok(doc)
    }

    /// List documents for a source
    pub async fn list_documents(&self, source_id: &str) -> Result<Vec<Document>> {
        let docs = sqlx::query_as::<_, Document>(
            "SELECT * FROM documents WHERE source_id = ? ORDER BY uri",
        )
        .bind(source_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(docs)
    }

    /// Delete documents not in the given URI list
    pub async fn delete_stale_documents(
        &self,
        source_id: &str,
        current_uris: &[String],
    ) -> Result<Vec<String>> {
        // Get stale doc IDs first
        let placeholders = current_uris
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(",");
        let query = if current_uris.is_empty() {
            "SELECT id FROM documents WHERE source_id = ?".to_string()
        } else {
            format!(
                "SELECT id FROM documents WHERE source_id = ? AND uri NOT IN ({})",
                placeholders
            )
        };

        let mut query_builder = sqlx::query_scalar::<_, String>(&query).bind(source_id);
        for uri in current_uris {
            query_builder = query_builder.bind(uri);
        }
        let stale_ids: Vec<String> = query_builder.fetch_all(&self.pool).await?;

        // Delete chunks for stale docs
        for id in &stale_ids {
            sqlx::query("DELETE FROM chunks WHERE doc_id = ?")
                .bind(id)
                .execute(&self.pool)
                .await?;
            sqlx::query("DELETE FROM documents WHERE id = ?")
                .bind(id)
                .execute(&self.pool)
                .await?;
        }

        Ok(stale_ids)
    }

    // ===== Chunk Operations =====

    /// Insert or update a chunk
    pub async fn upsert_chunk(&self, chunk: &Chunk) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO chunks (id, doc_id, chunk_index, chunk_hash, chunk_text, char_start, char_end, headings_json, qdrant_point_id, modality, media_url, media_hash, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id, chunk_index) DO UPDATE SET
                chunk_hash = excluded.chunk_hash,
                chunk_text = excluded.chunk_text,
                char_start = excluded.char_start,
                char_end = excluded.char_end,
                headings_json = excluded.headings_json,
                qdrant_point_id = excluded.qdrant_point_id,
                modality = excluded.modality,
                media_url = excluded.media_url,
                media_hash = excluded.media_hash,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(&chunk.id)
        .bind(&chunk.doc_id)
        .bind(chunk.chunk_index)
        .bind(&chunk.chunk_hash)
        .bind(&chunk.chunk_text)
        .bind(chunk.char_start)
        .bind(chunk.char_end)
        .bind(&chunk.headings_json)
        .bind(&chunk.qdrant_point_id)
        .bind(&chunk.modality)
        .bind(&chunk.media_url)
        .bind(&chunk.media_hash)
        .bind(&chunk.created_at)
        .bind(&chunk.updated_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Get chunks for a document
    pub async fn get_chunks(&self, doc_id: &str) -> Result<Vec<Chunk>> {
        let chunks = sqlx::query_as::<_, Chunk>(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
        )
        .bind(doc_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(chunks)
    }

    /// Get chunks for a document filtered by modality
    pub async fn get_chunks_by_modality(&self, doc_id: &str, modality: &str) -> Result<Vec<Chunk>> {
        let chunks = sqlx::query_as::<_, Chunk>(
            "SELECT * FROM chunks WHERE doc_id = ? AND modality = ? ORDER BY chunk_index",
        )
        .bind(doc_id)
        .bind(modality)
        .fetch_all(&self.pool)
        .await?;
        Ok(chunks)
    }

    /// Get chunk by Qdrant point ID
    pub async fn get_chunk_by_point_id(&self, point_id: &str) -> Result<Option<Chunk>> {
        let chunk = sqlx::query_as::<_, Chunk>("SELECT * FROM chunks WHERE qdrant_point_id = ?")
            .bind(point_id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(chunk)
    }

    /// Get chunk by document ID and content hash
    pub async fn get_chunk_by_hash(
        &self,
        doc_id: &str,
        content_hash: &str,
    ) -> Result<Option<Chunk>> {
        let chunk =
            sqlx::query_as::<_, Chunk>("SELECT * FROM chunks WHERE doc_id = ? AND chunk_hash = ?")
                .bind(doc_id)
                .bind(content_hash)
                .fetch_optional(&self.pool)
                .await?;
        Ok(chunk)
    }

    /// Delete chunks with index >= given value
    pub async fn delete_chunks_from_index(
        &self,
        doc_id: &str,
        from_index: i32,
    ) -> Result<Vec<String>> {
        let point_ids: Vec<String> = sqlx::query_scalar(
            "SELECT qdrant_point_id FROM chunks WHERE doc_id = ? AND chunk_index >= ? AND (modality IS NULL OR modality = 'text')",
        )
        .bind(doc_id)
        .bind(from_index)
        .fetch_all(&self.pool)
        .await?;

        sqlx::query("DELETE FROM chunks WHERE doc_id = ? AND chunk_index >= ? AND (modality IS NULL OR modality = 'text')")
            .bind(doc_id)
            .bind(from_index)
            .execute(&self.pool)
            .await?;

        Ok(point_ids)
    }

    /// Delete chunks for a document filtered by modality
    pub async fn delete_chunks_by_modality(
        &self,
        doc_id: &str,
        modality: &str,
    ) -> Result<Vec<String>> {
        let point_ids: Vec<String> = sqlx::query_scalar(
            "SELECT qdrant_point_id FROM chunks WHERE doc_id = ? AND modality = ?",
        )
        .bind(doc_id)
        .bind(modality)
        .fetch_all(&self.pool)
        .await?;

        sqlx::query("DELETE FROM chunks WHERE doc_id = ? AND modality = ?")
            .bind(doc_id)
            .bind(modality)
            .execute(&self.pool)
            .await?;

        Ok(point_ids)
    }

    /// Get all Qdrant point IDs for a source
    pub async fn get_source_point_ids(&self, source_id: &str) -> Result<Vec<String>> {
        let ids: Vec<String> = sqlx::query_scalar(
            r#"
            SELECT c.qdrant_point_id 
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            WHERE d.source_id = ?
            "#,
        )
        .bind(source_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(ids)
    }

    // ===== Ingestion Run Operations =====

    /// Start a new ingestion run
    pub async fn start_ingestion_run(
        &self,
        source_id: &str,
        operation: RunOperation,
    ) -> Result<IngestionRun> {
        let run = IngestionRun::new(source_id.to_string(), operation);
        sqlx::query(
            r#"
            INSERT INTO ingestion_runs (id, source_id, operation, started_at, status, docs_processed, chunks_created, chunks_updated, chunks_deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&run.id)
        .bind(&run.source_id)
        .bind(&run.operation)
        .bind(&run.started_at)
        .bind(&run.status)
        .bind(run.docs_processed)
        .bind(run.chunks_created)
        .bind(run.chunks_updated)
        .bind(run.chunks_deleted)
        .execute(&self.pool)
        .await?;
        Ok(run)
    }

    /// Complete an ingestion run
    #[allow(clippy::too_many_arguments)]
    pub async fn complete_ingestion_run(
        &self,
        id: &str,
        status: RunStatus,
        docs_processed: i32,
        chunks_created: i32,
        chunks_updated: i32,
        chunks_deleted: i32,
        errors: Option<Vec<String>>,
    ) -> Result<()> {
        let errors_json = errors.map(|e| serde_json::to_string(&e).unwrap_or_default());
        sqlx::query(
            r#"
            UPDATE ingestion_runs SET
                completed_at = ?,
                status = ?,
                docs_processed = ?,
                chunks_created = ?,
                chunks_updated = ?,
                chunks_deleted = ?,
                errors_json = ?
            WHERE id = ?
            "#,
        )
        .bind(Utc::now().to_rfc3339())
        .bind(status.to_string())
        .bind(docs_processed)
        .bind(chunks_created)
        .bind(chunks_updated)
        .bind(chunks_deleted)
        .bind(errors_json)
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Get latest ingestion run for a source
    pub async fn get_latest_run(&self, source_id: &str) -> Result<Option<IngestionRun>> {
        let run = sqlx::query_as::<_, IngestionRun>(
            "SELECT * FROM ingestion_runs WHERE source_id = ? ORDER BY started_at DESC LIMIT 1",
        )
        .bind(source_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(run)
    }

    // ===== Statistics =====

    /// Get source statistics
    pub async fn get_source_stats(&self, source_id: &str) -> Result<SourceStats> {
        let doc_count: i32 =
            sqlx::query_scalar("SELECT COUNT(*) FROM documents WHERE source_id = ?")
                .bind(source_id)
                .fetch_one(&self.pool)
                .await?;

        let chunk_count: i32 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*) FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            WHERE d.source_id = ?
            "#,
        )
        .bind(source_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(SourceStats {
            document_count: doc_count as usize,
            chunk_count: chunk_count as usize,
        })
    }

    /// Get global statistics
    pub async fn get_global_stats(&self) -> Result<GlobalStats> {
        let source_count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM sources")
            .fetch_one(&self.pool)
            .await?;

        let doc_count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM documents")
            .fetch_one(&self.pool)
            .await?;

        let chunk_count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM chunks")
            .fetch_one(&self.pool)
            .await?;

        Ok(GlobalStats {
            source_count: source_count as usize,
            document_count: doc_count as usize,
            chunk_count: chunk_count as usize,
        })
    }

    /// Create database with path directly (without full config)
    pub async fn new(db_path: &std::path::Path) -> Result<Self> {
        // Create parent directory if needed
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let options = SqliteConnectOptions::new()
            .filename(db_path)
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
            .synchronous(sqlx::sqlite::SqliteSynchronous::Normal);

        debug!("Connecting to SQLite database at {:?}", db_path);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;

        let db = Self { pool };

        // Auto-initialize schema if needed
        if !db.is_initialized().await? {
            db.init_schema().await?;
        }

        Ok(db)
    }

    /// List documents for a source (alias for list_documents)
    pub async fn list_source_documents(&self, source_id: &str) -> Result<Vec<Document>> {
        self.list_documents(source_id).await
    }

    /// List chunks for a document (alias for get_chunks)
    pub async fn list_document_chunks(&self, doc_id: &str) -> Result<Vec<ChunkRecord>> {
        let chunks = sqlx::query_as::<_, Chunk>(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
        )
        .bind(doc_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(chunks
            .into_iter()
            .map(|c| ChunkRecord {
                id: c.qdrant_point_id,
                text: c.chunk_text,
                chunk_index: c.chunk_index,
                content_hash: c.chunk_hash,
                headings: c.headings_json,
                modality: c.modality,
                media_url: c.media_url,
                media_hash: c.media_hash,
            })
            .collect())
    }

    /// Delete a document and its chunks
    pub async fn delete_document(&self, doc_id: &str) -> Result<()> {
        sqlx::query("DELETE FROM chunks WHERE doc_id = ?")
            .bind(doc_id)
            .execute(&self.pool)
            .await?;

        sqlx::query("DELETE FROM documents WHERE id = ?")
            .bind(doc_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// List all chunk IDs (for orphan detection)
    pub async fn list_all_chunk_ids(&self) -> Result<Vec<String>> {
        let ids: Vec<String> = sqlx::query_scalar("SELECT qdrant_point_id FROM chunks")
            .fetch_all(&self.pool)
            .await?;
        Ok(ids)
    }

    /// Retrieve corpus-level statistics for BM25 IDF computation.
    ///
    /// Returns the total number of distinct documents that have chunks and,
    /// for each query term, the number of distinct documents whose chunk text
    /// contains that term (case-insensitive via LIKE).
    pub async fn get_corpus_stats(&self, terms: &[String]) -> Result<crate::rank::CorpusStats> {
        let total_docs: i32 = sqlx::query_scalar("SELECT COUNT(DISTINCT doc_id) FROM chunks")
            .fetch_one(&self.pool)
            .await?;

        let mut doc_frequencies = std::collections::HashMap::new();
        for term in terms {
            let pattern = format!("%{}%", term);
            let count: i32 = sqlx::query_scalar(
                "SELECT COUNT(DISTINCT doc_id) FROM chunks WHERE chunk_text LIKE ?",
            )
            .bind(&pattern)
            .fetch_one(&self.pool)
            .await?;
            doc_frequencies.insert(term.to_lowercase(), count as usize);
        }

        Ok(crate::rank::CorpusStats {
            total_docs: total_docs as usize,
            doc_frequencies,
        })
    }
}

/// A simplified chunk record for API use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub id: String,
    pub text: String,
    pub chunk_index: i32,
    pub content_hash: String,
    pub headings: Option<String>,
    pub modality: String,
    pub media_url: Option<String>,
    pub media_hash: Option<String>,
}

impl ChunkRecord {
    /// Return the canonical Qdrant point UUID for this chunk record.
    pub fn point_uuid(&self) -> Uuid {
        Uuid::try_parse(&self.id)
            .unwrap_or_else(|_| Uuid::new_v5(&Uuid::NAMESPACE_OID, self.id.as_bytes()))
    }
}

/// Statistics for a single source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceStats {
    pub document_count: usize,
    pub chunk_count: usize,
}

/// Global statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalStats {
    pub source_count: usize,
    pub document_count: usize,
    pub chunk_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup_test_db() -> (MetaDb, TempDir) {
        let tmp = TempDir::new().unwrap();
        let mut config = Config::default();
        config.paths.db_file = tmp.path().join("test.db");

        let db = MetaDb::connect(&config).await.unwrap();
        db.init_schema().await.unwrap();
        (db, tmp)
    }

    #[tokio::test]
    async fn test_source_crud() {
        let (db, _tmp) = setup_test_db().await;

        let source = Source::new(
            SourceType::Dir,
            "/path/to/docs".to_string(),
            Some("Test Docs".to_string()),
        );
        db.insert_source(&source).await.unwrap();

        let loaded = db.get_source(&source.id).await.unwrap().unwrap();
        assert_eq!(loaded.uri, "/path/to/docs");
        assert_eq!(loaded.name, Some("Test Docs".to_string()));

        let sources = db.list_sources().await.unwrap();
        assert_eq!(sources.len(), 1);

        db.delete_source(&source.id).await.unwrap();
        let sources = db.list_sources().await.unwrap();
        assert_eq!(sources.len(), 0);
    }

    #[tokio::test]
    async fn test_document_upsert() {
        let (db, _tmp) = setup_test_db().await;

        let source = Source::new(SourceType::Dir, "/docs".to_string(), None);
        db.insert_source(&source).await.unwrap();

        let mut doc = Document::new(
            source.id.clone(),
            "/docs/file.md".to_string(),
            "hash1".to_string(),
        );
        doc.title = Some("Test File".to_string());
        let stored = db.upsert_document(&doc).await.unwrap();

        let loaded = db
            .get_document_by_uri(&source.id, "/docs/file.md")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.content_hash, "hash1");
        assert_eq!(stored.id, loaded.id);

        // Update the document with a new Document instance (simulates re-ingest)
        let mut doc2 = Document::new(
            source.id.clone(),
            "/docs/file.md".to_string(),
            "hash2".to_string(),
        );
        doc2.title = Some("Updated File".to_string());
        let stored2 = db.upsert_document(&doc2).await.unwrap();

        // The returned doc should have the ORIGINAL id, not the new one
        assert_eq!(stored2.id, stored.id);
        assert_ne!(stored2.id, doc2.id);
        assert_eq!(stored2.content_hash, "hash2");

        let loaded = db
            .get_document_by_uri(&source.id, "/docs/file.md")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.content_hash, "hash2");
        assert_eq!(loaded.id, stored.id);
    }

    #[tokio::test]
    async fn test_chunk_operations() {
        let (db, _tmp) = setup_test_db().await;

        let source = Source::new(SourceType::Dir, "/docs".to_string(), None);
        db.insert_source(&source).await.unwrap();

        let doc = Document::new(
            source.id.clone(),
            "/docs/file.md".to_string(),
            "hash1".to_string(),
        );
        let doc = db.upsert_document(&doc).await.unwrap();

        let chunk1 = Chunk::new(
            doc.id.clone(),
            0,
            "chunk_hash_1".to_string(),
            "First chunk text".to_string(),
            0,
            15,
            Some(vec!["Introduction".to_string()]),
        );
        let chunk2 = Chunk::new(
            doc.id.clone(),
            1,
            "chunk_hash_2".to_string(),
            "Second chunk text".to_string(),
            16,
            32,
            None,
        );

        db.upsert_chunk(&chunk1).await.unwrap();
        db.upsert_chunk(&chunk2).await.unwrap();

        let chunks = db.get_chunks(&doc.id).await.unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].headings(), vec!["Introduction".to_string()]);
    }

    /// Regression test: re-ingesting an existing document with new chunks must use the canonical
    /// document ID, otherwise the FK constraint on chunks.doc_id -> documents.id fails.
    #[tokio::test]
    async fn test_reingest_document_uses_canonical_id_for_chunks() {
        let (db, _tmp) = setup_test_db().await;

        let source = Source::new(SourceType::Dir, "/docs".to_string(), None);
        db.insert_source(&source).await.unwrap();

        // First ingest: create document and chunk
        let doc1 = Document::new(
            source.id.clone(),
            "/docs/file.md".to_string(),
            "hash_v1".to_string(),
        );
        let stored1 = db.upsert_document(&doc1).await.unwrap();
        let original_doc_id = stored1.id.clone();

        let chunk1 = Chunk::new(
            stored1.id.clone(),
            0,
            "chunk_hash_v1".to_string(),
            "Version 1 content".to_string(),
            0,
            17,
            None,
        );
        db.upsert_chunk(&chunk1).await.unwrap();

        // Simulate re-ingest: new Document instance with new UUID but same (source_id, uri)
        let doc2 = Document::new(
            source.id.clone(),
            "/docs/file.md".to_string(),
            "hash_v2".to_string(),
        );
        // This is a different UUID
        assert_ne!(doc2.id, original_doc_id);

        // upsert_document MUST return the canonical (original) document ID
        let stored2 = db.upsert_document(&doc2).await.unwrap();
        assert_eq!(
            stored2.id, original_doc_id,
            "upsert_document must return canonical doc ID"
        );

        // Now we can safely create a chunk using the returned ID
        let chunk2 = Chunk::new(
            stored2.id.clone(),
            1,
            "chunk_hash_v2".to_string(),
            "Version 2 content".to_string(),
            0,
            17,
            None,
        );
        // This should NOT fail with FK violation
        db.upsert_chunk(&chunk2).await.unwrap();

        // Verify both chunks exist under the same document
        let chunks = db.get_chunks(&original_doc_id).await.unwrap();
        assert_eq!(chunks.len(), 2);
    }

    // ===== Migration and Collection Config Tests =====

    #[tokio::test]
    async fn test_schema_version_fresh_db() {
        let (db, _tmp) = setup_test_db().await;
        let version = db.get_schema_version().await.unwrap();
        // After init_schema, we should be at version 2
        assert_eq!(version, 2);
    }

    #[tokio::test]
    async fn test_migrations_idempotent() {
        let (db, _tmp) = setup_test_db().await;
        // Run migrations multiple times - should not fail
        db.run_migrations().await.unwrap();
        db.run_migrations().await.unwrap();
        let version = db.get_schema_version().await.unwrap();
        assert_eq!(version, 2);
    }

    #[tokio::test]
    async fn test_collection_config_crud() {
        let (db, _tmp) = setup_test_db().await;

        // Initially no config
        let config = db.get_collection_config("test_collection").await.unwrap();
        assert!(config.is_none());

        // Insert config
        let record = CollectionConfigRecord {
            collection_name: "test_collection".to_string(),
            vector_dimension: 384,
            embedding_model: "test-model".to_string(),
            embedding_family: "test-family".to_string(),
            distance_metric: "cosine".to_string(),
            created_at: "2026-01-30T00:00:00Z".to_string(),
            updated_at: "2026-01-30T00:00:00Z".to_string(),
            verified_at: None,
        };
        db.upsert_collection_config(&record).await.unwrap();

        // Retrieve config
        let config = db.get_collection_config("test_collection").await.unwrap();
        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.vector_dimension, 384);
        assert_eq!(config.embedding_model, "test-model");
        assert_eq!(config.embedding_family, "test-family");
        assert!(config.verified_at.is_none());

        // Update verification timestamp
        db.update_collection_verified_at("test_collection")
            .await
            .unwrap();
        let config = db
            .get_collection_config("test_collection")
            .await
            .unwrap()
            .unwrap();
        assert!(config.verified_at.is_some());
    }

    #[tokio::test]
    async fn test_collection_config_upsert_updates_existing() {
        let (db, _tmp) = setup_test_db().await;

        // Insert initial config
        let record1 = CollectionConfigRecord {
            collection_name: "test_collection".to_string(),
            vector_dimension: 384,
            embedding_model: "model-v1".to_string(),
            embedding_family: "family-v1".to_string(),
            distance_metric: "cosine".to_string(),
            created_at: "2026-01-30T00:00:00Z".to_string(),
            updated_at: "2026-01-30T00:00:00Z".to_string(),
            verified_at: None,
        };
        db.upsert_collection_config(&record1).await.unwrap();

        // Upsert with updated values
        let record2 = CollectionConfigRecord {
            collection_name: "test_collection".to_string(),
            vector_dimension: 768,
            embedding_model: "model-v2".to_string(),
            embedding_family: "family-v2".to_string(),
            distance_metric: "cosine".to_string(),
            created_at: "2026-01-30T00:00:00Z".to_string(),
            updated_at: "2026-01-30T01:00:00Z".to_string(),
            verified_at: Some("2026-01-30T01:00:00Z".to_string()),
        };
        db.upsert_collection_config(&record2).await.unwrap();

        // Verify update occurred
        let config = db
            .get_collection_config("test_collection")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(config.vector_dimension, 768);
        assert_eq!(config.embedding_model, "model-v2");
        assert_eq!(config.embedding_family, "family-v2");
    }

    #[tokio::test]
    async fn test_document_links_crud() {
        let (db, _tmp) = setup_test_db().await;

        let source = Source::new(SourceType::Dir, "/docs".to_string(), None);
        db.insert_source(&source).await.unwrap();

        let doc1 = Document::new(
            source.id.clone(),
            "/docs/a.md".to_string(),
            "h1".to_string(),
        );
        let doc1 = db.upsert_document(&doc1).await.unwrap();

        let doc2 = Document::new(
            source.id.clone(),
            "/docs/b.md".to_string(),
            "h2".to_string(),
        );
        let doc2 = db.upsert_document(&doc2).await.unwrap();

        // Store links from doc1 → doc2
        use crate::parse::ExtractedLink;
        let links = vec![
            ExtractedLink {
                url: "/docs/b.md".to_string(),
                text: Some("See B".to_string()),
                is_internal: true,
            },
            ExtractedLink {
                url: "https://external.com".to_string(),
                text: Some("External".to_string()),
                is_internal: false,
            },
        ];

        let count = db.store_document_links(&doc1.id, &links).await.unwrap();
        assert_eq!(count, 2);

        // Check outgoing
        let outgoing = db.get_outgoing_links(&doc1.id).await.unwrap();
        assert_eq!(outgoing.len(), 2);
        assert_eq!(outgoing[0].link_type, "internal");
        assert_eq!(outgoing[1].link_type, "external");

        // Resolve references
        let resolved = db.resolve_document_links(&source.id).await.unwrap();
        assert_eq!(resolved, 1); // doc1 → doc2

        // Check incoming
        let incoming = db.get_incoming_links(&doc2.id).await.unwrap();
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].from_doc_id, doc1.id);

        // Check related
        let related = db.get_related_documents(&doc1.id).await.unwrap();
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].id, doc2.id);
    }

    #[tokio::test]
    async fn test_store_document_links_replaces_existing() {
        let (db, _tmp) = setup_test_db().await;

        let source = Source::new(SourceType::Dir, "/docs".to_string(), None);
        db.insert_source(&source).await.unwrap();

        let doc = Document::new(
            source.id.clone(),
            "/docs/a.md".to_string(),
            "h1".to_string(),
        );
        let doc = db.upsert_document(&doc).await.unwrap();

        use crate::parse::ExtractedLink;
        let links1 = vec![ExtractedLink {
            url: "/b".to_string(),
            text: None,
            is_internal: true,
        }];
        db.store_document_links(&doc.id, &links1).await.unwrap();
        assert_eq!(db.get_outgoing_links(&doc.id).await.unwrap().len(), 1);

        // Re-store with different links replaces
        let links2 = vec![
            ExtractedLink {
                url: "/c".to_string(),
                text: None,
                is_internal: true,
            },
            ExtractedLink {
                url: "/d".to_string(),
                text: None,
                is_internal: true,
            },
        ];
        db.store_document_links(&doc.id, &links2).await.unwrap();
        let outgoing = db.get_outgoing_links(&doc.id).await.unwrap();
        assert_eq!(outgoing.len(), 2);
        assert_eq!(outgoing[0].to_uri, "/c");
    }

    #[tokio::test]
    async fn test_query_log() {
        let (db, _tmp) = setup_test_db().await;

        db.log_query(
            "how to use rust",
            Some("factual"),
            None,
            5,
            Some(0.85),
            Some(120),
        )
        .await
        .unwrap();
        db.log_query(
            "compare python java",
            Some("comparative"),
            None,
            3,
            Some(0.72),
            Some(95),
        )
        .await
        .unwrap();
        db.log_query("no results query", None, None, 0, None, Some(50))
            .await
            .unwrap();

        let recent = db.get_recent_queries(10).await.unwrap();
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].query_text, "no results query"); // Most recent first

        let analytics = db.get_query_analytics(30).await.unwrap();
        assert_eq!(analytics.total_queries, 3);
        assert_eq!(analytics.zero_result_queries, 1);
        assert!(analytics.avg_top_score.is_some());
    }

    #[tokio::test]
    async fn test_migration_v2_idempotent() {
        let (db, _tmp) = setup_test_db().await;
        // Run migrations twice - should not fail
        db.run_migrations().await.unwrap();
        db.run_migrations().await.unwrap();
        let version = db.get_schema_version().await.unwrap();
        assert_eq!(version, 2);
    }

    #[tokio::test]
    async fn test_get_corpus_stats() {
        let (db, _tmp) = setup_test_db().await;

        let source = Source::new(SourceType::Dir, "/docs".to_string(), None);
        db.insert_source(&source).await.unwrap();

        // Two documents
        let doc1 = Document::new(
            source.id.clone(),
            "/docs/a.md".to_string(),
            "h1".to_string(),
        );
        let doc1 = db.upsert_document(&doc1).await.unwrap();

        let doc2 = Document::new(
            source.id.clone(),
            "/docs/b.md".to_string(),
            "h2".to_string(),
        );
        let doc2 = db.upsert_document(&doc2).await.unwrap();

        // Chunk in doc1 contains "rust" and "programming"
        let c1 = Chunk::new(
            doc1.id.clone(),
            0,
            "ch1".to_string(),
            "Rust is a systems programming language".to_string(),
            0,
            38,
            None,
        );
        db.upsert_chunk(&c1).await.unwrap();

        // Chunk in doc2 contains "programming" but NOT "rust"
        let c2 = Chunk::new(
            doc2.id.clone(),
            0,
            "ch2".to_string(),
            "Programming in Python is fun".to_string(),
            0,
            28,
            None,
        );
        db.upsert_chunk(&c2).await.unwrap();

        let stats = db
            .get_corpus_stats(&["rust".to_string(), "programming".to_string()])
            .await
            .unwrap();

        assert_eq!(stats.total_docs, 2);
        assert_eq!(stats.doc_frequencies["rust"], 1);
        assert_eq!(stats.doc_frequencies["programming"], 2);
    }
}
