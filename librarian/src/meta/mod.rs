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
use tracing::{debug, info};
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

impl std::fmt::Display for RunStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunStatus::Running => write!(f, "running"),
            RunStatus::Completed => write!(f, "completed"),
            RunStatus::Failed => write!(f, "failed"),
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
}

/// An ingestion run record
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct IngestionRun {
    pub id: String,
    pub source_id: String,
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
    pub fn new(source_id: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_id,
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
        info!("Initializing database schema");
        sqlx::query(SCHEMA_SQL).execute(&self.pool).await?;
        Ok(())
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

    /// Insert or update a document
    pub async fn upsert_document(&self, doc: &Document) -> Result<()> {
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
        Ok(())
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
            INSERT INTO chunks (id, doc_id, chunk_index, chunk_hash, chunk_text, char_start, char_end, headings_json, qdrant_point_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id, chunk_index) DO UPDATE SET
                chunk_hash = excluded.chunk_hash,
                chunk_text = excluded.chunk_text,
                char_start = excluded.char_start,
                char_end = excluded.char_end,
                headings_json = excluded.headings_json,
                qdrant_point_id = excluded.qdrant_point_id,
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

    /// Get chunk by Qdrant point ID
    pub async fn get_chunk_by_point_id(&self, point_id: &str) -> Result<Option<Chunk>> {
        let chunk = sqlx::query_as::<_, Chunk>("SELECT * FROM chunks WHERE qdrant_point_id = ?")
            .bind(point_id)
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
            "SELECT qdrant_point_id FROM chunks WHERE doc_id = ? AND chunk_index >= ?",
        )
        .bind(doc_id)
        .bind(from_index)
        .fetch_all(&self.pool)
        .await?;

        sqlx::query("DELETE FROM chunks WHERE doc_id = ? AND chunk_index >= ?")
            .bind(doc_id)
            .bind(from_index)
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
    pub async fn start_ingestion_run(&self, source_id: &str) -> Result<IngestionRun> {
        let run = IngestionRun::new(source_id.to_string());
        sqlx::query(
            r#"
            INSERT INTO ingestion_runs (id, source_id, started_at, status, docs_processed, chunks_created, chunks_updated, chunks_deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&run.id)
        .bind(&run.source_id)
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
}

/// A simplified chunk record for API use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub id: String,
    pub text: String,
    pub chunk_index: i32,
    pub content_hash: String,
    pub headings: Option<String>,
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
        db.upsert_document(&doc).await.unwrap();

        let loaded = db
            .get_document_by_uri(&source.id, "/docs/file.md")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.content_hash, "hash1");

        // Update the document
        doc.content_hash = "hash2".to_string();
        doc.updated_at = Utc::now().to_rfc3339();
        db.upsert_document(&doc).await.unwrap();

        let loaded = db
            .get_document_by_uri(&source.id, "/docs/file.md")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.content_hash, "hash2");
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
        db.upsert_document(&doc).await.unwrap();

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
}
