//! Reindex command - re-embed all documents

use crate::config::Config;
use crate::embed::Embedder;
use crate::error::Result;
use crate::meta::{MetaDb, RunOperation, RunStatus};
use crate::store::{ChunkPayload, ChunkPoint, QdrantStore};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use uuid::Uuid;

/// Reindex statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReindexStats {
    pub sources_processed: usize,
    pub documents_processed: usize,
    pub chunks_reindexed: usize,
    pub errors: usize,
}

/// Reindex options
#[derive(Debug, Clone, Default)]
pub struct ReindexOptions {
    /// Only reindex specific source IDs
    pub source_ids: Option<Vec<String>>,
    /// Batch size for embedding
    pub batch_size: usize,
}

impl ReindexOptions {
    pub fn new() -> Self {
        Self {
            source_ids: None,
            batch_size: 32,
        }
    }
}

/// Execute reindex command - re-embed all chunks
pub async fn cmd_reindex<E: Embedder>(
    _config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
    embedder: &E,
    options: ReindexOptions,
) -> Result<ReindexStats> {
    info!("Starting reindex operation");

    let mut stats = ReindexStats::default();

    // Get sources to reindex
    let sources = match &options.source_ids {
        Some(ids) => {
            let mut sources = Vec::new();
            for id in ids {
                if let Some(source) = db.get_source(id).await? {
                    sources.push(source);
                }
            }
            sources
        }
        None => db.list_sources().await?,
    };

    stats.sources_processed = sources.len();

    // Process each source
    for source in sources {
        let run = db
            .start_ingestion_run(&source.id, RunOperation::Reindex)
            .await?;

        let mut run_errors: Vec<String> = Vec::new();
        let mut run_docs_processed = 0usize;
        let mut run_chunks_updated = 0usize;

        let documents = db.list_source_documents(&source.id).await?;

        for doc in documents {
            match reindex_document(
                db,
                store,
                embedder,
                &source.id,
                &source.source_type,
                &source.uri,
                &doc.id,
                options.batch_size,
            )
            .await
            {
                Ok(chunk_count) => {
                    stats.documents_processed += 1;
                    stats.chunks_reindexed += chunk_count;
                    run_docs_processed += 1;
                    run_chunks_updated += chunk_count;
                }
                Err(e) => {
                    warn!(
                        doc_id = %doc.id,
                        error = %e,
                        "Failed to reindex document"
                    );
                    stats.errors += 1;
                    run_errors.push(format!("{}: {}", doc.id, e));
                }
            }
        }

        let status = if run_errors.is_empty() {
            RunStatus::Completed
        } else {
            RunStatus::Failed
        };

        let _ = db
            .complete_ingestion_run(
                &run.id,
                status,
                run_docs_processed as i32,
                0,
                run_chunks_updated as i32,
                0,
                if run_errors.is_empty() {
                    None
                } else {
                    Some(run_errors.clone())
                },
            )
            .await;
    }

    info!(
        documents = stats.documents_processed,
        chunks = stats.chunks_reindexed,
        errors = stats.errors,
        "Reindex complete"
    );

    Ok(stats)
}

/// Reindex a single document's chunks
async fn reindex_document<E: Embedder>(
    db: &MetaDb,
    store: &QdrantStore,
    embedder: &E,
    source_id: &str,
    source_type: &str,
    source_uri: &str,
    doc_id: &str,
    batch_size: usize,
) -> Result<usize> {
    let doc = db
        .get_document(doc_id)
        .await?
        .ok_or_else(|| crate::error::Error::DocumentNotFound(doc_id.to_string()))?;

    let chunks = db.list_document_chunks(doc_id).await?;

    if chunks.is_empty() {
        return Ok(0);
    }

    // Collect chunk texts
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();

    // Generate embeddings in batches
    let mut all_embeddings = Vec::with_capacity(texts.len());
    for batch in texts.chunks(batch_size) {
        let batch_vec: Vec<String> = batch.to_vec();
        let batch_embeddings = embedder.embed(batch_vec).await?;
        all_embeddings.extend(batch_embeddings);
    }

    // Build Qdrant points
    let mut points = Vec::with_capacity(chunks.len());
    for (chunk, embedding) in chunks.iter().zip(all_embeddings.into_iter()) {
        let point_id = Uuid::try_parse(&chunk.id)
            .unwrap_or_else(|_| Uuid::new_v5(&Uuid::NAMESPACE_OID, chunk.id.as_bytes()));

        let headings: Option<Vec<String>> = chunk
            .headings
            .as_ref()
            .and_then(|h| serde_json::from_str(h).ok());

        let mut payload = ChunkPayload::new(
            source_id.to_string(),
            source_type.to_string(),
            source_uri.to_string(),
            doc_id.to_string(),
            doc.uri.clone(),
            chunk.chunk_index,
            chunk.content_hash.clone(),
            chrono::Utc::now().to_rfc3339(),
        );
        payload.title = doc.title.clone();
        payload.headings = headings;

        points.push(ChunkPoint {
            id: point_id,
            vector: embedding,
            payload,
        });
    }

    // Upsert to Qdrant
    store.upsert_points(points).await?;

    Ok(chunks.len())
}

/// Print reindex stats to console
pub fn print_reindex_stats(stats: &ReindexStats) {
    println!("\n🔄 Reindex Complete\n");
    println!("Sources processed: {}", stats.sources_processed);
    println!("Documents processed: {}", stats.documents_processed);
    println!("Chunks reindexed: {}", stats.chunks_reindexed);
    if stats.errors > 0 {
        println!("Errors: {}", stats.errors);
    }
}
