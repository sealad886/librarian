//! Reindex command - re-embed all documents

use crate::config::{Config, ResolvedEmbeddingConfig};
use crate::embed::{
    embed_image_text_in_batches, embed_images_in_batches, embed_in_batches, Embedder,
    ImageEmbedInput, fuse_embeddings,
};
use crate::error::{Error, Result};
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
pub async fn cmd_reindex(
    config: &Config,
    embedding: &ResolvedEmbeddingConfig,
    db: &MetaDb,
    store: &QdrantStore,
    embedder: &dyn Embedder,
    options: ReindexOptions,
) -> Result<ReindexStats> {
    info!("Starting reindex operation");

    store.ensure_collection().await?;

    if embedder.dimension() != store.dimension() {
        return Err(Error::Embedding(format!(
            "Embedding dimension {} does not match Qdrant collection dimension {}",
            embedder.dimension(),
            store.dimension()
        )));
    }

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
                config,
                embedding,
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
async fn reindex_document(
    config: &Config,
    embedding: &ResolvedEmbeddingConfig,
    db: &MetaDb,
    store: &QdrantStore,
    embedder: &dyn Embedder,
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

    let (text_chunks, image_chunks): (Vec<_>, Vec<_>) = chunks
        .into_iter()
        .partition(|c| c.modality == "text");

    let mut points = Vec::new();
    let mut total = 0usize;

    let batch_size = embedding.effective_batch_size(batch_size);

    if !text_chunks.is_empty() {
        let texts: Vec<String> = text_chunks.iter().map(|c| c.text.clone()).collect();
        let mut all_embeddings = Vec::with_capacity(texts.len());
        for batch in texts.chunks(batch_size) {
            let batch_vec: Vec<String> = batch.to_vec();
            let batch_embeddings = embedder.embed(batch_vec).await?;
            all_embeddings.extend(batch_embeddings);
        }

        for (chunk, embedding) in text_chunks.iter().zip(all_embeddings.into_iter()) {
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

        total += text_chunks.len();
    }

    if !image_chunks.is_empty() {
        if !embedding.supports_image_inputs() {
            warn!(doc_id = %doc_id, model = %embedding.model_id, "Skipping image reindex (model not multimodal)");
            if !points.is_empty() {
                store.upsert_points(points).await?;
            }
            return Ok(total);
        }

        if embedding.supports_multi_vector {
            return Err(Error::Embedding(format!(
                "Late-interaction embedding model '{}' does not support image reindex",
                embedding.model_id
            )));
        }

        let assets_dir = config.paths.base_dir.join("assets");
        let mut image_paths = Vec::new();
        let mut image_meta = Vec::new();

        for chunk in image_chunks.iter() {
            let hash = chunk
                .media_hash
                .as_ref()
                .unwrap_or(&chunk.content_hash);
            if let Some(path) = find_cached_asset_path(&assets_dir, hash) {
                image_paths.push(path.to_string_lossy().to_string());
                image_meta.push(chunk);
            } else {
                warn!(
                    doc_id = %doc_id,
                    hash = %hash,
                    "Skipping image chunk (cached asset not found)"
                );
            }
        }

        if !image_paths.is_empty() {
            let expected_dim = store.dimension();
            if embedder.dimension() != expected_dim {
                return Err(Error::Embedding(format!(
                    "Embedding dimension mismatch for model '{}' (family '{}', source {}): embedder {} != collection {}",
                    embedding.model_id,
                    embedding.family,
                    embedding.dimension_source,
                    embedder.dimension(),
                    expected_dim
                )));
            }

            let batch_size = embedding.effective_batch_size(batch_size);
            let contexts: Vec<Option<String>> = image_meta
                .iter()
                .map(|chunk| {
                    let text = chunk.text.trim();
                    if text.is_empty() { None } else { Some(text.to_string()) }
                })
                .collect();

            let embeddings = if embedding.supports_joint_inputs {
                let inputs = image_paths
                    .iter()
                    .zip(contexts.iter())
                    .map(|(path, text)| ImageEmbedInput {
                        image_path: path.clone(),
                        text: text.clone(),
                    })
                    .collect::<Vec<_>>();
                embed_image_text_in_batches(embedder, inputs, batch_size).await?
            } else {
                let image_embeddings = embed_images_in_batches(embedder, image_paths, batch_size).await?;
                let mut fused_embeddings = image_embeddings.clone();

                let mut text_inputs = Vec::new();
                let mut text_indices = Vec::new();
                for (idx, context) in contexts.iter().enumerate() {
                    if let Some(text) = context {
                        if !text.trim().is_empty() {
                            text_indices.push(idx);
                            text_inputs.push(text.clone());
                        }
                    }
                }

                if !text_inputs.is_empty() {
                    let text_embeddings = embed_in_batches(embedder, text_inputs, batch_size).await?;
                    for (offset, idx) in text_indices.iter().enumerate() {
                        let image_vec = &image_embeddings[*idx];
                        let text_vec = &text_embeddings[offset];
                        if image_vec.len() != text_vec.len() {
                            return Err(Error::Embedding(format!(
                                "Dual-encoder fusion dimension mismatch for model '{}': image {} != text {}",
                                embedding.model_id,
                                image_vec.len(),
                                text_vec.len()
                            )));
                        }
                        fused_embeddings[*idx] = fuse_embeddings(image_vec, text_vec);
                    }
                }

                fused_embeddings
            };

            if embeddings.is_empty() {
                warn!(doc_id = %doc_id, "No image embeddings returned");
            } else if embeddings[0].len() != expected_dim {
                return Err(Error::Embedding(format!(
                    "Image embedding dimension mismatch for model '{}': expected {}, got {}",
                    embedding.model_id,
                    expected_dim,
                    embeddings[0].len()
                )));
            } else {
                for (chunk, embedding_vec) in image_meta.iter().zip(embeddings.into_iter()) {
                    let point_id = Uuid::try_parse(&chunk.id).unwrap_or_else(|_| {
                        Uuid::new_v5(&Uuid::NAMESPACE_OID, chunk.id.as_bytes())
                    });

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
                    payload.modality = Some(chunk.modality.clone());
                    payload.media_url = chunk.media_url.clone();
                    payload.media_hash = chunk.media_hash.clone();

                    points.push(ChunkPoint {
                        id: point_id,
                        vector: embedding_vec,
                        payload,
                    });
                }

                total += image_meta.len();
            }
        }
    }

    if !points.is_empty() {
        store.upsert_points(points).await?;
    }

    Ok(total)
}

fn find_cached_asset_path(base_dir: &std::path::Path, hash: &str) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(base_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if let Some(name_str) = name.to_str() {
            if name_str.starts_with(hash) {
                return Some(entry.path());
            }
        }
    }
    None
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
