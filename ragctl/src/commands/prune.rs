//! Prune command - remove stale/deleted documents

use crate::config::Config;
use crate::error::Result;
use crate::meta::{MetaDb, SourceType};
use crate::store::QdrantStore;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;
use tracing::{info, warn};
use uuid::Uuid;

/// Prune statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PruneStats {
    pub sources_checked: usize,
    pub documents_checked: usize,
    pub documents_removed: usize,
    pub chunks_removed: usize,
    pub orphan_points_removed: usize,
}

/// Prune options
#[derive(Debug, Clone, Default)]
pub struct PruneOptions {
    /// Only check specific source IDs
    pub source_ids: Option<Vec<String>>,
    /// Perform a dry run (don't actually delete)
    pub dry_run: bool,
    /// Also remove orphaned Qdrant points not in DB
    pub remove_orphans: bool,
}

/// Execute prune command
pub async fn cmd_prune(
    _config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
    options: PruneOptions,
) -> Result<PruneStats> {
    info!(dry_run = options.dry_run, "Starting prune operation");

    let mut stats = PruneStats::default();

    // Get sources to check
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

    stats.sources_checked = sources.len();

    // Check each source
    for source in sources {
        let source_type: SourceType = source.source_type.parse().unwrap_or(SourceType::Dir);

        match source_type {
            SourceType::Dir => {
                prune_directory_source(db, store, &source.id, &source.uri, &mut stats, options.dry_run)
                    .await?;
            }
            SourceType::Url | SourceType::Sitemap => {
                // For URL sources, we can't easily check if pages still exist
                // without re-crawling, so just log info
                info!(
                    source_id = %source.id,
                    "URL/Sitemap source - skipping file existence check"
                );
            }
        }
    }

    // Remove orphaned Qdrant points
    if options.remove_orphans {
        let orphans = find_orphan_points(db, store).await?;
        stats.orphan_points_removed = orphans.len();

        if !options.dry_run && !orphans.is_empty() {
            store.delete_points(&orphans).await?;
            info!(count = orphans.len(), "Removed orphaned Qdrant points");
        }
    }

    Ok(stats)
}

/// Prune a directory source - remove documents for files that no longer exist
async fn prune_directory_source(
    db: &MetaDb,
    store: &QdrantStore,
    source_id: &str,
    _base_path: &str,
    stats: &mut PruneStats,
    dry_run: bool,
) -> Result<()> {
    let documents = db.list_source_documents(source_id).await?;
    stats.documents_checked += documents.len();

    for doc in documents {
        let file_path = Path::new(&doc.uri);
        
        if !file_path.exists() {
            info!(
                doc_id = %doc.id,
                uri = %doc.uri,
                "Document file no longer exists"
            );

            if !dry_run {
                // Get chunks to delete from Qdrant
                let chunks = db.list_document_chunks(&doc.id).await?;
                let point_ids: Vec<Uuid> = chunks
                    .iter()
                    .filter_map(|c| Uuid::try_parse(&c.id).ok())
                    .collect();

                if !point_ids.is_empty() {
                    store.delete_points(&point_ids).await?;
                }

                // Delete from database
                db.delete_document(&doc.id).await?;
                stats.chunks_removed += chunks.len();
            }

            stats.documents_removed += 1;
        }
    }

    Ok(())
}

/// Find Qdrant points that are not in the database
async fn find_orphan_points(db: &MetaDb, store: &QdrantStore) -> Result<Vec<Uuid>> {
    // Get all chunk IDs from database
    let db_chunks = db.list_all_chunk_ids().await?;
    let db_ids: HashSet<String> = db_chunks.into_iter().collect();

    // Get all point IDs from Qdrant
    let qdrant_points = store.list_all_point_ids().await?;

    // Find orphans (in Qdrant but not in DB)
    let orphans: Vec<Uuid> = qdrant_points
        .into_iter()
        .filter(|id| !db_ids.contains(&id.to_string()))
        .collect();

    if !orphans.is_empty() {
        warn!(
            count = orphans.len(),
            "Found orphaned Qdrant points not in database"
        );
    }

    Ok(orphans)
}

/// Remove a specific source and all its data
pub async fn cmd_remove_source(
    db: &MetaDb,
    store: &QdrantStore,
    source_id: &str,
) -> Result<PruneStats> {
    info!(source_id = %source_id, "Removing source");

    let mut stats = PruneStats::default();
    stats.sources_checked = 1;

    // Get all documents for the source
    let documents = db.list_source_documents(source_id).await?;
    stats.documents_checked = documents.len();

    // Delete chunks from Qdrant
    for doc in &documents {
        let chunks = db.list_document_chunks(&doc.id).await?;
        let point_ids: Vec<Uuid> = chunks
            .iter()
            .filter_map(|c| Uuid::try_parse(&c.id).ok())
            .collect();

        if !point_ids.is_empty() {
            store.delete_points(&point_ids).await?;
            stats.chunks_removed += point_ids.len();
        }

        stats.documents_removed += 1;
    }

    // Delete source from database (cascades to documents and chunks)
    db.delete_source(source_id).await?;

    info!(
        source_id = %source_id,
        documents = stats.documents_removed,
        chunks = stats.chunks_removed,
        "Source removed"
    );

    Ok(stats)
}

/// Print prune stats to console
pub fn print_prune_stats(stats: &PruneStats, dry_run: bool) {
    println!("\n🧹 Prune {}\n", if dry_run { "(Dry Run)" } else { "Complete" });
    println!("Sources checked: {}", stats.sources_checked);
    println!("Documents checked: {}", stats.documents_checked);
    println!(
        "Documents {}: {}",
        if dry_run { "to remove" } else { "removed" },
        stats.documents_removed
    );
    println!(
        "Chunks {}: {}",
        if dry_run { "to remove" } else { "removed" },
        stats.chunks_removed
    );
    if stats.orphan_points_removed > 0 {
        println!(
            "Orphan points {}: {}",
            if dry_run { "to remove" } else { "removed" },
            stats.orphan_points_removed
        );
    }
}
