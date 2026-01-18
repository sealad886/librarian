//! Update command - incrementally refresh sources and prune vectors

use crate::commands::{
    cmd_ingest_dir, cmd_ingest_sitemap, cmd_ingest_url, CrawlOverrides, IngestStats,
};
use crate::commands::{cmd_prune, PruneOptions, PruneStats};
use crate::config::Config;
use crate::error::Result;
use crate::meta::{MetaDb, SourceType};
use crate::store::QdrantStore;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{info, warn};

/// Update statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpdateStats {
    pub sources_updated: usize,
    pub ingest: IngestStats,
    pub prune: Option<PruneStats>,
}

/// Update options
#[derive(Debug, Clone)]
pub struct UpdateOptions {
    /// Only update specific source IDs
    pub source_ids: Option<Vec<String>>,
    /// Also prune orphan vectors after updating
    pub prune_orphans: bool,
}

impl Default for UpdateOptions {
    fn default() -> Self {
        Self {
            source_ids: None,
            prune_orphans: true,
        }
    }
}

/// Execute update command - re-run ingestion for sources and prune orphaned vectors
pub async fn cmd_update(
    config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
    options: UpdateOptions,
) -> Result<UpdateStats> {
    info!("Starting update operation");

    let mut stats = UpdateStats::default();

    // Get sources to update
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

    stats.sources_updated = sources.len();

    for source in sources {
        let source_type: SourceType = source.source_type.parse().unwrap_or(SourceType::Dir);

        let ingest_result = match source_type {
            SourceType::Dir => {
                let path = Path::new(&source.uri);
                cmd_ingest_dir(config, db, store, path, source.name.clone()).await
            }
            SourceType::Url => {
                let overrides = CrawlOverrides::default();
                cmd_ingest_url(
                    config,
                    db,
                    store,
                    &source.uri,
                    source.name.clone(),
                    overrides,
                )
                .await
            }
            SourceType::Sitemap => {
                cmd_ingest_sitemap(config, db, store, &source.uri, source.name.clone(), None).await
            }
        };

        match ingest_result {
            Ok(ingest_stats) => {
                stats.ingest.docs_processed += ingest_stats.docs_processed;
                stats.ingest.docs_skipped += ingest_stats.docs_skipped;
                stats.ingest.chunks_created += ingest_stats.chunks_created;
                stats.ingest.chunks_updated += ingest_stats.chunks_updated;
                stats.ingest.chunks_deleted += ingest_stats.chunks_deleted;
                stats.ingest.errors.extend_from_slice(&ingest_stats.errors);
                stats
                    .ingest
                    .overlap_warnings
                    .extend_from_slice(&ingest_stats.overlap_warnings);
            }
            Err(e) => {
                let error_msg = format!("{}: {}", source.uri, e);
                warn!(%error_msg, "Update failed for source");
                stats.ingest.errors.push(error_msg);
            }
        }
    }

    // Prune orphan vectors (and per-source stale documents) after ingestion
    if options.prune_orphans {
        let prune_options = PruneOptions {
            source_ids: options.source_ids.clone(),
            dry_run: false,
            remove_orphans: true,
        };

        match cmd_prune(config, db, store, prune_options).await {
            Ok(prune_stats) => {
                stats.prune = Some(prune_stats);
            }
            Err(e) => {
                let error_msg = format!("prune: {}", e);
                warn!(%error_msg, "Prune step failed during update");
                stats.ingest.errors.push(error_msg);
            }
        }
    }

    Ok(stats)
}

/// Print update stats to console
pub fn print_update_stats(stats: &UpdateStats) {
    println!("\n♻️  Update Complete\n");
    println!("Sources updated: {}", stats.sources_updated);
    println!("Documents processed: {}", stats.ingest.docs_processed);
    println!("Documents skipped: {}", stats.ingest.docs_skipped);
    println!("Chunks created: {}", stats.ingest.chunks_created);
    println!("Chunks updated: {}", stats.ingest.chunks_updated);
    if stats.ingest.chunks_deleted > 0 {
        println!("Chunks deleted: {}", stats.ingest.chunks_deleted);
    }
    if let Some(prune) = &stats.prune {
        println!("\nPrune:");
        println!("  Documents removed: {}", prune.documents_removed);
        println!("  Chunks removed: {}", prune.chunks_removed);
        if prune.orphan_points_removed > 0 {
            println!("  Orphan points removed: {}", prune.orphan_points_removed);
        }
    }

    if !stats.ingest.overlap_warnings.is_empty() {
        println!("\nWarnings:");
        for warning in &stats.ingest.overlap_warnings {
            println!("- {}", warning);
        }
    }

    if !stats.ingest.errors.is_empty() {
        println!("\nErrors:");
        for error in &stats.ingest.errors {
            println!("- {}", error);
        }
    }
}
