//! Status command implementation

use crate::config::Config;
use crate::error::Result;
use crate::meta::{GlobalStats, MetaDb, SourceStats};
use crate::store::QdrantStore;
use clap_complete::Shell;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusInfo {
    pub config_path: String,
    pub db_path: String,
    pub qdrant_url: String,
    pub collection_name: String,
    pub embedding_model: String,
    pub qdrant_connected: bool,
    pub collection_exists: bool,
    pub qdrant_points: usize,
    pub db_stats: GlobalStats,
}

/// Get system status
pub async fn cmd_status(config: &Config, db: &MetaDb, store: &QdrantStore) -> Result<StatusInfo> {
    info!("Getting status");

    let db_stats = db.get_global_stats().await?;

    // Check if we can connect to Qdrant and if collection exists
    let (qdrant_connected, collection_exists, qdrant_points) = match store.collection_exists().await
    {
        Ok(true) => {
            // Collection exists, get stats
            match store.get_stats().await {
                Ok(stats) => (true, true, stats.points_count),
                Err(e) => {
                    tracing::debug!("Qdrant stats error: {:?}", e);
                    (true, true, 0)
                }
            }
        }
        Ok(false) => (true, false, 0), // Connected but collection doesn't exist
        Err(e) => {
            tracing::debug!("Qdrant connection error: {:?}", e);
            (false, false, 0)
        }
    };

    Ok(StatusInfo {
        config_path: config.paths.config_file.display().to_string(),
        db_path: config.paths.db_file.display().to_string(),
        qdrant_url: config.qdrant_url.clone(),
        collection_name: config.collection_name.clone(),
        embedding_model: config.embedding.model.clone(),
        qdrant_connected,
        collection_exists,
        qdrant_points,
        db_stats,
    })
}

/// Source information with stats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub id: String,
    pub source_type: String,
    pub uri: String,
    pub name: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub stats: SourceStats,
}

/// List all sources with their stats
pub async fn cmd_list_sources(db: &MetaDb) -> Result<Vec<SourceInfo>> {
    info!("Listing sources");

    let sources = db.list_sources().await?;
    let mut result = Vec::with_capacity(sources.len());

    for source in sources {
        let stats = db.get_source_stats(&source.id).await?;
        result.push(SourceInfo {
            id: source.id,
            source_type: source.source_type,
            uri: source.uri,
            name: source.name,
            created_at: source.created_at,
            updated_at: source.updated_at,
            stats,
        });
    }

    Ok(result)
}

/// Print status to console
pub fn print_status(status: &StatusInfo) {
    println!("\n📊 librarian Status\n");
    println!("Configuration: {}", status.config_path);
    println!("Database: {}", status.db_path);
    println!("\nQdrant:");
    println!("  URL: {}", status.qdrant_url);
    println!("  Collection: {}", status.collection_name);

    let connection_status = if status.qdrant_connected {
        if status.collection_exists {
            "✓ Connected"
        } else {
            "⚠ Connected (collection not created - run 'librarian ingest' to create)"
        }
    } else {
        "✗ Not connected"
    };
    println!("  Status: {}", connection_status);
    println!("  Points: {}", status.qdrant_points);
    println!("\nEmbedding Model: {}", status.embedding_model);
    println!("\nDatabase Stats:");
    println!("  Sources: {}", status.db_stats.source_count);
    println!("  Documents: {}", status.db_stats.document_count);
    println!("  Chunks: {}", status.db_stats.chunk_count);
}

/// Print sources list to console
pub fn print_sources(sources: &[SourceInfo]) {
    println!("\n📚 Registered Sources\n");

    if sources.is_empty() {
        println!("No sources registered. Use 'librarian ingest' to add sources.");
        return;
    }

    for source in sources {
        println!(
            "• {} [{}]",
            source.name.as_deref().unwrap_or(&source.uri),
            source.source_type
        );
        println!("  ID: {}", source.id);
        println!("  URI: {}", source.uri);
        println!(
            "  Documents: {}, Chunks: {}",
            source.stats.document_count, source.stats.chunk_count
        );
        println!("  Created: {}", source.created_at);
        println!();
    }
}

/// Print source IDs with descriptions for shell completions
pub fn print_source_completions(sources: &[SourceInfo], shell: Shell) {
    for source in sources {
        let display_name = source.name.as_deref().unwrap_or(&source.uri);
        let mut description = format!(
            "{} (at {}), Created {}",
            display_name, source.uri, source.created_at
        );
        description = description.replace('\n', " ");

        match shell {
            Shell::Zsh => {
                let sanitized = description.replace(':', "\\:");
                println!("{}:{}", source.id, sanitized);
            }
            Shell::Fish => {
                let sanitized = description.replace('\t', " ");
                println!("{}\t{}", source.id, sanitized);
            }
            _ => {
                println!("{}", source.id);
            }
        }
    }
}
