//! Init command implementation

use crate::config::Config;
use crate::error::{Error, Result};
use crate::meta::MetaDb;
use crate::store::QdrantStore;
use std::path::PathBuf;
use tracing::info;

/// Initialize librarian configuration and database
pub async fn cmd_init(base_dir: Option<PathBuf>, force: bool) -> Result<()> {
    let mut config = Config::default();

    if let Some(dir) = base_dir {
        config.paths.base_dir = dir.clone();
        config.paths.config_file = dir.join("config.toml");
        config.paths.db_file = dir.join("metadata.db");
    } else {
        let default_base = Config::default_base_dir();
        config.paths.base_dir = default_base.clone();
        config.paths.config_file = default_base.join("config.toml");
        config.paths.db_file = default_base.join("metadata.db");
    }

    // Check if already initialized
    if config.paths.config_file.exists() && !force {
        return Err(Error::AlreadyInitialized(
            config.paths.base_dir.display().to_string(),
        ));
    }

    // Create directory
    std::fs::create_dir_all(&config.paths.base_dir)?;

    // Validate config
    config.validate()?;

    // Save config
    config.save()?;
    info!("Created config at {:?}", config.paths.config_file);

    // Initialize database
    let db = MetaDb::connect(&config).await?;
    db.init_schema().await?;
    info!("Created database at {:?}", config.paths.db_file);

    // Try to connect to Qdrant and create collection
    match QdrantStore::connect(&config).await {
        Ok(store) => match store.ensure_collection().await {
            Ok(_) => info!("Qdrant collection '{}' ready", config.collection_name),
            Err(e) => {
                tracing::warn!(
                    "Could not create Qdrant collection: {}. You can create it later.",
                    e
                );
            }
        },
        Err(e) => {
            tracing::warn!(
                "Could not connect to Qdrant at {}: {}. Make sure Qdrant is running.",
                config.qdrant_url,
                e
            );
        }
    }

    println!("✓ Initialized librarian at {:?}", config.paths.base_dir);
    println!("\nConfiguration: {:?}", config.paths.config_file);
    println!("Database: {:?}", config.paths.db_file);
    println!("\nNext steps:");
    println!("  librarian ingest dir ./path/to/docs    # Index local docs");
    println!("  librarian ingest url https://docs.rs   # Index web docs");
    println!("  librarian query \"how to use X\"         # Search the index");

    Ok(())
}
