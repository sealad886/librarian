//! librarian CLI entry point

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Shell};
use librarian::{
    commands::{
        cmd_ingest_dir, cmd_ingest_sitemap, cmd_ingest_url, cmd_init, cmd_list_sources, cmd_prune,
        cmd_query, cmd_reindex, cmd_remove_source, cmd_status, print_prune_stats,
        print_query_results, print_reindex_stats, print_sources, print_status, PruneOptions,
        QueryOptions, ReindexOptions,
    },
    config::Config,
    embed::FastEmbedder,
    error::Result,
    mcp::McpServer,
    meta::MetaDb,
    store::QdrantStore,
};
use std::path::PathBuf;
use tracing::error;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[derive(Parser)]
#[command(name = "librarian")]
#[command(version, about = "Local RAG CLI tool with MCP server support", long_about = None)]
struct Cli {
    /// Path to config file
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Output as JSON
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize librarian configuration and database
    Init {
        /// Force overwrite existing config
        #[arg(long)]
        force: bool,
    },

    /// Ingest documentation into the RAG index
    Ingest {
        #[command(subcommand)]
        source: IngestSource,
    },

    /// Query the RAG index
    Query {
        /// The search query
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Minimum similarity score (0-1)
        #[arg(short, long)]
        min_score: Option<f32>,

        /// Filter to specific source IDs
        #[arg(long)]
        source: Option<Vec<String>>,

        /// Deduplicate results by document
        #[arg(long)]
        dedupe: bool,
    },

    /// Show system status
    Status,

    /// List registered sources
    Sources {
        /// Output only source IDs (one per line, for scripting)
        #[arg(long)]
        ids_only: bool,
    },

    /// Remove stale documents and orphan points
    Prune {
        /// Dry run - show what would be removed
        #[arg(long)]
        dry_run: bool,

        /// Also remove orphan Qdrant points
        #[arg(long)]
        remove_orphans: bool,

        /// Only prune specific source IDs
        #[arg(long)]
        source: Option<Vec<String>>,
    },

    /// Re-embed all documents
    Reindex {
        /// Only reindex specific source IDs
        #[arg(long)]
        source: Option<Vec<String>>,

        /// Batch size for embedding
        #[arg(long, default_value = "32")]
        batch_size: usize,
    },

    /// Remove a source and all its data
    /// 
    /// Use 'librarian sources --ids-only' to list available source IDs
    Remove {
        /// Source ID to remove (use 'librarian sources' to list)
        source_id: String,
    },

    /// Start MCP server on stdio
    Mcp,

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[derive(Subcommand)]
enum IngestSource {
    /// Ingest a local directory
    Dir {
        /// Path to directory
        path: PathBuf,

        /// Source name (defaults to directory name)
        #[arg(short, long)]
        name: Option<String>,

        /// File extensions to include (e.g., md,txt,html)
        #[arg(long)]
        extensions: Option<String>,

        /// Exclude patterns (glob)
        #[arg(long)]
        exclude: Option<Vec<String>>,
    },

    /// Ingest a URL (with crawling)
    Url {
        /// URL to ingest
        url: String,

        /// Source name (defaults to domain)
        #[arg(short, long)]
        name: Option<String>,

        /// Maximum pages to crawl
        #[arg(long, default_value = "100")]
        max_pages: u32,

        /// Maximum crawl depth
        #[arg(long, default_value = "3")]
        max_depth: u32,

        /// Restrict crawling to this path prefix (e.g., /docs/)
        /// If not specified, defaults to the seed URL's directory path
        #[arg(long)]
        path_prefix: Option<String>,
    },

    /// Ingest URLs from a sitemap
    Sitemap {
        /// Sitemap URL
        url: String,

        /// Source name
        #[arg(short, long)]
        name: Option<String>,

        /// Maximum pages to fetch
        #[arg(long)]
        max_pages: Option<u32>,
    },
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        error!("{}", e);
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let filter = if cli.verbose {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"))
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    };

    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(filter)
        .init();

    // Handle init command specially (doesn't need existing config)
    if matches!(cli.command, Commands::Init { .. }) {
        return handle_init(cli).await;
    }

    // Handle completions command (doesn't need config/db/store)
    if let Commands::Completions { shell } = cli.command {
        let mut cmd = Cli::command();
        generate(shell, &mut cmd, "librarian", &mut std::io::stdout());
        return Ok(());
    }

    // Load configuration
    let config = load_config(cli.config.as_deref()).await?;

    // Initialize components
    let db = MetaDb::new(&config.paths.db_file).await?;
    let store = QdrantStore::new(&config.qdrant_url, &config.collection_name).await?;

    // Handle commands
    match cli.command {
        Commands::Init { .. } => unreachable!(),

        Commands::Ingest { source } => {
            handle_ingest(&config, &db, &store, source).await?;
        }

        Commands::Query {
            query,
            limit,
            min_score,
            source,
            dedupe,
        } => {
            let options = QueryOptions {
                k: Some(limit),
                min_score,
                source_ids: source,
                dedupe_docs: dedupe,
                ..Default::default()
            };

            let results = cmd_query(&config, &db, &store, &query, options).await?;

            if cli.json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                print_query_results(&results);
            }
        }

        Commands::Status => {
            let status = cmd_status(&config, &db, &store).await?;

            if cli.json {
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                print_status(&status);
            }
        }

        Commands::Sources { ids_only } => {
            let sources = cmd_list_sources(&db).await?;

            if ids_only {
                // Output only IDs for scripting/completions
                for source in &sources {
                    println!("{}", source.id);
                }
            } else if cli.json {
                println!("{}", serde_json::to_string_pretty(&sources)?);
            } else {
                print_sources(&sources);
            }
        }

        Commands::Prune {
            dry_run,
            remove_orphans,
            source,
        } => {
            let options = PruneOptions {
                source_ids: source,
                dry_run,
                remove_orphans,
            };

            let stats = cmd_prune(&config, &db, &store, options).await?;

            if cli.json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                print_prune_stats(&stats, dry_run);
            }
        }

        Commands::Reindex { source, batch_size } => {
            let embedder = FastEmbedder::new(&config.embedding)?;

            let options = ReindexOptions {
                source_ids: source,
                batch_size,
            };

            let stats = cmd_reindex(&config, &db, &store, &embedder, options).await?;

            if cli.json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                print_reindex_stats(&stats);
            }
        }

        Commands::Remove { source_id } => {
            let stats = cmd_remove_source(&db, &store, &source_id).await?;

            if cli.json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                println!("✓ Source '{}' removed successfully", source_id);
                print_prune_stats(&stats, false);
            }
        }

        Commands::Mcp => {
            let server = McpServer::new(config, db, store);
            server.run().await.map_err(|e| librarian::error::Error::McpProtocol(e.to_string()))?;
        }

        Commands::Completions { .. } => unreachable!(),
    }

    Ok(())
}

async fn handle_init(cli: Cli) -> Result<()> {
    let Commands::Init { force } = cli.command else {
        unreachable!()
    };

    // Get the base directory: if user specifies config file, use its parent dir
    // Otherwise use default base dir
    let (base_dir, config_path) = if let Some(path) = cli.config {
        let base = path.parent().map(PathBuf::from).unwrap_or_else(Config::default_base_dir);
        let config = if path.extension().map_or(false, |e| e == "toml") {
            path  // User specified a .toml file
        } else {
            path.join("config.toml")  // User specified a directory
        };
        (base, config)
    } else {
        let base = Config::default_base_dir();
        (base.clone(), base.join("config.toml"))
    };

    if config_path.exists() && !force {
        eprintln!(
            "Config file already exists at: {}\nUse --force to overwrite.",
            config_path.display()
        );
        std::process::exit(1);
    }

    cmd_init(Some(base_dir), force).await?;

    println!("✓ librarian initialized successfully");
    println!("  Config: {}", config_path.display());
    println!("\nNext steps:");
    println!("  1. Edit the config file to customize settings");
    println!("  2. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant");
    println!("  3. Ingest docs: librarian ingest dir /path/to/docs");

    Ok(())
}

async fn load_config(path: Option<&std::path::Path>) -> Result<Config> {
    let config_path = path
        .map(PathBuf::from)
        .unwrap_or_else(Config::default_config_path);

    if !config_path.exists() {
        eprintln!(
            "Config file not found: {}\nRun 'librarian init' first.",
            config_path.display()
        );
        std::process::exit(1);
    }

    Config::load(&config_path)
}

async fn handle_ingest(
    config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
    source: IngestSource,
) -> Result<()> {
    match source {
        IngestSource::Dir {
            path,
            name,
            extensions: _,
            exclude: _,
        } => {
            let stats = cmd_ingest_dir(
                config, db, store, &path, name,
            )
            .await?;

            // Display overlap warnings
            for warning in &stats.overlap_warnings {
                println!("{}", warning);
            }

            println!("\n✓ Directory ingestion complete");
            println!("  Documents processed: {}", stats.docs_processed);
            println!("  Chunks created: {}", stats.chunks_created);
            println!("  Chunks updated: {}", stats.chunks_updated);
            println!("  Chunks deleted: {}", stats.chunks_deleted);
        }

        IngestSource::Url {
            url,
            name,
            max_pages,
            max_depth,
            path_prefix,
        } => {
            use librarian::commands::CrawlOverrides;
            let overrides = CrawlOverrides {
                max_pages: Some(max_pages),
                max_depth: Some(max_depth),
                path_prefix,
            };
            let stats = cmd_ingest_url(
                config, db, store, &url, name, overrides,
            )
            .await?;

            // Display overlap warnings
            for warning in &stats.overlap_warnings {
                println!("{}", warning);
            }

            println!("\n✓ URL ingestion complete");
            println!("  Pages processed: {}", stats.docs_processed);
            println!("  Chunks created: {}", stats.chunks_created);
            println!("  Chunks updated: {}", stats.chunks_updated);
        }

        IngestSource::Sitemap {
            url,
            name,
            max_pages,
        } => {
            let stats = cmd_ingest_sitemap(
                config,
                db,
                store,
                &url,
                name,
                max_pages,
            )
            .await?;

            // Display overlap warnings
            for warning in &stats.overlap_warnings {
                println!("{}", warning);
            }

            println!("\n✓ Sitemap ingestion complete");
            println!("  Pages processed: {}", stats.docs_processed);
            println!("  Chunks created: {}", stats.chunks_created);
            println!("  Chunks updated: {}", stats.chunks_updated);
        }
    }

    Ok(())
}
