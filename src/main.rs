//! librarian CLI entry point

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Shell};
use librarian::{
    commands::{
        cmd_ingest_dir, cmd_ingest_sitemap, cmd_ingest_url, cmd_init, cmd_list_sources, cmd_prune,
        cmd_query, cmd_reindex, cmd_remove_source, cmd_rename_source, cmd_status, cmd_update,
        cmd_xinference_sync_models, print_prune_stats, print_query_results, print_reindex_stats,
        print_source_completions, print_sources, print_status, print_update_stats, PruneOptions,
        QueryOptions, ReindexOptions, UpdateOptions, XinferenceSyncOptions,
    },
    config::Config,
    embed::create_embedder_auto,
    error::{Error, Result},
    mcp::McpServer,
    meta::{MetaDb, RunOperation},
    models::embedding_model_spec,
    progress::LogWriterFactory,
    store::QdrantStore,
};
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use tokio::process::Command;
use tokio::time::sleep;
use tracing::{debug, error};
use tracing_subscriber::fmt::writer::BoxMakeWriter;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use url::Url;

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

        /// Run without interactive prompts (writes defaults)
        #[arg(long)]
        non_interactive: bool,

        /// Accept defaults and skip confirmation
        #[arg(long, short = 'y')]
        yes: bool,
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

        /// Output source IDs with descriptions for shell completions
        #[arg(long, value_enum, hide = true)]
        completion: Option<Shell>,
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

    /// Incrementally update sources and prune embeddings
    Update {
        /// Only update specific source IDs
        #[arg(long)]
        source: Option<Vec<String>>,

        /// Skip pruning orphan vectors after updating
        #[arg(long)]
        skip_prune: bool,
    },

    /// Remove a source and all its data
    ///
    /// Use 'librarian sources --ids-only' to list available source IDs
    Remove {
        /// Source ID to remove (use 'librarian sources' to list)
        source_id: String,
    },

    /// Rename an existing source
    Rename {
        /// Source ID to rename
        source_id: String,
        /// New name to set
        name: String,
    },

    /// Start MCP server on stdio
    Mcp,

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },

    /// Manage Qdrant vector database
    Db {
        #[command(subcommand)]
        action: DbAction,
    },

    /// Configuration management commands
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Xinference registry tools
    Xinference {
        #[command(subcommand)]
        action: XinferenceAction,
    },
}

/// Configuration management actions
#[derive(Subcommand)]
enum ConfigAction {
    /// Print default configuration with all comments
    Print {
        /// Print all options including defaults (not just non-default values)
        #[arg(long)]
        all: bool,
    },
}

/// Xinference registry actions
#[derive(Subcommand)]
enum XinferenceAction {
    /// Sync Xinference model registry snapshots to the local cache
    SyncModels {
        /// Xinference base URL
        #[arg(long, default_value = "http://127.0.0.1:9997")]
        endpoint: String,

        /// Comma-separated registry types to sync (embedding,rerank,audio,video,image,llm)
        #[arg(long, default_value = "embedding,rerank,audio,video")]
        types: String,

        /// Write snapshots to disk (default is dry-run)
        #[arg(long)]
        write: bool,

        /// Skip the update/refresh call (list only)
        #[arg(long)]
        skip_update: bool,

        /// Override cache directory (defaults to ~/.librarian/xinference)
        #[arg(long)]
        cache_dir: Option<PathBuf>,

        /// HTTP retries for each request
        #[arg(long, default_value = "3")]
        retries: usize,

        /// Timeout in seconds for each request
        #[arg(long, default_value = "30")]
        timeout_secs: u64,
    },
}

/// Database management actions
#[derive(Subcommand)]
enum DbAction {
    /// Initialize/create the Qdrant collection
    Init,

    /// Show Qdrant collection status
    Status,

    /// Check collection health and configuration consistency
    Check,

    /// Reset the collection (delete all vectors and recreate)
    Reset {
        /// Skip confirmation prompt
        #[arg(long)]
        yes: bool,
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
        log_error_chain(&e);
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

    let is_mcp = matches!(&cli.command, &Commands::Mcp);
    let use_progress_writer = std::io::stderr().is_terminal() && !is_mcp;
    let writer: BoxMakeWriter = if use_progress_writer {
        BoxMakeWriter::new(LogWriterFactory)
    } else {
        BoxMakeWriter::new(std::io::stderr)
    };
    let fmt_layer = fmt::layer().with_writer(writer);

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(filter)
        .init();

    debug!(
        command = command_label(&cli.command),
        "Dispatching librarian command"
    );

    // Handle init command specially (doesn't need existing config)
    if matches!(cli.command, Commands::Init { .. }) {
        return handle_init(cli).await;
    }

    // Handle config command (doesn't need existing config)
    if let Commands::Config { action } = cli.command {
        return handle_config_action(action);
    }

    // Handle completions command (doesn't need config/db/store)
    if let Commands::Completions { shell } = cli.command {
        let mut cmd = Cli::command();
        generate(shell, &mut cmd, "librarian", &mut std::io::stdout());
        print_completion_extras(shell);
        return Ok(());
    }

    // Load configuration
    let config = load_config(cli.config.as_deref()).await?;

    debug!(
        config_path = %config.paths.config_file.display(),
        db_path = %config.paths.db_file.display(),
        "Resolved config paths"
    );

    debug!(
        qdrant_url = %config.qdrant_url,
        collection = %config.collection_name,
        "Resolved Qdrant configuration"
    );

    if config.qdrant_api_key_env.is_empty() {
        debug!("Qdrant API key env var not configured");
    } else {
        let present = std::env::var(&config.qdrant_api_key_env).is_ok();
        debug!(
            qdrant_api_key_env = %config.qdrant_api_key_env,
            present,
            "Checked Qdrant API key env var"
        );
    }

    // Initialize components
    let db = MetaDb::new(&config.paths.db_file).await?;

    if is_mcp {
        debug!(
            embedding_model = %config.embedding.model,
            embedding_backend = %config.embedding.backend,
            embedding_url = %config.embedding.url,
            custom_embedding_url = %config.embedding.custom.url,
            "Resolved embedding configuration for MCP startup"
        );

        let dimension = resolve_mcp_store_dimension(&config, &db).await?;
        debug!(
            dimension = dimension.dimension,
            source = dimension.source,
            "Resolved embedding dimension for MCP store"
        );

        let api_key = config.qdrant_api_key();
        let store = QdrantStore::new(
            &config.qdrant_url,
            &config.collection_name,
            dimension.dimension,
            None,
            api_key.as_deref(),
        )
        .await?;

        let server = McpServer::new(config, db, store);
        server
            .run()
            .await
            .map_err(|e| librarian::error::Error::McpProtocol(e.to_string()))?;
        return Ok(());
    }

    // Resolve embedding config and only create embedder when needed
    let embedding_config = config.resolve_embedding_config().await?;

    // Commands that don't need a store connection
    if let Commands::Xinference { action } = cli.command {
        handle_xinference_action(&config, action, cli.json).await?;
        return Ok(());
    }

    if let Commands::Db { action } = cli.command {
        handle_db_action(&config, action, cli.json).await?;
        return Ok(());
    }

    let needs_embedder = matches!(
        cli.command,
        Commands::Ingest { .. }
            | Commands::Query { .. }
            | Commands::Reindex { .. }
            | Commands::Update { .. }
    );
    let embedder = if needs_embedder {
        Some(create_embedder_auto(&embedding_config, &config.paths.base_dir).await?)
    } else {
        None
    };

    // Determine if this command needs validated connection (write operations)
    let needs_validation = matches!(
        cli.command,
        Commands::Ingest { .. }
            | Commands::Reindex { .. }
            | Commands::Update { .. }
            | Commands::Remove { .. }
    ) || matches!(&cli.command, Commands::Prune { remove_orphans, .. } if *remove_orphans);

    let api_key = config.qdrant_api_key();
    let store = if needs_validation {
        // Write operations use validated connection
        debug!(
            "Validating collection '{}' for write operation...",
            config.collection_name
        );
        QdrantStore::connect_validated(&config, &embedding_config, &db).await?
    } else {
        // Read-only operations use regular connection
        QdrantStore::new(
            &config.qdrant_url,
            &config.collection_name,
            embedding_config.dimension,
            Some(&embedding_config),
            api_key.as_deref(),
        )
        .await?
    };

    // Handle commands
    match cli.command {
        Commands::Init { .. } => unreachable!(),

        Commands::Ingest { source } => {
            let embedder = embedder
                .as_ref()
                .expect("embedder must be initialized for ingest");
            handle_ingest(
                &config,
                &embedding_config,
                embedder.as_ref(),
                &db,
                &store,
                source,
            )
            .await?;
        }

        Commands::Query {
            query,
            limit,
            min_score,
            source,
            dedupe,
        } => {
            let embedder = embedder
                .as_ref()
                .expect("embedder must be initialized for query");
            let options = QueryOptions {
                k: Some(limit),
                min_score,
                source_ids: source,
                dedupe_docs: dedupe,
                ..Default::default()
            };

            let results = cmd_query(
                &config,
                &embedding_config,
                embedder.as_ref(),
                &db,
                &store,
                &query,
                options,
            )
            .await?;

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

        Commands::Sources {
            ids_only,
            completion,
        } => {
            let sources = cmd_list_sources(&db).await?;

            if let Some(shell) = completion {
                print_source_completions(&sources, shell);
            } else if ids_only {
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
            let embedder = embedder
                .as_ref()
                .expect("embedder must be initialized for reindex");
            let options = ReindexOptions {
                source_ids: source,
                batch_size,
            };

            let stats = cmd_reindex(
                &config,
                &embedding_config,
                &db,
                &store,
                embedder.as_ref(),
                options,
            )
            .await?;

            if cli.json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                print_reindex_stats(&stats);
            }
        }

        Commands::Update { source, skip_prune } => {
            let embedder = embedder
                .as_ref()
                .expect("embedder must be initialized for update");
            let options = UpdateOptions {
                source_ids: source,
                prune_orphans: !skip_prune,
            };

            let stats = cmd_update(
                &config,
                &embedding_config,
                embedder.as_ref(),
                &db,
                &store,
                options,
            )
            .await?;

            if cli.json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                print_update_stats(&stats);
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

        Commands::Rename { source_id, name } => {
            let updated = cmd_rename_source(&db, &source_id, name).await?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&updated)?);
            } else {
                println!(
                    "✓ Renamed source '{}': {}",
                    updated.id,
                    updated.name.as_deref().unwrap_or(&updated.uri)
                );
            }
        }

        Commands::Config { action } => {
            handle_config_action(action)?;
        }
        Commands::Db { .. } | Commands::Xinference { .. } => unreachable!(),

        Commands::Mcp => unreachable!(),

        Commands::Completions { .. } => unreachable!(),
    }

    Ok(())
}

fn handle_config_action(action: ConfigAction) -> Result<()> {
    match action {
        ConfigAction::Print { all: _ } => {
            use librarian::config::render_config_toml;
            use std::collections::HashSet;

            let config = Config::default();
            let defaults = Config::default();
            // Show all fields as non-default by using empty irrelevant set
            let irrelevant = HashSet::new();
            let rendered = render_config_toml(&config, &defaults, &irrelevant);
            println!("{}", rendered);
            Ok(())
        }
    }
}

async fn handle_xinference_action(
    config: &Config,
    action: XinferenceAction,
    json: bool,
) -> Result<()> {
    match action {
        XinferenceAction::SyncModels {
            endpoint,
            types,
            write,
            skip_update,
            cache_dir,
            retries,
            timeout_secs,
        } => {
            let out_dir = cache_dir.unwrap_or_else(|| config.paths.base_dir.join("xinference"));
            let report = cmd_xinference_sync_models(XinferenceSyncOptions {
                endpoint,
                types,
                out_dir: out_dir.clone(),
                write,
                skip_update,
                retries,
                timeout_secs,
            })
            .await?;

            if json {
                let payload = serde_json::json!({
                    "out_dir": out_dir,
                    "write": write,
                    "has_changes": report.has_changes,
                    "diffs": report.diffs.iter().map(|diff| {
                        serde_json::json!({
                            "type": diff.registry_type,
                            "total_old": diff.total_old,
                            "total_new": diff.total_new,
                            "added": diff.added,
                            "removed": diff.removed,
                        })
                    }).collect::<Vec<_>>(),
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else {
                println!("Xinference registry sync:");
                for diff in &report.diffs {
                    println!(
                        "  {}: {} → {} (added {}, removed {})",
                        diff.registry_type,
                        diff.total_old,
                        diff.total_new,
                        diff.added.len(),
                        diff.removed.len()
                    );
                }

                if write {
                    println!("  Updated snapshots at {}", out_dir.display());
                } else {
                    println!("  Dry-run (no files written)");
                }
            }

            if !write && report.has_changes {
                return Err(librarian::error::Error::Config(
                    "Xinference registry snapshots are out of date. Re-run with --write to update the local cache.".to_string(),
                ));
            }
        }
    }

    Ok(())
}

fn command_label(command: &Commands) -> &'static str {
    match command {
        Commands::Init { .. } => "init",
        Commands::Ingest { .. } => "ingest",
        Commands::Query { .. } => "query",
        Commands::Status => "status",
        Commands::Sources { .. } => "sources",
        Commands::Prune { .. } => "prune",
        Commands::Reindex { .. } => "reindex",
        Commands::Update { .. } => "update",
        Commands::Remove { .. } => "remove",
        Commands::Rename { .. } => "rename",
        Commands::Mcp => "mcp",
        Commands::Completions { .. } => "completions",
        Commands::Db { .. } => "db",
        Commands::Config { .. } => "config",
        Commands::Xinference { .. } => "xinference",
    }
}

fn log_error_chain(err: &dyn std::error::Error) {
    error!("{}", err);
    let mut source = err.source();
    while let Some(cause) = source {
        error!("caused by: {}", cause);
        source = cause.source();
    }
}

struct StoreDimension {
    dimension: usize,
    source: &'static str,
}

async fn resolve_mcp_store_dimension(config: &Config, db: &MetaDb) -> Result<StoreDimension> {
    if let Some(dimension) = config.embedding.dimension {
        return Ok(StoreDimension {
            dimension,
            source: "config.embedding.dimension",
        });
    }

    if let Some(dimension) = config.embedding.custom.dimension {
        return Ok(StoreDimension {
            dimension,
            source: "config.embedding.custom.dimension",
        });
    }

    if let Some(record) = db.get_collection_config(&config.collection_name).await? {
        return Ok(StoreDimension {
            dimension: record.vector_dimension as usize,
            source: "metadata.collection_config",
        });
    }

    if let Some(spec) = embedding_model_spec(&config.embedding.model) {
        if let Some(dimension) = spec.default_dimension {
            return Ok(StoreDimension {
                dimension,
                source: "model_registry",
            });
        }
    }

    Err(librarian::error::Error::Config(
        "Embedding dimension could not be resolved for MCP startup. Set embedding.dimension in config.toml or ensure collection metadata exists."
            .to_string(),
    ))
}
#[allow(clippy::print_literal)]
fn print_completion_extras(shell: Shell) {
    match shell {
        Shell::Bash => {
            println!();
            println!("# Dynamic completion for 'librarian remove' source IDs");
            println!("_librarian_dynamic() {{");
            println!("    local cur prev words cword");
            println!("    if declare -F _init_completion >/dev/null; then");
            println!("        _init_completion -n : || return");
            println!("    else");
            println!("        cur=\"${{COMP_WORDS[COMP_CWORD]}}\"");
            println!("        words=(\"${{COMP_WORDS[@]}}\")");
            println!("        cword=$COMP_CWORD");
            println!("    fi");
            println!("    local remove_index=-1");
            println!("    for i in \"${{!words[@]}}\"; do");
            println!("        if [[ \"${{words[i]}}\" == \"remove\" ]]; then");
            println!("            remove_index=$i");
            println!("            break");
            println!("        fi");
            println!("    done");
            println!("    if [[ $remove_index -ge 0 && $cword -eq $((remove_index + 1)) ]]; then");
            println!(
                "        COMPREPLY=( $(compgen -W \"$(librarian sources --completion bash 2>/dev/null)\" -- \"$cur\") )"
            );
            println!("        return 0");
            println!("    fi");
            println!("    _librarian \"$@\"");
            println!("}}");
            println!(
                "if [[ \"${{BASH_VERSINFO[0]}}\" -eq 4 && \"${{BASH_VERSINFO[1]}}\" -ge 4 || \"${{BASH_VERSINFO[0]}}\" -gt 4 ]]; then"
            );
            println!(
                "    complete -F _librarian_dynamic -o nosort -o bashdefault -o default librarian"
            );
            println!("else");
            println!("    complete -F _librarian_dynamic -o bashdefault -o default librarian");
            println!("fi");
        }
        Shell::Zsh => {
            println!();
            println!("# Dynamic completion for 'librarian remove' source IDs");
            println!("_librarian_source_ids() {{");
            println!("    local -a entries");
            println!(
                "    entries=(\"${{(@f)$(librarian sources --completion zsh 2>/dev/null)}}\")"
            );
            println!("    _describe -t sources 'source ids' entries");
            println!("}}");
            println!("compdef _librarian_source_ids 'librarian remove'");
        }
        Shell::Fish => {
            println!();
            println!("# Dynamic completion for 'librarian remove' source IDs");
            println!(
                "complete -c librarian -n '__fish_seen_subcommand_from remove' -a '(librarian sources --completion fish 2>/dev/null)'"
            );
        }
        _ => {}
    }
}

async fn handle_init(cli: Cli) -> Result<()> {
    let Commands::Init {
        force,
        non_interactive,
        yes,
    } = cli.command
    else {
        unreachable!()
    };

    // Get the base directory: if user specifies config file, use its parent dir
    // Otherwise use default base dir
    let (base_dir, config_path) = if let Some(path) = cli.config {
        let base = path
            .parent()
            .map(PathBuf::from)
            .unwrap_or_else(Config::default_base_dir);
        let config = if path.extension().is_some_and(|e| e == "toml") {
            path // User specified a .toml file
        } else {
            path.join("config.toml") // User specified a directory
        };
        (base, config)
    } else {
        let base = Config::default_base_dir();
        (base.clone(), base.join("config.toml"))
    };

    cmd_init(librarian::commands::InitOptions {
        base_dir,
        config_path,
        force,
        non_interactive,
        yes,
    })
    .await?;

    Ok(())
}

async fn handle_db_action(config: &Config, action: DbAction, json: bool) -> Result<()> {
    let embedding_config = config.resolve_embedding_config().await?;
    let db = MetaDb::new(&config.paths.db_file).await?;

    match action {
        DbAction::Init => {
            ensure_collection_with_autostart(config, &embedding_config).await?;
            if json {
                println!(r#"{{"status": "ok", "message": "Collection initialized"}}"#);
            } else {
                println!("✓ Qdrant collection initialized");
            }
        }
        DbAction::Check => {
            let store = qdrant_store_or_error(config, &embedding_config).await?;
            use librarian::store::{CollectionConfig, ValidationResult};

            let expected =
                CollectionConfig::from_embedding_config(&config.collection_name, &embedding_config);

            let validation = store.validate_collection_state(&db, &expected).await?;
            let stored_config = db.get_collection_config(&config.collection_name).await?;

            if json {
                let status = serde_json::json!({
                    "collection": config.collection_name,
                    "validation_result": format!("{:?}", validation),
                    "stored_config": stored_config.as_ref().map(|c| serde_json::json!({
                        "dimension": c.vector_dimension,
                        "model": c.embedding_model,
                        "family": c.embedding_family,
                        "verified_at": c.verified_at,
                    })),
                    "current_config": {
                        "dimension": embedding_config.dimension,
                        "model": &embedding_config.model_id,
                        "family": &embedding_config.family,
                    },
                });
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                match validation {
                    ValidationResult::Valid => {
                        println!("✓ Collection '{}' is healthy", config.collection_name);
                        println!("  Dimension: {}", embedding_config.dimension);
                        println!(
                            "  Model: {} ({})",
                            embedding_config.model_id, embedding_config.family
                        );
                        if let Some(cfg) = stored_config {
                            if let Some(verified) = cfg.verified_at {
                                println!("  Last verified: {}", verified);
                            }
                        }
                    }
                    ValidationResult::NotFound => {
                        println!("⚠ Collection '{}' does not exist", config.collection_name);
                        println!("  Run 'librarian db init' to create it");
                    }
                    ValidationResult::NotFoundButConfigured {
                        stored_dimension,
                        stored_model,
                    } => {
                        println!(
                            "⚠ Collection '{}' is missing from Qdrant",
                            config.collection_name
                        );
                        println!(
                            "  Previously configured: {} dimensions, model '{}'",
                            stored_dimension, stored_model
                        );
                        if stored_dimension != embedding_config.dimension {
                            println!(
                                "  ❌ Current config has different dimension: {}",
                                embedding_config.dimension
                            );
                            println!(
                                "  Remediation: Run 'librarian db reset' to align with current config"
                            );
                        } else {
                            println!("  Run 'librarian db init' to recreate");
                        }
                    }
                    ValidationResult::DimensionMismatch {
                        qdrant_dimension,
                        expected_dimension,
                        source,
                    } => {
                        println!(
                            "❌ Collection '{}' has dimension mismatch",
                            config.collection_name
                        );
                        println!("  Qdrant dimension: {}", qdrant_dimension);
                        println!("  Expected dimension: {} ({})", expected_dimension, source);
                        println!("  Remediation: Change 'collection_name' in config to use a new collection,");
                        println!(
                            "               or run 'librarian db reset' to delete and recreate (WARNING: data loss)"
                        );
                    }
                    ValidationResult::ConfigConflict {
                        stored_dimension,
                        stored_model,
                        expected_dimension,
                        expected_model,
                    } => {
                        println!(
                            "❌ Collection '{}' has configuration conflict",
                            config.collection_name
                        );
                        println!(
                            "  Stored: {} dimensions, model '{}'",
                            stored_dimension, stored_model
                        );
                        println!(
                            "  Current: {} dimensions, model '{}'",
                            expected_dimension, expected_model
                        );
                        println!(
                            "  Remediation: Run 'librarian db reset' and reindex, or revert config"
                        );
                    }
                }
            }
        }
        DbAction::Status => {
            let store = qdrant_store_or_error(config, &embedding_config).await?;
            let stored_config = db.get_collection_config(&config.collection_name).await?;

            match store.get_collection_info().await? {
                Some(info) => {
                    if json {
                        let status = serde_json::json!({
                            "exists": true,
                            "points_count": info.points_count,
                            "indexed_vectors_count": info.indexed_vectors_count,
                            "status": info.status,
                            "config": stored_config.as_ref().map(|c| serde_json::json!({
                                "dimension": c.vector_dimension,
                                "model": c.embedding_model,
                                "family": c.embedding_family,
                                "verified_at": c.verified_at,
                            })),
                            "current_dimension": embedding_config.dimension,
                            "current_model": &embedding_config.model_id,
                        });
                        println!("{}", serde_json::to_string_pretty(&status)?);
                    } else {
                        println!("Qdrant Collection Status:");
                        println!("  Status: {}", info.status);
                        println!("  Points: {}", info.points_count);
                        println!("  Indexed Vectors: {}", info.indexed_vectors_count);
                        if let Some(cfg) = stored_config {
                            println!("  Configured dimension: {}", cfg.vector_dimension);
                            println!(
                                "  Configured model: {} ({})",
                                cfg.embedding_model, cfg.embedding_family
                            );
                            if let Some(verified) = cfg.verified_at {
                                println!("  Last verified: {}", verified);
                            }
                        }
                    }
                }
                None => {
                    if json {
                        println!(r#"{{"exists": false}}"#);
                    } else {
                        println!(
                            "Collection does not exist. Run 'librarian db init' to create it."
                        );
                    }
                }
            }
        }
        DbAction::Reset { yes } => {
            let store = qdrant_store_or_error(config, &embedding_config).await?;
            if !yes {
                eprintln!("⚠️  This will delete ALL indexed data!");
                eprintln!("Run with --yes to confirm.");
                std::process::exit(1);
            }
            store.reset_collection().await?;
            if json {
                println!(r#"{{"status": "ok", "message": "Collection reset"}}"#);
            } else {
                println!("✓ Qdrant collection reset (all data deleted and collection recreated)");
            }
        }
    }

    Ok(())
}

fn is_qdrant_connection_refused(err: &Error) -> bool {
    match err {
        Error::Qdrant(msg) => {
            msg.contains("Connection refused")
                || msg.contains("tcp connect error")
                || msg.contains("Failed to connect")
        }
        _ => false,
    }
}

async fn ensure_collection_with_autostart(
    config: &Config,
    embedding_config: &librarian::config::ResolvedEmbeddingConfig,
) -> Result<()> {
    let store = QdrantStore::connect(config, embedding_config).await?;
    match store.ensure_collection().await {
        Ok(()) => Ok(()),
        Err(err) if is_qdrant_connection_refused(&err) => {
            ensure_local_qdrant_running(config).await?;
            let store = QdrantStore::connect(config, embedding_config).await?;
            store.ensure_collection().await
        }
        Err(err) => Err(err),
    }
}

async fn qdrant_store_or_error(
    config: &Config,
    embedding_config: &librarian::config::ResolvedEmbeddingConfig,
) -> Result<QdrantStore> {
    match QdrantStore::connect(config, embedding_config).await {
        Ok(store) => Ok(store),
        Err(err) if is_qdrant_connection_refused(&err) => Err(Error::Config(format!(
            "Qdrant is not reachable at {}. Start Qdrant and try again.",
            config.qdrant_url
        ))),
        Err(err) => Err(err),
    }
}

async fn ensure_local_qdrant_running(config: &Config) -> Result<()> {
    if !is_local_qdrant_url(&config.qdrant_url) {
        return Err(Error::Config(format!(
            "Qdrant is not reachable at {} and auto-start is only supported for local URLs.",
            config.qdrant_url
        )));
    }

    let base_dir = if config.paths.base_dir.as_os_str().is_empty() {
        Config::default_base_dir()
    } else {
        config.paths.base_dir.clone()
    };
    let storage_dir = base_dir.join("qdrant_storage");
    fs::create_dir_all(&storage_dir)?;

    let compose_path = base_dir.join("qdrant-compose.yml");
    let compose_contents = build_qdrant_compose(&storage_dir, &config.qdrant_api_key_env);
    if fs::read_to_string(&compose_path).ok().as_deref() != Some(&compose_contents) {
        fs::write(&compose_path, compose_contents)?;
    }

    start_qdrant_compose(&compose_path).await?;
    wait_for_qdrant_ready(&config.qdrant_url).await?;
    Ok(())
}

fn is_local_qdrant_url(qdrant_url: &str) -> bool {
    let Ok(url) = Url::parse(qdrant_url) else {
        return false;
    };
    matches!(
        url.host_str(),
        Some("127.0.0.1") | Some("localhost") | Some("0.0.0.0") | Some("::1")
    )
}

fn build_qdrant_compose(storage_dir: &Path, api_key_env: &str) -> String {
    let mut compose = format!(
        r#"version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant
    container_name: qdrant_librarian
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - "{storage}:/qdrant/storage"
    restart: unless-stopped
"#,
        storage = storage_dir.display()
    );

    if !api_key_env.is_empty() && std::env::var(api_key_env).is_ok() {
        compose.push_str("    environment:\n");
        compose.push_str(&format!(
            "      - QDRANT__SERVICE__API_KEY=${{{}}}\n",
            api_key_env
        ));
    }

    compose
}

async fn start_qdrant_compose(compose_path: &Path) -> Result<()> {
    let mut compose_cmd = Command::new("docker");
    compose_cmd
        .arg("compose")
        .arg("-f")
        .arg(compose_path)
        .arg("up")
        .arg("-d")
        .arg("qdrant");

    match compose_cmd.output().await {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            if docker_compose_unavailable(&stderr, &stdout) {
                start_qdrant_compose_legacy(compose_path).await
            } else {
                Err(Error::Config(format!(
                    "Failed to start Qdrant via docker compose: {}",
                    stderr.trim()
                )))
            }
        }
        Err(_) => start_qdrant_compose_legacy(compose_path).await,
    }
}

fn docker_compose_unavailable(stderr: &str, stdout: &str) -> bool {
    let combined = format!("{}{}", stdout, stderr).to_lowercase();
    combined.contains("unknown command")
        || combined.contains("not a docker command")
        || combined.contains("docker: 'compose'")
}

async fn start_qdrant_compose_legacy(compose_path: &Path) -> Result<()> {
    let mut legacy_cmd = Command::new("docker-compose");
    legacy_cmd
        .arg("-f")
        .arg(compose_path)
        .arg("up")
        .arg("-d")
        .arg("qdrant");
    let output = legacy_cmd.output().await.map_err(|e| {
        Error::Config(format!(
            "Docker is required to auto-start Qdrant but was not found: {}",
            e
        ))
    })?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(Error::Config(format!(
            "Failed to start Qdrant via docker-compose: {}",
            stderr.trim()
        )))
    }
}

async fn wait_for_qdrant_ready(qdrant_url: &str) -> Result<()> {
    let health_url = qdrant_health_url(qdrant_url).ok_or_else(|| {
        Error::Config("Unable to derive Qdrant health URL for readiness check".to_string())
    })?;
    let client = reqwest::Client::new();
    for _ in 0..30 {
        if let Ok(resp) = client.get(&health_url).send().await {
            if resp.status().is_success() {
                return Ok(());
            }
        }
        sleep(std::time::Duration::from_millis(500)).await;
    }
    Err(Error::Config(format!(
        "Qdrant did not become ready at {}",
        health_url
    )))
}

fn qdrant_health_url(qdrant_url: &str) -> Option<String> {
    let url = Url::parse(qdrant_url).ok()?;
    let host = url.host_str()?;
    let scheme = url.scheme();
    let port = url.port().unwrap_or(6334);
    let rest_port = if port == 6334 { 6333 } else { port };
    Some(format!("{}://{}:{}/health", scheme, host, rest_port))
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
    embedding: &librarian::config::ResolvedEmbeddingConfig,
    embedder: &dyn librarian::embed::Embedder,
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
                config,
                embedding,
                embedder,
                db,
                store,
                &path,
                name,
                RunOperation::Ingest,
                true,
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
                config,
                embedding,
                embedder,
                db,
                store,
                &url,
                name,
                overrides,
                RunOperation::Ingest,
                true,
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
                embedding,
                embedder,
                db,
                store,
                &url,
                name,
                max_pages,
                RunOperation::Ingest,
                true,
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
