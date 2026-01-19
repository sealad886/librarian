//! MCP tool definitions and handlers

use super::types::ToolResult;
use crate::commands::{
    cmd_ingest_dir, cmd_ingest_sitemap, cmd_ingest_url, cmd_list_sources, cmd_query, cmd_reindex,
    cmd_update, CrawlOverrides, QueryOptions, ReindexOptions, UpdateOptions,
};
use crate::config::Config;
use crate::embed::FastEmbedder;
use crate::error::Error;
use crate::meta::{MetaDb, RunOperation, SourceType};
use crate::store::QdrantStore;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use tracing::error;

/// Tool definition for MCP
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// Get all available tool definitions
pub fn get_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "rag_search".to_string(),
            description: "Search the RAG index for relevant documentation chunks. Returns the most relevant passages matching your query.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query - natural language question or keywords"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5, max: 20)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "source_ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional: Filter to specific source IDs"
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum similarity score (0-1, default: 0.5)",
                        "default": 0.5,
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "rag_sources".to_string(),
            description: "List all registered documentation sources in the RAG index.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "rag_status".to_string(),
            description: "Get the current status of the RAG system including database stats and Qdrant connection.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "rag_ingest_source".to_string(),
            description: "Start an ingestion run for a source (dir/url/sitemap). Runs asynchronously.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "source_type": {
                        "type": "string",
                        "enum": ["dir", "url", "sitemap"],
                        "description": "Type of source to ingest"
                    },
                    "uri": {
                        "type": "string",
                        "description": "Directory path or URL to ingest"
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional friendly name for the source"
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Maximum pages to crawl (for url/sitemap)"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum crawl depth (for url sources)"
                    },
                    "path_prefix": {
                        "type": "string",
                        "description": "Limit URL crawling to this path prefix (for url sources)"
                    }
                },
                "required": ["source_type", "uri"]
            }),
        },
        ToolDefinition {
            name: "rag_update".to_string(),
            description: "Trigger an update run for registered sources asynchronously.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "source_ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional list of source IDs to update (default: all)"
                    },
                    "skip_prune": {
                        "type": "boolean",
                        "description": "If true, skip pruning orphaned vectors after update",
                        "default": false
                    }
                }
            }),
        },
        ToolDefinition {
            name: "rag_reindex".to_string(),
            description: "Trigger a full reindex of chunks for selected sources asynchronously.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "source_ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional list of source IDs to reindex (default: all)"
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size to use for embedding during reindex",
                        "default": 32
                    }
                }
            }),
        },
    ]
}

/// Handle a tool call
pub async fn handle_tool_call(
    name: &str,
    arguments: &HashMap<String, Value>,
    config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
) -> ToolResult {
    match name {
        "rag_search" => handle_search(arguments, config, db, store).await,
        "rag_sources" => handle_sources(db).await,
        "rag_status" => handle_status(config, db, store).await,
        "rag_ingest_source" => handle_ingest_trigger(arguments, config).await,
        "rag_update" => handle_update_trigger(arguments, config).await,
        "rag_reindex" => handle_reindex_trigger(arguments, config).await,
        _ => ToolResult::error(format!("Unknown tool: {}", name)),
    }
}

/// Handle rag_search tool
async fn handle_search(
    arguments: &HashMap<String, Value>,
    config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
) -> ToolResult {
    // Extract query parameter
    let query = match arguments.get("query") {
        Some(Value::String(q)) => q.clone(),
        _ => return ToolResult::error("Missing required parameter: query"),
    };

    // Extract optional parameters
    let limit = arguments
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|v| v.min(20) as usize)
        .unwrap_or(5);

    let source_ids = arguments.get("source_ids").and_then(|v| {
        v.as_array().map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
    });

    let min_score = arguments
        .get("min_score")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(0.5);

    // Build query options
    let options = QueryOptions {
        k: Some(limit),
        source_ids,
        min_score: Some(min_score),
        dedupe_docs: true,
        ..Default::default()
    };

    // Execute query
    match cmd_query(config, db, store, &query, options).await {
        Ok(result) => {
            if result.results.is_empty() {
                return ToolResult::text("No results found matching your query.");
            }

            // Format results as markdown
            let mut output = String::new();
            output.push_str(&format!("Found {} results:\n\n", result.results.len()));

            for (i, r) in result.results.iter().enumerate() {
                output.push_str(&format!("## Result {} (score: {:.2})\n", i + 1, r.score));
                output.push_str(&format!("**Source:** {}\n", r.doc_uri));
                if let Some(title) = &r.title {
                    output.push_str(&format!("**Title:** {}\n", title));
                }
                if let Some(headings) = &r.headings {
                    if !headings.is_empty() {
                        output.push_str(&format!("**Section:** {}\n", headings.join(" > ")));
                    }
                }
                output.push_str("\n```\n");
                output.push_str(&r.chunk_text);
                output.push_str("\n```\n\n");
            }

            ToolResult::text(output)
        }
        Err(e) => ToolResult::error(format!("Search failed: {}", e)),
    }
}

/// Handle rag_sources tool
async fn handle_sources(db: &MetaDb) -> ToolResult {
    match cmd_list_sources(db).await {
        Ok(sources) => {
            if sources.is_empty() {
                return ToolResult::text(
                    "No sources registered. Use 'librarian ingest' to add documentation sources.",
                );
            }

            let mut output = String::new();
            output.push_str(&format!("Registered Sources ({}):\n\n", sources.len()));

            for source in &sources {
                let display_name = source
                    .name
                    .as_deref()
                    .unwrap_or_else(|| source.uri.as_str());
                let last_updated = source.last_updated.as_deref().unwrap_or("unknown");
                let last_operation = source
                    .last_run
                    .as_ref()
                    .map(|r| format!("{} ({})", r.operation, r.status))
                    .unwrap_or_else(|| "none".to_string());
                output.push_str(&format!(
                    "- **{}** [{}]\n  - URI: {}\n  - State: {}\n  - Last update: {}\n  - Last operation: {}\n  - Documents: {}, Chunks: {}\n\n",
                    display_name,
                    source.source_type,
                    source.uri,
                    source.state,
                    last_updated,
                    last_operation,
                    source.stats.document_count,
                    source.stats.chunk_count
                ));
            }

            ToolResult::text(output)
        }
        Err(e) => ToolResult::error(format!("Failed to list sources: {}", e)),
    }
}

/// Handle rag_status tool
async fn handle_status(config: &Config, db: &MetaDb, store: &QdrantStore) -> ToolResult {
    let db_stats = match db.get_global_stats().await {
        Ok(s) => s,
        Err(e) => return ToolResult::error(format!("Failed to get DB stats: {}", e)),
    };

    let sources = match cmd_list_sources(db).await {
        Ok(s) => s,
        Err(e) => return ToolResult::error(format!("Failed to list sources: {}", e)),
    };

    let (qdrant_status, qdrant_points) = match store.get_stats().await {
        Ok(stats) => ("Connected".to_string(), stats.points_count),
        Err(e) => (format!("Error: {}", e), 0),
    };

    let mut output = format!(
        r#"RAG System Status:

**Qdrant:**
- URL: {}
- Collection: {}
- Status: {}
- Points: {}

**Embedding Model:** {}

**Database Stats:**
- Sources: {}
- Documents: {}
- Chunks: {}"#,
        config.qdrant_url,
        config.collection_name,
        qdrant_status,
        qdrant_points,
        config.embedding.model,
        db_stats.source_count,
        db_stats.document_count,
        db_stats.chunk_count
    );

    if !sources.is_empty() {
        output.push_str("\n\n**Sources:**\n");
        for source in &sources {
            let last_updated = source.last_updated.as_deref().unwrap_or("unknown");
            let last_operation = source
                .last_run
                .as_ref()
                .map(|r| format!("{} ({})", r.operation, r.status))
                .unwrap_or_else(|| "none".to_string());
            output.push_str(&format!(
                "- {} [{}]\n  - State: {}\n  - Last update: {}\n  - Last operation: {}\n  - Documents: {}, Chunks: {}\n",
                source.name.as_deref().unwrap_or_else(|| source.uri.as_str()),
                source.source_type,
                source.state,
                last_updated,
                last_operation,
                source.stats.document_count,
                source.stats.chunk_count
            ));
        }
    }

    ToolResult::text(output)
}

async fn handle_ingest_trigger(arguments: &HashMap<String, Value>, config: &Config) -> ToolResult {
    let source_type_str = match arguments.get("source_type").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return ToolResult::error("Missing required parameter: source_type"),
    };

    let source_type = match SourceType::from_str(source_type_str) {
        Ok(t) => t,
        Err(_) => return ToolResult::error("Invalid source_type. Use dir, url, or sitemap."),
    };

    let uri = match arguments.get("uri").and_then(|v| v.as_str()) {
        Some(u) => u.to_string(),
        None => return ToolResult::error("Missing required parameter: uri"),
    };

    let name = arguments
        .get("name")
        .and_then(|v| v.as_str())
        .map(ToString::to_string);
    let max_pages = arguments
        .get("max_pages")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    let max_depth = arguments
        .get("max_depth")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    let path_prefix = arguments
        .get("path_prefix")
        .and_then(|v| v.as_str())
        .map(ToString::to_string);

    let config_clone = config.clone();
    tokio::spawn(async move {
        if let Err(e) = run_ingest_background(
            config_clone,
            source_type,
            uri,
            name,
            max_pages,
            max_depth,
            path_prefix,
        )
        .await
        {
            error!(error=?e, "Background ingestion failed");
        }
    });

    ToolResult::text(format!(
        "Started {} ingestion. Check rag_status for progress.",
        source_type
    ))
}

async fn handle_update_trigger(arguments: &HashMap<String, Value>, config: &Config) -> ToolResult {
    let source_ids = parse_string_array(arguments.get("source_ids"));
    let prune_orphans = !arguments
        .get("skip_prune")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let config_clone = config.clone();
    tokio::spawn(async move {
        if let Err(e) = run_update_background(config_clone, source_ids, prune_orphans).await {
            error!(error=?e, "Background update failed");
        }
    });

    ToolResult::text("Update started. Check rag_status for progress.".to_string())
}

async fn handle_reindex_trigger(arguments: &HashMap<String, Value>, config: &Config) -> ToolResult {
    let source_ids = parse_string_array(arguments.get("source_ids"));
    let batch_size = arguments
        .get("batch_size")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(32);

    let config_clone = config.clone();
    tokio::spawn(async move {
        if let Err(e) = run_reindex_background(config_clone, source_ids, batch_size).await {
            error!(error=?e, "Background reindex failed");
        }
    });

    ToolResult::text("Reindex started. Check rag_status for progress.".to_string())
}

fn parse_string_array(value: Option<&Value>) -> Option<Vec<String>> {
    value.and_then(|v| {
        v.as_array().map(|arr| {
            arr.iter()
                .filter_map(|item| item.as_str().map(ToString::to_string))
                .collect::<Vec<_>>()
        })
    })
}

type AppResult<T> = std::result::Result<T, Error>;

async fn run_ingest_background(
    config: Config,
    source_type: SourceType,
    uri: String,
    name: Option<String>,
    max_pages: Option<u32>,
    max_depth: Option<u32>,
    path_prefix: Option<String>,
) -> AppResult<()> {
    let db = MetaDb::connect(&config).await?;
    db.init_schema().await?;
    let store = QdrantStore::connect(&config).await?;

    match source_type {
        SourceType::Dir => {
            let path = PathBuf::from(&uri);
            cmd_ingest_dir(
                &config,
                &db,
                &store,
                &path,
                name,
                RunOperation::Ingest,
                false,
            )
            .await?;
        }
        SourceType::Url => {
            let overrides = CrawlOverrides {
                max_pages,
                max_depth,
                path_prefix,
            };
            cmd_ingest_url(
                &config,
                &db,
                &store,
                &uri,
                name,
                overrides,
                RunOperation::Ingest,
                false,
            )
            .await?;
        }
        SourceType::Sitemap => {
            cmd_ingest_sitemap(
                &config,
                &db,
                &store,
                &uri,
                name,
                max_pages,
                RunOperation::Ingest,
                false,
            )
            .await?;
        }
    }

    Ok(())
}

async fn run_update_background(
    config: Config,
    source_ids: Option<Vec<String>>,
    prune_orphans: bool,
) -> AppResult<()> {
    let db = MetaDb::connect(&config).await?;
    db.init_schema().await?;
    let store = QdrantStore::connect(&config).await?;

    let options = UpdateOptions {
        source_ids,
        prune_orphans,
    };

    cmd_update(&config, &db, &store, options).await?;
    Ok(())
}

async fn run_reindex_background(
    config: Config,
    source_ids: Option<Vec<String>>,
    batch_size: usize,
) -> AppResult<()> {
    let db = MetaDb::connect(&config).await?;
    db.init_schema().await?;
    let store = QdrantStore::connect(&config).await?;
    let embedder = FastEmbedder::new(&config.embedding)?;

    let options = ReindexOptions {
        source_ids,
        batch_size,
    };

    cmd_reindex(&config, &db, &store, &embedder, options).await?;
    Ok(())
}
