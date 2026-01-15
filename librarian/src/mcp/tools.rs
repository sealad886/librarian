//! MCP tool definitions and handlers

use super::types::ToolResult;
use crate::commands::{cmd_list_sources, cmd_query, QueryOptions};
use crate::config::Config;
use crate::meta::MetaDb;
use crate::store::QdrantStore;
use serde_json::{json, Value};
use std::collections::HashMap;

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
                output.push_str(&format!(
                    "## Result {} (score: {:.2})\n",
                    i + 1,
                    r.score
                ));
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
                output.push_str(&format!(
                    "- **{}** [{}]\n  - URI: {}\n  - Documents: {}, Chunks: {}\n\n",
                    source.name.as_deref().unwrap_or(&source.id),
                    source.source_type,
                    source.uri,
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

    let (qdrant_status, qdrant_points) = match store.get_stats().await {
        Ok(stats) => ("Connected".to_string(), stats.points_count),
        Err(e) => (format!("Error: {}", e), 0),
    };

    let output = format!(
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

    ToolResult::text(output)
}
