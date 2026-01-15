//! Query command implementation

use crate::config::Config;
use crate::embed::create_embedder;
use crate::error::Result;
use crate::meta::MetaDb;
use crate::rank::{RankedResult, Ranker};
use crate::store::{QdrantStore, SearchFilter};
use serde::Serialize;
use tracing::{debug, info};

/// Query options
#[derive(Debug, Clone, Default)]
pub struct QueryOptions {
    /// Number of results to return
    pub k: Option<usize>,
    /// Minimum score threshold
    pub min_score: Option<f32>,
    /// Filter by source IDs
    pub source_ids: Option<Vec<String>>,
    /// Filter by source types
    pub source_types: Option<Vec<String>>,
    /// Filter by path prefix
    pub path_prefix: Option<String>,
    /// Deduplicate by document
    pub dedupe_docs: bool,
}

/// Query result for CLI display
#[derive(Debug, Clone, Serialize)]
pub struct QueryResult {
    pub results: Vec<RankedResult>,
    pub query: String,
    pub total_chunks_searched: usize,
}

/// Execute a query
pub async fn cmd_query(
    config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
    query: &str,
    options: QueryOptions,
) -> Result<QueryResult> {
    info!("Querying: {}", query);

    let k = options.k.unwrap_or(config.query.default_k);
    let min_score = options.min_score.unwrap_or(config.query.min_score);

    // Create embedder and embed query
    let embedder = create_embedder(&config.embedding)?;
    let query_embeddings = embedder.embed(vec![query.to_string()]).await?;
    let query_vector = query_embeddings
        .into_iter()
        .next()
        .ok_or_else(|| crate::error::Error::Embedding("No embedding returned".to_string()))?;

    // Build search filter
    let filter = if options.source_ids.is_some()
        || options.source_types.is_some()
        || options.path_prefix.is_some()
    {
        Some(SearchFilter {
            source_ids: options.source_ids,
            source_types: options.source_types,
            path_prefix: options.path_prefix,
        })
    } else {
        None
    };

    // Search Qdrant
    let search_results = store.search(query_vector, k * 2, filter).await?;
    debug!("Got {} raw results from Qdrant", search_results.len());

    // Rank results
    let ranker = Ranker::new(config.query.bm25_weight);
    let mut ranked = ranker.rank_vector_only(search_results);

    // Enrich with chunk text from SQLite
    for result in &mut ranked {
        if let Ok(Some(chunk)) = db.get_chunk_by_point_id(&result.id).await {
            result.chunk_text = chunk.chunk_text;
        }
    }

    // Filter by score
    ranked = ranker.filter_by_score(ranked, min_score);

    // Deduplicate if requested
    if options.dedupe_docs {
        ranked = ranker.dedupe_by_doc(ranked);
    }

    // Limit to k results
    ranked.truncate(k);

    let total = ranked.len();
    info!("Returning {} results", total);

    Ok(QueryResult {
        results: ranked,
        query: query.to_string(),
        total_chunks_searched: total,
    })
}

/// Print query results to console
pub fn print_query_results(result: &QueryResult) {
    println!("\n🔍 Query: {}\n", result.query);
    println!("Found {} results:\n", result.results.len());

    for (i, r) in result.results.iter().enumerate() {
        println!("{}. [score: {:.3}] {}", i + 1, r.score, r.doc_uri);

        if let Some(title) = &r.title {
            println!("   Title: {}", title);
        }

        if let Some(headings) = &r.headings {
            if !headings.is_empty() {
                println!("   Section: {}", headings.join(" > "));
            }
        }

        // Print preview (first 200 chars)
        let preview = if r.chunk_text.len() > 200 {
            format!("{}...", &r.chunk_text[..200].trim())
        } else {
            r.chunk_text.trim().to_string()
        };
        println!("   {}\n", preview.replace('\n', " "));
    }
}
