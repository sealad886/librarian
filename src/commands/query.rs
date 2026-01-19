//! Query command implementation

use crate::config::Config;
use crate::embed::create_embedder;
use crate::error::Result;
use crate::meta::MetaDb;
use crate::rank::{RankedResult, Ranker};
use crate::rerank::{create_reranker, Reranker};
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

    // Optional reranking
    if config.reranker.enabled && !ranked.is_empty() {
        let reranker = create_reranker(&config.reranker)?;
        if config.reranker.supports_multimodal {
            ranked = apply_reranker(reranker.as_ref(), query, ranked, config.reranker.top_k).await?;
        } else {
            let (text_results, other_results): (Vec<_>, Vec<_>) = ranked
                .into_iter()
                .partition(|r| r.modality.as_deref().unwrap_or("text") == "text");

            let mut reranked_text =
                apply_reranker(reranker.as_ref(), query, text_results, config.reranker.top_k)
                    .await?;
            reranked_text.extend(other_results);
            ranked = reranked_text;
        }
    }

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

async fn apply_reranker(
    reranker: &dyn Reranker,
    query: &str,
    mut results: Vec<RankedResult>,
    top_k: usize,
) -> Result<Vec<RankedResult>> {
    if results.is_empty() {
        return Ok(results);
    }

    let docs: Vec<String> = results.iter().map(|r| r.chunk_text.clone()).collect();
    let mut reranked = reranker.rerank(query, docs).await?;

    if reranked.is_empty() {
        results.truncate(top_k);
        return Ok(results);
    }

    reranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ordered = Vec::new();
    for r in reranked {
        if let Some(item) = results.get(r.index) {
            let mut updated = item.clone();
            updated.score = r.score;
            ordered.push(updated);
        }
    }

    ordered.truncate(top_k);
    Ok(ordered)
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

        if r.modality.as_deref() == Some("image") {
            let label = r
                .media_url
                .as_deref()
                .unwrap_or_else(|| r.doc_uri.as_str());
            println!("   [image] {}\n", label);
        } else {
            let preview = if r.chunk_text.len() > 200 {
                format!("{}...", &r.chunk_text[..200].trim())
            } else {
                r.chunk_text.trim().to_string()
            };
            println!("   {}\n", preview.replace('\n', " "));
        }
    }
}
