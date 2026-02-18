//! Query command implementation

use crate::classify::{classify_query, QueryIntent};
use crate::config::{Config, ResolvedEmbeddingConfig};
use crate::embed::Embedder;
use crate::error::Result;
use crate::meta::MetaDb;
use crate::models::is_multimodal_reranker_model;
use crate::rank::{RankedResult, Ranker};
use crate::rerank::{create_reranker, Reranker};
use crate::store::{QdrantStore, SearchFilter};
use serde::Serialize;
use std::time::Instant;
use tracing::{debug, warn};

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
    pub result_count: usize,
    pub intent: QueryIntent,
}

/// Execute a query
pub async fn cmd_query(
    config: &Config,
    embedding: &ResolvedEmbeddingConfig,
    embedder: &dyn Embedder,
    db: &MetaDb,
    store: &QdrantStore,
    query: &str,
    options: QueryOptions,
) -> Result<QueryResult> {
    debug!("Querying: {}", query);
    let start_time = Instant::now();
    let intent = classify_query(query);
    debug!(intent = %intent, "Classified query intent");

    let k = options.k.unwrap_or(config.query.default_k);
    let min_score = options.min_score.unwrap_or(config.query.min_score);

    let query_embeddings = embedder.embed(vec![query.to_string()]).await?;
    let query_vector = query_embeddings
        .into_iter()
        .next()
        .ok_or_else(|| crate::error::Error::Embedding("No embedding returned".to_string()))?;

    // Capture source filter string for logging before moving
    let source_filter_str = options.source_ids.as_ref().map(|ids| ids.join(","));

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
    let total_searched = search_results.len();
    debug!("Got {} raw results from Qdrant", total_searched);

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
        let reranker = create_reranker(&config.reranker, &embedding.backend.url)?;
        if is_multimodal_reranker_model(&config.reranker.model) {
            ranked =
                apply_reranker(reranker.as_ref(), query, ranked, config.reranker.top_k).await?;
        } else {
            // Partition by whether chunk has meaningful text content for reranking
            // Audio/video transcript chunks have substantial text and should be reranked
            // Image/video keyframe chunks have minimal text and should be skipped
            const MIN_TEXT_LENGTH_FOR_RERANK: usize = 50;
            let (text_results, other_results): (Vec<_>, Vec<_>) = ranked
                .into_iter()
                .partition(|r| r.chunk_text.len() >= MIN_TEXT_LENGTH_FOR_RERANK);

            let mut reranked_text = apply_reranker(
                reranker.as_ref(),
                query,
                text_results,
                config.reranker.top_k,
            )
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
    let latency = start_time.elapsed();
    debug!("Returning {} results in {:?}", total, latency);

    // Log query for analytics (best-effort, don't fail the query)
    let top_score = ranked.first().map(|r| r.score);
    if let Err(e) = db
        .log_query(
            query,
            Some(&intent.to_string()),
            source_filter_str.as_deref(),
            total as i32,
            top_score,
            Some(latency.as_millis() as i64),
        )
        .await
    {
        warn!("Failed to log query: {}", e);
    }

    Ok(QueryResult {
        results: ranked,
        query: query.to_string(),
        total_chunks_searched: total_searched,
        result_count: total,
        intent,
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

fn chunk_preview(chunk_text: &str) -> String {
    if chunk_text.len() > 200 {
        // Find a safe UTF-8 character boundary: include chars starting before byte 200
        let end = chunk_text
            .char_indices()
            .take_while(|&(idx, _)| idx < 200)
            .last()
            .map(|(idx, ch)| idx + ch.len_utf8())
            .unwrap_or(0);
        format!("{}...", chunk_text[..end].trim())
    } else {
        chunk_text.trim().to_string()
    }
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

        match r.modality.as_deref() {
            Some("image") => {
                let label = r.media_url.as_deref().unwrap_or(r.doc_uri.as_str());
                println!("   [image] {}\n", label);
            }
            Some("audio") => {
                let label = r.media_url.as_deref().unwrap_or(r.doc_uri.as_str());
                println!("   [audio] {}", label);
                // Show transcript preview
                let preview = chunk_preview(&r.chunk_text);
                if !preview.is_empty() {
                    println!("   Transcript: {}\n", preview.replace('\n', " "));
                } else {
                    println!();
                }
            }
            Some("video") => {
                let label = r.media_url.as_deref().unwrap_or(r.doc_uri.as_str());
                // Distinguish keyframe vs transcript based on text content
                if r.chunk_text.len() > 50 {
                    // Likely a transcript chunk
                    println!("   [video transcript] {}", label);
                    let preview = chunk_preview(&r.chunk_text);
                    println!("   {}\n", preview.replace('\n', " "));
                } else {
                    // Likely a keyframe chunk
                    println!("   [video keyframe] {}\n", label);
                }
            }
            _ => {
                // Text chunks (default)
                let preview = chunk_preview(&r.chunk_text);
                println!("   {}\n", preview.replace('\n', " "));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::chunk_preview;

    #[test]
    fn chunk_preview_trims_short_text() {
        let preview = chunk_preview("  hello world  ");
        assert_eq!(preview, "hello world");
    }

    #[test]
    fn chunk_preview_truncates_long_text() {
        let input = format!("{}{}", "x".repeat(200), "tail");
        let preview = chunk_preview(&input);
        assert_eq!(preview, format!("{}...", "x".repeat(200)));
    }

    #[test]
    fn chunk_preview_handles_multibyte_utf8_at_boundary() {
        // Place a 4-byte emoji right at the 200-byte boundary
        // 198 ASCII bytes + "🔍" (4 bytes) + more text
        let input = format!("{}🔍rest of the text", "x".repeat(198));
        let preview = chunk_preview(&input);
        // Should not panic, and should include the emoji since 198+4=202 > 200
        // but boundary search finds the char boundary at 198 or 202
        assert!(preview.ends_with("..."));
        assert!(!preview.is_empty());
    }

    #[test]
    fn chunk_preview_handles_cjk_at_boundary() {
        // CJK characters are 3 bytes each in UTF-8
        // 66 CJK chars = 198 bytes, then add more CJK
        let input: String = std::iter::repeat('中').take(70).collect();
        let preview = chunk_preview(&input);
        assert!(preview.ends_with("..."));
        assert!(!preview.is_empty());
    }
}
