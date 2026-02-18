//! Result ranking and hybrid retrieval
//!
//! This module handles:
//! - Merging vector search results
//! - Optional BM25 keyword scoring
//! - Score normalization

use crate::store::SearchResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A ranked search result with combined scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedResult {
    pub id: String,
    pub score: f32,
    pub vector_score: f32,
    pub bm25_score: Option<f32>,
    pub doc_uri: String,
    pub title: Option<String>,
    pub chunk_text: String,
    pub headings: Option<Vec<String>>,
    pub chunk_index: i32,
    pub source_id: String,
    pub source_type: String,
    pub source_uri: String,
    pub modality: Option<String>,
    pub media_url: Option<String>,
    pub media_hash: Option<String>,
}

impl From<SearchResult> for RankedResult {
    fn from(result: SearchResult) -> Self {
        Self {
            id: result.id,
            score: result.score,
            vector_score: result.score,
            bm25_score: None,
            doc_uri: result.payload.doc_uri,
            title: result.payload.title,
            chunk_text: String::new(), // Will be filled from SQLite
            headings: result.payload.headings,
            chunk_index: result.payload.chunk_index,
            source_id: result.payload.source_id,
            source_type: result.payload.source_type,
            source_uri: result.payload.source_uri,
            modality: result.payload.modality,
            media_url: result.payload.media_url,
            media_hash: result.payload.media_hash,
        }
    }
}

/// Rank and merge search results using vector + optional BM25 hybrid scoring.
///
/// The ranker normalises and combines vector similarity scores with BM25
/// keyword scores according to configurable weights.
pub struct Ranker {
    bm25_weight: f32,
    vector_weight: f32,
}

impl Ranker {
    /// Create a new ranker
    pub fn new(bm25_weight: f32) -> Self {
        Self {
            bm25_weight,
            vector_weight: 1.0 - bm25_weight,
        }
    }

    /// Rank results using vector scores only
    pub fn rank_vector_only(&self, results: Vec<SearchResult>) -> Vec<RankedResult> {
        let mut ranked: Vec<RankedResult> = results.into_iter().map(RankedResult::from).collect();
        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }

    /// Merge vector results with BM25 scores
    pub fn rank_hybrid(
        &self,
        vector_results: Vec<SearchResult>,
        bm25_scores: &HashMap<String, f32>,
    ) -> Vec<RankedResult> {
        let mut ranked: Vec<RankedResult> = vector_results
            .into_iter()
            .map(|r| {
                let mut result = RankedResult::from(r);
                result.bm25_score = bm25_scores.get(&result.id).copied();

                // Combine scores
                let bm25 = result.bm25_score.unwrap_or(0.0);
                result.score = self.vector_weight * result.vector_score + self.bm25_weight * bm25;

                result
            })
            .collect();

        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }

    /// Filter results by minimum score
    pub fn filter_by_score(&self, results: Vec<RankedResult>, min_score: f32) -> Vec<RankedResult> {
        results
            .into_iter()
            .filter(|r| r.score >= min_score)
            .collect()
    }

    /// Deduplicate results by doc_uri (keep highest scoring chunk per doc)
    pub fn dedupe_by_doc(&self, results: Vec<RankedResult>) -> Vec<RankedResult> {
        let mut by_doc: HashMap<String, RankedResult> = HashMap::new();

        for result in results {
            let entry = by_doc.entry(result.doc_uri.clone());
            match entry {
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    if result.score > e.get().score {
                        e.insert(result);
                    }
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(result);
                }
            }
        }

        let mut deduped: Vec<RankedResult> = by_doc.into_values().collect();
        deduped.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        deduped
    }
}

/// Corpus-level statistics for IDF computation in BM25 scoring.
///
/// Collect these once from the metadata database and pass to
/// [`Bm25Scorer::score_with_idf`] so rare terms receive higher weight.
pub struct CorpusStats {
    pub total_docs: usize,
    pub doc_frequencies: HashMap<String, usize>,
}

/// BM25 (Okapi) scorer with configurable k1 and b parameters.
///
/// Supports both a simplified single-document mode ([`score`](Self::score))
/// and a corpus-aware mode ([`score_with_idf`](Self::score_with_idf)) that
/// uses real inverse document frequency from [`CorpusStats`].
pub struct Bm25Scorer {
    k1: f32,
    b: f32,
}

impl Bm25Scorer {
    pub fn new() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }

    /// Score a single document against a query (simplified, no real IDF).
    ///
    /// Prefer [`score_with_idf`](Self::score_with_idf) when corpus statistics
    /// are available.
    pub fn score(&self, query_terms: &[String], doc_text: &str, avg_doc_len: f32) -> f32 {
        let doc_lower = doc_text.to_lowercase();
        let doc_len = doc_text.len() as f32;
        let mut total_score = 0.0;

        for term in query_terms {
            let term_lower = term.to_lowercase();
            let tf = doc_lower.matches(&term_lower).count() as f32;

            if tf > 0.0 {
                // Simplified BM25 - assumes single document collection
                let idf = 1.0; // Would need corpus statistics for real IDF
                let numerator = tf * (self.k1 + 1.0);
                let denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_doc_len));
                total_score += idf * (numerator / denominator);
            }
        }

        total_score
    }

    /// Tokenize query into terms (filters words < 3 chars as stop words)
    pub fn tokenize(&self, query: &str) -> Vec<String> {
        query
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .filter(|s| s.len() >= 3)
            .collect()
    }

    /// Score a document using real IDF from corpus statistics.
    ///
    /// IDF formula: `ln((N - n + 0.5) / (n + 0.5) + 1.0)` where N = total
    /// docs and n = number of docs containing the term.
    pub fn score_with_idf(
        &self,
        query_terms: &[String],
        doc_text: &str,
        avg_doc_len: f32,
        stats: &CorpusStats,
    ) -> f32 {
        let doc_lower = doc_text.to_lowercase();
        let doc_len = doc_text.len() as f32;
        let n_total = stats.total_docs as f32;
        let mut total_score = 0.0;

        for term in query_terms {
            let term_lower = term.to_lowercase();
            let tf = doc_lower.matches(&term_lower).count() as f32;

            if tf > 0.0 {
                let n_docs = stats.doc_frequencies.get(&term_lower).copied().unwrap_or(0) as f32;
                let idf = ((n_total - n_docs + 0.5) / (n_docs + 0.5) + 1.0).ln();
                let numerator = tf * (self.k1 + 1.0);
                let denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_doc_len));
                total_score += idf * (numerator / denominator);
            }
        }

        total_score
    }
}

impl Default for Bm25Scorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::ChunkPayload;

    fn make_search_result(id: &str, score: f32, doc_uri: &str) -> SearchResult {
        let payload = ChunkPayload::new(
            "src".to_string(),
            "dir".to_string(),
            "/docs".to_string(),
            "doc".to_string(),
            doc_uri.to_string(),
            0,
            "hash".to_string(),
            "2024-01-01".to_string(),
        );

        SearchResult {
            id: id.to_string(),
            score,
            payload,
        }
    }

    #[test]
    fn test_rank_vector_only() {
        let ranker = Ranker::new(0.0);
        let results = vec![
            make_search_result("1", 0.5, "/doc1"),
            make_search_result("2", 0.9, "/doc2"),
            make_search_result("3", 0.7, "/doc3"),
        ];

        let ranked = ranker.rank_vector_only(results);

        assert_eq!(ranked[0].id, "2");
        assert_eq!(ranked[1].id, "3");
        assert_eq!(ranked[2].id, "1");
    }

    #[test]
    fn test_dedupe_by_doc() {
        let ranker = Ranker::new(0.0);
        let results = vec![
            make_search_result("1", 0.9, "/doc1"),
            make_search_result("2", 0.7, "/doc1"),
            make_search_result("3", 0.8, "/doc2"),
        ];

        let ranked = ranker.rank_vector_only(results);
        let deduped = ranker.dedupe_by_doc(ranked);

        assert_eq!(deduped.len(), 2);
        assert_eq!(
            deduped.iter().find(|r| r.doc_uri == "/doc1").unwrap().score,
            0.9
        );
    }

    #[test]
    fn test_bm25_tokenize() {
        let scorer = Bm25Scorer::new();
        let terms = scorer.tokenize("How to configure X?");

        assert!(terms.contains(&"how".to_string()));
        assert!(terms.contains(&"configure".to_string()));
        assert!(!terms.contains(&"to".to_string())); // Too short
    }

    #[test]
    fn test_bm25_score() {
        let scorer = Bm25Scorer::new();
        let terms = scorer.tokenize("rust programming");

        let score1 = scorer.score(&terms, "Rust is a systems programming language", 100.0);
        let score2 = scorer.score(&terms, "Python is great", 100.0);

        assert!(score1 > score2);
    }

    #[test]
    fn test_bm25_score_with_idf_rare_term_higher() {
        let scorer = Bm25Scorer::new();

        let mut doc_frequencies = HashMap::new();
        // "rust" appears in 2 of 100 docs (rare)
        doc_frequencies.insert("rust".to_string(), 2);
        // "the" appears in 90 of 100 docs (common)
        doc_frequencies.insert("the".to_string(), 90);

        let stats = CorpusStats {
            total_docs: 100,
            doc_frequencies,
        };

        let doc_text = "Rust is the best language for the task";
        let rare_score = scorer.score_with_idf(&["rust".to_string()], doc_text, 100.0, &stats);
        let common_score = scorer.score_with_idf(&["the".to_string()], doc_text, 100.0, &stats);

        assert!(
            rare_score > common_score,
            "Rare term 'rust' ({rare_score}) should score higher than common term 'the' ({common_score})"
        );
    }

    #[test]
    fn test_bm25_score_with_idf_backward_compat() {
        let scorer = Bm25Scorer::new();
        let terms = scorer.tokenize("rust programming");

        // Existing score() method still works
        let score = scorer.score(&terms, "Rust is a systems programming language", 100.0);
        assert!(score > 0.0);
    }
}
