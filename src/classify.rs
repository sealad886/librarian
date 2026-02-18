//! Query intent classification
//!
//! Rule-based classifier that categorises search queries into intent types.
//! Used to log query analytics and, in the future, to tune retrieval strategy
//! (e.g. BM25 weight, over-fetch ratio, filter behaviour) per intent.

use serde::Serialize;
use std::fmt;

/// Detected intent of a search query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QueryIntent {
    /// Seeking a specific fact or definition ("what is X", "define Y")
    Factual,
    /// Seeking step-by-step instructions ("how to", "guide", "setup")
    Procedural,
    /// Comparing or contrasting two or more concepts ("X vs Y", "difference between")
    Comparative,
    /// Looking for a specific page, section, or resource ("show me", "find the config")
    Navigational,
    /// Seeking broad understanding of a topic ("explain", "overview", "why does")
    Conceptual,
    /// Default when no strong signal is detected
    General,
}

impl fmt::Display for QueryIntent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Factual => write!(f, "factual"),
            Self::Procedural => write!(f, "procedural"),
            Self::Comparative => write!(f, "comparative"),
            Self::Navigational => write!(f, "navigational"),
            Self::Conceptual => write!(f, "conceptual"),
            Self::General => write!(f, "general"),
        }
    }
}

/// Classify a query string into a [`QueryIntent`] using keyword heuristics.
///
/// The classifier checks for characteristic phrases in priority order:
/// comparative → procedural → factual → navigational → conceptual → general.
pub fn classify_query(query: &str) -> QueryIntent {
    let lower = query.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    // Comparative signals (highest priority — very specific)
    if has_comparative_signal(&lower) {
        return QueryIntent::Comparative;
    }

    // Procedural signals
    if has_procedural_signal(&lower, &words) {
        return QueryIntent::Procedural;
    }

    // Factual signals
    if has_factual_signal(&lower, &words) {
        return QueryIntent::Factual;
    }

    // Navigational signals
    if has_navigational_signal(&lower, &words) {
        return QueryIntent::Navigational;
    }

    // Conceptual signals
    if has_conceptual_signal(&lower, &words) {
        return QueryIntent::Conceptual;
    }

    QueryIntent::General
}

fn has_comparative_signal(lower: &str) -> bool {
    let patterns = [
        " vs ",
        " vs. ",
        " versus ",
        "compared to",
        "comparison of",
        "difference between",
        "differences between",
        "differ from",
        "better than",
        "worse than",
        "pros and cons",
        "trade-off",
        "tradeoff",
        "advantages of",
    ];
    patterns.iter().any(|p| lower.contains(p))
}

fn has_procedural_signal(lower: &str, words: &[&str]) -> bool {
    let starts = ["how to", "how do", "how can", "steps to", "guide to"];
    if starts.iter().any(|p| lower.starts_with(p)) {
        return true;
    }

    let phrases = [
        "step by step",
        "tutorial",
        "walkthrough",
        "instructions for",
        "set up",
        "setup",
        "install",
        "configure",
        "migrate",
        "getting started",
        "quickstart",
        "quick start",
    ];
    if phrases.iter().any(|p| lower.contains(p)) {
        return true;
    }

    let imperative_verbs = [
        "create",
        "build",
        "deploy",
        "run",
        "start",
        "stop",
        "enable",
        "disable",
        "add",
        "remove",
        "update",
        "upgrade",
        "import",
        "export",
        "connect",
        "initialize",
    ];
    words
        .first()
        .is_some_and(|first| imperative_verbs.contains(first))
}

fn has_factual_signal(lower: &str, words: &[&str]) -> bool {
    let starts = [
        "what is",
        "what are",
        "what does",
        "what's",
        "who is",
        "who are",
        "when is",
        "when was",
        "when did",
        "where is",
        "where are",
        "which",
        "define",
        "definition of",
    ];
    if starts.iter().any(|p| lower.starts_with(p)) {
        return true;
    }

    // Questions ending with "?"
    if lower.ends_with('?') && words.len() <= 8 {
        let question_words = ["what", "who", "when", "where", "which"];
        if words.first().is_some_and(|w| question_words.contains(w)) {
            return true;
        }
    }

    false
}

fn has_navigational_signal(lower: &str, words: &[&str]) -> bool {
    let phrases = [
        "show me",
        "find the",
        "find me",
        "where is the",
        "locate",
        "link to",
        "go to",
        "navigate to",
        "open the",
        "see the",
    ];
    if phrases.iter().any(|p| lower.contains(p)) {
        return true;
    }

    // Short queries (1-3 words) that look like resource references
    if words.len() <= 3 {
        let resource_indicators = [
            "config",
            "configuration",
            "settings",
            "api",
            "endpoint",
            "schema",
            "table",
            "file",
            "module",
            "class",
            "function",
            "readme",
            "changelog",
            "license",
        ];
        if words.iter().any(|w| resource_indicators.contains(w)) {
            return true;
        }
    }

    false
}

fn has_conceptual_signal(lower: &str, _words: &[&str]) -> bool {
    let starts = [
        "explain", "why is", "why does", "why do", "why are", "how does", "how is", "how are",
    ];
    if starts.iter().any(|p| lower.starts_with(p)) {
        return true;
    }

    let phrases = [
        "overview of",
        "introduction to",
        "concept of",
        "in depth",
        "deep dive",
        "under the hood",
        "architecture of",
        "design of",
        "philosophy",
        "purpose of",
        "motivation for",
        "rationale",
    ];
    phrases.iter().any(|p| lower.contains(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factual_queries() {
        assert_eq!(
            classify_query("what is a vector database"),
            QueryIntent::Factual
        );
        assert_eq!(classify_query("What are embeddings?"), QueryIntent::Factual);
        assert_eq!(classify_query("define chunking"), QueryIntent::Factual);
        assert_eq!(
            classify_query("who is the maintainer"),
            QueryIntent::Factual
        );
    }

    #[test]
    fn test_procedural_queries() {
        assert_eq!(
            classify_query("how to ingest a directory"),
            QueryIntent::Procedural
        );
        assert_eq!(
            classify_query("steps to configure qdrant"),
            QueryIntent::Procedural
        );
        assert_eq!(classify_query("install librarian"), QueryIntent::Procedural);
        assert_eq!(
            classify_query("getting started with MCP"),
            QueryIntent::Procedural
        );
        assert_eq!(
            classify_query("create a new source"),
            QueryIntent::Procedural
        );
    }

    #[test]
    fn test_comparative_queries() {
        assert_eq!(
            classify_query("BM25 vs vector search"),
            QueryIntent::Comparative
        );
        assert_eq!(
            classify_query("difference between update and reindex"),
            QueryIntent::Comparative
        );
        assert_eq!(
            classify_query("pros and cons of chunking strategies"),
            QueryIntent::Comparative
        );
    }

    #[test]
    fn test_navigational_queries() {
        assert_eq!(
            classify_query("show me the config"),
            QueryIntent::Navigational
        );
        assert_eq!(classify_query("api endpoint"), QueryIntent::Navigational);
        assert_eq!(classify_query("readme"), QueryIntent::Navigational);
        assert_eq!(
            classify_query("find the schema file"),
            QueryIntent::Navigational
        );
    }

    #[test]
    fn test_conceptual_queries() {
        assert_eq!(
            classify_query("explain how reranking works"),
            QueryIntent::Conceptual
        );
        assert_eq!(
            classify_query("why does librarian use Qdrant"),
            QueryIntent::Conceptual
        );
        assert_eq!(
            classify_query("architecture of the MCP server"),
            QueryIntent::Conceptual
        );
        assert_eq!(
            classify_query("overview of the embedding pipeline"),
            QueryIntent::Conceptual
        );
    }

    #[test]
    fn test_general_queries() {
        assert_eq!(
            classify_query("qdrant performance tuning tips"),
            QueryIntent::General
        );
        assert_eq!(
            classify_query("error handling patterns"),
            QueryIntent::General
        );
    }

    #[test]
    fn test_display_format() {
        assert_eq!(QueryIntent::Factual.to_string(), "factual");
        assert_eq!(QueryIntent::Procedural.to_string(), "procedural");
        assert_eq!(QueryIntent::Comparative.to_string(), "comparative");
        assert_eq!(QueryIntent::Navigational.to_string(), "navigational");
        assert_eq!(QueryIntent::Conceptual.to_string(), "conceptual");
        assert_eq!(QueryIntent::General.to_string(), "general");
    }
}
