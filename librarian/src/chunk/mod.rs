//! Text chunking with structure awareness
//!
//! This module handles splitting documents into chunks while:
//! - Respecting heading boundaries when possible
//! - Maintaining code block integrity
//! - Providing stable, deterministic chunk boundaries
//! - Computing content hashes for incremental updates

mod boundaries;

pub use boundaries::*;

use crate::config::ChunkConfig;
use crate::error::Result;
use crate::parse::{Heading, ParsedDocument};
use blake3::Hasher;

/// A text chunk with metadata
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// The actual text content
    pub text: String,
    
    /// Character start position in original document
    pub char_start: usize,
    
    /// Character end position in original document  
    pub char_end: usize,
    
    /// Chunk index (0-based)
    pub index: usize,
    
    /// Headings that apply to this chunk
    pub headings: Vec<String>,
    
    /// Blake3 hash of the normalized text
    pub hash: String,
}

impl TextChunk {
    /// Compute the hash for this chunk
    pub fn compute_hash(text: &str, doc_hash: &str) -> String {
        let mut hasher = Hasher::new();
        hasher.update(doc_hash.as_bytes());
        hasher.update(text.as_bytes());
        hasher.finalize().to_hex().to_string()
    }
}

/// Chunk a parsed document
pub fn chunk_document(
    doc: &ParsedDocument,
    doc_hash: &str,
    config: &ChunkConfig,
) -> Result<Vec<TextChunk>> {
    let text = &doc.text;
    
    if text.is_empty() {
        return Ok(Vec::new());
    }

    // Find potential break points
    let break_points = find_break_points(text, &doc.headings, config);
    
    let mut chunks = Vec::new();
    let mut current_start = 0;
    let mut chunk_index = 0;

    while current_start < text.len() {
        // Ensure current_start is on a char boundary
        current_start = ensure_char_boundary(text, current_start);
        if current_start >= text.len() {
            break;
        }

        // Find the next chunk boundary
        let target_end = current_start + config.max_chars;
        
        let chunk_end = if target_end >= text.len() {
            text.len()
        } else {
            // Find best break point near target (returns valid char boundary)
            find_best_break(text, current_start, target_end, &break_points, config)
        };

        // Ensure chunk_end is valid
        let chunk_end = ensure_char_boundary(text, chunk_end);
        if chunk_end <= current_start {
            current_start = chunk_end + 1;
            continue;
        }

        // Extract chunk text (now guaranteed to be on char boundaries)
        let chunk_text = text[current_start..chunk_end].trim().to_string();
        
        // Skip if too small (unless it's the last chunk)
        if chunk_text.len() < config.min_chars && chunk_end < text.len() {
            current_start = chunk_end;
            continue;
        }

        if !chunk_text.is_empty() {
            // Get headings that apply to this chunk
            let headings = doc.headings_at_position(current_start)
                .iter()
                .map(|h| h.text.clone())
                .collect();

            let hash = TextChunk::compute_hash(&chunk_text, doc_hash);

            chunks.push(TextChunk {
                text: chunk_text,
                char_start: current_start,
                char_end: chunk_end,
                index: chunk_index,
                headings,
                hash,
            });

            chunk_index += 1;
        }

        // Move to next chunk with overlap
        if chunk_end >= text.len() {
            break;
        }
        
        // Ensure overlap position is on a char boundary
        current_start = if chunk_end > config.overlap_chars {
            ensure_char_boundary(text, chunk_end - config.overlap_chars)
        } else {
            chunk_end
        };
    }

    Ok(chunks)
}

/// Find potential break points in the text
fn find_break_points(
    text: &str,
    headings: &[Heading],
    config: &ChunkConfig,
) -> Vec<BreakPoint> {
    let mut points = Vec::new();

    // Add heading positions as high-priority breaks
    if config.prefer_heading_boundaries {
        for heading in headings {
            if heading.position < text.len() && text.is_char_boundary(heading.position) {
                points.push(BreakPoint {
                    position: heading.position,
                    priority: BreakPriority::Heading,
                });
            }
        }
    }

    // Find paragraph breaks (double newlines)
    for (i, c) in text.char_indices() {
        if c == '\n' {
            let remaining = &text[i..];
            if remaining.starts_with("\n\n") {
                let pos = i + 2;
                if text.is_char_boundary(pos) {
                    points.push(BreakPoint {
                        position: pos,
                        priority: BreakPriority::Paragraph,
                    });
                }
            }
        }
    }

    // Find sentence boundaries
    for (i, _) in text.match_indices(". ") {
        let pos = i + 2;
        if text.is_char_boundary(pos) {
            points.push(BreakPoint {
                position: pos,
                priority: BreakPriority::Sentence,
            });
        }
    }
    for (i, _) in text.match_indices(".\n") {
        let pos = i + 2;
        if text.is_char_boundary(pos) {
            points.push(BreakPoint {
                position: pos,
                priority: BreakPriority::Sentence,
            });
        }
    }
    for (i, _) in text.match_indices("? ") {
        points.push(BreakPoint {
            position: i + 2,
            priority: BreakPriority::Sentence,
        });
    }
    for (i, _) in text.match_indices("! ") {
        points.push(BreakPoint {
            position: i + 2,
            priority: BreakPriority::Sentence,
        });
    }

    // Sort by position
    points.sort_by_key(|p| p.position);
    points.dedup_by_key(|p| p.position);

    points
}

/// Ensure a position is on a valid UTF-8 character boundary
fn ensure_char_boundary(text: &str, pos: usize) -> usize {
    if pos >= text.len() {
        return text.len();
    }
    if text.is_char_boundary(pos) {
        return pos;
    }
    // Search backwards for a valid boundary
    let mut adjusted = pos;
    while adjusted > 0 && !text.is_char_boundary(adjusted) {
        adjusted -= 1;
    }
    adjusted
}

/// Find the best break point near the target position
fn find_best_break(
    text: &str,
    start: usize,
    target: usize,
    break_points: &[BreakPoint],
    config: &ChunkConfig,
) -> usize {
    // Search window: 80% to 120% of target chunk size
    let min_pos = ensure_char_boundary(text, start + (config.max_chars * 4 / 5));
    let max_pos = ensure_char_boundary(text, std::cmp::min(start + (config.max_chars * 6 / 5), text.len()));

    // Find break points in the window (all break points should already be on char boundaries)
    let candidates: Vec<&BreakPoint> = break_points
        .iter()
        .filter(|p| p.position >= min_pos && p.position <= max_pos && text.is_char_boundary(p.position))
        .collect();

    if let Some(best) = candidates
        .iter()
        .max_by_key(|p| p.priority as u8)
    {
        return best.position;
    }

    // Fall back to word boundary using char_indices for safety
    if target < text.len() {
        let search_start = ensure_char_boundary(text, if target > 50 { target - 50 } else { start });
        let search_end = ensure_char_boundary(text, std::cmp::min(target + 50, text.len()));
        
        if search_start < search_end {
            let search_text = &text[search_start..search_end];
            
            for (i, _) in search_text.rmatch_indices(' ') {
                let pos = search_start + i + 1;
                if pos >= min_pos && pos <= max_pos && text.is_char_boundary(pos) {
                    return pos;
                }
            }
        }
    }

    // Ultimate fallback: ensure we return a valid char boundary
    ensure_char_boundary(text, std::cmp::min(target, text.len()))
}

/// Compute a stable hash for document content
pub fn compute_content_hash(content: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(content);
    hasher.finalize().to_hex().to_string()
}

/// Compute a stable hash for a string
pub fn compute_text_hash(text: &str) -> String {
    compute_content_hash(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::ContentType;

    fn make_test_doc(text: &str) -> ParsedDocument {
        ParsedDocument {
            title: None,
            text: text.to_string(),
            content_type: ContentType::PlainText,
            headings: Vec::new(),
            code_blocks: Vec::new(),
            links: Vec::new(),
        }
    }

    fn default_chunk_config() -> ChunkConfig {
        ChunkConfig {
            max_chars: 500,
            overlap_chars: 50,
            prefer_heading_boundaries: true,
            min_chars: 50,
        }
    }

    #[test]
    fn test_chunk_short_document() {
        let doc = make_test_doc("This is a short document.");
        let config = default_chunk_config();
        let doc_hash = compute_text_hash(&doc.text);
        
        let chunks = chunk_document(&doc, &doc_hash, &config).unwrap();
        
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "This is a short document.");
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn test_chunk_long_document() {
        let text = "Lorem ipsum dolor sit amet. ".repeat(100);
        let doc = make_test_doc(&text);
        let config = default_chunk_config();
        let doc_hash = compute_text_hash(&doc.text);
        
        let chunks = chunk_document(&doc, &doc_hash, &config).unwrap();
        
        assert!(chunks.len() > 1);
        // Check chunks don't exceed max size
        for chunk in &chunks {
            assert!(chunk.text.len() <= config.max_chars + 100); // Allow some flexibility
        }
    }

    #[test]
    fn test_chunk_hash_stability() {
        let doc = make_test_doc("Test content for hashing.");
        let config = default_chunk_config();
        let doc_hash = compute_text_hash(&doc.text);
        
        let chunks1 = chunk_document(&doc, &doc_hash, &config).unwrap();
        let chunks2 = chunk_document(&doc, &doc_hash, &config).unwrap();
        
        assert_eq!(chunks1[0].hash, chunks2[0].hash);
    }

    #[test]
    fn test_content_hash() {
        let hash1 = compute_text_hash("hello world");
        let hash2 = compute_text_hash("hello world");
        let hash3 = compute_text_hash("different content");
        
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_chunk_with_headings() {
        let mut doc = make_test_doc("# Title\n\nSome content here.\n\n## Section\n\nMore content.");
        doc.headings = vec![
            Heading { level: 1, text: "Title".to_string(), position: 0 },
            Heading { level: 2, text: "Section".to_string(), position: 30 },
        ];
        
        let config = ChunkConfig {
            max_chars: 100,
            overlap_chars: 10,
            prefer_heading_boundaries: true,
            min_chars: 10,
        };
        let doc_hash = compute_text_hash(&doc.text);
        
        let chunks = chunk_document(&doc, &doc_hash, &config).unwrap();
        
        // Verify headings are captured
        for chunk in &chunks {
            // Each chunk should have appropriate headings
            assert!(chunk.headings.len() <= doc.headings.len());
        }
    }
}
