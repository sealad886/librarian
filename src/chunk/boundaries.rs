//! Break point detection for chunking

/// Priority levels for break points
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BreakPriority {
    /// Word boundary (lowest)
    Word = 1,
    /// Sentence boundary
    Sentence = 2,
    /// Paragraph boundary
    Paragraph = 3,
    /// Heading boundary (highest)
    Heading = 4,
}

/// A potential break point in text
#[derive(Debug, Clone)]
pub struct BreakPoint {
    /// Character position
    pub position: usize,
    /// Priority of this break point
    pub priority: BreakPriority,
}

impl BreakPoint {
    pub fn new(position: usize, priority: BreakPriority) -> Self {
        Self { position, priority }
    }
}

/// Detect code block boundaries (positions to avoid breaking)
pub fn find_code_blocks(text: &str) -> Vec<(usize, usize)> {
    let mut blocks = Vec::new();
    let mut in_block = false;
    let mut block_start = 0;

    for (i, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            if in_block {
                // End of block
                let byte_pos = text.lines().take(i + 1).map(|l| l.len() + 1).sum::<usize>();
                blocks.push((block_start, byte_pos));
                in_block = false;
            } else {
                // Start of block
                block_start = text.lines().take(i).map(|l| l.len() + 1).sum::<usize>();
                in_block = true;
            }
        }
    }

    blocks
}

/// Check if a position is inside a code block
pub fn is_in_code_block(position: usize, code_blocks: &[(usize, usize)]) -> bool {
    code_blocks
        .iter()
        .any(|(start, end)| position >= *start && position < *end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_break_priority_ordering() {
        assert!(BreakPriority::Heading > BreakPriority::Paragraph);
        assert!(BreakPriority::Paragraph > BreakPriority::Sentence);
        assert!(BreakPriority::Sentence > BreakPriority::Word);
    }

    #[test]
    fn test_find_code_blocks() {
        let text = "Some text\n```\ncode here\n```\nMore text";
        let blocks = find_code_blocks(text);

        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn test_is_in_code_block() {
        let blocks = vec![(10, 30), (50, 70)];

        assert!(!is_in_code_block(5, &blocks));
        assert!(is_in_code_block(15, &blocks));
        assert!(!is_in_code_block(35, &blocks));
        assert!(is_in_code_block(60, &blocks));
    }
}
