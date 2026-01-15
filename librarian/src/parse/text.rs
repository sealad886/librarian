//! Plain text parsing

use super::{ContentType, ParsedDocument, normalize_whitespace};

/// Parse plain text content
pub fn parse_plain_text(content: &str) -> ParsedDocument {
    let text = normalize_whitespace(content);
    
    // Try to extract a title from the first line
    let title = text.lines().next().map(|line| {
        let trimmed = line.trim();
        if trimmed.len() < 100 && !trimmed.is_empty() {
            Some(trimmed.to_string())
        } else {
            None
        }
    }).flatten();

    ParsedDocument {
        title,
        text,
        content_type: ContentType::PlainText,
        headings: Vec::new(),
        code_blocks: Vec::new(),
        links: Vec::new(),
    }
}

/// Clean text for embedding (remove excessive formatting)
pub fn clean_for_embedding(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_was_newline = false;

    for c in text.chars() {
        match c {
            '\n' => {
                if !last_was_newline {
                    result.push(' ');
                    last_was_newline = true;
                }
            }
            '\r' | '\t' => {
                if !last_was_newline {
                    result.push(' ');
                }
            }
            _ if c.is_whitespace() => {
                if !last_was_newline {
                    result.push(' ');
                }
            }
            _ => {
                result.push(c);
                last_was_newline = false;
            }
        }
    }

    // Collapse multiple spaces
    let mut final_result = String::with_capacity(result.len());
    let mut last_was_space = false;
    
    for c in result.chars() {
        if c == ' ' {
            if !last_was_space {
                final_result.push(c);
                last_was_space = true;
            }
        } else {
            final_result.push(c);
            last_was_space = false;
        }
    }

    final_result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_plain_text() {
        let text = "Title Line\n\nSome content here.\n\nMore content.";
        let doc = parse_plain_text(text);
        
        assert_eq!(doc.title, Some("Title Line".to_string()));
        assert!(doc.text.contains("content"));
    }

    #[test]
    fn test_clean_for_embedding() {
        let text = "Hello\n\n\nWorld\t\tTest";
        let result = clean_for_embedding(text);
        assert_eq!(result, "Hello World Test");
    }

    #[test]
    fn test_no_title_for_long_first_line() {
        let text = "This is a very long first line that should not be considered a title because it exceeds the character limit we set for reasonable titles and would be truncated awkwardly if used.";
        let doc = parse_plain_text(text);
        assert_eq!(doc.title, None);
    }
}
