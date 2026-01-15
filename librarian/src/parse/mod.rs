//! Document parsing and text extraction
//!
//! This module handles:
//! - HTML parsing and text extraction
//! - Markdown processing
//! - Plain text normalization
//! - Content type detection

mod html;
mod markdown;
mod text;

pub use html::*;
pub use markdown::*;
pub use text::*;

use crate::error::Result;
use std::path::Path;

/// Content types we can parse
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    Html,
    Markdown,
    PlainText,
    Unknown,
}

impl ContentType {
    /// Detect content type from file extension
    pub fn from_extension(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("html") | Some("htm") => ContentType::Html,
            Some("md") | Some("markdown") | Some("mdx") => ContentType::Markdown,
            Some("txt") | Some("text") => ContentType::PlainText,
            Some("rst") => ContentType::PlainText, // Treat RST as plain text for now
            _ => ContentType::Unknown,
        }
    }

    /// Detect content type from MIME type
    pub fn from_mime(mime: &str) -> Self {
        let mime_lower = mime.to_lowercase();
        if mime_lower.contains("text/html") || mime_lower.contains("application/xhtml") {
            ContentType::Html
        } else if mime_lower.contains("text/markdown") {
            ContentType::Markdown
        } else if mime_lower.contains("text/plain") {
            ContentType::PlainText
        } else {
            ContentType::Unknown
        }
    }

    /// Detect from both path and optional MIME type
    pub fn detect(path: Option<&Path>, mime: Option<&str>) -> Self {
        // MIME takes precedence for web content
        if let Some(m) = mime {
            let detected = Self::from_mime(m);
            if detected != ContentType::Unknown {
                return detected;
            }
        }

        // Fall back to extension
        if let Some(p) = path {
            let detected = Self::from_extension(p);
            if detected != ContentType::Unknown {
                return detected;
            }
        }

        ContentType::Unknown
    }
}

/// Parsed document with extracted content
#[derive(Debug, Clone)]
pub struct ParsedDocument {
    /// Extracted title (if found)
    pub title: Option<String>,

    /// Main text content
    pub text: String,

    /// Detected content type
    pub content_type: ContentType,

    /// Extracted headings with their levels and positions
    pub headings: Vec<Heading>,

    /// Code blocks with language info
    pub code_blocks: Vec<CodeBlock>,

    /// Links found in the document
    pub links: Vec<ExtractedLink>,
}

/// A heading in the document
#[derive(Debug, Clone)]
pub struct Heading {
    /// Heading level (1-6)
    pub level: u8,

    /// Heading text
    pub text: String,

    /// Character position in the extracted text
    pub position: usize,
}

/// A code block
#[derive(Debug, Clone)]
pub struct CodeBlock {
    /// Language identifier (if specified)
    pub language: Option<String>,

    /// Code content
    pub content: String,

    /// Character position in the extracted text
    pub position: usize,
}

/// An extracted link
#[derive(Debug, Clone)]
pub struct ExtractedLink {
    /// Link URL
    pub url: String,

    /// Link text
    pub text: Option<String>,

    /// Whether this is internal (same domain) or external
    pub is_internal: bool,
}

impl ParsedDocument {
    pub fn new(text: String, content_type: ContentType) -> Self {
        Self {
            title: None,
            text,
            content_type,
            headings: Vec::new(),
            code_blocks: Vec::new(),
            links: Vec::new(),
        }
    }

    /// Get headings at or above a certain level
    pub fn headings_at_position(&self, position: usize) -> Vec<&Heading> {
        let mut current_levels: Vec<&Heading> = Vec::new();

        for heading in &self.headings {
            if heading.position > position {
                break;
            }

            // Remove headings at same or lower level
            current_levels.retain(|h| h.level < heading.level);
            current_levels.push(heading);
        }

        current_levels
    }
}

/// Parse content based on detected type
pub fn parse_content(
    content: &str,
    content_type: ContentType,
    base_url: Option<&str>,
) -> Result<ParsedDocument> {
    match content_type {
        ContentType::Html => parse_html(content, base_url),
        ContentType::Markdown => parse_markdown(content),
        ContentType::PlainText | ContentType::Unknown => Ok(parse_plain_text(content)),
    }
}

/// Check if content appears to be binary
pub fn is_binary_content(data: &[u8]) -> bool {
    // Check for null bytes in the first 8KB
    let check_len = std::cmp::min(data.len(), 8192);
    data[..check_len].iter().any(|&b| b == 0)
}

/// Check if file should be skipped based on extension
pub fn should_skip_file(path: &Path) -> bool {
    let skip_extensions = [
        "png", "jpg", "jpeg", "gif", "bmp", "ico", "svg", "webp", "mp3", "mp4", "wav", "ogg",
        "webm", "avi", "mov", "zip", "tar", "gz", "bz2", "xz", "7z", "rar", "exe", "dll", "so",
        "dylib", "bin", "woff", "woff2", "ttf", "otf", "eot", "pyc", "pyo", "class", "o", "obj",
        "lock", "DS_Store",
    ];

    #[cfg(not(feature = "pdf"))]
    let skip_extensions = {
        let mut exts = skip_extensions.to_vec();
        exts.push("pdf");
        exts
    };

    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        skip_extensions.contains(&ext.to_lowercase().as_str())
    } else {
        false
    }
}

/// Normalize whitespace in text
pub fn normalize_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_was_whitespace = true;
    let mut newline_count = 0;

    for c in text.chars() {
        if c.is_whitespace() {
            if c == '\n' {
                newline_count += 1;
            }
            last_was_whitespace = true;
        } else {
            // Before adding a non-whitespace char, handle accumulated whitespace
            if last_was_whitespace && !result.is_empty() {
                if newline_count >= 2 {
                    // Multiple newlines = paragraph break, preserve as double newline
                    result.push_str("\n\n");
                } else if newline_count == 1 {
                    // Single newline = line break
                    result.push('\n');
                } else {
                    // Other whitespace = single space
                    result.push(' ');
                }
            }
            newline_count = 0;
            result.push(c);
            last_was_whitespace = false;
        }
    }

    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_type_detection() {
        assert_eq!(
            ContentType::from_extension(Path::new("test.html")),
            ContentType::Html
        );
        assert_eq!(
            ContentType::from_extension(Path::new("test.md")),
            ContentType::Markdown
        );
        assert_eq!(
            ContentType::from_extension(Path::new("test.txt")),
            ContentType::PlainText
        );
        assert_eq!(
            ContentType::from_extension(Path::new("test.rs")),
            ContentType::Unknown
        );
    }

    #[test]
    fn test_mime_detection() {
        assert_eq!(
            ContentType::from_mime("text/html; charset=utf-8"),
            ContentType::Html
        );
        assert_eq!(
            ContentType::from_mime("text/markdown"),
            ContentType::Markdown
        );
        assert_eq!(ContentType::from_mime("text/plain"), ContentType::PlainText);
    }

    #[test]
    fn test_normalize_whitespace() {
        let input = "Hello   world\n\n\n\ntest";
        let result = normalize_whitespace(input);
        assert_eq!(result, "Hello world\n\ntest");
    }

    #[test]
    fn test_should_skip_file() {
        assert!(should_skip_file(Path::new("image.png")));
        assert!(should_skip_file(Path::new("archive.zip")));
        assert!(!should_skip_file(Path::new("readme.md")));
        assert!(!should_skip_file(Path::new("index.html")));
    }

    #[test]
    fn test_binary_detection() {
        assert!(is_binary_content(&[0x00, 0x01, 0x02]));
        assert!(!is_binary_content(b"Hello world"));
    }
}
