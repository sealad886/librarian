//! reStructuredText (RST) parsing and text extraction
//!
//! Extracts headings, code blocks (``.. code-block::``, literal blocks),
//! links, and images from RST documents. Uses regex-based heuristic parsing
//! since no pulldown-cmark equivalent exists for RST in the Rust ecosystem.

use super::{
    normalize_whitespace, CodeBlock, ContentType, ExtractedLink, ExtractedMedia, Heading,
    MediaModality, ParsedDocument,
};
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

// Pre-compiled regexes for inline link extraction (called per-line)
static RST_INLINE_REF_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"`([^`]+)\s+<([^>]+)>`_").unwrap());
static RST_BARE_URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?://[^\s)>\]]+").unwrap());

/// RST section adornment characters in priority order.
///
/// RST allows any punctuation character as an adornment. The heading level is
/// determined by the order of first appearance, not by which character is used.
/// We track first-seen order to assign levels 1-6.
const ADORNMENT_CHARS: &str = "=-~`'^\"#*+:._";

/// Parse reStructuredText content into a [`ParsedDocument`].
///
/// Extracts:
/// - **Headings**: RST uses over/underline adornment characters. Level is
///   assigned by order of first appearance.
/// - **Code blocks**: `.. code-block::` directives and `::` literal blocks.
/// - **Links**: Inline references (`` `text <url>`_ ``), standalone hyperlinks,
///   and named reference targets (``.. _name: url``).
/// - **Images**: `.. image::` and `.. figure::` directives with alt text.
/// - **Metadata**: RST field lists at the document start (`:key: value`).
pub fn parse_rst(content: &str) -> ParsedDocument {
    let mut text_parts: Vec<String> = Vec::new();
    let mut headings: Vec<Heading> = Vec::new();
    let mut code_blocks: Vec<CodeBlock> = Vec::new();
    let mut links: Vec<ExtractedLink> = Vec::new();
    let mut media: Vec<ExtractedMedia> = Vec::new();
    let mut title: Option<String> = None;

    // Track adornment character → heading level mapping
    let mut adornment_levels: Vec<(char, bool)> = Vec::new(); // (char, has_overline)

    let mut i = 0;
    let mut char_pos: usize = 0;

    // Extract field-list metadata from document start
    let (body, metadata) = extract_rst_metadata(content);
    let lines: Vec<&str> = body.lines().collect();

    while i < lines.len() {
        let line = lines[i];

        // Detect headings: RST headings have underline (and optional overline)
        // of adornment characters matching the title width
        if let Some((heading_text, adorn_char, has_overline, consumed)) =
            detect_rst_heading(&lines, i)
        {
            let level = get_or_assign_level(&mut adornment_levels, adorn_char, has_overline);
            if title.is_none() && level == 1 {
                title = Some(heading_text.clone());
            }
            headings.push(Heading {
                level,
                text: heading_text.clone(),
                position: char_pos,
            });
            let heading_len = heading_text.len();
            text_parts.push(heading_text);
            text_parts.push("\n\n".to_string());
            char_pos += heading_len + 2;
            // Skip consumed lines for overline + title + underline
            i += consumed;
            continue;
        }

        // Detect code-block directive
        if line.trim_start().starts_with(".. code-block::") || line.trim_start().ends_with("::") {
            let language = if line.trim_start().starts_with(".. code-block::") {
                line.trim_start()
                    .strip_prefix(".. code-block::")
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            } else {
                None
            };

            // Find the indented block
            i += 1;
            // Skip blank lines after directive
            while i < lines.len() && lines[i].trim().is_empty() {
                i += 1;
            }
            let mut code_lines: Vec<&str> = Vec::new();
            let indent = if i < lines.len() {
                lines[i].len() - lines[i].trim_start().len()
            } else {
                0
            };
            if indent > 0 {
                while i < lines.len()
                    && (lines[i].trim().is_empty()
                        || lines[i].len() - lines[i].trim_start().len() >= indent)
                {
                    // Remove the common indent from code
                    let stripped = if lines[i].len() >= indent {
                        &lines[i][indent..]
                    } else {
                        lines[i]
                    };
                    code_lines.push(stripped);
                    i += 1;
                }
            }
            let code_content = code_lines.join("\n").trim_end().to_string();
            if !code_content.is_empty() {
                code_blocks.push(CodeBlock {
                    language,
                    content: code_content.clone(),
                    position: char_pos,
                });
                let code_len = code_content.len();
                text_parts.push(code_content);
                text_parts.push("\n\n".to_string());
                char_pos += code_len + 2;
            }
            continue;
        }

        // Detect image/figure directives
        if line.trim_start().starts_with(".. image::")
            || line.trim_start().starts_with(".. figure::")
        {
            let url = line
                .split("::")
                .nth(1)
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            // Look for :alt: option in subsequent indented lines
            let mut alt = None;
            let mut j = i + 1;
            while j < lines.len() && (lines[j].starts_with("   ") || lines[j].trim().is_empty()) {
                if lines[j].trim_start().starts_with(":alt:") {
                    alt = lines[j]
                        .trim_start()
                        .strip_prefix(":alt:")
                        .map(|s| s.trim().to_string());
                }
                j += 1;
            }
            if !url.is_empty() {
                media.push(ExtractedMedia {
                    url: url.clone(),
                    alt,
                    tag: if line.contains("figure") {
                        "figure".to_string()
                    } else {
                        "img".to_string()
                    },
                    css_background: false,
                    modality: MediaModality::Image,
                    mime_type: None,
                });
            }
            i = j;
            continue;
        }

        // Detect reference targets: .. _name: url
        if line.trim_start().starts_with(".. _") {
            if let Some(colon_pos) = line.find(": ") {
                let label = line
                    .trim_start()
                    .strip_prefix(".. _")
                    .and_then(|s| s.split(':').next())
                    .map(|s| s.to_string());
                let url = line[colon_pos + 2..].trim().to_string();
                if !url.is_empty() && (url.starts_with("http") || url.starts_with('/')) {
                    links.push(ExtractedLink {
                        url,
                        text: label,
                        is_internal: false,
                    });
                }
            }
            i += 1;
            continue;
        }

        // Extract inline links: `text <url>`_
        extract_inline_rst_links(line, &mut links);

        // Regular text line
        text_parts.push(line.to_string());
        text_parts.push("\n".to_string());
        char_pos += line.len() + 1;
        i += 1;
    }

    let text = normalize_whitespace(&text_parts.join(""));

    // Fall back to metadata title if no heading found
    if title.is_none() {
        title = metadata.get("title").cloned();
    }

    ParsedDocument {
        title,
        text,
        content_type: ContentType::Rst,
        headings,
        code_blocks,
        links,
        media,
        metadata,
    }
}

/// Extract RST field-list metadata from the start of the document.
///
/// Field lists look like:
/// ```rst
/// :Author: John Doe
/// :Version: 1.0
/// ```
fn extract_rst_metadata(content: &str) -> (&str, HashMap<String, String>) {
    let mut metadata = HashMap::new();
    let mut end_offset = 0;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            end_offset += line.len() + 1; // +1 for newline
            continue;
        }
        if trimmed.starts_with(':') && trimmed.len() > 1 {
            if let Some(colon2) = trimmed[1..].find(':') {
                let key = trimmed[1..1 + colon2].trim().to_lowercase();
                let value = trimmed[2 + colon2..].trim().to_string();
                if !key.is_empty() {
                    metadata.insert(key, value);
                    end_offset += line.len() + 1;
                    continue;
                }
            }
        }
        // First non-metadata line found
        break;
    }

    if metadata.is_empty() {
        (content, metadata)
    } else {
        let body = if end_offset < content.len() {
            &content[end_offset..]
        } else {
            ""
        };
        (body, metadata)
    }
}

/// Detect an RST heading at the given line index.
///
/// Returns `(heading_text, adornment_char, has_overline, lines_consumed)`.
fn detect_rst_heading(lines: &[&str], i: usize) -> Option<(String, char, bool, usize)> {
    // Pattern 1: overline + title + underline
    if i + 2 < lines.len() {
        let over = lines[i];
        let text = lines[i + 1];
        let under = lines[i + 2];
        if is_adornment(over)
            && is_adornment(under)
            && over.chars().next() == under.chars().next()
            && !text.trim().is_empty()
            && over.len() >= text.trim().len()
        {
            return Some((text.trim().to_string(), over.chars().next()?, true, 3));
        }
    }

    // Pattern 2: title + underline
    if i + 1 < lines.len() {
        let text = lines[i];
        let under = lines[i + 1];
        if !text.trim().is_empty()
            && is_adornment(under)
            && under.len() >= text.trim().len()
            && !text.starts_with(' ')
        {
            return Some((text.trim().to_string(), under.chars().next()?, false, 2));
        }
    }

    None
}

/// Check if a line is an RST section adornment (all same punctuation char, >= 2 chars).
fn is_adornment(line: &str) -> bool {
    let trimmed = line.trim_end();
    if trimmed.len() < 2 {
        return false;
    }
    let first = trimmed.chars().next().unwrap_or(' ');
    ADORNMENT_CHARS.contains(first) && trimmed.chars().all(|c| c == first)
}

/// Assign a heading level based on (adornment_char, has_overline) pair.
///
/// RST heading levels are defined by the order of first appearance of each
/// unique adornment style, not by which character is used. Returns 1-6,
/// clamped at 6 for deeply nested headings.
fn get_or_assign_level(levels: &mut Vec<(char, bool)>, adorn_char: char, has_overline: bool) -> u8 {
    for (idx, &(c, o)) in levels.iter().enumerate() {
        if c == adorn_char && o == has_overline {
            return (idx as u8 + 1).min(6);
        }
    }
    levels.push((adorn_char, has_overline));
    (levels.len() as u8).min(6)
}

/// Extract inline RST links from a text line.
///
/// Handles:
/// - Inline references: `` `link text <http://example.com>`_ ``
/// - Bare URLs: `http://` or `https://` followed by non-whitespace
fn extract_inline_rst_links(line: &str, links: &mut Vec<ExtractedLink>) {
    // Pattern: `text <url>`_
    for cap in RST_INLINE_REF_RE.captures_iter(line) {
        links.push(ExtractedLink {
            url: cap[2].to_string(),
            text: Some(cap[1].to_string()),
            is_internal: false,
        });
    }

    // Bare URLs
    for mat in RST_BARE_URL_RE.find_iter(line) {
        let url = mat.as_str().to_string();
        // Skip if already captured as an inline ref
        if !links.iter().any(|l| l.url == url) {
            links.push(ExtractedLink {
                url,
                text: None,
                is_internal: false,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rst_headings() {
        let content = "\
============
Main Title
============

Introduction
------------

Some text here.

Subsection
~~~~~~~~~~

More text.
";
        let doc = parse_rst(content);
        assert_eq!(doc.title, Some("Main Title".to_string()));
        assert_eq!(doc.headings.len(), 3);
        assert_eq!(doc.headings[0].level, 1);
        assert_eq!(doc.headings[0].text, "Main Title");
        assert_eq!(doc.headings[1].level, 2);
        assert_eq!(doc.headings[1].text, "Introduction");
        assert_eq!(doc.headings[2].level, 3);
        assert_eq!(doc.headings[2].text, "Subsection");
    }

    #[test]
    fn test_rst_code_block() {
        let content = "\
Example
-------

.. code-block:: python

    def hello():
        print(\"Hello\")

Some text after.
";
        let doc = parse_rst(content);
        assert_eq!(doc.code_blocks.len(), 1);
        assert_eq!(doc.code_blocks[0].language, Some("python".to_string()));
        assert!(doc.code_blocks[0].content.contains("def hello()"));
    }

    #[test]
    fn test_rst_literal_block() {
        let content = "\
Example code::

    x = 42
    print(x)

Done.
";
        let doc = parse_rst(content);
        assert_eq!(doc.code_blocks.len(), 1);
        assert!(doc.code_blocks[0].content.contains("x = 42"));
    }

    #[test]
    fn test_rst_image_directive() {
        let content = "\
.. image:: images/logo.png
   :alt: Project Logo
   :width: 200px
";
        let doc = parse_rst(content);
        assert_eq!(doc.media.len(), 1);
        assert_eq!(doc.media[0].url, "images/logo.png");
        assert_eq!(doc.media[0].alt, Some("Project Logo".to_string()));
    }

    #[test]
    fn test_rst_inline_links() {
        let content = "See `Python docs <https://docs.python.org>`_ for details.\n";
        let doc = parse_rst(content);
        assert_eq!(doc.links.len(), 1);
        assert_eq!(doc.links[0].url, "https://docs.python.org");
        assert_eq!(doc.links[0].text, Some("Python docs".to_string()));
    }

    #[test]
    fn test_rst_metadata() {
        let content = "\
:Author: John Doe
:Version: 2.0

Title
=====

Content here.
";
        let doc = parse_rst(content);
        assert_eq!(doc.metadata.get("author"), Some(&"John Doe".to_string()));
        assert_eq!(doc.metadata.get("version"), Some(&"2.0".to_string()));
    }

    #[test]
    fn test_rst_reference_target() {
        let content = ".. _example-link: https://example.com\n\nSome text.\n";
        let doc = parse_rst(content);
        assert_eq!(doc.links.len(), 1);
        assert_eq!(doc.links[0].url, "https://example.com");
    }
}
