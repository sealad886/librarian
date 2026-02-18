//! AsciiDoc parsing and text extraction
//!
//! Extracts headings, code blocks (listing/source blocks), links, and images
//! from AsciiDoc documents. Uses regex-based heuristic parsing.

use super::{
    normalize_whitespace, CodeBlock, ContentType, ExtractedLink, ExtractedMedia, Heading,
    MediaModality, ParsedDocument,
};
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

// Pre-compiled regexes for link/image extraction (called per-line)
static ADOC_LINK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"link:([^\[]+)\[([^\]]*)\]").unwrap());
static ADOC_XREF_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<<([^,>]+)(?:,([^>]+))>>").unwrap());
static ADOC_BARE_URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?://[^\s\[\])+>]+").unwrap());
static ADOC_INLINE_IMG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"image:([^:\[]+)\[([^\]]*)\]").unwrap());

/// Parse AsciiDoc content into a [`ParsedDocument`].
///
/// Extracts:
/// - **Headings**: `= Title` (level 1) through `====== Heading` (level 6).
/// - **Code blocks**: Delimited listing blocks (`----`), source blocks
///   with `[source,lang]` attribute, and literal blocks (`....`).
/// - **Links**: `link:url[text]`, `https://url`, `<<anchor,text>>`.
/// - **Images**: `image::url[alt]` block and inline macros.
/// - **Metadata**: Document attributes (`:key: value`).
pub fn parse_asciidoc(content: &str) -> ParsedDocument {
    let mut text_parts: Vec<String> = Vec::new();
    let mut headings: Vec<Heading> = Vec::new();
    let mut code_blocks: Vec<CodeBlock> = Vec::new();
    let mut links: Vec<ExtractedLink> = Vec::new();
    let mut media: Vec<ExtractedMedia> = Vec::new();
    let mut metadata: HashMap<String, String> = HashMap::new();
    let mut title: Option<String> = None;

    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;
    let mut char_pos: usize = 0;
    let mut pending_source_lang: Option<String> = None;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        // Document attributes: :key: value
        if trimmed.starts_with(':') && trimmed.len() > 1 && !trimmed.starts_with("::") {
            if let Some((key, value)) = parse_attribute_line(trimmed) {
                metadata.insert(key, value);
                i += 1;
                continue;
            }
        }

        // ATX-style headings: = Title, == Section, etc.
        if let Some((level, heading_text)) = parse_atx_heading(trimmed) {
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
            i += 1;
            continue;
        }

        // Source block attribute: [source,lang]
        if trimmed.starts_with("[source") {
            pending_source_lang = trimmed
                .strip_prefix("[source")
                .and_then(|s| s.strip_suffix(']'))
                .and_then(|s| s.strip_prefix(','))
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            i += 1;
            continue;
        }

        // Delimited listing block (----)
        if trimmed.starts_with("----") && trimmed.chars().all(|c| c == '-') && trimmed.len() >= 4 {
            let language = pending_source_lang.take();
            i += 1;
            let mut code_lines: Vec<&str> = Vec::new();
            while i < lines.len() {
                if lines[i].trim().starts_with("----") && lines[i].trim().chars().all(|c| c == '-')
                {
                    i += 1;
                    break;
                }
                code_lines.push(lines[i]);
                i += 1;
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

        // Literal block (....)
        if trimmed.starts_with("....") && trimmed.chars().all(|c| c == '.') && trimmed.len() >= 4 {
            i += 1;
            let mut code_lines: Vec<&str> = Vec::new();
            while i < lines.len() {
                if lines[i].trim().starts_with("....") && lines[i].trim().chars().all(|c| c == '.')
                {
                    i += 1;
                    break;
                }
                code_lines.push(lines[i]);
                i += 1;
            }
            let code_content = code_lines.join("\n").trim_end().to_string();
            if !code_content.is_empty() {
                code_blocks.push(CodeBlock {
                    language: None,
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

        // Block image: image::path[alt]
        if trimmed.starts_with("image::") {
            if let Some((url, alt)) = parse_image_macro(trimmed, "image::") {
                media.push(ExtractedMedia {
                    url,
                    alt,
                    tag: "img".to_string(),
                    css_background: false,
                    modality: MediaModality::Image,
                    mime_type: None,
                });
            }
            i += 1;
            continue;
        }

        // Clear pending source lang if we reach a non-delimiter line
        pending_source_lang = None;

        // Extract inline links and images
        extract_asciidoc_links(line, &mut links);
        extract_inline_images(line, &mut media);

        text_parts.push(line.to_string());
        text_parts.push("\n".to_string());
        char_pos += line.len() + 1;
        i += 1;
    }

    let text = normalize_whitespace(&text_parts.join(""));

    if title.is_none() {
        title = metadata.get("doctitle").cloned();
    }

    ParsedDocument {
        title,
        text,
        content_type: ContentType::AsciiDoc,
        headings,
        code_blocks,
        links,
        media,
        metadata,
    }
}

/// Parse an AsciiDoc attribute line `:key: value`.
fn parse_attribute_line(line: &str) -> Option<(String, String)> {
    let without_leading = line.strip_prefix(':')?;
    let colon_pos = without_leading.find(':')?;
    let key = without_leading[..colon_pos].trim().to_lowercase();
    let value = without_leading[colon_pos + 1..].trim().to_string();
    if key.is_empty() || key.contains(' ') {
        return None;
    }
    Some((key, value))
}

/// Parse an ATX-style AsciiDoc heading: `= Title` to `====== H6`.
fn parse_atx_heading(line: &str) -> Option<(u8, String)> {
    if !line.starts_with('=') {
        return None;
    }
    let level = line.chars().take_while(|&c| c == '=').count();
    if level == 0 || level > 6 {
        return None;
    }
    let text = line[level..].trim().to_string();
    if text.is_empty() {
        return None;
    }
    Some((level as u8, text))
}

/// Parse an image macro: `image::path[alt text]`.
fn parse_image_macro(line: &str, prefix: &str) -> Option<(String, Option<String>)> {
    let rest = line.strip_prefix(prefix)?;
    let bracket_start = rest.find('[')?;
    let url = rest[..bracket_start].trim().to_string();
    let alt = rest[bracket_start + 1..]
        .strip_suffix(']')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if url.is_empty() {
        return None;
    }
    Some((url, alt))
}

/// Extract links from an AsciiDoc text line.
///
/// Handles:
/// - `link:url[text]` macros
/// - `https://...` bare URLs
/// - `<<anchor,text>>` cross-references
fn extract_asciidoc_links(line: &str, links: &mut Vec<ExtractedLink>) {
    // link:url[text]
    for cap in ADOC_LINK_RE.captures_iter(line) {
        links.push(ExtractedLink {
            url: cap[1].to_string(),
            text: if cap[2].is_empty() {
                None
            } else {
                Some(cap[2].to_string())
            },
            is_internal: false,
        });
    }

    // Cross-references: <<anchor,text>>
    for cap in ADOC_XREF_RE.captures_iter(line) {
        links.push(ExtractedLink {
            url: format!("#{}", &cap[1]),
            text: cap.get(2).map(|m| m.as_str().to_string()),
            is_internal: true,
        });
    }

    // Bare URLs
    for mat in ADOC_BARE_URL_RE.find_iter(line) {
        let url = mat.as_str().to_string();
        if !links.iter().any(|l| l.url == url) {
            links.push(ExtractedLink {
                url,
                text: None,
                is_internal: false,
            });
        }
    }
}

/// Extract inline image macros: `image:path[alt]` (single colon = inline).
fn extract_inline_images(line: &str, media: &mut Vec<ExtractedMedia>) {
    for cap in ADOC_INLINE_IMG_RE.captures_iter(line) {
        let url = cap[1].to_string();
        let alt = if cap[2].is_empty() {
            None
        } else {
            Some(cap[2].to_string())
        };
        media.push(ExtractedMedia {
            url,
            alt,
            tag: "img".to_string(),
            css_background: false,
            modality: MediaModality::Image,
            mime_type: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asciidoc_headings() {
        let content = "\
= Document Title

== Chapter One

Some text.

=== Section 1.1

More text.
";
        let doc = parse_asciidoc(content);
        assert_eq!(doc.title, Some("Document Title".to_string()));
        assert_eq!(doc.headings.len(), 3);
        assert_eq!(doc.headings[0].level, 1);
        assert_eq!(doc.headings[1].level, 2);
        assert_eq!(doc.headings[2].level, 3);
    }

    #[test]
    fn test_asciidoc_source_block() {
        let content = "\
[source,rust]
----
fn main() {
    println!(\"Hello\");
}
----
";
        let doc = parse_asciidoc(content);
        assert_eq!(doc.code_blocks.len(), 1);
        assert_eq!(doc.code_blocks[0].language, Some("rust".to_string()));
        assert!(doc.code_blocks[0].content.contains("fn main()"));
    }

    #[test]
    fn test_asciidoc_listing_block() {
        let content = "\
----
plain listing
no language
----
";
        let doc = parse_asciidoc(content);
        assert_eq!(doc.code_blocks.len(), 1);
        assert!(doc.code_blocks[0].language.is_none());
        assert!(doc.code_blocks[0].content.contains("plain listing"));
    }

    #[test]
    fn test_asciidoc_literal_block() {
        let content = "\
....
literal content
preserved
....
";
        let doc = parse_asciidoc(content);
        assert_eq!(doc.code_blocks.len(), 1);
        assert!(doc.code_blocks[0].content.contains("literal content"));
    }

    #[test]
    fn test_asciidoc_image() {
        let content = "image::images/diagram.png[Architecture Diagram]\n";
        let doc = parse_asciidoc(content);
        assert_eq!(doc.media.len(), 1);
        assert_eq!(doc.media[0].url, "images/diagram.png");
        assert_eq!(doc.media[0].alt, Some("Architecture Diagram".to_string()));
    }

    #[test]
    fn test_asciidoc_links() {
        let content = "See link:https://example.com[Example Site] for details.\n";
        let doc = parse_asciidoc(content);
        assert!(doc.links.iter().any(|l| l.url == "https://example.com"));
    }

    #[test]
    fn test_asciidoc_cross_reference() {
        let content = "See <<chapter1,Chapter 1>> for the introduction.\n";
        let doc = parse_asciidoc(content);
        assert_eq!(doc.links.len(), 1);
        assert_eq!(doc.links[0].url, "#chapter1");
        assert!(doc.links[0].is_internal);
    }

    #[test]
    fn test_asciidoc_attributes() {
        let content = "\
:author: Jane Smith
:version: 3.0

= Title

Content.
";
        let doc = parse_asciidoc(content);
        assert_eq!(doc.metadata.get("author"), Some(&"Jane Smith".to_string()));
        assert_eq!(doc.metadata.get("version"), Some(&"3.0".to_string()));
    }
}
