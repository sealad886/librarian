//! Markdown parsing and text extraction

use super::{
    CodeBlock, ContentType, ExtractedLink, ExtractedMedia, Heading, MediaModality, ParsedDocument,
};
use crate::error::Result;
use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};
use std::collections::HashMap;

/// Strip YAML frontmatter from markdown content.
///
/// Detects `---` delimiters at the start of the content, parses simple
/// `key: value` pairs between them, and returns the remaining body with the
/// extracted metadata map. Returns the original content and an empty map when
/// no valid frontmatter is found.
fn strip_frontmatter(content: &str) -> (&str, HashMap<String, String>) {
    let fence = if content.starts_with("---\n") {
        "---\n"
    } else if content.starts_with("---\r\n") {
        "---\r\n"
    } else {
        return (content, HashMap::new());
    };

    let after_open = &content[fence.len()..];

    let close_pos = if let Some(pos) = after_open.find("\n---\n") {
        pos + 1 // include the leading \n so the closing --- starts at the right spot
    } else if let Some(pos) = after_open.find("\n---\r\n") {
        pos + 1
    } else if after_open.ends_with("\n---") || after_open.ends_with("\n---\n") {
        // Frontmatter at very end of content
        if let Some(pos) = after_open.rfind("\n---") {
            pos + 1
        } else {
            return (content, HashMap::new());
        }
    } else {
        return (content, HashMap::new());
    };

    let yaml_block = &after_open[..close_pos - 1]; // exclude the \n before ---
    let mut metadata = HashMap::new();

    for line in yaml_block.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once(": ") {
            let key = key.trim().to_string();
            let value = value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();
            if !key.is_empty() {
                metadata.insert(key, value);
            }
        }
    }

    // Body starts after closing fence
    let body_start = fence.len() + close_pos + "---".len();
    let body = if body_start < content.len() {
        let rest = &content[body_start..];
        // Skip the newline immediately after closing ---
        if let Some(stripped) = rest.strip_prefix("\r\n") {
            stripped
        } else if let Some(stripped) = rest.strip_prefix('\n') {
            stripped
        } else {
            rest
        }
    } else {
        ""
    };

    (body, metadata)
}

/// Parse Markdown content and extract text
pub fn parse_markdown(content: &str) -> Result<ParsedDocument> {
    let (body, metadata) = strip_frontmatter(content);

    let parser = Parser::new(body);
    let mut doc = ParsedDocument::new(String::new(), ContentType::Markdown);

    let mut text_parts: Vec<String> = Vec::new();
    let mut current_heading: Option<(u8, Vec<String>)> = None;
    let mut in_code_block = false;
    let mut current_code: Vec<String> = Vec::new();
    let mut code_language: Option<String> = None;
    let mut current_link_url: Option<String> = None;
    let mut current_link_text: Vec<String> = Vec::new();
    let mut current_image_url: Option<String> = None;
    let mut current_image_alt: Vec<String> = Vec::new();
    let mut char_position = 0;

    for event in parser {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                current_heading = Some((heading_level_to_u8(level), Vec::new()));
            }
            Event::End(TagEnd::Heading(_)) => {
                if let Some((level, parts)) = current_heading.take() {
                    let heading_text = parts.join("").trim().to_string();
                    if !heading_text.is_empty() {
                        // First heading is often the title
                        if doc.title.is_none() && level == 1 {
                            doc.title = Some(heading_text.clone());
                        }

                        doc.headings.push(Heading {
                            level,
                            text: heading_text.clone(),
                            position: char_position,
                        });

                        text_parts.push(format!("\n{}\n", heading_text));
                        char_position += heading_text.len() + 2;
                    }
                }
            }
            Event::Start(Tag::CodeBlock(kind)) => {
                in_code_block = true;
                code_language = match kind {
                    pulldown_cmark::CodeBlockKind::Fenced(lang) => {
                        let lang_str = lang.to_string();
                        if lang_str.is_empty() {
                            None
                        } else {
                            Some(lang_str)
                        }
                    }
                    pulldown_cmark::CodeBlockKind::Indented => None,
                };
            }
            Event::End(TagEnd::CodeBlock) if in_code_block => {
                let code_content = current_code.join("");
                doc.code_blocks.push(CodeBlock {
                    language: code_language.take(),
                    content: code_content.clone(),
                    position: char_position,
                });
                text_parts.push(format!("\n```\n{}\n```\n", code_content));
                char_position += code_content.len() + 10;
                current_code.clear();
                in_code_block = false;
            }
            Event::Start(Tag::Link { dest_url, .. }) => {
                current_link_url = Some(dest_url.to_string());
            }
            Event::End(TagEnd::Link) => {
                if let Some(url) = current_link_url.take() {
                    let link_text = current_link_text.join("");
                    let link_text = if link_text.is_empty() {
                        None
                    } else {
                        Some(link_text)
                    };

                    let is_internal = !url.contains("://") || url.starts_with('#');

                    doc.links.push(ExtractedLink {
                        url,
                        text: link_text,
                        is_internal,
                    });
                    current_link_text.clear();
                }
            }
            Event::Start(Tag::Image { dest_url, .. }) => {
                current_image_url = Some(dest_url.to_string());
            }
            Event::End(TagEnd::Image) => {
                if let Some(url) = current_image_url.take() {
                    let alt = current_image_alt.join("");
                    let alt = if alt.is_empty() { None } else { Some(alt) };
                    doc.media.push(ExtractedMedia {
                        url,
                        alt: alt.clone(),
                        tag: "img".to_string(),
                        css_background: false,
                        modality: MediaModality::Image,
                        mime_type: None,
                    });
                    // Include alt text in document text for searchability
                    if let Some(ref alt_text) = alt {
                        text_parts.push(alt_text.clone());
                        char_position += alt_text.len();
                    }
                    current_image_alt.clear();
                }
            }
            Event::Text(text) => {
                let text_str = text.to_string();

                if let Some((_, ref mut parts)) = current_heading {
                    parts.push(text_str.clone());
                } else if in_code_block {
                    current_code.push(text_str);
                } else if current_image_url.is_some() {
                    current_image_alt.push(text_str);
                } else if current_link_url.is_some() {
                    current_link_text.push(text_str.clone());
                    text_parts.push(text_str.clone());
                    char_position += text_str.len();
                } else {
                    text_parts.push(text_str.clone());
                    char_position += text_str.len();
                }
            }
            Event::Code(code) => {
                let code_str = format!("`{}`", code);
                if let Some((_, ref mut parts)) = current_heading {
                    parts.push(code.to_string());
                } else {
                    text_parts.push(code_str.clone());
                    char_position += code_str.len();
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                text_parts.push(" ".to_string());
                char_position += 1;
            }
            Event::Start(Tag::Paragraph) => {}
            Event::End(TagEnd::Paragraph) => {
                text_parts.push("\n\n".to_string());
                char_position += 2;
            }
            Event::Start(Tag::List(_)) => {}
            Event::End(TagEnd::List(_)) => {
                text_parts.push("\n".to_string());
                char_position += 1;
            }
            Event::Start(Tag::Item) => {
                text_parts.push("• ".to_string());
                char_position += 2;
            }
            Event::End(TagEnd::Item) => {
                text_parts.push("\n".to_string());
                char_position += 1;
            }
            _ => {}
        }
    }

    doc.text = text_parts.join("").trim().to_string();
    doc.metadata = metadata;

    // Fall back to frontmatter title when no h1 heading was found
    if doc.title.is_none() {
        if let Some(fm_title) = doc.metadata.get("title") {
            doc.title = Some(fm_title.clone());
        }
    }

    Ok(doc)
}

fn heading_level_to_u8(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_markdown_basic() {
        let markdown = r#"
# Main Title

This is a paragraph with some text.

## Section One

More content here.

```rust
fn main() {
    println!("Hello");
}
```

### Subsection

- Item 1
- Item 2

[Link text](https://example.com)
"#;

        let doc = parse_markdown(markdown).unwrap();

        assert_eq!(doc.title, Some("Main Title".to_string()));
        assert!(doc.text.contains("paragraph"));
        assert!(doc.headings.len() >= 3);
        assert!(!doc.code_blocks.is_empty());
        assert_eq!(doc.code_blocks[0].language, Some("rust".to_string()));
        assert!(!doc.links.is_empty());
    }

    #[test]
    fn test_heading_hierarchy() {
        let markdown = "# H1\n## H2\n### H3\n## Another H2";
        let doc = parse_markdown(markdown).unwrap();

        assert_eq!(doc.headings.len(), 4);
        assert_eq!(doc.headings[0].level, 1);
        assert_eq!(doc.headings[1].level, 2);
        assert_eq!(doc.headings[2].level, 3);
        assert_eq!(doc.headings[3].level, 2);
    }

    #[test]
    fn test_code_blocks() {
        let markdown = "```python\nprint('hello')\n```\n\n```\nplain code\n```";
        let doc = parse_markdown(markdown).unwrap();

        assert_eq!(doc.code_blocks.len(), 2);
        assert_eq!(doc.code_blocks[0].language, Some("python".to_string()));
        assert_eq!(doc.code_blocks[1].language, None);
    }

    #[test]
    fn test_image_extraction_with_alt() {
        let markdown = "![Alt text](image.png)";
        let doc = parse_markdown(markdown).unwrap();

        assert_eq!(doc.media.len(), 1);
        assert_eq!(doc.media[0].url, "image.png");
        assert_eq!(doc.media[0].alt, Some("Alt text".to_string()));
        assert_eq!(doc.media[0].tag, "img");
        assert!(!doc.media[0].css_background);
    }

    #[test]
    fn test_image_extraction_multiple() {
        let markdown = "![A](a.png)\n\n![B](b.jpg)\n\n![C](c.svg)";
        let doc = parse_markdown(markdown).unwrap();

        assert_eq!(doc.media.len(), 3);
        assert_eq!(doc.media[0].url, "a.png");
        assert_eq!(doc.media[1].url, "b.jpg");
        assert_eq!(doc.media[2].url, "c.svg");
    }

    #[test]
    fn test_image_extraction_no_alt() {
        let markdown = "![](image.png)";
        let doc = parse_markdown(markdown).unwrap();

        assert_eq!(doc.media.len(), 1);
        assert_eq!(doc.media[0].url, "image.png");
        assert_eq!(doc.media[0].alt, None);
    }

    #[test]
    fn test_links_and_images_both_extracted() {
        let markdown = "[Link](https://example.com)\n\n![Photo](photo.jpg)";
        let doc = parse_markdown(markdown).unwrap();

        assert_eq!(doc.links.len(), 1);
        assert_eq!(doc.links[0].url, "https://example.com");
        assert_eq!(doc.media.len(), 1);
        assert_eq!(doc.media[0].url, "photo.jpg");
    }

    #[test]
    fn test_frontmatter_with_h1_heading() {
        let markdown = "---\ntitle: Test\ndescription: A test doc\n---\n# Heading\nContent";
        let doc = parse_markdown(markdown).unwrap();

        // Title comes from h1, not frontmatter
        assert_eq!(doc.title, Some("Heading".to_string()));
        assert_eq!(doc.metadata.get("title").unwrap(), "Test");
        assert_eq!(doc.metadata.get("description").unwrap(), "A test doc");
    }

    #[test]
    fn test_no_frontmatter() {
        let markdown = "# Normal Doc\n\nJust some content.";
        let doc = parse_markdown(markdown).unwrap();

        assert!(doc.metadata.is_empty());
        assert_eq!(doc.title, Some("Normal Doc".to_string()));
        assert!(doc.text.contains("Just some content"));
    }

    #[test]
    fn test_frontmatter_title_fallback() {
        let markdown = "---\ntitle: My Title\n---\nJust content";
        let doc = parse_markdown(markdown).unwrap();

        // No h1, so title falls back to frontmatter
        assert_eq!(doc.title, Some("My Title".to_string()));
    }

    #[test]
    fn test_frontmatter_not_in_text() {
        let markdown = "---\ntitle: Hidden\nauthor: Someone\n---\n# Visible\nBody text";
        let doc = parse_markdown(markdown).unwrap();

        assert!(!doc.text.contains("Hidden"));
        assert!(!doc.text.contains("Someone"));
        assert!(doc.text.contains("Visible"));
        assert!(doc.text.contains("Body text"));
    }
}
