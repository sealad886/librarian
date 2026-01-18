//! Markdown parsing and text extraction

use super::{CodeBlock, ContentType, ExtractedLink, Heading, ParsedDocument};
use crate::error::Result;
use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};

/// Parse Markdown content and extract text
pub fn parse_markdown(content: &str) -> Result<ParsedDocument> {
    let parser = Parser::new(content);
    let mut doc = ParsedDocument::new(String::new(), ContentType::Markdown);

    let mut text_parts: Vec<String> = Vec::new();
    let mut current_heading: Option<(u8, Vec<String>)> = None;
    let mut in_code_block = false;
    let mut current_code: Vec<String> = Vec::new();
    let mut code_language: Option<String> = None;
    let mut current_link_url: Option<String> = None;
    let mut current_link_text: Vec<String> = Vec::new();
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
            Event::End(TagEnd::CodeBlock) => {
                if in_code_block {
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
            Event::Text(text) => {
                let text_str = text.to_string();

                if let Some((_, ref mut parts)) = current_heading {
                    parts.push(text_str.clone());
                } else if in_code_block {
                    current_code.push(text_str);
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
        assert!(doc.code_blocks.len() >= 1);
        assert_eq!(doc.code_blocks[0].language, Some("rust".to_string()));
        assert!(doc.links.len() >= 1);
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
}
