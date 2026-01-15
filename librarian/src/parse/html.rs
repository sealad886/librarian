//! HTML parsing and text extraction

use super::{CodeBlock, ContentType, ExtractedLink, Heading, ParsedDocument, normalize_whitespace};
use crate::error::Result;
use scraper::{Html, Selector};
use url::Url;

/// Parse HTML content and extract text
pub fn parse_html(content: &str, base_url: Option<&str>) -> Result<ParsedDocument> {
    let document = Html::parse_document(content);
    let mut doc = ParsedDocument::new(String::new(), ContentType::Html);

    // Extract title
    if let Ok(selector) = Selector::parse("title") {
        if let Some(title_elem) = document.select(&selector).next() {
            doc.title = Some(title_elem.text().collect::<String>().trim().to_string());
        }
    }

    // Remove script and style elements from consideration
    let body_selector = Selector::parse("body").ok();
    let root = body_selector
        .as_ref()
        .and_then(|s| document.select(s).next())
        .map(|e| e.html())
        .unwrap_or_else(|| content.to_string());

    // Use html2text for main content extraction
    let text = html2text::from_read(root.as_bytes(), 80)
        .unwrap_or_else(|_| root.clone());
    doc.text = normalize_whitespace(&text);

    // Extract headings
    for level in 1..=6 {
        if let Ok(selector) = Selector::parse(&format!("h{}", level)) {
            for elem in document.select(&selector) {
                let heading_text = elem.text().collect::<String>().trim().to_string();
                if !heading_text.is_empty() {
                    // Approximate position based on text content
                    let position = doc.text.find(&heading_text).unwrap_or(0);
                    doc.headings.push(Heading {
                        level,
                        text: heading_text,
                        position,
                    });
                }
            }
        }
    }

    // Sort headings by position
    doc.headings.sort_by_key(|h| h.position);

    // Extract code blocks
    if let Ok(selector) = Selector::parse("pre code, pre") {
        for elem in document.select(&selector) {
            let code_text = elem.text().collect::<String>();
            let language = elem
                .value()
                .attr("class")
                .and_then(|c| {
                    c.split_whitespace()
                        .find(|cls| cls.starts_with("language-") || cls.starts_with("lang-"))
                        .map(|cls| cls.trim_start_matches("language-").trim_start_matches("lang-").to_string())
                });
            
            let position = doc.text.find(&code_text).unwrap_or(0);
            doc.code_blocks.push(CodeBlock {
                language,
                content: code_text,
                position,
            });
        }
    }

    // Extract links
    if let Ok(selector) = Selector::parse("a[href]") {
        let base = base_url.and_then(|u| Url::parse(u).ok());
        
        for elem in document.select(&selector) {
            if let Some(href) = elem.value().attr("href") {
                let link_text = elem.text().collect::<String>().trim().to_string();
                let link_text = if link_text.is_empty() { None } else { Some(link_text) };

                // Resolve relative URLs
                let url = if let Some(ref base) = base {
                    base.join(href).map(|u| u.to_string()).unwrap_or_else(|_| href.to_string())
                } else {
                    href.to_string()
                };

                // Determine if internal
                let is_internal = if let Some(ref base) = base {
                    if let Ok(link_url) = Url::parse(&url) {
                        link_url.host() == base.host()
                    } else {
                        href.starts_with('/') || href.starts_with('#') || !href.contains("://")
                    }
                } else {
                    !href.contains("://")
                };

                doc.links.push(ExtractedLink {
                    url,
                    text: link_text,
                    is_internal,
                });
            }
        }
    }

    Ok(doc)
}

/// Extract just the text content from HTML (simpler version)
pub fn extract_text_from_html(content: &str) -> String {
    let text = html2text::from_read(content.as_bytes(), 80)
        .unwrap_or_else(|_| content.to_string());
    normalize_whitespace(&text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_html_basic() {
        let html = r#"
        <!DOCTYPE html>
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Main Heading</h1>
            <p>Some paragraph text here.</p>
            <h2>Sub Heading</h2>
            <p>More text.</p>
            <pre><code class="language-rust">fn main() {}</code></pre>
            <a href="/other">Link</a>
        </body>
        </html>
        "#;

        let doc = parse_html(html, Some("https://example.com")).unwrap();
        
        assert_eq!(doc.title, Some("Test Page".to_string()));
        assert!(doc.text.contains("Main Heading"));
        assert!(doc.text.contains("paragraph text"));
        assert!(doc.headings.len() >= 2);
    }

    #[test]
    fn test_link_extraction() {
        let html = r#"
        <html>
        <body>
            <a href="/internal">Internal</a>
            <a href="https://external.com/page">External</a>
            <a href="relative/path">Relative</a>
        </body>
        </html>
        "#;

        let doc = parse_html(html, Some("https://example.com")).unwrap();
        
        assert_eq!(doc.links.len(), 3);
        assert!(doc.links[0].is_internal);
        assert!(!doc.links[1].is_internal);
    }

    #[test]
    fn test_extract_text_simple() {
        let html = "<html><body><p>Hello <strong>world</strong>!</p></body></html>";
        let text = extract_text_from_html(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
    }
}
