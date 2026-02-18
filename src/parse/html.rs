//! HTML parsing and text extraction

use super::{
    normalize_whitespace, CodeBlock, ContentType, ExtractedLink, ExtractedMedia, Heading,
    MediaModality, ParsedDocument,
};
use crate::error::Result;
use regex::Regex;
use scraper::{Html, Selector};
use std::collections::HashSet;
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
    let text = html2text::from_read(root.as_bytes(), 80).unwrap_or_else(|_| root.clone());
    doc.text = normalize_whitespace(&text);

    // Extract headings in DOM order using a combined selector, then assign
    // positions by scanning the extracted text sequentially so that repeated
    // heading text (e.g. two "Parameters" sections) gets distinct positions.
    if let Ok(selector) = Selector::parse("h1, h2, h3, h4, h5, h6") {
        let mut raw_headings: Vec<(u8, String)> = Vec::new();
        for elem in document.select(&selector) {
            let tag = elem.value().name();
            let level = tag.chars().nth(1).and_then(|c| c.to_digit(10)).unwrap_or(0) as u8;
            let heading_text = elem.text().collect::<String>().trim().to_string();
            if !heading_text.is_empty() && (1..=6).contains(&level) {
                raw_headings.push((level, heading_text));
            }
        }

        let mut search_offset = 0;
        for (level, heading_text) in &raw_headings {
            let position = doc.text[search_offset..]
                .find(heading_text.as_str())
                .map(|pos| pos + search_offset)
                .unwrap_or(search_offset);
            doc.headings.push(Heading {
                level: *level,
                text: heading_text.clone(),
                position,
            });
            search_offset = position + heading_text.len();
        }
    }

    // Extract code blocks
    if let Ok(selector) = Selector::parse("pre code, pre") {
        for elem in document.select(&selector) {
            let code_text = elem.text().collect::<String>();
            let language = elem.value().attr("class").and_then(|c| {
                c.split_whitespace()
                    .find(|cls| cls.starts_with("language-") || cls.starts_with("lang-"))
                    .map(|cls| {
                        cls.trim_start_matches("language-")
                            .trim_start_matches("lang-")
                            .to_string()
                    })
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
                let link_text = if link_text.is_empty() {
                    None
                } else {
                    Some(link_text)
                };

                // Resolve relative URLs
                let url = if let Some(ref base) = base {
                    base.join(href)
                        .map(|u| u.to_string())
                        .unwrap_or_else(|_| href.to_string())
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

    // Extract image/media candidates (img, picture/srcset, inline CSS backgrounds)
    // IMG tags with src
    if let Ok(selector) = Selector::parse("img") {
        let base = base_url.and_then(|u| Url::parse(u).ok());
        for elem in document.select(&selector) {
            let src = elem.value().attr("src");
            let alt = elem.value().attr("alt").map(|s| s.trim().to_string());
            let mut candidates: Vec<String> = Vec::new();
            if let Some(s) = src {
                candidates.push(s.to_string());
            }

            // Parse srcset and collect URLs (choose highest resolution later)
            if let Some(srcset) = elem.value().attr("srcset") {
                candidates.extend(parse_srcset_urls(srcset));
            }

            // Resolve and dedupe
            let mut seen: HashSet<String> = HashSet::new();
            for raw in candidates {
                let resolved = if let Some(ref base) = base {
                    base.join(&raw)
                        .map(|u| u.to_string())
                        .unwrap_or_else(|_| raw.clone())
                } else {
                    raw.clone()
                };
                if seen.insert(resolved.clone()) {
                    doc.media.push(ExtractedMedia {
                        url: resolved,
                        alt: alt.clone(),
                        tag: "img".to_string(),
                        css_background: false,
                        modality: MediaModality::Image,
                        mime_type: None,
                    });
                }
            }
        }
    }

    // picture/source with srcset
    if let Ok(selector) = Selector::parse("source") {
        let base = base_url.and_then(|u| Url::parse(u).ok());
        for elem in document.select(&selector) {
            if let Some(srcset) = elem.value().attr("srcset") {
                let mut seen: HashSet<String> = HashSet::new();
                for raw in parse_srcset_urls(srcset) {
                    let resolved = if let Some(ref base) = base {
                        base.join(&raw)
                            .map(|u| u.to_string())
                            .unwrap_or_else(|_| raw.clone())
                    } else {
                        raw.clone()
                    };
                    if seen.insert(resolved.clone()) {
                        doc.media.push(ExtractedMedia {
                            url: resolved,
                            alt: None,
                            tag: "source".to_string(),
                            css_background: false,
                            modality: MediaModality::Image,
                            mime_type: None,
                        });
                    }
                }
            }
        }
    }

    // Inline CSS background-image: url(...)
    if let Ok(selector) = Selector::parse("*[style]") {
        let base = base_url.and_then(|u| Url::parse(u).ok());
        let re = Regex::new(r#"background-image\s*:\s*url\(([^)]+)\)"#).ok();
        if let Some(ref regex) = re {
            for elem in document.select(&selector) {
                if let Some(style) = elem.value().attr("style") {
                    if let Some(caps) = regex.captures(style) {
                        if let Some(m) = caps.get(1) {
                            let mut raw = m.as_str().trim().to_string();
                            // Trim surrounding quotes if present
                            if (raw.starts_with('"') && raw.ends_with('"'))
                                || (raw.starts_with('\'') && raw.ends_with('\''))
                            {
                                raw = raw[1..raw.len() - 1].to_string();
                            }
                            let resolved = if let Some(ref base) = base {
                                base.join(&raw)
                                    .map(|u| u.to_string())
                                    .unwrap_or_else(|_| raw.clone())
                            } else {
                                raw.clone()
                            };
                            doc.media.push(ExtractedMedia {
                                url: resolved,
                                alt: None,
                                tag: "style".to_string(),
                                css_background: true,
                                modality: MediaModality::Image,
                                mime_type: None,
                            });
                        }
                    }
                }
            }
        }
    }

    // Extract <audio> elements with src or <source> children
    extract_audio_elements(&document, base_url, &mut doc);

    // Extract <video> elements with src or <source> children
    extract_video_elements(&document, base_url, &mut doc);

    Ok(doc)
}

/// Extract just the text content from HTML (simpler version)
pub fn extract_text_from_html(content: &str) -> String {
    let text = html2text::from_read(content.as_bytes(), 80).unwrap_or_else(|_| content.to_string());
    normalize_whitespace(&text)
}

/// Parse a srcset string into individual URLs (best-effort)
fn parse_srcset_urls(srcset: &str) -> Vec<String> {
    // srcset format: "url1 1x, url2 2x" or "url1 500w, url2 1000w"
    let mut urls = Vec::new();
    for part in srcset.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Split by whitespace and take the first token as URL
        if let Some((url, _descriptor)) = trimmed.split_once(' ') {
            urls.push(url.to_string());
        } else {
            // No descriptor, treat whole part as URL
            urls.push(trimmed.to_string());
        }
    }
    urls
}

/// Extract audio elements from HTML document
fn extract_audio_elements(document: &Html, base_url: Option<&str>, doc: &mut ParsedDocument) {
    let base = base_url.and_then(|u| Url::parse(u).ok());

    // <audio src="..."> with optional type attribute
    if let Ok(selector) = Selector::parse("audio[src]") {
        for elem in document.select(&selector) {
            if let Some(src) = elem.value().attr("src") {
                let resolved = resolve_url(&base, src);
                let mime_type = elem.value().attr("type").map(|s| s.to_string());
                doc.media.push(ExtractedMedia {
                    url: resolved,
                    alt: None,
                    tag: "audio".to_string(),
                    css_background: false,
                    modality: MediaModality::Audio,
                    mime_type,
                });
            }
        }
    }

    // <audio> with <source> children
    if let Ok(audio_selector) = Selector::parse("audio") {
        if let Ok(source_selector) = Selector::parse("source") {
            for audio_elem in document.select(&audio_selector) {
                // Skip if the audio element itself has src (already handled above)
                if audio_elem.value().attr("src").is_some() {
                    continue;
                }
                let mut seen: HashSet<String> = HashSet::new();
                for source_elem in audio_elem.select(&source_selector) {
                    if let Some(src) = source_elem.value().attr("src") {
                        let resolved = resolve_url(&base, src);
                        if seen.insert(resolved.clone()) {
                            let mime_type = source_elem.value().attr("type").map(|s| s.to_string());
                            doc.media.push(ExtractedMedia {
                                url: resolved,
                                alt: None,
                                tag: "audio-source".to_string(),
                                css_background: false,
                                modality: MediaModality::Audio,
                                mime_type,
                            });
                        }
                    }
                }
            }
        }
    }
}

/// Extract video elements from HTML document
fn extract_video_elements(document: &Html, base_url: Option<&str>, doc: &mut ParsedDocument) {
    let base = base_url.and_then(|u| Url::parse(u).ok());

    // <video src="..."> with optional type and poster attributes
    if let Ok(selector) = Selector::parse("video[src]") {
        for elem in document.select(&selector) {
            if let Some(src) = elem.value().attr("src") {
                let resolved = resolve_url(&base, src);
                let mime_type = elem.value().attr("type").map(|s| s.to_string());
                doc.media.push(ExtractedMedia {
                    url: resolved,
                    alt: elem.value().attr("poster").map(|s| s.to_string()),
                    tag: "video".to_string(),
                    css_background: false,
                    modality: MediaModality::Video,
                    mime_type,
                });
            }
        }
    }

    // <video> with <source> children
    if let Ok(video_selector) = Selector::parse("video") {
        if let Ok(source_selector) = Selector::parse("source") {
            for video_elem in document.select(&video_selector) {
                // Skip if the video element itself has src (already handled above)
                if video_elem.value().attr("src").is_some() {
                    continue;
                }
                let mut seen: HashSet<String> = HashSet::new();
                for source_elem in video_elem.select(&source_selector) {
                    if let Some(src) = source_elem.value().attr("src") {
                        let resolved = resolve_url(&base, src);
                        if seen.insert(resolved.clone()) {
                            let mime_type = source_elem.value().attr("type").map(|s| s.to_string());
                            doc.media.push(ExtractedMedia {
                                url: resolved,
                                alt: video_elem.value().attr("poster").map(|s| s.to_string()),
                                tag: "video-source".to_string(),
                                css_background: false,
                                modality: MediaModality::Video,
                                mime_type,
                            });
                        }
                    }
                }
            }
        }
    }
}

/// Resolve a URL against a base URL
fn resolve_url(base: &Option<Url>, raw: &str) -> String {
    if let Some(ref base) = base {
        base.join(raw)
            .map(|u| u.to_string())
            .unwrap_or_else(|_| raw.to_string())
    } else {
        raw.to_string()
    }
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

    #[test]
    fn test_image_extraction_basic() {
        let html = r#"
        <html><body>
            <img src="/images/diagram.png" alt="System Diagram" />
            <picture>
                <source srcset="/images/diagram-1x.png 1x, /images/diagram-2x.png 2x">
            </picture>
            <div style="background-image: url('/images/bg.jpg'); width:100px; height:100px"></div>
        </body></html>
        "#;
        let doc =
            parse_html(html, Some("https://example.com/docs")).expect("parse_html should succeed");
        assert!(doc
            .media
            .iter()
            .any(|m| m.url.ends_with("/images/diagram.png")));
        assert!(doc
            .media
            .iter()
            .any(|m| m.url.ends_with("/images/diagram-2x.png")));
        assert!(doc.media.iter().any(|m| m.url.ends_with("/images/bg.jpg")));
    }

    #[test]
    fn test_audio_extraction_basic() {
        let html = r#"
        <html><body>
            <audio src="/audio/podcast.mp3" type="audio/mpeg"></audio>
            <audio>
                <source src="/audio/music.ogg" type="audio/ogg">
                <source src="/audio/music.mp3" type="audio/mpeg">
            </audio>
        </body></html>
        "#;
        let doc = parse_html(html, Some("https://example.com")).expect("parse_html should succeed");

        let audio_media: Vec<_> = doc
            .media
            .iter()
            .filter(|m| m.modality == MediaModality::Audio)
            .collect();

        assert_eq!(audio_media.len(), 3);
        assert!(audio_media
            .iter()
            .any(|m| m.url.ends_with("/audio/podcast.mp3")));
        assert!(audio_media
            .iter()
            .any(|m| m.url.ends_with("/audio/music.ogg")));
        assert!(audio_media
            .iter()
            .any(|m| m.url.ends_with("/audio/music.mp3")));

        // Check MIME types are captured
        let podcast = audio_media
            .iter()
            .find(|m| m.url.ends_with("/audio/podcast.mp3"))
            .unwrap();
        assert_eq!(podcast.mime_type, Some("audio/mpeg".to_string()));
    }

    #[test]
    fn test_video_extraction_basic() {
        let html = r#"
        <html><body>
            <video src="/video/intro.mp4" type="video/mp4" poster="/images/poster.jpg"></video>
            <video poster="/images/poster2.jpg">
                <source src="/video/demo.webm" type="video/webm">
                <source src="/video/demo.mp4" type="video/mp4">
            </video>
        </body></html>
        "#;
        let doc = parse_html(html, Some("https://example.com")).expect("parse_html should succeed");

        let video_media: Vec<_> = doc
            .media
            .iter()
            .filter(|m| m.modality == MediaModality::Video)
            .collect();

        assert_eq!(video_media.len(), 3);
        assert!(video_media
            .iter()
            .any(|m| m.url.ends_with("/video/intro.mp4")));
        assert!(video_media
            .iter()
            .any(|m| m.url.ends_with("/video/demo.webm")));
        assert!(video_media
            .iter()
            .any(|m| m.url.ends_with("/video/demo.mp4")));

        // Check MIME types are captured
        let intro = video_media
            .iter()
            .find(|m| m.url.ends_with("/video/intro.mp4"))
            .unwrap();
        assert_eq!(intro.mime_type, Some("video/mp4".to_string()));

        // Check poster is captured as alt text
        assert!(intro
            .alt
            .as_ref()
            .is_some_and(|a| a.ends_with("/images/poster.jpg")));
    }

    #[test]
    fn test_repeated_headings_get_distinct_positions() {
        let html = r#"
        <html><body>
            <h2>Parameters</h2>
            <p>First parameters section content.</p>
            <h2>Parameters</h2>
            <p>Second parameters section content.</p>
        </body></html>
        "#;
        let doc = parse_html(html, None).unwrap();

        let param_headings: Vec<_> = doc
            .headings
            .iter()
            .filter(|h| h.text == "Parameters")
            .collect();
        assert_eq!(param_headings.len(), 2);
        assert!(
            param_headings[1].position > param_headings[0].position,
            "Second 'Parameters' heading must have a greater position than the first"
        );
    }

    #[test]
    fn test_repeated_headings_interleaved_order() {
        let html = r#"
        <html><body>
            <h2>Parameters</h2>
            <p>Params A.</p>
            <h2>Returns</h2>
            <p>Returns info.</p>
            <h2>Parameters</h2>
            <p>Params B.</p>
        </body></html>
        "#;
        let doc = parse_html(html, None).unwrap();

        assert_eq!(doc.headings.len(), 3);
        assert_eq!(doc.headings[0].text, "Parameters");
        assert_eq!(doc.headings[1].text, "Returns");
        assert_eq!(doc.headings[2].text, "Parameters");

        // Positions must be strictly increasing
        for window in doc.headings.windows(2) {
            assert!(
                window[1].position > window[0].position,
                "Heading '{}' at {} should come after '{}' at {}",
                window[1].text,
                window[1].position,
                window[0].text,
                window[0].position,
            );
        }
    }

    #[test]
    fn test_unique_headings_regression() {
        let html = r#"
        <html><body>
            <h1>Title</h1>
            <p>Intro text.</p>
            <h2>Section A</h2>
            <p>Content A.</p>
            <h2>Section B</h2>
            <p>Content B.</p>
        </body></html>
        "#;
        let doc = parse_html(html, None).unwrap();

        assert_eq!(doc.headings.len(), 3);
        assert_eq!(doc.headings[0].text, "Title");
        assert_eq!(doc.headings[1].text, "Section A");
        assert_eq!(doc.headings[2].text, "Section B");

        for window in doc.headings.windows(2) {
            assert!(window[1].position > window[0].position);
        }
    }
}
