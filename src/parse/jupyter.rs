//! Jupyter Notebook (`.ipynb`) parsing and text extraction
//!
//! Parses the JSON notebook format, extracting markdown cells as prose,
//! code cells as code blocks, and cell outputs as supplementary text.
//! Notebook metadata is captured as document metadata.

use super::{
    normalize_whitespace, CodeBlock, ContentType, ExtractedLink, ExtractedMedia, Heading,
    ParsedDocument,
};
use crate::error::{Error, Result};
use serde::Deserialize;
use std::collections::HashMap;

/// Top-level notebook structure (nbformat 4).
#[derive(Debug, Deserialize)]
struct Notebook {
    cells: Vec<Cell>,
    metadata: Option<NotebookMetadata>,
    #[serde(default)]
    nbformat: u32,
}

/// A single notebook cell.
#[derive(Debug, Deserialize)]
struct Cell {
    cell_type: String,
    source: CellSource,
    #[serde(default)]
    outputs: Vec<CellOutput>,
}

/// Cell source can be a single string or an array of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CellSource {
    Single(String),
    Lines(Vec<String>),
}

impl std::fmt::Display for CellSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CellSource::Single(s) => f.write_str(s),
            CellSource::Lines(lines) => {
                for line in lines {
                    f.write_str(line)?;
                }
                Ok(())
            }
        }
    }
}

/// Cell output (simplified — only text outputs are captured).
#[derive(Debug, Deserialize)]
struct CellOutput {
    #[allow(dead_code)]
    output_type: Option<String>,
    text: Option<CellSource>,
    data: Option<OutputData>,
}

/// Output data MIME bundle.
#[derive(Debug, Deserialize)]
struct OutputData {
    #[serde(rename = "text/plain")]
    text_plain: Option<CellSource>,
    #[allow(dead_code)]
    #[serde(rename = "image/png")]
    image_png: Option<String>,
}

/// Notebook-level metadata.
#[derive(Debug, Deserialize)]
struct NotebookMetadata {
    kernelspec: Option<Kernelspec>,
    language_info: Option<LanguageInfo>,
}

#[derive(Debug, Deserialize)]
struct Kernelspec {
    display_name: Option<String>,
    language: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LanguageInfo {
    name: Option<String>,
}

/// Parse a Jupyter Notebook (`.ipynb`) JSON string into a [`ParsedDocument`].
///
/// # Errors
///
/// Returns an error if the content is not valid notebook JSON.
pub fn parse_jupyter(content: &str) -> Result<ParsedDocument> {
    let notebook: Notebook = serde_json::from_str(content)
        .map_err(|e| Error::Parse(format!("Invalid notebook JSON: {}", e)))?;

    let language = infer_language(&notebook);
    let mut text_parts: Vec<String> = Vec::new();
    let mut headings: Vec<Heading> = Vec::new();
    let mut code_blocks: Vec<CodeBlock> = Vec::new();
    let mut links: Vec<ExtractedLink> = Vec::new();
    let mut media: Vec<ExtractedMedia> = Vec::new();
    let mut metadata: HashMap<String, String> = HashMap::new();
    let mut title: Option<String> = None;
    let mut char_pos: usize = 0;

    // Extract notebook metadata
    if let Some(meta) = &notebook.metadata {
        if let Some(ks) = &meta.kernelspec {
            if let Some(name) = &ks.display_name {
                metadata.insert("kernel".to_string(), name.clone());
            }
            if let Some(lang) = &ks.language {
                metadata.insert("language".to_string(), lang.clone());
            }
        }
        if let Some(li) = &meta.language_info {
            if let Some(name) = &li.name {
                metadata.insert("language".to_string(), name.clone());
            }
        }
    }
    metadata.insert("nbformat".to_string(), notebook.nbformat.to_string());

    for cell in &notebook.cells {
        let source = cell.source.to_string();
        if source.trim().is_empty() {
            continue;
        }

        match cell.cell_type.as_str() {
            "markdown" => {
                // Extract headings from markdown cells
                for line in source.lines() {
                    if line.starts_with('#') {
                        let level = line.chars().take_while(|&c| c == '#').count() as u8;
                        let text = line[level as usize..].trim().to_string();
                        if !text.is_empty() && level <= 6 {
                            if title.is_none() && level == 1 {
                                title = Some(text.clone());
                            }
                            headings.push(Heading {
                                level,
                                text,
                                position: char_pos,
                            });
                        }
                    }

                    // Extract markdown links: [text](url)
                    extract_markdown_links(line, &mut links);
                    // Extract markdown images: ![alt](url)
                    extract_markdown_images(line, &mut media);
                }

                text_parts.push(source.clone());
                text_parts.push("\n\n".to_string());
                char_pos += source.len() + 2;
            }
            "code" => {
                code_blocks.push(CodeBlock {
                    language: language.clone(),
                    content: source.clone(),
                    position: char_pos,
                });

                // Include code in text for search but mark it
                text_parts.push(source.clone());
                text_parts.push("\n\n".to_string());
                char_pos += source.len() + 2;

                // Extract text from outputs
                for output in &cell.outputs {
                    if let Some(text_output) = extract_output_text(output) {
                        if !text_output.trim().is_empty() {
                            text_parts.push(text_output.clone());
                            text_parts.push("\n\n".to_string());
                            char_pos += text_output.len() + 2;
                        }
                    }
                }
            }
            "raw" => {
                text_parts.push(source.clone());
                text_parts.push("\n\n".to_string());
                char_pos += source.len() + 2;
            }
            _ => {}
        }
    }

    let text = normalize_whitespace(&text_parts.join(""));

    Ok(ParsedDocument {
        title,
        text,
        content_type: ContentType::Jupyter,
        headings,
        code_blocks,
        links,
        media,
        metadata,
    })
}

/// Infer the programming language from notebook metadata.
fn infer_language(notebook: &Notebook) -> Option<String> {
    notebook.metadata.as_ref().and_then(|m| {
        m.language_info
            .as_ref()
            .and_then(|li| li.name.clone())
            .or_else(|| m.kernelspec.as_ref().and_then(|ks| ks.language.clone()))
    })
}

/// Extract text content from a cell output.
fn extract_output_text(output: &CellOutput) -> Option<String> {
    // stream or execute_result text
    if let Some(text) = &output.text {
        return Some(text.to_string());
    }
    // data bundle text/plain
    if let Some(data) = &output.data {
        if let Some(text) = &data.text_plain {
            return Some(text.to_string());
        }
    }
    None
}

/// Extract markdown-style links from a line: `[text](url)`.
fn extract_markdown_links(line: &str, links: &mut Vec<ExtractedLink>) {
    let re = regex::Regex::new(r"\[([^\]]*)\]\(([^)]+)\)").unwrap();
    for cap in re.captures_iter(line) {
        // Skip image links (prefixed with !)
        let start = cap.get(0).unwrap().start();
        if start > 0 && line.as_bytes()[start - 1] == b'!' {
            continue;
        }
        links.push(ExtractedLink {
            url: cap[2].to_string(),
            text: if cap[1].is_empty() {
                None
            } else {
                Some(cap[1].to_string())
            },
            is_internal: cap[2].starts_with('#') || cap[2].starts_with('.'),
        });
    }
}

/// Extract markdown-style images from a line: `![alt](url)`.
fn extract_markdown_images(line: &str, media: &mut Vec<ExtractedMedia>) {
    let re = regex::Regex::new(r"!\[([^\]]*)\]\(([^)]+)\)").unwrap();
    for cap in re.captures_iter(line) {
        media.push(ExtractedMedia {
            url: cap[2].to_string(),
            alt: if cap[1].is_empty() {
                None
            } else {
                Some(cap[1].to_string())
            },
            tag: "img".to_string(),
            css_background: false,
            modality: super::MediaModality::Image,
            mime_type: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_notebook() -> &'static str {
        r###"{
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python"
                },
                "language_info": {
                    "name": "python"
                }
            },
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Notebook Title\n", "\n", "Some intro text.\n"],
                    "metadata": {}
                },
                {
                    "cell_type": "code",
                    "source": "import pandas as pd\ndf = pd.read_csv('data.csv')",
                    "metadata": {},
                    "outputs": [
                        {
                            "output_type": "execute_result",
                            "data": {
                                "text/plain": "   col1  col2\n0     1     2"
                            }
                        }
                    ]
                },
                {
                    "cell_type": "markdown",
                    "source": "## Analysis\n\nSee [pandas docs](https://pandas.pydata.org) for help.\n\n![chart](output.png)\n",
                    "metadata": {}
                },
                {
                    "cell_type": "code",
                    "source": ["df.describe()"],
                    "metadata": {},
                    "outputs": []
                }
            ]
        }"###
    }

    #[test]
    fn test_jupyter_parse_basic() {
        let doc = parse_jupyter(sample_notebook()).unwrap();
        assert_eq!(doc.title, Some("Notebook Title".to_string()));
        assert_eq!(doc.content_type, ContentType::Jupyter);
    }

    #[test]
    fn test_jupyter_headings() {
        let doc = parse_jupyter(sample_notebook()).unwrap();
        assert_eq!(doc.headings.len(), 2);
        assert_eq!(doc.headings[0].text, "Notebook Title");
        assert_eq!(doc.headings[0].level, 1);
        assert_eq!(doc.headings[1].text, "Analysis");
        assert_eq!(doc.headings[1].level, 2);
    }

    #[test]
    fn test_jupyter_code_blocks() {
        let doc = parse_jupyter(sample_notebook()).unwrap();
        assert_eq!(doc.code_blocks.len(), 2);
        assert_eq!(doc.code_blocks[0].language, Some("python".to_string()));
        assert!(doc.code_blocks[0].content.contains("import pandas"));
    }

    #[test]
    fn test_jupyter_links() {
        let doc = parse_jupyter(sample_notebook()).unwrap();
        assert!(doc
            .links
            .iter()
            .any(|l| l.url == "https://pandas.pydata.org"));
    }

    #[test]
    fn test_jupyter_images() {
        let doc = parse_jupyter(sample_notebook()).unwrap();
        assert!(doc.media.iter().any(|m| m.url == "output.png"));
    }

    #[test]
    fn test_jupyter_metadata() {
        let doc = parse_jupyter(sample_notebook()).unwrap();
        assert_eq!(doc.metadata.get("language"), Some(&"python".to_string()));
        assert_eq!(doc.metadata.get("nbformat"), Some(&"4".to_string()));
    }

    #[test]
    fn test_jupyter_output_text_included() {
        let doc = parse_jupyter(sample_notebook()).unwrap();
        // Output text should be included in the searchable text
        assert!(doc.text.contains("col1"));
    }

    #[test]
    fn test_jupyter_invalid_json() {
        let result = parse_jupyter("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_jupyter_empty_notebook() {
        let content = r#"{"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}"#;
        let doc = parse_jupyter(content).unwrap();
        assert!(doc.title.is_none());
        assert!(doc.headings.is_empty());
        assert!(doc.code_blocks.is_empty());
    }
}
