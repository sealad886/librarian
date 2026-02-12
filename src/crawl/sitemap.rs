//! Sitemap XML parsing
//!
//! Supports:
//! - Standard sitemap.xml format
//! - Sitemap index files (sitemapindex)
//! - Recursive sitemap index resolution
//! - SSRF protection via async DNS validation

use super::ssrf::validate_url_ssrf;
use crate::error::{Error, Result};
use reqwest::Client;
use std::time::Duration;
use tracing::{debug, info, warn};
use url::Url;

/// A URL entry from a sitemap
#[derive(Debug, Clone)]
pub struct SitemapEntry {
    /// The page URL
    pub loc: String,
    /// Last modification time (optional)
    pub lastmod: Option<String>,
    /// Change frequency (optional)
    pub changefreq: Option<String>,
    /// Priority (optional)
    pub priority: Option<f32>,
}

/// Sitemap parser
pub struct SitemapParser {
    client: Client,
    #[allow(dead_code)]
    user_agent: String,
    max_sitemaps: usize,
}

impl SitemapParser {
    /// Create a new sitemap parser
    pub fn new(user_agent: &str) -> Result<Self> {
        let client = Client::builder()
            .user_agent(user_agent)
            .timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none()) // Disable redirects to prevent SSRF bypass
            .gzip(true)
            .build()
            .map_err(|e| Error::Crawl(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            user_agent: user_agent.to_string(),
            max_sitemaps: 50, // Limit recursion for sitemap indexes
        })
    }

    /// Parse a sitemap URL and return all page URLs
    pub async fn parse(&self, sitemap_url: &str) -> Result<Vec<SitemapEntry>> {
        debug!("Parsing sitemap: {}", sitemap_url);

        // Validate initial sitemap URL for SSRF protection
        validate_url_ssrf(sitemap_url).await?;

        let mut all_entries = Vec::new();
        let mut sitemaps_processed = 0;

        // Start with the initial sitemap
        let mut sitemap_queue = vec![sitemap_url.to_string()];

        while let Some(url) = sitemap_queue.pop() {
            if sitemaps_processed >= self.max_sitemaps {
                warn!(
                    "Reached max sitemap limit ({}), stopping",
                    self.max_sitemaps
                );
                break;
            }

            match self.fetch_and_parse(&url).await {
                Ok(ParseResult::UrlSet(entries)) => {
                    debug!("Found {} URLs in sitemap: {}", entries.len(), url);
                    all_entries.extend(entries);
                }
                Ok(ParseResult::SitemapIndex(sitemaps)) => {
                    debug!(
                        "Found sitemap index with {} sitemaps: {}",
                        sitemaps.len(),
                        url
                    );
                    sitemap_queue.extend(sitemaps);
                }
                Err(e) => {
                    warn!("Failed to parse sitemap {}: {}", url, e);
                }
            }

            sitemaps_processed += 1;
        }

        info!(
            "Parsed {} URLs from {} sitemaps",
            all_entries.len(),
            sitemaps_processed
        );
        Ok(all_entries)
    }

    /// Fetch and parse a single sitemap
    async fn fetch_and_parse(&self, url: &str) -> Result<ParseResult> {
        // Validate URL for SSRF protection before fetching
        validate_url_ssrf(url).await?;

        let response = self.client.get(url).send().await?;

        if !response.status().is_success() {
            return Err(Error::Crawl(format!("HTTP {}: {}", response.status(), url)));
        }

        let content = response.text().await?;

        // Detect sitemap type and parse
        if content.contains("<sitemapindex") {
            self.parse_sitemap_index(&content).await
        } else if content.contains("<urlset") {
            self.parse_urlset(&content).await
        } else {
            // Try to parse as plain text list of URLs
            self.parse_plain_text(&content).await
        }
    }

    /// Parse a urlset sitemap
    async fn parse_urlset(&self, content: &str) -> Result<ParseResult> {
        let mut entries = Vec::new();

        // Simple XML parsing using string operations
        // A full XML parser would be better but this keeps dependencies minimal
        for url_block in content.split("<url>").skip(1) {
            if let Some(end) = url_block.find("</url>") {
                let block = &url_block[..end];

                let loc = extract_tag(block, "loc");
                if let Some(loc) = loc {
                    // Validate URL
                    if Url::parse(&loc).is_ok() {
                        // Validate URL safety (SSRF protection)
                        if let Err(e) = validate_url_ssrf(&loc).await {
                            debug!(url = %loc, reason = %e, "Skipping URL from urlset sitemap (unsafe URL)");
                            continue;
                        }

                        entries.push(SitemapEntry {
                            loc,
                            lastmod: extract_tag(block, "lastmod"),
                            changefreq: extract_tag(block, "changefreq"),
                            priority: extract_tag(block, "priority").and_then(|s| s.parse().ok()),
                        });
                    }
                }
            }
        }

        Ok(ParseResult::UrlSet(entries))
    }

    /// Parse a sitemap index
    async fn parse_sitemap_index(&self, content: &str) -> Result<ParseResult> {
        let mut sitemaps = Vec::new();

        for sitemap_block in content.split("<sitemap>").skip(1) {
            if let Some(end) = sitemap_block.find("</sitemap>") {
                let block = &sitemap_block[..end];

                if let Some(loc) = extract_tag(block, "loc") {
                    if Url::parse(&loc).is_ok() {
                        // Validate URL safety (SSRF protection)
                        if let Err(e) = validate_url_ssrf(&loc).await {
                            debug!(url = %loc, reason = %e, "Skipping sitemap from index (unsafe URL)");
                            continue;
                        }

                        sitemaps.push(loc);
                    }
                }
            }
        }

        Ok(ParseResult::SitemapIndex(sitemaps))
    }

    /// Parse plain text list of URLs
    async fn parse_plain_text(&self, content: &str) -> Result<ParseResult> {
        let mut entries = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if (line.starts_with("http://") || line.starts_with("https://"))
                && Url::parse(line).is_ok()
            {
                // Validate URL safety (SSRF protection)
                if let Err(e) = validate_url_ssrf(line).await {
                    debug!(url = %line, reason = %e, "Skipping URL from plain text sitemap (unsafe URL)");
                    continue;
                }

                entries.push(SitemapEntry {
                    loc: line.to_string(),
                    lastmod: None,
                    changefreq: None,
                    priority: None,
                });
            }
        }

        Ok(ParseResult::UrlSet(entries))
    }
}

/// Result of parsing a sitemap
enum ParseResult {
    /// A urlset containing page URLs
    UrlSet(Vec<SitemapEntry>),
    /// A sitemap index containing links to other sitemaps
    SitemapIndex(Vec<String>),
}

/// Extract text content from an XML tag
fn extract_tag(content: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{}>", tag);
    let end_tag = format!("</{}>", tag);

    content.find(&start_tag).and_then(|start| {
        let value_start = start + start_tag.len();
        content[value_start..]
            .find(&end_tag)
            .map(|end| content[value_start..value_start + end].trim().to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_tag() {
        let xml = "<loc>https://example.com/page</loc>";
        assert_eq!(
            extract_tag(xml, "loc"),
            Some("https://example.com/page".to_string())
        );
    }

    #[tokio::test]
    async fn test_parse_urlset() {
        let parser = SitemapParser::new("test-agent").unwrap();
        let xml = r#"
        <?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://8.8.8.8/page1</loc>
                <lastmod>2024-01-01</lastmod>
                <priority>0.8</priority>
            </url>
            <url>
                <loc>https://1.1.1.1/page2</loc>
            </url>
        </urlset>
        "#;

        let result = parser.parse_urlset(xml).await.unwrap();
        if let ParseResult::UrlSet(entries) = result {
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].loc, "https://8.8.8.8/page1");
            assert_eq!(entries[0].priority, Some(0.8));
        } else {
            panic!("Expected UrlSet");
        }
    }

    #[tokio::test]
    async fn test_validate_url_ssrf_accepts_public_urls() {
        // Test with well-known public IP addresses to avoid DNS resolution in tests
        // Google DNS 8.8.8.8 - using direct IP to ensure it's public
        let result = validate_url_ssrf("http://8.8.8.8/page").await;
        assert!(result.is_ok(), "Should accept public IP 8.8.8.8");

        // Cloudflare DNS 1.1.1.1
        let result = validate_url_ssrf("https://1.1.1.1/").await;
        assert!(result.is_ok(), "Should accept public IP 1.1.1.1");
    }

    #[tokio::test]
    async fn test_validate_url_ssrf_rejects_private_ips_directly() {
        // Direct IP URLs should be rejected if they're private
        assert!(validate_url_ssrf("http://127.0.0.1/page").await.is_err());
        assert!(validate_url_ssrf("http://10.0.0.1/page").await.is_err());
        assert!(validate_url_ssrf("http://192.168.1.1/page").await.is_err());
        assert!(validate_url_ssrf("http://169.254.169.254/latest/meta-data")
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_validate_url_ssrf_rejects_unsupported_schemes() {
        assert!(validate_url_ssrf("ftp://example.com/file").await.is_err());
        assert!(validate_url_ssrf("file:///etc/passwd").await.is_err());
    }

    #[tokio::test]
    async fn test_parse_plain_text_filters_unsafe_urls() {
        let parser = SitemapParser::new("test-agent").unwrap();
        let content = r#"
https://8.8.8.8/page1
http://127.0.0.1/admin
https://1.1.1.1/page2
http://10.0.0.1/internal
http://169.254.169.254/metadata
        "#;

        let result = parser.parse_plain_text(content).await.unwrap();
        if let ParseResult::UrlSet(entries) = result {
            // Should only include the safe public URLs
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].loc, "https://8.8.8.8/page1");
            assert_eq!(entries[1].loc, "https://1.1.1.1/page2");
        } else {
            panic!("Expected UrlSet");
        }
    }

    #[tokio::test]
    async fn test_parse_urlset_filters_unsafe_urls() {
        let parser = SitemapParser::new("test-agent").unwrap();
        let xml = r#"
        <?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://8.8.8.8/page1</loc>
            </url>
            <url>
                <loc>http://127.0.0.1/admin</loc>
            </url>
            <url>
                <loc>https://1.1.1.1/page2</loc>
            </url>
            <url>
                <loc>http://192.168.1.1/internal</loc>
            </url>
        </urlset>
        "#;

        let result = parser.parse_urlset(xml).await.unwrap();
        if let ParseResult::UrlSet(entries) = result {
            // Should only include the safe public URLs
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].loc, "https://8.8.8.8/page1");
            assert_eq!(entries[1].loc, "https://1.1.1.1/page2");
        } else {
            panic!("Expected UrlSet");
        }
    }

    #[tokio::test]
    async fn test_parse_sitemap_index_filters_unsafe_urls() {
        let parser = SitemapParser::new("test-agent").unwrap();
        let xml = r#"
        <?xml version="1.0" encoding="UTF-8"?>
        <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <sitemap>
                <loc>https://8.8.8.8/sitemap1.xml</loc>
            </sitemap>
            <sitemap>
                <loc>http://10.0.0.1/internal-sitemap.xml</loc>
            </sitemap>
            <sitemap>
                <loc>https://1.1.1.1/sitemap2.xml</loc>
            </sitemap>
        </sitemapindex>
        "#;

        let result = parser.parse_sitemap_index(xml).await.unwrap();
        if let ParseResult::SitemapIndex(sitemaps) = result {
            // Should only include the safe public sitemap URLs
            assert_eq!(sitemaps.len(), 2);
            assert_eq!(sitemaps[0], "https://8.8.8.8/sitemap1.xml");
            assert_eq!(sitemaps[1], "https://1.1.1.1/sitemap2.xml");
        } else {
            panic!("Expected SitemapIndex");
        }
    }
}
