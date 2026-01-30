//! Sitemap XML parsing
//!
//! Supports:
//! - Standard sitemap.xml format
//! - Sitemap index files (sitemapindex)
//! - Recursive sitemap index resolution

use crate::error::{Error, Result};
use reqwest::Client;
use std::net::{IpAddr, ToSocketAddrs};
use std::time::Duration;
use tracing::{debug, info, warn};
use url::Url;

/// Check if an IP address is in a private or reserved range to prevent SSRF attacks
fn is_ip_address_safe(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ipv4) => {
            // Reject private IP ranges (RFC 1918)
            if ipv4.is_private() {
                return false;
            }
            // Reject loopback (127.0.0.0/8)
            if ipv4.is_loopback() {
                return false;
            }
            // Reject link-local (169.254.0.0/16) - includes cloud metadata endpoints
            if ipv4.is_link_local() {
                return false;
            }
            // Reject broadcast
            if ipv4.is_broadcast() {
                return false;
            }
            // Reject unspecified (0.0.0.0)
            if ipv4.is_unspecified() {
                return false;
            }
            // Reject multicast
            if ipv4.is_multicast() {
                return false;
            }
            // Reject reserved (240.0.0.0/4) - check for class E addresses
            let octets = ipv4.octets();
            if octets[0] >= 240 {
                return false;
            }

            true
        }
        IpAddr::V6(ipv6) => {
            // Check for IPv4-mapped IPv6 addresses (::ffff:0:0/96)
            if let Some(ipv4) = ipv6.to_ipv4_mapped() {
                return is_ip_address_safe(IpAddr::V4(ipv4));
            }

            // Reject loopback (::1)
            if ipv6.is_loopback() {
                return false;
            }
            // Reject unspecified (::)
            if ipv6.is_unspecified() {
                return false;
            }
            // Reject multicast
            if ipv6.is_multicast() {
                return false;
            }
            // Reject unique local addresses (fc00::/7)
            let segments = ipv6.segments();
            if (segments[0] & 0xfe00) == 0xfc00 {
                return false;
            }
            // Reject link-local (fe80::/10)
            if (segments[0] & 0xffc0) == 0xfe80 {
                return false;
            }

            true
        }
    }
}

/// Validate that a URL's resolved IP address is safe to request (not private/reserved)
fn validate_url_safety(url: &str) -> std::result::Result<(), String> {
    let parsed = Url::parse(url).map_err(|e| format!("Failed to parse URL: {}", e))?;

    let host = parsed
        .host_str()
        .ok_or_else(|| "URL has no host".to_string())?;

    // Only validate http/https URLs
    let scheme = parsed.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(format!("Unsupported URL scheme: {}", scheme));
    }

    // Resolve hostname to IP addresses
    let port = parsed.port_or_known_default().unwrap_or(80);
    let socket_addrs = format!("{}:{}", host, port)
        .to_socket_addrs()
        .map_err(|e| format!("Failed to resolve hostname '{}': {}", host, e))?;

    // Check all resolved IPs - if any are unsafe, reject the URL
    let mut has_valid_ip = false;
    for socket_addr in socket_addrs {
        let ip = socket_addr.ip();
        if !is_ip_address_safe(ip) {
            return Err(format!(
                "URL resolves to unsafe IP address {}: private or reserved ranges are not allowed",
                ip
            ));
        }
        has_valid_ip = true;
    }

    if !has_valid_ip {
        return Err("URL did not resolve to any IP addresses".to_string());
    }

    Ok(())
}

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
        info!("Parsing sitemap: {}", sitemap_url);

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
        let response = self.client.get(url).send().await?;

        if !response.status().is_success() {
            return Err(Error::Crawl(format!("HTTP {}: {}", response.status(), url)));
        }

        let content = response.text().await?;

        // Detect sitemap type and parse
        if content.contains("<sitemapindex") {
            self.parse_sitemap_index(&content)
        } else if content.contains("<urlset") {
            self.parse_urlset(&content)
        } else {
            // Try to parse as plain text list of URLs
            self.parse_plain_text(&content)
        }
    }

    /// Parse a urlset sitemap
    fn parse_urlset(&self, content: &str) -> Result<ParseResult> {
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
                        if let Err(e) = validate_url_safety(&loc) {
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
    fn parse_sitemap_index(&self, content: &str) -> Result<ParseResult> {
        let mut sitemaps = Vec::new();

        for sitemap_block in content.split("<sitemap>").skip(1) {
            if let Some(end) = sitemap_block.find("</sitemap>") {
                let block = &sitemap_block[..end];

                if let Some(loc) = extract_tag(block, "loc") {
                    if Url::parse(&loc).is_ok() {
                        // Validate URL safety (SSRF protection)
                        if let Err(e) = validate_url_safety(&loc) {
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
    fn parse_plain_text(&self, content: &str) -> Result<ParseResult> {
        let mut entries = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if (line.starts_with("http://") || line.starts_with("https://"))
                && Url::parse(line).is_ok()
            {
                // Validate URL safety (SSRF protection)
                if let Err(e) = validate_url_safety(line) {
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

    #[test]
    fn test_parse_urlset() {
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

        let result = parser.parse_urlset(xml).unwrap();
        if let ParseResult::UrlSet(entries) = result {
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].loc, "https://8.8.8.8/page1");
            assert_eq!(entries[0].priority, Some(0.8));
        } else {
            panic!("Expected UrlSet");
        }
    }

    #[test]
    fn test_is_ip_address_safe_accepts_public_ipv4() {
        use std::net::Ipv4Addr;

        // Public IPs should be accepted
        assert!(is_ip_address_safe(IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1))));
        assert!(is_ip_address_safe(IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8))));
        assert!(is_ip_address_safe(IpAddr::V4(Ipv4Addr::new(
            93, 184, 216, 34
        ))));
    }

    #[test]
    fn test_is_ip_address_safe_rejects_private_ipv4() {
        use std::net::Ipv4Addr;

        // Private IPs (RFC 1918) should be rejected
        assert!(!is_ip_address_safe(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1))));
        assert!(!is_ip_address_safe(IpAddr::V4(Ipv4Addr::new(
            172, 16, 0, 1
        ))));
        assert!(!is_ip_address_safe(IpAddr::V4(Ipv4Addr::new(
            192, 168, 1, 1
        ))));
    }

    #[test]
    fn test_is_ip_address_safe_rejects_loopback() {
        use std::net::Ipv4Addr;

        // Loopback should be rejected
        assert!(!is_ip_address_safe(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))));
        assert!(!is_ip_address_safe(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2))));
    }

    #[test]
    fn test_is_ip_address_safe_rejects_link_local() {
        use std::net::Ipv4Addr;

        // Link-local (cloud metadata endpoints) should be rejected
        assert!(!is_ip_address_safe(IpAddr::V4(Ipv4Addr::new(
            169, 254, 169, 254
        ))));
        assert!(!is_ip_address_safe(IpAddr::V4(Ipv4Addr::new(
            169, 254, 0, 1
        ))));
    }

    #[test]
    fn test_is_ip_address_safe_rejects_reserved() {
        use std::net::Ipv4Addr;

        // Reserved ranges should be rejected
        assert!(!is_ip_address_safe(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0))));
        assert!(!is_ip_address_safe(IpAddr::V4(Ipv4Addr::new(240, 0, 0, 1))));
        assert!(!is_ip_address_safe(IpAddr::V4(Ipv4Addr::new(
            255, 255, 255, 255
        ))));
    }

    #[test]
    fn test_is_ip_address_safe_rejects_ipv6_loopback() {
        use std::net::Ipv6Addr;

        assert!(!is_ip_address_safe(IpAddr::V6(Ipv6Addr::new(
            0, 0, 0, 0, 0, 0, 0, 1
        ))));
    }

    #[test]
    fn test_is_ip_address_safe_rejects_ipv6_link_local() {
        use std::net::Ipv6Addr;

        assert!(!is_ip_address_safe(IpAddr::V6(Ipv6Addr::new(
            0xfe80, 0, 0, 0, 0, 0, 0, 1
        ))));
    }

    #[test]
    fn test_validate_url_safety_accepts_public_urls() {
        // Test with well-known public IP addresses to avoid DNS resolution in tests
        // Google DNS 8.8.8.8 - using direct IP to ensure it's public
        let result = validate_url_safety("http://8.8.8.8/page");
        assert!(result.is_ok(), "Should accept public IP 8.8.8.8");

        // Cloudflare DNS 1.1.1.1
        let result = validate_url_safety("https://1.1.1.1/");
        assert!(result.is_ok(), "Should accept public IP 1.1.1.1");
    }

    #[test]
    fn test_validate_url_safety_rejects_private_ips_directly() {
        // Direct IP URLs should be rejected if they're private
        assert!(validate_url_safety("http://127.0.0.1/page").is_err());
        assert!(validate_url_safety("http://10.0.0.1/page").is_err());
        assert!(validate_url_safety("http://192.168.1.1/page").is_err());
        assert!(validate_url_safety("http://169.254.169.254/latest/meta-data").is_err());
    }

    #[test]
    fn test_validate_url_safety_rejects_unsupported_schemes() {
        assert!(validate_url_safety("ftp://example.com/file").is_err());
        assert!(validate_url_safety("file:///etc/passwd").is_err());
    }

    #[test]
    fn test_parse_plain_text_filters_unsafe_urls() {
        let parser = SitemapParser::new("test-agent").unwrap();
        let content = r#"
https://8.8.8.8/page1
http://127.0.0.1/admin
https://1.1.1.1/page2
http://10.0.0.1/internal
http://169.254.169.254/metadata
        "#;

        let result = parser.parse_plain_text(content).unwrap();
        if let ParseResult::UrlSet(entries) = result {
            // Should only include the safe public URLs
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].loc, "https://8.8.8.8/page1");
            assert_eq!(entries[1].loc, "https://1.1.1.1/page2");
        } else {
            panic!("Expected UrlSet");
        }
    }

    #[test]
    fn test_parse_urlset_filters_unsafe_urls() {
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

        let result = parser.parse_urlset(xml).unwrap();
        if let ParseResult::UrlSet(entries) = result {
            // Should only include the safe public URLs
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].loc, "https://8.8.8.8/page1");
            assert_eq!(entries[1].loc, "https://1.1.1.1/page2");
        } else {
            panic!("Expected UrlSet");
        }
    }

    #[test]
    fn test_parse_sitemap_index_filters_unsafe_urls() {
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

        let result = parser.parse_sitemap_index(xml).unwrap();
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
