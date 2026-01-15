//! Web crawling with robots.txt support and rate limiting
//!
//! This module provides:
//! - URL fetching with configurable timeouts
//! - robots.txt parsing and respect
//! - Per-host rate limiting
//! - Crawl depth and page limits
//! - Sitemap XML parsing

mod robots;
mod rate_limit;
mod sitemap;

pub use robots::*;
pub use rate_limit::*;
pub use sitemap::*;

use crate::config::CrawlConfig;
use crate::error::{Error, Result};
use crate::parse::{parse_html, ContentType, ExtractedLink};
use reqwest::Client;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use url::Url;

/// A crawled page
#[derive(Debug, Clone)]
pub struct CrawledPage {
    pub url: String,
    pub content: String,
    pub content_type: ContentType,
    pub title: Option<String>,
    pub links: Vec<ExtractedLink>,
    pub depth: u32,
}

/// Web crawler state
pub struct Crawler {
    client: Client,
    config: CrawlConfig,
    robots_cache: Arc<RwLock<HashMap<String, RobotsRules>>>,
    rate_limiters: Arc<RwLock<HashMap<String, HostRateLimiter>>>,
    visited: Arc<RwLock<HashSet<String>>>,
}

impl Crawler {
    /// Create a new crawler
    pub fn new(config: CrawlConfig) -> Result<Self> {
        let client = Client::builder()
            .user_agent(&config.user_agent)
            .timeout(Duration::from_secs(config.timeout_secs))
            .gzip(true)
            .brotli(true)
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .map_err(|e| Error::Crawl(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            config,
            robots_cache: Arc::new(RwLock::new(HashMap::new())),
            rate_limiters: Arc::new(RwLock::new(HashMap::new())),
            visited: Arc::new(RwLock::new(HashSet::new())),
        })
    }

    /// Fetch a single URL
    pub async fn fetch(&self, url: &str) -> Result<CrawledPage> {
        let parsed_url = Url::parse(url)?;
        let host = parsed_url.host_str()
            .ok_or_else(|| Error::Crawl("URL has no host".to_string()))?
            .to_string();

        // Check robots.txt
        if self.config.respect_robots_txt {
            self.ensure_robots_loaded(&host, &parsed_url).await?;
            let rules = self.robots_cache.read().await;
            if let Some(r) = rules.get(&host) {
                if !r.is_allowed(parsed_url.path(), &self.config.user_agent) {
                    return Err(Error::RobotsDisallowed(url.to_string()));
                }
            }
        }

        // Rate limiting
        self.rate_limit(&host).await?;

        debug!("Fetching: {}", url);

        let response = self.client.get(url).send().await?;
        
        let status = response.status();
        if !status.is_success() {
            return Err(Error::Crawl(format!("HTTP {}: {}", status, url)));
        }

        let content_type_header = response.headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        let content = response.text().await?;

        // Detect content type
        let ct = ContentType::detect(
            Some(std::path::Path::new(parsed_url.path())),
            content_type_header.as_deref(),
        );

        // Parse if HTML
        let (title, links) = if ct == ContentType::Html {
            let parsed = parse_html(&content, Some(url))?;
            (parsed.title, parsed.links)
        } else {
            (None, Vec::new())
        };

        Ok(CrawledPage {
            url: url.to_string(),
            content,
            content_type: ct,
            title,
            links,
            depth: 0,
        })
    }

    /// Crawl from a seed URL
    pub async fn crawl(
        &self,
        seed_url: &str,
        callback: impl Fn(CrawledPage) -> bool + Send + Sync,
    ) -> Result<Vec<CrawledPage>> {
        let seed = Url::parse(seed_url)?;
        let seed_host = seed.host_str()
            .ok_or_else(|| Error::Crawl("Seed URL has no host".to_string()))?
            .to_string();

        // Determine allowed domains
        let mut allowed_hosts: HashSet<String> = self.config.allowed_domains
            .iter()
            .cloned()
            .collect();
        if allowed_hosts.is_empty() {
            allowed_hosts.insert(seed_host.clone());
        }

        let mut queue: VecDeque<(String, u32)> = VecDeque::new();
        queue.push_back((seed_url.to_string(), 0));

        let mut results = Vec::new();
        let mut pages_crawled = 0u32;

        while let Some((url, depth)) = queue.pop_front() {
            // Check limits
            if depth > self.config.max_depth {
                continue;
            }
            if pages_crawled >= self.config.max_pages {
                info!("Reached max pages limit ({})", self.config.max_pages);
                break;
            }

            // Normalize and check if visited
            let normalized = normalize_url(&url);
            {
                let mut visited = self.visited.write().await;
                if visited.contains(&normalized) {
                    continue;
                }
                visited.insert(normalized.clone());
            }

            // Fetch the page
            match self.fetch(&url).await {
                Ok(mut page) => {
                    page.depth = depth;
                    
                    // Queue internal links
                    for link in &page.links {
                        if !link.is_internal {
                            continue;
                        }
                        if let Ok(link_url) = Url::parse(&link.url) {
                            if let Some(host) = link_url.host_str() {
                                if allowed_hosts.contains(host) {
                                    let link_normalized = normalize_url(&link.url);
                                    let visited = self.visited.read().await;
                                    if !visited.contains(&link_normalized) {
                                        queue.push_back((link.url.clone(), depth + 1));
                                    }
                                }
                            }
                        }
                    }

                    // Call callback
                    let should_continue = callback(page.clone());
                    results.push(page);
                    pages_crawled += 1;

                    if !should_continue {
                        break;
                    }
                }
                Err(e) => {
                    warn!("Failed to fetch {}: {}", url, e);
                }
            }
        }

        info!("Crawled {} pages from {}", results.len(), seed_url);
        Ok(results)
    }

    async fn ensure_robots_loaded(&self, host: &str, url: &Url) -> Result<()> {
        {
            let cache = self.robots_cache.read().await;
            if cache.contains_key(host) {
                return Ok(());
            }
        }

        let robots_url = format!("{}://{}/robots.txt", url.scheme(), host);
        debug!("Fetching robots.txt from {}", robots_url);

        let rules = match self.client.get(&robots_url).send().await {
            Ok(response) if response.status().is_success() => {
                let text = response.text().await.unwrap_or_default();
                RobotsRules::parse(&text)
            }
            _ => {
                // No robots.txt or error - allow all
                RobotsRules::allow_all()
            }
        };

        let mut cache = self.robots_cache.write().await;
        cache.insert(host.to_string(), rules);
        Ok(())
    }

    async fn rate_limit(&self, host: &str) -> Result<()> {
        let limiter = {
            let mut limiters = self.rate_limiters.write().await;
            limiters.entry(host.to_string())
                .or_insert_with(|| HostRateLimiter::new(self.config.rate_limit_per_host))
                .clone()
        };

        limiter.wait().await;
        Ok(())
    }
}

/// Normalize a URL for deduplication
pub fn normalize_url(url: &str) -> String {
    if let Ok(parsed) = Url::parse(url) {
        let mut normalized = parsed.clone();
        
        // Remove fragment
        normalized.set_fragment(None);
        
        // Remove trailing slash from path
        let path = parsed.path().trim_end_matches('/');
        if path.is_empty() {
            normalized.set_path("/");
        } else {
            normalized.set_path(path);
        }

        // Sort query parameters for consistency
        // (simplified - full implementation would parse and sort)
        
        normalized.to_string()
    } else {
        url.to_string()
    }
}

/// Check if a URL should be crawled based on patterns
pub fn should_crawl_url(url: &str) -> bool {
    let lower = url.to_lowercase();
    
    // Skip common non-document URLs
    let skip_patterns = [
        "/login", "/logout", "/signin", "/signout", "/register",
        "/admin", "/wp-admin", "/api/", "/cgi-bin/",
        ".xml", ".json", ".rss", ".atom",
        "javascript:", "mailto:", "tel:",
        "#", "?page=", "?sort=", "?filter=",
    ];

    for pattern in skip_patterns {
        if lower.contains(pattern) {
            return false;
        }
    }

    // Skip calendar-like URLs with dates
    let date_pattern = regex::Regex::new(r"/\d{4}/\d{2}/\d{2}/").ok();
    if let Some(re) = date_pattern {
        if re.is_match(&lower) {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_url() {
        assert_eq!(
            normalize_url("https://example.com/path/"),
            "https://example.com/path"
        );
        assert_eq!(
            normalize_url("https://example.com/path#fragment"),
            "https://example.com/path"
        );
        assert_eq!(
            normalize_url("https://example.com/"),
            "https://example.com/"
        );
    }

    #[test]
    fn test_should_crawl_url() {
        assert!(should_crawl_url("https://example.com/docs/intro"));
        assert!(!should_crawl_url("https://example.com/login"));
        assert!(!should_crawl_url("https://example.com/api/users"));
        assert!(!should_crawl_url("javascript:void(0)"));
    }
}
