//! Web crawling with robots.txt support and rate limiting
//!
//! This module provides:
//! - URL fetching with configurable timeouts
//! - robots.txt parsing and respect
//! - Per-host rate limiting
//! - Crawl depth and page limits
//! - Sitemap XML parsing
//! - SPA detection and JavaScript rendering

mod detection;
mod rate_limit;
mod renderer;
mod robots;
mod sitemap;

pub use detection::*;
pub use rate_limit::*;
pub use renderer::*;
pub use robots::*;
pub use sitemap::*;

use crate::config::CrawlConfig;
use crate::error::{Error, Result};
use crate::parse::{parse_html, ContentType, ExtractedLink};
use reqwest::Client;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
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
    renderer: Option<Arc<tokio::sync::Mutex<HeadlessRenderer>>>,
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

        // Initialize renderer if auto JS rendering is enabled and feature is available
        let renderer = if config.auto_js_rendering && is_js_rendering_available() {
            let renderer_config = RendererConfig {
                page_load_timeout_ms: config.js_page_load_timeout_ms,
                render_wait_ms: config.js_render_wait_ms,
                headless: true,
                wait_for_selector: None,
                sandbox: !config.js_no_sandbox,
            };
            Some(Arc::new(tokio::sync::Mutex::new(HeadlessRenderer::new(
                renderer_config,
            ))))
        } else {
            None
        };

        Ok(Self {
            client,
            config,
            robots_cache: Arc::new(RwLock::new(HashMap::new())),
            rate_limiters: Arc::new(RwLock::new(HashMap::new())),
            visited: Arc::new(RwLock::new(HashSet::new())),
            renderer,
        })
    }

    /// Fetch a single URL with automatic SPA detection and JS rendering fallback
    pub async fn fetch(&self, url: &str) -> Result<CrawledPage> {
        let parsed_url = Url::parse(url)?;
        let host = parsed_url
            .host_str()
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

        // Initial fetch with plain HTTP
        let response = self.client.get(url).send().await?;

        let status = response.status();
        if !status.is_success() {
            return Err(Error::Crawl(format!("HTTP {}: {}", status, url)));
        }

        let content_type_header = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        let content = response.text().await?;

        // Detect content type
        let ct = ContentType::detect(
            Some(std::path::Path::new(parsed_url.path())),
            content_type_header.as_deref(),
        );

        // For HTML pages, check if SPA rendering is needed
        if ct == ContentType::Html && self.config.auto_js_rendering {
            let analysis = analyze_page(&content, url);

            if analysis.needs_js_rendering {
                info!(
                    "SPA detected ({}, confidence: {:.0}%): {}",
                    match &analysis.technology {
                        PageTechnology::Spa(fw) => fw.to_string(),
                        _ => "Dynamic".to_string(),
                    },
                    analysis.confidence * 100.0,
                    url
                );

                for indicator in &analysis.indicators {
                    debug!("  - {}", indicator);
                }

                // Try JS rendering if available
                if let Some(renderer) = &self.renderer {
                    info!("Rendering with headless browser...");
                    let renderer_guard = renderer.lock().await;
                    match renderer_guard.render(url).await {
                        Ok(rendered) => {
                            info!(
                                "Rendered in {}ms: {} ({} bytes)",
                                rendered.render_time_ms,
                                url,
                                rendered.html.len()
                            );

                            // Parse the rendered HTML and extract hash routes
                            let parsed = parse_html(&rendered.html, Some(&rendered.url))?;

                            // Check for hash routes in rendered content
                            let hash_routes =
                                extract_hash_routes_from_rendered(&rendered.html, &rendered.url);
                            if !hash_routes.is_empty() {
                                info!(
                                    "Discovered {} hash routes in rendered page",
                                    hash_routes.len()
                                );
                                for route in &hash_routes {
                                    debug!("  Hash route: {}", route);
                                }
                            }

                            // Convert hash routes to full URLs as links
                            let mut links = parsed.links;
                            for route in hash_routes {
                                let hash_url = build_hash_route_url(&rendered.url, &route);
                                links.push(ExtractedLink {
                                    url: hash_url,
                                    text: None,
                                    is_internal: true,
                                });
                            }

                            return Ok(CrawledPage {
                                url: rendered.url,
                                content: rendered.html,
                                content_type: ContentType::Html,
                                title: rendered.title.or(parsed.title),
                                links,
                                depth: 0,
                            });
                        }
                        Err(e) => {
                            warn!("JS rendering failed, using static content: {}", e);
                            // Fall through to use static content
                        }
                    }
                } else {
                    warn!(
                        "SPA detected but JS rendering not available. \
                         Compile with --features js-rendering or disable auto_js_rendering. \
                         URL: {}",
                        url
                    );
                }
            }
        }

        // Parse HTML (either static content or non-SPA)
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

    /// Fetch a specific hash route (renders the page at that hash location)
    pub async fn fetch_hash_route(&self, base_url: &str, hash_route: &str) -> Result<CrawledPage> {
        let full_url = build_hash_route_url(base_url, hash_route);

        debug!("Fetching hash route: {}", full_url);

        if let Some(renderer) = &self.renderer {
            let renderer_guard = renderer.lock().await;
            match renderer_guard.render(&full_url).await {
                Ok(rendered) => {
                    let parsed = parse_html(&rendered.html, Some(&rendered.url))?;

                    // Extract more hash routes from this page
                    let hash_routes =
                        extract_hash_routes_from_rendered(&rendered.html, &rendered.url);
                    let mut links = parsed.links;
                    for route in hash_routes {
                        let hash_url = build_hash_route_url(&rendered.url, &route);
                        links.push(ExtractedLink {
                            url: hash_url,
                            text: None,
                            is_internal: true,
                        });
                    }

                    return Ok(CrawledPage {
                        url: full_url,
                        content: rendered.html,
                        content_type: ContentType::Html,
                        title: rendered.title.or(parsed.title),
                        links,
                        depth: 0,
                    });
                }
                Err(e) => {
                    return Err(Error::Crawl(format!(
                        "Failed to render hash route {}: {}",
                        full_url, e
                    )));
                }
            }
        }

        Err(Error::Crawl(
            "JS rendering not available for hash routes".to_string(),
        ))
    }

    /// Close the renderer (call this when done crawling)
    pub async fn close(&self) -> Result<()> {
        if let Some(renderer) = &self.renderer {
            let renderer_guard = renderer.lock().await;
            renderer_guard.close().await?;
        }
        Ok(())
    }

    /// Crawl from a seed URL
    pub async fn crawl(
        &self,
        seed_url: &str,
        callback: impl Fn(CrawledPage) -> bool + Send + Sync,
    ) -> Result<Vec<CrawledPage>> {
        let seed = Url::parse(seed_url)?;
        let seed_host = seed
            .host_str()
            .ok_or_else(|| Error::Crawl("Seed URL has no host".to_string()))?
            .to_string();

        // Determine allowed domains
        let mut allowed_hosts: HashSet<String> =
            self.config.allowed_domains.iter().cloned().collect();
        if allowed_hosts.is_empty() {
            allowed_hosts.insert(seed_host.clone());
        }

        // Determine path prefix restriction
        // If not explicitly set, use the seed URL's path
        let path_prefix = self.config.path_prefix.clone().unwrap_or_else(|| {
            let seed_path = seed.path();
            // Get the parent directory path from the seed URL
            // e.g., /docs/intro/getting-started -> /docs/intro/
            // e.g., /docs/ -> /docs/
            if seed_path.ends_with('/') {
                seed_path.to_string()
            } else {
                // Get directory part
                match seed_path.rfind('/') {
                    Some(idx) => seed_path[..=idx].to_string(),
                    None => "/".to_string(),
                }
            }
        });

        if path_prefix != "/" {
            info!("Restricting crawl to path prefix: {}", path_prefix);
        }

        let mut queue: VecDeque<(String, u32)> = VecDeque::new();
        queue.push_back((seed_url.to_string(), 0));

        let mut results = Vec::new();
        let mut pages_crawled = 0u32;
        let mut attempts = 0u32;
        let mut hash_routes_queued = 0u32;
        let mut warned_hash_route_cap = false;

        let per_page_budget_secs = self
            .config
            .timeout_secs
            .max(self.config.js_page_load_timeout_ms / 1000)
            .max(1);
        let max_crawl_seconds = per_page_budget_secs
            .saturating_mul(self.config.max_pages.max(1) as u64);
        let crawl_deadline = Instant::now() + Duration::from_secs(max_crawl_seconds);
        let max_attempts = self.config.max_pages.saturating_mul(5).max(1);
        let max_hash_routes = self.config.max_pages.max(1);

        // Track if we're crawling a hash-routed SPA
        let mut is_hash_routed_spa = false;

        while let Some((url, depth)) = queue.pop_front() {
            // Check limits
            if depth > self.config.max_depth {
                continue;
            }
            if pages_crawled >= self.config.max_pages {
                info!("Reached max pages limit ({})", self.config.max_pages);
                break;
            }
            if attempts >= max_attempts {
                warn!(
                    "Reached crawl attempt limit ({}); stopping to avoid stalling",
                    max_attempts
                );
                break;
            }
            if Instant::now() > crawl_deadline {
                warn!(
                    "Reached crawl time budget ({}s); stopping to avoid stalling",
                    max_crawl_seconds
                );
                break;
            }
            attempts += 1;

            // Normalize URL - use hash-aware normalization if we know this is a hash-routed SPA
            let normalized = if is_hash_routed_spa {
                normalize_url_with_hash(&url)
            } else {
                normalize_url(&url)
            };

            {
                let mut visited = self.visited.write().await;
                if visited.contains(&normalized) {
                    continue;
                }
                visited.insert(normalized.clone());
            }

            // Determine if this is a hash route URL
            let is_hash_route = url.contains("#/");

            // Fetch the page
            let fetch_result = if is_hash_route {
                // Extract base URL and hash route
                if let Some(hash_idx) = url.find('#') {
                    let base = &url[..hash_idx];
                    let route = &url[hash_idx + 1..];
                    self.fetch_hash_route(base, route).await
                } else {
                    self.fetch(&url).await
                }
            } else {
                self.fetch(&url).await
            };

            match fetch_result {
                Ok(mut page) => {
                    page.depth = depth;

                    // Check if we discovered hash routes - if so, switch to hash-aware mode
                    let has_hash_routes = page.links.iter().any(|l| l.url.contains("#/"));
                    if has_hash_routes && !is_hash_routed_spa {
                        info!("Detected hash-routed SPA, switching to hash-aware crawling");
                        is_hash_routed_spa = true;
                    }

                    // Queue internal links
                    for link in &page.links {
                        if !link.is_internal {
                            continue;
                        }

                        // Skip URLs that shouldn't be crawled
                        if !should_crawl_url(&link.url) {
                            continue;
                        }

                        if let Ok(link_url) = Url::parse(&link.url) {
                            if let Some(host) = link_url.host_str() {
                                if allowed_hosts.contains(host) {
                                    // Check path prefix restriction
                                    let link_path = link_url.path();
                                    if !link_path.starts_with(&path_prefix) {
                                        debug!(
                                            "Skipping {} - outside path prefix {}",
                                            link.url, path_prefix
                                        );
                                        continue;
                                    }

                                    // Use hash-aware normalization for SPAs
                                    let link_normalized = if is_hash_routed_spa {
                                        normalize_url_with_hash(&link.url)
                                    } else {
                                        normalize_url(&link.url)
                                    };
                                    let visited = self.visited.read().await;
                                    if !visited.contains(&link_normalized) {
                                        if is_hash_routed_spa && link.url.contains("#/") {
                                            if hash_routes_queued >= max_hash_routes {
                                                if !warned_hash_route_cap {
                                                    warn!(
                                                        "Reached hash-route cap ({}); skipping additional hash routes",
                                                        max_hash_routes
                                                    );
                                                    warned_hash_route_cap = true;
                                                }
                                                continue;
                                            }
                                            hash_routes_queued += 1;
                                        }
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
            limiters
                .entry(host.to_string())
                .or_insert_with(|| HostRateLimiter::new(self.config.rate_limit_per_host))
                .clone()
        };

        limiter.wait().await;
        Ok(())
    }
}

/// Normalize a URL for deduplication
pub fn normalize_url(url: &str) -> String {
    normalize_url_impl(url, false)
}

/// Normalize a URL for deduplication, optionally preserving hash routes
/// Hash routes are preserved when they look like SPA paths (e.g., #/api, #/docs)
pub fn normalize_url_with_hash(url: &str) -> String {
    normalize_url_impl(url, true)
}

fn normalize_url_impl(url: &str, preserve_hash_routes: bool) -> String {
    if let Ok(parsed) = Url::parse(url) {
        let mut normalized = parsed.clone();

        // Handle fragment
        if let Some(fragment) = parsed.fragment() {
            // Check if this looks like a hash route (starts with /)
            if preserve_hash_routes && fragment.starts_with('/') {
                // Keep the fragment - it's a hash route
            } else {
                // Remove regular fragments
                normalized.set_fragment(None);
            }
        }

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

fn build_hash_route_url(base_url: &str, hash_route: &str) -> String {
    let route = hash_route.trim_start_matches('#');
    if let Ok(mut base) = Url::parse(base_url) {
        base.set_fragment(Some(route));
        base.to_string()
    } else {
        format!("{}#{}", base_url.trim_end_matches('/'), route)
    }
}

/// Check if a URL should be crawled based on patterns
pub fn should_crawl_url(url: &str) -> bool {
    let lower = url.to_lowercase();

    // Skip common non-document URLs
    let skip_patterns = [
        "/login",
        "/logout",
        "/signin",
        "/signout",
        "/register",
        "/admin",
        "/wp-admin",
        "/api/",
        "/cgi-bin/",
        ".xml",
        ".json",
        ".rss",
        ".atom",
        "javascript:",
        "mailto:",
        "tel:",
        "?page=",
        "?sort=",
        "?filter=",
    ];

    for pattern in skip_patterns {
        if lower.contains(pattern) {
            return false;
        }
    }

    // Skip plain anchor fragments (but NOT hash routes like #/path)
    if let Some(hash_idx) = lower.find('#') {
        let after_hash = &lower[hash_idx + 1..];
        // Allow hash routes that look like paths (start with /)
        // Skip plain anchors like #section or #top
        if !after_hash.is_empty() && !after_hash.starts_with('/') {
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
    use wiremock::{Mock, MockServer, ResponseTemplate};
    use wiremock::matchers::{method, path, path_regex};

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
    fn test_normalize_url_with_hash_routes() {
        // Hash routes should be preserved when using hash-aware normalization
        assert_eq!(
            normalize_url_with_hash("https://example.com/#/api"),
            "https://example.com/#/api"
        );
        assert_eq!(
            normalize_url_with_hash("https://example.com/#/service/Lightbulb"),
            "https://example.com/#/service/Lightbulb"
        );
        // Regular fragments should still be stripped
        assert_eq!(
            normalize_url_with_hash("https://example.com/page#section"),
            "https://example.com/page"
        );
        // Empty fragments should be stripped
        assert_eq!(
            normalize_url_with_hash("https://example.com/page#"),
            "https://example.com/page"
        );
    }

    #[test]
    fn test_should_crawl_url() {
        assert!(should_crawl_url("https://example.com/docs/intro"));
        assert!(!should_crawl_url("https://example.com/login"));
        assert!(!should_crawl_url("https://example.com/api/users"));
        assert!(!should_crawl_url("javascript:void(0)"));
        // Hash routes should be crawled
        assert!(should_crawl_url("https://example.com/#/api"));
        assert!(should_crawl_url("https://example.com/#/service/Lightbulb"));
        // Plain anchors should not
        assert!(!should_crawl_url("https://example.com/page#section"));
    }

    #[test]
    fn test_build_hash_route_url_preserves_path() {
        let base = "https://example.com/docs/";
        let route = "/api";
        assert_eq!(
            build_hash_route_url(base, route),
            "https://example.com/docs/#/api"
        );
    }

    #[tokio::test]
    async fn test_crawl_attempt_cap_limits_failed_fetches() {
        let mock_server = MockServer::start().await;

        let mut links = String::new();
        for i in 0..20 {
            links.push_str(&format!("<a href=\"/missing/{}\">missing</a>", i));
        }
        let html = format!("<html><body>{}</body></html>", links);

        Mock::given(method("GET"))
            .and(path("/index.html"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                html.clone().into_bytes(),
                "text/html",
            ))
            .mount(&mock_server)
            .await;

        let missing_guard = Mock::given(method("GET"))
            .and(path_regex(r"^/missing/\d+$"))
            .respond_with(ResponseTemplate::new(404))
            .mount_as_scoped(&mock_server)
            .await;

        let max_pages = 2;
        let mut crawl_config = CrawlConfig::default();
        crawl_config.max_pages = max_pages;
        crawl_config.max_depth = 2;
        crawl_config.auto_js_rendering = false;
        crawl_config.rate_limit_per_host = 1000.0;
        crawl_config.timeout_secs = 5;

        let max_attempts = max_pages.saturating_mul(5).max(1);
        let crawler = Crawler::new(crawl_config).expect("crawler should build");
        let seed = format!("{}/index.html", mock_server.uri());
        let results = crawler
            .crawl(&seed, |_page| true)
            .await
            .expect("crawl should complete");
        let expected_missing = max_attempts.saturating_sub(1) as usize;
        let missing_requests = missing_guard.received_requests().await;

        assert_eq!(results[0].content_type, ContentType::Html);
        assert_eq!(
            results[0].links.iter().filter(|link| link.is_internal).count(),
            20
        );

        assert_eq!(results.len(), 1);
        assert_eq!(missing_requests.len(), expected_missing);
    }
}
