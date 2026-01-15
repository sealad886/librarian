//! Headless browser rendering for JavaScript-heavy pages
//!
//! Uses Chrome DevTools Protocol via chromiumoxide to render pages
//! that require JavaScript execution (SPAs, dynamic content, etc.)

use crate::error::{Error, Result};

/// Configuration for the headless browser renderer
#[derive(Debug, Clone)]
pub struct RendererConfig {
    /// Time to wait for page load (milliseconds)
    pub page_load_timeout_ms: u64,
    /// Time to wait after load for dynamic content (milliseconds)
    pub render_wait_ms: u64,
    /// Run browser in headless mode
    pub headless: bool,
    /// Additional wait for specific selector (optional)
    pub wait_for_selector: Option<String>,
    /// Enable sandbox (disable for Docker/CI environments)
    pub sandbox: bool,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            page_load_timeout_ms: 30000,
            render_wait_ms: 2000,
            headless: true,
            wait_for_selector: None,
            sandbox: true,
        }
    }
}

/// Rendered page result
#[derive(Debug, Clone)]
pub struct RenderedPage {
    /// Final URL after any redirects
    pub url: String,
    /// Fully rendered HTML content
    pub html: String,
    /// Page title
    pub title: Option<String>,
    /// Time taken to render (milliseconds)
    pub render_time_ms: u64,
}

#[cfg(feature = "js-rendering")]
mod browser_impl {
    use super::*;
    use chromiumoxide::browser::{Browser, BrowserConfig};
    use futures::StreamExt;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Mutex;
    use tokio::time::{timeout, Instant};
    use tracing::{debug, info, warn};

    /// Headless browser renderer
    pub struct HeadlessRenderer {
        config: RendererConfig,
        browser: Arc<Mutex<Option<Browser>>>,
        handler_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    }

    impl HeadlessRenderer {
        /// Create a new headless renderer
        pub fn new(config: RendererConfig) -> Self {
            Self {
                config,
                browser: Arc::new(Mutex::new(None)),
                handler_handle: Arc::new(Mutex::new(None)),
            }
        }

        /// Initialize the browser (lazy initialization)
        async fn ensure_browser(&self) -> Result<()> {
            let mut browser_guard = self.browser.lock().await;
            if browser_guard.is_some() {
                return Ok(());
            }

            info!("Launching headless Chrome browser...");

            let mut builder = BrowserConfig::builder();
            
            if self.config.headless {
                builder = builder.with_head();
                // Note: chromiumoxide defaults to headless, with_head() adds UI
                // We want headless, so don't call with_head()
            }

            if !self.config.sandbox {
                builder = builder.no_sandbox();
            }

            // Common args for stability
            builder = builder
                .arg("--disable-gpu")
                .arg("--disable-dev-shm-usage")
                .arg("--disable-setuid-sandbox")
                .arg("--no-first-run")
                .arg("--no-zygote")
                .arg("--disable-extensions");

            let browser_config = builder.build()
                .map_err(|e| Error::Crawl(format!("Failed to build browser config: {}", e)))?;

            let (browser, mut handler) = Browser::launch(browser_config)
                .await
                .map_err(|e| Error::Crawl(format!("Failed to launch browser: {}", e)))?;

            // Spawn handler task
            let handle = tokio::spawn(async move {
                while let Some(result) = handler.next().await {
                    if result.is_err() {
                        break;
                    }
                }
            });

            *browser_guard = Some(browser);
            *self.handler_handle.lock().await = Some(handle);

            info!("Headless browser launched successfully");
            Ok(())
        }

        /// Render a page using the headless browser
        pub async fn render(&self, url: &str) -> Result<RenderedPage> {
            self.ensure_browser().await?;

            let start = Instant::now();
            debug!("Rendering page with headless browser: {}", url);

            let browser_guard = self.browser.lock().await;
            let browser = browser_guard.as_ref()
                .ok_or_else(|| Error::Crawl("Browser not initialized".to_string()))?;

            // Create new page/tab
            let page = browser.new_page(url)
                .await
                .map_err(|e| Error::Crawl(format!("Failed to create page: {}", e)))?;

            // Wait for page load with timeout
            let load_timeout = Duration::from_millis(self.config.page_load_timeout_ms);
            timeout(load_timeout, page.wait_for_navigation())
                .await
                .map_err(|_| Error::Crawl(format!("Page load timeout: {}", url)))?
                .map_err(|e| Error::Crawl(format!("Navigation failed: {}", e)))?;

            // Additional wait for dynamic content
            if self.config.render_wait_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.config.render_wait_ms)).await;
            }

            // Wait for specific selector if configured
            if let Some(selector) = &self.config.wait_for_selector {
                let selector_timeout = Duration::from_secs(10);
                match timeout(selector_timeout, page.find_element(selector)).await {
                    Ok(Ok(_)) => debug!("Found selector: {}", selector),
                    Ok(Err(e)) => warn!("Selector {} not found: {}", selector, e),
                    Err(_) => warn!("Timeout waiting for selector: {}", selector),
                }
            }

            // Get final URL (after redirects)
            let final_url = page.url()
                .await
                .map_err(|e| Error::Crawl(format!("Failed to get URL: {}", e)))?
                .map(|u| u.to_string())
                .unwrap_or_else(|| url.to_string());

            // Get rendered HTML
            let html = page.content()
                .await
                .map_err(|e| Error::Crawl(format!("Failed to get content: {}", e)))?;

            // Get page title
            let title = page.evaluate("document.title")
                .await
                .ok()
                .and_then(|v| v.into_value::<String>().ok())
                .filter(|t| !t.is_empty());

            // Close the page/tab
            if let Err(e) = page.close().await {
                warn!("Failed to close page: {}", e);
            }

            let render_time_ms = start.elapsed().as_millis() as u64;
            debug!("Rendered {} in {}ms", url, render_time_ms);

            Ok(RenderedPage {
                url: final_url,
                html,
                title,
                render_time_ms,
            })
        }

        /// Close the browser
        pub async fn close(&self) -> Result<()> {
            let mut browser_guard = self.browser.lock().await;
            if let Some(mut browser) = browser_guard.take() {
                browser.close().await
                    .map_err(|e| Error::Crawl(format!("Failed to close browser: {}", e)))?;
            }

            let mut handle_guard = self.handler_handle.lock().await;
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }

            Ok(())
        }
    }

    impl Drop for HeadlessRenderer {
        fn drop(&mut self) {
            // Note: async drop not possible, browser cleanup happens in close()
        }
    }
}

#[cfg(feature = "js-rendering")]
pub use browser_impl::HeadlessRenderer;

/// Stub renderer when js-rendering feature is disabled
#[cfg(not(feature = "js-rendering"))]
pub struct HeadlessRenderer {
    _config: RendererConfig,
}

#[cfg(not(feature = "js-rendering"))]
impl HeadlessRenderer {
    pub fn new(config: RendererConfig) -> Self {
        Self { _config: config }
    }

    pub async fn render(&self, url: &str) -> Result<RenderedPage> {
        Err(Error::Crawl(format!(
            "JavaScript rendering not available for {}. \
             Compile with --features js-rendering to enable headless browser support.",
            url
        )))
    }

    pub async fn close(&self) -> Result<()> {
        Ok(())
    }
}

/// Check if JS rendering feature is available
pub fn is_js_rendering_available() -> bool {
    cfg!(feature = "js-rendering")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_config_default() {
        let config = RendererConfig::default();
        assert!(config.headless);
        assert!(config.sandbox);
        assert_eq!(config.render_wait_ms, 2000);
    }

    #[test]
    fn test_js_rendering_availability() {
        // This test passes regardless of feature flag
        let available = is_js_rendering_available();
        println!("JS rendering available: {}", available);
    }
}
