//! Page technology detection for intelligent crawling
//!
//! Detects various page rendering technologies that require special handling:
//! - Single Page Applications (React, Angular, Vue, Svelte, etc.)
//! - Dynamic content loading (infinite scroll, lazy load)
//! - Client-side rendering indicators
//! - Framework-specific markers

use regex::Regex;
use tracing::debug;

/// Detected page technology/rendering method
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageTechnology {
    /// Standard server-rendered HTML
    Static,
    /// Single Page Application requiring JS rendering
    Spa(SpaFramework),
    /// Page uses significant dynamic content loading
    DynamicContent,
    /// Content behind authentication wall
    AuthWall,
    /// Anti-bot protection detected
    BotProtection,
}

/// Known SPA frameworks
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpaFramework {
    React,
    Angular,
    Vue,
    Svelte,
    NextJs,
    Nuxt,
    Gatsby,
    Ember,
    Unknown,
}

impl std::fmt::Display for SpaFramework {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpaFramework::React => write!(f, "React"),
            SpaFramework::Angular => write!(f, "Angular"),
            SpaFramework::Vue => write!(f, "Vue"),
            SpaFramework::Svelte => write!(f, "Svelte"),
            SpaFramework::NextJs => write!(f, "Next.js"),
            SpaFramework::Nuxt => write!(f, "Nuxt"),
            SpaFramework::Gatsby => write!(f, "Gatsby"),
            SpaFramework::Ember => write!(f, "Ember"),
            SpaFramework::Unknown => write!(f, "Unknown SPA"),
        }
    }
}

/// Result of page analysis
#[derive(Debug, Clone)]
pub struct PageAnalysis {
    /// Primary detected technology
    pub technology: PageTechnology,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Whether JS rendering is recommended
    pub needs_js_rendering: bool,
    /// Detected indicators that led to this conclusion
    pub indicators: Vec<String>,
    /// Estimated content ratio (actual text / total HTML size)
    pub content_ratio: f32,
    /// Whether the site uses hash-based routing (e.g., #/path)
    pub uses_hash_routing: bool,
    /// Discovered hash routes if hash routing is detected
    pub hash_routes: Vec<String>,
}

impl PageAnalysis {
    /// Create analysis for static page
    pub fn static_page(content_ratio: f32) -> Self {
        Self {
            technology: PageTechnology::Static,
            confidence: 1.0,
            needs_js_rendering: false,
            indicators: vec!["Normal content ratio".to_string()],
            content_ratio,
            uses_hash_routing: false,
            hash_routes: Vec::new(),
        }
    }

    /// Create analysis for SPA
    pub fn spa(
        framework: SpaFramework,
        confidence: f32,
        indicators: Vec<String>,
        content_ratio: f32,
    ) -> Self {
        Self {
            technology: PageTechnology::Spa(framework),
            confidence,
            needs_js_rendering: true,
            indicators,
            content_ratio,
            uses_hash_routing: false,
            hash_routes: Vec::new(),
        }
    }

    /// Create analysis for SPA with hash routing
    pub fn spa_with_hash_routes(
        framework: SpaFramework,
        confidence: f32,
        indicators: Vec<String>,
        content_ratio: f32,
        hash_routes: Vec<String>,
    ) -> Self {
        Self {
            technology: PageTechnology::Spa(framework),
            confidence,
            needs_js_rendering: true,
            indicators,
            content_ratio,
            uses_hash_routing: true,
            hash_routes,
        }
    }
}

/// Analyze HTML to detect page technology
pub fn analyze_page(html: &str, url: &str) -> PageAnalysis {
    let html_lower = html.to_lowercase();
    let mut indicators = Vec::new();
    let mut spa_score: f32 = 0.0;
    let mut detected_framework = SpaFramework::Unknown;

    // Calculate content ratio (text content vs HTML size)
    let content_ratio = calculate_content_ratio(html);
    debug!("Content ratio for {}: {:.2}", url, content_ratio);

    // Very low content ratio is a strong SPA indicator
    if content_ratio < 0.05 {
        spa_score += 0.4;
        indicators.push(format!(
            "Very low content ratio: {:.1}%",
            content_ratio * 100.0
        ));
    } else if content_ratio < 0.15 {
        spa_score += 0.2;
        indicators.push(format!("Low content ratio: {:.1}%", content_ratio * 100.0));
    }

    // Check for SPA root elements (empty shells)
    let spa_roots = check_spa_root_elements(html);
    if !spa_roots.is_empty() {
        spa_score += 0.3;
        indicators.extend(spa_roots);
    }

    // Detect specific frameworks
    if let Some((framework, framework_indicators)) = detect_framework(html, &html_lower) {
        detected_framework = framework;
        spa_score += 0.2;
        indicators.extend(framework_indicators);
    }

    // Check for heavy script loading patterns
    let script_analysis = analyze_scripts(html, &html_lower);
    if script_analysis.heavy_js {
        spa_score += 0.15;
        indicators.push(format!(
            "Heavy JS: {} scripts, {:.0}KB estimated",
            script_analysis.script_count, script_analysis.estimated_size_kb
        ));
    }

    // Check for hydration/client-side rendering markers
    if check_hydration_markers(&html_lower) {
        spa_score += 0.1;
        indicators.push("Client-side hydration markers detected".to_string());
    }

    // Check for auth walls
    if check_auth_wall(&html_lower) {
        return PageAnalysis {
            technology: PageTechnology::AuthWall,
            confidence: 0.8,
            needs_js_rendering: false,
            indicators: vec!["Login/authentication form detected".to_string()],
            content_ratio,
            uses_hash_routing: false,
            hash_routes: Vec::new(),
        };
    }

    // Check for bot protection
    if check_bot_protection(&html_lower) {
        return PageAnalysis {
            technology: PageTechnology::BotProtection,
            confidence: 0.9,
            needs_js_rendering: true,
            indicators: vec!["Bot protection/CAPTCHA detected".to_string()],
            content_ratio,
            uses_hash_routing: false,
            hash_routes: Vec::new(),
        };
    }

    // Check for hash-based routing (common in Angular, older React apps)
    let hash_routes = extract_hash_routes(html);
    let uses_hash_routing = !hash_routes.is_empty();
    if uses_hash_routing {
        spa_score += 0.3;
        indicators.push(format!(
            "Hash-based routing detected: {} routes",
            hash_routes.len()
        ));
    }

    // Decision threshold
    if spa_score >= 0.5 {
        if uses_hash_routing {
            PageAnalysis::spa_with_hash_routes(
                detected_framework,
                spa_score.min(1.0),
                indicators,
                content_ratio,
                hash_routes,
            )
        } else {
            PageAnalysis::spa(
                detected_framework,
                spa_score.min(1.0),
                indicators,
                content_ratio,
            )
        }
    } else {
        PageAnalysis::static_page(content_ratio)
    }
}

/// Calculate ratio of visible text content to total HTML size
fn calculate_content_ratio(html: &str) -> f32 {
    let total_size = html.len() as f32;
    if total_size == 0.0 {
        return 0.0;
    }

    // First, strip scripts and styles (must do this BEFORE stripping tags)
    let script_re = Regex::new(r"(?is)<script[^>]*>.*?</script>").unwrap();
    let style_re = Regex::new(r"(?is)<style[^>]*>.*?</style>").unwrap();
    let cleaned = script_re.replace_all(html, "");
    let cleaned = style_re.replace_all(&cleaned, "");

    // Now strip remaining tags
    let tag_re = Regex::new(r"<[^>]+>").unwrap();
    let text_only = tag_re.replace_all(&cleaned, " ");

    // Normalize whitespace and count
    let text_content: String = text_only.split_whitespace().collect::<Vec<_>>().join(" ");

    let text_size = text_content.len() as f32;
    text_size / total_size
}

/// Check for common SPA root elements that indicate client-side rendering
fn check_spa_root_elements(html: &str) -> Vec<String> {
    let mut indicators = Vec::new();

    let spa_patterns = [
        // Angular
        (r"<app-root[^>]*>\s*</app-root>", "Angular <app-root> shell"),
        (
            r"<app-root[^>]*>Loading",
            "Angular <app-root> with loading state",
        ),
        // React
        (
            r#"<div\s+id\s*=\s*["']root["'][^>]*>\s*</div>"#,
            "React #root shell",
        ),
        (
            r#"<div\s+id\s*=\s*["']app["'][^>]*>\s*</div>"#,
            "React/Vue #app shell",
        ),
        (
            r#"<div\s+id\s*=\s*["']__next["'][^>]*>\s*</div>"#,
            "Next.js #__next shell",
        ),
        // Vue
        (
            r#"<div\s+id\s*=\s*["']__nuxt["'][^>]*>"#,
            "Nuxt #__nuxt shell",
        ),
        // Svelte
        (
            r#"<div\s+id\s*=\s*["']svelte["'][^>]*>\s*</div>"#,
            "Svelte #svelte shell",
        ),
        // Generic
        (
            r#"<div\s+id\s*=\s*["']main-app["'][^>]*>\s*</div>"#,
            "SPA #main-app shell",
        ),
    ];

    for (pattern, description) in spa_patterns {
        if let Ok(re) = Regex::new(&format!("(?i){}", pattern)) {
            if re.is_match(html) {
                indicators.push(description.to_string());
            }
        }
    }

    indicators
}

/// Detect specific SPA framework
fn detect_framework(html: &str, html_lower: &str) -> Option<(SpaFramework, Vec<String>)> {
    let mut indicators = Vec::new();

    // Angular
    if html_lower.contains("ng-version") || html.contains("_ngcontent") || html.contains("_nghost")
    {
        indicators.push("Angular markers (ng-version, _ngcontent)".to_string());
        return Some((SpaFramework::Angular, indicators));
    }

    // Next.js
    if html_lower.contains("__next") || html_lower.contains("_next/static") {
        indicators.push("Next.js markers (__next, _next/static)".to_string());
        return Some((SpaFramework::NextJs, indicators));
    }

    // Nuxt
    if html_lower.contains("__nuxt") || html_lower.contains("/_nuxt/") {
        indicators.push("Nuxt markers (__nuxt, /_nuxt/)".to_string());
        return Some((SpaFramework::Nuxt, indicators));
    }

    // Gatsby
    if html_lower.contains("___gatsby") || html_lower.contains("/page-data/") {
        indicators.push("Gatsby markers (___gatsby, /page-data/)".to_string());
        return Some((SpaFramework::Gatsby, indicators));
    }

    // React (generic, check after specific React frameworks)
    if html_lower.contains("data-reactroot") || html_lower.contains("data-reactid") {
        indicators.push("React markers (data-reactroot)".to_string());
        return Some((SpaFramework::React, indicators));
    }

    // Vue
    if html_lower.contains("data-v-") || html_lower.contains("v-cloak") {
        indicators.push("Vue markers (data-v-, v-cloak)".to_string());
        return Some((SpaFramework::Vue, indicators));
    }

    // Svelte
    if html_lower.contains("svelte-") || html.contains("__svelte") {
        indicators.push("Svelte markers (svelte-)".to_string());
        return Some((SpaFramework::Svelte, indicators));
    }

    // Ember
    if html_lower.contains("ember-view") || html_lower.contains("data-ember") {
        indicators.push("Ember markers (ember-view)".to_string());
        return Some((SpaFramework::Ember, indicators));
    }

    None
}

/// Analysis of script tags
struct ScriptAnalysis {
    script_count: usize,
    heavy_js: bool,
    estimated_size_kb: f32,
}

/// Analyze script loading patterns
fn analyze_scripts(html: &str, _html_lower: &str) -> ScriptAnalysis {
    let script_re = Regex::new(r#"<script[^>]*(?:src\s*=\s*["'][^"']+["'])?[^>]*>"#).unwrap();
    let scripts: Vec<_> = script_re.find_iter(html).collect();
    let script_count = scripts.len();

    // Estimate total script size from inline scripts
    let inline_re = Regex::new(r"(?is)<script[^>]*>(.+?)</script>").unwrap();
    let inline_size: usize = inline_re
        .captures_iter(html)
        .map(|c| c.get(1).map_or(0, |m| m.as_str().len()))
        .sum();

    let estimated_size_kb = inline_size as f32 / 1024.0;

    // Heavy JS: many scripts or large inline scripts
    let heavy_js = script_count > 5 || estimated_size_kb > 50.0;

    ScriptAnalysis {
        script_count,
        heavy_js,
        estimated_size_kb,
    }
}

/// Check for client-side hydration markers
fn check_hydration_markers(html_lower: &str) -> bool {
    let markers = [
        "data-server-rendered",
        "data-hydrate",
        "__preload_data__",
        "__ssg__",
        "__ssr__",
        "window.__initial_state__",
        "window.__state__",
        "window.__data__",
    ];

    markers.iter().any(|m| html_lower.contains(m))
}

/// Check for authentication walls
fn check_auth_wall(html_lower: &str) -> bool {
    // Look for login forms with minimal other content
    let auth_indicators = [
        r#"<form[^>]*action[^>]*login"#,
        r#"<input[^>]*type\s*=\s*["']password["']"#,
        "sign in to continue",
        "log in to continue",
        "please log in",
        "authentication required",
    ];

    let auth_count = auth_indicators
        .iter()
        .filter(|p| {
            if p.contains('<') {
                Regex::new(&format!("(?i){}", p))
                    .map(|re| re.is_match(html_lower))
                    .unwrap_or(false)
            } else {
                html_lower.contains(*p)
            }
        })
        .count();

    // Need multiple auth indicators to avoid false positives
    auth_count >= 2
}

/// Check for bot protection (Cloudflare, reCAPTCHA, etc.)
fn check_bot_protection(html_lower: &str) -> bool {
    let bot_indicators = [
        "captcha",
        "recaptcha",
        "hcaptcha",
        "cf-browser-verification",
        "cloudflare",
        "ddos-guard",
        "challenge-platform",
        "please wait while we verify",
        "checking your browser",
        "just a moment",
        "enable javascript and cookies",
    ];

    let protection_count = bot_indicators
        .iter()
        .filter(|p| html_lower.contains(*p))
        .count();

    // Bot protection usually has multiple markers
    protection_count >= 2
}

/// Extract hash-based routes from HTML (e.g., href="#/api", href="#/service/Lightbulb")
pub fn extract_hash_routes(html: &str) -> Vec<String> {
    let mut routes = std::collections::HashSet::new();

    // Match href="#/..." patterns (hash-based routing)
    // This captures Angular, Vue hash mode, and other hash-routed SPAs
    let hash_route_re = Regex::new(r#"href\s*=\s*["']#(/[^"'#]*)["']"#).unwrap();

    for cap in hash_route_re.captures_iter(html) {
        if let Some(route) = cap.get(1) {
            let route_str = route.as_str().to_string();
            // Skip empty or just "/" routes
            if route_str != "/" && !route_str.is_empty() {
                routes.insert(route_str);
            }
        }
    }

    // Also check for programmatic routes in JavaScript (common patterns)
    let js_route_patterns = [
        // router.navigate(['...'])
        r#"navigate\s*\(\s*\[\s*['"](/[^'"]+)['"]"#,
        // this.$router.push('...')
        r#"\$router\.push\s*\(\s*['"]#?(/[^'"]+)['"]"#,
        // { path: '/...' }
        r#"path\s*:\s*['"](/[^'"]+)['"]"#,
    ];

    for pattern in js_route_patterns {
        if let Ok(re) = Regex::new(pattern) {
            for cap in re.captures_iter(html) {
                if let Some(route) = cap.get(1) {
                    let route_str = route.as_str().to_string();
                    if route_str != "/" && !route_str.is_empty() {
                        routes.insert(route_str);
                    }
                }
            }
        }
    }

    let mut routes_vec: Vec<String> = routes.into_iter().collect();
    routes_vec.sort();
    routes_vec
}

/// Analyze rendered HTML to extract hash routes (call after JS rendering)
pub fn extract_hash_routes_from_rendered(html: &str, base_url: &str) -> Vec<String> {
    use url::Url;

    let mut routes = std::collections::HashSet::new();
    let base = Url::parse(base_url).ok();

    // Match all href attributes
    let href_re = Regex::new(r#"href\s*=\s*["']([^"']+)["']"#).unwrap();

    for cap in href_re.captures_iter(html) {
        if let Some(href) = cap.get(1) {
            let href_str = href.as_str();

            // Check for hash routes
            if href_str.starts_with("#/") {
                let route = &href_str[1..]; // Remove leading #
                if route != "/" && !route.is_empty() {
                    routes.insert(route.to_string());
                }
            }
            // Check for full URLs with hash fragments
            else if let Some(ref base) = base {
                if let Ok(full_url) = base.join(href_str) {
                    if full_url.host() == base.host() {
                        if let Some(fragment) = full_url.fragment() {
                            if fragment.starts_with('/') && fragment != "/" {
                                routes.insert(fragment.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    let mut routes_vec: Vec<String> = routes.into_iter().collect();
    routes_vec.sort();
    routes_vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_angular_spa() {
        let html = r#"
            <!DOCTYPE html>
            <html>
            <head><title>App</title></head>
            <body>
                <app-root></app-root>
                <script src="runtime.js"></script>
                <script src="polyfills.js"></script>
                <script src="main.js"></script>
            </body>
            </html>
        "#;

        let analysis = analyze_page(html, "https://example.com");
        assert!(analysis.needs_js_rendering);
        assert!(matches!(
            analysis.technology,
            PageTechnology::Spa(SpaFramework::Angular) | PageTechnology::Spa(SpaFramework::Unknown)
        ));
    }

    #[test]
    fn test_detect_react_spa() {
        let html = r#"
            <!DOCTYPE html>
            <html>
            <head><title>React App</title></head>
            <body>
                <div id="root"></div>
                <script src="bundle.js"></script>
            </body>
            </html>
        "#;

        let analysis = analyze_page(html, "https://example.com");
        assert!(analysis.needs_js_rendering);
    }

    #[test]
    fn test_detect_static_page() {
        let html = r#"
            <!DOCTYPE html>
            <html>
            <head><title>Documentation</title></head>
            <body>
                <h1>Welcome to the Documentation</h1>
                <p>This is a comprehensive guide to our product. It covers installation,
                configuration, and advanced usage patterns. You'll find examples and
                best practices throughout this documentation.</p>
                <h2>Getting Started</h2>
                <p>First, install the package using npm or yarn. Then configure your
                environment variables according to the setup guide.</p>
                <h2>Configuration</h2>
                <p>The configuration file supports multiple formats including JSON, YAML,
                and TOML. Choose the format that best fits your workflow.</p>
            </body>
            </html>
        "#;

        let analysis = analyze_page(html, "https://example.com");
        assert!(!analysis.needs_js_rendering);
        assert!(matches!(analysis.technology, PageTechnology::Static));
    }

    #[test]
    fn test_content_ratio() {
        // Mostly content
        let content_heavy = "<html><body><p>This is a lot of meaningful text content that should be indexed.</p></body></html>";
        assert!(calculate_content_ratio(content_heavy) > 0.3);

        // Mostly markup
        let markup_heavy =
            "<html><body><div><div><div><span></span></div></div></div></body></html>";
        assert!(calculate_content_ratio(markup_heavy) < 0.1);
    }
}
