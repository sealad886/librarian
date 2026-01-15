//! robots.txt parsing and handling

use robotstxt::DefaultMatcher;
use tracing::debug;

/// Parsed robots.txt rules
#[derive(Debug, Clone)]
pub struct RobotsRules {
    content: String,
}

impl RobotsRules {
    /// Parse robots.txt content
    pub fn parse(content: &str) -> Self {
        Self {
            content: content.to_string(),
        }
    }

    /// Create rules that allow everything
    pub fn allow_all() -> Self {
        Self {
            content: String::new(),
        }
    }

    /// Check if a path is allowed for a user agent
    pub fn is_allowed(&self, path: &str, user_agent: &str) -> bool {
        if self.content.is_empty() {
            return true;
        }

        let mut matcher = DefaultMatcher::default();
        let allowed = matcher.one_agent_allowed_by_robots(&self.content, user_agent, path);
        
        if !allowed {
            debug!("robots.txt disallows {} for {}", path, user_agent);
        }
        
        allowed
    }

    /// Get crawl delay if specified
    pub fn crawl_delay(&self, user_agent: &str) -> Option<f64> {
        // Simple parsing for Crawl-delay directive
        let ua_lower = user_agent.to_lowercase();
        let mut in_matching_block = false;
        let default_delay = None;

        for line in self.content.lines() {
            let line = line.trim();
            
            if line.starts_with("User-agent:") {
                let agent = line.trim_start_matches("User-agent:").trim().to_lowercase();
                in_matching_block = agent == "*" || ua_lower.contains(&agent);
                if agent == "*" && default_delay.is_none() {
                    // We'll capture the default block's delay
                }
            }
            
            if in_matching_block && line.starts_with("Crawl-delay:") {
                if let Some(delay_str) = line.strip_prefix("Crawl-delay:") {
                    if let Ok(delay) = delay_str.trim().parse::<f64>() {
                        return Some(delay);
                    }
                }
            }
        }

        default_delay
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robots_allow_all() {
        let rules = RobotsRules::allow_all();
        assert!(rules.is_allowed("/any/path", "MyBot"));
    }

    #[test]
    fn test_robots_basic() {
        let content = r#"
User-agent: *
Disallow: /admin/
Disallow: /private/

User-agent: BadBot
Disallow: /
"#;
        let rules = RobotsRules::parse(content);
        
        assert!(rules.is_allowed("/public/page", "GoodBot"));
        assert!(!rules.is_allowed("/admin/secret", "GoodBot"));
        assert!(!rules.is_allowed("/anything", "BadBot"));
    }

    #[test]
    fn test_crawl_delay() {
        let content = r#"
User-agent: *
Crawl-delay: 2.5

User-agent: SpecialBot
Crawl-delay: 1.0
"#;
        let rules = RobotsRules::parse(content);
        
        assert_eq!(rules.crawl_delay("SpecialBot"), Some(1.0));
        assert_eq!(rules.crawl_delay("RandomBot"), Some(2.5));
    }
}
