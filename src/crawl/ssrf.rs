//! SSRF (Server-Side Request Forgery) protection
//!
//! Provides async DNS resolution and IP validation to prevent SSRF attacks.
//! Blocks requests to private networks, localhost, and other restricted addresses.

use crate::error::{Error, Result};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use tokio::net::lookup_host;
use tracing::debug;
use url::Url;

/// Configuration for SSRF validation
#[derive(Debug, Clone, Default)]
pub struct SsrfConfig {
    /// Allow localhost addresses (for testing)
    pub allow_localhost: bool,
}

/// Check if an IPv4 address is in a private/restricted range
fn is_private_ipv4(ip: Ipv4Addr, allow_localhost: bool) -> bool {
    // Loopback (127.0.0.0/8) - conditionally blocked
    if ip.is_loopback() && !allow_localhost {
        return true;
    }

    // Private ranges (RFC 1918)
    ip.is_private()
        // Link-local (169.254.0.0/16)
        || ip.is_link_local()
        // Broadcast
        || ip.is_broadcast()
        // Documentation ranges (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24)
        || ip.is_documentation()
        // Unspecified (0.0.0.0)
        || ip.is_unspecified()
        // Shared address space (100.64.0.0/10) - RFC 6598
        || (ip.octets()[0] == 100 && (ip.octets()[1] & 0xC0) == 64)
        // Benchmarking (198.18.0.0/15)
        || (ip.octets()[0] == 198 && (ip.octets()[1] & 0xFE) == 18)
}

/// Check if an IPv6 address is in a private/restricted range
fn is_private_ipv6(ip: Ipv6Addr, allow_localhost: bool) -> bool {
    // Loopback (::1) - conditionally blocked
    if ip.is_loopback() && !allow_localhost {
        return true;
    }

    // Unspecified (::)
    ip.is_unspecified()
        // Unique local (fc00::/7)
        || ((ip.segments()[0] & 0xFE00) == 0xFC00)
        // Link-local (fe80::/10)
        || ((ip.segments()[0] & 0xFFC0) == 0xFE80)
        // Documentation (2001:db8::/32)
        || (ip.segments()[0] == 0x2001 && ip.segments()[1] == 0x0DB8)
        // IPv4-mapped addresses - check the embedded IPv4
        || ip.to_ipv4_mapped().map(|v4| is_private_ipv4(v4, allow_localhost)).unwrap_or(false)
}

/// Check if an IP address is private/restricted
fn is_restricted_ip(ip: IpAddr, allow_localhost: bool) -> bool {
    match ip {
        IpAddr::V4(v4) => is_private_ipv4(v4, allow_localhost),
        IpAddr::V6(v6) => is_private_ipv6(v6, allow_localhost),
    }
}

/// Validate a URL for SSRF protection
///
/// Performs async DNS resolution and checks that all resolved IP addresses
/// are not in private/restricted ranges.
///
/// # Arguments
///
/// * `url` - The URL to validate
///
/// # Returns
///
/// Returns `Ok(())` if the URL is safe, or an error describing the issue.
///
/// # Errors
///
/// Returns error if:
/// - URL scheme is not http or https
/// - URL has no host
/// - DNS resolution fails
/// - Any resolved IP is in a restricted range
pub async fn validate_url_ssrf(url: &str) -> Result<()> {
    validate_url_ssrf_with_config(url, &SsrfConfig::default()).await
}

/// Validate a URL for SSRF protection with custom configuration
///
/// Performs async DNS resolution and checks that all resolved IP addresses
/// are not in private/restricted ranges.
///
/// # Arguments
///
/// * `url` - The URL to validate
/// * `config` - SSRF configuration options
///
/// # Returns
///
/// Returns `Ok(())` if the URL is safe, or an error describing the issue.
pub async fn validate_url_ssrf_with_config(url: &str, config: &SsrfConfig) -> Result<()> {
    let parsed =
        Url::parse(url).map_err(|e| Error::Crawl(format!("Invalid URL '{}': {}", url, e)))?;

    // Only allow http/https schemes
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(Error::Crawl(format!(
                "URL scheme '{}' not allowed (only http/https permitted)",
                scheme
            )));
        }
    }

    // Extract host
    let host = parsed
        .host_str()
        .ok_or_else(|| Error::Crawl(format!("URL '{}' has no host", url)))?;

    // Check for obvious localhost patterns before DNS (unless allow_localhost is set)
    if !config.allow_localhost {
        let host_lower = host.to_lowercase();
        if host_lower == "localhost"
            || host_lower == "localhost.localdomain"
            || host_lower.ends_with(".localhost")
            || host_lower.ends_with(".local")
        {
            return Err(Error::Crawl(format!(
                "URL host '{}' resolves to localhost (SSRF blocked)",
                host
            )));
        }
    }

    // Get port (default based on scheme)
    let port = parsed.port().unwrap_or(match parsed.scheme() {
        "https" => 443,
        _ => 80,
    });

    // Perform async DNS resolution
    let host_port = format!("{}:{}", host, port);
    let addrs: Vec<_> = lookup_host(&host_port)
        .await
        .map_err(|e| Error::Crawl(format!("DNS resolution failed for '{}': {}", host, e)))?
        .collect();

    if addrs.is_empty() {
        return Err(Error::Crawl(format!(
            "DNS resolution for '{}' returned no addresses",
            host
        )));
    }

    debug!(host = %host, addresses = ?addrs, "SSRF validation: resolved addresses");

    // Check all resolved IPs
    for addr in &addrs {
        if is_restricted_ip(addr.ip(), config.allow_localhost) {
            return Err(Error::Crawl(format!(
                "URL host '{}' resolves to restricted IP {} (SSRF blocked)",
                host,
                addr.ip()
            )));
        }
    }

    Ok(())
}

/// Validate a URL for SSRF protection (convenience wrapper that takes Url)
///
/// # Arguments
///
/// * `url` - The parsed URL to validate
///
/// # Returns
///
/// Returns `Ok(())` if the URL is safe, or an error describing the issue.
pub async fn validate_url_ssrf_parsed(url: &Url) -> Result<()> {
    validate_url_ssrf(url.as_str()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_private_ipv4() {
        // Private ranges
        assert!(is_private_ipv4(Ipv4Addr::new(10, 0, 0, 1), false));
        assert!(is_private_ipv4(Ipv4Addr::new(172, 16, 0, 1), false));
        assert!(is_private_ipv4(Ipv4Addr::new(172, 31, 255, 255), false));
        assert!(is_private_ipv4(Ipv4Addr::new(192, 168, 1, 1), false));

        // Loopback (blocked when allow_localhost = false)
        assert!(is_private_ipv4(Ipv4Addr::new(127, 0, 0, 1), false));
        assert!(is_private_ipv4(Ipv4Addr::new(127, 255, 255, 255), false));
        // Loopback (allowed when allow_localhost = true)
        assert!(!is_private_ipv4(Ipv4Addr::new(127, 0, 0, 1), true));

        // Link-local
        assert!(is_private_ipv4(Ipv4Addr::new(169, 254, 0, 1), false));

        // Shared address space (RFC 6598)
        assert!(is_private_ipv4(Ipv4Addr::new(100, 64, 0, 1), false));
        assert!(is_private_ipv4(Ipv4Addr::new(100, 127, 255, 255), false));

        // Public IPs should not be private
        assert!(!is_private_ipv4(Ipv4Addr::new(8, 8, 8, 8), false));
        assert!(!is_private_ipv4(Ipv4Addr::new(1, 1, 1, 1), false));
        assert!(!is_private_ipv4(Ipv4Addr::new(93, 184, 216, 34), false));
    }

    #[test]
    fn test_is_private_ipv6() {
        // Loopback (blocked when allow_localhost = false)
        assert!(is_private_ipv6(
            Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1),
            false
        ));
        // Loopback (allowed when allow_localhost = true)
        assert!(!is_private_ipv6(
            Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1),
            true
        ));

        // Unspecified
        assert!(is_private_ipv6(
            Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0),
            false
        ));

        // Unique local
        assert!(is_private_ipv6(
            Ipv6Addr::new(0xFC00, 0, 0, 0, 0, 0, 0, 1),
            false
        ));
        assert!(is_private_ipv6(
            Ipv6Addr::new(0xFD00, 0, 0, 0, 0, 0, 0, 1),
            false
        ));

        // Link-local
        assert!(is_private_ipv6(
            Ipv6Addr::new(0xFE80, 0, 0, 0, 0, 0, 0, 1),
            false
        ));

        // Documentation
        assert!(is_private_ipv6(
            Ipv6Addr::new(0x2001, 0x0DB8, 0, 0, 0, 0, 0, 1),
            false
        ));

        // Public IPv6 should not be private
        assert!(!is_private_ipv6(
            Ipv6Addr::new(0x2607, 0xF8B0, 0x4004, 0x0800, 0, 0, 0, 0x200E),
            false
        )); // Google
    }

    #[tokio::test]
    async fn test_validate_url_ssrf_scheme() {
        // File scheme should be blocked
        let result = validate_url_ssrf("file:///etc/passwd").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("scheme"));

        // FTP should be blocked
        let result = validate_url_ssrf("ftp://example.com/file").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validate_url_ssrf_localhost() {
        // localhost should be blocked
        let result = validate_url_ssrf("http://localhost/").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("localhost"));

        // .local should be blocked
        let result = validate_url_ssrf("http://myhost.local/").await;
        assert!(result.is_err());
    }

    // Note: Tests that require actual DNS resolution are not included here
    // as they would make unit tests flaky. Integration tests should cover
    // real DNS resolution scenarios.
}
