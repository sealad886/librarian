//! Rate limiting for web crawling

use governor::{Quota, RateLimiter};
use nonzero_ext::nonzero;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time::Instant;
use tracing::trace;

/// Per-host rate limiter
#[derive(Clone)]
pub struct HostRateLimiter {
    inner: Arc<Mutex<RateLimiterInner>>,
}

struct RateLimiterInner {
    last_request: Option<Instant>,
    min_interval: Duration,
}

impl HostRateLimiter {
    /// Create a new rate limiter for the given requests per second
    pub fn new(requests_per_second: f64) -> Self {
        let interval_ms = (1000.0 / requests_per_second) as u64;
        
        Self {
            inner: Arc::new(Mutex::new(RateLimiterInner {
                last_request: None,
                min_interval: Duration::from_millis(interval_ms),
            })),
        }
    }

    /// Wait until the next request is allowed
    pub async fn wait(&self) {
        let mut inner = self.inner.lock().await;
        
        if let Some(last) = inner.last_request {
            let elapsed = last.elapsed();
            if elapsed < inner.min_interval {
                let wait_time = inner.min_interval - elapsed;
                trace!("Rate limiting: waiting {:?}", wait_time);
                tokio::time::sleep(wait_time).await;
            }
        }
        
        inner.last_request = Some(Instant::now());
    }
}

/// Global rate limiter for all requests
pub struct GlobalRateLimiter {
    limiter: RateLimiter<governor::state::NotKeyed, governor::state::InMemoryState, governor::clock::DefaultClock>,
}

impl GlobalRateLimiter {
    /// Create a new global rate limiter
    pub fn new(requests_per_second: u32) -> Self {
        let rps = NonZeroU32::new(requests_per_second).unwrap_or(nonzero!(1u32));
        let quota = Quota::per_second(rps);
        let limiter = RateLimiter::direct(quota);
        
        Self { limiter }
    }

    /// Wait until a request is allowed
    pub async fn wait(&self) {
        self.limiter.until_ready().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn test_host_rate_limiter() {
        let limiter = HostRateLimiter::new(10.0); // 10 req/s = 100ms between requests
        
        let start = Instant::now();
        limiter.wait().await;
        limiter.wait().await;
        limiter.wait().await;
        let elapsed = start.elapsed();
        
        // Should take at least 200ms for 3 requests (2 intervals)
        assert!(elapsed >= Duration::from_millis(180));
    }

    #[tokio::test]
    async fn test_global_rate_limiter() {
        let limiter = GlobalRateLimiter::new(100);
        
        // Should be able to make many requests quickly
        for _ in 0..10 {
            limiter.wait().await;
        }
    }
}
