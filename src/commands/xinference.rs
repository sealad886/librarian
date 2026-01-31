//! Xinference command helpers.

use crate::error::{Error, Result};
use crate::xinference::{sync_xinference_snapshots, RegistryType, SyncOptions, SyncReport};
use std::path::PathBuf;
use std::time::Duration;
use url::Url;

pub struct XinferenceSyncOptions {
    pub endpoint: String,
    pub types: String,
    pub out_dir: PathBuf,
    pub write: bool,
    pub skip_update: bool,
    pub retries: usize,
    pub timeout_secs: u64,
}

pub async fn cmd_xinference_sync_models(options: XinferenceSyncOptions) -> Result<SyncReport> {
    let endpoint = Url::parse(&options.endpoint)?;
    let registry_types = parse_registry_types(&options.types)?;

    let sync_options = SyncOptions {
        endpoint,
        registry_types,
        out_dir: options.out_dir,
        refresh: !options.skip_update,
        write: options.write,
        retries: options.retries,
        timeout: Duration::from_secs(options.timeout_secs),
    };

    sync_xinference_snapshots(&sync_options).await
}

pub fn parse_registry_types(input: &str) -> Result<Vec<RegistryType>> {
    let mut types = Vec::new();
    for raw in input.split(',') {
        let value = raw.trim();
        if value.is_empty() {
            continue;
        }
        let parsed = value
            .parse::<RegistryType>()
            .map_err(|_| Error::Config(format!("Unknown registry type '{}'.", value)))?;
        types.push(parsed);
    }

    if types.is_empty() {
        return Err(Error::Config(
            "No registry types provided for Xinference sync".to_string(),
        ));
    }

    Ok(types)
}
