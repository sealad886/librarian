//! Xinference model registry snapshots and loading helpers.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::OnceLock;

pub const SNAPSHOT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegistryType {
    Llm,
    Embedding,
    Rerank,
    Image,
    Audio,
    Video,
}

impl RegistryType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RegistryType::Llm => "llm",
            RegistryType::Embedding => "embedding",
            RegistryType::Rerank => "rerank",
            RegistryType::Image => "image",
            RegistryType::Audio => "audio",
            RegistryType::Video => "video",
        }
    }

    pub fn api_label(&self) -> &'static str {
        match self {
            RegistryType::Llm => "LLM",
            RegistryType::Embedding => "embedding",
            RegistryType::Rerank => "rerank",
            RegistryType::Image => "image",
            RegistryType::Audio => "audio",
            RegistryType::Video => "video",
        }
    }
}

impl FromStr for RegistryType {
    type Err = ();

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_lowercase().as_str() {
            "llm" => Ok(Self::Llm),
            "embedding" => Ok(Self::Embedding),
            "rerank" | "reranker" => Ok(Self::Rerank),
            "image" => Ok(Self::Image),
            "audio" => Ok(Self::Audio),
            "video" => Ok(Self::Video),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrySnapshot {
    pub schema_version: u32,
    pub model_type: String,
    pub registrations: Vec<RegistryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub model_id: String,
    pub model_name: String,
    pub model_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimension: Option<usize>,
    #[serde(default)]
    pub modalities: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_batch: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_mrl: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_multi_vector: Option<bool>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryMetadata {
    pub schema_version: u32,
    pub content_hash: String,
    pub last_updated: String,
}

static EMBEDDING_SNAPSHOT: OnceLock<std::result::Result<RegistrySnapshot, String>> =
    OnceLock::new();
static RERANK_SNAPSHOT: OnceLock<std::result::Result<RegistrySnapshot, String>> = OnceLock::new();
static AUDIO_SNAPSHOT: OnceLock<std::result::Result<RegistrySnapshot, String>> = OnceLock::new();
static VIDEO_SNAPSHOT: OnceLock<std::result::Result<RegistrySnapshot, String>> = OnceLock::new();
static IMAGE_SNAPSHOT: OnceLock<std::result::Result<RegistrySnapshot, String>> = OnceLock::new();
static LLM_SNAPSHOT: OnceLock<std::result::Result<RegistrySnapshot, String>> = OnceLock::new();

pub fn registry_cache_dir() -> PathBuf {
    if let Ok(value) = env::var("LIBRARIAN_XINFERENCE_CACHE_DIR") {
        return PathBuf::from(value);
    }

    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".librarian")
        .join("xinference")
}

pub fn registry_snapshot_path(base_dir: &Path, registry_type: RegistryType) -> PathBuf {
    base_dir.join(format!("registrations.{}.json", registry_type.as_str()))
}

pub fn registry_snapshot(registry_type: RegistryType) -> Result<&'static RegistrySnapshot> {
    let cache = match registry_type {
        RegistryType::Embedding => &EMBEDDING_SNAPSHOT,
        RegistryType::Rerank => &RERANK_SNAPSHOT,
        RegistryType::Audio => &AUDIO_SNAPSHOT,
        RegistryType::Video => &VIDEO_SNAPSHOT,
        RegistryType::Image => &IMAGE_SNAPSHOT,
        RegistryType::Llm => &LLM_SNAPSHOT,
    };

    let loaded =
        cache.get_or_init(|| load_registry_snapshot(registry_type).map_err(|e| e.to_string()));
    loaded
        .as_ref()
        .map_err(|msg| Error::Config(format!("Failed to load Xinference registry: {}", msg)))
}

/// Load registry snapshot from embedded resources (authoritative for allowlists).
///
/// This function always uses the embedded snapshot included at build time.
/// User cache files are NOT consulted here to ensure deterministic allowlists
/// per CONVENTIONS.md.
pub fn load_registry_snapshot(registry_type: RegistryType) -> Result<RegistrySnapshot> {
    let embedded = embedded_snapshot_json(registry_type);
    let snapshot: RegistrySnapshot = serde_json::from_str(embedded)?;
    validate_snapshot(&snapshot, registry_type)?;
    Ok(snapshot)
}

/// Load registry snapshot from user cache directory if it exists, otherwise fall back to embedded.
///
/// This function is intended for sync operations that need to compare against user-synced data.
pub fn load_user_registry_snapshot(registry_type: RegistryType) -> Result<RegistrySnapshot> {
    let path = registry_snapshot_path(&registry_cache_dir(), registry_type);
    if path.exists() {
        let content = std::fs::read_to_string(&path)?;
        let snapshot: RegistrySnapshot = serde_json::from_str(&content)?;
        validate_snapshot(&snapshot, registry_type)?;
        return Ok(snapshot);
    }

    // Fall back to embedded snapshot
    load_registry_snapshot(registry_type)
}

pub fn embedded_snapshot_json(registry_type: RegistryType) -> &'static str {
    match registry_type {
        RegistryType::Embedding => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/resources/xinference/registrations.embedding.json"
        )),
        RegistryType::Rerank => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/resources/xinference/registrations.rerank.json"
        )),
        RegistryType::Audio => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/resources/xinference/registrations.audio.json"
        )),
        RegistryType::Video => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/resources/xinference/registrations.video.json"
        )),
        RegistryType::Image => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/resources/xinference/registrations.image.json"
        )),
        RegistryType::Llm => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/resources/xinference/registrations.llm.json"
        )),
    }
}

pub fn validate_snapshot(snapshot: &RegistrySnapshot, registry_type: RegistryType) -> Result<()> {
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION {
        return Err(Error::Config(format!(
            "Xinference registry schema version {} is not supported (expected {})",
            snapshot.schema_version, SNAPSHOT_SCHEMA_VERSION
        )));
    }

    if snapshot.model_type.trim().is_empty() {
        return Err(Error::Config(
            "Xinference registry snapshot model_type is empty".to_string(),
        ));
    }

    if snapshot.model_type.to_lowercase() != registry_type.as_str() {
        return Err(Error::Config(format!(
            "Xinference registry snapshot type mismatch: expected {}, got {}",
            registry_type.as_str(),
            snapshot.model_type
        )));
    }

    for entry in &snapshot.registrations {
        if entry.model_id.trim().is_empty() {
            return Err(Error::Config(
                "Xinference registry entry has empty model_id".to_string(),
            ));
        }
        if entry.model_name.trim().is_empty() {
            return Err(Error::Config(
                "Xinference registry entry has empty model_name".to_string(),
            ));
        }
        if entry.model_type.trim().is_empty() {
            return Err(Error::Config(
                "Xinference registry entry has empty model_type".to_string(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_registry_type_roundtrip() {
        assert_eq!(
            "embedding".parse::<RegistryType>().ok(),
            Some(RegistryType::Embedding)
        );
        assert_eq!(
            "RERANK".parse::<RegistryType>().ok(),
            Some(RegistryType::Rerank)
        );
        assert_eq!(
            "video".parse::<RegistryType>().ok(),
            Some(RegistryType::Video)
        );
        assert_eq!(RegistryType::Embedding.as_str(), "embedding");
        assert_eq!(RegistryType::Embedding.api_label(), "embedding");
    }

    #[test]
    fn test_snapshot_loads() {
        for registry_type in [
            RegistryType::Embedding,
            RegistryType::Rerank,
            RegistryType::Audio,
            RegistryType::Video,
        ] {
            let snapshot = load_registry_snapshot(registry_type).unwrap();
            assert_eq!(snapshot.model_type, registry_type.as_str());
        }
    }

    #[test]
    fn test_snapshot_no_duplicate_ids() {
        let snapshot = load_registry_snapshot(RegistryType::Embedding).unwrap();
        let mut ids = HashSet::new();
        for entry in &snapshot.registrations {
            assert!(ids.insert(entry.model_id.as_str()));
        }
    }

    #[test]
    fn test_snapshot_known_models_present() {
        let embedding = load_registry_snapshot(RegistryType::Embedding).unwrap();
        let embedding_ids = embedding
            .registrations
            .iter()
            .map(|entry| entry.model_id.as_str())
            .collect::<HashSet<_>>();
        assert!(embedding_ids.contains("BAAI/bge-small-en-v1.5"));
        assert!(embedding_ids.contains("Qwen/Qwen3-VL-Embedding-2B"));

        let rerank = load_registry_snapshot(RegistryType::Rerank).unwrap();
        let rerank_ids = rerank
            .registrations
            .iter()
            .map(|entry| entry.model_id.as_str())
            .collect::<HashSet<_>>();
        assert!(rerank_ids.contains("BAAI/bge-reranker-base"));
    }

    #[test]
    fn test_all_snapshot_contains_entries() {
        let all = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/resources/xinference/registrations.all.json"
        ));
        let snapshot: RegistrySnapshot = serde_json::from_str(all).unwrap();
        assert_eq!(snapshot.model_type, "all");
        assert!(snapshot
            .registrations
            .iter()
            .any(|entry| entry.model_type == "embedding"));
        assert!(snapshot
            .registrations
            .iter()
            .any(|entry| entry.model_type == "rerank"));
    }
}
