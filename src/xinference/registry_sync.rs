//! Xinference registry synchronization logic used by xtask and CLI.

use crate::error::{Error, Result};
use crate::xinference::registry::{
    registry_snapshot_path, RegistryEntry, RegistryMetadata, RegistrySnapshot, RegistryType,
    SNAPSHOT_SCHEMA_VERSION,
};
use blake3::Hasher;
use chrono::Utc;
use regex::Regex;
use reqwest::{Client, Method, Url};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct SyncOptions {
    pub endpoint: Url,
    pub registry_types: Vec<RegistryType>,
    pub out_dir: PathBuf,
    pub refresh: bool,
    pub write: bool,
    pub retries: usize,
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct SyncReport {
    pub diffs: Vec<SnapshotDiff>,
    pub wrote: bool,
    pub has_changes: bool,
}

#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    pub registry_type: String,
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub total_old: usize,
    pub total_new: usize,
    pub content_changed: bool,
}

#[derive(Debug, Clone)]
struct OpenApiOperation {
    path: String,
    method: Method,
    operation_id: Option<String>,
    parameters: Vec<OpenApiParameter>,
    request_body_model_type: bool,
}

#[derive(Debug, Clone)]
struct OpenApiParameter {
    name: String,
    location: String,
}

#[derive(Debug, Clone)]
struct DiscoveredEndpoints {
    list_operation: OpenApiOperation,
    update_operation: Option<OpenApiOperation>,
}

#[derive(Debug, Default, Clone, Copy)]
struct RegistryOverrides {
    max_batch: Option<usize>,
    supports_mrl: Option<bool>,
    supports_multi_vector: Option<bool>,
    strategy: Option<&'static str>,
}

pub async fn sync_xinference_snapshots(options: &SyncOptions) -> Result<SyncReport> {
    if options.registry_types.is_empty() {
        return Err(Error::Config(
            "No registry types specified for Xinference sync".to_string(),
        ));
    }

    let client = Client::builder()
        .timeout(options.timeout)
        .build()
        .map_err(|e| Error::Embedding(format!("Failed to build HTTP client: {}", e)))?;

    let openapi = fetch_openapi_spec(&client, &options.endpoint).await?;
    let endpoints = discover_registry_endpoints(&openapi)?;

    let mut snapshots = Vec::new();
    for registry_type in &options.registry_types {
        let registration_values = fetch_registry_for_type(
            &client,
            &options.endpoint,
            &endpoints,
            *registry_type,
            options.refresh,
            options.retries,
        )
        .await?;

        let snapshot = normalize_registry_snapshot(*registry_type, registration_values)?;
        snapshots.push((*registry_type, snapshot));
    }

    let merged = merge_snapshots(&snapshots);
    let (metadata, content_hash) = build_metadata(&snapshots, &merged)?;

    let mut diffs = Vec::new();
    let mut has_changes = false;

    for (registry_type, snapshot) in &snapshots {
        let path = registry_snapshot_path(&options.out_dir, *registry_type);
        let diff = diff_snapshot_file(&path, snapshot)?;
        if diff.content_changed
            || !diff.added.is_empty()
            || !diff.removed.is_empty()
            || diff.total_old != diff.total_new
        {
            has_changes = true;
        }
        diffs.push(diff);
    }

    let all_path = options.out_dir.join("registrations.all.json");
    let all_diff = diff_snapshot_file(&all_path, &merged)?;
    if all_diff.content_changed
        || !all_diff.added.is_empty()
        || !all_diff.removed.is_empty()
        || all_diff.total_old != all_diff.total_new
    {
        has_changes = true;
    }
    diffs.push(all_diff);

    if options.write {
        std::fs::create_dir_all(&options.out_dir)?;

        for (registry_type, snapshot) in &snapshots {
            let path = registry_snapshot_path(&options.out_dir, *registry_type);
            write_snapshot_file(&path, snapshot)?;
        }

        write_snapshot_file(&all_path, &merged)?;
        write_metadata(&options.out_dir, &metadata, &content_hash)?;

        return Ok(SyncReport {
            diffs,
            wrote: true,
            has_changes,
        });
    }

    Ok(SyncReport {
        diffs,
        wrote: false,
        has_changes,
    })
}

async fn fetch_openapi_spec(client: &Client, endpoint: &Url) -> Result<Value> {
    let direct = endpoint.join("/openapi.json")?;
    let direct_resp = client.get(direct.clone()).send().await;
    if let Ok(resp) = direct_resp {
        if resp.status().is_success() {
            return Ok(resp.json::<Value>().await?);
        }
    }

    let docs_url = endpoint.join("/docs")?;
    let docs_resp = client
        .get(docs_url)
        .send()
        .await
        .map_err(|e| Error::Embedding(format!("Failed to fetch Xinference docs: {}", e)))?;

    if !docs_resp.status().is_success() {
        return Err(Error::Embedding(format!(
            "Xinference docs endpoint returned HTTP {}",
            docs_resp.status()
        )));
    }

    let body = docs_resp.text().await?;
    let re = Regex::new(r"(?i)(/openapi\.json)\b")
        .map_err(|e| Error::Embedding(format!("Failed to build openapi regex: {}", e)))?;
    let Some(mat) = re.find(&body) else {
        return Err(Error::Embedding(
            "Unable to locate openapi.json in Xinference /docs HTML".to_string(),
        ));
    };

    let openapi_path = mat.as_str();
    let openapi_url = endpoint.join(openapi_path)?;
    let resp = client
        .get(openapi_url)
        .send()
        .await
        .map_err(|e| Error::Embedding(format!("Failed to fetch OpenAPI spec: {}", e)))?;
    if !resp.status().is_success() {
        return Err(Error::Embedding(format!(
            "OpenAPI spec returned HTTP {}",
            resp.status()
        )));
    }

    Ok(resp.json::<Value>().await?)
}

fn discover_registry_endpoints(spec: &Value) -> Result<DiscoveredEndpoints> {
    let paths = spec
        .get("paths")
        .and_then(Value::as_object)
        .ok_or_else(|| Error::Embedding("OpenAPI spec missing 'paths'".to_string()))?;

    let mut operations = Vec::new();
    for (path, path_item) in paths {
        let Some(methods) = path_item.as_object() else {
            continue;
        };

        for (method, details) in methods {
            let method = method.to_lowercase();
            let method = match method.as_str() {
                "get" => Method::GET,
                "post" => Method::POST,
                "put" => Method::PUT,
                "patch" => Method::PATCH,
                _ => continue,
            };

            let details_obj = details.as_object().cloned().unwrap_or_default();
            let operation_id = details_obj
                .get("operationId")
                .and_then(Value::as_str)
                .map(|s| s.to_string());

            let parameters = details_obj
                .get("parameters")
                .and_then(Value::as_array)
                .map(|params| {
                    params
                        .iter()
                        .filter_map(|param| {
                            let obj = param.as_object()?;
                            let name = obj.get("name")?.as_str()?.to_string();
                            let location = obj.get("in")?.as_str()?.to_string();
                            Some(OpenApiParameter { name, location })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            let request_body_model_type = details_obj
                .get("requestBody")
                .and_then(|body| body.get("content"))
                .and_then(Value::as_object)
                .and_then(|content| content.values().next())
                .and_then(|media| media.get("schema"))
                .and_then(Value::as_object)
                .and_then(|schema| schema.get("properties"))
                .and_then(Value::as_object)
                .map(|props| {
                    props
                        .keys()
                        .any(|key| key.eq_ignore_ascii_case("model_type"))
                })
                .unwrap_or(false);

            operations.push(OpenApiOperation {
                path: path.to_string(),
                method,
                operation_id,
                parameters,
                request_body_model_type,
            });
        }
    }

    let list_operation = select_list_operation(&operations)?;
    let update_operation = select_update_operation(&operations);

    Ok(DiscoveredEndpoints {
        list_operation,
        update_operation,
    })
}

fn select_list_operation(operations: &[OpenApiOperation]) -> Result<OpenApiOperation> {
    let mut candidates = Vec::new();
    for op in operations {
        if op.method == Method::GET && op.path.contains("model_registrations") {
            let mut score = 0;
            if op.path.contains('{') {
                score += 3;
            }
            if op
                .parameters
                .iter()
                .any(|p| p.name.eq_ignore_ascii_case("model_type"))
            {
                score += 2;
            }
            if op
                .operation_id
                .as_deref()
                .map(|id| id.to_lowercase().contains("registration"))
                .unwrap_or(false)
            {
                score += 1;
            }
            candidates.push((score, op.clone()));
        }
    }

    candidates
        .into_iter()
        .max_by_key(|(score, _)| *score)
        .map(|(_, op)| op)
        .ok_or_else(|| Error::Embedding("Failed to locate registry list endpoint".to_string()))
}

fn select_update_operation(operations: &[OpenApiOperation]) -> Option<OpenApiOperation> {
    let mut candidates = Vec::new();
    for op in operations {
        if !matches!(op.method, Method::POST | Method::PUT | Method::PATCH) {
            continue;
        }
        if !op.path.contains("model_registrations") {
            continue;
        }

        let path_lower = op.path.to_lowercase();
        let op_lower = op
            .operation_id
            .as_deref()
            .map(|id| id.to_lowercase())
            .unwrap_or_default();

        let is_update = path_lower.contains("update") || op_lower.contains("update");
        let is_refresh = path_lower.contains("refresh") || op_lower.contains("refresh");
        if !is_update && !is_refresh {
            continue;
        }

        let mut score = 0;
        if op.path.contains('{') {
            score += 3;
        }
        if is_update {
            score += 2;
        }
        if is_refresh {
            score += 1;
        }
        candidates.push((score, op.clone()));
    }

    candidates
        .into_iter()
        .max_by_key(|(score, _)| *score)
        .map(|(_, op)| op)
}

async fn fetch_registry_for_type(
    client: &Client,
    endpoint: &Url,
    endpoints: &DiscoveredEndpoints,
    registry_type: RegistryType,
    refresh: bool,
    retries: usize,
) -> Result<Vec<Value>> {
    if refresh {
        if let Some(update_op) = &endpoints.update_operation {
            let request = build_request_for_type(client, endpoint, update_op, registry_type)?;
            send_with_retry::<()>(request, retries).await?;
        } else {
            return Err(Error::Embedding(
                "Xinference OpenAPI spec does not expose a model update endpoint".to_string(),
            ));
        }
    }

    let list_request =
        build_request_for_type(client, endpoint, &endpoints.list_operation, registry_type)?;
    let response: Value = send_with_retry(list_request, retries).await?;
    extract_registrations(response)
}

fn build_request_for_type(
    client: &Client,
    endpoint: &Url,
    operation: &OpenApiOperation,
    registry_type: RegistryType,
) -> Result<reqwest::RequestBuilder> {
    let path = apply_type_to_path(&operation.path, registry_type);
    let url = endpoint.join(&path)?;
    let mut request = client.request(operation.method.clone(), url);

    if operation
        .parameters
        .iter()
        .any(|p| p.location == "query" && p.name.eq_ignore_ascii_case("model_type"))
    {
        request = request.query(&[("model_type", registry_type.api_label())]);
    }

    if operation.request_body_model_type {
        request = request.json(&serde_json::json!({
            "model_type": registry_type.api_label()
        }));
    }

    Ok(request)
}

fn apply_type_to_path(path: &str, registry_type: RegistryType) -> String {
    if !path.contains('{') {
        return path.to_string();
    }

    let re = Regex::new(r"\{[^}]+\}").unwrap_or_else(|_| Regex::new(r"\{\}").unwrap());
    re.replace_all(path, registry_type.api_label()).to_string()
}

async fn send_with_retry<T: serde::de::DeserializeOwned>(
    request: reqwest::RequestBuilder,
    retries: usize,
) -> Result<T> {
    let mut last_err: Option<Error> = None;
    for attempt in 0..=retries {
        let req = request
            .try_clone()
            .ok_or_else(|| Error::Embedding("Failed to clone xinference request".to_string()))?;
        match req.send().await {
            Ok(response) => match response.error_for_status() {
                Ok(ok) => return Ok(ok.json::<T>().await?),
                Err(e) => last_err = Some(Error::Embedding(e.to_string())),
            },
            Err(e) => last_err = Some(Error::Embedding(e.to_string())),
        }

        if attempt < retries {
            let delay = Duration::from_millis(250 * (attempt + 1) as u64);
            tokio::time::sleep(delay).await;
        }
    }

    Err(last_err.unwrap_or_else(|| Error::Embedding("Xinference request failed".to_string())))
}

fn extract_registrations(response: Value) -> Result<Vec<Value>> {
    match response {
        Value::Array(values) => Ok(values),
        Value::Object(mut map) => {
            for key in ["data", "models", "registrations", "items"] {
                if let Some(Value::Array(values)) = map.remove(key) {
                    return Ok(values);
                }
            }
            Err(Error::Embedding(
                "Xinference registry response did not contain a list".to_string(),
            ))
        }
        _ => Err(Error::Embedding(
            "Unexpected Xinference registry response shape".to_string(),
        )),
    }
}

fn normalize_registry_snapshot(
    registry_type: RegistryType,
    registrations: Vec<Value>,
) -> Result<RegistrySnapshot> {
    let mut normalized = Vec::new();

    for value in registrations {
        let Value::Object(map) = value else {
            continue;
        };

        let model_name = extract_string(&map, &["model_name", "name", "model", "id"])
            .unwrap_or_else(|| registry_type.as_str().to_string());

        let model_id = extract_string(&map, &["model_id", "model_uid", "id", "name"])
            .unwrap_or_else(|| model_name.clone());

        let model_family = extract_string(&map, &["model_family", "family"]);
        let dimension = extract_usize(
            &map,
            &[
                "dimension",
                "embedding_dim",
                "embedding_size",
                "vector_size",
            ],
        );
        let modalities = extract_string_list(
            &map,
            &["modalities", "modality", "abilities", "model_abilities"],
        );
        let max_batch = extract_usize(&map, &["max_batch", "max_batch_size"]);
        let supports_mrl = extract_bool(&map, &["supports_mrl", "mrl"]);
        let supports_multi_vector = extract_bool(&map, &["supports_multi_vector", "multivector"]);

        let mut aliases = BTreeSet::new();
        if model_name != model_id {
            aliases.insert(model_name.clone());
        }
        if let Some(alias) = extract_string(&map, &["model_name", "name"]) {
            if alias != model_id {
                aliases.insert(alias);
            }
        }

        normalized.push(RegistryEntry {
            model_id,
            model_name,
            model_type: registry_type.as_str().to_string(),
            model_family,
            dimension,
            modalities,
            aliases: aliases.into_iter().collect(),
            max_batch,
            supports_mrl,
            supports_multi_vector,
            metadata: BTreeMap::new(),
        });
    }

    apply_registry_overrides(registry_type, &mut normalized);

    normalized.sort_by(|a, b| {
        let name_cmp = a
            .model_name
            .to_lowercase()
            .cmp(&b.model_name.to_lowercase());
        if name_cmp == std::cmp::Ordering::Equal {
            return a.model_id.to_lowercase().cmp(&b.model_id.to_lowercase());
        }
        name_cmp
    });

    Ok(RegistrySnapshot {
        schema_version: SNAPSHOT_SCHEMA_VERSION,
        model_type: registry_type.as_str().to_string(),
        registrations: normalized,
    })
}

fn apply_registry_overrides(registry_type: RegistryType, entries: &mut [RegistryEntry]) {
    for entry in entries {
        let overrides = match registry_type {
            RegistryType::Embedding => embedding_overrides(entry.model_id.as_str()),
            RegistryType::Rerank => rerank_overrides(entry.model_id.as_str()),
            _ => RegistryOverrides::default(),
        };

        if let Some(max_batch) = overrides.max_batch {
            entry.max_batch = Some(max_batch);
        }
        if let Some(supports_mrl) = overrides.supports_mrl {
            entry.supports_mrl = Some(supports_mrl);
        }
        if let Some(supports_multi_vector) = overrides.supports_multi_vector {
            entry.supports_multi_vector = Some(supports_multi_vector);
        }
        if let Some(strategy) = overrides.strategy {
            entry
                .metadata
                .insert("strategy".to_string(), Value::String(strategy.to_string()));
        }
    }
}

fn embedding_overrides(model_id: &str) -> RegistryOverrides {
    match model_id {
        "BAAI/bge-small-en-v1.5" => RegistryOverrides {
            max_batch: Some(32),
            ..RegistryOverrides::default()
        },
        "BAAI/bge-base-en-v1.5" => RegistryOverrides {
            max_batch: Some(32),
            ..RegistryOverrides::default()
        },
        "BAAI/bge-large-en-v1.5" => RegistryOverrides {
            max_batch: Some(16),
            ..RegistryOverrides::default()
        },
        "sentence-transformers/all-MiniLM-L6-v2" => RegistryOverrides {
            max_batch: Some(32),
            ..RegistryOverrides::default()
        },
        "Qwen/Qwen3-VL-Embedding-2B" => RegistryOverrides {
            max_batch: Some(8),
            strategy: Some("vl_embedding"),
            ..RegistryOverrides::default()
        },
        "Qwen/Qwen3-VL-Embedding-8B" => RegistryOverrides {
            max_batch: Some(4),
            strategy: Some("vl_embedding"),
            ..RegistryOverrides::default()
        },
        "jinaai/jina-clip-v2" => RegistryOverrides {
            max_batch: Some(16),
            ..RegistryOverrides::default()
        },
        "google/siglip2-base-patch16-224" => RegistryOverrides {
            max_batch: Some(16),
            ..RegistryOverrides::default()
        },
        "vidore/colpali" => RegistryOverrides {
            max_batch: Some(8),
            supports_multi_vector: Some(true),
            strategy: Some("late_interaction"),
            ..RegistryOverrides::default()
        },
        "laion/clap-htsat-unfused" => RegistryOverrides {
            max_batch: Some(16),
            ..RegistryOverrides::default()
        },
        "OpenGVLab/InternVideo2-Stage2_1B-224p-f4" => RegistryOverrides {
            max_batch: Some(4),
            ..RegistryOverrides::default()
        },
        "Salesforce/blip2-itm-vit-g" => RegistryOverrides {
            max_batch: Some(4),
            ..RegistryOverrides::default()
        },
        "facebook/ImageBind" => RegistryOverrides {
            max_batch: Some(4),
            ..RegistryOverrides::default()
        },
        _ => RegistryOverrides::default(),
    }
}

fn rerank_overrides(model_id: &str) -> RegistryOverrides {
    match model_id {
        "BAAI/bge-reranker-base" => RegistryOverrides {
            max_batch: Some(32),
            ..RegistryOverrides::default()
        },
        "Qwen/Qwen3-VL-Reranker-2B" => RegistryOverrides {
            max_batch: Some(8),
            ..RegistryOverrides::default()
        },
        "Qwen/Qwen3-VL-Reranker-8B" => RegistryOverrides {
            max_batch: Some(4),
            ..RegistryOverrides::default()
        },
        "jinaai/jina-reranker-m0" => RegistryOverrides {
            max_batch: Some(16),
            ..RegistryOverrides::default()
        },
        "lightonai/MonoQwen2-VL-v0.1" => RegistryOverrides {
            max_batch: Some(8),
            ..RegistryOverrides::default()
        },
        _ => RegistryOverrides::default(),
    }
}

fn merge_snapshots(snapshots: &[(RegistryType, RegistrySnapshot)]) -> RegistrySnapshot {
    let mut entries = Vec::new();
    for (_ty, snapshot) in snapshots {
        entries.extend(snapshot.registrations.clone());
    }

    entries.sort_by(|a, b| {
        let type_cmp = a.model_type.cmp(&b.model_type);
        if type_cmp == std::cmp::Ordering::Equal {
            let name_cmp = a
                .model_name
                .to_lowercase()
                .cmp(&b.model_name.to_lowercase());
            if name_cmp == std::cmp::Ordering::Equal {
                return a.model_id.to_lowercase().cmp(&b.model_id.to_lowercase());
            }
            return name_cmp;
        }
        type_cmp
    });

    RegistrySnapshot {
        schema_version: SNAPSHOT_SCHEMA_VERSION,
        model_type: "all".to_string(),
        registrations: entries,
    }
}

fn build_metadata(
    snapshots: &[(RegistryType, RegistrySnapshot)],
    merged: &RegistrySnapshot,
) -> Result<(RegistryMetadata, String)> {
    let mut hasher = Hasher::new();
    for (_ty, snapshot) in snapshots {
        let serialized = serde_json::to_vec(snapshot)?;
        hasher.update(&serialized);
    }
    let merged_serialized = serde_json::to_vec(merged)?;
    hasher.update(&merged_serialized);

    let content_hash = hasher.finalize().to_hex().to_string();

    let metadata = RegistryMetadata {
        schema_version: SNAPSHOT_SCHEMA_VERSION,
        content_hash: content_hash.clone(),
        last_updated: Utc::now().to_rfc3339(),
    };

    Ok((metadata, content_hash))
}

fn diff_snapshot_file(path: &Path, new_snapshot: &RegistrySnapshot) -> Result<SnapshotDiff> {
    let new_json = serde_json::to_string_pretty(new_snapshot)?;

    let (old_count, old_ids, content_changed) = if path.exists() {
        let content = std::fs::read_to_string(path)?;
        let old: RegistrySnapshot = serde_json::from_str(&content)?;
        let ids = old
            .registrations
            .iter()
            .map(|entry| entry.model_id.clone())
            .collect::<BTreeSet<_>>();
        let content_changed = content.trim_end() != new_json.trim_end();
        (old.registrations.len(), ids, content_changed)
    } else {
        (0, BTreeSet::new(), true)
    };

    let new_ids = new_snapshot
        .registrations
        .iter()
        .map(|entry| entry.model_id.clone())
        .collect::<BTreeSet<_>>();

    let added = new_ids.difference(&old_ids).cloned().collect::<Vec<_>>();
    let removed = old_ids.difference(&new_ids).cloned().collect::<Vec<_>>();

    Ok(SnapshotDiff {
        registry_type: new_snapshot.model_type.clone(),
        added,
        removed,
        total_old: old_count,
        total_new: new_snapshot.registrations.len(),
        content_changed,
    })
}

fn write_snapshot_file(path: &Path, snapshot: &RegistrySnapshot) -> Result<()> {
    let json = serde_json::to_string_pretty(snapshot)?;
    std::fs::write(path, format!("{}\n", json))?;
    Ok(())
}

fn write_metadata(out_dir: &Path, metadata: &RegistryMetadata, content_hash: &str) -> Result<()> {
    let metadata_path = out_dir.join("metadata.json");
    let mut metadata = metadata.clone();

    if metadata_path.exists() {
        if let Ok(existing) = std::fs::read_to_string(&metadata_path) {
            if let Ok(existing_meta) = serde_json::from_str::<RegistryMetadata>(&existing) {
                if existing_meta.content_hash == content_hash {
                    metadata.last_updated = existing_meta.last_updated;
                }
            }
        }
    }

    let json = serde_json::to_string_pretty(&metadata)?;
    std::fs::write(metadata_path, format!("{}\n", json))?;
    Ok(())
}

fn extract_string(map: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(value) = map.get(*key) {
            if let Some(s) = value.as_str() {
                return Some(s.to_string());
            }
        }
    }
    None
}

fn extract_usize(map: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<usize> {
    for key in keys {
        if let Some(value) = map.get(*key) {
            if let Some(num) = value.as_u64() {
                return Some(num as usize);
            }
        }
    }
    None
}

fn extract_bool(map: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<bool> {
    for key in keys {
        if let Some(value) = map.get(*key) {
            if let Some(b) = value.as_bool() {
                return Some(b);
            }
        }
    }
    None
}

fn extract_string_list(map: &serde_json::Map<String, Value>, keys: &[&str]) -> Vec<String> {
    for key in keys {
        if let Some(value) = map.get(*key) {
            match value {
                Value::Array(items) => {
                    let list = items
                        .iter()
                        .filter_map(|item| item.as_str().map(|s| s.to_string()))
                        .collect::<Vec<_>>();
                    if !list.is_empty() {
                        return list;
                    }
                }
                Value::String(value) => {
                    return vec![value.to_string()];
                }
                _ => {}
            }
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_registry_snapshot_is_stable() {
        let raw = vec![
            serde_json::json!({
                "model_name": "b",
                "model_id": "b",
                "dimension": 1
            }),
            serde_json::json!({
                "model_name": "a",
                "model_id": "a",
                "dimension": 2
            }),
        ];

        let snapshot = normalize_registry_snapshot(RegistryType::Embedding, raw).unwrap();
        let first = serde_json::to_string_pretty(&snapshot).unwrap();
        let second = serde_json::to_string_pretty(&snapshot).unwrap();
        assert_eq!(first, second);
        assert_eq!(snapshot.registrations[0].model_name, "a");
        assert_eq!(snapshot.registrations[1].model_name, "b");
    }
}
