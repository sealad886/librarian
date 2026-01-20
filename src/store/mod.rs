//! Qdrant vector database integration
//!
//! This module wraps the Qdrant client and provides:
//! - Collection management
//! - Point upsert/delete operations
//! - Vector search

mod payload;

pub use payload::*;

use crate::config::{Config, EmbeddingDimensionSource, ResolvedEmbeddingConfig};
use crate::error::{Error, Result};
use qdrant_client::qdrant::{
    CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, GetCollectionInfoResponse,
    PointId, PointStruct, ScalarQuantizationBuilder, SearchPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use serde_json::Value;
use tracing::{debug, info};
use uuid::Uuid;

/// Information about a Qdrant collection
#[derive(Debug, Clone)]
pub struct CollectionInfo {
    pub points_count: u64,
    pub indexed_vectors_count: u64,
    pub status: String,
}

/// Qdrant store handle
pub struct QdrantStore {
    client: Qdrant,
    collection: String,
    dimension: usize,
    embedding_context: Option<EmbeddingContext>,
}

#[derive(Debug, Clone)]
pub struct EmbeddingContext {
    pub model_id: String,
    pub family: String,
    pub dimension: usize,
    pub dimension_source: EmbeddingDimensionSource,
}

impl From<&ResolvedEmbeddingConfig> for EmbeddingContext {
    fn from(config: &ResolvedEmbeddingConfig) -> Self {
        Self {
            model_id: config.model_id.clone(),
            family: config.family.clone(),
            dimension: config.dimension,
            dimension_source: config.dimension_source,
        }
    }
}

impl QdrantStore {
    /// Connect to Qdrant using config
    pub async fn connect(config: &Config, embedding: &ResolvedEmbeddingConfig) -> Result<Self> {
        Self::new(
            &config.qdrant_url,
            &config.collection_name,
            embedding.dimension,
            Some(embedding),
        )
        .await
    }

    /// Create a new store connection directly with URL and collection name
    pub async fn new(
        url: &str,
        collection: &str,
        dimension: usize,
        embedding: Option<&ResolvedEmbeddingConfig>,
    ) -> Result<Self> {
        debug!("Connecting to Qdrant at {}", url);

        let client = Qdrant::from_url(url)
            .skip_compatibility_check()
            .build()
            .map_err(|e| Error::Qdrant(e.to_string()))?;

        let store = Self {
            client,
            collection: collection.to_string(),
            dimension,
            embedding_context: embedding.map(EmbeddingContext::from),
        };

        Ok(store)
    }

    /// Get the expected vector dimension for this store
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Ensure the collection exists with correct configuration
    pub async fn ensure_collection(&self) -> Result<()> {
        // Check if collection exists
        let exists = self.client.collection_exists(&self.collection).await?;

        if exists {
            debug!("Collection {} already exists", self.collection);

            if let Some(sizes) = self.collection_vector_sizes().await? {
                if sizes.len() > 1 {
                    let detail = describe_vector_sizes(&sizes);
                    return Err(Error::Qdrant(format!(
                        "Collection '{}' uses named vectors ({}) which are not supported by this store",
                        self.collection, detail
                    )));
                }

                if let Some((_, size)) = sizes.first() {
                    let size = *size as usize;
                    if size != self.dimension {
                        let detail = if let Some(ctx) = &self.embedding_context {
                            format!(
                                "model '{}' (family '{}') expects {} from {}",
                                ctx.model_id, ctx.family, ctx.dimension, ctx.dimension_source
                            )
                        } else {
                            format!("config expects {}", self.dimension)
                        };
                        return Err(Error::Qdrant(format!(
                            "Collection '{}' has vector size {}, but {}. Remediation: set a new collection name or migrate/reindex with the expected dimension.",
                            self.collection, size, detail
                        )));
                    }
                }
            }

            return Ok(());
        }

        info!(
            "Creating collection {} with dimension {}",
            self.collection, self.dimension
        );

        let vectors_config = VectorParamsBuilder::new(self.dimension as u64, Distance::Cosine);

        self.client
            .create_collection(
                CreateCollectionBuilder::new(&self.collection)
                    .vectors_config(vectors_config)
                    .quantization_config(ScalarQuantizationBuilder::default()),
            )
            .await?;

        info!("Collection {} created successfully", self.collection);
        Ok(())
    }

    /// Check if the collection exists
    pub async fn collection_exists(&self) -> Result<bool> {
        let exists = self.client.collection_exists(&self.collection).await?;
        Ok(exists)
    }

    /// Delete the collection if it exists
    pub async fn delete_collection(&self) -> Result<bool> {
        let exists = self.client.collection_exists(&self.collection).await?;

        if !exists {
            return Ok(false);
        }

        info!("Deleting collection {}", self.collection);
        self.client.delete_collection(&self.collection).await?;
        Ok(true)
    }

    /// Reset the collection (delete and recreate)
    pub async fn reset_collection(&self) -> Result<()> {
        // Delete if exists
        if self.client.collection_exists(&self.collection).await? {
            info!("Deleting existing collection {}", self.collection);
            self.client.delete_collection(&self.collection).await?;
        }

        // Recreate
        self.ensure_collection().await?;
        Ok(())
    }

    /// Get collection info (point count, etc)
    pub async fn get_collection_info(&self) -> Result<Option<CollectionInfo>> {
        if !self.client.collection_exists(&self.collection).await? {
            return Ok(None);
        }

        let info = self.client.collection_info(&self.collection).await?;
        if let Some(result) = info.result {
            Ok(Some(CollectionInfo {
                points_count: result.points_count.unwrap_or(0),
                indexed_vectors_count: result.indexed_vectors_count.unwrap_or(0),
                status: format!("{:?}", result.status()),
            }))
        } else {
            Ok(None)
        }
    }

    /// Upsert ChunkPoint objects (converts to PointStruct internally)
    pub async fn upsert_points(&self, points: Vec<ChunkPoint>) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        if let Some(mismatch) = points.iter().find(|p| p.vector.len() != self.dimension) {
            let detail = if let Some(ctx) = &self.embedding_context {
                format!(
                    "model '{}' (family '{}') expects {} from {}",
                    ctx.model_id, ctx.family, ctx.dimension, ctx.dimension_source
                )
            } else {
                format!("expected {}", self.dimension)
            };
            return Err(Error::Qdrant(format!(
                "Vector dimension mismatch for collection '{}': {} (got {})",
                self.collection,
                detail,
                mismatch.vector.len()
            )));
        }

        debug!(
            "Upserting {} points to collection {}",
            points.len(),
            self.collection
        );

        let point_structs: Vec<PointStruct> =
            points.into_iter().map(|p| p.to_point_struct()).collect();

        self.client
            .upsert_points(qdrant_client::qdrant::UpsertPointsBuilder::new(
                &self.collection,
                point_structs,
            ))
            .await?;

        Ok(())
    }

    /// Delete points by UUID
    pub async fn delete_points(&self, point_ids: &[Uuid]) -> Result<()> {
        if point_ids.is_empty() {
            return Ok(());
        }

        debug!(
            "Deleting {} points from collection {}",
            point_ids.len(),
            self.collection
        );

        let ids: Vec<PointId> = point_ids
            .iter()
            .map(|id| PointId::from(id.to_string()))
            .collect();

        self.client
            .delete_points(DeletePointsBuilder::new(&self.collection).points(ids))
            .await?;

        Ok(())
    }

    /// Search for similar vectors
    pub async fn search(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
        filter: Option<SearchFilter>,
    ) -> Result<Vec<SearchResult>> {
        debug!(
            "Searching collection {} with limit {}",
            self.collection, limit
        );

        let mut search_builder =
            SearchPointsBuilder::new(&self.collection, query_vector, limit as u64)
                .with_payload(true);

        if let Some(f) = filter {
            if let Some(qdrant_filter) = f.to_qdrant_filter() {
                search_builder = search_builder.filter(qdrant_filter);
            }
        }

        let response = self.client.search_points(search_builder).await?;

        let results: Vec<SearchResult> = response
            .result
            .into_iter()
            .map(|p| {
                let payload: ChunkPayload = p
                    .payload
                    .into_iter()
                    .map(|(k, v)| (k, json_from_qdrant_value(v)))
                    .collect::<serde_json::Map<String, Value>>()
                    .into();

                SearchResult {
                    id: point_id_to_string(p.id),
                    score: p.score,
                    payload,
                }
            })
            .collect();

        Ok(results)
    }

    /// Get collection statistics
    pub async fn get_stats(&self) -> Result<CollectionStats> {
        let info = self.client.collection_info(&self.collection).await?;

        let points_count = info
            .result
            .map(|r| r.points_count.unwrap_or(0))
            .unwrap_or(0);

        Ok(CollectionStats {
            collection: self.collection.clone(),
            points_count: points_count as usize,
        })
    }

    /// List all point IDs (for orphan detection) - scrolls through all points
    pub async fn list_all_point_ids(&self) -> Result<Vec<Uuid>> {
        use qdrant_client::qdrant::ScrollPointsBuilder;

        let mut all_ids = Vec::new();
        let mut offset: Option<PointId> = None;
        let batch_size = 1000u32;

        loop {
            let mut scroll_builder = ScrollPointsBuilder::new(&self.collection)
                .limit(batch_size)
                .with_payload(false)
                .with_vectors(false);

            if let Some(ref o) = offset {
                scroll_builder = scroll_builder.offset(o.clone());
            }

            let response = self.client.scroll(scroll_builder).await?;

            let points = response.result;
            if points.is_empty() {
                break;
            }

            for point in &points {
                if let Some(ref id) = point.id {
                    if let Some(uuid) = point_id_to_uuid(id) {
                        all_ids.push(uuid);
                    }
                }
            }

            offset = response.next_page_offset;
            if offset.is_none() {
                break;
            }
        }

        Ok(all_ids)
    }
}

fn describe_vector_sizes(sizes: &[(String, u64)]) -> String {
    sizes
        .iter()
        .map(|(name, size)| format!("{}:{}", name, size))
        .collect::<Vec<_>>()
        .join(", ")
}

fn extract_vector_sizes(info: &GetCollectionInfoResponse) -> Option<Vec<(String, u64)>> {
    let result = info.result.as_ref()?;
    let config = result.config.as_ref()?;
    let params = config.params.as_ref()?;
    let vectors_config = params.vectors_config.as_ref()?;
    let config = vectors_config.config.as_ref()?;

    match config {
        qdrant_client::qdrant::vectors_config::Config::Params(params) => {
            Some(vec![("default".to_string(), params.size as u64)])
        }
        qdrant_client::qdrant::vectors_config::Config::ParamsMap(map) => {
            let mut sizes = Vec::new();
            for (name, params) in &map.map {
                sizes.push((name.clone(), params.size as u64));
            }
            if sizes.is_empty() {
                None
            } else {
                Some(sizes)
            }
        }
    }
}

impl QdrantStore {
    async fn collection_vector_sizes(&self) -> Result<Option<Vec<(String, u64)>>> {
        let info = self.client.collection_info(&self.collection).await?;
        Ok(extract_vector_sizes(&info))
    }
}

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub payload: ChunkPayload,
}

/// Search filter options
#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    pub source_ids: Option<Vec<String>>,
    pub source_types: Option<Vec<String>>,
    pub path_prefix: Option<String>,
}

impl SearchFilter {
    fn to_qdrant_filter(&self) -> Option<Filter> {
        use qdrant_client::qdrant::Condition;

        let mut must_conditions: Vec<Condition> = Vec::new();

        if let Some(ref source_ids) = self.source_ids {
            if source_ids.len() == 1 {
                must_conditions.push(Condition::matches("source_id", source_ids[0].clone()));
            }
        }

        if let Some(ref source_types) = self.source_types {
            if source_types.len() == 1 {
                must_conditions.push(Condition::matches("source_type", source_types[0].clone()));
            }
        }

        if must_conditions.is_empty() {
            return None;
        }

        Some(Filter {
            must: must_conditions,
            should: vec![],
            must_not: vec![],
            min_should: None,
        })
    }
}

/// Collection statistics
#[derive(Debug, Clone)]
pub struct CollectionStats {
    pub collection: String,
    pub points_count: usize,
}

/// Convert PointId to string
fn point_id_to_string(id: Option<PointId>) -> String {
    match id {
        Some(PointId {
            point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)),
        }) => uuid,
        Some(PointId {
            point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)),
        }) => num.to_string(),
        _ => String::new(),
    }
}

/// Convert PointId to UUID
fn point_id_to_uuid(id: &PointId) -> Option<Uuid> {
    match &id.point_id_options {
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid_str)) => {
            Uuid::try_parse(uuid_str).ok()
        }
        _ => None,
    }
}

/// Convert Qdrant value to serde_json Value
fn json_from_qdrant_value(v: qdrant_client::qdrant::Value) -> Value {
    use qdrant_client::qdrant::value::Kind;

    match v.kind {
        Some(Kind::NullValue(_)) => Value::Null,
        Some(Kind::BoolValue(b)) => Value::Bool(b),
        Some(Kind::IntegerValue(i)) => Value::Number(i.into()),
        Some(Kind::DoubleValue(d)) => serde_json::Number::from_f64(d)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Some(Kind::StringValue(s)) => Value::String(s),
        Some(Kind::ListValue(list)) => Value::Array(
            list.values
                .into_iter()
                .map(json_from_qdrant_value)
                .collect(),
        ),
        Some(Kind::StructValue(s)) => Value::Object(
            s.fields
                .into_iter()
                .map(|(k, v)| (k, json_from_qdrant_value(v)))
                .collect(),
        ),
        None => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use uuid::Uuid;

    #[test]
    fn test_search_filter_to_qdrant() {
        let filter = SearchFilter {
            source_ids: Some(vec!["test-source".to_string()]),
            source_types: Some(vec!["dir".to_string()]),
            path_prefix: None,
        };

        let qdrant_filter = filter.to_qdrant_filter();
        assert!(qdrant_filter.is_some());
        assert_eq!(qdrant_filter.unwrap().must.len(), 2);
    }

    #[tokio::test]
    async fn test_upsert_points_rejects_dimension_mismatch() {
        let store = QdrantStore::new("http://127.0.0.1:6334", "test_collection", 3, None)
            .await
            .expect("store should initialize");

        let payload = ChunkPayload::new(
            "source-123".to_string(),
            "dir".to_string(),
            "/docs".to_string(),
            "doc-456".to_string(),
            "/docs/readme.md".to_string(),
            0,
            "hash123".to_string(),
            "2024-01-01T00:00:00Z".to_string(),
        );

        let point = ChunkPoint {
            id: Uuid::new_v4(),
            vector: vec![0.1, 0.2],
            payload,
        };

        let err = store
            .upsert_points(vec![point])
            .await
            .expect_err("should reject mismatched vector length");

        match err {
            Error::Qdrant(message) => assert!(message.contains("Vector dimension mismatch")),
            other => panic!("expected qdrant error, got {other:?}"),
        }
    }
}
