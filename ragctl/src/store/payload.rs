//! Payload schema for Qdrant points

use qdrant_client::qdrant::{PointStruct, Value as QdrantValue};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use uuid::Uuid;

/// A point ready to be upserted to Qdrant
#[derive(Debug, Clone)]
pub struct ChunkPoint {
    pub id: Uuid,
    pub vector: Vec<f32>,
    pub payload: ChunkPayload,
}

impl ChunkPoint {
    /// Convert to qdrant-client PointStruct
    pub fn to_point_struct(self) -> PointStruct {
        let payload_map = self.payload.to_qdrant_payload();
        PointStruct::new(self.id.to_string(), self.vector, payload_map)
    }
}

/// Payload stored with each chunk in Qdrant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkPayload {
    /// Source ID (stable UUID)
    pub source_id: String,

    /// Source type ("dir", "url", "sitemap")
    pub source_type: String,

    /// Source URI (directory path or base URL)
    pub source_uri: String,

    /// Document ID (stable per file/page)
    pub doc_id: String,

    /// Document URI (file path or URL)
    pub doc_uri: String,

    /// Document title (if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Headings hierarchy above this chunk
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headings: Option<Vec<String>>,

    /// Chunk index within the document
    pub chunk_index: i32,

    /// Hash of the chunk content
    pub chunk_hash: String,

    /// When this chunk was last updated
    pub updated_at: String,
}

impl ChunkPayload {
    pub fn new(
        source_id: String,
        source_type: String,
        source_uri: String,
        doc_id: String,
        doc_uri: String,
        chunk_index: i32,
        chunk_hash: String,
        updated_at: String,
    ) -> Self {
        Self {
            source_id,
            source_type,
            source_uri,
            doc_id,
            doc_uri,
            title: None,
            headings: None,
            chunk_index,
            chunk_hash,
            updated_at,
        }
    }

    /// Convert to Qdrant payload format
    pub fn to_qdrant_payload(self) -> HashMap<String, QdrantValue> {
        let mut map = HashMap::new();

        map.insert("source_id".to_string(), string_to_qdrant(&self.source_id));
        map.insert("source_type".to_string(), string_to_qdrant(&self.source_type));
        map.insert("source_uri".to_string(), string_to_qdrant(&self.source_uri));
        map.insert("doc_id".to_string(), string_to_qdrant(&self.doc_id));
        map.insert("doc_uri".to_string(), string_to_qdrant(&self.doc_uri));
        map.insert("chunk_index".to_string(), int_to_qdrant(self.chunk_index as i64));
        map.insert("chunk_hash".to_string(), string_to_qdrant(&self.chunk_hash));
        map.insert("updated_at".to_string(), string_to_qdrant(&self.updated_at));

        if let Some(ref title) = self.title {
            map.insert("title".to_string(), string_to_qdrant(title));
        }

        if let Some(ref headings) = self.headings {
            let values: Vec<QdrantValue> = headings.iter().map(|s| string_to_qdrant(s)).collect();
            map.insert(
                "headings".to_string(),
                QdrantValue {
                    kind: Some(qdrant_client::qdrant::value::Kind::ListValue(
                        qdrant_client::qdrant::ListValue { values },
                    )),
                },
            );
        }

        map
    }
}

fn string_to_qdrant(s: &str) -> QdrantValue {
    QdrantValue {
        kind: Some(qdrant_client::qdrant::value::Kind::StringValue(s.to_string())),
    }
}

fn int_to_qdrant(i: i64) -> QdrantValue {
    QdrantValue {
        kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)),
    }
}

impl From<Map<String, Value>> for ChunkPayload {
    fn from(map: Map<String, Value>) -> Self {
        serde_json::from_value(Value::Object(map)).unwrap_or_else(|_| ChunkPayload {
            source_id: String::new(),
            source_type: String::new(),
            source_uri: String::new(),
            doc_id: String::new(),
            doc_uri: String::new(),
            title: None,
            headings: None,
            chunk_index: 0,
            chunk_hash: String::new(),
            updated_at: String::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_serialization() {
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

        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("source_id"));
        assert!(json.contains("source-123"));

        let parsed: ChunkPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.source_id, "source-123");
    }
}
