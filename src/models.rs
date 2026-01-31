//! Model capability registry for multimodal embeddings and rerankers.

use crate::xinference::{registry_snapshot, RegistryEntry, RegistryType};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultimodalStrategy {
    /// Vision-language embedding model that jointly encodes text + image inputs.
    VlEmbedding,
    /// Dual-encoder model that produces separate text/image embeddings.
    DualEncoder,
    /// Late-interaction model that emits multi-vector representations.
    LateInteraction,
    /// Audio-language model that jointly encodes text + audio inputs.
    AudioLanguage,
    /// Video-language model that jointly encodes text + video inputs.
    VideoLanguage,
    /// Omni-modal model supporting multiple modalities (text, image, audio, video).
    OmniModal,
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddingModelCapabilities {
    pub strategy: MultimodalStrategy,
    pub supports_text: bool,
    pub supports_image: bool,
    pub supports_audio: bool,
    pub supports_video: bool,
    pub supports_joint_inputs: bool,
    pub supports_multi_vector: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct RerankerModelCapabilities {
    pub supports_text: bool,
    pub supports_image: bool,
    pub supports_audio: bool,
    pub supports_video: bool,
    pub supports_joint_inputs: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddingModelSpec {
    pub id: &'static str,
    pub family: &'static str,
    pub default_dimension: Option<usize>,
    pub modalities: &'static [&'static str],
    pub capabilities: EmbeddingModelCapabilities,
    pub supports_mrl: bool,
    pub max_batch: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct RerankerModelSpec {
    pub id: &'static str,
    pub family: &'static str,
    pub modalities: &'static [&'static str],
    pub capabilities: RerankerModelCapabilities,
    pub max_batch: usize,
}

struct ModelRegistry {
    embedding: Vec<EmbeddingModelSpec>,
    reranker: Vec<RerankerModelSpec>,
    embedding_index: HashMap<&'static str, usize>,
    reranker_index: HashMap<&'static str, usize>,
}

static MODEL_REGISTRY: OnceLock<ModelRegistry> = OnceLock::new();

fn model_registry() -> &'static ModelRegistry {
    MODEL_REGISTRY.get_or_init(build_registry)
}

fn build_registry() -> ModelRegistry {
    let mut embedding = Vec::new();
    let mut reranker = Vec::new();
    let mut embedding_index = HashMap::new();
    let mut reranker_index = HashMap::new();

    let snapshot = registry_snapshot(RegistryType::Embedding)
        .expect("Xinference embedding registry snapshot is missing or invalid");
    for entry in &snapshot.registrations {
        let spec = embedding_spec_from_entry(entry);
        let index = embedding.len();
        embedding_index.insert(spec.id, index);
        if spec.id != entry.model_name.as_str() {
            embedding_index.insert(leak_str(entry.model_name.clone()), index);
        }
        for alias in &entry.aliases {
            embedding_index.insert(leak_str(alias.clone()), index);
        }
        embedding.push(spec);
    }

    let snapshot = registry_snapshot(RegistryType::Rerank)
        .expect("Xinference reranker registry snapshot is missing or invalid");
    for entry in &snapshot.registrations {
        let spec = reranker_spec_from_entry(entry);
        let index = reranker.len();
        reranker_index.insert(spec.id, index);
        if spec.id != entry.model_name.as_str() {
            reranker_index.insert(leak_str(entry.model_name.clone()), index);
        }
        for alias in &entry.aliases {
            reranker_index.insert(leak_str(alias.clone()), index);
        }
        reranker.push(spec);
    }

    ModelRegistry {
        embedding,
        reranker,
        embedding_index,
        reranker_index,
    }
}

fn leak_str(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn leak_str_list(values: Vec<String>) -> &'static [&'static str] {
    let leaked = values
        .into_iter()
        .map(leak_str)
        .collect::<Vec<&'static str>>();
    Box::leak(leaked.into_boxed_slice())
}

fn embedding_spec_from_entry(entry: &RegistryEntry) -> EmbeddingModelSpec {
    let modalities = if entry.modalities.is_empty() {
        vec!["text".to_string()]
    } else {
        entry.modalities.clone()
    };
    let modalities_ref = leak_str_list(modalities.clone());
    let supports_multi_vector = entry.supports_multi_vector.unwrap_or(false);
    let mut capabilities = capabilities_from_modalities(modalities_ref, supports_multi_vector);
    apply_strategy_override(&mut capabilities, entry);

    EmbeddingModelSpec {
        id: leak_str(entry.model_id.clone()),
        family: leak_str(
            entry
                .model_family
                .clone()
                .unwrap_or_else(|| "xinference".to_string()),
        ),
        default_dimension: entry.dimension,
        modalities: modalities_ref,
        capabilities,
        supports_mrl: entry.supports_mrl.unwrap_or(false),
        max_batch: entry.max_batch.unwrap_or(32),
    }
}

fn reranker_spec_from_entry(entry: &RegistryEntry) -> RerankerModelSpec {
    let modalities = if entry.modalities.is_empty() {
        vec!["text".to_string()]
    } else {
        entry.modalities.clone()
    };
    let modalities_ref = leak_str_list(modalities.clone());
    let capabilities = reranker_caps_from_modalities(modalities_ref);

    RerankerModelSpec {
        id: leak_str(entry.model_id.clone()),
        family: leak_str(
            entry
                .model_family
                .clone()
                .unwrap_or_else(|| "xinference".to_string()),
        ),
        modalities: modalities_ref,
        capabilities,
        max_batch: entry.max_batch.unwrap_or(32),
    }
}

fn apply_strategy_override(
    capabilities: &mut EmbeddingModelCapabilities,
    entry: &RegistryEntry,
) {
    let Some(Value::String(strategy)) = entry.metadata.get("strategy") else {
        return;
    };

    match strategy.as_str() {
        "vl_embedding" => capabilities.strategy = MultimodalStrategy::VlEmbedding,
        "dual_encoder" => capabilities.strategy = MultimodalStrategy::DualEncoder,
        "late_interaction" => {
            capabilities.strategy = MultimodalStrategy::LateInteraction;
            capabilities.supports_multi_vector = true;
        }
        "audio_language" => capabilities.strategy = MultimodalStrategy::AudioLanguage,
        "video_language" => capabilities.strategy = MultimodalStrategy::VideoLanguage,
        "omni_modal" => capabilities.strategy = MultimodalStrategy::OmniModal,
        _ => {}
    }
}

fn capabilities_from_modalities(
    modalities: &[&'static str],
    supports_multi_vector: bool,
) -> EmbeddingModelCapabilities {
    let supports_joint_inputs = modalities.contains(&"multimode");
    let supports_image = supports_joint_inputs || modalities.contains(&"image");
    let supports_text = supports_joint_inputs || modalities.contains(&"text");
    let supports_audio = modalities.contains(&"audio");
    let supports_video = modalities.contains(&"video");

    let strategy = if supports_multi_vector {
        MultimodalStrategy::LateInteraction
    } else if supports_joint_inputs {
        MultimodalStrategy::VlEmbedding
    } else {
        MultimodalStrategy::DualEncoder
    };

    EmbeddingModelCapabilities {
        strategy,
        supports_text,
        supports_image,
        supports_audio,
        supports_video,
        supports_joint_inputs,
        supports_multi_vector,
    }
}

fn reranker_caps_from_modalities(modalities: &[&'static str]) -> RerankerModelCapabilities {
    let supports_joint_inputs = modalities.contains(&"multimode");
    let supports_image = supports_joint_inputs || modalities.contains(&"image");
    let supports_text = supports_joint_inputs || modalities.contains(&"text");
    let supports_audio = modalities.contains(&"audio");
    let supports_video = modalities.contains(&"video");

    RerankerModelCapabilities {
        supports_text,
        supports_image,
        supports_audio,
        supports_video,
        supports_joint_inputs,
    }
}

pub fn embedding_model_spec(model: &str) -> Option<&'static EmbeddingModelSpec> {
    let registry = model_registry();
    registry
        .embedding_index
        .get(model)
        .and_then(|idx| registry.embedding.get(*idx))
}

pub fn reranker_model_spec(model: &str) -> Option<&'static RerankerModelSpec> {
    let registry = model_registry();
    registry
        .reranker_index
        .get(model)
        .and_then(|idx| registry.reranker.get(*idx))
}

pub fn embedding_model_capabilities(model: &str) -> Option<EmbeddingModelCapabilities> {
    embedding_model_spec(model).map(|spec| spec.capabilities)
}

pub fn reranker_model_capabilities(model: &str) -> Option<RerankerModelCapabilities> {
    reranker_model_spec(model).map(|spec| spec.capabilities)
}

pub fn multimodal_strategy_for_embedding(model: &str) -> Option<MultimodalStrategy> {
    embedding_model_capabilities(model).map(|caps| caps.strategy)
}

pub fn supported_multimodal_embedding_models() -> Vec<&'static str> {
    model_registry()
        .embedding
        .iter()
        .filter(|spec| spec.capabilities.supports_image)
        .map(|spec| spec.id)
        .collect()
}

pub fn supported_multimodal_reranker_models() -> Vec<&'static str> {
    model_registry()
        .reranker
        .iter()
        .filter(|spec| spec.capabilities.supports_image)
        .map(|spec| spec.id)
        .collect()
}

pub fn is_multimodal_embedding_model(model: &str) -> bool {
    embedding_model_capabilities(model)
        .map(|caps| caps.supports_image)
        .unwrap_or(false)
}

pub fn is_multimodal_reranker_model(model: &str) -> bool {
    reranker_model_capabilities(model)
        .map(|caps| caps.supports_image)
        .unwrap_or(false)
}

pub fn allowlisted_embedding_models() -> Vec<&'static str> {
    model_registry()
        .embedding
        .iter()
        .map(|spec| spec.id)
        .collect()
}

pub fn allowlisted_reranker_models() -> Vec<&'static str> {
    model_registry()
        .reranker
        .iter()
        .map(|spec| spec.id)
        .collect()
}

/// Get all embedding models that support audio
pub fn supported_audio_embedding_models() -> Vec<&'static str> {
    model_registry()
        .embedding
        .iter()
        .filter(|spec| spec.capabilities.supports_audio)
        .map(|spec| spec.id)
        .collect()
}

/// Get all embedding models that support video
pub fn supported_video_embedding_models() -> Vec<&'static str> {
    model_registry()
        .embedding
        .iter()
        .filter(|spec| spec.capabilities.supports_video)
        .map(|spec| spec.id)
        .collect()
}

/// Check if an embedding model supports audio
pub fn is_audio_embedding_model(model: &str) -> bool {
    embedding_model_capabilities(model)
        .map(|caps| caps.supports_audio)
        .unwrap_or(false)
}

/// Check if an embedding model supports video
pub fn is_video_embedding_model(model: &str) -> bool {
    embedding_model_capabilities(model)
        .map(|caps| caps.supports_video)
        .unwrap_or(false)
}

/// Check if a reranker model supports audio
pub fn is_audio_reranker_model(model: &str) -> bool {
    reranker_model_capabilities(model)
        .map(|caps| caps.supports_audio)
        .unwrap_or(false)
}

/// Check if a reranker model supports video
pub fn is_video_reranker_model(model: &str) -> bool {
    reranker_model_capabilities(model)
        .map(|caps| caps.supports_video)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_capabilities() {
        let caps = embedding_model_capabilities("Qwen/Qwen3-VL-Embedding-2B").unwrap();
        assert_eq!(caps.strategy, MultimodalStrategy::VlEmbedding);
        assert!(!caps.supports_joint_inputs); // Joint inputs not supported by this model
        assert!(caps.supports_image);
        assert!(!caps.supports_video); // Video embeddings not supported by this model

        let caps = embedding_model_capabilities("jinaai/jina-clip-v2").unwrap();
        assert_eq!(caps.strategy, MultimodalStrategy::DualEncoder);
        assert!(!caps.supports_joint_inputs);

        let caps = embedding_model_capabilities("vidore/colpali").unwrap();
        assert_eq!(caps.strategy, MultimodalStrategy::LateInteraction);
        assert!(caps.supports_multi_vector);
    }

    #[test]
    fn test_reranker_capabilities() {
        assert!(reranker_model_capabilities("jinaai/jina-reranker-m0").is_some());
        assert!(reranker_model_capabilities("unknown-reranker").is_none());
    }

    #[test]
    fn test_audio_video_model_detection() {
        assert!(!is_audio_embedding_model("BAAI/bge-small-en-v1.5"));
        assert!(!is_video_embedding_model("BAAI/bge-small-en-v1.5"));
    }

    #[test]
    fn test_allowlisted_models_present() {
        let embed_models = allowlisted_embedding_models();
        assert!(embed_models.contains(&"BAAI/bge-small-en-v1.5"));

        let rerank_models = allowlisted_reranker_models();
        assert!(rerank_models.contains(&"BAAI/bge-reranker-base"));
    }
}
