//! Model capability registry for multimodal embeddings and rerankers.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultimodalStrategy {
    /// Vision-language embedding model that jointly encodes text + image inputs.
    VlEmbedding,
    /// Dual-encoder model that produces separate text/image embeddings.
    DualEncoder,
    /// Late-interaction model that emits multi-vector representations.
    LateInteraction,
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddingModelCapabilities {
    pub strategy: MultimodalStrategy,
    pub supports_text: bool,
    pub supports_image: bool,
    pub supports_joint_inputs: bool,
    pub supports_multi_vector: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct RerankerModelCapabilities {
    pub supports_text: bool,
    pub supports_image: bool,
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

const EMBED_MODELS: &[EmbeddingModelSpec] = &[
    EmbeddingModelSpec {
        id: "BAAI/bge-small-en-v1.5",
        family: "bge",
        default_dimension: Some(384),
        modalities: &["text"],
        capabilities: EmbeddingModelCapabilities {
            strategy: MultimodalStrategy::DualEncoder,
            supports_text: true,
            supports_image: false,
            supports_joint_inputs: false,
            supports_multi_vector: false,
        },
        supports_mrl: false,
        max_batch: 32,
    },
    EmbeddingModelSpec {
        id: "BAAI/bge-base-en-v1.5",
        family: "bge",
        default_dimension: Some(768),
        modalities: &["text"],
        capabilities: EmbeddingModelCapabilities {
            strategy: MultimodalStrategy::DualEncoder,
            supports_text: true,
            supports_image: false,
            supports_joint_inputs: false,
            supports_multi_vector: false,
        },
        supports_mrl: false,
        max_batch: 32,
    },
    EmbeddingModelSpec {
        id: "BAAI/bge-large-en-v1.5",
        family: "bge",
        default_dimension: Some(1024),
        modalities: &["text"],
        capabilities: EmbeddingModelCapabilities {
            strategy: MultimodalStrategy::DualEncoder,
            supports_text: true,
            supports_image: false,
            supports_joint_inputs: false,
            supports_multi_vector: false,
        },
        supports_mrl: false,
        max_batch: 16,
    },
    EmbeddingModelSpec {
        id: "sentence-transformers/all-MiniLM-L6-v2",
        family: "minilm",
        default_dimension: Some(384),
        modalities: &["text"],
        capabilities: EmbeddingModelCapabilities {
            strategy: MultimodalStrategy::DualEncoder,
            supports_text: true,
            supports_image: false,
            supports_joint_inputs: false,
            supports_multi_vector: false,
        },
        supports_mrl: false,
        max_batch: 32,
    },
    EmbeddingModelSpec {
        id: "Qwen/Qwen3-VL-Embedding-2B",
        family: "qwen3-vl",
        default_dimension: None,
        modalities: &["text", "image_text"],
        capabilities: EmbeddingModelCapabilities {
            strategy: MultimodalStrategy::VlEmbedding,
            supports_text: true,
            supports_image: true,
            supports_joint_inputs: true,
            supports_multi_vector: false,
        },
        supports_mrl: false,
        max_batch: 8,
    },
    EmbeddingModelSpec {
        id: "Qwen/Qwen3-VL-Embedding-8B",
        family: "qwen3-vl",
        default_dimension: None,
        modalities: &["text", "image_text"],
        capabilities: EmbeddingModelCapabilities {
            strategy: MultimodalStrategy::VlEmbedding,
            supports_text: true,
            supports_image: true,
            supports_joint_inputs: true,
            supports_multi_vector: false,
        },
        supports_mrl: false,
        max_batch: 4,
    },
    EmbeddingModelSpec {
        id: "jinaai/jina-clip-v2",
        family: "jina-clip",
        default_dimension: None,
        modalities: &["text", "image"],
        capabilities: EmbeddingModelCapabilities {
            strategy: MultimodalStrategy::DualEncoder,
            supports_text: true,
            supports_image: true,
            supports_joint_inputs: false,
            supports_multi_vector: false,
        },
        supports_mrl: false,
        max_batch: 16,
    },
    EmbeddingModelSpec {
        id: "google/siglip2-base-patch16-224",
        family: "siglip2",
        default_dimension: None,
        modalities: &["text", "image"],
        capabilities: EmbeddingModelCapabilities {
            strategy: MultimodalStrategy::DualEncoder,
            supports_text: true,
            supports_image: true,
            supports_joint_inputs: false,
            supports_multi_vector: false,
        },
        supports_mrl: false,
        max_batch: 16,
    },
    EmbeddingModelSpec {
        id: "vidore/colpali",
        family: "colpali",
        default_dimension: None,
        modalities: &["text", "image"],
        capabilities: EmbeddingModelCapabilities {
            strategy: MultimodalStrategy::LateInteraction,
            supports_text: true,
            supports_image: true,
            supports_joint_inputs: false,
            supports_multi_vector: true,
        },
        supports_mrl: false,
        max_batch: 8,
    },
];

const RERANK_MODELS: &[RerankerModelSpec] = &[
    RerankerModelSpec {
        id: "BAAI/bge-reranker-base",
        family: "bge",
        modalities: &["text"],
        capabilities: RerankerModelCapabilities {
            supports_text: true,
            supports_image: false,
            supports_joint_inputs: false,
        },
        max_batch: 32,
    },
    RerankerModelSpec {
        id: "Qwen/Qwen3-VL-Reranker-2B",
        family: "qwen3-vl",
        modalities: &["text", "image_text"],
        capabilities: RerankerModelCapabilities {
            supports_text: true,
            supports_image: true,
            supports_joint_inputs: true,
        },
        max_batch: 8,
    },
    RerankerModelSpec {
        id: "Qwen/Qwen3-VL-Reranker-8B",
        family: "qwen3-vl",
        modalities: &["text", "image_text"],
        capabilities: RerankerModelCapabilities {
            supports_text: true,
            supports_image: true,
            supports_joint_inputs: true,
        },
        max_batch: 4,
    },
    RerankerModelSpec {
        id: "jinaai/jina-reranker-m0",
        family: "jina",
        modalities: &["text", "image_text"],
        capabilities: RerankerModelCapabilities {
            supports_text: true,
            supports_image: true,
            supports_joint_inputs: true,
        },
        max_batch: 16,
    },
    RerankerModelSpec {
        id: "lightonai/MonoQwen2-VL-v0.1",
        family: "monoqwen2",
        modalities: &["text", "image_text"],
        capabilities: RerankerModelCapabilities {
            supports_text: true,
            supports_image: true,
            supports_joint_inputs: true,
        },
        max_batch: 8,
    },
];

pub fn embedding_model_spec(model: &str) -> Option<&'static EmbeddingModelSpec> {
    EMBED_MODELS.iter().find(|spec| spec.id == model)
}

pub fn reranker_model_spec(model: &str) -> Option<&'static RerankerModelSpec> {
    RERANK_MODELS.iter().find(|spec| spec.id == model)
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
    EMBED_MODELS
        .iter()
        .filter(|spec| spec.capabilities.supports_image)
        .map(|spec| spec.id)
        .collect()
}

pub fn supported_multimodal_reranker_models() -> Vec<&'static str> {
    RERANK_MODELS
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
    EMBED_MODELS.iter().map(|spec| spec.id).collect()
}

pub fn allowlisted_reranker_models() -> Vec<&'static str> {
    RERANK_MODELS.iter().map(|spec| spec.id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_capabilities() {
        let caps = embedding_model_capabilities("Qwen/Qwen3-VL-Embedding-2B").unwrap();
        assert_eq!(caps.strategy, MultimodalStrategy::VlEmbedding);
        assert!(caps.supports_joint_inputs);

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
}
