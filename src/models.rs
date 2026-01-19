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

const QWEN3_VL_EMBEDDING_2B: &str = "Qwen/Qwen3-VL-Embedding-2B";
const QWEN3_VL_EMBEDDING_8B: &str = "Qwen/Qwen3-VL-Embedding-8B";
const QWEN3_VL_RERANKER_2B: &str = "Qwen/Qwen3-VL-Reranker-2B";
const QWEN3_VL_RERANKER_8B: &str = "Qwen/Qwen3-VL-Reranker-8B";
const JINA_CLIP_V2: &str = "jinaai/jina-clip-v2";
const JINA_RERANKER_M0: &str = "jinaai/jina-reranker-m0";
const COLPALI: &str = "vidore/colpali";
const MONOQWEN2_VL: &str = "lightonai/MonoQwen2-VL-v0.1";

fn is_siglip2(model: &str) -> bool {
    model.starts_with("google/siglip2-")
}

pub fn embedding_model_capabilities(model: &str) -> Option<EmbeddingModelCapabilities> {
    match model {
        QWEN3_VL_EMBEDDING_2B | QWEN3_VL_EMBEDDING_8B => Some(EmbeddingModelCapabilities {
            strategy: MultimodalStrategy::VlEmbedding,
            supports_text: true,
            supports_image: true,
            supports_joint_inputs: true,
            supports_multi_vector: false,
        }),
        JINA_CLIP_V2 => Some(EmbeddingModelCapabilities {
            strategy: MultimodalStrategy::DualEncoder,
            supports_text: true,
            supports_image: true,
            supports_joint_inputs: false,
            supports_multi_vector: false,
        }),
        COLPALI => Some(EmbeddingModelCapabilities {
            strategy: MultimodalStrategy::LateInteraction,
            supports_text: true,
            supports_image: true,
            supports_joint_inputs: false,
            supports_multi_vector: true,
        }),
        _ if is_siglip2(model) => Some(EmbeddingModelCapabilities {
            strategy: MultimodalStrategy::DualEncoder,
            supports_text: true,
            supports_image: true,
            supports_joint_inputs: false,
            supports_multi_vector: false,
        }),
        _ => None,
    }
}

pub fn reranker_model_capabilities(model: &str) -> Option<RerankerModelCapabilities> {
    match model {
        QWEN3_VL_RERANKER_2B | QWEN3_VL_RERANKER_8B | JINA_RERANKER_M0 | MONOQWEN2_VL => {
            Some(RerankerModelCapabilities {
                supports_text: true,
                supports_image: true,
                supports_joint_inputs: true,
            })
        }
        _ => None,
    }
}

pub fn multimodal_strategy_for_embedding(model: &str) -> Option<MultimodalStrategy> {
    embedding_model_capabilities(model).map(|caps| caps.strategy)
}

pub fn supported_multimodal_embedding_models() -> Vec<&'static str> {
    vec![
        QWEN3_VL_EMBEDDING_2B,
        QWEN3_VL_EMBEDDING_8B,
        JINA_CLIP_V2,
        COLPALI,
        "google/siglip2-*",
    ]
}

pub fn supported_multimodal_reranker_models() -> Vec<&'static str> {
    vec![
        QWEN3_VL_RERANKER_2B,
        QWEN3_VL_RERANKER_8B,
        JINA_RERANKER_M0,
        MONOQWEN2_VL,
    ]
}

pub fn is_multimodal_embedding_model(model: &str) -> bool {
    embedding_model_capabilities(model).is_some()
}

pub fn is_multimodal_reranker_model(model: &str) -> bool {
    reranker_model_capabilities(model).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_capabilities() {
        let caps = embedding_model_capabilities(QWEN3_VL_EMBEDDING_2B).unwrap();
        assert_eq!(caps.strategy, MultimodalStrategy::VlEmbedding);
        assert!(caps.supports_joint_inputs);

        let caps = embedding_model_capabilities(JINA_CLIP_V2).unwrap();
        assert_eq!(caps.strategy, MultimodalStrategy::DualEncoder);
        assert!(!caps.supports_joint_inputs);

        let caps = embedding_model_capabilities(COLPALI).unwrap();
        assert_eq!(caps.strategy, MultimodalStrategy::LateInteraction);
        assert!(caps.supports_multi_vector);
    }

    #[test]
    fn test_siglip2_prefix() {
        let caps = embedding_model_capabilities("google/siglip2-base-patch16-224").unwrap();
        assert_eq!(caps.strategy, MultimodalStrategy::DualEncoder);
    }

    #[test]
    fn test_reranker_capabilities() {
        assert!(reranker_model_capabilities(JINA_RERANKER_M0).is_some());
        assert!(reranker_model_capabilities("unknown-reranker").is_none());
    }
}
