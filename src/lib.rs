//! librarian — high-performance local RAG for documentation comprehension
//!
//! This crate provides:
//! - CLI commands for ingesting documentation (local directories + web URLs + sitemaps)
//! - An MCP server over stdio for VS Code / editor integration
//! - Hybrid BM25 + vector similarity search with optional cross-encoder reranking
//! - Multimodal indexing (text, images, audio, video) with capability-gated configuration
//! - Incremental updates via content hashing and canonical document IDs
//! - Integration with Qdrant vector database and Xinference embedding backends

pub mod chunk;
pub mod classify;
pub mod commands;
pub mod config;
pub mod crawl;
pub mod embed;
pub mod embedding_backend;
pub mod error;
pub mod mcp;
pub mod meta;
pub mod models;
pub mod parse;
pub mod progress;
pub mod rank;
pub mod rerank;
pub mod store;
pub mod xinference;

pub use config::Config;
pub use error::{Error, Result};
