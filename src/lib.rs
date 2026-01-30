//! librarian - A CLI tool for local RAG (Retrieval-Augmented Generation)
//!
//! This crate provides:
//! - CLI commands for ingesting documentation (local directories + web URLs)
//! - An MCP server over stdio for VS Code integration
//! - Integration with Qdrant vector database for semantic search

pub mod chunk;
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
