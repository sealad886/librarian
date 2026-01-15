//! ragctl - A CLI tool for local RAG (Retrieval-Augmented Generation)
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
pub mod error;
pub mod mcp;
pub mod meta;
pub mod parse;
pub mod rank;
pub mod store;

pub use config::Config;
pub use error::{Error, Result};
