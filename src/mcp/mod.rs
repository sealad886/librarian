//! MCP (Model Context Protocol) server implementation
//!
//! Exposes RAG functionality over stdio for VS Code integration.

mod server;
mod tools;
mod types;

pub use server::McpServer;
pub use types::{McpError, McpRequest, McpResponse};
