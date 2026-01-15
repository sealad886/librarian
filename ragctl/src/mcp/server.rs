//! MCP stdio server implementation

use super::tools::{get_tool_definitions, handle_tool_call};
use super::types::{McpError, McpMessage, McpNotification, McpRequest, McpResponse};
use crate::config::Config;
use crate::meta::MetaDb;
use crate::store::QdrantStore;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use tracing::{debug, error, info, warn};

/// MCP Server implementation
pub struct McpServer {
    config: Config,
    db: MetaDb,
    store: QdrantStore,
}

impl McpServer {
    /// Create a new MCP server
    pub fn new(config: Config, db: MetaDb, store: QdrantStore) -> Self {
        Self { config, db, store }
    }

    /// Run the MCP server loop over stdio
    pub async fn run(&self) -> Result<(), McpError> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        info!("MCP server starting on stdio");

        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    error!("Failed to read line: {}", e);
                    continue;
                }
            };

            if line.is_empty() {
                continue;
            }

            debug!("Received: {}", line);

            // Parse the message
            let message: McpMessage = match serde_json::from_str(&line) {
                Ok(m) => m,
                Err(e) => {
                    error!("Failed to parse message: {}", e);
                    let error_response = json!({
                        "jsonrpc": "2.0",
                        "id": null,
                        "error": {
                            "code": -32700,
                            "message": format!("Parse error: {}", e)
                        }
                    });
                    writeln!(stdout, "{}", error_response)?;
                    stdout.flush()?;
                    continue;
                }
            };

            // Handle the message
            match message {
                McpMessage::Request(req) => {
                    let response = self.handle_request(req).await;
                    let response_str = serde_json::to_string(&response)?;
                    debug!("Sending: {}", response_str);
                    writeln!(stdout, "{}", response_str)?;
                    stdout.flush()?;
                }
                McpMessage::Notification(notif) => {
                    self.handle_notification(notif).await;
                }
                McpMessage::Response(_) => {
                    warn!("Unexpected response message received");
                }
            }
        }

        info!("MCP server shutting down");
        Ok(())
    }

    /// Handle an MCP request
    async fn handle_request(&self, request: McpRequest) -> McpResponse {
        let id = request.id.clone();

        match request.method.as_str() {
            "initialize" => self.handle_initialize(id, request.params),
            "tools/list" => self.handle_tools_list(id).await,
            "tools/call" => self.handle_tools_call(id, request.params).await,
            "resources/list" => self.handle_resources_list(id).await,
            "prompts/list" => self.handle_prompts_list(id).await,
            _ => McpResponse::error_with_code(
                id,
                -32601,
                format!("Method not found: {}", request.method),
            ),
        }
    }

    /// Handle notifications (fire-and-forget)
    async fn handle_notification(&self, notification: McpNotification) {
        match notification.method.as_str() {
            "notifications/initialized" => {
                info!("Client initialized");
            }
            "notifications/cancelled" => {
                info!("Request cancelled");
            }
            _ => {
                debug!("Unknown notification: {}", notification.method);
            }
        }
    }

    /// Handle initialize request
    fn handle_initialize(&self, id: Option<Value>, _params: Option<Value>) -> McpResponse {
        McpResponse::success(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": false
                    },
                    "resources": {
                        "subscribe": false,
                        "listChanged": false
                    },
                    "prompts": {
                        "listChanged": false
                    }
                },
                "serverInfo": {
                    "name": "ragctl",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        )
    }

    /// Handle tools/list request
    async fn handle_tools_list(&self, id: Option<Value>) -> McpResponse {
        let tools = get_tool_definitions();
        McpResponse::success(id, json!({ "tools": tools }))
    }

    /// Handle tools/call request
    async fn handle_tools_call(&self, id: Option<Value>, params: Option<Value>) -> McpResponse {
        let params = match params {
            Some(p) => p,
            None => return McpResponse::error_with_code(id, -32602, "Missing params"),
        };

        let name = match params.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => return McpResponse::error_with_code(id, -32602, "Missing tool name"),
        };

        let arguments: HashMap<String, Value> = params
            .get("arguments")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        debug!("Calling tool: {} with args: {:?}", name, arguments);

        let result =
            handle_tool_call(&name, &arguments, &self.config, &self.db, &self.store).await;

        McpResponse::success(
            id,
            json!({
                "content": result.content,
                "isError": result.is_error
            }),
        )
    }

    /// Handle resources/list request
    async fn handle_resources_list(&self, id: Option<Value>) -> McpResponse {
        McpResponse::success(id, json!({ "resources": [] }))
    }

    /// Handle prompts/list request
    async fn handle_prompts_list(&self, id: Option<Value>) -> McpResponse {
        McpResponse::success(id, json!({ "prompts": [] }))
    }
}
