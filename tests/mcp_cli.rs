use std::process::Stdio;
use std::time::Duration;

use tempfile::tempdir;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;

fn write_test_config() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("config.toml");
    let config = r#"qdrant_url = "http://127.0.0.1:6334"
collection_name = "librarian_test"

[embedding]
model = "BAAI/bge-small-en-v1.5"
dimension = 384
"#;
    std::fs::write(&config_path, config).expect("write config");
    (dir, config_path)
}

#[tokio::test]
async fn mcp_initialize_responds_on_stdout_and_logs_to_stderr() {
    let (_dir, config_path) = write_test_config();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_librarian"));
    cmd.arg("--config")
        .arg(config_path)
        .arg("mcp")
        .env("RUST_LOG", "debug")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("spawn mcp");
    let mut stdin = child.stdin.take().expect("stdin");
    stdin
        .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}\n")
        .await
        .expect("write initialize");
    drop(stdin);

    let output = timeout(Duration::from_secs(5), child.wait_with_output())
        .await
        .expect("mcp timeout")
        .expect("mcp output");

    assert!(output.status.success(), "mcp exited non-zero");

    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    let mut responses = Vec::new();
    for line in stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let value: serde_json::Value =
            serde_json::from_str(line).expect("stdout line should be json");
        responses.push(value);
    }

    assert!(
        responses.iter().any(|value| {
            value.get("id") == Some(&serde_json::json!(1))
                && value
                    .get("result")
                    .and_then(|result| result.get("protocolVersion"))
                    .and_then(|version| version.as_str())
                    == Some("2024-11-05")
        }),
        "missing initialize response in stdout"
    );

    let stderr = String::from_utf8(output.stderr).expect("stderr utf8");
    assert!(
        stderr.contains("MCP server starting on stdio"),
        "expected logs on stderr"
    );
}

#[tokio::test]
async fn mcp_help_exits_quickly() {
    let output = timeout(
        Duration::from_secs(2),
        Command::new(env!("CARGO_BIN_EXE_librarian"))
            .arg("mcp")
            .arg("--help")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output(),
    )
    .await
    .expect("help timeout")
    .expect("help output");

    assert!(output.status.success(), "mcp --help failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("mcp"), "help output missing mcp");
}
