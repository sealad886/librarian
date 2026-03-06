use std::time::Duration;

use tempfile::tempdir;
use tokio::process::Command;
use tokio::time::timeout;

fn write_config(contents: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("config.toml");
    std::fs::write(&config_path, contents).expect("write config");
    (dir, config_path)
}

fn local_config(qdrant_url: &str) -> String {
    format!(
        r#"qdrant_url = "{qdrant_url}"
collection_name = "librarian_test"

[embedding]
model = "BAAI/bge-small-en-v1.5"
dimension = 384
"#
    )
}

fn custom_backend_config(qdrant_url: &str, backend_url: &str) -> String {
    format!(
        r#"qdrant_url = "{qdrant_url}"
collection_name = "librarian_test"

[embedding]
model = "custom"
allow_custom = true

[embedding.custom]
id = "custom/test-model"
backend = "http"
url = "{backend_url}"
family = "custom"
dimension = 384
modalities = ["text"]
"#
    )
}

#[tokio::test]
async fn sources_accepts_config_directory_path() {
    let (dir, _config_path) = write_config(&local_config("http://127.0.0.1:65534"));

    let output = timeout(
        Duration::from_secs(5),
        Command::new(env!("CARGO_BIN_EXE_librarian"))
            .arg("--config")
            .arg(dir.path())
            .arg("sources")
            .arg("--json")
            .output(),
    )
    .await
    .expect("sources timeout")
    .expect("sources output");

    assert!(output.status.success(), "sources failed");
    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    assert_eq!(stdout.trim(), "[]");
}

#[tokio::test]
async fn sources_skip_unreachable_custom_embedding_backend() {
    let (dir, _config_path) = write_config(&custom_backend_config(
        "http://127.0.0.1:65534",
        "http://127.0.0.1:65535",
    ));

    let output = timeout(
        Duration::from_secs(5),
        Command::new(env!("CARGO_BIN_EXE_librarian"))
            .arg("--config")
            .arg(dir.path())
            .arg("sources")
            .arg("--json")
            .output(),
    )
    .await
    .expect("sources timeout")
    .expect("sources output");

    assert!(output.status.success(), "sources failed");
    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    assert_eq!(stdout.trim(), "[]");
}

#[tokio::test]
async fn status_finishes_without_bootstrapping_xinference() {
    let (dir, _config_path) = write_config(&local_config("http://127.0.0.1:65534"));

    let output = timeout(
        Duration::from_secs(5),
        Command::new(env!("CARGO_BIN_EXE_librarian"))
            .arg("--config")
            .arg(dir.path())
            .arg("status")
            .arg("--json")
            .output(),
    )
    .await
    .expect("status timeout")
    .expect("status output");

    assert!(output.status.success(), "status failed");
    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    let status: serde_json::Value = serde_json::from_str(&stdout).expect("status json");
    assert_eq!(
        status.get("qdrant_connected"),
        Some(&serde_json::Value::Bool(false))
    );
}

#[tokio::test]
async fn list_alias_help_works() {
    let output = timeout(
        Duration::from_secs(5),
        Command::new(env!("CARGO_BIN_EXE_librarian"))
            .arg("list")
            .arg("--help")
            .output(),
    )
    .await
    .expect("list help timeout")
    .expect("list help output");

    assert!(output.status.success(), "list --help failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("List registered sources"));
}

#[tokio::test]
async fn prune_legacy_flag_alias_parses() {
    let output = timeout(
        Duration::from_secs(5),
        Command::new(env!("CARGO_BIN_EXE_librarian"))
            .arg("prune")
            .arg("--orphans")
            .arg("--help")
            .output(),
    )
    .await
    .expect("prune help timeout")
    .expect("prune help output");

    assert!(output.status.success(), "prune --orphans --help failed");
}

#[tokio::test]
async fn ingest_url_legacy_same_domain_flag_parses() {
    let output = timeout(
        Duration::from_secs(5),
        Command::new(env!("CARGO_BIN_EXE_librarian"))
            .arg("ingest")
            .arg("url")
            .arg("--same-domain")
            .arg("--help")
            .output(),
    )
    .await
    .expect("ingest url help timeout")
    .expect("ingest url help output");

    assert!(
        output.status.success(),
        "ingest url --same-domain --help failed"
    );
}
