use clap::{Parser, Subcommand};
use librarian::xinference::{sync_xinference_snapshots, RegistryType, SyncOptions};
use std::path::PathBuf;
use std::time::Duration;
use url::Url;

#[derive(Parser)]
#[command(name = "xtask")]
struct XtaskCli {
    #[command(subcommand)]
    command: XtaskCommand,
}

#[derive(Subcommand)]
enum XtaskCommand {
    /// Sync Xinference model registry snapshots
    XinferenceSync {
        /// Xinference base URL
        #[arg(long, default_value = "http://127.0.0.1:9997")]
        endpoint: String,

        /// Comma-separated registry types to sync (embedding,rerank,audio,video,image,llm)
        #[arg(long, default_value = "embedding,rerank,audio,video")]
        types: String,

        /// Output directory for snapshot files
        #[arg(long, default_value = "resources/xinference")]
        out: PathBuf,

        /// Write snapshots to disk (default is dry-run)
        #[arg(long)]
        write: bool,

        /// Skip the update/refresh call (list only)
        #[arg(long)]
        skip_update: bool,

        /// HTTP retries for each request
        #[arg(long, default_value = "3")]
        retries: usize,

        /// Timeout in seconds for each request
        #[arg(long, default_value = "30")]
        timeout_secs: u64,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = XtaskCli::parse();

    match cli.command {
        XtaskCommand::XinferenceSync {
            endpoint,
            types,
            out,
            write,
            skip_update,
            retries,
            timeout_secs,
        } => {
            let endpoint = Url::parse(&endpoint)?;
            let registry_types = parse_registry_types(&types)?;
            let options = SyncOptions {
                endpoint,
                registry_types,
                out_dir: out,
                refresh: !skip_update,
                write,
                retries,
                timeout: Duration::from_secs(timeout_secs),
            };

            let report = sync_xinference_snapshots(&options).await?;
            for diff in &report.diffs {
                println!(
                    "{}: {} → {} (added {}, removed {})",
                    diff.registry_type,
                    diff.total_old,
                    diff.total_new,
                    diff.added.len(),
                    diff.removed.len()
                );
            }

            if !write && report.has_changes {
                anyhow::bail!(
                    "Xinference registry snapshots are out of date. Re-run with --write."
                );
            }

            if write && !report.has_changes {
                println!("Snapshots already up to date");
            }
        }
    }

    Ok(())
}

fn parse_registry_types(input: &str) -> anyhow::Result<Vec<RegistryType>> {
    let mut types = Vec::new();
    for raw in input.split(',') {
        let value = raw.trim();
        if value.is_empty() {
            continue;
        }
        let parsed = value
            .parse::<RegistryType>()
            .map_err(|_| anyhow::anyhow!("Unknown registry type '{}'.", value))?;
        types.push(parsed);
    }

    if types.is_empty() {
        anyhow::bail!("No registry types provided.");
    }

    Ok(types)
}
