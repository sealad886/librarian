//! Init command implementation

use crate::config::{render_config_toml, Config};
use crate::error::{Error, Result};
use crate::meta::MetaDb;
use crate::store::QdrantStore;
use crossterm::cursor;
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal;
use crossterm::execute;
use std::collections::HashSet;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use tracing::info;
use url::Url;

#[derive(Debug, Clone)]
pub struct InitOptions {
    pub base_dir: PathBuf,
    pub config_path: PathBuf,
    pub force: bool,
    pub non_interactive: bool,
    pub yes: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InitAction {
    Abort,
    Overwrite,
    Merge,
}

/// Initialize librarian configuration and database
pub async fn cmd_init(options: InitOptions) -> Result<()> {
    let InitOptions {
        base_dir,
        config_path,
        force,
        non_interactive,
        yes,
    } = options;

    let is_tty = io::stdin().is_terminal();
    let interactive = resolve_interactive(is_tty, non_interactive)?;
    let auto_accept = yes || non_interactive;

    let config_exists = config_path.exists();
    let action = if config_exists {
        if force {
            InitAction::Overwrite
        } else if !interactive {
            return Err(Error::Config(format!(
                "Config already exists at {}. Use --force or run interactively.",
                config_path.display()
            )));
        } else {
            prompt_init_action(auto_accept)?
        }
    } else {
        InitAction::Overwrite
    };

    if action == InitAction::Abort {
        println!("Initialization aborted.");
        return Ok(());
    }

    let mut config = match action {
        InitAction::Merge => Config::load(&config_path)?,
        InitAction::Overwrite => Config::default(),
        InitAction::Abort => unreachable!(),
    };

    config.paths.base_dir = base_dir.clone();
    config.paths.config_file = config_path.clone();
    config.paths.db_file = base_dir.join("metadata.db");

    let mut defaults = Config::default();
    defaults.paths.base_dir = base_dir.clone();
    defaults.paths.config_file = config_path.clone();
    defaults.paths.db_file = base_dir.join("metadata.db");

    if interactive && !auto_accept {
        run_init_wizard(&mut config)?;
    }

    config.validate()?;

    let irrelevant = compute_irrelevant_paths(&config);
    let rendered = render_config_toml(&config, &defaults, &irrelevant);

    if interactive && !auto_accept {
        println!("\nConfiguration preview:\n");
        println!("{}", rendered);
        let confirm = prompt_confirm("Write this configuration?", true, auto_accept)?;
        if !confirm {
            println!("Initialization aborted.");
            return Ok(());
        }
    }

    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&config_path, rendered)?;
    info!("Created config at {:?}", config_path);

    let db = MetaDb::connect(&config).await?;
    db.init_schema().await?;
    info!("Created database at {:?}", config.paths.db_file);

    match QdrantStore::connect(&config).await {
        Ok(store) => match store.ensure_collection().await {
            Ok(_) => info!("Qdrant collection '{}' ready", config.collection_name),
            Err(e) => {
                tracing::warn!(
                    "Could not create Qdrant collection: {}. You can create it later.",
                    e
                );
            }
        },
        Err(e) => {
            tracing::warn!(
                "Could not connect to Qdrant at {}: {}. Make sure Qdrant is running.",
                config.qdrant_url,
                e
            );
        }
    }

    println!("✓ Initialized librarian at {:?}", config.paths.base_dir);
    println!("\nConfiguration: {:?}", config.paths.config_file);
    println!("Database: {:?}", config.paths.db_file);
    println!("\nNext steps:");
    println!("  librarian ingest dir ./path/to/docs    # Index local docs");
    println!("  librarian ingest url https://docs.rs   # Index web docs");
    println!("  librarian query \"how to use X\"         # Search the index");

    Ok(())
}

fn resolve_interactive(is_tty: bool, non_interactive: bool) -> Result<bool> {
    if !is_tty && !non_interactive {
        return Err(Error::Config(
            "stdin is not a TTY. Use --non-interactive to write defaults.".to_string(),
        ));
    }
    Ok(is_tty && !non_interactive)
}

fn prompt_init_action(auto_accept: bool) -> Result<InitAction> {
    if auto_accept {
        return Ok(InitAction::Overwrite);
    }

    let options = ["Abort", "Overwrite", "Merge/update interactively"];
    let selection = prompt_select("Config exists. Choose an action:", &options, 0, auto_accept)?;
    Ok(match selection {
        0 => InitAction::Abort,
        1 => InitAction::Overwrite,
        _ => InitAction::Merge,
    })
}

fn run_init_wizard(config: &mut Config) -> Result<()> {
    println!("\nWelcome to the librarian setup wizard.\n");

    // Core storage
    config.qdrant_url = prompt_string(
        "Qdrant URL",
        &config.qdrant_url,
        |value| Url::parse(value).map(|_| ()).map_err(|_| "Invalid URL".to_string()),
        false,
    )?;

    config.collection_name = prompt_string(
        "Qdrant collection name",
        &config.collection_name,
        |value| {
            if value.trim().is_empty() {
                Err("Collection name cannot be empty".to_string())
            } else {
                Ok(())
            }
        },
        false,
    )?;

    config.qdrant_api_key_env = prompt_string(
        "Qdrant API key env var (leave empty if not needed)",
        &config.qdrant_api_key_env,
        |_| Ok(()),
        false,
    )?;

    // Embedding
    let embedding_models = embedding_model_choices(&config.embedding.model);
    config.embedding.model = prompt_select_with_custom(
        "Embedding model",
        &embedding_models,
        &config.embedding.model,
    )?;

    let dimension_options = [384, 512, 768, 1024];
    config.embedding.dimension = prompt_select_dimension(
        "Embedding dimension",
        config.embedding.dimension,
        &dimension_options,
    )?;

    config.embedding.batch_size = prompt_usize(
        "Embedding batch size",
        config.embedding.batch_size,
        |value| {
            if value == 0 {
                Err("Batch size must be > 0".to_string())
            } else {
                Ok(())
            }
        },
        false,
    )?;

    // Ranking/search
    config.query.default_k = prompt_usize(
        "Default result count",
        config.query.default_k,
        |value| {
            if value == 0 {
                Err("Default_k must be > 0".to_string())
            } else {
                Ok(())
            }
        },
        false,
    )?;
    config.query.max_results = prompt_usize(
        "Maximum results",
        config.query.max_results,
        |value| {
            if value == 0 {
                Err("max_results must be > 0".to_string())
            } else {
                Ok(())
            }
        },
        false,
    )?;
    config.query.min_score = prompt_f32(
        "Minimum similarity score (0-1)",
        config.query.min_score,
        |value| {
            if (0.0..=1.0).contains(&value) {
                Ok(())
            } else {
                Err("min_score must be between 0 and 1".to_string())
            }
        },
        false,
    )?;

    config.query.hybrid_search = prompt_confirm("Enable hybrid BM25 + vector search?", config.query.hybrid_search, false)?;
    if config.query.hybrid_search {
        config.query.bm25_weight = prompt_f32(
            "BM25 weight (0-1)",
            config.query.bm25_weight,
            |value| {
                if (0.0..=1.0).contains(&value) {
                    Ok(())
                } else {
                    Err("bm25_weight must be between 0 and 1".to_string())
                }
            },
            false,
        )?;
    }

    config.reranker.enabled = prompt_confirm("Enable reranking?", config.reranker.enabled, false)?;
    if config.reranker.enabled {
        let reranker_models = reranker_model_choices(&config.reranker.model);
        config.reranker.model = prompt_select_with_custom(
            "Reranker model",
            &reranker_models,
            &config.reranker.model,
        )?;
        config.reranker.top_k = prompt_usize(
            "Reranker top_k",
            config.reranker.top_k,
            |value| {
                if value == 0 {
                    Err("top_k must be > 0".to_string())
                } else {
                    Ok(())
                }
            },
            false,
        )?;
    }

    // Chunking
    config.chunk.max_chars = prompt_usize(
        "Chunk max chars",
        config.chunk.max_chars,
        |value| {
            if value == 0 {
                Err("max_chars must be > 0".to_string())
            } else {
                Ok(())
            }
        },
        false,
    )?;
    config.chunk.min_chars = prompt_usize(
        "Chunk min chars",
        config.chunk.min_chars,
        |value| {
            if value >= config.chunk.max_chars {
                Err("min_chars must be < max_chars".to_string())
            } else {
                Ok(())
            }
        },
        false,
    )?;
    config.chunk.overlap_chars = prompt_usize(
        "Chunk overlap chars",
        config.chunk.overlap_chars,
        |value| {
            if value >= config.chunk.max_chars {
                Err("overlap_chars must be < max_chars".to_string())
            } else {
                Ok(())
            }
        },
        false,
    )?;
    config.chunk.prefer_heading_boundaries = prompt_confirm(
        "Prefer heading boundaries?",
        config.chunk.prefer_heading_boundaries,
        false,
    )?;

    // Crawling
    config.crawl.user_agent = prompt_string(
        "User agent",
        &config.crawl.user_agent,
        |value| {
            if value.trim().is_empty() {
                Err("User agent cannot be empty".to_string())
            } else {
                Ok(())
            }
        },
        false,
    )?;
    config.crawl.timeout_secs = prompt_u64(
        "Request timeout (seconds)",
        config.crawl.timeout_secs,
        |value| {
            if value == 0 {
                Err("timeout_secs must be > 0".to_string())
            } else {
                Ok(())
            }
        },
        false,
    )?;
    config.crawl.max_pages = prompt_u32(
        "Max pages per source",
        config.crawl.max_pages,
        |value| {
            if value == 0 {
                Err("max_pages must be > 0".to_string())
            } else {
                Ok(())
            }
        },
        false,
    )?;
    config.crawl.max_depth = prompt_u32(
        "Max crawl depth",
        config.crawl.max_depth,
        |value| {
            if value == 0 {
                Err("max_depth must be > 0".to_string())
            } else {
                Ok(())
            }
        },
        false,
    )?;
    config.crawl.rate_limit_per_host = prompt_f64(
        "Requests per second per host",
        config.crawl.rate_limit_per_host,
        |value| {
            if value > 0.0 {
                Ok(())
            } else {
                Err("rate_limit_per_host must be > 0".to_string())
            }
        },
        false,
    )?;
    config.crawl.respect_robots_txt = prompt_confirm(
        "Respect robots.txt?",
        config.crawl.respect_robots_txt,
        false,
    )?;

    let configure_advanced = prompt_confirm(
        "Configure advanced crawl settings?",
        false,
        false,
    )?;
    if configure_advanced {
        config.crawl.allowed_domains = prompt_string_list(
            "Allowed domains (comma-separated, empty for same domain)",
            &config.crawl.allowed_domains,
            false,
        )?;
        let path_prefix = prompt_string(
            "Path prefix (leave empty for none)",
            config.crawl.path_prefix.as_deref().unwrap_or(""),
            |_| Ok(()),
            false,
        )?;
        config.crawl.path_prefix = if path_prefix.trim().is_empty() {
            None
        } else {
            Some(path_prefix)
        };

        config.crawl.auto_js_rendering = prompt_confirm(
            "Auto-detect SPAs and use JS rendering?",
            config.crawl.auto_js_rendering,
            false,
        )?;
        if config.crawl.auto_js_rendering {
            config.crawl.js_page_load_timeout_ms = prompt_u64(
                "JS page load timeout (ms)",
                config.crawl.js_page_load_timeout_ms,
                |value| {
                    if value > 0 {
                        Ok(())
                    } else {
                        Err("js_page_load_timeout_ms must be > 0".to_string())
                    }
                },
                false,
            )?;
            config.crawl.js_render_wait_ms = prompt_u64(
                "JS render wait (ms)",
                config.crawl.js_render_wait_ms,
                |value| {
                    if value > 0 {
                        Ok(())
                    } else {
                        Err("js_render_wait_ms must be > 0".to_string())
                    }
                },
                false,
            )?;
            config.crawl.js_no_sandbox = prompt_confirm(
                "Disable browser sandbox?",
                config.crawl.js_no_sandbox,
                false,
            )?;
        }
    }

    config.crawl.multimodal.enabled = prompt_confirm(
        "Enable multimodal image discovery?",
        config.crawl.multimodal.enabled,
        false,
    )?;
    if config.crawl.multimodal.enabled {
        config.crawl.multimodal.include_images = prompt_confirm(
            "Include images?",
            config.crawl.multimodal.include_images,
            false,
        )?;
        if config.crawl.multimodal.include_images {
            config.crawl.multimodal.max_asset_bytes = prompt_usize(
                "Max image size (bytes)",
                config.crawl.multimodal.max_asset_bytes,
                |value| {
                    if value == 0 {
                        Err("max_asset_bytes must be > 0".to_string())
                    } else {
                        Ok(())
                    }
                },
                false,
            )?;
            config.crawl.multimodal.min_asset_bytes = prompt_usize(
                "Min image size (bytes)",
                config.crawl.multimodal.min_asset_bytes,
                |value| {
                    if value > config.crawl.multimodal.max_asset_bytes {
                        Err("min_asset_bytes must be <= max_asset_bytes".to_string())
                    } else {
                        Ok(())
                    }
                },
                false,
            )?;
            config.crawl.multimodal.max_assets_per_page = prompt_usize(
                "Max images per page",
                config.crawl.multimodal.max_assets_per_page,
                |value| {
                    if value == 0 {
                        Err("max_assets_per_page must be > 0".to_string())
                    } else {
                        Ok(())
                    }
                },
                false,
            )?;
            config.crawl.multimodal.allowed_mime_prefixes = prompt_string_list(
                "Allowed image MIME prefixes (comma-separated)",
                &config.crawl.multimodal.allowed_mime_prefixes,
                false,
            )?;
            config.crawl.multimodal.min_relevance_score = prompt_f32(
                "Minimum relevance score (0-1)",
                config.crawl.multimodal.min_relevance_score,
                |value| {
                    if (0.0..=1.0).contains(&value) {
                        Ok(())
                    } else {
                        Err("min_relevance_score must be between 0 and 1".to_string())
                    }
                },
                false,
            )?;
            config.crawl.multimodal.include_css_background_images = prompt_confirm(
                "Include CSS background images?",
                config.crawl.multimodal.include_css_background_images,
                false,
            )?;
        }
    }

    Ok(())
}

fn compute_irrelevant_paths(config: &Config) -> HashSet<String> {
    let mut irrelevant = HashSet::new();

    if !config.reranker.enabled {
        irrelevant.insert("reranker.model".to_string());
        irrelevant.insert("reranker.top_k".to_string());
    }

    if !config.query.hybrid_search {
        irrelevant.insert("query.bm25_weight".to_string());
    }

    if !config.crawl.auto_js_rendering {
        irrelevant.insert("crawl.js_page_load_timeout_ms".to_string());
        irrelevant.insert("crawl.js_render_wait_ms".to_string());
        irrelevant.insert("crawl.js_no_sandbox".to_string());
    }

    if !config.crawl.multimodal.enabled {
        for key in [
            "crawl.multimodal.include_images",
            "crawl.multimodal.include_audio",
            "crawl.multimodal.include_video",
            "crawl.multimodal.max_asset_bytes",
            "crawl.multimodal.min_asset_bytes",
            "crawl.multimodal.max_assets_per_page",
            "crawl.multimodal.allowed_mime_prefixes",
            "crawl.multimodal.min_relevance_score",
            "crawl.multimodal.include_css_background_images",
        ] {
            irrelevant.insert(key.to_string());
        }
    } else if !config.crawl.multimodal.include_images {
        for key in [
            "crawl.multimodal.max_asset_bytes",
            "crawl.multimodal.min_asset_bytes",
            "crawl.multimodal.max_assets_per_page",
            "crawl.multimodal.allowed_mime_prefixes",
            "crawl.multimodal.min_relevance_score",
            "crawl.multimodal.include_css_background_images",
        ] {
            irrelevant.insert(key.to_string());
        }
    }

    irrelevant
}

fn embedding_model_choices(current: &str) -> Vec<String> {
    let mut models = vec![
        "BAAI/bge-small-en-v1.5".to_string(),
        "BAAI/bge-base-en-v1.5".to_string(),
        "BAAI/bge-large-en-v1.5".to_string(),
        "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        "Qwen/Qwen3-VL-Embedding-2B".to_string(),
        "Qwen/Qwen3-VL-Embedding-8B".to_string(),
        "jinaai/jina-clip-v2".to_string(),
        "google/siglip2-base-patch16-224".to_string(),
        "vidore/colpali".to_string(),
    ];

    if !models.iter().any(|m| m == current) {
        models.insert(0, current.to_string());
    }

    models
}

fn reranker_model_choices(current: &str) -> Vec<String> {
    let mut models = vec![
        "BAAI/bge-reranker-base".to_string(),
        "Qwen/Qwen3-VL-Reranker-2B".to_string(),
        "Qwen/Qwen3-VL-Reranker-8B".to_string(),
        "jinaai/jina-reranker-m0".to_string(),
        "lightonai/MonoQwen2-VL-v0.1".to_string(),
    ];

    if !models.iter().any(|m| m == current) {
        models.insert(0, current.to_string());
    }

    models
}

fn prompt_select_with_custom(
    label: &str,
    options: &[String],
    default_value: &str,
) -> Result<String> {
    let mut choices = options.to_vec();
    let custom_label = "Custom...".to_string();
    let default_index = choices
        .iter()
        .position(|item| item == default_value)
        .unwrap_or(choices.len());
    if !choices.contains(&custom_label) {
        choices.push(custom_label.clone());
    }

    let selection = prompt_select(label, &choices, default_index, false)?;
    if choices[selection] == custom_label {
        return prompt_string(
            "Enter custom value",
            default_value,
            |value| {
                if value.trim().is_empty() {
                    Err("Value cannot be empty".to_string())
                } else {
                    Ok(())
                }
            },
            false,
        );
    }

    Ok(choices[selection].clone())
}

fn prompt_select_dimension(
    label: &str,
    current: usize,
    options: &[usize],
) -> Result<usize> {
    let mut choices = options.iter().map(|v| v.to_string()).collect::<Vec<_>>();
    let default_index = options
        .iter()
        .position(|v| *v == current)
        .unwrap_or(choices.len());
    choices.push("Custom...".to_string());

    let selection = prompt_select(label, &choices, default_index, false)?;
    if selection == choices.len() - 1 {
        return prompt_usize(
            "Enter custom dimension",
            current,
            |value| {
                if value == 0 {
                    Err("Dimension must be > 0".to_string())
                } else {
                    Ok(())
                }
            },
            false,
        );
    }

    choices[selection]
        .parse::<usize>()
        .map_err(|_| Error::Config("Invalid dimension selection".to_string()))
}

fn prompt_confirm(label: &str, default: bool, auto_accept: bool) -> Result<bool> {
    if auto_accept {
        return Ok(default);
    }
    let options = ["Yes", "No"];
    let default_index = if default { 0 } else { 1 };
    let selection = prompt_select(label, &options, default_index, auto_accept)?;
    Ok(selection == 0)
}

fn prompt_select(label: &str, options: &[impl AsRef<str>], default_index: usize, auto_accept: bool) -> Result<usize> {
    if auto_accept {
        return Ok(default_index.min(options.len().saturating_sub(1)));
    }

    let mut stdout = io::stdout();
    let mut selected = default_index.min(options.len().saturating_sub(1));
    let _raw_mode = RawModeGuard::new()?;

    loop {
        execute!(
            stdout,
            terminal::Clear(terminal::ClearType::All),
            cursor::MoveTo(0, 0)
        )?;
        writeln!(stdout, "{}", label)?;
        for (idx, option) in options.iter().enumerate() {
            if idx == selected {
                writeln!(stdout, "> {}", option.as_ref())?;
            } else {
                writeln!(stdout, "  {}", option.as_ref())?;
            }
        }
        stdout.flush()?;

        match event::read()? {
            Event::Key(key) => match key.code {
                KeyCode::Up => {
                    if selected > 0 {
                        selected -= 1;
                    }
                }
                KeyCode::Down => {
                    if selected + 1 < options.len() {
                        selected += 1;
                    }
                }
                KeyCode::Enter => {
                    execute!(
                        stdout,
                        terminal::Clear(terminal::ClearType::FromCursorDown),
                        cursor::MoveToColumn(0)
                    )?;
                    return Ok(selected);
                }
                _ => {}
            },
            _ => {}
        }
    }
}

fn prompt_string<F>(label: &str, default: &str, validate: F, auto_accept: bool) -> Result<String>
where
    F: Fn(&str) -> std::result::Result<(), String>,
{
    if auto_accept {
        return Ok(default.to_string());
    }

    loop {
        print!("{} [{}]: ", label, default);
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let value = input.trim();
        let value = if value.is_empty() { default } else { value };

        if let Err(message) = validate(value) {
            println!("{}", message);
            continue;
        }
        return Ok(value.to_string());
    }
}

fn prompt_usize<F>(label: &str, default: usize, validate: F, auto_accept: bool) -> Result<usize>
where
    F: Fn(usize) -> std::result::Result<(), String>,
{
    if auto_accept {
        return Ok(default);
    }
    loop {
        let value = prompt_string(label, &default.to_string(), |_| Ok(()), false)?;
        match value.parse::<usize>() {
            Ok(parsed) => {
                if let Err(message) = validate(parsed) {
                    println!("{}", message);
                    continue;
                }
                return Ok(parsed);
            }
            Err(_) => println!("Enter a valid number."),
        }
    }
}

fn prompt_u32<F>(label: &str, default: u32, validate: F, auto_accept: bool) -> Result<u32>
where
    F: Fn(u32) -> std::result::Result<(), String>,
{
    if auto_accept {
        return Ok(default);
    }
    loop {
        let value = prompt_string(label, &default.to_string(), |_| Ok(()), false)?;
        match value.parse::<u32>() {
            Ok(parsed) => {
                if let Err(message) = validate(parsed) {
                    println!("{}", message);
                    continue;
                }
                return Ok(parsed);
            }
            Err(_) => println!("Enter a valid number."),
        }
    }
}

fn prompt_u64<F>(label: &str, default: u64, validate: F, auto_accept: bool) -> Result<u64>
where
    F: Fn(u64) -> std::result::Result<(), String>,
{
    if auto_accept {
        return Ok(default);
    }
    loop {
        let value = prompt_string(label, &default.to_string(), |_| Ok(()), false)?;
        match value.parse::<u64>() {
            Ok(parsed) => {
                if let Err(message) = validate(parsed) {
                    println!("{}", message);
                    continue;
                }
                return Ok(parsed);
            }
            Err(_) => println!("Enter a valid number."),
        }
    }
}

fn prompt_f32<F>(label: &str, default: f32, validate: F, auto_accept: bool) -> Result<f32>
where
    F: Fn(f32) -> std::result::Result<(), String>,
{
    if auto_accept {
        return Ok(default);
    }
    loop {
        let value = prompt_string(label, &default.to_string(), |_| Ok(()), false)?;
        match value.parse::<f32>() {
            Ok(parsed) => {
                if let Err(message) = validate(parsed) {
                    println!("{}", message);
                    continue;
                }
                return Ok(parsed);
            }
            Err(_) => println!("Enter a valid number."),
        }
    }
}

fn prompt_f64<F>(label: &str, default: f64, validate: F, auto_accept: bool) -> Result<f64>
where
    F: Fn(f64) -> std::result::Result<(), String>,
{
    if auto_accept {
        return Ok(default);
    }
    loop {
        let value = prompt_string(label, &default.to_string(), |_| Ok(()), false)?;
        match value.parse::<f64>() {
            Ok(parsed) => {
                if let Err(message) = validate(parsed) {
                    println!("{}", message);
                    continue;
                }
                return Ok(parsed);
            }
            Err(_) => println!("Enter a valid number."),
        }
    }
}

fn prompt_string_list(label: &str, default: &[String], auto_accept: bool) -> Result<Vec<String>> {
    let default_value = if default.is_empty() {
        "".to_string()
    } else {
        default.join(",")
    };
    let value = prompt_string(label, &default_value, |_| Ok(()), auto_accept)?;
    let items = value
        .split(',')
        .map(|item| item.trim())
        .filter(|item| !item.is_empty())
        .map(|item| item.to_string())
        .collect::<Vec<_>>();
    Ok(items)
}

struct RawModeGuard;

impl RawModeGuard {
    fn new() -> Result<Self> {
        terminal::enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_interactive_requires_tty() {
        let result = resolve_interactive(false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_interactive_non_interactive_ok() {
        let result = resolve_interactive(false, true).unwrap();
        assert!(!result);
    }
}
