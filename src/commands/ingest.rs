//! Ingest command implementation

use crate::chunk::{chunk_document, compute_content_hash, TextChunk};
use crate::config::Config;
use crate::crawl::{CrawledPage, Crawler};
use crate::embed::{create_embedder, embed_images_in_batches, embed_in_batches, Embedder};
use crate::error::{Error, Result};
use crate::meta::{Chunk, Document, MetaDb, RunOperation, RunStatus, Source, SourceType};
use crate::models::{embedding_model_capabilities, is_multimodal_embedding_model, MultimodalStrategy};
use crate::parse::{is_binary_content, parse_content, should_skip_file, ContentType};
use crate::parse::{ParsedDocument, ExtractedMedia};
use crate::progress::add_progress_bar;
use crate::store::{ChunkPayload, ChunkPoint, QdrantStore};
use chrono::Utc;
use ignore::WalkBuilder;
use image::imageops::FilterType;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing::{debug, info, warn};
use url::Url;
use uuid::Uuid;

/// Statistics from an ingestion run
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestStats {
    pub docs_processed: i32,
    pub docs_skipped: i32,
    pub chunks_created: i32,
    pub chunks_updated: i32,
    pub chunks_deleted: i32,
    pub errors: Vec<String>,
    /// Warnings about source overlaps (potential duplicates)
    pub overlap_warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct CachedAsset {
    media: ExtractedMedia,
    hash: String,
    path: PathBuf,
}

/// Determine if a URL looks like an allowed image type based on extension and config
fn url_is_allowed_image(url: &str, allowed_prefixes: &[String]) -> bool {
    if let Ok(parsed) = Url::parse(url) {
        let scheme = parsed.scheme();
        if scheme != "http" && scheme != "https" {
            return false;
        }
    } else {
        return false;
    }

    // If allowed prefixes contain "image/", accept common image extensions
    let lower = url.to_lowercase();
    let exts = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".avif"];
    let has_img_ext = exts.iter().any(|e| lower.ends_with(e));
    let allows_images = allowed_prefixes.iter().any(|p| p.starts_with("image/"));
    has_img_ext && allows_images
}

fn normalize_media_url(url: &str) -> String {
    Url::parse(url)
        .map(|u| u.to_string())
        .unwrap_or_else(|_| url.to_string())
}

fn is_svg_url(url: &str) -> bool {
    Url::parse(url)
        .ok()
        .and_then(|u| {
            std::path::Path::new(u.path())
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| matches!(ext.to_lowercase().as_str(), "svg" | "svgz"))
        })
        .unwrap_or(false)
}

/// Very simple relevance scoring for image candidates based on alt text and heading overlap
fn score_image_candidate(doc: &ParsedDocument, media: &ExtractedMedia) -> f32 {
    let mut score: f32 = 0.0;
    let mut has_main_signal = false;
    if let Some(ref alt) = media.alt {
        let trimmed = alt.trim();
        if !trimmed.is_empty() {
            score += 0.35;
            has_main_signal = true;
        }
        let alt_lower = trimmed.to_lowercase();
        if doc.headings.iter().any(|h| alt_lower.contains(&h.text.to_lowercase())) {
            score += 0.2;
            has_main_signal = true;
        }
        if doc
            .title
            .as_ref()
            .map(|t| alt_lower.contains(&t.to_lowercase()))
            .unwrap_or(false)
        {
            score += 0.1;
            has_main_signal = true;
        }
    }

    let url_lower = media.url.to_lowercase();
    for kw in [
        "diagram",
        "architecture",
        "overview",
        "flow",
        "guide",
        "example",
        "schema",
        "chart",
        "figure",
        "graph",
        "plot",
        "screenshot",
    ]
    .iter()
    {
        if url_lower.contains(kw) {
            score += 0.2;
            has_main_signal = true;
            break;
        }
    }

    for kw in [
        "logo",
        "icon",
        "sprite",
        "avatar",
        "placeholder",
        "spinner",
        "favicon",
        "badge",
        "thumbnail",
        "thumb",
    ]
    .iter()
    {
        if url_lower.contains(kw) {
            score -= 0.3;
            break;
        }
    }

    if media.css_background {
        score -= 0.1;
    }

    if !has_main_signal {
        return 0.0;
    }

    // Cap at 1.0
    score.min(1.0f32)
}

/// Select image candidates according to config thresholds and limits
fn select_image_candidates(config: &Config, doc: &ParsedDocument) -> Vec<(ExtractedMedia, f32)> {
    let mm = &config.crawl.multimodal;
    if !mm.enabled || !mm.include_images { return Vec::new(); }
    if !is_multimodal_embedding_model(&config.embedding.model) {
        debug!(model = %config.embedding.model, "Skipping image candidates (model not multimodal)");
        return Vec::new();
    }
    if let Some(caps) = embedding_model_capabilities(&config.embedding.model) {
        if caps.strategy == MultimodalStrategy::LateInteraction {
            debug!(model = %config.embedding.model, "Skipping image candidates (late-interaction strategy not supported)");
            return Vec::new();
        }
    }

    // Collect, filter, score, and dedupe by normalized URL (keep highest score)
    let mut by_url: HashMap<String, (ExtractedMedia, f32)> = HashMap::new();
    for m in &doc.media {
        if m.css_background && !mm.include_css_background_images {
            continue;
        }
        if is_svg_url(&m.url) {
            continue;
        }
        if !url_is_allowed_image(&m.url, &mm.allowed_mime_prefixes) {
            continue;
        }

        let score = score_image_candidate(doc, m);
        if score < mm.min_relevance_score {
            debug!(url = %m.url, score, threshold = mm.min_relevance_score, "Rejected image candidate (below threshold)");
            continue;
        }

        let key = normalize_media_url(&m.url);
        match by_url.get(&key) {
            Some((_, existing_score)) if *existing_score >= score => {}
            _ => {
                by_url.insert(key, (m.clone(), score));
            }
        }
    }

    let mut scored: Vec<(ExtractedMedia, f32)> = by_url.into_values().collect();

    // Sort by score desc and take up to max_assets_per_page
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if scored.len() > mm.max_assets_per_page {
        scored.truncate(mm.max_assets_per_page);
    }
    scored
}

const PERCEPTUAL_HASH_SIZE: u32 = 8;
const PERCEPTUAL_HASH_MAX_DISTANCE: u32 = 5;

fn compute_perceptual_hash(bytes: &[u8]) -> Option<u64> {
    let image = image::load_from_memory(bytes).ok()?;
    let gray = image.to_luma8();
    let resized = image::imageops::resize(
        &gray,
        PERCEPTUAL_HASH_SIZE,
        PERCEPTUAL_HASH_SIZE,
        FilterType::Triangle,
    );
    let mut total: u32 = 0;
    for pixel in resized.pixels() {
        total += pixel[0] as u32;
    }
    let avg = total / (PERCEPTUAL_HASH_SIZE * PERCEPTUAL_HASH_SIZE) as u32;
    let mut hash: u64 = 0;
    for (idx, pixel) in resized.pixels().enumerate() {
        if pixel[0] as u32 >= avg {
            hash |= 1u64 << idx;
        }
    }
    Some(hash)
}

fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

fn is_perceptual_duplicate(hash: u64, seen: &[u64]) -> bool {
    seen.iter()
        .any(|seen_hash| hamming_distance(*seen_hash, hash) <= PERCEPTUAL_HASH_MAX_DISTANCE)
}

/// Fetch accepted image candidates and cache them under base_dir/assets
async fn fetch_and_cache_images(config: &Config, images: &[(ExtractedMedia, f32)]) -> Vec<CachedAsset> {
    use reqwest::header::CONTENT_TYPE;
    use reqwest::Client;
    use tokio::fs;

    let mm = &config.crawl.multimodal;
    if images.is_empty() {
        return Vec::new();
    }

    let client = Client::builder()
        .user_agent(&config.crawl.user_agent)
        .timeout(Duration::from_secs(config.crawl.timeout_secs))
        .gzip(true)
        .brotli(true)
        .build();
    let client = match client {
        Ok(c) => c,
        Err(e) => {
            warn!("Failed to create HTTP client: {}", e);
            return Vec::new();
        }
    };

    let assets_dir = config.paths.base_dir.join("assets");
    if let Err(e) = fs::create_dir_all(&assets_dir).await { warn!("Failed to create assets dir: {}", e); }

    let mut cached = Vec::new();
    let mut seen_hashes: HashSet<String> = HashSet::new();
    let mut seen_phashes: Vec<u64> = Vec::new();

    for (m, _score) in images.iter() {
        if Url::parse(&m.url).map(|u| u.scheme().to_string()).map(|s| s != "http" && s != "https").unwrap_or(true) {
            continue;
        }
        // Fetch
        match client.get(&m.url).send().await {
            Ok(resp) => {
                if let Some(content_type) = resp.headers().get(CONTENT_TYPE) {
                    if content_type
                        .to_str()
                        .map(|ct| ct.contains("image/svg"))
                        .unwrap_or(false)
                    {
                        debug!(url = %m.url, "Skipping image (SVG content-type)");
                        continue;
                    }
                }
                // Content length check
                if let Some(len) = resp.content_length() {
                    if len < mm.min_asset_bytes as u64 {
                        debug!(url = %m.url, size = len, min = mm.min_asset_bytes, "Skipping image (content-length below minimum)");
                        continue;
                    }
                    if len as usize > mm.max_asset_bytes {
                        debug!(url = %m.url, size = len, limit = mm.max_asset_bytes, "Skipping image (content-length exceeds limit)");
                        continue;
                    }
                }
                match resp.bytes().await {
                    Ok(bytes) => {
                        if bytes.len() < mm.min_asset_bytes {
                            debug!(url = %m.url, size = bytes.len(), min = mm.min_asset_bytes, "Skipping image (downloaded size below minimum)");
                            continue;
                        }
                        if bytes.len() > mm.max_asset_bytes {
                            debug!(url = %m.url, size = bytes.len(), limit = mm.max_asset_bytes, "Skipping image (downloaded size exceeds limit)");
                            continue;
                        }
                        let hash = compute_content_hash(&bytes);
                        if !seen_hashes.insert(hash.clone()) {
                            continue;
                        }
                        if let Some(phash) = compute_perceptual_hash(&bytes) {
                            if is_perceptual_duplicate(phash, &seen_phashes) {
                                debug!(url = %m.url, "Skipping image (perceptual duplicate)");
                                continue;
                            }
                            seen_phashes.push(phash);
                        }
                        // Determine extension from URL (best-effort)
                        let ext = if let Some(pos) = m.url.rfind('.') { m.url[pos..].to_string() } else { ".bin".to_string() };
                        let file_name = format!("{}{}", hash, ext);
                        let target = assets_dir.join(file_name);
                        let exists = fs::metadata(&target).await.is_ok();
                        if !exists {
                            if let Err(e) = fs::write(&target, &bytes).await {
                                warn!(url = %m.url, path = %target.display(), "Failed to write cached image: {}", e);
                                continue;
                            }
                        }
                        debug!(url = %m.url, path = %target.display(), size = bytes.len(), "Cached image asset");
                        cached.push(CachedAsset {
                            media: m.clone(),
                            hash,
                            path: target,
                        });
                    }
                    Err(e) => {
                        warn!(url = %m.url, "Failed to read image bytes: {}", e);
                    }
                }
            }
            Err(e) => {
                warn!(url = %m.url, "Failed to fetch image: {}", e);
            }
        }
    }

    cached
}

async fn embed_cached_images(
    config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
    embedder: &dyn Embedder,
    source: &Source,
    doc: &Document,
    doc_uri: &str,
    cached_images: Vec<CachedAsset>,
) -> Result<(i32, i32)> {
    if cached_images.is_empty() {
        return Ok((0, 0));
    }

    if !is_multimodal_embedding_model(&config.embedding.model) {
        debug!(uri = %doc_uri, model = %config.embedding.model, "Skipping image embedding (model not multimodal)");
        return Ok((0, 0));
    }

    let image_paths: Vec<String> = cached_images
        .iter()
        .map(|c| c.path.to_string_lossy().to_string())
        .collect();

    let embeddings = match embed_images_in_batches(embedder, image_paths, config.embedding.batch_size).await {
        Ok(embeddings) => embeddings,
        Err(e) => {
            warn!(uri = %doc_uri, "Image embedding failed: {}", e);
            return Ok((0, 0));
        }
    };

    if embeddings.is_empty() {
        return Ok((0, 0));
    }

    if embeddings[0].len() != embedder.dimension() {
        warn!(
            uri = %doc_uri,
            image_dim = embeddings[0].len(),
            expected_dim = embedder.dimension(),
            "Skipping image embeddings (dimension mismatch)"
        );
        return Ok((0, 0));
    }

    if embeddings.len() != cached_images.len() {
        warn!(
            uri = %doc_uri,
            embeddings = embeddings.len(),
            cached = cached_images.len(),
            "Image embedding count mismatch; truncating to shortest"
        );
    }

    let deleted_point_ids = db.delete_chunks_by_modality(&doc.id, "image").await?;
    if !deleted_point_ids.is_empty() {
        let deleted_uuids: Vec<Uuid> = deleted_point_ids
            .iter()
            .filter_map(|id| Uuid::try_parse(id).ok())
            .collect();
        if !deleted_uuids.is_empty() {
            store.delete_points(&deleted_uuids).await?;
        }
    }

    let mut points: Vec<ChunkPoint> = Vec::new();
    let mut created = 0i32;

    for (i, (asset, embedding)) in cached_images.iter().zip(embeddings.iter()).enumerate() {
        let chunk_index = -(i as i32) - 1;
        let chunk_text = asset
            .media
            .alt
            .clone()
            .unwrap_or_else(|| asset.media.url.clone());

        let meta_chunk = Chunk::new_media(
            doc.id.clone(),
            chunk_index,
            asset.hash.clone(),
            chunk_text,
            asset.media.url.clone(),
            Some(asset.hash.clone()),
        );

        db.upsert_chunk(&meta_chunk).await?;

        let mut payload = ChunkPayload::new(
            source.id.clone(),
            source.source_type.clone(),
            source.uri.clone(),
            doc.id.clone(),
            doc_uri.to_string(),
            chunk_index,
            asset.hash.clone(),
            Utc::now().to_rfc3339(),
        );
        payload.title = doc.title.clone();
        payload.modality = Some("image".to_string());
        payload.media_url = Some(asset.media.url.clone());
        payload.media_hash = Some(asset.hash.clone());

        let point_id = Uuid::try_parse(&meta_chunk.qdrant_point_id).unwrap_or_else(|_| {
            Uuid::new_v5(&Uuid::NAMESPACE_OID, meta_chunk.qdrant_point_id.as_bytes())
        });

        points.push(ChunkPoint {
            id: point_id,
            vector: embedding.clone(),
            payload,
        });

        created += 1;
    }

    store.upsert_points(points).await?;

    Ok((created, 0))
}

/// CLI overrides for crawl configuration
#[derive(Debug, Default)]
pub struct CrawlOverrides {
    pub max_pages: Option<u32>,
    pub max_depth: Option<u32>,
    pub path_prefix: Option<String>,
}

/// Describes an overlap between two sources
#[derive(Debug)]
pub struct SourceOverlap {
    pub existing_source: Source,
    pub overlap_type: OverlapType,
}

/// Type of overlap between sources
#[derive(Debug)]
pub enum OverlapType {
    /// New source is a subdirectory/subpath of existing
    SubsetOf,
    /// New source is a parent directory/path of existing
    SupersetOf,
    /// Sources have the exact same URI
    Identical,
}

/// Check if a new directory source overlaps with existing sources
pub async fn check_dir_overlap(db: &MetaDb, new_path: &Path) -> Result<Vec<SourceOverlap>> {
    let sources = db.list_sources().await?;
    let mut overlaps = Vec::new();

    for source in sources {
        // Only check Dir sources
        if source.get_type().ok() != Some(SourceType::Dir) {
            continue;
        }

        let existing_path = Path::new(&source.uri);

        // Check if paths overlap - determine overlap type first
        let overlap_type = if new_path == existing_path {
            Some(OverlapType::Identical)
        } else if new_path.starts_with(existing_path) {
            Some(OverlapType::SubsetOf)
        } else if existing_path.starts_with(new_path) {
            Some(OverlapType::SupersetOf)
        } else {
            None
        };

        if let Some(ot) = overlap_type {
            overlaps.push(SourceOverlap {
                existing_source: source,
                overlap_type: ot,
            });
        }
    }

    Ok(overlaps)
}

/// Check if a new URL source overlaps with existing URL sources
pub async fn check_url_overlap(db: &MetaDb, new_url: &str) -> Result<Vec<SourceOverlap>> {
    let sources = db.list_sources().await?;
    let mut overlaps = Vec::new();

    let new_parsed = match Url::parse(new_url) {
        Ok(u) => u,
        Err(_) => return Ok(overlaps),
    };

    let new_host = new_parsed.host_str().unwrap_or("");
    let new_path = new_parsed.path();

    for source in sources {
        // Check Url and Sitemap sources
        let source_type = source.get_type().ok();
        if source_type != Some(SourceType::Url) && source_type != Some(SourceType::Sitemap) {
            continue;
        }

        let existing_parsed = match Url::parse(&source.uri) {
            Ok(u) => u,
            Err(_) => continue,
        };

        let existing_host = existing_parsed.host_str().unwrap_or("");
        let existing_path = existing_parsed.path();

        // Only check overlaps for same domain
        if new_host != existing_host {
            continue;
        }

        // Determine overlap type
        let overlap_type = if new_path == existing_path {
            Some(OverlapType::Identical)
        } else if new_path.starts_with(existing_path) {
            Some(OverlapType::SubsetOf)
        } else if existing_path.starts_with(new_path) {
            Some(OverlapType::SupersetOf)
        } else {
            None
        };

        if let Some(ot) = overlap_type {
            overlaps.push(SourceOverlap {
                existing_source: source,
                overlap_type: ot,
            });
        }
    }

    Ok(overlaps)
}

/// Format overlap warnings for display
pub fn format_overlap_warnings(overlaps: &[SourceOverlap], new_uri: &str) -> Vec<String> {
    overlaps.iter().map(|o| {
        match o.overlap_type {
            OverlapType::Identical => {
                format!(
                    "Source already exists: {} (id: {})",
                    o.existing_source.name.as_deref().unwrap_or(&o.existing_source.uri),
                    o.existing_source.id
                )
            }
            OverlapType::SubsetOf => {
                format!(
                    "⚠ New source '{}' is inside existing source '{}' (id: {}) - documents may be duplicated",
                    new_uri,
                    o.existing_source.name.as_deref().unwrap_or(&o.existing_source.uri),
                    o.existing_source.id
                )
            }
            OverlapType::SupersetOf => {
                format!(
                    "⚠ New source '{}' contains existing source '{}' (id: {}) - documents may be duplicated",
                    new_uri,
                    o.existing_source.name.as_deref().unwrap_or(&o.existing_source.uri),
                    o.existing_source.id
                )
            }
        }
    }).collect()
}

/// Ingest a local directory
pub async fn cmd_ingest_dir(
    config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
    path: &Path,
    name: Option<String>,
    operation: RunOperation,
    interactive: bool,
) -> Result<IngestStats> {
    let canonical_path = path
        .canonicalize()
        .map_err(|e| Error::InvalidPath(format!("{}: {}", path.display(), e)))?;

    let uri = canonical_path.display().to_string();
    info!("Ingesting directory: {}", uri);

    let mut stats = IngestStats::default();

    // Check for overlaps with existing sources
    let overlaps = check_dir_overlap(db, &canonical_path).await?;
    if !overlaps.is_empty() {
        stats.overlap_warnings = format_overlap_warnings(&overlaps, &uri);
        for warning in &stats.overlap_warnings {
            warn!("{}", warning);
        }
    }

    // Resolve source interactively on conflicts
    let source = resolve_source(db, SourceType::Dir, &uri, name.clone(), interactive).await?;

    // Start ingestion run
    let run = db.start_ingestion_run(&source.id, operation).await?;

    // Create embedder
    let embedder = create_embedder(&config.embedding)?;

    // Collect all files
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let walker = WalkBuilder::new(&canonical_path)
        .hidden(false)
        .git_ignore(true)
        .git_exclude(true)
        .build();

    for entry in walker {
        match entry {
            Ok(e) if e.file_type().map(|t| t.is_file()).unwrap_or(false) => {
                let path = e.path().to_path_buf();
                if !should_skip_file(&path) {
                    files.push(path);
                }
            }
            _ => {}
        }
    }

    info!("Found {} files to process", files.len());

    let mut current_uris: Vec<String> = Vec::new();
    let file_progress = start_progress_bar(files.len(), "Processing files");

    for file_path in files {
        let file_uri = file_path.display().to_string();
        current_uris.push(file_uri.clone());

        match process_file(config, db, store, embedder.as_ref(), &source, &file_path).await {
            Ok((created, updated)) => {
                stats.docs_processed += 1;
                stats.chunks_created += created;
                stats.chunks_updated += updated;
            }
            Err(e) => {
                let error_msg = format!("{}: {}", file_path.display(), e);
                warn!("{}", error_msg);
                stats.errors.push(error_msg);
                stats.docs_skipped += 1;
            }
        }

        advance_progress(&file_progress);
    }

    finish_progress(file_progress, "Files processed");

    // Delete stale documents
    let stale_ids = db.delete_stale_documents(&source.id, &current_uris).await?;
    if !stale_ids.is_empty() {
        info!("Deleted {} stale documents", stale_ids.len());
        // Get point IDs for stale docs and delete from Qdrant
        for doc_id in &stale_ids {
            if let Ok(chunks) = db.get_chunks(doc_id).await {
                let point_ids: Vec<Uuid> = chunks
                    .iter()
                    .filter_map(|c| Uuid::try_parse(&c.qdrant_point_id).ok())
                    .collect();
                if !point_ids.is_empty() {
                    if let Err(e) = store.delete_points(&point_ids).await {
                        warn!("Failed to delete Qdrant points: {}", e);
                    }
                }
            }
        }
    }

    // Complete ingestion run
    let errors = if stats.errors.is_empty() {
        None
    } else {
        Some(stats.errors.clone())
    };

    db.complete_ingestion_run(
        &run.id,
        if stats.errors.is_empty() {
            RunStatus::Completed
        } else {
            RunStatus::Failed
        },
        stats.docs_processed,
        stats.chunks_created,
        stats.chunks_updated,
        stats.chunks_deleted,
        errors,
    )
    .await?;

    info!(
        "Ingestion complete: {} docs, {} chunks created, {} chunks updated",
        stats.docs_processed, stats.chunks_created, stats.chunks_updated
    );

    Ok(stats)
}

/// Process a single file
async fn process_file(
    config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
    embedder: &dyn Embedder,
    source: &Source,
    path: &Path,
) -> Result<(i32, i32)> {
    let file_uri = path.display().to_string();
    debug!("Processing file: {}", file_uri);

    // Read file content
    let content = std::fs::read(path)?;

    // Skip binary files
    if is_binary_content(&content) {
        debug!("Skipping binary file: {}", file_uri);
        return Ok((0, 0));
    }

    // Convert to string
    let text = String::from_utf8_lossy(&content).to_string();
    let content_hash = compute_content_hash(text.as_bytes());

    // Check if content changed
    let existing_doc = db.get_document_by_uri(&source.id, &file_uri).await?;
    if let Some(existing_doc) = existing_doc.as_ref() {
        if existing_doc.content_hash == content_hash {
            debug!("File unchanged: {}", file_uri);
            return Ok((0, 0));
        }
    }
    let was_existing = existing_doc.is_some();

    // Detect content type and parse
    let content_type = ContentType::from_extension(path);
    let parsed = parse_content(&text, content_type, None)?;

    // Create/update document
    let mut doc = Document::new(source.id.clone(), file_uri.clone(), content_hash.clone());
    doc.title = parsed.title.clone();
    doc.content_type = Some(format!("{:?}", content_type).to_lowercase());
    let doc = db.upsert_document(&doc).await?;
    debug!(
        doc_id = %doc.id,
        source_id = %doc.source_id,
        existing = was_existing,
        uri = %doc.uri,
        "Upserted document for file ingestion"
    );

    // Chunk the document
    let chunks = chunk_document(&parsed, &content_hash, &config.chunk)?;

    if chunks.is_empty() {
        debug!("No chunks generated for: {}", file_uri);
        return Ok((0, 0));
    }

    // Process chunks
    let (created, updated) =
        process_chunks(config, db, store, embedder, source, &doc, &file_uri, chunks).await?;

    Ok((created, updated))
}

/// Process chunks for a document
async fn process_chunks(
    config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
    embedder: &dyn Embedder,
    source: &Source,
    doc: &Document,
    doc_uri: &str,
    chunks: Vec<TextChunk>,
) -> Result<(i32, i32)> {
    let mut created = 0i32;
    let mut updated = 0i32;
    let mut chunks_to_embed: Vec<(usize, TextChunk)> = Vec::new();
    let existing_chunks = db.get_chunks_by_modality(&doc.id, "text").await?;
    let existing_hashes: HashSet<String> = existing_chunks
        .iter()
        .map(|c| c.chunk_hash.clone())
        .collect();

    // Find chunks that need embedding
    for (i, chunk) in chunks.iter().enumerate() {
        if !existing_hashes.contains(&chunk.hash) {
            chunks_to_embed.push((i, chunk.clone()));
        }
    }

    if chunks_to_embed.is_empty() {
        debug!("All chunks unchanged for: {}", doc_uri);
        return Ok((0, 0));
    }

    debug!(
        "Embedding {} new/changed chunks for: {}",
        chunks_to_embed.len(),
        doc_uri
    );

    // Embed in batches
    let texts: Vec<String> = chunks_to_embed
        .iter()
        .map(|(_, c)| c.text.clone())
        .collect();
    let embeddings = embed_in_batches(embedder, texts, config.embedding.batch_size).await?;

    // Prepare points for Qdrant
    let mut points: Vec<ChunkPoint> = Vec::new();

    for ((chunk_index, chunk), embedding) in chunks_to_embed.iter().zip(embeddings.iter()) {
        let meta_chunk = Chunk::new(
            doc.id.clone(),
            *chunk_index as i32,
            chunk.hash.clone(),
            chunk.text.clone(),
            chunk.char_start as i32,
            chunk.char_end as i32,
            if chunk.headings.is_empty() {
                None
            } else {
                Some(chunk.headings.clone())
            },
        );

        // Save chunk to SQLite
        db.upsert_chunk(&meta_chunk).await?;

        // Create Qdrant payload (defaults to text modality)
        let mut payload = ChunkPayload::new(
            source.id.clone(),
            source.source_type.clone(),
            source.uri.clone(),
            doc.id.clone(),
            doc_uri.to_string(),
            *chunk_index as i32,
            chunk.hash.clone(),
            Utc::now().to_rfc3339(),
        );
        // Attach optional metadata
        payload.title = doc.title.clone();
        payload.headings = if chunk.headings.is_empty() {
            None
        } else {
            Some(chunk.headings.clone())
        };

        // Parse qdrant_point_id string to Uuid
        let point_id = Uuid::try_parse(&meta_chunk.qdrant_point_id).unwrap_or_else(|_| {
            Uuid::new_v5(&Uuid::NAMESPACE_OID, meta_chunk.qdrant_point_id.as_bytes())
        });

        points.push(ChunkPoint {
            id: point_id,
            vector: embedding.clone(),
            payload,
        });

        if existing_hashes.contains(&chunk.hash) {
            updated += 1;
        } else {
            created += 1;
        }
    }

    // Upsert to Qdrant
    store.upsert_points(points).await?;

    // Delete extra chunks if document shrunk
    let deleted_point_strings = db
        .delete_chunks_from_index(&doc.id, chunks.len() as i32)
        .await?;
    if !deleted_point_strings.is_empty() {
        let deleted_uuids: Vec<Uuid> = deleted_point_strings
            .iter()
            .filter_map(|s| Uuid::try_parse(s).ok())
            .collect();
        if !deleted_uuids.is_empty() {
            store.delete_points(&deleted_uuids).await?;
        }
    }

    Ok((created, updated))
}

/// Ingest from a URL
pub async fn cmd_ingest_url(
    config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
    url: &str,
    name: Option<String>,
    overrides: CrawlOverrides,
    operation: RunOperation,
    interactive: bool,
) -> Result<IngestStats> {
    info!("Ingesting URL: {}", url);

    let mut stats = IngestStats::default();

    // Check for overlaps with existing sources
    let overlaps = check_url_overlap(db, url).await?;
    if !overlaps.is_empty() {
        stats.overlap_warnings = format_overlap_warnings(&overlaps, url);
        for warning in &stats.overlap_warnings {
            warn!("{}", warning);
        }
    }

    // Resolve source interactively on conflicts
    let source = resolve_source(db, SourceType::Url, url, name.clone(), interactive).await?;

    // Start ingestion run
    let run = db.start_ingestion_run(&source.id, operation).await?;

    // Create embedder
    let embedder = create_embedder(&config.embedding)?;

    // Build crawl config with CLI overrides
    let mut crawl_config = config.crawl.clone();
    if let Some(max_pages) = overrides.max_pages {
        crawl_config.max_pages = max_pages;
    }
    if let Some(max_depth) = overrides.max_depth {
        crawl_config.max_depth = max_depth;
    }
    if overrides.path_prefix.is_some() {
        crawl_config.path_prefix = overrides.path_prefix;
    }

    // Create crawler
    let crawler = Crawler::new(crawl_config)?;

    let mut current_uris: Vec<String> = Vec::new();

    // Crawl and process pages
    let pages = crawler
        .crawl(url, |_page| {
            // Continue callback - return true to keep crawling
            true
        })
        .await?;

    let page_progress = start_progress_bar(pages.len(), "Processing pages");

    for page in pages {
        current_uris.push(page.url.clone());

        match process_page(config, db, store, embedder.as_ref(), &source, &page).await {
            Ok((created, updated)) => {
                stats.docs_processed += 1;
                stats.chunks_created += created;
                stats.chunks_updated += updated;
            }
            Err(e) => {
                let error_msg = format!("{}: {}", page.url, e);
                warn!("{}", error_msg);
                stats.errors.push(error_msg);
                stats.docs_skipped += 1;
            }
        }

        advance_progress(&page_progress);
    }

    finish_progress(page_progress, "Pages processed");

    // Delete stale documents
    let stale_ids = db.delete_stale_documents(&source.id, &current_uris).await?;
    if !stale_ids.is_empty() {
        info!("Deleted {} stale documents", stale_ids.len());
        for doc_id in &stale_ids {
            if let Ok(chunks) = db.get_chunks(doc_id).await {
                let point_ids: Vec<Uuid> = chunks
                    .iter()
                    .filter_map(|c| Uuid::try_parse(&c.qdrant_point_id).ok())
                    .collect();
                if !point_ids.is_empty() {
                    if let Err(e) = store.delete_points(&point_ids).await {
                        warn!("Failed to delete Qdrant points: {}", e);
                    }
                }
            }
        }
    }

    // Complete ingestion run
    let errors = if stats.errors.is_empty() {
        None
    } else {
        Some(stats.errors.clone())
    };

    db.complete_ingestion_run(
        &run.id,
        if stats.errors.is_empty() {
            RunStatus::Completed
        } else {
            RunStatus::Failed
        },
        stats.docs_processed,
        stats.chunks_created,
        stats.chunks_updated,
        stats.chunks_deleted,
        errors,
    )
    .await?;

    info!(
        "Ingestion complete: {} docs, {} chunks created, {} chunks updated",
        stats.docs_processed, stats.chunks_created, stats.chunks_updated
    );

    Ok(stats)
}

/// Ingest from a sitemap URL
pub async fn cmd_ingest_sitemap(
    config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
    sitemap_url: &str,
    name: Option<String>,
    max_pages: Option<u32>,
    operation: RunOperation,
    interactive: bool,
) -> Result<IngestStats> {
    use crate::crawl::SitemapParser;

    info!("Ingesting sitemap: {}", sitemap_url);

    let mut stats = IngestStats::default();

    // Parse sitemap to get URLs
    let parser = SitemapParser::new(&config.crawl.user_agent)?;
    let entries = parser.parse(sitemap_url).await?;

    if entries.is_empty() {
        warn!("No URLs found in sitemap: {}", sitemap_url);
        return Ok(stats);
    }

    let max = max_pages.unwrap_or(config.crawl.max_pages);
    let entries: Vec<_> = entries.into_iter().take(max as usize).collect();
    info!(
        "Found {} URLs in sitemap (limited to {})",
        entries.len(),
        max
    );

    // Check for overlaps - use the first entry URL as representative for the sitemap domain
    if let Some(first_entry) = entries.first() {
        let overlaps = check_url_overlap(db, &first_entry.loc).await?;
        if !overlaps.is_empty() {
            stats.overlap_warnings = format_overlap_warnings(&overlaps, sitemap_url);
            for warning in &stats.overlap_warnings {
                warn!("{}", warning);
            }
        }
    }

    // Resolve source interactively on conflicts
    let source = resolve_source(
        db,
        SourceType::Sitemap,
        sitemap_url,
        name.clone(),
        interactive,
    )
    .await?;

    // Start ingestion run
    let run = db.start_ingestion_run(&source.id, operation).await?;

    // Create embedder and crawler
    let embedder = create_embedder(&config.embedding)?;
    let crawler = Crawler::new(config.crawl.clone())?;

    let mut current_uris: Vec<String> = Vec::new();
    let url_progress = start_progress_bar(entries.len(), "Processing URLs");

    // Process each URL from sitemap
    for entry in entries {
        current_uris.push(entry.loc.clone());

        // Fetch the page
        match crawler.fetch(&entry.loc).await {
            Ok(page) => {
                match process_page(config, db, store, embedder.as_ref(), &source, &page).await {
                    Ok((created, updated)) => {
                        stats.docs_processed += 1;
                        stats.chunks_created += created;
                        stats.chunks_updated += updated;
                    }
                    Err(e) => {
                        let error_msg = format!("{}: {}", entry.loc, e);
                        warn!("{}", error_msg);
                        stats.errors.push(error_msg);
                        stats.docs_skipped += 1;
                    }
                }
            }
            Err(e) => {
                let error_msg = format!("{}: {}", entry.loc, e);
                warn!("{}", error_msg);
                stats.errors.push(error_msg);
                stats.docs_skipped += 1;
            }
        }

        advance_progress(&url_progress);
    }

    finish_progress(url_progress, "URLs processed");

    // Delete stale documents
    let stale_ids = db.delete_stale_documents(&source.id, &current_uris).await?;
    if !stale_ids.is_empty() {
        info!("Deleted {} stale documents", stale_ids.len());
        for doc_id in &stale_ids {
            if let Ok(chunks) = db.get_chunks(doc_id).await {
                let point_ids: Vec<Uuid> = chunks
                    .iter()
                    .filter_map(|c| Uuid::try_parse(&c.qdrant_point_id).ok())
                    .collect();
                if !point_ids.is_empty() {
                    if let Err(e) = store.delete_points(&point_ids).await {
                        warn!("Failed to delete Qdrant points: {}", e);
                    }
                }
            }
        }
    }

    // Complete ingestion run
    let errors = if stats.errors.is_empty() {
        None
    } else {
        Some(stats.errors.clone())
    };

    db.complete_ingestion_run(
        &run.id,
        if stats.errors.is_empty() {
            RunStatus::Completed
        } else {
            RunStatus::Failed
        },
        stats.docs_processed,
        stats.chunks_created,
        stats.chunks_updated,
        stats.chunks_deleted,
        errors,
    )
    .await?;

    info!(
        "Sitemap ingestion complete: {} docs, {} chunks created, {} chunks updated",
        stats.docs_processed, stats.chunks_created, stats.chunks_updated
    );

    Ok(stats)
}

/// Process a crawled page
async fn process_page(
    config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
    embedder: &dyn Embedder,
    source: &Source,
    page: &CrawledPage,
) -> Result<(i32, i32)> {
    debug!("Processing page: {}", page.url);

    let content_hash = compute_content_hash(page.content.as_bytes());

    // Check if content changed
    let existing_doc = db.get_document_by_uri(&source.id, &page.url).await?;
    if let Some(existing_doc) = existing_doc.as_ref() {
        if existing_doc.content_hash == content_hash {
            debug!("Page unchanged: {}", page.url);
            return Ok((0, 0));
        }
    }
    let was_existing = existing_doc.is_some();

    // Parse content
    let parsed = parse_content(&page.content, page.content_type, Some(&page.url))?;

    // Create/update document
    let mut doc = Document::new(source.id.clone(), page.url.clone(), content_hash.clone());
    doc.title = page.title.clone().or(parsed.title.clone());
    doc.content_type = Some(format!("{:?}", page.content_type).to_lowercase());
    let doc = db.upsert_document(&doc).await?;
    debug!(
        doc_id = %doc.id,
        source_id = %doc.source_id,
        existing = was_existing,
        uri = %doc.uri,
        "Upserted document for page ingestion"
    );

    // Chunk the document
    let chunks = chunk_document(&parsed, &content_hash, &config.chunk)?;

    // Multimodal image selection + caching (optional)
    let images = select_image_candidates(config, &parsed);
    let cached_images = if images.is_empty() {
        Vec::new()
    } else {
        debug!(count = images.len(), uri = %page.url, "Selected image candidates for ingestion");
        for (m, score) in &images {
            debug!(url = %m.url, css = m.css_background, score, "Image candidate accepted");
        }
        fetch_and_cache_images(config, &images).await
    };

    if chunks.is_empty() {
        debug!("No chunks generated for: {}", page.url);
        if cached_images.is_empty() {
            return Ok((0, 0));
        }

        let (image_created, image_updated) = match embed_cached_images(
            config,
            db,
            store,
            embedder,
            source,
            &doc,
            &page.url,
            cached_images,
        )
        .await
        {
            Ok(counts) => counts,
            Err(e) => {
                warn!(uri = %page.url, "Failed to embed images: {}", e);
                (0, 0)
            }
        };

        return Ok((image_created, image_updated));
    }

    // Process text chunks
    let (created, updated) =
        process_chunks(config, db, store, embedder, source, &doc, &page.url, chunks).await?;

    // Embed cached images after text processing
    let (image_created, image_updated) = if cached_images.is_empty() {
        (0, 0)
    } else {
        match embed_cached_images(
            config,
            db,
            store,
            embedder,
            source,
            &doc,
            &page.url,
            cached_images,
        )
        .await
        {
            Ok(counts) => counts,
            Err(e) => {
                warn!(uri = %page.url, "Failed to embed images: {}", e);
                (0, 0)
            }
        }
    };

    Ok((created + image_created, updated + image_updated))
}

fn start_progress_bar(len: usize, message: &str) -> Option<ProgressBar> {
    if len == 0 {
        return None;
    }

    let pb = add_progress_bar(len as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} {msg}",
        )
        .unwrap()
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
    );
    pb.set_message(message.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    Some(pb)
}

fn advance_progress(pb: &Option<ProgressBar>) {
    if let Some(pb) = pb {
        pb.inc(1);
    }
}

fn finish_progress(pb: Option<ProgressBar>, message: &str) {
    if let Some(pb) = pb {
        pb.finish_with_message(message.to_string());
    }
}

async fn resolve_source(
    db: &MetaDb,
    source_type: SourceType,
    uri: &str,
    desired_name: Option<String>,
    interactive: bool,
) -> Result<Source> {
    if interactive {
        return resolve_source_interactive(db, source_type, uri, desired_name).await;
    }

    if let Some(existing) = db.get_source_by_uri(uri).await? {
        return Ok(existing);
    }

    let source = Source::new(source_type, uri.to_string(), desired_name);
    db.insert_source(&source).await?;
    Ok(source)
}

async fn resolve_source_interactive(
    db: &MetaDb,
    source_type: SourceType,
    uri: &str,
    desired_name: Option<String>,
) -> Result<Source> {
    // Case 1: Source with same URI already exists
    if let Some(existing) = db.get_source_by_uri(uri).await? {
        println!(
            "A source with this URI already exists: {} (name: {})",
            existing.id,
            existing.name.as_deref().unwrap_or(&existing.uri)
        );
        println!("Choose an action:");
        println!("  [1] Use existing source");
        println!("  [2] Rename existing source");
        println!("  [3] Abort");
        let choice = prompt_choice(1, 3);
        match choice {
            1 => {
                return Ok(existing);
            }
            2 => {
                let new_name = prompt_string(
                    "Enter new name for existing source",
                    existing.name.as_deref(),
                );
                db.update_source_name(&existing.id, Some(new_name)).await?;
                let updated = db.get_source(&existing.id).await?.unwrap();
                return Ok(updated);
            }
            _ => {
                return Err(Error::Config("Ingestion aborted".to_string()));
            }
        }
    }

    // Case 2: Name collision with another source
    let mut final_name = desired_name.clone();
    if let Some(ref name) = desired_name {
        if let Some(named) = db.get_source_by_name(name).await? {
            println!(
                "Another source already uses the name '{}': {} (URI: {})",
                name, named.id, named.uri
            );
            println!("Choose an action:");
            println!("  [1] Keep duplicate name");
            println!("  [2] Enter a new name for this source");
            println!("  [3] Rename the existing source");
            println!("  [4] Abort");
            let choice = prompt_choice(1, 4);
            match choice {
                1 => {}
                2 => {
                    let new_name = prompt_string("Enter new name", None);
                    final_name = Some(new_name);
                }
                3 => {
                    let new_name = prompt_string("Enter new name for existing source", None);
                    db.update_source_name(&named.id, Some(new_name)).await?;
                }
                _ => return Err(Error::Config("Ingestion aborted".to_string())),
            }
        }
    }

    let s = Source::new(
        source_type,
        uri.to_string(),
        final_name.or(Some(uri.to_string())),
    );
    db.insert_source(&s).await?;
    info!("Created new source: {}", s.id);
    Ok(s)
}

fn prompt_choice(default: usize, max: usize) -> usize {
    use std::io::{self, Write};
    loop {
        print!("Enter choice [{}-{}] (default {}): ", 1, max, default);
        let _ = io::stdout().flush();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_ok() {
            let trimmed = input.trim();
            if trimmed.is_empty() {
                return default;
            }
            if let Ok(n) = trimmed.parse::<usize>() {
                if n >= 1 && n <= max {
                    return n;
                }
            }
        }
        println!(
            "Invalid input. Please enter a number between {} and {}.",
            1, max
        );
    }
}

fn prompt_string(prompt: &str, default: Option<&str>) -> String {
    use std::io::{self, Write};
    loop {
        match default {
            Some(d) => {
                print!("{} [{}]: ", prompt, d);
            }
            None => {
                print!("{}: ", prompt);
            }
        }
        let _ = io::stdout().flush();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_ok() {
            let trimmed = input.trim();
            if trimmed.is_empty() {
                if let Some(d) = default {
                    return d.to_string();
                }
            } else {
                return trimmed.to_string();
            }
        }
        println!("Invalid input. Please try again.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{ContentType, ExtractedMedia, Heading, ParsedDocument};

    fn multimodal_config() -> Config {
        let mut config = Config::default();
        config.crawl.multimodal.enabled = true;
        config.embedding.model = "jinaai/jina-clip-v2".to_string();
        config.crawl.multimodal.min_relevance_score = 0.0;
        config
    }

    #[test]
    fn test_select_image_candidates_dedupes_urls() {
        let config = multimodal_config();
        let mut doc = ParsedDocument::new("text".to_string(), ContentType::Html);
        doc.title = Some("Architecture".to_string());
        doc.headings.push(Heading {
            level: 1,
            text: "Architecture".to_string(),
            position: 0,
        });
        doc.media = vec![
            ExtractedMedia {
                url: "https://example.com/diagram.png".to_string(),
                alt: Some("Architecture diagram".to_string()),
                tag: "img".to_string(),
                css_background: false,
            },
            ExtractedMedia {
                url: "https://example.com/diagram.png".to_string(),
                alt: Some("Diagram".to_string()),
                tag: "img".to_string(),
                css_background: false,
            },
        ];

        let candidates = select_image_candidates(&config, &doc);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0.url, "https://example.com/diagram.png");
    }

    #[test]
    fn test_select_image_candidates_respects_css_toggle() {
        let mut config = multimodal_config();
        config.crawl.multimodal.include_css_background_images = false;

        let mut doc = ParsedDocument::new("text".to_string(), ContentType::Html);
        doc.media = vec![ExtractedMedia {
            url: "https://example.com/bg.png".to_string(),
            alt: Some("Background".to_string()),
            tag: "div".to_string(),
            css_background: true,
        }];

        let candidates = select_image_candidates(&config, &doc);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_select_image_candidates_skips_late_interaction() {
        let mut config = multimodal_config();
        config.embedding.model = "vidore/colpali".to_string();

        let mut doc = ParsedDocument::new("text".to_string(), ContentType::Html);
        doc.media = vec![ExtractedMedia {
            url: "https://example.com/diagram.png".to_string(),
            alt: Some("Diagram".to_string()),
            tag: "img".to_string(),
            css_background: false,
        }];

        let candidates = select_image_candidates(&config, &doc);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_url_is_allowed_image_filters_svg() {
        let allowed = vec!["image/".to_string()];
        assert!(!url_is_allowed_image("https://example.com/icon.svg", &allowed));
        assert!(url_is_allowed_image("https://example.com/photo.png", &allowed));
    }

    #[test]
    fn test_is_svg_url_detects_svg_and_svgz() {
        assert!(is_svg_url("https://example.com/icon.svg"));
        assert!(is_svg_url("https://example.com/icon.svgz"));
        assert!(!is_svg_url("https://example.com/photo.png"));
    }

    #[test]
    fn test_perceptual_duplicate_threshold() {
        let base: u64 = 0b1010_1010;
        let near = base ^ 0b11; // distance 2
        let far = base ^ 0b1111_1111; // distance 8
        let seen = vec![base];

        assert!(is_perceptual_duplicate(near, &seen));
        assert!(!is_perceptual_duplicate(far, &seen));
    }
}
