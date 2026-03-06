//! Ingest command implementation

use crate::chunk::{chunk_document, compute_content_hash, TextChunk};
use crate::config::{AudioConfig, AudioTranscriptionBackend, Config, ResolvedEmbeddingConfig};
use crate::crawl::{validate_url_ssrf, CrawledPage, Crawler};
use crate::embed::{embed_images_with_optional_text_fusion, embed_in_batches, Embedder};
use crate::error::{Error, Result};
use crate::meta::{Chunk, Document, MetaDb, RunOperation, RunStatus, Source, SourceType};
use crate::parse::{is_audio_file, is_video_file, ExtractedMedia, ParsedDocument};
use crate::parse::{is_binary_content, parse_content, should_skip_file, ContentType};
use crate::progress::add_progress_bar;
use crate::store::{ChunkPayload, ChunkPoint, QdrantStore};
use crate::xinference::{
    ensure_xinference_ready, get_or_init_xinference_manager, xinference_auth_token,
    DEFAULT_XINFERENCE_PORT,
};
use base64::Engine;
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
        if doc
            .headings
            .iter()
            .any(|h| alt_lower.contains(&h.text.to_lowercase()))
        {
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

fn build_image_context(doc: &ParsedDocument, media: &ExtractedMedia) -> Option<String> {
    let mut parts = Vec::new();

    if let Some(ref alt) = media.alt {
        let trimmed = alt.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }

    if let Some(ref title) = doc.title {
        let trimmed = title.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }

    if !doc.headings.is_empty() {
        let headings = doc
            .headings
            .iter()
            .map(|h| h.text.trim())
            .filter(|h| !h.is_empty())
            .collect::<Vec<_>>();
        if !headings.is_empty() {
            parts.push(headings.join(" > "));
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" | "))
    }
}

/// Select image candidates according to config thresholds and limits
fn select_image_candidates(
    config: &Config,
    embedding: &ResolvedEmbeddingConfig,
    doc: &ParsedDocument,
) -> Vec<(ExtractedMedia, f32)> {
    let mm = &config.crawl.multimodal;
    if !mm.enabled || !mm.include_images {
        return Vec::new();
    }
    if !embedding.supports_image_inputs() {
        debug!(model = %embedding.model_id, "Skipping image candidates (model not multimodal)");
        return Vec::new();
    }
    if embedding.supports_multi_vector {
        debug!(model = %embedding.model_id, "Skipping image candidates (late-interaction strategy not supported)");
        return Vec::new();
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

/// Select audio candidates from parsed document media
///
/// Note: Currently infrastructure for future web crawl audio support.
/// Web crawl audio processing is planned for a future release.
#[allow(dead_code)]
fn select_audio_candidates(config: &Config, doc: &ParsedDocument) -> Vec<ExtractedMedia> {
    use crate::parse::MediaModality;

    let mm = &config.crawl.multimodal;
    select_media_candidates(
        config,
        doc,
        MediaModality::Audio,
        &mm.audio.allowed_mime_types,
        mm.include_audio,
        "audio",
    )
}

/// Select video candidates from parsed document media
///
/// Note: Currently infrastructure for future web crawl video support.
/// Web crawl video processing is planned for a future release.
#[allow(dead_code)]
fn select_video_candidates(config: &Config, doc: &ParsedDocument) -> Vec<ExtractedMedia> {
    use crate::parse::MediaModality;

    let mm = &config.crawl.multimodal;
    select_media_candidates(
        config,
        doc,
        MediaModality::Video,
        &mm.video.allowed_mime_types,
        mm.include_video,
        "video",
    )
}

fn select_media_candidates(
    config: &Config,
    doc: &ParsedDocument,
    modality: crate::parse::MediaModality,
    allowed_mime_types: &[String],
    enabled_for_modality: bool,
    label: &str,
) -> Vec<ExtractedMedia> {
    let mm = &config.crawl.multimodal;
    if !mm.enabled || !enabled_for_modality {
        return Vec::new();
    }

    let mut candidates: Vec<ExtractedMedia> = Vec::new();
    let mut seen_urls: HashSet<String> = HashSet::new();

    for m in &doc.media {
        if m.modality != modality {
            continue;
        }

        // Check MIME type if available
        if let Some(ref mime) = m.mime_type {
            if !allowed_mime_types
                .iter()
                .any(|allowed| mime.starts_with(allowed))
            {
                debug!(url = %m.url, mime = %mime, modality = %label, "Rejected media candidate (MIME not allowed)");
                continue;
            }
        }

        // Dedupe by URL
        let key = normalize_media_url(&m.url);
        if seen_urls.contains(&key) {
            continue;
        }
        seen_urls.insert(key);

        candidates.push(m.clone());

        // Limit to max_assets_per_page
        if candidates.len() >= mm.max_assets_per_page {
            break;
        }
    }

    debug!(count = candidates.len(), modality = %label, "Selected media candidates");
    candidates
}

// =============================================================================
// Audio/Video Processing Pipeline
// =============================================================================

/// Metadata extracted from audio/video files via ffprobe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaMetadata {
    /// Duration in seconds
    pub duration_secs: f64,
    /// Format name (e.g., "mp3", "wav", "mp4")
    pub format_name: String,
    /// Bitrate in bits/s (optional)
    pub bit_rate: Option<i64>,
    /// Number of audio streams
    pub audio_streams: usize,
    /// Number of video streams  
    pub video_streams: usize,
    /// Sample rate for audio (Hz)
    pub sample_rate: Option<i64>,
    /// Channel layout (e.g., "stereo", "mono")
    pub channels: Option<i32>,
}

/// Run ffprobe to extract metadata from a media file
async fn extract_media_metadata(path: &Path) -> Result<MediaMetadata> {
    use tokio::process::Command;

    let output = Command::new("ffprobe")
        .args([
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path.to_str().ok_or_else(|| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Invalid path encoding",
                ))
            })?,
        ])
        .output()
        .await
        .map_err(|e| {
            Error::Io(std::io::Error::other(format!(
                "Failed to run ffprobe: {}. Is ffmpeg installed?",
                e
            )))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::Io(std::io::Error::other(format!(
            "ffprobe failed: {}",
            stderr
        ))));
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).map_err(|e| {
        Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse ffprobe output: {}", e),
        ))
    })?;

    // Extract format info
    let format = json.get("format").ok_or_else(|| {
        Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "ffprobe output missing format info",
        ))
    })?;

    let duration_secs = format
        .get("duration")
        .and_then(|d| d.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);

    let format_name = format
        .get("format_name")
        .and_then(|f| f.as_str())
        .unwrap_or("unknown")
        .to_string();

    let bit_rate = format
        .get("bit_rate")
        .and_then(|b| b.as_str())
        .and_then(|s| s.parse::<i64>().ok());

    // Count streams by type
    let streams = json
        .get("streams")
        .and_then(|s| s.as_array())
        .cloned()
        .unwrap_or_default();

    let mut audio_streams = 0;
    let mut video_streams = 0;
    let mut sample_rate = None;
    let mut channels = None;

    for stream in &streams {
        let codec_type = stream.get("codec_type").and_then(|c| c.as_str());
        match codec_type {
            Some("audio") => {
                audio_streams += 1;
                if sample_rate.is_none() {
                    sample_rate = stream
                        .get("sample_rate")
                        .and_then(|s| s.as_str())
                        .and_then(|s| s.parse::<i64>().ok());
                }
                if channels.is_none() {
                    channels = stream
                        .get("channels")
                        .and_then(|c| c.as_i64())
                        .map(|c| c as i32);
                }
            }
            Some("video") => video_streams += 1,
            _ => {}
        }
    }

    Ok(MediaMetadata {
        duration_secs,
        format_name,
        bit_rate,
        audio_streams,
        video_streams,
        sample_rate,
        channels,
    })
}

/// Transcription result from the API
#[derive(Debug, Clone, Deserialize)]
struct TranscriptionResponse {
    text: String,
}

#[derive(Debug, Clone)]
struct AudioTranscriber {
    backend: AudioTranscriptionBackend,
    transcription_url: Url,
    model: String,
    auth_token: Option<String>,
}

impl AudioTranscriber {
    async fn from_config(config: &Config) -> Result<Option<Self>> {
        let audio_config = &config.crawl.multimodal.audio;
        if !audio_config.transcription_enabled {
            return Ok(None);
        }

        let backend = resolve_transcription_backend(audio_config);
        let configured_model = resolve_transcription_model(audio_config, backend).to_string();

        match backend {
            AudioTranscriptionBackend::Http => Ok(Some(Self {
                backend,
                transcription_url: normalize_http_transcription_url(
                    &audio_config.transcription_url,
                )?,
                model: configured_model,
                auth_token: None,
            })),
            AudioTranscriptionBackend::Xinference => {
                let base_url =
                    normalize_xinference_transcription_base_url(&audio_config.transcription_url)?;
                let auth_token = xinference_auth_token();
                let manager_lock =
                    get_or_init_xinference_manager(&base_url, auth_token.clone()).await?;

                let needs_prepare = {
                    let mut guard = manager_lock.lock().await;
                    let mgr = guard.as_mut().ok_or_else(|| {
                        Error::Config("Xinference manager not initialized".to_string())
                    })?;
                    !mgr.health_check().await? && is_localhost_url(&base_url)
                };
                if needs_prepare {
                    ensure_xinference_ready(&config.paths.base_dir)?;
                }

                let model_uid = {
                    let mut guard = manager_lock.lock().await;
                    let mgr = guard.as_mut().ok_or_else(|| {
                        Error::Config("Xinference manager not initialized".to_string())
                    })?;
                    mgr.ensure_running().await?;
                    mgr.ensure_model_launched(&configured_model, "audio", false)
                        .await?
                };

                Ok(Some(Self {
                    backend,
                    transcription_url: base_url.join("/v1/audio/transcriptions").map_err(|e| {
                        Error::Config(format!("Invalid Xinference transcription URL: {}", e))
                    })?,
                    model: model_uid,
                    auth_token,
                }))
            }
            AudioTranscriptionBackend::Auto => unreachable!("auto backend is resolved earlier"),
        }
    }

    async fn transcribe(&self, path: &Path, http_client: &reqwest::Client) -> Result<String> {
        let file_bytes = tokio::fs::read(path).await?;
        let filename = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("audio.bin")
            .to_string();
        let mime_type = mime_guess::from_path(path).first_or_octet_stream();

        let part = reqwest::multipart::Part::bytes(file_bytes)
            .file_name(filename)
            .mime_str(mime_type.essence_str())?;

        let form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("model", self.model.clone());

        debug!(
            backend = %self.backend,
            model = %self.model,
            url = %self.transcription_url,
            path = %path.display(),
            "Submitting audio transcription request"
        );

        let mut request = http_client
            .post(self.transcription_url.clone())
            .multipart(form)
            .timeout(Duration::from_secs(300));
        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Io(std::io::Error::other(format!(
                "Transcription API error {}: {}",
                status, body
            ))));
        }

        let result: TranscriptionResponse = response.json().await?;
        Ok(result.text)
    }
}

fn resolve_transcription_backend(audio_config: &AudioConfig) -> AudioTranscriptionBackend {
    match audio_config.transcription_backend {
        AudioTranscriptionBackend::Auto => {
            if transcription_url_looks_like_xinference(&audio_config.transcription_url) {
                AudioTranscriptionBackend::Xinference
            } else {
                AudioTranscriptionBackend::Http
            }
        }
        backend => backend,
    }
}

fn transcription_url_looks_like_xinference(raw: &str) -> bool {
    Url::parse(raw)
        .ok()
        .map(|url| {
            matches!(
                url.host_str(),
                Some("127.0.0.1") | Some("localhost") | Some("0.0.0.0") | Some("::1")
            ) && url.port_or_known_default() == Some(DEFAULT_XINFERENCE_PORT)
        })
        .unwrap_or(false)
}

fn normalize_http_transcription_url(raw: &str) -> Result<Url> {
    Url::parse(raw)
        .map_err(|e| Error::Config(format!("Invalid transcription URL '{}': {}", raw, e)))
}

fn normalize_xinference_transcription_base_url(raw: &str) -> Result<Url> {
    let mut url = Url::parse(raw)
        .map_err(|e| Error::Config(format!("Invalid Xinference URL '{}': {}", raw, e)))?;
    match url.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(Error::Config(format!(
                "Unsupported Xinference URL scheme '{}'; use http or https",
                scheme
            )));
        }
    }

    if url.port().is_none() {
        url.set_port(Some(DEFAULT_XINFERENCE_PORT))
            .map_err(|_| Error::Config("Failed to apply default Xinference port".to_string()))?;
    }

    let path = url.path().trim_end_matches('/');
    match path {
        "" | "/" | "/v1" | "/v1/audio/transcriptions" => url.set_path("/"),
        other => {
            return Err(Error::Config(format!(
                "Xinference transcription URL must be a base URL or /v1/audio/transcriptions endpoint (got '{}')",
                other
            )));
        }
    }

    url.set_query(None);
    url.set_fragment(None);
    Ok(url)
}

fn resolve_transcription_model(
    audio_config: &AudioConfig,
    backend: AudioTranscriptionBackend,
) -> &str {
    let configured = audio_config.transcription_model.trim();
    if !configured.eq_ignore_ascii_case("auto") {
        return configured;
    }

    match backend {
        AudioTranscriptionBackend::Http => "whisper-1",
        AudioTranscriptionBackend::Xinference => {
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                "whisper-large-v3-turbo-mlx"
            } else {
                "whisper-large-v3-turbo"
            }
        }
        AudioTranscriptionBackend::Auto => "auto",
    }
}

fn is_localhost_url(url: &Url) -> bool {
    matches!(
        url.host_str(),
        Some("127.0.0.1") | Some("localhost") | Some("0.0.0.0") | Some("::1")
    )
}

/// Process an audio file: extract metadata, transcribe, and create chunks
#[allow(clippy::too_many_arguments)]
async fn process_audio_file(
    config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
    embedder: &dyn Embedder,
    source: &Source,
    path: &Path,
    transcriber: Option<&AudioTranscriber>,
    http_client: &reqwest::Client,
) -> Result<(i32, i32)> {
    let file_uri = path.display().to_string();
    let audio_config = &config.crawl.multimodal.audio;

    debug!(path = %file_uri, "Processing audio file");

    // Extract metadata with ffprobe
    let metadata = extract_media_metadata(path).await?;
    debug!(
        path = %file_uri,
        duration = metadata.duration_secs,
        format = %metadata.format_name,
        "Extracted audio metadata"
    );

    // Check duration limit
    if metadata.duration_secs > audio_config.max_duration_secs as f64 {
        debug!(
            path = %file_uri,
            duration = metadata.duration_secs,
            max = audio_config.max_duration_secs,
            "Skipping audio file (exceeds max duration)"
        );
        return Ok((0, 0));
    }

    // Compute content hash from file
    let file_bytes = tokio::fs::read(path).await?;
    let content_hash = compute_content_hash(&file_bytes);

    // Check if content changed
    let existing_doc = db.get_document_by_uri(&source.id, &file_uri).await?;
    if let Some(ref doc) = existing_doc {
        if doc.content_hash == content_hash {
            debug!(path = %file_uri, "Audio file unchanged");
            return Ok((0, 0));
        }
    }

    // Create/update document
    let mut doc = Document::new(source.id.clone(), file_uri.clone(), content_hash.clone());
    doc.title = Some(
        path.file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("audio")
            .to_string(),
    );
    doc.content_type = Some(format!("audio/{}", metadata.format_name));
    let doc = db.upsert_document(&doc).await?;

    // Transcribe if enabled
    let transcript = if let Some(transcriber) = transcriber {
        match transcriber.transcribe(path, http_client).await {
            Ok(text) => {
                debug!(
                    path = %file_uri,
                    chars = text.len(),
                    "Transcribed audio"
                );
                Some(text)
            }
            Err(e) => {
                warn!(path = %file_uri, error = %e, "Failed to transcribe audio");
                None
            }
        }
    } else {
        None
    };

    // If no transcript, create a minimal metadata chunk
    let chunk_text = transcript.unwrap_or_else(|| {
        format!(
            "[Audio: {} | Duration: {:.1}s | Format: {}]",
            path.file_name().and_then(|f| f.to_str()).unwrap_or("audio"),
            metadata.duration_secs,
            metadata.format_name
        )
    });

    // Create chunk with audio modality
    let chunk_hash = compute_content_hash(chunk_text.as_bytes());
    let media_hash = Some(content_hash.clone());

    let chunk = Chunk::new_with_modality(
        doc.id.clone(),
        0,
        chunk_hash.clone(),
        chunk_text.clone(),
        "audio",
        Some(file_uri.clone()),
        media_hash,
    );

    // Check if chunk exists
    let existing = db.get_chunk_by_hash(&doc.id, &chunk_hash).await?;
    if existing.is_some() {
        debug!(path = %file_uri, "Audio chunk unchanged");
        return Ok((0, 0));
    }

    // Delete old audio chunks for this document
    let old_chunks = db.get_chunks_by_modality(&doc.id, "audio").await?;
    if !old_chunks.is_empty() {
        let point_ids: Vec<Uuid> = old_chunks.iter().map(|c| c.point_uuid()).collect();
        if !point_ids.is_empty() {
            store.delete_points(&point_ids).await?;
        }
        db.delete_chunks_by_modality(&doc.id, "audio").await?;
    }

    // Embed the transcript text (audio chunks use text embeddings)
    let embeddings = embed_in_batches(
        embedder,
        vec![chunk_text.clone()],
        config.embedding.batch_size,
    )
    .await?;
    if embeddings.is_empty() {
        warn!(path = %file_uri, "Failed to embed audio transcript");
        return Ok((0, 0));
    }

    // Create Qdrant point
    let point_id = chunk.point_uuid();

    let mut payload = ChunkPayload::new(
        source.id.clone(),
        source.source_type.clone(),
        source.uri.clone(),
        doc.id.clone(),
        file_uri.clone(),
        0, // chunk_index
        chunk_hash.clone(),
        Utc::now().to_rfc3339(),
    );
    payload.title = doc.title.clone();
    payload.modality = Some("audio".to_string());
    payload.media_url = Some(file_uri.clone());
    payload.media_hash = Some(content_hash.clone());

    let points = vec![ChunkPoint {
        id: point_id,
        vector: embeddings[0].clone(),
        payload,
    }];

    store.upsert_points(points).await?;
    db.upsert_chunk(&chunk).await?;

    debug!(
        path = %file_uri,
        duration = metadata.duration_secs,
        "Processed audio file"
    );

    Ok((1, 0))
}

/// Extract keyframes from a video file using ffmpeg
async fn extract_keyframes(
    video_path: &Path,
    output_dir: &Path,
    max_keyframes: usize,
) -> Result<Vec<PathBuf>> {
    use tokio::process::Command;

    // Create output directory if it doesn't exist
    tokio::fs::create_dir_all(output_dir).await?;

    // Extract keyframes using ffmpeg's select filter for I-frames
    // Output pattern: output_dir/keyframe_%03d.jpg
    let output_pattern = output_dir.join("keyframe_%03d.jpg");

    let output = Command::new("ffmpeg")
        .args([
            "-i",
            video_path.to_str().ok_or_else(|| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Invalid video path encoding",
                ))
            })?,
            "-vf",
            "select=eq(pict_type\\,I),scale=640:-1",
            "-frames:v",
            &max_keyframes.to_string(),
            "-vsync",
            "vfr",
            "-q:v",
            "2",
            output_pattern.to_str().ok_or_else(|| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Invalid output path encoding",
                ))
            })?,
            "-y", // Overwrite existing files
        ])
        .output()
        .await
        .map_err(|e| {
            Error::Io(std::io::Error::other(format!(
                "Failed to run ffmpeg for keyframe extraction: {}. Is ffmpeg installed?",
                e
            )))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::Io(std::io::Error::other(format!(
            "ffmpeg keyframe extraction failed: {}",
            stderr
        ))));
    }

    // Collect the extracted keyframe paths
    let mut keyframes = Vec::new();
    for i in 1..=max_keyframes {
        let frame_path = output_dir.join(format!("keyframe_{:03}.jpg", i));
        if frame_path.exists() {
            keyframes.push(frame_path);
        } else {
            break; // No more frames
        }
    }

    Ok(keyframes)
}

/// Extract audio track from video file to a temporary file for transcription
async fn extract_audio_track(video_path: &Path, output_path: &Path) -> Result<bool> {
    use tokio::process::Command;

    let output = Command::new("ffmpeg")
        .args([
            "-i",
            video_path.to_str().ok_or_else(|| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Invalid video path encoding",
                ))
            })?,
            "-vn", // No video
            "-acodec",
            "libmp3lame",
            "-q:a",
            "4",
            output_path.to_str().ok_or_else(|| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Invalid output path encoding",
                ))
            })?,
            "-y", // Overwrite existing files
        ])
        .output()
        .await
        .map_err(|e| {
            Error::Io(std::io::Error::other(format!(
                "Failed to run ffmpeg for audio extraction: {}. Is ffmpeg installed?",
                e
            )))
        })?;

    // Return false if no audio stream (common for some video files)
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("does not contain any stream") || stderr.contains("Output file is empty")
        {
            return Ok(false);
        }
        return Err(Error::Io(std::io::Error::other(format!(
            "ffmpeg audio extraction failed: {}",
            stderr
        ))));
    }

    // Check if output file was created and has content
    Ok(output_path.exists() && output_path.metadata().map(|m| m.len() > 0).unwrap_or(false))
}

/// Process a video file: extract metadata, keyframes, audio track, and create chunks
#[allow(clippy::too_many_arguments)]
async fn process_video_file(
    config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
    embedder: &dyn Embedder,
    source: &Source,
    path: &Path,
    transcriber: Option<&AudioTranscriber>,
    http_client: &reqwest::Client,
    temp_dir: &Path,
) -> Result<(i32, i32)> {
    let file_uri = path.display().to_string();
    let video_config = &config.crawl.multimodal.video;

    debug!(path = %file_uri, "Processing video file");

    // Extract metadata with ffprobe
    let metadata = extract_media_metadata(path).await?;
    debug!(
        path = %file_uri,
        duration = metadata.duration_secs,
        format = %metadata.format_name,
        video_streams = metadata.video_streams,
        audio_streams = metadata.audio_streams,
        "Extracted video metadata"
    );

    // Check duration limit
    if metadata.duration_secs > video_config.max_duration_secs as f64 {
        debug!(
            path = %file_uri,
            duration = metadata.duration_secs,
            max = video_config.max_duration_secs,
            "Skipping video file (exceeds max duration)"
        );
        return Ok((0, 0));
    }

    // Compute content hash from file
    let file_bytes = tokio::fs::read(path).await?;
    let content_hash = compute_content_hash(&file_bytes);

    // Check if content changed
    let existing_doc = db.get_document_by_uri(&source.id, &file_uri).await?;
    if let Some(ref doc) = existing_doc {
        if doc.content_hash == content_hash {
            debug!(path = %file_uri, "Video file unchanged");
            return Ok((0, 0));
        }
    }

    // Create/update document
    let mut doc = Document::new(source.id.clone(), file_uri.clone(), content_hash.clone());
    doc.title = Some(
        path.file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("video")
            .to_string(),
    );
    doc.content_type = Some(format!("video/{}", metadata.format_name));
    let doc = db.upsert_document(&doc).await?;

    // Delete old video chunks for this document before creating new ones
    let old_chunks = db.get_chunks_by_modality(&doc.id, "video").await?;
    if !old_chunks.is_empty() {
        let point_ids: Vec<Uuid> = old_chunks.iter().map(|c| c.point_uuid()).collect();
        if !point_ids.is_empty() {
            store.delete_points(&point_ids).await?;
        }
        db.delete_chunks_by_modality(&doc.id, "video").await?;
    }

    let mut chunks_created = 0;
    let video_temp_dir = temp_dir.join(format!("video_{}", Uuid::new_v4()));
    tokio::fs::create_dir_all(&video_temp_dir).await?;

    // Extract keyframes if video has video streams
    if metadata.video_streams > 0 && video_config.max_keyframes > 0 {
        match extract_keyframes(path, &video_temp_dir, video_config.max_keyframes).await {
            Ok(keyframe_paths) => {
                debug!(
                    path = %file_uri,
                    keyframes = keyframe_paths.len(),
                    "Extracted keyframes from video"
                );

                // Embed each keyframe
                for (frame_idx, frame_path) in keyframe_paths.iter().enumerate() {
                    let frame_bytes = match tokio::fs::read(frame_path).await {
                        Ok(b) => b,
                        Err(e) => {
                            warn!(path = %frame_path.display(), error = %e, "Failed to read keyframe");
                            continue;
                        }
                    };

                    let frame_hash = compute_content_hash(&frame_bytes);
                    let base64_img = base64::engine::general_purpose::STANDARD.encode(&frame_bytes);
                    let data_url = format!("data:image/jpeg;base64,{}", base64_img);

                    // Embed as image
                    let embeddings = match embedder.embed_images(vec![data_url.clone()]).await {
                        Ok(e) if !e.is_empty() => e,
                        Ok(_) => {
                            warn!(path = %file_uri, frame = frame_idx, "Failed to embed keyframe (empty result)");
                            continue;
                        }
                        Err(e) => {
                            warn!(path = %file_uri, frame = frame_idx, error = %e, "Failed to embed keyframe");
                            continue;
                        }
                    };

                    // Create chunk for keyframe with video modality
                    let chunk = Chunk::new_with_modality(
                        doc.id.clone(),
                        frame_idx as i32,
                        frame_hash.clone(),
                        format!(
                            "[Video keyframe {} from {}]",
                            frame_idx + 1,
                            path.file_name().and_then(|f| f.to_str()).unwrap_or("video")
                        ),
                        "video",
                        Some(file_uri.clone()),
                        Some(content_hash.clone()),
                    );

                    let point_id = chunk.point_uuid();

                    let mut payload = ChunkPayload::new(
                        source.id.clone(),
                        source.source_type.clone(),
                        source.uri.clone(),
                        doc.id.clone(),
                        file_uri.clone(),
                        frame_idx as i32,
                        frame_hash.clone(),
                        Utc::now().to_rfc3339(),
                    );
                    payload.title = doc.title.clone();
                    payload.modality = Some("video".to_string());
                    payload.media_url = Some(file_uri.clone());
                    payload.media_hash = Some(content_hash.clone());

                    let points = vec![ChunkPoint {
                        id: point_id,
                        vector: embeddings[0].clone(),
                        payload,
                    }];

                    if let Err(e) = store.upsert_points(points).await {
                        warn!(path = %file_uri, frame = frame_idx, error = %e, "Failed to store keyframe");
                        continue;
                    }

                    if let Err(e) = db.upsert_chunk(&chunk).await {
                        warn!(path = %file_uri, frame = frame_idx, error = %e, "Failed to save keyframe chunk");
                        continue;
                    }

                    chunks_created += 1;
                }
            }
            Err(e) => {
                warn!(path = %file_uri, error = %e, "Failed to extract keyframes");
            }
        }
    }

    // Extract and transcribe audio track if enabled and video has audio
    if metadata.audio_streams > 0 && video_config.extract_audio {
        let audio_path = video_temp_dir.join("audio_track.mp3");

        match extract_audio_track(path, &audio_path).await {
            Ok(true) => {
                // Transcribe the extracted audio
                match transcriber {
                    Some(transcriber) => {
                        match transcriber.transcribe(&audio_path, http_client).await {
                            Ok(transcript) if !transcript.is_empty() => {
                                debug!(
                                    path = %file_uri,
                                    chars = transcript.len(),
                                    "Transcribed video audio track"
                                );

                                // Create a text chunk for the transcript
                                let transcript_hash = compute_content_hash(transcript.as_bytes());
                                let chunk_index = chunks_created; // After keyframe chunks

                                let chunk = Chunk::new_with_modality(
                                    doc.id.clone(),
                                    chunk_index,
                                    transcript_hash.clone(),
                                    transcript.clone(),
                                    "video", // Still video modality since it's from video
                                    Some(file_uri.clone()),
                                    Some(content_hash.clone()),
                                );

                                // Embed as text
                                let embeddings = embed_in_batches(
                                    embedder,
                                    vec![transcript.clone()],
                                    config.embedding.batch_size,
                                )
                                .await?;
                                if !embeddings.is_empty() {
                                    let point_id = chunk.point_uuid();

                                    let mut payload = ChunkPayload::new(
                                        source.id.clone(),
                                        source.source_type.clone(),
                                        source.uri.clone(),
                                        doc.id.clone(),
                                        file_uri.clone(),
                                        chunk_index,
                                        transcript_hash.clone(),
                                        Utc::now().to_rfc3339(),
                                    );
                                    payload.title = doc.title.clone();
                                    payload.modality = Some("video".to_string());
                                    payload.media_url = Some(file_uri.clone());
                                    payload.media_hash = Some(content_hash.clone());

                                    let points = vec![ChunkPoint {
                                        id: point_id,
                                        vector: embeddings[0].clone(),
                                        payload,
                                    }];

                                    store.upsert_points(points).await?;
                                    db.upsert_chunk(&chunk).await?;
                                    chunks_created += 1;
                                }
                            }
                            Ok(_) => {
                                debug!(path = %file_uri, "Video audio track transcription returned empty");
                            }
                            Err(e) => {
                                warn!(path = %file_uri, error = %e, "Failed to transcribe video audio track");
                            }
                        }
                    }
                    None => {
                        debug!(
                            path = %file_uri,
                            "Video audio track extraction succeeded, but transcription is disabled"
                        );
                    }
                }
            }
            Ok(false) => {
                debug!(path = %file_uri, "No audio track in video");
            }
            Err(e) => {
                warn!(path = %file_uri, error = %e, "Failed to extract audio track from video");
            }
        }
    }

    // Clean up temp directory
    if let Err(e) = tokio::fs::remove_dir_all(&video_temp_dir).await {
        debug!(path = %video_temp_dir.display(), error = %e, "Failed to clean up video temp directory");
    }

    debug!(
        path = %file_uri,
        duration = metadata.duration_secs,
        chunks = chunks_created,
        "Processed video file"
    );

    Ok((chunks_created, 0))
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
    let avg = total / (PERCEPTUAL_HASH_SIZE * PERCEPTUAL_HASH_SIZE);
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
#[allow(clippy::too_many_arguments)]
async fn fetch_and_cache_images(
    config: &Config,
    images: &[(ExtractedMedia, f32)],
) -> Vec<CachedAsset> {
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
        .redirect(reqwest::redirect::Policy::none()) // Disable redirects to prevent SSRF bypass
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
    if let Err(e) = fs::create_dir_all(&assets_dir).await {
        warn!("Failed to create assets dir: {}", e);
    }

    let mut cached = Vec::new();
    let mut seen_hashes: HashSet<String> = HashSet::new();
    let mut seen_phashes: Vec<u64> = Vec::new();

    for (m, _score) in images.iter() {
        if Url::parse(&m.url)
            .map(|u| u.scheme().to_string())
            .map(|s| s != "http" && s != "https")
            .unwrap_or(true)
        {
            continue;
        }
        // SSRF validation before fetching
        if let Err(e) = validate_url_ssrf(&m.url).await {
            debug!(url = %m.url, error = %e, "Skipping image (SSRF validation failed)");
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
                        let ext = if let Some(pos) = m.url.rfind('.') {
                            m.url[pos..].to_string()
                        } else {
                            ".bin".to_string()
                        };
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

#[allow(clippy::too_many_arguments)]
async fn embed_cached_images(
    config: &Config,
    embedding: &ResolvedEmbeddingConfig,
    db: &MetaDb,
    store: &QdrantStore,
    embedder: &dyn Embedder,
    source: &Source,
    doc: &Document,
    doc_uri: &str,
    parsed: &ParsedDocument,
    cached_images: Vec<CachedAsset>,
) -> Result<(i32, i32)> {
    if cached_images.is_empty() {
        return Ok((0, 0));
    }

    if !embedding.supports_image_inputs() {
        debug!(uri = %doc_uri, model = %embedding.model_id, "Skipping image embedding (model not multimodal)");
        return Ok((0, 0));
    }

    if embedding.supports_multi_vector {
        return Err(Error::Embedding(format!(
            "Late-interaction embedding model '{}' does not support image ingestion",
            embedding.model_id
        )));
    }

    let batch_size = embedding.effective_batch_size(config.embedding.batch_size);
    let contexts: Vec<Option<String>> = cached_images
        .iter()
        .map(|asset| build_image_context(parsed, &asset.media))
        .collect();
    let image_paths: Vec<String> = cached_images
        .iter()
        .map(|asset| asset.path.to_string_lossy().to_string())
        .collect();

    let embeddings = embed_images_with_optional_text_fusion(
        embedder,
        embedding,
        image_paths,
        contexts,
        batch_size,
    )
    .await?;

    if embeddings.is_empty() {
        return Ok((0, 0));
    }

    let expected_dim = store.dimension();
    if embedder.dimension() != expected_dim {
        return Err(Error::Embedding(format!(
            "Embedding dimension mismatch for model '{}' (family '{}', source {}): embedder {} != collection {}",
            embedding.model_id,
            embedding.family,
            embedding.dimension_source,
            embedder.dimension(),
            expected_dim
        )));
    }

    if embeddings[0].len() != expected_dim {
        return Err(Error::Embedding(format!(
            "Image embedding dimension mismatch for model '{}': expected {}, got {}",
            embedding.model_id,
            expected_dim,
            embeddings[0].len()
        )));
    }

    if embeddings.len() != cached_images.len() {
        return Err(Error::Embedding(format!(
            "Image embedding count mismatch for model '{}': expected {}, got {}",
            embedding.model_id,
            cached_images.len(),
            embeddings.len()
        )));
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

        let point_id = meta_chunk.point_uuid();

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
#[allow(clippy::too_many_arguments)]
pub async fn cmd_ingest_dir(
    config: &Config,
    embedding: &ResolvedEmbeddingConfig,
    embedder: &dyn Embedder,
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

    store.ensure_collection().await?;

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

    validate_dimensions(embedder, store, embedding)?;

    // Collect all files
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let mut audio_files: Vec<std::path::PathBuf> = Vec::new();
    let mut video_files: Vec<std::path::PathBuf> = Vec::new();

    let mm = &config.crawl.multimodal;
    let include_audio = mm.enabled && mm.include_audio;
    let include_video = mm.enabled && mm.include_video;

    let walker = WalkBuilder::new(&canonical_path)
        .hidden(false)
        .git_ignore(true)
        .git_exclude(true)
        .build();

    for entry in walker {
        match entry {
            Ok(e) if e.file_type().map(|t| t.is_file()).unwrap_or(false) => {
                let path = e.path().to_path_buf();
                // Check for audio/video files before the generic skip check
                if include_audio && is_audio_file(&path) {
                    audio_files.push(path);
                } else if include_video && is_video_file(&path) {
                    video_files.push(path);
                } else if !should_skip_file(&path) {
                    files.push(path);
                }
            }
            _ => {}
        }
    }

    info!(
        "Found {} text files, {} audio files, {} video files to process",
        files.len(),
        audio_files.len(),
        video_files.len()
    );

    let mut current_uris: Vec<String> = Vec::new();
    let total_files = files.len() + audio_files.len() + video_files.len();
    let file_progress = start_progress_bar(total_files, "Processing files");

    for file_path in files {
        let file_uri = file_path.display().to_string();
        current_uris.push(file_uri.clone());

        match process_file(config, embedding, db, store, embedder, &source, &file_path).await {
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

    // Process audio files with ffprobe metadata and transcription
    let http_client = reqwest::Client::new();
    let needs_audio_transcriber = (!audio_files.is_empty()
        || (!video_files.is_empty() && config.crawl.multimodal.video.extract_audio))
        && config.crawl.multimodal.audio.transcription_enabled;
    let audio_transcriber = if needs_audio_transcriber {
        match AudioTranscriber::from_config(config).await {
            Ok(transcriber) => transcriber,
            Err(error) => {
                warn!(
                    error = %error,
                    "Failed to initialize audio transcriber; audio and video transcription will be skipped"
                );
                None
            }
        }
    } else {
        None
    };
    for audio_path in audio_files {
        let file_uri = audio_path.display().to_string();
        current_uris.push(file_uri.clone());

        match process_audio_file(
            config,
            db,
            store,
            embedder,
            &source,
            &audio_path,
            audio_transcriber.as_ref(),
            &http_client,
        )
        .await
        {
            Ok((created, updated)) => {
                stats.chunks_created += created;
                stats.chunks_updated += updated;
                if created > 0 || updated > 0 {
                    stats.docs_processed += 1;
                } else {
                    stats.docs_skipped += 1;
                }
            }
            Err(e) => {
                let msg = format!("Error processing audio {}: {}", file_uri, e);
                warn!("{}", msg);
                stats.errors.push(msg);
            }
        }
        advance_progress(&file_progress);
    }

    // Create temp directory for video processing
    let video_temp_dir = std::env::temp_dir().join(format!("librarian_video_{}", Uuid::new_v4()));

    // Process video files with ffprobe metadata and keyframe extraction
    for video_path in video_files {
        let file_uri = video_path.display().to_string();
        current_uris.push(file_uri.clone());

        match process_video_file(
            config,
            db,
            store,
            embedder,
            &source,
            &video_path,
            audio_transcriber.as_ref(),
            &http_client,
            &video_temp_dir,
        )
        .await
        {
            Ok((created, updated)) => {
                stats.chunks_created += created;
                stats.chunks_updated += updated;
                if created > 0 || updated > 0 {
                    stats.docs_processed += 1;
                } else {
                    stats.docs_skipped += 1;
                }
            }
            Err(e) => {
                let msg = format!("Error processing video {}: {}", file_uri, e);
                warn!("{}", msg);
                stats.errors.push(msg);
            }
        }
        advance_progress(&file_progress);
    }

    // Clean up video temp directory
    if video_temp_dir.exists() {
        let _ = tokio::fs::remove_dir_all(&video_temp_dir).await;
    }

    finish_progress(file_progress, "Files processed");

    // Resolve document cross-references now that all docs are ingested
    match db.resolve_document_links(&source.id).await {
        Ok(resolved) if resolved > 0 => {
            info!(resolved, source_id = %source.id, "Resolved document links");
        }
        Err(e) => warn!("Failed to resolve document links: {}", e),
        _ => {}
    }

    cleanup_and_complete_run(
        db,
        store,
        &source.id,
        &run.id,
        &current_uris,
        &stats,
        "Ingestion",
    )
    .await?;

    Ok(stats)
}

/// Process a single file
async fn process_file(
    config: &Config,
    embedding: &ResolvedEmbeddingConfig,
    db: &MetaDb,
    store: &QdrantStore,
    embedder: &dyn Embedder,
    source: &Source,
    path: &Path,
) -> Result<(i32, i32)> {
    let file_uri = path.display().to_string();
    debug!("Processing file: {}", file_uri);

    // Read file content
    let content = tokio::fs::read(path).await?;

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

    // Store document links for cross-reference resolution
    if !parsed.links.is_empty() {
        if let Err(e) = db.store_document_links(&doc.id, &parsed.links).await {
            warn!(doc_id = %doc.id, "Failed to store document links: {}", e);
        }
    }

    if chunks.is_empty() {
        debug!("No chunks generated for: {}", file_uri);
        return Ok((0, 0));
    }

    // Process chunks
    let (created, updated) = process_chunks(
        config, embedding, db, store, embedder, source, &doc, &file_uri, chunks,
    )
    .await?;

    Ok((created, updated))
}

/// Process chunks for a document
#[allow(clippy::too_many_arguments)]
async fn process_chunks(
    config: &Config,
    embedding: &ResolvedEmbeddingConfig,
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

    let expected_dim = store.dimension();
    if embedder.dimension() != expected_dim {
        return Err(Error::Embedding(format!(
            "Embedding dimension mismatch for model '{}' (family '{}', source {}): embedder {} != collection {}",
            embedding.model_id,
            embedding.family,
            embedding.dimension_source,
            embedder.dimension(),
            expected_dim
        )));
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
    let batch_size = embedding.effective_batch_size(config.embedding.batch_size);
    let embeddings = embed_in_batches(embedder, texts, batch_size).await?;

    if embeddings
        .iter()
        .any(|embedding| embedding.len() != expected_dim)
    {
        let mismatched = embeddings
            .iter()
            .find(|embedding| embedding.len() != expected_dim)
            .map(|embedding| embedding.len())
            .unwrap_or(0);
        return Err(Error::Embedding(format!(
            "Embedding dimension mismatch: expected {}, got {}",
            expected_dim, mismatched
        )));
    }

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
        let point_id = meta_chunk.point_uuid();

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
#[allow(clippy::too_many_arguments)]
pub async fn cmd_ingest_url(
    config: &Config,
    embedding: &ResolvedEmbeddingConfig,
    embedder: &dyn Embedder,
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

    store.ensure_collection().await?;

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

    validate_dimensions(embedder, store, embedding)?;

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

        match process_page(config, embedding, db, store, embedder, &source, &page).await {
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

    // Resolve document cross-references now that all pages are ingested
    match db.resolve_document_links(&source.id).await {
        Ok(resolved) if resolved > 0 => {
            info!(resolved, source_id = %source.id, "Resolved document links");
        }
        Err(e) => warn!("Failed to resolve document links: {}", e),
        _ => {}
    }

    cleanup_and_complete_run(
        db,
        store,
        &source.id,
        &run.id,
        &current_uris,
        &stats,
        "Ingestion",
    )
    .await?;

    Ok(stats)
}

/// Ingest from a sitemap URL
#[allow(clippy::too_many_arguments)]
pub async fn cmd_ingest_sitemap(
    config: &Config,
    embedding: &ResolvedEmbeddingConfig,
    embedder: &dyn Embedder,
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

    store.ensure_collection().await?;

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

    validate_dimensions(embedder, store, embedding)?;
    let crawler = Crawler::new(config.crawl.clone())?;

    let mut current_uris: Vec<String> = Vec::new();
    let url_progress = start_progress_bar(entries.len(), "Processing URLs");

    // Process each URL from sitemap
    for entry in entries {
        current_uris.push(entry.loc.clone());

        // Fetch the page
        match crawler.fetch(&entry.loc).await {
            Ok(page) => {
                match process_page(config, embedding, db, store, embedder, &source, &page).await {
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

    // Resolve document cross-references now that all pages are ingested
    match db.resolve_document_links(&source.id).await {
        Ok(resolved) if resolved > 0 => {
            info!(resolved, source_id = %source.id, "Resolved document links");
        }
        Err(e) => warn!("Failed to resolve document links: {}", e),
        _ => {}
    }

    cleanup_and_complete_run(
        db,
        store,
        &source.id,
        &run.id,
        &current_uris,
        &stats,
        "Sitemap ingestion",
    )
    .await?;

    Ok(stats)
}

/// Process a crawled page
async fn process_page(
    config: &Config,
    embedding: &ResolvedEmbeddingConfig,
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

    // Store document links for cross-reference resolution
    if !parsed.links.is_empty() {
        if let Err(e) = db.store_document_links(&doc.id, &parsed.links).await {
            warn!(doc_id = %doc.id, "Failed to store document links: {}", e);
        }
    }

    // Multimodal image selection + caching (optional)
    let images = select_image_candidates(config, embedding, &parsed);
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
            embedding,
            db,
            store,
            embedder,
            source,
            &doc,
            &page.url,
            &parsed,
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
    let (created, updated) = process_chunks(
        config, embedding, db, store, embedder, source, &doc, &page.url, chunks,
    )
    .await?;

    // Embed cached images after text processing
    let (image_created, image_updated) = if cached_images.is_empty() {
        (0, 0)
    } else {
        match embed_cached_images(
            config,
            embedding,
            db,
            store,
            embedder,
            source,
            &doc,
            &page.url,
            &parsed,
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

/// Validate that the embedder and store share the same vector dimension.
///
/// # Errors
/// Returns `Error::Embedding` when the dimensions differ, including the
/// model id, family, and dimension source for diagnostics.
pub(crate) fn validate_dimensions(
    embedder: &dyn Embedder,
    store: &QdrantStore,
    embedding: &ResolvedEmbeddingConfig,
) -> Result<()> {
    if embedder.dimension() != store.dimension() {
        return Err(Error::Embedding(format!(
            "Embedding dimension mismatch for model '{}' (family '{}', source {}): embedder {} != collection {}",
            embedding.model_id,
            embedding.family,
            embedding.dimension_source,
            embedder.dimension(),
            store.dimension()
        )));
    }
    Ok(())
}

/// Remove documents that no longer exist in the source and delete their Qdrant
/// points, then finalise the ingestion run with the given stats.
///
/// # Errors
/// Propagates database or store errors.
async fn cleanup_and_complete_run(
    db: &MetaDb,
    store: &QdrantStore,
    source_id: &str,
    run_id: &str,
    current_uris: &[String],
    stats: &IngestStats,
    completion_label: &str,
) -> Result<()> {
    let stale_ids = db.delete_stale_documents(source_id, current_uris).await?;
    if !stale_ids.is_empty() {
        info!("Deleted {} stale documents", stale_ids.len());
        for doc_id in &stale_ids {
            if let Ok(chunks) = db.get_chunks(doc_id).await {
                let point_ids: Vec<Uuid> = chunks.iter().map(|c| c.point_uuid()).collect();
                if !point_ids.is_empty() {
                    if let Err(e) = store.delete_points(&point_ids).await {
                        warn!("Failed to delete Qdrant points: {}", e);
                    }
                }
            }
        }
    }

    let errors = if stats.errors.is_empty() {
        None
    } else {
        Some(stats.errors.clone())
    };

    db.complete_ingestion_run(
        run_id,
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
        "{} complete: {} docs, {} chunks created, {} chunks updated",
        completion_label, stats.docs_processed, stats.chunks_created, stats.chunks_updated
    );

    Ok(())
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
    use crate::config::{
        AudioTranscriptionBackend, EmbeddingDimensionSource, ResolvedEmbeddingConfig,
    };
    use crate::embedding_backend::{EmbeddingBackendConfig, EmbeddingBackendKind};
    use crate::models::MultimodalStrategy;
    use crate::parse::{ContentType, ExtractedMedia, Heading, MediaModality, ParsedDocument};
    use serde_json::json;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use wiremock::matchers::{body_string_contains, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn multimodal_config() -> Config {
        let mut config = Config::default();
        config.crawl.multimodal.enabled = true;
        config.embedding.model = "jinaai/jina-clip-v2".to_string();
        config.crawl.multimodal.min_relevance_score = 0.0;
        config
    }

    fn test_embedding_config(supports_image: bool, multi_vector: bool) -> ResolvedEmbeddingConfig {
        ResolvedEmbeddingConfig {
            model_id: "test-model".to_string(),
            family: "test".to_string(),
            modalities: if supports_image {
                vec!["text".to_string(), "image".to_string()]
            } else {
                vec!["text".to_string()]
            },
            dimension: 384,
            dimension_source: EmbeddingDimensionSource::Config,
            backend: EmbeddingBackendConfig {
                kind: EmbeddingBackendKind::Http,
                url: "http://localhost:9997".to_string(),
            },
            allow_custom: false,
            strategy: if multi_vector {
                MultimodalStrategy::LateInteraction
            } else {
                MultimodalStrategy::DualEncoder
            },
            supports_text: true,
            supports_image,
            supports_audio: false,
            supports_video: false,
            supports_joint_inputs: false,
            supports_multi_vector: multi_vector,
            supports_mrl: false,
            max_batch: 32,
        }
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
                modality: MediaModality::Image,
                mime_type: None,
            },
            ExtractedMedia {
                url: "https://example.com/diagram.png".to_string(),
                alt: Some("Diagram".to_string()),
                tag: "img".to_string(),
                css_background: false,
                modality: MediaModality::Image,
                mime_type: None,
            },
        ];

        let embedding = test_embedding_config(true, false);
        let candidates = select_image_candidates(&config, &embedding, &doc);
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
            modality: MediaModality::Image,
            mime_type: None,
        }];

        let embedding = test_embedding_config(true, false);
        let candidates = select_image_candidates(&config, &embedding, &doc);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_select_image_candidates_skips_late_interaction() {
        let config = multimodal_config();

        let mut doc = ParsedDocument::new("text".to_string(), ContentType::Html);
        doc.media = vec![ExtractedMedia {
            url: "https://example.com/diagram.png".to_string(),
            alt: Some("Diagram".to_string()),
            tag: "img".to_string(),
            css_background: false,
            modality: MediaModality::Image,
            mime_type: None,
        }];

        // Use late-interaction model config
        let embedding = test_embedding_config(true, true);
        let candidates = select_image_candidates(&config, &embedding, &doc);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_url_is_allowed_image_filters_svg() {
        let allowed = vec!["image/".to_string()];
        assert!(!url_is_allowed_image(
            "https://example.com/icon.svg",
            &allowed
        ));
        assert!(url_is_allowed_image(
            "https://example.com/photo.png",
            &allowed
        ));
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

    #[tokio::test]
    async fn test_validate_url_ssrf_rejects_non_http() {
        assert!(validate_url_ssrf("file:///etc/passwd").await.is_err());
        assert!(validate_url_ssrf("ftp://example.com/file").await.is_err());
    }

    #[tokio::test]
    async fn test_validate_url_ssrf_accepts_valid_urls() {
        let result = validate_url_ssrf("http://8.8.8.8/image.png").await;
        assert!(result.is_ok(), "Should accept public IP 8.8.8.8");

        let result = validate_url_ssrf("http://1.1.1.1/image.png").await;
        assert!(result.is_ok(), "Should accept public IP 1.1.1.1");
    }

    #[tokio::test]
    async fn test_validate_url_ssrf_rejects_private_ips_directly() {
        assert!(validate_url_ssrf("http://127.0.0.1/image.png")
            .await
            .is_err());
        assert!(validate_url_ssrf("http://10.0.0.1/image.png")
            .await
            .is_err());
        assert!(validate_url_ssrf("http://192.168.1.1/image.png")
            .await
            .is_err());
        assert!(validate_url_ssrf("http://169.254.169.254/latest/meta-data")
            .await
            .is_err());
    }

    #[test]
    fn test_auto_transcription_backend_detects_local_xinference() {
        let mut config = Config::default();
        config.crawl.multimodal.audio.transcription_backend = AudioTranscriptionBackend::Auto;
        config.crawl.multimodal.audio.transcription_url =
            "http://127.0.0.1:9997/v1/audio/transcriptions".to_string();
        assert_eq!(
            resolve_transcription_backend(&config.crawl.multimodal.audio),
            AudioTranscriptionBackend::Xinference
        );

        config.crawl.multimodal.audio.transcription_url =
            "http://127.0.0.1:8000/v1/audio/transcriptions".to_string();
        assert_eq!(
            resolve_transcription_backend(&config.crawl.multimodal.audio),
            AudioTranscriptionBackend::Http
        );
    }

    #[test]
    fn test_auto_transcription_model_resolves_per_backend() {
        let mut audio_config = Config::default().crawl.multimodal.audio;
        audio_config.transcription_model = "auto".to_string();
        assert_eq!(
            resolve_transcription_model(&audio_config, AudioTranscriptionBackend::Http),
            "whisper-1"
        );
        let xinference_model =
            resolve_transcription_model(&audio_config, AudioTranscriptionBackend::Xinference);
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert_eq!(xinference_model, "whisper-large-v3-turbo-mlx");
        } else {
            assert_eq!(xinference_model, "whisper-large-v3-turbo");
        }
    }

    #[tokio::test]
    async fn test_audio_transcriber_http_auto_uses_whisper_1() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(body_string_contains("whisper-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "text": "hello over http"
            })))
            .mount(&mock_server)
            .await;

        let mut config = Config::default();
        config.crawl.multimodal.audio.transcription_enabled = true;
        config.crawl.multimodal.audio.transcription_backend = AudioTranscriptionBackend::Auto;
        config.crawl.multimodal.audio.transcription_url =
            format!("{}/v1/audio/transcriptions", mock_server.uri());
        config.crawl.multimodal.audio.transcription_model = "auto".to_string();

        let transcriber = AudioTranscriber::from_config(&config)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(transcriber.backend, AudioTranscriptionBackend::Http);
        assert_eq!(transcriber.model, "whisper-1");

        let mut audio_file = NamedTempFile::new().unwrap();
        audio_file.write_all(b"fake mp3 bytes").unwrap();

        let transcript = transcriber
            .transcribe(audio_file.path(), &reqwest::Client::new())
            .await
            .unwrap();
        assert_eq!(transcript, "hello over http");
    }

    #[tokio::test]
    async fn test_audio_transcriber_launches_xinference_audio_model() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/cluster/version"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "version": "test"
            })))
            .mount(&mock_server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([])))
            .mount(&mock_server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/models"))
            .and(body_string_contains("whisper-large-v3-turbo"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model_uid": "audio-model-uid"
            })))
            .mount(&mock_server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(body_string_contains("audio-model-uid"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "text": "hello from xinference"
            })))
            .mount(&mock_server)
            .await;

        let mut config = Config::default();
        config.crawl.multimodal.audio.transcription_enabled = true;
        config.crawl.multimodal.audio.transcription_backend = AudioTranscriptionBackend::Xinference;
        config.crawl.multimodal.audio.transcription_url =
            format!("{}/v1/audio/transcriptions", mock_server.uri());
        config.crawl.multimodal.audio.transcription_model = "whisper-large-v3-turbo".to_string();

        let transcriber = AudioTranscriber::from_config(&config)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(transcriber.backend, AudioTranscriptionBackend::Xinference);
        assert_eq!(transcriber.model, "audio-model-uid");
        assert_eq!(
            transcriber.transcription_url.as_str(),
            format!("{}/v1/audio/transcriptions", mock_server.uri())
        );

        let mut audio_file = NamedTempFile::new().unwrap();
        audio_file.write_all(b"fake mp3 bytes").unwrap();

        let transcript = transcriber
            .transcribe(audio_file.path(), &reqwest::Client::new())
            .await
            .unwrap();
        assert_eq!(transcript, "hello from xinference");
    }
}
