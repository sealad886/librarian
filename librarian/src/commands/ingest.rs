//! Ingest command implementation

use crate::chunk::{chunk_document, compute_content_hash, TextChunk};
use crate::config::Config;
use crate::crawl::{CrawledPage, Crawler};
use crate::embed::{create_embedder, embed_in_batches, Embedder};
use crate::error::{Error, Result};
use crate::meta::{Chunk, Document, MetaDb, RunStatus, Source, SourceType};
use crate::parse::{
    is_binary_content, parse_content, should_skip_file, ContentType,
};
use crate::store::{ChunkPayload, ChunkPoint, QdrantStore};
use chrono::Utc;
use ignore::WalkBuilder;
use std::collections::HashSet;
use std::path::Path;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Statistics from an ingestion run
#[derive(Debug, Default)]
pub struct IngestStats {
    pub docs_processed: i32,
    pub docs_skipped: i32,
    pub chunks_created: i32,
    pub chunks_updated: i32,
    pub chunks_deleted: i32,
    pub errors: Vec<String>,
}

/// Ingest a local directory
pub async fn cmd_ingest_dir(
    config: &Config,
    db: &MetaDb,
    store: &QdrantStore,
    path: &Path,
    name: Option<String>,
) -> Result<IngestStats> {
    let canonical_path = path.canonicalize().map_err(|e| {
        Error::InvalidPath(format!("{}: {}", path.display(), e))
    })?;

    let uri = canonical_path.display().to_string();
    info!("Ingesting directory: {}", uri);

    // Get or create source
    let source = match db.get_source_by_uri(&uri).await? {
        Some(s) => {
            info!("Found existing source: {}", s.id);
            s
        }
        None => {
            let s = Source::new(SourceType::Dir, uri.clone(), name.or(Some(uri.clone())));
            db.insert_source(&s).await?;
            info!("Created new source: {}", s.id);
            s
        }
    };

    // Start ingestion run
    let run = db.start_ingestion_run(&source.id).await?;

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

    let mut stats = IngestStats::default();
    let mut current_uris: Vec<String> = Vec::new();

    for file_path in files {
        let file_uri = file_path.display().to_string();
        current_uris.push(file_uri.clone());

        match process_file(
            config,
            db,
            store,
            embedder.as_ref(),
            &source,
            &file_path,
        )
        .await
        {
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
    }

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
        if stats.errors.is_empty() { RunStatus::Completed } else { RunStatus::Failed },
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
    if let Some(existing_doc) = db.get_document_by_uri(&source.id, &file_uri).await? {
        if existing_doc.content_hash == content_hash {
            debug!("File unchanged: {}", file_uri);
            return Ok((0, 0));
        }
    }

    // Detect content type and parse
    let content_type = ContentType::from_extension(path);
    let parsed = parse_content(&text, content_type, None)?;

    // Create/update document
    let mut doc = Document::new(source.id.clone(), file_uri.clone(), content_hash.clone());
    doc.title = parsed.title.clone();
    doc.content_type = Some(format!("{:?}", content_type).to_lowercase());
    db.upsert_document(&doc).await?;

    // Chunk the document
    let chunks = chunk_document(&parsed, &content_hash, &config.chunk)?;

    if chunks.is_empty() {
        debug!("No chunks generated for: {}", file_uri);
        return Ok((0, 0));
    }

    // Process chunks
    let (created, updated) = process_chunks(
        config, db, store, embedder, source, &doc, &file_uri, chunks,
    )
    .await?;

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
    let existing_chunks = db.get_chunks(&doc.id).await?;
    let existing_hashes: HashSet<String> = existing_chunks.iter().map(|c| c.chunk_hash.clone()).collect();

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
    let texts: Vec<String> = chunks_to_embed.iter().map(|(_, c)| c.text.clone()).collect();
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
            if chunk.headings.is_empty() { None } else { Some(chunk.headings.clone()) },
        );

        // Save chunk to SQLite
        db.upsert_chunk(&meta_chunk).await?;

        // Create Qdrant payload
        let payload = ChunkPayload {
            source_id: source.id.clone(),
            source_type: source.source_type.clone(),
            source_uri: source.uri.clone(),
            doc_id: doc.id.clone(),
            doc_uri: doc_uri.to_string(),
            title: doc.title.clone(),
            headings: if chunk.headings.is_empty() { None } else { Some(chunk.headings.clone()) },
            chunk_index: *chunk_index as i32,
            chunk_hash: chunk.hash.clone(),
            updated_at: Utc::now().to_rfc3339(),
        };

        // Parse qdrant_point_id string to Uuid
        let point_id = Uuid::try_parse(&meta_chunk.qdrant_point_id)
            .unwrap_or_else(|_| Uuid::new_v5(&Uuid::NAMESPACE_OID, meta_chunk.qdrant_point_id.as_bytes()));

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
    let deleted_point_strings = db.delete_chunks_from_index(&doc.id, chunks.len() as i32).await?;
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
) -> Result<IngestStats> {
    info!("Ingesting URL: {}", url);

    // Get or create source
    let source = match db.get_source_by_uri(url).await? {
        Some(s) => {
            info!("Found existing source: {}", s.id);
            s
        }
        None => {
            let s = Source::new(SourceType::Url, url.to_string(), name.or(Some(url.to_string())));
            db.insert_source(&s).await?;
            info!("Created new source: {}", s.id);
            s
        }
    };

    // Start ingestion run
    let run = db.start_ingestion_run(&source.id).await?;

    // Create embedder
    let embedder = create_embedder(&config.embedding)?;

    // Create crawler
    let crawler = Crawler::new(config.crawl.clone())?;

    let mut stats = IngestStats::default();
    let mut current_uris: Vec<String> = Vec::new();

    // Crawl and process pages
    let pages = crawler.crawl(url, |_page| {
        // Continue callback - return true to keep crawling
        true
    }).await?;

    for page in pages {
        current_uris.push(page.url.clone());

        match process_page(
            config,
            db,
            store,
            embedder.as_ref(),
            &source,
            &page,
        )
        .await
        {
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
    }

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
        if stats.errors.is_empty() { RunStatus::Completed } else { RunStatus::Failed },
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
) -> Result<IngestStats> {
    use crate::crawl::SitemapParser;

    info!("Ingesting sitemap: {}", sitemap_url);

    // Parse sitemap to get URLs
    let parser = SitemapParser::new(&config.crawl.user_agent)?;
    let entries = parser.parse(sitemap_url).await?;
    
    if entries.is_empty() {
        warn!("No URLs found in sitemap: {}", sitemap_url);
        return Ok(IngestStats::default());
    }

    let max = max_pages.unwrap_or(config.crawl.max_pages);
    let entries: Vec<_> = entries.into_iter().take(max as usize).collect();
    info!("Found {} URLs in sitemap (limited to {})", entries.len(), max);

    // Get or create source
    let source = match db.get_source_by_uri(sitemap_url).await? {
        Some(s) => {
            info!("Found existing source: {}", s.id);
            s
        }
        None => {
            let s = Source::new(
                SourceType::Sitemap, 
                sitemap_url.to_string(), 
                name.or(Some(sitemap_url.to_string()))
            );
            db.insert_source(&s).await?;
            info!("Created new source: {}", s.id);
            s
        }
    };

    // Start ingestion run
    let run = db.start_ingestion_run(&source.id).await?;

    // Create embedder and crawler
    let embedder = create_embedder(&config.embedding)?;
    let crawler = Crawler::new(config.crawl.clone())?;

    let mut stats = IngestStats::default();
    let mut current_uris: Vec<String> = Vec::new();

    // Process each URL from sitemap
    for entry in entries {
        current_uris.push(entry.loc.clone());

        // Fetch the page
        match crawler.fetch(&entry.loc).await {
            Ok(page) => {
                match process_page(
                    config,
                    db,
                    store,
                    embedder.as_ref(),
                    &source,
                    &page,
                )
                .await
                {
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
    }

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
        if stats.errors.is_empty() { RunStatus::Completed } else { RunStatus::Failed },
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
    if let Some(existing_doc) = db.get_document_by_uri(&source.id, &page.url).await? {
        if existing_doc.content_hash == content_hash {
            debug!("Page unchanged: {}", page.url);
            return Ok((0, 0));
        }
    }

    // Parse content
    let parsed = parse_content(&page.content, page.content_type, Some(&page.url))?;

    // Create/update document
    let mut doc = Document::new(source.id.clone(), page.url.clone(), content_hash.clone());
    doc.title = page.title.clone().or(parsed.title.clone());
    doc.content_type = Some(format!("{:?}", page.content_type).to_lowercase());
    db.upsert_document(&doc).await?;

    // Chunk the document
    let chunks = chunk_document(&parsed, &content_hash, &config.chunk)?;

    if chunks.is_empty() {
        debug!("No chunks generated for: {}", page.url);
        return Ok((0, 0));
    }

    // Process chunks
    let (created, updated) = process_chunks(
        config, db, store, embedder, source, &doc, &page.url, chunks,
    )
    .await?;

    Ok((created, updated))
}
