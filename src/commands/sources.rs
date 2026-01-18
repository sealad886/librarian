//! Source management commands (rename, etc.)

use crate::error::Result;
use crate::meta::{MetaDb, Source};

/// Rename a source's display name
pub async fn cmd_rename_source(db: &MetaDb, source_id: &str, new_name: String) -> Result<Source> {
    db.update_source_name(source_id, Some(new_name)).await?;
    let updated = db
        .get_source(source_id)
        .await?
        .ok_or_else(|| crate::error::Error::SourceNotFound(source_id.to_string()))?;
    Ok(updated)
}

/// Clear a source's name (use URI as display)
pub async fn cmd_clear_source_name(db: &MetaDb, source_id: &str) -> Result<Source> {
    db.update_source_name(source_id, None).await?;
    let updated = db
        .get_source(source_id)
        .await?
        .ok_or_else(|| crate::error::Error::SourceNotFound(source_id.to_string()))?;
    Ok(updated)
}
