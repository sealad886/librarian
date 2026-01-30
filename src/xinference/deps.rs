//! Xinference dependency management
//!
//! Handles checking and installing Python dependencies for Xinference.

use crate::error::{Error, Result};
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use tracing::{debug, info, warn};

/// Cached result of xinference installation check
static XINFERENCE_READY: OnceLock<bool> = OnceLock::new();

/// Check if Python is available and return the path to the Python executable
pub fn check_python() -> Result<PathBuf> {
    // Try python3 first, then python
    for cmd in ["python3", "python"] {
        if let Ok(output) = Command::new(cmd).args(["--version"]).output() {
            if output.status.success() {
                debug!("Found Python at: {}", cmd);
                return Ok(PathBuf::from(cmd));
            }
        }
    }
    Err(Error::Embedding(
        "Python not found in PATH. Please install Python 3.8+ to use Xinference backend.".into(),
    ))
}

/// Check if pip is available for the given Python
pub fn check_pip(python: &PathBuf) -> Result<()> {
    let output = Command::new(python)
        .args(["-m", "pip", "--version"])
        .output()
        .map_err(|e| Error::Embedding(format!("Failed to check pip: {}", e)))?;

    if output.status.success() {
        debug!("pip is available");
        Ok(())
    } else {
        Err(Error::Embedding(
            "pip not available. Please install pip to use Xinference backend.".into(),
        ))
    }
}

/// Check if xinference is installed
pub fn check_xinference_installed(python: &PathBuf) -> Result<bool> {
    let output = Command::new(python)
        .args(["-c", "import xinference; print(xinference.__version__)"])
        .output()
        .map_err(|e| Error::Embedding(format!("Failed to check xinference: {}", e)))?;

    Ok(output.status.success())
}

/// Get the installed xinference version
pub fn get_xinference_version(python: &PathBuf) -> Result<Option<String>> {
    let output = Command::new(python)
        .args(["-c", "import xinference; print(xinference.__version__)"])
        .output()
        .map_err(|e| Error::Embedding(format!("Failed to get xinference version: {}", e)))?;

    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(Some(version))
    } else {
        Ok(None)
    }
}

/// Install xinference with transformers support
pub fn install_xinference(python: &PathBuf) -> Result<()> {
    info!("Installing xinference[transformers]... This may take a few minutes.");

    let output = Command::new(python)
        .args([
            "-m",
            "pip",
            "install",
            "--quiet",
            "xinference[transformers]",
        ])
        .output()
        .map_err(|e| Error::Embedding(format!("Failed to run pip install: {}", e)))?;

    if output.status.success() {
        info!("xinference installed successfully");
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(Error::Embedding(format!(
            "Failed to install xinference: {}",
            stderr
        )))
    }
}

/// Ensure xinference is installed, installing it if necessary
pub fn ensure_xinference_installed(python: &PathBuf) -> Result<()> {
    if check_xinference_installed(python)? {
        if let Ok(Some(version)) = get_xinference_version(python) {
            debug!("xinference version {} is installed", version);
        }
        return Ok(());
    }

    info!("xinference not found, installing...");
    install_xinference(python)?;

    // Verify installation
    if !check_xinference_installed(python)? {
        return Err(Error::Embedding(
            "xinference installation failed verification".into(),
        ));
    }

    Ok(())
}

/// Check if xinference-local command is available in PATH
pub fn check_xinference_command() -> Result<bool> {
    match Command::new("xinference-local").arg("--help").output() {
        Ok(output) => Ok(output.status.success()),
        Err(_) => Ok(false),
    }
}

/// Ensure all xinference dependencies are ready.
/// Returns the Python path to use.
/// Results are cached for the lifetime of the process.
pub fn ensure_xinference_ready() -> Result<PathBuf> {
    // Fast path: already checked
    if XINFERENCE_READY.get().copied() == Some(true) {
        return check_python();
    }

    let python = check_python()?;
    check_pip(&python)?;
    ensure_xinference_installed(&python)?;

    // Verify the xinference-local command is available
    if !check_xinference_command()? {
        warn!("xinference-local command not found in PATH after installation");
        return Err(Error::Embedding(
            "xinference-local not found in PATH. Try restarting your terminal or activating your Python environment.".into(),
        ));
    }

    // Mark as ready
    let _ = XINFERENCE_READY.set(true);

    Ok(python)
}

/// Check if xinference dependencies are ready without installing
pub fn is_xinference_ready() -> bool {
    if XINFERENCE_READY.get().copied() == Some(true) {
        return true;
    }

    let Ok(python) = check_python() else {
        return false;
    };

    check_xinference_installed(&python).unwrap_or(false)
        && check_xinference_command().unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_python() {
        // This test assumes Python is installed on the test system
        // It's okay if it fails in environments without Python
        let result = check_python();
        // Just verify it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_check_xinference_command_when_not_installed() {
        // This should return Ok(false) or Ok(true) depending on system state
        let result = check_xinference_command();
        assert!(result.is_ok());
    }
}
