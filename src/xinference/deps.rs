//! Xinference dependency management
//!
//! Handles checking and installing Python dependencies for Xinference.

use crate::error::{Error, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use tracing::{debug, info, warn};

/// Cached Python path for the Xinference environment
static XINFERENCE_PYTHON: OnceLock<PathBuf> = OnceLock::new();

/// Ensure uv is available to manage the dedicated Python environment.
pub fn ensure_uv_available() -> Result<()> {
    match Command::new("uv").arg("--version").output() {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(Error::Embedding(format!(
                "uv is required to manage the Xinference Python environment. Install uv and retry. ({})",
                stderr.trim()
            )))
        }
        Err(e) => Err(Error::Embedding(format!(
            "uv is required to manage the Xinference Python environment but was not found: {}",
            e
        ))),
    }
}

fn xinference_venv_dir(base_dir: &Path) -> PathBuf {
    base_dir.join("xinference").join(".venv")
}

fn venv_bin_dir(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir.join("Scripts")
    } else {
        venv_dir.join("bin")
    }
}

fn venv_python_path(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_bin_dir(venv_dir).join("python.exe")
    } else {
        venv_bin_dir(venv_dir).join("python")
    }
}

fn venv_pip_path(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_bin_dir(venv_dir).join("pip.exe")
    } else {
        venv_bin_dir(venv_dir).join("pip")
    }
}

fn xinference_local_path(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_bin_dir(venv_dir).join("xinference-local.exe")
    } else {
        venv_bin_dir(venv_dir).join("xinference-local")
    }
}

fn python_is_310(python: &Path) -> Result<bool> {
    let output = Command::new(python)
        .args([
            "-c",
            "import sys; print(f\"{sys.version_info[0]}.{sys.version_info[1]}\")",
        ])
        .output()
        .map_err(|e| Error::Embedding(format!("Failed to check Python version: {}", e)))?;

    if !output.status.success() {
        return Ok(false);
    }

    let version = String::from_utf8_lossy(&output.stdout);
    Ok(version.trim() == "3.10")
}

fn ensure_venv_pip(python: &Path) -> Result<()> {
    let output = Command::new(python)
        .args(["-m", "ensurepip"])
        .output()
        .map_err(|e| Error::Embedding(format!("Failed to run ensurepip: {}", e)))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_ascii_lowercase();
    if output.status.success() || stdout.contains("a new release of pip is available") {
        let _pip_upgrade_output = Command::new(python)
            .args(["-m", "pip", "install", "--upgrade", "pip"])
            .output()
            .map_err(|e| Error::Embedding(format!("Failed to upgrade pip: {}", e)))?;
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(Error::Embedding(format!(
            "Failed to bootstrap pip in Xinference venv: {}",
            stderr.trim()
        )))
    }
}

fn ensure_xinference_venv(base_dir: &Path) -> Result<PathBuf> {
    ensure_uv_available()?;

    let venv_dir = xinference_venv_dir(base_dir);
    let python = venv_python_path(&venv_dir);
    let pip = venv_pip_path(&venv_dir);

    if python.exists() {
        if python_is_310(&python)? {
            debug!("Using existing Xinference venv at {}", venv_dir.display());
            return Ok(python);
        }

        warn!(
            "Xinference venv at {} is not Python 3.10; recreating",
            venv_dir.display()
        );
        fs::remove_dir_all(&venv_dir)?;
    }

    if let Some(parent) = venv_dir.parent() {
        fs::create_dir_all(parent)?;
    }

    let venv_dir_str = venv_dir
        .to_str()
        .ok_or_else(|| Error::Embedding("Invalid Xinference venv path (non-UTF8)".to_string()))?;

    let output = Command::new("uv")
        .args(["venv", "--python", "3.10", venv_dir_str])
        .output()
        .map_err(|e| Error::Embedding(format!("Failed to create Xinference venv: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::Embedding(format!(
            "Failed to create Xinference Python 3.10 environment: {}",
            stderr.trim()
        )));
    }

    if !python.exists() || !python_is_310(&python)? {
        return Err(Error::Embedding(
            "Xinference venv does not contain Python 3.10 after creation".into(),
        ));
    }

    if !pip.exists() {
        ensure_venv_pip(&python)?;
    }

    Ok(python)
}

fn ensure_venv_on_path(venv_dir: &Path) -> Result<()> {
    let bin_dir = venv_bin_dir(venv_dir);
    let bin_str = bin_dir.to_str().ok_or_else(|| {
        Error::Embedding("Invalid Xinference venv bin path (non-UTF8)".to_string())
    })?;

    let current = std::env::var_os("PATH").unwrap_or_default();
    let mut paths: Vec<PathBuf> = std::env::split_paths(&current).collect();
    if paths.iter().any(|p| p == &bin_dir) {
        return Ok(());
    }

    paths.insert(0, PathBuf::from(bin_str));
    let new_path = std::env::join_paths(paths).map_err(|e| {
        Error::Embedding(format!("Failed to update PATH for Xinference venv: {}", e))
    })?;
    std::env::set_var("PATH", new_path);
    Ok(())
}

/// Check if xinference is installed
pub fn check_xinference_installed(python: &Path) -> Result<bool> {
    let output = Command::new(python)
        .args(["-c", "import xinference; print(xinference.__version__)"])
        .output()
        .map_err(|e| Error::Embedding(format!("Failed to check xinference: {}", e)))?;

    Ok(output.status.success())
}

/// Get the installed xinference version
pub fn get_xinference_version(python: &Path) -> Result<Option<String>> {
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

/// Install xinference with all extras
pub fn install_xinference(python: &Path) -> Result<()> {
    info!("Installing xinference[all]... This may take a few minutes.");

    let python_str = python
        .to_str()
        .ok_or_else(|| Error::Embedding("Invalid Xinference Python path (non-UTF8)".to_string()))?;
    let output = Command::new("uv")
        .args([
            "pip",
            "install",
            "--python",
            python_str,
            "--quiet",
            "xinference[all]",
        ])
        .output()
        .map_err(|e| Error::Embedding(format!("Failed to run uv pip install: {}", e)))?;

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
pub fn ensure_xinference_installed(python: &Path) -> Result<()> {
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

/// Prepare the dedicated Xinference environment (used by init).
pub fn prepare_xinference_env(base_dir: &Path) -> Result<PathBuf> {
    let python = ensure_xinference_venv(base_dir)?;
    let venv_dir = xinference_venv_dir(base_dir);
    ensure_venv_on_path(&venv_dir)?;
    install_xinference(&python)?;
    ensure_xoscar_compatible(&python)?;

    if !check_xinference_command(&venv_dir)? {
        warn!("xinference-local command not found in Xinference venv after installation");
        return Err(Error::Embedding(
            "xinference-local not found in Xinference venv. Try reinstalling xinference or removing the venv to recreate it.".into(),
        ));
    }

    let _ = XINFERENCE_PYTHON.set(python.clone());
    Ok(python)
}

/// Check if xinference-local command is available in PATH
pub fn check_xinference_command(venv_dir: &Path) -> Result<bool> {
    let path = xinference_local_path(venv_dir);
    if !path.exists() {
        return Ok(false);
    }

    match Command::new(path).arg("--help").output() {
        Ok(output) => Ok(output.status.success()),
        Err(_) => Ok(false),
    }
}

/// Ensure all xinference dependencies are ready.
/// Returns the Python path to use.
/// Results are cached for the lifetime of the process.
/// Ensure all xinference dependencies are ready for a given base directory.
/// Returns the Python path to use.
/// Results are cached for the lifetime of the process.
pub fn ensure_xinference_ready(base_dir: &Path) -> Result<PathBuf> {
    // Fast path: already checked
    if let Some(python) = XINFERENCE_PYTHON.get() {
        return Ok(python.clone());
    }

    let python = ensure_xinference_venv(base_dir)?;
    let venv_dir = xinference_venv_dir(base_dir);
    ensure_venv_on_path(&venv_dir)?;
    ensure_xinference_installed(&python)?;
    ensure_xoscar_compatible(&python)?;

    // Verify the xinference-local command is available
    if !check_xinference_command(&venv_dir)? {
        warn!("xinference-local command not found in Xinference venv after installation");
        return Err(Error::Embedding(
            "xinference-local not found in Xinference venv. Try reinstalling xinference or removing the venv to recreate it.".into(),
        ));
    }

    // Mark as ready
    let _ = XINFERENCE_PYTHON.set(python.clone());

    Ok(python)
}

fn xoscar_supports_start_method(python: &Path) -> Result<Option<bool>> {
    let output = Command::new(python)
        .args([
            "-c",
            "import inspect; from xoscar.core.pool import MainActorPool; sig = inspect.signature(MainActorPool.append_sub_pool); print('start_method' in sig.parameters)",
        ])
        .output()
        .map_err(|e| Error::Embedding(format!("Failed to check xoscar: {}", e)))?;

    if !output.status.success() {
        return Ok(None);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let value = stdout.trim();
    if value.eq_ignore_ascii_case("true") {
        Ok(Some(true))
    } else if value.eq_ignore_ascii_case("false") {
        Ok(Some(false))
    } else {
        Ok(None)
    }
}

fn ensure_xoscar_compatible(python: &Path) -> Result<()> {
    let supports = xoscar_supports_start_method(python)?;
    if supports == Some(true) {
        return Ok(());
    }

    info!("Upgrading xoscar for Xinference compatibility...");
    let output = Command::new(python)
        .args(["-m", "pip", "install", "--quiet", "--upgrade", "xoscar"])
        .output()
        .map_err(|e| Error::Embedding(format!("Failed to run pip install: {}", e)))?;

    if output.status.success() && xoscar_supports_start_method(python)? == Some(true) {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(Error::Embedding(format!(
        "xoscar upgrade did not resolve Xinference compatibility: {}",
        stderr.trim()
    )))
}

/// Check if xinference dependencies are ready without installing
pub fn is_xinference_ready(base_dir: &Path) -> bool {
    if XINFERENCE_PYTHON.get().is_some() {
        return true;
    }

    let venv_dir = xinference_venv_dir(base_dir);
    let python = venv_python_path(&venv_dir);
    if !python.exists() {
        return false;
    }

    if python_is_310(&python).unwrap_or(false) {
        check_xinference_installed(&python).unwrap_or(false)
            && check_xinference_command(&venv_dir).unwrap_or(false)
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_python() {
        // This test assumes uv is installed and accessible on the test system
        // It's okay if it fails in environments without uv
        let base_dir = std::env::temp_dir().join("librarian-xinference-test");
        let result = ensure_xinference_venv(&base_dir);
        // Just verify it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_check_xinference_command_when_not_installed() {
        // This should return Ok(false) or Ok(true) depending on system state
        let base_dir = std::env::temp_dir().join("librarian-xinference-test");
        let result = check_xinference_command(&xinference_venv_dir(&base_dir));
        assert!(result.is_ok());
    }
}
