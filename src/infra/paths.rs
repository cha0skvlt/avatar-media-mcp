//! Path policy for tool arguments.
//!
//! The MCP server is driven by an agent: a prompt-injected call must not be
//! able to write outside the workspace. Reads stay unrestricted (the operator
//! points at photos wherever they live); writes are confined.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Resolve `path` and require its parent to be inside `root` (the workspace).
/// The parent is created when missing.
pub fn confine_output(path: &Path, root: &Path) -> Result<PathBuf> {
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("output path {:?} has no file name", path))?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create {}", parent.display()))?;
    let parent = parent
        .canonicalize()
        .with_context(|| format!("resolve {}", parent.display()))?;
    let root = root
        .canonicalize()
        .with_context(|| format!("resolve workspace {}", root.display()))?;
    anyhow::ensure!(
        parent.starts_with(&root),
        "output path {:?} is outside the workspace {}",
        path,
        root.display()
    );
    Ok(parent.join(file_name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_inside_the_workspace_is_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.png");
        let resolved = confine_output(&target, dir.path()).unwrap();
        assert!(resolved.starts_with(dir.path().canonicalize().unwrap()));
    }

    #[test]
    fn output_outside_the_workspace_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("out.png");
        let error = confine_output(&target, dir.path()).unwrap_err().to_string();
        assert!(error.contains("outside the workspace"), "{error}");
    }

    #[test]
    fn traversal_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("..").join("escape.png");
        assert!(confine_output(&target, dir.path()).is_err());
    }
}
