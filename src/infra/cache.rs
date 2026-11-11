//! Content-addressed cache (ARCHITECTURE.md 2.6, 8.1).
//!
//! Key: hash(provider + model + params + input hashes). A crashed server
//! resumes without paying twice.

pub struct Cache {
    root: std::path::PathBuf,
}

impl Cache {
    pub fn new(root: std::path::PathBuf) -> Self {
        Self { root }
    }

    /// Cache key from provider, model, params and input file hashes.
    pub fn key(
        provider: &str,
        model_id: &str,
        params: &serde_json::Value,
        input_hashes: &[String],
    ) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(provider.as_bytes());
        hasher.update(model_id.as_bytes());
        hasher.update(params.to_string().as_bytes());
        for h in input_hashes {
            hasher.update(h.as_bytes());
        }
        hex::encode(hasher.finalize())
    }

    pub fn get(&self, key: &str) -> Option<std::path::PathBuf> {
        let path = self.root.join(key);
        path.exists().then_some(path)
    }

    pub fn put(&self, key: &str, bytes: &[u8]) -> anyhow::Result<std::path::PathBuf> {
        std::fs::create_dir_all(&self.root)?;
        let path = self.root.join(key);
        std::fs::write(&path, bytes)?;
        Ok(path)
    }
}

/// Hash a file's contents.
pub fn file_hash(path: &std::path::Path) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(hex::encode(hasher.finalize()))
}
