//! Content-addressed cache.
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
    /// Every field is length-prefixed: plain concatenation makes
    /// (`provider="ab"`, `model="c"`) and (`provider="a"`, `model="bc"`)
    /// hash identically, which would serve one model's artifact for another.
    pub fn key(
        provider: &str,
        model_id: &str,
        params: &serde_json::Value,
        input_hashes: &[String],
    ) -> String {
        use sha2::{Digest, Sha256};
        fn feed(hasher: &mut Sha256, bytes: &[u8]) {
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        }
        let mut hasher = Sha256::new();
        feed(&mut hasher, provider.as_bytes());
        feed(&mut hasher, model_id.as_bytes());
        feed(&mut hasher, params.to_string().as_bytes());
        feed(&mut hasher, &(input_hashes.len() as u64).to_le_bytes());
        for h in input_hashes {
            feed(&mut hasher, h.as_bytes());
        }
        hex::encode(hasher.finalize())
    }

    pub fn get(&self, key: &str) -> Option<std::path::PathBuf> {
        let path = self.root.join(key);
        path.exists().then_some(path)
    }

    /// Write through a temporary file and rename: a crash mid-write must not
    /// leave a truncated artifact that `get` later reports as a cache hit.
    pub fn put(&self, key: &str, bytes: &[u8]) -> anyhow::Result<std::path::PathBuf> {
        std::fs::create_dir_all(&self.root)?;
        let path = self.root.join(key);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp = self
            .root
            .join(format!(".{key}.{}.{nanos}.tmp", std::process::id()));
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, &path)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_boundaries_do_not_collide() {
        let params = serde_json::json!({});
        let left = Cache::key("ab", "c", &params, &[]);
        let right = Cache::key("a", "bc", &params, &[]);
        assert_ne!(left, right);
        let one = Cache::key("fal", "m", &params, &["h1".into()]);
        let two = Cache::key("fal", "m", &params, &["h1h2".into()]);
        assert_ne!(one, two);
    }

    #[test]
    fn put_then_get_roundtrip_leaves_no_temp_file() {
        let root = std::env::temp_dir().join(format!("avatar-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let cache = Cache::new(root.clone());
        let key = Cache::key("fal", "m", &serde_json::json!({"w": 1}), &[]);
        assert!(cache.get(&key).is_none());
        let path = cache.put(&key, b"payload").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"payload");
        assert_eq!(cache.get(&key).unwrap(), path);
        let leftovers: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temp files left behind: {leftovers:?}");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
