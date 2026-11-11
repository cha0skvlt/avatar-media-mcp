//! Canonical avatar layout.
//!
//! ```text
//! avatars/<slug>/
//! ├── avatar.json          # every path inside is relative to this directory
//! ├── consent.json
//! ├── identity_model.json
//! └── refs/NN.jpg          # normalized copies — the canon
//! ```
//!
//! One writer, one reader. `save` refuses an avatar whose refs are absolute or
//! missing, so an avatar directory is always self-contained; `load` refuses a
//! directory whose refs disappeared. There is no second layout.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::domain::avatar::{Avatar, Consent, IdentityModel, SCHEMA_VERSION};

pub const AVATAR_FILE: &str = "avatar.json";
pub const CONSENT_FILE: &str = "consent.json";
pub const MODEL_FILE: &str = "identity_model.json";
pub const REFS_DIR: &str = "refs";

/// A loaded avatar with its refs resolved to absolute paths.
#[derive(Debug, Clone)]
pub struct LoadedAvatar {
    pub avatar: Avatar,
    pub model: IdentityModel,
    pub refs: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct AvatarStore {
    root: PathBuf,
}

impl Default for AvatarStore {
    fn default() -> Self {
        Self::new("avatars")
    }
}

impl AvatarStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn dir(&self, slug: &str) -> PathBuf {
        self.root.join(slug)
    }

    pub fn refs_dir(&self, slug: &str) -> PathBuf {
        self.dir(slug).join(REFS_DIR)
    }

    /// Slug is a path component: reject anything that could escape the root.
    pub fn validate_slug(slug: &str) -> Result<()> {
        anyhow::ensure!(
            !slug.is_empty() && slug.len() <= 64,
            "slug must be 1..=64 characters"
        );
        anyhow::ensure!(
            slug.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_'),
            "slug may contain only a-z, 0-9, '-' and '_' (got {slug:?})"
        );
        Ok(())
    }

    /// Write `avatar.json` + `identity_model.json` in place. Prefer
    /// `begin_staging` for a rebuild: an in-place write leaves a half-updated
    /// avatar if the process dies between the two files.
    pub fn save(&self, avatar: &Avatar, model: &IdentityModel) -> Result<PathBuf> {
        Self::validate_slug(&avatar.slug)?;
        let dir = self.dir(&avatar.slug);
        std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        Self::save_into(&dir, avatar, model)
    }

    /// Build a new avatar directory beside the live one and swap it in on
    /// `commit`. A failed rebuild leaves the existing avatar untouched.
    pub fn begin_staging(&self, slug: &str) -> Result<Staging> {
        Self::validate_slug(slug)?;
        std::fs::create_dir_all(&self.root)
            .with_context(|| format!("create {}", self.root.display()))?;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = self.root.join(format!(".staging-{slug}-{nanos}"));
        std::fs::create_dir_all(dir.join(REFS_DIR))
            .with_context(|| format!("create {}", dir.display()))?;
        Ok(Staging {
            root: self.root.clone(),
            slug: slug.to_string(),
            dir,
            committed: false,
        })
    }

    /// Read and validate `consent.json`. The consent gate is a document, not a
    /// file-existence check.
    pub fn load_consent(&self, slug: &str) -> Result<Consent> {
        Self::validate_slug(slug)?;
        let path = self.dir(slug).join(CONSENT_FILE);
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        let consent: Consent = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", path.display()))?;
        Ok(consent)
    }

    fn save_into(dir: &Path, avatar: &Avatar, model: &IdentityModel) -> Result<PathBuf> {
        for reference in &avatar.refs.set {
            let path = Path::new(reference);
            anyhow::ensure!(
                path.is_relative(),
                "ref {reference:?} must be relative to the avatar directory"
            );
            anyhow::ensure!(
                dir.join(path).is_file(),
                "ref {reference:?} does not exist under {}",
                dir.display()
            );
        }
        anyhow::ensure!(
            !avatar.refs.primary.is_empty(),
            "avatar {} has no primary ref",
            avatar.slug
        );
        anyhow::ensure!(
            avatar.refs.set.contains(&avatar.refs.primary),
            "primary ref {:?} is not in refs.set",
            avatar.refs.primary
        );

        let avatar_path = dir.join(AVATAR_FILE);
        write_atomic(&avatar_path, &serde_json::to_vec_pretty(avatar)?)?;
        let model_path = dir.join(MODEL_FILE);
        write_atomic(&model_path, &serde_json::to_vec_pretty(model)?)?;
        Ok(avatar_path)
    }

    /// Read an avatar back. Fails when the model file is missing or when a ref
    /// no longer resolves — a half-avatar is an error, not a warning.
    pub fn load(&self, slug: &str) -> Result<LoadedAvatar> {
        Self::validate_slug(slug)?;
        let dir = self.dir(slug);
        let avatar_path = dir.join(AVATAR_FILE);
        let model_path = dir.join(MODEL_FILE);
        let avatar: Avatar = serde_json::from_str(
            &std::fs::read_to_string(&avatar_path)
                .with_context(|| format!("read {}", avatar_path.display()))?,
        )
        .with_context(|| format!("parse {}", avatar_path.display()))?;
        let model: IdentityModel = serde_json::from_str(
            &std::fs::read_to_string(&model_path)
                .with_context(|| format!("read {}", model_path.display()))?,
        )
        .with_context(|| format!("parse {}", model_path.display()))?;

        anyhow::ensure!(
            avatar.schema_version == SCHEMA_VERSION,
            "avatar {slug} has schema_version {}, this build reads {SCHEMA_VERSION}",
            avatar.schema_version
        );
        anyhow::ensure!(
            avatar.slug == slug,
            "avatar directory {slug} contains slug {:?}",
            avatar.slug
        );

        let mut refs = Vec::with_capacity(avatar.refs.set.len());
        let mut missing = Vec::new();
        for reference in &avatar.refs.set {
            let path = dir.join(reference);
            if path.is_file() {
                refs.push(path);
            } else {
                missing.push(reference.clone());
            }
        }
        anyhow::ensure!(
            missing.is_empty(),
            "avatar {slug} is not self-contained: missing refs {missing:?}"
        );
        Ok(LoadedAvatar {
            avatar,
            model,
            refs,
        })
    }

    /// Slugs of every directory that holds an `avatar.json`, sorted.
    pub fn list(&self) -> Result<Vec<String>> {
        if !self.root.is_dir() {
            return Ok(Vec::new());
        }
        let mut slugs = Vec::new();
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            if entry.path().join(AVATAR_FILE).is_file() {
                slugs.push(entry.file_name().to_string_lossy().to_string());
            }
        }
        slugs.sort();
        Ok(slugs)
    }
}

/// A staged avatar directory. Dropping it without `commit` removes it.
#[derive(Debug)]
pub struct Staging {
    root: PathBuf,
    slug: String,
    dir: PathBuf,
    committed: bool,
}

impl Staging {
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn refs_dir(&self) -> PathBuf {
        self.dir.join(REFS_DIR)
    }

    /// Validate and write the staged files, then swap the directory into place.
    pub fn commit(mut self, avatar: &Avatar, model: &IdentityModel) -> Result<PathBuf> {
        anyhow::ensure!(
            avatar.slug == self.slug,
            "staged slug {} does not match avatar slug {}",
            self.slug,
            avatar.slug
        );
        AvatarStore::save_into(&self.dir, avatar, model)?;
        let final_dir = self.root.join(&self.slug);
        let backup = self.root.join(format!(
            ".old-{}-{}",
            self.slug,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let had_previous = final_dir.exists();
        if had_previous {
            std::fs::rename(&final_dir, &backup)
                .with_context(|| format!("move {} aside", final_dir.display()))?;
        }
        if let Err(error) = std::fs::rename(&self.dir, &final_dir) {
            if had_previous {
                let _ = std::fs::rename(&backup, &final_dir);
            }
            return Err(error).with_context(|| format!("swap into {}", final_dir.display()));
        }
        if had_previous {
            let _ = std::fs::remove_dir_all(&backup);
        }
        self.committed = true;
        Ok(final_dir.join(AVATAR_FILE))
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_extension(format!("{}.{nanos}.tmp", std::process::id()));
    std::fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("rename into {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::avatar::{Hair, Identity, PromptBlocks, RefBinding, Refs, SCHEMA_VERSION};

    fn identity() -> Identity {
        Identity {
            apparent_age_range: "30-35".into(),
            gender_presentation: "androgynous".into(),
            face_shape: "oval".into(),
            skin_tone: "olive".into(),
            eyes: "brown".into(),
            brows: "straight".into(),
            nose: "narrow".into(),
            lips: "medium".into(),
            hair: Hair {
                color: "black".into(),
                length: "short".into(),
                style: "crop".into(),
            },
            facial_hair: None,
            distinguishing_marks: vec![],
            build: "medium".into(),
        }
    }

    fn avatar(slug: &str, refs: Vec<String>) -> Avatar {
        Avatar {
            schema_version: SCHEMA_VERSION,
            slug: slug.into(),
            display_name: slug.into(),
            consent_ref: CONSENT_FILE.into(),
            identity: identity(),
            persona: None,
            voice: None,
            prompt_blocks: PromptBlocks {
                identity: String::new(),
                negative: String::new(),
                do_not_change: vec![],
                ref_binding: RefBinding {
                    gemini_image: None,
                    omni_video: None,
                    generic: "same person".into(),
                },
            },
            refs: Refs {
                primary: refs.first().cloned().unwrap_or_default(),
                set: refs,
            },
            created_at: "2026-11-12T00:00:00Z".into(),
            canon_approved_at: None,
        }
    }

    fn model() -> IdentityModel {
        IdentityModel {
            face_embeddings: [("refs/01.jpg".to_string(), vec![1.0, 0.0])]
                .into_iter()
                .collect(),
            face_model: None,
            body_model: None,
            identity_threshold: Some(0.72),
        }
    }

    fn store_with_refs() -> (tempfile::TempDir, AvatarStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = AvatarStore::new(dir.path());
        std::fs::create_dir_all(store.refs_dir("smoke")).unwrap();
        std::fs::write(store.refs_dir("smoke").join("01.jpg"), b"jpg").unwrap();
        (dir, store)
    }

    #[test]
    fn save_then_load_roundtrips() {
        let (_dir, store) = store_with_refs();
        let avatar = avatar("smoke", vec!["refs/01.jpg".into()]);
        store.save(&avatar, &model()).unwrap();

        let loaded = store.load("smoke").unwrap();
        assert_eq!(loaded.avatar.identity.eyes, "brown");
        assert_eq!(loaded.model.identity_threshold, Some(0.72));
        assert_eq!(loaded.refs.len(), 1);
        assert!(loaded.refs[0].is_absolute());
        assert_eq!(store.list().unwrap(), vec!["smoke".to_string()]);
    }

    #[test]
    fn absolute_refs_are_refused() {
        let (_dir, store) = store_with_refs();
        let avatar = avatar("smoke", vec!["/tmp/elsewhere/01.png".into()]);
        let error = store.save(&avatar, &model()).unwrap_err().to_string();
        assert!(error.contains("relative"), "unexpected error: {error}");
    }

    #[test]
    fn missing_ref_is_refused_on_save() {
        let (_dir, store) = store_with_refs();
        let avatar = avatar("smoke", vec!["refs/99.png".into()]);
        let error = store.save(&avatar, &model()).unwrap_err().to_string();
        assert!(error.contains("does not exist"), "unexpected error: {error}");
    }

    #[test]
    fn primary_must_be_in_the_set() {
        let (_dir, store) = store_with_refs();
        let mut avatar = avatar("smoke", vec!["refs/01.jpg".into()]);
        avatar.refs.primary = "refs/02.png".into();
        let error = store.save(&avatar, &model()).unwrap_err().to_string();
        assert!(error.contains("primary"), "unexpected error: {error}");
    }

    #[test]
    fn load_reports_a_ref_that_disappeared() {
        let (_dir, store) = store_with_refs();
        let avatar = avatar("smoke", vec!["refs/01.jpg".into()]);
        store.save(&avatar, &model()).unwrap();
        std::fs::remove_file(store.refs_dir("smoke").join("01.jpg")).unwrap();
        let error = store.load("smoke").unwrap_err().to_string();
        assert!(error.contains("not self-contained"), "unexpected error: {error}");
    }

    #[test]
    fn load_without_a_model_file_is_an_error() {
        let (_dir, store) = store_with_refs();
        let avatar = avatar("smoke", vec!["refs/01.jpg".into()]);
        store.save(&avatar, &model()).unwrap();
        std::fs::remove_file(store.dir("smoke").join(MODEL_FILE)).unwrap();
        assert!(store.load("smoke").is_err());
    }

    #[test]
    fn staging_commit_replaces_the_live_avatar() {
        let (_dir, store) = store_with_refs();
        let first = avatar("smoke", vec!["refs/01.jpg".into()]);
        store.save(&first, &model()).unwrap();

        let staging = store.begin_staging("smoke").unwrap();
        std::fs::write(staging.refs_dir().join("01.jpg"), b"new").unwrap();
        let mut second = avatar("smoke", vec!["refs/01.jpg".into()]);
        second.identity.eyes = "green".into();
        staging.commit(&second, &model()).unwrap();

        let loaded = store.load("smoke").unwrap();
        assert_eq!(loaded.avatar.identity.eyes, "green");
        assert_eq!(std::fs::read(store.refs_dir("smoke").join("01.jpg")).unwrap(), b"new");
        // No staging or backup directories survive a commit.
        let leftovers: Vec<String> = std::fs::read_dir(store.root())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with('.'))
            .collect();
        assert!(leftovers.is_empty(), "leftovers: {leftovers:?}");
    }

    #[test]
    fn dropped_staging_leaves_the_live_avatar_untouched() {
        let (_dir, store) = store_with_refs();
        let live = avatar("smoke", vec!["refs/01.jpg".into()]);
        store.save(&live, &model()).unwrap();
        {
            let staging = store.begin_staging("smoke").unwrap();
            std::fs::write(staging.refs_dir().join("01.jpg"), b"broken").unwrap();
            // Dropped without commit.
        }
        let loaded = store.load("smoke").unwrap();
        assert_eq!(loaded.avatar.identity.eyes, "brown");
        assert_eq!(std::fs::read(store.refs_dir("smoke").join("01.jpg")).unwrap(), b"jpg");
    }

    #[test]
    fn schema_version_mismatch_is_refused() {
        let (_dir, store) = store_with_refs();
        let mut avatar = avatar("smoke", vec!["refs/01.jpg".into()]);
        store.save(&avatar, &model()).unwrap();
        avatar.schema_version = SCHEMA_VERSION + 1;
        // Write the newer document directly, bypassing save's own checks.
        std::fs::write(
            store.dir("smoke").join(AVATAR_FILE),
            serde_json::to_vec_pretty(&avatar).unwrap(),
        )
        .unwrap();
        let error = store.load("smoke").unwrap_err().to_string();
        assert!(error.contains("schema_version"), "unexpected error: {error}");
    }

    #[test]
    fn slug_mismatch_is_refused() {
        let (_dir, store) = store_with_refs();
        let avatar = avatar("smoke", vec!["refs/01.jpg".into()]);
        store.save(&avatar, &model()).unwrap();
        let mut wrong = avatar.clone();
        wrong.slug = "other".into();
        std::fs::write(
            store.dir("smoke").join(AVATAR_FILE),
            serde_json::to_vec_pretty(&wrong).unwrap(),
        )
        .unwrap();
        let error = store.load("smoke").unwrap_err().to_string();
        assert!(error.contains("contains slug"), "unexpected error: {error}");
    }

    #[test]
    fn consent_is_parsed_not_just_present() {
        let (_dir, store) = store_with_refs();
        let avatar = avatar("smoke", vec!["refs/01.jpg".into()]);
        store.save(&avatar, &model()).unwrap();
        std::fs::write(store.dir("smoke").join(CONSENT_FILE), b"not json").unwrap();
        assert!(store.load_consent("smoke").is_err());

        let consent = crate::domain::avatar::Consent {
            subject: crate::domain::avatar::ConsentSubject::ThirdParty,
            statement: "signed".into(),
            date: "2026-11-12".into(),
            scope: vec!["avatar".into()],
            written_consent_path: None,
        };
        std::fs::write(
            store.dir("smoke").join(CONSENT_FILE),
            serde_json::to_vec(&consent).unwrap(),
        )
        .unwrap();
        let parsed = store.load_consent("smoke").unwrap();
        assert_eq!(parsed.subject, crate::domain::avatar::ConsentSubject::ThirdParty);
    }

    #[test]
    fn slug_traversal_is_refused() {
        let (_dir, store) = store_with_refs();
        assert!(AvatarStore::validate_slug("smoke").is_ok());
        for bad in ["", "../x", "a/b", "A", ".", ".."] {
            assert!(AvatarStore::validate_slug(bad).is_err(), "accepted {bad:?}");
        }
        assert!(store.load("../x").is_err());
    }
}
