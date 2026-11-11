//! Minimal avatar build: normalize photos → embed → dedupe → write
//! `avatars/<slug>/` through the same `AvatarStore` the MCP tool uses.
//!
//! Usage: cargo run --features local-identity --bin avatar_build -- \
//!   --name <slug> --photos <dir> --consent self

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use avatar_media_mcp::domain::avatar::{
    Avatar, Consent, ConsentSubject, Identity, IdentityModel, PromptBlocks, RefBinding, Refs,
    SCHEMA_VERSION,
};
use avatar_media_mcp::identity::arcface::FaceEngine;
use avatar_media_mcp::identity::embeddings::cosine_similarity_checked;
use avatar_media_mcp::infra::avatar_store::{AvatarStore, CONSENT_FILE};
use avatar_media_mcp::pipeline::avatar_build::normalize_photo;

fn main() -> Result<()> {
    let mut name = String::new();
    let mut photos_dir = PathBuf::new();
    let mut consent = ConsentSubject::Self_;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--name" => name = args.next().context("--name needs a value")?,
            "--photos" => photos_dir = PathBuf::from(args.next().context("--photos needs a value")?),
            "--consent" => {
                consent = match args.next().context("--consent needs a value")?.as_str() {
                    "self" => ConsentSubject::Self_,
                    "fictional" => ConsentSubject::Fictional,
                    other => bail!("unsupported consent for CLI test: {other} (use self|fictional)"),
                }
            }
            other => bail!("unknown arg: {other}"),
        }
    }
    if name.is_empty() || !photos_dir.exists() {
        bail!("usage: avatar_build --name <slug> --photos <dir> [--consent self|fictional]");
    }

    // 1. Collect photos and normalize them into refs/ (the canon lives inside
    //    the avatar directory — no absolute paths to the source folder).
    let store = AvatarStore::default();
    AvatarStore::validate_slug(&name)?;
    let refs_dir = store.refs_dir(&name);
    std::fs::create_dir_all(&refs_dir)?;

    let mut files: Vec<PathBuf> = std::fs::read_dir(&photos_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    anyhow::ensure!(!files.is_empty(), "no images in {}", photos_dir.display());

    // 2. Engine + embed every normalized photo.
    let mut engine = FaceEngine::load(Path::new("models/insightface/buffalo_l"))?;

    let mut kept: Vec<(String, Vec<f32>)> = Vec::new();
    let mut rejected: Vec<String> = Vec::new();
    for f in &files {
        let norm_path = refs_dir.join(format!("{:02}.jpg", kept.len() + rejected.len() + 1));
        normalize_photo(f, &norm_path)?;

        let faces = engine.embed_image(&norm_path)?;
        if faces.is_empty() {
            rejected.push(format!("{}: no face", f.display()));
            std::fs::remove_file(&norm_path)?;
            continue;
        }
        if faces.len() > 1 {
            rejected.push(format!("{}: {} faces", f.display(), faces.len()));
            std::fs::remove_file(&norm_path)?;
            continue;
        }
        let v = faces[0].vector.clone();
        // Same near-duplicate rule as the MCP tool: only identical frames go.
        if kept
            .iter()
            .any(|(_, kv)| cosine_similarity_checked(kv, &v).map(|c| c.get()).unwrap_or(0.0) > 0.9999)
        {
            rejected.push(format!("{}: duplicate", f.display()));
            std::fs::remove_file(&norm_path)?;
            continue;
        }
        // Re-number refs sequentially for kept photos.
        let final_path = refs_dir.join(format!("{:02}.jpg", kept.len() + 1));
        if norm_path != final_path {
            std::fs::rename(&norm_path, &final_path)?;
        }
        kept.push((
            final_path.file_name().unwrap().to_string_lossy().to_string(),
            v,
        ));
    }
    anyhow::ensure!(!kept.is_empty(), "no usable photos");

    // 3. Median self-similarity as the canon anchor score.
    let mut sims = Vec::new();
    for i in 0..kept.len() {
        for j in (i + 1)..kept.len() {
            sims.push(cosine_similarity_checked(&kept[i].1, &kept[j].1)?.get());
        }
    }
    sims.sort_by(|a, b| a.total_cmp(b));
    // One usable photo has no pair to compare against: the anchor is 1.0.
    let median = if sims.is_empty() { 1.0 } else { sims[sims.len() / 2] };

    // 4. consent.json + avatar.json + identity_model.json via the store.
    let consent_doc = Consent {
        subject: consent,
        statement: "CLI test build; operator asserts rights to the reference images".into(),
        date: chrono::Utc::now().date_naive().to_string(),
        scope: vec!["test".into()],
        written_consent_path: None,
    };
    std::fs::write(
        store.dir(&name).join(CONSENT_FILE),
        serde_json::to_vec_pretty(&consent_doc)?,
    )?;

    let model = IdentityModel {
        face_embeddings: kept
            .iter()
            .map(|(n, v)| (format!("refs/{n}"), v.clone()))
            .collect(),
        face_model: None,
        body_model: None,
        // Same band as the clustering path: a lucky pair cannot unlock the gate.
        identity_threshold: Some((median - 0.10).clamp(0.55, 0.93)),
    };

    let avatar = Avatar {
        schema_version: SCHEMA_VERSION,
        slug: name.clone(),
        display_name: name.clone(),
        consent_ref: CONSENT_FILE.into(),
        // The CLI has no vision channel: it must not invent a biometric
        // description. The numeric lock is the canon until a channel fills it.
        identity: Identity {
            apparent_age_range: String::new(),
            gender_presentation: String::new(),
            face_shape: String::new(),
            skin_tone: String::new(),
            eyes: String::new(),
            brows: String::new(),
            nose: String::new(),
            lips: String::new(),
            hair: avatar_media_mcp::domain::avatar::Hair {
                color: String::new(),
                length: String::new(),
                style: String::new(),
            },
            facial_hair: None,
            distinguishing_marks: vec![],
            build: String::new(),
        },
        persona: None,
        voice: None,
        prompt_blocks: PromptBlocks {
            identity: format!(
                "The same person in every frame. {} reference photos, median self-cosine {:.3}",
                kept.len(),
                median
            ),
            negative: "different person, altered face shape, changed eye colour, beauty-filter skin, extra fingers, deformed hands".into(),
            do_not_change: vec![],
            ref_binding: RefBinding {
                gemini_image: None,
                omni_video: None,
                generic: "same person as in the reference image(s)".into(),
            },
        },
        refs: Refs {
            primary: format!("refs/{}", kept[0].0),
            set: kept.iter().map(|(n, _)| format!("refs/{n}")).collect(),
        },
        created_at: chrono::Utc::now().to_rfc3339(),
        canon_approved_at: None,
    };
    store.save(&avatar, &model)?;

    println!(
        "avatar '{name}': {} refs kept, {} rejected, median self-cosine {:.3}, threshold {:.3}\nrejected: {}",
        kept.len(),
        rejected.len(),
        median,
        model.identity_threshold.unwrap(),
        if rejected.is_empty() { "-".into() } else { rejected.join("; ") }
    );
    Ok(())
}
