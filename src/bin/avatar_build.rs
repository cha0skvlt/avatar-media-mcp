//! Minimal avatar build: normalize photos → embed → dedupe → write
//! avatars/<slug>/ with avatar.json, consent.json, identity/face_embedding.json.
//!
//! Usage: cargo run --features local-identity --bin avatar_build -- \
//!   --name ksenia --photos ../Ksenia --consent self

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use avatar_media_mcp::domain::avatar::{Avatar, Consent, ConsentSubject, Identity, IdentityModel, PromptBlocks, RefBinding, Refs, SCHEMA_VERSION};
use avatar_media_mcp::identity::arcface::FaceEngine;
use avatar_media_mcp::identity::embeddings::{cosine_similarity_checked, EmbeddingSource};

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

    // 1. Collect + normalize photos into refs/.
    let avatar_dir = PathBuf::from("avatars").join(&name);
    let refs_dir = avatar_dir.join("refs");
    std::fs::create_dir_all(&refs_dir)?;
    let identity_dir = avatar_dir.join("identity");
    std::fs::create_dir_all(&identity_dir)?;

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
        // Normalize: RGB, long side ≤2048, EXIF stripped (image crate decode drops it).
        let img = image::open(f).with_context(|| f.display().to_string())?.to_rgb8();
        let (w, h) = (img.width(), img.height());
        let img = if w.max(h) > 2048 {
            let s = 2048.0 / w.max(h) as f32;
            image::imageops::resize(&img, (w as f32 * s) as u32, (h as f32 * s) as u32, image::imageops::FilterType::Triangle)
        } else {
            img
        };
        let norm_path = refs_dir.join(format!("{:02}.png", kept.len() + rejected.len() + 1));
        img.save(&norm_path)?;

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
        // Dedupe: cosine >0.95 against an already-kept vector = duplicate.
        if kept.iter().any(|(_, kv)| cosine_similarity_checked(&kv, &v).map(|c| c.get()).unwrap_or(0.0) > 0.95) {
            rejected.push(format!("{}: duplicate", f.display()));
            std::fs::remove_file(&norm_path)?;
            continue;
        }
        // Re-number refs sequentially for kept photos.
        let final_path = refs_dir.join(format!("{:02}.png", kept.len() + 1));
        if norm_path != final_path {
            std::fs::rename(&norm_path, &final_path)?;
        }
        kept.push((final_path.file_name().unwrap().to_string_lossy().to_string(), v));
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
    let median = sims[sims.len() / 2];

    // 4. Write identity/face_embedding.json.
    let model = IdentityModel {
        face_embeddings: kept.iter().map(|(n, v)| (n.clone(), v.clone())).collect(),
        face_model: None,
        body_model: None,
        identity_threshold: Some(median - 0.10), // conservative per-avatar pass floor
    };
    std::fs::write(
        identity_dir.join("face_embedding.json"),
        serde_json::to_vec_pretty(&model)?,
    )?;

    // 5. consent.json + avatar.json.
    let consent_doc = Consent {
        subject: consent,
        statement: "CLI test build; operator asserts rights to the reference images".into(),
        date: chrono::Utc::now().date_naive().to_string(),
        scope: vec!["test".into()],
        written_consent_path: None,
    };
    std::fs::write(
        avatar_dir.join("consent.json"),
        serde_json::to_vec_pretty(&consent_doc)?,
    )?;

    let avatar = Avatar {
        schema_version: SCHEMA_VERSION,
        slug: name.clone(),
        display_name: name.clone(),
        consent_ref: "consent.json".into(),
        identity: Identity {
            apparent_age_range: "25-30".into(),
            gender_presentation: "female".into(),
            face_shape: "oval, tapering to a pointed chin".into(),
            skin_tone: "fair, warm undertone".into(),
            eyes: "almond, slightly hooded, grey-blue".into(),
            brows: "natural, medium thickness".into(),
            nose: "straight, narrow bridge, slightly rounded tip".into(),
            lips: "medium, defined cupid's bow".into(),
            hair: avatar_media_mcp::domain::avatar::Hair {
                color: "auburn/copper".into(),
                length: "mid-back".into(),
                style: "dreadlocks".into(),
            },
            facial_hair: None,
            distinguishing_marks: vec!["black geometric tattoo right shoulder".into()],
            build: "slim, ectomorph".into(),
        },
        persona: None,
        voice: None,
        prompt_blocks: avatar_media_mcp::domain::avatar::PromptBlocks {
            identity: format!(
                "A woman with shoulder-length auburn dreadlocks, {} reference photos, median self-cosine {:.3}",
                kept.len(), median
            ),
            negative: "different person, altered face shape, extra fingers, beauty-filter skin".into(),
            do_not_change: vec!["face shape".into(), "hair color".into(), "eye color".into()],
            ref_binding: avatar_media_mcp::domain::avatar::RefBinding {
                gemini_image: None,
                omni_video: None,
                generic: "same person as in the reference image(s)".into(),
            },
        },
        refs: avatar_media_mcp::domain::avatar::Refs {
            primary: format!("refs/{}", kept[0].0),
            set: kept.iter().map(|(n, _)| format!("refs/{n}")).collect(),
        },
        created_at: chrono::Utc::now().to_rfc3339(),
        canon_approved_at: None,
    };
    std::fs::write(
        avatar_dir.join("avatar.json"),
        serde_json::to_vec_pretty(&avatar)?,
    )?;

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
