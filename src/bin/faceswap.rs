//! Face-swap CLI (FS1): swap the identity of image A onto the
//! face in image B, save the result, and report timing.
//!
//! Usage: `cargo run --features local-identity --bin faceswap -- \
//!   <source_face_img> <target_img> [out.png]`
//!
//! Requires ORT_DYLIB_PATH pointing at libonnxruntime.dylib.

use std::time::Instant;

use avatar_media_mcp::identity::arcface::FaceEngine;
use avatar_media_mcp::pipeline::faceswap::SwapEngine;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 3,
        "usage: faceswap <source_face_img> <target_img> [out.png]"
    );
    let src_path = std::path::Path::new(&args[1]);
    let tgt_path = std::path::Path::new(&args[2]);
    let out_path = std::path::Path::new(args.get(3).map(String::as_str).unwrap_or("swap_out.png"));

    // 1. Source identity: ArcFace embedding of the main face.
    let mut engine = FaceEngine::load(std::path::Path::new("models/insightface/buffalo_l"))?;
    let t0 = Instant::now();
    let src_faces = engine.embed_image(src_path)?;
    anyhow::ensure!(!src_faces.is_empty(), "no face in source {}", src_path.display());
    let source_embedding = src_faces[0].vector.clone();
    println!(
        "source: embedded in {:.2?} (dim {}, norm {:.4})",
        t0.elapsed(),
        source_embedding.len(),
        source_embedding.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    // 2. Detect the target face.
    let t1 = Instant::now();
    let detections = engine.detect_faces(tgt_path)?;
    anyhow::ensure!(!detections.is_empty(), "no face in target {}", tgt_path.display());
    let det = &detections[0];
    println!(
        "target: {} face(s), best score {:.3}, bbox [{:.0},{:.0},{:.0},{:.0}] in {:.2?}",
        detections.len(),
        det.score,
        det.bbox[0],
        det.bbox[1],
        det.bbox[2],
        det.bbox[3],
        t1.elapsed()
    );

    // 3. Swap.
    let mut swap = SwapEngine::load(std::path::Path::new("models/faceswap"))?;
    println!("gpen available: {}", swap.gpen_available());
    let target = avatar_media_mcp::infra::image_io::open_limited(tgt_path)
        .with_context(|| format!("open {}", tgt_path.display()))?
        .to_rgb8();
    let t2 = Instant::now();
    let result = swap.swap_face(&target, det.bbox, det.kps, &source_embedding)?;
    println!("swap done in {:.2?}", t2.elapsed());

    result.save(out_path)?;
    println!("saved: {}", out_path.display());
    Ok(())
}

use anyhow::Context;
