//! One-shot identity test: embed an image, print the vector norm and
//! self-similarity. Usage: `cargo run --features local-identity --bin
//! identity_test -- <image> [<image2>]`
//!
//! Requires ORT_DYLIB_PATH pointing at libonnxruntime.dylib
//! (models/ort/lib/onnxruntime.dylib in this repo).

use avatar_media_mcp::identity::{arcface::FaceEngine, embeddings};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 2, "usage: identity_test <image> [<image2>]");
    let models_dir = std::path::Path::new("models/insightface/buffalo_l");
    let mut engine = FaceEngine::load(models_dir)?;

    let a = &engine.embed_image(std::path::Path::new(&args[1]))?;
    anyhow::ensure!(!a.is_empty(), "no face detected in {}", args[1]);
    println!(
        "image1: {} face(s), main vector dim={}, norm={:.4}",
        a.len(),
        a[0].vector.len(),
        a[0].vector.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    if args.len() > 2 {
        let b = &engine.embed_image(std::path::Path::new(&args[2]))?;
        anyhow::ensure!(!b.is_empty(), "no face detected in {}", args[2]);
        let sim = embeddings::cosine_similarity(&a[0].vector, &b[0].vector);
        println!("image2: {} face(s)", b.len());
        println!("cosine similarity (main faces): {sim:.4}");
        let t = avatar_media_mcp::identity::scoring::IdentityThresholds::default();
        match avatar_media_mcp::identity::scoring::stage1(sim, &t) {
            avatar_media_mcp::identity::scoring::Stage1Verdict::Pass(_) => println!("verdict: PASS (same person)"),
            avatar_media_mcp::identity::scoring::Stage1Verdict::Borderline(_) => println!("verdict: BORDERLINE (send to judge)"),
            avatar_media_mcp::identity::scoring::Stage1Verdict::Fail(_) => println!("verdict: FAIL (different person)"),
        }
    }
    Ok(())
}
