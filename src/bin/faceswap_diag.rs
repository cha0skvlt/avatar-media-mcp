//! Diagnostic: dump inswapper output for the aligned crop.
use std::path::Path;
use avatar_media_mcp::identity::arcface::FaceEngine;
use avatar_media_mcp::pipeline::faceswap::{SwapEngine, umeyama_similarity, warp_sample, SWAP_TEMPLATE};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 3, "usage: faceswap_diag <src> <tgt>");
    let mut engine = FaceEngine::load(Path::new("models/insightface/buffalo_l"))?;
    let src_faces = engine.embed_image(Path::new(&args[1]))?;
    let emb = src_faces[0].vector.clone();
    let dets = engine.detect_faces(Path::new(&args[2]))?;
    let det = &dets[0];
    let target = avatar_media_mcp::infra::image_io::open_rgb8(Path::new(&args[2]))?;
    let kps: Vec<[f32;2]> = det.kps.to_vec();
    let m = umeyama_similarity(&kps, &SWAP_TEMPLATE)?;
    let inv = m.invert()?;
    let aligned = warp_sample(&target, 128, 128, &inv);
    aligned.save("/tmp/diag_aligned.png")?;

    let mut swap = SwapEngine::load(Path::new("models/faceswap"))?;
    let out = swap.inswapper_forward(&aligned, &emb)?;
    out.save("/tmp/diag_swapped.png")?;
    let mut min=255f32; let mut max=0f32; let mut sum=0f32;
    for (_,_,p) in out.enumerate_pixels() { for c in 0..3 { let v=p[c] as f32; min=min.min(v); max=max.max(v); sum+=v; }}
    println!("swapped min {} max {} mean {:.1}", min, max, sum/(128*128*3) as f32);
    Ok(())
}
