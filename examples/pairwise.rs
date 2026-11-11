//! Pairwise cosine distribution over an identity model.
//!
//! Usage: cargo run --example pairwise -- avatars/<slug>/identity_model.json

use avatar_media_mcp::identity::embeddings::cosine_similarity_checked;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: pairwise <identity_model.json>"))?;
    let raw = std::fs::read_to_string(&path)?;
    let model: serde_json::Value = serde_json::from_str(&raw)?;
    let embeddings = model["face_embeddings"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{path}: no face_embeddings object"))?;

    let vectors: Vec<Vec<f32>> = embeddings
        .values()
        .map(|value| {
            value
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("embedding is not an array"))?
                .iter()
                .map(|component| {
                    component
                        .as_f64()
                        .map(|v| v as f32)
                        .ok_or_else(|| anyhow::anyhow!("embedding component is not a number"))
                })
                .collect::<anyhow::Result<Vec<f32>>>()
        })
        .collect::<anyhow::Result<Vec<Vec<f32>>>>()?;

    let mut scores = Vec::new();
    for left in 0..vectors.len() {
        for right in (left + 1)..vectors.len() {
            scores.push(cosine_similarity_checked(&vectors[left], &vectors[right])?.get());
        }
    }
    if scores.is_empty() {
        anyhow::bail!("{path}: need at least two embeddings, got {}", vectors.len());
    }
    scores.sort_by(f32::total_cmp);
    let n = scores.len();
    println!(
        "pairs: {n}\nmin: {:.3}\np25: {:.3}\nmedian: {:.3}\np75: {:.3}\nmax: {:.3}",
        scores[0],
        scores[n / 4],
        scores[n / 2],
        scores[3 * n / 4],
        scores[n - 1]
    );
    Ok(())
}
