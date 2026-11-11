//! Local ArcFace inference: SCRFD face detection + alignment + w600k_r50
//! embeddings via ONNX Runtime (`ort`), fully local, no network at runtime.
//!
//! Pipeline (standard InsightFace recipe):
//! 1. SCRFD (`det_10g.onnx`) detects the face + 5 landmarks on a 640px
//!    letterboxed copy.
//! 2. A least-squares similarity transform maps the 5 landmarks onto the
//!    canonical ArcFace 112×112 template.
//! 3. ArcFace (`w600k_r50.onnx`) embeds the aligned crop; the vector is
//!    L2-normalized so cosine similarity is a plain dot product.
//!
//! Model files live in `models/insightface/buffalo_l/` (downloaded once,
//! gitignored). The ONNX Runtime shared library is resolved via
//! `ORT_DYLIB_PATH` (feature `load-dynamic`).

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use ndarray::Array4;

pub const EMBEDDING_DIM: usize = 512;
/// Canonical ArcFace landmark template on the 112×112 aligned crop.
const TEMPLATE: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [45.0316, 92.3655],
    [69.4576, 92.2041],
];
const DET_THRESHOLD: f32 = 0.5;

pub struct FaceEngine {
    env: Arc<ort::environment::Environment>,
    det: ort::session::Session,
    rec: ort::session::Session,
}

impl FaceEngine {
    /// Load both models from a directory containing det_10g.onnx and
    /// w600k_r50.onnx.
    pub fn load(models_dir: &std::path::Path) -> Result<Self> {
        let det_path = models_dir.join("det_10g.onnx");
        let rec_path = models_dir.join("w600k_r50.onnx");
        for p in [&det_path, &rec_path] {
            if !p.exists() {
                bail!("model missing: {} — download buffalo_l first", p.display());
            }
        }
        let env = Arc::new(
            ort::init()
                .with_name("avatar-media-identity")
                // CoreML EP mis-shapes SCRFD's dynamic outputs; CPU is fast
                // enough for one-shot embeddings (~50 ms/face).
                .build()?,
        );
        let det = ort::session::Session::builder(&env)
            .map_err(|e| anyhow::anyhow!("ort env: {e}"))?;
        // CoreML EP mis-shapes SCRFD's dynamic outputs; CPU is fast
        // enough for one-shot embeddings (~50 ms/face).
        let det = det
            .with_auto_device(ort::session::builder::AutoDevicePolicy::PreferCPU)
            .map_err(|e| anyhow::anyhow!("ort det ep: {e}"))?
            .commit_from_file(&det_path)
            .map_err(|e| anyhow::anyhow!("load {}: {e}", det_path.display()))?;
        let rec = ort::session::Session::builder(&env)
            .map_err(|e| anyhow::anyhow!("ort env: {e}"))?
            .with_auto_device(ort::session::builder::AutoDevicePolicy::PreferCPU)
            .map_err(|e| anyhow::anyhow!("ort rec ep: {e}"))?
            .commit_from_file(&rec_path)
            .map_err(|e| anyhow::anyhow!("load {}: {e}", rec_path.display()))?;
        Ok(Self { env, det, rec })
    }

    /// Embed every face found in the image, largest (main person) first.
    pub fn embed_image(&mut self, image_path: &std::path::Path) -> Result<Vec<FaceEmbedding>> {
        let img = image::open(image_path)
            .with_context(|| format!("open {}", image_path.display()))?
            .to_rgb8();
        let (w, h) = (img.width() as f32, img.height() as f32);

        // Letterbox the long side to 640 for detection.
        let scale = 640.0 / w.max(h);
        let rw = ((w * scale) as u32 / 2 * 2).max(2);
        let rh = ((h * scale) as u32 / 2 * 2).max(2);
        let small = image::imageops::resize(&img, rw, rh, image::imageops::FilterType::Triangle);

        let faces = self.detect(&small, scale)?;
        let mut out = Vec::with_capacity(faces.len());
        for kps in faces {
            let aligned = align_affine(&img, &kps)?;
            let vector = self.embed_aligned(&aligned)?;
            out.push(FaceEmbedding { vector, landmarks: kps });
        }
        // Largest inter-ocular distance first (main person heuristic).
        out.sort_by(|a, b| {
            face_size(&b.landmarks)
                .total_cmp(&face_size(&a.landmarks))
        });
        Ok(out)
    }

    /// SCRFD forward pass. Returns landmark sets (5×2) in ORIGINAL image
    /// coordinates. det_10g emits 3 stride groups × (scores, boxes, kps).
    fn detect(&mut self, small: &image::RgbImage, scale: f32) -> Result<Vec<[[f32; 2]; 5]>> {
        let (rw, rh) = (small.width() as usize, small.height() as usize);
        let mut blob = Array4::<f32>::zeros((1, 3, rh, rw));
        for (x, y, px) in small.enumerate_pixels() {
            let (x, y) = (x as usize, y as usize);
            blob[[0, 0, y, x]] = px[0] as f32;
            blob[[0, 1, y, x]] = px[1] as f32;
            blob[[0, 2, y, x]] = px[2] as f32;
        }
        blob.mapv_inplace(|v| (v - 127.5) / 127.5);

        let input_name = self
            .det
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input.1".into());
        let outputs = self
            .det
            .run(ort::inputs![ort::value::TensorRef::from_array_view(&blob)?])
            .context("SCRFD forward")?;

        let mut best: Option<(f32, [[f32; 2]; 5])> = None;
        // 9 outputs: for stride group g in 0..3 → [scores, boxes, kps].
        for g in 0..3 {
            let scores = outputs[g * 3]
                .try_extract_array::<f32>()?
                .view()
                .iter()
                .copied()
                .collect::<Vec<f32>>();
            let kps_flat = outputs[g * 3 + 2]
                .try_extract_array::<f32>()?
                .view()
                .iter()
                .copied()
                .collect::<Vec<f32>>();
            let n = scores.len();
            for i in 0..n {
                let s = scores[i];
                if s < DET_THRESHOLD {
                    continue;
                }
                let base = i * 10;
                if base + 10 > kps_flat.len() {
                    continue;
                }
                let mut kps = [[0f32; 2]; 5];
                for (j, kp) in kps.iter_mut().enumerate() {
                    kp[0] = kps_flat[base + j * 2] / scale;
                    kp[1] = kps_flat[base + j * 2 + 1] / scale;
                }
                if best.map(|(bs, _)| s > bs).unwrap_or(true) {
                    best = Some((s, kps));
                }
            }
        }
        Ok(best.map(|(_, kps)| vec![kps]).unwrap_or_default())
    }

    /// ArcFace forward on the aligned 112×112 crop; L2-normalized 512-d.
    fn embed_aligned(&mut self, aligned: &image::RgbImage) -> Result<Vec<f32>> {
        let mut blob = Array4::<f32>::zeros((1, 3, 112, 112));
        for (x, y, px) in aligned.enumerate_pixels() {
            let (x, y) = (x as usize, y as usize);
            blob[[0, 0, y, x]] = px[0] as f32;
            blob[[0, 1, y, x]] = px[1] as f32;
            blob[[0, 2, y, x]] = px[2] as f32;
        }
        blob.mapv_inplace(|v| (v - 127.5) / 127.5);
        let input_name = self
            .rec
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input.1".into());
        let outputs = self
            .rec
            .run(ort::inputs![ort::value::TensorRef::from_array_view(&blob)?])
            .context("ArcFace forward")?;
        let v: Vec<f32> = outputs[0]
            .try_extract_array::<f32>()?
            .view()
            .iter()
            .copied()
            .collect();
        anyhow::ensure!(v.len() == EMBEDDING_DIM, "unexpected embedding dim {}", v.len());
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        anyhow::ensure!(norm > 1e-6, "zero embedding");
        Ok(v.into_iter().map(|x| x / norm).collect())
    }
}

#[derive(Debug, Clone)]
pub struct FaceEmbedding {
    pub vector: Vec<f32>,
    pub landmarks: [[f32; 2]; 5],
}

/// Face size proxy: inter-ocular distance.
fn face_size(kps: &[[f32; 2]; 5]) -> f32 {
    let dx = kps[1][0] - kps[0][0];
    let dy = kps[1][1] - kps[0][1];
    (dx * dx + dy * dy).sqrt()
}

/// Similarity transform: map detected 5 landmarks onto the canonical
/// 112×112 ArcFace template (least-squares affine, no OpenCV). The result
/// is the INVERSE mapping (output px → source px), which is what
/// `image::imageops::transform` samples with.
fn align_affine(img: &image::RgbImage, kps: &[[f32; 2]; 5]) -> Result<image::RgbImage> {
    // Solve M (2×3) minimizing ||M·detected − template||² over 5 points.
    let mut ata = [[0f64; 6]; 6];
    let mut atb = [0f64; 6];
    for ([x, y], [sx, sy]) in kps.iter().zip(TEMPLATE.iter()) {
        let row = [*x as f64, *y as f64, 1.0, 0.0, 0.0, 0.0];
        for r in 0..6 {
            for c in 0..6 {
                ata[r][c] += row[r] * row[c];
            }
            atb[r] += row[r] * (*sx as f64);
        }
        let row2 = [0.0, 0.0, 0.0, *x as f64, *y as f64, 1.0];
        for r in 0..6 {
            for c in 0..6 {
                ata[r][c] += row2[r] * row2[c];
            }
            atb[r] += row2[r] * (*sy as f64);
        }
    }
    let m = solve6(&ata, &atb)?;
    let inv = invert_affine(&m)?;
    // image::imageops::transform samples with the given 3×2 matrix
    // [a c; b d; tx ty] column-major as (a, b, c, d, tx, ty) mapping
    // output → input. Our inv is [ia ib itx ic id ity] (row-major 2×3).
    Ok(align_affine_sample(img, kps))
}

/// Affine warp via manual bilinear sampling: for each output pixel (ox, oy)
/// of the 112×112 crop, sample the source at inv·(ox, oy).
fn align_affine_sample(img: &image::RgbImage, kps: &[[f32; 2]; 5]) -> image::RgbImage {
    // Solve M (2×3) minimizing ||M·detected − template||² over 5 points.
    let mut ata = [[0f64; 6]; 6];
    let mut atb = [0f64; 6];
    for ([x, y], [sx, sy]) in kps.iter().zip(TEMPLATE.iter()) {
        let row = [*x as f64, *y as f64, 1.0, 0.0, 0.0, 0.0];
        for r in 0..6 {
            for c in 0..6 {
                ata[r][c] += row[r] * row[c];
            }
            atb[r] += row[r] * (*sx as f64);
        }
        let row2 = [0.0, 0.0, 0.0, *x as f64, *y as f64, 1.0];
        for r in 0..6 {
            for c in 0..6 {
                ata[r][c] += row2[r] * row2[c];
            }
            atb[r] += row2[r] * (*sy as f64);
        }
    }
    let m = match solve6(&ata, &atb) {
        Ok(m) => m,
        Err(_) => return image::RgbImage::new(112, 112),
    };
    let inv = match invert_affine(&m) {
        Ok(v) => v,
        Err(_) => return image::RgbImage::new(112, 112),
    };
    let (sw, sh) = (img.width() as i64, img.height() as i64);
    let mut out = image::RgbImage::new(112, 112);
    for oy in 0..112u32 {
        for ox in 0..112u32 {
            let sx = inv[0] * ox as f64 + inv[1] * oy as f64 + inv[2];
            let sy = inv[3] * ox as f64 + inv[4] * oy as f64 + inv[5];
            if sx < 0.0 || sy < 0.0 || sx >= (sw - 1) as f64 || sy >= (sh - 1) as f64 {
                continue; // stays black (0) — standard for out-of-bounds
            }
            let x0 = sx as u32;
            let y0 = sy as u32;
            let fx = (sx - x0 as f64) as f32;
            let fy = (sy - y0 as f64) as f32;
            let p00 = img.get_pixel(x0, y0);
            let p10 = img.get_pixel(x0 + 1, y0);
            let p01 = img.get_pixel(x0, y0 + 1);
            let p11 = img.get_pixel(x0 + 1, y0 + 1);
            let mut px = [0u8; 3];
            for c in 0..3 {
                let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
                let bot = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
                px[c] = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(ox, oy, image::Rgb(px));
        }
    }
    out
}

/// Solve a 6×6 linear system by Gaussian elimination with partial pivoting.
fn solve6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Result<[f64; 6]> {
    let mut m = *a;
    let mut rhs = *b;
    for col in 0..6 {
        let pivot = (col..6usize)
            .max_by(|r1, r2| m[*r1][col].abs().total_cmp(&m[*r2][col].abs()))
            .context("singular system")?;
        if m[pivot][col].abs() < 1e-9 {
            bail!("singular landmark system");
        }
        m.swap(col, pivot);
        rhs.swap(col, pivot);
        let d = m[col][col];
        for r in 0..6 {
            if r != col {
                let f = m[r][col] / d;
                for c in col..6 {
                    m[r][c] -= f * m[col][c];
                }
                rhs[r] -= f * rhs[col];
            }
        }
    }
    let mut out = [0.0; 6];
    for (i, o) in out.iter_mut().enumerate() {
        *o = rhs[i] / m[i][i];
    }
    Ok(out)
}

/// Invert a 2×3 affine [a b tx; c d ty] (row-major flat [a,b,tx,c,d,ty]).
fn invert_affine(m: &[f64; 6]) -> Result<[f64; 6]> {
    let [a, b, tx, c, d, ty] = *m;
    let det = a * d - b * c;
    if det.abs() < 1e-12 {
        bail!("degenerate affine");
    }
    let ia = d / det;
    let ib = -b / det;
    let ic = -c / det;
    let id = a / det;
    let itx = -(ia * tx + ib * ty);
    let ity = -(ic * tx + id * ty);
    Ok([ia, ib, itx, ic, id, ity])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_roundtrip() {
        // A pure translation+scale must invert to itself applied twice.
        let m = [2.0f64, 0.0, 5.0, 0.0, 2.0, 7.0];
        let inv = invert_affine(&m).unwrap();
        // Forward: p' = M·p. Inverse of (1,1): M·(1,1) = (7,9); inv·(7,9) = (1,1).
        let px = inv[0] * 7.0 + inv[1] * 9.0 + inv[2];
        let py = inv[3] * 7.0 + inv[4] * 9.0 + inv[5];
        assert!((px - 1.0).abs() < 1e-9 && (py - 1.0).abs() < 1e-9);
    }

    #[test]
    fn solve6_identity() {
        let a = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(solve6(&a, &b).unwrap(), b);
    }
}
