//! Fast face swap (milestone FS1).
//!
//! Port of the magic-red-bikini `swapper.py` methodology to pure Rust `ort`,
//! no Python, no OpenCV:
//!
//! 1. SCRFD detection + 5 landmarks (shared `FaceEngine`, `arcface.rs`).
//! 2. `2d106det` dense 106-landmark fit on the detected face.
//! 3. `inswapper_128`: aligned 128×128 target crop + source ArcFace embedding
//!    projected through the model's `emap` (buff2fs) matrix → swapped crop.
//! 4. Reinhard per-channel color transfer onto the aligned target crop.
//! 5. Optional GPEN bfr-512 restoration (blend 0.8).
//! 6. Landmark-driven paste-back: face-contour mask (outline 0–32 + brows +
//!    forehead arc, jaw pull, differential erode, feather), open-mouth keep
//!    hole (inner lips stay original), bilinear composite through the
//!    frame→crop sampler.
//!
//! All weights live under `models/faceswap/` (symlinks into the mrb data
//! tree are fine; nothing is committed). `inswapper_emap.f32` is the raw
//! 512×512 f32 `buff2fs` initializer extracted from inswapper_128.onnx.

use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::Array4;

use crate::identity::arcface::shared_ort_environment;

const LM_SIZE: u32 = 192;
const SWAP_SIZE: usize = 128;
const GPEN_SIZE: usize = 512;
const GPEN_BLEND: f32 = 0.8;
/// Inner-lip gap / mouth width above which the mouth hole opens.
const MOUTH_MAR_THRESHOLD: f32 = 0.35;

/// inswapper alignment template (InsightFace `face_align.arcface_dst`),
/// 128×128 variant: 112-template scaled by 1 with +8 x-offset.
pub const SWAP_TEMPLATE: [[f32; 2]; 5] = [
    [38.2946 + 8.0, 51.6963],
    [73.5318 + 8.0, 51.5014],
    [56.0252 + 8.0, 71.7366],
    [41.5493 + 8.0, 92.3655],
    [70.7299 + 8.0, 92.2041],
];

/// Row-major 2×3 affine: x' = a·x + b·y + tx, y' = c·x + d·y + ty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine2x3 {
    pub a: f64,
    pub b: f64,
    pub tx: f64,
    pub c: f64,
    pub d: f64,
    pub ty: f64,
}

impl Affine2x3 {
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.b * y + self.tx,
            self.c * x + self.d * y + self.ty,
        )
    }

    pub fn invert(&self) -> Result<Affine2x3> {
        anyhow::ensure!(
            [self.a, self.b, self.tx, self.c, self.d, self.ty]
                .iter()
                .all(|v| v.is_finite()),
            "affine contains non-finite values"
        );
        let det = self.a * self.d - self.b * self.c;
        if !det.is_finite() || det.abs() < 1e-12 {
            bail!("degenerate affine");
        }
        let ia = self.d / det;
        let ib = -self.b / det;
        let ic = -self.c / det;
        let id = self.a / det;
        Ok(Affine2x3 {
            a: ia,
            b: ib,
            tx: -(ia * self.tx + ib * self.ty),
            c: ic,
            d: id,
            ty: -(ic * self.tx + id * self.ty),
        })
    }

    /// Scale the whole matrix (mrb `_scale_affine`: `out *= scale` — ALL six
    /// elements, translation included; a 512-crop pixel is 4× a 128-crop
    /// pixel, so the forward map scales uniformly).
    fn scaled(&self, s: f64) -> Affine2x3 {
        Affine2x3 {
            a: self.a * s,
            b: self.b * s,
            tx: self.tx * s,
            c: self.c * s,
            d: self.d * s,
            ty: self.ty * s,
        }
    }
}

/// Umeyama least-squares similarity transform src → dst (skimage
/// `SimilarityTransform.estimate` convention, scaling allowed, no reflection).
pub fn umeyama_similarity(src: &[[f32; 2]], dst: &[[f32; 2]]) -> Result<Affine2x3> {
    anyhow::ensure!(
        src.len() == dst.len() && src.len() >= 2,
        "umeyama needs ≥2 point pairs"
    );
    anyhow::ensure!(
        src.iter().chain(dst).flatten().all(|v| v.is_finite()),
        "umeyama point sets contain non-finite coordinates"
    );
    let n = src.len() as f64;
    let mu_src = (
        src.iter().map(|p| p[0] as f64).sum::<f64>() / n,
        src.iter().map(|p| p[1] as f64).sum::<f64>() / n,
    );
    let mu_dst = (
        dst.iter().map(|p| p[0] as f64).sum::<f64>() / n,
        dst.iter().map(|p| p[1] as f64).sum::<f64>() / n,
    );
    let mut sigma_src = 0.0f64;
    let mut h = [[0.0f64; 2]; 2]; // cov = Σ dst' ⊗ src'
    for (s, d) in src.iter().zip(dst) {
        let sx = s[0] as f64 - mu_src.0;
        let sy = s[1] as f64 - mu_src.1;
        let dx = d[0] as f64 - mu_dst.0;
        let dy = d[1] as f64 - mu_dst.1;
        sigma_src += sx * sx + sy * sy;
        h[0][0] += dx * sx;
        h[0][1] += dx * sy;
        h[1][0] += dy * sx;
        h[1][1] += dy * sy;
    }
    sigma_src /= n;
    for r in &mut h {
        for v in r {
            *v /= n;
        }
    }
    anyhow::ensure!(sigma_src > 1e-12, "degenerate source point set");

    // 2×2 SVD via eigen-decomposition of HᵀH (symmetric): V = eigvecs, σ = √λ.
    let mtm = [
        [h[0][0] * h[0][0] + h[1][0] * h[1][0], h[0][0] * h[0][1] + h[1][0] * h[1][1]],
        [h[0][0] * h[0][1] + h[1][0] * h[1][1], h[0][1] * h[0][1] + h[1][1] * h[1][1]],
    ];
    let trace = mtm[0][0] + mtm[1][1];
    let det = mtm[0][0] * mtm[1][1] - mtm[0][1] * mtm[0][1];
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let (l1, l2) = ((trace + disc) / 2.0, (trace - disc) / 2.0);
    let sv1 = l1.max(0.0).sqrt();
    let sv2 = l2.max(0.0).sqrt();

    // Eigenvector for an eigenvalue λ of [[p,q],[q,r]].
    let eigvec = |p: f64, q: f64, r: f64, lam: f64| -> (f64, f64) {
        // (p-λ)x + qy = 0 → v = (q, λ-p) or (λ-r, q), pick the larger norm.
        let v1 = (q, lam - p);
        let v2 = (lam - r, q);
        let n1 = v1.0 * v1.0 + v1.1 * v1.1;
        let n2 = v2.0 * v2.0 + v2.1 * v2.1;
        let v = if n1 >= n2 { v1 } else { v2 };
        let nv = (v.0 * v.0 + v.1 * v.1).sqrt();
        if nv < 1e-12 {
            (1.0, 0.0)
        } else {
            (v.0 / nv, v.1 / nv)
        }
    };
    let v1 = eigvec(mtm[0][0], mtm[0][1], mtm[1][1], l1);
    // Construct the second eigenvector as the exact perpendicular. For an
    // isotropic covariance (equal eigenvalues), independently solving both
    // eigenvectors is underdetermined and can return the same vector twice.
    let v2 = (-v1.1, v1.0);

    // U columns: u_i = H·v_i / σ_i (fallback to an orthogonal complement
    // when σ_i ≈ 0).
    let hv = |v: (f64, f64)| (h[0][0] * v.0 + h[0][1] * v.1, h[1][0] * v.0 + h[1][1] * v.1);
    let mut u1 = hv(v1);
    let n1 = (u1.0 * u1.0 + u1.1 * u1.1).sqrt();
    if sv1 > 1e-12 && n1 > 1e-12 {
        u1 = (u1.0 / n1, u1.1 / n1);
    } else {
        u1 = (1.0, 0.0);
    }
    let mut u2 = hv(v2);
    let n2 = (u2.0 * u2.0 + u2.1 * u2.1).sqrt();
    if sv2 > 1e-12 && n2 > 1e-12 {
        u2 = (u2.0 / n2, u2.1 / n2);
    } else {
        // Orthogonal complement of u1.
        u2 = (-u1.1, u1.0);
    }

    // R = U·S·Vᵀ, with S correcting a possible reflection.
    // det(U) where U = [u1 u2] columns: u1.x*u2.y - u1.y*u2.x
    let det_u = u1.0 * u2.1 - u1.1 * u2.0;
    let det_v = v1.0 * v2.1 - v1.1 * v2.0;
    let s2 = if det_u * det_v < 0.0 { -1.0 } else { 1.0 };

    // R = Σ_i σ-weighted? No: R = U·diag(1,s2)·Vᵀ (pure rotation, scale separate).
    // R = u1·v1ᵀ + s2·u2·v2ᵀ
    let r00 = u1.0 * v1.0 + s2 * u2.0 * v2.0;
    let r01 = u1.0 * v1.1 + s2 * u2.0 * v2.1;
    let r10 = u1.1 * v1.0 + s2 * u2.1 * v2.0;
    let r11 = u1.1 * v1.1 + s2 * u2.1 * v2.1;
    // c = trace(Σ·S)/σ_src² where Σ = diag(σ1, σ2)
    let c = (sv1 + s2 * sv2) / sigma_src;
    anyhow::ensure!(c > 1e-12, "degenerate similarity scale");
    let tx = mu_dst.0 - c * (r00 * mu_src.0 + r01 * mu_src.1);
    let ty = mu_dst.1 - c * (r10 * mu_src.0 + r11 * mu_src.1);
    Ok(Affine2x3 {
        a: c * r00,
        b: c * r01,
        tx,
        c: c * r10,
        d: c * r11,
        ty,
    })
}

/// Bilinear warp: dst(x, y) = src(mat·(x, y)), out-of-bounds stays black.
/// This matches cv2.warpAffine's default semantics (mat is the forward
/// src→dst map; sampling uses its inverse internally — here we pass the
/// dst→src sampling matrix directly).
pub fn warp_sample(src: &image::RgbImage, w: u32, h: u32, mat: &Affine2x3) -> image::RgbImage {
    let (sw, sh) = (src.width() as i64, src.height() as i64);
    let mut out = image::RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = mat.apply(x as f64, y as f64);
            if sx < 0.0 || sy < 0.0 || sx >= (sw - 1) as f64 || sy >= (sh - 1) as f64 {
                continue;
            }
            let x0 = sx as u32;
            let y0 = sy as u32;
            let fx = (sx - x0 as f64) as f32;
            let fy = (sy - y0 as f64) as f32;
            let p00 = src.get_pixel(x0, y0);
            let p10 = src.get_pixel(x0 + 1, y0);
            let p01 = src.get_pixel(x0, y0 + 1);
            let p11 = src.get_pixel(x0 + 1, y0 + 1);
            let mut px = [0u8; 3];
            for c in 0..3 {
                let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
                let bot = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
                px[c] = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(x, y, image::Rgb(px));
        }
    }
    out
}

/// Reinhard per-channel color transfer: match crop's mean/std to ref's.
pub fn reinhard_color_match(crop: &mut image::RgbImage, reference: &image::RgbImage) {
    let (cw, ch) = (crop.width() as usize, crop.height() as usize);
    let (rw, rh) = (reference.width() as usize, reference.height() as usize);
    for c in 0..3 {
        let mut cs = 0f64;
        let mut css = 0f64;
        let mut rs = 0f64;
        let mut rss = 0f64;
        for y in 0..ch {
            for x in 0..cw {
                let v = crop.get_pixel(x as u32, y as u32)[c] as f64;
                cs += v;
                css += v * v;
            }
        }
        for y in 0..rh {
            for x in 0..rw {
                let v = reference.get_pixel(x as u32, y as u32)[c] as f64;
                rs += v;
                rss += v * v;
            }
        }
        let cn = (cw * ch) as f64;
        let rn = (rw * rh) as f64;
        let c_mean = cs / cn;
        let c_std = (css / cn - c_mean * c_mean).sqrt().max(1e-6);
        let r_mean = rs / rn;
        let r_std = (rss / rn - r_mean * r_mean).sqrt().max(1e-6);
        let gain = r_std / c_std;
        for y in 0..ch {
            for x in 0..cw {
                let p = crop.get_pixel_mut(x as u32, y as u32);
                let v = (p[c] as f64 - c_mean) * gain + r_mean;
                p[c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

fn odd(mut n: usize, minimum: usize) -> usize {
    n = n.max(minimum);
    if n.is_multiple_of(2) {
        n += 1;
    }
    n
}

/// Separable min filter (square kernel) — erosion approximation on [0,1] masks.
fn erode_mask(mask: &mut [f32], w: usize, h: usize, k: usize) {
    if w == 0 || h == 0 || mask.len() != w.saturating_mul(h) || k < 3 {
        return;
    }
    let r = k / 2;
    let mut tmp = vec![0.0f32; w * h];
    // Horizontal pass.
    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            let mut m = 1.0f32;
            for xx in x0..=x1 {
                m = m.min(mask[y * w + xx]);
            }
            tmp[y * w + x] = m;
        }
    }
    // Vertical pass.
    for y in 0..h {
        for x in 0..w {
            let y0 = y.saturating_sub(r);
            let y1 = (y + r).min(h - 1);
            let mut m = 1.0f32;
            for yy in y0..=y1 {
                m = m.min(tmp[yy * w + x]);
            }
            mask[y * w + x] = m;
        }
    }
}

/// Separable box blur, one pass (Gaussian approximation when applied 3×).
fn box_blur_pass(mask: &mut [f32], w: usize, h: usize, k: usize) {
    if w == 0 || h == 0 || mask.len() != w.saturating_mul(h) || k < 3 {
        return;
    }
    let r = k / 2;
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            let n = (x1 - x0 + 1) as f32;
            let s: f32 = (x0..=x1).map(|xx| mask[y * w + xx]).sum();
            tmp[y * w + x] = s / n;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let y0 = y.saturating_sub(r);
            let y1 = (y + r).min(h - 1);
            let n = (y1 - y0 + 1) as f32;
            let s: f32 = (y0..=y1).map(|yy| tmp[yy * w + x]).sum();
            mask[y * w + x] = s / n;
        }
    }
}

fn gaussian_blur_approx(mask: &mut [f32], w: usize, h: usize, k: usize) {
    for _ in 0..3 {
        box_blur_pass(mask, w, h, k);
    }
}

/// Convex hull (Andrew monotone chain), returns hull in CCW order.
pub fn convex_hull(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: &(f32, f32), a: &(f32, f32), b: &(f32, f32)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut lower = Vec::new();
    for p in &pts {
        while lower.len() >= 2 && cross(&lower[lower.len() - 2], lower.last().unwrap(), p) <= 0.0 {
            lower.pop();
        }
        lower.push(*p);
    }
    let mut upper = Vec::new();
    for p in pts.iter().rev() {
        while upper.len() >= 2 && cross(&upper[upper.len() - 2], upper.last().unwrap(), p) <= 0.0 {
            upper.pop();
        }
        upper.push(*p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Even-odd scanline polygon fill into an [0,1] mask.
fn fill_polygon(mask: &mut [f32], w: usize, h: usize, poly: &[(f32, f32)]) {
    if w == 0 || h == 0 || mask.len() != w.saturating_mul(h) || poly.len() < 3
        || !poly.iter().all(|v| v.0.is_finite() && v.1.is_finite())
    {
        return;
    }
    let min_y_f = poly.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
    let max_y_f = poly.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
    if max_y_f < 0.0 || min_y_f > (h - 1) as f32 {
        return;
    }
    let min_y = min_y_f.floor().clamp(0.0, (h - 1) as f32) as usize;
    let max_y = max_y_f.ceil().clamp(0.0, (h - 1) as f32) as usize;
    for y in min_y..=max_y {
        let yc = y as f32 + 0.5;
        let mut xs: Vec<f32> = Vec::new();
        for i in 0..poly.len() {
            let (x1, y1) = poly[i];
            let (x2, y2) = poly[(i + 1) % poly.len()];
            if (y1 <= yc && y2 > yc) || (y2 <= yc && y1 > yc) {
                let t = (yc - y1) / (y2 - y1);
                xs.push(x1 + t * (x2 - x1));
            }
        }
        xs.sort_by(|a, b| a.total_cmp(b));
        for pair in xs.chunks_exact(2) {
            let left = pair[0].min(pair[1]);
            let right = pair[0].max(pair[1]);
            if right < 0.0 || left > (w - 1) as f32 {
                continue;
            }
            let x0 = left.ceil().clamp(0.0, (w - 1) as f32) as usize;
            let x1 = right.floor().clamp(0.0, (w - 1) as f32) as usize;
            if x0 <= x1 {
                for x in x0..=x1 {
                    mask[y * w + x] = 1.0;
                }
            }
        }
    }
}

/// Mouth aspect ratio: inner-lip gap / mouth width (indices 52–63 outer,
/// 64–71 inner). Outer bbox fires on every closed mouth; this does not.
pub fn mouth_aspect_ratio(lmk: &[[f32; 2]]) -> f32 {
    if lmk.len() < 72 {
        return 0.0;
    }
    let outer = &lmk[52..64];
    let inner = &lmk[64..72];
    let left = outer.iter().min_by(|a, b| a[0].total_cmp(&b[0])).unwrap();
    let right = outer.iter().max_by(|a, b| a[0].total_cmp(&b[0])).unwrap();
    let width = ((right[0] - left[0]).powi(2) + (right[1] - left[1]).powi(2)).sqrt();
    if width < 1.0 {
        return 0.0;
    }
    // Gap = vertical extent of the inner-lip ring (robust for any pose).
    let gap = inner.iter().map(|p| p[1]).fold(f32::MIN, f32::max)
        - inner.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
    (gap.max(0.0) / width).min(1.0)
}

/// Face-contour paste mask (port of mrb `_face_contour_mask`): hull of the
/// outline (0–32) scaled 0.97 toward the centroid, quadratic jaw pull,
/// brow chords pushed out, forehead arc up to hair_y; differential erode
/// (jaw eroded, forehead keeps the extended hull) and feather.
pub fn face_contour_mask(w: usize, h: usize, lmk: &[[f32; 2]]) -> Vec<f32> {
    let mut mask = vec![0.0f32; w.saturating_mul(h)];
    if w == 0 || h == 0 {
        return mask;
    }
    // Unusable landmarks return an empty mask, not a fallback oval: this
    // function works in full-frame coordinates, while `ellipse_fallback_mask`
    // is anchored in 128-crop space and would paste a tiny oval at the top
    // centre of the frame. The caller substitutes a bbox-anchored ellipse.
    if lmk.len() < 33 || !lmk.iter().flatten().all(|v| v.is_finite()) {
        return mask;
    }
    let contour = &lmk[..33];
    let cx = contour.iter().map(|p| p[0]).sum::<f32>() / contour.len() as f32;
    let cy = contour.iter().map(|p| p[1]).sum::<f32>() / contour.len() as f32;
    let face_h = contour.iter().map(|p| p[1]).fold(f32::MIN, f32::max)
        - contour.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
    if !face_h.is_finite() || face_h < 1.0 {
        return mask;
    }
    let mut pts: Vec<(f32, f32)> = contour
        .iter()
        .map(|p| (cx + (p[0] - cx) * 0.97, cy + (p[1] - cy) * 0.97))
        .collect();
    let y0 = pts.iter().map(|p| p.1).fold(f32::MAX, f32::min);
    for p in pts.iter_mut() {
        let jaw_t = ((p.1 - y0) / face_h.max(1.0)).clamp(0.0, 1.0);
        p.1 -= jaw_t * jaw_t * face_h * 0.06;
    }
    if lmk.len() >= 106 {
        let mut brows: Vec<[f32; 2]> = lmk[43..52].to_vec();
        brows.extend_from_slice(&lmk[97..106]);
        let brow_top = brows.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
        let mid_x = cx;
        let mut brows_out: Vec<(f32, f32)> = brows
            .iter()
            .map(|p| {
                (
                    p[0] + (p[0] - mid_x).signum() * face_h * 0.03,
                    p[1] - face_h * 0.10,
                )
            })
            .collect();
        let bx0 = brows.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
        let bx1 = brows.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
        let hair_y = brow_top - face_h * 0.22;
        for i in 0..7usize {
            let t = i as f32 / 6.0;
            let arch = (t * std::f32::consts::PI).sin();
            let x = bx0 + t * (bx1 - bx0);
            let y = hair_y + (1.0 - arch) * (brow_top - hair_y) * 0.25;
            brows_out.push((x, y));
        }
        pts.extend(brows_out);
    }
    let hull = convex_hull(&pts);
    fill_polygon(&mut mask, w, h, &hull);
    let erode = odd((face_h * 0.035) as usize, 3);
    erode_mask(&mut mask, w, h, erode);
    // Differential: rows above brow_top keep the full hull, below blends to
    // the eroded mask over 20% of face height.
    let brow_top = if lmk.len() >= 106 {
        lmk[43..52]
            .iter()
            .chain(lmk[97..106].iter())
            .map(|p| p[1])
            .fold(f32::MAX, f32::min)
    } else {
        y0
    };
    let full = mask.clone();
    // Re-fill the full (non-eroded) hull for the forehead region blend.
    let mut full_mask = vec![0.0f32; w * h];
    fill_polygon(&mut full_mask, w, h, &convex_hull(&pts));
    for y in 0..h {
        let mix = (((y as f32) - brow_top) / (face_h * 0.20).max(1.0)).clamp(0.0, 1.0);
        for x in 0..w {
            let i = y * w + x;
            mask[i] = full_mask[i] * (1.0 - mix) + full[i] * mix;
        }
    }
    let blur = odd((face_h * 0.07) as usize, 9);
    gaussian_blur_approx(&mut mask, w, h, blur);
    mask.iter().map(|v| v.clamp(0.0, 1.0)).collect()
}

/// Tight outer-lip hole (indices 52–63) only when the mouth is clearly open.
pub fn mouth_keep_mask(w: usize, h: usize, lmk: &[[f32; 2]]) -> Vec<f32> {
    let mut mask = vec![0.0f32; w * h];
    if lmk.len() < 72 {
        return mask;
    }
    if mouth_aspect_ratio(lmk) < MOUTH_MAR_THRESHOLD {
        return mask;
    }
    let pts: Vec<(f32, f32)> = lmk[52..64].iter().map(|p| (p[0], p[1])).collect();
    let mouth_w = pts.iter().map(|p| p.0).fold(f32::MIN, f32::max)
        - pts.iter().map(|p| p.0).fold(f32::MAX, f32::min);
    // Degenerate (collinear) rings: expand into a thin closed band so the
    // hull has area; real 106-point rings are never collinear.
    let ys: Vec<f32> = pts.iter().map(|p| p.1).collect();
    let flat = ys.iter().all(|y| (y - ys[0]).abs() < 1e-3);
    let poly = if flat {
        let half = (mouth_w * 0.04).max(2.0);
        let x0 = pts.iter().map(|p| p.0).fold(f32::MAX, f32::min);
        let x1 = pts.iter().map(|p| p.0).fold(f32::MIN, f32::max);
        let y = ys[0];
        vec![(x0, y - half), (x1, y - half), (x1, y + half), (x0, y + half)]
    } else {
        convex_hull(&pts)
    };
    fill_polygon(&mut mask, w, h, &poly);
    let blur = odd((mouth_w * 0.08) as usize, 7);
    gaussian_blur_approx(&mut mask, w, h, blur);
    mask.iter().map(|v| v.clamp(0.0, 1.0)).collect()
}

/// Ellipse fallback mask (no reliable landmarks): centered oval, eroded,
/// feathered — mrb `_ellipse_crop_mask` in full-frame coordinates.
pub fn ellipse_fallback_mask(w: usize, h: usize, size: f32) -> Vec<f32> {
    let mut mask = vec![0.0f32; w * h];
    let cx = w as f32 / 2.0;
    let cy = size * 0.46;
    let ax = (size * 0.32).max(8.0);
    let ay = (size * 0.36).max(8.0);
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f32 - cx) / ax;
            let dy = (y as f32 - cy) / ay;
            if dx * dx + dy * dy <= 1.0 {
                mask[y * w + x] = 1.0;
            }
        }
    }
    erode_mask(&mut mask, w, h, odd((size * 0.04) as usize, 3));
    gaussian_blur_approx(&mut mask, w, h, odd((size * 0.06) as usize, 15));
    mask.iter().map(|v| v.clamp(0.0, 1.0)).collect()
}

/// Face-local ellipse fallback. Unlike the legacy centered helper above,
/// this is anchored to the detected target bbox, so a failed 106-point fit
/// cannot paste a face-sized oval at the center of an unrelated image.
fn ellipse_face_fallback_mask(w: usize, h: usize, bbox: [f32; 4]) -> Vec<f32> {
    let mut mask = vec![0.0f32; w.saturating_mul(h)];
    if w == 0 || h == 0 {
        return mask;
    }
    let bw = (bbox[2] - bbox[0]).max(1.0);
    let bh = (bbox[3] - bbox[1]).max(1.0);
    let cx = (bbox[0] + bbox[2]) * 0.5;
    let cy = bbox[1] + bh * 0.52;
    let ax = (bw * 0.52).max(4.0);
    let ay = (bh * 0.58).max(4.0);
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f32 - cx) / ax;
            let dy = (y as f32 - cy) / ay;
            if dx * dx + dy * dy <= 1.0 {
                mask[y * w + x] = 1.0;
            }
        }
    }
    let scale = bw.min(bh);
    erode_mask(&mut mask, w, h, odd((scale * 0.025) as usize, 3));
    gaussian_blur_approx(&mut mask, w, h, odd((scale * 0.045) as usize, 7));
    mask.iter().map(|v| v.clamp(0.0, 1.0)).collect()
}

/// Composite: out = mask·warped_crop + (1−mask)·frame.
fn paste_crop(
    frame: &image::RgbImage,
    crop: &image::RgbImage,
    sampling: &Affine2x3,
    mask: &[f32],
) -> image::RgbImage {
    let (w, h) = frame.dimensions();
    let warped = warp_sample(crop, w, h, sampling);
    let mut out = frame.clone();
    for y in 0..h {
        for x in 0..w {
            let a = mask[y as usize * w as usize + x as usize].clamp(0.0, 1.0);
            if a <= 0.0 {
                continue;
            }
            let wp = warped.get_pixel(x, y);
            let fp = out.get_pixel(x, y);
            let mut px = [0u8; 3];
            for c in 0..3 {
                px[c] = (a * wp[c] as f32 + (1.0 - a) * fp[c] as f32).round().clamp(0.0, 255.0)
                    as u8;
            }
            out.put_pixel(x, y, image::Rgb(px));
        }
    }
    out
}

/// A face located by SCRFD: bbox (x0, y0, x1, y1) in original image
/// coordinates plus the 5 landmarks.
#[derive(Debug, Clone)]
pub struct DetectedFace {
    pub bbox: [f32; 4],
    pub kps: [[f32; 2]; 5],
    pub score: f32,
}

/// Loaded swap stack: 2d106det + inswapper_128 (+ emap) + optional GPEN.
pub struct SwapEngine {
    lmk: ort::session::Session,
    swap: ort::session::Session,
    gpen: Option<ort::session::Session>,
    emap: Vec<f32>, // 512×512 row-major
}

impl SwapEngine {
    /// Load from a directory holding 2d106det.onnx, inswapper_128.onnx,
    /// inswapper_emap.f32 and optionally gpen_bfr_512.onnx.
    pub fn load(models_dir: &Path) -> Result<Self> {
        let lmk_path = models_dir.join("2d106det.onnx");
        let swap_path = models_dir.join("inswapper_128.onnx");
        let emap_path = models_dir.join("inswapper_emap.f32");
        let gpen_path = models_dir.join("gpen_bfr_512.onnx");
        for p in [&lmk_path, &swap_path, &emap_path] {
            if !p.exists() {
                bail!("faceswap model missing: {}", p.display());
            }
        }
        // FaceEngine and SwapEngine share one process-wide ORT environment;
        // ort intentionally rejects a second environment in the same process.
        let env = shared_ort_environment()?;
        let mk = |p: &Path| -> Result<ort::session::Session> {
            ort::session::Session::builder(&env)
                .map_err(|e| anyhow::anyhow!("ort builder: {e}"))?
                .with_auto_device(ort::session::builder::AutoDevicePolicy::PreferCPU)
                .map_err(|e| anyhow::anyhow!("ort ep: {e}"))?
                .commit_from_file(p)
                .with_context(|| format!("load {}", p.display()))
        };
        let lmk = mk(&lmk_path)?;
        let swap = mk(&swap_path)?;
        let gpen = if gpen_path.exists() { Some(mk(&gpen_path)?) } else { None };
        let emap_bytes = std::fs::read(&emap_path).with_context(|| emap_path.display().to_string())?;
        anyhow::ensure!(
            emap_bytes.len() == 512 * 512 * 4,
            "emap file must be 512×512 f32, got {} bytes",
            emap_bytes.len()
        );
        let emap: Vec<f32> = emap_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        anyhow::ensure!(emap.iter().all(|v| v.is_finite()), "emap contains non-finite values");
        Ok(Self { lmk, swap, gpen, emap })
    }

    pub fn gpen_available(&self) -> bool {
        self.gpen.is_some()
    }

    /// Dense 106 landmarks for one face, original image coordinates.
    /// Input: raw RGB 0..255 (the graph starts with Sub/Mul ⇒ mean 0, std 1).
    pub fn landmarks_106(
        &mut self,
        img: &image::RgbImage,
        bbox: [f32; 4],
    ) -> Result<Vec<[f32; 2]>> {
        anyhow::ensure!(img.width() >= 2 && img.height() >= 2, "landmark input image is too small");
        anyhow::ensure!(bbox.iter().all(|v| v.is_finite()), "face bbox contains non-finite coordinates");
        let w = bbox[2] - bbox[0];
        let h = bbox[3] - bbox[1];
        anyhow::ensure!(w > 1.0 && h > 1.0, "degenerate face bbox");
        let cx = (bbox[2] + bbox[0]) / 2.0;
        let cy = (bbox[3] + bbox[1]) / 2.0;
        let s = LM_SIZE as f32 / (w.max(h) * 1.5);
        // face_align.transform with rotate=0: scale about origin, then center
        // the scaled face at (96, 96).
        let m = Affine2x3 {
            a: s as f64,
            b: 0.0,
            tx: (LM_SIZE as f32 / 2.0 - cx * s) as f64,
            c: 0.0,
            d: s as f64,
            ty: (LM_SIZE as f32 / 2.0 - cy * s) as f64,
        };
        // warp_sample expects destination→source coordinates, while `m`
        // maps original-image coordinates→aligned crop coordinates.
        let sampling = m.invert()?;
        let aligned = warp_sample(img, LM_SIZE, LM_SIZE, &sampling);

        let mut blob = Array4::<f32>::zeros((1, 3, LM_SIZE as usize, LM_SIZE as usize));
        for (x, y, px) in aligned.enumerate_pixels() {
            let (x, y) = (x as usize, y as usize);
            blob[[0, 0, y, x]] = px[0] as f32;
            blob[[0, 1, y, x]] = px[1] as f32;
            blob[[0, 2, y, x]] = px[2] as f32;
        }
        let outputs = self
            .lmk
            .run(ort::inputs![ort::value::TensorRef::from_array_view(&blob)?])
            .context("2d106det forward")?;
        anyhow::ensure!(outputs.len() > 0, "2d106det returned no outputs");
        let flat: Vec<f32> = outputs[0]
            .try_extract_array::<f32>()?
            .view()
            .iter()
            .copied()
            .collect();
        anyhow::ensure!(flat.len() >= 212, "2d106det output too small: {}", flat.len());
        // Take the LAST 212 values (network may pad), denormalize: +1, ×96.
        let tail = &flat[flat.len() - 212..];
        let inv = sampling;
        let mut out = Vec::with_capacity(106);
        for i in 0..106 {
            let nx = tail[i * 2];
            let ny = tail[i * 2 + 1];
            anyhow::ensure!(nx.is_finite() && ny.is_finite(), "2d106det returned non-finite landmark {i}");
            let ax = (nx + 1.0) * (LM_SIZE as f32 / 2.0);
            let ay = (ny + 1.0) * (LM_SIZE as f32 / 2.0);
            let (ox, oy) = inv.apply(ax as f64, ay as f64);
            anyhow::ensure!(ox.is_finite() && oy.is_finite(), "2d106det transform produced non-finite landmark {i}");
            out.push([ox as f32, oy as f32]);
        }
        Ok(out)
    }

    /// inswapper forward: aligned 128 target crop + projected source
    /// embedding → swapped 128 crop (RGB).
    pub fn inswapper_forward(
        &mut self,
        aligned: &image::RgbImage,
        source_embedding: &[f32],
    ) -> Result<image::RgbImage> {
        anyhow::ensure!(
            source_embedding.len() == 512,
            "source embedding must be 512-d, got {}",
            source_embedding.len()
        );
        anyhow::ensure!(
            source_embedding.iter().all(|v| v.is_finite()),
            "source embedding contains non-finite values"
        );
        anyhow::ensure!(
            aligned.dimensions() == (SWAP_SIZE as u32, SWAP_SIZE as u32),
            "inswapper input must be 128×128, got {}×{}",
            aligned.width(), aligned.height()
        );
        // latent = emb · emap, renormalized.
        let mut latent = vec![0.0f32; 512];
        for (i, v) in source_embedding.iter().enumerate() {
            let row = &self.emap[i * 512..(i + 1) * 512];
            for (l, r) in latent.iter_mut().zip(row) {
                *l += v * r;
            }
        }
        let norm: f32 = latent.iter().map(|x| x * x).sum::<f32>().sqrt();
        anyhow::ensure!(norm > 1e-6, "zero latent");
        for l in latent.iter_mut() {
            *l /= norm;
        }

        let mut blob = Array4::<f32>::zeros((1, 3, SWAP_SIZE, SWAP_SIZE));
        for (x, y, px) in aligned.enumerate_pixels() {
            let (x, y) = (x as usize, y as usize);
            blob[[0, 0, y, x]] = px[0] as f32 / 255.0;
            blob[[0, 1, y, x]] = px[1] as f32 / 255.0;
            blob[[0, 2, y, x]] = px[2] as f32 / 255.0;
        }
        let latent_arr = ndarray::Array2::<f32>::from_shape_vec((1, 512), latent)?;
        let outputs = self
            .swap
            .run(ort::inputs![
                ort::value::TensorRef::from_array_view(&blob)?,
                ort::value::TensorRef::from_array_view(&latent_arr)?
            ])
            .context("inswapper forward")?;
        anyhow::ensure!(outputs.len() > 0, "inswapper returned no outputs");
        let out: Vec<f32> = outputs[0]
            .try_extract_array::<f32>()?
            .view()
            .iter()
            .copied()
            .collect();
        anyhow::ensure!(out.len() == 3 * SWAP_SIZE * SWAP_SIZE, "bad inswapper output: {} values", out.len());
        anyhow::ensure!(out.iter().all(|v| v.is_finite()), "inswapper output contains non-finite values");
        let mut img = image::RgbImage::new(SWAP_SIZE as u32, SWAP_SIZE as u32);
        for y in 0..SWAP_SIZE {
            for x in 0..SWAP_SIZE {
                let mut px = [0u8; 3];
                for c in 0..3 {
                    // NCHW → HWC, model emits RGB in [0,1].
                    let v = out[c * SWAP_SIZE * SWAP_SIZE + y * SWAP_SIZE + x];
                    px[c] = (v * 255.0).round().clamp(0.0, 255.0) as u8;
                }
                img.put_pixel(x as u32, y as u32, image::Rgb(px));
            }
        }
        Ok(img)
    }

    /// GPEN bfr-512 restoration on a square crop. Input RGB/255 → [-1,1];
    /// output (pred+1)/2·255 RGB.
    fn gpen_enhance(&mut self, crop: &image::RgbImage) -> Result<image::RgbImage> {
        let gpen = self
            .gpen
            .as_mut()
            .context("GPEN weights not loaded")?;
        let resized = image::imageops::resize(
            crop,
            GPEN_SIZE as u32,
            GPEN_SIZE as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let mut blob = Array4::<f32>::zeros((1, 3, GPEN_SIZE, GPEN_SIZE));
        for (x, y, px) in resized.enumerate_pixels() {
            let (x, y) = (x as usize, y as usize);
            for c in 0..3 {
                blob[[0, c, y, x]] = (px[c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
        let outputs = gpen
            .run(ort::inputs![ort::value::TensorRef::from_array_view(&blob)?])
            .context("GPEN forward")?;
        anyhow::ensure!(outputs.len() > 0, "GPEN returned no outputs");
        let out: Vec<f32> = outputs[0]
            .try_extract_array::<f32>()?
            .view()
            .iter()
            .copied()
            .collect();
        anyhow::ensure!(out.len() == 3 * GPEN_SIZE * GPEN_SIZE, "bad GPEN output: {} values", out.len());
        anyhow::ensure!(out.iter().all(|v| v.is_finite()), "GPEN output contains non-finite values");
        let mut img = image::RgbImage::new(GPEN_SIZE as u32, GPEN_SIZE as u32);
        for y in 0..GPEN_SIZE {
            for x in 0..GPEN_SIZE {
                let mut px = [0u8; 3];
                for c in 0..3 {
                    let v = out[c * GPEN_SIZE * GPEN_SIZE + y * GPEN_SIZE + x];
                    px[c] = ((v + 1.0) / 2.0 * 255.0).round().clamp(0.0, 255.0) as u8;
                }
                img.put_pixel(x as u32, y as u32, image::Rgb(px));
            }
        }
        Ok(img)
    }

    /// Swap the identity of `source_embedding` onto the given face of
    /// `target`. `face_bbox`/`face_kps5` come from SCRFD detection; the
    /// 106-landmark fit runs internally. Falls back to the ellipse mask when
    /// the landmark fit is unreliable.
    pub fn swap_face(
        &mut self,
        target: &image::RgbImage,
        face_bbox: [f32; 4],
        face_kps5: [[f32; 2]; 5],
        source_embedding: &[f32],
    ) -> Result<image::RgbImage> {
        anyhow::ensure!(target.width() >= 2 && target.height() >= 2, "target image is too small");
        anyhow::ensure!(
            face_kps5.iter().flatten().all(|v| v.is_finite()),
            "target face landmarks contain non-finite coordinates"
        );
        anyhow::ensure!(
            face_bbox.iter().all(|v| v.is_finite())
                && face_bbox[2] > face_bbox[0]
                && face_bbox[3] > face_bbox[1],
            "target face bbox is invalid"
        );
        // Alignment: similarity transform kps5 → 128-template (Umeyama).
        let kps: Vec<[f32; 2]> = face_kps5.to_vec();
        let m = umeyama_similarity(&kps, &SWAP_TEMPLATE)
            .context("inswapper alignment")?;
        // `m` is target→crop. The crop sampler needs crop→target.
        let sampling = m.invert().context("invert inswapper alignment")?;
        let aligned = warp_sample(target, SWAP_SIZE as u32, SWAP_SIZE as u32, &sampling);

        let mut swapped = self.inswapper_forward(&aligned, source_embedding)?;
        reinhard_color_match(&mut swapped, &aligned);

        // GPEN restore at 512, pasted through the scaled affine.
        let (paste_crop_img, paste_m) = if self.gpen.is_some() {
            let enhanced = self.gpen_enhance(&swapped)?;
            // GPEN returns 512² while inswapper returns 128². Resize the raw
            // swap to the same canvas before blending; zipping the original
            // 128² iterator would otherwise blend only the first 16,384
            // pixels and leave the rest of the GPEN image untouched.
            let raw_512 = image::imageops::resize(
                &swapped,
                GPEN_SIZE as u32,
                GPEN_SIZE as u32,
                image::imageops::FilterType::CatmullRom,
            );
            let mut blended = enhanced;
            for (bp, sp) in blended
                .enumerate_pixels_mut()
                .zip(raw_512.enumerate_pixels())
            {
                for c in 0..3 {
                    bp.2[c] = (GPEN_BLEND * bp.2[c] as f32 + (1.0 - GPEN_BLEND) * sp.2[c] as f32)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            }
            let scale = GPEN_SIZE as f64 / SWAP_SIZE as f64;
            (blended, m.scaled(scale))
        } else {
            (swapped, m)
        };

        // Mask: 106-landmark contour + open-mouth hole; ellipse fallback.
        let (w, h) = (target.width() as usize, target.height() as usize);
        let lmk = self.landmarks_106(target, face_bbox);
        let mask = match lmk {
            Ok(l)
                if l.len() >= 33
                    && l.iter().flatten().all(|v| v.is_finite())
                    && l[..33].iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max)
                        - l[..33].iter().map(|p| p[0]).fold(f32::INFINITY, f32::min) > 1.0
                    && l[..33].iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max)
                        - l[..33].iter().map(|p| p[1]).fold(f32::INFINITY, f32::min) > 1.0 =>
            {
                let contour = face_contour_mask(w, h, &l);
                let keep = mouth_keep_mask(w, h, &l);
                let mask: Vec<f32> = contour
                    .into_iter()
                    .zip(keep)
                    .map(|(c, k)| (c * (1.0 - k)).clamp(0.0, 1.0))
                    .collect();
                // Degenerate contour (collapsed landmarks) yields an empty
                // mask; fall back to the face-local ellipse instead of
                // silently pasting nothing.
                if mask.iter().sum::<f32>() < 1.0 {
                    ellipse_face_fallback_mask(w, h, face_bbox)
                } else {
                    mask
                }
            }
            _ => ellipse_face_fallback_mask(w, h, face_bbox),
        };

        // Paste: `paste_m` already maps frame coordinates → crop coordinates,
        // which is exactly the dst(frame)→src(crop) sampler paste_crop needs.
        // mrb's _paste_crop inverts this same forward matrix and hands the
        // result to cv2.warpAffine as src→dst; inverting here would apply the
        // crop→frame map to frame pixels, sampling the crop out of bounds and
        // pasting a black blob. For the GPEN path the crop is 512² and the
        // forward map is 4·M (all elements, mrb _scale_affine).
        Ok(paste_crop(target, &paste_crop_img, &paste_m, &mask))
    }
}

#[cfg(test)]
// Synthetic 106-point fixtures are written by landmark id (`lmk[i] = ...`),
// which is the readable form here; the index is the landmark, not a position.
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    fn affine(a: f64, b: f64, tx: f64, c: f64, d: f64, ty: f64) -> Affine2x3 {
        Affine2x3 { a, b, tx, c, d, ty }
    }

    #[test]
    fn affine_invert_roundtrip() {
        let m = affine(2.0, 0.5, 7.0, -0.5, 2.0, -3.0);
        let inv = m.invert().unwrap();
        for (x, y) in [(1.0, 2.0), (-4.0, 0.5), (100.0, -20.0)] {
            let (mx, my) = m.apply(x, y);
            let (rx, ry) = inv.apply(mx, my);
            assert!((rx - x).abs() < 1e-9 && (ry - y).abs() < 1e-9);
        }
    }

    #[test]
    fn affine_scaled_matches_mrb() {
        // mrb _scale_affine multiplies ALL elements including translation.
        let m = affine(1.0, 0.0, 5.0, 0.0, 1.0, 9.0);
        let s = m.scaled(4.0);
        assert_eq!(s.tx, 20.0);
        assert_eq!(s.ty, 36.0);
        assert_eq!(s.a, 4.0);
    }

    #[test]
    fn umeyama_recovers_similarity() {
        // Ground truth: rotate 30°, scale 2, translate (10, -5).
        let (ang, sc) = (30f32.to_radians(), 2.0f32);
        let (a, b, tx, ty) = (
            (sc * ang.cos()) as f64,
            (-sc * ang.sin()) as f64,
            10.0,
            -5.0,
        );
        let (c, d) = ((sc * ang.sin()) as f64, (sc * ang.cos()) as f64);
        let m_true = affine(a, b, tx, c, d, ty);
        let src: Vec<[f32; 2]> = vec![[10.0, 3.0], [40.0, 60.0], [70.0, 20.0], [25.0, 90.0], [55.0, 45.0]];
        let dst: Vec<[f32; 2]> = src
            .iter()
            .map(|p| {
                let (x, y) = m_true.apply(p[0] as f64, p[1] as f64);
                [x as f32, y as f32]
            })
            .collect();
        let est = umeyama_similarity(&src, &dst).unwrap();
        for p in src.iter().take(3) {
            let (ex, ey) = est.apply(p[0] as f64, p[1] as f64);
            let (gx, gy) = m_true.apply(p[0] as f64, p[1] as f64);
            assert!((ex - gx).abs() < 1e-6 && (ey - gy).abs() < 1e-6);
        }
    }

    #[test]
    fn umeyama_identity() {
        let pts: Vec<[f32; 2]> = vec![[0.0, 0.0], [10.0, 0.0], [5.0, 8.0], [3.0, 1.0]];
        let m = umeyama_similarity(&pts, &pts).unwrap();
        assert!((m.a - 1.0).abs() < 1e-6 && (m.d - 1.0).abs() < 1e-6);
        assert!(m.b.abs() < 1e-6 && m.c.abs() < 1e-6);
        assert!(m.tx.abs() < 1e-6 && m.ty.abs() < 1e-6);
    }

    #[test]
    fn umeyama_isotropic_rotation_and_scale() {
        // A centered square has isotropic covariance; this is the repeated-
        // eigenvalue case where independently chosen eigenvectors collapse.
        let src = vec![[-1.0f32, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let angle = 0.7f64;
        let scale = 1.8f64;
        let truth = affine(
            scale * angle.cos(),
            -scale * angle.sin(),
            13.0,
            scale * angle.sin(),
            scale * angle.cos(),
            -4.0,
        );
        let dst: Vec<[f32; 2]> = src.iter().map(|p| {
            let (x, y) = truth.apply(p[0] as f64, p[1] as f64);
            [x as f32, y as f32]
        }).collect();
        let estimated = umeyama_similarity(&src, &dst).unwrap();
        for point in &src {
            let actual = estimated.apply(point[0] as f64, point[1] as f64);
            let expected = truth.apply(point[0] as f64, point[1] as f64);
            assert!((actual.0 - expected.0).abs() < 1e-5);
            assert!((actual.1 - expected.1).abs() < 1e-5);
        }
    }

    #[test]
    fn reinhard_identity_is_noop() {
        let img = image::RgbImage::from_fn(16, 16, |x, y| {
            image::Rgb([(x * 7 % 255) as u8, (y * 13 % 255) as u8, 128])
        });
        let mut out = img.clone();
        reinhard_color_match(&mut out, &img);
        for (a, b) in out.enumerate_pixels().zip(img.enumerate_pixels()) {
            assert_eq!(a.1, b.1);
        }
    }

    #[test]
    fn reinhard_shifts_mean() {
        let dark = image::RgbImage::from_fn(32, 32, |_, _| image::Rgb([40, 40, 40]));
        let bright = image::RgbImage::from_fn(32, 32, |_, _| image::Rgb([200, 200, 200]));
        let mut out = dark.clone();
        reinhard_color_match(&mut out, &bright);
        let mean: f64 = out.enumerate_pixels().map(|(_, _, p)| p[0] as f64).sum::<f64>()
            / (32 * 32) as f64;
        assert!((mean - 200.0).abs() < 2.0, "mean {mean}");
    }

    #[test]
    fn hull_square() {
        let pts = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (5.0, 5.0), (2.0, 3.0)];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn polygon_fill_counts_area() {
        let w = 20usize;
        let h = 20usize;
        let mut mask = vec![0.0f32; w * h];
        fill_polygon(&mut mask, w, h, &[(2.0, 2.0), (12.0, 2.0), (12.0, 12.0), (2.0, 12.0)]);
        let area: f32 = mask.iter().sum();
        // 10×10 square ± boundary rounding.
        assert!((area - 100.0).abs() < 12.0, "area {area}");
    }

    #[test]
    fn contour_mask_inside_image_and_positive() {
        // Synthetic 106 landmarks: an oval outline + brows etc.
        let mut lmk = vec![[0.0f32; 2]; 106];
        for i in 0..33 {
            let t = i as f32 / 33.0 * std::f32::consts::TAU;
            lmk[i] = [50.0 + 30.0 * t.cos(), 50.0 + 40.0 * t.sin()];
        }
        for i in 43..52 {
            lmk[i] = [40.0 + (i - 43) as f32 * 2.0, 40.0];
        }
        for i in 97..106 {
            lmk[i] = [40.0 + (i - 97) as f32 * 2.0, 42.0];
        }
        for i in 52..64 {
            lmk[i] = [45.0 + (i - 52) as f32, 60.0];
        }
        for i in 64..72 {
            lmk[i] = [46.0 + (i - 64) as f32, 66.0];
        }
        let mask = face_contour_mask(100, 100, &lmk);
        assert_eq!(mask.len(), 100 * 100);
        let sum: f32 = mask.iter().sum();
        assert!(sum > 100.0, "mask too small: {sum}");
        assert!(mask.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn mouth_mar_closed_vs_open() {
        // Closed mouth: inner lips coincide with outer.
        let mut lmk = vec![[0.0f32; 2]; 106];
        for i in 52..64 {
            lmk[i] = [30.0 + (i - 52) as f32, 60.0];
        }
        for i in 64..72 {
            lmk[i] = [42.0 + (i - 64) as f32, 60.0];
        }
        assert!(mouth_aspect_ratio(&lmk) < MOUTH_MAR_THRESHOLD);
        // Open mouth: inner lips 10px below.
        for i in 64..72 {
            lmk[i] = [42.0 + (i - 64) as f32, 70.0 + ((i - 64) % 2) as f32 * 12.0];
        }
        assert!(mouth_aspect_ratio(&lmk) >= MOUTH_MAR_THRESHOLD);
    }

    #[test]
    fn mouth_keep_mask_opens_only_when_open() {
        let mut lmk = vec![[0.0f32; 2]; 106];
        for i in 52..64 {
            lmk[i] = [30.0 + (i - 52) as f32, 60.0];
        }
        for i in 64..72 {
            lmk[i] = [42.0 + (i - 64) as f32, 60.0];
        }
        let closed = mouth_keep_mask(100, 100, &lmk);
        assert!(closed.iter().sum::<f32>() < 1e-6);
        for i in 64..72 {
            lmk[i] = [42.0 + (i - 64) as f32, 72.0 + ((i - 64) % 2) as f32 * 12.0];
        }
        let open = mouth_keep_mask(100, 100, &lmk);
        assert!(open.iter().sum::<f32>() > 10.0);
    }

    #[test]
    fn ellipse_fallback_is_centered() {
        let mask = ellipse_fallback_mask(100, 100, 100.0);
        assert!(mask[50 * 100 + 50] > 0.9);
        assert_eq!(mask[0], 0.0);
    }

    #[test]
    fn paste_respects_mask() {
        let frame = image::RgbImage::from_fn(32, 32, |_, _| image::Rgb([10, 10, 10]));
        let crop = image::RgbImage::from_fn(32, 32, |_, _| image::Rgb([200, 200, 200]));
        let mut mask = vec![0.0f32; 32 * 32];
        mask[16 * 32 + 16] = 1.0;
        let out = paste_crop(&frame, &crop, &affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), &mask);
        assert_eq!(out.get_pixel(16, 16)[0], 200);
        assert_eq!(out.get_pixel(0, 0)[0], 10);
    }

    #[test]
    fn paste_roundtrip_uses_forward_sampler() {
        // paste_crop samples the crop with the frame→crop matrix directly.
        // Passing its inverse pastes out-of-bounds black — the FS1 bug.
        let target = image::RgbImage::from_fn(200, 200, |x, y| {
            image::Rgb([(x / 2) as u8, (y / 2) as u8, 100])
        });
        let kps: Vec<[f32; 2]> = SWAP_TEMPLATE
            .iter()
            .map(|p| [p[0] * 1.2 + 40.0, p[1] * 1.2 + 20.0])
            .collect();
        let m = umeyama_similarity(&kps, &SWAP_TEMPLATE).unwrap();
        let aligned = warp_sample(
            &target,
            SWAP_SIZE as u32,
            SWAP_SIZE as u32,
            &m.invert().unwrap(),
        );
        let mask = vec![1.0f32; 200 * 200];
        let out = paste_crop(&target, &aligned, &m, &mask);
        for (i, p) in kps.iter().enumerate() {
            let (x, y) = (p[0].round() as u32, p[1].round() as u32);
            let (a, b) = (out.get_pixel(x, y), target.get_pixel(x, y));
            let diff: i32 = (0..3).map(|c| (a[c] as i32 - b[c] as i32).abs()).sum();
            assert!(diff < 18, "kps {i} diverged: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn warp_sample_identity() {
        let img = image::RgbImage::from_fn(8, 8, |x, y| {
            image::Rgb([(x * 30) as u8, (y * 30) as u8, 7])
        });
        let out = warp_sample(&img, 8, 8, &affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0));
        assert_eq!(out.get_pixel(3, 5), img.get_pixel(3, 5));
    }

    #[test]
    fn warp_sample_translation() {
        let img = image::RgbImage::from_fn(8, 8, |x, _| image::Rgb([(x * 30) as u8, 0, 0]));
        // dst(x) = src(x - 2): sampling matrix maps dst→src with tx = -2.
        let out = warp_sample(&img, 8, 8, &affine(1.0, 0.0, -2.0, 0.0, 1.0, 0.0));
        assert_eq!(out.get_pixel(5, 0), img.get_pixel(3, 0));
    }

    #[test]
    fn contour_mask_is_empty_when_landmarks_are_unusable() {
        // Full-frame coordinates: a 128-crop-space ellipse here would paste a
        // tiny oval at the top centre of the frame. The caller substitutes a
        // bbox-anchored ellipse instead.
        assert_eq!(face_contour_mask(64, 64, &[]).iter().sum::<f32>(), 0.0);
        let mut lmk = vec![[0.0f32; 2]; 106];
        lmk[3][0] = f32::NAN;
        assert_eq!(face_contour_mask(64, 64, &lmk).iter().sum::<f32>(), 0.0);
        // Collapsed contour (zero height) is equally unusable.
        let flat = vec![[10.0f32, 10.0]; 106];
        assert_eq!(face_contour_mask(64, 64, &flat).iter().sum::<f32>(), 0.0);
    }
}
