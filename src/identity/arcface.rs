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

use std::sync::{Arc, OnceLock};

use anyhow::{bail, Context, Result};
use ndarray::Array4;


/// Return the single process-wide ONNX Runtime environment.
///
/// `ort` permits only one committed environment per process. Both the
/// embedding engine and the face-swap engine must therefore share this
/// instance; sessions remain independent and may be loaded from different
/// model directories. Cache initialization failures as text so callers get
/// an ordinary anyhow error without requiring `anyhow::Error: Clone`.
pub(crate) fn shared_ort_environment() -> Result<Arc<ort::environment::Environment>> {
    static ENV: OnceLock<std::result::Result<Arc<ort::environment::Environment>, String>> =
        OnceLock::new();

    match ENV.get_or_init(|| {
        ort::init()
            .with_name("avatar-media-mcp")
            .build()
            .map(Arc::new)
            .map_err(|error| error.to_string())
    }) {
        Ok(env) => Ok(Arc::clone(env)),
        Err(error) => bail!("initialize shared ONNX Runtime environment: {error}"),
    }
}

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
const DET_INPUT_SIZE: u32 = 640;
const NMS_THRESHOLD: f32 = 0.4;

/// Resize without changing aspect ratio and place the image at the top-left
/// of the detector's fixed 640×640 canvas (InsightFace SCRFD convention).
/// `valid` is the resized image extent; detections in the padded region are
/// discarded before coordinates are mapped back to the source image.
fn prepare_detector_input(
    image: &image::RgbImage,
) -> Result<(image::RgbImage, f32, u32, u32)> {
    anyhow::ensure!(image.width() > 0 && image.height() > 0, "input image is empty");
    let scale = DET_INPUT_SIZE as f32 / image.width().max(image.height()) as f32;
    let rw = ((image.width() as f32 * scale).round() as u32).clamp(1, DET_INPUT_SIZE);
    let rh = ((image.height() as f32 * scale).round() as u32).clamp(1, DET_INPUT_SIZE);
    let resized = image::imageops::resize(
        image,
        rw,
        rh,
        image::imageops::FilterType::Triangle,
    );
    let mut canvas = image::RgbImage::from_pixel(
        DET_INPUT_SIZE,
        DET_INPUT_SIZE,
        image::Rgb([0, 0, 0]),
    );
    image::imageops::replace(&mut canvas, &resized, 0, 0);
    Ok((canvas, scale, rw, rh))
}

pub struct FaceEngine {
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
        let env = shared_ort_environment()?;
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
        Ok(Self { det, rec })
    }

    /// Embed every face found in the image, largest (main person) first.
    pub fn embed_image(&mut self, image_path: &std::path::Path) -> Result<Vec<FaceEmbedding>> {
        let img = image::open(image_path)
            .with_context(|| format!("open {}", image_path.display()))?
            .to_rgb8();
        let (small, scale, valid_w, valid_h) = prepare_detector_input(&img)?;
        let faces = self.detect(&small, scale, valid_w, valid_h)?;
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

    /// Detect faces with bounding boxes + 5 landmarks (original coords),
    /// best score first. Used by the face-swap pipeline (faceswap.rs).
    pub fn detect_faces(
        &mut self,
        image_path: &std::path::Path,
    ) -> Result<Vec<DetectedFaceBox>> {
        let img = image::open(image_path)
            .with_context(|| format!("open {}", image_path.display()))?
            .to_rgb8();
        let (small, scale, valid_w, valid_h) = prepare_detector_input(&img)?;
        let (source_w, source_h) = (img.width() as f32, img.height() as f32);
        let raw = self.detect_raw(&small, valid_w, valid_h)?;
        let mut out: Vec<DetectedFaceBox> = raw
            .into_iter()
            .map(|(score, bbox, kps)| DetectedFaceBox {
                bbox: [
                    (bbox[0] / scale).clamp(0.0, source_w),
                    (bbox[1] / scale).clamp(0.0, source_h),
                    (bbox[2] / scale).clamp(0.0, source_w),
                    (bbox[3] / scale).clamp(0.0, source_h),
                ],
                kps: [
                    [(kps[0][0] / scale).clamp(0.0, source_w), (kps[0][1] / scale).clamp(0.0, source_h)],
                    [(kps[1][0] / scale).clamp(0.0, source_w), (kps[1][1] / scale).clamp(0.0, source_h)],
                    [(kps[2][0] / scale).clamp(0.0, source_w), (kps[2][1] / scale).clamp(0.0, source_h)],
                    [(kps[3][0] / scale).clamp(0.0, source_w), (kps[3][1] / scale).clamp(0.0, source_h)],
                    [(kps[4][0] / scale).clamp(0.0, source_w), (kps[4][1] / scale).clamp(0.0, source_h)],
                ],
                score,
            })
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(out)
    }

    /// SCRFD forward pass. Returns landmark sets (5×2) in ORIGINAL image
    /// coordinates. det_10g emits 3 stride groups × (scores, boxes, kps).
    fn detect(
        &mut self,
        small: &image::RgbImage,
        scale: f32,
        valid_w: u32,
        valid_h: u32,
    ) -> Result<Vec<[[f32; 2]; 5]>> {
        let raw = self.detect_raw(small, valid_w, valid_h)?;
        Ok(raw
            .into_iter()
            .map(|(_, _, kps)| {
                let mut kps = kps;
                for kp in kps.iter_mut() {
                    kp[0] /= scale;
                    kp[1] /= scale;
                }
                kps
            })
            .collect())
    }

    /// Raw SCRFD pass: (score, bbox, kps) in 640×640 detector-canvas coordinates.
    ///
    /// The supplied model graph exposes three output triplets in stride order
    /// (8, 16, 32): scores, box distances, and landmark offsets. SCRFD's
    /// exported heads are not decoded boxes: each feature-map location has
    /// one or more anchors, and box/keypoint values are distances/offsets
    /// measured in feature cells. Decode them against the stride-scaled
    /// feature-map centers here, exactly once.
    fn detect_raw(
        &mut self,
        small: &image::RgbImage,
        valid_w: u32,
        valid_h: u32,
    ) -> Result<Vec<(f32, [f32; 4], [[f32; 2]; 5])>> {
        let (rw, rh) = (small.width() as usize, small.height() as usize);
        anyhow::ensure!(rw > 0 && rh > 0, "SCRFD input image is empty");
        anyhow::ensure!(
            valid_w > 0 && valid_h > 0 && valid_w <= small.width() && valid_h <= small.height(),
            "invalid SCRFD valid image extent {valid_w}×{valid_h} for {}×{} canvas",
            small.width(), small.height()
        );

        let mut blob = Array4::<f32>::zeros((1, 3, rh, rw));
        for (x, y, px) in small.enumerate_pixels() {
            let (x, y) = (x as usize, y as usize);
            blob[[0, 0, y, x]] = px[0] as f32;
            blob[[0, 1, y, x]] = px[1] as f32;
            blob[[0, 2, y, x]] = px[2] as f32;
        }
        blob.mapv_inplace(|v| (v - 127.5) / 127.5);

        let outputs = self
            .det
            .run(ort::inputs![ort::value::TensorRef::from_array_view(&blob)?])
            .context("SCRFD forward")?;
        anyhow::ensure!(outputs.len() == 9, "SCRFD expected exactly 9 outputs, got {}", outputs.len());

        let strides = [8usize, 16, 32];
        let mut all = Vec::<(f32, [f32; 4], [[f32; 2]; 5])>::new();

        // Output order is grouped by KIND, not by stride triplet:
        // [scores s8, s16, s32, boxes s8, s16, s32, kps s8, s16, s32]
        // (verified live: shapes (12800,1)/(3200,1)/(800,1) then (·,4)/(·,10)).
        for (group, stride) in strides.into_iter().enumerate() {
            let score_tensor = outputs[group]
                .try_extract_array::<f32>()
                .context("SCRFD score output")?;
            let box_tensor = outputs[3 + group]
                .try_extract_array::<f32>()
                .context("SCRFD box output")?;
            let kps_tensor = outputs[6 + group]
                .try_extract_array::<f32>()
                .context("SCRFD landmark output")?;

            let box_shape = box_tensor.shape().to_vec();
            let kps_shape = kps_tensor.shape().to_vec();
            let scores: Vec<f32> = score_tensor.iter().copied().collect();
            let boxes: Vec<f32> = box_tensor.iter().copied().collect();
            let landmarks: Vec<f32> = kps_tensor.iter().copied().collect();

            let feat_w = rw.div_ceil(stride);
            let feat_h = rh.div_ceil(stride);
            let cells = feat_w
                .checked_mul(feat_h)
                .context("SCRFD feature map size overflow")?;
            anyhow::ensure!(cells > 0 && scores.len() % cells == 0,
                "SCRFD stride-{stride} score count {} is not divisible by feature map {}×{}",
                scores.len(), feat_w, feat_h);
            let anchors = scores.len() / cells;
            anyhow::ensure!(anchors == 1 || anchors == 2,
                "unsupported SCRFD anchor count {anchors} at stride {stride}");
            anyhow::ensure!(boxes.len() == scores.len() * 4,
                "SCRFD stride-{stride} boxes have {} values for {} scores",
                boxes.len(), scores.len());
            anyhow::ensure!(landmarks.len() == scores.len() * 10,
                "SCRFD stride-{stride} landmarks have {} values for {} scores",
                landmarks.len(), scores.len());

            // Some exports include sigmoid in the graph, others expose logits.
            // Detect logits by range instead of applying sigmoid twice.
            let scores_are_logits = scores.iter().any(|v| *v < 0.0 || *v > 1.0);
            for (i, raw_score) in scores.iter().copied().enumerate() {
                if !raw_score.is_finite() {
                    continue;
                }
                let score = if scores_are_logits {
                    if raw_score >= 0.0 {
                        1.0 / (1.0 + (-raw_score).exp())
                    } else {
                        let e = raw_score.exp();
                        e / (1.0 + e)
                    }
                } else {
                    raw_score
                };
                if !(DET_THRESHOLD..=1.0).contains(&score) {
                    continue;
                }

                // SCRFD ordering is row-major (y, x, anchor). Two anchors at
                // each cell share the same center.
                let cell = i / anchors;
                let anchor_x = (cell % feat_w) as f32 * stride as f32;
                let anchor_y = (cell / feat_w) as f32 * stride as f32;
                // The 640-square canvas is padded on the right/bottom. Do not
                // emit detections whose anchor is wholly inside that padding.
                if anchor_x >= valid_w as f32 || anchor_y >= valid_h as f32 {
                    continue;
                }

                let box_at = |component| {
                    scrfd_component_index(&box_shape, scores.len(), 4, i, component)
                        .map(|index| boxes[index])
                };
                let distances = [box_at(0)?, box_at(1)?, box_at(2)?, box_at(3)?];
                if !distances.iter().all(|v| v.is_finite() && *v >= 0.0) {
                    continue;
                }
                let [left, top, right, bottom] = distances.map(|v| v * stride as f32);
                let bbox = [
                    (anchor_x - left).clamp(0.0, valid_w as f32),
                    (anchor_y - top).clamp(0.0, valid_h as f32),
                    (anchor_x + right).clamp(0.0, valid_w as f32),
                    (anchor_y + bottom).clamp(0.0, valid_h as f32),
                ];
                if !bbox.iter().all(|v| v.is_finite()) || bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
                    continue;
                }

                let mut kps = [[0.0f32; 2]; 5];
                let mut valid = true;
                for (j, kp) in kps.iter_mut().enumerate() {
                    let x_index = scrfd_component_index(&kps_shape, scores.len(), 10, i, j * 2)?;
                    let y_index = scrfd_component_index(&kps_shape, scores.len(), 10, i, j * 2 + 1)?;
                    kp[0] = (anchor_x + landmarks[x_index] * stride as f32)
                        .clamp(0.0, valid_w as f32);
                    kp[1] = (anchor_y + landmarks[y_index] * stride as f32)
                        .clamp(0.0, valid_h as f32);
                    valid &= kp[0].is_finite() && kp[1].is_finite();
                }
                if valid {
                    all.push((score, bbox, kps));
                }
            }
        }

        // Suppress duplicate boxes across anchors and feature-map scales.
        all.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut kept: Vec<(f32, [f32; 4], [[f32; 2]; 5])> = Vec::with_capacity(all.len());
        for candidate in all {
            if kept.iter().all(|prior| bbox_iou(&candidate.1, &prior.1) <= NMS_THRESHOLD) {
                kept.push(candidate);
            }
        }
        Ok(kept)
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
        let outputs = self
            .rec
            .run(ort::inputs![ort::value::TensorRef::from_array_view(&blob)?])
            .context("ArcFace forward")?;
        anyhow::ensure!(outputs.len() > 0, "ArcFace model returned no outputs");
        let v: Vec<f32> = outputs[0]
            .try_extract_array::<f32>()?
            .view()
            .iter()
            .copied()
            .collect();
        anyhow::ensure!(v.len() == EMBEDDING_DIM, "unexpected embedding dim {}", v.len());
        anyhow::ensure!(v.iter().all(|x| x.is_finite()), "ArcFace embedding contains non-finite values");
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

/// A detected face with bounding box, 5 landmarks and detection score,
/// all in original image coordinates (used by the face-swap pipeline).
#[derive(Debug, Clone)]
pub struct DetectedFaceBox {
    pub bbox: [f32; 4],
    pub kps: [[f32; 2]; 5],
    pub score: f32,
}

/// Resolve an item/component offset across SCRFD output layouts. Supports
/// interleaved, channel-first, and flattened ONNX Reshape outputs.
fn scrfd_component_index(
    shape: &[usize],
    count: usize,
    components: usize,
    item: usize,
    component: usize,
) -> Result<usize> {
    anyhow::ensure!(item < count && component < components, "SCRFD tensor index out of range");
    let expected = count
        .checked_mul(components)
        .context("SCRFD component count overflow")?;
    let actual = shape.iter().try_fold(1usize, |n, dim| n.checked_mul(*dim));
    anyhow::ensure!(actual == Some(expected), "SCRFD output shape {shape:?} has {:?} values; expected {expected}", actual);

    // Handle a component axis in any non-batch position. This covers
    // [1,N,C], [1,C,N], and reshaped variants such as [1,N,C,1] or
    // [1,1,C,N] without assuming that the flattened layout is interleaved.
    let mut component_axes = Vec::new();
    for axis in 1..shape.len() {
        if shape[axis] != components {
            continue;
        }
        let before = shape[1..axis].iter().product::<usize>();
        let after = shape[axis + 1..].iter().product::<usize>();
        if before.checked_mul(after) == Some(count) {
            component_axes.push((before, after));
        }
    }
    anyhow::ensure!(component_axes.len() <= 1,
        "ambiguous SCRFD component axis in output shape {shape:?}");
    if let Some((before, after)) = component_axes.first().copied() {
        let group = item / after;
        let offset = item % after;
        return Ok(group * components * after + component * after + offset);
    }

    // Some exporter versions collapse all feature dimensions and channels
    // into [1,N*C]; the graph's flatten order is item-major in that case.
    anyhow::ensure!(shape.len() <= 2,
        "unsupported SCRFD output shape {shape:?}; no {components}-wide component axis");
    Ok(item * components + component)
}

/// Intersection-over-union for SCRFD NMS.
fn bbox_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x0 = a[0].max(b[0]);
    let y0 = a[1].max(b[1]);
    let x1 = a[2].min(b[2]);
    let y1 = a[3].min(b[3]);
    let intersection = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - intersection;
    if union > 0.0 { intersection / union } else { 0.0 }
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
    anyhow::ensure!(
        kps.iter().flatten().all(|v| v.is_finite()),
        "face landmarks contain non-finite coordinates"
    );
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
    let forward = solve6(&ata, &atb)?;
    let inverse = invert_affine(&forward)?;
    align_affine_sample(img, &inverse)
}

/// Affine warp via manual bilinear sampling: for each output pixel (ox, oy)
/// of the 112×112 crop, sample the source at the inverse transform.
fn align_affine_sample(img: &image::RgbImage, inverse: &[f64; 6]) -> Result<image::RgbImage> {
    anyhow::ensure!(img.width() >= 2 && img.height() >= 2, "source image is too small to align");
    anyhow::ensure!(inverse.iter().all(|v| v.is_finite()), "non-finite alignment transform");
    let (sw, sh) = (img.width() as i64, img.height() as i64);
    let mut out = image::RgbImage::new(112, 112);
    for oy in 0..112u32 {
        for ox in 0..112u32 {
            let sx = inverse[0] * ox as f64 + inverse[1] * oy as f64 + inverse[2];
            let sy = inverse[3] * ox as f64 + inverse[4] * oy as f64 + inverse[5];
            if !sx.is_finite() || !sy.is_finite() || sx < 0.0 || sy < 0.0
                || sx >= (sw - 1) as f64 || sy >= (sh - 1) as f64
            {
                continue;
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
    Ok(out)
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
    #[test]
    fn scrfd_component_index_supports_channel_last_and_channel_first() {
        assert_eq!(scrfd_component_index(&[1, 7, 4], 7, 4, 3, 2).unwrap(), 14);
        assert_eq!(scrfd_component_index(&[1, 4, 7], 7, 4, 3, 2).unwrap(), 17);
        assert_eq!(scrfd_component_index(&[1, 28], 7, 4, 3, 2).unwrap(), 14);
    }

    #[test]
    fn scrfd_component_index_rejects_ambiguous_shape() {
        assert!(scrfd_component_index(&[1, 3, 5], 7, 4, 0, 0).is_err());
    }

    #[test]
    fn bbox_iou_is_exact_for_identical_and_disjoint_boxes() {
        let a = [0.0, 0.0, 10.0, 10.0];
        assert!((bbox_iou(&a, &a) - 1.0).abs() < 1e-6);
        let b = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&a, &b), 0.0);
    }

}
