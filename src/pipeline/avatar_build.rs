//! Avatar build pipeline.
//!
//! `normalize_photo` → `check_photo` → canon sheet (`build_canon`, M3).

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Normalize a photo: ≤2048px long side, metadata stripped, sRGB assumed.
///
/// Colour note: the `image` crate has no ICC engine, so a Display-P3 or Adobe
/// RGB source is reinterpreted as sRGB — the profile is dropped, not converted.
/// Wide-gamut photos therefore shift slightly; fixing that needs an lcms2 pass.
///
/// Encoding: `.jpg`/`.jpeg` output is JPEG q95 (a 30-photo canon stays ~15 MB
/// instead of ~150 MB of PNG), anything else is PNG.
pub fn normalize_photo(input: &Path, output: &Path) -> Result<()> {
    let img = crate::infra::image_io::open_rgb8(input)?;
    const MAX_SIDE: u32 = 2048;
    let (w, h) = (img.width(), img.height());
    let longest = w.max(h);
    let img = if longest > MAX_SIDE {
        let scale = MAX_SIDE as f32 / longest as f32;
        let nw = ((w as f32 * scale).round() as u32).max(1);
        let nh = ((h as f32 * scale).round() as u32).max(1);
        image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };
    let jpeg = output
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
        .unwrap_or(false);
    if jpeg {
        let file = std::fs::File::create(output)
            .with_context(|| format!("create {}", output.display()))?;
        let mut writer = std::io::BufWriter::new(file);
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut writer, 95)
            .encode_image(&img)
            .with_context(|| format!("encode {}", output.display()))?;
    } else {
        img.save_with_format(output, image::ImageFormat::Png)
            .with_context(|| format!("save {}", output.display()))?;
    }
    Ok(())
}

/// Quality check over a set: decodable, minimum size, mean brightness.
/// Returns one remark per problem, prefixed with the file name.
pub fn check_photo_quality(photos: &[PathBuf]) -> Result<Vec<String>> {
    let mut remarks = Vec::new();
    for path in photos {
        remarks.extend(check_photo(path)?);
    }
    Ok(remarks)
}

/// Quality remarks for one photo (empty = usable).
pub fn check_photo(path: &Path) -> Result<Vec<String>> {
    let mut remarks = Vec::new();
    let img = crate::infra::image_io::open_rgb8(path)?;
    let name = path.display().to_string();
    if img.width().max(img.height()) < 256 {
        remarks.push(format!("{name}: too small ({}×{})", img.width(), img.height()));
    }
    // Mean brightness over a decimated grid, strided by linear pixel index.
    let pixels = u64::from(img.width()) * u64::from(img.height());
    let step = (pixels / 4096).max(1);
    let mut sum = 0u64;
    let mut n = 0u64;
    for (index, (_, _, px)) in img.enumerate_pixels().enumerate() {
        if (index as u64).is_multiple_of(step) {
            sum += (px[0] as u64 + px[1] as u64 + px[2] as u64) / 3;
            n += 1;
        }
    }
    let mean = sum.checked_div(n).unwrap_or(0);
    if mean < 40 {
        remarks.push(format!("{name}: too dark (mean {mean}/255)"));
    } else if mean > 235 {
        remarks.push(format!("{name}: blown out (mean {mean}/255)"));
    }
    Ok(remarks)
}

/// Generate the canon sheet (~6–8 images, ~$1); each sheet is ArcFace-scored
/// against references before operator approval. M3 (needs keyframe provider).
pub async fn build_canon(
    _slug: &str,
    _tier: crate::config::models::Tier,
) -> Result<Vec<PathBuf>> {
    anyhow::bail!("canon generation needs the image provider (M3); references are the canon until then")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_check_flags_small_and_dark_photos() {
        let dir = tempfile::tempdir().unwrap();
        let small = dir.path().join("small.png");
        image::RgbImage::from_pixel(64, 64, image::Rgb([128, 128, 128]))
            .save(&small)
            .unwrap();
        let dark = dir.path().join("dark.png");
        image::RgbImage::from_pixel(512, 512, image::Rgb([5, 5, 5]))
            .save(&dark)
            .unwrap();
        let fine = dir.path().join("fine.png");
        image::RgbImage::from_pixel(512, 512, image::Rgb([128, 128, 128]))
            .save(&fine)
            .unwrap();

        let remarks = check_photo_quality(&[small, dark, fine.clone()]).unwrap();
        assert!(remarks.iter().any(|remark| remark.contains("too small")));
        assert!(remarks.iter().any(|remark| remark.contains("too dark")));
        assert!(!remarks.iter().any(|remark| remark.contains("fine.png")));
    }

    #[test]
    fn normalize_photo_caps_the_long_side() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.png");
        image::RgbImage::from_pixel(3000, 1500, image::Rgb([10, 20, 30]))
            .save(&input)
            .unwrap();
        let output = dir.path().join("out.png");
        normalize_photo(&input, &output).unwrap();
        let normalized = crate::infra::image_io::open_rgb8(&output).unwrap();
        assert_eq!(normalized.width(), 2048);
        assert_eq!(normalized.height(), 1024);
    }
}
