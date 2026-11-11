//! Bounded image decoding.
//!
//! `image::open` decodes whatever the file claims to be: a 20000×20000 PNG is
//! a few hundred KB on disk and ~1.6 GB in memory. Every decode in this crate
//! goes through `open_limited`, which caps both dimensions and total
//! allocation before a single pixel is read.

use std::path::Path;

use anyhow::{Context, Result};
use image::{DynamicImage, ImageReader};

/// Longest side accepted from any input file.
pub const MAX_SIDE: u32 = 12_000;
/// Hard cap on the decoded buffer (RGBA8 worst case).
pub const MAX_ALLOC_BYTES: u64 = 512 * 1024 * 1024;

pub fn open_limited(path: &Path) -> Result<DynamicImage> {
    let mut reader = ImageReader::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("guess format {}", path.display()))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_SIDE);
    limits.max_image_height = Some(MAX_SIDE);
    limits.max_alloc = Some(MAX_ALLOC_BYTES);
    reader.limits(limits);
    reader
        .decode()
        .with_context(|| format!("decode {}", path.display()))
}

/// Decode and convert to RGB8 in one step (the form every pipeline stage wants).
pub fn open_rgb8(path: &Path) -> Result<image::RgbImage> {
    Ok(open_limited(path)?.to_rgb8())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_an_oversized_image() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge.png");
        // 13000×1 is over the side limit but tiny on disk.
        image::RgbImage::from_pixel(13_000, 1, image::Rgb([0, 0, 0]))
            .save(&path)
            .unwrap();
        let error = open_limited(&path).unwrap_err().to_string();
        assert!(error.contains("decode"), "unexpected error: {error}");
    }

    #[test]
    fn accepts_a_normal_photo() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ok.png");
        image::RgbImage::from_pixel(640, 480, image::Rgb([10, 20, 30]))
            .save(&path)
            .unwrap();
        let image = open_rgb8(&path).unwrap();
        assert_eq!((image.width(), image.height()), (640, 480));
    }

    #[test]
    fn missing_file_is_an_error() {
        assert!(open_limited(Path::new("/nonexistent/x.png")).is_err());
    }
}
