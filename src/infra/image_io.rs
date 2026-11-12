//! Bounded, orientation-correct image decoding.
//!
//! `image::open` decodes whatever the file claims to be: a 20000×20000 PNG is
//! a few hundred KB on disk and ~1.6 GB in memory. Every decode in this crate
//! goes through `open_limited`, which caps both dimensions and total
//! allocation before a single pixel is read.
//!
//! Orientation: phones store pixels in sensor order and record the intended
//! rotation in EXIF. The decoder does not apply it on its own, and every
//! later stage drops metadata (`normalize_photo` re-encodes without it), so
//! the rotation has to be baked into the pixels here, at the single entry
//! point. Otherwise a portrait shot reaches SCRFD sideways and is reported as
//! "no face".

use std::path::Path;

use anyhow::{Context, Result};
use image::{DynamicImage, ImageDecoder, ImageReader};

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
    reader.limits(limits.clone());

    let mut decoder = reader
        .into_decoder()
        .with_context(|| format!("decode {}", path.display()))?;
    // Set explicitly as well: the bomb guard must not depend on whether
    // `into_decoder` forwards the reader's limits.
    decoder
        .set_limits(limits)
        .with_context(|| format!("decode {}", path.display()))?;
    // Read before `from_decoder` consumes the decoder. A missing or broken
    // EXIF block is not an error: the image is simply taken as upright.
    let orientation = decoder.orientation();
    let mut img = DynamicImage::from_decoder(decoder)
        .with_context(|| format!("decode {}", path.display()))?;
    if let Ok(orientation) = orientation {
        img.apply_orientation(orientation);
    }
    Ok(img)
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

    /// A real JPEG with a hand-built EXIF APP1 segment carrying only the
    /// Orientation tag (0x0112), spliced in right after SOI.
    fn jpeg_with_orientation(width: u32, height: u32, orientation: u16) -> Vec<u8> {
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 90)
            .encode_image(&image::RgbImage::from_pixel(width, height, image::Rgb([90, 120, 150])))
            .unwrap();
        let mut app1 = vec![0xFF, 0xE1, 0x00, 0x22]; // APP1, length 34
        app1.extend_from_slice(b"Exif\0\0");
        app1.extend_from_slice(&[0x4D, 0x4D, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08]); // TIFF header, big-endian
        app1.extend_from_slice(&[0x00, 0x01]); // one IFD entry
        app1.extend_from_slice(&[0x01, 0x12, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01]); // Orientation, SHORT, count 1
        app1.extend_from_slice(&orientation.to_be_bytes());
        app1.extend_from_slice(&[0x00, 0x00]); // value padding
        app1.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // next IFD offset
        jpeg.splice(2..2, app1);
        jpeg
    }

    #[test]
    fn exif_rotation_is_baked_into_the_pixels() {
        let dir = tempfile::tempdir().unwrap();
        // Orientation 6 = "rotate 90° clockwise to display": a 40×20 sensor
        // frame is a 20×40 portrait.
        let rotated = dir.path().join("portrait.jpg");
        std::fs::write(&rotated, jpeg_with_orientation(40, 20, 6)).unwrap();
        let image = open_rgb8(&rotated).unwrap();
        assert_eq!((image.width(), image.height()), (20, 40));

        // Orientation 1 = upright: untouched.
        let upright = dir.path().join("upright.jpg");
        std::fs::write(&upright, jpeg_with_orientation(40, 20, 1)).unwrap();
        let image = open_rgb8(&upright).unwrap();
        assert_eq!((image.width(), image.height()), (40, 20));
    }

    #[test]
    fn missing_file_is_an_error() {
        assert!(open_limited(Path::new("/nonexistent/x.png")).is_err());
    }
}
