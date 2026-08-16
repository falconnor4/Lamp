use crate::cursor::CursorState;
use anyhow::Result;

/// Multimodal encoder: text + vision + audio → unified latent
pub struct VisionEncoder;

impl VisionEncoder {
    pub fn encode(screenshot: Vec<u8>, cursor: &CursorState) -> Result<Vec<f32>> {
        // 1. Decode image
        let img = image::load_from_memory(&screenshot)?;
        let img = img.to_rgb8();

        // 2. Crop region around LLM's cursor for spatial awareness
        let region = Self::crop_around_cursor(&img, cursor);

        // 3. Patch embedding (ViT-style)
        let patches = Self::patchify(region);

        // 4. Concatenate cursor position encoding
        let pos_encoding = cursor.position_encoding();
        let mut latent = patches;
        latent.extend(pos_encoding);

        Ok(latent)
    }

    fn crop_around_cursor(_img: &image::RgbImage, _cursor: &CursorState) -> Vec<u8> {
        // Extract viewport centered on LLM cursor
        // Allows zoom in/out for detailed examination
        vec![]
    }

    fn patchify(_region: Vec<u8>) -> Vec<f32> {
        // Split into patches, linear project
        vec![]
    }
}

/// Audio encoder for voice input
pub struct AudioEncoder;

impl AudioEncoder {
    pub fn encode(audio: Vec<u8>) -> Result<Vec<f32>> {
        let reader = hound::WavReader::new(&audio[..])?;
        let samples: Vec<f32> = reader.into_samples::<i16>()
            .filter_map(Result::ok)
            .map(|s| s as f32 / i16::MAX as f32)
            .collect();
        // Mel spectrogram + encoder
        Ok(samples)
    }
}