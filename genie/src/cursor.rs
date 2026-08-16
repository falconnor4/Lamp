use serde::{Deserialize, Serialize};

/// LLM cursor state — spatial anchor for the AI's perception
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorState {
    pub x: f64,
    pub y: f64,
    pub zoom: f64,       // 1.0 = normal, 2.0 = 2x zoom around cursor
    pub viewport_w: f64, // visible screen width in logical pixels
    pub viewport_h: f64, // visible screen height in logical pixels
    pub owner: String,   // which AI agent owns this cursor
}

impl CursorState {
    pub fn new(owner: String) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            zoom: 1.0,
            viewport_w: 1920.0,
            viewport_h: 1080.0,
            owner,
        }
    }

    /// Encode cursor position as a positional embedding vector
    pub fn position_encoding(&self) -> Vec<f32> {
        let mut encoding = Vec::with_capacity(16);
        encoding.push(self.x as f32 / self.viewport_w as f32);
        encoding.push(self.y as f32 / self.viewport_h as f32);
        encoding.push(1.0 / self.zoom as f32);
        encoding.push(self.viewport_w as f32);
        encoding.push(self.viewport_h as f32);
        encoding
    }

    /// Move cursor (with bounds clamping)
    pub fn move_to(&mut self, x: f64, y: f64) {
        self.x = x.clamp(0.0, self.viewport_w);
        self.y = y.clamp(0.0, self.viewport_h);
    }

    pub fn zoom_in(&mut self) {
        self.zoom = (self.zoom * 1.25).min(16.0);
    }

    pub fn zoom_out(&mut self) {
        self.zoom = (self.zoom / 1.25).max(0.25);
    }
}