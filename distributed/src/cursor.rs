use anyhow::Result;
use std::collections::HashMap;

/// Cross-device cursor synchronization
/// Allows each device to see every other device's LLM cursors
pub struct CursorSync {
    local_cursor: CursorState,
    peer_cursors: HashMap<String, CursorState>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct CursorState {
    pub node: String,
    pub agent_id: String,        // "genie-0", "genie-1", etc
    pub x: f64,
    pub y: f64,
    pub zoom: f64,
    pub viewport_w: f64,
    pub viewport_h: f64,
    pub screenshot_hash: String, // for change detection
}

impl CursorSync {
    pub fn new(node: &str) -> Self {
        Self {
            local_cursor: CursorState {
                node: node.to_string(),
                agent_id: "genie-0".into(),
                x: 0.0,
                y: 0.0,
                zoom: 1.0,
                viewport_w: 1920.0,
                viewport_h: 1080.0,
                screenshot_hash: String::new(),
            },
            peer_cursors: HashMap::new(),
        }
    }

    pub fn state(&self) -> Vec<u8> {
        serde_json::to_vec(&self.local_cursor).unwrap_or_default()
    }

    pub fn update_peer(&mut self, data: Vec<u8>) -> Result<()> {
        let state: CursorState = serde_json::from_slice(&data)?;
        self.peer_cursors.insert(state.node.clone(), state);
        Ok(())
    }

    pub fn visible_cursors(&self) -> Vec<&CursorState> {
        self.peer_cursors.values().collect()
    }
}