use anyhow::Result;
use std::process::Command as StdCommand;

/// DriftWM compositor lifecycle management
pub struct DriftWmHandle;

impl DriftWmHandle {
    pub fn spawn() -> Result<Self> {
        // DriftWM is the Wayland compositor — it handles:
        // - Infinite tiling (dynamic grid with infinite expansion)
        // - LLM cursor management (separate cursor per AI agent)
        // - Multi-monitor + multi-device display mapping
        // - Keyboard shortcuts & input routing
        //
        // This handle ensures driftwm starts and we can
        // communicate with it via the compositor protocol.
        let _child = StdCommand::new("driftwm")
            .spawn()?;

        tracing::info!("DriftWM compositor started");
        Ok(Self)
    }
}