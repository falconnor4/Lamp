use anyhow::Result;
use std::process::Command;

/// Sync layer: JuiceFS mount operations
pub struct Sync;

impl Sync {
    pub async fn mount(endpoint: &str, mount_point: &str) -> Result<()> {
        let output = Command::new("juicefs")
            .args(["mount", endpoint, mount_point])
            .output()?;
        if !output.status.success() {
            anyhow::bail!("JuiceFS mount failed: {}", String::from_utf8_lossy(&output.stderr));
        }
        Ok(())
    }

    pub async fn unmount(mount_point: &str) -> Result<()> {
        Command::new("juicefs")
            .args(["umount", mount_point])
            .output()?;
        Ok(())
    }
}