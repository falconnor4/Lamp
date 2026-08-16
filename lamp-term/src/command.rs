use anyhow::Result;
use std::process::Command;

/// Executes a system command (everything after the '/' in the Lamp terminal)
pub async fn execute(cmd: &str) -> Result<String> {
    // Split command into program and args
    let parts = shell_words::split(cmd)?;
    if parts.is_empty() {
        return Ok(String::new());
    }

    let output = if cfg!(target_os = "linux") {
        Command::new(&parts[0])
            .args(&parts[1..])
            .output()?
    } else {
        Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .output()?
    };

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    Ok(if stderr.is_empty() { stdout } else { format!("{}{}", stdout, stderr) })
}