use anyhow::Result;

/// Lamp terminal widget (embedded in the top bar area)
pub struct LampTermWidget;

impl LampTermWidget {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    pub async fn read_line(&mut self) -> Result<String> {
        todo!("Read from input field")
    }

    pub async fn write(&mut self, text: String) -> Result<()> {
        todo!("Write to terminal output")
    }

    pub async fn write_genie_response(&mut self, response: &str) -> Result<()> {
        self.write(format!("✦ {}\n", response)).await
    }
}