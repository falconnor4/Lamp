use anyhow::Result;

/// System tray — Wi-Fi, battery, time, notifications
pub struct SystemTray;

impl SystemTray {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    pub async fn get_wifi_status() -> Result<String> {
        // Read from NetworkManager
        Ok("WiFi".into())
    }

    pub async fn get_battery() -> Result<String> {
        // Read from /sys/class/power_supply/
        Ok("🔋 87%".into())
    }

    pub async fn get_time() -> Result<String> {
        Ok("🕐 14:23".into())
    }

    pub async fn render() -> Result<String> {
        let wifi = Self::get_wifi_status().await?;
        let battery = Self::get_battery().await?;
        let time = Self::get_time().await?;
        Ok(format!("[{}] [{}] [{}]", wifi, battery, time))
    }
}