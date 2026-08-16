mod bar;
mod tray;
mod compositor;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "mdns")]
    discovery: String,
    #[arg(long)]
    node_name: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let _args = Args::parse();

    tracing::info!("Starting lamp-shell");

    // DriftWM is the compositor — lamp-shell wraps it with
    // our custom top bar and service integration
    let compositor = compositor::DriftWmHandle::spawn()?;
    let mut top_bar = bar::TopBar::new()?;

    loop {
        top_bar.render().await?;
        tokio::time::sleep(tokio::time::Duration::from_millis(16)).await; // ~60fps
    }
}