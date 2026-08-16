mod peer;
mod sync;
mod cursor;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    node_name: String,
    #[arg(long, default_value = "mdns")]
    discovery: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    tracing::info!("Starting distributed mesh node: {}", args.node_name);

    let mesh = peer::Mesh::new(&args.node_name, &args.discovery).await?;

    // Broadcast our cursor state to the mesh
    let cursor_sync = cursor::CursorSync::new(&args.node_name);
    mesh.broadcast(cursor_sync.state()).await?;

    // Listen for peer cursor updates
    loop {
        if let Some(peer_state) = mesh.recv().await? {
            cursor_sync.update_peer(peer_state).await?;
        }
    }
}