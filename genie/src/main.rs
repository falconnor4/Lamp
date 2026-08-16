mod model;
mod liquid;
mod multimodal;
mod cursor;
mod ipc;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:4210")]
    listen: String,
    #[arg(long, default_value = "/var/lib/genie/models")]
    model_dir: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    tracing::info!("Genie daemon starting on {}", args.listen);

    let model = model::GenieModel::load(&args.model_dir).await?;
    let ipc = ipc::IpcServer::bind(&args.listen).await?;

    loop {
        let request = ipc.accept().await?;
        let response = handle_request(request, &model).await?;
        ipc.respond(response).await?;
    }
}

async fn handle_request(request: ipc::Request, model: &model::GenieModel) -> Result<ipc::Response> {
    match request {
        ipc::Request::Chat { messages } => {
            let output = model.generate(messages).await?;
            Ok(ipc::Response::Text { content: output })
        }
        ipc::Request::Act { action } => {
            let plan = model.plan_action(action).await?;
            Ok(ipc::Response::Action { plan })
        }
        ipc::Request::Screen { screenshot, cursor } => {
            let analysis = model.analyze_screen(screenshot, cursor).await?;
            Ok(ipc::Response::ScreenAnalysis { analysis })
        }
    }
}