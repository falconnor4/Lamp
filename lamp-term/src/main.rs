mod terminal;
mod command;
mod chat;
mod overlay;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:4210")]
    genie_addr: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let mut app = LampTerminal::new(&args.genie_addr).await?;
    app.run().await?;
    Ok(())
}

pub struct LampTerminal {
    genie_client: genie::ipc::IpcClient,
    term: terminal::LampTermWidget,
}

impl LampTerminal {
    async fn new(genie_addr: &str) -> Result<Self> {
        let client = genie::ipc::IpcClient::connect(genie_addr).await?;
        let term = terminal::LampTermWidget::new()?;
        Ok(Self { genie_client: client, term })
    }

    async fn run(&mut self) -> Result<()> {
        loop {
            let input = self.term.read_line().await?;

            if input.starts_with('/') {
                let output = command::execute(&input[1..]).await?;
                self.term.write(output).await?;
            } else {
                let response = self.genie_client
                    .send(genie::ipc::Request::Chat {
                        messages: vec![genie::ipc::Message {
                            role: "user".into(),
                            content: input,
                        }],
                    }).await?;

                if let genie::ipc::Response::Text { content } = response {
                    self.term.write_genie_response(&content).await?;
                }
            }
        }
    }
}