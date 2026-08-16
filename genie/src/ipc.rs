use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use anyhow::Result;

/// IPC protocol between the Lamp terminal and Genie
#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    Chat { messages: Vec<Message> },
    Act { action: String },
    Screen { screenshot: Vec<u8>, cursor: crate::cursor::CursorState },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    Text { content: String },
    Action { plan: String },
    ScreenAnalysis { analysis: String },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,   // "user", "genie", "system"
    pub content: String,
}

pub struct IpcServer {
    listener: TcpListener,
}

impl IpcServer {
    pub async fn bind(addr: &str) -> Result<Self> {
        let listener = TcpListener::bind(addr).await?;
        tracing::info!("IPC server listening on {}", addr);
        Ok(Self { listener })
    }

    pub async fn accept(&self) -> Result<Request> {
        let (stream, _) = self.listener.accept().await?;
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        Ok(serde_json::from_str(&line)?)
    }

    pub async fn respond(&self, response: Response) -> Result<()> {
        let json = serde_json::to_string(&response)?;
        // TODO: track active connections
        Ok(())
    }
}

pub struct IpcClient {
    stream: TcpStream,
}

impl IpcClient {
    pub async fn connect(addr: &str) -> Result<Self> {
        let stream = TcpStream::connect(addr).await?;
        Ok(Self { stream })
    }

    pub async fn send(&mut self, request: Request) -> Result<Response> {
        let mut payload = serde_json::to_string(&request)?;
        payload.push('\n');
        self.stream.write_all(payload.as_bytes()).await?;
        let mut reader = BufReader::new(&mut self.stream);
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        Ok(serde_json::from_str(&line)?)
    }
}