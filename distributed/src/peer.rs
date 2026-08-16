use anyhow::Result;
use std::collections::HashMap;
use tokio::net::UdpSocket;

/// P2P mesh network for distributed Lamp nodes
pub struct Mesh {
    node_name: String,
    socket: UdpSocket,
    peers: HashMap<String, PeerInfo>,
}

struct PeerInfo {
    addr: std::net::SocketAddr,
    last_seen: std::time::Instant,
}

impl Mesh {
    pub async fn new(node_name: &str, discovery: &str) -> Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:4211").await?;
        socket.set_broadcast(true)?;

        match discovery {
            "mdns" => Self::discover_mdns(node_name).await?,
            "dht" => Self::discover_dht(node_name).await?,
            _ => {} // manual
        }

        Ok(Self {
            node_name: node_name.to_string(),
            socket,
            peers: HashMap::new(),
        })
    }

    async fn discover_mdns(_name: &str) -> Result<()> {
        // mDNS service _lamp._tcp
        Ok(())
    }

    async fn discover_dht(_name: &str) -> Result<()> {
        // Kademlia DHT for WAN peer discovery
        Ok(())
    }

    pub async fn broadcast(&self, data: Vec<u8>) -> Result<()> {
        self.socket.send_to(&data, "255.255.255.255:4211").await?;
        Ok(())
    }

    pub async fn recv(&self) -> Result<Option<Vec<u8>>> {
        let mut buf = [0u8; 4096];
        match self.socket.recv_from(&mut buf).await {
            Ok((len, addr)) => {
                self.register_peer(addr);
                Ok(Some(buf[..len].to_vec()))
            }
            Err(_) => Ok(None),
        }
    }

    fn register_peer(&mut self, addr: std::net::SocketAddr) {
        self.peers.insert(addr.to_string(), PeerInfo {
            addr,
            last_seen: std::time::Instant::now(),
        });
    }
}