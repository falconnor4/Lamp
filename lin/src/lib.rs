//! Lin — unified CPU/GPU programming language bindings
//!
//! Lin provides a single-threaded-async model that maps seamlessly
//! onto both CPU cores and GPU shader units, enabling massive
//! parallelism without traditional GPU programming complexity.
//!
//! Architecture:
//! - Kernel language compiles to SPIR-V (GPU) and native (CPU)
//! - Automatic work-group distribution across available devices
//! - Zero-overhead shared memory between CPU and GPU heaps
//! - All devices on the mesh contribute compute (distributed compute)

/// A Lin kernel — runs on both CPU and GPU transparently
pub struct Kernel {
    spirv: Vec<u32>,
    native: Vec<u8>,
}

impl Kernel {
    pub fn compile(source: &str) -> Self {
        // Compile Lin source to both SPIR-V and native
        todo!("Lin compiler")
    }

    pub fn dispatch(&self, global_size: (u64, u64, u64)) -> ComputeStream {
        todo!("Dispatch kernel across all available devices")
    }
}

/// A stream of compute work, distributed across CPU + GPU + mesh peers
pub struct ComputeStream;

impl ComputeStream {
    pub async fn collect<T: Send>(self) -> Vec<T> {
        todo!("Collect results from all devices")
    }
}

/// Device abstraction — a CPU core, GPU, or remote peer
pub enum Device {
    Cpu { cores: u32 },
    Gpu { name: String, compute_units: u32 },
    Remote { node: String, address: String },
}

/// Automatic device discovery
pub fn available_devices() -> Vec<Device> {
    let mut devices = vec![];

    // Local CPU cores
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    devices.push(Device::Cpu { cores: cores as u32 });

    // GPU via Vulkan
    // TODO: enumerate Vulkan physical devices

    // Remote peers via distributed mesh
    // TODO: query mesh for available compute

    devices
}