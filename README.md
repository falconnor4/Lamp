# Lamp Linux

**Lamp** is a distributed, AI-native Linux distribution built on NixOS. It is the operating system for **Genie** — a 1-bit multimodal liquid diffusion LLM that can perceive and control your desktop.

## Vision

Lamp reimagines the desktop OS for the age of local AI. Instead of bolging AI onto an existing OS, Lamp is built from the ground up with Genie as a first-class citizen — not an app, but a peer that shares your screen with spatial awareness, its own cursor, and distributed compute across all your devices.

### Principles

- **Everything local.** AI runs on your hardware. Outsourcing intelligence is discouraged.
- **Backwards compatible.** Traditional Linux apps work through XWayland and POSIX shims.
- **Distributed by default.** Your desktop and phone combine into one logical machine.
- **1-bit efficiency.** BitNet b1.58 + diffusion language modeling makes LLM inference CPU-friendly.
- **Fruiger Aero.** Glassmorphism, gradients, and early-2000s optimism.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ ┌─ Lamp ─────────────────────────────────────────┐  │
│ │ ✦ Talk to Genie...         [WiFi] [🔋 87%]    │  │
│ └────────────────────────────────────────────────┘  │
│                          ┌──────────────────────────┐│
│                          │ Genie cursor (spatially  ││
│                          │ aware, zoom-able, has    ││
│                          │ its own viewport)        ││
│                          └──────────────────────────┘│
│ ┌─ DriftWM (infinite tiling) ──────────────────────┐│
│ │                                                   ││
│ │   ┌──────┐ ┌──────┐ ┌──────┐                     ││
│ │   │ term │ │ code │ │ chat │                     ││
│ │   └──────┘ └──────┘ └──────┘                     ││
│ │           ┌──────────┐                            ││
│ │           │ browser  │    ← tiles expand forever  ││
│ │           └──────────┘                            ││
│ └───────────────────────────────────────────────────┘│
│ ┌─ Distributed Mesh ────────────────────────────────┐│
│ │ Phone ←→ Desktop ←→ Laptop (shared cursors,       ││
│ │ combined compute via Lin)                         ││
│ └───────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

### Components

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Base** | NixOS | Declarative, reproducible Linux foundation |
| **Compositor** | DriftWM | Infinite tiling Wayland compositor |
| **Shell** | lamp-shell | Fruiger Aero top bar + system tray |
| **Terminal** | Lamp | Chat with Genie; `/` for classic commands |
| **LLM** | Genie | 1-bit multimodal liquid diffusion model |
| **Sync** | JuiceFS + Garage | Distributed filesystem across devices |
| **Compute** | Lin | Unified CPU/GPU/mesh programming language |
| **Mesh** | lamp-distributed | Peer discovery + cursor sync |

## Genie

Genie is a 1-bit (BitNet b1.58) diffusion language model with liquid neural network temporal processing.

- **1-bit weights** — ternary {-1, 0, +1} for minimal memory
- **Diffusion head** — iterative denoising (dLLM-style) rather than autoregressive
- **Liquid NCP** — closed-form continuous-time cells for temporal awareness
- **Multimodal** — text, vision (screen), and audio input
- **Spatially aware** — has its own cursor, can zoom, sees the screen through its viewport

Genie runs as a systemd service (`genied`) and communicates with the Lamp terminal over local TCP.

The full model spec and training stack live in [`genie/ARCHITECTURE.md`](genie/ARCHITECTURE.md). The trainable PyTorch implementation (ternary QAT backbone, masked-diffusion head, CfC liquid cells, vision/audio encoders, BitNet export) is in `genie/training/`:

```bash
python genie/training/train.py --config genie/training/configs/smoke.yaml --out /tmp/genie-ckpt
python genie/training/export_bitnet.py --ckpt /tmp/genie-ckpt/final.pt --out /tmp/genie-export
```

### Multi-cursor system

Since DriftWM is an infinite tiling compositor, Genie gets its own cursor. When Genie takes a screenshot, it's centered around its cursor position at its current zoom level. This lets Genie:

- Navigate spatially without interfering with the user's cursor
- Zoom in to examine details or zoom out for context
- Multiple AI agents can coexist with their own cursors
- Across the distributed mesh, all devices see each other's cursors

## Lamp terminal

The Lamp terminal is both a terminal emulator and a Genie chat interface. You chat with Genie directly in the top bar. To run a classic terminal command, prefix it with `/`:

```
✦ what's the weather today?     ← talks to Genie
  ✦ I can't see the weather directly, but I can check
     your location and fetch it if you have internet.
/ls -la ~/Documents             ← runs ls as a normal command
  output from ls...
```

## Development

```bash
# Build all Rust packages
./scripts/dev.sh dev

# Build NixOS ISO
./scripts/dev.sh iso

# Run in VM
./scripts/dev.sh vm
```

### Repo structure

```
├── flake.nix                # NixOS flake
├── nixos/
│   ├── configuration.nix    # Base system config
│   └── modules/
│       ├── driftwm.nix      # DriftWM compositor
│       ├── genie.nix        # Genie service
│       ├── terminal.nix     # Lamp terminal
│       ├── fruiger-aero.nix # Theming
│       ├── distributed.nix  # Mesh networking
│       ├── sync.nix         # JuiceFS + Garage sync
│       └── backwards-compat.nix # Legacy support
├── genie/                   # 1-bit LLM (Rust)
├── lamp-term/               # Terminal (Rust)
├── shell/                   # Compositor wrapper (Rust)
├── distributed/             # Mesh layer (Rust)
├── lin/                     # Lin bindings (Rust)
├── nix/packages/            # Nix package definitions
└── scripts/                 # Dev helpers
```

### External dependencies

| Project | Use |
|---------|-----|
| [driftwm](https://github.com/malbiruk/driftwm) | Infinite tiling Wayland compositor |
| [BitNet](https://github.com/microsoft/BitNet) | 1-bit LLM inference framework |
| [dLLM](https://github.com/ZHZisZZ/dllm) | Diffusion language modeling |
| [ncps](https://github.com/mlech26l/ncps) | Liquid neural networks (NCP/LTC/CfC) |
| [Lin](https://github.com/Studio-Todos/Lin) | Unified CPU/GPU programming |
| [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) | Vision-language multimodal backbone |
| [waypipe](https://github.com/deepin-community/waypipe) | Wayland compositor utilities |
| JuiceFS | POSIX-compatible distributed filesystem |
| Garage | S3-compatible object store for sync |

## License

MIT