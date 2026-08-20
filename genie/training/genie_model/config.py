from dataclasses import dataclass, field, asdict


@dataclass
class GenieConfig:
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 64
    mlp_ratio: float = 3.5
    max_seq_len: int = 2048
    block_size: int = 32
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

    ternary_weights: bool = True
    act_quant: bool = False

    liquid_every: int = 0

    vision_enabled: bool = True
    patch_size: int = 16
    image_size: int = 224
    vision_dim: int = 384
    vision_layers: int = 6
    max_frames: int = 4

    audio_enabled: bool = True
    audio_dim: int = 256

    pad_id: int = 0
    mask_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    image_id: int = 5
    audio_id: int = 6

    diffusion_weight_clamp: float = 5.0
    diffusion_steps: int = 8
    loss_chunk: int = 256

    affect_dim: int = 16
    affect_weight: float = 0.1

    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warmup_steps: int = 2000
    total_steps: int = 100000
    grad_clip: float = 1.0
    batch_size: int = 64
    seq_len: int = 2048
    grad_checkpoint: bool = False
    bf16: bool = True

    def validate(self):
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        assert self.max_seq_len % self.block_size == 0, "max_seq_len must be a multiple of block_size"
        assert self.seq_len % self.block_size == 0, "seq_len must be a multiple of block_size"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
