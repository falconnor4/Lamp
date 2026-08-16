import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GenieConfig
from .liquid import LiquidScan


class VisionBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class VisionEncoder(nn.Module):
    def __init__(self, cfg: GenieConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.vision_dim
        self.patch = nn.Conv2d(3, d, cfg.patch_size, cfg.patch_size)
        self.grid = cfg.image_size // cfg.patch_size
        self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([VisionBlock(d, heads=max(1, min(8, d // 64))) for _ in range(cfg.vision_layers)])
        self.temporal = LiquidScan(d) if cfg.max_frames > 1 else None
        self.proj = nn.Linear(d, cfg.d_model)
        self.cursor_mlp = nn.Sequential(nn.Linear(4, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model))

    def forward(self, images, cursor=None):
        B, F_, C, H, W = images.shape
        assert H == self.cfg.image_size and W == self.cfg.image_size, "images must match cfg.image_size"
        x = self.patch(images.reshape(B * F_, C, H, W)).flatten(2).transpose(1, 2)
        x = x + self.pos
        for block in self.blocks:
            x = block(x)
        if self.temporal is not None and F_ > 1:
            P = x.shape[1]
            x = x.reshape(B, F_, P, -1).permute(0, 2, 1, 3).reshape(B * P, F_, -1)
            x = self.temporal(x)
            x = x.reshape(B, P, F_, -1).permute(0, 2, 1, 3).reshape(B, F_ * P, -1)
        x = self.proj(x)
        if cursor is not None:
            x = x + self.cursor_mlp(cursor).unsqueeze(1)
        return x


class AudioEncoder(nn.Module):
    def __init__(self, cfg: GenieConfig):
        super().__init__()
        d = cfg.audio_dim
        self.frontend = nn.Sequential(
            nn.Conv1d(1, d // 2, 10, 5), nn.GELU(),
            nn.Conv1d(d // 2, d, 8, 4), nn.GELU(),
            nn.Conv1d(d, d, 4, 2),
        )
        self.temporal = LiquidScan(d)
        self.proj = nn.Linear(d, cfg.d_model)

    def forward(self, wave):
        x = self.frontend(wave.unsqueeze(1) if wave.dim() == 2 else wave)
        x = self.temporal(x.transpose(1, 2))
        return self.proj(x)


def scatter_features(emb, ids, token_id, feats):
    B, L, D = emb.shape
    placeholder = ids == token_id
    counts = placeholder.sum(1)
    assert feats.shape[0] == B and feats.shape[1] == counts.max().item() and counts.min().item() == counts.max().item(), (
        "each sample must contain the same number of placeholder tokens matching the feature count"
    )
    idx = placeholder.nonzero()
    emb[idx[:, 0], idx[:, 1]] = feats.reshape(-1, D).to(emb.dtype)
    return emb
