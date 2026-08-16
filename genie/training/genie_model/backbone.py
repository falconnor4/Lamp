import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import GenieConfig
from .liquid import LiquidScan
from .quant import TernaryLinear


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        n = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * n).type_as(x) * self.weight


def precompute_rope(head_dim, max_len, base=10000.0, device=None):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    ang = torch.outer(torch.arange(max_len, device=device).float(), inv)
    return torch.polar(torch.ones_like(ang), ang)


def apply_rope(x, freqs, pos_ids):
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs[pos_ids][None, None]).flatten(-2).type_as(x)


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, H, L, D = x.shape
    return x[:, :, None].expand(B, H, n_rep, L, D).reshape(B, H * n_rep, L, D)


def block_causal_mask(L, block_size, device):
    blk = torch.arange(L, device=device) // block_size
    return blk[None, :] <= blk[:, None]


class Attention(nn.Module):
    def __init__(self, cfg: GenieConfig, quantize, act_quant):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.q = TernaryLinear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False, quantize=quantize, act_quant=act_quant)
        self.k = TernaryLinear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False, quantize=quantize, act_quant=act_quant)
        self.v = TernaryLinear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False, quantize=quantize, act_quant=act_quant)
        self.o = TernaryLinear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False, quantize=quantize, act_quant=act_quant)

    def forward(self, x, freqs, pos_ids, attn_mask):
        B, L, _ = x.shape
        q = self.q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, L, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, L, self.n_kv, self.head_dim).transpose(1, 2)
        q = apply_rope(q, freqs, pos_ids)
        k = apply_rope(k, freqs, pos_ids)
        k = repeat_kv(k, self.n_heads // self.n_kv)
        v = repeat_kv(v, self.n_heads // self.n_kv)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o(out.transpose(1, 2).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, cfg: GenieConfig, quantize, act_quant):
        super().__init__()
        hidden = int(cfg.d_model * cfg.mlp_ratio)
        self.gate = TernaryLinear(cfg.d_model, hidden, bias=False, quantize=quantize, act_quant=act_quant)
        self.up = TernaryLinear(cfg.d_model, hidden, bias=False, quantize=quantize, act_quant=act_quant)
        self.down = TernaryLinear(hidden, cfg.d_model, bias=False, quantize=quantize, act_quant=act_quant)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TernaryBlock(nn.Module):
    def __init__(self, cfg: GenieConfig):
        super().__init__()
        q = cfg.ternary_weights
        a = cfg.act_quant
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, q, a)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg, q, a)

    def forward(self, x, freqs, pos_ids, attn_mask):
        x = x + self.attn(self.attn_norm(x), freqs, pos_ids, attn_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GenieBackbone(nn.Module):
    def __init__(self, cfg: GenieConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([TernaryBlock(cfg) for _ in range(cfg.n_layers)])
        self.liquid = nn.ModuleList(
            [LiquidScan(cfg.d_model) if (cfg.liquid_every and (i + 1) % cfg.liquid_every == 0) else None for i in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.freqs = precompute_rope(cfg.head_dim, cfg.block_size, cfg.rope_base)
        self.register_buffer("pos_ids", torch.arange(cfg.max_seq_len) % cfg.block_size, persistent=False)

    def forward(self, x, attn_mask=None):
        cfg = self.cfg
        L = x.shape[1]
        freqs = self.freqs.to(x.device)
        pos_ids = self.pos_ids[:L].to(x.device)
        if attn_mask is None:
            attn_mask = block_causal_mask(L, cfg.block_size, x.device)
        for block, liquid in zip(self.blocks, self.liquid):
            if cfg.grad_checkpoint and self.training:
                x = checkpoint(block, x, freqs, pos_ids, attn_mask, use_reentrant=False)
            else:
                x = block(x, freqs, pos_ids, attn_mask)
            if liquid is not None:
                x = x + liquid(x)
        return self.norm(x)
