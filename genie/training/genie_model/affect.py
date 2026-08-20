import torch
import torch.nn as nn


class AffectHead(nn.Module):
    def __init__(self, dim, affect_dim):
        super().__init__()
        self.readout = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, affect_dim),
            nn.Tanh(),
        )
        self.proj = nn.Linear(affect_dim, dim)

    def forward(self, pooled):
        return self.readout(pooled)

    def inject(self, context, affect):
        return context + self.proj(affect)


class SurprisePredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def _valid(self, L, block_size, pad_mask, device):
        pos = torch.arange(1, L, device=device)
        boundary = (pos % block_size) == 0
        valid = (~boundary)[None, :]
        return valid & pad_mask[:, 1:] & pad_mask[:, :-1]

    def surprise(self, h, pad_mask, block_size):
        B, L, D = h.shape
        err = (self.proj(h[:, :-1]) - h[:, 1:]).pow(2).mean(-1)
        valid = self._valid(L, block_size, pad_mask, h.device)
        denom = valid.sum(1).clamp_min(1)
        return (err * valid).sum(1) / denom

    def loss(self, h, pad_mask, block_size):
        return self.surprise(h, pad_mask, block_size).mean()