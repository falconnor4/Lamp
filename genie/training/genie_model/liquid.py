import torch
import torch.nn as nn


class CfC(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.synapse = nn.Linear(input_dim + hidden_dim, hidden_dim * 2)

    def forward(self, x, h):
        ff1, ff2 = self.synapse(torch.cat([x, h], -1)).chunk(2, -1)
        return torch.tanh(ff1) * (1 - torch.sigmoid(ff2)) + h * torch.sigmoid(ff2)


class LiquidScan(nn.Module):
    def __init__(self, dim, bidirectional=False):
        super().__init__()
        self.fwd = CfC(dim, dim)
        self.bwd = CfC(dim, dim) if bidirectional else None
        self.merge = nn.Linear(dim * 2, dim) if bidirectional else None
        self.norm = nn.LayerNorm(dim)

    def _scan(self, cell, x, reverse):
        B, T, D = x.shape
        h = x.new_zeros(B, D)
        steps = range(T - 1, -1, -1) if reverse else range(T)
        out = []
        for t in steps:
            h = cell(x[:, t], h)
            out.append(h)
        if reverse:
            out = out[::-1]
        return torch.stack(out, 1)

    def forward(self, x):
        y = self._scan(self.fwd, x, reverse=False)
        if self.bwd is not None:
            y = self.merge(torch.cat([y, self._scan(self.bwd, x, reverse=True)], -1))
        return self.norm(y)


class LiquidMemory(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cell = CfC(dim * 2, dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def step(self, h, context):
        return self.cell(torch.cat([context, h], -1), h)

    def inject(self, emb, h):
        return emb + self.gate(h).unsqueeze(1) * h.unsqueeze(1)

    def zero_state(self, batch, device):
        return torch.zeros(batch, self.cell.hidden_dim, device=device)
