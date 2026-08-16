import torch
import torch.nn as nn
import torch.nn.functional as F


class TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        gamma = weight.abs().mean()
        q = torch.round(torch.clamp(weight / gamma.clamp_min(1e-12), -1.0, 1.0))
        return q * gamma

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Int8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
        return torch.round(x / scale).clamp(-127, 127) * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ternary_quantize(weight):
    gamma = weight.abs().mean()
    q = torch.round(torch.clamp(weight / gamma.clamp_min(1e-12), -1.0, 1.0))
    return q.to(torch.int8), gamma.detach()


def int8_quantize(weight):
    scale = weight.abs().amax().clamp_min(1e-12) / 127.0
    q = torch.round(weight / scale).clamp(-127, 127)
    return q.to(torch.int8), scale.detach()


class TernaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, quantize=True, act_quant=False):
        super().__init__(in_features, out_features, bias)
        self.quantize = quantize
        self.act_quant = act_quant

    def forward(self, x):
        w = TernarySTE.apply(self.weight) if self.quantize else self.weight
        if self.act_quant:
            x = Int8STE.apply(x)
        return F.linear(x, w, self.bias)

    @torch.no_grad()
    def export_quantized(self):
        if self.quantize:
            q, scale = ternary_quantize(self.weight)
        else:
            q, scale = int8_quantize(self.weight)
        return q, scale
