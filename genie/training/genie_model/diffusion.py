import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt


class MaskedDiffusion:
    def __init__(self, mask_id, weight_clamp=5.0):
        self.mask_id = mask_id
        self.weight_clamp = weight_clamp

    def corrupt(self, input_ids, block_size, protect_ids=()):
        B, L = input_ids.shape
        device = input_ids.device
        num_blocks = L // block_size
        t = torch.rand(B, num_blocks, device=device)
        mask_prob = t.repeat_interleave(block_size, dim=1)
        mask = torch.rand(B, L, device=device) < mask_prob
        for pid in protect_ids:
            mask = mask & (input_ids != pid)
        corrupted = torch.where(mask, torch.full_like(input_ids, self.mask_id), input_ids)
        return corrupted, mask, t

    def loss(self, hidden, head_fn, target, mask, t, block_size, chunk=256):
        B, L, _ = hidden.shape
        weight = (1.0 / (1.0 - t).clamp_min(1e-4)).clamp_max(self.weight_clamp)
        weight = weight.repeat_interleave(block_size, dim=1)
        denom = mask.sum().clamp_min(1.0)

        def chunk_ce(h_c, tgt_c, msk_c, w_c):
            logits = head_fn(h_c)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tgt_c.reshape(-1),
                reduction="none",
            ).reshape(B, -1)
            return (ce * msk_c * w_c).sum()

        total = hidden.new_zeros(())
        for i in range(0, L, chunk):
            total = total + _ckpt(
                chunk_ce,
                hidden[:, i : i + chunk],
                target[:, i : i + chunk],
                mask[:, i : i + chunk],
                weight[:, i : i + chunk],
                use_reentrant=False,
            )
        return total / denom

    @torch.no_grad()
    def reverse_step(self, x, x0, t, s):
        masked = x == self.mask_id
        if s <= 0.0:
            return torch.where(masked, x0, x)
        reveal = masked & (torch.rand(x.shape, device=x.device) < (t - s) / t)
        keep_prob = (1.0 - s) / (1.0 - t) if t < 1.0 else 1.0
        remask = ~masked & (keep_prob < 1.0) & (torch.rand(x.shape, device=x.device) > keep_prob)
        x = torch.where(reveal, x0, x)
        x = torch.where(remask, torch.full_like(x, self.mask_id), x)
        return x


class BlockSampler:
    def __init__(self, diffusion, block_size, steps=8):
        self.diffusion = diffusion
        self.block_size = block_size
        self.steps = steps

    @torch.no_grad()
    def sample_block(self, logits_fn, batch, device, temperature=0.0):
        x = torch.full((batch, self.block_size), self.diffusion.mask_id, dtype=torch.long, device=device)
        for i in range(self.steps):
            t = 1.0 - i / self.steps
            s = 1.0 - (i + 1) / self.steps
            logits = logits_fn(x)
            logits = logits.clone()
            logits[..., self.diffusion.mask_id] = float("-inf")
            if temperature <= 0.0:
                x0 = logits.argmax(-1)
            else:
                x0 = torch.distributions.Categorical(logits=logits / temperature).sample()
            x = self.diffusion.reverse_step(x, x0, t, s)
        return x
