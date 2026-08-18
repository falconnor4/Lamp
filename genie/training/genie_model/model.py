import torch
import torch.nn as nn

from .backbone import GenieBackbone, block_causal_mask
from .config import GenieConfig
from .diffusion import BlockSampler, MaskedDiffusion
from .liquid import LiquidMemory
from .multimodal import AudioEncoder, VisionEncoder, scatter_features
from .quant import TernaryLinear


class GenieLM(nn.Module):
    def __init__(self, cfg: GenieConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        with torch.no_grad():
            self.tok_emb.weight[cfg.pad_id].zero_()
        self.modality_emb = nn.Embedding(3, cfg.d_model)
        self.backbone = GenieBackbone(cfg)
        self.head = None if cfg.tie_embeddings else TernaryLinear(cfg.d_model, cfg.vocab_size, bias=False, quantize=cfg.ternary_weights)
        self.vision = VisionEncoder(cfg) if cfg.vision_enabled else None
        self.audio = AudioEncoder(cfg) if cfg.audio_enabled else None
        self.memory = LiquidMemory(cfg.d_model)
        self.diffusion = MaskedDiffusion(cfg.mask_id, cfg.diffusion_weight_clamp)
        self.sampler = BlockSampler(self.diffusion, cfg.block_size, cfg.diffusion_steps)

    def embed(self, ids, images=None, audio=None, cursor=None, memory=None):
        emb = self.tok_emb(ids)
        modality = torch.zeros_like(ids)
        if images is not None and self.vision is not None:
            feats = self.vision(images, cursor)
            emb = scatter_features(emb, ids, self.cfg.image_id, feats)
            modality = torch.where(ids == self.cfg.image_id, 1, modality)
        if audio is not None and self.audio is not None:
            feats = self.audio(audio)
            emb = scatter_features(emb, ids, self.cfg.audio_id, feats)
            modality = torch.where(ids == self.cfg.audio_id, 2, modality)
        emb = emb + self.modality_emb(modality)
        if memory is not None:
            emb = self.memory.inject(emb, memory)
        return emb

    def _head(self, h):
        if self.head is None:
            return torch.nn.functional.linear(h, self.tok_emb.weight)
        return self.head(h)

    def logits(self, ids, images=None, audio=None, cursor=None, memory=None, attn_mask=None):
        h = self.backbone(self.embed(ids, images, audio, cursor, memory), attn_mask)
        return self._head(h)

    def forward(self, input_ids, images=None, audio=None, cursor=None, memory=None):
        protect = [self.cfg.pad_id]
        if images is not None:
            protect.append(self.cfg.image_id)
        if audio is not None:
            protect.append(self.cfg.audio_id)
        corrupted, mask, t = self.diffusion.corrupt(input_ids, self.cfg.block_size, protect)
        h = self.backbone(self.embed(corrupted, images, audio, cursor, memory), None)
        loss = self.diffusion.loss(h, self._head, input_ids, mask, t, self.cfg.block_size, self.cfg.loss_chunk)
        return loss, h

    @torch.no_grad()
    def update_memory(self, memory, ids, images=None, audio=None, cursor=None):
        h = self.backbone(self.embed(ids, images, audio, cursor, memory), None)
        context = h.mean(1)
        return self.memory.step(memory, context)

    @torch.no_grad()
    def generate(self, prefix, num_tokens, steps=None, temperature=0.0, memory=None, stop_at_eos=True):
        cfg = self.cfg
        self.sampler.steps = steps or cfg.diffusion_steps
        generated = prefix.clone()
        n_new = 0
        device = prefix.device
        while n_new < num_tokens:
            def block_logits(block):
                ids = torch.cat([generated, block], 1)
                if ids.shape[1] > cfg.max_seq_len:
                    ids = ids[:, -cfg.max_seq_len:]
                mask = block_causal_mask(ids.shape[1], cfg.block_size, device)
                return self.logits(ids, memory=memory, attn_mask=mask)[:, -cfg.block_size:]

            block = self.sampler.sample_block(block_logits, prefix.shape[0], device, temperature)
            generated = torch.cat([generated, block], 1)
            n_new += cfg.block_size
            if stop_at_eos and (block == cfg.eos_id).any():
                break
        return generated[:, prefix.shape[1]:]
