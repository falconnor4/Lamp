import torch

from genie_model.config import GenieConfig
from genie_model.model import GenieLM


def smoke_config():
    return GenieConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        mlp_ratio=2.0,
        max_seq_len=32,
        block_size=8,
        seq_len=32,
        vision_dim=32,
        vision_layers=1,
        image_size=32,
        patch_size=8,
        max_frames=2,
        audio_dim=32,
        diffusion_steps=3,
    )


def test_forward_backward():
    cfg = smoke_config()
    model = GenieLM(cfg)
    ids = torch.randint(10, cfg.vocab_size, (2, cfg.seq_len))
    loss, logits = model(ids)
    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
    loss.backward()
    grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_generate():
    cfg = smoke_config()
    model = GenieLM(cfg).eval()
    prefix = torch.randint(10, cfg.vocab_size, (1, 8))
    out = model.generate(prefix, num_tokens=16, stop_at_eos=False)
    assert out.shape[1] >= 16
    assert (out != cfg.mask_id).all()


def test_multimodal_forward():
    cfg = smoke_config()
    model = GenieLM(cfg)
    n_img = (cfg.image_size // cfg.patch_size) ** 2 * cfg.max_frames
    ids = torch.randint(10, cfg.vocab_size, (2, cfg.seq_len))
    ids[:, :n_img] = cfg.image_id
    images = torch.randn(2, cfg.max_frames, 3, cfg.image_size, cfg.image_size)
    cursor = torch.rand(2, 4)
    loss, _ = model(ids, images=images, cursor=cursor)
    loss.backward()


def test_memory_loop():
    cfg = smoke_config()
    model = GenieLM(cfg).eval()
    h = model.memory.zero_state(1, "cpu")
    ids = torch.randint(10, cfg.vocab_size, (1, cfg.seq_len))
    h2 = model.update_memory(h, ids)
    assert h2.shape == h.shape
    loss, _ = model(ids, memory=h2)
    assert torch.isfinite(loss)


def test_ternary_export():
    cfg = smoke_config()
    model = GenieLM(cfg)
    lin = model.backbone.blocks[0].attn.q
    q, gamma = lin.export_quantized()
    assert set(q.unique().tolist()) <= {-1, 0, 1}
    assert gamma.item() > 0
