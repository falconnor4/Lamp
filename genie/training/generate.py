import argparse

import torch

from genie_model import tokenizer as tok_mod
from genie_model.config import GenieConfig
from genie_model.model import GenieLM


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = GenieConfig.from_dict(ckpt["config"])
    model = GenieLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg, ckpt.get("step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt (latest.pt / final.pt / stepN.pt)")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json used during training")
    ap.add_argument("--prompt", default="", help="text prefix to condition on")
    ap.add_argument("--num-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8, help="0.0 = greedy")
    ap.add_argument("--steps", type=int, default=None, help="diffusion denoise steps per block")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, cfg, step = load_model(args.ckpt, device)
    tok = tok_mod.load(args.tokenizer)

    ids = tok.encode(args.prompt).ids if args.prompt else [cfg.bos_id]
    prefix = torch.tensor([ids], dtype=torch.long, device=device)

    use_amp = device.type == "cuda" and cfg.bf16
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        out = model.generate(prefix, args.num_tokens, steps=args.steps, temperature=args.temperature)

    text = tok.decode(out[0].tolist(), skip_special_tokens=True)
    print(f"--- ckpt step {step} | temp {args.temperature} | {args.num_tokens} tokens ---")
    print(args.prompt + text)


if __name__ == "__main__":
    main()
