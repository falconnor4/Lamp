import argparse
import json
import os

import torch
from safetensors.torch import save_file

from genie_model.config import GenieConfig
from genie_model.model import GenieLM
from genie_model.quant import int8_quantize


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="export")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = GenieConfig.from_dict(ckpt["config"])
    model = GenieLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tensors = {}
    for name, module in model.named_modules():
        if hasattr(module, "export_quantized") and isinstance(module, torch.nn.Linear):
            q, scale = module.export_quantized()
            tensors[f"{name}.weight"] = q.contiguous()
            tensors[f"{name}.scale"] = scale.reshape(1)
            if module.bias is not None:
                tensors[f"{name}.bias"] = module.bias.contiguous()

    q, scale = int8_quantize(model.tok_emb.weight)
    tensors["tok_emb.weight"] = q.contiguous()
    tensors["tok_emb.scale"] = scale.reshape(1)
    if model.head is None:
        tensors["head.weight"] = tensors["tok_emb.weight"].clone()
        tensors["head.scale"] = tensors["tok_emb.scale"].clone()

    os.makedirs(args.out, exist_ok=True)
    save_file(tensors, os.path.join(args.out, "genie.safetensors"))

    manifest = cfg.to_dict()
    manifest["quant"] = "ternary1.58"
    manifest["n_params"] = sum(p.numel() for p in model.parameters())
    with open(os.path.join(args.out, "genie.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"exported {len(tensors)} tensors -> {args.out}/genie.safetensors")


if __name__ == "__main__":
    main()
