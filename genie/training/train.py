import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from genie_model.config import GenieConfig
from genie_model.data import PackedTokenDataset, SyntheticDataset, collate
from genie_model.model import GenieLM


def lr_lambda(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def evaluate(model, loader, device, use_amp, n_batches):
    model.eval()
    total = 0.0
    seen = 0
    for batch in loader:
        if seen >= n_batches:
            break
        batch = batch.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss, _ = model(batch)
        total += loss.item()
        seen += 1
    model.train()
    return total / max(1, seen)


def save_ckpt(path, model, opt, sched, step, cfg):
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "step": step,
            "config": cfg.to_dict(),
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default=None, help="path to uint16 token .bin (synthetic data if omitted)")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--eval-data", default=None, help="held-out uint16 token .bin for validation loss")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=20)
    args = ap.parse_args()

    cfg = GenieConfig.from_yaml(args.config)
    cfg.validate()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    model = GenieLM(cfg).to(device)

    start_step = 0
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        cfg = GenieConfig.from_dict(ckpt["config"])
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"] + 1
    if args.steps is not None:
        cfg.total_steps = args.steps

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg.warmup_steps, cfg.total_steps))
    if ckpt is not None:
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])

    dataset = PackedTokenDataset(args.data, cfg.seq_len, cfg.pad_id) if args.data else SyntheticDataset(10000, cfg.seq_len, cfg.vocab_size)
    use_cuda = device.type == "cuda"
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=use_cuda,
    )
    if use_cuda and args.workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    eval_loader = None
    if args.eval_data:
        eval_ds = PackedTokenDataset(args.eval_data, cfg.seq_len, cfg.pad_id)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=args.workers,
            drop_last=True,
            pin_memory=use_cuda,
        )

    use_amp = cfg.bf16 and use_cuda
    scaler = torch.amp.GradScaler(enabled=use_amp)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")

    step = start_step
    model.train()
    t0 = time.time()
    while step < cfg.total_steps:
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss, _ = model(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            if step % args.log_every == 0:
                rec = {
                    "step": step,
                    "loss": round(loss.item(), 4),
                    "lr": opt.param_groups[0]["lr"],
                    "tok_s": round(cfg.batch_size * cfg.seq_len * args.log_every / max(1e-6, time.time() - t0)),
                }
                t0 = time.time()
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()

            if eval_loader is not None and step and step % args.eval_every == 0:
                ev = evaluate(model, eval_loader, device, use_amp, args.eval_batches)
                rec = {"step": step, "eval_loss": round(ev, 4)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()

            if step and step % args.save_every == 0:
                save_ckpt(os.path.join(args.out, f"step{step}.pt"), model, opt, sched, step, cfg)
                save_ckpt(os.path.join(args.out, "latest.pt"), model, opt, sched, step, cfg)

            step += 1
            if step >= cfg.total_steps:
                break

    save_ckpt(os.path.join(args.out, "final.pt"), model, opt, sched, step, cfg)
    save_ckpt(os.path.join(args.out, "latest.pt"), model, opt, sched, step, cfg)
    log.close()


if __name__ == "__main__":
    main()
