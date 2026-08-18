import argparse
import os

from datasets import load_dataset


def flush(path, buf):
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(buf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--total-gb", type=float, default=10.0)
    ap.add_argument("--tokenizer-train-mb", type=int, default=400)
    ap.add_argument("--shard-mb", type=int, default=256)
    args = ap.parse_args()

    tt_dir = os.path.join(args.out, "tokenizer")
    shard_dir = os.path.join(args.out, "shards")
    os.makedirs(tt_dir, exist_ok=True)
    os.makedirs(shard_dir, exist_ok=True)

    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)

    tt_target = args.tokenizer_train_mb * 1024 * 1024
    shard_target = args.shard_mb * 1024 * 1024
    total_target = int(args.total_gb * 1024 ** 3)

    tt_buf, tt_size, tt_done = [], 0, False
    shard_buf, shard_size, shard_idx, total_bytes = [], 0, 0, 0

    for ex in ds:
        text = ex["text"] + "\n"
        nb = len(text.encode("utf-8"))
        if not tt_done:
            tt_buf.append(text)
            tt_size += nb
            if tt_size >= tt_target:
                flush(os.path.join(tt_dir, "tokenizer_train.txt"), tt_buf)
                tt_buf = []
                tt_done = True
        else:
            shard_buf.append(text)
            shard_size += nb
            total_bytes += nb
            if shard_size >= shard_target:
                flush(os.path.join(shard_dir, f"shard_{shard_idx:05d}.txt"), shard_buf)
                shard_buf = []
                shard_idx += 1
                shard_size = 0
        if total_bytes >= total_target:
            break

    if tt_buf:
        flush(os.path.join(tt_dir, "tokenizer_train.txt"), tt_buf)
    if shard_buf:
        flush(os.path.join(shard_dir, f"shard_{shard_idx:05d}.txt"), shard_buf)

    n_shards = shard_idx + (1 if shard_buf else 0)
    print(f"tokenizer train sample: {tt_size / 2**20:.1f} MiB; shards: {total_bytes / 2**30:.2f} GiB ({n_shards} files)")


if __name__ == "__main__":
    main()
