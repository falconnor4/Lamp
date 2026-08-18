import argparse
import glob
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from genie_model.tokenizer import load, train_bpe


CHUNK_CHARS = 1_000_000


def tokenize_file(path, tokenizer_path, tmp_dir):
    tok = load(tokenizer_path)
    part = os.path.join(tmp_dir, os.path.basename(path) + ".part.bin")
    buf = []
    buf_chars = 0
    with open(path, encoding="utf-8", errors="ignore") as f, open(part, "wb") as out:
        for line in f:
            buf.append(line)
            buf_chars += len(line)
            if buf_chars >= CHUNK_CHARS:
                np.asarray(tok.encode("".join(buf)).ids, dtype=np.uint16).tofile(out)
                buf = []
                buf_chars = 0
        if buf:
            np.asarray(tok.encode("".join(buf)).ids, dtype=np.uint16).tofile(out)
    return part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="text file, directory of .txt, or glob")
    ap.add_argument("--tokenizer", default=None, help="existing tokenizer json; trained if omitted")
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--out", required=True, help="output .bin path (uint16 token ids)")
    ap.add_argument("--save-tokenizer", default="tokenizer.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--holdout-fraction", type=float, default=0.0, help="fraction of tokens held out for eval")
    ap.add_argument("--eval-out", default=None, help="optional eval .bin path (requires --holdout-fraction)")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "**", "*.txt"), recursive=True))
    elif any(c in args.input for c in "*?["):
        files = sorted(glob.glob(args.input))
    else:
        files = [args.input]
    assert files, f"no input files matched {args.input}"

    if args.tokenizer:
        tok_path = args.tokenizer
    else:
        train_bpe(files, args.vocab_size, args.save_tokenizer)
        tok_path = args.save_tokenizer
        print(f"trained tokenizer -> {args.save_tokenizer}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, ".parts")
    os.makedirs(tmp_dir, exist_ok=True)

    worker = partial(tokenize_file, tokenizer_path=tok_path, tmp_dir=tmp_dir)
    if len(files) == 1 or args.workers <= 1:
        parts = [worker(f) for f in files]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            parts = list(ex.map(worker, files))

    full = os.path.join(tmp_dir, "full.bin")
    with open(full, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
            os.remove(p)

    total = os.path.getsize(full) // 2
    if args.holdout_fraction > 0 and args.eval_out:
        eval_n = max(1, int(total * args.holdout_fraction))
        train_n = total - eval_n
        m = np.memmap(full, dtype=np.uint16, mode="r")
        m[:train_n].tofile(args.out)
        m[train_n:].tofile(args.eval_out)
        del m
        print(f"{train_n} train tokens -> {args.out}, {eval_n} eval tokens -> {args.eval_out}")
    else:
        shutil.move(full, args.out)
        print(f"{total} tokens -> {args.out}")
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
