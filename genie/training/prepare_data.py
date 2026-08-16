import argparse
import glob
import os

import numpy as np

from genie_model.tokenizer import load, train_bpe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="text file, directory of .txt, or glob")
    ap.add_argument("--tokenizer", default=None, help="existing tokenizer json; trained if omitted")
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--out", required=True, help="output .bin path (uint16 token ids)")
    ap.add_argument("--save-tokenizer", default="tokenizer.json")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "**", "*.txt"), recursive=True))
    elif any(c in args.input for c in "*?["):
        files = sorted(glob.glob(args.input))
    else:
        files = [args.input]
    assert files, f"no input files matched {args.input}"

    if args.tokenizer:
        tok = load(args.tokenizer)
    else:
        tok = train_bpe(files, args.vocab_size, args.save_tokenizer)
        print(f"trained tokenizer -> {args.save_tokenizer}")

    ids = []
    for path in files:
        with open(path, encoding="utf-8", errors="ignore") as f:
            ids.extend(tok.encode(f.read()).ids)
    arr = np.asarray(ids, dtype=np.uint16)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    arr.tofile(args.out)
    print(f"{len(arr)} tokens -> {args.out}")


if __name__ == "__main__":
    main()
