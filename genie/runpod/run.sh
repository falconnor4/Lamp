#!/usr/bin/env bash
set -euo pipefail

# Genie pretrain entrypoint for a RunPod pod.
# Override these via the pod's environment variables if desired.
REPO_DIR="${REPO_DIR:-/workspace/Lamp}"
DATA_DIR="${DATA_DIR:-/workspace/genie-data}"
CKPT_DIR="${CKPT_DIR:-/workspace/genie-ckpt}"
CONFIG="${CONFIG:-configs/pretrain-100m-fineweb.yaml}"
STEPS="${STEPS:-16000}"
TOTAL_GB="${TOTAL_GB:-10}"
WORKERS="${WORKERS:-4}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-20}"

cd "$REPO_DIR"
python -m pip install --no-cache-dir -r genie/training/requirements.txt
mkdir -p "$DATA_DIR" "$CKPT_DIR"

echo "=== 1/4 download FineWeb-Edu sample ==="
python genie/training/download_fineweb.py --out "$DATA_DIR/raw" --total-gb "$TOTAL_GB"

echo "=== 2/4 train BPE tokenizer ==="
python genie/training/prepare_data.py \
  --input "$DATA_DIR/raw/tokenizer/tokenizer_train.txt" \
  --vocab-size 32768 --save-tokenizer "$DATA_DIR/tokenizer.json" \
  --out "$DATA_DIR/tok_train_sample.bin" --workers "$WORKERS"

echo "=== 3/4 tokenize shards ==="
python genie/training/prepare_data.py \
  --input "$DATA_DIR/raw/shards" \
  --tokenizer "$DATA_DIR/tokenizer.json" --vocab-size 32768 \
  --out "$DATA_DIR/tokens.bin" --eval-out "$DATA_DIR/eval.bin" \
  --holdout-fraction 0.005 --workers "$WORKERS"

echo "=== 4/4 train ==="
python genie/training/train.py \
  --config "genie/training/$CONFIG" \
  --data "$DATA_DIR/tokens.bin" \
  --eval-data "$DATA_DIR/eval.bin" --eval-every "$EVAL_EVERY" \
  --out "$CKPT_DIR" --steps "$STEPS" \
  --save-every "$SAVE_EVERY" --log-every "$LOG_EVERY" --workers 2

echo "Training complete. Checkpoints + train_log.jsonl in $CKPT_DIR"
