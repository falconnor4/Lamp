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
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
FORCE_TOKENIZE="${FORCE_TOKENIZE:-0}"

# Reduce CUDA allocator fragmentation (recommended by torch OOM messages).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$REPO_DIR"
python -m pip install --no-cache-dir -r genie/training/requirements.txt
mkdir -p "$DATA_DIR" "$CKPT_DIR"

RAW_SHARDS="$DATA_DIR/raw/shards"
TOKENIZER="$DATA_DIR/tokenizer.json"
TOKENS="$DATA_DIR/tokens.bin"
EVAL_BIN="$DATA_DIR/eval.bin"

echo "=== 1/4 download FineWeb-Edu sample ==="
if [ "$FORCE_DOWNLOAD" = "1" ] || [ ! -d "$RAW_SHARDS" ] || [ -z "$(ls -A "$RAW_SHARDS" 2>/dev/null)" ]; then
  python genie/training/download_fineweb.py --out "$DATA_DIR/raw" --total-gb "$TOTAL_GB"
else
  echo "shards already present ($(ls "$RAW_SHARDS" | wc -l) files); skipping download"
fi

echo "=== 2/4 train BPE tokenizer ==="
if [ "$FORCE_TOKENIZE" = "1" ] || [ ! -f "$TOKENIZER" ]; then
  python genie/training/prepare_data.py \
    --input "$DATA_DIR/raw/tokenizer/tokenizer_train.txt" \
    --vocab-size 32768 --save-tokenizer "$TOKENIZER" \
    --out "$DATA_DIR/tok_train_sample.bin" --workers "$WORKERS"
else
  echo "tokenizer already present; skipping BPE training"
fi

echo "=== 3/4 tokenize shards ==="
if [ "$FORCE_TOKENIZE" = "1" ] || [ ! -f "$TOKENS" ] || [ ! -f "$EVAL_BIN" ]; then
  python genie/training/prepare_data.py \
    --input "$RAW_SHARDS" \
    --tokenizer "$TOKENIZER" --vocab-size 32768 \
    --out "$TOKENS" --eval-out "$EVAL_BIN" \
    --holdout-fraction 0.005 --workers "$WORKERS"
else
  echo "token bins already present; skipping tokenization"
fi

echo "=== 4/4 train ==="
python genie/training/train.py \
  --config "genie/training/$CONFIG" \
  --data "$TOKENS" \
  --eval-data "$EVAL_BIN" --eval-every "$EVAL_EVERY" \
  --out "$CKPT_DIR" --steps "$STEPS" \
  --save-every "$SAVE_EVERY" --log-every "$LOG_EVERY" --workers 2

echo "Training complete. Checkpoints + train_log.jsonl in $CKPT_DIR"
