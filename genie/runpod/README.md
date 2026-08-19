# Training Genie on RunPod (~$20 overnight)

Goal: pretrain the 100M text-only Genie model on a FineWeb-Edu sample for ~12h
on a single RTX 4090/3090, landing checkpoints on a persistent volume.

## 0. Push the code

RunPod pulls from GitHub, so commit and push first:

```bash
git add genie/training genie/runpod
git commit -m "Add RunPod pretrain pipeline (FineWeb-Edu, 100M text-only)"
git push origin main
```

## 1. Create a Network Volume (persistent storage)

RunPod → Storage → Network Volume. Create one (e.g. 100 GB) in the same region
as your GPU pod. This keeps `genie-data` and `genie-ckpt` alive when the pod
stops — critical, since pod disks are wiped on termination.

## 2. Create the pod

- **Template / image:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.0-devel-ubuntu22.04`
  (any recent PyTorch CUDA image works; it ships `torch` + `pip`).
- **GPU:** RTX 4090 (24 GB, ~$0.69/hr) or RTX 3090 (~$0.22/hr). 24 GB is plenty
  for the 100M config at batch 64 / seq 2048.
- **Attach the Network Volume at `/workspace`.**
- **Container start command** (fresh clone each start — code only, data/ckpts
  live outside the repo):

```bash
cd /workspace && rm -rf Lamp && git clone https://github.com/falconnor4/Lamp.git Lamp && bash Lamp/genie/runpod/run.sh
```

- Leave all other settings default. Start the pod.

## 3. What the run does

`run.sh` runs four stages, everything landing in `/workspace` (your volume):

1. `download_fineweb.py` streams `HuggingFaceFW/fineweb-edu` (`sample-10BT`) and
   writes ~10 GB of text shards + a 400 MB tokenizer-training sample.
2. `prepare_data.py` trains a 32k BPE on the sample.
3. `prepare_data.py` tokenizes all shards in parallel into `tokens.bin`
   (`uint16` memmap) and holds out 0.5% into `eval.bin`.
4. `train.py` runs 16000 steps (~2.1B tokens) on
   `configs/pretrain-100m-fineweb.yaml`, logging JSONL and saving `stepN.pt` +
   `latest.pt` every 500 steps plus `final.pt`.

Progress is logged to stdout (RunPod's Logs tab) and to
`/workspace/genie-ckpt/train_log.jsonl` (train + eval loss every 500 steps).

## 4. Tuning the run

`run.sh` reads these env vars (set them in the pod's "Environment Variables"):

| Var | Default | Meaning |
| --- | --- | --- |
| `STEPS` | 16000 | total training steps (16000 × 64 × 2048 ≈ 2.1B tokens) |
| `TOTAL_GB` | 10 | GB of text to download |
| `CONFIG` | `configs/pretrain-100m-fineweb.yaml` | model config |
| `EVAL_EVERY` | 500 | steps between eval-loss checks |
| `SAVE_EVERY` | 500 | steps between checkpoints |
| `FORCE_DOWNLOAD` | 0 | set `1` to re-download even if shards exist |
| `FORCE_TOKENIZE` | 0 | set `1` to re-tokenize even if token bins exist |

`run.sh` is idempotent: on a restarted pod it skips download/tokenization if
the artifacts already exist on the volume, so a resume goes straight to
training.

To smoke-test the pipeline first (before committing to 12h), set `STEPS=50`
and `TOTAL_GB=1`.

## 5. Stopping / resuming

- Pods bill per second. **Stop or delete the pod in the morning** — the Network
  Volume keeps your checkpoints.
- To resume an interrupted run, restart the pod (start command re-clones and
  re-runs; data/ckpts persist on the volume) and set `STEPS` higher, or resume
  manually with (pass `--steps` to extend beyond the checkpoint's total):

```bash
python genie/training/train.py --config genie/training/configs/pretrain-100m-fineweb.yaml \
  --data /workspace/genie-data/tokens.bin --resume /workspace/genie-ckpt/latest.pt \
  --out /workspace/genie-ckpt --steps 20000
```

## 6. Export to BitNet

Once you have a checkpoint, export the ternary weights for the Rust `genied`:

```bash
python genie/training/export_bitnet.py --ckpt /workspace/genie-ckpt/final.pt --out /workspace/genie-export
```
