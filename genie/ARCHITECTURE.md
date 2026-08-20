# Genie Architecture

Genie is a **1-bit multimodal liquid diffusion LLM**. It combines four ideas into one
trainable model:

1. **1-bit weights** — BitNet b1.58-style ternary weights `{-1, 0, +1}` trained from
   scratch with quantization-aware training (straight-through estimator).
2. **Diffusion language modeling** — masked discrete diffusion (MDLM-style) with a
   block-diffusion structure (BD3-LM-style), giving parallel decoding and infilling.
3. **Liquid dynamics** — CfC (closed-form continuous-time) liquid cells providing
   temporal processing in the multimodal encoders and a persistent recurrent memory
   for the agent loop.
4. **Multimodality + spatial awareness** — vision and audio encoders projecting into
   the language backbone, conditioned on Genie's cursor state (position + zoom) so
   the model knows *where* it is looking.

Training lives in `genie/training/` (PyTorch). Inference is exported to ternary
weights consumed by the Rust `genied` daemon (and later microsoft/BitNet kernels).

```
                        ┌────────────────────────────────────────────┐
 screenshot ───────────►│ VisionEncoder (ViT + CfC temporal scan)    │──┐
 (F frames, cursor-     │   + CursorEmbedding(x, y, zoom)            │  │
  centered crops)       └────────────────────────────────────────────┘  │
                                                                        ▼
 waveform ─────────────►┌ AudioEncoder (conv frontend + CfC) ────► GenieBackbone
                        └────────────────────────────────────┐    N × TernaryBlock
                                                             │    (ternary attn + SwiGLU,
 text tokens ──► Embedding + modality emb + memory gate ─────┘     block-causal mask,
                                                             ▲     RoPE within blocks)
 LiquidMemory (CfC state h, persistent across agent steps) ──┘           │
                                                                         ▼
                                                              diffusion loss (train)
                                                              block sampler (inference)
```

## 1. Tokenizer and special tokens

BPE, vocab 32,768 (configurable), byte-fallback. Special tokens:

| Token        | Role                                                        |
| ------------ | ----------------------------------------------------------- |
| `<\|pad\|>`  | padding                                                     |
| `<\|mask\|>` | diffusion mask state (the "noise" of masked diffusion)      |
| `<\|bos\|>` / `<\|eos\|>` | sequence boundaries                            |
| `<\|image\|>`| placeholder span replaced by vision features                |
| `<\|audio\|>`| placeholder span replaced by audio features                 |
| `<\|cursor\|>`| marks cursor telemetry tokens (agent context)              |
| `<\|state\|>`| reserved for liquid memory readout                          |
| `<\|action\|>`| reserved for the future action head (click/type/move)     |

## 2. Backbone: ternary transformer

`genie_model/backbone.py`

- Pre-norm transformer with RMSNorm.
- Every `nn.Linear` in the backbone is a `TernaryLinear`: in the forward pass the
  weight is quantized to ternary with the BitNet b1.58 rule

  ```
  γ = mean(|W|)                    (per-tensor abs-mean scale)
  W_q = round(clamp(W / γ, -1, 1)) ∈ {-1, 0, +1}
  W_eff = γ · W_q
  ```

  Gradients flow through the quantizer via STE (straight-through estimator).
  Training is **1-bit from step 0** (QAT from scratch, per the BitNet b1.58 recipe);
  there is no full-precision pretrain phase.
- Activations optionally quantized to int8 (per-token abs-max, STE) via
  `act_quant` — mirrors the 2-bit activation target of BitNet at inference.
- Grouped-query attention (configurable `n_kv_heads`), RoPE applied with
  **within-block position ids** (`pos % block_size`).
- SwiGLU MLP (`mlp_ratio` configurable).
- Embedding table and LM head stay high precision during training (head tied to the
  embedding by default) and are quantized to int8 only at export.
- Optional `LiquidScan` residual adapter every `liquid_every` blocks (default off;
  see §4 for where liquid dynamics live by default).

### Block-causal attention mask

The sequence of length `L` is split into contiguous blocks of `block_size`. Position
`i` attends to position `j` iff `block(j) <= block(i)`: fully bidirectional *inside*
a block, causal *across* blocks. This single mask supports both training and
block-by-block generation.

## 3. Diffusion objective (masked, block-structured)

`genie_model/diffusion.py`

Forward noising: sample a noise level `t_b ~ U(0, 1)` independently per block `b`;
each token in block `b` is replaced by `<|mask|>` independently with probability
`t_b`. The model predicts the original token at every masked position.

Loss (MDLM importance-weighted ELBO estimator):

```
L = E_{t, mask} [ (1 / (1 - t)) · CE(pred, target) ]      (averaged over masked positions,
                                                            weight clamped, default 5.0)
```

Sampling (reverse process, exact for masking diffusion), per block with `K` steps
(default 8), noise levels `t = 1 - i/K → s = 1 - (i+1)/K`:

- masked position: revealed to the sampled prediction with probability `(t - s) / t`
- unmasked position: re-masked with probability `1 - (1 - s) / (1 - t)`
- final step (`s = 0`): all remaining masks resolved.

Generation is **block diffusion**: decode one `block_size`-wide block at a time,
each block conditioned on all previously decoded (clean) blocks. Consequences:

- ~`block_size / K`× fewer forward passes than autoregressive decoding,
- arbitrary-length streaming (agent can keep appending blocks),
- free infilling/editing (mask any span, denoise it).

## 4. Liquid components (CfC)

`genie_model/liquid.py`

Closed-form continuous (CfC) cell, per Hasani et al. 2022:

```
z   = [x_t ; h_{t-1}]
ff1 = tanh(W_1 z + b_1)          # candidate update
ff2 = sigmoid(W_2 z + b_2)       # forget gate
h_t = ff1 ⊙ (1 - ff2) + h_{t-1} ⊙ ff2
```

Placement (chosen so recurrence never breaks the diffusion objective):

1. **Vision temporal scan** — CfC scan across frames of a screenshot/video clip
   inside the vision encoder.
2. **Audio temporal scan** — CfC scan across time in the audio encoder.
3. **LiquidMemory** — a persistent state `h ∈ R^d` held by the *daemon*, not the
   training batch. After each agent step the pooled context updates `h` via a CfC
   step; during the next forward pass `h` is injected into the backbone through a
   learned gate added to all embeddings. This is Genie's continuous memory across
   interactions — liquid dynamics in the agent loop, permutation-safe diffusion in
   the backbone.
4. Optional backbone adapter (`liquid_every > 0`) for experiments.

## 5. Multimodal encoders

`genie_model/multimodal.py`

**Vision.** `patch_size`-sized conv patch embedding → learned 2-D positional
embeddings → small ViT stack (full precision) → optional CfC temporal scan across
frames → linear projection to `d_model`. Encoded tokens replace contiguous
`<|image|>` placeholder spans in the embedding stream.

**Cursor conditioning.** The cursor state `(x, y, zoom, aspect)` — normalized
coordinates of Genie's own cursor in the infinite canvas, matching
`genie/src/cursor.rs` — is embedded by an MLP and **added to every vision token**
of that screenshot. The model therefore learns screenshots as "what I see *here*,
at *this* zoom", which is the basis of its spatial awareness.

**Audio.** Learned 1-D conv frontend (stride-downsampled raw waveform) → CfC
temporal scan → projection. Tokens replace `<|audio|>` spans.

A learned modality embedding (text / image / audio) is added per position.

## 6. Model sizes

| Config          | d_model | layers | heads | kv heads | head_dim | seq | block | ≈ params |
| --------------- | ------- | ------ | ----- | -------- | -------- | ---- | ----- | -------- |
| `smoke.yaml`    | 128     | 2      | 4     | 2        | 32       | 64   | 8     | ~1M      |
| `base-100m.yaml`| 768     | 12     | 12    | 4        | 64       | 2048 | 32    | ~130M    |
| `base-1b.yaml`  | 2048    | 24     | 16    | 4        | 128      | 4096 | 64    | ~1.1B    |

Ternary weights ⇒ ~`log2(3) ≈ 1.58` bits/weight at inference: a 1.1B-param Genie
fits in ~220 MB of weights.

## 7. Training recipe

`genie/training/train.py`

- AdamW (β = 0.9/0.95, wd 0.1), cosine schedule with linear warmup, grad clip 1.0.
- bf16 autocast on CUDA; fp32 on CPU. Gradient checkpointing optional.
- QAT ternary quantization active from step 0.
- Data: tokenized uint16 memmap files packed into fixed-length sequences
  (`prepare_data.py` builds them from text corpora with the trained BPE).
- Multimodal batches: image/audio tensors + placeholder spans (dataset stub in
  `data.py`, wire real screenshot/audio capture later).
- Checkpoints: `model.pt` + config; `export_bitnet.py` converts to the ternary
  serving format.

## 8. Export → Rust daemon / BitNet

`genie/training/export_bitnet.py` produces `genie.safetensors` + `genie.json`:

- every `TernaryLinear` → `int8` ternary tensor + per-tensor `fp32` scale γ
- embedding/head → int8 abs-max quantized
- config manifest (dims, block size, special token ids)

The Rust `genied` (candle) loads this directly; the layout matches
microsoft/BitNet's ternary expectations so BitNet kernels can be dropped in for
fast CPU/ARM inference on both the desktop and the phone.

## 9. Affect readout (predictive coding)

`genie_model/affect.py`

Two auxiliary modules give Genie a low-dimensional self-signal, in the spirit of
brain-encoding models like TRIBE v2 (which predict internal neural responses
rather than only the next token). Both live *after* the backbone — readouts on
the final hidden states — so they never add recurrence inside the diffusion
trunk and never break the masked-diffusion objective (§3).

- **`SurprisePredictor`** — a first-order latent predictor (`d → d`, bias-free
  linear) that predicts `h[t]` from `h[t-1]` *within* blocks (predictions are
  dropped at block boundaries). Its mean-squared error is the "surprise"
  signal. During training it contributes `affect_weight · surprise` to the
  loss, biasing the ternary representation toward predictability — the
  predictive-coding prior.
- **`AffectHead`** — maps the mean-pooled hidden state to a low-dim vector in
  `[-1, 1]` (`affect_dim`, default 16), the valence/arousal/salience readout.
  Its `proj` maps affect back into `d_model` so it can be injected into the
  agent-loop `LiquidMemory` context via `update_memory(..., affect=...)`.

Config knobs: `affect_dim` (0 disables) and `affect_weight`. Both modules are
full-precision readouts and are not part of the ternary BitNet export.

## 10. Roadmap

1. **Now:** text pretraining on packed corpora (100M config first), validate
   diffusion loss curves and block sampling quality.
2. Cursor-centered screenshot pretraining (contrastive-free: masked prediction of
   text given screenshots — the LLaVA-style alignment stage, but diffusion).
3. LiquidMemory fine-tuning on agent traces (multi-step PC-control sessions).
4. Action head on `<|action|>` tokens (structured actions decoded per block).
5. Multi-agent: shared cursor-state tokens so co-resident Genies see each other.
6. BitNet kernel integration in `genied`; int8/2-bit activation serving.

## 11. Quickstart

Verified on NixOS (CPU torch). Everything runs from `genie/training/`.

```bash
# environment (NixOS: system python has no pip)
nix-shell -p python312 --run "python -m venv /tmp/genie-venv"
/tmp/genie-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/genie-venv/bin/pip install numpy pyyaml tokenizers safetensors tqdm pytest
export LD_LIBRARY_PATH="$(nix eval --raw nixpkgs#stdenv.cc.cc.lib)/lib:$LD_LIBRARY_PATH"

# data: any .txt corpus -> BPE tokenizer + packed uint16 tokens
/tmp/genie-venv/bin/python prepare_data.py --input corpus.txt \
    --vocab-size 32000 --save-tokenizer tok.json --out data.bin

# train (synthetic data if --data omitted)
/tmp/genie-venv/bin/python train.py --config configs/smoke.yaml \
    --data data.bin --tokenizer tok.json --out /tmp/genie-ckpt

# export ternary BitNet weights for genied
/tmp/genie-venv/bin/python export_bitnet.py \
    --ckpt /tmp/genie-ckpt/final.pt --out /tmp/genie-export

# tests
cd genie/training && PYTHONPATH=. /tmp/genie-venv/bin/python -m pytest tests/ -q
```

Smoke run on CPU: loss starts at ~19 (≈ ln(512)·2.6, expected for the masked
diffusion objective) and falls immediately; export yields int8 tensors whose
unique values are exactly `{-1, 0, 1}` plus per-tensor fp32 scales — the
BitNet b1.58 format.
