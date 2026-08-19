use std::error::Error;

use safetensors::tensor::TensorView;
use safetensors::SafeTensors;

use crate::config::Config;

pub struct TernaryLinear {
    pub w: Vec<i8>, // [out, inp] row-major, ternary {-1, 0, 1}
    pub scale: f32,
    pub out: usize,
    pub inp: usize,
}

impl TernaryLinear {
    /// y = scale * (x @ w^T), x is [l, inp] row-major (already act-quantized by caller).
    pub fn matmul(&self, x: &[f32], l: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; l * self.out];
        for t in 0..l {
            let xrow = &x[t * self.inp..(t + 1) * self.inp];
            let yrow = &mut y[t * self.out..(t + 1) * self.out];
            for o in 0..self.out {
                let wrow = &self.w[o * self.inp..(o + 1) * self.inp];
                let mut acc = 0.0f32;
                for k in 0..self.inp {
                    acc += wrow[k] as f32 * xrow[k];
                }
                yrow[o] = acc * self.scale;
            }
        }
        y
    }
}

fn view_f32(t: &TensorView) -> Vec<f32> {
    t.data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn view_i8(t: &TensorView) -> Vec<i8> {
    t.data().iter().map(|&b| b as i8).collect()
}

fn scalar(t: &TensorView) -> f32 {
    view_f32(t)[0]
}

fn linear(
    t: &SafeTensors,
    prefix: &str,
    out: usize,
    inp: usize,
) -> Result<TernaryLinear, Box<dyn Error + Send + Sync>> {
    Ok(TernaryLinear {
        w: view_i8(&t.tensor(&format!("{prefix}.weight"))?),
        scale: scalar(&t.tensor(&format!("{prefix}.scale"))?),
        out,
        inp,
    })
}

pub struct Block {
    attn_norm: Vec<f32>,
    mlp_norm: Vec<f32>,
    q: TernaryLinear,
    k: TernaryLinear,
    v: TernaryLinear,
    o: TernaryLinear,
    gate: TernaryLinear,
    up: TernaryLinear,
    down: TernaryLinear,
    hidden: usize,
}

impl Block {
    fn load(t: &SafeTensors, p: &str, cfg: &Config) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let d = cfg.d_model;
        let hd = cfg.head_dim;
        let h = cfg.n_heads;
        let nkv = cfg.n_kv_heads;
        let hidden = cfg.mlp_hidden();
        Ok(Self {
            attn_norm: view_f32(&t.tensor(&format!("{p}.attn_norm.weight"))?),
            mlp_norm: view_f32(&t.tensor(&format!("{p}.mlp_norm.weight"))?),
            q: linear(t, &format!("{p}.attn.q"), h * hd, d)?,
            k: linear(t, &format!("{p}.attn.k"), nkv * hd, d)?,
            v: linear(t, &format!("{p}.attn.v"), nkv * hd, d)?,
            o: linear(t, &format!("{p}.attn.o"), d, h * hd)?,
            gate: linear(t, &format!("{p}.mlp.gate"), hidden, d)?,
            up: linear(t, &format!("{p}.mlp.up"), hidden, d)?,
            down: linear(t, &format!("{p}.mlp.down"), d, hidden)?,
            hidden,
        })
    }

    fn forward(&self, x: &[f32], cfg: &Config, cos: &[f32], sin: &[f32]) -> Vec<f32> {
        let d = cfg.d_model;
        let h = cfg.n_heads;
        let hd = cfg.head_dim;
        let nkv = cfg.n_kv_heads;
        let nrep = h / nkv;
        let l = x.len() / d;

        // attention: norm -> quantize -> q/k/v -> rope -> attention -> o
        let mut xn = x.to_vec();
        rmsnorm(&mut xn, &self.attn_norm, cfg.norm_eps);
        act_quantize(&mut xn, d);
        let mut q = self.q.matmul(&xn, l); // [l, h*hd]
        let mut k = self.k.matmul(&xn, l); // [l, nkv*hd]
        let v = self.v.matmul(&xn, l); // [l, nkv*hd]

        apply_rope(&mut q, cfg, cos, sin, l, h, hd);
        apply_rope(&mut k, cfg, cos, sin, l, nkv, hd);

        let sqrt_hd = (hd as f32).sqrt();
        let bs = cfg.block_size;
        let mut attn = vec![0.0f32; l * h * hd];
        for t in 0..l {
            let bt = (t / bs) as i32;
            for hh in 0..h {
                let hkv = hh / nrep;
                let mut scores = vec![0.0f32; l];
                let mut max = f32::NEG_INFINITY;
                for s in 0..l {
                    let bss = (s / bs) as i32;
                    if bss > bt {
                        scores[s] = f32::NEG_INFINITY;
                    } else {
                        let qo = t * h * hd + hh * hd;
                        let ko = s * nkv * hd + hkv * hd;
                        let mut dot = 0.0f32;
                        for i in 0..hd {
                            dot += q[qo + i] * k[ko + i];
                        }
                        scores[s] = dot / sqrt_hd;
                        if scores[s] > max {
                            max = scores[s];
                        }
                    }
                }
                let mut sum = 0.0f32;
                for s in 0..l {
                    if scores[s] != f32::NEG_INFINITY {
                        scores[s] = (scores[s] - max).exp();
                        sum += scores[s];
                    }
                }
                for s in 0..l {
                    scores[s] /= sum;
                }
                let ao = t * h * hd + hh * hd;
                for i in 0..hd {
                    let mut acc = 0.0f32;
                    for s in 0..l {
                        acc += scores[s] * v[s * nkv * hd + hkv * hd + i];
                    }
                    attn[ao + i] = acc;
                }
            }
        }

        act_quantize(&mut attn, h * hd);
        let attn_out = self.o.matmul(&attn, l);

        let mut r = vec![0.0f32; l * d];
        for i in 0..l * d {
            r[i] = x[i] + attn_out[i];
        }

        // mlp: norm -> quantize -> gate/up -> silu*up -> quantize -> down
        let mut mn = r.clone();
        rmsnorm(&mut mn, &self.mlp_norm, cfg.norm_eps);
        act_quantize(&mut mn, d);
        let gate = self.gate.matmul(&mn, l);
        let up = self.up.matmul(&mn, l);
        let hidden = self.hidden;
        let mut hu = vec![0.0f32; l * hidden];
        for i in 0..l * hidden {
            let g = gate[i];
            hu[i] = g / (1.0 + (-g).exp()) * up[i];
        }
        act_quantize(&mut hu, hidden);
        let down = self.down.matmul(&hu, l);
        for i in 0..l * d {
            r[i] += down[i];
        }
        r
    }
}

pub struct Model {
    cfg: Config,
    tok_emb: Vec<f32>,   // [V, D]
    modality0: Vec<f32>, // [D]
    norm_final: Vec<f32>,
    blocks: Vec<Block>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

impl Model {
    pub fn load(sf: &str, cfg: Config) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let data = std::fs::read(sf)?;
        let t = SafeTensors::deserialize(&data)?;
        let tok_emb = view_f32(&t.tensor("tok_emb.weight")?);
        let modality = view_f32(&t.tensor("modality_emb.weight")?);
        let modality0 = modality[0..cfg.d_model].to_vec();
        let norm_final = view_f32(&t.tensor("backbone.norm.weight")?);

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            blocks.push(Block::load(&t, &format!("backbone.blocks.{i}"), &cfg)?);
        }

        let n = cfg.head_dim / 2;
        let mut rope_cos = vec![0.0f32; cfg.block_size * n];
        let mut rope_sin = vec![0.0f32; cfg.block_size * n];
        for p in 0..cfg.block_size {
            for i in 0..n {
                let inv = cfg.rope_base.powf(-(2.0 * i as f64) / cfg.head_dim as f64);
                let ang = p as f64 * inv;
                rope_cos[p * n + i] = ang.cos() as f32;
                rope_sin[p * n + i] = ang.sin() as f32;
            }
        }

        Ok(Self {
            cfg,
            tok_emb,
            modality0,
            norm_final,
            blocks,
            rope_cos,
            rope_sin,
        })
    }

    pub fn cfg(&self) -> &Config {
        &self.cfg
    }

    /// Full forward pass: input token ids -> logits [L, V] row-major.
    pub fn logits(&self, ids: &[u32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let l = ids.len();
        let v = cfg.vocab_size;

        let mut x = vec![0.0f32; l * d];
        for t in 0..l {
            let id = ids[t] as usize;
            let erow = &self.tok_emb[id * d..(id + 1) * d];
            for i in 0..d {
                x[t * d + i] = erow[i] + self.modality0[i];
            }
        }

        for b in &self.blocks {
            x = b.forward(&x, cfg, &self.rope_cos, &self.rope_sin);
        }

        rmsnorm(&mut x, &self.norm_final, cfg.norm_eps);

        // tied head: logits[t][v] = <x[t], tok_emb[v]>
        let mut logits = vec![0.0f32; l * v];
        for t in 0..l {
            let xrow = &x[t * d..(t + 1) * d];
            for vv in 0..v {
                let erow = &self.tok_emb[vv * d..(vv + 1) * d];
                let mut acc = 0.0f32;
                for i in 0..d {
                    acc += xrow[i] * erow[i];
                }
                logits[t * v + vv] = acc;
            }
        }
        logits
    }
}

pub fn rmsnorm(x: &mut [f32], w: &[f32], eps: f64) {
    let d = w.len();
    let n = x.len() / d;
    for t in 0..n {
        let row = &x[t * d..(t + 1) * d];
        let mut s = 0.0f64;
        for &v in row {
            s += (v as f64) * (v as f64);
        }
        let inv = 1.0 / ((s / d as f64 + eps).sqrt());
        let inv = inv as f32;
        for i in 0..d {
            x[t * d + i] *= inv * w[i];
        }
    }
}

pub fn act_quantize(x: &mut [f32], d: usize) {
    let n = x.len() / d;
    for t in 0..n {
        let mut amax = 0.0f32;
        for i in 0..d {
            let a = x[t * d + i].abs();
            if a > amax {
                amax = a;
            }
        }
        let scale = amax.max(1e-12) / 127.0;
        for i in 0..d {
            let q = (x[t * d + i] / scale).round().clamp(-127.0, 127.0);
            x[t * d + i] = q * scale;
        }
    }
}

fn apply_rope(
    x: &mut [f32],
    cfg: &Config,
    cos: &[f32],
    sin: &[f32],
    l: usize,
    num_heads: usize,
    hd: usize,
) {
    let n = hd / 2;
    let bs = cfg.block_size;
    for t in 0..l {
        let pos = t % bs;
        let base = pos * n;
        for hh in 0..num_heads {
            let o = t * num_heads * hd + hh * hd;
            for i in 0..n {
                let a = x[o + 2 * i];
                let b = x[o + 2 * i + 1];
                x[o + 2 * i] = a * cos[base + i] - b * sin[base + i];
                x[o + 2 * i + 1] = a * sin[base + i] + b * cos[base + i];
            }
        }
    }
}