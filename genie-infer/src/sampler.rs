use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

pub struct BlockSampler {
    pub block_size: usize,
    pub steps: usize,
    pub mask_id: u32,
    pub temperature: f32,
    rng: StdRng,
}

impl BlockSampler {
    pub fn new(block_size: usize, steps: usize, mask_id: u32, temperature: f32, seed: u64) -> Self {
        Self {
            block_size,
            steps,
            mask_id,
            temperature,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Run block-diffusion generation for one block of `block_size` tokens.
    /// `logits_fn` receives the current block (mask tokens / revealed tokens)
    /// and must return the model's logits for those `block_size` positions,
    /// shape [block_size, vocab], row-major.
    pub fn sample_block(&mut self, logits_fn: impl Fn(&[u32]) -> Vec<f32>) -> Vec<u32> {
        let bs = self.block_size;
        let mut x = vec![self.mask_id; bs];
        for i in 0..self.steps {
            let t = 1.0 - i as f32 / self.steps as f32;
            let s = 1.0 - (i + 1) as f32 / self.steps as f32;

            let mut logits = logits_fn(&x);
            let v = logits.len() / bs;
            for p in 0..bs {
                logits[p * v + self.mask_id as usize] = f32::NEG_INFINITY;
            }

            let x0: Vec<u32> = if self.temperature <= 0.0 {
                (0..bs).map(|p| argmax(&logits[p * v..(p + 1) * v])).collect()
            } else {
                (0..bs)
                    .map(|p| sample(&logits[p * v..(p + 1) * v], self.temperature, &mut self.rng))
                    .collect()
            };

            x = reverse_step(&x, &x0, t, s, self.mask_id, &mut self.rng);
        }
        x
    }
}

fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    let mut bestv = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > bestv {
            bestv = v;
            best = i;
        }
    }
    best as u32
}

fn sample(logits: &[f32], temperature: f32, rng: &mut StdRng) -> u32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    let mut probs = vec![0.0f64; logits.len()];
    for (i, &l) in logits.iter().enumerate() {
        let p = (((l - max) / temperature) as f64).exp();
        probs[i] = p;
        sum += p;
    }
    let mut u = rng.gen::<f64>() * sum;
    for (i, &p) in probs.iter().enumerate() {
        u -= p;
        if u < 0.0 {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

fn reverse_step(x: &[u32], x0: &[u32], t: f32, s: f32, mask_id: u32, rng: &mut StdRng) -> Vec<u32> {
    let mut y = x.to_vec();
    if s <= 0.0 {
        for i in 0..x.len() {
            if x[i] == mask_id {
                y[i] = x0[i];
            }
        }
        return y;
    }
    let reveal_p = (t - s) / t;
    let keep_prob = if t < 1.0 { (1.0 - s) / (1.0 - t) } else { 1.0 };
    for i in 0..x.len() {
        let masked = x[i] == mask_id;
        if masked && rng.gen::<f32>() < reveal_p {
            y[i] = x0[i];
        } else if !masked && keep_prob < 1.0 && rng.gen::<f32>() > keep_prob {
            y[i] = mask_id;
        }
    }
    y
}