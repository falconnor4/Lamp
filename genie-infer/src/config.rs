use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub mlp_ratio: f32,
    pub max_seq_len: usize,
    pub block_size: usize,
    pub rope_base: f64,
    pub norm_eps: f64,
    pub tie_embeddings: bool,
    pub act_quant: bool,
    pub pad_id: u32,
    pub mask_id: u32,
    pub bos_id: u32,
    pub eos_id: u32,
    pub diffusion_steps: usize,
    pub diffusion_weight_clamp: f32,
}

impl Config {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let s = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&s)?)
    }

    pub fn mlp_hidden(&self) -> usize {
        (self.d_model as f32 * self.mlp_ratio) as usize
    }
}