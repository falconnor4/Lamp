mod config;
mod model;
mod sampler;

use std::error::Error;

use config::Config;
use model::Model;
use sampler::BlockSampler;

struct Args {
    model: String,
    manifest: String,
    tokenizer: String,
    prompt: String,
    num_tokens: usize,
    temperature: f32,
    seed: u64,
    reference: Option<String>,
}

fn parse_args() -> Result<Args, Box<dyn Error + Send + Sync>> {
    let mut a = Args {
        model: String::new(),
        manifest: String::new(),
        tokenizer: String::new(),
        prompt: "The meaning of life is".to_string(),
        num_tokens: 64,
        temperature: 0.8,
        seed: 0,
        reference: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut next = || it.next().ok_or(format!("missing value for {k}"));
        match k.as_str() {
            "--model" => a.model = next()?,
            "--manifest" => a.manifest = next()?,
            "--tokenizer" => a.tokenizer = next()?,
            "--prompt" => a.prompt = next()?,
            "--num-tokens" => a.num_tokens = next()?.parse()?,
            "--temperature" => a.temperature = next()?.parse()?,
            "--seed" => a.seed = next()?.parse()?,
            "--ref" => a.reference = Some(next()?),
            other => return Err(format!("unknown arg {other}").into()),
        }
    }
    if a.model.is_empty() || a.manifest.is_empty() || a.tokenizer.is_empty() {
        return Err("--model, --manifest and --tokenizer are required".into());
    }
    Ok(a)
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = parse_args()?;

    let cfg = Config::load(&args.manifest)?;
    let model = Model::load(&args.model, cfg.clone())?;

    let tok = tokenizers::Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;
    let ids: Vec<u32> = tok
        .encode(args.prompt.as_str(), false)
        .map_err(|e| format!("failed to encode prompt: {e}"))?
        .get_ids()
        .to_vec();

    println!("model: {} (d_model={}, layers={}, block_size={})", args.model, cfg.d_model, cfg.n_layers, cfg.block_size);
    println!("prompt: {args_prompt:?}", args_prompt = args.prompt);
    println!("prompt ids: {ids:?} (len {})", ids.len());

    // Deterministic logits comparison against a Python reference dump.
    let logits = model.logits(&ids);
    if let Some(ref_dir) = &args.reference {
        compare(&ref_dir, &ids, &logits, cfg.vocab_size)?;
    }

    // Block-diffusion generation.
    let mut sampler = BlockSampler::new(
        cfg.block_size,
        cfg.diffusion_steps,
        cfg.mask_id,
        args.temperature,
        args.seed,
    );
    let orig_len = ids.len();
    let mut generated = ids.clone();
    let mut n_new = 0usize;
    while n_new < args.num_tokens {
        let gen = generated.clone();
        let vocab = cfg.vocab_size;
        let bs = cfg.block_size;
        let max_seq_len = cfg.max_seq_len;
        let block = sampler.sample_block(|cur: &[u32]| {
            let mut full = gen.clone();
            full.extend_from_slice(cur);
            if full.len() > max_seq_len {
                full = full[full.len() - max_seq_len..].to_vec();
            }
            let l = full.len();
            let logits = model.logits(&full);
            let start = (l - bs) * vocab;
            logits[start..].to_vec()
        });
        if block.iter().any(|&x| x == cfg.eos_id) {
            generated.extend_from_slice(&block);
            n_new += bs;
            break;
        }
        generated.extend_from_slice(&block);
        n_new += bs;
    }

    let new_tokens = &generated[orig_len..];
    let text = tok
        .decode(new_tokens, true)
        .map_err(|e| format!("failed to decode: {e}"))?;
    println!();
    println!("{}{}", args.prompt, text);
    Ok(())
}

fn compare(
    dir: &str,
    ids: &[u32],
    mine: &[f32],
    vocab: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json"))?)?;
    let ref_ids: Vec<u32> = meta["ids"]
        .as_array()
        .ok_or("meta.json missing ids")?
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let bytes = std::fs::read(format!("{dir}/logits.bin"))?;
    let reference: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    if ref_ids != ids {
        return Err(format!(
            "reference ids {ref_ids:?} do not match mine {ids:?}"
        )
        .into());
    }
    if reference.len() != mine.len() {
        return Err(format!(
            "logits length mismatch: ref {} vs mine {}",
            reference.len(),
            mine.len()
        )
        .into());
    }

    let l = ids.len();
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut argmax_hit = 0usize;
    for t in 0..l {
        let mut best_ref = 0usize;
        let mut best_mine = 0usize;
        let mut br = f32::NEG_INFINITY;
        let mut bm = f32::NEG_INFINITY;
        for v in 0..vocab {
            let idx = t * vocab + v;
            let rv = reference[idx];
            let mv = mine[idx];
            let d = (rv - mv).abs();
            max_diff = max_diff.max(d);
            sum_diff += d as f64;
            if rv > br {
                br = rv;
                best_ref = v;
            }
            if mv > bm {
                bm = mv;
                best_mine = v;
            }
        }
        if best_ref == best_mine {
            argmax_hit += 1;
        }
    }
    let mean_diff = sum_diff / mine.len() as f64;
    println!();
    println!("=== reference comparison (L={l}, V={vocab}) ===");
    println!("max_abs_diff   = {max_diff:.3e}");
    println!("mean_abs_diff  = {mean_diff:.3e}");
    println!("argmax match   = {argmax_hit}/{l}");
    Ok(())
}