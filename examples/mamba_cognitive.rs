//! Mamba Cognitive Inference — first real-model delta modulation test.
//!
//! Run: `cargo run --features candle --example mamba_cognitive`
//!
//! Downloads mamba-130m from HuggingFace (~500MB), then runs inference with
//! three cognitive states (phasic, neutral, tonic) and compares the resulting
//! logit distributions.
//!
//! This is the PROOF that Nous's cognitive algorithms directly modulate how
//! a real language model processes information — not through text, not through
//! sampling tricks, but inside the forward pass via delta scaling.

#[cfg(feature = "candle")]
use nous_regulator::cognition::delta_modulation::compute_delta_modulation;
#[cfg(feature = "candle")]
use nous_regulator::inference::cognitive_model::CognitiveModel;
#[cfg(feature = "candle")]
use nous_regulator::inference::mamba::{CognitiveMambaModel, HfTokenizer, MambaConfig};
#[cfg(feature = "candle")]
use nous_regulator::inference::model::LocalModel;
#[cfg(feature = "candle")]
use nous_regulator::inference::tokenizer::NousTokenizer;
#[cfg(feature = "candle")]
use nous_regulator::types::intervention::{CognitiveState, DeltaModulation};
#[cfg(feature = "candle")]
use nous_regulator::types::world::GainMode;

fn main() {
    #[cfg(not(feature = "candle"))]
    {
        eprintln!("This example requires the `candle` feature: cargo run --features candle --example mamba_cognitive");
        return;
    }

    #[cfg(feature = "candle")]
    run();
}

#[cfg(feature = "candle")]
fn run() {
    println!("=== Nous Tầng 2: Real Model Delta Modulation ===\n");

    // HF-format variant has safetensors + tokenizer bundled.
    let model_id = "state-spaces/mamba-130m-hf";
    let config = MambaConfig::mamba_130m();

    // ── Step 1: Load tokenizer ──
    println!("Loading tokenizer from {model_id}...");
    let tokenizer = match HfTokenizer::from_pretrained(model_id) {
        Ok(t) => {
            println!("  Tokenizer loaded. Vocab size: {}", t.vocab_size());
            t
        }
        Err(e) => {
            eprintln!("Failed to load tokenizer: {e}");
            return;
        }
    };

    // ── Step 2: Load model ──
    println!("Loading model from {model_id} (~500MB, first run downloads)...");
    let mut model = match CognitiveMambaModel::from_pretrained(model_id, config.clone()) {
        Ok(m) => {
            println!(
                "  Model loaded. {} layers, d_model={}, vocab={}",
                m.num_layers(),
                config.d_model,
                config.vocab_size
            );
            m
        }
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            return;
        }
    };

    // ── Step 3: Tokenize a prompt ──
    let prompt = "The meaning of life is";
    let tokens = tokenizer.encode(prompt, false).unwrap();
    println!("\nPrompt: \"{prompt}\"");
    println!("Tokens: {tokens:?} ({} tokens)\n", tokens.len());

    // ── Step 4: Forward with three cognitive states ──
    println!("Running forward pass with 3 cognitive states...\n");

    let states: Vec<(&str, CognitiveState)> = vec![
        (
            "Neutral (gain=1.0)",
            CognitiveState::default(),
        ),
        (
            "Phasic  (focused)",
            CognitiveState {
                gain_mode: GainMode::Phasic,
                ..CognitiveState::default()
            },
        ),
        (
            "Tonic   (exploratory)",
            CognitiveState {
                gain_mode: GainMode::Tonic,
                ..CognitiveState::default()
            },
        ),
    ];

    let mut all_logits: Vec<(&str, Vec<f32>, DeltaModulation)> = Vec::new();

    for (name, state) in &states {
        // Reset model state for fair comparison.
        model.reset_cache();

        // Compute delta modulation.
        let dm = compute_delta_modulation(state, model.num_layers());

        // Forward all prompt tokens to fill SSM state.
        let result = model
            .forward_cognitive(&tokens, 0, &dm)
            .expect("Forward pass failed");

        println!(
            "  {name}: gain={:.3}, modulated={}, layers={:?}",
            dm.gain_factor,
            result.modulation_applied,
            if result.modulated_layers.len() > 6 {
                format!(
                    "{}..{} ({} layers)",
                    result.modulated_layers.first().unwrap_or(&0),
                    result.modulated_layers.last().unwrap_or(&0),
                    result.modulated_layers.len()
                )
            } else {
                format!("{:?}", result.modulated_layers)
            }
        );

        all_logits.push((name, result.logits.clone(), dm));
    }

    // ── Step 5: Compare logit distributions ──
    println!("\n── Top-5 Token Predictions ──\n");

    for (name, logits, _dm) in &all_logits {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        print!("  {name}: ");
        for (i, (tok_id, logit)) in indexed.iter().take(5).enumerate() {
            let tok_text = tokenizer
                .decode_token(*tok_id as u32)
                .unwrap_or_else(|_| format!("[{tok_id}]"));
            let sep = if i < 4 { " | " } else { "" };
            print!("{:.1}:{}{sep}", logit, tok_text.trim());
        }
        println!();
    }

    // ── Step 6: Quantify differences ──
    println!("\n── Logit Distribution Differences ──\n");

    let neutral_logits = &all_logits[0].1;
    for (name, logits, _) in all_logits.iter().skip(1) {
        let max_diff: f32 = neutral_logits
            .iter()
            .zip(logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let mean_diff: f32 = neutral_logits
            .iter()
            .zip(logits.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / logits.len() as f32;

        let rank_changes = count_rank_changes(neutral_logits, logits, 20);

        println!(
            "  {name} vs Neutral: max_diff={:.4}, mean_diff={:.6}, top-20 rank changes={}",
            max_diff, mean_diff, rank_changes
        );
    }

    // ── Step 7: Verdict ──
    let neutral = &all_logits[0].1;
    let phasic = &all_logits[1].1;
    let tonic = &all_logits[2].1;

    let phasic_differs = neutral
        .iter()
        .zip(phasic.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    let tonic_differs = neutral
        .iter()
        .zip(tonic.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);

    println!("\n── Verdict ──\n");
    if phasic_differs && tonic_differs {
        println!("  PROVEN: Delta modulation produces DIFFERENT logit distributions.");
        println!("  Nous's cognitive state directly modulates how Mamba processes information.");
        println!("  This is Tầng 2 — cognitive algorithms operating INSIDE the model.");
    } else if phasic_differs || tonic_differs {
        println!("  PARTIAL: One cognitive state changed logits, the other didn't.");
        println!("  Delta modulation has effect but may need tuning.");
    } else {
        println!("  NO EFFECT: Logits identical across cognitive states.");
        println!("  Investigation needed — check layer targeting and gain range.");
    }
}

#[cfg(feature = "candle")]
fn count_rank_changes(logits_a: &[f32], logits_b: &[f32], k: usize) -> usize {
    let top_a = top_k_indices(logits_a, k);
    let top_b = top_k_indices(logits_b, k);

    let mut changes = 0;
    for i in 0..k.min(top_a.len()).min(top_b.len()) {
        if top_a[i] != top_b[i] {
            changes += 1;
        }
    }
    changes
}

#[cfg(feature = "candle")]
fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|(i, _)| *i).collect()
}
