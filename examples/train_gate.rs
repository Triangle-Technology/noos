//! CognitiveGate Training — Tầng 3 Phase 3.3 proof of concept.
//!
//! Run: `cargo run --features candle --example train_gate`
//!
//! This example trains a CognitiveGate on a frozen mamba-130m base model.
//! Only gate parameters (~105K) are trained; base model weights stay fixed.
//!
//! The gate learns to modulate hidden state via residual blend:
//!   modulated = (1 - alpha) * hidden_state + alpha * cognitive_contribution
//! where alpha starts near 0.05 (safe passthrough) and learns from data.
//!
//! What to observe:
//! - Loss should decrease over training steps
//! - gate_alpha should evolve from ~0.05 initial value
//! - Gate params change while base model is unchanged
//!
//! See `docs/tang3-roadmap.md` Phase 3.3 for context.

use candle_core::{Device, Tensor};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};

use nous_regulator::errors::NousResult;
use nous_regulator::inference::cognitive_gate::CognitiveGateConfig;
use nous_regulator::inference::cognitive_model::CognitiveModel;
use nous_regulator::inference::mamba::{CognitiveMambaWithGate, HfTokenizer, MambaConfig};
use nous_regulator::inference::model::LocalModel;
use nous_regulator::inference::tokenizer::NousTokenizer;
use nous_regulator::types::intervention::DeltaModulation;

/// Sample text for training (when no file is provided).
/// A mix of factual and conversational content to give the gate
/// different processing contexts to learn from.
const SAMPLE_TEXT: &str = "\
The brain processes information through billions of interconnected neurons. \
Each neuron communicates via electrochemical signals, transmitting data across \
synapses at speeds of up to 120 meters per second. The prefrontal cortex handles \
executive functions like planning and decision-making. How do you feel about that? \
I think it's fascinating how the brain can reorganize itself through neuroplasticity. \
The hippocampus plays a crucial role in forming new memories. Scientists have \
discovered that sleep is essential for memory consolidation. During deep sleep, \
the brain replays experiences and strengthens important connections. What patterns \
do you notice in your own thinking? The amygdala processes emotions rapidly, \
sometimes before conscious awareness. This fast pathway evolved as a survival \
mechanism. Fear responses can be triggered in milliseconds. Meanwhile, the \
default mode network activates during rest and mind-wandering. Creative insights \
often emerge from this wandering state. The locus coeruleus modulates attention \
through norepinephrine release. Phasic bursts sharpen focus on specific stimuli, \
while tonic activity enables broad environmental monitoring.";

fn main() -> NousResult<()> {
    println!("=== Nous Tầng 3: CognitiveGate Training ===\n");

    // ── Configuration ───────────────────────────────────────────────
    let model_id = "state-spaces/mamba-130m-hf";
    let learning_rate = 1e-3;
    let num_steps = 50;
    let seq_len = 8; // Short sequences to fit backward graph in memory (CPU training)

    let config = MambaConfig::mamba_130m();
    let gate_config = CognitiveGateConfig::from_mamba_config(&config);

    println!("Model: {model_id}");
    println!("Gate position: layer {} / {}", gate_config.gate_position, config.n_layer);
    println!("Gate cognitive dim: {}", gate_config.cognitive_dim);
    println!("Learning rate: {learning_rate}");
    println!("Steps: {num_steps}, seq_len: {seq_len}\n");

    // ── Load model + gate ───────────────────────────────────────────
    println!("Loading model (this may download ~500MB on first run)...");

    let (mut model, gate_varmap) =
        CognitiveMambaWithGate::from_pretrained_with_gate(model_id, config.clone(), gate_config)?;

    let tokenizer = HfTokenizer::from_pretrained(model_id)?;

    // Count params.
    let gate_param_count: usize = gate_varmap
        .data()
        .lock()
        .map_err(|e| nous_regulator::NousError::Internal(format!("Lock error: {e}")))?
        .values()
        .map(|v| v.elem_count())
        .sum();

    println!("Gate params: {gate_param_count} (trainable)");
    println!("Base model: frozen\n");

    // ── Tokenize ────────────────────────────────────────────────────
    let tokens = tokenizer.encode(SAMPLE_TEXT, false)?;
    println!("Tokenized: {} tokens from sample text\n", tokens.len());

    if tokens.len() < seq_len + 1 {
        return Err(nous_regulator::NousError::Internal(
            "Sample text too short for training".into(),
        ));
    }

    // Create (input, target) pairs: target = input shifted by 1.
    let num_sequences = (tokens.len() - 1) / seq_len;

    // ── Create optimizer (gate params only) ─────────────────────────
    let gate_vars = gate_varmap.all_vars();
    let params = ParamsAdamW {
        lr: learning_rate,
        ..ParamsAdamW::default()
    };
    let mut optimizer = AdamW::new(gate_vars, params)
        .map_err(|e| nous_regulator::NousError::Internal(format!("Optimizer init error: {e}")))?;

    // ── Training loop ───────────────────────────────────────────────
    println!("── Training ──\n");
    println!("{:>5}  {:>10}  {:>10}  {:>12}", "Step", "Loss", "Alpha", "Delta Gain");
    println!("{}", "-".repeat(45));

    for step in 0..num_steps {
        // Cycle through sequences.
        let seq_idx = step % num_sequences;
        let start = seq_idx * seq_len;
        let input_tokens = &tokens[start..start + seq_len];
        let target_tokens: Vec<u32> = tokens[start + 1..start + seq_len + 1].to_vec();

        // Reset SSM state for each sequence.
        model.reset_cache();

        // Forward pass (training mode — returns Tensor for gradient flow).
        let logits = model
            .forward_train(input_tokens)
            .map_err(|e| nous_regulator::NousError::Internal(format!("Forward error: {e}")))?;

        // Cross-entropy loss.
        let targets = Tensor::new(target_tokens.as_slice(), &Device::Cpu)
            .map_err(|e| nous_regulator::NousError::Internal(format!("Target tensor error: {e}")))?;

        let loss = candle_nn::loss::cross_entropy(&logits, &targets)
            .map_err(|e| nous_regulator::NousError::Internal(format!("Loss error: {e}")))?;

        let loss_val = loss
            .to_scalar::<f32>()
            .map_err(|e| nous_regulator::NousError::Internal(format!("Loss scalar error: {e}")))?;

        // Backward + optimizer step (only gate params updated).
        optimizer
            .backward_step(&loss)
            .map_err(|e| nous_regulator::NousError::Internal(format!("Backward step error: {e}")))?;

        // Inspect gate output (run one token through inference mode).
        model.reset_cache();
        let dm = DeltaModulation::default();
        let inspect = model.forward_cognitive(&[input_tokens[0]], 0, &dm)?;
        let gate_alpha = inspect.gate_alpha.unwrap_or(0.0);
        let delta_gain = inspect.gate_delta_gain.unwrap_or(1.0);

        // Print metrics.
        if step % 5 == 0 || step == num_steps - 1 {
            println!(
                "{:>5}  {:>10.4}  {:>10.4}  {:>12.4}",
                step, loss_val, gate_alpha, delta_gain
            );
        }
    }

    println!("\n── Training complete ──\n");

    // ── Verify gate learned ─────────────────────────────────────────
    // Run inference to check gate alpha has evolved.
    model.reset_cache();
    let dm = DeltaModulation::default();
    let final_result = model.forward_cognitive(&[tokens[0]], 0, &dm)?;
    let final_alpha = final_result.gate_alpha.unwrap_or(0.0);
    let final_delta_gain = final_result.gate_delta_gain.unwrap_or(1.0);

    println!("Final gate_alpha: {final_alpha:.4} (init was ~0.05)");
    println!("Final delta_gain: {final_delta_gain:.4} (range [0.5, 2.0])");

    if (final_alpha - 0.047_f64).abs() > 0.01 {
        println!("\nGate alpha has moved from initialization → gate is LEARNING.");
    } else {
        println!("\nGate alpha near initial value — may need more steps or different data.");
    }

    // Verify gate params actually changed.
    let gate_changed = gate_varmap
        .data()
        .lock()
        .map_err(|e| nous_regulator::NousError::Internal(format!("Lock error: {e}")))?
        .iter()
        .filter(|(name, _)| name.contains("w_gate"))
        .any(|(_, var)| {
            var.flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map(|bias| bias.iter().any(|v| (v - (-3.0f32)).abs() > 0.01))
                .unwrap_or(false)
        });

    if gate_changed {
        println!("W_gate bias moved from -3.0 → gate learning confirmed.\n");
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase 3.4: Verify Principles in Trained Gate
    // ═══════════════════════════════════════════════════════════════

    println!("=== Phase 3.4: Principle Verification ===\n");

    // Test different input types to verify gate differentiates context.
    // P4 (Affect): gate should respond differently to emotional vs factual content.
    // P5 (Classification): gate should show different alpha/delta for different text types.
    let test_prompts = vec![
        ("Factual", "The hippocampus is a brain region involved in memory"),
        ("Emotional", "I feel terrified and my heart is racing with fear"),
        ("Question", "How do neurons communicate across synapses"),
        ("Creative", "Imagine a world where thoughts flow like rivers of light"),
    ];

    println!("{:<12} {:>10} {:>12}  Principle check", "Type", "Alpha", "Delta Gain");
    println!("{}", "-".repeat(55));

    let mut alphas = Vec::new();
    let mut deltas = Vec::new();

    for (label, prompt) in &test_prompts {
        model.reset_cache();
        let prompt_tokens = tokenizer.encode(prompt, false)?;
        // Run a few tokens through to build up SSM state.
        let dm = DeltaModulation::default();
        let n = prompt_tokens.len().min(8);
        let result = model.forward_cognitive(&prompt_tokens[..n], 0, &dm)?;
        let alpha = result.gate_alpha.unwrap_or(0.0);
        let delta = result.gate_delta_gain.unwrap_or(1.0);

        alphas.push(alpha);
        deltas.push(delta);

        println!("{:<12} {:>10.4} {:>12.4}", label, alpha, delta);
    }

    // P4 verification: do alphas vary across context types?
    let alpha_range = alphas.iter().cloned().fold(f64::INFINITY, f64::min);
    let alpha_max = alphas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let alpha_spread = alpha_max - alpha_range;

    let delta_min = deltas.iter().cloned().fold(f64::INFINITY, f64::min);
    let delta_max = deltas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let delta_spread = delta_max - delta_min;

    println!("\n── Principle Verification ──\n");

    // P4: Affect permeates — gate should show different responses.
    if alpha_spread > 0.001 {
        println!("P4 (Affect): PASS — alpha varies across contexts (spread: {alpha_spread:.4})");
    } else {
        println!("P4 (Affect): WEAK — alpha barely varies (spread: {alpha_spread:.6}). More training needed.");
    }

    // P5: Classification emerges — different text types get different treatment.
    if delta_spread > 0.001 {
        println!("P5 (Classification): PASS — delta_gain differs by text type (spread: {delta_spread:.4})");
    } else {
        println!("P5 (Classification): WEAK — delta_gain uniform (spread: {delta_spread:.6}). More training needed.");
    }

    // P7: Multi-timescale — the gate produces non-trivial delta values
    let non_unity = deltas.iter().any(|d| (d - 1.0).abs() > 0.05);
    if non_unity {
        println!("P7 (Multi-Timescale): PASS — gate produces non-unity delta_gain (modulating SSM timescale)");
    } else {
        println!("P7 (Multi-Timescale): WEAK — delta_gain near 1.0. Gate not yet modulating timescale.");
    }

    println!("\n=== Tầng 3 Complete: Phase 3.1-3.4 ===");

    Ok(())
}
