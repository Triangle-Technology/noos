//! Hidden State Probe — what does mamba-130m already know?
//!
//! Run: `cargo run --release --features candle --example probe_hidden_states`
//!
//! This analysis feeds different text types through a pretrained Mamba model
//! and records activations at every layer. The goal: understand what cognitive
//! information the model already encodes in its hidden states BEFORE any
//! Nous intervention.
//!
//! From intervention.md: "Model đã có implicit cognitive signals từ training."
//! This probe answers: What signals? Where? How strong?
//!
//! Parts:
//! 1. Content differentiation by layer — at which depth does the model
//!    start distinguishing emotional vs factual vs question text?
//! 2. Gate position analysis — is layer 12 (50% depth) optimal for reading?
//! 3. Gate W_read analysis — what dimensions did the trained gate learn to read?
//! 4. SSM state analysis — does recurrent memory differentiate content?

use nous_regulator::errors::NousResult;
use nous_regulator::inference::cognitive_gate::CognitiveGateConfig;
use nous_regulator::inference::mamba::{CognitiveMambaWithGate, HfTokenizer, MambaConfig};
use nous_regulator::inference::model::LocalModel;
use nous_regulator::inference::tokenizer::NousTokenizer;
use nous_regulator::math::vector::cosine_similarity;

/// Test prompts covering different cognitive contexts.
/// Each should activate different processing patterns if the model
/// has learned content-dependent representations.
const PROBES: &[(&str, &str)] = &[
    ("Factual", "The hippocampus is a brain region involved in the formation of new memories and spatial navigation"),
    ("Emotional", "I feel terrified and overwhelmed, my heart is racing and I cannot stop the panic from spreading"),
    ("Question", "How do neurons communicate across synapses and what role do neurotransmitters play in this process"),
    ("Creative", "Imagine a vast ocean of light where every wave carries a forgotten dream toward an infinite shore"),
    ("Technical", "The function iterates through the hash map, computing the running average with exponential decay"),
    ("Social", "Hey thanks so much for helping me yesterday, I really appreciate you taking the time to explain"),
];

fn main() -> NousResult<()> {
    println!("=== Nous: Hidden State Probe ===\n");

    let model_id = "state-spaces/mamba-130m-hf";
    let config = MambaConfig::mamba_130m();
    let gate_config = CognitiveGateConfig::from_mamba_config(&config);

    println!("Model: {model_id} ({} layers, d_model={})", config.n_layer, config.d_model);
    println!("Gate position: layer {}\n", gate_config.gate_position);

    // Load model with trained gate.
    println!("Loading model...");
    let (mut model, _varmap) =
        CognitiveMambaWithGate::from_pretrained_with_gate(model_id, config.clone(), gate_config)?;
    let tokenizer = HfTokenizer::from_pretrained(model_id)?;

    // ═══════════════════════════════════════════════════════════════
    // Collect probe results for all text types
    // ═══════════════════════════════════════════════════════════════

    println!("Probing {} text types...\n", PROBES.len());

    let mut all_results = Vec::new();
    for (label, text) in PROBES {
        model.reset_cache();
        let tokens = tokenizer.encode(text, false)?;
        let result = model
            .forward_probe(&tokens)
            .map_err(|e| nous_regulator::NousError::Internal(format!("Probe error for {label}: {e}")))?;
        all_results.push((*label, result));
    }

    // ═══════════════════════════════════════════════════════════════
    // Part 1: Content differentiation by layer
    // ═══════════════════════════════════════════════════════════════

    println!("=== Part 1: Content Differentiation by Layer ===\n");
    println!("Cosine similarity between text types at each layer depth.");
    println!("Lower similarity = model distinguishes them more.\n");

    let n_layers = all_results[0].1.layer_activations.len();
    let n_probes = all_results.len();

    // Print header.
    print!("{:>7}", "Layer");
    for i in 0..n_probes {
        for j in (i + 1)..n_probes {
            print!("  {:>4}-{:<4}", &all_results[i].0[..4], &all_results[j].0[..4]);
        }
    }
    println!("  {:>8}", "Avg Sim");
    println!("{}", "-".repeat(7 + 12 * (n_probes * (n_probes - 1) / 2) + 10));

    // Track which layer has lowest average similarity (most differentiation).
    let mut min_avg_sim = f64::INFINITY;
    let mut best_layer = 0;

    for layer in 0..n_layers {
        print!("{:>7}", layer);
        let mut sim_sum = 0.0f64;
        let mut sim_count = 0;

        for i in 0..n_probes {
            for j in (i + 1)..n_probes {
                let act_i = &all_results[i].1.layer_activations[layer];
                let act_j = &all_results[j].1.layer_activations[layer];
                let sim = cosine_similarity(act_i, act_j) as f64;
                print!("  {:>10.4}", sim);
                sim_sum += sim;
                sim_count += 1;
            }
        }

        let avg_sim = sim_sum / sim_count as f64;
        print!("  {:>8.4}", avg_sim);

        if layer == 11 {
            print!("  <-- gate position");
        }
        if avg_sim < min_avg_sim {
            min_avg_sim = avg_sim;
            best_layer = layer;
        }
        println!();
    }

    println!("\nMost differentiating layer: {} (avg sim = {:.4})", best_layer, min_avg_sim);
    println!(
        "Gate is at layer 12. {} for reading cognitive state.",
        if (best_layer as i32 - 12).unsigned_abs() <= 3 {
            "Good position"
        } else {
            "Consider repositioning gate"
        }
    );

    // ═══════════════════════════════════════════════════════════════
    // Part 2: Gate-level analysis
    // ═══════════════════════════════════════════════════════════════

    println!("\n=== Part 2: Gate Output by Text Type ===\n");
    println!(
        "{:<12} {:>10} {:>12} {:>10}",
        "Type", "Alpha", "Delta Gain", "CogSig L2"
    );
    println!("{}", "-".repeat(48));

    for (label, result) in &all_results {
        let cog_l2: f64 = result
            .gate_cog_signal
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt();
        println!(
            "{:<12} {:>10.4} {:>12.4} {:>10.4}",
            label, result.gate_alpha, result.gate_delta_gain, cog_l2
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Part 3: Cognitive signal similarity
    // ═══════════════════════════════════════════════════════════════

    println!("\n=== Part 3: Cognitive Signal Similarity (Gate W_read output) ===\n");
    println!("How similar are gate's 'readings' for different text types?\n");

    print!("{:<12}", "");
    for (label, _) in &all_results {
        print!(" {:>10}", &label[..label.len().min(10)]);
    }
    println!();

    for i in 0..n_probes {
        print!("{:<12}", all_results[i].0);
        for j in 0..n_probes {
            let sim = cosine_similarity(
                &all_results[i].1.gate_cog_signal,
                &all_results[j].1.gate_cog_signal,
            );
            print!(" {:>10.4}", sim);
        }
        println!();
    }

    // ═══════════════════════════════════════════════════════════════
    // Part 4: SSM state at gate layer
    // ═══════════════════════════════════════════════════════════════

    println!("\n=== Part 4: SSM State Similarity at Gate Layer ===\n");
    println!("Does the SSM recurrent memory differentiate content?\n");

    print!("{:<12}", "");
    for (label, _) in &all_results {
        print!(" {:>10}", &label[..label.len().min(10)]);
    }
    println!();

    for i in 0..n_probes {
        print!("{:<12}", all_results[i].0);
        for j in 0..n_probes {
            let sim = cosine_similarity(
                &all_results[i].1.ssm_state_at_gate,
                &all_results[j].1.ssm_state_at_gate,
            );
            print!(" {:>10.4}", sim);
        }
        println!();
    }

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════

    println!("\n=== Summary ===\n");

    // Compute average residual stream similarity at gate layer vs last layer.
    let gate_layer = 11.min(n_layers - 1);
    let last_layer = n_layers - 1;

    let mut gate_sim_sum = 0.0;
    let mut last_sim_sum = 0.0;
    let mut pair_count = 0;
    for i in 0..n_probes {
        for j in (i + 1)..n_probes {
            gate_sim_sum += cosine_similarity(
                &all_results[i].1.layer_activations[gate_layer],
                &all_results[j].1.layer_activations[gate_layer],
            ) as f64;
            last_sim_sum += cosine_similarity(
                &all_results[i].1.layer_activations[last_layer],
                &all_results[j].1.layer_activations[last_layer],
            ) as f64;
            pair_count += 1;
        }
    }
    let gate_avg = gate_sim_sum / pair_count as f64;
    let last_avg = last_sim_sum / pair_count as f64;

    println!("Avg similarity at gate layer ({}): {:.4}", gate_layer, gate_avg);
    println!("Avg similarity at final layer ({}): {:.4}", last_layer, last_avg);

    if gate_avg < last_avg {
        println!("Gate layer MORE differentiating than final layer.");
    } else {
        println!("Final layer more differentiating — deeper = more specialized.");
    }

    println!("\nThese results inform Tầng 4 design:");
    println!("- If layers show strong differentiation → model already computes cognitive signals");
    println!("- Gate should read at the layer with maximum content differentiation");
    println!("- SSM state vs residual stream tells us where cognitive memory lives");

    Ok(())
}
