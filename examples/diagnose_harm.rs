//! Diagnose: WHY does delta modulation hurt?
//!
//! Run: `cargo run --features candle --example diagnose_harm`
//!
//! Tests 3 hypotheses:
//! H1: Gain=1.2 too aggressive → try micro-gains (1.01, 1.05, 1.10)
//! H2: Mid-layer targeting wrong → try early, mid, late, all layers
//! H3: ALL external modulation hurts → if every gain≠1.0 hurts proportionally
//!
//! Uses emotional text from perplexity_eval (the text that showed +5.21% harm).

#[cfg(feature = "candle")]
use noos::inference::cognitive_model::CognitiveModel;
#[cfg(feature = "candle")]
use noos::inference::mamba::{CognitiveMambaModel, HfTokenizer, MambaConfig};
#[cfg(feature = "candle")]
use noos::inference::model::LocalModel;
#[cfg(feature = "candle")]
use noos::inference::tokenizer::NoosTokenizer;
#[cfg(feature = "candle")]
use noos::math::softmax::softmax_f32;
#[cfg(feature = "candle")]
use noos::types::intervention::{DeltaModulation, DeltaModulationSource, LayerTarget};

fn main() {
    #[cfg(not(feature = "candle"))]
    eprintln!("Requires `candle` feature.");

    #[cfg(feature = "candle")]
    run();
}

#[cfg(feature = "candle")]
fn run() {
    println!("=== Diagnose: WHY Does Delta Modulation Hurt? ===\n");

    let model_id = "state-spaces/mamba-130m-hf";
    let config = MambaConfig::mamba_130m();

    println!("Loading...");
    let tokenizer = HfTokenizer::from_pretrained(model_id).expect("tokenizer");
    let mut model = CognitiveMambaModel::from_pretrained(model_id, config).expect("model");
    let num_layers = model.num_layers(); // 24
    println!("  {} layers ready.\n", num_layers);

    let emotional_text = "I am feeling extremely stressed and anxious about the upcoming deadline. \
         Everything seems to be going wrong and I cannot focus on anything. \
         My heart is racing and I feel overwhelmed by the pressure. \
         I need help dealing with this terrible situation before it gets worse.";

    let tokens = tokenizer.encode(emotional_text, false).expect("tokenize");
    println!("Emotional text: {} tokens\n", tokens.len());

    // ── Baseline ──
    model.reset_cache();
    let baseline_ce = compute_ce(&mut model, &tokens, None);
    let baseline_ppl = baseline_ce.exp();
    println!("Baseline perplexity: {:.2}\n", baseline_ppl);

    // ══════════════════════════════════════════════════════════════
    // H1: Is gain magnitude the problem?
    // ══════════════════════════════════════════════════════════════
    println!("═══ H1: Gain Magnitude (mid-layers 40-60%) ═══\n");
    println!("{:<12} {:>10} {:>10} {:>10}",
        "Gain", "PPL", "Change", "Verdict");
    println!("{}", "-".repeat(44));

    let gains_h1 = vec![
        0.80, 0.90, 0.95, 0.99,
        1.00, // should be identical to baseline
        1.01, 1.05, 1.10, 1.20, 1.50,
    ];

    let mid_start = (num_layers as f64 * 0.4).round() as usize;
    let mid_end = (num_layers as f64 * 0.6).round() as usize;
    let mid_target = LayerTarget {
        start_layer: mid_start,
        end_layer: mid_end,
        total_layers: num_layers,
    };

    let mut h1_results: Vec<(f64, f64)> = Vec::new();

    for &gain in &gains_h1 {
        let dm = DeltaModulation {
            gain_factor: gain,
            target: mid_target.clone(),
            source: DeltaModulationSource::Combined,
        };
        model.reset_cache();
        let ce = compute_ce(&mut model, &tokens, Some(&dm));
        let ppl = ce.exp();
        let change = (ppl - baseline_ppl) / baseline_ppl * 100.0;
        let verdict = if change.abs() < 0.01 { "=" }
            else if change < 0.0 { "HELPS" }
            else { "HURTS" };
        println!("{:<12.2} {:>10.2} {:>+9.2}% {:>10}",
            gain, ppl, change, verdict);
        h1_results.push((gain, change));
    }

    // ══════════════════════════════════════════════════════════════
    // H2: Is layer targeting the problem?
    // ══════════════════════════════════════════════════════════════
    println!("\n═══ H2: Layer Targeting (gain=1.20 fixed) ═══\n");
    println!("{:<20} {:>10} {:>10} {:>10}",
        "Target", "PPL", "Change", "Layers");
    println!("{}", "-".repeat(52));

    let gain_fixed = 1.20;
    let layer_targets: Vec<(&str, usize, usize)> = vec![
        ("Early (0-30%)", 0, (num_layers as f64 * 0.3).round() as usize),
        ("Mid (40-60%)", (num_layers as f64 * 0.4).round() as usize,
                         (num_layers as f64 * 0.6).round() as usize),
        ("Late (70-100%)", (num_layers as f64 * 0.7).round() as usize, num_layers - 1),
        ("All (0-100%)", 0, num_layers - 1),
        ("Single (layer 12)", 12, 12),
    ];

    for (name, start, end) in &layer_targets {
        let target = LayerTarget {
            start_layer: *start,
            end_layer: *end,
            total_layers: num_layers,
        };
        let dm = DeltaModulation {
            gain_factor: gain_fixed,
            target,
            source: DeltaModulationSource::Combined,
        };
        model.reset_cache();
        let ce = compute_ce(&mut model, &tokens, Some(&dm));
        let ppl = ce.exp();
        let change = (ppl - baseline_ppl) / baseline_ppl * 100.0;
        let layer_count = end - start;
        println!("{:<20} {:>10.2} {:>+9.2}% {:>8}/{}",
            name, ppl, change, layer_count, num_layers);
    }

    // ══════════════════════════════════════════════════════════════
    // H3: Is ALL modulation harmful? (proportionality check)
    // ══════════════════════════════════════════════════════════════
    println!("\n═══ H3: Proportionality (does harm scale with deviation?) ═══\n");

    let deviations: Vec<f64> = h1_results.iter()
        .map(|(gain, _)| (gain - 1.0).abs())
        .collect();
    let harms: Vec<f64> = h1_results.iter()
        .map(|(_, change)| *change)
        .collect();

    // Check: do all non-zero deviations produce harm?
    let all_hurt = h1_results.iter()
        .filter(|(gain, _)| (gain - 1.0).abs() > 0.001)
        .all(|(_, change)| *change > 0.0);

    let both_directions_hurt = h1_results.iter()
        .filter(|(gain, _)| *gain < 1.0)
        .any(|(_, change)| *change > 0.0)
        && h1_results.iter()
        .filter(|(gain, _)| *gain > 1.0)
        .any(|(_, change)| *change > 0.0);

    // Simple correlation: deviation vs harm
    let n = deviations.len() as f64;
    let mean_d = deviations.iter().sum::<f64>() / n;
    let mean_h = harms.iter().sum::<f64>() / n;
    let cov: f64 = deviations.iter().zip(harms.iter())
        .map(|(d, h)| (d - mean_d) * (h - mean_h))
        .sum::<f64>() / n;
    let std_d = (deviations.iter().map(|d| (d - mean_d).powi(2)).sum::<f64>() / n).sqrt();
    let std_h = (harms.iter().map(|h| (h - mean_h).powi(2)).sum::<f64>() / n).sqrt();
    let correlation = if std_d > 1e-10 && std_h > 1e-10 { cov / (std_d * std_h) } else { 0.0 };

    println!("  All non-1.0 gains hurt: {}", if all_hurt { "YES" } else { "NO" });
    println!("  Both directions hurt (gain<1 AND gain>1): {}",
        if both_directions_hurt { "YES" } else { "NO" });
    println!("  Correlation(|deviation|, harm): {:.3}", correlation);

    // ── Final Diagnosis ──
    println!("\n{}", "=".repeat(56));
    println!("DIAGNOSIS\n");

    if all_hurt && both_directions_hurt && correlation > 0.5 {
        println!("  H3 CONFIRMED: ALL external modulation hurts.");
        println!("  Harm scales with deviation from 1.0 (r={:.2}).", correlation);
        println!("  The model IS a closed optimized system.");
        println!("  Any delta perturbation → worse prediction.");
        println!();
        println!("  IMPLICATION: Tầng 2 (delta modulation) is fundamentally");
        println!("  the wrong approach for pretrained models.");
    } else if all_hurt {
        println!("  H1+H3 PARTIAL: All tested gains hurt, but correlation");
        println!("  with deviation is weak (r={:.2}).", correlation);
        println!("  Harm exists but pattern unclear.");
    } else {
        // Some gains help!
        let helpers: Vec<_> = h1_results.iter()
            .filter(|(_, change)| *change < -0.01)
            .collect();
        if !helpers.is_empty() {
            println!("  SOME GAINS HELP:");
            for (gain, change) in &helpers {
                println!("    gain={:.2} → {:.2}% (lower perplexity)", gain, change);
            }
            println!("  H1 or H2 may explain the pattern.");
        }
    }
}

#[cfg(feature = "candle")]
fn compute_ce(
    model: &mut CognitiveMambaModel,
    tokens: &[u32],
    delta_mod: Option<&DeltaModulation>,
) -> f64 {
    if tokens.len() < 2 { return 0.0; }

    let mut total_ce = 0.0;
    let mut count = 0;

    for i in 0..tokens.len() - 1 {
        let input = tokens[i];
        let target = tokens[i + 1] as usize;

        let logits = match delta_mod {
            Some(dm) => model.forward_cognitive(&[input], i, dm)
                .map(|r| r.logits).unwrap_or_default(),
            None => model.forward(&[input], i).unwrap_or_default(),
        };

        if target >= logits.len() || logits.is_empty() { continue; }

        let probs = softmax_f32(&logits);
        let p = (probs[target] as f64).max(1e-10);
        total_ce -= p.ln();
        count += 1;
    }

    if count == 0 { 0.0 } else { total_ce / count as f64 }
}
