//! Bottleneck Steering Eval — Tầng 4 structural compensation.
//!
//! Run: `cargo run --features candle --example bottleneck_eval`
//!
//! Phase 1: Calibrates bottleneck channels via variance analysis, then
//! steers only high-variance (delta-sensitive) channels at Layer 20.
//!
//! Paper: Mohan et al. 2026 (arXiv 2602.22719):
//! - 435/768 high-variance channels → scale 5×
//! - 155/768 moderate channels → scale 2×
//! - Remaining → no scaling
//! Result: +8.27% avg across 5 SSMs / 6 benchmarks.

#[cfg(feature = "candle")]
use noos::cognition::delta_modulation::compute_delta_modulation;
#[cfg(feature = "candle")]
use noos::inference::bottleneck::{
    compute_channel_variance, BottleneckConfig, BottleneckSteering, CalibrationResult,
};
#[cfg(feature = "candle")]
use noos::inference::cognitive_model::CognitiveModel;
#[cfg(feature = "candle")]
use noos::inference::mamba::{CognitiveMambaModel, HfTokenizer, MambaConfig};
#[cfg(feature = "candle")]
use noos::inference::model::LocalModel;
#[cfg(feature = "candle")]
use noos::inference::tokenizer::NousTokenizer;
#[cfg(feature = "candle")]
use noos::math::softmax::softmax_f32;
#[cfg(feature = "candle")]
use noos::session::CognitiveSession;

fn main() {
    #[cfg(not(feature = "candle"))]
    {
        eprintln!("Requires `candle` feature: cargo run --features candle --example bottleneck_eval");
        return;
    }

    #[cfg(feature = "candle")]
    run();
}

#[cfg(feature = "candle")]
fn run() {
    println!("=== Bottleneck Steering Eval (Tầng 4) ===\n");
    println!("Paper: Mohan et al. 2026, arXiv 2602.22719");
    println!("Method: calibrate delta-sensitive channels, scale selectively\n");

    let model_id = "state-spaces/mamba-130m-hf";
    let config = MambaConfig::mamba_130m();
    let target_layer = 20;
    let d_model = 768;

    println!("Loading tokenizer...");
    let tokenizer = match HfTokenizer::from_pretrained(model_id) {
        Ok(t) => t,
        Err(e) => { eprintln!("Tokenizer failed: {e}"); return; }
    };

    println!("Loading model...");
    let mut model = match CognitiveMambaModel::from_pretrained(model_id, config) {
        Ok(m) => {
            println!("  {} layers, d_model={}, ready.\n", m.num_layers(), d_model);
            m
        }
        Err(e) => { eprintln!("Model failed: {e}"); return; }
    };

    // ── Step 1: Calibration — find delta-sensitive channels ──
    println!("=== CALIBRATION ===");
    println!("Running calibration text through Layer {} to identify delta-sensitive channels...",
        target_layer);

    // Calibration corpus: diverse text to capture activation variance.
    let calibration_texts = vec![
        "The president announced new economic policies that would affect millions of workers.",
        "In quantum mechanics, the wave function collapses upon measurement.",
        "She walked through the rain-soaked streets, remembering their last conversation.",
        "The recursive algorithm has O(n log n) time complexity for the average case.",
        "Anger surged through him as he read the message, his hands trembling.",
        "Add two cups of flour, one egg, and a tablespoon of vanilla extract.",
        "The ancient temple stood silently among the towering trees of the forest.",
        "Machine learning models require careful hyperparameter tuning for optimal results.",
    ];

    let mut mixer_samples: Vec<Vec<f32>> = Vec::new();

    for text in &calibration_texts {
        let tokens = match tokenizer.encode(text, false) {
            Ok(t) if t.len() >= 3 => t,
            _ => continue,
        };

        model.reset_cache();
        for &token_id in &tokens {
            match model.forward_capture_mixer(token_id, target_layer) {
                Ok((_logits, mixer_out)) if !mixer_out.is_empty() => {
                    mixer_samples.push(mixer_out);
                }
                _ => {}
            }
        }
    }

    println!("  Collected {} token samples across {} texts.",
        mixer_samples.len(), calibration_texts.len());

    let variances = compute_channel_variance(&mixer_samples, d_model);
    let calibration = CalibrationResult {
        channel_variances: variances.clone(),
        num_tokens: mixer_samples.len(),
        layer_index: target_layer,
    };

    // Stats on variance distribution.
    let max_var = variances.iter().cloned().fold(0.0_f64, f64::max);
    let min_var = variances.iter().cloned().fold(f64::MAX, f64::min);
    let mean_var: f64 = variances.iter().sum::<f64>() / variances.len() as f64;
    println!("  Channel variance: min={:.6}, mean={:.6}, max={:.6}", min_var, mean_var, max_var);

    // ── Step 2: Build calibrated configs ──
    // Paper defaults: 57% channels at 5×, 20% at 2×, rest at 1×.
    let calibrated_5x = calibration.to_config_paper_defaults();
    let num_5x = calibrated_5x.channel_scales.iter().filter(|&&s| (s - 5.0).abs() < 0.01).count();
    let num_2x = calibrated_5x.channel_scales.iter().filter(|&&s| (s - 2.0).abs() < 0.01).count();
    let num_1x = calibrated_5x.channel_scales.iter().filter(|&&s| (s - 1.0).abs() < 0.01).count();
    println!("  Calibrated config: {} channels at 5×, {} at 2×, {} at 1×\n",
        num_5x, num_2x, num_1x);

    // Also test compensatory variant: high-variance channels at 0.8×.
    let calibrated_comp = calibration.to_config(0.57, 0.8, 0.20, 0.9);

    // ── Reference texts ──
    let categories = vec![
        ("Emotional",
         "I am feeling extremely stressed and anxious about the upcoming deadline. \
          Everything seems to be going wrong and I cannot focus on anything. \
          My heart is racing and I feel overwhelmed by the pressure. \
          I need help dealing with this terrible situation before it gets worse."),
        ("Technical",
         "The binary search algorithm works by repeatedly dividing the search interval \
          in half. If the value of the search key is less than the item in the middle \
          of the interval, narrow the interval to the lower half. Otherwise, narrow it \
          to the upper half. Repeatedly check until the value is found or the interval is empty."),
        ("Creative",
         "In a world where colors had sounds and music had shape, there lived a painter \
          who could hear the sunset sing. Every evening she would climb the tallest hill \
          and listen to the sky transform from gold to crimson, each hue a different note \
          in an endless symphony that only she could perceive."),
        ("Routine",
         "Hello, how are you doing today? I hope you are having a good day. \
          The weather has been quite nice lately. I was thinking about going for \
          a walk in the park this afternoon. Maybe we could grab some coffee later \
          if you are free. Let me know what works for you."),
    ];

    // ── Step 3: Eval ──
    println!("=== PERPLEXITY EVAL ===\n");
    println!("{:<12} {:>10} {:>12} {:>10} {:>12} {:>10} {:>12} {:>10}",
        "Category", "Baseline", "Uniform 5×", "Δ", "Calib 5×/2×", "Δ", "Calib comp", "Δ");
    println!("{}", "=".repeat(100));

    struct Configs {
        name: &'static str,
        config: BottleneckConfig,
    }

    let configs = vec![
        Configs { name: "Uniform 5×", config: BottleneckConfig::uniform(target_layer, d_model, 5.0) },
        Configs { name: "Calib 5×/2×", config: calibrated_5x },
        Configs { name: "Calib comp", config: calibrated_comp },
    ];

    for (name, text) in &categories {
        let tokens = match tokenizer.encode(text, false) {
            Ok(t) if t.len() >= 5 => t,
            _ => continue,
        };

        // Baseline.
        model.reset_cache();
        model.clear_bottleneck();
        let baseline_ce = compute_avg_cross_entropy(&mut model, &tokens);
        let baseline_ppl = baseline_ce.exp();

        print!("{:<12} {:>10.1}", name, baseline_ppl);

        for cfg in &configs {
            let steering = match BottleneckSteering::new(cfg.config.clone(), &candle_core::Device::Cpu) {
                Ok(s) => s,
                Err(e) => { print!("  ERR:{e}"); continue; }
            };

            model.reset_cache();
            model.set_bottleneck(steering);
            let ce = compute_avg_cross_entropy(&mut model, &tokens);
            let ppl = ce.exp();
            let change = (ppl - baseline_ppl) / baseline_ppl * 100.0;

            print!(" {:>12.1} {:>+9.2}%", ppl, change);
        }
        println!();
    }

    // ── Step 4: Compound test — Tầng 2 + Tầng 4 ──
    println!("\n=== COMPOUND TEST: Tầng 2 (delta) + Tầng 4 (bottleneck) ===\n");
    println!("{:<12} {:>10} {:>12} {:>10} {:>12} {:>10} {:>12} {:>10}",
        "Category", "Baseline", "T2 only", "Δ", "T4 only", "Δ", "T2+T4", "Δ");
    println!("{}", "=".repeat(100));

    let num_layers = model.num_layers();

    for (name, text) in &categories {
        let tokens = match tokenizer.encode(text, false) {
            Ok(t) if t.len() >= 5 => t,
            _ => continue,
        };

        // Get cognitive state for this text.
        let mut session = CognitiveSession::with_model_layers(num_layers);
        let turn = session.process_message(text);
        let cognitive_state = turn.cognitive_state;
        let delta_mod = compute_delta_modulation(&cognitive_state, num_layers);

        // Baseline (no modulation).
        model.reset_cache();
        model.clear_bottleneck();
        let baseline_ppl = compute_avg_cross_entropy(&mut model, &tokens).exp();

        // Tầng 2 only (delta modulation at mid-layers).
        model.reset_cache();
        model.clear_bottleneck();
        let t2_ppl = compute_avg_cross_entropy_cognitive(&mut model, &tokens, &delta_mod).exp();
        let t2_change = (t2_ppl - baseline_ppl) / baseline_ppl * 100.0;

        // Tầng 4 only (calibrated compensatory at Layer 20).
        let comp_cfg = calibration.to_config(0.57, 0.8, 0.20, 0.9);
        let steering = BottleneckSteering::new(comp_cfg, &candle_core::Device::Cpu).unwrap();
        model.reset_cache();
        model.set_bottleneck(steering);
        let t4_ppl = compute_avg_cross_entropy(&mut model, &tokens).exp();
        let t4_change = (t4_ppl - baseline_ppl) / baseline_ppl * 100.0;

        // Compound: Tầng 2 + Tầng 4 simultaneously.
        let comp_cfg2 = calibration.to_config(0.57, 0.8, 0.20, 0.9);
        let steering2 = BottleneckSteering::new(comp_cfg2, &candle_core::Device::Cpu).unwrap();
        model.reset_cache();
        model.set_bottleneck(steering2);
        let compound_ppl = compute_avg_cross_entropy_cognitive(&mut model, &tokens, &delta_mod).exp();
        let compound_change = (compound_ppl - baseline_ppl) / baseline_ppl * 100.0;

        println!("{:<12} {:>10.1} {:>12.1} {:>+9.2}% {:>12.1} {:>+9.2}% {:>12.1} {:>+9.2}%",
            name, baseline_ppl, t2_ppl, t2_change, t4_ppl, t4_change, compound_ppl, compound_change);
    }

    // ── Verdict ──
    println!("\n{}", "=".repeat(100));
    println!("ANALYSIS\n");
    println!("  T2 only: Tầng 2 delta modulation at layers 10-14 (gain from cognitive state).");
    println!("  T4 only: Tầng 4 calibrated compensatory (0.8×/0.9× selective at Layer 20).");
    println!("  T2+T4: Both active simultaneously — do they compound?");
    println!();
    println!("  If T2+T4 < T2 alone → Tầng 4 adds value on top of Tầng 2.");
    println!("  If T2+T4 ≈ T2 → Tầng 4 doesn't contribute when T2 is active.");
    println!("  If T2+T4 > T2 → Tầng 4 interferes with T2 (they conflict).");

    model.clear_bottleneck();
}

/// Compute average cross-entropy over reference tokens.
#[cfg(feature = "candle")]
fn compute_avg_cross_entropy(
    model: &mut CognitiveMambaModel,
    tokens: &[u32],
) -> f64 {
    if tokens.len() < 2 {
        return 0.0;
    }

    let mut total_ce = 0.0;
    let mut count = 0;

    for i in 0..tokens.len() - 1 {
        let input_token = tokens[i];
        let target_token = tokens[i + 1] as usize;

        let logits = match model.forward(&[input_token], i) {
            Ok(l) => l,
            Err(_) => continue,
        };

        if target_token >= logits.len() {
            continue;
        }

        let probs = softmax_f32(&logits);
        let target_prob = probs[target_token] as f64;
        let ce = -(target_prob.max(1e-10)).ln();
        total_ce += ce;
        count += 1;
    }

    if count == 0 { 0.0 } else { total_ce / count as f64 }
}

/// Compute average cross-entropy with cognitive delta modulation (Tầng 2).
#[cfg(feature = "candle")]
fn compute_avg_cross_entropy_cognitive(
    model: &mut CognitiveMambaModel,
    tokens: &[u32],
    delta_mod: &noos::types::intervention::DeltaModulation,
) -> f64 {
    if tokens.len() < 2 {
        return 0.0;
    }

    let mut total_ce = 0.0;
    let mut count = 0;

    for i in 0..tokens.len() - 1 {
        let input_token = tokens[i];
        let target_token = tokens[i + 1] as usize;

        let logits = match model.forward_cognitive(&[input_token], i, delta_mod) {
            Ok(result) => result.logits,
            Err(_) => continue,
        };

        if target_token >= logits.len() {
            continue;
        }

        let probs = softmax_f32(&logits);
        let target_prob = probs[target_token] as f64;
        let ce = -(target_prob.max(1e-10)).ln();
        total_ce += ce;
        count += 1;
    }

    if count == 0 { 0.0 } else { total_ce / count as f64 }
}
