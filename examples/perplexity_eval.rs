//! Perplexity Eval — Does Nous Actually Help?
//!
//! Run: `cargo run --features candle --example perplexity_eval`
//!
//! Feeds reference texts through mamba-130m token-by-token.
//! Three-way comparison:
//! - Baseline: no modulation
//! - Regex: arousal from emotional.rs regex → uniform delta modulation
//! - HS: arousal from SSM hidden state churn → per-token delta modulation
//!
//! Lower perplexity = model predicts text better.

#[cfg(feature = "candle")]
use noos::cognition::delta_modulation::compute_delta_modulation;
#[cfg(feature = "candle")]
use noos::cognition::hs_arousal::arousal_from_hs;
#[cfg(feature = "candle")]
use noos::inference::cognitive_model::CognitiveModel;
#[cfg(feature = "candle")]
use noos::inference::mamba::{compute_hs_stats, CognitiveMambaModel, HfTokenizer, MambaConfig};
#[cfg(feature = "candle")]
use noos::inference::model::LocalModel;
#[cfg(feature = "candle")]
use noos::inference::tokenizer::NousTokenizer;
#[cfg(feature = "candle")]
use noos::math::softmax::softmax_f32;
#[cfg(feature = "candle")]
use noos::session::CognitiveSession;
#[cfg(feature = "candle")]
use noos::types::intervention::CognitiveState;

fn main() {
    #[cfg(not(feature = "candle"))]
    {
        eprintln!("Requires `candle` feature: cargo run --features candle --example perplexity_eval");
        return;
    }

    #[cfg(feature = "candle")]
    run();
}

#[cfg(feature = "candle")]
fn run() {
    println!("=== Nous Perplexity Eval: Three-Way Comparison ===\n");
    println!("Baseline (no modulation) vs Regex (emotional.rs) vs HS (hidden state churn)\n");

    let model_id = "state-spaces/mamba-130m-hf";
    let config = MambaConfig::mamba_130m();

    println!("Loading tokenizer...");
    let tokenizer = match HfTokenizer::from_pretrained(model_id) {
        Ok(t) => t,
        Err(e) => { eprintln!("Tokenizer failed: {e}"); return; }
    };

    println!("Loading model...");
    let mut model = match CognitiveMambaModel::from_pretrained(model_id, config) {
        Ok(m) => {
            println!("  {} layers, ready.\n", m.num_layers());
            m
        }
        Err(e) => { eprintln!("Model failed: {e}"); return; }
    };

    let num_layers = model.num_layers();

    // ── Reference texts ──
    let categories = vec![
        (
            "Emotional",
            "I am feeling extremely stressed and anxious about the upcoming deadline. \
             Everything seems to be going wrong and I cannot focus on anything. \
             My heart is racing and I feel overwhelmed by the pressure. \
             I need help dealing with this terrible situation before it gets worse.",
        ),
        (
            "Technical",
            "The binary search algorithm works by repeatedly dividing the search interval \
             in half. If the value of the search key is less than the item in the middle \
             of the interval, narrow the interval to the lower half. Otherwise, narrow it \
             to the upper half. Repeatedly check until the value is found or the interval is empty.",
        ),
        (
            "Creative",
            "In a world where colors had sounds and music had shape, there lived a painter \
             who could hear the sunset sing. Every evening she would climb the tallest hill \
             and listen to the sky transform from gold to crimson, each hue a different note \
             in an endless symphony that only she could perceive.",
        ),
        (
            "Routine",
            "Hello, how are you doing today? I hope you are having a good day. \
             The weather has been quite nice lately. I was thinking about going for \
             a walk in the park this afternoon. Maybe we could grab some coffee later \
             if you are free. Let me know what works for you.",
        ),
    ];

    println!("{:<12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "Category", "Base PPL", "Regex%", "HS%", "RegGain", "AvgChurn", "RegAro", "HSAro");
    println!("{}", "=".repeat(88));

    #[allow(dead_code)]
    struct EvalResult {
        name: String,
        baseline_ppl: f64,
        regex_ppl: f64,
        hs_ppl: f64,
        regex_gain: f64,
        regex_arousal: f64,
        avg_churn: f64,
        avg_hs_arousal: f64,
        tokens: usize,
    }

    let mut results: Vec<EvalResult> = Vec::new();

    for (name, text) in &categories {
        let tokens = match tokenizer.encode(text, false) {
            Ok(t) if t.len() >= 5 => t,
            _ => { println!("  {name}: tokenization failed, skip"); continue; }
        };

        // Process through CognitiveSession to get regex-based cognitive state.
        let mut session = CognitiveSession::with_model_layers(num_layers);
        let turn = session.process_message(text);
        let cognitive_state = turn.cognitive_state;
        let regex_delta_mod = compute_delta_modulation(&cognitive_state, num_layers);

        // ── Pass 1: Baseline (no modulation) ──
        model.reset_cache();
        let baseline_ce = compute_avg_cross_entropy(&mut model, &tokens, None);

        // ── Pass 2: Regex arousal → uniform delta modulation ──
        model.reset_cache();
        let regex_ce = compute_avg_cross_entropy(&mut model, &tokens, Some(&regex_delta_mod));

        // ── Pass 3: HS arousal → per-token delta modulation ──
        model.reset_cache();
        let (hs_ce, avg_churn, avg_hs_arousal) =
            compute_avg_cross_entropy_hs(&mut model, &tokens, num_layers);

        let baseline_ppl = baseline_ce.exp();
        let regex_ppl = regex_ce.exp();
        let hs_ppl = hs_ce.exp();
        let regex_pct = pct_change(baseline_ppl, regex_ppl);
        let hs_pct = pct_change(baseline_ppl, hs_ppl);

        println!("{:<12} {:>10.1} {:>+9.2}% {:>+9.2}% {:>10.4} {:>10.4} {:>8.3} {:>8.3}",
            name, baseline_ppl, regex_pct, hs_pct,
            regex_delta_mod.gain_factor, avg_churn,
            cognitive_state.arousal, avg_hs_arousal);

        results.push(EvalResult {
            name: name.to_string(),
            baseline_ppl,
            regex_ppl,
            hs_ppl,
            regex_gain: regex_delta_mod.gain_factor,
            regex_arousal: cognitive_state.arousal,
            avg_churn,
            avg_hs_arousal,
            tokens: tokens.len(),
        });
    }

    // ── Verdict ──
    println!("\n{}", "=".repeat(88));
    println!("VERDICT\n");

    for r in &results {
        let regex_chg = pct_change(r.baseline_ppl, r.regex_ppl);
        let hs_chg = pct_change(r.baseline_ppl, r.hs_ppl);
        let better = if hs_chg < regex_chg { "HS BETTER" }
            else if hs_chg > regex_chg { "REGEX BETTER" }
            else { "EQUAL" };
        println!("  {:<12} Regex: {:>+7.2}%  HS: {:>+7.2}%  → {:<12} [churn={:.3}, {} tokens]",
            r.name, regex_chg, hs_chg, better, r.avg_churn, r.tokens);
    }

    // Key question: does HS extend benefit to non-emotional categories?
    println!();
    let non_emotional_hs_helps = results.iter()
        .filter(|r| r.name != "Emotional")
        .any(|r| pct_change(r.baseline_ppl, r.hs_ppl) < -0.01);

    if non_emotional_hs_helps {
        println!("  HS extends benefit to non-emotional text → SSM readout works!");
    } else {
        println!("  HS does not yet help non-emotional text → calibration may be needed.");
        println!("  Check avg_churn values to tune CHURN_FLOOR/CEILING in hs_arousal.rs.");
    }
}

#[cfg(feature = "candle")]
fn pct_change(baseline: f64, modulated: f64) -> f64 {
    if baseline.abs() > 1e-10 {
        (modulated - baseline) / baseline * 100.0
    } else {
        0.0
    }
}

/// Compute average cross-entropy: baseline or uniform regex delta modulation.
#[cfg(feature = "candle")]
fn compute_avg_cross_entropy(
    model: &mut CognitiveMambaModel,
    tokens: &[u32],
    delta_mod: Option<&noos::types::intervention::DeltaModulation>,
) -> f64 {
    if tokens.len() < 2 {
        return 0.0;
    }

    let mut total_ce = 0.0;
    let mut count = 0;

    for i in 0..tokens.len() - 1 {
        let input_token = tokens[i];
        let target_token = tokens[i + 1] as usize;

        let logits = match delta_mod {
            Some(dm) => {
                match model.forward_cognitive(&[input_token], i, dm) {
                    Ok(result) => result.logits,
                    Err(_) => continue,
                }
            }
            None => {
                match model.forward(&[input_token], i) {
                    Ok(l) => l,
                    Err(_) => continue,
                }
            }
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

/// Compute average cross-entropy with per-token HS-derived delta modulation.
///
/// For each token: read hs stats from previous token's SSM state → compute
/// arousal from churn → build CognitiveState → compute per-token delta mod.
///
/// One-token delay is biologically correct: LC modulates the NEXT cycle
/// based on current cortical state (Grella 2024).
///
/// Returns: (avg_cross_entropy, avg_churn, avg_hs_arousal)
#[cfg(feature = "candle")]
fn compute_avg_cross_entropy_hs(
    model: &mut CognitiveMambaModel,
    tokens: &[u32],
    num_layers: usize,
) -> (f64, f64, f64) {
    use noos::cognition::delta_modulation::compute_layer_targets;
    use noos::cognition::locus_coeruleus::LocusCoeruleus;

    if tokens.len() < 2 {
        return (0.0, 0.0, 0.0);
    }

    let layer_target = compute_layer_targets(num_layers);
    let mut lc = LocusCoeruleus::new();
    let mut total_ce = 0.0;
    let mut total_churn = 0.0;
    let mut total_hs_arousal = 0.0;
    let mut count = 0;
    let mut hs_count = 0;
    let mut prev_snapshot: Option<Vec<f64>> = None;
    let mut prev_arousal = 0.0_f64;
    // Churn from the PREVIOUS token's forward pass — feeds arousal for current token.
    // One-token delay: biologically correct (LC modulates NEXT cycle, not current).
    let mut pending_arousal: Option<f64> = None;

    for i in 0..tokens.len() - 1 {
        let input_token = tokens[i];
        let target_token = tokens[i + 1] as usize;

        // Use arousal computed from PREVIOUS token's churn for this token's modulation.
        let delta_mod = if let Some(hs_arousal) = pending_arousal {
            lc.set_arousal(hs_arousal);
            let sensory_pe = (hs_arousal - prev_arousal).abs();
            prev_arousal = hs_arousal;

            let cog_state = CognitiveState {
                arousal: hs_arousal,
                gain_mode: lc.gain_mode(),
                sensory_pe,
                ..CognitiveState::default()
            };
            Some(compute_delta_modulation(&cog_state, num_layers))
        } else {
            None // First token: no previous churn, no modulation.
        };

        // Forward one token.
        let logits = match &delta_mod {
            Some(dm) => {
                match model.forward_cognitive(&[input_token], i, dm) {
                    Ok(result) => result.logits,
                    Err(_) => continue,
                }
            }
            None => {
                match model.forward(&[input_token], i) {
                    Ok(l) => l,
                    Err(_) => continue,
                }
            }
        };

        // AFTER forward: compute churn by comparing current state with prev snapshot.
        // This captures how much THIS token changed the SSM state.
        // The resulting arousal feeds the NEXT token (one-token delay).
        match compute_hs_stats(model.state(), prev_snapshot.as_deref(), &layer_target) {
            Ok((stats, snapshot)) => {
                if stats.valid {
                    let hs_arousal = arousal_from_hs(&stats).unwrap_or(0.0);
                    total_churn += stats.state_churn;
                    total_hs_arousal += hs_arousal;
                    hs_count += 1;
                    pending_arousal = Some(hs_arousal);
                }
                prev_snapshot = Some(snapshot);
            }
            Err(_) => {}
        }

        if target_token >= logits.len() {
            continue;
        }

        let probs = softmax_f32(&logits);
        let target_prob = probs[target_token] as f64;
        let ce = -(target_prob.max(1e-10)).ln();
        total_ce += ce;
        count += 1;
    }

    let avg_ce = if count == 0 { 0.0 } else { total_ce / count as f64 };
    let avg_churn = if hs_count == 0 { 0.0 } else { total_churn / hs_count as f64 };
    let avg_hs_arousal = if hs_count == 0 { 0.0 } else { total_hs_arousal / hs_count as f64 };

    (avg_ce, avg_churn, avg_hs_arousal)
}
