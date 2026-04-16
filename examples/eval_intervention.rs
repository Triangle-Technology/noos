//! Eval: Does Non-Cortical Processing Matter?
//!
//! Run: `cargo run --features candle --example eval_intervention`
//!
//! Tests whether Noos's subcortical modulation (Tầng 1 + 2) creates
//! measurable, meaningful differences in model behavior.
//!
//! For each scenario:
//! 1. CognitiveSession processes message → TurnResult (arousal, gain, budget, etc.)
//! 2. Model forward pass in 2 modes: baseline vs cognitive (Tầng 2 delta modulation)
//! 3. Metrics: KL divergence, entropy change, top-k overlap
//!
//! Neuroscience predictions:
//! - Phasic gain → entropy should DECREASE (focused)
//! - Tonic gain → entropy should INCREASE (exploratory)
//! - High arousal → delta gain > 1.0 (attend to current input)
//! - Depleted budget → delta gain < 1.0 (conservation)

#[cfg(feature = "candle")]
use noos::cognition::delta_modulation::compute_delta_modulation;
#[cfg(feature = "candle")]
use noos::inference::cognitive_model::CognitiveModel;
#[cfg(feature = "candle")]
use noos::inference::cognitive_gate::CognitiveGateConfig;
#[cfg(feature = "candle")]
use noos::inference::mamba::{CognitiveMambaModel, CognitiveMambaWithGate, HfTokenizer, MambaConfig};
#[cfg(feature = "candle")]
use noos::inference::model::LocalModel;
#[cfg(feature = "candle")]
use noos::inference::tokenizer::NoosTokenizer;
#[cfg(feature = "candle")]
use noos::math::softmax::softmax_f32;
#[cfg(feature = "candle")]
use noos::session::{CognitiveSession, TurnResult};
#[cfg(feature = "candle")]
use noos::types::world::GainMode;

fn main() {
    #[cfg(not(feature = "candle"))]
    eprintln!("Requires `candle` feature: cargo run --features candle --example eval_intervention");

    #[cfg(feature = "candle")]
    run();
}

#[cfg(feature = "candle")]
struct Scenario {
    name: &'static str,
    /// Messages to process (last message is the test message).
    messages: Vec<&'static str>,
    /// What we expect to see.
    prediction: &'static str,
}

#[cfg(feature = "candle")]
struct ScenarioResult {
    name: String,
    // Cognitive state from session
    arousal: f64,
    gain_mode: String,
    body_budget: f64,
    sensory_pe: f64,
    // Tầng 1 params
    temperature: f64,
    top_p: f64,
    freq_penalty: f64,
    // Tầng 2 params
    delta_gain: f64,
    modulated_layers: usize,
    // Metrics
    kl_divergence: f64,
    entropy_baseline: f64,
    entropy_tang2: f64,
    entropy_tang1_baseline: f64,
    entropy_tang1_tang2: f64,
    top10_overlap: usize,
}

#[cfg(feature = "candle")]
fn run() {
    println!("=== Noos Eval: Does Non-Cortical Processing Matter? ===\n");

    // ── Load model ──
    let model_id = "state-spaces/mamba-130m-hf";
    let config = MambaConfig::mamba_130m();

    println!("Loading tokenizer...");
    let tokenizer = match HfTokenizer::from_pretrained(model_id) {
        Ok(t) => t,
        Err(e) => { eprintln!("Tokenizer failed: {e}"); return; }
    };

    println!("Loading model (~500MB, first run downloads)...");
    let mut model = match CognitiveMambaModel::from_pretrained(model_id, config) {
        Ok(m) => {
            println!("  {} layers, ready.\n", m.num_layers());
            m
        }
        Err(e) => { eprintln!("Model failed: {e}"); return; }
    };

    let num_layers = model.num_layers();

    // ── Define scenarios ──
    let scenarios = vec![
        Scenario {
            name: "1. Routine greeting (control)",
            messages: vec!["Hello, how are you?"],
            prediction: "Minimal modulation — gain ~1.0, low arousal",
        },
        Scenario {
            name: "2. Emotional stress",
            messages: vec!["I'm really stressed about this deadline and everything is going wrong!!!"],
            prediction: "High arousal, negative → delta gain > 1.0 (focus on current threat)",
        },
        Scenario {
            name: "3. Creative exploration",
            messages: vec!["What if we could redesign cities completely from scratch, reimagining how people live and interact?"],
            prediction: "Exploratory → if tonic, entropy should increase (broader distribution)",
        },
        Scenario {
            name: "4. Technical question",
            messages: vec!["How does the TCP protocol handle packet loss and retransmission?"],
            prediction: "Focused → if phasic, entropy should decrease (narrower distribution)",
        },
        Scenario {
            name: "5. Vietnamese philosophical",
            messages: vec!["Khám phá ý nghĩa của cuộc sống và sự tồn tại con người"],
            prediction: "Novel + Vietnamese detection → high PE, possible NE burst",
        },
        Scenario {
            name: "6. Multi-turn stress (depleted budget)",
            messages: vec![
                "This is terrible, I lost all my work!",
                "And now the server is down too!",
                "Everything keeps getting worse and I don't know what to do!!!",
            ],
            prediction: "Body budget depleted → delta gain < 1.0 (conservation mode)",
        },
    ];

    let prompt = "The meaning of life is";
    let prompt_tokens = tokenizer.encode(prompt, false).unwrap_or_default();
    if prompt_tokens.is_empty() {
        eprintln!("Failed to tokenize prompt");
        return;
    }

    println!("Probe prompt: \"{prompt}\" ({} tokens)", prompt_tokens.len());
    println!("Comparing: baseline forward() vs cognitive forward_cognitive(delta)\n");
    println!("{}", "=".repeat(72));

    let mut results: Vec<ScenarioResult> = Vec::new();

    for scenario in &scenarios {
        // Fresh session for each scenario.
        let mut session = CognitiveSession::with_model_layers(num_layers);

        // Process all messages through CognitiveSession.
        let mut turn_opt: Option<TurnResult> = None;
        for (i, msg) in scenario.messages.iter().enumerate() {
            let t = session.process_message(msg);
            // Simulate response for multi-turn (so consolidate runs).
            if i < scenario.messages.len() - 1 {
                session.process_response("I understand, that's difficult.", 0.5);
            }
            turn_opt = Some(t);
        }
        let turn = turn_opt.expect("Scenario must have at least one message");

        // ── Baseline: forward() with no cognitive modulation ──
        model.reset_cache();
        let logits_baseline = match model.forward(&prompt_tokens, 0) {
            Ok(l) => l,
            Err(e) => { eprintln!("Baseline forward failed: {e}"); continue; }
        };

        // ── Tầng 2: forward_cognitive() with delta modulation ──
        model.reset_cache();
        let delta_mod = compute_delta_modulation(&turn.cognitive_state, num_layers);
        let result_tang2 = match model.forward_cognitive(&prompt_tokens, 0, &delta_mod) {
            Ok(r) => r,
            Err(e) => { eprintln!("Cognitive forward failed: {e}"); continue; }
        };
        let logits_tang2 = &result_tang2.logits;

        // ── Compute metrics ──

        // Raw logit comparison (Tầng 2 effect on model processing).
        let probs_baseline = softmax_f32(&logits_baseline);
        let probs_tang2 = softmax_f32(logits_tang2);

        let kl = kl_divergence(&probs_baseline, &probs_tang2);
        let ent_baseline = entropy(&probs_baseline);
        let ent_tang2 = entropy(&probs_tang2);
        let overlap = top_k_overlap(&logits_baseline, logits_tang2, 10);

        // Tầng 1 effect: apply temperature to raw logits → compare entropy.
        let temp = turn.sampling.temperature as f32;
        let scaled_baseline: Vec<f32> = logits_baseline.iter().map(|l| l / temp).collect();
        let scaled_tang2: Vec<f32> = logits_tang2.iter().map(|l| l / temp).collect();
        let probs_t1_baseline = softmax_f32(&scaled_baseline);
        let probs_t1_tang2 = softmax_f32(&scaled_tang2);
        let ent_t1_baseline = entropy(&probs_t1_baseline);
        let ent_t1_tang2 = entropy(&probs_t1_tang2);

        // ── Print ──
        println!("\n{}", scenario.name);
        println!("{}", "-".repeat(72));
        if scenario.messages.len() > 1 {
            for msg in &scenario.messages {
                println!("  msg: \"{}\"", truncate(msg, 60));
            }
        } else {
            println!("  msg: \"{}\"", truncate(scenario.messages[0], 60));
        }
        println!("  prediction: {}", scenario.prediction);
        println!();

        println!("  Cognitive state:");
        println!("    arousal={:.3}  valence={:?}  gain={:?}  budget={:.3}",
            turn.arousal, turn.valence, turn.gain_mode, turn.body_budget);
        println!("    sensory_pe={:.3}  pe_volatility={:.3}  gate={:?}",
            turn.sensory_pe, turn.cognitive_state.pe_volatility, turn.gate_type);
        println!();

        println!("  Tầng 1 (sampling): temp={:.3}  top_p={:.3}  freq_penalty={:.3}  presence_penalty={:.3}",
            turn.sampling.temperature, turn.sampling.top_p,
            turn.sampling.frequency_penalty, turn.sampling.presence_penalty);
        println!("  Tầng 2 (delta):    gain={:.4}  layers={}/{} (40-60% depth)",
            delta_mod.gain_factor,
            result_tang2.modulated_layers.len(), num_layers);
        println!();

        println!("  Metrics (baseline vs Tầng 2):");
        println!("    KL divergence:    {:.6}", kl);
        println!("    Entropy baseline: {:.4}", ent_baseline);
        println!("    Entropy Tầng 2:   {:.4} ({:+.1}%)",
            ent_tang2, pct_change(ent_baseline, ent_tang2));
        println!("    Top-10 overlap:   {}/10", overlap);
        println!();

        println!("  Tầng 1 temperature effect (temp={:.3}):", temp);
        println!("    Entropy baseline+temp: {:.4}", ent_t1_baseline);
        println!("    Entropy Tầng1+2:       {:.4} ({:+.1}% from baseline+temp)",
            ent_t1_tang2, pct_change(ent_t1_baseline, ent_t1_tang2));

        // Top-5 tokens comparison.
        println!();
        print_top_tokens("    Baseline", &logits_baseline, &tokenizer, 5);
        print_top_tokens("    Tầng 1+2", logits_tang2, &tokenizer, 5);

        results.push(ScenarioResult {
            name: scenario.name.to_string(),
            arousal: turn.arousal,
            gain_mode: format!("{:?}", turn.gain_mode),
            body_budget: turn.body_budget,
            sensory_pe: turn.sensory_pe,
            temperature: turn.sampling.temperature,
            top_p: turn.sampling.top_p,
            freq_penalty: turn.sampling.frequency_penalty,
            delta_gain: delta_mod.gain_factor,
            modulated_layers: result_tang2.modulated_layers.len(),
            kl_divergence: kl,
            entropy_baseline: ent_baseline,
            entropy_tang2: ent_tang2,
            entropy_tang1_baseline: ent_t1_baseline,
            entropy_tang1_tang2: ent_t1_tang2,
            top10_overlap: overlap,
        });
    }

    // ══════════════════════════════════════════════════════════════════
    // PART B: Manual cognitive states (prove Tầng 2 works independently)
    // ══════════════════════════════════════════════════════════════════
    println!("\n{}", "=".repeat(72));
    println!("PART B: Manual Cognitive States (bypass text processing)");
    println!("{}", "=".repeat(72));
    println!();
    println!("Testing Tầng 2 with hand-crafted states to isolate the");
    println!("delta modulation effect from CognitiveSession signal generation.\n");

    use noos::types::belief::AffectValence;
    use noos::types::intervention::CognitiveState;

    let manual_states: Vec<(&str, CognitiveState)> = vec![
        ("Neutral (control)", CognitiveState::default()),
        ("Phasic (focused)", CognitiveState {
            gain_mode: GainMode::Phasic,
            arousal: 0.3,
            certainty: 0.8,
            ..CognitiveState::default()
        }),
        ("Tonic (exploratory)", CognitiveState {
            gain_mode: GainMode::Tonic,
            arousal: 0.2,
            certainty: 0.3,
            ..CognitiveState::default()
        }),
        ("High arousal threat", CognitiveState {
            arousal: 0.9,
            valence: AffectValence::Negative,
            body_budget: 0.6,
            sensory_pe: 0.8,
            ..CognitiveState::default()
        }),
        ("Depleted budget", CognitiveState {
            body_budget: 0.2,
            arousal: 0.4,
            valence: AffectValence::Negative,
            ..CognitiveState::default()
        }),
        ("Volatile environment", CognitiveState {
            pe_volatility: 0.8,
            sensory_pe: 0.7,
            arousal: 0.5,
            ..CognitiveState::default()
        }),
    ];

    // Baseline logits (neutral state, already computed — recompute for clarity).
    model.reset_cache();
    let logits_neutral = model.forward(&prompt_tokens, 0)
        .expect("Baseline forward failed");
    let probs_neutral = softmax_f32(&logits_neutral);
    let ent_neutral = entropy(&probs_neutral);

    println!("{:<25} {:>7} {:>8} {:>8} {:>7} {:>7}",
        "State", "Gain", "KL div", "Ent %", "Top10", "Layers");
    println!("{}", "-".repeat(72));

    for (name, state) in &manual_states {
        model.reset_cache();
        let dm = compute_delta_modulation(state, num_layers);
        let result = model.forward_cognitive(&prompt_tokens, 0, &dm)
            .expect("Cognitive forward failed");

        let probs = softmax_f32(&result.logits);
        let kl = kl_divergence(&probs_neutral, &probs);
        let ent = entropy(&probs);
        let overlap = top_k_overlap(&logits_neutral, &result.logits, 10);

        println!("{:<25} {:>7.4} {:>8.6} {:>+7.1}% {:>5}/10 {:>3}/{}",
            name, dm.gain_factor, kl, pct_change(ent_neutral, ent),
            overlap, result.modulated_layers.len(), num_layers);
    }

    // ══════════════════════════════════════════════════════════════════
    // PART C: Closed-loop with CognitiveGate (thalamocortical feedback)
    // ══════════════════════════════════════════════════════════════════
    println!("\n{}", "=".repeat(72));
    println!("PART C: Closed-Loop with CognitiveGate (thalamocortical feedback)");
    println!("{}", "=".repeat(72));
    println!();
    println!("Loading CognitiveMambaWithGate (fresh gate, untrained)...");

    use noos::cognition::intervention::build_cognitive_state;
    use noos::types::intervention::ForwardResult;

    let gate_config = CognitiveGateConfig::from_mamba_config(&MambaConfig::mamba_130m());
    let gate_model = CognitiveMambaWithGate::from_pretrained_with_gate(
        model_id, MambaConfig::mamba_130m(), gate_config,
    );

    match gate_model {
        Ok((mut gmodel, _varmap)) => {
            let gate_layers = gmodel.num_layers();
            println!("  Gate model loaded. {} layers, gate at mid-depth.\n", gate_layers);

            let test_msg = "I'm really stressed about this deadline and everything is going wrong!!!";
            let gen_tokens = 15;

            println!("Comparing open-loop (base model) vs closed-loop (gate model + feedback)");
            println!("Message: \"{}\"\n", truncate(test_msg, 60));

            // ── Open-loop: base model, fixed cognitive state ──
            let mut session_open = CognitiveSession::with_model_layers(gate_layers);
            let turn_open = session_open.process_message(test_msg);
            let state_open = turn_open.cognitive_state.clone();

            model.reset_cache();
            let _ = model.forward(&prompt_tokens, 0);
            let mut open_alphas: Vec<f64> = Vec::new();
            let mut open_arousals: Vec<f64> = Vec::new();

            for _ in 0..gen_tokens {
                let dm = compute_delta_modulation(&state_open, num_layers);
                let result = model.forward_cognitive(&[0], 0, &dm)
                    .unwrap_or_else(|_| ForwardResult::from_logits(vec![0.0; 50280]));
                open_alphas.push(result.gate_alpha.unwrap_or(0.0));
                open_arousals.push(state_open.arousal);
            }

            // ── Closed-loop: gate model, feedback per token ──
            let mut session_closed = CognitiveSession::with_model_layers(gate_layers);
            let _turn_closed = session_closed.process_message(test_msg);

            gmodel.reset_cache();
            let _ = <CognitiveMambaWithGate as LocalModel>::forward(&mut gmodel, &prompt_tokens, 0);
            let mut closed_alphas: Vec<f64> = Vec::new();
            let mut closed_arousals: Vec<f64> = Vec::new();
            let mut closed_deltas: Vec<f64> = Vec::new();

            for _ in 0..gen_tokens {
                let current_state = build_cognitive_state(
                    session_closed.world_model(),
                    session_closed.world_model().learned.gain_mode,
                );
                let dm = compute_delta_modulation(&current_state, gate_layers);
                let result = gmodel.forward_cognitive(&[0], 0, &dm)
                    .unwrap_or_else(|_| ForwardResult::from_logits(vec![0.0; 50280]));

                let ga = result.gate_alpha.unwrap_or(0.0);
                let gd = result.gate_delta_gain.unwrap_or(1.0);
                closed_alphas.push(ga);
                closed_arousals.push(current_state.arousal);
                closed_deltas.push(gd);

                session_closed.inject_gate_feedback(ga, gd);
            }

            // ── Print comparison ──
            println!("{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}",
                "Token", "Base α", "Gate α", "Base arou", "Gate arou", "Gate δ");
            println!("{}", "-".repeat(62));
            for i in 0..gen_tokens {
                println!("{:<6} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                    i + 1,
                    open_alphas[i], closed_alphas[i],
                    open_arousals[i], closed_arousals[i],
                    closed_deltas[i]);
            }

            let open_arousal_change = open_arousals.last().unwrap_or(&0.0)
                - open_arousals.first().unwrap_or(&0.0);
            let closed_arousal_change = closed_arousals.last().unwrap_or(&0.0)
                - closed_arousals.first().unwrap_or(&0.0);

            println!();
            println!("  Open-loop arousal:   {:+.4} (fixed)", open_arousal_change);
            println!("  Closed-loop arousal: {:+.4} (gate feedback)", closed_arousal_change);

            let avg_gate_alpha: f64 = closed_alphas.iter().sum::<f64>() / gen_tokens as f64;
            let avg_gate_delta: f64 = closed_deltas.iter().sum::<f64>() / gen_tokens as f64;
            println!("  Avg gate α: {:.4}, avg gate δ: {:.4}", avg_gate_alpha, avg_gate_delta);

            if avg_gate_alpha > 0.01 {
                println!("  GATE ACTIVE: thalamocortical feedback loop producing real signals.");
            }
            if (closed_arousal_change - open_arousal_change).abs() > 1e-6 {
                println!("  FEEDBACK LOOP: arousal evolves differently with gate feedback.");
            }
        }
        Err(e) => {
            println!("  Failed to load gate model: {e}");
            println!("  Skipping Part C (gate model required).");
        }
    }

    // ── Summary ──
    println!("\n{}", "=".repeat(72));
    println!("SUMMARY");
    println!("{}", "=".repeat(72));
    println!();
    println!("{:<35} {:>6} {:>6} {:>8} {:>8} {:>7}",
        "Scenario", "Arousal", "Gain", "KL div", "Ent %", "Top10");
    println!("{}", "-".repeat(72));

    for r in &results {
        println!("{:<35} {:>6.3} {:>6} {:>8.6} {:>+7.1}% {:>5}/10",
            truncate(&r.name, 35),
            r.arousal,
            &r.gain_mode[..r.gain_mode.len().min(6)],
            r.kl_divergence,
            pct_change(r.entropy_baseline, r.entropy_tang2),
            r.top10_overlap,
        );
    }

    // ── Verdict ──
    println!("\n{}", "-".repeat(72));
    println!("VERDICT\n");

    let any_kl = results.iter().any(|r| r.kl_divergence > 1e-6);
    let routine_minimal = results.first()
        .map(|r| r.kl_divergence < results.iter().skip(1)
            .map(|r2| r2.kl_divergence)
            .fold(f64::MAX, f64::min))
        .unwrap_or(false);

    if any_kl {
        println!("  Tầng 2 (delta modulation) produces MEASURABLY DIFFERENT logit distributions.");
        if routine_minimal {
            println!("  Control (routine) shows LESS modulation than aroused scenarios.");
            println!("  Non-cortical processing IS context-dependent — not random noise.");
        }
    } else {
        println!("  No measurable difference. Investigate: gain range, layer targeting.");
    }

    // Check neuroscience predictions.
    let stress = results.iter().find(|r| r.name.contains("Emotional"));
    let routine = results.first();
    if let (Some(s), Some(r)) = (stress, routine) {
        if s.delta_gain > r.delta_gain {
            println!("  Stress → higher delta gain than routine (attention to threat).");
        }
    }

    let depleted = results.iter().find(|r| r.name.contains("Multi-turn"));
    if let Some(d) = depleted {
        if d.body_budget < 0.95 {
            println!("  Multi-turn stress depletes body budget to {:.3} (allostasis active).", d.body_budget);
        }
        if d.delta_gain < 1.0 {
            println!("  Depleted budget → delta gain < 1.0 (conservation mode confirmed).");
        }
    }

    println!();
}

// ── Metrics ──────────────────────────────────────────────────────────────

#[cfg(feature = "candle")]
fn kl_divergence(p: &[f32], q: &[f32]) -> f64 {
    let mut kl = 0.0f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let pi = pi as f64;
        let qi = qi as f64;
        if pi > 1e-10 && qi > 1e-10 {
            kl += pi * (pi / qi).ln();
        }
    }
    kl.max(0.0)
}

#[cfg(feature = "candle")]
fn entropy(probs: &[f32]) -> f64 {
    let mut h = 0.0f64;
    for &p in probs {
        let p = p as f64;
        if p > 1e-10 {
            h -= p * p.ln();
        }
    }
    h
}

#[cfg(feature = "candle")]
fn top_k_overlap(logits_a: &[f32], logits_b: &[f32], k: usize) -> usize {
    let top_a = top_k_indices(logits_a, k);
    let top_b = top_k_indices(logits_b, k);
    top_a.iter().filter(|idx| top_b.contains(idx)).count()
}

#[cfg(feature = "candle")]
fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|(i, _)| *i).collect()
}

#[cfg(feature = "candle")]
fn pct_change(from: f64, to: f64) -> f64 {
    if from.abs() < 1e-10 { 0.0 } else { (to - from) / from * 100.0 }
}

// ── Display helpers ─────────────────────────────────────────────────────

#[cfg(feature = "candle")]
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() }
    else { format!("{}...", &s[..max.saturating_sub(3)]) }
}

#[cfg(feature = "candle")]
fn print_top_tokens(label: &str, logits: &[f32], tokenizer: &HfTokenizer, k: usize) {
    let indices = top_k_indices(logits, k);
    print!("{}: ", label);
    for (i, &idx) in indices.iter().enumerate() {
        let tok = tokenizer.decode_token(idx as u32)
            .unwrap_or_else(|_| format!("[{}]", idx));
        let sep = if i < k - 1 { " | " } else { "" };
        print!("{:.1}:{}{sep}", logits[idx], tok.trim());
    }
    println!();
}
