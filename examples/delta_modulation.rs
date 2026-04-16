//! Delta Modulation Explorer — demonstrates Tầng 2 cognitive intervention.
//!
//! Run: `cargo run --example delta_modulation`
//!
//! This example shows how Nous's cognitive state maps to SSM delta modulation:
//! - Phasic mode → higher delta → model attends to current input (focused)
//! - Tonic mode → lower delta → model preserves history (exploratory)
//! - Low body budget → conservation → reduced delta
//! - High arousal → emergency override → boosted delta
//! - All thresholds are adaptive (P2: precision as universal currency)
//!
//! Tầng 1 (sampling) + Tầng 2 (delta modulation) stack together:
//! delta modulation changes HOW the model thinks,
//! then sampling modulation changes WHAT token is selected.
//!
//! See `docs/intervention.md` for the full intervention architecture.

use noos::cognition::delta_modulation::{compute_delta_modulation, compute_layer_targets};
use noos::session::CognitiveSession;
use noos::types::intervention::CognitiveState;
use noos::types::world::GainMode;

fn main() {
    println!("=== Nous Tầng 2: Delta Modulation Explorer ===\n");

    // ── Part 1: Direct computation from CognitiveState ──────────────

    println!("── Part 1: CognitiveState → DeltaModulation ──\n");

    let scenarios: Vec<(&str, CognitiveState)> = vec![
        (
            "Calm neutral (default)",
            CognitiveState::default(),
        ),
        (
            "Phasic mode (focused task)",
            CognitiveState {
                gain_mode: GainMode::Phasic,
                ..CognitiveState::default()
            },
        ),
        (
            "Tonic mode (broad exploration)",
            CognitiveState {
                gain_mode: GainMode::Tonic,
                ..CognitiveState::default()
            },
        ),
        (
            "Depleted body budget (conservation)",
            CognitiveState {
                body_budget: 0.1,
                ..CognitiveState::default()
            },
        ),
        (
            "High volatility (unstable environment)",
            CognitiveState {
                pe_volatility: 0.9,
                ..CognitiveState::default()
            },
        ),
        (
            "Emergency arousal (threat detected)",
            CognitiveState {
                arousal: 0.95,
                ..CognitiveState::default()
            },
        ),
        (
            "Tonic + depleted (worst case conservation)",
            CognitiveState {
                gain_mode: GainMode::Tonic,
                body_budget: 0.05,
                ..CognitiveState::default()
            },
        ),
        (
            "Phasic + volatile + aroused (maximum focus)",
            CognitiveState {
                gain_mode: GainMode::Phasic,
                pe_volatility: 0.9,
                arousal: 0.95,
                ..CognitiveState::default()
            },
        ),
    ];

    let num_layers = 64; // Falcon Mamba 7B

    println!(
        "{:<45} {:>8} {:>12} {:>8}",
        "Scenario", "Gain", "Layers", "Effect"
    );
    println!("{}", "-".repeat(80));

    for (name, state) in &scenarios {
        let dm = compute_delta_modulation(state, num_layers);
        let effect = if dm.gain_factor > 1.05 {
            "ATTEND ↑"
        } else if dm.gain_factor < 0.95 {
            "PRESERVE ↓"
        } else {
            "neutral"
        };
        println!(
            "{:<45} {:>8.3} {:>5}-{:<5} {:>8}",
            name, dm.gain_factor, dm.target.start_layer, dm.target.end_layer, effect
        );
    }

    // ── Part 2: Layer targeting across model sizes ──────────────────

    println!("\n── Part 2: Layer Targeting (40-60% depth) ──\n");

    let model_sizes = [12, 24, 32, 48, 64, 96];
    println!("{:<15} {:>8} {:>8} {:>8}", "Model layers", "Start", "End", "Count");
    println!("{}", "-".repeat(45));
    for &n in &model_sizes {
        let target = compute_layer_targets(n);
        println!(
            "{:<15} {:>8} {:>8} {:>8}",
            n,
            target.start_layer,
            target.end_layer,
            target.modulated_count()
        );
    }

    // ── Part 3: Full session with Tầng 1 + Tầng 2 output ──────────

    println!("\n── Part 3: CognitiveSession → Tầng 1 + Tầng 2 ──\n");

    let mut session = CognitiveSession::new();

    let messages = [
        "Hello, how are you?",
        "I need help with a complex algorithm",
        "THIS IS URGENT!!! Everything is broken!!!",
        "Wait, I think I found the issue",
        "Thanks, that worked perfectly",
    ];

    println!(
        "{:<45} {:>5} {:>6} {:>6} {:>6} {:>8}",
        "Message", "Gain", "Temp", "TopP", "Budget", "Mode"
    );
    println!("{}", "-".repeat(90));

    for msg in &messages {
        let result = session.process_message(msg);

        let truncated: String = if msg.len() > 42 {
            format!("{}...", &msg[..39])
        } else {
            msg.to_string()
        };

        println!(
            "{:<45} {:>5.2} {:>6.2} {:>6.2} {:>6.2} {:>8?}",
            truncated,
            result.delta_modulation.gain_factor,
            result.sampling.temperature,
            result.sampling.top_p,
            result.body_budget,
            result.gain_mode,
        );

        // Simulate model response.
        session.process_response("Here is a helpful response with clear steps.", 0.8);
    }

    println!("\n── Summary ──");
    println!(
        "Tầng 1 (sampling): temperature + top_p modulate WHAT token is selected"
    );
    println!(
        "Tầng 2 (delta):    gain_factor modulates HOW the SSM processes input"
    );
    println!("Both tầng stack — they complement, not replace each other.");
    println!(
        "\nNext: connect to real Mamba model (candle) for end-to-end cognitive inference."
    );
}
