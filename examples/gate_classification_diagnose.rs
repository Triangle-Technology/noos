//! Diagnostic: what does `thalamic_gate` classify the perplexity-eval
//! text categories as?
//!
//! Run: `cargo run --example gate_classification_diagnose`
//!
//! The CR5 gate-conditioning fix in `compute_delta_modulation` only activates
//! on `GateType::Routine` inputs. For the fix to help the Technical-category
//! regression observed in `examples/perplexity_eval.rs` (HS arousal produced
//! +2.37% perplexity), thalamic_gate must classify the Technical text as
//! Routine when it passes through `CognitiveSession::process_message`.
//!
//! If Technical → Routine: the fix helps end-to-end (just needs a pipeline-
//! integrated perplexity eval to demonstrate).
//!
//! If Technical → Novel: the fix doesn't help perplexity on this text
//! because the gate still says "process this". The regression needs a
//! different surgical fix (e.g., valence-conditioning, or confidence-based
//! damping).

use nous_regulator::session::CognitiveSession;

fn main() {
    println!("=== Gate classification diagnostic ===\n");
    println!("Four text categories from perplexity_eval.rs run through");
    println!("CognitiveSession::process_message. We want to know:");
    println!("  (a) What gate_type does thalamic_gate produce?");
    println!("  (b) What delta_modulation.gain_factor results?");
    println!("  (c) Does Routine classification short-circuit to passthrough?\n");

    let categories = [
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

    println!(
        "  {:<12} {:<12} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "Category", "gate_type", "gate_conf", "arousal", "valence", "gain", "short-circuit?"
    );
    println!("  {}", "─".repeat(92));

    for (name, text) in &categories {
        let mut session = CognitiveSession::new();
        let turn = session.process_message(text);

        let gate = format!("{:?}", turn.gate_type);
        let valence = format!("{:?}", turn.valence);
        let gain = turn.delta_modulation.gain_factor;
        let short_circuit = (gain - 1.0).abs() < 1e-9 && matches!(turn.gate_type, nous_regulator::types::gate::GateType::Routine);
        let marker = if short_circuit { "✓ PASSTHROUGH" } else { "(compensated)" };

        println!(
            "  {:<12} {:<12} {:>10.3} {:>10.3} {:>10} {:>10.3} {:>12}",
            name, gate, turn.gate_confidence, turn.arousal, valence, gain, marker
        );
    }

    println!("\nInterpretation:");
    println!("  • Each category's gate_type is what thalamic_gate returned.");
    println!("  • gain=1.000 on a Routine row means the CR5 short-circuit fired.");
    println!("  • If Technical is NOT Routine, the gate-conditioning fix does NOT");
    println!("    help the Technical perplexity regression — a different");
    println!("    surgical fix is needed (e.g., valence-conditioning, low-");
    println!("    confidence damping, or widening the 'skip' predicate).");
    println!();
    println!("  This diagnostic does NOT run the model (no candle). Perplexity");
    println!("  numbers require `perplexity_eval.rs` which loads mamba-130m.");
}
