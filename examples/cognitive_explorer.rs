//! Cognitive Explorer — interactive demo of Nous's cognitive pipeline.
//!
//! Run: `cargo run --example cognitive_explorer`
//!
//! Type messages and watch how Nous's brain reacts:
//! - Emotional arousal rises on stressed/excited input
//! - Body budget depletes under sustained stress
//! - Gain mode shifts between focused (phasic) and exploratory (tonic)
//! - Sampling parameters adapt to cognitive state
//! - Strategy recommendations emerge from learned patterns

use nous_regulator::session::CognitiveSession;
use std::io::{self, BufRead, Write};

fn main() {
    println!("=== Nous Cognitive Explorer ===");
    println!("Type messages to see cognitive state. Type 'quit' to exit.");
    println!("After each message, type a simulated response quality (0.0-1.0) or press Enter to skip.\n");

    let mut session = CognitiveSession::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Prompt for user message.
        print!("> ");
        stdout.flush().unwrap();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).is_err() || input.trim().is_empty() {
            continue;
        }
        let message = input.trim();

        if message == "quit" || message == "exit" {
            println!("Session ended. {} turns processed.", session.turn_count());
            break;
        }

        // Process through cognitive pipeline.
        let result = session.process_message(message);

        // Display cognitive state.
        println!("\n--- Cognitive State (Turn {}) ---", session.turn_count());
        println!("  Arousal:      {:.2} {}", result.arousal, valence_label(result.valence));
        println!("  Body budget:  {:.2}", result.body_budget);
        println!("  Sensory PE:   {:.2}", result.sensory_pe);
        println!("  Gain mode:    {:?}", result.gain_mode);
        println!("  Gate:         {:?} (confidence: {:.2})", result.gate_type, result.gate_confidence);

        println!("--- Sampling Override ---");
        println!("  Temperature:       {:.3}", result.sampling.temperature);
        println!("  Top-p:             {:.3}", result.sampling.top_p);
        println!("  Frequency penalty: {:.3}", result.sampling.frequency_penalty);
        println!("  Presence penalty:  {:.3}", result.sampling.presence_penalty);

        println!("--- Strategy ---");
        match result.recommended_strategy {
            Some(strategy) => println!("  Recommended: {:?}", strategy),
            None => println!("  Recommended: None (no learned data for this topic)"),
        }

        println!("--- Convergence ---");
        println!(
            "  Iterations: {}, converged: {}",
            result.convergence_iterations, result.converged
        );

        // Simulate response + learning.
        print!("\nResponse quality (0.0-1.0, Enter to skip): ");
        stdout.flush().unwrap();

        let mut quality_input = String::new();
        if stdin.lock().read_line(&mut quality_input).is_ok() {
            let quality_str = quality_input.trim();
            if let Ok(quality) = quality_str.parse::<f64>() {
                let quality = quality.clamp(0.0, 1.0);
                // Simulate a response based on recommended strategy.
                let simulated_response = match result.recommended_strategy {
                    Some(nous_regulator::types::world::ResponseStrategy::StepByStep) => {
                        "Here's a step-by-step guide:\n1. First, do this\n2. Then, do that\n3. Finally, finish"
                    }
                    Some(nous_regulator::types::world::ResponseStrategy::ClarifyFirst) => {
                        "Before I answer, let me ask:\n- What is your goal?\n- What have you tried?"
                    }
                    Some(nous_regulator::types::world::ResponseStrategy::StructuredAnalysis) => {
                        "## Analysis\n### Pros\n- Good point\n### Cons\n- Bad point\n### Recommendation\nDo X."
                    }
                    _ => "Here's a direct answer to your question.",
                };
                session.process_response(simulated_response, quality);
                println!(
                    "  Learned: strategy={:?}, RPE={:.2}",
                    session.world_model().last_response_strategy,
                    session.world_model().response_rpe,
                );
            }
        }
        println!();
    }
}

fn valence_label(v: nous_regulator::types::belief::AffectValence) -> &'static str {
    match v {
        nous_regulator::types::belief::AffectValence::Positive => "(positive)",
        nous_regulator::types::belief::AffectValence::Negative => "(negative)",
        nous_regulator::types::belief::AffectValence::Neutral => "(neutral)",
    }
}
