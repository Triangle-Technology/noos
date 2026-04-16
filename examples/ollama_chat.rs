//! Ollama Chat — Nous cognitive engine + real model via Ollama API.
//!
//! Prerequisites:
//!   1. Install Ollama: https://ollama.com
//!   2. Pull a model: `ollama pull phi3:mini` (or any model)
//!   3. Ollama runs at http://localhost:11434 by default
//!
//! Run: `cargo run --example ollama_chat`
//! Or with custom model: `cargo run --example ollama_chat -- mistral`
//!
//! This demonstrates Nous's cognitive modulation affecting REAL model output:
//! - Calm input → balanced temperature → varied responses
//! - Stressed input → lower temperature + frequency penalty → focused responses
//! - Over time: strategy learning, body budget depletion/recovery

use noos::session::{CognitiveSession, TurnResult};
use noos::types::belief::AffectValence;
use serde_json::json;
use std::io::{self, BufRead, Write};

const DEFAULT_MODEL: &str = "phi3:mini";
const OLLAMA_URL: &str = "http://localhost:11434/api/chat";

fn main() {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    println!("=== Nous + Ollama Chat ===");
    println!("Model: {model}");
    println!("Cognitive modulation: ON (temperature, top_p, penalties from brain state)");
    println!("Type 'quit' to exit, 'state' to show cognitive state.\n");

    let mut session = CognitiveSession::new();
    let mut conversation: Vec<serde_json::Value> = Vec::new();

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).is_err() || input.trim().is_empty() {
            continue;
        }
        let message = input.trim().to_string();

        if message == "quit" || message == "exit" {
            break;
        }

        if message == "state" {
            print_state(&session);
            continue;
        }

        // 1. Cognitive pipeline: perceive → converge → sampling override.
        let turn = session.process_message(&message);
        print_cognitive_summary(&turn);

        // 2. Build Ollama request with cognitive sampling parameters.
        conversation.push(json!({
            "role": "user",
            "content": message,
        }));

        let body = json!({
            "model": model,
            "messages": conversation,
            "stream": false,
            "options": {
                "temperature": turn.sampling.temperature,
                "top_p": turn.sampling.top_p,
                "frequency_penalty": turn.sampling.frequency_penalty,
                "presence_penalty": turn.sampling.presence_penalty
            }
        });

        // 3. Call Ollama.
        print!(
            "\n[Generating with temp={:.2}, top_p={:.2}...] ",
            turn.sampling.temperature, turn.sampling.top_p
        );
        stdout.flush().unwrap();

        match ureq::post(OLLAMA_URL).send_json(&body) {
            Ok(response) => match response.into_json::<serde_json::Value>() {
                Ok(data) => {
                    let content = data["message"]["content"]
                        .as_str()
                        .unwrap_or("(empty response)")
                        .to_string();

                    println!("\n\n{content}\n");

                    // 4. Learning: process response through cognitive pipeline.
                    let quality = estimate_quality(&content, &turn);
                    session.process_response(&content, quality);

                    conversation.push(json!({
                        "role": "assistant",
                        "content": content,
                    }));

                    // Show learning.
                    if let Some(strategy) = session.world_model().last_response_strategy {
                        println!(
                            "[Strategy: {:?}, RPE: {:.2}, Budget: {:.2}]",
                            strategy,
                            session.world_model().response_rpe,
                            session.world_model().body_budget,
                        );
                    }
                }
                Err(e) => println!("\nParse error: {e}"),
            },
            Err(e) => {
                println!("\nOllama connection failed: {e}");
                println!("Is Ollama running? Try: ollama serve");
                conversation.pop();
            }
        }
    }
}

fn print_cognitive_summary(turn: &TurnResult) {
    let valence = match turn.valence {
        AffectValence::Positive => "+",
        AffectValence::Negative => "-",
        AffectValence::Neutral => "~",
    };
    println!(
        "[Brain: arousal={:.2}{} budget={:.2} gate={:?} gain={:?}]",
        turn.arousal, valence, turn.body_budget, turn.gate_type, turn.gain_mode
    );
    if let Some(strategy) = turn.recommended_strategy {
        println!("[Recommending: {:?}]", strategy);
    }
}

fn print_state(session: &CognitiveSession) {
    let m = session.world_model();
    println!("\n--- Full Cognitive State ---");
    println!("  Body budget:    {:.3}", m.body_budget);
    println!(
        "  Arousal:        {:.3} ({:?})",
        m.belief.affect.arousal, m.belief.affect.valence
    );
    println!("  Sensory PE:     {:.3}", m.sensory_pe);
    println!("  PE volatility:  {:.3}", m.pe_volatility);
    println!("  Resource press: {:.3}", m.resource_pressure);
    println!("  Response RPE:   {:.3}", m.response_rpe);
    println!("  Turn:           {}", m.belief.turn);
    println!("  Learned strats: {}", m.learned.response_strategies.len());
    if let Some(strategy) = m.recommended_strategy {
        println!("  Recommended:    {:?}", strategy);
    }
    println!();
}

/// Simple quality heuristic for learning feedback.
fn estimate_quality(response: &str, turn: &TurnResult) -> f64 {
    let mut quality: f64 = 0.6;

    if response.len() > 200 && turn.sensory_pe > 0.3 {
        quality += 0.1;
    }
    if response.contains('#') || response.contains("1.") {
        quality += 0.05;
    }
    if response.len() < 50 && turn.gate_confidence > 0.6 {
        quality -= 0.1;
    }

    quality.clamp(0.1, 1.0)
}
