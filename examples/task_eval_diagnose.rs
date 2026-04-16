//! Diagnostic for task_eval_synthetic.rs failure.
//!
//! Prints 4 diagnostic signals used to debug cluster-mismatch bugs:
//!
//! 1. Cluster hash for each of the 3 synthetic queries (collision check).
//! 2. Detected strategy from each of the 3 simulated responses (which the
//!    training path actually records).
//! 3. `learned.response_strategies` after training (what got stored).
//! 4. `turn.signals.strategy` for each eval query (what Nous recommends).
//!
//! Run: `cargo run --example task_eval_diagnose`

use noos::cognition::detector::{detect_response_strategy, extract_topics};
use noos::session::CognitiveSession;

fn cluster_for(text: &str) -> String {
    // Mirrors world_model.rs::build_topic_cluster (private).
    let topics = extract_topics(text);
    let mut sorted: Vec<String> = topics
        .iter()
        .filter(|t| t.len() >= 3)
        .map(|t| t.to_lowercase())
        .collect();
    sorted.sort();
    sorted.truncate(2);
    sorted.join("+")
}

fn main() {
    println!("=== Diagnostic 1: topics + cluster hash per query ===\n");
    let queries = [
        ("NumericalDebug", "Help me debug this numerical issue."),
        ("QuickLookup", "What is the default port for Postgres?"),
        ("AmbiguousRequest", "Make this better somehow."),
    ];
    for (label, q) in &queries {
        let topics = extract_topics(q);
        let cluster = cluster_for(q);
        println!("  {:<20} query = {:?}", label, q);
        println!("  {:<20} topics = {:?}", "", topics);
        println!("  {:<20} cluster = {:?}\n", "", cluster);
    }

    println!("=== Diagnostic 2: strategy detected from each simulated response ===\n");
    let responses = [
        ("DirectAnswer (for QuickLookup)", "Short answer."),
        ("AskClarifying (for AmbiguousRequest)", "What do you mean exactly?"),
        (
            "StepByStep (for NumericalDebug)",
            "1. First. 2. Then. 3. Finally.",
        ),
    ];
    for (label, resp) in &responses {
        let detected = detect_response_strategy(resp);
        println!("  {:<40} response = {:?}", label, resp);
        println!("  {:<40} detected = {:?}\n", "", detected);
    }

    println!("=== Diagnostic 3: learned.response_strategies after training ===\n");
    // Same training as task_eval_synthetic.rs: 6 rounds × 3 categories.
    let mut session = CognitiveSession::new();
    for _round in 0..6 {
        // NumericalDebug → StepByStep
        let _ = session.process_message("Help me debug this numerical issue.");
        session.track_cost(0.8);
        session.process_response("1. First. 2. Then. 3. Finally.", 0.9);
        // QuickLookup → DirectAnswer
        let _ = session.process_message("What is the default port for Postgres?");
        session.track_cost(0.1);
        session.process_response("Short answer.", 0.9);
        // AmbiguousRequest → AskClarifying
        let _ = session.process_message("Make this better somehow.");
        session.track_cost(0.3);
        session.process_response("What do you mean exactly?", 0.9);
    }
    let learned = session.export_learned();
    println!("  response_strategies entries: {}", learned.response_strategies.len());
    for (cluster, strategies) in &learned.response_strategies {
        println!("  cluster = {:?}", cluster);
        for (strategy, entry) in strategies {
            println!(
                "    strategy = {:<20} count = {:>3}   success_rate = {:.3}",
                strategy, entry.count, entry.success_rate
            );
        }
    }
    println!();

    println!("=== Diagnostic 4: turn.signals.strategy for each eval query ===\n");
    // Fresh session with imported learned state — mirrors warm-start path.
    let mut eval_session = CognitiveSession::with_learned(learned.clone(), 64);
    for (label, q) in &queries {
        let turn = eval_session.process_message(q);
        println!(
            "  {:<20} query = {:?}  recommendation = {:?}",
            label, q, turn.signals.strategy
        );
        // Close the loop with a placeholder response so the session advances cleanly.
        eval_session.process_response("(diagnostic probe)", 0.5);
    }
}
