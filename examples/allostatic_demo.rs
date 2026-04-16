//! Advanced: `CognitiveSession` walkthrough — the signal layer underneath
//! `Regulator`.
//!
//! Run: `cargo run --example allostatic_demo`
//!
//! Demonstrates the `CognitiveSession` API end-to-end:
//! 1. Per-turn signals (salience, confidence, conservation)
//! 2. Body budget — slow-timescale depletion signal
//! 3. Strategy learning via EMA per topic cluster
//! 4. Cross-session persistence via JSON round-trip
//! 5. Closed-loop cost tracking (`track_cost` feeds body budget)
//! 6. Quality-history signal (`recent_quality` + `rpe`)
//! 7. Memory retrieval composes with signals (sync `hybrid_recall`)
//!
//! No candle/model required — pure text-level cognitive pipeline
//! (convergence loop, LC-style gain, body-budget accounting, reward learning).
//!
//! Most integrators should prefer `Regulator` (see README). This demo is
//! for users building custom decision policies on raw continuous signals
//! or running local Mamba inference that needs delta modulation.

use nous::session::{CognitiveSession, TurnResult};
use nous::{hybrid_recall, AtomSource, AtomType, MemoryAtom, RecallOptions};
use std::collections::HashMap;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Nous Allostatic Demo — Closed-Loop Allostatic Controller    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    scenario_1_salience_tracking();
    scenario_2_conservation_under_sustained_stress();
    scenario_3_strategy_accumulation();
    scenario_4_cross_session_persistence();
    scenario_5_closed_loop_cost_tracking();
    scenario_6_failure_detection();
    scenario_7_memory_informs_decisions();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  All signals available via turn.signals (CognitiveSignals)   ║");
    println!("║  Application decides how to act — Nous provides the state.   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

/// Scenario 1: Salience signal tracks emotional intensity per turn.
///
/// Shows that salience (the LC-NE phasic burst analog) responds immediately
/// to input content — unlike body_budget which is slow-timescale.
fn scenario_1_salience_tracking() {
    println!("── Scenario 1: Per-Turn Salience Tracking ──");
    println!("Salience responds immediately to input intensity (LC phasic signal).\n");

    let mut session = CognitiveSession::new();
    let messages = [
        ("Hello, I have a question.", "calm"),
        ("I'm so overwhelmed!!! I can't focus!!!", "stressed"),
        ("OK, let me think about this calmly.", "calm"),
        ("Everything is TERRIBLE!!! DISASTER!!!", "panicked"),
        ("Thanks for helping, this is great.", "positive"),
    ];

    println!("  {:<4} {:<12} {:<10} {:<10} {:<10}",
        "Turn", "Tone", "Salience", "Confid.", "Arousal");
    println!("  {}", "─".repeat(50));

    for (i, (msg, tone)) in messages.iter().enumerate() {
        let turn = session.process_message(msg);
        println!("  {:<4} {:<12} {:<10.3} {:<10.3} {:<10.3}",
            i + 1, tone, turn.signals.salience, turn.signals.confidence, turn.arousal);
    }

    println!("\n  ✓ Salience tracks emotional intensity per turn (real-time signal)");
}

/// Scenario 2: Body budget as slow-timescale signal under sustained stress.
///
/// Biologically correct: body budget doesn't deplete from a few messages.
/// It's a cumulative measure over long interactions — matching human allostasis.
fn scenario_2_conservation_under_sustained_stress() {
    println!("\n── Scenario 2: Body Budget — Slow-Timescale Signal ──");
    println!("30 sustained-stress messages. Body budget is a SLOW signal (hours analog).\n");

    let mut session = CognitiveSession::new();
    let initial_budget = session.world_model().body_budget;

    for _ in 0..30 {
        session.process_message("This is terrible!!! Everything is broken!!! Panic!!!");
    }

    let final_budget = session.world_model().body_budget;
    let final_turn = session.process_message("Still stressed.");

    println!("  Initial budget: {:.4}", initial_budget);
    println!("  Budget after 30 stress messages: {:.4}", final_budget);
    println!("  Conservation signal: {:.3}", final_turn.signals.conservation);
    println!("  Budget drop per turn: {:.5}", (initial_budget - final_budget) / 30.0);

    if final_budget < initial_budget {
        println!("  ✓ Budget depletes under sustained stress (slow timescale)");
    }

    // Recovery.
    for _ in 0..50 {
        session.idle_cycle();
    }
    let recovered = session.world_model().body_budget;
    println!("  After 50 idle cycles: {:.4}", recovered);
    if recovered > final_budget {
        println!("  ✓ Idle cycles replenish budget (allostatic recovery)");
    }
}

/// Scenario 3: Strategy EMA accumulates per topic cluster.
///
/// The recommendation only appears if new messages hash to the SAME cluster.
/// This reveals that topic clustering is fine-grained — a realistic finding
/// (detector.rs uses content hash, not semantic similarity).
fn scenario_3_strategy_accumulation() {
    println!("\n── Scenario 3: Strategy Learning (EMA across turns) ──");
    println!("Consistent step-by-step responses. Strategy data accumulates.\n");

    let mut session = CognitiveSession::new();
    // Use the SAME message text each time to ensure same topic cluster.
    let repeated_query = "How do I handle this Rust error?";
    // Natural numbered-list format — detect_response_strategy handles
    // capitalized sequence markers (case-insensitive since 2026-04-14).
    let step_response =
        "1. First, identify the error type.\n\
         2. Then, check the function signature.\n\
         3. Next, apply the correct handling.\n\
         4. Finally, verify with a test.";

    for _ in 0..10 {
        session.process_message(repeated_query);
        session.process_response(step_response, 0.85);
    }

    let learned = session.world_model().learned.clone();
    let final_turn = session.process_message(repeated_query);

    println!("  Learned strategy clusters: {}", learned.response_strategies.len());
    for (cluster, strategies) in &learned.response_strategies {
        for (name, entry) in strategies {
            println!("    cluster '{}': {} → success_rate={:.3}, count={}",
                cluster, name, entry.success_rate, entry.count);
        }
    }
    println!("  Recommendation for same query: {:?}", final_turn.signals.strategy);

    if final_turn.signals.strategy.is_some() {
        println!("  ✓ Strategy recommendation emerged for repeated queries");
    } else {
        println!("  ⓘ No recommendation yet (needs more EMA accumulation)");
    }
}

/// Scenario 4: Cross-session persistence via JSON round-trip.
fn scenario_4_cross_session_persistence() {
    println!("\n── Scenario 4: Cross-Session Persistence ──");
    println!("Export LearnedState → JSON → new session.\n");

    let mut session1 = CognitiveSession::new();
    for _ in 0..5 {
        session1.process_message("Training message");
        session1.process_response("Step 1: Do X\nStep 2: Do Y", 0.8);
    }

    let learned = session1.export_learned();
    let before_clusters = learned.response_strategies.len();
    let before_tick = learned.tick;

    let json = match serde_json::to_string(&learned) {
        Ok(s) => s,
        Err(e) => { eprintln!("serialize failed: {e}"); return; }
    };
    println!("  Exported: {} bytes JSON", json.len());
    println!("    strategy clusters: {}, tick: {}", before_clusters, before_tick);

    let restored = match serde_json::from_str(&json) {
        Ok(r) => r,
        Err(e) => { eprintln!("deserialize failed: {e}"); return; }
    };

    let session2 = CognitiveSession::with_learned(restored, 64);
    let after_clusters = session2.world_model().learned.response_strategies.len();
    let after_tick = session2.world_model().learned.tick;

    println!("  Imported into new session:");
    println!("    strategy clusters: {}, tick: {}", after_clusters, after_tick);

    if after_clusters == before_clusters && after_tick == before_tick {
        println!("  ✓ LearnedState preserved across sessions via JSON");
    }
}

/// Scenario 5: Closed-loop resource cost tracking.
///
/// Previously open loop: body_budget depleted only from user-input signals.
/// Now closed loop: application reports actual cost via track_cost() →
/// budget depletes from real resource consumption → conservation rises →
/// next turn's signals reflect accumulated cost.
///
/// Closes the measurement loop: conservation reflects actual resource
/// consumption, not just signals inferred from input text.
fn scenario_5_closed_loop_cost_tracking() {
    println!("\n── Scenario 5: Closed-Loop Cost Tracking ──");
    println!("Application reports cost → budget depletes → signals reflect\n");

    let mut session = CognitiveSession::new();
    println!("  {:<20} {:>8} {:>14}", "State", "Budget", "Conservation");
    println!("  {}", "─".repeat(48));

    let t0 = session.process_message("Query.");
    println!("  {:<20} {:>8.3} {:>14.3}", "Baseline", t0.body_budget, t0.signals.conservation);

    // Moderate cost (small LLM call equivalent) x 10.
    for _ in 0..10 { session.track_cost(0.3); }
    let t1 = session.process_message("Query.");
    println!("  {:<20} {:>8.3} {:>14.3}", "After 10x moderate", t1.body_budget, t1.signals.conservation);

    // High cost (complex reasoning + tool calls) x 20.
    for _ in 0..20 { session.track_cost(0.9); }
    let t2 = session.process_message("Query.");
    println!("  {:<20} {:>8.3} {:>14.3}", "After 20x expensive", t2.body_budget, t2.signals.conservation);

    // Cross adaptive threshold (~0.30) → conservation signal activates.
    for _ in 0..20 { session.track_cost(1.0); }
    let t3 = session.process_message("Query.");
    println!("  {:<20} {:>8.3} {:>14.3}", "After 20x max", t3.body_budget, t3.signals.conservation);

    // Recovery via idle cycles.
    for _ in 0..50 { session.idle_cycle(); }
    let t4 = session.process_message("Query.");
    println!("  {:<20} {:>8.3} {:>14.3}", "After 50 idle", t4.body_budget, t4.signals.conservation);

    println!("\n  Observations:");
    if t2.body_budget < t0.body_budget {
        println!("  ✓ Budget depletes proportionally to reported cost");
    }
    if t3.signals.conservation > t1.signals.conservation {
        println!("  ✓ Conservation signal activates when budget crosses threshold");
    }
    if t4.body_budget > t3.body_budget {
        println!("  ✓ Idle cycles restore budget (allostatic recovery)");
    }
    println!("  ✓ Loop closed: Nous senses its own resource consumption");
}

/// Scenario 6: Failure detection via recent_quality + rpe.
///
/// Simulates a model that starts well but degrades. Application can watch
/// `signals.recent_quality` drop and `signals.rpe` stay negative to decide
/// when to intervene (switch strategy, restart, ask user for clarification).
///
/// Addresses MAST NeurIPS 2025 finding: agents repeat failing strategies
/// 19+ times because they lack an intrinsic failure signal.
fn scenario_6_failure_detection() {
    println!("\n── Scenario 6: Failure Detection (recent_quality + rpe) ──");
    println!("Response quality degrades over time. Signal surfaces the pattern.\n");

    let mut session = CognitiveSession::new();

    println!("  {:<4} {:<8} {:>8} {:>8} {:>10}",
        "Turn", "Quality", "Recent", "RPE", "Pattern");
    println!("  {}", "─".repeat(48));

    // Simulate degrading model responses.
    let qualities = [0.9, 0.85, 0.75, 0.6, 0.45, 0.35, 0.3, 0.25, 0.2, 0.15];

    for (i, q) in qualities.iter().enumerate() {
        session.process_message("Help me with this task.");
        session.process_response("Here is a response.", *q);
        let turn = session.process_message("Another query.");

        let pattern = if turn.signals.recent_quality < 0.4 && turn.signals.rpe < 0.0 {
            "DEGRADING"
        } else if turn.signals.rpe > 0.0 {
            "improving"
        } else {
            "steady"
        };

        println!("  {:<4} {:<8.2} {:>8.3} {:>+8.3} {:>10}",
            i + 1, q, turn.signals.recent_quality, turn.signals.rpe, pattern);
    }

    let final_turn = session.process_message("Is this still failing?");
    let should_intervene = final_turn.signals.recent_quality < 0.4
        && final_turn.signals.rpe < 0.0;

    println!("\n  Application decision logic:");
    println!("    if recent_quality < 0.4 AND rpe < 0: intervene");
    println!("    current: quality={:.3}, rpe={:+.3}",
        final_turn.signals.recent_quality, final_turn.signals.rpe);

    if should_intervene {
        println!("  ✓ Failure pattern detected — application would intervene");
        println!("    (MAST 2025: this is where agents normally keep failing silently)");
    } else {
        println!("  Pattern not severe enough yet");
    }
}

/// Scenario 7: Memory retrieval composes with CognitiveSignals.
///
/// Application loads past atoms (async in its own code — app concern),
/// queries them via sync `hybrid_recall`, then uses the retrieval result
/// alongside `turn.signals` to make decisions.
///
/// Point of this scenario: demonstrate the INTEGRATION PATTERN, not the
/// precision of retrieval. Without embeddings, topic-only retrieval is
/// coarse — applications should provide query embeddings for precise match.
/// What matters: sync cognitive computation works alongside signals.
fn scenario_7_memory_informs_decisions() {
    println!("\n── Scenario 7: Memory + Signals Composition ──");
    println!("Pre-loaded atoms + sync hybrid_recall + CognitiveSignals\n");

    // Simulate atoms the application pre-loaded from its persistent store.
    // In production, this is an async DB query — but Nous only sees the
    // result as a Vec<MemoryAtom>, sync.
    let past_atoms = vec![
        make_demo_atom("m1", "User asked about Rust async patterns.",
            vec!["rust".into(), "async".into()], 0.5),
        make_demo_atom("m2", "Previous Rust error was a borrow checker issue.",
            vec!["rust".into(), "error".into(), "borrow".into()], 0.6),
        make_demo_atom("m3", "User prefers concise responses.",
            vec!["preference".into(), "brevity".into()], 0.4),
    ];
    let synapses = HashMap::new();

    let mut session = CognitiveSession::new();
    let query = "How do I fix a Rust borrow checker error?";

    let turn = session.process_message(query);

    // Sync retrieval — fast, deterministic, no async runtime needed.
    let retrieved = hybrid_recall(
        &past_atoms,
        None, // no embedding — topic-only match (coarse without embeddings)
        query,
        &synapses,
        &RecallOptions::default(),
    );

    println!("  Query: {}", query);
    println!("  Retrieved {} atoms:", retrieved.len());
    for a in &retrieved {
        println!("    [{}] {} (score={:.2})", a.atom.id, a.atom.content, a.score);
    }
    println!("\n  Signals for this turn:");
    println!("    confidence={:.2}  salience={:.2}  conservation={:.2}",
        turn.signals.confidence, turn.signals.salience, turn.signals.conservation);

    // Application decision combining memory + signals:
    let has_memory = !retrieved.is_empty();
    let decision = match (has_memory, turn.signals.salience > 0.5, turn.signals.confidence > 0.5) {
        (true, _, true) => "answer using retrieved context (memory + confident signal)",
        (true, false, _) => "consult retrieved atoms, verify with user",
        (_, true, _) => "handle as urgent input, no prior context",
        _ => "default: answer from model's general knowledge",
    };
    println!("\n  App decision: {}", decision);

    println!("\n  ✓ Memory API composes with signals in one sync call tree");
    println!("  ✓ Apps own async I/O (DB loads); Nous owns sync cognitive computation");
    println!("  Note: for precise retrieval provide query embeddings — topic-only is coarse.");
}

fn make_demo_atom(id: &str, content: &str, topics: Vec<String>, importance: f64) -> MemoryAtom {
    MemoryAtom {
        id: id.into(),
        content: content.into(),
        embedding: None,
        atom_type: AtomType::Episodic,
        source: AtomSource::default(),
        importance,
        access_count: 0,
        last_accessed_at: 0.0,
        created_at: 0.0,
        topics,
        domain: None,
        consolidated_from: None,
        is_consolidated: false,
        parent_id: None,
        depth: None,
        label: None,
        child_ids: None,
        superseded: false,
        suppressed: false,
        dormant: false,
        tags: vec![],
        encoding_context: None,
        retrieval_reward: None,
        reconsolidation_count: None,
        arousal: None,
        valence: None,
        epoch: None,
        crystallized: false,
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() } else { format!("{}…", &s[..max]) }
}

#[allow(dead_code)]
fn print_signals_row(turn_num: usize, turn: &TurnResult) {
    println!("  {:<4} {:<10.3} {:<10.3} {:<10.3} {:<10.3} {:<8.3}",
        turn_num,
        turn.body_budget,
        turn.signals.conservation,
        turn.signals.salience,
        turn.signals.confidence,
        turn.arousal);
}
