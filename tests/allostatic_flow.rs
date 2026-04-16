//! Integration tests — exercise Phase 7 allostatic API as an external consumer.
//!
//! These tests use only public re-exports from nous_regulator::* and verify that the
//! full flow (session creation → per-turn signals → cost tracking → reward
//! learning → cross-session persistence) works as documented.
//!
//! Distinct from module tests: module tests validate individual components.
//! These validate the COMPOSITION — that parts work together through the
//! intended public API surface.

use nous_regulator::session::CognitiveSession;
use nous_regulator::types::belief::AffectValence;
use nous_regulator::types::world::{GainMode, LearnedState};

#[test]
fn full_turn_cycle_produces_signals() {
    let mut session = CognitiveSession::new();
    let turn = session.process_message("Hello, how are you?");

    // All 8 CognitiveSignals fields should be populated.
    assert!(turn.signals.conservation >= 0.0 && turn.signals.conservation <= 1.0);
    assert!(turn.signals.salience >= 0.0 && turn.signals.salience <= 1.0);
    assert!(turn.signals.confidence >= 0.0 && turn.signals.confidence <= 1.0);
    assert_eq!(turn.signals.gain_mode, GainMode::Neutral);
    assert_eq!(turn.signals.valence, AffectValence::Neutral);
    assert!(turn.signals.recent_quality >= 0.0);
    // rpe can be negative, just finite.
    assert!(turn.signals.rpe.is_finite());
    // First turn: no learned strategy yet.
    assert!(turn.signals.strategy.is_none());
}

#[test]
fn emotional_input_raises_salience_and_arousal() {
    let mut session = CognitiveSession::new();
    let calm = session.process_message("Hello.");
    let stressed = session.process_message("Everything is TERRIBLE!!! I'm in PANIC!!!");

    assert!(
        stressed.signals.salience > calm.signals.salience,
        "Emotional content should raise salience (LC phasic burst analog)"
    );
    assert!(stressed.arousal > calm.arousal);
    assert_eq!(stressed.valence, AffectValence::Negative);
}

#[test]
fn track_cost_depletes_budget_and_raises_conservation() {
    let mut session = CognitiveSession::new();
    let before = session.process_message("Query").signals.conservation;

    // Expensive operations — push below adaptive threshold.
    for _ in 0..40 {
        session.track_cost(1.0);
    }

    let after = session.process_message("Query").signals.conservation;
    assert!(
        after > before,
        "Heavy cost tracking should raise conservation signal"
    );
    assert!(session.world_model().body_budget < 0.5);
}

#[test]
fn idle_cycles_recover_from_stress() {
    let mut session = CognitiveSession::new();
    for _ in 0..30 {
        session.track_cost(1.0);
    }
    let depleted = session.world_model().body_budget;
    assert!(depleted < 0.5);

    for _ in 0..100 {
        session.idle_cycle();
    }
    let recovered = session.world_model().body_budget;
    assert!(
        recovered > depleted,
        "Idle cycles should replenish budget (allostasis recovery)"
    );
}

#[test]
fn reward_learning_populates_strategies() {
    let mut session = CognitiveSession::new();
    let repeated = "How do I fix this bug?";
    let step_response =
        "1. First, reproduce the bug.\n\
         2. Then, identify the failing condition.\n\
         3. Next, patch the code.\n\
         4. Finally, verify with a test.";

    for _ in 0..8 {
        session.process_message(repeated);
        session.process_response(step_response, 0.85);
    }

    let turn = session.process_message(repeated);
    // After 8 successful consistent turns with quality 0.85, a recommendation
    // should surface for the same query's topic cluster.
    assert!(
        turn.signals.strategy.is_some(),
        "Repeated success should surface strategy recommendation"
    );
}

#[test]
fn cross_session_persistence_preserves_state() {
    let mut session1 = CognitiveSession::new();
    for i in 0..5 {
        session1.process_message(&format!("Training message {i}"));
        session1.process_response("1. First do X\n2. Then do Y\n3. Next do Z", 0.8);
    }

    // Export + serialize + deserialize + import — simulate full persistence cycle.
    let learned: LearnedState = session1.export_learned();
    let json = serde_json::to_string(&learned).expect("LearnedState serializable");
    let restored: LearnedState = serde_json::from_str(&json).expect("LearnedState deserializable");

    let session2 = CognitiveSession::with_learned(restored, 64);
    assert_eq!(
        session2.world_model().learned.response_strategies.len(),
        learned.response_strategies.len()
    );
    assert_eq!(session2.world_model().learned.tick, learned.tick);
}

#[test]
fn failure_pattern_detectable_from_signals() {
    let mut session = CognitiveSession::new();
    let qualities = [0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1];

    for q in qualities {
        session.process_message("Help me.");
        session.process_response("attempt", q);
    }

    let turn = session.process_message("Another attempt?");
    // After consistently degrading quality, the EMA should be clearly low
    // and RPE negative — application can detect the pattern from signals.
    assert!(
        turn.signals.recent_quality < 0.5,
        "recent_quality EMA should reflect sustained low quality, got {}",
        turn.signals.recent_quality
    );
    assert!(
        turn.signals.rpe < 0.0,
        "Most recent RPE should be negative, got {}",
        turn.signals.rpe
    );
}

#[test]
fn signals_serialize_for_telemetry() {
    // Application use case: serialize signals to JSON for logging / telemetry.
    let mut session = CognitiveSession::new();
    let turn = session.process_message("Hello");

    let json = serde_json::to_string(&turn.signals).expect("signals should serialize");
    let restored: nous_regulator::types::intervention::CognitiveSignals =
        serde_json::from_str(&json).expect("signals should deserialize");

    assert_eq!(restored.gain_mode, turn.signals.gain_mode);
    assert_eq!(restored.valence, turn.signals.valence);
    assert!((restored.conservation - turn.signals.conservation).abs() < 1e-10);
}
