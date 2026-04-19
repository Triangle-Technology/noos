//! Regulator persistence envelope.
//!
//! **Scope note (P1 / P9b)**: pure data-shape module, not cognitive.
//! Everything here is `#[derive(Serialize, Deserialize)]` on fields
//! that already exist in [`LearnedState`] (Path 1) or
//! [`CorrectionPattern`] (Path 2 structural patterns) — no new signal
//! computation happens at this layer.
//! Snapshot / restore only. P1 applies to the wrapped
//! [`CognitiveSession`](crate::session::CognitiveSession) that owns
//! `LearnedState`; P9b is satisfied by construction.
//!
//! Session 20 split `RegulatorState` out of `regulator::mod` into its
//! own module so the persistence surface grows independently of the
//! dispatch surface. Older snapshots round-trip cleanly because every
//! added field carries `#[serde(default)]`.
//!
//! ## What stays inside LearnedState vs RegulatorState
//!
//! Per-session cognitive state (strategy EMA, response success per
//! cluster, `tick`, `gain_mode`) stays in
//! [`LearnedState`] — it's
//! Path-1-native and `CognitiveSession::import_learned` restores it.
//!
//! Path-2-only state (procedural correction patterns, future cost /
//! reputation envelopes) lives directly on `RegulatorState`. That
//! keeps the Layer 0 (`types/`) boundary clean: `LearnedState` never
//! has to know about `CorrectionPattern`, and external Path 1 users
//! who only touch `CognitiveSession::export_learned` never see Path 2
//! additions leak in.
//!
//! Deviation from the original Session 15 plan: the architecture memo
//! proposed `LearnedState.correction_patterns: HashMap<(String,
//! String), CorrectionPattern>`, keyed by `(user_id, topic_cluster)`.
//! Session 20 moved the field here (keyed by `topic_cluster` alone,
//! `user_id` implicit from the enclosing Regulator) so the Layer-0
//! types module stays free of Path 2 types, and so snapshots don't
//! need tuple-key JSON tricks. See the Session 20 status memo for the
//! full rationale.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::world::LearnedState;

use super::CorrectionPattern;

/// Serialisable snapshot of a [`Regulator`](super::Regulator).
///
/// Wraps the underlying Path 1 [`LearnedState`] plus Path-2-only
/// persistent state. Persist via `serde_json` (or any `serde`
/// backend); restore with [`Regulator::import`](super::Regulator::import).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatorState {
    pub user_id: String,
    pub learned: LearnedState,
    /// Session 20: per-topic-cluster procedural patterns learned from
    /// repeated user corrections. Keyed by the same
    /// [`build_topic_cluster`](crate::cognition::detector::build_topic_cluster)
    /// hash that `LearnedState.response_strategies` uses, so strategy
    /// learning + correction patterns agree on cluster identity.
    ///
    /// Pre-Session-20 snapshots omit this field — `#[serde(default)]`
    /// supplies an empty map so old exports still load.
    #[serde(default)]
    pub correction_patterns: HashMap<String, CorrectionPattern>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state_round_trips() {
        let s = RegulatorState {
            user_id: "alice".into(),
            learned: LearnedState::default(),
            correction_patterns: HashMap::new(),
        };
        let json = serde_json::to_string(&s).expect("serialise");
        let back: RegulatorState = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(back.user_id, "alice");
        assert!(back.correction_patterns.is_empty());
    }

    #[test]
    fn pre_session_20_snapshot_deserialises_with_empty_patterns() {
        // Snapshot shape from Sessions 16–19: no `correction_patterns`
        // field. `#[serde(default)]` must fill it with an empty map so
        // old exports still load cleanly after Session 20 upgrades the
        // struct.
        let legacy_json = r#"{
            "user_id": "legacy_user",
            "learned": {
                "gain_mode": "neutral",
                "tick": 0,
                "response_success": {},
                "response_strategies": {}
            }
        }"#;
        let state: RegulatorState =
            serde_json::from_str(legacy_json).expect("legacy snapshot must load");
        assert_eq!(state.user_id, "legacy_user");
        assert!(
            state.correction_patterns.is_empty(),
            "missing field must default to empty map"
        );
    }

    #[test]
    fn correction_patterns_survive_round_trip() {
        let mut patterns = HashMap::new();
        patterns.insert(
            "refactor+async".to_string(),
            CorrectionPattern {
                user_id: "alice".into(),
                topic_cluster: "refactor+async".into(),
                pattern_name: "corrections_on_refactor+async".into(),
                learned_from_turns: 3,
                confidence: 0.15,
                example_corrections: vec![
                    "no more logs".into(),
                    "stop adding logging".into(),
                    "don't add logging".into(),
                ],
            },
        );
        let s = RegulatorState {
            user_id: "alice".into(),
            learned: LearnedState::default(),
            correction_patterns: patterns,
        };
        let json = serde_json::to_string(&s).expect("serialise");
        let back: RegulatorState = serde_json::from_str(&json).expect("deserialise");
        let restored = back
            .correction_patterns
            .get("refactor+async")
            .expect("pattern should round-trip");
        assert_eq!(restored.learned_from_turns, 3);
        assert_eq!(restored.example_corrections.len(), 3);
        assert_eq!(restored.example_corrections[0], "no more logs");
    }
}
