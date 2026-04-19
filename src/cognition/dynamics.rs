//! Conversation Dynamics — Level 3 temporal hierarchy.
//!
//! Brain analog: cortical hierarchy processes at multiple timescales (Murray 2014).
//! Higher levels (L3) predict REGIMES of lower-level (L2) dynamics.
//! Sparse updates via accumulated turn PE (DPC 2024, THICK ICLR 2024).
//!
//! ## 2026-04-11 audit refactor
//!
//! Previously used `model.belief.topic.familiarity` (regex-based topic
//! overlap with context) as the familiarity signal. Per P9, topic tracking
//! duplicates cortical attention. Replaced with a PE-based proxy:
//! **low sensory_pe = familiar territory, high sensory_pe = unfamiliar**.
//! This is the dorsolateral prefrontal surprise signal approximation
//! (Behrens 2007), which is non-cortical in nature (ACC tracks volatility,
//! not semantic content).
//!
//! Pure functions, <1ms, $0 LLM cost.
//!
//! ## Gating (P10)
//!
//! Regime classifier, not a competing signal. Output
//! ([`DynamicsState`]) is read by downstream gating consumers
//! (`adaptive_thresholds`, `resource_allocator`) as a modulation input;
//! priority competition happens there.
//!
//! - **Fires when**: [`detect_regime`] is called once per
//!   [`converge`](super::convergence::converge) step.
//! - **Inactive when**: accumulated PE is below
//!   [`REGIME_REASSESS_PE_THRESHOLD`] — the existing regime persists
//!   rather than being recomputed (Murray 2014 slow-timescale
//!   integration; sparse-update prior from DPC 2024 / THICK ICLR 2024).
//! - **Suppresses**: nothing directly. Regime IS an input to gating
//!   decisions but does not itself suppress another signal.
//! - **Suppressed by**: nothing. No upstream module can veto regime
//!   classification on a given turn.

use crate::types::world::{ConversationRegime, DynamicsState, WorldModel};

// ── Constants ──────────────────────────────────────────────────────────

/// Turns 0-2 = opening regime.
pub const OPENING_MAX_TURNS: u32 = 2;
/// Minimum consecutive turns for deep-dive classification.
pub const DEEP_DIVE_MIN_CONSECUTIVE: u32 = 3;
/// Accumulated PE threshold before regime reassessment (sparse updates).
pub const REGIME_REASSESS_PE_THRESHOLD: f64 = 1.5;
/// Depth increment per continuation turn.
pub const DEPTH_INCREMENT: f64 = 0.15;
/// Depth multiplier on topic switch (decay).
pub const DEPTH_DECAY_ON_SWITCH: f64 = 0.8;

/// Sensory PE below this = familiar territory (low surprise = known domain).
/// Replaces the deprecated `topic.familiarity` regex signal (2026-04-11 audit).
/// Matches Behrens 2007 — ACC tracks prediction-error as a volatility proxy.
pub const FAMILIAR_PE_THRESHOLD: f64 = 0.4;
/// Sensory PE below this + sustained focus = deeply familiar (problem-solving).
pub const DEEP_FAMILIAR_PE_THRESHOLD: f64 = 0.25;

// ── Core Functions ─────────────────────────────────────────────────────

/// Create initial dynamics state.
pub fn create_dynamics_state() -> DynamicsState {
    DynamicsState::default()
}

/// Detect current conversation regime from world model state.
///
/// Implements DPC (Dynamic Predictive Coding): higher level predicts
/// the regime of lower-level dynamics, not the next state directly.
pub fn detect_regime(model: &WorldModel, previous: Option<&DynamicsState>) -> DynamicsState {
    // Phase 1: Opening
    if model.belief.turn <= OPENING_MAX_TURNS {
        return DynamicsState {
            regime: ConversationRegime::Opening,
            depth: 0.0,
            turns_in_regime: model.belief.turn,
            accumulated_turn_pe: 0.0,
        };
    }

    let prev = match previous {
        Some(p) => p,
        None => {
            return DynamicsState {
                regime: ConversationRegime::Exploration,
                depth: 0.0,
                turns_in_regime: 1,
                accumulated_turn_pe: 0.0,
            };
        }
    };

    // Phase 2: Accumulate turn PE
    let new_accumulated_pe = prev.accumulated_turn_pe + model.sensory_pe;

    // Phase 3: Update depth
    // Continuation = same-thread turns + low surprise (PE-based familiarity proxy).
    // Replaced deprecated topic.familiarity (cortical) with sensory_pe threshold.
    let is_continuation =
        model.turns_since_switch > 0 && model.sensory_pe < FAMILIAR_PE_THRESHOLD;
    let new_depth = if is_continuation {
        (prev.depth + DEPTH_INCREMENT).min(1.0)
    } else {
        prev.depth * (1.0 - DEPTH_DECAY_ON_SWITCH)
    };

    // Phase 4: Decide reassessment (sparse — only when enough surprise accumulated)
    let should_reassess = new_accumulated_pe >= REGIME_REASSESS_PE_THRESHOLD
        || prev.regime == ConversationRegime::Opening
        || (model.turns_since_switch == 0 && prev.regime == ConversationRegime::DeepDive);

    if !should_reassess {
        // Maintain current regime, grow confidence
        return DynamicsState {
            regime: prev.regime,
            depth: new_depth,
            turns_in_regime: prev.turns_in_regime + 1,
            accumulated_turn_pe: new_accumulated_pe,
        };
    }

    // Phase 5: Classify new regime
    let new_regime = classify_regime(model, new_depth, prev);

    let turns_in_regime = if new_regime == prev.regime {
        prev.turns_in_regime + 1
    } else {
        1
    };

    DynamicsState {
        regime: new_regime,
        depth: new_depth,
        turns_in_regime,
        accumulated_turn_pe: 0.0, // Reset after reassessment
    }
}

/// Classify the conversation regime from current state.
fn classify_regime(
    model: &WorldModel,
    depth: f64,
    previous: &DynamicsState,
) -> ConversationRegime {
    // Divergent: switch away from deep-dive
    if model.turns_since_switch == 0 && previous.regime == ConversationRegime::DeepDive {
        return ConversationRegime::Divergent;
    }

    // Problem-solving: sustained focus + deep familiarity + low arousal
    // Deep familiarity proxy: very low sensory_pe = model is not surprised,
    // domain is well-known (Behrens 2007 ACC volatility signal inverted).
    if model.turns_since_switch >= DEEP_DIVE_MIN_CONSECUTIVE
        && depth > 0.4
        && model.sensory_pe < DEEP_FAMILIAR_PE_THRESHOLD
        && model.belief.affect.arousal < 0.4
    {
        return ConversationRegime::ProblemSolving;
    }

    // Deep-dive: sustained focus + moderate depth
    if model.turns_since_switch >= DEEP_DIVE_MIN_CONSECUTIVE && depth > 0.3 {
        return ConversationRegime::DeepDive;
    }

    // Exploration: recent switch or not enough depth
    ConversationRegime::Exploration
}

/// Format regime for LLM context (compact summary).
pub fn format_regime_for_llm(state: &DynamicsState) -> Option<String> {
    match state.regime {
        ConversationRegime::Opening => None, // Don't clutter opening turns
        _ => Some(format!(
            "Phase: {:?} (depth {:.1}, {} turns)",
            state.regime, state.depth, state.turns_in_regime
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test helper. `sensory_pe` serves as the familiarity proxy:
    // low PE = familiar, high PE = unfamiliar (per 2026-04-11 audit).
    fn make_model(turn: u32, turns_since_switch: u32, sensory_pe: f64) -> WorldModel {
        let mut model = WorldModel::new("test".into());
        model.belief.turn = turn;
        model.turns_since_switch = turns_since_switch;
        model.sensory_pe = sensory_pe;
        model
    }

    #[test]
    fn opening_regime() {
        // Opening: first few turns, regardless of PE
        let model = make_model(1, 0, 0.3);
        let state = detect_regime(&model, None);
        assert_eq!(state.regime, ConversationRegime::Opening);
    }

    #[test]
    fn exploration_after_opening() {
        // Moderate PE after opening → exploration (not familiar enough for deep-dive)
        let model = make_model(3, 1, 0.5);
        let prev = DynamicsState {
            regime: ConversationRegime::Opening,
            depth: 0.0,
            turns_in_regime: 2,
            accumulated_turn_pe: 0.0,
        };
        let state = detect_regime(&model, Some(&prev));
        assert_eq!(state.regime, ConversationRegime::Exploration);
    }

    #[test]
    fn deep_dive_sustained_focus() {
        // Low PE (0.2 < FAMILIAR_PE_THRESHOLD 0.4) = continuation builds depth.
        // Arousal 0.5 prevents problem-solving classification.
        // After enough focus + depth → deep-dive.
        let mut model = make_model(8, 5, 0.2);
        model.belief.affect.arousal = 0.5;
        let prev = DynamicsState {
            regime: ConversationRegime::Exploration,
            depth: 0.5,
            turns_in_regime: 3,
            accumulated_turn_pe: 2.0, // Enough PE to reassess
        };
        let state = detect_regime(&model, Some(&prev));
        assert_eq!(state.regime, ConversationRegime::DeepDive);
    }

    #[test]
    fn problem_solving_calm_deep_familiar() {
        // Very low PE (0.1 < DEEP_FAMILIAR_PE_THRESHOLD 0.25) = deep familiarity
        // + low arousal + sustained focus + depth → problem-solving regime.
        let mut model = make_model(10, 5, 0.1);
        model.belief.affect.arousal = 0.2;
        let prev = DynamicsState {
            regime: ConversationRegime::DeepDive,
            depth: 0.6,
            turns_in_regime: 4,
            accumulated_turn_pe: 2.0,
        };
        let state = detect_regime(&model, Some(&prev));
        assert_eq!(state.regime, ConversationRegime::ProblemSolving);
    }

    #[test]
    fn divergent_after_deep_dive_switch() {
        // High PE (0.8) + topic switch (turns_since_switch=0) from deep-dive → divergent.
        let model = make_model(10, 0, 0.8);
        let prev = DynamicsState {
            regime: ConversationRegime::DeepDive,
            depth: 0.6,
            turns_in_regime: 5,
            accumulated_turn_pe: 0.5,
        };
        let state = detect_regime(&model, Some(&prev));
        assert_eq!(state.regime, ConversationRegime::Divergent);
    }

    #[test]
    fn sparse_update_no_reassess() {
        // Low PE per-turn + accumulated below threshold → maintain regime.
        let model = make_model(5, 3, 0.1);
        let prev = DynamicsState {
            regime: ConversationRegime::Exploration,
            depth: 0.3,
            turns_in_regime: 2,
            accumulated_turn_pe: 0.2, // Below threshold
        };
        let state = detect_regime(&model, Some(&prev));
        assert_eq!(state.regime, ConversationRegime::Exploration);
        assert_eq!(state.turns_in_regime, 3);
        assert!(state.accumulated_turn_pe > 0.2);
    }

    #[test]
    fn depth_increases_on_continuation() {
        // Low PE (0.1 < FAMILIAR_PE_THRESHOLD 0.4) = familiar territory =
        // continuation → depth grows.
        let model = make_model(5, 3, 0.1);
        let prev = DynamicsState {
            regime: ConversationRegime::Exploration,
            depth: 0.3,
            turns_in_regime: 2,
            accumulated_turn_pe: 0.1,
        };
        let state = detect_regime(&model, Some(&prev));
        assert!(state.depth > 0.3);
    }

    #[test]
    fn format_opening_returns_none() {
        let state = DynamicsState {
            regime: ConversationRegime::Opening,
            depth: 0.0,
            turns_in_regime: 1,
            accumulated_turn_pe: 0.0,
        };
        assert!(format_regime_for_llm(&state).is_none());
    }

    #[test]
    fn format_deep_dive_returns_some() {
        let state = DynamicsState {
            regime: ConversationRegime::DeepDive,
            depth: 0.6,
            turns_in_regime: 5,
            accumulated_turn_pe: 0.3,
        };
        let formatted = format_regime_for_llm(&state).unwrap();
        assert!(formatted.contains("DeepDive"));
    }
}
