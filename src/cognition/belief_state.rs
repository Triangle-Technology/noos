//! Belief state update functions — affect and next-turn predictions.
//!
//! ## Scope after 2026-04-11 non-cortical audit
//!
//! This module previously contained `update_topic_beliefs()` and
//! `compute_topic_pe()` — regex-based topic extraction from text + topic-
//! prediction-error computation. Per P9 (compensate, don't amplify) and the
//! 2026-04-11 SSM research synthesis, these duplicated cortical work:
//! the model's attention tracks topics across context far better than any
//! regex, and topic-level prediction errors are computed implicitly in the
//! model's residual stream.
//!
//! What remains:
//! - `update_affect()` — wraps the arousal heuristic (see `emotional.rs` for
//!   honest naming). Output feeds into compensatory Tầng 2 modulation.
//! - `update_predictions()` — stores next-turn topic cluster key for
//!   cross-session strategy learning. Topic extraction here is NOT a
//!   cognitive claim; it is used purely as a stable hash for the learned
//!   strategy EMA store (dorsomedial striatum analog, Daw 2005).
//!
//! Pure functions, <1ms, $0 LLM cost.

use crate::cognition::detector::extract_topics;
use crate::cognition::emotional::compute_arousal;
use crate::math::clamp;
use crate::types::belief::*;

/// Update affect state from message content.
///
/// Wraps `compute_arousal` heuristic and tracks sustained arousal
/// (Yerkes-Dodson inverted-U). See `emotional.rs` for honest naming of the
/// scalar-arousal interim heuristic.
pub fn update_affect(state: &SharedBeliefState, message: &str) -> AffectState {
    let result = compute_arousal(message);

    // Certainty: higher arousal = higher certainty in the reading
    let certainty = if result.arousal >= 0.6 {
        HIGH_AROUSAL_CERTAINTY
    } else if result.arousal >= 0.3 {
        (LOW_AROUSAL_CERTAINTY + HIGH_AROUSAL_CERTAINTY) / 2.0
    } else {
        LOW_AROUSAL_CERTAINTY
    };

    // Sustained modulator (Yerkes-Dodson inverted-U).
    //
    // Semantics: `sustained` represents remaining arousal-coping CAPACITY in
    // [0, 1], defaulting to 1.0 (full reserve). Prolonged high/moderate arousal
    // DEPLETES capacity; calm intervals RESTORE capacity toward 1.0. Matches:
    // `types/belief.rs` doc ("decays over time"), `signals.rs::compute_signals`
    // formula `(1.0 - sustained) * 0.2` (low sustained → penalty), and
    // `app-contract.md` §1.1 ("prolonged stress depletes capacity").
    //
    // Pre-2026-04-14 bug: both branches ADDED to sustained, pinning it at 1.0
    // forever. Surfaced + fixed via `examples/task_eval_conservation.rs`.
    //
    // Graded threshold (not binary): the regex arousal heuristic rarely exceeds
    // 0.6 for typical stress markers ("HELP!!!" scores ~0.4 on mamba-130m-sized
    // inputs). Using a graded response lets moderate arousal still deplete
    // reserve, matching the biological reality that sub-threatening-but-
    // unpleasant signals also tax allostatic resources (Sterling 2012).
    let sustained_delta = if result.arousal >= 0.6 {
        // High arousal: strong capacity depletion.
        -0.10
    } else if result.arousal >= 0.3 {
        // Moderate arousal: slower depletion — still a tax, just less severe.
        -0.05
    } else {
        // Calm: recovery toward 1.0 baseline.
        0.05
    };
    let sustained = clamp(state.affect.sustained + sustained_delta, 0.0, 1.0);

    AffectState {
        arousal: result.arousal,
        valence: result.valence,
        certainty,
        sustained,
    }
}

/// Update predictions from response content (for next turn).
///
/// Extracts topic cluster candidates from the response. These are used
/// purely as opaque cluster keys for cross-session strategy learning
/// (`world_model::consolidate`'s per-cluster success EMA). They are NOT a
/// claim that Nous is tracking topics cognitively — the model does that.
pub fn update_predictions(response_content: &str) -> Predictions {
    let topics = extract_topics(response_content);
    let next_topics = topics
        .into_iter()
        .take(PREDICTION_TOPIC_COUNT)
        .collect();

    Predictions { next_topics }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_affect_neutral() {
        let state = SharedBeliefState::new("test".into());
        let affect = update_affect(&state, "Hello world");
        assert!(affect.arousal < 0.3);
        assert_eq!(affect.valence, AffectValence::Neutral);
    }

    #[test]
    fn update_affect_high_arousal() {
        let state = SharedBeliefState::new("test".into());
        let affect = update_affect(&state, "This is TERRIBLE and frustrating!!!");
        assert!(affect.arousal > 0.3);
        assert_eq!(affect.certainty, HIGH_AROUSAL_CERTAINTY);
    }

    #[test]
    fn predictions_from_response() {
        let preds = update_predictions("Here is the solution using Rust and tokio runtime");
        assert!(!preds.next_topics.is_empty());
    }
}
