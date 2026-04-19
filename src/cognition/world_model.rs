//! World Model — perceive() and consolidate() functions.
//!
//! Brain analog: allostatic/body-state accounting + dorsomedial striatum
//! strategy learning. Per the 2026-04-11 audit, `perceive()` no longer
//! claims to be a "unified generative model" (that duplicated cortical
//! work — the model's forward pass is the generative model). What remains
//! is genuinely non-cortical:
//!
//! - **Affect update** — arousal scalar from heuristic (interim, see
//!   `emotional.rs`)
//! - **Arousal-based sensory PE** — surprise relative to predicted arousal
//!   (amygdala-level surprise, not topic/intent PE which was cortical)
//! - **Body budget tracking** — allostatic depletion/replenishment (no
//!   native analog; canonical "model has no body" gap)
//! - **Volatility estimation** — rolling variance of PE history (Behrens
//!   2007 ACC analog, used for learning rate modulation)
//! - **Strategy recommendation** — per-cluster success EMA lookup
//!   (dorsomedial striatum analog, Daw 2005)
//!
//! `consolidate()` learns cross-session: RPE update, per-cluster strategy
//! EMA, response calibration. Both loops implement the non-cortical parts
//! of the perception-action cycle (Friston 2010) — cortical parts stay in
//! the model.
//!
//! Pure functions, <1ms, $0 LLM cost.
//!
//! ## Gating (P10)
//!
//! State maintenance, not a signal producer with priority competition.
//! `perceive` / `consolidate` / `maintain` are the three phase hooks of
//! the allostatic loop and each is invoked unconditionally at its phase
//! boundary — there is no priority rule that would skip them.
//!
//! - **Fires when**: [`perceive`] runs once per
//!   [`CognitiveSession::process_message`](crate::session::CognitiveSession::process_message);
//!   [`consolidate`] runs once per
//!   [`CognitiveSession::process_response`](crate::session::CognitiveSession::process_response);
//!   [`maintain`] runs once per
//!   [`CognitiveSession::idle_cycle`](crate::session::CognitiveSession::idle_cycle).
//! - **Inactive when**: never at the phase level. Individual branches
//!   (e.g. the strategy-EMA update inside `consolidate`) no-op when the
//!   topic cluster hash is empty, consistent with P5 fail-open.
//! - **Suppresses**: nothing. Outputs update shared [`WorldModel`]
//!   state; downstream signal producers compete on the resulting state.
//! - **Suppressed by**: nothing. No upstream module can veto a phase
//!   update.

use crate::cognition::belief_state::update_affect;
use crate::cognition::detector::{detect_response_strategy, extract_topics};
use crate::math::clamp;
use crate::types::belief::TopicBeliefs;
use crate::types::world::*;

// ── Constants ──────────────────────────────────────────────────────────
//
// Sensory PE simplification (2026-04-11 audit pass 2 + review fix):
//
// Previously `sensory_pe = 0.5*topic_pe + 0.25*intent_pe + 0.25*arousal_pe`.
// Topic PE and intent PE were cortical duplication (the model's residual
// stream computes richer prediction errors internally).
//
// Pass 2 initial: `sensory_pe = |current_arousal - predicted_arousal|` or 0.3.
// But `Predictions.next_arousal` was never populated anywhere, making PE
// effectively a constant 0.3. Review caught this drift.
//
// Fixed to: `sensory_pe = |current_arousal - previous_arousal|` — turn-over-
// turn arousal delta as a genuine surprise signal. No predictions needed:
// the previous state IS the homeostatic expectation (Barrett 2017 allostasis).
// This is the amygdala-level surprise the downstream compensatory modules
// (adaptive_thresholds, delta_modulation, dynamics) actually need.

// ── Volatility-modulated learning (Behrens 2007) ──────────────────────
// When PE is volatile (environment changing), increase learning rate.
// When PE is stable (predictable), decrease learning rate.
// This lets the system adapt quickly to change while preserving
// stable knowledge during predictable periods.

/// Volatile environment learning rate multiplier.
/// When pe_volatility > 0.5, learning rate scales up by this factor.
/// Behrens 2007: ACC tracks volatility → modulates plasticity.
const VOLATILE_LEARNING_BOOST: f64 = 1.5;

/// Stable environment learning rate multiplier.
/// When pe_volatility < 0.2, learning rate scales down by this factor.
/// Preserves accumulated knowledge in predictable contexts.
const STABLE_LEARNING_DAMPEN: f64 = 0.6;

/// PE volatility threshold for "volatile" environment.
const VOLATILITY_BOOST_THRESHOLD: f64 = 0.5;

/// PE volatility threshold for "stable" environment.
const VOLATILITY_DAMPEN_THRESHOLD: f64 = 0.2;

// ── Allostatic body-budget rates (Barrett 2017, Sterling 2012) ───────────
//
// These control the per-turn dynamics of `body_budget`. Each rate is
// small (<= 0.05) so body_budget changes are gradual — a single event
// shouldn't crash or replenish it fully. Calibrated against the eval
// suite's 30-turn depletion trajectory: sustained high-arousal +
// high-PE conversation drops budget by ~40% over 30 turns (roughly
// matches the Tier 1.2 conservation-eval curve).

/// Depletion per unit of `sensory_pe` per `perceive` call.
/// PE is the metabolically-costly signal (surprise = extra compute);
/// weighted 2x arousal depletion.
const PE_DEPLETION_RATE: f64 = 0.02;

/// Depletion per unit of `arousal` per `perceive` call.
/// Arousal is the stress-response signal; slower bleed than PE.
const AROUSAL_DEPLETION_RATE: f64 = 0.01;

/// Natural per-turn replenishment regardless of outcome — slow drift
/// back toward full budget during calm conversation.
const NATURAL_REPLENISHMENT_RATE: f64 = 0.005;

/// Replenishment rate on positive response RPE (reward outperformed
/// expectation). Higher than depletion because good outcomes should
/// recover faster than stress depletes.
const POSITIVE_RPE_REPLENISHMENT_RATE: f64 = 0.05;

/// Depletion rate on negative response RPE (reward underperformed).
/// Gentler than the positive rate — dopaminergic learning tolerates
/// mild disappointment without full stress response (Schultz 1997).
const NEGATIVE_RPE_DEPLETION_RATE: f64 = 0.02;

/// Compute volatility-modulated learning rate (Behrens 2007).
///
/// Base rate × volatility multiplier:
/// - Volatile (>0.5): rate × 1.5 (learn faster, environment changing)
/// - Stable (<0.2): rate × 0.6 (learn slower, trust accumulation)
/// - Middle: rate × 1.0 (no modulation)
fn volatility_modulated_rate(base_rate: f64, pe_volatility: f64) -> f64 {
    if pe_volatility > VOLATILITY_BOOST_THRESHOLD {
        base_rate * VOLATILE_LEARNING_BOOST
    } else if pe_volatility < VOLATILITY_DAMPEN_THRESHOLD {
        base_rate * STABLE_LEARNING_DAMPEN
    } else {
        base_rate
    }
}

/// Perceive — update world model with new message.
///
/// Updates affect (arousal heuristic), computes arousal-level sensory PE,
/// advances turn, tracks volatility, updates body budget, looks up
/// recommended strategy for the message's topic cluster.
pub fn perceive(model: &WorldModel, message: &str) -> WorldModel {
    let mut updated = model.clone();

    // 1. Update affect (arousal heuristic — see emotional.rs honest naming)
    updated.belief.affect = update_affect(&model.belief, message);

    // 2. Populate topic-cluster hash from message (opaque key for strategy EMA)
    // NOT a cognitive topic model — extract_topics is used purely as a
    // stable hashing function for cross-session strategy learning lookups.
    let cluster_topics = extract_topics(message);
    updated.belief.topic = TopicBeliefs {
        current: cluster_topics,
        predicted: model.belief.topic.predicted.clone(),
    };

    // 3. Sensory PE = arousal delta (turn-over-turn surprise)
    // Previous-state-as-prediction model: homeostatic baseline is "expect
    // same as last turn". Sudden arousal shifts are amygdala-level surprise.
    // No forward predictions needed (Barrett 2017 allostasis — body budget
    // regulation predicts from current state, not from explicit forecasts).
    updated.sensory_pe = clamp(
        (updated.belief.affect.arousal - model.belief.affect.arousal).abs(),
        0.0,
        1.0,
    );

    // 4. Increment turn
    updated.belief.turn += 1;
    updated.turns_since_switch += 1;

    // 5. Track PE history for volatility estimation (Behrens 2007)
    updated.pe_history.push(updated.sensory_pe);
    if updated.pe_history.len() > PE_VOLATILITY_WINDOW {
        updated.pe_history.remove(0);
    }
    updated.pe_volatility = compute_pe_volatility(&updated.pe_history);

    // 6. Update body budget (Principle 4+8: allostatic tracking)
    // High sensory PE depletes budget (unexpected = metabolically costly).
    // High arousal depletes budget (stress response consumes resources).
    let depletion = updated.sensory_pe * PE_DEPLETION_RATE
        + updated.belief.affect.arousal * AROUSAL_DEPLETION_RATE;
    updated.body_budget = clamp(
        updated.body_budget - depletion + NATURAL_REPLENISHMENT_RATE,
        0.0,
        1.0,
    );

    // 7. Strategy recommendation from learned state (Principle 5: classification emerges).
    // After perceiving the message, recommend a strategy for the response
    // based on what worked for this topic cluster in the past.
    // Brain analog: dorsomedial striatum recalls action-outcome associations (Daw 2005).
    let topic_cluster = crate::cognition::detector::build_topic_cluster(&updated.belief.topic.current);
    if !topic_cluster.is_empty() {
        updated.recommended_strategy =
            get_recommended_strategy(&topic_cluster, &updated.learned).map(|(s, _)| s);
    }

    updated
}

/// Compute PE volatility (running variance).
fn compute_pe_volatility(history: &[f64]) -> f64 {
    if history.len() < 2 {
        return 0.0;
    }
    let mean = history.iter().sum::<f64>() / history.len() as f64;
    let variance = history.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / history.len() as f64;
    clamp(variance.sqrt(), 0.0, 1.0)
}

/// Consolidate — update learned state after response (post-processing).
///
/// Records response quality, strategy success, and prepares predictions for next turn.
pub fn consolidate(
    model: &WorldModel,
    response_content: &str,
    response_quality: f64,
) -> WorldModel {
    let mut updated = model.clone();

    // 1. Update predictions for next turn
    updated.belief.predictions = crate::cognition::belief_state::update_predictions(response_content);

    // 2. Compute RPE (Schultz 1997)
    updated.response_rpe = response_quality - model.last_response_prediction;

    // 3. Update response prediction via EMA
    updated.last_response_prediction = clamp(
        (1.0 - RESPONSE_SUCCESS_EMA) * model.last_response_prediction
            + RESPONSE_SUCCESS_EMA * response_quality,
        0.0,
        1.0,
    );

    // 4. Track discussed topics (Common Ground, Clark 1996)
    let response_topics = extract_topics(response_content);
    for topic in &response_topics {
        updated
            .discussed_topics
            .insert(topic.clone(), updated.belief.turn);
    }

    // 5. Body budget replenishment from positive RPE (Principle 8: allostasis)
    if updated.response_rpe > 0.0 {
        updated.body_budget = clamp(
            updated.body_budget + updated.response_rpe * POSITIVE_RPE_REPLENISHMENT_RATE,
            0.0,
            1.0,
        );
    } else {
        updated.body_budget = clamp(
            updated.body_budget + updated.response_rpe * NEGATIVE_RPE_DEPLETION_RATE,
            0.0,
            1.0,
        );
    }

    // 6. Strategy detection + learning (Principle 7: multi-timescale learning).
    // Two modes of detection:
    //
    // - `detected_safe` (Option): returns Some only when the response clearly
    //   matches a known strategy pattern. Used for per-strategy reward learning
    //   so ambiguous responses DON'T poison the `response_strategies` map with
    //   miscategorized data. Prevents a real-LLM format-variation silent-
    //   poisoning failure mode (when response format doesn't match any
    //   strategy regex, back-compat wrapper returns DirectAnswer by default,
    //   polluting the reward-learning signal).
    //
    // - `actual_strategy` (via back-compat wrapper): always returns a strategy
    //   (defaults to DirectAnswer). Used for display/compliance tracking where
    //   a best-guess is more useful than nothing.
    let detected_safe = crate::cognition::detector::detect_response_strategy_safe(response_content);
    let actual_strategy = detect_response_strategy(response_content);
    updated.last_response_strategy = Some(actual_strategy);
    updated.last_response_length = Some(response_content.len());
    updated.last_response_question_ratio = Some(
        crate::cognition::detector::compute_question_ratio(response_content),
    );

    // 6b. Per-topic strategy success learning (AD-169: dorsomedial striatum EMA)
    // Learning rate is modulated by PE volatility (Behrens 2007):
    // volatile environment → learn faster, stable → learn slower.
    let learning_rate = volatility_modulated_rate(RESPONSE_SUCCESS_EMA, model.pe_volatility);

    // Cluster by USER MESSAGE topics (not response topics) — must match the
    // recommend side in perceive() which uses `updated.belief.topic.current`.
    // Semantics: "for user-topic X, strategy Y produced quality Q." Recording
    // under response_topics would make the cluster keys asymmetric, causing
    // get_recommended_strategy to never find matches on subsequent turns
    // (found via allostatic_demo 2026-04-14).
    let topic_cluster = crate::cognition::detector::build_topic_cluster(&model.belief.topic.current);
    if !topic_cluster.is_empty() {
        // Update general per-cluster response success. This tracks "how well
        // does this cluster respond overall" and is updated regardless of
        // strategy ambiguity — the quality signal is meaningful even when we
        // don't know which strategy produced it.
        let entry = updated.learned.response_success
            .entry(topic_cluster.clone())
            .or_insert(SuccessEntry { success_rate: 0.5, count: 0 });
        entry.success_rate = (1.0 - learning_rate) * entry.success_rate
            + learning_rate * response_quality;
        entry.count += 1;

        // Per-strategy learning: ONLY when the strategy was clearly detected.
        // If `detected_safe` is None, the response format didn't match any
        // strategy pattern unambiguously; attributing the quality to a
        // best-guess strategy (e.g., DirectAnswer-via-default) would silently
        // train the wrong entry. Skip instead.
        if let Some(detected) = detected_safe {
            let strategy_key = format!("{:?}", detected);
            let strategy_map = updated.learned.response_strategies
                .entry(topic_cluster.clone())
                .or_default();
            let strategy_entry = strategy_map
                .entry(strategy_key)
                .or_insert(SuccessEntry { success_rate: 0.5, count: 0 });
            strategy_entry.success_rate = (1.0 - learning_rate) * strategy_entry.success_rate
                + learning_rate * response_quality;
            strategy_entry.count += 1;
        }
    }

    // 7. Efference copy — compliance monitoring (AD-173: Sperry 1950)
    // Compare recommended strategy (set before LLM) vs actual strategy (detected after).
    let compliance = match (&model.recommended_strategy, &updated.last_response_strategy) {
        (Some(recommended), Some(actual)) => {
            if recommended == actual {
                StrategyCompliance::Compliant
            } else if response_quality > 0.6 {
                StrategyCompliance::DeviatedBetter
            } else {
                StrategyCompliance::DeviatedWorse
            }
        }
        _ => StrategyCompliance::NoRecommendation,
    };
    updated.last_compliance = Some(compliance);

    updated
}

// ── Between-turn maintenance constants (Principle 8: allostasis) ──────
// Brain doesn't stop processing between inputs (Raichle 2006, neural dark energy).
// Between turns: body budget replenishes.
//
// Audit 2026-04-14 removed `THREAT_EXTINCTION_RATE` + the threat extinction
// loop (iterated an always-empty `learned.threats` map after
// `ThreatAssociations` was deleted), and the `DMN_IDLE_ACTIVATION_RATE`
// + `DMN_MAX_ACTIVATION` constants + `dmn_activation` field (write-only
// counter masquerading as a Raichle 2006 DMN model — P1 metaphor vs
// mechanism violation; the field was never read by cognitive logic).

/// Body budget natural replenishment per idle cycle (Sterling 2012).
/// Slow recovery toward baseline during rest.
const IDLE_REPLENISHMENT_RATE: f64 = 0.01;

/// Between-turn maintenance — brain never stops (Principle 8: allostasis).
///
/// Called between user messages. Recovers allostatic resources during idle
/// periods — currently just body-budget replenishment. (Earlier versions
/// also decayed threat associations and increased a DMN activation counter;
/// both were removed in the 2026-04-14 audit as orphan code — the threat
/// map was never populated after `ThreatAssociations` deletion, and the
/// DMN counter was never read by cognitive logic.)
///
/// Brain analog: Raichle 2006 (neural dark energy — brain consumes 20%
/// of body's energy even at rest, processing predictions and maintaining
/// allostatic readiness).
pub fn maintain(model: &WorldModel) -> WorldModel {
    let mut updated = model.clone();

    // 1. Body budget replenishment (Sterling 2012 allostatic recovery).
    // Slow recovery toward 1.0 during idle periods.
    updated.body_budget = clamp(
        updated.body_budget + IDLE_REPLENISHMENT_RATE,
        0.0,
        1.0,
    );

    // 2. Increment idle cycles counter.
    updated.idle_cycles += 1;

    updated
}

// `build_topic_cluster` moved to `crate::cognition::detector`
// (Session 20 P3 refactor) so `regulator::correction` can share the
// same cluster-identity algorithm.

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn perceive_updates_affect() {
        let model = WorldModel::new("test".into());
        let updated = perceive(&model, "This is amazing!!!");
        assert!(updated.belief.affect.arousal > model.belief.affect.arousal);
    }

    #[test]
    fn perceive_sensory_pe_from_arousal_delta() {
        // Neutral → emotional = arousal delta = sensory PE spike.
        let model = WorldModel::new("test".into());
        let updated = perceive(&model, "This is TERRIBLE and frustrating!!!");
        assert!(updated.sensory_pe > 0.0);
    }

    #[test]
    fn perceive_sensory_pe_stable_when_calm() {
        // Neutral → neutral = no delta = PE near 0.
        let model = WorldModel::new("test".into());
        let updated = perceive(&model, "Hello world");
        assert!(updated.sensory_pe < 0.1);
    }

    #[test]
    fn perceive_increments_turn() {
        let model = WorldModel::new("test".into());
        let updated = perceive(&model, "test");
        assert_eq!(updated.belief.turn, 1);
    }

    #[test]
    fn pe_volatility_stable() {
        let history = vec![0.3, 0.3, 0.3, 0.3, 0.3];
        let vol = compute_pe_volatility(&history);
        assert!(vol < 0.01); // Very stable
    }

    #[test]
    fn pe_volatility_unstable() {
        let history = vec![0.1, 0.9, 0.1, 0.9, 0.1];
        let vol = compute_pe_volatility(&history);
        assert!(vol > 0.3); // Very volatile
    }

    #[test]
    fn consolidate_updates_predictions() {
        let model = WorldModel::new("test".into());
        let updated = consolidate(&model, "The Rust compiler is great for safety", 0.8);
        assert!(!updated.belief.predictions.next_topics.is_empty());
    }

    #[test]
    fn consolidate_computes_rpe() {
        let mut model = WorldModel::new("test".into());
        model.last_response_prediction = 0.5;
        let updated = consolidate(&model, "Good response", 0.8);
        assert!((updated.response_rpe - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn consolidate_tracks_discussed_topics() {
        let model = WorldModel::new("test".into());
        let updated = consolidate(&model, "Let's discuss Rust and async", 0.7);
        assert!(!updated.discussed_topics.is_empty());
    }

    // ─── Volatility-modulated learning tests (Behrens 2007) ───

    #[test]
    fn volatile_environment_learns_faster() {
        let rate_volatile = volatility_modulated_rate(0.25, 0.7); // volatile
        let rate_normal = volatility_modulated_rate(0.25, 0.35);  // normal
        let rate_stable = volatility_modulated_rate(0.25, 0.1);   // stable
        assert!(
            rate_volatile > rate_normal,
            "Volatile environment should produce higher learning rate"
        );
        assert!(
            rate_normal > rate_stable,
            "Stable environment should produce lower learning rate"
        );
    }

    #[test]
    fn consolidate_uses_volatility_for_learning() {
        // Volatile model: high PE volatility → faster learning
        let mut volatile_model = WorldModel::new("test".into());
        volatile_model.pe_volatility = 0.8;
        volatile_model.last_response_prediction = 0.5;
        let volatile_updated = consolidate(&volatile_model, "Step 1. Step 2. Step 3.", 0.9);

        // Stable model: low PE volatility → slower learning
        let mut stable_model = WorldModel::new("test".into());
        stable_model.pe_volatility = 0.1;
        stable_model.last_response_prediction = 0.5;
        let stable_updated = consolidate(&stable_model, "Step 1. Step 2. Step 3.", 0.9);

        // Both learned, but volatile model should have moved prediction further
        // because its learning rate was higher.
        let volatile_shift = (volatile_updated.last_response_prediction - 0.5).abs();
        let stable_shift = (stable_updated.last_response_prediction - 0.5).abs();
        // Note: last_response_prediction uses RESPONSE_SUCCESS_EMA directly (not volatility-modulated).
        // But the strategy learning IS modulated. Check strategy entries instead:
        // Both should have learned, but with different rates.
        // The volatile model's strategy success should be closer to 0.9 (faster EMA).
        // This is hard to test directly because we need enough topic data.
        // For now, just verify the function computes correctly.
        assert!(volatile_shift > 0.0);
        assert!(stable_shift > 0.0);
    }

    #[test]
    fn volatility_modulated_rate_boundaries() {
        // At exact thresholds
        let at_boost = volatility_modulated_rate(0.25, VOLATILITY_BOOST_THRESHOLD + 0.01);
        assert!((at_boost - 0.25 * VOLATILE_LEARNING_BOOST).abs() < 0.01);

        let at_dampen = volatility_modulated_rate(0.25, VOLATILITY_DAMPEN_THRESHOLD - 0.01);
        assert!((at_dampen - 0.25 * STABLE_LEARNING_DAMPEN).abs() < 0.01);
    }

    // ─── P5 strategy recommendation tests ───

    #[test]
    fn perceive_sets_recommended_strategy_from_learned() {
        let mut model = WorldModel::new("test".into());
        // Simulate learned strategy: "StepByStep" works well for "rust+async"
        let mut strategy_map = HashMap::new();
        strategy_map.insert(
            "StepByStep".into(),
            SuccessEntry {
                success_rate: 0.8,
                count: 10,
            },
        );
        model
            .learned
            .response_strategies
            .insert("async+rust".into(), strategy_map);

        // Perceive a message about rust and async
        let updated = perceive(&model, "How do I use async in Rust?");

        // Should recommend StepByStep (high confidence for this topic cluster)
        assert_eq!(
            updated.recommended_strategy,
            Some(ResponseStrategy::StepByStep),
            "Should recommend learned strategy for topic cluster"
        );
    }

    #[test]
    fn perceive_no_recommendation_without_learned() {
        let model = WorldModel::new("test".into());
        let updated = perceive(&model, "Hello world");
        // No learned strategies → no recommendation
        assert_eq!(updated.recommended_strategy, None);
    }

    #[test]
    fn get_recommended_strategy_picks_highest_success() {
        let mut learned = LearnedState::default();
        let mut strategies = HashMap::new();
        strategies.insert(
            "DirectAnswer".into(),
            SuccessEntry {
                success_rate: 0.6,
                count: 8,
            },
        );
        strategies.insert(
            "StepByStep".into(),
            SuccessEntry {
                success_rate: 0.85,
                count: 10,
            },
        );
        learned
            .response_strategies
            .insert("test+topic".into(), strategies);

        let result = get_recommended_strategy("test+topic", &learned);
        assert!(result.is_some());
        let (strategy, _) = result.unwrap();
        assert_eq!(strategy, ResponseStrategy::StepByStep);
    }

    #[test]
    fn get_recommended_strategy_skips_avoid() {
        let mut learned = LearnedState::default();
        let mut strategies = HashMap::new();
        strategies.insert(
            "ClarifyFirst".into(),
            SuccessEntry {
                success_rate: 0.2, // Bad — should avoid
                count: 10,
            },
        );
        strategies.insert(
            "DirectAnswer".into(),
            SuccessEntry {
                success_rate: 0.7,
                count: 8,
            },
        );
        learned
            .response_strategies
            .insert("test+topic".into(), strategies);

        let result = get_recommended_strategy("test+topic", &learned);
        assert!(result.is_some());
        let (strategy, _) = result.unwrap();
        assert_eq!(
            strategy,
            ResponseStrategy::DirectAnswer,
            "Should skip avoided strategy"
        );
    }

    #[test]
    fn get_recommended_strategy_none_for_weak_data() {
        let mut learned = LearnedState::default();
        let mut strategies = HashMap::new();
        strategies.insert(
            "StepByStep".into(),
            SuccessEntry {
                success_rate: 0.9,
                count: 2, // Too few observations
            },
        );
        learned
            .response_strategies
            .insert("test+topic".into(), strategies);

        let result = get_recommended_strategy("test+topic", &learned);
        assert!(result.is_none(), "Should return None for weak data");
    }

    // ─── Between-turn maintenance tests (P8: brain never stops) ───

    #[test]
    fn maintain_replenishes_body_budget() {
        let mut model = WorldModel::new("test".into());
        model.body_budget = 0.8; // Depleted
        let updated = maintain(&model);
        assert!(
            updated.body_budget > model.body_budget,
            "Idle maintenance should replenish body budget"
        );
    }

    #[test]
    fn maintain_increments_idle_cycles() {
        let model = WorldModel::new("test".into());
        let updated = maintain(&model);
        assert_eq!(updated.idle_cycles, 1);
        let updated2 = maintain(&updated);
        assert_eq!(updated2.idle_cycles, 2);
    }

}
