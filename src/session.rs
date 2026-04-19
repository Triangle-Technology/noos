//! Cognitive Session — the public API for applications.
//!
//! Brain analog: a complete cognitive cycle. Each turn mirrors how the brain
//! processes a stimulus: perceive → attend → decide → respond → learn.
//! CognitiveSession manages all internal state (WorldModel, LocusCoeruleus,
//! conversation history) so the application only needs to call two methods:
//! `process_message()` and `process_response()`.
//!
//! Key papers: Friston 2010 (perception-action cycle), Doya 1999 (modulatory loops).
//!
//! See `docs/intervention.md` for how TurnResult feeds into model generation.

use crate::cognition::convergence::{converge, ConvergenceContext};
use crate::cognition::delta_modulation::compute_delta_modulation;
use crate::cognition::intervention::{build_cognitive_state, compute_sampling_override};
use crate::cognition::locus_coeruleus::LocusCoeruleus;
use crate::cognition::resource_allocator::ModelTier;
use crate::cognition::signals::compute_signals;
use crate::cognition::world_model::{consolidate, maintain, perceive};
use crate::math::clamp;
use crate::types::belief::AffectValence;
use crate::types::gate::{GateType, RecentMessage};
use crate::types::intervention::{CognitiveSignals, CognitiveState, DeltaModulation, SamplingOverride};
use crate::types::world::{GainMode, LearnedState, ResponseStrategy, WorldModel};

/// Maximum recent messages kept for gate context (working memory capacity).
const MAX_RECENT_MESSAGES: usize = 10;

/// Body budget depletion per unit of reported cost (track_cost).
/// 0.02 at cost=1.0 depletes budget by 2% — comparable to a high-arousal
/// turn. Low enough that single actions don't crash the system, high enough
/// that sustained high-cost operations meaningfully accumulate depletion.
const COST_DEPLETION_RATE: f64 = 0.02;

// ── Gate feedback EMA (inject_gate_feedback) ─────────────────────────────
//
// Per-token feedback from the cognitive gate into affect + sensory PE.
// Values below are EMA blend weights — chosen so a single salient token
// bumps the state by ~20%, while sustained salience drives the state
// toward the gate's own signal over ~4 tokens. Matches the gamma-cycle
// thalamocortical latency from the convergence-loop derivation (Lamme 2000).

/// Gate activation threshold. Below this, gate is considered passive and
/// arousal decays toward baseline. Above, gate is actively modulating and
/// arousal EMA-blends toward the gate signal.
const GATE_ACTIVE_MIN_ALPHA: f64 = 0.1;

/// Prior weight on arousal during active EMA blend. `0.8` means each
/// token advances the arousal ~20% toward the gate's alpha — 4 tokens
/// to saturate.
const GATE_AROUSAL_PRIOR_WEIGHT: f64 = 0.8;

/// Decay factor applied per token when the gate is passive. `0.95` gives
/// a ~14-token half-life — enough to remember recent salience without
/// locking arousal high after a single burst.
const GATE_PASSIVE_DECAY: f64 = 0.95;

/// Minimum `|delta_gain - 1.0|` before it counts as a gate-surprise
/// signal. Small deviations from unit gain are treated as noise.
const GATE_SURPRISE_MIN_DELTA: f64 = 0.05;

/// Prior weight on sensory_pe during gate-surprise EMA blend. `0.7`
/// advances PE ~30% per token — faster than arousal because PE is a
/// more transient signal (surprise should decay quickly after the
/// triggering token).
const GATE_PE_PRIOR_WEIGHT: f64 = 0.7;

/// High-level cognitive session — the application-facing API.
///
/// Manages all internal cognitive state. Applications call:
/// 1. `process_message(user_input)` → get cognitive state + sampling params
/// 2. [Application generates response using sampling params]
/// 3. `process_response(response, quality)` → learning + strategy update
///
/// This is the perception-action cycle (Friston 2010) in one struct.
pub struct CognitiveSession {
    model: WorldModel,
    lc: LocusCoeruleus,
    history: Vec<RecentMessage>,
    turn_count: usize,
    /// Number of layers in the target model (for delta modulation layer targeting).
    /// Default 64 (Falcon Mamba 7B). Applications should set this to match their model.
    num_model_layers: usize,
}

/// Result of processing a user message through the full cognitive pipeline.
///
/// Contains everything an application needs to generate a cognitively-modulated response.
/// Includes both Tầng 1 (sampling) and Tầng 2 (delta modulation) intervention signals.
#[derive(Debug, Clone)]
pub struct TurnResult {
    /// Unified cognitive state snapshot (for InferenceEngine or inspection).
    pub cognitive_state: CognitiveState,
    /// Tầng 1: Sampling parameter overrides (temperature, top_p, penalties).
    pub sampling: SamplingOverride,
    /// Tầng 2: Delta modulation for SSM state injection.
    /// Application passes this to CognitiveModel::forward_cognitive().
    /// If the model doesn't support ActivationAccess, this is informational only.
    pub delta_modulation: DeltaModulation,
    /// How many convergence iterations were needed.
    pub convergence_iterations: usize,
    /// Whether convergence loop settled (delta < epsilon).
    pub converged: bool,
    /// Recommended response strategy from learned state (None if first turn or weak data).
    pub recommended_strategy: Option<ResponseStrategy>,
    /// Current body budget (0-1, allostatic resources).
    pub body_budget: f64,
    /// Sensory prediction error (0-1, how surprising this message was).
    pub sensory_pe: f64,
    /// Current LC-NE gain mode.
    pub gain_mode: GainMode,
    /// Thalamic gate classification.
    pub gate_type: GateType,
    /// Gate classification confidence (0-1).
    pub gate_confidence: f64,
    /// Emotional arousal (0-1).
    pub arousal: f64,
    /// Affective valence.
    pub valence: AffectValence,
    /// Application-facing allostatic signals (Phase 7).
    /// Organized around application DECISIONS: conservation, salience, confidence.
    pub signals: CognitiveSignals,
}

impl CognitiveSession {
    /// Create a new cognitive session for a conversation.
    /// Uses default 64 layers (Falcon Mamba 7B) for delta modulation targeting.
    pub fn new() -> Self {
        Self {
            model: WorldModel::new("session".into()),
            lc: LocusCoeruleus::new(),
            history: Vec::new(),
            turn_count: 0,
            num_model_layers: 64,
        }
    }

    /// Create a session configured for a specific model size.
    /// `num_layers`: total layers in the target model (e.g., 64 for Falcon Mamba 7B,
    /// 32 for smaller models). Used for delta modulation layer targeting.
    pub fn with_model_layers(num_layers: usize) -> Self {
        Self {
            num_model_layers: num_layers,
            ..Self::new()
        }
    }

    /// Mutable: runs full cognitive pipeline on user message.
    /// Updates WorldModel, LocusCoeruleus, history.
    /// Requires mutation because cognitive state accumulates across turns
    /// (body budget, gain mode, topic context, learned strategies).
    ///
    /// Pipeline:
    /// 1. perceive() → affect, topics, sensory PE, body budget, strategy recommendation
    /// 2. converge() → damped iterative settling (thalamocortical gamma cycles)
    /// 3. build_cognitive_state() → unified snapshot
    /// 4. compute_sampling_override() → Tầng 1: temperature, top_p, penalties
    /// 5. compute_delta_modulation() → Tầng 2: SSM delta gain_factor + layer targeting
    pub fn process_message(&mut self, message: &str) -> TurnResult {
        // 1. Perceive — update world model with new sensory input.
        self.model = perceive(&self.model, message);

        // 2. Converge — damped iterative settling.
        let ctx = ConvergenceContext {
            model_tier: ModelTier::Medium,
            fok_average: None, // no memory system connected yet
            has_graph_data: false,
            message_count: self.turn_count,
        };
        let convergence = converge(
            &self.model,
            message,
            &self.history,
            &mut self.lc,
            &ctx,
        );
        self.model = convergence.model;

        // 3. Build cognitive state + Tầng 1 sampling + Tầng 2 delta modulation + Phase 7 signals.
        let cognitive_state = build_cognitive_state(&self.model, self.lc.gain_mode());
        let sampling = compute_sampling_override(&cognitive_state);
        let delta_modulation =
            compute_delta_modulation(&cognitive_state, self.num_model_layers);
        let signals = compute_signals(&self.model, self.lc.gain_mode());

        // 4. Record message in history (for next turn's gate context).
        self.history.push(RecentMessage {
            role: "user".into(),
            content: message.to_string(),
        });
        if self.history.len() > MAX_RECENT_MESSAGES {
            self.history.remove(0);
        }
        self.turn_count += 1;

        TurnResult {
            cognitive_state,
            sampling,
            delta_modulation,
            convergence_iterations: convergence.iterations,
            converged: convergence.converged,
            recommended_strategy: self.model.recommended_strategy,
            body_budget: self.model.body_budget,
            sensory_pe: self.model.sensory_pe,
            gain_mode: self.lc.gain_mode(),
            gate_type: self.model.gate.gate,
            gate_confidence: self.model.gate.confidence,
            arousal: self.model.belief.affect.arousal,
            valence: self.model.belief.affect.valence,
            signals,
        }
    }

    /// Mutable: processes model response for learning.
    /// Updates learned state (strategy success, RPE, predictions).
    /// Requires mutation because learning accumulates across responses
    /// (strategy EMA, calibration, compliance tracking).
    pub fn process_response(&mut self, response: &str, quality: f64) {
        self.model = consolidate(&self.model, response, quality);

        // Record response in history.
        self.history.push(RecentMessage {
            role: "assistant".into(),
            content: response.to_string(),
        });
        if self.history.len() > MAX_RECENT_MESSAGES {
            self.history.remove(0);
        }
    }

    /// Mutable: runs between-turn maintenance (Principle 8: brain never stops).
    /// Predictions decay, body budget replenishes, threats extinguish, DMN activates.
    /// Call between turns when the system is idle.
    ///
    /// Brain analog: Raichle 2006 (neural dark energy — brain processes even at rest).
    pub fn idle_cycle(&mut self) {
        self.model = maintain(&self.model);
    }

    /// Mutable: report actual resource cost of a completed operation.
    /// Depletes body_budget proportional to cost, closing the allostatic loop.
    ///
    /// Closes the open loop: previously `body_budget` only depleted from
    /// user-input signals (arousal, PE). Now the application can report
    /// actual resource consumption (tokens, latency, API calls — normalized
    /// to [0, 1]), making Noos sense its own cost.
    ///
    /// `cost`: [0, 1] normalized effort. 0 = trivial (cached response).
    /// 1 = exhausting (long reasoning chain, many tool calls). Application
    /// decides the normalization — Noos uses the scalar as an effort signal.
    ///
    /// Brain analog: anterior cingulate cortex (ACC) effort monitoring +
    /// hypothalamic metabolic cost tracking. The organism senses caloric
    /// expenditure independently of external stressors (Shenhav 2013,
    /// Expected Value of Control).
    ///
    /// Typical call pattern:
    ///   session.process_message(user_msg)
    ///   // application generates response, knows cost
    ///   session.track_cost(actual_cost)
    ///   session.process_response(response, quality)
    pub fn track_cost(&mut self, cost: f64) {
        // `math::clamp` absorbs NaN → 0.0 (P5 fail-open). A user-provided
        // NaN cost must not cascade into body_budget (CR4 invariants).
        let cost = clamp(cost, 0.0, 1.0);
        // Depletion rate matches the scale of per-turn depletion in perceive()
        // (which is ~0.015 per high-arousal turn). Cost acts as a parallel
        // channel — real effort depletes just like stress does.
        let depletion = cost * COST_DEPLETION_RATE;
        self.model.body_budget = clamp(self.model.body_budget - depletion, 0.0, 1.0);
    }

    /// Mutable: inject gate feedback from model inference into cognitive
    /// state. Updates `model.belief.affect.arousal` and
    /// `model.sensory_pe` via per-token EMA blend. Requires mutation
    /// because arousal and PE accumulate across tokens in the
    /// thalamocortical loop (CR4 invariants apply — clamped to `[0, 1]`).
    ///
    /// Closes the thalamocortical loop: model (cortex) → gate (thalamus) →
    /// subcortical state update → next token's cognitive modulation.
    ///
    /// gate_alpha: gate blend factor [0, 1]. Higher = gate actively modulating
    /// = model detected salient content. Maps to arousal (subcortical interpretation
    /// of cortical salience signal).
    ///
    /// gate_delta_gain: gate's learned state update speed [0.5, 2.0].
    /// > 1.0 = attend to current input. < 1.0 = preserve history.
    ///
    /// Call after each generate_next_cognitive() to create per-token feedback.
    pub fn inject_gate_feedback(&mut self, gate_alpha: f64, gate_delta_gain: f64) {
        // Gate alpha → arousal: high alpha = gate actively modulating = salient input.
        // EMA blend so arousal accumulates across tokens, not jumps.
        if gate_alpha > GATE_ACTIVE_MIN_ALPHA {
            self.model.belief.affect.arousal = self.model.belief.affect.arousal
                * GATE_AROUSAL_PRIOR_WEIGHT
                + gate_alpha * (1.0 - GATE_AROUSAL_PRIOR_WEIGHT);
        } else {
            // Gate passive (alpha ≈ 0) — decay arousal toward baseline.
            self.model.belief.affect.arousal *= GATE_PASSIVE_DECAY;
        }

        // Gate delta_gain → sensory PE proxy: deviation from 1.0 = surprise.
        // |gain - 1.0| measures how much the gate wants to change processing.
        let gate_surprise = (gate_delta_gain - 1.0).abs();
        if gate_surprise > GATE_SURPRISE_MIN_DELTA {
            self.model.sensory_pe = self.model.sensory_pe * GATE_PE_PRIOR_WEIGHT
                + gate_surprise * (1.0 - GATE_PE_PRIOR_WEIGHT);
        }

        // Arousal-based gain trigger (same as convergence loop — Aston-Jones 2005).
        self.lc.set_arousal(self.model.belief.affect.arousal);
    }

    /// Access current world model (read-only, for inspection/debugging).
    pub fn world_model(&self) -> &WorldModel {
        &self.model
    }

    /// Current turn count.
    pub fn turn_count(&self) -> usize {
        self.turn_count
    }

    // ── Cross-Session Persistence (Phase 7: Allostatic Controller) ────

    /// Export learned state for cross-session persistence.
    ///
    /// Returns a serializable snapshot of everything Noos learned this session:
    /// strategy success rates (striatal EMA), response calibration, LC gain
    /// mode, and LC tick count.
    ///
    /// Application responsibility: serialize (serde_json) and store to disk/DB.
    /// Call at session end or periodically.
    ///
    /// Brain analog: hippocampal→neocortical consolidation — ephemeral learning
    /// becomes persistent through explicit transfer (Diekelmann & Born 2010).
    ///
    /// The LC (tick + gain_mode) mutates in-memory during `process_message`
    /// via `nudge_gain_from_confidence` and `set_arousal`, but those mutations
    /// live on the LC struct rather than on `model.learned`. This method
    /// flushes the LC state into the returned snapshot so a subsequent
    /// `import_learned` restores the exact gain trajectory.
    pub fn export_learned(&self) -> LearnedState {
        let mut snapshot = self.model.learned.clone();
        self.lc.sync_to_learned(&mut snapshot);
        snapshot
    }

    /// Mutable: restores cross-session learning from a previous session.
    /// Requires mutation because it overwrites the current learned state
    /// (threats, strategies, calibration, gain mode) and syncs LC state.
    /// Call before processing any messages in a new session.
    ///
    /// Brain analog: retrieving consolidated knowledge at the start of a new day.
    /// The organism wakes with its learned associations intact.
    pub fn import_learned(&mut self, learned: LearnedState) {
        self.model.learned = learned;
        self.lc.sync_from_learned(&self.model.learned);
    }

    /// Create a new session pre-loaded with learned state from a previous session.
    ///
    /// Equivalent to `new()` followed by `import_learned(learned)`.
    pub fn with_learned(learned: LearnedState, num_layers: usize) -> Self {
        let mut session = Self::with_model_layers(num_layers);
        session.import_learned(learned);
        session
    }
}

impl Default for CognitiveSession {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_session_starts_calm() {
        let session = CognitiveSession::new();
        assert_eq!(session.model.body_budget, 1.0);
        assert_eq!(session.model.belief.affect.arousal, 0.0);
        assert_eq!(session.turn_count, 0);
    }

    #[test]
    fn process_message_returns_turn_result() {
        let mut session = CognitiveSession::new();
        let result = session.process_message("Hello, how are you?");
        assert!(result.converged);
        assert!(result.body_budget > 0.0);
        assert!(result.sampling.temperature > 0.0);
        assert_eq!(session.turn_count, 1);
    }

    #[test]
    fn arousal_rises_on_emotional_input() {
        let mut session = CognitiveSession::new();
        let calm = session.process_message("Hello");
        let angry = session.process_message("I'm so frustrated!!! Nothing works!!!");
        assert!(
            angry.arousal > calm.arousal,
            "Emotional input should raise arousal"
        );
    }

    #[test]
    fn body_budget_depletes_under_stress() {
        let mut session = CognitiveSession::new();
        let initial_budget = session.world_model().body_budget;

        // Repeated high-arousal messages should deplete budget.
        for _ in 0..5 {
            session.process_message("This is terrible!!! Everything is broken!!!");
        }

        assert!(
            session.world_model().body_budget < initial_budget,
            "Repeated stress should deplete body budget"
        );
    }

    #[test]
    fn process_response_updates_learning() {
        let mut session = CognitiveSession::new();
        session.process_message("How do I use async in Rust?");
        session.process_response(
            "Here's a step-by-step guide:\n1. First, add tokio\n2. Then, use async fn",
            0.8,
        );

        // Should have detected strategy and updated learned state.
        assert!(session.world_model().last_response_strategy.is_some());
    }

    #[test]
    fn strategy_learning_across_turns() {
        let mut session = CognitiveSession::new();

        // Simulate multiple turns with consistent strategy success.
        for i in 0..10 {
            session.process_message(&format!("Tell me about Rust topic {i}"));
            session.process_response(
                "Here's a step-by-step explanation:\n1. First step\n2. Second step\n3. Third step",
                0.85,
            );
        }

        // After enough turns, next message should get a recommendation.
        let _result = session.process_message("Another Rust question");
        // May or may not have recommendation depending on topic cluster matching,
        // but the learning system should have accumulated data.
        assert!(!session.world_model().learned.response_strategies.is_empty());
    }

    #[test]
    fn turn_result_includes_delta_modulation() {
        let mut session = CognitiveSession::new();
        let result = session.process_message("Hello");
        // Neutral state → gain_factor should be 1.0 (no modulation).
        assert_eq!(
            result.delta_modulation.gain_factor, 1.0,
            "Calm message should produce neutral delta modulation"
        );
        assert_eq!(result.delta_modulation.target.total_layers, 64);
    }

    #[test]
    fn custom_model_layers() {
        let mut session = CognitiveSession::with_model_layers(32);
        let result = session.process_message("Hello");
        assert_eq!(
            result.delta_modulation.target.total_layers, 32,
            "Should target 32-layer model"
        );
    }

    #[test]
    fn history_limited_to_max() {
        let mut session = CognitiveSession::new();
        for i in 0..20 {
            session.process_message(&format!("Message {i}"));
            session.process_response(&format!("Response {i}"), 0.7);
        }
        assert!(session.history.len() <= MAX_RECENT_MESSAGES);
    }

    #[test]
    fn convergence_settles_quickly() {
        let mut session = CognitiveSession::new();
        let result = session.process_message("Simple hello");
        // Should converge in ≤5 iterations (CR2: max iterations).
        assert!(result.convergence_iterations <= 5);
        assert!(result.converged);
    }

    #[test]
    fn idle_cycle_replenishes_budget() {
        let mut session = CognitiveSession::new();
        // Deplete budget with stress.
        for _ in 0..5 {
            session.process_message("Everything is terrible!!!");
        }
        let depleted = session.world_model().body_budget;

        // Run idle cycles.
        for _ in 0..10 {
            session.idle_cycle();
        }

        assert!(
            session.world_model().body_budget > depleted,
            "Idle cycles should replenish body budget"
        );
    }

    // ─── Gate feedback tests (thalamocortical loop) ───

    #[test]
    fn gate_feedback_high_alpha_increases_arousal() {
        let mut session = CognitiveSession::new();
        let initial = session.world_model().belief.affect.arousal;

        // High gate_alpha = gate actively modulating = salient input.
        session.inject_gate_feedback(0.8, 1.0);

        assert!(
            session.world_model().belief.affect.arousal > initial,
            "High gate_alpha should increase arousal"
        );
    }

    #[test]
    fn gate_feedback_low_alpha_decays_arousal() {
        let mut session = CognitiveSession::new();
        // First set some arousal.
        session.inject_gate_feedback(0.9, 1.0);
        let elevated = session.world_model().belief.affect.arousal;
        assert!(elevated > 0.0);

        // Low gate_alpha = gate passive = decay arousal.
        for _ in 0..5 {
            session.inject_gate_feedback(0.0, 1.0);
        }

        assert!(
            session.world_model().belief.affect.arousal < elevated,
            "Low gate_alpha should decay arousal"
        );
    }

    #[test]
    fn gate_feedback_delta_deviation_updates_pe() {
        let mut session = CognitiveSession::new();
        let initial_pe = session.world_model().sensory_pe;

        // gate_delta_gain far from 1.0 = surprise signal.
        session.inject_gate_feedback(0.0, 1.5);

        assert!(
            session.world_model().sensory_pe > initial_pe,
            "Delta gain deviation from 1.0 should increase sensory PE"
        );
    }

    // ─── Cross-session persistence tests ───

    #[test]
    fn export_import_preserves_learned_state() {
        let mut session = CognitiveSession::new();
        // Train strategy learning — needs enough turns for EMA to accumulate.
        for i in 0..8 {
            session.process_message(&format!("How to fix Rust error {i}?"));
            session.process_response(
                "Step 1: Check the error\nStep 2: Fix it\nStep 3: Test",
                0.8,
            );
        }

        let learned = session.export_learned();
        // Strategy data should have accumulated.
        let has_data = !learned.response_strategies.is_empty()
            || !learned.response_success.is_empty()
            || learned.tick > 0;
        assert!(has_data, "Should have learned some cross-session data");

        // `export_learned` must flush LC tick — 8 converge() calls should
        // have ticked nudge_gain_from_confidence at least once. A tick of 0
        // means the LC→LearnedState sync is missing (silent data loss bug).
        assert!(
            learned.tick > 0,
            "LC tick must be flushed into exported LearnedState, got {}",
            learned.tick
        );

        // New session with imported state.
        let session2 = CognitiveSession::with_learned(learned.clone(), 64);
        assert_eq!(
            session2.world_model().learned.tick,
            learned.tick,
            "Imported session should preserve tick count"
        );
    }

    #[test]
    fn export_import_preserves_strategies() {
        let mut session = CognitiveSession::new();
        // Train strategy success.
        for i in 0..10 {
            session.process_message(&format!("Rust question {i}"));
            session.process_response(
                "Step 1: First\nStep 2: Second\nStep 3: Third",
                0.85,
            );
        }

        let learned = session.export_learned();
        assert!(
            !learned.response_strategies.is_empty(),
            "Should have learned strategy data"
        );

        // New session inherits strategies.
        let session2 = CognitiveSession::with_learned(learned.clone(), 64);
        assert_eq!(
            session2.world_model().learned.response_strategies.len(),
            learned.response_strategies.len(),
        );
    }

    #[test]
    fn signals_reflect_cognitive_state() {
        let mut session = CognitiveSession::new();
        let calm = session.process_message("Hello, how are you?");
        assert!(calm.signals.conservation < 0.3, "Calm message → low conservation");
        assert!(calm.signals.salience < 0.5, "Calm message → low salience");

        // Stress should increase conservation (budget depletes).
        for _ in 0..5 {
            session.process_message("Everything is terrible!!! PANIC!!!");
        }
        let stressed = session.process_message("Still terrible!!!");
        assert!(
            stressed.signals.conservation > calm.signals.conservation,
            "Stressed session should have higher conservation than calm"
        );
    }

    // ─── Resource cost tracking tests (closed-loop allostasis) ───

    #[test]
    fn track_cost_depletes_budget() {
        let mut session = CognitiveSession::new();
        let initial = session.world_model().body_budget;
        session.track_cost(1.0); // Max cost.
        let after = session.world_model().body_budget;
        assert!(after < initial, "Full cost should deplete body budget");
    }

    #[test]
    fn track_cost_zero_no_change() {
        let mut session = CognitiveSession::new();
        let initial = session.world_model().body_budget;
        session.track_cost(0.0);
        assert_eq!(session.world_model().body_budget, initial,
            "Zero cost should not deplete budget");
    }

    #[test]
    fn track_cost_clamps_invalid_input() {
        let mut session = CognitiveSession::new();
        session.track_cost(10.0); // Out of range — should clamp to 1.0.
        // Budget should drop by at most COST_DEPLETION_RATE (2%).
        assert!(
            session.world_model().body_budget >= 1.0 - 0.021,
            "Cost must clamp — over-range should not deplete more than 1.0 worth"
        );
    }

    #[test]
    fn track_cost_accumulates_across_calls() {
        let mut session = CognitiveSession::new();
        let initial = session.world_model().body_budget;
        // 30 calls at cost=0.8 should show meaningful cumulative depletion.
        for _ in 0..30 {
            session.track_cost(0.8);
        }
        let depleted = session.world_model().body_budget;
        assert!(
            depleted < initial - 0.1,
            "Sustained cost should accumulate depletion, initial={} final={}",
            initial, depleted
        );
    }

    #[test]
    fn track_cost_reflected_in_conservation_signal() {
        let mut session = CognitiveSession::new();
        let baseline = session.process_message("Hello");
        // Simulate expensive operations.
        for _ in 0..20 {
            session.track_cost(1.0);
        }
        let after_cost = session.process_message("Hello");
        assert!(
            after_cost.signals.conservation > baseline.signals.conservation,
            "Cost-depleted budget should raise conservation signal"
        );
    }
}
