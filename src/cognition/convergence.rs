//! Convergence Loop — damped iterative belief propagation.
//!
//! Brain analog: thalamocortical gamma cycle settling (Lamme 2000, CORnet-S).
//! 5 bidirectional feedback connections create circular dependencies
//! resolved by damped iteration with EMA blending (α=0.5).
//!
//! The 5 connections:
//! 1. Gate ↔ Arousal amplification (LeDoux 1996)
//! 2. Resource pressure → Gate selectivity (Barrett 2017)
//! 3. Resource pressure → Sensory PE modulation (allostatic perception)
//! 4. Gate confidence → Gain mode (Aston-Jones & Cohen 2005)
//! 5. Allocator → Resource pressure (allostatic load feedback)
//!
//! Converges in 2-5 iterations, <25ms, 0 LLM calls.
//!
//! ## Gating (P10)
//!
//! Currently the 5 connections run in parallel within each iteration without
//! explicit priority. This is a known gap (SC SIGNALS v1 failure mode): at
//! high arousal, breadth-oriented signals (resource allocator expansion,
//! lowered ignition threshold) should be **suppressed** rather than averaged
//! with salient signals. Real thalamocortical dynamics implement this via
//! amygdala low-road dominance (LeDoux 1996) suppressing insular breadth
//! checks during emotional salience.
//!
//! Partial gating exists via `adaptive_thresholds`:
//! - `threshold_resource_pressure` has `arousal_weight = -0.05` (threshold
//!   lowers under arousal → pressure signal fires sooner when agent is stressed)
//! - `threshold_delta_arousal_emergency` has `volatility_weight = -0.10`
//!   (threshold lowers in volatile environments → faster phasic override)
//! - `build_threshold_context` applies affect modulation before any threshold
//!   is computed (Principle 2+4 — affect pre-colors precision)
//!
//! Explicit per-connection priority rules are a TODO (gating corollary
//! owed from the precision-signal audit).
//!
//! Pure function (except LocusCoeruleus mutation via gain nudge), <25ms, $0 LLM cost.

use crate::cognition::locus_coeruleus::LocusCoeruleus;
use crate::cognition::resource_allocator::{
    allocate_context_budget, compute_resource_pressure, AllocatorContext, ModelTier,
};
use crate::cognition::adaptive_thresholds::build_threshold_context;
use crate::cognition::thalamic_gate::classify_gate_with_feedback;
use crate::cognition::world_model::perceive;
use crate::math::clamp;
use crate::types::gate::{GateContext, GateContextWithFeedback, RecentMessage};
use crate::types::world::WorldModel;

// ── Constants ──────────────────────────────────────────────────────────

/// Convergence threshold: 1% change = perceptually insignificant (Lamme 2000).
/// Below this delta, further iterations yield no meaningful perceptual update.
pub const CONVERGENCE_EPSILON: f64 = 0.01;
/// Max settling iterations (CORnet-S, Kubilius 2018: gamma cycle saturation).
/// Brain thalamocortical loops settle in 3-6 gamma cycles (~25ms).
pub const MAX_ITERATIONS: usize = 5;
/// EMA damping factor (0.5 = equal blend of new and old).
/// Prevents oscillation while allowing meaningful updates per iteration.
pub const DAMPING_ALPHA: f64 = 0.5;

// ── Result ─────────────────────────────────────────────────────────────

/// Result of the convergence loop.
#[derive(Debug, Clone)]
pub struct ConvergenceResult {
    pub model: WorldModel,
    pub iterations: usize,
    pub converged: bool,
    pub final_delta: f64,
}

/// Context parameters for convergence (grouping reduces argument count — P8).
///
/// Separates the immutable contextual facts about the conversation from the
/// mutable cognitive state. Use `ConvergenceContext::default()` for defaults.
#[derive(Debug, Clone, Default)]
pub struct ConvergenceContext {
    /// Model quality tier (affects budget allocation).
    pub model_tier: ModelTier,
    /// Average feeling-of-knowing confidence (from memory retrieval, if any).
    pub fok_average: Option<f64>,
    /// Whether project graph data is loaded.
    pub has_graph_data: bool,
    /// Total messages in conversation (affects pressure baseline).
    pub message_count: usize,
}

// ── Core Algorithm ─────────────────────────────────────────────────────

/// Run the convergence loop — iterative settling of fast cognitive modules.
///
/// Takes the current world model and a new message, returns the settled model.
/// Pure function (except locus coeruleus state mutation via `lc`).
///
/// # Arguments
/// * `model` — Current world model state
/// * `message` — New user message
/// * `recent_messages` — Recent conversation history
/// * `lc` — Locus coeruleus state (mutated for gain nudge)
/// * `ctx` — Convergence context (model tier, fok, graph data, message count)
pub fn converge(
    model: &WorldModel,
    message: &str,
    recent_messages: &[RecentMessage],
    lc: &mut LocusCoeruleus,
    ctx: &ConvergenceContext,
) -> ConvergenceResult {
    let mut current = model.clone();
    let mut final_delta = 1.0;
    let mut iterations = 0;
    let mut previous_computed: Option<WorldModel> = None;

    for i in 0..MAX_ITERATIONS {
        iterations = i + 1;

        // Run one iteration of the feedback loop
        let computed = run_one_iteration(
            &current,
            message,
            recent_messages,
            lc,
            ctx,
        );

        // Convergence check: compare consecutive raw computations
        if let Some(ref prev) = previous_computed {
            final_delta = compute_max_delta(prev, &computed);
            if final_delta < CONVERGENCE_EPSILON {
                current = damp_state(&computed, &current, DAMPING_ALPHA);
                break;
            }
        }

        previous_computed = Some(computed.clone());
        current = damp_state(&computed, &current, DAMPING_ALPHA);
    }

    ConvergenceResult {
        model: current,
        iterations,
        converged: final_delta < CONVERGENCE_EPSILON,
        final_delta,
    }
}

/// Run one iteration of the feedback loop.
fn run_one_iteration(
    model: &WorldModel,
    message: &str,
    recent_messages: &[RecentMessage],
    lc: &mut LocusCoeruleus,
    ctx: &ConvergenceContext,
) -> WorldModel {
    // Step 1: PERCEIVE — sensory PE + allostatic feedback (#3)
    let after_perceive = perceive(model, message);

    // Step 1b: BUILD THRESHOLD CONTEXT — affect permeation (Principle 2+4)
    // Affect pre-colors all precision signals before any threshold is computed.
    let threshold_ctx = build_threshold_context(
        after_perceive.sensory_pe,
        after_perceive.belief.affect.arousal,
        model.gate.confidence,
        after_perceive.pe_volatility,
        Some(after_perceive.belief.affect.valence),
        Some(after_perceive.body_budget),
    );

    // Step 2: GATE — classification with arousal + pressure feedback (#1, #2)
    let base_ctx = GateContext {
        message,
        recent_messages,
        arousal: threshold_ctx.arousal, // Affect-modulated arousal (Principle 4)
    };
    let gate_ctx = GateContextWithFeedback {
        base: base_ctx,
        resource_pressure: after_perceive.resource_pressure,
        previous_gate: Some(&model.gate),
    };
    let gate_result = classify_gate_with_feedback(&gate_ctx);

    // Step 3: GAIN NUDGE — LC arousal + gate confidence → NE mode (#4)
    // Aston-Jones 2005: LC responds to arousal (salient stimuli) and confidence (task utility).
    lc.set_arousal(after_perceive.belief.affect.arousal);
    let rpe_adjusted = clamp(
        gate_result.confidence + after_perceive.response_rpe * 0.15,
        0.0,
        1.0,
    );
    lc.nudge_gain_from_confidence(rpe_adjusted);

    // Step 4: ALLOCATE — resource budget based on gate + gain
    let alloc_ctx = AllocatorContext {
        query: message,
        gate_result: Some(&gate_result),
        gain_mode: lc.gain_mode(),
        arousal: after_perceive.belief.affect.arousal,
        fok_average: ctx.fok_average,
        model_tier: ctx.model_tier,
        has_graph_data: ctx.has_graph_data,
        active_file_count: 0, // Not available in convergence loop
        pinned_count: 0,
        has_prospective: false,
        message_count: ctx.message_count,
        has_threat_topics: false,
    };
    let allocation = allocate_context_budget(&alloc_ctx, &[]);

    // Step 5: COMPUTE PRESSURE — allostatic load feedback (#5)
    let resource_pressure = compute_resource_pressure(allocation.as_ref());

    // Build updated model
    let mut result = after_perceive;
    result.gate = gate_result;
    result.resource_pressure = resource_pressure;
    result
}

/// Compute maximum delta between two model states.
///
/// Checks all numeric fields that participate in the feedback loop.
fn compute_max_delta(before: &WorldModel, after: &WorldModel) -> f64 {
    let deltas = [
        (after.belief.affect.arousal - before.belief.affect.arousal).abs(),
        (after.belief.affect.certainty - before.belief.affect.certainty).abs(),
        (after.belief.affect.sustained - before.belief.affect.sustained).abs(),
        (after.sensory_pe - before.sensory_pe).abs(),
        (after.resource_pressure - before.resource_pressure).abs(),
        (after.gate.confidence - before.gate.confidence).abs(),
        if after.gate.gate != before.gate.gate {
            1.0
        } else {
            0.0
        },
    ];

    deltas
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max)
}

/// Damp state with EMA blending (prevents oscillation).
///
/// Numeric fields: α × new + (1-α) × old
/// Discrete fields (gate type, topics): use new value
fn damp_state(computed: &WorldModel, previous: &WorldModel, alpha: f64) -> WorldModel {
    let blend = |new: f64, old: f64| alpha * new + (1.0 - alpha) * old;

    let mut result = computed.clone();

    // Damp numeric fields
    result.belief.affect.arousal = blend(computed.belief.affect.arousal, previous.belief.affect.arousal);
    result.belief.affect.certainty = blend(computed.belief.affect.certainty, previous.belief.affect.certainty);
    result.belief.affect.sustained = blend(computed.belief.affect.sustained, previous.belief.affect.sustained);
    result.sensory_pe = blend(computed.sensory_pe, previous.sensory_pe);
    result.resource_pressure = blend(computed.resource_pressure, previous.resource_pressure);
    result.gate.confidence = blend(computed.gate.confidence, previous.gate.confidence);

    // Discrete fields: use computed values (no damping)
    // gate.gate, topics, user state — already from computed

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::gate::GateType;

    #[test]
    fn convergence_simple_message() {
        let model = WorldModel::new("test".into());
        let mut lc = LocusCoeruleus::new();

        let result = converge(
            &model,
            "Hello world",
            &[],
            &mut lc,
            &ConvergenceContext::default(),
        );

        assert!(result.iterations >= 1);
        assert!(result.iterations <= MAX_ITERATIONS);
        // Simple message should converge quickly
        assert!(result.converged || result.iterations <= 3);
    }

    #[test]
    fn convergence_high_arousal() {
        let model = WorldModel::new("test".into());
        let mut lc = LocusCoeruleus::new();

        let result = converge(
            &model,
            "ERROR! Everything is broken!!! HELP!!!",
            &[],
            &mut lc,
            &ConvergenceContext::default(),
        );

        assert!(result.model.belief.affect.arousal > 0.0);
        assert_eq!(result.model.gate.gate, GateType::Urgent);
    }

    #[test]
    fn convergence_max_iterations() {
        let model = WorldModel::new("test".into());
        let mut lc = LocusCoeruleus::new();

        let result = converge(
            &model,
            "Test message",
            &[],
            &mut lc,
            &ConvergenceContext::default(),
        );

        assert!(result.iterations <= MAX_ITERATIONS);
    }

    #[test]
    fn damping_blends_values() {
        let mut a = WorldModel::new("test".into());
        let mut b = WorldModel::new("test".into());
        a.belief.affect.arousal = 1.0;
        b.belief.affect.arousal = 0.0;

        let damped = damp_state(&a, &b, 0.5);
        assert!((damped.belief.affect.arousal - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn max_delta_identical() {
        let model = WorldModel::new("test".into());
        assert!(compute_max_delta(&model, &model) < f64::EPSILON);
    }

    #[test]
    fn max_delta_gate_change() {
        let mut a = WorldModel::new("test".into());
        let mut b = WorldModel::new("test".into());
        a.gate.gate = GateType::Novel;
        b.gate.gate = GateType::Urgent;
        assert_eq!(compute_max_delta(&a, &b), 1.0);
    }

    #[test]
    fn convergence_result_model_has_gate() {
        let model = WorldModel::new("test".into());
        let mut lc = LocusCoeruleus::new();

        let result = converge(
            &model,
            "Just a normal message about programming",
            &[],
            &mut lc,
            &ConvergenceContext::default(),
        );

        // Gate should be classified
        assert!(!result.model.gate.reason.is_empty());
    }

    // ─── LC mutation invariants (CR2: convergence invariants are sacred) ───
    //
    // `converge()` mutates `LocusCoeruleus` via `&mut lc` each iteration. The
    // damping logic only EMA-blends `WorldModel` fields; LC state (gain_mode
    // enum, arousal scalar) is NOT damped. Without these tests, a regression
    // where LC oscillates between iterations could pass `final_delta <
    // CONVERGENCE_EPSILON` (WorldModel fields quiesce) while LC gain_mode
    // flips Phasic ↔ Tonic each turn, producing a silently unstable output.

    #[test]
    fn lc_gain_mode_stable_on_repeated_converge() {
        // Similarly, gain_mode should not flip between Phasic/Tonic/Neutral
        // just by re-running convergence on identical input (short of a
        // genuinely ambiguous borderline case).
        let model = WorldModel::new("test".into());
        let mut lc = LocusCoeruleus::new();

        let _ = converge(
            &model,
            "Neutral message about weather",
            &[],
            &mut lc,
            &ConvergenceContext::default(),
        );
        let mode_first = lc.gain_mode();

        let _ = converge(
            &model,
            "Neutral message about weather",
            &[],
            &mut lc,
            &ConvergenceContext::default(),
        );
        let mode_second = lc.gain_mode();

        assert_eq!(
            mode_first, mode_second,
            "LC gain_mode should be stable on repeated convergence; got \
             {mode_first:?} then {mode_second:?}"
        );
    }

    #[test]
    fn convergence_reports_converged_flag_meaningfully() {
        // `converged = true` claims the iterative loop reached delta < epsilon.
        // Verify: when flag is true, the ACTUAL delta is indeed below epsilon.
        // This guards against a regression where `converged` could report true
        // on MAX_ITERATIONS timeout with a large final_delta.
        let model = WorldModel::new("test".into());
        let mut lc = LocusCoeruleus::new();

        let result = converge(
            &model,
            "Simple test message",
            &[],
            &mut lc,
            &ConvergenceContext::default(),
        );

        if result.converged {
            assert!(
                result.final_delta < CONVERGENCE_EPSILON,
                "converged=true claim violated: final_delta={} >= epsilon={}",
                result.final_delta, CONVERGENCE_EPSILON
            );
        }
    }
}
