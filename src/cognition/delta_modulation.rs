//! Delta Modulation — cognitive state → SSM delta scaling.
//!
//! Compensatory modulation: help model retain more state, don't amplify processing.
//!
//! Perplexity eval finding (2026-04-10, `examples/diagnose_harm.rs`):
//! - gain > 1.0 (amplify) HURTS: +5.21% at 1.2, +15.7% at 1.5
//! - gain < 1.0 (compensate) HELPS: -2.87% at 0.8, -1.86% at 0.9
//! - Model's learned delta is a FLOOR, not an optimum
//! - State retention can be enhanced; processing speed should not increase
//!
//! Principle (P9): Don't amplify what model already does. Compensate structural limits.
//! All modulation targets gain ≤ 1.0 (enhance retention, preserve context).
//!
//! ## Gating (P10)
//!
//! Delta modulation implements gating along THREE axes:
//!
//! - **Classification** (gate-type conditioning): `GateType::Routine` inputs
//!   short-circuit to `gain = 1.0` passthrough (CR5-motivated defensive fix
//!   2026-04-14). Aligns with P9 — "stay passive when cortex is sufficient."
//!   **Note**: in the production `CognitiveSession` pipeline, Technical text
//!   is classified Novel (not Routine) so this short-circuit does not fire
//!   on the Technical regression case it was originally named for — that
//!   regression lives in the per-token HS path which bypasses the thalamic
//!   gate. See `compute_delta_modulation` inline comment for full validation
//!   status.
//! - **Spatial** (layer targeting): modulation applies only to layers 40-60%
//!   of depth (HiSPA 2026 "critical transport corridor", Block 29 r = -0.9707).
//!   Early and late layers are gated OUT — they do encoding and decoding
//!   that should not be touched.
//! - **Directional** (compensatory only): all signals push gain DOWNWARD
//!   (more retention). No signal can increase gain above 1.0. The safety
//!   rails (`GAIN_MIN = 0.5`, `GAIN_MAX = 2.0`) are physics bounds per
//!   Mamba Modulation NeurIPS 2025 + Lyapunov ICLR 2025, but the cognitive
//!   logic itself never reaches above 1.0.
//!
//! When `GainMode::Neutral` or `GateType::Routine`, gain is exactly 1.0 and
//! no further modulation applies — the module stays out of the way when
//! cortex is sufficient. This matches CognitiveGate self-learning alpha ≈ 0
//! for routine content.
//!
//! Key papers:
//! - Hidden Attention of Mamba Models (ACL 2025: delta = implicit attention decay)
//! - HiSPA 2026 (mid-layers 40-60% = critical transport corridor)
//! - Mamba Modulation (NeurIPS 2025: uniform 2-3× = catastrophic)
//! - Lyapunov Stability of Mamba (ICLR 2025: small perturbations stable)
//!
//! Pure function, O(1), $0 LLM cost.

use crate::cognition::adaptive_thresholds::{
    build_threshold_context, get_adaptive_threshold, threshold_body_budget_conservation,
    threshold_delta_arousal_emergency, threshold_delta_volatility,
};
use crate::math::vector::clamp;
use crate::types::gate::GateType;
use crate::types::intervention::{
    CognitiveState, DeltaModulation, DeltaModulationSource, LayerTarget,
};
use crate::types::world::GainMode;

// ─── Layer Targeting (HiSPA 2026) ───────────────────────────────────
//
// HiSPA found "critical transport corridor" in Mamba at 44-58% depth.
// Block 29 (out of 64) showed highest correlation with downstream
// behavior (r = -0.9707). Targeting 40-60% is conservative encompass.
//
// Applied to any model size: fraction of total layers, not fixed indices.
const TARGET_DEPTH_START: f64 = 0.40;
const TARGET_DEPTH_END: f64 = 0.60;

// ─── Gain Mode → Delta Scale (corrected by perplexity eval data) ───
//
// All modes target gain ≤ 1.0 (compensatory retention enhancement).
// Data: gain > 1.0 HURTS prediction. gain < 1.0 HELPS.
//
// Phasic: focused retention — preserve relevant context more strongly.
// Tonic: broad retention — preserve wide context for exploration.
// Both are < 1.0 because model always benefits from more retention.
const GAIN_PHASIC: f64 = 0.90;
const GAIN_TONIC: f64 = 0.80;
const GAIN_NEUTRAL: f64 = 1.0;

// ─── Modulation Strengths ──────────────────────────────────────────
//
// All signals push gain DOWNWARD (more retention). No signal increases gain.
// This is the compensatory principle: help model retain, don't amplify.

/// Max delta reduction when body budget fully depleted (Barrett 2017).
/// Low budget → preserve state even more (conserve resources).
const BUDGET_DELTA_REDUCTION: f64 = 0.10;
/// Max delta reduction under high PE volatility (Behrens 2007).
/// Volatile environment → hold onto what you know (stabilize state).
const VOLATILITY_DELTA_REDUCTION: f64 = 0.08;
/// Max delta reduction under high arousal (LeDoux 1996).
/// Arousal → preserve emotional/salient context across tokens.
const AROUSAL_DELTA_REDUCTION: f64 = 0.10;

// ─── Safety Bounds (Mamba Modulation NeurIPS 2025, HiSPA 2026) ─────
//
// Hard safety rails. These are NOT tunable — they are physics.
// Below 0.5×: state updates too slow, model effectively ignores input.
// Above 2.0×: catastrophic — Mamba Modulation showed perplexity 1648
//   at 2× uniform scaling. Layer-selective is safer, but 2× is ceiling.
// Lyapunov stability (ICLR 2025): perturbations <1.5% provably stable.
// Our operating range [0.7, 1.3] stays well within stable regime.
const GAIN_MIN: f64 = 0.5;
const GAIN_MAX: f64 = 2.0;

/// Compute delta modulation parameters from settled cognitive state.
///
/// Maps cognitive signals to compensatory delta scaling at mid-layers.
/// All modulation targets gain ≤ 1.0 (enhance retention, preserve context).
///
/// Principle (from data): Don't amplify what model already does.
/// Compensate what model structurally lacks (state retention capacity).
///
/// Profiles:
/// - Phasic (focused): gain 0.90 → retain relevant context
/// - Tonic (exploring): gain 0.80 → retain broad context
/// - Conservation (depleted): gain further reduced → preserve resources
/// - Arousal (salient): gain reduced → preserve emotional context
///
/// Returns DeltaModulation with gain_factor in [GAIN_MIN, GAIN_MAX].
pub fn compute_delta_modulation(
    state: &CognitiveState,
    num_layers: usize,
) -> DeltaModulation {
    // Step 0a: Gate-type conditioning (CR5-motivated defensive fix, 2026-04-14).
    //
    // If the thalamic gate classified this input as Routine, skip compensatory
    // modulation entirely — return passthrough (gain = 1.0). The gate says
    // "nothing special, cortex handles it"; delta modulation should respect
    // that classification over the arousal-driven signals downstream.
    //
    // **Validation status (honest framing, updated 2026-04-14 phase 4)**:
    //
    // The fix was originally motivated by the hs_arousal Technical +2.37%
    // perplexity regression (per-token HS path bypassing CognitiveSession).
    // Phase-4 diagnostic (`gate_classification_diagnose.rs`) confirmed that
    // in the production `CognitiveSession` pipeline, Technical text is
    // classified as **Novel** (not Routine), so this short-circuit never
    // fires in the default path. The +2.37% regression lives in the
    // per-token HS path which constructs `CognitiveState` directly without
    // routing through the thalamic gate — this fix does not reach that path.
    //
    // What this fix IS: principled defensive code aligned with P9 — "stay
    // passive when cortex is sufficient, like CognitiveGate alpha ≈ 0 for
    // routine content." If any future code path classifies content as
    // Routine, delta modulation will correctly abstain.
    //
    // What this fix IS NOT: empirical validation against the specific
    // failure case it was named for. Addressing the per-token HS regression
    // requires either routing per-token arousal through gate classification
    // (new plumbing) or tuning `CHURN_FLOOR` / `CHURN_CEILING` on
    // `hs_arousal.rs`. Both are separate work items.
    //
    // This short-circuit is a defensive choice (trust the gate classification
    // when it says the input is routine) rather than a validated mechanism.
    if matches!(state.gate_type, GateType::Routine) {
        return DeltaModulation {
            gain_factor: GAIN_NEUTRAL,
            target: compute_layer_targets(num_layers),
            source: DeltaModulationSource::Combined,
        };
    }

    // Step 0b: Build adaptive threshold context (P2: precision as universal currency).
    // All thresholds below are modulated by current cognitive state, not hardcoded.
    // Reuses build_threshold_context() from adaptive_thresholds (P3: single source).
    let threshold_ctx = build_threshold_context(
        state.sensory_pe,
        state.arousal,
        state.gate_confidence,
        state.pe_volatility,
        Some(state.valence),
        Some(state.body_budget),
    );

    // Compute adaptive thresholds for each signal.
    // P3: body budget threshold reused from Tầng 1 (same brain basis: Barrett 2017).
    let budget_threshold =
        get_adaptive_threshold(&threshold_body_budget_conservation(), &threshold_ctx);
    let volatility_threshold =
        get_adaptive_threshold(&threshold_delta_volatility(), &threshold_ctx);
    let arousal_emergency_threshold =
        get_adaptive_threshold(&threshold_delta_arousal_emergency(), &threshold_ctx);

    // Step 1: Base gain from mode (compensatory retention).
    // Phasic: focused retention (0.90). Tonic: broad retention (0.80).
    // Both < 1.0 — principle: compensate, don't amplify.
    let mut gain = match state.gain_mode {
        GainMode::Phasic => GAIN_PHASIC,
        GainMode::Tonic => GAIN_TONIC,
        GainMode::Neutral => GAIN_NEUTRAL,
    };

    // Step 2: Body budget conservation (Barrett 2017).
    // Low budget → reduce delta further → preserve state (conserve resources).
    if state.body_budget < budget_threshold {
        let factor =
            (budget_threshold - state.body_budget) / budget_threshold.max(0.01);
        gain -= factor * BUDGET_DELTA_REDUCTION;
    }

    // Step 3: PE volatility stabilization (Behrens 2007).
    // High volatility → reduce delta → hold onto known state (stabilize).
    // Corrected from boost: data shows gain < 1.0 helps in uncertain environments.
    if state.pe_volatility > volatility_threshold {
        let factor = (state.pe_volatility - volatility_threshold)
            / (1.0 - volatility_threshold).max(0.01);
        gain -= factor * VOLATILITY_DELTA_REDUCTION;
    }

    // Step 4: Arousal retention (LeDoux 1996).
    // High arousal → reduce delta → preserve emotional/salient context.
    // Corrected from boost: data shows phasic (gain > 1.0) hurts prediction.
    // Emotional context needs RETENTION across tokens, not faster processing.
    if state.arousal > arousal_emergency_threshold {
        let factor = (state.arousal - arousal_emergency_threshold)
            / (1.0 - arousal_emergency_threshold).max(0.01);
        gain -= factor * AROUSAL_DELTA_REDUCTION;
    }

    // Step 5: Clamp to safety bounds (CR4: clamping bounds are safety rails).
    gain = clamp(gain, GAIN_MIN, GAIN_MAX);

    // Step 6: Compute target layers from model depth.
    let target = compute_layer_targets(num_layers);

    DeltaModulation {
        gain_factor: gain,
        target,
        source: DeltaModulationSource::Combined,
    }
}

/// Compute which layers to target for delta modulation.
///
/// Maps the abstract depth range (40-60%) to concrete layer indices
/// for a model with the given number of layers.
///
/// Brain analog: neuromodulatory nuclei project to specific cortical
/// layers — not uniformly across all layers.
pub fn compute_layer_targets(num_layers: usize) -> LayerTarget {
    let start = (num_layers as f64 * TARGET_DEPTH_START).round() as usize;
    let end = ((num_layers as f64 * TARGET_DEPTH_END).round() as usize).min(num_layers.saturating_sub(1));
    LayerTarget {
        start_layer: start,
        end_layer: end,
        total_layers: num_layers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::belief::AffectValence;

    // ─── Gain mode tests ───

    #[test]
    fn phasic_gain_below_neutral() {
        let state = CognitiveState {
            gain_mode: GainMode::Phasic,
            ..CognitiveState::default()
        };
        let dm = compute_delta_modulation(&state, 64);
        assert!(
            dm.gain_factor < 1.0,
            "Phasic mode should reduce delta (compensatory retention, data: gain<1.0 helps)"
        );
    }

    #[test]
    fn tonic_gain_below_neutral() {
        let state = CognitiveState {
            gain_mode: GainMode::Tonic,
            ..CognitiveState::default()
        };
        let dm = compute_delta_modulation(&state, 64);
        assert!(
            dm.gain_factor < 1.0,
            "Tonic mode should decrease delta (broad exploration)"
        );
    }

    #[test]
    fn neutral_gain_is_unity() {
        let state = CognitiveState::default();
        let dm = compute_delta_modulation(&state, 64);
        assert_eq!(
            dm.gain_factor, 1.0,
            "Neutral mode with no signals should be pass-through"
        );
    }

    // ─── Body budget tests ───

    #[test]
    fn low_budget_reduces_gain() {
        let normal = CognitiveState::default();
        let depleted = CognitiveState {
            body_budget: 0.1,
            ..CognitiveState::default()
        };
        let dm_normal = compute_delta_modulation(&normal, 64);
        let dm_depleted = compute_delta_modulation(&depleted, 64);
        assert!(
            dm_depleted.gain_factor < dm_normal.gain_factor,
            "Low body budget should reduce delta (conservation)"
        );
    }

    #[test]
    fn full_budget_no_reduction() {
        let state = CognitiveState {
            body_budget: 1.0,
            ..CognitiveState::default()
        };
        let dm = compute_delta_modulation(&state, 64);
        // Full budget above threshold — no conservation effect.
        assert_eq!(dm.gain_factor, 1.0);
    }

    // ─── Volatility tests ───

    #[test]
    fn high_volatility_reduces_gain() {
        let stable = CognitiveState::default();
        let volatile = CognitiveState {
            pe_volatility: 0.9,
            ..CognitiveState::default()
        };
        let dm_stable = compute_delta_modulation(&stable, 64);
        let dm_volatile = compute_delta_modulation(&volatile, 64);
        assert!(
            dm_volatile.gain_factor < dm_stable.gain_factor,
            "High volatility should reduce delta (stabilize, preserve known state)"
        );
    }

    // ─── Arousal emergency tests ───

    #[test]
    fn emergency_arousal_reduces_gain() {
        let calm = CognitiveState::default();
        let threat = CognitiveState {
            arousal: 0.95,
            valence: AffectValence::Negative,
            ..CognitiveState::default()
        };
        let dm_calm = compute_delta_modulation(&calm, 64);
        let dm_threat = compute_delta_modulation(&threat, 64);
        assert!(
            dm_threat.gain_factor < dm_calm.gain_factor,
            "High arousal should reduce delta (preserve emotional context)"
        );
    }

    #[test]
    fn moderate_arousal_no_override() {
        let state = CognitiveState {
            arousal: 0.5, // Below emergency threshold (0.8).
            ..CognitiveState::default()
        };
        let dm = compute_delta_modulation(&state, 64);
        assert_eq!(
            dm.gain_factor, 1.0,
            "Moderate arousal should not trigger emergency override"
        );
    }

    // ─── Safety bounds tests ───

    #[test]
    fn gain_clamped_to_safe_range() {
        // Extreme phasic + high volatility + emergency arousal → should hit ceiling.
        let extreme = CognitiveState {
            gain_mode: GainMode::Phasic,
            pe_volatility: 1.0,
            arousal: 1.0,
            ..CognitiveState::default()
        };
        let dm = compute_delta_modulation(&extreme, 64);
        assert!(
            dm.gain_factor <= GAIN_MAX,
            "Gain must never exceed GAIN_MAX ({})",
            GAIN_MAX
        );
        assert!(
            dm.gain_factor >= GAIN_MIN,
            "Gain must never go below GAIN_MIN ({})",
            GAIN_MIN
        );
    }

    #[test]
    fn depleted_tonic_stays_above_floor() {
        // Extreme tonic + depleted budget → should hit floor.
        let extreme = CognitiveState {
            gain_mode: GainMode::Tonic,
            body_budget: 0.0,
            ..CognitiveState::default()
        };
        let dm = compute_delta_modulation(&extreme, 64);
        assert!(
            dm.gain_factor >= GAIN_MIN,
            "Even extreme depletion must stay above GAIN_MIN"
        );
    }

    // ─── Layer targeting tests ───

    #[test]
    fn layer_targeting_midrange_64() {
        let target = compute_layer_targets(64);
        assert_eq!(target.total_layers, 64);
        // 40% of 64 = 25.6 → 26, 60% of 64 = 38.4 → 38
        assert!(target.start_layer >= 25 && target.start_layer <= 26);
        assert!(target.end_layer >= 38 && target.end_layer <= 39);
    }

    #[test]
    fn layer_targeting_small_model() {
        let target = compute_layer_targets(12);
        // 40% of 12 = 4.8 → 5, 60% of 12 = 7.2 → 7
        assert!(target.start_layer >= 4 && target.start_layer <= 5);
        assert!(target.end_layer >= 7 && target.end_layer <= 8);
        assert_eq!(target.total_layers, 12);
    }

    #[test]
    fn layer_targeting_single_layer() {
        let target = compute_layer_targets(1);
        assert_eq!(target.total_layers, 1);
        // Should not overflow for minimal models.
        assert!(target.end_layer < 1);
    }

    // ─── Combined signal tests ───

    #[test]
    fn signals_compose_additively() {
        // Phasic + high volatility should reduce MORE than phasic alone.
        let phasic = CognitiveState {
            gain_mode: GainMode::Phasic,
            ..CognitiveState::default()
        };
        let phasic_volatile = CognitiveState {
            gain_mode: GainMode::Phasic,
            pe_volatility: 0.9,
            ..CognitiveState::default()
        };
        let dm_p = compute_delta_modulation(&phasic, 64);
        let dm_pv = compute_delta_modulation(&phasic_volatile, 64);
        assert!(
            dm_pv.gain_factor < dm_p.gain_factor,
            "Volatility should compound with phasic reduction (both push gain down)"
        );
    }

    #[test]
    fn tonic_plus_budget_depletion_compounds() {
        // Tonic + low budget should reduce more than tonic alone.
        let tonic = CognitiveState {
            gain_mode: GainMode::Tonic,
            ..CognitiveState::default()
        };
        let tonic_depleted = CognitiveState {
            gain_mode: GainMode::Tonic,
            body_budget: 0.1,
            ..CognitiveState::default()
        };
        let dm_t = compute_delta_modulation(&tonic, 64);
        let dm_td = compute_delta_modulation(&tonic_depleted, 64);
        assert!(
            dm_td.gain_factor < dm_t.gain_factor,
            "Budget depletion should compound with tonic reduction"
        );
    }

    #[test]
    fn source_is_combined() {
        let state = CognitiveState::default();
        let dm = compute_delta_modulation(&state, 64);
        assert_eq!(dm.source, DeltaModulationSource::Combined);
    }

    // ─── Gate-type conditioning tests (CR5 surgical fix 2026-04-14) ───

    #[test]
    fn routine_gate_forces_passthrough_over_phasic() {
        // Even with Phasic gain mode (which alone would reduce gain to 0.90),
        // Routine gate classification should short-circuit to gain = 1.0.
        let state = CognitiveState {
            gain_mode: GainMode::Phasic,
            gate_type: GateType::Routine,
            ..CognitiveState::default()
        };
        let dm = compute_delta_modulation(&state, 64);
        assert_eq!(
            dm.gain_factor, 1.0,
            "Routine gate must override Phasic mode to passthrough (P9: stay out when cortex sufficient)"
        );
    }

    #[test]
    fn routine_gate_ignores_high_arousal() {
        // High arousal would normally trigger emergency reduction,
        // but Routine classification says "nothing urgent" — trust the gate.
        let state = CognitiveState {
            gain_mode: GainMode::Phasic,
            arousal: 0.95,
            gate_type: GateType::Routine,
            ..CognitiveState::default()
        };
        let dm = compute_delta_modulation(&state, 64);
        assert_eq!(
            dm.gain_factor, 1.0,
            "Routine gate should pass through even when arousal scalar is high (churn ≠ salience)"
        );
    }

    #[test]
    fn novel_gate_preserves_compensatory_behavior() {
        // Regression check: Novel gate (the default) must still allow
        // the existing compensatory modulation to fire.
        let state = CognitiveState {
            gain_mode: GainMode::Phasic,
            gate_type: GateType::Novel,
            ..CognitiveState::default()
        };
        let dm = compute_delta_modulation(&state, 64);
        assert!(
            dm.gain_factor < 1.0,
            "Novel gate must NOT short-circuit — existing Phasic→0.90 reduction should still apply"
        );
    }

    #[test]
    fn urgent_gate_preserves_compensatory_behavior() {
        // Urgent inputs (e.g., error/crisis) explicitly do need modulation.
        let state = CognitiveState {
            gain_mode: GainMode::Phasic,
            arousal: 0.9,
            gate_type: GateType::Urgent,
            ..CognitiveState::default()
        };
        let dm = compute_delta_modulation(&state, 64);
        assert!(
            dm.gain_factor < 1.0,
            "Urgent gate must allow compensatory reduction — crisis content needs retention"
        );
    }

    #[test]
    fn routine_gate_still_targets_correct_layers() {
        // Short-circuit shouldn't break layer targeting — return still uses
        // the same spatial gating (HiSPA corridor) so downstream consumers
        // can rely on the target fields regardless of gain.
        let state = CognitiveState {
            gate_type: GateType::Routine,
            ..CognitiveState::default()
        };
        let dm = compute_delta_modulation(&state, 32);
        assert_eq!(dm.target.total_layers, 32);
        assert_eq!(dm.gain_factor, 1.0);
    }
}
