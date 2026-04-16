//! Adaptive Thresholds — universal precision gain control.
//!
//! Brain analog: Friston 2010 (precision modulates all processing),
//! Moran 2013 (gain control), Ernst & Banks 2002 (Bayesian optimal combination).
//!
//! Instead of hardcoded thresholds, each threshold adapts based on current
//! cognitive state (sensory PE, arousal, gate confidence, PE volatility).
//!
//! ## Gating (P10) — weights ARE the gating rules
//!
//! Each `ThresholdProfile` encodes priority/gating **implicitly** via its
//! weights. Signal dominance is defined by weight magnitude and sign:
//!
//! - `pe_weight > 0`: high PE raises the threshold (become conservative
//!   under surprise — match SC SIGNALS v2 "don't redirect when uncertain")
//! - `arousal_weight < 0`: high arousal LOWERS the threshold (amygdala
//!   low-road makes salient signals easier to ignite — LeDoux 1996)
//! - `confidence_weight > 0`: high confidence raises the threshold (trust
//!   the current precision, don't re-ignite spurious signals)
//! - `volatility_weight`: widens precision window when environment changes
//!   (Behrens 2007 — volatile world needs broader priors)
//!
//! This is the **single source of gating** in Nous: all other modules read
//! adaptive thresholds and inherit their priority structure. To add a new
//! gating rule, register a new threshold with the appropriate profile.
//!
//! Pure functions, <1ms, $0 LLM cost.

use crate::math::clamp;
use crate::types::belief::AffectValence;

// ── Shared Constants ───────────────────────────────────────────────────

/// Body budget level considered "healthy" — above this, no conservation pressure.
/// Below this, depletion drives both `threshold_body_budget_conservation` (when
/// to start conserving in delta modulation / sampling) AND threat-perception
/// modulation in `build_threshold_context` (deficit increases sensory PE bias).
///
/// **Calibrated 2026-04-14** (phase 6) — see
/// `threshold_body_budget_conservation` for the calibration analysis.
/// Two consumers MUST share this value to stay coherent: if conservation
/// triggers at budget < X, threat-perception bias should also key off X.
/// Previously the value was hardcoded inline at both sites (P3 violation
/// found in 2026-04-14 audit).
pub const BODY_BUDGET_HEALTHY_THRESHOLD: f64 = 0.70;

// ── Types ──────────────────────────────────────────────────────────────

/// Weights for how each state signal modulates a threshold.
#[derive(Debug, Clone)]
pub struct ThresholdProfile {
    /// Sensory PE weight (positive = conservative when surprised).
    pub pe_weight: f64,
    /// Arousal weight (negative = more sensitive under stress).
    pub arousal_weight: f64,
    /// Gate confidence weight (negative = lower threshold when confident).
    pub confidence_weight: f64,
    /// PE volatility weight (positive = widen threshold in volatile contexts).
    pub volatility_weight: f64,
}

/// Current cognitive state for threshold adaptation.
#[derive(Debug, Clone)]
pub struct ThresholdContext {
    /// 0-1, sensory prediction error.
    pub sensory_pe: f64,
    /// 0-1, emotional arousal.
    pub arousal: f64,
    /// 0-1, thalamic gate confidence.
    pub gate_confidence: f64,
    /// 0-1, PE volatility (Behrens 2007).
    pub pe_volatility: f64,
}

/// A registered threshold with base value and adaptation profile.
#[derive(Debug, Clone)]
pub struct ThresholdRegistration {
    pub id: String,
    pub base: f64,
    pub profile: ThresholdProfile,
    pub min: f64,
    pub max: f64,
}

/// Precision-weighted prediction error from a dimension.
#[derive(Debug, Clone)]
pub struct DimensionPE {
    /// 0-1, prediction error magnitude.
    pub pe: f64,
    /// Inverse variance — higher = more reliable.
    pub precision: f64,
}

// ── Pre-Registered Thresholds ──────────────────────────────────────────
//
// Each threshold is a ThresholdRegistration consumed via `get_adaptive_threshold(&t, &ctx)`.
// Removed orphans (phase 13, 2026-04-14):
//   - `threshold_ignition` — was wired by `integration.rs` Global Workspace ignition; module
//     removed, no remaining callers. Re-add with a live caller if Dehaene-style ignition gating
//     is re-introduced.
//   - `threshold_arousal_encoding` — flashbulb-memory encoding threshold, never wired to any
//     caller. Available in git history if memory-consolidation gating needs it.

// ── Intervention Thresholds (Tier 1) ──────────────────────────────────

/// Body budget conservation onset (Barrett 2017, Sterling 2012).
/// Below this threshold, system conserves (lower temp/top_p).
/// High PE and low confidence accelerate conservation.
///
/// **Calibrated 2026-04-14** (phase 6) to reach signal saturation (~0.5+ on
/// `signals.conservation`) under realistic sustained-stress + low-quality
/// work (simulating an agent struggling through hard content). Previous
/// base 0.30 kept `budget_factor` at 0 for typical session lengths —
/// conservation signal topped out ~0.30, never reaching the decision-grade
/// range implied by `docs/app-contract.md`. Raised base to 0.70 so
/// `budget_factor` starts contributing at 70% remaining budget — an "early
/// warning" zone rather than "near empty".
///
/// Relation to `COST_DEPLETION_RATE` (0.02 in `src/session.rs`): per-turn
/// cost depletion × avg cost ≈ 0.011/turn. With low reported quality
/// (negative RPE, no replenishment), body_budget drops below 0.70 after
/// ~30 turns of sustained high-cost reporting. At that point, sustained
/// (0.20) + resource_pressure (0.045) + budget_factor (rising) combine to
/// cross the 0.5 decision point documented in app-contract.md §1.1.
///
/// High-quality responses replenish body_budget via positive RPE in
/// consolidate — this is the correct allostatic behavior: successful work
/// doesn't trigger conservation even if it's costly. Conservation fires
/// when cost AND poor outcomes BOTH apply.
pub fn threshold_body_budget_conservation() -> ThresholdRegistration {
    ThresholdRegistration {
        id: "body-budget-conservation".into(),
        base: BODY_BUDGET_HEALTHY_THRESHOLD,
        profile: ThresholdProfile {
            pe_weight: 0.10,      // More surprise → conserve sooner
            arousal_weight: 0.05, // High arousal → conserve sooner
            confidence_weight: -0.10, // Low confidence → conserve sooner
            volatility_weight: 0.0,
        },
        min: 0.40,
        max: 0.85,
    }
}

/// Arousal intervention onset — frequency penalty under threat (LeDoux 1996).
/// High arousal + negative valence triggers tunnel vision prevention.
/// Base 0.40: calibrated to emotional.rs output range (stress ≈ 0.5-0.6).
/// Volatile environments lower the threshold (more sensitive to threat).
pub fn threshold_arousal_intervention() -> ThresholdRegistration {
    ThresholdRegistration {
        id: "arousal-intervention".into(),
        base: 0.40,
        profile: ThresholdProfile {
            pe_weight: -0.05,     // High PE → lower threshold (more vigilant)
            arousal_weight: 0.0,  // Arousal is the signal itself, no self-modulation
            confidence_weight: 0.10, // High confidence → higher threshold (less reactive)
            volatility_weight: -0.10, // Volatile env → lower threshold (more alert)
        },
        min: 0.25,
        max: 0.70,
    }
}

/// Resource pressure onset — presence penalty under allostatic load (Sterling 2012).
/// High pressure triggers concise output to conserve tokens.
/// High PE lowers the threshold (conserve under surprise).
pub fn threshold_resource_pressure() -> ThresholdRegistration {
    ThresholdRegistration {
        id: "resource-pressure".into(),
        base: 0.70,
        profile: ThresholdProfile {
            pe_weight: -0.05,     // High PE → conserve sooner
            arousal_weight: -0.05, // High arousal → conserve sooner
            confidence_weight: 0.10, // High confidence → tolerate more pressure
            volatility_weight: -0.05, // Volatile → conserve sooner
        },
        min: 0.50,
        max: 0.85,
    }
}

// ── Delta Modulation Thresholds (Tầng 2) ────────────────────────────

/// PE volatility threshold for delta exploration boost (Behrens 2007).
/// Above this, environment is unstable → increase delta → attend to new input.
/// High PE and volatile context lower the threshold (become exploration-sensitive sooner).
pub fn threshold_delta_volatility() -> ThresholdRegistration {
    ThresholdRegistration {
        id: "delta-volatility".into(),
        base: 0.60,
        profile: ThresholdProfile {
            pe_weight: -0.10,      // High PE → explore sooner (lower threshold)
            arousal_weight: -0.05, // Aroused → explore sooner
            confidence_weight: 0.10, // Confident → tolerate more volatility before exploring
            volatility_weight: 0.0,  // Volatility IS the signal, no self-modulation
        },
        min: 0.35,
        max: 0.80,
    }
}

/// Arousal emergency threshold for delta override (LeDoux 1996 fast pathway).
/// Above this, threat detected → override to phasic delta (attend to threat).
/// Base 0.55: calibrated to emotional.rs arousal output range (stress ≈ 0.5-0.6).
/// Volatile environment lowers threshold (more alert).
pub fn threshold_delta_arousal_emergency() -> ThresholdRegistration {
    ThresholdRegistration {
        id: "delta-arousal-emergency".into(),
        base: 0.55,
        profile: ThresholdProfile {
            pe_weight: -0.05,      // High PE → lower threshold (more vigilant)
            arousal_weight: 0.0,   // Arousal IS the signal, no self-modulation
            confidence_weight: 0.05, // Confident → slightly higher threshold
            volatility_weight: -0.10, // Volatile env → lower threshold (more alert)
        },
        min: 0.40,
        max: 0.80,
    }
}

// ── Core Functions ─────────────────────────────────────────────────────

/// Compute adaptive threshold from base + cognitive state modulation.
///
/// Formula (centered around neutral = 0.5):
/// ```text
/// modulation = (PE - 0.5) × peWeight
///            + arousal × arousalWeight
///            + (confidence - 0.5) × confidenceWeight
///            + (volatility - 0.3) × volatilityWeight
/// effective = clamp(base × (1 + modulation), min, max)
/// ```
pub fn get_adaptive_threshold(threshold: &ThresholdRegistration, ctx: &ThresholdContext) -> f64 {
    let pe = safe_num(ctx.sensory_pe, 0.5);
    let arousal = safe_num(ctx.arousal, 0.0);
    let conf = safe_num(ctx.gate_confidence, 0.5);
    let vol = safe_num(ctx.pe_volatility, 0.3);

    let modulation = (pe - 0.5) * threshold.profile.pe_weight
        + arousal * threshold.profile.arousal_weight
        + (conf - 0.5) * threshold.profile.confidence_weight
        + (vol - 0.3) * threshold.profile.volatility_weight;

    clamp(
        threshold.base * (1.0 + modulation),
        threshold.min,
        threshold.max,
    )
}

/// Build threshold context with optional affect modulation (Barrett 2025, Seth 2024).
///
/// Negative valence → tunnel vision (arousal amplified).
/// Positive valence → broadened attention (arousal dampened).
/// Low body budget → increased threat perception (PE bias).
pub fn build_threshold_context(
    sensory_pe: f64,
    arousal: f64,
    gate_confidence: f64,
    pe_volatility: f64,
    valence: Option<AffectValence>,
    body_budget: Option<f64>,
) -> ThresholdContext {
    let mut adj_arousal = arousal;
    let mut adj_pe = sensory_pe;
    let mut adj_conf = gate_confidence;

    // Affect modulation (AD-180)
    if let Some(v) = valence {
        match v {
            AffectValence::Negative => adj_arousal *= 1.15, // Tunnel vision
            AffectValence::Positive => adj_arousal *= 0.9,  // Broadened attention
            AffectValence::Neutral => {}
        }
    }

    // Body budget modulation (allostatic). P3: same threshold as
    // `threshold_body_budget_conservation` so that "depleted enough to
    // shift threat perception" and "depleted enough to trigger conservation"
    // co-fire from the same trigger point. Don't drift these.
    if let Some(budget) = body_budget {
        if budget < BODY_BUDGET_HEALTHY_THRESHOLD {
            let deficit = BODY_BUDGET_HEALTHY_THRESHOLD - budget;
            adj_pe += deficit * 0.15;   // Perceive more threat
            adj_conf -= deficit * 0.1;  // Less model trust
        }
    }

    ThresholdContext {
        sensory_pe: clamp(adj_pe, 0.0, 1.0),
        arousal: clamp(adj_arousal, 0.0, 1.0),
        gate_confidence: clamp(adj_conf, 0.0, 1.0),
        pe_volatility: clamp(pe_volatility, 0.0, 1.0),
    }
}

/// Bayesian-optimal combination of multiple dimension PEs (Ernst & Banks 2002).
///
/// Unified PE = precision-weighted average: Σ(precision_i × PE_i) / Σ(precision_i).
/// Falls back to simple mean if total precision = 0.
pub fn compute_unified_pe(dimensions: &[DimensionPE]) -> f64 {
    if dimensions.is_empty() {
        return 0.5; // Neutral
    }

    let total_precision: f64 = dimensions.iter().map(|d| d.precision.max(0.0)).sum();

    if total_precision == 0.0 {
        // Fallback: simple mean
        return dimensions.iter().map(|d| d.pe).sum::<f64>() / dimensions.len() as f64;
    }

    let weighted: f64 = dimensions
        .iter()
        .map(|d| d.precision.max(0.0) * d.pe)
        .sum();

    weighted / total_precision
}

/// Safe number with fallback for NaN/undefined.
fn safe_num(value: f64, fallback: f64) -> f64 {
    if value.is_nan() || value.is_infinite() {
        fallback
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neutral_ctx() -> ThresholdContext {
        ThresholdContext {
            sensory_pe: 0.5,
            arousal: 0.0,
            gate_confidence: 0.5,
            pe_volatility: 0.3,
        }
    }

    // These 4 tests cover the core `get_adaptive_threshold` formula. Prior to phase 13 they
    // used `threshold_ignition` as a fixture; switched to `threshold_resource_pressure` (a live
    // threshold with negative `arousal_weight`, matching the original semantics) after
    // `threshold_ignition` was removed as orphan. Semantics preserved: arousal lowers, confidence
    // raises, neutral context returns base, output clamps to [min, max].

    #[test]
    fn neutral_context_returns_base() {
        let t = threshold_resource_pressure();
        let effective = get_adaptive_threshold(&t, &neutral_ctx());
        assert!((effective - t.base).abs() < 0.05);
    }

    #[test]
    fn high_arousal_lowers_threshold() {
        let t = threshold_resource_pressure();
        let base = get_adaptive_threshold(&t, &neutral_ctx());
        let aroused = get_adaptive_threshold(
            &t,
            &ThresholdContext {
                arousal: 0.9,
                ..neutral_ctx()
            },
        );
        assert!(aroused < base); // Arousal weight is negative → lowers threshold
    }

    #[test]
    fn high_confidence_raises_threshold() {
        let t = threshold_resource_pressure();
        let base = get_adaptive_threshold(&t, &neutral_ctx());
        let confident = get_adaptive_threshold(
            &t,
            &ThresholdContext {
                gate_confidence: 0.9,
                ..neutral_ctx()
            },
        );
        assert!(confident > base); // Confidence weight is positive → raises threshold
    }

    #[test]
    fn threshold_clamped_to_bounds() {
        let t = threshold_resource_pressure();
        let extreme = get_adaptive_threshold(
            &t,
            &ThresholdContext {
                sensory_pe: 1.0,
                arousal: 1.0,
                gate_confidence: 0.0,
                pe_volatility: 1.0,
            },
        );
        assert!(extreme >= t.min);
        assert!(extreme <= t.max);
    }

    #[test]
    fn unified_pe_precision_weighted() {
        let dims = vec![
            DimensionPE { pe: 0.8, precision: 0.9 }, // High PE, high precision
            DimensionPE { pe: 0.2, precision: 0.1 }, // Low PE, low precision
        ];
        let unified = compute_unified_pe(&dims);
        // Should be close to 0.8 (high-precision dimension dominates)
        assert!(unified > 0.6);
    }

    #[test]
    fn unified_pe_equal_precision() {
        let dims = vec![
            DimensionPE { pe: 0.8, precision: 1.0 },
            DimensionPE { pe: 0.2, precision: 1.0 },
        ];
        let unified = compute_unified_pe(&dims);
        // Equal precision → simple mean
        assert!((unified - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn unified_pe_empty() {
        assert!((compute_unified_pe(&[]) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn unified_pe_zero_precision_fallback() {
        let dims = vec![
            DimensionPE { pe: 0.3, precision: 0.0 },
            DimensionPE { pe: 0.7, precision: 0.0 },
        ];
        let unified = compute_unified_pe(&dims);
        assert!((unified - 0.5).abs() < f64::EPSILON); // Fallback to mean
    }

    #[test]
    fn negative_valence_amplifies_arousal() {
        let ctx = build_threshold_context(0.5, 0.5, 0.5, 0.3, Some(AffectValence::Negative), None);
        assert!(ctx.arousal > 0.5); // Amplified
    }

    #[test]
    fn positive_valence_dampens_arousal() {
        let ctx = build_threshold_context(0.5, 0.5, 0.5, 0.3, Some(AffectValence::Positive), None);
        assert!(ctx.arousal < 0.5); // Dampened
    }

    #[test]
    fn low_body_budget_increases_pe() {
        let ctx = build_threshold_context(0.5, 0.3, 0.5, 0.3, None, Some(0.3));
        assert!(ctx.sensory_pe > 0.5); // More threat perception
        assert!(ctx.gate_confidence < 0.5); // Less model trust
    }

    #[test]
    fn safe_num_handles_nan() {
        assert_eq!(safe_num(f64::NAN, 0.5), 0.5);
        assert_eq!(safe_num(0.7, 0.5), 0.7);
    }
}
