//! Allostatic Signals — application-facing cognitive state summary.
//!
//! Translates internal cognitive state into actionable application decisions.
//! The bridge between Noos's non-cortical processing and the application layer.
//!
//! ## Brain analog
//!
//! Hypothalamic output drives organism behavior through 3 axes:
//! - Conservation: energy allocation (Sterling 2012 allostatic regulation)
//! - Salience: threat/novelty detection (LeDoux 1996 amygdala, Aston-Jones 2005 LC)
//! - Confidence: interoceptive certainty (Seth 2024 control-oriented interoception)
//!
//! Barrett 2025 (Neuron): allostasis is the brain's primary function. All other
//! cognition serves body-budget regulation. CognitiveSignals implements this:
//! Noos's cognitive processing serves the SYSTEM's adaptive functioning.
//!
//! ## Gating (P10)
//!
//! - **Fires when**: CognitiveSession.process_message() completes convergence.
//! - **Does NOT fire**: between turns (idle cycles don't produce signals).
//! - **Suppresses**: nothing — this is an output interface, not a competing signal.
//! - **Suppressed by**: nothing — always computed after convergence settles.
//! - **Inactive when**: never — signals are always produced (fail-open with neutral defaults).
//!
//! Pure function, O(1), $0 LLM cost.

use crate::cognition::adaptive_thresholds::{
    build_threshold_context, get_adaptive_threshold, threshold_body_budget_conservation,
};
use crate::math::clamp;
use crate::types::gate::GateType;
use crate::types::intervention::CognitiveSignals;
use crate::types::world::{GainMode, WorldModel};

// ── Constants ──────────────────────────────────────────────────────────

/// Weight of resource_pressure contribution to conservation signal.
/// 0.3 = moderate — pressure alone doesn't drive full conservation.
/// Pressure comes from resource_allocator (softmax competition, Yerkes-Dodson).
const PRESSURE_CONSERVATION_WEIGHT: f64 = 0.3;

/// Salience contribution from URGENT gate classification.
/// 0.3 = strong — urgent gate is the amygdala fast-pathway signal (LeDoux 1996).
const GATE_URGENT_SALIENCE: f64 = 0.3;

/// Salience contribution from NOVEL gate classification.
/// 0.15 = moderate — novelty draws attention but isn't threat-level.
const GATE_NOVEL_SALIENCE: f64 = 0.15;

/// Weight of sensory PE contribution to salience.
/// 0.2 = moderate — PE adds to salience but doesn't dominate over arousal/gate.
const PE_SALIENCE_WEIGHT: f64 = 0.2;

/// How much PE volatility reduces confidence.
/// 0.5 = substantial — in volatile environments, trust your assessments less.
/// Brain analog: high environmental volatility → increase learning rate,
/// decrease reliance on current model (Behrens 2007).
const VOLATILITY_CONFIDENCE_REDUCTION: f64 = 0.5;

/// Weight of "lost sustained arousal capacity" on the conservation signal.
/// Kept at 0.2 so sustained-arousal penalty is a modulator, not a driver
/// — body_budget and resource_pressure are the primary conservation
/// drivers; this is the slow-timescale fatigue tax on top (Yerkes-Dodson
/// inverted-U on prolonged stress).
const SUSTAINED_AROUSAL_CONSERVATION_WEIGHT: f64 = 0.2;

/// Minimum divisor when normalizing budget_factor. Prevents a near-zero
/// adaptive threshold from dividing by 0 (which would produce ±inf and
/// cascade through the clamp). 0.01 is well below any realistic
/// conservation threshold (typical adaptive threshold ∈ [0.2, 0.5]).
const BUDGET_FACTOR_MIN_DIVISOR: f64 = 0.01;

// ── Core Function ──────────────────────────────────────────────────────

/// Compute application-facing allostatic signals from settled cognitive state.
///
/// Pure function. Takes converged WorldModel + LC gain mode, produces
/// CognitiveSignals organized around application decisions.
///
/// Called after convergence loop settles, alongside `build_cognitive_state()`.
pub fn compute_signals(model: &WorldModel, gain_mode: GainMode) -> CognitiveSignals {
    // ── Conservation axis ──
    // P3: uses adaptive_thresholds (same threshold source as intervention.rs
    // and delta_modulation.rs) to determine conservation level. All three
    // consumers share the same predicate: "body_budget < adaptive threshold."
    let threshold_ctx = build_threshold_context(
        model.sensory_pe,
        model.belief.affect.arousal,
        model.gate.confidence,
        model.pe_volatility,
        Some(model.belief.affect.valence),
        Some(model.body_budget),
    );
    let budget_threshold =
        get_adaptive_threshold(&threshold_body_budget_conservation(), &threshold_ctx);

    // Budget factor: how far below the adaptive threshold.
    // 0 = budget above threshold (healthy). 1 = budget at zero (critical).
    let budget_factor = if model.body_budget < budget_threshold {
        (budget_threshold - model.body_budget) / budget_threshold.max(BUDGET_FACTOR_MIN_DIVISOR)
    } else {
        0.0
    };
    // Sustained arousal penalty: prolonged stress depletes capacity.
    let sustained_penalty =
        (1.0 - model.belief.affect.sustained).max(0.0) * SUSTAINED_AROUSAL_CONSERVATION_WEIGHT;
    let conservation = clamp(
        budget_factor + model.resource_pressure * PRESSURE_CONSERVATION_WEIGHT + sustained_penalty,
        0.0,
        1.0,
    );

    // ── Salience axis ──
    // How urgent/novel is this input? High = prioritize.
    // Arousal is the primary signal (amygdala activation).
    // Gate type adds a classification-based component.
    // Sensory PE adds a surprise component.
    let gate_salience = match model.gate.gate {
        GateType::Urgent => GATE_URGENT_SALIENCE,
        GateType::Novel => GATE_NOVEL_SALIENCE,
        GateType::Routine => 0.0,
    };
    let salience = clamp(
        model.belief.affect.arousal + gate_salience + model.sensory_pe * PE_SALIENCE_WEIGHT,
        0.0,
        1.0,
    );

    // ── Confidence axis ──
    // How reliable is the system's assessment? Low = hedge.
    // Base: gate classification confidence (how sure is the thalamic classifier).
    // Reduced by: PE volatility (unstable environment → lower trust in assessment).
    let confidence = clamp(
        model.gate.confidence * (1.0 - model.pe_volatility * VOLATILITY_CONFIDENCE_REDUCTION),
        0.0,
        1.0,
    );

    CognitiveSignals {
        conservation,
        salience,
        confidence,
        strategy: model.recommended_strategy,
        gain_mode,
        valence: model.belief.affect.valence,
        // Direct pass-through from WorldModel — no composition.
        // Application decides how to combine these for failure detection.
        recent_quality: model.last_response_prediction,
        rpe: model.response_rpe,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::belief::AffectValence;

    fn default_model() -> WorldModel {
        WorldModel::new("test".into())
    }

    #[test]
    fn default_produces_neutral_signals() {
        let signals = compute_signals(&default_model(), GainMode::Neutral);
        // Default: full budget, no pressure, no arousal → low conservation, low salience.
        assert!(signals.conservation < 0.2, "Default should have low conservation, got {}", signals.conservation);
        assert!(signals.salience < 0.2, "Default should have low salience, got {}", signals.salience);
        assert_eq!(signals.gain_mode, GainMode::Neutral);
        assert_eq!(signals.valence, AffectValence::Neutral);
        assert!(signals.strategy.is_none());
    }

    #[test]
    fn depleted_budget_high_conservation() {
        let mut model = default_model();
        model.body_budget = 0.1; // Nearly depleted.
        let signals = compute_signals(&model, GainMode::Neutral);
        // Budget 0.1 < adaptive threshold (base 0.70 since phase-6 calibration,
        // modulated by state) → budget_factor ≈ 0.85 in this default ctx.
        // Conservation should be meaningfully above neutral (0.0).
        assert!(
            signals.conservation > 0.5,
            "Depleted budget should produce high conservation, got {}",
            signals.conservation
        );
    }

    #[test]
    fn high_pressure_increases_conservation() {
        let mut model = default_model();
        model.resource_pressure = 0.9;
        let signals = compute_signals(&model, GainMode::Neutral);
        assert!(
            signals.conservation > 0.2,
            "High pressure should increase conservation, got {}",
            signals.conservation
        );
    }

    #[test]
    fn urgent_gate_high_salience() {
        let mut model = default_model();
        model.gate.gate = GateType::Urgent;
        model.belief.affect.arousal = 0.7;
        let signals = compute_signals(&model, GainMode::Phasic);
        assert!(
            signals.salience > 0.8,
            "Urgent gate + high arousal should produce high salience, got {}",
            signals.salience
        );
    }

    #[test]
    fn routine_low_salience() {
        let mut model = default_model();
        model.gate.gate = GateType::Routine;
        model.belief.affect.arousal = 0.1;
        let signals = compute_signals(&model, GainMode::Neutral);
        assert!(
            signals.salience < 0.2,
            "Routine gate + low arousal should produce low salience, got {}",
            signals.salience
        );
    }

    #[test]
    fn high_volatility_reduces_confidence() {
        let mut model = default_model();
        model.gate.confidence = 0.8;
        model.pe_volatility = 0.9;
        let signals = compute_signals(&model, GainMode::Neutral);
        // High volatility should reduce confidence below what gate_confidence alone gives.
        assert!(
            signals.confidence < model.gate.confidence,
            "Volatility should reduce confidence below gate_confidence, got {}",
            signals.confidence
        );
        // But not to zero — still some confidence from gate.
        assert!(signals.confidence > 0.0);
    }

    #[test]
    fn recent_quality_and_rpe_passed_through() {
        let mut model = default_model();
        model.last_response_prediction = 0.35;
        model.response_rpe = -0.2;
        let signals = compute_signals(&model, GainMode::Neutral);
        assert_eq!(signals.recent_quality, 0.35);
        assert_eq!(signals.rpe, -0.2);
    }

    #[test]
    fn conservation_clamped_to_unit() {
        let mut model = default_model();
        model.body_budget = 0.0;
        model.resource_pressure = 1.0;
        let signals = compute_signals(&model, GainMode::Neutral);
        assert!(
            signals.conservation <= 1.0,
            "Conservation must be clamped to [0, 1]"
        );
    }

    #[test]
    fn signals_serde_round_trip() {
        // Applications need to log/telemetry CognitiveSignals — verify full
        // round-trip through JSON preserves every field (including Option and
        // nested enums).
        use crate::types::world::ResponseStrategy;

        let signals = CognitiveSignals {
            conservation: 0.42,
            salience: 0.78,
            confidence: 0.65,
            strategy: Some(ResponseStrategy::StepByStep),
            gain_mode: GainMode::Phasic,
            valence: AffectValence::Negative,
            recent_quality: 0.55,
            rpe: -0.12,
        };

        let json = serde_json::to_string(&signals)
            .expect("CognitiveSignals should serialize to JSON");
        let restored: CognitiveSignals = serde_json::from_str(&json)
            .expect("CognitiveSignals should deserialize from JSON");

        assert!((restored.conservation - signals.conservation).abs() < 1e-10);
        assert!((restored.salience - signals.salience).abs() < 1e-10);
        assert!((restored.confidence - signals.confidence).abs() < 1e-10);
        assert_eq!(restored.strategy, signals.strategy);
        assert_eq!(restored.gain_mode, signals.gain_mode);
        assert_eq!(restored.valence, signals.valence);
        assert!((restored.recent_quality - signals.recent_quality).abs() < 1e-10);
        assert!((restored.rpe - signals.rpe).abs() < 1e-10);
    }

    #[test]
    fn signals_with_no_strategy_serializes() {
        // First-session case: strategy is None, must still round-trip cleanly.
        let signals = CognitiveSignals {
            conservation: 0.0,
            salience: 0.0,
            confidence: 0.5,
            strategy: None,
            gain_mode: GainMode::Neutral,
            valence: AffectValence::Neutral,
            recent_quality: 0.5,
            rpe: 0.0,
        };

        let json = serde_json::to_string(&signals).unwrap();
        let restored: CognitiveSignals = serde_json::from_str(&json).unwrap();
        assert!(restored.strategy.is_none());
    }
}
