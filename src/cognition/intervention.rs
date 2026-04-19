//! Cognitive Intervention — translates brain state to model control signals.
//!
//! Brain analog: neuromodulatory system. Dopamine, norepinephrine, serotonin,
//! and acetylcholine don't carry specific information — they modulate HOW
//! neural circuits process. This module is the digital equivalent: cognitive
//! state modulates how the model generates output.
//!
//! Key papers:
//! - Aston-Jones & Cohen 2005 (LC-NE gain control → temperature/top_p)
//! - Barrett 2017 (allostasis → conservation under low body budget)
//! - Behrens 2007 (volatility → exploration)
//! - LeDoux 1996 (amygdala fast pathway → frequency penalty under threat)
//! - Sterling 2012 (allostatic regulation → presence penalty under load)
//!
//! Pure functions, <1ms, $0 LLM cost.
//!
//! ## Gating (P10)
//!
//! Tầng 1 sampling adapter: converts settled
//! [`CognitiveState`] into a [`SamplingOverride`]. Not a signal
//! producer — a consumer that reads adaptive thresholds (the single
//! source of gating per §P10) and emits sampling knobs.
//!
//! - **Fires when**: [`compute_sampling_override`] is called once per
//!   turn after the convergence loop settles.
//! - **Inactive when**: cognitive signals are at neutral defaults —
//!   per-threshold guards (e.g. `arousal >= threshold_arousal_intervention`)
//!   individually return `None` and every field of the returned
//!   [`SamplingOverride`] stays `None`.
//! - **Suppresses**: nothing — sampling overrides compose, they do not
//!   replace peer signals.
//! - **Suppressed by**: [`adaptive_thresholds`](super::adaptive_thresholds)
//!   refusing to cross a per-signal intervention threshold. Gating is
//!   delegated upstream (P10 single source of gating rule).

use crate::cognition::adaptive_thresholds::{
    get_adaptive_threshold, threshold_arousal_intervention, threshold_body_budget_conservation,
    threshold_resource_pressure, ThresholdContext,
};
use crate::math::vector::clamp;
use crate::types::belief::AffectValence;
use crate::types::intervention::{CognitiveState, SamplingOverride};
use crate::types::world::{GainMode, WorldModel};

// ─── Temperature constants (Aston-Jones & Cohen 2005) ───
// LC-NE gain mode determines base generation temperature.
// Phasic = high gain, narrow tuning → low temperature (focused).
// Tonic = low gain, broad tuning → high temperature (exploratory).
//
// NOTE (P3): resource_allocator.rs also maps gain_mode → temperature (0.15/0.30/0.50)
// but for SOFTMAX competition between context layers (internal budget allocation).
// These constants are for LLM GENERATION sampling — a wider range (0.3–0.9) is
// appropriate because generation temperature has a larger perceptual effect than
// softmax temperature on fixed-size budget pools.

/// Phasic mode temperature — focused exploitation (Aston-Jones 2005).
/// Low temperature concentrates probability on high-confidence tokens.
const TEMP_PHASIC: f64 = 0.3;

/// Tonic mode temperature — broad exploration (Aston-Jones 2005).
/// High temperature spreads probability across more token candidates.
const TEMP_TONIC: f64 = 0.9;

/// Neutral mode temperature — balanced generation.
/// Midpoint of phasic-tonic range. No single paper; follows the
/// Aston-Jones 2005 continuum where moderate NE = moderate gain.
const TEMP_NEUTRAL: f64 = 0.5;

// ─── Top-p constants (nucleus sampling, Aston-Jones & Cohen 2005) ───
// Parallels temperature: phasic narrows candidate pool, tonic widens it.

/// Phasic top-p — fewer candidates, higher precision.
const TOP_P_PHASIC: f64 = 0.8;

/// Tonic top-p — more candidates, higher diversity.
const TOP_P_TONIC: f64 = 0.95;

/// Neutral top-p — standard nucleus sampling.
/// 0.9 is the industry default (Holtzman 2020, The Curious Case of Neural Text Degeneration).
const TOP_P_NEUTRAL: f64 = 0.9;

// ─── Body budget conservation (Barrett 2017, Sterling 2012) ───
// Low body budget = organism under stress = conserve resources.
// Reduces exploration (lower temp/top_p) to avoid costly errors.
// Threshold is adaptive via threshold_body_budget_conservation() (theories.md P2).

/// Maximum temperature reduction under full conservation.
/// At body_budget=0, temperature drops by this amount from base.
const CONSERVATION_TEMP_REDUCTION: f64 = 0.15;

/// Maximum top_p reduction under full conservation.
const CONSERVATION_TOP_P_REDUCTION: f64 = 0.1;

// ─── PE volatility exploration (Behrens 2007) ───
// High environmental volatility = predictions unreliable = explore more.
// Increases temperature to broaden search when world is changing.

/// Volatility threshold above which exploration boost activates.
/// 0.6 = environment is changing faster than model can track.
const VOLATILITY_HIGH_THRESHOLD: f64 = 0.6;

/// Maximum temperature boost under maximum volatility.
const VOLATILITY_TEMP_BOOST: f64 = 0.2;

// ─── Arousal frequency penalty (LeDoux 1996) ───
// High arousal + negative valence = threat detected = tunnel vision.
// Frequency penalty prevents repetitive/stuck generation under stress.
// Threshold is adaptive via threshold_arousal_intervention() (theories.md P2).

/// Maximum frequency penalty under maximum threat arousal.
/// Prevents token repetition during high-stress generation.
const AROUSAL_FREQUENCY_PENALTY: f64 = 0.3;

// ─── Resource pressure presence penalty (Sterling 2012) ───
// High allostatic load = system overloaded = conserve output tokens.
// Presence penalty encourages concise, non-redundant generation.
// Threshold is adaptive via threshold_resource_pressure() (theories.md P2).

/// Maximum presence penalty under maximum resource pressure.
const PRESSURE_PRESENCE_PENALTY: f64 = 0.2;

// ─── Clamp bounds for sampling parameters ───

/// Minimum temperature — never fully deterministic (preserves stochasticity).
/// Stochastic resonance: small noise improves signal detection (Benzi 1981).
/// Without minimum noise, model output becomes degenerate/repetitive.
const TEMP_MIN: f64 = 0.1;

/// Maximum temperature — never fully random.
/// Beyond 1.0, softmax becomes nearly uniform → incoherent generation.
const TEMP_MAX: f64 = 1.0;

/// Minimum top_p — always consider some candidates.
/// Below 0.5, greedy-like sampling dominates → loss of nuance.
/// Floor ensures at least moderate token diversity (Holtzman 2020).
const TOP_P_MIN: f64 = 0.5;

/// Maximum top_p — full nucleus.
/// 1.0 = consider all tokens in vocabulary.
const TOP_P_MAX: f64 = 1.0;

/// Compute sampling parameter overrides from cognitive state.
///
/// This is the core Tier 1 intervention function. It translates the settled
/// cognitive state (output of convergence loop) into model sampling parameters.
///
/// The function composes multiple neuromodulatory signals:
/// 1. Gain mode → base temperature + top_p (Aston-Jones 2005)
/// 2. Body budget → conservation reduction (Barrett 2017) — adaptive threshold
/// 3. PE volatility → exploration boost (Behrens 2007)
/// 4. Arousal + valence → frequency penalty (LeDoux 1996) — adaptive threshold
/// 5. Resource pressure → presence penalty (Sterling 2012) — adaptive threshold
///
/// Thresholds 2/4/5 are adaptive via `get_adaptive_threshold()` (theories.md P2:
/// precision as universal currency — no hardcoded thresholds).
///
/// All outputs clamped to safe ranges (CR4: clamping bounds are safety rails).
pub fn compute_sampling_override(state: &CognitiveState) -> SamplingOverride {
    // Build threshold context from cognitive state (theories.md P2).
    // This makes all thresholds adaptive to current cognitive conditions.
    let threshold_ctx = ThresholdContext {
        sensory_pe: state.sensory_pe,
        arousal: state.arousal,
        gate_confidence: state.gate_confidence,
        pe_volatility: state.pe_volatility,
    };

    // Compute adaptive thresholds (not hardcoded).
    let budget_threshold =
        get_adaptive_threshold(&threshold_body_budget_conservation(), &threshold_ctx);
    let arousal_threshold =
        get_adaptive_threshold(&threshold_arousal_intervention(), &threshold_ctx);
    let pressure_threshold =
        get_adaptive_threshold(&threshold_resource_pressure(), &threshold_ctx);

    // Step 1: Base temperature and top_p from gain mode (Aston-Jones & Cohen 2005).
    // Phasic = focused exploitation, Tonic = broad exploration.
    let (base_temp, base_top_p) = match state.gain_mode {
        GainMode::Phasic => (TEMP_PHASIC, TOP_P_PHASIC),
        GainMode::Tonic => (TEMP_TONIC, TOP_P_TONIC),
        GainMode::Neutral => (TEMP_NEUTRAL, TOP_P_NEUTRAL),
    };

    let mut temperature = base_temp;
    let mut top_p = base_top_p;
    let mut frequency_penalty = 0.0;
    let mut presence_penalty = 0.0;

    // Step 2: Body budget conservation (Barrett 2017, Sterling 2012).
    // Low body budget → organism under stress → conserve by reducing exploration.
    // Threshold is adaptive: high PE/low confidence → conserve sooner.
    if state.body_budget < budget_threshold {
        let conservation = (budget_threshold - state.body_budget) / budget_threshold.max(0.01);
        temperature -= conservation * CONSERVATION_TEMP_REDUCTION;
        top_p -= conservation * CONSERVATION_TOP_P_REDUCTION;
    }

    // Step 3: PE volatility exploration (Behrens 2007).
    // High volatility → predictions unreliable → explore more to update model.
    // Boost scales linearly from 0 (at threshold) to 1 (at volatility=1).
    if state.pe_volatility > VOLATILITY_HIGH_THRESHOLD {
        let explore_drive =
            (state.pe_volatility - VOLATILITY_HIGH_THRESHOLD) / (1.0 - VOLATILITY_HIGH_THRESHOLD);
        temperature += explore_drive * VOLATILITY_TEMP_BOOST;
    }

    // Step 4: Arousal + negative valence → tunnel vision (LeDoux 1996).
    // High arousal with negative valence = threat detected.
    // Threshold is adaptive: volatile env → lower threshold (more alert).
    if state.arousal > arousal_threshold && state.valence == AffectValence::Negative {
        let intensity =
            (state.arousal - arousal_threshold) / (1.0 - arousal_threshold).max(0.01);
        frequency_penalty += intensity * AROUSAL_FREQUENCY_PENALTY;
    }

    // Step 5: Resource pressure → conserve tokens (Sterling 2012).
    // High allostatic load → encourage concise output.
    // Threshold is adaptive: high PE/arousal → conserve sooner.
    if state.resource_pressure > pressure_threshold {
        let pressure_drive = (state.resource_pressure - pressure_threshold)
            / (1.0 - pressure_threshold).max(0.01);
        presence_penalty += pressure_drive * PRESSURE_PRESENCE_PENALTY;
    }

    // Step 6: Clamp all values (CR4: clamping bounds are safety rails).
    SamplingOverride {
        temperature: clamp(temperature, TEMP_MIN, TEMP_MAX),
        top_p: clamp(top_p, TOP_P_MIN, TOP_P_MAX),
        frequency_penalty: clamp(frequency_penalty, 0.0, 1.0),
        presence_penalty: clamp(presence_penalty, 0.0, 1.0),
        logit_biases: Vec::new(), // Tier 1 Phase 1: no token-level biases yet
    }
}

/// Build CognitiveState snapshot from WorldModel + gain mode.
///
/// Assembles the unified cognitive state from the convergence loop's settled
/// WorldModel and the locus coeruleus gain mode. Called after convergence completes,
/// before passing state to InferenceProvider.
///
/// This is the bridge function: convergence loop output → intervention input.
pub fn build_cognitive_state(model: &WorldModel, gain_mode: GainMode) -> CognitiveState {
    CognitiveState {
        arousal: model.belief.affect.arousal,
        valence: model.belief.affect.valence,
        certainty: model.belief.affect.certainty,
        sustained_arousal: model.belief.affect.sustained,
        gain_mode,
        body_budget: model.body_budget,
        sensory_pe: model.sensory_pe,
        resource_pressure: model.resource_pressure,
        pe_volatility: model.pe_volatility,
        gate_confidence: model.gate.confidence,
        gate_type: model.gate.gate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ─── compute_sampling_override tests ───

    #[test]
    fn phasic_gain_lowers_temperature() {
        let state = CognitiveState {
            gain_mode: GainMode::Phasic,
            ..CognitiveState::default()
        };
        let s = compute_sampling_override(&state);
        assert!(s.temperature <= 0.3, "Phasic should produce low temperature");
        assert!(s.top_p <= 0.8, "Phasic should produce low top_p");
    }

    #[test]
    fn tonic_gain_raises_temperature() {
        let state = CognitiveState {
            gain_mode: GainMode::Tonic,
            ..CognitiveState::default()
        };
        let s = compute_sampling_override(&state);
        assert!(s.temperature >= 0.7, "Tonic should produce high temperature");
        assert!(s.top_p >= 0.9, "Tonic should produce high top_p");
    }

    #[test]
    fn neutral_state_produces_balanced_params() {
        let state = CognitiveState::default();
        let s = compute_sampling_override(&state);
        assert_relative_eq!(s.temperature, TEMP_NEUTRAL, epsilon = 0.01);
        assert_relative_eq!(s.top_p, TOP_P_NEUTRAL, epsilon = 0.01);
        assert_eq!(s.frequency_penalty, 0.0);
        assert_eq!(s.presence_penalty, 0.0);
    }

    #[test]
    fn high_arousal_negative_raises_frequency_penalty() {
        let state = CognitiveState {
            arousal: 0.8,
            valence: AffectValence::Negative,
            ..CognitiveState::default()
        };
        let s = compute_sampling_override(&state);
        assert!(
            s.frequency_penalty > 0.0,
            "High arousal + negative valence should raise frequency penalty"
        );
    }

    #[test]
    fn high_arousal_positive_no_frequency_penalty() {
        let state = CognitiveState {
            arousal: 0.8,
            valence: AffectValence::Positive,
            ..CognitiveState::default()
        };
        let s = compute_sampling_override(&state);
        assert_eq!(
            s.frequency_penalty, 0.0,
            "High arousal + positive valence should NOT raise frequency penalty"
        );
    }

    #[test]
    fn low_body_budget_conserves() {
        let state = CognitiveState {
            body_budget: 0.1,
            ..CognitiveState::default()
        };
        let s = compute_sampling_override(&state);
        let neutral = compute_sampling_override(&CognitiveState::default());
        assert!(
            s.temperature < neutral.temperature,
            "Low body budget should lower temperature"
        );
        assert!(
            s.top_p < neutral.top_p,
            "Low body budget should lower top_p"
        );
    }

    #[test]
    fn high_volatility_explores() {
        let state = CognitiveState {
            pe_volatility: 0.9,
            ..CognitiveState::default()
        };
        let s = compute_sampling_override(&state);
        let neutral = compute_sampling_override(&CognitiveState::default());
        assert!(
            s.temperature > neutral.temperature,
            "High PE volatility should raise temperature for exploration"
        );
    }

    #[test]
    fn high_resource_pressure_raises_presence_penalty() {
        let state = CognitiveState {
            resource_pressure: 0.9,
            ..CognitiveState::default()
        };
        let s = compute_sampling_override(&state);
        assert!(
            s.presence_penalty > 0.0,
            "High resource pressure should raise presence penalty"
        );
    }

    #[test]
    fn temperature_clamped_to_safe_range() {
        // Phasic + low body budget → temperature pushed very low
        let state = CognitiveState {
            gain_mode: GainMode::Phasic,
            body_budget: 0.0,
            ..CognitiveState::default()
        };
        let s = compute_sampling_override(&state);
        assert!(
            s.temperature >= TEMP_MIN,
            "Temperature must not go below minimum"
        );

        // Tonic + high volatility → temperature pushed very high
        let state = CognitiveState {
            gain_mode: GainMode::Tonic,
            pe_volatility: 1.0,
            ..CognitiveState::default()
        };
        let s = compute_sampling_override(&state);
        assert!(
            s.temperature <= TEMP_MAX,
            "Temperature must not exceed maximum"
        );
    }

    #[test]
    fn multiple_signals_compose() {
        // High arousal + negative + high pressure → both penalties active
        let state = CognitiveState {
            arousal: 0.9,
            valence: AffectValence::Negative,
            resource_pressure: 0.9,
            ..CognitiveState::default()
        };
        let s = compute_sampling_override(&state);
        assert!(s.frequency_penalty > 0.0);
        assert!(s.presence_penalty > 0.0);
    }

    // ─── build_cognitive_state tests ───

    #[test]
    fn build_cognitive_state_from_world_model() {
        let model = WorldModel::new("test".into());
        let state = build_cognitive_state(&model, GainMode::Phasic);

        assert_eq!(state.arousal, model.belief.affect.arousal);
        assert_eq!(state.valence, model.belief.affect.valence);
        assert_eq!(state.body_budget, model.body_budget);
        assert_eq!(state.sensory_pe, model.sensory_pe);
        assert_eq!(state.gain_mode, GainMode::Phasic);
        assert_eq!(state.gate_confidence, model.gate.confidence);
    }
}
