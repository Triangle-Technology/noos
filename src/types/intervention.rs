//! Intervention types — data structures for cognitive model intervention.
//!
//! These types bridge cognitive state (from convergence loop) to model control
//! signals (sampling parameters, logit biases). Part of the intervention
//! architecture that moves Noos from text I/O wrapping to model-internal
//! modulation.
//!
//! See `docs/intervention.md` for the full paradigm.

use serde::{Deserialize, Serialize};

use super::belief::AffectValence;
use super::gate::GateType;
use super::world::GainMode;

/// Model capability levels — what intervention depth is available.
///
/// Higher tiers subsume lower tiers. A model supporting ActivationAccess
/// also supports LogitAccess and TextOnly.
///
/// Brain analog: different levels of access to neural substrate.
/// TextOnly = observing behavior. LogitAccess = reading EEG.
/// ActivationAccess = single-neuron recording. MultiModel = whole-brain imaging.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum InterventionDepth {
    /// Text I/O only (closed API models: Claude, GPT). SC-equivalent.
    #[default]
    TextOnly,
    /// Logit access before sampling (open models via API or local).
    LogitAccess,
    /// Hidden state read/write between transformer layers (local inference).
    ActivationAccess,
    /// Cognitive gate layer embedded in model architecture (Tầng 3).
    /// Brain analog: thalamic relay + LC-NE gain integrated into cortical circuit.
    ArchitectureIntegration,
    /// Full multi-model orchestration with shared representation space (local ensemble).
    MultiModel,
}

/// Unified cognitive state snapshot — assembled from convergence loop output.
///
/// This is the bridge between Noos's cognitive algorithms and model intervention.
/// After the convergence loop settles, CognitiveState captures the final state
/// and passes it to intervention hooks that modulate model generation.
///
/// Brain analog: the neuromodulatory "context" that colors all downstream
/// processing — a summary of arousal, confidence, resource state, and
/// environmental stability that modulates how the brain generates responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveState {
    // --- From AffectState (belief.affect) ---
    /// Amygdala activation level (LeDoux 1996). 0 = calm, 1 = maximum arousal.
    pub arousal: f64,
    /// Affective polarity (Barrett 2017). Pre-colors all downstream processing.
    pub valence: AffectValence,
    /// Confidence in current perceptual reading. 0 = uncertain, 1 = certain.
    pub certainty: f64,
    /// Sustained arousal modulator (Yerkes-Dodson). Penalizes prolonged high arousal.
    pub sustained_arousal: f64,

    // --- From LocusCoeruleus ---
    /// LC-NE gain mode (Aston-Jones & Cohen 2005).
    /// Phasic = focused exploitation, Tonic = broad exploration.
    pub gain_mode: GainMode,

    // --- From WorldModel ---
    /// Allostatic body budget (Barrett 2017, Sterling 2012).
    /// 1.0 = full resources, depletes on stress/PE, replenishes on success.
    pub body_budget: f64,
    /// Feature-specific prediction error (Friston 2010). 0 = expected, 1 = maximally surprising.
    pub sensory_pe: f64,
    /// Allostatic load from resource allocator. 0 = no pressure, 1 = overloaded.
    pub resource_pressure: f64,
    /// Environmental stability estimate (Behrens 2007). 0 = stable, 1 = volatile.
    pub pe_volatility: f64,

    // --- From GateResult ---
    /// Thalamic gate classification confidence. 0 = uncertain, 1 = certain.
    pub gate_confidence: f64,
    /// Thalamic gate classification (Routine / Novel / Urgent).
    /// Added 2026-04-14 to support gate-type-conditional delta modulation
    /// (CR5 surgical fix for hs_arousal Technical regression).
    pub gate_type: GateType,
}

impl Default for CognitiveState {
    /// Safe default: calm, neutral, full resources, no prediction error.
    /// Used as fail-open fallback (P5).
    fn default() -> Self {
        Self {
            arousal: 0.0,
            valence: AffectValence::Neutral,
            certainty: 0.5,
            sustained_arousal: 0.0,
            gain_mode: GainMode::Neutral,
            body_budget: 1.0,
            sensory_pe: 0.0,
            resource_pressure: 0.0,
            pe_volatility: 0.0,
            gate_confidence: 0.5,
            // Default matches GateType::default() (Novel). Routine would short-circuit
            // all delta modulation per the gate-conditioning rule in
            // `compute_delta_modulation`; existing tests that assert phasic/tonic
            // modes produce gain < 1.0 rely on a non-Routine default.
            gate_type: GateType::Novel,
        }
    }
}

/// Sampling parameter overrides derived from cognitive state.
///
/// These parameters modulate model generation at the sampling level.
/// Brain analog: neuromodulators (DA, NE, ACh, 5-HT) don't carry specific
/// information — they change gain, threshold, and sensitivity of neural circuits.
/// SamplingOverride is Noos's neuromodulatory output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingOverride {
    /// Generation temperature. Lower = focused/deterministic, higher = creative/exploratory.
    /// Modulated by gain_mode (Aston-Jones 2005) and body_budget (Barrett 2017).
    pub temperature: f64,
    /// Nucleus sampling threshold. Lower = fewer token candidates, higher = more diverse.
    /// Modulated by gain_mode and pe_volatility (Behrens 2007).
    pub top_p: f64,
    /// Frequency penalty. Reduces probability of recently-used tokens.
    /// Elevated under high arousal + negative valence (LeDoux 1996: tunnel vision → avoid repetition).
    pub frequency_penalty: f64,
    /// Presence penalty. Reduces probability of any token that appeared in output.
    /// Elevated under high resource pressure (Sterling 2012: conserve under allostatic load).
    pub presence_penalty: f64,
    /// Token-level logit biases. Applied before sampling.
    /// Currently empty in Tier 1 Phase 1 (requires tokenizer integration).
    pub logit_biases: Vec<LogitBias>,
}

impl Default for SamplingOverride {
    /// Neutral sampling — no intervention. Equivalent to standard model behavior.
    fn default() -> Self {
        Self {
            temperature: 0.5,
            top_p: 0.9,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_biases: Vec::new(),
        }
    }
}

/// A single logit bias entry — modifies probability of a specific token.
///
/// Brain analog: synaptic facilitation/depression at the output layer.
/// Positive bias = pre-activate token (prime), negative = suppress token (inhibit).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitBias {
    /// Token ID in the model's vocabulary.
    pub token_id: u32,
    /// Bias magnitude. Positive = amplify, negative = suppress.
    /// Typical range: [-5.0, 5.0].
    pub bias: f64,
    /// Which cognitive module generated this bias (for debugging/tracing).
    pub source: String,
}

// ═══════════════════════════════════════════════════════════════════
// Tầng 2: Delta Modulation Types — SSM state injection
// ═══════════════════════════════════════════════════════════════════

/// Target layer range for SSM delta modulation.
///
/// Brain analog: neuromodulatory projections target specific cortical layers.
/// Mid-layers (40-60% depth) are the critical transport corridor where
/// information is actively processed, not just embedded or retrieved.
///
/// Key paper: HiSPA 2026 (blocks 28-37/64 = highest correlation with
/// downstream behavior, r = -0.9707 at block 29).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerTarget {
    /// First layer to modulate (inclusive, 0-indexed).
    pub start_layer: usize,
    /// Last layer to modulate (inclusive, 0-indexed).
    pub end_layer: usize,
    /// Total layers in the model (for bounds checking and percentage calculations).
    pub total_layers: usize,
}

impl Default for LayerTarget {
    /// Default: targets 40-60% depth of a 64-layer model (Falcon Mamba).
    /// HiSPA 2026: critical transport corridor.
    fn default() -> Self {
        Self {
            start_layer: 25,
            end_layer: 38,
            total_layers: 64,
        }
    }
}

impl LayerTarget {
    /// Check if a given layer index should be modulated.
    pub fn contains(&self, layer: usize) -> bool {
        layer >= self.start_layer && layer <= self.end_layer
    }

    /// Number of layers being modulated.
    pub fn modulated_count(&self) -> usize {
        if self.end_layer >= self.start_layer {
            self.end_layer - self.start_layer + 1
        } else {
            0
        }
    }
}

/// What cognitive signal drove this delta modulation.
///
/// Tracking source enables debugging and future per-source tuning.
/// Brain analog: different neuromodulators (NE, DA, ACh, 5-HT) have
/// distinct projection patterns and functional effects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeltaModulationSource {
    /// LC-NE gain mode is the primary driver (Aston-Jones 2005).
    GainMode,
    /// Body budget conservation (Barrett 2017, Sterling 2012).
    BodyBudget,
    /// PE volatility exploration (Behrens 2007).
    Volatility,
    /// Arousal emergency override (LeDoux 1996).
    Arousal,
    /// Multiple signals combined (typical case).
    Combined,
}

/// Delta modulation parameters for SSM state injection.
///
/// Brain analog: noradrenergic gain control (Aston-Jones & Cohen 2005).
/// Delta (dt) in Mamba's selective scan controls state update speed:
/// - Higher delta → state updates more from current input (reactive, focused)
/// - Lower delta → state preserves more history (inertial, exploratory)
///
/// Hidden Attention (ACL 2025) proves: delta modulation = modulating
/// implicit attention's temporal decay.
///
/// Safety bounds (Mamba Modulation, NeurIPS 2025):
/// - Uniform 2-3× scaling across all layers = catastrophic
/// - Layer-selective 0.7-1.3× = safe (Lyapunov-stable regime)
/// - ~25% state norm change at mid-layers = catastrophic forgetting (HiSPA 2026)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaModulation {
    /// Multiplicative scale factor for delta (dt) parameter.
    /// - `1.0` = no modulation (pass-through).
    /// - `> 1.0` = attend more to current input (phasic, focused).
    /// - `< 1.0` = preserve more history (tonic, exploratory).
    ///
    /// Clamped to [0.5, 2.0] by compute_delta_modulation().
    pub gain_factor: f64,
    /// Which layers to target for modulation.
    pub target: LayerTarget,
    /// What cognitive signal drove this modulation (for debugging/tracing).
    pub source: DeltaModulationSource,
}

impl Default for DeltaModulation {
    /// No modulation — pure pass-through. Used as fail-open fallback (P5).
    fn default() -> Self {
        Self {
            gain_factor: 1.0,
            target: LayerTarget::default(),
            source: DeltaModulationSource::GainMode,
        }
    }
}

/// Result from a cognitive forward pass — logits plus modulation metadata.
///
/// Extends the basic logits output with information about what intervention
/// was applied, enabling downstream inspection and predictive coding
/// comparisons between modulated and unmodulated distributions.
///
/// Brain analog: cortical output includes not just the signal but also
/// a corollary discharge (efference copy) of the modulation applied
/// (Crapse & Sommer 2008).
#[derive(Debug, Clone)]
pub struct ForwardResult {
    /// Logit distribution over vocabulary (same as LocalModel::forward output).
    pub logits: Vec<f32>,
    /// Whether delta modulation was actually applied during this forward pass.
    pub modulation_applied: bool,
    /// Which layers were modulated (empty if modulation_applied is false).
    pub modulated_layers: Vec<usize>,
    /// The gain factor that was applied (1.0 if no modulation).
    pub applied_gain_factor: f64,
    /// Tầng 3: learned delta gain from CognitiveGate (None if no gate present).
    /// Maps gate's sigmoid output to [0.5, 2.0] gain range.
    pub gate_delta_gain: Option<f64>,
    /// Tầng 3: gate blend factor (None if no gate, 0.0 = passthrough, 1.0 = full modulation).
    /// Initialized near 0.05 via W_gate bias = -3.0.
    pub gate_alpha: Option<f64>,
    /// Hidden state statistics for cognitive signal extraction (Tầng 2 enhancement).
    /// None when model does not support ActivationAccess or hs not captured.
    /// When present, `cognition::hs_arousal::arousal_from_hs()` derives arousal from these.
    pub hs_stats: Option<HiddenStateStats>,
}

impl ForwardResult {
    /// Create a result from a standard (unmodulated) forward pass.
    /// Used when CognitiveModel falls back to LocalModel behavior.
    pub fn from_logits(logits: Vec<f32>) -> Self {
        Self {
            logits,
            modulation_applied: false,
            modulated_layers: Vec::new(),
            applied_gain_factor: 1.0,
            gate_delta_gain: None,
            gate_alpha: None,
            hs_stats: None,
        }
    }
}

/// Statistics derived from SSM hidden state for cognitive signal extraction.
///
/// Bridge between inference layer (candle tensors) and cognition layer (f64 scalars).
/// Replaces regex-based arousal with model-derived signal when SSM state is available.
///
/// Brain analog: cortical state summary read by LC for aggregate unsigned PE
/// detection (Grella 2024, Jordan & Keller eLife 2023). LC doesn't classify
/// content — it measures HOW MUCH cortical state is changing.
///
/// Key metric: state_churn maps to LC unsigned prediction error.
/// High churn = state being overwritten = needs compensatory retention.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct HiddenStateStats {
    /// State churn rate: `||hs[t] - hs[t-1]|| / ||hs[t]||`.
    /// How fast SSM state is changing across target layers (40-60% depth).
    /// High churn = new information overwriting old = state decay = needs compensation.
    /// Range: typically [0, 2]. 0 = static, >1 = changing faster than own magnitude.
    pub state_churn: f64,
    /// State magnitude: L2 norm of mean-pooled hs across target layers.
    /// Measures working memory capacity utilization. Scale is model-dependent.
    pub state_magnitude: f64,
    /// Whether this reading is valid. False on first token (no previous hs to compare).
    /// When false, consumers should fall back to regex arousal.
    pub valid: bool,
}

/// Application-facing allostatic signals.
///
/// Organized around application DECISIONS, not internal module state.
/// Brain analog: hypothalamic output signals that drive organism behavior
/// (Barrett 2017 — allostasis as core brain function, Neuron 2025).
///
/// Three decision axes:
/// 1. Conservation: should the system invest more or conserve?
/// 2. Salience: how urgent/novel is this input?
/// 3. Confidence: how reliable is the system's own assessment?
///
/// Plus: learned strategy recommendation, gain mode, emotional valence.
///
/// Consumed by application layer for: adaptive prompting, model routing,
/// resource allocation, strategy selection, UX feedback.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CognitiveSignals {
    /// Should the system invest more or conserve resources? [0, 1]
    /// 0 = fully engaged (plenty of budget, low pressure).
    /// 1 = conservation mode (depleted budget, high pressure, sustained stress).
    /// Application: limit context window, use cheaper model, simplify response.
    /// Brain analog: hypothalamic conservation signal under allostatic overload.
    pub conservation: f64,

    /// How urgent/salient is this input? [0, 1]
    /// 0 = routine (low arousal, familiar gate).
    /// 1 = urgent (high arousal, urgent gate, or novel with high PE).
    /// Application: prioritize processing, interrupt other tasks, alert user.
    /// Brain analog: LC-NE phasic burst on salient input (Aston-Jones 2005).
    pub salience: f64,

    /// How confident is the system in its own state assessment? [0, 1]
    /// Low confidence = system may be wrong about everything above.
    /// Application: add verification step, request user confirmation, hedge.
    /// Brain analog: interoceptive certainty (Seth 2024).
    pub confidence: f64,

    /// Recommended response strategy (from cross-session learning).
    /// None = no prior data for this topic cluster.
    pub strategy: Option<crate::types::world::ResponseStrategy>,

    /// Gain mode — exploit (phasic) or explore (tonic)?
    /// Application: phasic = focused single-path. tonic = broad multi-path.
    pub gain_mode: GainMode,

    /// Emotional valence of the input.
    /// Application: adjust tone, enable empathy mode, flag sensitive content.
    pub valence: AffectValence,

    /// EMA-smoothed recent response quality [0, 1].
    /// 0.5 = uninitialized or mixed. High = consistently good responses.
    /// Low = consistently poor — application may want to intervene.
    /// Brain analog: striatal value estimate (Schultz 1997) — running
    /// estimate of outcome quality from action-outcome pairs.
    pub recent_quality: f64,

    /// Most recent reward prediction error [-1, +1].
    /// Positive = response exceeded expectation. Negative = disappointed.
    /// Brain analog: dopaminergic RPE (Schultz 1997). Drives learning rate
    /// in volatile environments (Behrens 2007).
    ///
    /// Application: use with recent_quality to detect failure patterns.
    /// E.g. recent_quality < 0.4 AND rpe < 0 → pattern is degrading, intervene.
    pub rpe: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cognitive_state_default_is_calm_neutral() {
        let state = CognitiveState::default();
        assert_eq!(state.arousal, 0.0);
        assert_eq!(state.valence, AffectValence::Neutral);
        assert_eq!(state.body_budget, 1.0);
        assert_eq!(state.gain_mode, GainMode::Neutral);
        assert_eq!(state.sensory_pe, 0.0);
        assert_eq!(state.resource_pressure, 0.0);
    }

    #[test]
    fn sampling_override_default_is_neutral() {
        let s = SamplingOverride::default();
        assert_eq!(s.temperature, 0.5);
        assert_eq!(s.top_p, 0.9);
        assert_eq!(s.frequency_penalty, 0.0);
        assert_eq!(s.presence_penalty, 0.0);
        assert!(s.logit_biases.is_empty());
    }

    #[test]
    fn intervention_depth_ordering() {
        assert!(InterventionDepth::TextOnly < InterventionDepth::LogitAccess);
        assert!(InterventionDepth::LogitAccess < InterventionDepth::ActivationAccess);
        assert!(InterventionDepth::ActivationAccess < InterventionDepth::ArchitectureIntegration);
        assert!(InterventionDepth::ArchitectureIntegration < InterventionDepth::MultiModel);
    }

    #[test]
    fn intervention_depth_default_is_text_only() {
        assert_eq!(InterventionDepth::default(), InterventionDepth::TextOnly);
    }

    #[test]
    fn logit_bias_construction() {
        let bias = LogitBias {
            token_id: 42,
            bias: -2.5,
            source: "emotional".into(),
        };
        assert_eq!(bias.token_id, 42);
        assert_eq!(bias.bias, -2.5);
        assert_eq!(bias.source, "emotional");
    }

    // ─── Tầng 2 type tests ───

    #[test]
    fn delta_modulation_default_is_no_modulation() {
        let dm = DeltaModulation::default();
        assert_eq!(dm.gain_factor, 1.0, "Default should be pass-through");
        assert_eq!(dm.source, DeltaModulationSource::GainMode);
    }

    #[test]
    fn layer_target_default_targets_midrange() {
        let target = LayerTarget::default();
        assert_eq!(target.total_layers, 64, "Default is Falcon Mamba 64 layers");
        assert!(target.start_layer >= 25, "Should target ~40% depth");
        assert!(target.end_layer <= 40, "Should target ~60% depth");
    }

    #[test]
    fn layer_target_contains() {
        let target = LayerTarget {
            start_layer: 10,
            end_layer: 20,
            total_layers: 64,
        };
        assert!(!target.contains(9));
        assert!(target.contains(10));
        assert!(target.contains(15));
        assert!(target.contains(20));
        assert!(!target.contains(21));
    }

    #[test]
    fn layer_target_modulated_count() {
        let target = LayerTarget {
            start_layer: 10,
            end_layer: 20,
            total_layers: 64,
        };
        assert_eq!(target.modulated_count(), 11);
    }

    #[test]
    fn forward_result_from_logits_is_unmodulated() {
        let result = ForwardResult::from_logits(vec![1.0, 2.0, 3.0]);
        assert!(!result.modulation_applied);
        assert!(result.modulated_layers.is_empty());
        assert_eq!(result.applied_gain_factor, 1.0);
        assert_eq!(result.logits.len(), 3);
        assert!(result.gate_delta_gain.is_none());
        assert!(result.gate_alpha.is_none());
        assert!(result.hs_stats.is_none());
    }

    #[test]
    fn hidden_state_stats_default_is_invalid() {
        let stats = HiddenStateStats::default();
        assert_eq!(stats.state_churn, 0.0);
        assert_eq!(stats.state_magnitude, 0.0);
        assert!(!stats.valid, "Default stats should be invalid (no previous hs)");
    }

    #[test]
    fn hidden_state_stats_serde_round_trip() {
        let stats = HiddenStateStats {
            state_churn: 0.73,
            state_magnitude: 2.45,
            valid: true,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let restored: HiddenStateStats = serde_json::from_str(&json).unwrap();
        assert!((restored.state_churn - stats.state_churn).abs() < 1e-10);
        assert!((restored.state_magnitude - stats.state_magnitude).abs() < 1e-10);
        assert_eq!(restored.valid, stats.valid);
    }
}
