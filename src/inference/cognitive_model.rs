//! Cognitive Model — model that integrates cognitive state during inference.
//!
//! Brain analog: cortical tissue modulated by neuromodulators. The cortex
//! doesn't just receive input — its processing is shaped by NE (arousal/gain),
//! DA (reward prediction), ACh (sensory precision), 5-HT (mood/inhibition).
//!
//! CognitiveModel extends LocalModel with cognitive awareness. Instead of
//! opaque `forward() → logits`, it accepts DeltaModulation and applies it
//! during the forward pass — the model itself receives the modulation
//! signal, rather than being wrapped by an external cognition layer.
//!
//! Key papers:
//! - Aston-Jones & Cohen 2005 (LC-NE gain modulation of cortical processing)
//! - Yamins & DiCarlo 2016 (deep nets as cortical models)
//! - Hidden Attention of Mamba Models (ACL 2025: delta = implicit attention)

use crate::errors::NousResult;
use crate::inference::model::LocalModel;
use crate::types::intervention::{DeltaModulation, ForwardResult, InterventionDepth};

/// A model that integrates cognitive state during its forward pass.
///
/// Extends LocalModel with the ability to receive and apply delta modulation
/// during inference. The model decides HOW to apply the modulation based on
/// its architecture (SSM delta scaling, attention steering, etc.).
///
/// Brain analog: cortex receiving neuromodulatory input. The NE projection
/// from locus coeruleus doesn't tell the cortex WHAT to compute — it
/// changes HOW the cortex computes (gain, threshold, temporal dynamics).
///
/// Implementations must:
/// 1. Report their intervention depth via `intervention_depth()`
/// 2. Apply delta modulation in `forward_cognitive()` at targeted layers
/// 3. Report which layers were actually modulated in `ForwardResult`
///
/// Graceful degradation (P5): if a model only supports LogitAccess,
/// `forward_cognitive()` should still work — just ignore the delta
/// modulation and return unmodulated logits via `ForwardResult::from_logits()`.
pub trait CognitiveModel: LocalModel {
    /// What depth of intervention this model supports.
    ///
    /// - `ActivationAccess`: can read/write hidden states, apply delta modulation
    /// - `LogitAccess`: can provide logits but not modulate internals
    /// - `TextOnly`: API-only, cognitive model trait should not be used
    fn intervention_depth(&self) -> InterventionDepth;

    /// Forward pass with cognitive delta modulation.
    ///
    /// The model applies `delta_modulation.gain_factor` to the SSM delta (dt)
    /// parameter at layers within `delta_modulation.target`.
    ///
    /// For SSM (Mamba): `dt_modulated = dt * gain_factor` before softplus.
    /// For Transformer: could map to attention temperature scaling.
    /// For unknown arch: ignore modulation, return logits only (P5 fail-open).
    ///
    /// Returns ForwardResult with logits + modulation metadata (efference copy).
    fn forward_cognitive(
        &mut self,
        tokens: &[u32],
        position: usize,
        delta_modulation: &DeltaModulation,
    ) -> NousResult<ForwardResult>;

    /// Number of layers in the model (for layer targeting computation).
    fn num_layers(&self) -> usize;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::inference::model::tests::MockModel;

    /// Mock cognitive model for testing — simulates delta modulation.
    ///
    /// Uses MockModel internally for logit generation, adds modulation
    /// tracking for test assertions.
    pub(crate) struct MockCognitiveModel {
        inner: MockModel,
        layers: usize,
        depth: InterventionDepth,
    }

    impl MockCognitiveModel {
        pub fn new(vocab_size: usize, layers: usize) -> Self {
            Self {
                inner: MockModel::new(vocab_size),
                layers,
                depth: InterventionDepth::ActivationAccess,
            }
        }
    }

    impl LocalModel for MockCognitiveModel {
        fn forward(&mut self, tokens: &[u32], position: usize) -> NousResult<Vec<f32>> {
            self.inner.forward(tokens, position)
        }

        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }

        fn reset_cache(&mut self) {
            self.inner.reset_cache();
        }
    }

    impl CognitiveModel for MockCognitiveModel {
        fn intervention_depth(&self) -> InterventionDepth {
            self.depth
        }

        fn forward_cognitive(
            &mut self,
            tokens: &[u32],
            position: usize,
            delta_modulation: &DeltaModulation,
        ) -> NousResult<ForwardResult> {
            // Get base logits from inner model.
            let logits = self.inner.forward(tokens, position)?;

            // Simulate modulation: track which layers were targeted.
            let modulated_layers: Vec<usize> = (0..self.layers)
                .filter(|&l| delta_modulation.target.contains(l))
                .collect();

            let modulation_applied =
                !modulated_layers.is_empty() && delta_modulation.gain_factor != 1.0;

            Ok(ForwardResult {
                logits,
                modulation_applied,
                modulated_layers,
                applied_gain_factor: delta_modulation.gain_factor,
                gate_delta_gain: None,
                gate_alpha: None,
                hs_stats: None,
            })
        }

        fn num_layers(&self) -> usize {
            self.layers
        }
    }

    #[test]
    fn mock_cognitive_model_reports_depth() {
        let model = MockCognitiveModel::new(100, 64);
        assert_eq!(model.intervention_depth(), InterventionDepth::ActivationAccess);
    }

    #[test]
    fn mock_cognitive_model_applies_modulation() {
        let mut model = MockCognitiveModel::new(100, 64);
        let dm = DeltaModulation {
            gain_factor: 1.2,
            ..DeltaModulation::default()
        };
        let result = model.forward_cognitive(&[1], 0, &dm).unwrap();
        assert!(result.modulation_applied);
        assert!(!result.modulated_layers.is_empty());
        assert_eq!(result.applied_gain_factor, 1.2);
    }

    #[test]
    fn mock_cognitive_model_no_modulation_at_unity() {
        let mut model = MockCognitiveModel::new(100, 64);
        let dm = DeltaModulation::default(); // gain_factor = 1.0
        let result = model.forward_cognitive(&[1], 0, &dm).unwrap();
        assert!(
            !result.modulation_applied,
            "Unity gain should not count as modulation"
        );
    }

    #[test]
    fn forward_result_tracks_correct_layers() {
        let mut model = MockCognitiveModel::new(100, 64);
        let dm = DeltaModulation {
            gain_factor: 1.2,
            target: crate::types::intervention::LayerTarget {
                start_layer: 25,
                end_layer: 38,
                total_layers: 64,
            },
            ..DeltaModulation::default()
        };
        let result = model.forward_cognitive(&[1], 0, &dm).unwrap();
        assert_eq!(result.modulated_layers.len(), 14); // 25..=38 = 14 layers
        assert_eq!(*result.modulated_layers.first().unwrap(), 25);
        assert_eq!(*result.modulated_layers.last().unwrap(), 38);
    }

    #[test]
    fn cognitive_model_still_works_as_local_model() {
        let mut model = MockCognitiveModel::new(100, 64);
        // LocalModel interface still works.
        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.len(), 100);
    }
}
