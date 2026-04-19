//! Resource Allocator — application context-budget allocation via softmax.
//!
//! ## Scope clarification (2026-04-11 audit)
//!
//! This is **application-level infrastructure**, not neural computation.
//! The "layers" being allocated (memory/graph/artifact/recap) are retrieval
//! subsystems of a host application, not brain regions. The softmax
//! competition mechanism IS precise (Desimone & Duncan 1995 biased
//! competition is a real neural mechanism), but what it allocates to is
//! not. Keep the mechanism, drop the metaphorical claim that this is
//! "neural".
//!
//! Where this IS compensatory: the model has no native mechanism to
//! decide how much char-budget to spend on each retrieval subsystem.
//! This module provides that decision using cognitive state (gain mode,
//! arousal, resource pressure) as modulation signals — which is a valid
//! non-cortical role (resource allocation is a genuine hypothalamic
//! function, Sterling 2012 allostasis).
//!
//! Brain citations below are for the softmax-competition mathematics
//! (Desimone & Duncan, Eldar) and for the arousal-modulation curves
//! (Yerkes-Dodson, Lavie). They are NOT a claim that allocating
//! char-budget between memory/graph/artifact retrieval is a neural
//! computation.
//!
//! Key papers: Friston 2010 (Active Inference / EFE), Desimone & Duncan
//! 1995 (biased competition softmax), Eldar 2013 (neural gain →
//! temperature), Lavie 2004 (perceptual load), Yerkes-Dodson 1908
//! (arousal × complexity inverted-U).
//!
//! Pure functions, <1ms, $0 LLM cost.
//!
//! ## Gating (P10)
//!
//! Application-infrastructure allocator, not a cognitive signal
//! producer. Output ([`AllocationResult`]) is a budget vector consumed
//! by the host application's retrieval subsystems; it does not enter
//! the convergence-loop priority chain.
//!
//! - **Fires when**: [`allocate_context_budget`] is called by the host
//!   application for a retrieval decision. [`compute_resource_pressure`]
//!   is called from [`convergence`](super::convergence) to drive
//!   downstream arousal modulation.
//! - **Inactive when**: caller passes `None` context — allocator
//!   returns `None` and the application falls back to fixed defaults
//!   (P5 fail-open).
//! - **Suppresses**: nothing. Allocation vectors do not dominate peer
//!   signals.
//! - **Suppressed by**: [`adaptive_thresholds`](super::adaptive_thresholds)
//!   indirectly — arousal and gain-mode modulation of budget shares
//!   flow through the threshold layer before reaching the allocator
//!   (single-source-of-gating rule).

use crate::math::{clamp, softmax};
use crate::types::gate::{GateResult, GateType};
use crate::types::world::GainMode;

// ── Constants ──────────────────────────────────────────────────────────

/// Simple query budget (chars).
pub const BUDGET_SIMPLE: f64 = 3000.0;
/// Moderate query budget.
pub const BUDGET_MODERATE: f64 = 5000.0;
/// Complex query budget.
pub const BUDGET_COMPLEX: f64 = 8000.0;
/// Absolute max (Cognitive Load Theory — Sweller 1988).
pub const BUDGET_CEILING: f64 = 12000.0;

/// High arousal boost for simple tasks (Yerkes-Dodson).
const YERKES_SIMPLE_BOOST: f64 = 0.15;
/// High arousal penalty for complex tasks.
const YERKES_COMPLEX_PENALTY: f64 = 0.20;
/// Arousal threshold for Yerkes-Dodson.
const YERKES_AROUSAL_THRESHOLD: f64 = 0.6;

/// Estimated chars per pinned atom.
pub const PINNED_CHARS_PER_ATOM: f64 = 400.0;
/// Estimated chars for prospective memory.
pub const PROSPECTIVE_CHARS: f64 = 200.0;

/// Softmax temperature for phasic (focused, winner-take-more).
const TEMP_PHASIC: f64 = 0.15;
/// Softmax temperature for neutral.
const TEMP_NEUTRAL: f64 = 0.30;
/// Softmax temperature for tonic (broad, equal competition).
const TEMP_TONIC: f64 = 0.50;

/// FOK below this = uncertain → explore.
const UNCERTAINTY_THRESHOLD: f64 = 0.45;
/// Min effective precision when uncertain.
const EXPLORATION_FLOOR: f64 = 0.3;
/// Epistemic bonus for low-precision layers.
const EPISTEMIC_BONUS: f64 = 0.15;
/// Default precision (neutral prior).
pub const TOPIC_PRECISION_DEFAULT: f64 = 0.5;

/// Memory boost for threat-associated topics.
const THREAT_MEMORY_BOOST: f64 = 0.25;

// Layer IDs
pub const LAYER_MEMORY: &str = "layer:memory";
pub const LAYER_GRAPH: &str = "layer:graph";
pub const LAYER_ARTIFACT: &str = "layer:artifact";
pub const LAYER_RECAP: &str = "layer:recap";

/// Model tier scaling factors.
const MODEL_TIER_NANO: f64 = 0.6;
/// Medium model baseline (1.0 = no scaling).
const MODEL_TIER_MEDIUM: f64 = 1.0;
/// Large model scaling — 50% more budget (more context capacity).
const MODEL_TIER_LARGE: f64 = 1.5;

// ── Types ──────────────────────────────────────────────────────────────

/// Task complexity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskLoad {
    Simple,
    Moderate,
    Complex,
}

/// Model tier for budget scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelTier {
    Nano,
    #[default]
    Medium,
    Large,
}

/// Context for budget allocation.
#[derive(Debug, Clone)]
pub struct AllocatorContext<'a> {
    pub query: &'a str,
    pub gate_result: Option<&'a GateResult>,
    pub gain_mode: GainMode,
    pub arousal: f64,
    pub fok_average: Option<f64>,
    pub model_tier: ModelTier,
    pub has_graph_data: bool,
    pub active_file_count: usize,
    pub pinned_count: usize,
    pub has_prospective: bool,
    pub message_count: usize,
    /// Whether any query topics are threat-associated.
    pub has_threat_topics: bool,
}

/// Per-layer budget allocation result.
#[derive(Debug, Clone)]
pub struct LayerBudget {
    pub layer_id: String,
    /// Allocated chars.
    pub budget: f64,
    /// 0-1, estimated relevance to query.
    pub query_relevance: f64,
    /// Raw per-topic precision.
    pub precision: f64,
    /// After exploration adjustment.
    pub effective_precision: f64,
    /// Relevance × effective precision.
    pub expected_utility: f64,
}

/// Complete allocation result.
#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub total_budget: f64,
    pub guaranteed_used: f64,
    pub competitive_pool: f64,
    pub task_load: TaskLoad,
    pub layers: Vec<LayerBudget>,
}

// ── Core Functions ─────────────────────────────────────────────────────

/// Estimate task complexity from query.
pub fn estimate_task_load(query: &str, gate_result: Option<&GateResult>) -> TaskLoad {
    let question_count = query.matches('?').count();
    if question_count >= 2 || query.len() > 200 {
        return TaskLoad::Complex;
    }

    // Compare/contrast patterns
    let compare_patterns = ["compare", "difference", "versus", "vs", "pros and cons"];
    if compare_patterns
        .iter()
        .any(|p| query.to_lowercase().contains(p))
    {
        return TaskLoad::Complex;
    }

    if query.len() < 50 && question_count == 0 {
        return TaskLoad::Simple;
    }

    // Routine messages → simple load (short ack, minimal retrieval needed)
    if let Some(gr) = gate_result {
        if gr.gate == GateType::Routine {
            return TaskLoad::Simple;
        }
    }

    TaskLoad::Moderate
}

/// Compute total budget (Lavie + Yerkes-Dodson + model tier).
pub fn compute_total_budget(task_load: TaskLoad, arousal: f64, model_tier: ModelTier) -> f64 {
    let mut budget = match task_load {
        TaskLoad::Simple => BUDGET_SIMPLE,
        TaskLoad::Moderate => BUDGET_MODERATE,
        TaskLoad::Complex => BUDGET_COMPLEX,
    };

    // Yerkes-Dodson: arousal interacts with complexity
    if arousal >= YERKES_AROUSAL_THRESHOLD {
        match task_load {
            TaskLoad::Simple => budget *= 1.0 + YERKES_SIMPLE_BOOST,
            TaskLoad::Complex => budget *= 1.0 - YERKES_COMPLEX_PENALTY,
            TaskLoad::Moderate => {} // No adjustment
        }
    }

    // Model tier scaling
    let tier_scale = match model_tier {
        ModelTier::Nano => MODEL_TIER_NANO,
        ModelTier::Medium => MODEL_TIER_MEDIUM,
        ModelTier::Large => MODEL_TIER_LARGE,
    };
    budget *= tier_scale;

    budget.min(BUDGET_CEILING)
}

/// Get softmax temperature from gain mode.
fn get_temperature(gain_mode: GainMode) -> f64 {
    match gain_mode {
        GainMode::Phasic => TEMP_PHASIC,
        GainMode::Tonic => TEMP_TONIC,
        GainMode::Neutral => TEMP_NEUTRAL,
    }
}

/// Compute effective precision (Active Inference: epistemic value).
pub fn compute_effective_precision(
    raw_precision: f64,
    fok_average: Option<f64>,
    layer_has_content: bool,
) -> f64 {
    if !layer_has_content {
        return 0.0;
    }

    match fok_average {
        None => raw_precision,
        Some(fok) if fok >= UNCERTAINTY_THRESHOLD => raw_precision, // Exploit
        Some(_) => {
            // Uncertain → explore (boost unproven layers)
            let bonus = if raw_precision < TOPIC_PRECISION_DEFAULT {
                EPISTEMIC_BONUS
            } else {
                0.0
            };
            (raw_precision + bonus).max(EXPLORATION_FLOOR)
        }
    }
}

/// Estimate query relevance for a layer (pragmatic value).
fn estimate_query_relevance(layer_id: &str, ctx: &AllocatorContext) -> f64 {
    match layer_id {
        LAYER_MEMORY => {
            let mut relevance = 0.3; // Baseline
            // Threat boost
            if ctx.has_threat_topics {
                relevance += THREAT_MEMORY_BOOST;
            }
            // Past-reference boost (P9b: English-only interim heuristic;
            // a language-aware version belongs in the application layer).
            let q = ctx.query.to_lowercase();
            if q.contains("before") || q.contains("earlier") || q.contains("previous") {
                relevance += 0.3;
            }
            relevance.min(1.0)
        }
        LAYER_GRAPH => {
            if !ctx.has_graph_data {
                0.0
            } else {
                0.4
            }
        }
        LAYER_ARTIFACT => {
            if ctx.active_file_count == 0 {
                0.0
            } else {
                0.4
            }
        }
        LAYER_RECAP => {
            if ctx.message_count >= 20 {
                0.4
            } else {
                0.0
            }
        }
        _ => 0.3,
    }
}

/// Allocate context budget across layers.
///
/// Returns `None` on error (fail-open: use defaults).
pub fn allocate_context_budget(
    ctx: &AllocatorContext,
    layer_precisions: &[(String, f64)],
) -> Option<AllocationResult> {
    // Step 1-2: Task load and total budget
    let task_load = estimate_task_load(ctx.query, ctx.gate_result);
    let total_budget = compute_total_budget(task_load, ctx.arousal, ctx.model_tier);

    // Step 3: Guaranteed layers
    let pinned_chars = ctx.pinned_count as f64 * PINNED_CHARS_PER_ATOM;
    let prospective_chars = if ctx.has_prospective {
        PROSPECTIVE_CHARS
    } else {
        0.0
    };
    let guaranteed_used = pinned_chars + prospective_chars;
    let competitive_pool = (total_budget - guaranteed_used).max(0.0);

    // Step 4-6: Per-layer estimation
    let layer_ids = [LAYER_MEMORY, LAYER_GRAPH, LAYER_ARTIFACT, LAYER_RECAP];
    let mut layers = Vec::new();

    for layer_id in &layer_ids {
        let query_relevance = estimate_query_relevance(layer_id, ctx);
        let raw_precision = layer_precisions
            .iter()
            .find(|(id, _)| id == *layer_id)
            .map(|(_, p)| *p)
            .unwrap_or(TOPIC_PRECISION_DEFAULT);
        let has_content = match *layer_id {
            LAYER_GRAPH => ctx.has_graph_data,
            LAYER_ARTIFACT => ctx.active_file_count > 0,
            LAYER_RECAP => ctx.message_count >= 20,
            _ => true, // Memory always has potential content
        };
        let effective_precision =
            compute_effective_precision(raw_precision, ctx.fok_average, has_content);
        let expected_utility = query_relevance * effective_precision;

        layers.push(LayerBudget {
            layer_id: layer_id.to_string(),
            budget: 0.0, // Set below
            query_relevance,
            precision: raw_precision,
            effective_precision,
            expected_utility,
        });
    }

    // Step 7: Softmax competitive allocation
    let utilities: Vec<f64> = layers.iter().map(|l| l.expected_utility).collect();
    let temp = get_temperature(ctx.gain_mode);
    let weights = softmax(&utilities, temp);

    for (layer, weight) in layers.iter_mut().zip(weights.iter()) {
        layer.budget = competitive_pool * weight;
    }

    Some(AllocationResult {
        total_budget,
        guaranteed_used,
        competitive_pool,
        task_load,
        layers,
    })
}

/// Compute resource pressure (allostatic load signal).
///
/// Feeds back into thalamic gate and sensory PE modulation.
///
/// **Math note (audit 2026-04-14)**: Previously this function multiplied the
/// summed per-layer utility by `competitive_pool` then divided by
/// `competitive_pool` — those operations cancel. The function in practice
/// returns `clamp(Σ query_relevance × effective_precision, 0.0, 1.0)`. With
/// 4 layers each ∈ [0, 1], the natural ceiling is the layer count; the clamp
/// turns "sum exceeds 1" into "saturated pressure." Whether this matches the
/// original "demand vs pool capacity" intent is a FIXME — the cancelling
/// pool factor was likely a stale refactor remnant. Behaviour preserved
/// (output unchanged) pending a proper pressure-semantic redesign.
pub fn compute_resource_pressure(allocation: Option<&AllocationResult>) -> f64 {
    match allocation {
        None => 0.5, // Neutral when no allocation data
        Some(alloc) => {
            if alloc.competitive_pool <= 0.0 {
                return 1.0;
            }
            // Equivalent to the previous (sum * pool) / pool — pool cancels.
            // Kept as explicit sum to make the actual computation legible.
            let summed_utility: f64 = alloc
                .layers
                .iter()
                .map(|l| l.query_relevance * l.effective_precision)
                .sum();
            clamp(summed_utility, 0.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_load_simple() {
        assert_eq!(estimate_task_load("ok", None), TaskLoad::Simple);
    }

    #[test]
    fn task_load_complex_multiple_questions() {
        assert_eq!(
            estimate_task_load("What is X? And how does Y work?", None),
            TaskLoad::Complex
        );
    }

    #[test]
    fn task_load_complex_compare() {
        assert_eq!(
            estimate_task_load("Compare React vs Vue", None),
            TaskLoad::Complex
        );
    }

    #[test]
    fn total_budget_simple() {
        let budget = compute_total_budget(TaskLoad::Simple, 0.0, ModelTier::Medium);
        assert!((budget - BUDGET_SIMPLE).abs() < f64::EPSILON);
    }

    #[test]
    fn total_budget_yerkes_dodson_simple_boost() {
        let budget = compute_total_budget(TaskLoad::Simple, 0.8, ModelTier::Medium);
        assert!(budget > BUDGET_SIMPLE); // Arousal helps simple tasks
    }

    #[test]
    fn total_budget_yerkes_dodson_complex_penalty() {
        let budget = compute_total_budget(TaskLoad::Complex, 0.8, ModelTier::Medium);
        assert!(budget < BUDGET_COMPLEX); // Arousal hurts complex tasks
    }

    #[test]
    fn total_budget_model_tier_scaling() {
        let nano = compute_total_budget(TaskLoad::Moderate, 0.0, ModelTier::Nano);
        let large = compute_total_budget(TaskLoad::Moderate, 0.0, ModelTier::Large);
        assert!(nano < large);
    }

    #[test]
    fn total_budget_ceiling() {
        let budget = compute_total_budget(TaskLoad::Complex, 0.0, ModelTier::Large);
        assert!(budget <= BUDGET_CEILING);
    }

    #[test]
    fn effective_precision_exploit_mode() {
        // High FOK → just use raw precision
        let eff = compute_effective_precision(0.8, Some(0.7), true);
        assert!((eff - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn effective_precision_explore_mode() {
        // Low FOK → boost low-precision layers
        let eff = compute_effective_precision(0.2, Some(0.3), true);
        assert!(eff >= EXPLORATION_FLOOR);
        assert!(eff > 0.2); // Should be boosted
    }

    #[test]
    fn effective_precision_no_content() {
        let eff = compute_effective_precision(0.8, Some(0.7), false);
        assert_eq!(eff, 0.0);
    }

    #[test]
    fn allocate_budget_returns_result() {
        let ctx = AllocatorContext {
            query: "Tell me about Rust",
            gate_result: None,
            gain_mode: GainMode::Neutral,
            arousal: 0.3,
            fok_average: Some(0.5),
            model_tier: ModelTier::Medium,
            has_graph_data: false,
            active_file_count: 0,
            pinned_count: 0,
            has_prospective: false,
            message_count: 5,
            has_threat_topics: false,
        };
        let result = allocate_context_budget(&ctx, &[]);
        assert!(result.is_some());
        let alloc = result.unwrap();
        assert!(alloc.total_budget > 0.0);
        assert_eq!(alloc.layers.len(), 4);
    }

    #[test]
    fn allocate_budget_softmax_sums_to_pool() {
        let ctx = AllocatorContext {
            query: "Complex question about multiple topics here?",
            gate_result: None,
            gain_mode: GainMode::Neutral,
            arousal: 0.0,
            fok_average: None,
            model_tier: ModelTier::Medium,
            has_graph_data: true,
            active_file_count: 3,
            pinned_count: 0,
            has_prospective: false,
            message_count: 25,
            has_threat_topics: false,
        };
        let result = allocate_context_budget(&ctx, &[]).unwrap();
        let sum: f64 = result.layers.iter().map(|l| l.budget).sum();
        assert!((sum - result.competitive_pool).abs() < 1.0); // Allow small float error
    }

    #[test]
    fn phasic_concentrates_budget() {
        let make_ctx = |gain| AllocatorContext {
            query: "Tell me about Rust",
            gate_result: None,
            gain_mode: gain,
            arousal: 0.0,
            fok_average: None,
            model_tier: ModelTier::Medium,
            has_graph_data: true,
            active_file_count: 3,
            pinned_count: 0,
            has_prospective: false,
            message_count: 25,
            has_threat_topics: false,
        };

        let phasic = allocate_context_budget(&make_ctx(GainMode::Phasic), &[]).unwrap();
        let tonic = allocate_context_budget(&make_ctx(GainMode::Tonic), &[]).unwrap();

        // Phasic should give more to the top layer
        let phasic_max = phasic
            .layers
            .iter()
            .map(|l| l.budget)
            .fold(0.0_f64, f64::max);
        let tonic_max = tonic
            .layers
            .iter()
            .map(|l| l.budget)
            .fold(0.0_f64, f64::max);

        // Phasic should concentrate more in the winner
        let phasic_ratio = phasic_max / phasic.competitive_pool;
        let tonic_ratio = tonic_max / tonic.competitive_pool;
        assert!(phasic_ratio > tonic_ratio);
    }

    #[test]
    fn resource_pressure_within_bounds() {
        let ctx = AllocatorContext {
            query: "test query",
            gate_result: None,
            gain_mode: GainMode::Neutral,
            arousal: 0.5,
            fok_average: None,
            model_tier: ModelTier::Medium,
            has_graph_data: true,
            active_file_count: 3,
            pinned_count: 2,
            has_prospective: true,
            message_count: 30,
            has_threat_topics: true,
        };
        let alloc = allocate_context_budget(&ctx, &[]).unwrap();
        let pressure = compute_resource_pressure(Some(&alloc));
        assert!((0.0..=1.0).contains(&pressure));
    }
}
