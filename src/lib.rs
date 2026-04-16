//! # Nous — Reliability infrastructure for Rust LLM agents
//!
//! [`Regulator`] sits between your agent's retry loop and your LLM. Every
//! turn, you emit a handful of events — user message, LLM response,
//! tokens spent, quality signal when you have one. `Regulator` returns a
//! [`Decision`]: `Continue`, `ScopeDriftWarn`, `CircuitBreak`,
//! `ProceduralWarning`, or `LowConfidenceSpans`. Your loop branches on
//! the variant and keeps moving.
//!
//! Nothing in Nous wraps your LLM client. There is no framework lock-in
//! and no runtime dependency on a specific model. The event surface is a
//! single enum your code owns.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use noos::{Decision, LLMEvent, Regulator};
//!
//! let mut regulator = Regulator::for_user("alice").with_cost_cap(2_000);
//!
//! regulator.on_event(LLMEvent::TurnStart {
//!     user_message: "Refactor fetch_user to be async".into(),
//! });
//!
//! // ... call your LLM of choice ...
//! # let response_text = String::new();
//! # let (tokens_in, tokens_out, wallclock_ms): (u32, u32, u32) = (0, 0, 0);
//!
//! regulator.on_event(LLMEvent::TurnComplete {
//!     full_response: response_text,
//! });
//! regulator.on_event(LLMEvent::Cost {
//!     tokens_in, tokens_out, wallclock_ms, provider: None,
//! });
//!
//! match regulator.decide() {
//!     Decision::Continue => { /* send response to user */ }
//!     Decision::CircuitBreak { .. } => { /* halt the retry loop */ }
//!     _ => { /* handle warning variants */ }
//! }
//! ```
//!
//! See `docs/regulator-guide.md` for the full event contract, decision
//! handling recipes, and gotchas; `docs/app-contract.md` for the semantic
//! contract between Nous and your application.
//!
//! ## Advanced: direct cognitive-session access
//!
//! Underneath `Regulator` runs [`session::CognitiveSession`], a pipeline
//! producing continuous signals (`conservation`, `confidence`, strategy
//! recommendation, gain mode) plus delta-modulation output for local
//! Mamba/SSM inference (requires the `candle` feature flag).
//!
//! Most integrations do not need this layer. Use `CognitiveSession`
//! directly only if you need raw continuous signals for a custom policy
//! or are running local Mamba inference (perplexity −1.86 % on emotional
//! text, 3 runs bit-identical).

pub mod errors;
pub mod types;
pub mod math;
pub mod cognition;
pub mod kernel;
pub mod ai;
pub mod memory;
pub mod inference;
pub mod session;
pub mod regulator;

// ── Primary API: Regulator (event-driven reliability layer) ──────────────
//
// Sessions 16–20 built the current surface: event dispatch, token-stats
// confidence, scope-drift tracker, cost accumulator + CircuitBreak,
// correction store + ProceduralWarning + persistence split. New
// integrations should start here.
pub use regulator::{
    CircuitBreakReason, ConfidenceSpan, CorrectionPattern, Decision, LLMEvent, Regulator,
    RegulatorState,
};
pub use regulator::correction::CorrectionStore;
pub use regulator::cost::CostAccumulator;
pub use regulator::scope::ScopeTracker;
pub use regulator::token_stats::TokenStatsAccumulator;

pub use errors::{NousError, NousResult};

// ── Advanced: direct cognitive-session access ────────────────────────────
//
// `Regulator` wraps [`session::CognitiveSession`] and feeds it
// LLM-operational events. Most integrations do not need the layer below.
// Use these re-exports only when you need raw continuous signals for a
// custom decision policy, or are running local Mamba/SSM inference and
// want delta-modulation hints (requires the `candle` feature flag).
pub use types::belief::{AffectState, AffectValence, SharedBeliefState};
pub use types::gate::{GateContext, GateResult, GateType};
pub use types::intervention::{
    CognitiveSignals, CognitiveState, DeltaModulation, ForwardResult, HiddenStateStats,
    InterventionDepth, LayerTarget, SamplingOverride,
};
pub use types::world::{GainMode, LearnedState, WorldModel};

pub use cognition::convergence::{converge, ConvergenceContext, ConvergenceResult};
pub use cognition::delta_modulation::compute_delta_modulation;
pub use cognition::intervention::{build_cognitive_state, compute_sampling_override};
pub use cognition::signals::compute_signals;

// Memory system — applications own async I/O, Nous provides sync cognitive
// computation over pre-loaded atoms (P4 trait boundary). Used by
// `CognitiveSession` for recall; `Regulator` does not touch this path.
pub use memory::retrieval::{hybrid_recall, ActivatedAtom, ActivationSource, RecallOptions};
pub use memory::store::{AtomUpdate, InMemoryStore, MemoryStore};
pub use types::memory::{AtomSource, AtomType, MemoryAtom, Synapse, SynapseType};
