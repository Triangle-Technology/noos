//! Regulator — reliability layer for LLM agent loops (Path 2).
//!
//! Reads LLM operation events (token stream, user corrections, cost accounting,
//! quality feedback) and emits regulatory decisions (continue, circuit-break,
//! scope-drift warning, low-confidence fragment flagging, procedural correction
//! warning).
//!
//! **Scope note (P1 / P9b)**: this module is an I/O adapter, not a cognitive
//! module. All cognitive mechanisms — convergence loop, LC-NE gain modulation,
//! body-budget allostasis, per-cluster strategy EMA — live inside the wrapped
//! [`CognitiveSession`], which carries the brain-analog framing (Friston 2010
//! perception-action cycle, Aston-Jones 2005 LC-NE). `Regulator` dispatches
//! LLM-operational events into that pipeline; it does not itself compute
//! anything cognitive. P1 (neuroscience grounding) applies to the wrapped
//! session; P9b (don't duplicate cortical work) is satisfied by construction
//! because the event stream is LLM-generated, never regex-on-user-text.
//!
//! ## Positioning vs Path 1
//!
//! `session::CognitiveSession` (Path 1) builds cognitive state from user text
//! via regex/lexicon adapters. `Regulator` (Path 2, 2026-04-15) wraps
//! `CognitiveSession` and replaces the input pathway with LLM-operational
//! events. Downstream signals (`CognitiveSignals`, `body_budget`, etc.) stay
//! the same — only the source of evidence changes.
//!
//! Path 1 APIs remain callable for backwards compatibility. New integrations
//! should prefer `Regulator`.
//!
//! ## Implementation status (Sessions 16–20)
//!
//! - **Session 16** — public API shape: [`Regulator`], [`LLMEvent`],
//!   [`Decision`], [`RegulatorState`]. Event dispatch wires `TurnStart`
//!   into `CognitiveSession::process_message` and routes
//!   `TurnComplete` + `QualityFeedback` through `process_response` via
//!   a response buffer.
//! - **Session 17** — [`token_stats::TokenStatsAccumulator`] added.
//!   `Token` events populate a rolling logprob window, and
//!   [`Regulator::confidence`] produces a hybrid confidence readout.
//! - **Session 18** — [`scope::ScopeTracker`] added. `TurnStart` /
//!   `TurnComplete` populate task / response keyword bags and
//!   [`Regulator::decide`] emits [`Decision::ScopeDriftWarn`] on
//!   high drift.
//! - **Session 19** — [`cost::CostAccumulator`] added. `Cost` events
//!   fold into cumulative counters and [`cost::normalize_cost`]
//!   feeds `CognitiveSession::track_cost`. `decide()` gained
//!   [`Decision::CircuitBreak`] predicates with an explicit P10
//!   priority order.
//! - **Session 20** — [`correction::CorrectionStore`] added.
//!   `UserCorrection` events with `corrects_last = true` record
//!   against the current scope cluster; once
//!   [`correction::MIN_CORRECTIONS_FOR_PATTERN`] is reached,
//!   `decide()` emits [`Decision::ProceduralWarning`] before the next
//!   generation. [`RegulatorState`] moved to the new [`state`]
//!   submodule and gained a `correction_patterns` field so patterns
//!   survive process restarts.
//!
//! See `docs/regulator-design.md` for the authoritative spec.

pub mod correction;
pub mod cost;
pub mod scope;
pub mod state;
pub mod token_stats;

use serde::{Deserialize, Serialize};

use crate::session::CognitiveSession;

use self::correction::CorrectionStore;
use self::cost::{
    normalize_cost, CostAccumulator, POOR_QUALITY_MEAN, QUALITY_DECLINE_MIN_DELTA,
    QUALITY_DECLINE_WINDOW,
};
use self::scope::{ScopeTracker, DRIFT_WARN_THRESHOLD};
use self::token_stats::{confidence_with_fallback, TokenStatsAccumulator};

pub use self::state::RegulatorState;

// ── Events ─────────────────────────────────────────────────────────────

/// Input event from an LLM agent loop.
///
/// Callers emit one or more events per turn. Typical ordering for a
/// successful turn:
///
/// 1. `TurnStart` — user message arrives.
/// 2. Zero or more `Token` events as the LLM streams output.
///    Non-streaming clients can skip this and emit a single `TurnComplete`.
/// 3. `TurnComplete` — final response text.
/// 4. `Cost` — token counts and wallclock for the turn.
/// 5. `QualityFeedback` (optional) — explicit ground-truth signal.
/// 6. `UserCorrection` (optional, next turn) — if the user pushes back.
///
/// `Regulator` is forgiving about missing or out-of-order events. Fields are
/// structured so higher-fidelity providers (logprobs, fragment spans) can
/// populate more detail without breaking callers that have less information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LLMEvent {
    /// A new turn begins. The user's message.
    TurnStart {
        user_message: String,
    },

    /// One token emitted by the LLM.
    ///
    /// Streaming clients emit per-token. Non-streaming clients can skip
    /// `Token` entirely or emit a single aggregate at end of turn.
    ///
    /// `logprob` is the natural-log probability of this token. When a
    /// provider does not expose per-token logprobs (e.g., Anthropic as of
    /// 2026-04), callers pass `0.0` to signal "unknown" — the
    /// [`token_stats`] accumulator treats any non-finite or
    /// non-negative value as unavailable and falls back to the
    /// structural confidence heuristic (see `token_stats` module docs).
    Token {
        token: String,
        logprob: f64,
        index: usize,
    },

    /// Turn response complete. Full text regardless of whether the caller
    /// streamed tokens.
    TurnComplete {
        full_response: String,
    },

    /// Cost accounting for the turn. Emit after `TurnComplete`.
    Cost {
        tokens_in: u32,
        tokens_out: u32,
        wallclock_ms: u32,
        /// Optional provider tag for multi-provider agents.
        provider: Option<String>,
    },

    /// User corrected the previous response. Used for procedural learning.
    ///
    /// `corrects_last == true` means "this user message is a correction of
    /// the previous response"; `false` means "new independent query" (in
    /// which case callers should prefer `TurnStart`).
    UserCorrection {
        correction_message: String,
        corrects_last: bool,
    },

    /// Ground-truth signal about response quality. Closes the learning loop.
    ///
    /// Typical sources: thumbs-up/down, automated evaluators, task
    /// completion checks. When present, drains any buffered
    /// `TurnComplete` into the underlying session's strategy-learning path.
    QualityFeedback {
        /// Quality in `[0, 1]`. Callers clamp out-of-range values at the
        /// boundary; the regulator does not assume a specific evaluator.
        quality: f64,
        /// Optional: response fragments that triggered this feedback.
        /// Currently unused — reserved for a future `LowConfidenceSpans`
        /// predicate that will use span-local logprobs + feedback to
        /// flag specific response ranges. Callers may pass `None`.
        fragment_spans: Option<Vec<(usize, usize)>>,
    },
}

// ── Decisions ──────────────────────────────────────────────────────────

/// Regulatory decision returned by [`Regulator::decide`].
///
/// A single turn may warrant multiple concerns (drift AND low confidence).
/// v1 `decide()` returns a single `Decision`; callers may call `decide()`
/// repeatedly or branch on variant. Multi-concern aggregation is a Session
/// 19+ refinement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Decision {
    /// Continue normally. No intervention required.
    Continue,

    /// Stop the agent loop. Further LLM calls are unlikely to help.
    CircuitBreak {
        reason: CircuitBreakReason,
        /// Human-readable suggestion for the application to surface, e.g.
        /// "ask user to clarify scope".
        suggestion: String,
    },

    /// Response drifted beyond task scope. App may accept, strip, or
    /// re-prompt.
    ScopeDriftWarn {
        drift_tokens: Vec<String>,
        /// How far out of scope, in `[0, 1]`.
        drift_score: f64,
        /// Original task keywords for caller reference.
        task_tokens: Vec<String>,
    },

    /// Specific response fragments have low confidence. App may highlight
    /// for user review or re-generate those spans.
    LowConfidenceSpans {
        spans: Vec<ConfidenceSpan>,
    },

    /// Apply learned procedural pattern before next generation.
    ///
    /// Example: "user_123 + refactor cluster → do not add error handling".
    /// Patterns come from `CorrectionStore` (Session 20).
    ProceduralWarning {
        patterns: Vec<CorrectionPattern>,
    },
}

/// Why a `CircuitBreak` fired.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitBreakReason {
    /// Budget cap reached with response quality still poor.
    CostCapReached {
        tokens_spent: u32,
        tokens_cap: u32,
        mean_quality_last_n: f64,
    },
    /// Quality trending down across N consecutive turns without recovery.
    QualityDeclineNoRecovery {
        turns: usize,
        mean_delta: f64,
    },
    /// Repeated failure on the same topic cluster.
    RepeatedFailurePattern {
        cluster: String,
        failure_count: usize,
    },
}

/// A span of response text flagged as low-confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceSpan {
    pub start_char: usize,
    pub end_char: usize,
    /// Confidence in `[0, 1]`. Low values indicate the model was uncertain.
    pub confidence: f64,
    /// Mean natural-log probability of tokens in the span.
    pub mean_token_logprob: f64,
}

/// A learned procedural rule extracted from repeated `UserCorrection`
/// events.
///
/// Session 20 MVP: pattern identity is STRUCTURAL (cluster-based count
/// threshold), not semantic. `pattern_name` is opaque
/// (`corrections_on_{cluster}`) — no English-regex rule extraction,
/// P9b-compliant. The `example_corrections` field carries raw
/// correction texts the app / LLM can pass through for rule
/// interpretation at generation time.
///
/// `#[non_exhaustive]` is set so future sessions can add fields (e.g.
/// per-example timestamps, extracted rule strings from an optional LLM
/// classifier pass) without breaking downstream matchers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CorrectionPattern {
    pub user_id: String,
    pub topic_cluster: String,
    /// Opaque identifier, currently `corrections_on_{cluster}`.
    pub pattern_name: String,
    pub learned_from_turns: usize,
    /// Confidence in `[0, 1]`.
    pub confidence: f64,
    /// Up to [`correction::MAX_EXAMPLE_CORRECTIONS`] most-recent raw
    /// correction texts, newest first. Apps pass these to the LLM for
    /// rule interpretation. Empty when deserialized from a
    /// pre-Session-20 snapshot (`#[serde(default)]`).
    #[serde(default)]
    pub example_corrections: Vec<String>,
}

// ── Regulator ──────────────────────────────────────────────────────────

// `RegulatorState` moved to the `state` submodule in Session 20 so the
// persistence envelope grows independently of the dispatch surface.
// Re-exported above for backcompat.

/// External regulatory layer for an LLM agent loop.
///
/// See module docs for the event-driven contract and positioning relative
/// to Path 1 (`CognitiveSession`).
pub struct Regulator {
    session: CognitiveSession,
    user_id: String,
    /// Response text buffered by the last `TurnComplete`, awaiting a
    /// `QualityFeedback` signal to close the learning loop. `None` when
    /// idle or already consolidated.
    pending_response: Option<String>,
    /// Per-turn rolling logprob window + coverage counters. Reset on
    /// each `TurnStart`. Drives [`Self::confidence`] via
    /// [`token_stats::confidence_with_fallback`].
    token_stats: TokenStatsAccumulator,
    /// Per-turn task / response keyword bags. Reset on each `TurnStart`
    /// and populated on `TurnComplete`. Drives
    /// [`Decision::ScopeDriftWarn`] emission in [`Self::decide`].
    scope: ScopeTracker,
    /// Cumulative token / wallclock counters + rolling quality history.
    /// Unlike `token_stats` and `scope` (both per-turn), this one
    /// persists across turns for the Regulator's lifetime — the "agent
    /// task" cost budget is cumulative. Drives
    /// [`Decision::CircuitBreak`] emission in [`Self::decide`].
    cost: CostAccumulator,
    /// Per-(user, topic-cluster) record of user corrections + a
    /// structural pattern extractor. Persists across turns and across
    /// process restarts via [`RegulatorState::correction_patterns`].
    /// Drives [`Decision::ProceduralWarning`] emission in
    /// [`Self::decide`].
    correction: CorrectionStore,
    /// Most-recent topic cluster key computed from `TurnStart.user_message`.
    /// Used by the `UserCorrection` handler to attribute a correction to
    /// the cluster that was active when the flawed response was
    /// generated. Empty string before the first `TurnStart`, or when the
    /// message has no extractable top-2 topics.
    current_topic_cluster: String,
}

impl Regulator {
    /// Create a fresh regulator bound to a user identity.
    ///
    /// Use `import` to restore a saved snapshot instead.
    pub fn for_user(user_id: impl Into<String>) -> Self {
        Self {
            session: CognitiveSession::new(),
            user_id: user_id.into(),
            pending_response: None,
            token_stats: TokenStatsAccumulator::new(),
            scope: ScopeTracker::new(),
            cost: CostAccumulator::new(),
            correction: CorrectionStore::new(),
            current_topic_cluster: String::new(),
        }
    }

    /// Builder: override the default cumulative output-token cap used
    /// by the cost-cap CircuitBreak predicate.
    ///
    /// Does not reset accumulated counters — the running totals stay
    /// intact. Useful for demos / tests (the plan test target uses
    /// 1_000) and for agents with tight budget envelopes. Default cap
    /// is [`cost::DEFAULT_TOKEN_CAP`] (10_000).
    pub fn with_cost_cap(mut self, cap_tokens: u32) -> Self {
        self.cost.set_cap(cap_tokens);
        self
    }

    /// Mutable: ingests one LLM operation event and forwards to the wrapped
    /// session when the event type warrants it. Requires mutation because the
    /// wrapped `CognitiveSession` accumulates state per turn (world model,
    /// LC gain, history) and this regulator buffers the most recent response
    /// between `TurnComplete` and `QualityFeedback`.
    ///
    /// Does not return a decision — call [`decide`](Self::decide) after the
    /// turn settles.
    ///
    /// Session 16 dispatch behaviour:
    ///
    /// - `TurnStart` runs the cognitive pipeline via `process_message`.
    /// - `TurnComplete` buffers the response for later consolidation.
    /// - `QualityFeedback` drains the buffer into `process_response`, so
    ///   existing strategy learning continues to fire through Path 2.
    /// - `Token`, `Cost`, `UserCorrection` are recorded but inert until
    ///   Sessions 17–20 wire their accumulators.
    pub fn on_event(&mut self, event: LLMEvent) {
        match event {
            LLMEvent::TurnStart { user_message } => {
                // Reset per-turn statistics before the new turn runs.
                // The logprob window is per-turn: confidence should not
                // drag forward from the previous turn's tokens.
                self.token_stats.begin_turn();
                // Scope tracker is also per-turn; `set_task` loads the
                // new task keywords and clears any stale response
                // keywords from the previous turn.
                self.scope.set_task(&user_message);
                // Compute the topic cluster for this turn using the
                // same `build_topic_cluster` algorithm LearnedState
                // uses (P3) — the `UserCorrection` handler and the
                // `decide()` ProceduralWarning predicate both key off
                // this cluster to attribute corrections consistently.
                self.current_topic_cluster = crate::cognition::detector::build_topic_cluster(
                    self.scope.task_tokens(),
                );
                // Path 1 cognitive pipeline still runs; downstream signals
                // continue to update while the Path 2 input adapters grow.
                let _ = self.session.process_message(&user_message);
            }

            LLMEvent::Token { logprob, .. } => {
                // Feed the rolling window. Callers that can't provide
                // real logprobs pass `LOGPROB_UNAVAILABLE` (= 0.0); the
                // accumulator handles that case and
                // `Self::confidence` falls back to the structural
                // heuristic when the window ends up empty.
                self.token_stats.on_token(logprob);
            }

            LLMEvent::TurnComplete { full_response } => {
                // Populate scope keywords for drift detection. Doing
                // this before the buffer move lets us borrow
                // `full_response` for keyword extraction without
                // cloning.
                self.scope.set_response(&full_response);
                // Buffer until we learn the turn's quality. If a caller
                // emits `TurnComplete` twice without intervening feedback,
                // the later response replaces the earlier one — mirrors
                // "last response wins" semantics for retry loops.
                self.pending_response = Some(full_response);
            }

            LLMEvent::Cost {
                tokens_in,
                tokens_out,
                wallclock_ms,
                provider: _,
            } => {
                // Record raw counters for circuit-break predicates.
                self.cost.record_cost(tokens_in, tokens_out, wallclock_ms);
                // Feed normalised [0, 1] cost into Path 1 body-budget
                // allostasis — closes the loop the Path 1 design
                // intended (see `docs/app-contract.md`).
                let normalised = normalize_cost(tokens_out, wallclock_ms);
                self.session.track_cost(normalised);
            }

            LLMEvent::UserCorrection {
                correction_message,
                corrects_last,
            } => {
                // Only treat as a correction when the caller flagged it
                // as one. `corrects_last == false` means "new
                // independent query"; the correct channel for that is
                // `TurnStart` and we drop rather than misattribute.
                if !corrects_last {
                    return;
                }
                if self.current_topic_cluster.is_empty() {
                    // No active cluster — the corrected turn had no
                    // identifiable top-2 topics, so the correction
                    // can't be attributed. Drop rather than pollute
                    // the any-cluster path.
                    return;
                }
                self.correction
                    .record_correction(&self.current_topic_cluster, correction_message);
            }

            LLMEvent::QualityFeedback { quality, .. } => {
                // Record in cost accumulator for trend analysis before
                // (possibly) consuming the pending response — quality
                // belongs to the agent's overall recent performance
                // regardless of whether this specific turn produced a
                // buffered response.
                self.cost.record_quality(quality);
                // Close the learning loop if we have a buffered response.
                // Without one there is no turn to score against; dropping
                // the signal is safer than inventing a target.
                if let Some(response) = self.pending_response.take() {
                    self.session.process_response(&response, quality);
                }
            }
        }
    }

    /// Current turn-level confidence estimate in `[0, 1]`.
    ///
    /// Primary path: mean negative-log-likelihood over the rolling
    /// logprob window, when any tokens have logprobs available.
    /// Fallback path: language-neutral structural heuristic on the
    /// buffered response text. Returns
    /// [`token_stats::NEUTRAL_CONFIDENCE`] (0.5) when neither signal is
    /// available (e.g., before any turn completes).
    ///
    /// Path 1 users who read `turn.signals.confidence` from
    /// `CognitiveSession::process_message` see the legacy 0.5 base
    /// unchanged — wiring this dynamic value through to
    /// `CognitiveSignals` is deferred to a later session (would require
    /// a new mutation hook on the wrapped session).
    pub fn confidence(&self) -> f64 {
        confidence_with_fallback(&self.token_stats, self.pending_response.as_deref())
    }

    /// Fraction of tokens in the current turn whose logprobs were
    /// available, in `[0, 1]`. Useful for callers that want to discount
    /// the confidence signal when coverage is low.
    pub fn logprob_coverage(&self) -> f64 {
        self.token_stats.logprob_coverage()
    }

    /// Cumulative output tokens recorded via `Cost` events so far.
    /// Useful for callers that want to surface an explicit budget
    /// progress bar to the user without reaching through `session_mut`.
    pub fn total_tokens_out(&self) -> u32 {
        self.cost.total_tokens_out()
    }

    /// Current cost cap in cumulative output tokens. See
    /// [`cost::CostAccumulator::cap_tokens`] for the default and
    /// [`cost::CostAccumulator::with_cap`] for overriding.
    pub fn cost_cap_tokens(&self) -> u32 {
        self.cost.cap_tokens()
    }

    /// Query the current regulatory decision.
    ///
    /// ## Priority order (P10)
    ///
    /// Multiple predicates can fire on the same turn (a cost-cap hit
    /// alongside scope drift, alongside a learned correction pattern).
    /// The single [`Decision`] returned follows a strict priority
    /// order, highest first:
    ///
    /// 1. [`Decision::CircuitBreak`] with
    ///    [`CircuitBreakReason::CostCapReached`] — hard budget ceiling
    ///    reached with poor recent quality. Most urgent: the agent
    ///    should stop, not just warn.
    /// 2. [`Decision::CircuitBreak`] with
    ///    [`CircuitBreakReason::QualityDeclineNoRecovery`] — quality
    ///    trending down over [`QUALITY_DECLINE_WINDOW`] turns with mean
    ///    still below [`POOR_QUALITY_MEAN`]. Urgent enough to halt a
    ///    retry loop.
    /// 3. [`Decision::ScopeDriftWarn`] — response keywords disjoint
    ///    from task keywords (semantic warning, not a stop).
    /// 4. [`Decision::ProceduralWarning`] — the current topic cluster
    ///    has a learned [`CorrectionPattern`] from repeated past user
    ///    corrections (advisory: app / LLM should consult the
    ///    `example_corrections` before generating).
    /// 5. [`Decision::Continue`] — no fired predicates.
    ///
    /// Rationale: urgent stop signals dominate semantic warnings which
    /// dominate historical-pattern advisories. A future session will
    /// add [`Decision::LowConfidenceSpans`] below `ProceduralWarning`
    /// in the order (both advisory, but spans are span-local while
    /// procedural warnings are turn-wide — so procedural fires first
    /// when both are live).
    ///
    /// All predicates read accumulated state — `decide()` is idempotent
    /// within a turn and safe to call repeatedly.
    pub fn decide(&self) -> Decision {
        // ── 1. Cost-cap circuit break (top priority) ──
        if self.cost.cap_reached() {
            let mean_quality_last_n = self
                .cost
                .mean_quality_last_n(QUALITY_DECLINE_WINDOW)
                .unwrap_or(1.0);
            if mean_quality_last_n < POOR_QUALITY_MEAN {
                return Decision::CircuitBreak {
                    reason: CircuitBreakReason::CostCapReached {
                        tokens_spent: self.cost.total_tokens_out(),
                        tokens_cap: self.cost.cap_tokens(),
                        mean_quality_last_n,
                    },
                    suggestion:
                        "Cost cap reached with poor recent quality. Ask the user to clarify scope or abandon this task."
                            .into(),
                };
            }
        }

        // ── 2. Quality-decline circuit break ──
        if let Some(delta) = self
            .cost
            .quality_decline_over_n(QUALITY_DECLINE_WINDOW, QUALITY_DECLINE_MIN_DELTA)
        {
            let mean = self
                .cost
                .mean_quality_last_n(QUALITY_DECLINE_WINDOW)
                .unwrap_or(1.0);
            if mean < POOR_QUALITY_MEAN {
                return Decision::CircuitBreak {
                    reason: CircuitBreakReason::QualityDeclineNoRecovery {
                        turns: QUALITY_DECLINE_WINDOW,
                        mean_delta: delta,
                    },
                    suggestion:
                        "Response quality is declining without recovery. Consider redirecting or simplifying the task."
                            .into(),
                };
            }
        }

        // ── 3. Scope-drift warning ──
        if let Some(drift) = self.scope.drift_score() {
            if drift >= DRIFT_WARN_THRESHOLD {
                return Decision::ScopeDriftWarn {
                    drift_tokens: self.scope.drift_tokens(),
                    drift_score: drift,
                    task_tokens: self.scope.task_tokens().to_vec(),
                };
            }
        }

        // ── 4. Procedural-correction warning ──
        if !self.current_topic_cluster.is_empty() {
            if let Some(pattern) = self
                .correction
                .pattern_for(&self.user_id, &self.current_topic_cluster)
            {
                return Decision::ProceduralWarning {
                    patterns: vec![pattern],
                };
            }
        }

        // ── 5. Fall through ──
        Decision::Continue
    }

    /// Export persistent state for storage. Callers serialise with
    /// `serde_json` or similar.
    ///
    /// Session 20: includes `correction_patterns` — every cluster with
    /// ≥ [`correction::MIN_CORRECTIONS_FOR_PATTERN`] corrections is
    /// exported as a [`CorrectionPattern`]. Below-threshold records
    /// do not survive export (they would on import devolve back to
    /// below-threshold), so an exported → imported regulator has the
    /// same *patterns* but not the exact pending-count-building-up
    /// state. This trade-off keeps the snapshot focused on stable
    /// learned rules.
    pub fn export(&self) -> RegulatorState {
        let correction_patterns = self
            .correction
            .all_patterns(&self.user_id)
            .into_iter()
            .map(|p| (p.topic_cluster.clone(), p))
            .collect();
        RegulatorState {
            user_id: self.user_id.clone(),
            learned: self.session.export_learned(),
            correction_patterns,
        }
    }

    /// Restore a regulator from a previously exported snapshot.
    ///
    /// Only durable cross-session state (`LearnedState` via the wrapped
    /// session) is restored. Per-turn state (buffered response, token
    /// stats window) starts fresh — it's not meaningful to resume a
    /// half-generated turn across process restarts.
    pub fn import(state: RegulatorState) -> Self {
        let mut session = CognitiveSession::new();
        session.import_learned(state.learned);
        // Session 20: restore persisted correction patterns by replaying
        // each example-correction back into a fresh CorrectionStore.
        // Replaying gives us the exact same count-based confidence
        // signal the store would produce live; storing only the
        // `pattern_for` output would lose the underlying records needed
        // to extend patterns with new corrections post-restart.
        let mut correction = CorrectionStore::new();
        for (cluster, pattern) in &state.correction_patterns {
            for text in &pattern.example_corrections {
                correction.record_correction(cluster, text.clone());
            }
        }
        Self {
            session,
            user_id: state.user_id,
            pending_response: None,
            token_stats: TokenStatsAccumulator::new(),
            scope: ScopeTracker::new(),
            cost: CostAccumulator::new(),
            correction,
            current_topic_cluster: String::new(),
        }
    }

    /// Identity this regulator is bound to.
    pub fn user_id(&self) -> &str {
        &self.user_id
    }

    /// Escape hatch: read-only access to the wrapped Path 1 session for
    /// callers that want raw cognitive signals, Tầng 2 delta modulation, or
    /// the world-model snapshot.
    pub fn session(&self) -> &CognitiveSession {
        &self.session
    }

    /// Mutable: escape hatch for callers that need to invoke Path 1 methods
    /// directly (e.g., `track_cost`, `idle_cycle`) until Sessions 17–20 wire
    /// them through events. Requires `&mut self` because callers expect to
    /// mutate the wrapped session through the returned reference.
    pub fn session_mut(&mut self) -> &mut CognitiveSession {
        &mut self.session
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_user_starts_with_fresh_session() {
        // Contract: identity is preserved and cognitive state is zeroed.
        // "No buffered response" is tested indirectly by
        // `quality_feedback_without_turn_complete_is_noop` — here we only
        // verify the observable session state.
        let reg = Regulator::for_user("user_42");
        assert_eq!(reg.user_id(), "user_42");
        assert_eq!(reg.session().turn_count(), 0);
        assert!(reg.session().world_model().last_response_strategy.is_none());
    }

    #[test]
    fn turn_start_runs_cognitive_pipeline() {
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "Explain async Rust".into(),
        });
        // `process_message` increments turn count — proves dispatch reached
        // the cognitive pipeline, not just the match arm.
        assert_eq!(reg.session().turn_count(), 1);
    }

    #[test]
    fn turn_complete_without_feedback_does_not_learn() {
        // Contract: learning fires only when a quality signal lands, not on
        // `TurnComplete` alone. Mirrors Path 1's `process_response(text,
        // quality)` signature — quality is required input.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "How to use async?".into(),
        });
        reg.on_event(LLMEvent::TurnComplete {
            full_response:
                "Here's a step-by-step guide:\n1. Add tokio\n2. Write async fn\n3. Await"
                    .into(),
        });
        assert!(reg.session().world_model().last_response_strategy.is_none());
    }

    #[test]
    fn quality_feedback_consolidates_buffered_response() {
        // Contract: a full TurnStart → TurnComplete → QualityFeedback cycle
        // records a response strategy, matching Path 1 `process_response`
        // behaviour. The buffer-drain is verified indirectly by
        // `second_quality_feedback_after_drain_is_noop` below.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "How do I use async in Rust?".into(),
        });
        reg.on_event(LLMEvent::TurnComplete {
            full_response:
                "Here's a step-by-step guide:\n1. Add tokio\n2. Write async fn\n3. Await"
                    .into(),
        });
        reg.on_event(LLMEvent::QualityFeedback {
            quality: 0.85,
            fragment_spans: None,
        });
        assert!(reg
            .session()
            .world_model()
            .last_response_strategy
            .is_some());
    }

    #[test]
    fn second_quality_feedback_after_drain_is_noop() {
        // Contract: once a response is consolidated, a stray second
        // `QualityFeedback` (no intervening `TurnComplete`) does not fire
        // learning a second time. This is the observable complement to the
        // buffer-drain mechanism. We compare the serialised `LearnedState`
        // rather than peek at the buffer field.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "How to async?".into(),
        });
        reg.on_event(LLMEvent::TurnComplete {
            full_response: "Step 1: Add tokio\nStep 2: async fn\nStep 3: await".into(),
        });
        reg.on_event(LLMEvent::QualityFeedback {
            quality: 0.85,
            fragment_spans: None,
        });
        let learned_after_first = serde_json::to_string(
            &reg.session().world_model().learned,
        )
        .expect("serialise LearnedState");

        // Second QF with nothing buffered + a different quality — if the
        // buffer were still full, `process_response` would fire again with
        // quality=0.1 and the EMA would shift, producing a different JSON.
        reg.on_event(LLMEvent::QualityFeedback {
            quality: 0.1,
            fragment_spans: None,
        });
        let learned_after_second = serde_json::to_string(
            &reg.session().world_model().learned,
        )
        .expect("serialise LearnedState");

        assert_eq!(
            learned_after_first, learned_after_second,
            "drained buffer must not be re-consolidated by a stray feedback"
        );
    }

    #[test]
    fn quality_feedback_without_turn_complete_is_noop() {
        // Contract: with no buffered response, `QualityFeedback` is a no-op;
        // the regulator's state is not corrupted (a subsequent valid cycle
        // still learns). Matches the Path 1 invariant that
        // `process_response` is never called with an empty or stale
        // response.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "How to async?".into(),
        });
        reg.on_event(LLMEvent::QualityFeedback {
            quality: 0.5,
            fragment_spans: None,
        });
        assert!(reg.session().world_model().last_response_strategy.is_none());

        // Recovery — a valid cycle afterwards still fires learning.
        reg.on_event(LLMEvent::TurnComplete {
            full_response: "Step 1: First\nStep 2: Second\nStep 3: Third".into(),
        });
        reg.on_event(LLMEvent::QualityFeedback {
            quality: 0.8,
            fragment_spans: None,
        });
        assert!(reg.session().world_model().last_response_strategy.is_some());
    }

    #[test]
    fn inert_events_do_not_panic_or_mutate_turn_count() {
        let mut reg = Regulator::for_user("user_a");
        // Establish a turn so turn_count is >0 — lets us assert the inert
        // variants truly don't advance it.
        reg.on_event(LLMEvent::TurnStart { user_message: "hi".into() });
        let before = reg.session().turn_count();

        reg.on_event(LLMEvent::Token {
            token: "hello".into(),
            logprob: -0.5,
            index: 0,
        });
        reg.on_event(LLMEvent::Cost {
            tokens_in: 10,
            tokens_out: 20,
            wallclock_ms: 500,
            provider: Some("anthropic".into()),
        });
        reg.on_event(LLMEvent::UserCorrection {
            correction_message: "don't add docstrings".into(),
            corrects_last: true,
        });

        // No new turn should have been started by the skeleton dispatch.
        assert_eq!(reg.session().turn_count(), before);
    }

    #[test]
    fn decide_returns_continue_by_default() {
        let reg = Regulator::for_user("user_a");
        assert!(matches!(reg.decide(), Decision::Continue));
    }

    #[test]
    fn export_import_roundtrip_preserves_learning() {
        let mut reg = Regulator::for_user("user_persist");
        // Train enough turns to populate response_strategies via
        // `process_response` through QualityFeedback routing.
        for i in 0..10 {
            reg.on_event(LLMEvent::TurnStart {
                user_message: format!("Rust question {i}"),
            });
            reg.on_event(LLMEvent::TurnComplete {
                full_response: "Step 1: First\nStep 2: Second\nStep 3: Third".into(),
            });
            reg.on_event(LLMEvent::QualityFeedback {
                quality: 0.85,
                fragment_spans: None,
            });
        }

        let snapshot = reg.export();
        assert_eq!(snapshot.user_id, "user_persist");
        assert!(
            !snapshot.learned.response_strategies.is_empty(),
            "training loop should have populated strategy EMA"
        );

        // Restore into a new regulator — learning carries across.
        let restored = Regulator::import(snapshot.clone());
        assert_eq!(restored.user_id(), "user_persist");
        assert_eq!(
            restored.session().world_model().learned.response_strategies.len(),
            snapshot.learned.response_strategies.len(),
        );
    }

    #[test]
    fn roundtrip_via_serde_json() {
        // Skeleton guarantee: `RegulatorState` round-trips through JSON.
        // Sessions 20+ add fields; serde `#[serde(default)]` will keep old
        // snapshots loading against the new struct.
        let mut reg = Regulator::for_user("user_json");
        reg.on_event(LLMEvent::TurnStart { user_message: "hi".into() });
        reg.on_event(LLMEvent::TurnComplete { full_response: "hello".into() });
        reg.on_event(LLMEvent::QualityFeedback {
            quality: 0.7,
            fragment_spans: None,
        });

        let snapshot = reg.export();
        let json = serde_json::to_string(&snapshot).expect("serialise");
        let decoded: RegulatorState =
            serde_json::from_str(&json).expect("deserialise");
        assert_eq!(decoded.user_id, snapshot.user_id);
        assert_eq!(decoded.learned.tick, snapshot.learned.tick);
    }

    #[test]
    fn session_mut_exposes_path1_escape_hatch() {
        // Confirms the escape hatch works — Tầng 2 users and callers that
        // want `track_cost` before Session 19 lands can reach the wrapped
        // session directly.
        let mut reg = Regulator::for_user("user_a");
        let initial = reg.session().world_model().body_budget;
        reg.session_mut().track_cost(1.0);
        let after = reg.session().world_model().body_budget;
        assert!(after < initial, "track_cost via session_mut should deplete budget");
    }

    // ── Session 17: confidence wiring ──────────────────────────────────

    #[test]
    fn confidence_starts_neutral() {
        // Contract: before any events, confidence is the documented
        // neutral default (matches the legacy `signals.confidence` base
        // so Path 1 and Path 2 agree at startup).
        let reg = Regulator::for_user("user_a");
        assert!((reg.confidence() - 0.5).abs() < 1e-9);
        assert_eq!(reg.logprob_coverage(), 0.0);
    }

    #[test]
    fn confident_token_stream_raises_confidence() {
        // Contract: a run of high-probability tokens (small magnitude
        // negative logprobs) pushes `confidence()` high.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart { user_message: "explain async".into() });
        for i in 0..15 {
            reg.on_event(LLMEvent::Token {
                token: "tok".into(),
                logprob: -0.2,
                index: i,
            });
        }
        assert!(
            reg.confidence() > 0.8,
            "confident token stream should drive confidence >0.8 (got {})",
            reg.confidence()
        );
        assert!((reg.logprob_coverage() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn gibberish_token_stream_lowers_confidence() {
        // Contract: a run of uncertain tokens (high negative-log-likelihood,
        // typical of an LLM responding to OOD / gibberish input) drives
        // `confidence()` into the low band. This is the Session 17
        // test-target scenario at the Regulator level.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "asdfkjh qwer zxcvb".into(),
        });
        for i in 0..15 {
            reg.on_event(LLMEvent::Token {
                token: "?".into(),
                logprob: -6.5,
                index: i,
            });
        }
        assert!(
            reg.confidence() < 0.2,
            "high-NLL stream should drive confidence <0.2 (got {})",
            reg.confidence()
        );
    }

    #[test]
    fn turn_start_resets_token_window() {
        // Contract: token statistics are per-turn. A new TurnStart
        // clears the previous turn's window so confidence doesn't drag
        // forward.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart { user_message: "q1".into() });
        for i in 0..10 {
            reg.on_event(LLMEvent::Token {
                token: "t".into(),
                logprob: -0.1,
                index: i,
            });
        }
        let confident = reg.confidence();
        assert!(confident > 0.8);

        // Next turn starts — window should be empty, confidence back
        // to neutral (no tokens observed yet, no response buffered).
        reg.on_event(LLMEvent::TurnStart { user_message: "q2".into() });
        assert!(
            (reg.confidence() - 0.5).abs() < 1e-9,
            "new turn should reset confidence to neutral (got {})",
            reg.confidence()
        );
    }

    #[test]
    fn unavailable_logprobs_fall_back_to_structural_signal() {
        // Contract: providers that don't expose logprobs pass 0.0 per
        // the `LLMEvent::Token` convention. Confidence then reads from
        // the buffered response text via the structural heuristic.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "how to refactor?".into(),
        });
        for i in 0..5 {
            reg.on_event(LLMEvent::Token {
                token: "t".into(),
                logprob: 0.0, // unavailable
                index: i,
            });
        }
        reg.on_event(LLMEvent::TurnComplete {
            full_response:
                "Here's the refactored function. It preserves the original signature."
                    .into(),
        });
        // No logprobs + clean long response → structural default (0.7).
        assert!(
            (reg.confidence() - 0.7).abs() < 0.01,
            "structural fallback on unremarkable response should be ~0.7 (got {})",
            reg.confidence()
        );
        assert_eq!(reg.logprob_coverage(), 0.0);
    }

    #[test]
    fn unavailable_logprobs_short_response_fallback_is_low() {
        // Contract: without logprobs + a short response (refusal
        // heuristic), structural fallback caps confidence in the low
        // band.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "can you do X?".into(),
        });
        reg.on_event(LLMEvent::Token {
            token: "No.".into(),
            logprob: 0.0,
            index: 0,
        });
        reg.on_event(LLMEvent::TurnComplete {
            full_response: "No.".into(),
        });
        assert!(
            reg.confidence() < 0.5,
            "short unremarkable response should fall in low band (got {})",
            reg.confidence()
        );
    }

    // ── Session 18: scope-drift wiring ─────────────────────────────────

    #[test]
    fn decide_emits_scope_drift_warn_on_plan_example() {
        // Contract: the Session 18 plan test target — task "refactor
        // function" vs response "add logging + error handling" —
        // produces a ScopeDriftWarn from `decide()`. This is the
        // end-to-end check that TurnStart + TurnComplete populate
        // scope state and `decide()` reads it correctly.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "refactor this function to be async".into(),
        });
        reg.on_event(LLMEvent::TurnComplete {
            full_response: "add logging and error handling".into(),
        });
        match reg.decide() {
            Decision::ScopeDriftWarn {
                drift_score,
                drift_tokens,
                task_tokens,
            } => {
                assert!(
                    drift_score > 0.3,
                    "plan target requires drift > 0.3 (got {drift_score})"
                );
                assert!(!drift_tokens.is_empty(), "drift_tokens must be populated");
                assert!(!task_tokens.is_empty(), "task_tokens must be populated");
            }
            other => panic!(
                "plan example must emit ScopeDriftWarn, got {other:?}"
            ),
        }
    }

    #[test]
    fn decide_continues_when_response_stays_on_task() {
        // Contract: when response keywords overlap the task
        // sufficiently (drift < threshold), `decide()` returns
        // Continue rather than firing a false-positive warning.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "refactor the async function".into(),
        });
        reg.on_event(LLMEvent::TurnComplete {
            full_response: "refactor async function".into(),
        });
        assert!(
            matches!(reg.decide(), Decision::Continue),
            "on-task response must not emit ScopeDriftWarn"
        );
    }

    #[test]
    fn decide_continues_before_turn_complete() {
        // Contract: scope drift requires both task and response
        // keywords. Before TurnComplete fires, drift_score is None and
        // decide() should emit Continue (not a false warning about an
        // unfinished turn).
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "refactor this function to be async".into(),
        });
        // No TurnComplete yet — scope has task but no response.
        assert!(matches!(reg.decide(), Decision::Continue));
    }

    #[test]
    fn turn_start_resets_scope_state() {
        // Contract: a new TurnStart clears stale response keywords from
        // the previous turn, so drift isn't computed across turn
        // boundaries. This mirrors the token_stats per-turn reset.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "refactor this function".into(),
        });
        reg.on_event(LLMEvent::TurnComplete {
            full_response: "add logging and telemetry".into(),
        });
        // First turn clearly drifts.
        assert!(matches!(reg.decide(), Decision::ScopeDriftWarn { .. }));

        // New turn starts — scope reset, response keywords cleared.
        // Before the new response arrives, decide() must return
        // Continue (stale response from previous turn must not leak).
        reg.on_event(LLMEvent::TurnStart {
            user_message: "explain tokio runtime".into(),
        });
        assert!(
            matches!(reg.decide(), Decision::Continue),
            "new turn must clear stale response before new response arrives"
        );
    }

    // ── Session 19: cost + circuit-break wiring ────────────────────────

    #[test]
    fn cost_event_accumulates_totals() {
        // Contract: LLMEvent::Cost folds into cumulative counters
        // accessible via the observability accessors.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::Cost {
            tokens_in: 100,
            tokens_out: 400,
            wallclock_ms: 2_000,
            provider: None,
        });
        reg.on_event(LLMEvent::Cost {
            tokens_in: 50,
            tokens_out: 200,
            wallclock_ms: 1_500,
            provider: Some("anthropic".into()),
        });
        assert_eq!(reg.total_tokens_out(), 600);
    }

    #[test]
    fn cost_event_feeds_track_cost_via_normalisation() {
        // Contract: Cost event triggers `session.track_cost(normalised)`
        // which depletes body_budget — the Path 1 ↔ Path 2 bridge
        // `docs/app-contract.md` describes.
        let mut reg = Regulator::for_user("user_a");
        let initial_budget = reg.session().world_model().body_budget;
        reg.on_event(LLMEvent::Cost {
            tokens_in: 0,
            tokens_out: cost::TYPICAL_TURN_TOKENS_OUT,
            wallclock_ms: cost::TYPICAL_TURN_WALLCLOCK_MS,
            provider: None,
        });
        let after_budget = reg.session().world_model().body_budget;
        assert!(
            after_budget < initial_budget,
            "Cost event must deplete body_budget via track_cost (before {initial_budget}, after {after_budget})"
        );
    }

    #[test]
    fn quality_feedback_records_in_cost_accumulator() {
        // Contract: QualityFeedback is tracked in cost_accumulator for
        // trend analysis independently of whether a pending response
        // was consolidated.
        let mut reg = Regulator::for_user("user_a");
        // Three QualityFeedback events without any TurnComplete — cost
        // accumulator should still record them.
        for q in [0.9, 0.8, 0.7] {
            reg.on_event(LLMEvent::QualityFeedback {
                quality: q,
                fragment_spans: None,
            });
        }
        // After three recorded values, trend queries work.
        // Mean of last 3 = 0.8. Delta = 0.9 - 0.7 = 0.2 > 0.15 default
        // min_delta — decline fires. But quality mean 0.8 > 0.5 poor
        // threshold, so CircuitBreak doesn't fire. decide() stays
        // Continue — but the accumulator recorded the quality.
        assert!(matches!(reg.decide(), Decision::Continue));
    }

    #[test]
    fn decide_emits_circuit_break_on_cost_cap_with_poor_quality() {
        // Contract: plan test target — when cumulative tokens exceed
        // cap AND recent quality is poor, CircuitBreak fires with
        // CostCapReached reason.
        let mut reg = Regulator::for_user("user_a").with_cost_cap(1_000);

        // Three turns of poor quality accompanied by cost accumulation.
        for _ in 0..3 {
            reg.on_event(LLMEvent::Cost {
                tokens_in: 0,
                tokens_out: 400,
                wallclock_ms: 1_000,
                provider: None,
            });
            reg.on_event(LLMEvent::QualityFeedback {
                quality: 0.3, // poor
                fragment_spans: None,
            });
        }
        // Cumulative tokens_out = 1200 > 1000 cap, quality mean = 0.3 < 0.5.
        match reg.decide() {
            Decision::CircuitBreak { reason, .. } => match reason {
                CircuitBreakReason::CostCapReached {
                    tokens_spent,
                    tokens_cap,
                    mean_quality_last_n,
                } => {
                    assert_eq!(tokens_spent, 1_200);
                    assert_eq!(tokens_cap, 1_000);
                    assert!(mean_quality_last_n < 0.5);
                }
                other => panic!("expected CostCapReached, got {other:?}"),
            },
            other => panic!("expected CircuitBreak, got {other:?}"),
        }
    }

    #[test]
    fn decide_does_not_emit_circuit_break_when_quality_recovers() {
        // Contract: cost cap alone isn't enough — quality must also be
        // poor. A high-quality agent that happens to use a lot of
        // tokens should not be halted.
        let mut reg = Regulator::for_user("user_a").with_cost_cap(1_000);

        for _ in 0..3 {
            reg.on_event(LLMEvent::Cost {
                tokens_in: 0,
                tokens_out: 400,
                wallclock_ms: 1_000,
                provider: None,
            });
            reg.on_event(LLMEvent::QualityFeedback {
                quality: 0.9, // high
                fragment_spans: None,
            });
        }
        assert!(
            matches!(reg.decide(), Decision::Continue),
            "high-quality agent at over-cap must not be halted"
        );
    }

    #[test]
    fn decide_emits_circuit_break_on_quality_decline_no_recovery() {
        // Contract: even under the cost cap, a sharp quality decline
        // (with mean dropping below poor threshold) triggers the
        // QualityDeclineNoRecovery variant.
        let mut reg = Regulator::for_user("user_a").with_cost_cap(u32::MAX);

        // Monotonically declining quality ending well below 0.5.
        for q in [0.8, 0.5, 0.3, 0.2, 0.15] {
            reg.on_event(LLMEvent::QualityFeedback {
                quality: q,
                fragment_spans: None,
            });
        }
        // Last 3 = [0.3, 0.2, 0.15]. Mean ≈ 0.22 < 0.5 (poor).
        // Delta = 0.3 - 0.15 = 0.15 ≥ MIN_DELTA → decline fires.
        match reg.decide() {
            Decision::CircuitBreak {
                reason:
                    CircuitBreakReason::QualityDeclineNoRecovery { turns, mean_delta },
                ..
            } => {
                assert_eq!(turns, cost::QUALITY_DECLINE_WINDOW);
                assert!(mean_delta >= cost::QUALITY_DECLINE_MIN_DELTA);
            }
            other => panic!("expected QualityDeclineNoRecovery, got {other:?}"),
        }
    }

    #[test]
    fn decide_priority_circuit_break_dominates_scope_drift() {
        // Contract (P10): when cost-cap CircuitBreak AND ScopeDriftWarn
        // both fire on the same turn, CircuitBreak wins — urgent stop
        // beats semantic warning.
        let mut reg = Regulator::for_user("user_a").with_cost_cap(500);

        // Set up a drifting turn.
        reg.on_event(LLMEvent::TurnStart {
            user_message: "refactor this function".into(),
        });
        reg.on_event(LLMEvent::TurnComplete {
            full_response: "add logging and error handling".into(),
        });

        // Drive the cost cap over with poor quality.
        for _ in 0..3 {
            reg.on_event(LLMEvent::Cost {
                tokens_in: 0,
                tokens_out: 400,
                wallclock_ms: 0,
                provider: None,
            });
            reg.on_event(LLMEvent::QualityFeedback {
                quality: 0.2,
                fragment_spans: None,
            });
        }

        // Both ScopeDriftWarn and CostCapReached would fire; priority
        // rule says CostCapReached wins.
        match reg.decide() {
            Decision::CircuitBreak {
                reason: CircuitBreakReason::CostCapReached { .. },
                ..
            } => {}
            other => panic!(
                "priority rule must emit CircuitBreak CostCapReached, got {other:?}"
            ),
        }
    }

    #[test]
    fn with_cost_cap_preserves_prior_accumulation() {
        // Contract: `with_cost_cap` only updates the cap. Counters
        // accumulated before the call survive — callers can tune the
        // cap mid-task without losing history.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::Cost {
            tokens_in: 100,
            tokens_out: 200,
            wallclock_ms: 500,
            provider: None,
        });
        assert_eq!(reg.total_tokens_out(), 200);

        let reg = reg.with_cost_cap(10_000);
        assert_eq!(reg.total_tokens_out(), 200, "cap change must not reset counters");
        assert_eq!(reg.cost_cap_tokens(), 10_000);
    }

    // ── Session 20: correction + procedural-warning wiring ─────────────

    /// Helper: drive three corrections on the same cluster through the
    /// Regulator. Used by several Session 20 tests.
    fn drive_three_corrections_on(reg: &mut Regulator, task_message: &str) {
        reg.on_event(LLMEvent::TurnStart {
            user_message: task_message.into(),
        });
        for msg in [
            "don't add logging",
            "stop adding logging please",
            "no more logs",
        ] {
            reg.on_event(LLMEvent::UserCorrection {
                correction_message: msg.into(),
                corrects_last: true,
            });
        }
    }

    #[test]
    fn user_correction_requires_corrects_last_true() {
        // Contract: UserCorrection with `corrects_last=false` is
        // treated as a new query (should go via TurnStart) and must
        // not attribute the text as a correction on the previous
        // cluster.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "refactor this function to be async".into(),
        });
        for _ in 0..5 {
            reg.on_event(LLMEvent::UserCorrection {
                correction_message: "something entirely different".into(),
                corrects_last: false,
            });
        }
        // No correction should have been recorded.
        assert!(matches!(reg.decide(), Decision::Continue));
    }

    #[test]
    fn user_correction_dropped_when_no_active_cluster() {
        // Contract: if TurnStart produced an empty cluster (no
        // extractable top-2 topics), UserCorrection is dropped — the
        // record has no cluster to attribute to.
        let mut reg = Regulator::for_user("user_a");
        // Short-message TurnStart with only stop-words — cluster is
        // empty.
        reg.on_event(LLMEvent::TurnStart {
            user_message: "is it ok?".into(),
        });
        for _ in 0..5 {
            reg.on_event(LLMEvent::UserCorrection {
                correction_message: "never do X".into(),
                corrects_last: true,
            });
        }
        assert!(matches!(reg.decide(), Decision::Continue));
    }

    #[test]
    fn decide_emits_procedural_warning_at_pattern_threshold() {
        // Plan test target: 3 similar corrections in different words
        // on the same topic → single CorrectionPattern extracted, and
        // `decide()` on a NEW turn against the same cluster emits
        // ProceduralWarning.
        let mut reg = Regulator::for_user("user_42");
        drive_three_corrections_on(&mut reg, "refactor this function to be async");

        // Simulate the app querying decide() before the next
        // generation — at this point current_topic_cluster still
        // reflects "refactor+async".
        match reg.decide() {
            Decision::ProceduralWarning { patterns } => {
                assert_eq!(patterns.len(), 1);
                let pattern = &patterns[0];
                assert_eq!(pattern.user_id, "user_42");
                assert_eq!(pattern.learned_from_turns, 3);
                // Opaque pattern name — no English rule extraction.
                assert!(pattern
                    .pattern_name
                    .starts_with("corrections_on_"));
                // Example corrections preserved, newest first.
                assert_eq!(pattern.example_corrections.len(), 3);
                assert_eq!(pattern.example_corrections[0], "no more logs");
            }
            other => panic!("expected ProceduralWarning, got {other:?}"),
        }
    }

    #[test]
    fn decide_does_not_emit_procedural_warning_below_threshold() {
        // Only 2 corrections — below MIN_CORRECTIONS_FOR_PATTERN.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "refactor this function to be async".into(),
        });
        for msg in ["don't add logging", "no more logs"] {
            reg.on_event(LLMEvent::UserCorrection {
                correction_message: msg.into(),
                corrects_last: true,
            });
        }
        assert!(matches!(reg.decide(), Decision::Continue));
    }

    #[test]
    fn procedural_warning_fires_only_on_matching_cluster() {
        // Contract: ProceduralWarning is cluster-scoped. Three
        // corrections on cluster A must not fire a warning when the
        // current turn is on cluster B.
        let mut reg = Regulator::for_user("user_a");
        drive_three_corrections_on(&mut reg, "refactor this function to be async");
        assert!(matches!(reg.decide(), Decision::ProceduralWarning { .. }));

        // New turn on an unrelated cluster — no warning.
        reg.on_event(LLMEvent::TurnStart {
            user_message: "explain docker containers to me".into(),
        });
        assert!(matches!(reg.decide(), Decision::Continue));
    }

    #[test]
    fn decide_priority_scope_drift_dominates_procedural_warning() {
        // Contract (P10): when BOTH ScopeDriftWarn and
        // ProceduralWarning would fire on the same turn, the semantic
        // warning wins over the historical-pattern advisory per the
        // priority order documented on `decide`.
        let mut reg = Regulator::for_user("user_a");

        // Build up three corrections on the refactor+async cluster.
        drive_three_corrections_on(&mut reg, "refactor this function to be async");

        // New turn on the same cluster but with a drifting response.
        reg.on_event(LLMEvent::TurnStart {
            user_message: "refactor this function to be async".into(),
        });
        reg.on_event(LLMEvent::TurnComplete {
            full_response: "add logging and error handling".into(),
        });
        // Both ScopeDriftWarn and ProceduralWarning would fire —
        // priority rule says ScopeDriftWarn wins.
        assert!(matches!(reg.decide(), Decision::ScopeDriftWarn { .. }));
    }

    #[test]
    fn export_includes_correction_patterns() {
        // Contract: Regulator::export surfaces every cluster that has
        // reached the pattern threshold in the exported
        // RegulatorState.correction_patterns map.
        let mut reg = Regulator::for_user("user_a");
        drive_three_corrections_on(&mut reg, "refactor this function to be async");

        let snapshot = reg.export();
        assert!(
            !snapshot.correction_patterns.is_empty(),
            "at-threshold cluster must appear in exported patterns"
        );
        let cluster_key = snapshot
            .correction_patterns
            .keys()
            .next()
            .expect("one pattern");
        let pattern = &snapshot.correction_patterns[cluster_key];
        assert_eq!(pattern.learned_from_turns, 3);
        assert_eq!(pattern.example_corrections.len(), 3);
    }

    #[test]
    fn import_restores_patterns_via_example_replay() {
        // Contract: Regulator::import reconstructs the pattern state
        // from persisted example_corrections so a restored regulator
        // surfaces the same ProceduralWarning as the original.
        let mut source = Regulator::for_user("user_persist");
        drive_three_corrections_on(&mut source, "refactor this function to be async");
        let snapshot = source.export();

        // JSON round-trip proves the snapshot is self-contained.
        let json = serde_json::to_string(&snapshot).expect("serialise");
        let decoded: RegulatorState =
            serde_json::from_str(&json).expect("deserialise");
        let mut restored = Regulator::import(decoded);

        // Seed the restored regulator with the same TurnStart so
        // current_topic_cluster matches the persisted pattern's
        // cluster.
        restored.on_event(LLMEvent::TurnStart {
            user_message: "refactor this function to be async".into(),
        });
        match restored.decide() {
            Decision::ProceduralWarning { patterns } => {
                assert_eq!(patterns.len(), 1);
                assert_eq!(patterns[0].learned_from_turns, 3);
                assert_eq!(patterns[0].example_corrections.len(), 3);
            }
            other => panic!("restored regulator must fire ProceduralWarning, got {other:?}"),
        }
    }

    #[test]
    fn legacy_snapshot_loads_without_correction_patterns() {
        // Contract: RegulatorState snapshots from Sessions 16–19 lack
        // the correction_patterns field. `#[serde(default)]` must
        // populate it with an empty map and the restored Regulator
        // behaves as if fresh.
        let legacy = r#"{
            "user_id": "legacy",
            "learned": {
                "gain_mode": "neutral",
                "tick": 0,
                "response_success": {},
                "response_strategies": {}
            }
        }"#;
        let state: RegulatorState =
            serde_json::from_str(legacy).expect("legacy snapshot must load");
        let reg = Regulator::import(state);
        assert_eq!(reg.user_id(), "legacy");
        assert!(matches!(reg.decide(), Decision::Continue));
    }
}
