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
pub mod otel;
pub mod scope;
pub mod state;
pub mod token_stats;
pub mod tools;

use serde::{Deserialize, Serialize};

use crate::session::CognitiveSession;

use self::correction::CorrectionStore;
use self::cost::{
    normalize_cost, CostAccumulator, POOR_QUALITY_MEAN, QUALITY_DECLINE_MIN_DELTA,
    QUALITY_DECLINE_WINDOW,
};
use self::scope::{ScopeTracker, DRIFT_WARN_THRESHOLD};
use self::token_stats::{confidence_with_fallback, TokenStatsAccumulator};
use self::tools::ToolStatsAccumulator;

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
///
/// `#[non_exhaustive]` — future sessions may add event variants (e.g. tool
/// calls, streaming deltas). Callers matching on `LLMEvent` must include a
/// wildcard arm.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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

    /// The agent invoked a tool. Emit once per tool call.
    ///
    /// Drives per-turn tool-loop detection (see
    /// [`CircuitBreakReason::RepeatedToolCallLoop`] and
    /// [`tools::TOOL_LOOP_THRESHOLD`]). `args_json` is opaque to the
    /// regulator — stored only for downstream observability by the
    /// app; pass `None` when you do not want to thread args through.
    ToolCall {
        tool_name: String,
        args_json: Option<String>,
    },

    /// The tool call just returned. Emit after `ToolCall` for each
    /// completed tool call. Carries latency + success signal for
    /// observability via [`Regulator::tool_total_duration_ms`] and
    /// [`Regulator::tool_failure_count`]. Does not influence
    /// [`Decision::CircuitBreak`] in 0.3.0 (loop detection uses
    /// `ToolCall` only); future sessions may wire failure-pattern
    /// detection.
    ToolResult {
        tool_name: String,
        success: bool,
        duration_ms: u64,
        /// Short description of the failure when `success == false`.
        /// Free-form — the regulator does not parse the text.
        error_summary: Option<String>,
    },
}

// ── Decisions ──────────────────────────────────────────────────────────

/// Regulatory decision returned by [`Regulator::decide`].
///
/// A single turn may warrant multiple concerns (drift AND low confidence).
/// v1 `decide()` returns a single `Decision`; callers may call `decide()`
/// repeatedly or branch on variant. Multi-concern aggregation is a Session
/// 19+ refinement.
///
/// `#[non_exhaustive]` — future sessions may add variants (e.g. a
/// multi-concern aggregator). Callers matching on `Decision` must include
/// a wildcard arm.
///
/// `#[must_use]` — `Regulator::decide()` produces a control-flow signal
/// the app is expected to act on. Dropping it on the floor is almost
/// always a bug.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
#[must_use]
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
///
/// `#[non_exhaustive]` — future sessions may add new circuit-break
/// reasons. Callers matching on `CircuitBreakReason` must include a
/// wildcard arm.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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
    /// The agent called the same tool [`tools::TOOL_LOOP_THRESHOLD`]+
    /// consecutive times in a single turn without interleaving a
    /// different tool or a reasoning step. Signature pattern of a
    /// runaway retry loop — halt before the cost spirals.
    RepeatedToolCallLoop {
        tool_name: String,
        consecutive_count: usize,
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
    /// Per-turn tool-call + tool-result history. Reset on `TurnStart`
    /// (tool loops are measured within a single turn). Drives
    /// [`CircuitBreakReason::RepeatedToolCallLoop`] emission in
    /// [`Self::decide`] via [`tools::ToolStatsAccumulator::detected_loop`].
    tools: ToolStatsAccumulator,
    /// Implicit-correction detection window. When set via
    /// [`Self::with_implicit_correction_window`], a `TurnStart` that
    /// arrives within this duration of the previous `TurnComplete` AND
    /// maps to the same topic cluster is treated as a retry — the
    /// regulator synthesises a correction record into
    /// [`CorrectionStore`] using the new `user_message` as the
    /// correction text. `None` disables the feature (default) — no
    /// behaviour change from pre-Session-32 versions.
    implicit_correction_window: Option<std::time::Duration>,
    /// Timestamp of the most recent `TurnComplete`, captured only when
    /// [`Self::implicit_correction_window`] is set. `Instant` is not
    /// `Serialize` and by design does NOT survive
    /// [`Self::export`] / [`Self::import`] — implicit-correction signal
    /// is ephemeral (retry-within-seconds never spans process
    /// restarts).
    last_turn_complete_at: Option<std::time::Instant>,
    /// Counter for implicit corrections synthesised since creation /
    /// last `import`. Exposed via
    /// [`Self::implicit_corrections_count`] for observability. Not
    /// persisted (per-process counter).
    implicit_corrections_count: usize,
    /// Drift-score threshold at or above which
    /// [`Decision::ScopeDriftWarn`] fires in [`Self::decide`]. Defaults
    /// to [`scope::DRIFT_WARN_THRESHOLD`] (0.5). Configurable via
    /// [`Self::with_scope_drift_threshold`]: verbose LLMs (e.g., Gemini)
    /// often exceed 0.5 even when on-topic, so the 2026-04-20 empirical
    /// finding calls for a higher cutoff — typical adjusted value 0.7.
    ///
    /// Not persisted by [`Self::export`] — it's a caller-driven
    /// configuration, not accumulated state.
    scope_drift_threshold: f64,
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
            tools: ToolStatsAccumulator::new(),
            implicit_correction_window: None,
            last_turn_complete_at: None,
            implicit_corrections_count: 0,
            scope_drift_threshold: DRIFT_WARN_THRESHOLD,
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

    /// Builder: enable implicit correction detection. A `TurnStart`
    /// arriving within `window` of the previous `TurnComplete` AND
    /// mapping to the same topic cluster is treated as a retry — the
    /// regulator synthesises a correction record (using the new
    /// `user_message` as the correction text) against the matching
    /// cluster.
    ///
    /// ## Why this exists
    ///
    /// Most chat UIs do NOT emit an explicit
    /// [`LLMEvent::UserCorrection`] when the user re-asks. They just
    /// submit another message. Without implicit detection, the
    /// [`Decision::ProceduralWarning`] path never activates for those
    /// apps — the strongest wedge of the regulator (procedural memory
    /// from corrections) stays inert.
    ///
    /// ## Heuristic
    ///
    /// Two independent signals must both hold:
    ///
    /// 1. **Temporal proximity** — retry within the configured window.
    ///    Typical value: 30–60 seconds. Longer windows catch more
    ///    deliberate "let me rephrase" edits; shorter windows keep
    ///    false-positive rate low.
    /// 2. **Topic continuity** — the new turn maps to the SAME
    ///    `current_topic_cluster`. Unrelated follow-up questions on a
    ///    different topic don't fire.
    ///
    /// Both gates are required. A 3-second retry on an unrelated topic
    /// is not a correction; a 3-minute retry on the same topic is
    /// probably not either.
    ///
    /// ## Interaction with explicit `UserCorrection`
    ///
    /// Implicit and explicit paths are additive — apps that use both
    /// will double-count corrections on retries where they also emit
    /// an explicit event. The expected integration pattern is
    /// **implicit OR explicit, not both**: choose implicit when the
    /// app has no native correction signal, explicit when the app can
    /// distinguish (e.g., a dedicated "this was wrong" button).
    ///
    /// ## Persistence
    ///
    /// The window setting itself is NOT persisted by
    /// [`Self::export`] — it's a caller-driven configuration, not
    /// accumulated state. After `import`, callers must re-apply
    /// `with_implicit_correction_window(...)` if they want the
    /// feature active in the restored regulator. The implicit
    /// counter also resets (per-process counter).
    pub fn with_implicit_correction_window(mut self, window: std::time::Duration) -> Self {
        self.implicit_correction_window = Some(window);
        self
    }

    /// Builder: override the default scope-drift threshold used by
    /// [`Decision::ScopeDriftWarn`] emission.
    ///
    /// ## Why this exists
    ///
    /// The default threshold [`scope::DRIFT_WARN_THRESHOLD`] (0.5) was
    /// tuned against the Session 18 hand-crafted audit (≤ 20% total
    /// error rate on 10 cases). The 2026-04-20 empirical Gemini smoke
    /// run surfaced a real-world false-positive regime: verbose
    /// pedagogical responses on technical prompts routinely score in
    /// the `[0.5, 0.7)` band even when the response is on-topic — the
    /// LLM is simply expanding the task keywords with peripheral
    /// vocabulary. Apps targeting verbose model families can raise the
    /// threshold to `0.7` to suppress the false-positive class without
    /// losing sensitivity to the truly-disjoint-vocabulary case (which
    /// scores ≥ 0.8).
    ///
    /// ## Guard
    ///
    /// Non-finite (`NaN` / `±inf`) and out-of-range values are silently
    /// dropped — the regulator keeps the previously-set (or default)
    /// threshold. This matches the fail-open pattern of the other
    /// boundary inputs in the crate (see `track_cost`,
    /// `record_quality`, `Token.logprob`).
    ///
    /// ## Persistence
    ///
    /// The threshold is NOT persisted by [`Self::export`] — it's a
    /// caller-driven configuration, not accumulated state. After
    /// `import`, callers must re-apply
    /// `with_scope_drift_threshold(...)` if they want a non-default
    /// value in the restored regulator.
    pub fn with_scope_drift_threshold(mut self, threshold: f64) -> Self {
        if threshold.is_finite() && (0.0..=1.0).contains(&threshold) {
            self.scope_drift_threshold = threshold;
        }
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
                // Capture the PREVIOUS cluster before we overwrite it —
                // needed for the implicit-correction check below.
                let previous_cluster = self.current_topic_cluster.clone();
                // Evaluate the temporal gate now (before `Instant::now()`
                // drifts further). `within_window` is `false` when the
                // feature is off OR no previous TurnComplete has landed.
                let within_window = match (
                    self.implicit_correction_window,
                    self.last_turn_complete_at,
                ) {
                    (Some(window), Some(last_complete_at)) => {
                        last_complete_at.elapsed() <= window
                    }
                    _ => false,
                };

                // Reset per-turn statistics before the new turn runs.
                // The logprob window is per-turn: confidence should not
                // drag forward from the previous turn's tokens.
                self.token_stats.begin_turn();
                // Tool-call history is per-turn too — a tool-call loop
                // is defined as consecutive same-tool calls within one
                // turn, so leaking tool history across turns would
                // conflate "legitimate retry on this turn" with
                // "stuck in a multi-turn loop" (which is a different
                // signal, not yet wired).
                self.tools.reset_turn();
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

                // Implicit correction gate: both temporal proximity AND
                // topic continuity must hold. Skip on empty cluster
                // (build_topic_cluster couldn't identify a topic —
                // attribution would be garbage) and on first-ever turn
                // (previous_cluster is empty, never matches).
                if within_window
                    && !self.current_topic_cluster.is_empty()
                    && self.current_topic_cluster == previous_cluster
                {
                    // The new `user_message` is the "correction text" —
                    // it's what the user said after the previous
                    // response didn't satisfy.
                    self.correction.record_correction(
                        &self.current_topic_cluster,
                        user_message.clone(),
                    );
                    self.implicit_corrections_count += 1;
                }

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
                // Stamp the implicit-correction timer only when the
                // feature is enabled. Calling `Instant::now()` is
                // cheap but we still gate on the `Option` so idle
                // regulators don't accrue syscalls they don't need.
                if self.implicit_correction_window.is_some() {
                    self.last_turn_complete_at = Some(std::time::Instant::now());
                }
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

            LLMEvent::ToolCall {
                tool_name,
                args_json,
            } => {
                // Record for loop detection + observability. The call
                // itself does not advance any cost / quality state —
                // the app's Cost event (or its own tool cost tracking)
                // remains the source of truth for token accounting.
                self.tools.record_call(tool_name, args_json);
            }

            LLMEvent::ToolResult {
                tool_name,
                success,
                duration_ms,
                error_summary,
            } => {
                // Observability only in 0.3.0. Future work: feed the
                // success flag into a failure-pattern detector that
                // raises a dedicated `RepeatedFailurePattern` subtype.
                self.tools
                    .record_result(tool_name, success, duration_ms, error_summary);
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

    // ── Tool-call observability (Path A, 0.3.0) ───────────────────────

    /// Total tool calls observed in the current turn. Reset on every
    /// `TurnStart`. Zero when no tool calls have been emitted yet.
    pub fn tool_total_calls(&self) -> usize {
        self.tools.total_calls()
    }

    /// Per-tool call counts in the current turn (e.g.
    /// `{"search": 3, "db.query": 1}`).
    pub fn tool_counts_by_name(&self) -> std::collections::HashMap<String, usize> {
        self.tools.counts_by_tool()
    }

    /// Sum of `duration_ms` across all `ToolResult` events in the
    /// current turn. Zero when no results have been emitted.
    pub fn tool_total_duration_ms(&self) -> u64 {
        self.tools.total_duration_ms()
    }

    /// Count of `ToolResult` events in the current turn with
    /// `success == false`.
    pub fn tool_failure_count(&self) -> usize {
        self.tools.failure_count()
    }

    /// Count of implicit corrections the regulator has synthesised
    /// since creation / last `import`. Zero when
    /// [`Self::with_implicit_correction_window`] has never been called.
    ///
    /// Useful for observability: surface "Noos auto-detected N user
    /// corrections on this session" alongside explicit-correction
    /// counts. The counter is per-process and resets on `import` (it
    /// is not persisted in `RegulatorState`).
    pub fn implicit_corrections_count(&self) -> usize {
        self.implicit_corrections_count
    }

    /// Current drift-score threshold used by [`Self::decide`] when
    /// deciding whether to emit [`Decision::ScopeDriftWarn`].
    ///
    /// Matches whatever [`Self::with_scope_drift_threshold`] last
    /// accepted, or [`scope::DRIFT_WARN_THRESHOLD`] (0.5) if the builder
    /// was never called. Not persisted in [`RegulatorState`].
    pub fn scope_drift_threshold(&self) -> f64 {
        self.scope_drift_threshold
    }

    /// One-call snapshot of every numeric observability signal the
    /// regulator exposes, as a map keyed by stable metric name. Drop
    /// straight into Prometheus / Datadog / StatsD pipelines without
    /// calling eight individual accessors.
    ///
    /// Keys (all `f64` — counts are widened from their native integer
    /// types so callers can feed a single type into metrics APIs):
    ///
    /// - `noos.confidence` — `[0, 1]`
    /// - `noos.logprob_coverage` — `[0, 1]`
    /// - `noos.total_tokens_out` — cumulative output tokens since last `import`
    /// - `noos.cost_cap_tokens` — current cap
    /// - `noos.tool_total_calls` — per-turn tool call count
    /// - `noos.tool_total_duration_ms` — per-turn tool wallclock sum
    /// - `noos.tool_failure_count` — per-turn tool failures
    /// - `noos.implicit_corrections_count` — per-process synthesised correction count
    /// - `noos.scope_drift_threshold` — current `[0, 1]` cutoff for [`Decision::ScopeDriftWarn`]
    ///
    /// Does NOT call [`Self::decide`] — the decision is a separate,
    /// explicit call point. Snapshots are cheap (adds up individual
    /// accessor calls); safe to sample every turn or every N turns.
    ///
    /// **CAUTION — common misuse**: a monitoring loop that polls
    /// `metrics_snapshot()` alone will NOT see `CircuitBreak`,
    /// `ScopeDriftWarn`, or `ProceduralWarning` events. Those require
    /// an explicit [`Self::decide`] call. Anti-pattern:
    /// ```ignore
    /// // WRONG — silently misses every decision the regulator makes
    /// loop {
    ///     let snapshot = r.metrics_snapshot();
    ///     send_to_prometheus(snapshot);  // decisions never emitted
    /// }
    /// ```
    /// Correct pattern: call `decide()` per turn, alert on the
    /// `Decision` variant; sample `metrics_snapshot()` independently
    /// for time-series metrics.
    ///
    /// Metric names are prefixed with `noos.` so they don't collide
    /// with your app's own namespace in a shared metrics system.
    pub fn metrics_snapshot(&self) -> std::collections::HashMap<String, f64> {
        let mut m = std::collections::HashMap::new();
        m.insert("noos.confidence".into(), self.confidence());
        m.insert("noos.logprob_coverage".into(), self.logprob_coverage());
        m.insert(
            "noos.total_tokens_out".into(),
            self.total_tokens_out() as f64,
        );
        m.insert(
            "noos.cost_cap_tokens".into(),
            self.cost_cap_tokens() as f64,
        );
        m.insert(
            "noos.tool_total_calls".into(),
            self.tool_total_calls() as f64,
        );
        m.insert(
            "noos.tool_total_duration_ms".into(),
            self.tool_total_duration_ms() as f64,
        );
        m.insert(
            "noos.tool_failure_count".into(),
            self.tool_failure_count() as f64,
        );
        m.insert(
            "noos.implicit_corrections_count".into(),
            self.implicit_corrections_count() as f64,
        );
        m.insert(
            "noos.scope_drift_threshold".into(),
            self.scope_drift_threshold,
        );
        m
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
    /// 3. [`Decision::CircuitBreak`] with
    ///    [`CircuitBreakReason::RepeatedToolCallLoop`] — agent called
    ///    the same tool [`tools::TOOL_LOOP_THRESHOLD`]+ consecutive
    ///    times within a single turn. Signature of a runaway retry
    ///    loop — halt before the cost spirals.
    /// 4. [`Decision::ScopeDriftWarn`] — response keywords disjoint
    ///    from task keywords (semantic warning, not a stop).
    /// 5. [`Decision::ProceduralWarning`] — the current topic cluster
    ///    has a learned [`CorrectionPattern`] from repeated past user
    ///    corrections (advisory: app / LLM should consult the
    ///    `example_corrections` before generating).
    /// 6. [`Decision::Continue`] — no fired predicates.
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

        // ── 3. Tool-call loop circuit break ──
        // Slots between the quality-based circuit breaks above and the
        // semantic ScopeDriftWarn below. Same rationale as 1/2: the
        // agent is wasting cost in a way that will not self-correct
        // without intervention — halt rather than just warn.
        if let Some((tool_name, consecutive_count)) = self.tools.detected_loop() {
            return Decision::CircuitBreak {
                reason: CircuitBreakReason::RepeatedToolCallLoop {
                    tool_name,
                    consecutive_count,
                },
                suggestion: "Agent repeatedly called the same tool without progress. \
                             Break the loop: re-prompt with a different approach, \
                             escalate to the user, or mark the task unresolved."
                    .into(),
            };
        }

        // ── 4. Scope-drift warning ──
        if let Some(drift) = self.scope.drift_score() {
            if drift >= self.scope_drift_threshold {
                return Decision::ScopeDriftWarn {
                    drift_tokens: self.scope.drift_tokens(),
                    drift_score: drift,
                    task_tokens: self.scope.task_tokens().to_vec(),
                };
            }
        }

        // ── 5. Procedural-correction warning ──
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

        // ── 6. Fall through ──
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
        //
        // `example_corrections` is stored newest-first, but
        // `record_correction` appends to the end of the internal vec
        // (so the vec is oldest-first). Iterate in reverse so the
        // replay order is oldest→newest, matching the store shape
        // `pattern_for` originally saw. Without the `.rev()`, a single
        // export→import roundtrip would silently invert the ordering
        // reported by subsequent `pattern_for` calls.
        let mut correction = CorrectionStore::new();
        for (cluster, pattern) in &state.correction_patterns {
            for text in pattern.example_corrections.iter().rev() {
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
            tools: ToolStatsAccumulator::new(),
            // Implicit-correction config + timer + counter are
            // per-process — callers re-apply `with_implicit_correction_window`
            // if they want the feature on in the restored regulator.
            implicit_correction_window: None,
            last_turn_complete_at: None,
            implicit_corrections_count: 0,
            // Scope-drift threshold is a caller-driven configuration too;
            // callers re-apply `with_scope_drift_threshold` if they want a
            // non-default cutoff.
            scope_drift_threshold: DRIFT_WARN_THRESHOLD,
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

    // ── Procedural pattern injection (Path B, 0.2.2) ──
    //
    // Before 0.2.2, apps had to hand-thread `ProceduralWarning`
    // `example_corrections` into the next prompt themselves (see the
    // recipe in regulator-guide.md §3.4). That boilerplate is the same
    // in every integration — these helpers lift it into the crate so
    // the app layer shrinks to a single method call.

    /// Return a prelude text block if the current turn has a learned
    /// correction pattern, otherwise `None`.
    ///
    /// Call after `LLMEvent::TurnStart` and before invoking the LLM.
    /// The return value is suitable for prepending to the user's prompt
    /// (or placing in a system message) so the model can adjust generation
    /// *before* producing tokens that would repeat a past mistake.
    ///
    /// Format (3 example corrections shown):
    /// ```text
    /// User has previously corrected responses on this topic with:
    /// - <most recent correction>
    /// - <second most recent>
    /// - <third most recent>
    /// ```
    ///
    /// When the current cluster has multiple patterns (rare — one
    /// cluster usually maps to one pattern), all `example_corrections`
    /// are concatenated in pattern order.
    ///
    /// Returns `None` when `decide()` is not `ProceduralWarning`, which
    /// also covers the cases where a higher-priority decision (cost-cap,
    /// scope-drift) would have intercepted the turn. In those cases the
    /// app should handle the higher-priority decision before considering
    /// prompt injection.
    #[must_use]
    pub fn corrections_prelude(&self) -> Option<String> {
        let patterns = match self.decide() {
            Decision::ProceduralWarning { patterns } => patterns,
            _ => return None,
        };
        let lines: Vec<String> = patterns
            .iter()
            .flat_map(|p| &p.example_corrections)
            .map(|ex| format!("- {ex}"))
            .collect();
        if lines.is_empty() {
            None
        } else {
            Some(format!(
                "User has previously corrected responses on this topic with:\n{}",
                lines.join("\n")
            ))
        }
    }

    /// Convenience wrapper around [`Regulator::corrections_prelude`]: given
    /// the user's raw prompt, return it with the correction prelude
    /// prepended, or return the prompt unchanged if no pattern applies.
    ///
    /// Call after `LLMEvent::TurnStart` and before invoking the LLM.
    ///
    /// Equivalent to:
    /// ```ignore
    /// match regulator.corrections_prelude() {
    ///     Some(prelude) => format!("{prelude}\n\nCurrent request: {user_prompt}"),
    ///     None => user_prompt.to_string(),
    /// }
    /// ```
    ///
    /// For custom templating (different header, system-message placement,
    /// multi-turn conversation formats) use [`Regulator::corrections_prelude`]
    /// directly and splice the returned block into your own prompt layout.
    #[must_use]
    pub fn inject_corrections(&self, user_prompt: &str) -> String {
        match self.corrections_prelude() {
            Some(prelude) => format!("{prelude}\n\nCurrent request: {user_prompt}"),
            None => user_prompt.to_string(),
        }
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

    // ── Path B: procedural pattern INJECTION (0.2.2) ──────────────────

    #[test]
    fn corrections_prelude_none_when_no_pattern() {
        // Contract: with no prior corrections the helper returns None so
        // the caller doesn't prepend an empty header.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "refactor this function to be async".into(),
        });
        assert!(reg.corrections_prelude().is_none());
    }

    #[test]
    fn corrections_prelude_returns_formatted_block_after_threshold() {
        let mut reg = Regulator::for_user("user_a");
        drive_three_corrections_on(&mut reg, "refactor this function to be async");
        // Same cluster, new turn — ProceduralWarning should fire
        // pre-generation (no TurnComplete yet).
        // Same cluster (keyword-hash of `refactor + async`) — matches
        // the drive_three_corrections_on seed message so the stored
        // pattern applies. Changing any topic-bearing word would hash
        // to a different cluster and the pattern would not fire.
        reg.on_event(LLMEvent::TurnStart {
            user_message: "refactor this function to be async".into(),
        });

        let prelude = reg.corrections_prelude().expect("pattern should apply");
        assert!(
            prelude.starts_with("User has previously corrected responses on this topic with:"),
            "unexpected header: {prelude}"
        );
        // All three example_corrections should appear as bulleted lines.
        for expected in &["no more logs", "stop adding logging please", "don't add logging"] {
            assert!(
                prelude.contains(&format!("- {expected}")),
                "prelude missing correction {expected:?}: {prelude}"
            );
        }
    }

    #[test]
    fn inject_corrections_returns_unchanged_when_no_pattern() {
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "refactor this function to be async".into(),
        });
        let prompt = "write me a hello world";
        assert_eq!(reg.inject_corrections(prompt), prompt);
    }

    #[test]
    fn inject_corrections_wraps_prompt_after_threshold() {
        let mut reg = Regulator::for_user("user_a");
        drive_three_corrections_on(&mut reg, "refactor this function to be async");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "refactor this function to be async".into(),
        });

        let prompt = "refactor this function to be async";
        let injected = reg.inject_corrections(prompt);
        assert!(
            injected.contains("User has previously corrected responses"),
            "injected prompt missing prelude header"
        );
        assert!(
            injected.contains(&format!("Current request: {prompt}")),
            "injected prompt missing current-request marker"
        );
        assert!(injected.len() > prompt.len(), "expected expansion");
    }

    // ── Path A: tool-call observation channel (0.3.0) ─────────────────

    #[test]
    fn tool_call_event_accumulates_stats() {
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "do a search".into(),
        });
        for _ in 0..3 {
            reg.on_event(LLMEvent::ToolCall {
                tool_name: "search".into(),
                args_json: None,
            });
        }
        assert_eq!(reg.tool_total_calls(), 3);
        let counts = reg.tool_counts_by_name();
        assert_eq!(counts.get("search"), Some(&3));
    }

    #[test]
    fn tool_call_loop_fires_circuit_break() {
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "do a search".into(),
        });
        // 5 consecutive same-tool calls = threshold
        for _ in 0..5 {
            reg.on_event(LLMEvent::ToolCall {
                tool_name: "search".into(),
                args_json: None,
            });
        }
        match reg.decide() {
            Decision::CircuitBreak {
                reason:
                    CircuitBreakReason::RepeatedToolCallLoop {
                        tool_name,
                        consecutive_count,
                    },
                ..
            } => {
                assert_eq!(tool_name, "search");
                assert_eq!(consecutive_count, 5);
            }
            other => panic!("expected RepeatedToolCallLoop, got {other:?}"),
        }
    }

    #[test]
    fn tool_call_interleaved_does_not_fire_loop() {
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "do a search".into(),
        });
        // 4 search + 1 db.query + 4 search — trailing run of 4 < 5
        for _ in 0..4 {
            reg.on_event(LLMEvent::ToolCall {
                tool_name: "search".into(),
                args_json: None,
            });
        }
        reg.on_event(LLMEvent::ToolCall {
            tool_name: "db.query".into(),
            args_json: None,
        });
        for _ in 0..4 {
            reg.on_event(LLMEvent::ToolCall {
                tool_name: "search".into(),
                args_json: None,
            });
        }
        assert!(matches!(reg.decide(), Decision::Continue));
    }

    #[test]
    fn turn_start_resets_tool_stats() {
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "turn 1".into(),
        });
        for _ in 0..5 {
            reg.on_event(LLMEvent::ToolCall {
                tool_name: "search".into(),
                args_json: None,
            });
        }
        // Loop would fire at this point.
        assert!(matches!(reg.decide(), Decision::CircuitBreak { .. }));
        // New turn resets tool stats — no loop in turn 2.
        reg.on_event(LLMEvent::TurnStart {
            user_message: "turn 2".into(),
        });
        assert_eq!(reg.tool_total_calls(), 0);
        assert!(!matches!(reg.decide(), Decision::CircuitBreak { .. }));
    }

    #[test]
    fn tool_result_observability_counters() {
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "do a search".into(),
        });
        reg.on_event(LLMEvent::ToolResult {
            tool_name: "search".into(),
            success: true,
            duration_ms: 120,
            error_summary: None,
        });
        reg.on_event(LLMEvent::ToolResult {
            tool_name: "db.query".into(),
            success: false,
            duration_ms: 250,
            error_summary: Some("timeout".into()),
        });
        assert_eq!(reg.tool_total_duration_ms(), 370);
        assert_eq!(reg.tool_failure_count(), 1);
    }

    #[test]
    fn tool_loop_has_higher_priority_than_scope_drift() {
        // Contract (P10): tool-call loop is a CircuitBreak and
        // dominates the semantic ScopeDriftWarn.
        let mut reg = Regulator::for_user("user_a");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "refactor this async function".into(),
        });
        for _ in 0..5 {
            reg.on_event(LLMEvent::ToolCall {
                tool_name: "search".into(),
                args_json: None,
            });
        }
        // Produce a drift-triggering response too.
        reg.on_event(LLMEvent::TurnComplete {
            full_response: "added logging database migration new schema totally unrelated".into(),
        });
        assert!(matches!(
            reg.decide(),
            Decision::CircuitBreak {
                reason: CircuitBreakReason::RepeatedToolCallLoop { .. },
                ..
            }
        ));
    }

    #[test]
    fn corrections_prelude_none_when_higher_priority_decision_dominates() {
        // Higher-priority decisions (CircuitBreak, ScopeDriftWarn)
        // suppress pattern injection. The app is expected to handle
        // those first — returning None here keeps the API contract
        // honest ("injection only when decide() is ProceduralWarning").
        // We use ScopeDriftWarn as the dominator because it requires
        // only one turn of setup (CostCap needs 3+ quality samples to
        // fill the QUALITY_DECLINE_WINDOW mean).
        let mut reg = Regulator::for_user("user_a");
        drive_three_corrections_on(&mut reg, "refactor this function to be async");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "refactor this function to be async".into(),
        });
        reg.on_event(LLMEvent::TurnComplete {
            full_response: "add logging plus database migration new schema".into(),
        });
        assert!(matches!(reg.decide(), Decision::ScopeDriftWarn { .. }));
        assert!(reg.corrections_prelude().is_none());
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
    fn import_preserves_example_corrections_order() {
        // Contract: `example_corrections` is documented newest-first.
        // An export → JSON round-trip → import cycle must preserve that
        // ordering so the value at `example_corrections[0]` stays
        // the newest correction before and after persistence.
        // Regression guard for a pre-0.1.1 bug where a single roundtrip
        // inverted the order.
        let mut source = Regulator::for_user("user_order");
        drive_three_corrections_on(&mut source, "refactor this function to be async");
        let snapshot_before = source.export();
        let (cluster, pattern_before) = snapshot_before
            .correction_patterns
            .iter()
            .next()
            .expect("one pattern after 3 corrections");
        let before = pattern_before.example_corrections.clone();

        let json = serde_json::to_string(&snapshot_before).expect("serialise");
        let decoded: RegulatorState =
            serde_json::from_str(&json).expect("deserialise");
        let restored = Regulator::import(decoded);
        let snapshot_after = restored.export();
        let pattern_after = snapshot_after
            .correction_patterns
            .get(cluster)
            .expect("pattern restored under same cluster key");
        assert_eq!(pattern_after.example_corrections, before);
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

    // ── Implicit correction detection (Session 32) ─────────────────
    //
    // These tests use real `Instant` / `std::thread::sleep` with short
    // windows (50–200 ms). Rust's test harness runs them concurrently
    // by default so the sleeps overlap, keeping wall-clock cost low.

    #[test]
    fn implicit_correction_off_by_default_no_synthetic_records() {
        // Default regulator has no implicit correction window — a
        // fast same-cluster retry should NOT record a correction.
        let mut r = Regulator::for_user("u");
        r.on_event(LLMEvent::TurnStart {
            user_message: "Refactor fetch_user to be async".into(),
        });
        r.on_event(LLMEvent::TurnComplete {
            full_response: "resp".into(),
        });
        // Re-ask immediately — no window configured, no correction.
        r.on_event(LLMEvent::TurnStart {
            user_message: "Fix the fetch_user async refactoring".into(),
        });
        assert_eq!(r.implicit_corrections_count(), 0);
    }

    #[test]
    fn implicit_correction_fires_on_fast_same_cluster_retry() {
        let mut r = Regulator::for_user("u")
            .with_implicit_correction_window(std::time::Duration::from_millis(500));

        r.on_event(LLMEvent::TurnStart {
            user_message: "Refactor fetch_user to be async".into(),
        });
        r.on_event(LLMEvent::TurnComplete {
            full_response: "(unsatisfactory response)".into(),
        });

        // User re-asks on same topic within 20ms — well under the
        // 500ms window.
        std::thread::sleep(std::time::Duration::from_millis(20));
        r.on_event(LLMEvent::TurnStart {
            user_message: "Fix the fetch_user async refactoring".into(),
        });

        assert_eq!(r.implicit_corrections_count(), 1);
        // Pattern threshold is 3 — this one correction won't yet fire
        // ProceduralWarning, but the count should reflect it.
    }

    #[test]
    fn implicit_correction_skips_when_window_expires() {
        // Window is 50ms; we wait 150ms before the retry. Should NOT
        // fire.
        let mut r = Regulator::for_user("u")
            .with_implicit_correction_window(std::time::Duration::from_millis(50));

        r.on_event(LLMEvent::TurnStart {
            user_message: "Refactor fetch_user to be async".into(),
        });
        r.on_event(LLMEvent::TurnComplete {
            full_response: "resp".into(),
        });

        std::thread::sleep(std::time::Duration::from_millis(150));
        r.on_event(LLMEvent::TurnStart {
            user_message: "Fix the fetch_user async refactoring".into(),
        });

        assert_eq!(
            r.implicit_corrections_count(),
            0,
            "retry outside window must not synthesise a correction"
        );
    }

    #[test]
    fn implicit_correction_skips_on_different_cluster() {
        // Fast retry but on a different topic → not a correction.
        let mut r = Regulator::for_user("u")
            .with_implicit_correction_window(std::time::Duration::from_millis(500));

        r.on_event(LLMEvent::TurnStart {
            user_message: "Refactor fetch_user to be async".into(),
        });
        r.on_event(LLMEvent::TurnComplete {
            full_response: "resp".into(),
        });
        std::thread::sleep(std::time::Duration::from_millis(10));
        r.on_event(LLMEvent::TurnStart {
            user_message: "Explain tokio channels and scheduling".into(),
        });

        assert_eq!(
            r.implicit_corrections_count(),
            0,
            "different-topic follow-up must not count as a correction"
        );
    }

    #[test]
    fn implicit_correction_skips_on_first_ever_turn() {
        // No previous TurnComplete — last_turn_complete_at is None,
        // temporal gate should fail closed.
        let mut r = Regulator::for_user("u")
            .with_implicit_correction_window(std::time::Duration::from_millis(500));
        r.on_event(LLMEvent::TurnStart {
            user_message: "Refactor fetch_user to be async".into(),
        });
        assert_eq!(r.implicit_corrections_count(), 0);
    }

    #[test]
    fn implicit_correction_skips_when_cluster_is_empty() {
        // A message with no extractable top-2 topics has an empty
        // cluster — attribution would be garbage, must fail closed.
        let mut r = Regulator::for_user("u")
            .with_implicit_correction_window(std::time::Duration::from_millis(500));

        // "ok" and "hi" are both stop-words-like (short) — produce
        // empty cluster.
        r.on_event(LLMEvent::TurnStart {
            user_message: "ok hi".into(),
        });
        r.on_event(LLMEvent::TurnComplete {
            full_response: "greetings".into(),
        });
        std::thread::sleep(std::time::Duration::from_millis(10));
        r.on_event(LLMEvent::TurnStart {
            user_message: "hi ok".into(),
        });

        // Both clusters are empty (or at least the new one is) → skip.
        assert_eq!(r.implicit_corrections_count(), 0);
    }

    #[test]
    fn implicit_correction_accumulates_to_pattern_at_threshold() {
        // Three fast same-cluster retries → after the third TurnStart
        // the correction count on the cluster hits
        // MIN_CORRECTIONS_FOR_PATTERN, and a subsequent `decide()`
        // called pre-generation (before TurnComplete) should fire
        // ProceduralWarning.
        //
        // Message choice: every retry must produce the SAME top-2
        // alphabetical meaningful words so `build_topic_cluster`
        // returns the same key. Cluster for all four is
        // "async+fetch_user" (async < fetch_user alphabetically, and
        // neither is beaten by any other meaningful word in these
        // messages — stop-word filter drops "to"/"be"/"the"/"for").
        let mut r = Regulator::for_user("u")
            .with_implicit_correction_window(std::time::Duration::from_millis(500));

        // Turn 1 — no prior complete, no implicit correction.
        r.on_event(LLMEvent::TurnStart {
            user_message: "Refactor fetch_user to be async".into(),
        });
        r.on_event(LLMEvent::TurnComplete {
            full_response: "try 1".into(),
        });

        // Turn 2 — first implicit correction.
        // {fix, fetch_user, async, refactoring} → top-2 = {async, fetch_user}
        std::thread::sleep(std::time::Duration::from_millis(10));
        r.on_event(LLMEvent::TurnStart {
            user_message: "Fix the fetch_user async refactoring".into(),
        });
        r.on_event(LLMEvent::TurnComplete {
            full_response: "try 2".into(),
        });

        // Turn 3 — second implicit correction.
        // {make, fetch_user, async, properly} → top-2 = {async, fetch_user}
        std::thread::sleep(std::time::Duration::from_millis(10));
        r.on_event(LLMEvent::TurnStart {
            user_message: "Make fetch_user async properly".into(),
        });
        r.on_event(LLMEvent::TurnComplete {
            full_response: "try 3".into(),
        });

        // Turn 4 — third implicit correction. This brings the stored
        // correction count on cluster "async+fetch_user" to 3 =
        // MIN_CORRECTIONS_FOR_PATTERN.
        // {update, fetch_user, async, version} → top-2 = {async, fetch_user}
        std::thread::sleep(std::time::Duration::from_millis(10));
        r.on_event(LLMEvent::TurnStart {
            user_message: "Update fetch_user to async version".into(),
        });
        assert_eq!(
            r.implicit_corrections_count(),
            3,
            "3 fast same-cluster retries → 3 synthesised corrections"
        );

        // `decide()` called BEFORE TurnComplete (pre-generation probe)
        // must fire ProceduralWarning — this is the integration the
        // feature unlocks.
        let d = r.decide();
        match d {
            Decision::ProceduralWarning { patterns } => {
                assert!(!patterns.is_empty());
                assert_eq!(patterns[0].learned_from_turns, 3);
            }
            other => panic!(
                "expected ProceduralWarning after 3 implicit corrections, got {other:?}"
            ),
        }
    }

    #[test]
    fn metrics_snapshot_exposes_stable_keys() {
        // Contract: every documented key is present, prefixed with
        // `noos.`, and the value type is f64. Integrators rely on
        // stable key names to route metrics into observability
        // systems — this test pins the contract.
        let r = Regulator::for_user("m").with_cost_cap(5_000);
        let snap = r.metrics_snapshot();
        for key in [
            "noos.confidence",
            "noos.logprob_coverage",
            "noos.total_tokens_out",
            "noos.cost_cap_tokens",
            "noos.tool_total_calls",
            "noos.tool_total_duration_ms",
            "noos.tool_failure_count",
            "noos.implicit_corrections_count",
            "noos.scope_drift_threshold",
        ] {
            assert!(snap.contains_key(key), "missing metric key {key:?}");
            let v = snap[key];
            assert!(v.is_finite(), "metric {key} produced non-finite {v}");
        }
        // cost_cap_tokens should reflect the builder setting.
        assert!((snap["noos.cost_cap_tokens"] - 5_000.0).abs() < 1e-9);
    }

    #[test]
    fn scope_drift_threshold_defaults_to_constant() {
        // Contract: a fresh regulator uses the crate-default
        // `DRIFT_WARN_THRESHOLD`. Exposed both via the accessor and via
        // metrics_snapshot so observability tooling can alert on drift.
        let r = Regulator::for_user("u");
        assert!((r.scope_drift_threshold() - DRIFT_WARN_THRESHOLD).abs() < f64::EPSILON);
        let snap = r.metrics_snapshot();
        assert!(
            (snap["noos.scope_drift_threshold"] - DRIFT_WARN_THRESHOLD).abs() < f64::EPSILON
        );
    }

    #[test]
    fn scope_drift_threshold_builder_accepts_in_range() {
        // 2026-04-20 empirical finding: verbose LLMs often score in
        // [0.5, 0.7) on on-topic responses. Bumping the threshold to
        // 0.7 suppresses that FP class.
        let r = Regulator::for_user("u").with_scope_drift_threshold(0.7);
        assert!((r.scope_drift_threshold() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn scope_drift_threshold_builder_rejects_invalid() {
        // Contract: NaN, ±inf, or out-of-range inputs are dropped
        // silently — the default is preserved. Fail-open mirrors the
        // other boundary inputs in the crate (track_cost,
        // record_quality, Token.logprob).
        let r = Regulator::for_user("u")
            .with_scope_drift_threshold(f64::NAN)
            .with_scope_drift_threshold(f64::INFINITY)
            .with_scope_drift_threshold(-0.1)
            .with_scope_drift_threshold(1.1);
        assert!(
            (r.scope_drift_threshold() - DRIFT_WARN_THRESHOLD).abs() < f64::EPSILON,
            "invalid inputs must leave the default in place"
        );
    }

    #[test]
    fn scope_drift_threshold_raises_suppresses_borderline_drift() {
        // Contract: a borderline-drift case scoring in [0.5, 0.9)
        // fires under the default 0.5 threshold but is suppressed by a
        // 0.9 threshold. Exercised to make sure the builder actually
        // feeds into the decide() gate.
        //
        // Task keywords: {async, explain, rust, runtime, tokio}
        // Response keywords (top-10 alphabetical):
        //   {async, driven, event, rust, runtime, scheduling, tasks,
        //    timers, tokio}
        // Anchored: {async, rust, runtime, tokio} → 4 of 9
        // Drift = 5/9 ≈ 0.556 (in range).
        let mut r_default = Regulator::for_user("u");
        let mut r_lax = Regulator::for_user("u").with_scope_drift_threshold(0.9);

        for r in [&mut r_default, &mut r_lax] {
            r.on_event(LLMEvent::TurnStart {
                user_message: "explain tokio runtime rust async".into(),
            });
            r.on_event(LLMEvent::TurnComplete {
                full_response: "Tokio is async Rust runtime for scheduling \
                    event driven tasks and timers".into(),
            });
        }

        // Default threshold: the response scores ≥ 0.5 → fires.
        assert!(
            matches!(r_default.decide(), Decision::ScopeDriftWarn { .. }),
            "default 0.5 threshold should fire on the borderline case"
        );
        // Lax threshold (0.9): same response passes as Continue.
        assert!(
            matches!(r_lax.decide(), Decision::Continue),
            "threshold 0.9 should suppress the same borderline case"
        );
    }

    #[test]
    fn metrics_snapshot_tracks_state_changes() {
        // Snapshot values should move with accumulated events —
        // covers the common observability pattern "sample before and
        // after a turn, diff the counters."
        let mut r = Regulator::for_user("m").with_cost_cap(5_000);
        let before = r.metrics_snapshot();
        assert!((before["noos.total_tokens_out"] - 0.0).abs() < 1e-9);

        r.on_event(LLMEvent::TurnStart {
            user_message: "hello".into(),
        });
        r.on_event(LLMEvent::Cost {
            tokens_in: 10,
            tokens_out: 150,
            wallclock_ms: 500,
            provider: None,
        });
        let after = r.metrics_snapshot();
        assert!((after["noos.total_tokens_out"] - 150.0).abs() < 1e-9);
    }

    #[test]
    fn implicit_counter_not_persisted_across_import() {
        // The implicit counter is a per-process observability number,
        // not part of the snapshot. Round-trip a regulator and confirm
        // the restored counter starts at zero (while the correction
        // records themselves DO survive).
        let mut r = Regulator::for_user("u")
            .with_implicit_correction_window(std::time::Duration::from_millis(500));

        r.on_event(LLMEvent::TurnStart {
            user_message: "Refactor fetch_user to be async".into(),
        });
        r.on_event(LLMEvent::TurnComplete {
            full_response: "r1".into(),
        });
        std::thread::sleep(std::time::Duration::from_millis(10));
        // Same cluster: {fix, fetch_user, async, refactoring} → top-2
        // = {async, fetch_user}. Matches prior turn.
        r.on_event(LLMEvent::TurnStart {
            user_message: "Fix the fetch_user async refactoring".into(),
        });
        assert_eq!(r.implicit_corrections_count(), 1);

        let snapshot = r.export();
        let restored = Regulator::import(snapshot);

        // Counter resets; the feature must be re-enabled by the
        // caller via `with_implicit_correction_window` if wanted.
        assert_eq!(
            restored.implicit_corrections_count(),
            0,
            "counter is per-process; not in RegulatorState"
        );
    }
}
