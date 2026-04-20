//! Node.js / TypeScript bindings for the Noos LLM-agent reliability
//! crate, via napi-rs.
//!
//! Exposes a minimal, TS-friendly API mirroring the Rust `Regulator`:
//!
//! ```typescript
//! import { Regulator, LLMEvent } from '@triangle-technology/noos'
//!
//! const r = Regulator.forUser('alice')
//! r.withCostCap(2_000)
//!
//! r.onEvent(LLMEvent.turnStart('Refactor fetch_user to be async'))
//! // ... call your LLM ...
//! r.onEvent(LLMEvent.turnComplete(responseText))
//! r.onEvent(LLMEvent.cost(25, 800, 500, 'anthropic'))
//!
//! const decision = r.decide()
//! switch (decision.kind) {
//!     case 'scope_drift_warn':
//!         console.log(`drift ${decision.driftScore}`)
//!         break
//!     case 'circuit_break':
//!         console.log(`halt: ${decision.suggestion} (${decision.reason?.kind})`)
//!         break
//!     case 'procedural_warning':
//!         decision.patterns?.forEach(p => console.log(p.patternName))
//!         break
//! }
//! ```
//!
//! Design notes (parallel to the Python binding):
//!
//! - **`.kind: string`** + variant-specific getters that return
//!   `T | null` when the variant doesn't carry that field. napi-rs
//!   maps `Option<T>` to `T | null` in generated `.d.ts`, which
//!   matches modern TS nullability patterns.
//! - **No builder chaining** across JS boundary (napi-rs doesn't
//!   expose `&mut Self` return ergonomically). Call `withCostCap` +
//!   `withImplicitCorrectionWindowSecs` as separate statements.
//! - **JSON-string persistence** — `exportJson()` + `fromJson()` —
//!   same as Python, portable across all bindings.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use noos::regulator::otel as rust_otel;
use noos::{
    CircuitBreakReason as RustCBR, CorrectionPattern as RustCP, Decision as RustDecision,
    LLMEvent as RustLLMEvent, Regulator as RustRegulator, RegulatorState as RustRS,
};

// ── LLMEvent ──────────────────────────────────────────────────────────

/// An event to feed into `Regulator.onEvent`. Construct via static
/// factory methods. The object is opaque; the only inspection
/// property is `.kind`.
///
/// `js_name = "LLMEvent"` preserves the all-caps acronym — napi-rs 3.x
/// runs struct names through `Case::Pascal`, which would rewrite
/// `LLMEvent` → `LlmEvent`. Python binding uses the same class name;
/// this keeps the JS / Python APIs symmetric.
#[napi(js_name = "LLMEvent")]
pub struct LLMEvent {
    inner: RustLLMEvent,
}

#[napi]
impl LLMEvent {
    /// A new turn begins. `userMessage` is the user's input text.
    #[napi(factory)]
    pub fn turn_start(user_message: String) -> Self {
        Self {
            inner: RustLLMEvent::TurnStart { user_message },
        }
    }

    /// One token emitted by the LLM. Pass `logprob = 0` if the
    /// provider does not expose per-token logprobs — the regulator
    /// treats any non-finite or non-negative value as unavailable and
    /// falls back to the structural confidence heuristic.
    #[napi(factory)]
    pub fn token(token: String, logprob: f64, index: u32) -> Self {
        Self {
            inner: RustLLMEvent::Token {
                token,
                logprob,
                index: index as usize,
            },
        }
    }

    /// Turn response complete. Full text regardless of whether you
    /// streamed tokens.
    #[napi(factory)]
    pub fn turn_complete(full_response: String) -> Self {
        Self {
            inner: RustLLMEvent::TurnComplete { full_response },
        }
    }

    /// Cost accounting for the turn. Emit after `turnComplete`.
    #[napi(factory)]
    pub fn cost(
        tokens_in: u32,
        tokens_out: u32,
        wallclock_ms: u32,
        provider: Option<String>,
    ) -> Self {
        Self {
            inner: RustLLMEvent::Cost {
                tokens_in,
                tokens_out,
                wallclock_ms,
                provider,
            },
        }
    }

    /// Ground-truth signal about response quality. `quality` is in
    /// `[0, 1]`. `fragmentSpans` is reserved for a future
    /// `lowConfidenceSpans` predicate — pass `null` for now.
    #[napi(factory)]
    pub fn quality_feedback(quality: f64, fragment_spans: Option<Vec<Vec<u32>>>) -> Self {
        // napi-rs maps `[number, number]` tuples clunkily; accept
        // `number[][]` where each inner vec is `[start, end]`.
        let fragment_spans = fragment_spans.map(|spans| {
            spans
                .into_iter()
                .filter_map(|pair| {
                    if pair.len() == 2 {
                        Some((pair[0] as usize, pair[1] as usize))
                    } else {
                        None
                    }
                })
                .collect()
        });
        Self {
            inner: RustLLMEvent::QualityFeedback {
                quality,
                fragment_spans,
            },
        }
    }

    /// User corrected the previous response. `correctsLast = true`
    /// attributes the correction to the prior turn's topic cluster.
    #[napi(factory)]
    pub fn user_correction(correction_message: String, corrects_last: bool) -> Self {
        Self {
            inner: RustLLMEvent::UserCorrection {
                correction_message,
                corrects_last,
            },
        }
    }

    /// The agent invoked a tool. Drives per-turn loop detection.
    #[napi(factory)]
    pub fn tool_call(tool_name: String, args_json: Option<String>) -> Self {
        Self {
            inner: RustLLMEvent::ToolCall {
                tool_name,
                args_json,
            },
        }
    }

    /// The tool call just returned. `success` and `durationMs` feed
    /// `toolFailureCount` and `toolTotalDurationMs`.
    #[napi(factory)]
    pub fn tool_result(
        tool_name: String,
        success: bool,
        duration_ms: BigInt,
        error_summary: Option<String>,
    ) -> Result<Self> {
        let duration_ms = duration_ms.get_u64().1;
        Ok(Self {
            inner: RustLLMEvent::ToolResult {
                tool_name,
                success,
                duration_ms,
                error_summary,
            },
        })
    }

    /// Variant name: `turn_start`, `token`, `turn_complete`, `cost`,
    /// `quality_feedback`, `user_correction`, `tool_call`,
    /// `tool_result`, or `unknown` (future variants).
    #[napi(getter)]
    pub fn kind(&self) -> &'static str {
        match &self.inner {
            RustLLMEvent::TurnStart { .. } => "turn_start",
            RustLLMEvent::Token { .. } => "token",
            RustLLMEvent::TurnComplete { .. } => "turn_complete",
            RustLLMEvent::Cost { .. } => "cost",
            RustLLMEvent::QualityFeedback { .. } => "quality_feedback",
            RustLLMEvent::UserCorrection { .. } => "user_correction",
            RustLLMEvent::ToolCall { .. } => "tool_call",
            RustLLMEvent::ToolResult { .. } => "tool_result",
            _ => "unknown",
        }
    }
}

/// Parse an OpenTelemetry GenAI span (JSON string) into a list of
/// `LLMEvent`s ready to feed `Regulator.onEvent`. Returns an empty
/// array when the span has no recognized `gen_ai.*` signals.
/// Throws on malformed JSON.
///
/// Exposed as a freestanding function (not a `LLMEvent` static method)
/// because napi-rs 3.x static methods returning `Vec<Self>` break class
/// registration — the Python binding's `LLMEvent.from_otel_span_json`
/// shape isn't portable to napi-rs. Usage:
/// ```js
/// import { llmEventsFromOtelSpanJson } from '@triangle-technology/noos';
/// for (const e of llmEventsFromOtelSpanJson(spanJson)) regulator.onEvent(e);
/// ```
#[napi]
pub fn llm_events_from_otel_span_json(span_json: String) -> Result<Vec<LLMEvent>> {
    let value: serde_json::Value = serde_json::from_str(&span_json)
        .map_err(|e| Error::new(Status::InvalidArg, format!("OTel span JSON parse failed: {e}")))?;
    Ok(rust_otel::events_from_span(&value)
        .into_iter()
        .map(|inner| LLMEvent { inner })
        .collect())
}

// ── Decision ──────────────────────────────────────────────────────────

/// The output of `Regulator.decide()`. Branch on `.kind`;
/// variant-specific getters return `null` when not applicable.
///
/// Variants:
/// - `continue` — no intervention required.
/// - `scope_drift_warn` — use `.driftScore`, `.driftTokens`, `.taskTokens`.
/// - `circuit_break` — use `.reason` (a `CircuitBreakReason`), `.suggestion`.
/// - `procedural_warning` — use `.patterns` (list of `CorrectionPattern`).
/// - `low_confidence_spans` — reserved for future use.
#[napi]
pub struct Decision {
    inner: RustDecision,
}

#[napi]
impl Decision {
    #[napi(getter)]
    pub fn kind(&self) -> &'static str {
        match &self.inner {
            RustDecision::Continue => "continue",
            RustDecision::CircuitBreak { .. } => "circuit_break",
            RustDecision::ScopeDriftWarn { .. } => "scope_drift_warn",
            RustDecision::LowConfidenceSpans { .. } => "low_confidence_spans",
            RustDecision::ProceduralWarning { .. } => "procedural_warning",
            _ => "unknown",
        }
    }

    // Scope drift accessors
    #[napi(getter)]
    pub fn drift_score(&self) -> Option<f64> {
        if let RustDecision::ScopeDriftWarn { drift_score, .. } = &self.inner {
            Some(*drift_score)
        } else {
            None
        }
    }

    #[napi(getter)]
    pub fn drift_tokens(&self) -> Option<Vec<String>> {
        if let RustDecision::ScopeDriftWarn { drift_tokens, .. } = &self.inner {
            Some(drift_tokens.clone())
        } else {
            None
        }
    }

    #[napi(getter)]
    pub fn task_tokens(&self) -> Option<Vec<String>> {
        if let RustDecision::ScopeDriftWarn { task_tokens, .. } = &self.inner {
            Some(task_tokens.clone())
        } else {
            None
        }
    }

    // CircuitBreak accessors
    #[napi(getter)]
    pub fn reason(&self) -> Option<CircuitBreakReason> {
        if let RustDecision::CircuitBreak { reason, .. } = &self.inner {
            Some(CircuitBreakReason {
                inner: reason.clone(),
            })
        } else {
            None
        }
    }

    #[napi(getter)]
    pub fn suggestion(&self) -> Option<String> {
        if let RustDecision::CircuitBreak { suggestion, .. } = &self.inner {
            Some(suggestion.clone())
        } else {
            None
        }
    }

    // ProceduralWarning accessors
    #[napi(getter)]
    pub fn patterns(&self) -> Option<Vec<CorrectionPattern>> {
        if let RustDecision::ProceduralWarning { patterns } = &self.inner {
            Some(
                patterns
                    .iter()
                    .map(|p| CorrectionPattern { inner: p.clone() })
                    .collect(),
            )
        } else {
            None
        }
    }

    // Convenience predicates (mirrors the Rust `matches!` idiom)
    #[napi]
    pub fn is_continue(&self) -> bool {
        matches!(self.inner, RustDecision::Continue)
    }
    #[napi]
    pub fn is_scope_drift(&self) -> bool {
        matches!(self.inner, RustDecision::ScopeDriftWarn { .. })
    }
    #[napi]
    pub fn is_circuit_break(&self) -> bool {
        matches!(self.inner, RustDecision::CircuitBreak { .. })
    }
    #[napi]
    pub fn is_procedural_warning(&self) -> bool {
        matches!(self.inner, RustDecision::ProceduralWarning { .. })
    }
    #[napi]
    pub fn is_low_confidence(&self) -> bool {
        matches!(self.inner, RustDecision::LowConfidenceSpans { .. })
    }
}

// ── CircuitBreakReason ────────────────────────────────────────────────

/// Why a `CircuitBreak` fired. Branch on `.kind`.
///
/// Variants:
/// - `cost_cap_reached` — use `.tokensSpent`, `.tokensCap`, `.meanQualityLastN`.
/// - `quality_decline_no_recovery` — use `.turns`, `.meanDelta`.
/// - `repeated_failure_pattern` — use `.cluster`, `.failureCount`.
/// - `repeated_tool_call_loop` — use `.toolName`, `.consecutiveCount`.
#[napi]
pub struct CircuitBreakReason {
    inner: RustCBR,
}

#[napi]
impl CircuitBreakReason {
    #[napi(getter)]
    pub fn kind(&self) -> &'static str {
        match &self.inner {
            RustCBR::CostCapReached { .. } => "cost_cap_reached",
            RustCBR::QualityDeclineNoRecovery { .. } => "quality_decline_no_recovery",
            RustCBR::RepeatedFailurePattern { .. } => "repeated_failure_pattern",
            RustCBR::RepeatedToolCallLoop { .. } => "repeated_tool_call_loop",
            _ => "unknown",
        }
    }

    // CostCapReached
    #[napi(getter)]
    pub fn tokens_spent(&self) -> Option<u32> {
        if let RustCBR::CostCapReached { tokens_spent, .. } = &self.inner {
            Some(*tokens_spent)
        } else {
            None
        }
    }
    #[napi(getter)]
    pub fn tokens_cap(&self) -> Option<u32> {
        if let RustCBR::CostCapReached { tokens_cap, .. } = &self.inner {
            Some(*tokens_cap)
        } else {
            None
        }
    }
    #[napi(getter)]
    pub fn mean_quality_last_n(&self) -> Option<f64> {
        if let RustCBR::CostCapReached {
            mean_quality_last_n,
            ..
        } = &self.inner
        {
            Some(*mean_quality_last_n)
        } else {
            None
        }
    }

    // QualityDeclineNoRecovery
    #[napi(getter)]
    pub fn turns(&self) -> Option<u32> {
        if let RustCBR::QualityDeclineNoRecovery { turns, .. } = &self.inner {
            Some(*turns as u32)
        } else {
            None
        }
    }
    #[napi(getter)]
    pub fn mean_delta(&self) -> Option<f64> {
        if let RustCBR::QualityDeclineNoRecovery { mean_delta, .. } = &self.inner {
            Some(*mean_delta)
        } else {
            None
        }
    }

    // RepeatedFailurePattern
    #[napi(getter)]
    pub fn cluster(&self) -> Option<String> {
        if let RustCBR::RepeatedFailurePattern { cluster, .. } = &self.inner {
            Some(cluster.clone())
        } else {
            None
        }
    }
    #[napi(getter)]
    pub fn failure_count(&self) -> Option<u32> {
        if let RustCBR::RepeatedFailurePattern { failure_count, .. } = &self.inner {
            Some(*failure_count as u32)
        } else {
            None
        }
    }

    // RepeatedToolCallLoop
    #[napi(getter)]
    pub fn tool_name(&self) -> Option<String> {
        if let RustCBR::RepeatedToolCallLoop { tool_name, .. } = &self.inner {
            Some(tool_name.clone())
        } else {
            None
        }
    }
    #[napi(getter)]
    pub fn consecutive_count(&self) -> Option<u32> {
        if let RustCBR::RepeatedToolCallLoop {
            consecutive_count, ..
        } = &self.inner
        {
            Some(*consecutive_count as u32)
        } else {
            None
        }
    }
}

// ── CorrectionPattern ─────────────────────────────────────────────────

/// A learned procedural rule extracted from repeated `userCorrection`
/// events on the same topic cluster. Read-only from JS.
#[napi]
pub struct CorrectionPattern {
    inner: RustCP,
}

#[napi]
impl CorrectionPattern {
    #[napi(getter)]
    pub fn user_id(&self) -> String {
        self.inner.user_id.clone()
    }
    #[napi(getter)]
    pub fn topic_cluster(&self) -> String {
        self.inner.topic_cluster.clone()
    }
    /// Opaque pattern identifier (currently `corrections_on_{cluster}`).
    /// Apps read `exampleCorrections` for the raw correction texts to
    /// inject into the next LLM prompt.
    #[napi(getter)]
    pub fn pattern_name(&self) -> String {
        self.inner.pattern_name.clone()
    }
    #[napi(getter)]
    pub fn learned_from_turns(&self) -> u32 {
        self.inner.learned_from_turns as u32
    }
    #[napi(getter)]
    pub fn confidence(&self) -> f64 {
        self.inner.confidence
    }
    #[napi(getter)]
    pub fn example_corrections(&self) -> Vec<String> {
        self.inner.example_corrections.clone()
    }
}

// ── Regulator ─────────────────────────────────────────────────────────

/// External regulatory layer for an LLM agent loop.
///
/// Construct with `Regulator.forUser(userId)`. Call `withCostCap(n)`
/// or `withImplicitCorrectionWindowSecs(s)` to configure. Feed
/// `LLMEvent`s via `onEvent(e)` and branch on `decide()`.
#[napi]
pub struct Regulator {
    inner: RustRegulator,
}

#[napi]
impl Regulator {
    /// Create a fresh regulator bound to a user identity. Use
    /// `fromJson(...)` to restore a saved snapshot instead.
    #[napi(factory)]
    pub fn for_user(user_id: String) -> Self {
        Self {
            inner: RustRegulator::for_user(user_id),
        }
    }

    /// Mutable: override the default cumulative output-token cap used
    /// by the `cost_cap_reached` CircuitBreak predicate. Swaps the
    /// inner Rust regulator via `mem::replace` (the Rust builder
    /// consumes `self`, and napi-rs can't return `&mut Self` across
    /// the JS boundary, so we mutate in place and return `()`).
    /// Chain via separate statements, not fluently.
    #[napi]
    pub fn with_cost_cap(&mut self, cap_tokens: u32) {
        // Swap inner out via a placeholder, apply the Rust consuming
        // builder, put the result back. `Regulator::for_user("")` is
        // cheap to construct.
        let uid = self.inner.user_id().to_string();
        let taken = std::mem::replace(&mut self.inner, RustRegulator::for_user(uid));
        self.inner = taken.with_cost_cap(cap_tokens);
    }

    /// Mutable: enable implicit correction detection with a window in
    /// seconds. A `turnStart` arriving within this window of the
    /// previous `turnComplete` AND mapping to the same topic cluster
    /// is treated as a retry — a synthetic correction is recorded
    /// against the cluster using the new user message as the
    /// correction text. Swaps the inner Rust regulator via
    /// `mem::replace` (same shape reason as `with_cost_cap`).
    ///
    /// Typical values: 30-60 seconds for chat UIs. Throws on
    /// non-finite or non-positive values.
    #[napi]
    pub fn with_implicit_correction_window_secs(&mut self, window_secs: f64) -> Result<()> {
        if !window_secs.is_finite() || window_secs <= 0.0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("windowSecs must be a positive finite number, got {window_secs}"),
            ));
        }
        let window = std::time::Duration::from_secs_f64(window_secs);
        let uid = self.inner.user_id().to_string();
        let taken = std::mem::replace(&mut self.inner, RustRegulator::for_user(uid));
        self.inner = taken.with_implicit_correction_window(window);
        Ok(())
    }

    /// Mutable: override the default scope-drift threshold used by
    /// `Decision.kind === "scopeDriftWarn"`. Default is 0.5 — raise to
    /// 0.7 on verbose models (e.g. Gemini) to suppress the on-topic
    /// false-positive regime.
    ///
    /// Accepts `[0.0, 1.0]`. Throws on non-finite or out-of-range
    /// values.
    #[napi]
    pub fn with_scope_drift_threshold(&mut self, threshold: f64) -> Result<()> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "threshold must be finite and within [0.0, 1.0], got {threshold}"
                ),
            ));
        }
        let uid = self.inner.user_id().to_string();
        let taken = std::mem::replace(&mut self.inner, RustRegulator::for_user(uid));
        self.inner = taken.with_scope_drift_threshold(threshold);
        Ok(())
    }

    /// Mutable: feed one event into the regulator. Requires
    /// mutation because the wrapped `CognitiveSession` accumulates
    /// state per turn and the regulator buffers responses between
    /// `turnComplete` and `qualityFeedback`. Typical per-turn
    /// ordering: `turnStart` → (optional `token` stream) →
    /// `turnComplete` → `cost` → optional `qualityFeedback` →
    /// optional `userCorrection`.
    #[napi]
    pub fn on_event(&mut self, event: &LLMEvent) {
        self.inner.on_event(event.inner.clone());
    }

    /// Return a regulatory decision for the current state. Call after
    /// you've fed events for the turn. Safe to call repeatedly — the
    /// result updates as new events come in.
    #[napi]
    pub fn decide(&self) -> Decision {
        Decision {
            inner: self.inner.decide(),
        }
    }

    // ── Continuous signal accessors ───────────────────────────────

    /// Hybrid confidence in `[0, 1]`.
    #[napi]
    pub fn confidence(&self) -> f64 {
        self.inner.confidence()
    }

    /// Fraction of the last turn's tokens that carried usable logprobs
    /// (in `[0, 1]`). Low values indicate the structural fallback is
    /// in use.
    #[napi]
    pub fn logprob_coverage(&self) -> f64 {
        self.inner.logprob_coverage()
    }

    /// Cumulative `tokensOut` across all `cost` events since creation
    /// / last `fromJson`.
    #[napi]
    pub fn total_tokens_out(&self) -> u32 {
        self.inner.total_tokens_out()
    }

    /// The current cost cap.
    #[napi]
    pub fn cost_cap_tokens(&self) -> u32 {
        self.inner.cost_cap_tokens()
    }

    // ── Tool-stats accessors ──────────────────────────────────────

    #[napi]
    pub fn tool_total_calls(&self) -> u32 {
        self.inner.tool_total_calls() as u32
    }

    /// Counts per tool name in the current turn.
    #[napi]
    pub fn tool_counts_by_name(&self) -> std::collections::HashMap<String, u32> {
        self.inner
            .tool_counts_by_name()
            .into_iter()
            .map(|(k, v)| (k, v as u32))
            .collect()
    }

    #[napi]
    pub fn tool_total_duration_ms(&self) -> BigInt {
        BigInt::from(self.inner.tool_total_duration_ms())
    }

    #[napi]
    pub fn tool_failure_count(&self) -> u32 {
        self.inner.tool_failure_count() as u32
    }

    /// Count of implicit corrections synthesised since creation /
    /// last `fromJson`. Zero when
    /// `withImplicitCorrectionWindowSecs` has not been called.
    #[napi]
    pub fn implicit_corrections_count(&self) -> u32 {
        self.inner.implicit_corrections_count() as u32
    }

    /// The current scope-drift threshold — either the crate default
    /// (0.5) or the last value accepted by
    /// `withScopeDriftThreshold(...)`. Not persisted in snapshots.
    #[napi]
    pub fn scope_drift_threshold(&self) -> f64 {
        self.inner.scope_drift_threshold()
    }

    // ── Identity + Path B helpers ─────────────────────────────────

    #[napi(getter)]
    pub fn user_id(&self) -> String {
        self.inner.user_id().to_string()
    }

    /// Returns the bulleted correction-history block when the current
    /// decision is `procedural_warning`, else `null`.
    #[napi]
    pub fn corrections_prelude(&self) -> Option<String> {
        self.inner.corrections_prelude()
    }

    /// One-call helper: prepends the correction-history block to
    /// `userPrompt` when patterns apply; returns `userPrompt`
    /// unchanged otherwise.
    #[napi]
    pub fn inject_corrections(&self, user_prompt: String) -> String {
        self.inner.inject_corrections(&user_prompt)
    }

    /// One-call snapshot of every numeric observability signal as a
    /// `Record<string, number>`. Drop-in for metrics pipelines.
    #[napi]
    pub fn metrics_snapshot(&self) -> std::collections::HashMap<String, f64> {
        self.inner.metrics_snapshot()
    }

    // ── Persistence ───────────────────────────────────────────────

    /// Export state as a JSON string. Persist via file / DB / Redis;
    /// restore with `fromJson`.
    #[napi]
    pub fn export_json(&self) -> Result<String> {
        let state = self.inner.export();
        serde_json::to_string(&state)
            .map_err(|e| Error::new(Status::GenericFailure, format!("JSON serialize: {e}")))
    }

    /// Restore a regulator from a snapshot produced by `exportJson`.
    /// Throws on malformed JSON.
    #[napi(factory)]
    pub fn from_json(json_str: String) -> Result<Self> {
        let state: RustRS = serde_json::from_str(&json_str)
            .map_err(|e| Error::new(Status::InvalidArg, format!("JSON parse: {e}")))?;
        Ok(Self {
            inner: RustRegulator::import(state),
        })
    }
}
