//! Python bindings for the Noos LLM-agent reliability crate.
//!
//! Exposes a minimal, Pythonic API mirroring the Rust `Regulator`:
//!
//! ```python
//! from noos_regulator import Regulator, LLMEvent
//!
//! r = Regulator.for_user("alice").with_cost_cap(2000)
//! r.on_event(LLMEvent.turn_start("Refactor fetch_user to be async"))
//! # ... call your LLM ...
//! r.on_event(LLMEvent.turn_complete(response_text))
//! r.on_event(LLMEvent.cost(tokens_in=25, tokens_out=800, wallclock_ms=500))
//!
//! decision = r.decide()
//! if decision.kind == "scope_drift_warn":
//!     print(f"drift_score={decision.drift_score}")
//! elif decision.kind == "circuit_break":
//!     print(f"halt: {decision.suggestion} ({decision.reason.kind})")
//! elif decision.kind == "procedural_warning":
//!     for p in decision.patterns:
//!         print(p.pattern_name, p.example_corrections)
//! ```
//!
//! Decision / LLMEvent / CircuitBreakReason carry a `.kind` string plus
//! variant-specific typed attributes that return `None` when not
//! applicable. This keeps pattern matching Pythonic without forcing
//! Python enum gymnastics that don't map cleanly to Rust data-bearing
//! variants.
//!
//! Persistence uses JSON strings via `Regulator.export_json()` and
//! `Regulator.from_json(...)` so snapshots are portable across Python,
//! Rust, or any other JSON-aware consumer.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use noos::regulator::otel as rust_otel;
use noos::{
    CircuitBreakReason as RustCBR, CorrectionPattern as RustCP, Decision as RustDecision,
    LLMEvent as RustLLMEvent, Regulator as RustRegulator, RegulatorState as RustRS,
};

// ── LLMEvent ──────────────────────────────────────────────────────────

/// An event to feed into [`Regulator.on_event`].
///
/// Construct via factory staticmethods. The object is opaque; the only
/// inspection method is `.kind` (returns the variant name as a string).
#[pyclass(name = "LLMEvent", frozen, module = "noos_regulator")]
#[derive(Clone)]
struct PyLLMEvent {
    inner: RustLLMEvent,
}

#[pymethods]
impl PyLLMEvent {
    /// A new turn begins. `user_message` is the user's input text.
    #[staticmethod]
    fn turn_start(user_message: String) -> Self {
        Self {
            inner: RustLLMEvent::TurnStart { user_message },
        }
    }

    /// One token emitted by the LLM. Pass `logprob=0.0` if the provider
    /// does not expose per-token logprobs (the regulator treats any
    /// non-finite or non-negative value as unavailable and falls back
    /// to the structural confidence heuristic).
    #[staticmethod]
    fn token(token: String, logprob: f64, index: usize) -> Self {
        Self {
            inner: RustLLMEvent::Token {
                token,
                logprob,
                index,
            },
        }
    }

    /// Turn response complete. Full text regardless of whether you
    /// streamed tokens.
    #[staticmethod]
    fn turn_complete(full_response: String) -> Self {
        Self {
            inner: RustLLMEvent::TurnComplete { full_response },
        }
    }

    /// Cost accounting for the turn. Emit after `turn_complete`.
    #[staticmethod]
    #[pyo3(signature = (tokens_in, tokens_out, wallclock_ms, provider=None))]
    fn cost(
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
    /// `[0, 1]`. `fragment_spans` is reserved for a future
    /// `low_confidence_spans` predicate — pass `None` for now.
    #[staticmethod]
    #[pyo3(signature = (quality, fragment_spans=None))]
    fn quality_feedback(quality: f64, fragment_spans: Option<Vec<(usize, usize)>>) -> Self {
        Self {
            inner: RustLLMEvent::QualityFeedback {
                quality,
                fragment_spans,
            },
        }
    }

    /// User corrected the previous response. `corrects_last=True`
    /// attributes the correction to the prior turn's topic cluster.
    #[staticmethod]
    fn user_correction(correction_message: String, corrects_last: bool) -> Self {
        Self {
            inner: RustLLMEvent::UserCorrection {
                correction_message,
                corrects_last,
            },
        }
    }

    /// The agent invoked a tool. Drives per-turn loop detection.
    #[staticmethod]
    #[pyo3(signature = (tool_name, args_json=None))]
    fn tool_call(tool_name: String, args_json: Option<String>) -> Self {
        Self {
            inner: RustLLMEvent::ToolCall {
                tool_name,
                args_json,
            },
        }
    }

    /// The tool call just returned. `success` and `duration_ms` feed
    /// `tool_failure_count` and `tool_total_duration_ms`.
    #[staticmethod]
    #[pyo3(signature = (tool_name, success, duration_ms, error_summary=None))]
    fn tool_result(
        tool_name: String,
        success: bool,
        duration_ms: u64,
        error_summary: Option<String>,
    ) -> Self {
        Self {
            inner: RustLLMEvent::ToolResult {
                tool_name,
                success,
                duration_ms,
                error_summary,
            },
        }
    }

    /// Parse an OpenTelemetry GenAI span (JSON string) into a list of
    /// `LLMEvent`s ready to feed `Regulator.on_event`.
    ///
    /// Input format: the SDK-idiomatic dict form (attributes as a map,
    /// events as a list of `{name, attributes}` objects). See the
    /// Rust crate's `noos::regulator::otel` module docs for the full
    /// attribute table.
    ///
    /// Example:
    ///
    /// ```python
    /// import json
    /// span = {
    ///     "attributes": {
    ///         "gen_ai.usage.input_tokens": 25,
    ///         "gen_ai.usage.output_tokens": 800,
    ///     },
    ///     "events": [
    ///         {"name": "gen_ai.user.message",
    ///          "attributes": {"content": "Refactor fetch_user"}},
    ///         {"name": "gen_ai.assistant.message",
    ///          "attributes": {"content": "async def fetch_user..."}},
    ///     ],
    ///     "start_time_unix_nano": 1700000000000000000,
    ///     "end_time_unix_nano":   1700000000500000000,
    /// }
    /// for event in LLMEvent.from_otel_span_json(json.dumps(span)):
    ///     regulator.on_event(event)
    /// ```
    ///
    /// Raises `ValueError` on malformed JSON. Returns an empty list
    /// when the span carries no recognized `gen_ai.*` signals.
    #[staticmethod]
    fn from_otel_span_json(span_json: &str) -> PyResult<Vec<Self>> {
        let value: serde_json::Value = serde_json::from_str(span_json)
            .map_err(|e| PyValueError::new_err(format!("OTel span JSON parse failed: {e}")))?;
        Ok(rust_otel::events_from_span(&value)
            .into_iter()
            .map(|inner| Self { inner })
            .collect())
    }

    /// Variant name: `turn_start`, `token`, `turn_complete`, `cost`,
    /// `quality_feedback`, `user_correction`, `tool_call`,
    /// `tool_result`, or `unknown` (future variants).
    #[getter]
    fn kind(&self) -> &'static str {
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

    fn __repr__(&self) -> String {
        format!("LLMEvent.{}", self.kind())
    }
}

// ── Decision ──────────────────────────────────────────────────────────

/// The output of [`Regulator.decide`]. Branch on `.kind`; variant-specific
/// attributes return `None` when not applicable.
///
/// Variants:
/// - `continue` — no intervention required.
/// - `scope_drift_warn` — use `.drift_score`, `.drift_tokens`, `.task_tokens`.
/// - `circuit_break` — use `.reason` (a `CircuitBreakReason`) and `.suggestion`.
/// - `procedural_warning` — use `.patterns` (list of `CorrectionPattern`).
/// - `low_confidence_spans` — reserved; `.spans` currently unused.
#[pyclass(name = "Decision", frozen, module = "noos_regulator")]
#[derive(Clone)]
struct PyDecision {
    inner: RustDecision,
}

#[pymethods]
impl PyDecision {
    #[getter]
    fn kind(&self) -> &'static str {
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
    #[getter]
    fn drift_score(&self) -> Option<f64> {
        if let RustDecision::ScopeDriftWarn { drift_score, .. } = &self.inner {
            Some(*drift_score)
        } else {
            None
        }
    }

    #[getter]
    fn drift_tokens(&self) -> Option<Vec<String>> {
        if let RustDecision::ScopeDriftWarn { drift_tokens, .. } = &self.inner {
            Some(drift_tokens.clone())
        } else {
            None
        }
    }

    #[getter]
    fn task_tokens(&self) -> Option<Vec<String>> {
        if let RustDecision::ScopeDriftWarn { task_tokens, .. } = &self.inner {
            Some(task_tokens.clone())
        } else {
            None
        }
    }

    // CircuitBreak accessors
    #[getter]
    fn reason(&self) -> Option<PyCircuitBreakReason> {
        if let RustDecision::CircuitBreak { reason, .. } = &self.inner {
            Some(PyCircuitBreakReason {
                inner: reason.clone(),
            })
        } else {
            None
        }
    }

    #[getter]
    fn suggestion(&self) -> Option<String> {
        if let RustDecision::CircuitBreak { suggestion, .. } = &self.inner {
            Some(suggestion.clone())
        } else {
            None
        }
    }

    // ProceduralWarning accessors
    #[getter]
    fn patterns(&self) -> Option<Vec<PyCorrectionPattern>> {
        if let RustDecision::ProceduralWarning { patterns } = &self.inner {
            Some(
                patterns
                    .iter()
                    .map(|p| PyCorrectionPattern { inner: p.clone() })
                    .collect(),
            )
        } else {
            None
        }
    }

    // Convenience predicates (mirrors the Rust `matches!` idiom)
    fn is_continue(&self) -> bool {
        matches!(self.inner, RustDecision::Continue)
    }
    fn is_scope_drift(&self) -> bool {
        matches!(self.inner, RustDecision::ScopeDriftWarn { .. })
    }
    fn is_circuit_break(&self) -> bool {
        matches!(self.inner, RustDecision::CircuitBreak { .. })
    }
    fn is_procedural_warning(&self) -> bool {
        matches!(self.inner, RustDecision::ProceduralWarning { .. })
    }
    fn is_low_confidence(&self) -> bool {
        matches!(self.inner, RustDecision::LowConfidenceSpans { .. })
    }

    fn __repr__(&self) -> String {
        format!("Decision.{}", self.kind())
    }
}

// ── CircuitBreakReason ────────────────────────────────────────────────

/// Why a `CircuitBreak` fired. Branch on `.kind`.
///
/// Variants:
/// - `cost_cap_reached` — use `.tokens_spent`, `.tokens_cap`, `.mean_quality_last_n`.
/// - `quality_decline_no_recovery` — use `.turns`, `.mean_delta`.
/// - `repeated_failure_pattern` — use `.cluster`, `.failure_count`.
/// - `repeated_tool_call_loop` — use `.tool_name`, `.consecutive_count`.
#[pyclass(name = "CircuitBreakReason", frozen, module = "noos_regulator")]
#[derive(Clone)]
struct PyCircuitBreakReason {
    inner: RustCBR,
}

#[pymethods]
impl PyCircuitBreakReason {
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            RustCBR::CostCapReached { .. } => "cost_cap_reached",
            RustCBR::QualityDeclineNoRecovery { .. } => "quality_decline_no_recovery",
            RustCBR::RepeatedFailurePattern { .. } => "repeated_failure_pattern",
            RustCBR::RepeatedToolCallLoop { .. } => "repeated_tool_call_loop",
            _ => "unknown",
        }
    }

    // CostCapReached
    #[getter]
    fn tokens_spent(&self) -> Option<u32> {
        if let RustCBR::CostCapReached { tokens_spent, .. } = &self.inner {
            Some(*tokens_spent)
        } else {
            None
        }
    }
    #[getter]
    fn tokens_cap(&self) -> Option<u32> {
        if let RustCBR::CostCapReached { tokens_cap, .. } = &self.inner {
            Some(*tokens_cap)
        } else {
            None
        }
    }
    #[getter]
    fn mean_quality_last_n(&self) -> Option<f64> {
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
    #[getter]
    fn turns(&self) -> Option<usize> {
        if let RustCBR::QualityDeclineNoRecovery { turns, .. } = &self.inner {
            Some(*turns)
        } else {
            None
        }
    }
    #[getter]
    fn mean_delta(&self) -> Option<f64> {
        if let RustCBR::QualityDeclineNoRecovery { mean_delta, .. } = &self.inner {
            Some(*mean_delta)
        } else {
            None
        }
    }

    // RepeatedFailurePattern
    #[getter]
    fn cluster(&self) -> Option<String> {
        if let RustCBR::RepeatedFailurePattern { cluster, .. } = &self.inner {
            Some(cluster.clone())
        } else {
            None
        }
    }
    #[getter]
    fn failure_count(&self) -> Option<usize> {
        if let RustCBR::RepeatedFailurePattern { failure_count, .. } = &self.inner {
            Some(*failure_count)
        } else {
            None
        }
    }

    // RepeatedToolCallLoop
    #[getter]
    fn tool_name(&self) -> Option<String> {
        if let RustCBR::RepeatedToolCallLoop { tool_name, .. } = &self.inner {
            Some(tool_name.clone())
        } else {
            None
        }
    }
    #[getter]
    fn consecutive_count(&self) -> Option<usize> {
        if let RustCBR::RepeatedToolCallLoop {
            consecutive_count, ..
        } = &self.inner
        {
            Some(*consecutive_count)
        } else {
            None
        }
    }

    fn __repr__(&self) -> String {
        format!("CircuitBreakReason.{}", self.kind())
    }
}

// ── CorrectionPattern ─────────────────────────────────────────────────

/// A learned procedural rule extracted from repeated `user_correction`
/// events on the same topic cluster.
///
/// Frozen / read-only from Python; the fields are set by the regulator
/// when a pattern emerges.
#[pyclass(name = "CorrectionPattern", frozen, module = "noos_regulator")]
#[derive(Clone)]
struct PyCorrectionPattern {
    inner: RustCP,
}

#[pymethods]
impl PyCorrectionPattern {
    #[getter]
    fn user_id(&self) -> String {
        self.inner.user_id.clone()
    }
    #[getter]
    fn topic_cluster(&self) -> String {
        self.inner.topic_cluster.clone()
    }
    /// Opaque pattern identifier (currently `corrections_on_{cluster}`).
    /// Apps read `example_corrections` for the raw correction texts to
    /// inject into the next LLM prompt.
    #[getter]
    fn pattern_name(&self) -> String {
        self.inner.pattern_name.clone()
    }
    #[getter]
    fn learned_from_turns(&self) -> usize {
        self.inner.learned_from_turns
    }
    #[getter]
    fn confidence(&self) -> f64 {
        self.inner.confidence
    }
    #[getter]
    fn example_corrections(&self) -> Vec<String> {
        self.inner.example_corrections.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "CorrectionPattern(user={}, cluster={}, turns={}, conf={:.2})",
            self.inner.user_id,
            self.inner.topic_cluster,
            self.inner.learned_from_turns,
            self.inner.confidence,
        )
    }
}

// ── Regulator ─────────────────────────────────────────────────────────

/// External regulatory layer for an LLM agent loop.
///
/// Construct with `Regulator.for_user(user_id)`, optionally chain
/// `.with_cost_cap(n)` to override the default cumulative-token cap,
/// then feed `LLMEvent` objects via `on_event(...)` and branch on
/// `decide()`.
///
/// `unsendable` — do not share instances across Python threads. Each
/// agent loop owns its own regulator.
#[pyclass(name = "Regulator", unsendable, module = "noos_regulator")]
struct PyRegulator {
    inner: RustRegulator,
}

#[pymethods]
impl PyRegulator {
    /// Create a fresh regulator bound to a user identity. Use
    /// `from_json(...)` to restore a saved snapshot instead.
    #[staticmethod]
    fn for_user(user_id: String) -> Self {
        Self {
            inner: RustRegulator::for_user(user_id),
        }
    }

    /// Mutable: override the default cumulative output-token cap used
    /// by the `cost_cap_reached` CircuitBreak predicate. Swaps the
    /// inner Rust regulator via `mem::replace` (the Rust builder
    /// consumes `self` and returns `Self`, incompatible with PyO3's
    /// `PyRefMut` borrow shape). Returns the same `PyRefMut` for
    /// Python-side method chaining.
    fn with_cost_cap<'py>(
        mut slf: PyRefMut<'py, Self>,
        cap_tokens: u32,
    ) -> PyRefMut<'py, Self> {
        // The Rust builder consumes self and returns Self; here we
        // swap the inner out with a throwaway placeholder, run the
        // builder, and put the result back. The placeholder only
        // exists for the duration of the swap and is dropped.
        let uid = slf.inner.user_id().to_string();
        let taken = std::mem::replace(&mut slf.inner, RustRegulator::for_user(uid));
        slf.inner = taken.with_cost_cap(cap_tokens);
        slf
    }

    /// Mutable: enable implicit correction detection with a window in
    /// seconds. A `turn_start` arriving within this window of the
    /// previous `turn_complete` AND mapping to the same topic cluster
    /// is treated as a retry — a synthetic correction is recorded
    /// against the cluster using the new user message as the
    /// correction text. Swaps the inner Rust regulator via
    /// `mem::replace` (same shape reason as `with_cost_cap`).
    ///
    /// Typical values: 30-60 seconds for chat UIs. Pass a non-finite
    /// or non-positive value to raise `ValueError`.
    ///
    /// Returns self for chaining.
    ///
    /// Example:
    ///
    /// ```python
    /// r = Regulator.for_user("alice").with_implicit_correction_window_secs(30.0)
    /// # Now fast same-cluster retries auto-record corrections.
    /// ```
    fn with_implicit_correction_window_secs<'py>(
        mut slf: PyRefMut<'py, Self>,
        window_secs: f64,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if !window_secs.is_finite() || window_secs <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "window_secs must be a positive finite number, got {window_secs}"
            )));
        }
        let window = std::time::Duration::from_secs_f64(window_secs);
        let uid = slf.inner.user_id().to_string();
        let taken = std::mem::replace(&mut slf.inner, RustRegulator::for_user(uid));
        slf.inner = taken.with_implicit_correction_window(window);
        Ok(slf)
    }

    /// Mutable: feed one event into the regulator. Requires
    /// mutation because the wrapped `CognitiveSession` accumulates
    /// state per turn and the regulator buffers responses between
    /// `turn_complete` and `quality_feedback`. Typical per-turn
    /// ordering: `turn_start` → (optional `token` stream) →
    /// `turn_complete` → `cost` → optional `quality_feedback` →
    /// optional `user_correction`.
    fn on_event(&mut self, event: &PyLLMEvent) {
        self.inner.on_event(event.inner.clone());
    }

    /// Return a regulatory decision for the current state. Call after
    /// you've fed events for the turn. Call repeatedly to re-probe —
    /// results update as new events come in.
    fn decide(&self) -> PyDecision {
        PyDecision {
            inner: self.inner.decide(),
        }
    }

    // ── Continuous signal accessors ───────────────────────────────

    /// Hybrid confidence `[0, 1]`. Uses mean-NLL over the rolling
    /// logprob window when available; falls back to a structural
    /// heuristic (length + `?` ratio) otherwise.
    fn confidence(&self) -> f64 {
        self.inner.confidence()
    }

    /// Fraction of the last turn's tokens that carried usable logprobs
    /// (`[0, 1]`). Low values indicate the provider didn't expose
    /// logprobs and the structural fallback is in use.
    fn logprob_coverage(&self) -> f64 {
        self.inner.logprob_coverage()
    }

    /// Cumulative `tokens_out` across all `cost` events since creation
    /// / last `import`.
    fn total_tokens_out(&self) -> u32 {
        self.inner.total_tokens_out()
    }

    /// The current cost cap (default or set via `with_cost_cap`).
    fn cost_cap_tokens(&self) -> u32 {
        self.inner.cost_cap_tokens()
    }

    // ── Tool-stats accessors (0.3.0) ──────────────────────────────

    /// Total tool calls observed in the current turn. Resets on
    /// `turn_start`.
    fn tool_total_calls(&self) -> usize {
        self.inner.tool_total_calls()
    }

    /// Counts per tool name in the current turn.
    fn tool_counts_by_name(&self) -> std::collections::HashMap<String, usize> {
        self.inner.tool_counts_by_name()
    }

    /// Sum of `duration_ms` from `tool_result` events in the current turn.
    fn tool_total_duration_ms(&self) -> u64 {
        self.inner.tool_total_duration_ms()
    }

    /// Number of `tool_result` events with `success == False` in the
    /// current turn.
    fn tool_failure_count(&self) -> usize {
        self.inner.tool_failure_count()
    }

    /// Count of implicit corrections synthesised by the
    /// implicit-correction detector since creation / last `from_json`.
    /// Always 0 when
    /// `with_implicit_correction_window_secs` has not been called.
    ///
    /// Per-process counter — resets on `from_json` (not persisted).
    fn implicit_corrections_count(&self) -> usize {
        self.inner.implicit_corrections_count()
    }

    /// One-call snapshot of every numeric observability signal as a
    /// `dict[str, float]`. Drop-in for Prometheus / Datadog / StatsD
    /// pipelines — iterate the dict and forward `(key, value)` pairs
    /// to your metrics client.
    ///
    /// Keys are stable and prefixed with `noos.` so they don't
    /// collide with your app namespace. See the Rust crate's
    /// `Regulator::metrics_snapshot` docs for the full key list.
    fn metrics_snapshot(&self) -> std::collections::HashMap<String, f64> {
        self.inner.metrics_snapshot()
    }

    // ── Identity + Path B helpers ─────────────────────────────────

    fn user_id(&self) -> String {
        self.inner.user_id().to_string()
    }

    /// Returns the bulleted correction-history block when the current
    /// decision is `procedural_warning`, else `None`. Primitive for
    /// `inject_corrections`.
    fn corrections_prelude(&self) -> Option<String> {
        self.inner.corrections_prelude()
    }

    /// One-call helper: prepends the correction-history block to
    /// `user_prompt` when patterns apply; returns `user_prompt`
    /// unchanged otherwise. Use in place of hand-threading
    /// `corrections_prelude` into your prompt builder.
    fn inject_corrections(&self, user_prompt: &str) -> String {
        self.inner.inject_corrections(user_prompt)
    }

    // ── Persistence ───────────────────────────────────────────────

    /// Export state as a JSON string. Contains `LearnedState` +
    /// correction patterns so patterns survive process restarts.
    /// Persist to file / DB / Redis; restore via `from_json`.
    fn export_json(&self) -> PyResult<String> {
        let state = self.inner.export();
        serde_json::to_string(&state)
            .map_err(|e| PyValueError::new_err(format!("JSON serialize failed: {e}")))
    }

    /// Restore a regulator from a snapshot produced by `export_json`.
    /// Raises `ValueError` on malformed JSON.
    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Self> {
        let state: RustRS = serde_json::from_str(json_str)
            .map_err(|e| PyValueError::new_err(format!("JSON parse failed: {e}")))?;
        Ok(Self {
            inner: RustRegulator::import(state),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Regulator(user={}, total_tokens_out={})",
            self.inner.user_id(),
            self.inner.total_tokens_out(),
        )
    }
}

// ── Module ────────────────────────────────────────────────────────────

#[pymodule]
fn noos_regulator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLLMEvent>()?;
    m.add_class::<PyDecision>()?;
    m.add_class::<PyCircuitBreakReason>()?;
    m.add_class::<PyCorrectionPattern>()?;
    m.add_class::<PyRegulator>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
