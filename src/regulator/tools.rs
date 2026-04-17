//! Tool-call observation channel (Path A, 0.3.0).
//!
//! Modern LLM agents make tool calls (function calls, API calls,
//! retrieval). Without visibility into these, the regulator cannot
//! detect a very common failure mode: the agent calls the same tool
//! over and over, burning cost without producing progress.
//!
//! This module tracks per-turn tool-call sequences. It exposes a
//! detector `detected_loop()` that fires `Some((tool, count))` when the
//! last [`TOOL_LOOP_THRESHOLD`] consecutive tool calls are the same
//! tool name — the signature pattern of "agent is stuck in a retry
//! loop on a single tool that is failing / not making progress."
//!
//! Counts reset on every [`LLMEvent::TurnStart`] so a new turn gets a
//! fresh view — accumulation across turns would conflate "legitimately
//! hit this tool 3 times across a multi-step plan" with "stuck in a
//! one-tool loop." The loop-detection unit of account is the turn.
//!
//! `ToolResult` events carry `duration_ms` + `success` for observability
//! hooks (accessible via [`ToolStatsAccumulator::total_calls`] and
//! related getters); they do not influence loop detection in 0.3.0.
//!
//! ## Gating (P10)
//!
//! This module produces the
//! [`CircuitBreakReason::RepeatedToolCallLoop`] variant of
//! [`Decision::CircuitBreak`] via [`Regulator::decide`].
//!
//! - **Suppresses**:
//!   [`Decision::ScopeDriftWarn`],
//!   [`Decision::ProceduralWarning`],
//!   [`Decision::Continue`]. Tool-loop halt is a circuit break — it
//!   dominates every advisory below it.
//! - **Suppressed by**:
//!   [`CircuitBreakReason::CostCapReached`] and
//!   [`CircuitBreakReason::QualityDeclineNoRecovery`]. The two
//!   cost / quality circuit breaks are considered more urgent in the
//!   priority chain — if the budget has already been blown or quality
//!   is collapsing, halting on tool pattern alone would mask the
//!   bigger problem.
//! - **Inactive when**: fewer than [`TOOL_LOOP_THRESHOLD`] consecutive
//!   [`LLMEvent::ToolCall`] events have landed since the last
//!   [`LLMEvent::TurnStart`], OR the trailing run includes at least
//!   two distinct tool names (interleaving breaks a loop).
//!
//! [`CircuitBreakReason::RepeatedToolCallLoop`]: super::CircuitBreakReason::RepeatedToolCallLoop
//! [`CircuitBreakReason::CostCapReached`]: super::CircuitBreakReason::CostCapReached
//! [`CircuitBreakReason::QualityDeclineNoRecovery`]: super::CircuitBreakReason::QualityDeclineNoRecovery
//! [`Decision::CircuitBreak`]: super::Decision::CircuitBreak
//! [`Decision::ScopeDriftWarn`]: super::Decision::ScopeDriftWarn
//! [`Decision::ProceduralWarning`]: super::Decision::ProceduralWarning
//! [`Decision::Continue`]: super::Decision::Continue
//! [`Regulator::decide`]: super::Regulator::decide
//! [`LLMEvent::ToolCall`]: super::LLMEvent::ToolCall
//! [`LLMEvent::TurnStart`]: super::LLMEvent::TurnStart

use std::collections::HashMap;

// ── Constants ──────────────────────────────────────────────────────────

/// Number of consecutive calls to the same tool that trip the loop
/// circuit-break. Chosen to balance "false alarms on a legitimate
/// 3-retry strategy" against "waste from a runaway loop."
///
/// 5 is conservative: a well-behaved agent almost never calls the
/// same tool 5× in a row without interleaving a reasoning or
/// different-tool step. Anthropic's agentic-retrieval loop
/// research observes loops typically exceed 10 repetitions when
/// they occur, so 5 catches them with margin.
pub const TOOL_LOOP_THRESHOLD: usize = 5;

// ── Records ────────────────────────────────────────────────────────────

/// One observed tool call.
#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    /// Tool name as reported by the agent (the caller owns the naming
    /// convention — `"search"`, `"db.query"`, `"exec_python"`, etc.).
    pub tool_name: String,
    /// Optional JSON-serialised args. Opaque to the regulator; stored
    /// only for observability / downstream inspection by the app.
    pub args_json: Option<String>,
    /// Index within the current turn (0-based). Resets on `TurnStart`.
    pub turn_local_index: usize,
}

/// One observed tool result.
#[derive(Debug, Clone)]
pub struct ToolResultRecord {
    pub tool_name: String,
    pub success: bool,
    pub duration_ms: u64,
    pub error_summary: Option<String>,
    pub turn_local_index: usize,
}

// ── Accumulator ────────────────────────────────────────────────────────

/// Per-turn tool-call history + loop detector.
///
/// Reset via [`ToolStatsAccumulator::reset_turn`] at every
/// `LLMEvent::TurnStart`. The regulator drives that lifecycle; callers
/// don't interact with this type directly.
#[derive(Debug, Clone, Default)]
pub struct ToolStatsAccumulator {
    /// All tool calls observed in the current turn, in emission order.
    calls: Vec<ToolCallRecord>,
    /// All tool results observed in the current turn, in emission order.
    results: Vec<ToolResultRecord>,
}

impl ToolStatsAccumulator {
    /// Fresh accumulator with empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mutable: clear per-turn history. Called by the regulator on
    /// every `LLMEvent::TurnStart` so tool-loop detection only fires
    /// within a single turn (cross-turn accumulation would conflate
    /// "stuck in a loop this turn" with "legitimately touched this
    /// tool in several adjacent turns").
    pub fn reset_turn(&mut self) {
        self.calls.clear();
        self.results.clear();
    }

    /// Mutable: append a tool-call record to the per-turn history.
    /// Requires mutation because the history is accumulated state
    /// feeding [`Self::detected_loop`] and the observability getters.
    pub fn record_call(&mut self, tool_name: String, args_json: Option<String>) {
        let turn_local_index = self.calls.len();
        self.calls.push(ToolCallRecord {
            tool_name,
            args_json,
            turn_local_index,
        });
    }

    /// Mutable: append a tool-result record to the per-turn history.
    /// Requires mutation because the history drives the observability
    /// accessors ([`Self::total_duration_ms`], [`Self::failure_count`]).
    /// Does not feed loop detection in 0.3.0.
    pub fn record_result(
        &mut self,
        tool_name: String,
        success: bool,
        duration_ms: u64,
        error_summary: Option<String>,
    ) {
        let turn_local_index = self.results.len();
        self.results.push(ToolResultRecord {
            tool_name,
            success,
            duration_ms,
            error_summary,
            turn_local_index,
        });
    }

    /// Total tool calls observed this turn.
    pub fn total_calls(&self) -> usize {
        self.calls.len()
    }

    /// Total tool results observed this turn.
    pub fn total_results(&self) -> usize {
        self.results.len()
    }

    /// Per-tool call counts this turn. Useful for observability —
    /// `{"db.query": 3, "search": 1}` style reporting.
    pub fn counts_by_tool(&self) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for call in &self.calls {
            *counts.entry(call.tool_name.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Detect a consecutive-same-tool loop.
    ///
    /// Returns `Some((tool_name, count))` when the last [`TOOL_LOOP_THRESHOLD`]
    /// calls were all the same tool. `count` is the number of consecutive
    /// same-tool calls counting back from the most recent.
    ///
    /// Returns `None` when:
    /// - Fewer than `TOOL_LOOP_THRESHOLD` calls have been observed.
    /// - The last `TOOL_LOOP_THRESHOLD` calls include at least two
    ///   distinct tools (the agent interleaved, which breaks a loop).
    pub fn detected_loop(&self) -> Option<(String, usize)> {
        if self.calls.len() < TOOL_LOOP_THRESHOLD {
            return None;
        }
        let last_tool = &self.calls.last()?.tool_name;
        // Count back from the end while the tool name matches. If we
        // hit >= TOOL_LOOP_THRESHOLD before seeing a different tool (or
        // running out), we have a loop.
        let mut consecutive = 0usize;
        for call in self.calls.iter().rev() {
            if &call.tool_name == last_tool {
                consecutive += 1;
            } else {
                break;
            }
        }
        if consecutive >= TOOL_LOOP_THRESHOLD {
            Some((last_tool.clone(), consecutive))
        } else {
            None
        }
    }

    /// Total wall-clock time spent in tool results this turn, in
    /// milliseconds. Zero when no results have been observed.
    pub fn total_duration_ms(&self) -> u64 {
        self.results.iter().map(|r| r.duration_ms).sum()
    }

    /// Count of failed results this turn.
    pub fn failure_count(&self) -> usize {
        self.results.iter().filter(|r| !r.success).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_has_no_loop() {
        let acc = ToolStatsAccumulator::new();
        assert!(acc.detected_loop().is_none());
        assert_eq!(acc.total_calls(), 0);
    }

    #[test]
    fn below_threshold_has_no_loop() {
        let mut acc = ToolStatsAccumulator::new();
        for _ in 0..(TOOL_LOOP_THRESHOLD - 1) {
            acc.record_call("search".into(), None);
        }
        assert!(acc.detected_loop().is_none());
    }

    #[test]
    fn threshold_same_tool_detects_loop() {
        let mut acc = ToolStatsAccumulator::new();
        for _ in 0..TOOL_LOOP_THRESHOLD {
            acc.record_call("search".into(), None);
        }
        let (tool, count) = acc.detected_loop().expect("loop should fire");
        assert_eq!(tool, "search");
        assert_eq!(count, TOOL_LOOP_THRESHOLD);
    }

    #[test]
    fn interleaved_tools_break_loop() {
        let mut acc = ToolStatsAccumulator::new();
        // 4 search + 1 db.query + 4 search ⇒ trailing same-tool run is 4,
        // below threshold.
        for _ in 0..4 {
            acc.record_call("search".into(), None);
        }
        acc.record_call("db.query".into(), None);
        for _ in 0..4 {
            acc.record_call("search".into(), None);
        }
        assert!(acc.detected_loop().is_none());
    }

    #[test]
    fn reset_turn_clears_state() {
        let mut acc = ToolStatsAccumulator::new();
        for _ in 0..TOOL_LOOP_THRESHOLD {
            acc.record_call("search".into(), None);
        }
        assert!(acc.detected_loop().is_some());
        acc.reset_turn();
        assert!(acc.detected_loop().is_none());
        assert_eq!(acc.total_calls(), 0);
    }

    #[test]
    fn counts_by_tool_aggregates_correctly() {
        let mut acc = ToolStatsAccumulator::new();
        acc.record_call("search".into(), None);
        acc.record_call("search".into(), None);
        acc.record_call("db.query".into(), None);
        let counts = acc.counts_by_tool();
        assert_eq!(counts.get("search"), Some(&2));
        assert_eq!(counts.get("db.query"), Some(&1));
    }

    #[test]
    fn duration_and_failure_counters() {
        let mut acc = ToolStatsAccumulator::new();
        acc.record_result("search".into(), true, 100, None);
        acc.record_result("db.query".into(), false, 250, Some("timeout".into()));
        acc.record_result("search".into(), true, 50, None);
        assert_eq!(acc.total_duration_ms(), 400);
        assert_eq!(acc.failure_count(), 1);
        assert_eq!(acc.total_results(), 3);
    }

    // ── Adversarial tests (Session 31) ─────────────────────────────
    //
    // Session 30 audit flagged the existing test suite as "staged" —
    // all cases were simple "N same-tool calls in a row" patterns.
    // These tests cover shapes that could plausibly fool the detector
    // one way or the other, and document the exact design boundary
    // ("consecutive same tool at the trailing run, by name, regardless
    // of args, ignoring historical loops that resolved").

    #[test]
    fn loop_fires_on_tail_run_after_mixed_prefix() {
        // Agent starts exploring (A, B), then falls into A×THRESHOLD
        // at the tail. Detection must look at the trailing run only —
        // the initial mixed prefix is irrelevant.
        let mut acc = ToolStatsAccumulator::new();
        acc.record_call("explore".into(), None);
        acc.record_call("db.query".into(), None);
        for _ in 0..TOOL_LOOP_THRESHOLD {
            acc.record_call("search".into(), None);
        }
        let (tool, count) = acc
            .detected_loop()
            .expect("tail same-tool run should fire regardless of prefix");
        assert_eq!(tool, "search");
        assert_eq!(count, TOOL_LOOP_THRESHOLD);
    }

    #[test]
    fn loop_fires_on_same_tool_with_diverging_args() {
        // The $47k LangChain incident shape: agent "refines" args on
        // each retry but keeps calling the same tool. Args are opaque
        // to the detector — only the tool NAME drives loop detection.
        let mut acc = ToolStatsAccumulator::new();
        for i in 0..TOOL_LOOP_THRESHOLD {
            acc.record_call(
                "search_orders".into(),
                Some(format!("{{\"user_id\":42,\"attempt\":{i}}}")),
            );
        }
        let (tool, count) = acc
            .detected_loop()
            .expect("same name + varying args is still a loop");
        assert_eq!(tool, "search_orders");
        assert_eq!(count, TOOL_LOOP_THRESHOLD);
    }

    #[test]
    fn over_threshold_returns_actual_count_not_threshold() {
        // 8 same-tool in a row → count is 8, not 5. The detector
        // reports the full run depth so apps can surface "loop depth"
        // for debugging / telemetry.
        let mut acc = ToolStatsAccumulator::new();
        for _ in 0..8 {
            acc.record_call("search".into(), None);
        }
        let (_, count) = acc.detected_loop().expect("well over threshold");
        assert_eq!(count, 8, "count reflects actual run depth, not threshold");
    }

    #[test]
    fn resolved_loop_followed_by_short_tail_does_not_fire() {
        // An earlier same-tool run that RESOLVED (agent switched
        // strategy) must not fire. Detection is on the trailing run —
        // a historical loop that the agent moved past is explicitly
        // out-of-scope.
        let mut acc = ToolStatsAccumulator::new();
        // Loop phase — but later interrupted
        for _ in 0..TOOL_LOOP_THRESHOLD {
            acc.record_call("search".into(), None);
        }
        // Agent recognizes the loop and switches
        acc.record_call("db.query".into(), None);
        acc.record_call("db.query".into(), None);
        assert!(
            acc.detected_loop().is_none(),
            "historical loops that resolved are explicitly out-of-scope"
        );
    }

    #[test]
    fn alternating_high_volume_tools_do_not_fire() {
        // 20 calls alternating A/B — agent IS stuck in a ping-pong,
        // but this is a different pathology ("A↔B ping-pong") that
        // RepeatedToolCallLoop does NOT claim to catch. Documented
        // boundary — a future CircuitBreakReason (e.g.,
        // AlternatingToolPingPong) could cover this; Noos 0.3.0
        // deliberately does not.
        let mut acc = ToolStatsAccumulator::new();
        for i in 0..20 {
            let tool = if i % 2 == 0 { "search" } else { "db.query" };
            acc.record_call(tool.into(), None);
        }
        assert!(
            acc.detected_loop().is_none(),
            "alternating A/B is not a same-tool consecutive loop by design"
        );
    }

    #[test]
    fn exact_threshold_boundary_both_sides() {
        // Paired boundary: THRESHOLD-1 silent, THRESHOLD fires. Guards
        // against an off-by-one regression.
        let mut under = ToolStatsAccumulator::new();
        for _ in 0..(TOOL_LOOP_THRESHOLD - 1) {
            under.record_call("t".into(), None);
        }
        assert!(
            under.detected_loop().is_none(),
            "THRESHOLD - 1 must not fire"
        );

        let mut at = ToolStatsAccumulator::new();
        for _ in 0..TOOL_LOOP_THRESHOLD {
            at.record_call("t".into(), None);
        }
        assert!(at.detected_loop().is_some(), "THRESHOLD must fire");
    }
}
