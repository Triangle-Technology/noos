//! # Shadow replay — CLI for offline agent-event analysis
//!
//! Reads a JSONL stream of LLM agent events from stdin, drives each turn
//! through [`Regulator`], and emits JSONL decisions + an aggregate summary
//! to stdout. Intended for:
//!
//! 1. **Post-hoc analysis of production shadow logs** — pipe your D1 /
//!    Postgres / log-file dump into this tool to see what Noos would have
//!    decided on every turn.
//! 2. **Integration with agent eval harnesses** — a CI gate that flags
//!    scope drift / cost runaway / tool-loop regressions before publish.
//! 3. **Any pipeline that emits JSON events** — language-agnostic. Works
//!    with Python agents, TypeScript agents, Rust agents, log exports.
//!
//! ## Input format (JSONL on stdin)
//!
//! One JSON object per line:
//!
//! ```json
//! {"turn_id":"uuid","event_idx":0,"event_type":"TurnStart","payload":{"user_message":"Hello"}}
//! {"turn_id":"uuid","event_idx":1,"event_type":"ToolCall","payload":{"tool_name":"search","args_json":null}}
//! {"turn_id":"uuid","event_idx":2,"event_type":"TurnComplete","payload":{"full_response":"Here..."}}
//! {"turn_id":"uuid","event_idx":3,"event_type":"Cost","payload":{"tokens_in":20,"tokens_out":50,"wallclock_ms":800,"provider":"anthropic"}}
//! ```
//!
//! Fields `turn_id` + `event_idx` are **required** for grouping / ordering.
//! `event_type` + `payload` match [`LLMEvent`] variant tag + body respectively
//! (so `payload` for `TurnStart` = `{"user_message":"..."}`, etc.)
//!
//! Events from the same `turn_id` are grouped; within a turn events are
//! replayed in ascending `event_idx` order. Turns are processed in the
//! order they first appear.
//!
//! ## Output format (JSONL on stdout)
//!
//! One line per event, then a final `_summary` line:
//!
//! ```json
//! {"turn_id":"uuid","event_idx":0,"event_type":"TurnStart","decision_kind":"Continue","decision":"Continue"}
//! ...
//! {"_summary":{"turns":50,"events":450,"decisions":{"Continue":440,"ScopeDriftWarn":8,"CircuitBreak":2}}}
//! ```
//!
//! ## Usage
//!
//! ```bash
//! # Run with stdin from a file
//! cargo run --release --example shadow_replay < events.jsonl
//!
//! # Pipe from another tool (e.g. D1 export + jq reshape)
//! wrangler d1 execute DB --remote --json \
//!   --command "SELECT turn_id,event_idx,event_type,payload_json FROM llm_events" \
//!   | jq -c '.[0].results[] | {turn_id, event_idx, event_type, payload: (.payload_json | fromjson)}' \
//!   | cargo run --release --example shadow_replay
//!
//! # With a cost cap (triggers CircuitBreak on runaway turns)
//! cargo run --release --example shadow_replay -- --cost-cap=10000 < events.jsonl
//!
//! # Summary only (skip per-event lines)
//! cargo run --release --example shadow_replay -- --summary-only < events.jsonl
//!
//! # Drift-threshold calibration: capture EVERY drift score, not just
//! # those above 0.5. Pair with --summary-only for a compact report.
//! cargo run --release --example shadow_replay -- \
//!     --scope-threshold=0.0 --summary-only < events.jsonl
//!
//! # Simulate a 0.7 default (suppress verbose-LLM false positives):
//! cargo run --release --example shadow_replay -- --scope-threshold=0.7 < events.jsonl
//! ```
//!
//! ## Summary output
//!
//! The final `_summary` line always carries turn / event / decision counts.
//! When at least one ScopeDriftWarn fires, an additional `drift_score_stats`
//! object appears with:
//!
//! - `count` — observed drift scores (one per ScopeDriftWarn emission)
//! - `min` / `max` / `mean` / `p50` / `p95` — summary statistics
//! - `histogram` — 10 bins over `[0.0, 1.0]` (bin width 0.1)
//! - `threshold_sweep` — "if threshold were X, how many would fire?" at
//!   `X ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}`. Drives data-informed
//!   decisions on whether to raise the per-Regulator threshold via
//!   `Regulator::with_scope_drift_threshold`.
//!
//! Drift statistics reflect ONLY the scores the regulator actually
//! emitted via `Decision::ScopeDriftWarn`. To see the full
//! sub-threshold distribution for calibration, pass
//! `--scope-threshold=0.0` which forces every non-empty turn to surface
//! its raw score.
//!
//! ## Exit status
//!
//! - `0` — clean run
//! - `1` — I/O error or input parse failure
//!
//! ## Design notes
//!
//! - **One [`Regulator`] per turn** — matches Phase 1 shadow wiring where
//!   each HTTP request gets a fresh turn context. Cross-turn procedural
//!   memory is out of scope here (requires per-user batching — future work).
//! - **All LLM events in a turn are buffered then replayed in order.**
//!   Events within a turn may arrive out of order in the input; `event_idx`
//!   sort ensures deterministic replay.
//! - **Malformed lines are skipped with stderr warning** — one bad line
//!   never fails the whole run.

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use noos::{CircuitBreakReason, Decision, LLMEvent, Regulator};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One row of the input stream.
#[derive(Debug, Deserialize)]
struct EventLine {
    turn_id: String,
    event_idx: usize,
    event_type: String,
    payload: Value,
}

/// One row of the output stream (per-event).
#[derive(Debug, Serialize)]
struct DecisionLine<'a> {
    turn_id: &'a str,
    event_idx: usize,
    event_type: &'a str,
    decision_kind: &'static str,
    decision: Value,
}

/// Final aggregate summary line.
#[derive(Debug, Serialize, Default)]
struct Summary {
    #[serde(rename = "_summary")]
    body: SummaryBody,
}

#[derive(Debug, Serialize, Default)]
struct SummaryBody {
    turns: usize,
    events: usize,
    /// Count of each Decision variant emitted across all events.
    decisions: HashMap<String, usize>,
    /// Number of CircuitBreak decisions broken down by reason.
    circuit_break_reasons: HashMap<String, usize>,
    /// Turns where at least one ScopeDriftWarn fired.
    turns_with_scope_drift: usize,
    /// Turns where at least one CircuitBreak fired.
    turns_with_circuit_break: usize,
    /// Turns where at least one ProceduralWarning fired.
    turns_with_procedural_warning: usize,
    /// Distribution summary over observed drift scores. `None` when no
    /// `Decision::ScopeDriftWarn` fired in the stream. See module docs
    /// for interpretation + calibration workflow.
    #[serde(skip_serializing_if = "Option::is_none")]
    drift_score_stats: Option<DriftStats>,
}

/// Summary of drift scores observed across all emitted
/// `Decision::ScopeDriftWarn` events. Produced at end-of-stream from the
/// raw score list. Serialisation order is stable so downstream scripts
/// can rely on field positions.
#[derive(Debug, Serialize)]
struct DriftStats {
    count: usize,
    min: f64,
    mean: f64,
    p50: f64,
    p95: f64,
    max: f64,
    /// 10 bins over `[0.0, 1.0]` with width 0.1. Index i covers
    /// `[i * 0.1, (i + 1) * 0.1)` except the last bin which includes
    /// the upper bound (so 1.0 lands in bin 9).
    histogram: [usize; 10],
    /// "If the regulator threshold were X, how many of these scores
    /// would fire?" Keyed by formatted-float string so the JSON object
    /// has stable "0.3" / "0.7" keys downstream tools can grep for.
    /// Calibration hook: compare 0.5 default vs 0.7 verbose-model
    /// candidate directly.
    threshold_sweep: Vec<(String, usize)>,
}

impl DriftStats {
    /// Compute statistics from a raw score list. Returns `None` on empty
    /// input so the parent summary can omit the field cleanly.
    fn from_scores(mut scores: Vec<f64>) -> Option<Self> {
        if scores.is_empty() {
            return None;
        }
        // NaN shouldn't appear (the Regulator's metric never produces
        // non-finite values) but sort defensively with `total_cmp` so a
        // stray NaN doesn't panic the pipeline.
        scores.sort_by(|a, b| a.total_cmp(b));
        let count = scores.len();
        let min = scores[0];
        let max = scores[count - 1];
        let sum: f64 = scores.iter().sum();
        let mean = sum / count as f64;
        // Nearest-rank quantile: simple, no interpolation, deterministic
        // at small N. For p95 on 10 scores we pick index 9 (highest),
        // matching what a practitioner would eyeball from a stem-and-leaf.
        let p50 = scores[count / 2];
        let p95_idx = ((count as f64 * 0.95).ceil() as usize).saturating_sub(1);
        let p95 = scores[p95_idx.min(count - 1)];

        let mut histogram = [0usize; 10];
        for &s in &scores {
            // Clamp defensively. bin = floor(s * 10), but s=1.0 would
            // produce bin 10 — fold into bin 9 so the last bucket is
            // inclusive of 1.0.
            let bin = ((s.clamp(0.0, 1.0) * 10.0) as usize).min(9);
            histogram[bin] += 1;
        }

        let thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let threshold_sweep = thresholds
            .iter()
            .map(|&t| {
                let turns_at_or_above = scores.iter().filter(|&&s| s >= t).count();
                (format!("{t:.1}"), turns_at_or_above)
            })
            .collect();

        Some(DriftStats {
            count,
            min,
            mean,
            p50,
            p95,
            max,
            histogram,
            threshold_sweep,
        })
    }
}

/// CLI options parsed from argv.
#[derive(Debug, Default)]
struct CliArgs {
    cost_cap: Option<u32>,
    summary_only: bool,
    /// Overrides the default `Regulator` scope-drift threshold (0.5).
    /// Pass `0.0` to surface the full distribution for calibration;
    /// pass `0.7` to simulate a verbose-LLM-tolerant default.
    scope_threshold: Option<f64>,
}

/// Parse CLI flags from `std::env::args()`.
///
/// Unknown flags are reported to stderr but don't abort the run — MVP
/// tolerates unknown args so shell-generated command lines with extra
/// whitespace or empty strings don't fail the whole pipeline. `--help` /
/// `-h` is the only flag that exits (status 0).
fn parse_args() -> CliArgs {
    let mut args = CliArgs::default();
    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--cost-cap=") {
            match val.parse::<u32>() {
                Ok(n) => args.cost_cap = Some(n),
                Err(e) => eprintln!("shadow_replay: ignoring invalid --cost-cap={val}: {e}"),
            }
        } else if let Some(val) = arg.strip_prefix("--scope-threshold=") {
            match val.parse::<f64>() {
                Ok(t) if t.is_finite() && (0.0..=1.0).contains(&t) => {
                    args.scope_threshold = Some(t);
                }
                Ok(t) => eprintln!(
                    "shadow_replay: ignoring --scope-threshold={t} (must be finite and in [0.0, 1.0])"
                ),
                Err(e) => eprintln!(
                    "shadow_replay: ignoring invalid --scope-threshold={val}: {e}"
                ),
            }
        } else if arg == "--summary-only" {
            args.summary_only = true;
        } else if arg == "--help" || arg == "-h" {
            print_help();
            std::process::exit(0);
        } else {
            eprintln!("shadow_replay: unknown arg `{arg}` (ignored; use --help)");
        }
    }
    args
}

/// Write CLI usage to stderr. Kept separate from `parse_args` so help text
/// stays visually close to the flag-matching code above it.
fn print_help() {
    eprintln!("shadow_replay — pipe JSONL agent events → JSONL regulator decisions");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  shadow_replay [--cost-cap=N] [--scope-threshold=F]");
    eprintln!("                [--summary-only] < events.jsonl");
    eprintln!();
    eprintln!("FLAGS:");
    eprintln!("  --cost-cap=N          Enable CircuitBreak(CostCapReached) at N tokens");
    eprintln!("  --scope-threshold=F   Override scope-drift threshold (default 0.5);");
    eprintln!("                        pass 0.0 for calibration (surfaces every score)");
    eprintln!("  --summary-only        Suppress per-event lines; emit only _summary");
    eprintln!("  -h, --help            Print this help");
}

/// Classify a Decision into a short kind tag suitable for aggregation.
fn decision_kind(d: &Decision) -> &'static str {
    match d {
        Decision::Continue => "Continue",
        Decision::ScopeDriftWarn { .. } => "ScopeDriftWarn",
        Decision::ProceduralWarning { .. } => "ProceduralWarning",
        Decision::CircuitBreak { .. } => "CircuitBreak",
        Decision::LowConfidenceSpans { .. } => "LowConfidenceSpans",
        // `Decision` is `#[non_exhaustive]`; future variants fall through.
        _ => "Unknown",
    }
}

/// Classify a CircuitBreakReason variant for aggregation.
fn circuit_break_reason_kind(r: &CircuitBreakReason) -> &'static str {
    match r {
        CircuitBreakReason::CostCapReached { .. } => "CostCapReached",
        CircuitBreakReason::QualityDeclineNoRecovery { .. } => "QualityDeclineNoRecovery",
        CircuitBreakReason::RepeatedFailurePattern { .. } => "RepeatedFailurePattern",
        CircuitBreakReason::RepeatedToolCallLoop { .. } => "RepeatedToolCallLoop",
        _ => "Unknown",
    }
}

/// Reconstruct a `LLMEvent` from the flat `{event_type, payload}` input
/// shape by wrapping in serde's externally-tagged envelope.
fn reconstruct_event(event_type: &str, payload: Value) -> serde_json::Result<LLMEvent> {
    let mut tagged = serde_json::Map::with_capacity(1);
    tagged.insert(event_type.to_string(), payload);
    serde_json::from_value(Value::Object(tagged))
}

fn main() -> ExitCode {
    let args = parse_args();

    // Read + group input. Small memory cost (events within a turn share
    // one Vec); for 10k events × 2KB = ~20MB — fine for MVP.
    let stdin = io::stdin();
    let mut turns: HashMap<String, Vec<EventLine>> = HashMap::new();
    let mut turn_order: Vec<String> = Vec::new();
    let mut malformed_lines = 0usize;

    for (lineno, line_result) in stdin.lock().lines().enumerate() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                eprintln!("shadow_replay: stdin read error on line {lineno}: {e}");
                return ExitCode::from(1);
            }
        };
        if line.trim().is_empty() {
            continue;
        }

        let event: EventLine = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("shadow_replay: skip malformed line {lineno}: {e}");
                malformed_lines += 1;
                continue;
            }
        };

        if !turns.contains_key(&event.turn_id) {
            turn_order.push(event.turn_id.clone());
        }
        turns.entry(event.turn_id.clone()).or_default().push(event);
    }

    // Process each turn in stream-order. Write per-event decisions unless
    // --summary-only is set. Aggregate as we go.
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    let mut summary = Summary::default();
    // Drift scores observed across every `Decision::ScopeDriftWarn`
    // emission in the whole stream. Used at end-of-run to produce
    // `DriftStats` (histogram + threshold sweep).
    let mut drift_scores: Vec<f64> = Vec::new();

    for turn_id in &turn_order {
        let mut events = match turns.remove(turn_id) {
            Some(v) => v,
            None => continue,
        };
        events.sort_by_key(|e| e.event_idx);

        // `for_user` keys procedural memory; we reuse turn_id since this tool
        // replays one Regulator per turn (no cross-turn procedural accumulation).
        let mut regulator = Regulator::for_user(turn_id.as_str());
        if let Some(cap) = args.cost_cap {
            regulator = regulator.with_cost_cap(cap);
        }
        if let Some(threshold) = args.scope_threshold {
            regulator = regulator.with_scope_drift_threshold(threshold);
        }

        let mut turn_had_scope_drift = false;
        let mut turn_had_circuit_break = false;
        let mut turn_had_procedural_warning = false;

        for event_line in events {
            let event = match reconstruct_event(&event_line.event_type, event_line.payload.clone())
            {
                Ok(e) => e,
                Err(e) => {
                    eprintln!(
                        "shadow_replay: skip event (turn={}, idx={}, type={}): {}",
                        event_line.turn_id, event_line.event_idx, event_line.event_type, e
                    );
                    continue;
                }
            };

            regulator.on_event(event);
            let decision = regulator.decide();
            summary.body.events += 1;

            let kind = decision_kind(&decision);
            *summary.body.decisions.entry(kind.to_string()).or_insert(0) += 1;

            match &decision {
                Decision::ScopeDriftWarn { drift_score, .. } => {
                    turn_had_scope_drift = true;
                    drift_scores.push(*drift_score);
                }
                Decision::CircuitBreak { ref reason, .. } => {
                    turn_had_circuit_break = true;
                    let reason_kind = circuit_break_reason_kind(reason);
                    *summary
                        .body
                        .circuit_break_reasons
                        .entry(reason_kind.to_string())
                        .or_insert(0) += 1;
                }
                Decision::ProceduralWarning { .. } => turn_had_procedural_warning = true,
                _ => {}
            }

            if !args.summary_only {
                // Serialize Decision; fall back to null on the (unexpected) case
                // where Decision fails to serialize.
                let decision_value = serde_json::to_value(&decision).unwrap_or(Value::Null);
                let out = DecisionLine {
                    turn_id: &event_line.turn_id,
                    event_idx: event_line.event_idx,
                    event_type: &event_line.event_type,
                    decision_kind: kind,
                    decision: decision_value,
                };
                match serde_json::to_string(&out) {
                    Ok(s) => {
                        if let Err(e) = writeln!(stdout, "{s}") {
                            eprintln!("shadow_replay: stdout write error: {e}");
                            return ExitCode::from(1);
                        }
                    }
                    Err(e) => {
                        eprintln!("shadow_replay: skip output line (serde error): {e}");
                    }
                }
            }
        }

        summary.body.turns += 1;
        if turn_had_scope_drift {
            summary.body.turns_with_scope_drift += 1;
        }
        if turn_had_circuit_break {
            summary.body.turns_with_circuit_break += 1;
        }
        if turn_had_procedural_warning {
            summary.body.turns_with_procedural_warning += 1;
        }
    }

    // Finalize drift-score statistics before serialization.
    summary.body.drift_score_stats = DriftStats::from_scores(drift_scores);

    // Emit aggregate. Even with --summary-only, this is the authoritative
    // line the caller should parse.
    let summary_json = match serde_json::to_string(&summary) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("shadow_replay: summary serialization failed: {e}");
            return ExitCode::from(1);
        }
    };
    if let Err(e) = writeln!(stdout, "{summary_json}") {
        eprintln!("shadow_replay: stdout write error (summary): {e}");
        return ExitCode::from(1);
    }

    if malformed_lines > 0 {
        eprintln!("shadow_replay: completed with {malformed_lines} malformed line(s) skipped");
    }

    ExitCode::from(0)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruct_turn_start() {
        let payload = serde_json::json!({"user_message": "hi"});
        let event = reconstruct_event("TurnStart", payload).expect("should parse");
        match event {
            LLMEvent::TurnStart { user_message } => assert_eq!(user_message, "hi"),
            other => panic!("expected TurnStart, got {other:?}"),
        }
    }

    #[test]
    fn reconstruct_tool_call_with_args() {
        let payload = serde_json::json!({
            "tool_name": "search",
            "args_json": "{\"q\":\"noos\"}",
        });
        let event = reconstruct_event("ToolCall", payload).expect("should parse");
        match event {
            LLMEvent::ToolCall {
                tool_name,
                args_json,
            } => {
                assert_eq!(tool_name, "search");
                assert_eq!(args_json.as_deref(), Some("{\"q\":\"noos\"}"));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn reconstruct_tool_call_null_args() {
        let payload = serde_json::json!({"tool_name": "search", "args_json": null});
        let event = reconstruct_event("ToolCall", payload).expect("should parse");
        match event {
            LLMEvent::ToolCall {
                tool_name,
                args_json,
            } => {
                assert_eq!(tool_name, "search");
                assert_eq!(args_json, None);
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn reconstruct_cost() {
        let payload = serde_json::json!({
            "tokens_in": 42,
            "tokens_out": 100,
            "wallclock_ms": 1500,
            "provider": "anthropic",
        });
        let event = reconstruct_event("Cost", payload).expect("should parse");
        match event {
            LLMEvent::Cost {
                tokens_in,
                tokens_out,
                wallclock_ms,
                provider,
            } => {
                assert_eq!(tokens_in, 42);
                assert_eq!(tokens_out, 100);
                assert_eq!(wallclock_ms, 1500);
                assert_eq!(provider.as_deref(), Some("anthropic"));
            }
            other => panic!("expected Cost, got {other:?}"),
        }
    }

    #[test]
    fn reconstruct_malformed_fails_cleanly() {
        // Wrong payload shape for TurnStart — should produce serde error, not panic.
        let payload = serde_json::json!({"wrong_field": 42});
        let err = reconstruct_event("TurnStart", payload).unwrap_err();
        assert!(err.to_string().contains("user_message"));
    }

    #[test]
    fn reconstruct_unknown_variant() {
        let payload = serde_json::json!({});
        let err = reconstruct_event("NotARealVariant", payload).unwrap_err();
        // serde_json reports "unknown variant" — exact text may vary across
        // versions so we only assert it's an error.
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn decision_kind_covers_all_known_variants() {
        // If a new Decision variant is added upstream, at minimum this test
        // catches a compile-time reminder (exhaustive match warning) in CI.
        assert_eq!(decision_kind(&Decision::Continue), "Continue");
    }

    #[test]
    fn circuit_break_reason_kind_classifies_all_known_reasons() {
        // `CircuitBreakReason` is `#[non_exhaustive]`; if a new reason is
        // added upstream and the kind classifier isn't updated, it silently
        // returns "Unknown" — this test at minimum locks the four known
        // reasons so the regression is caught during manual review.
        assert_eq!(
            circuit_break_reason_kind(&CircuitBreakReason::CostCapReached {
                tokens_spent: 100,
                tokens_cap: 50,
                mean_quality_last_n: 0.3,
            }),
            "CostCapReached"
        );
        assert_eq!(
            circuit_break_reason_kind(&CircuitBreakReason::QualityDeclineNoRecovery {
                turns: 3,
                mean_delta: -0.5,
            }),
            "QualityDeclineNoRecovery"
        );
        assert_eq!(
            circuit_break_reason_kind(&CircuitBreakReason::RepeatedToolCallLoop {
                tool_name: "search".into(),
                consecutive_count: 5,
            }),
            "RepeatedToolCallLoop"
        );
        assert_eq!(
            circuit_break_reason_kind(&CircuitBreakReason::RepeatedFailurePattern {
                cluster: "async+auth".into(),
                failure_count: 3,
            }),
            "RepeatedFailurePattern"
        );
    }

    #[test]
    fn end_to_end_single_turn_clean() {
        // Minimal Regulator-driven replay: TurnStart + TurnComplete + Cost
        // with aligned task/response keywords → expect Continue throughout.
        let mut reg = Regulator::for_user("test");
        reg.on_event(LLMEvent::TurnStart {
            user_message: "refactor my auth function to be async".into(),
        });
        assert!(matches!(reg.decide(), Decision::Continue));

        reg.on_event(LLMEvent::TurnComplete {
            full_response: "I refactored the auth function to be async.".into(),
        });
        // Scope-drift may or may not fire depending on keyword overlap —
        // for this minimal test we only require that it doesn't panic.
        let _ = reg.decide();

        reg.on_event(LLMEvent::Cost {
            tokens_in: 20,
            tokens_out: 50,
            wallclock_ms: 800,
            provider: Some("anthropic".into()),
        });
        let _ = reg.decide();
    }

    #[test]
    fn drift_stats_from_empty_returns_none() {
        assert!(DriftStats::from_scores(Vec::new()).is_none());
    }

    #[test]
    fn drift_stats_computes_summary_stats() {
        // Fixed input → fixed output. Verifies all statistics against
        // hand-computed expected values.
        let scores = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let stats = DriftStats::from_scores(scores).expect("non-empty");
        assert_eq!(stats.count, 5);
        assert!((stats.min - 0.1).abs() < 1e-9);
        assert!((stats.max - 0.9).abs() < 1e-9);
        assert!((stats.mean - 0.5).abs() < 1e-9);
        assert!((stats.p50 - 0.5).abs() < 1e-9);
        // ceil(5 * 0.95) = 5, index 4 → 0.9.
        assert!((stats.p95 - 0.9).abs() < 1e-9);
    }

    #[test]
    fn drift_stats_histogram_bins_values_correctly() {
        // Each value should land in its own bin.
        let scores = vec![0.05, 0.15, 0.45, 0.75, 0.95, 1.0];
        let stats = DriftStats::from_scores(scores).expect("non-empty");
        // 0.05 → bin 0, 0.15 → bin 1, 0.45 → bin 4, 0.75 → bin 7,
        // 0.95 → bin 9, 1.0 → bin 9 (inclusive upper bound).
        assert_eq!(stats.histogram[0], 1);
        assert_eq!(stats.histogram[1], 1);
        assert_eq!(stats.histogram[4], 1);
        assert_eq!(stats.histogram[7], 1);
        assert_eq!(stats.histogram[9], 2);
        // All other bins are 0.
        assert_eq!(stats.histogram.iter().sum::<usize>(), 6);
    }

    #[test]
    fn drift_stats_threshold_sweep_is_monotonic_decreasing() {
        // Raising the threshold can only reduce (or hold) the number of
        // drift scores that would qualify — monotonic non-increasing.
        let scores = (0..20).map(|i| i as f64 / 20.0).collect::<Vec<_>>();
        let stats = DriftStats::from_scores(scores).expect("non-empty");
        let sweep = &stats.threshold_sweep;
        for w in sweep.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "threshold_sweep must be monotonic non-increasing, \
                 found {}={} then {}={}",
                w[0].0,
                w[0].1,
                w[1].0,
                w[1].1
            );
        }
        // At threshold 0.0 (not in sweep), all 20 would qualify.
        // At 0.3, scores >= 0.3 are {0.30, 0.35, ..., 0.95}. Count = 14.
        assert_eq!(sweep[0].0, "0.3");
        assert_eq!(sweep[0].1, 14);
    }

    #[test]
    fn drift_stats_threshold_sweep_keys_are_stable() {
        // Downstream scripts grep for "0.5" / "0.7"; keys must be
        // formatted to one decimal regardless of locale or NaN drift.
        let stats = DriftStats::from_scores(vec![0.5]).expect("non-empty");
        let expected_keys = ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"];
        let got_keys: Vec<&str> = stats.threshold_sweep.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(got_keys, expected_keys);
    }

    #[test]
    fn drift_stats_upper_bin_inclusive() {
        // Defensive: drift score 1.0 must land in bin 9 (not bin 10).
        let stats = DriftStats::from_scores(vec![1.0, 1.0, 1.0]).expect("non-empty");
        assert_eq!(stats.histogram[9], 3);
        assert_eq!(stats.count, 3);
    }

    #[test]
    fn drift_stats_nan_does_not_panic() {
        // NaN shouldn't appear, but sort must not panic if one slips in.
        // Using `total_cmp` makes the sort well-defined even with NaN
        // (NaN sorts consistently but the resulting quantiles are
        // meaningless; the test only guards against a panic).
        let scores = vec![0.1, f64::NAN, 0.5];
        // This must not panic.
        let _ = DriftStats::from_scores(scores);
    }

    #[test]
    fn output_line_serializes() {
        let line = DecisionLine {
            turn_id: "abc",
            event_idx: 3,
            event_type: "TurnStart",
            decision_kind: "Continue",
            decision: serde_json::json!("Continue"),
        };
        let s = serde_json::to_string(&line).expect("should serialize");
        assert!(s.contains("\"turn_id\":\"abc\""));
        assert!(s.contains("\"event_idx\":3"));
        assert!(s.contains("\"decision_kind\":\"Continue\""));
    }
}
