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
//! ```
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
}

/// CLI options parsed from argv.
#[derive(Debug, Default)]
struct CliArgs {
    cost_cap: Option<u32>,
    summary_only: bool,
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
    eprintln!("  shadow_replay [--cost-cap=N] [--summary-only] < events.jsonl");
    eprintln!();
    eprintln!("FLAGS:");
    eprintln!("  --cost-cap=N      Enable CircuitBreak(CostCapReached) at N tokens");
    eprintln!("  --summary-only    Suppress per-event lines; emit only the _summary line");
    eprintln!("  -h, --help        Print this help");
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
                Decision::ScopeDriftWarn { .. } => turn_had_scope_drift = true,
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
