//! OpenTelemetry GenAI semantic conventions adapter.
//!
//! Converts OTel GenAI spans (JSON representation) into
//! [`LLMEvent`] vectors, letting teams already instrumented with the
//! stable [`gen_ai.*` semantic conventions](
//! https://opentelemetry.io/docs/specs/semconv/gen-ai/) feed
//! [`Regulator`](super::Regulator) without refactoring their agent
//! loop's event bus.
//!
//! **Scope note (P1 / P9b)**: like the other `regulator` sub-modules,
//! this is an I/O adapter, not a cognitive module. The computation is
//! JSON-value lookups against a fixed attribute vocabulary (no
//! language-specific regex, no lexicon) and a start/end-timestamp
//! subtraction. P1 applies to the wrapped
//! [`CognitiveSession`](crate::session::CognitiveSession); P9b is
//! satisfied by construction because every input field comes from the
//! OTel GenAI specification — no Noos-side text parsing.
//!
//! ## Input shape
//!
//! Expects the **SDK-idiomatic** JSON form (dict-style attributes):
//!
//! ```json
//! {
//!   "name": "chat anthropic",
//!   "attributes": {
//!     "gen_ai.system": "anthropic",
//!     "gen_ai.usage.input_tokens": 25,
//!     "gen_ai.usage.output_tokens": 800
//!   },
//!   "events": [
//!     { "name": "gen_ai.user.message",      "attributes": { "content": "Refactor fetch_user" } },
//!     { "name": "gen_ai.assistant.message", "attributes": { "content": "async fn ..." } }
//!   ],
//!   "start_time_unix_nano": 1700000000000000000,
//!   "end_time_unix_nano":   1700000000500000000
//! }
//! ```
//!
//! OTLP/JSON protobuf form (array-style `attributes[{key, value}]`) is
//! NOT supported directly — convert to dict form first with whatever
//! OTel client library you already use, or post-process with a one-line
//! transformer. Keeping the adapter restricted to one shape avoids
//! re-implementing OTel's wire-format handling here.
//!
//! ## Mapping
//!
//! | OTel signal                              | Noos [`LLMEvent`]                  |
//! |------------------------------------------|------------------------------------|
//! | event `gen_ai.user.message` `.content`   | [`TurnStart`](super::LLMEvent::TurnStart) |
//! | event `gen_ai.assistant.message` `.content` | [`TurnComplete`](super::LLMEvent::TurnComplete) |
//! | `gen_ai.usage.input_tokens` +            | [`Cost`](super::LLMEvent::Cost)    |
//! |   `gen_ai.usage.output_tokens` +         |                                    |
//! |   (end_time - start_time) / 1e6          |                                    |
//! | event `gen_ai.tool.message` with         | [`ToolCall`](super::LLMEvent::ToolCall) |
//! |   `gen_ai.tool.name`, `gen_ai.tool.arguments` |                               |
//! | event `gen_ai.tool.message` with         | [`ToolResult`](super::LLMEvent::ToolResult) |
//! |   `gen_ai.tool.name`, `gen_ai.tool.duration_ms`, |                            |
//! |   optional `error_summary`               |                                    |
//!
//! ## Event ordering
//!
//! Emitted in the order a regulator expects per turn:
//! `TurnStart` → `TurnComplete` → `Cost` → `ToolCall` / `ToolResult`.
//! This matches [`super::LLMEvent`] doc's typical-turn ordering.
//!
//! ## Unsupported (yet)
//!
//! - `gen_ai.system.message` — system prompts are not a regulator
//!   concern; dropped silently.
//! - Per-token streaming (`gen_ai.response.token` events) —
//!   [`LLMEvent::Token`] exists but most OTel instrumentation does not
//!   emit per-token logprobs, so the mapping is deferred until a
//!   concrete integration asks for it.
//! - Multi-turn spans. If your span records several conversational
//!   turns, split them at the OTel layer before calling this adapter —
//!   one span, one turn.

use serde_json::Value;

use super::LLMEvent;

/// Parse an OTel GenAI span (JSON value) into a vector of
/// [`LLMEvent`]s ready for [`Regulator::on_event`](super::Regulator::on_event).
///
/// Returns an empty vec when the span carries no recognized `gen_ai.*`
/// signals. Partial spans (cost only, messages only, etc.) produce
/// whatever subset of events is derivable — the regulator's event
/// contract is forgiving about missing pieces.
///
/// ## Example
///
/// ```
/// use noos::{regulator::otel, Regulator};
/// use serde_json::json;
///
/// let span = json!({
///     "name": "chat",
///     "attributes": {
///         "gen_ai.usage.input_tokens": 25,
///         "gen_ai.usage.output_tokens": 800
///     },
///     "events": [
///         { "name": "gen_ai.user.message", "attributes": { "content": "Refactor fetch_user" } },
///         { "name": "gen_ai.assistant.message", "attributes": { "content": "async fn..." } }
///     ],
///     "start_time_unix_nano": 1700000000000000000_u64,
///     "end_time_unix_nano":   1700000000500000000_u64,
/// });
///
/// let mut r = Regulator::for_user("alice");
/// for event in otel::events_from_span(&span) {
///     r.on_event(event);
/// }
/// ```
pub fn events_from_span(span: &Value) -> Vec<LLMEvent> {
    let mut out = Vec::new();
    let attrs = span.get("attributes").unwrap_or(&Value::Null);
    let span_events = span
        .get("events")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    // 1. TurnStart from the first user message, if any.
    if let Some(content) = first_event_content(&span_events, "gen_ai.user.message") {
        out.push(LLMEvent::TurnStart {
            user_message: content,
        });
    }

    // 2. TurnComplete from the first assistant message, if any.
    if let Some(content) = first_event_content(&span_events, "gen_ai.assistant.message") {
        out.push(LLMEvent::TurnComplete {
            full_response: content,
        });
    }

    // 3. Cost from usage attributes + span duration. Only emit when
    //    at least one token counter is present — cost with 0/0
    //    tokens is noise.
    let tokens_in = attrs
        .get("gen_ai.usage.input_tokens")
        .and_then(Value::as_u64)
        .map(|n| n as u32);
    let tokens_out = attrs
        .get("gen_ai.usage.output_tokens")
        .and_then(Value::as_u64)
        .map(|n| n as u32);

    if tokens_in.is_some() || tokens_out.is_some() {
        let wallclock_ms = span_duration_ms(span).unwrap_or(0);
        let provider = attrs
            .get("gen_ai.system")
            .and_then(Value::as_str)
            .map(String::from);
        out.push(LLMEvent::Cost {
            tokens_in: tokens_in.unwrap_or(0),
            tokens_out: tokens_out.unwrap_or(0),
            wallclock_ms,
            provider,
        });
    }

    // 4. ToolCall / ToolResult from tool message events. Each event
    //    can represent either a call, a result, or both — emit
    //    whichever is derivable from its attributes.
    for tool_event in span_events.iter().filter(|e| {
        e.get("name").and_then(Value::as_str) == Some("gen_ai.tool.message")
    }) {
        let tool_attrs = tool_event.get("attributes").unwrap_or(&Value::Null);
        let tool_name = tool_attrs
            .get("gen_ai.tool.name")
            .and_then(Value::as_str)
            .map(String::from);

        if let Some(name) = tool_name.as_ref() {
            // Call side: if arguments are present (even if empty
            // string), this event represents a call.
            if let Some(args) = tool_attrs.get("gen_ai.tool.arguments") {
                let args_json = match args {
                    Value::String(s) => Some(s.clone()),
                    Value::Null => None,
                    other => Some(other.to_string()),
                };
                out.push(LLMEvent::ToolCall {
                    tool_name: name.clone(),
                    args_json,
                });
            }
            // Result side: if duration is present, this event also
            // carries a completion signal.
            if let Some(duration_ms) = tool_attrs
                .get("gen_ai.tool.duration_ms")
                .and_then(Value::as_u64)
            {
                // Success defaults to true unless an error summary
                // is present; OTel's convention uses `error.type`
                // for failures.
                let error_summary = tool_attrs
                    .get("error.type")
                    .or_else(|| tool_attrs.get("gen_ai.tool.error"))
                    .and_then(Value::as_str)
                    .map(String::from);
                let success = error_summary.is_none();
                out.push(LLMEvent::ToolResult {
                    tool_name: name.clone(),
                    success,
                    duration_ms,
                    error_summary,
                });
            }
        }
    }

    out
}

// ── Helpers ────────────────────────────────────────────────────────

/// Pull the `content` string out of the first event with the given
/// name. Returns `None` if the event is absent or the `content`
/// attribute is missing / non-string.
fn first_event_content(events: &[Value], event_name: &str) -> Option<String> {
    for event in events {
        if event.get("name").and_then(Value::as_str) == Some(event_name) {
            if let Some(content) = event
                .get("attributes")
                .and_then(|a| a.get("content"))
                .and_then(Value::as_str)
            {
                return Some(content.to_string());
            }
        }
    }
    None
}

/// Derive a wallclock duration in milliseconds from span start / end
/// unix nanos. Returns `None` if either timestamp is missing or the
/// difference is negative (clock skew / corrupted span).
fn span_duration_ms(span: &Value) -> Option<u32> {
    let start = span.get("start_time_unix_nano").and_then(Value::as_u64)?;
    let end = span.get("end_time_unix_nano").and_then(Value::as_u64)?;
    if end < start {
        return None;
    }
    // nanos → millis. Saturate to u32::MAX on overflow (~49 days);
    // anything near that bound is almost certainly a bogus span.
    let diff_ms = (end - start) / 1_000_000;
    Some(diff_ms.min(u32::MAX as u64) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_span_yields_no_events() {
        let span = json!({});
        assert!(events_from_span(&span).is_empty());
    }

    #[test]
    fn span_with_only_random_attrs_yields_no_events() {
        let span = json!({
            "name": "chat",
            "attributes": {"http.status_code": 200}
        });
        assert!(events_from_span(&span).is_empty());
    }

    #[test]
    fn user_message_event_becomes_turn_start() {
        let span = json!({
            "events": [
                {"name": "gen_ai.user.message", "attributes": {"content": "hello"}}
            ]
        });
        let events = events_from_span(&span);
        assert_eq!(events.len(), 1);
        match &events[0] {
            LLMEvent::TurnStart { user_message } => assert_eq!(user_message, "hello"),
            other => panic!("expected TurnStart, got {other:?}"),
        }
    }

    #[test]
    fn assistant_message_event_becomes_turn_complete() {
        let span = json!({
            "events": [
                {"name": "gen_ai.assistant.message", "attributes": {"content": "world"}}
            ]
        });
        let events = events_from_span(&span);
        assert_eq!(events.len(), 1);
        match &events[0] {
            LLMEvent::TurnComplete { full_response } => assert_eq!(full_response, "world"),
            other => panic!("expected TurnComplete, got {other:?}"),
        }
    }

    #[test]
    fn usage_attributes_become_cost_event() {
        let span = json!({
            "attributes": {
                "gen_ai.system": "anthropic",
                "gen_ai.usage.input_tokens": 25_u32,
                "gen_ai.usage.output_tokens": 800_u32
            },
            "start_time_unix_nano": 1_000_000_000_u64,
            "end_time_unix_nano":   1_500_000_000_u64
        });
        let events = events_from_span(&span);
        assert_eq!(events.len(), 1);
        match &events[0] {
            LLMEvent::Cost {
                tokens_in,
                tokens_out,
                wallclock_ms,
                provider,
            } => {
                assert_eq!(*tokens_in, 25);
                assert_eq!(*tokens_out, 800);
                assert_eq!(*wallclock_ms, 500); // 0.5s
                assert_eq!(provider.as_deref(), Some("anthropic"));
            }
            other => panic!("expected Cost, got {other:?}"),
        }
    }

    #[test]
    fn missing_one_token_counter_still_emits_cost() {
        // Provider emitted only output tokens — still emit Cost with
        // input_tokens = 0. Forgiving-events contract.
        let span = json!({
            "attributes": {
                "gen_ai.usage.output_tokens": 100_u32
            }
        });
        let events = events_from_span(&span);
        assert_eq!(events.len(), 1);
        match &events[0] {
            LLMEvent::Cost {
                tokens_in: 0,
                tokens_out: 100,
                ..
            } => {}
            other => panic!("expected Cost with only output tokens, got {other:?}"),
        }
    }

    #[test]
    fn both_messages_emit_in_canonical_order() {
        let span = json!({
            "events": [
                {"name": "gen_ai.assistant.message", "attributes": {"content": "resp"}},
                {"name": "gen_ai.user.message", "attributes": {"content": "req"}}
            ]
        });
        let events = events_from_span(&span);
        assert_eq!(events.len(), 2);
        // Canonical order: TurnStart before TurnComplete even if the
        // OTel events array has them reversed.
        assert!(matches!(events[0], LLMEvent::TurnStart { .. }));
        assert!(matches!(events[1], LLMEvent::TurnComplete { .. }));
    }

    #[test]
    fn tool_call_event_emits_tool_call_llm_event() {
        let span = json!({
            "events": [
                {
                    "name": "gen_ai.tool.message",
                    "attributes": {
                        "gen_ai.tool.name": "search_orders",
                        "gen_ai.tool.arguments": "{\"user_id\": 42}"
                    }
                }
            ]
        });
        let events = events_from_span(&span);
        assert_eq!(events.len(), 1);
        match &events[0] {
            LLMEvent::ToolCall {
                tool_name,
                args_json,
            } => {
                assert_eq!(tool_name, "search_orders");
                assert_eq!(args_json.as_deref(), Some("{\"user_id\": 42}"));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn tool_event_with_duration_also_emits_tool_result() {
        let span = json!({
            "events": [
                {
                    "name": "gen_ai.tool.message",
                    "attributes": {
                        "gen_ai.tool.name": "search_orders",
                        "gen_ai.tool.arguments": "{}",
                        "gen_ai.tool.duration_ms": 120_u64
                    }
                }
            ]
        });
        let events = events_from_span(&span);
        // Both ToolCall and ToolResult are derivable from the same
        // OTel event. Emit both so the regulator gets matched pairs.
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], LLMEvent::ToolCall { .. }));
        match &events[1] {
            LLMEvent::ToolResult {
                tool_name,
                success,
                duration_ms,
                ..
            } => {
                assert_eq!(tool_name, "search_orders");
                assert!(*success);
                assert_eq!(*duration_ms, 120);
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn tool_event_with_error_type_marks_failure() {
        let span = json!({
            "events": [
                {
                    "name": "gen_ai.tool.message",
                    "attributes": {
                        "gen_ai.tool.name": "db.query",
                        "gen_ai.tool.duration_ms": 250_u64,
                        "error.type": "timeout"
                    }
                }
            ]
        });
        let events = events_from_span(&span);
        let tool_result = events
            .iter()
            .find_map(|e| {
                if let LLMEvent::ToolResult {
                    success,
                    error_summary,
                    ..
                } = e
                {
                    Some((success, error_summary))
                } else {
                    None
                }
            })
            .expect("tool result should be present");
        assert!(!*tool_result.0, "error.type should mark failure");
        assert_eq!(tool_result.1.as_deref(), Some("timeout"));
    }

    #[test]
    fn full_turn_span_produces_canonical_event_ordering() {
        // End-to-end: a realistic span with user/assistant messages +
        // usage attrs + one tool call + one tool result. Verify the
        // resulting event sequence is in turn-canonical order
        // (TurnStart → TurnComplete → Cost → ToolCall → ToolResult).
        let span = json!({
            "name": "agent.turn",
            "attributes": {
                "gen_ai.system": "openai",
                "gen_ai.usage.input_tokens": 50_u32,
                "gen_ai.usage.output_tokens": 400_u32
            },
            "events": [
                {"name": "gen_ai.user.message", "attributes": {"content": "find order 42"}},
                {
                    "name": "gen_ai.tool.message",
                    "attributes": {
                        "gen_ai.tool.name": "search_orders",
                        "gen_ai.tool.arguments": "{\"id\": 42}",
                        "gen_ai.tool.duration_ms": 80_u64
                    }
                },
                {"name": "gen_ai.assistant.message", "attributes": {"content": "order found"}}
            ],
            "start_time_unix_nano": 2_000_000_000_u64,
            "end_time_unix_nano":   2_750_000_000_u64
        });
        let events = events_from_span(&span);
        // Expected: [TurnStart, TurnComplete, Cost, ToolCall, ToolResult]
        assert_eq!(events.len(), 5, "events emitted: {events:?}");
        assert!(matches!(events[0], LLMEvent::TurnStart { .. }));
        assert!(matches!(events[1], LLMEvent::TurnComplete { .. }));
        assert!(matches!(events[2], LLMEvent::Cost { .. }));
        assert!(matches!(events[3], LLMEvent::ToolCall { .. }));
        assert!(matches!(events[4], LLMEvent::ToolResult { .. }));
    }

    #[test]
    fn negative_duration_clamps_to_none_then_zero() {
        // end < start — clock skew / broken span. Must not panic.
        let span = json!({
            "attributes": {"gen_ai.usage.output_tokens": 100_u32},
            "start_time_unix_nano": 1_500_000_000_u64,
            "end_time_unix_nano":   1_000_000_000_u64
        });
        let events = events_from_span(&span);
        assert_eq!(events.len(), 1);
        if let LLMEvent::Cost { wallclock_ms, .. } = &events[0] {
            // Corrupted duration → 0, not negative or panic.
            assert_eq!(*wallclock_ms, 0);
        } else {
            panic!("expected Cost event");
        }
    }

    #[test]
    fn end_to_end_feeds_regulator_cleanly() {
        // Integration smoke test: an OTel span flows through the
        // adapter and into a Regulator without errors. The Decision
        // outcome isn't asserted — this is a plumbing test.
        use crate::Regulator;

        let span = json!({
            "attributes": {
                "gen_ai.usage.input_tokens": 25_u32,
                "gen_ai.usage.output_tokens": 800_u32
            },
            "events": [
                {"name": "gen_ai.user.message", "attributes": {"content": "hello"}},
                {"name": "gen_ai.assistant.message", "attributes": {"content": "hi back"}}
            ],
            "start_time_unix_nano": 1_u64,
            "end_time_unix_nano":   500_000_000_u64
        });

        let mut r = Regulator::for_user("otel_user");
        for event in events_from_span(&span) {
            r.on_event(event);
        }
        // Continue is fine — the adapter plumbed through, that's all
        // we're asserting.
        let _ = r.decide();
    }
}
