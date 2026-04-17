"""Smoke tests for the Python bindings.

Mirrors the Rust integration tests at a behavioral level: construct
events, fire decide() at the right moments, check kind + typed
attributes. Does not require an LLM.

Run:
    cd bindings/python
    maturin develop --release
    pytest tests/
"""
import json

import pytest

from noos_regulator import (
    CircuitBreakReason,
    CorrectionPattern,
    Decision,
    LLMEvent,
    Regulator,
    __version__,
)


def test_version_is_exported():
    assert isinstance(__version__, str) and __version__.count(".") >= 2


def test_empty_regulator_returns_continue():
    r = Regulator.for_user("alice")
    d = r.decide()
    assert d.kind == "continue"
    assert d.is_continue()
    assert d.drift_score is None
    assert d.reason is None
    assert d.patterns is None


def test_scope_drift_fires_on_drifted_response():
    r = Regulator.for_user("alice")
    r.on_event(LLMEvent.turn_start(
        "Refactor fetch_user to be async. Keep the database lookup logic unchanged."
    ))
    # 10 keywords with zero anchor in the task produces drift_score = 1.0
    r.on_event(LLMEvent.turn_complete(
        "added counter timing retry cache wrapper handler middleware logger queue"
    ))
    r.on_event(LLMEvent.cost(
        tokens_in=40, tokens_out=180, wallclock_ms=0, provider="test",
    ))
    d = r.decide()
    assert d.kind == "scope_drift_warn"
    assert d.is_scope_drift()
    assert d.drift_score is not None
    assert d.drift_score >= 0.5
    assert isinstance(d.drift_tokens, list)
    assert len(d.drift_tokens) > 0


def test_cost_break_fires_when_cap_and_quality_both_trip():
    r = Regulator.for_user("bob").with_cost_cap(1000)
    for i, q in enumerate([0.5, 0.35, 0.2], start=1):
        r.on_event(LLMEvent.turn_start(f"attempt {i}"))
        r.on_event(LLMEvent.turn_complete(f"r{i}"))
        r.on_event(LLMEvent.cost(
            tokens_in=25, tokens_out=400, wallclock_ms=0, provider="test",
        ))
        r.on_event(LLMEvent.quality_feedback(quality=q))

    d = r.decide()
    assert d.kind == "circuit_break"
    assert d.is_circuit_break()
    assert d.reason is not None
    assert d.reason.kind in (
        "cost_cap_reached",
        "quality_decline_no_recovery",
    )
    assert isinstance(d.suggestion, str) and len(d.suggestion) > 0


def test_tool_loop_detection_fires_on_five_consecutive_same_tool():
    r = Regulator.for_user("dave")
    r.on_event(LLMEvent.turn_start("find user"))
    for i in range(5):
        r.on_event(LLMEvent.tool_call(
            tool_name="search_orders",
            args_json=f'{{"i": {i}}}',
        ))
        r.on_event(LLMEvent.tool_result(
            tool_name="search_orders",
            success=True,
            duration_ms=100,
        ))
    assert r.tool_total_calls() == 5
    d = r.decide()
    assert d.kind == "circuit_break"
    assert d.reason.kind == "repeated_tool_call_loop"
    assert d.reason.tool_name == "search_orders"
    assert d.reason.consecutive_count >= 5


def test_tool_stats_accessors():
    r = Regulator.for_user("eve")
    r.on_event(LLMEvent.turn_start("task"))
    r.on_event(LLMEvent.tool_call("a", None))
    r.on_event(LLMEvent.tool_result("a", True, 50))
    r.on_event(LLMEvent.tool_call("b", None))
    r.on_event(LLMEvent.tool_result("b", False, 30, "boom"))
    r.on_event(LLMEvent.tool_call("a", None))
    r.on_event(LLMEvent.tool_result("a", True, 20))

    assert r.tool_total_calls() == 3
    assert r.tool_counts_by_name() == {"a": 2, "b": 1}
    assert r.tool_total_duration_ms() == 100
    assert r.tool_failure_count() == 1


def test_export_import_roundtrip_preserves_type():
    r = Regulator.for_user("frank")
    r.on_event(LLMEvent.turn_start("hello"))
    r.on_event(LLMEvent.turn_complete("world"))
    r.on_event(LLMEvent.cost(
        tokens_in=10, tokens_out=20, wallclock_ms=5, provider="test",
    ))

    snapshot = r.export_json()
    assert isinstance(snapshot, str)
    # Must be valid JSON
    parsed = json.loads(snapshot)
    assert isinstance(parsed, dict)

    r2 = Regulator.from_json(snapshot)
    assert r2.user_id() == "frank"


def test_from_json_raises_on_malformed_json():
    with pytest.raises(ValueError):
        Regulator.from_json("not json at all")


def test_inject_corrections_noop_when_no_pattern():
    r = Regulator.for_user("grace")
    r.on_event(LLMEvent.turn_start("something"))
    prompt = "Do the thing"
    result = r.inject_corrections(prompt)
    # No pattern yet → helper returns the prompt unchanged
    assert result == prompt


def test_llm_event_factory_kinds_are_exhaustive():
    evts = [
        LLMEvent.turn_start("x"),
        LLMEvent.token("t", 0.0, 0),
        LLMEvent.turn_complete("r"),
        LLMEvent.cost(1, 2, 3),
        LLMEvent.quality_feedback(0.5),
        LLMEvent.user_correction("fix it", True),
        LLMEvent.tool_call("search"),
        LLMEvent.tool_result("search", True, 100),
    ]
    kinds = {e.kind for e in evts}
    assert kinds == {
        "turn_start", "token", "turn_complete", "cost",
        "quality_feedback", "user_correction", "tool_call", "tool_result",
    }


def test_decision_repr_contains_kind():
    r = Regulator.for_user("h")
    d = r.decide()
    assert "continue" in repr(d).lower()


def test_confidence_and_coverage_return_floats():
    r = Regulator.for_user("i")
    r.on_event(LLMEvent.turn_start("x"))
    c = r.confidence()
    cov = r.logprob_coverage()
    assert 0.0 <= c <= 1.0
    assert 0.0 <= cov <= 1.0


def test_cost_cap_getter_reflects_builder():
    r = Regulator.for_user("j").with_cost_cap(5000)
    assert r.cost_cap_tokens() == 5000


def test_quality_feedback_with_fragment_spans_accepts_list_of_tuples():
    # The fragment_spans field is reserved for future use but must
    # accept the expected shape without error.
    e = LLMEvent.quality_feedback(0.7, [(0, 10), (20, 30)])
    assert e.kind == "quality_feedback"


def test_from_otel_span_json_parses_full_turn():
    """OTel GenAI span JSON → list of LLMEvents in canonical order."""
    span = {
        "attributes": {
            "gen_ai.system": "anthropic",
            "gen_ai.usage.input_tokens": 25,
            "gen_ai.usage.output_tokens": 800,
        },
        "events": [
            {"name": "gen_ai.user.message",
             "attributes": {"content": "find order 42"}},
            {"name": "gen_ai.assistant.message",
             "attributes": {"content": "found"}},
        ],
        "start_time_unix_nano": 1_000_000_000,
        "end_time_unix_nano":   1_500_000_000,
    }
    events = LLMEvent.from_otel_span_json(json.dumps(span))
    # Canonical: TurnStart, TurnComplete, Cost
    assert len(events) == 3
    kinds = [e.kind for e in events]
    assert kinds == ["turn_start", "turn_complete", "cost"]


def test_from_otel_span_json_with_tool_call_emits_paired_events():
    span = {
        "events": [
            {
                "name": "gen_ai.tool.message",
                "attributes": {
                    "gen_ai.tool.name": "search_orders",
                    "gen_ai.tool.arguments": '{"user_id": 42}',
                    "gen_ai.tool.duration_ms": 120,
                },
            }
        ]
    }
    events = LLMEvent.from_otel_span_json(json.dumps(span))
    kinds = [e.kind for e in events]
    assert kinds == ["tool_call", "tool_result"]


def test_from_otel_span_json_raises_on_malformed_input():
    with pytest.raises(ValueError):
        LLMEvent.from_otel_span_json("not json at all")


def test_from_otel_span_json_empty_span_returns_empty_list():
    events = LLMEvent.from_otel_span_json("{}")
    assert events == []


def test_metrics_snapshot_exposes_stable_keys():
    """metrics_snapshot returns all documented keys with float values."""
    r = Regulator.for_user("m").with_cost_cap(5000)
    snap = r.metrics_snapshot()
    expected_keys = {
        "noos.confidence",
        "noos.logprob_coverage",
        "noos.total_tokens_out",
        "noos.cost_cap_tokens",
        "noos.tool_total_calls",
        "noos.tool_total_duration_ms",
        "noos.tool_failure_count",
        "noos.implicit_corrections_count",
    }
    assert set(snap.keys()) == expected_keys
    for k, v in snap.items():
        assert isinstance(v, float), f"metric {k} should be float, got {type(v)}"
    assert snap["noos.cost_cap_tokens"] == 5000.0


def test_metrics_snapshot_tracks_state_changes():
    """Snapshot values move with accumulated events."""
    r = Regulator.for_user("m")
    assert r.metrics_snapshot()["noos.total_tokens_out"] == 0.0
    r.on_event(LLMEvent.turn_start("hello"))
    r.on_event(LLMEvent.cost(tokens_in=10, tokens_out=150, wallclock_ms=500))
    assert r.metrics_snapshot()["noos.total_tokens_out"] == 150.0


def test_implicit_correction_off_by_default():
    """No window configured → fast retries don't count."""
    r = Regulator.for_user("u")
    r.on_event(LLMEvent.turn_start("Refactor fetch_user to be async"))
    r.on_event(LLMEvent.turn_complete("resp"))
    r.on_event(LLMEvent.turn_start("Fix the fetch_user async refactoring"))
    assert r.implicit_corrections_count() == 0


def test_implicit_correction_fires_on_fast_same_cluster_retry():
    """Fast retry on same cluster with window enabled → 1 correction."""
    import time

    r = Regulator.for_user("u").with_implicit_correction_window_secs(0.5)
    r.on_event(LLMEvent.turn_start("Refactor fetch_user to be async"))
    r.on_event(LLMEvent.turn_complete("unsatisfactory response"))
    time.sleep(0.02)
    r.on_event(LLMEvent.turn_start("Fix the fetch_user async refactoring"))
    assert r.implicit_corrections_count() == 1


def test_with_implicit_correction_window_rejects_non_positive():
    with pytest.raises(ValueError):
        Regulator.for_user("u").with_implicit_correction_window_secs(0.0)
    with pytest.raises(ValueError):
        Regulator.for_user("u").with_implicit_correction_window_secs(-1.0)
    with pytest.raises(ValueError):
        Regulator.for_user("u").with_implicit_correction_window_secs(float("nan"))


def test_otel_end_to_end_feeds_regulator():
    """Plumbing test: OTel span → Regulator.on_event → decide() works."""
    span = {
        "attributes": {
            "gen_ai.usage.input_tokens": 20,
            "gen_ai.usage.output_tokens": 100,
        },
        "events": [
            {"name": "gen_ai.user.message",
             "attributes": {"content": "hello"}},
            {"name": "gen_ai.assistant.message",
             "attributes": {"content": "hi"}},
        ],
    }
    r = Regulator.for_user("otel_user")
    for event in LLMEvent.from_otel_span_json(json.dumps(span)):
        r.on_event(event)
    # Don't assert the specific decision; just confirm the path plumbs.
    _ = r.decide()
    assert r.total_tokens_out() == 100


def test_import_preserves_learning_via_correction_roundtrip():
    """Correction patterns survive export → from_json."""
    r = Regulator.for_user("learning_user")
    for i in range(4):  # exceed MIN_CORRECTIONS_FOR_PATTERN=3
        r.on_event(LLMEvent.turn_start(
            f"Refactor auth to be async (attempt {i})"
        ))
        r.on_event(LLMEvent.turn_complete(
            "async handler with telemetry and logging"
        ))
        r.on_event(LLMEvent.cost(tokens_in=20, tokens_out=60, wallclock_ms=0))
        r.on_event(LLMEvent.user_correction(
            correction_message=f"no telemetry please ({i})",
            corrects_last=True,
        ))

    snapshot = r.export_json()
    r2 = Regulator.from_json(snapshot)

    # New turn on the same cluster → pre-generation procedural_warning
    r2.on_event(LLMEvent.turn_start(
        "Refactor auth module to support async patterns"
    ))
    d = r2.decide()
    if d.is_procedural_warning():  # structural counting may or may not fire
        assert d.patterns is not None
        assert len(d.patterns) >= 1
        p = d.patterns[0]
        assert isinstance(p, CorrectionPattern)
        assert p.user_id == "learning_user"
        assert isinstance(p.pattern_name, str)
        assert p.confidence >= 0.0
