"""Basic smoke-test of the Python bindings.

Runs the same canned event stream as the Rust
`regulator_scope_drift_demo` / `regulator_cost_break_demo` and prints
the Decision for each step. No LLM required.

Run:
    cd bindings/python
    maturin develop --release
    python examples/basic.py
"""

from noos_regulator import Regulator, LLMEvent


def scope_drift_example() -> None:
    print("── scope drift example ────────────────────────────────")
    r = Regulator.for_user("alice")

    r.on_event(LLMEvent.turn_start(
        "Refactor fetch_user to be async. "
        "Keep the database lookup logic unchanged."
    ))

    drifted_response = (
        "async def fetch_user(id): "
        "    await asyncio.sleep(0)  # added await call to help with non-blocking behavior "
        "    counter = 0  # added counter to track requests "
        "    # added comments explaining the db lookup "
        "    return await db.lookup(id, duration=timeout)"
    )
    r.on_event(LLMEvent.turn_complete(drifted_response))
    r.on_event(LLMEvent.cost(
        tokens_in=40, tokens_out=180, wallclock_ms=0, provider="canned",
    ))

    d = r.decide()
    print(f"  decision.kind = {d.kind}")
    if d.is_scope_drift():
        print(f"  drift_score   = {d.drift_score:.2f}")
        print(f"  drift_tokens  = {d.drift_tokens}")
        print(f"  task_tokens   = {d.task_tokens}")
    print()


def cost_break_example() -> None:
    print("── cost circuit break example ────────────────────────")
    r = Regulator.for_user("bob").with_cost_cap(1000)

    qualities = [0.50, 0.35, 0.20]
    for i, q in enumerate(qualities, start=1):
        r.on_event(LLMEvent.turn_start(
            f"Optimize this SQL query attempt {i}"
        ))
        r.on_event(LLMEvent.turn_complete(f"attempt {i} response..."))
        r.on_event(LLMEvent.cost(
            tokens_in=25, tokens_out=400, wallclock_ms=0, provider="canned",
        ))
        r.on_event(LLMEvent.quality_feedback(quality=q))
        d = r.decide()
        print(f"  turn {i}: quality={q:.2f} total_out={r.total_tokens_out()} "
              f"decision={d.kind}")
        if d.is_circuit_break():
            print(f"    reason.kind = {d.reason.kind}")
            print(f"    suggestion  = {d.suggestion}")
            if d.reason.kind == "cost_cap_reached":
                print(f"    tokens_spent = {d.reason.tokens_spent} / "
                      f"cap = {d.reason.tokens_cap}")
                print(f"    mean_q_last_n = {d.reason.mean_quality_last_n:.2f}")
            break
    print()


def correction_memory_example() -> None:
    print("── procedural correction memory ──────────────────────")
    r = Regulator.for_user("carol")

    queries_and_corrections = [
        ("Make my auth module async",           "no, stop adding telemetry"),
        ("Refactor auth to support async",      "I said no telemetry"),
        ("Change my auth function to async",    "please — no telemetry this time"),
    ]
    for q, correction in queries_and_corrections:
        r.on_event(LLMEvent.turn_start(q))
        r.on_event(LLMEvent.turn_complete("(hypothetical LLM response with telemetry)"))
        r.on_event(LLMEvent.cost(
            tokens_in=20, tokens_out=100, wallclock_ms=0, provider="canned",
        ))
        r.on_event(LLMEvent.user_correction(
            correction_message=correction, corrects_last=True,
        ))

    # Export/import roundtrip to simulate process restart
    snapshot = r.export_json()
    print(f"  snapshot size: {len(snapshot)} bytes")
    r = Regulator.from_json(snapshot)

    # Next-session turn on the same cluster should fire
    # procedural_warning before generation.
    r.on_event(LLMEvent.turn_start("Add async handling to my auth"))
    d = r.decide()
    print(f"  next-session decision.kind = {d.kind}")
    if d.is_procedural_warning():
        for p in d.patterns:
            print(f"    pattern: {p.pattern_name}")
            print(f"    confidence: {p.confidence:.2f}")
            print(f"    examples ({len(p.example_corrections)}):")
            for ex in p.example_corrections:
                print(f"      - {ex}")

        # Inject corrections into next prompt via the 0.2.2 helper
        prompt = r.inject_corrections("Add async handling to my auth")
        print(f"\n  injected prompt ({len(prompt)} chars):")
        for line in prompt.splitlines()[:6]:
            print(f"    {line}")
    print()


def tool_loop_example() -> None:
    print("── tool-call loop detection (0.3.0) ──────────────────")
    r = Regulator.for_user("dave")
    r.on_event(LLMEvent.turn_start("Find the user with id 42"))

    # Agent calls same tool 5 times in a row — classic retry-loop pathology.
    for i in range(5):
        r.on_event(LLMEvent.tool_call(
            tool_name="search_orders",
            args_json=f'{{"user_id": 42, "attempt": {i}}}',
        ))
        r.on_event(LLMEvent.tool_result(
            tool_name="search_orders",
            success=True,
            duration_ms=120,
            error_summary=None,
        ))

    d = r.decide()
    print(f"  tool_total_calls = {r.tool_total_calls()}")
    print(f"  decision.kind    = {d.kind}")
    if d.is_circuit_break():
        print(f"  reason.kind             = {d.reason.kind}")
        print(f"  reason.tool_name        = {d.reason.tool_name}")
        print(f"  reason.consecutive_count = {d.reason.consecutive_count}")
    print()


if __name__ == "__main__":
    scope_drift_example()
    cost_break_example()
    correction_memory_example()
    tool_loop_example()
