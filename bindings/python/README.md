# noos-regulator (Python)

> Python bindings for [Noos](https://crates.io/crates/noos), a reliability
> layer for LLM agents: scope drift, cost circuit breaks, procedural
> correction memory, and tool-call loop detection as event-driven
> decisions.

The Rust core is unchanged — this package ships a native extension
(`noos_regulator.abi3.*.so / .pyd`) built from the same crate published
on crates.io.

## Install

```bash
pip install noos-regulator
```

Wheels are published for CPython 3.9+ on Linux (x86_64, aarch64), macOS
(x86_64, arm64), and Windows (x86_64). Other platforms build from
source (requires a Rust toolchain).

## Quick start

```python
from noos_regulator import Regulator, LLMEvent

r = Regulator.for_user("alice").with_cost_cap(2000)

r.on_event(LLMEvent.turn_start("Refactor fetch_user to be async"))

# ... call your LLM of choice; collect response text ...

r.on_event(LLMEvent.turn_complete(response_text))
r.on_event(LLMEvent.cost(
    tokens_in=25,
    tokens_out=800,
    wallclock_ms=500,
    provider="anthropic",
))

decision = r.decide()

match decision.kind:
    case "continue":
        pass  # deliver response
    case "scope_drift_warn":
        print(f"drift {decision.drift_score:.2f}: {decision.drift_tokens}")
    case "circuit_break":
        print(f"halt: {decision.suggestion} (reason: {decision.reason.kind})")
    case "procedural_warning":
        for p in decision.patterns:
            print(f"learned rule {p.pattern_name}: {p.example_corrections}")
```

The full per-event contract + pre- vs post-generation `decide()` timing
rules lives in the Rust crate's
[`docs/regulator-guide.md`](https://github.com/Triangle-Technology/noos/blob/main/docs/regulator-guide.md).

## API summary

| Class | Purpose |
|-------|---------|
| `Regulator` | Main API. `for_user(id)`, `with_cost_cap(n)`, `on_event(e)`, `decide()`, `export_json()`, `from_json(s)` + accessors. |
| `LLMEvent` | Event constructors. `turn_start`, `token`, `turn_complete`, `cost`, `quality_feedback`, `user_correction`, `tool_call`, `tool_result`. |
| `Decision` | Output of `decide()`. `.kind` + variant-specific attributes. |
| `CircuitBreakReason` | Nested on `Decision.reason` when `kind == "circuit_break"`. |
| `CorrectionPattern` | Items of `Decision.patterns` when `kind == "procedural_warning"`. |

### Decision variants

`decision.kind` is one of:

| kind | Available attributes |
|------|----------------------|
| `continue` | — |
| `scope_drift_warn` | `drift_score`, `drift_tokens`, `task_tokens` |
| `circuit_break` | `reason` (a `CircuitBreakReason`), `suggestion` |
| `procedural_warning` | `patterns` (list of `CorrectionPattern`) |
| `low_confidence_spans` | reserved |

### `CircuitBreakReason` variants

`reason.kind` is one of:

| kind | Available attributes |
|------|----------------------|
| `cost_cap_reached` | `tokens_spent`, `tokens_cap`, `mean_quality_last_n` |
| `quality_decline_no_recovery` | `turns`, `mean_delta` |
| `repeated_failure_pattern` | `cluster`, `failure_count` |
| `repeated_tool_call_loop` | `tool_name`, `consecutive_count` |

### Persistence

```python
# Save
snapshot: str = r.export_json()
with open("regulator.json", "w") as f:
    f.write(snapshot)

# Restore
with open("regulator.json") as f:
    r = Regulator.from_json(f.read())
```

The snapshot carries cross-session learning (strategy EMA + correction
patterns above the emergence threshold). Malformed JSON raises
`ValueError`.

### Path B helper (prompt injection)

Once a `CorrectionPattern` has emerged on a topic cluster, the cheapest
way to apply it is to prepend the learned examples to the next user
prompt:

```python
prompt = r.inject_corrections(user_message)
response = my_llm.complete(prompt)
```

`inject_corrections` is a no-op when `decide()` wouldn't fire
`procedural_warning` — so it's safe to wrap every prompt.

## Build from source

Requires: Rust toolchain (1.75+), Python 3.9+, [maturin](https://maturin.rs/).

```bash
pip install maturin
cd bindings/python
maturin develop --release
```

For a wheel you can redistribute:

```bash
maturin build --release
# wheel lands in target/wheels/
```

## License

MIT. Same as the Rust crate.
