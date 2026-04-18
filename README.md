# Noos

[![CI](https://github.com/Triangle-Technology/noos/actions/workflows/ci.yml/badge.svg)](https://github.com/Triangle-Technology/noos/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/noos.svg)](https://crates.io/crates/noos)
[![docs.rs](https://docs.rs/noos/badge.svg)](https://docs.rs/noos)
[![license](https://img.shields.io/crates/l/noos.svg)](LICENSE)

> Reliability layer for Rust LLM agents. Detects tool-call retry loops,
> learns from user corrections across sessions, halts on cost × quality
> compound, flags scope drift — all as event-driven decisions, no
> framework lock-in.

`Regulator` sits between your agent's retry loop and your LLM. Every turn
your loop runs, you emit a handful of events — user message, LLM response,
tokens spent, tool calls, optional quality signal. `Regulator` returns a
`Decision`: `Continue`, `CircuitBreak`, `ProceduralWarning`,
`ScopeDriftWarn`, or `LowConfidenceSpans`. Your loop branches on the
variant and keeps moving.

Nothing in Noos wraps your LLM client. There is no framework lock-in and no
runtime dependency on a specific model. The event surface is a single enum
your code owns.

## Problem

Two failure modes cost real money in production:

- **Tool-call retry loops.** A November 2025 LangChain deployment looped
  between two agents for 11 days and burned through $47,000 before anyone
  noticed. `max_iterations` and transport-level retry crates don't catch
  this — the tool call *succeeds* at the protocol level every time; it's
  the agent's decision to re-invoke the same tool that needs to halt.
- **Wasted retries on non-retryable errors.** In a 200-task benchmark,
  90.8% of retries were wasted because systems re-ran hallucinated tool
  names and doomed re-prompts. A compound halt predicate (cost AND
  declining quality) catches this where a cost-only budget doesn't.

Two more are slower-burning but corrode trust over time:

- **Procedural correction amnesia.** Content-memory stores (Mem0, Letta,
  Zep) remember what the user said, but not that the user *keeps pushing
  back the same way*. The third time a user says "no telemetry",
  something structural should change before the fourth generation.
- **Scope drift.** The LLM answered a *different* question than the one
  asked — typically reaches the user and only gets caught if the user
  notices.

Noos surfaces all four signals *during* the loop so the app can act
before delivery: halt a runaway tool loop, halt a cost-burning
quality-dropping retry loop, consult learned corrections before the next
generation, strip drifted material.

## Quick start

```toml
[dependencies]
noos = "0.3"
```

```rust
use noos::{Decision, LLMEvent, Regulator};

let mut regulator = Regulator::for_user("alice").with_cost_cap(2_000);

regulator.on_event(LLMEvent::TurnStart {
    user_message: "Refactor fetch_user to be async".into(),
});

// ... call your LLM of choice ...

regulator.on_event(LLMEvent::TurnComplete {
    full_response: response_text.clone(),
});
regulator.on_event(LLMEvent::Cost {
    tokens_in,
    tokens_out,
    wallclock_ms,
    provider: Some("anthropic".into()),
});

match regulator.decide() {
    Decision::Continue => { /* send response to user */ }
    Decision::CircuitBreak { reason, suggestion } => {
        // Halt the retry loop; surface `suggestion` to the user. The
        // `reason` variant distinguishes CostCapReached /
        // QualityDeclineNoRecovery / RepeatedToolCallLoop.
    }
    Decision::ProceduralWarning { patterns } => {
        // Read `pattern.example_corrections` into the next prompt so
        // the LLM sees the user's prior pushback before generating.
        // Or use the 0.2.2 helper:
        //   let prompt = regulator.inject_corrections(&user_message);
    }
    Decision::ScopeDriftWarn { drift_tokens, drift_score, .. } => {
        // drift_score >= 0.50 means the response added keywords with no
        // anchor in the task. Strip, re-prompt, or annotate.
    }
    Decision::LowConfidenceSpans { .. } => { /* reserved for future */ }
    // `Decision` is `#[non_exhaustive]` since 0.2.1 — future variants
    // (e.g. a multi-concern aggregator) require a wildcard arm.
    _ => {}
}
```

Full per-event contract + timing rules (pre- vs post-generation `decide()`
calls, per-query vs per-task regulator lifetime): see
[`docs/regulator-guide.md`](docs/regulator-guide.md).

**Migrating from another tool?** See
[`docs/migrating.md`](docs/migrating.md) for concrete recipes replacing
LangChain `recursion_limit`, Mem0/Letta content memory, tenacity-style
retry wrappers, and Langfuse/Arize observability.

## What this closes that competitors don't

Leading with the two claims that are net-new across the Python and Rust
agent ecosystems as of 2026-Q2 — ordering follows strongest wedge first:

|                               | Tool-loop halt (structural) | Pattern extraction from corrections | Cost × quality halt | Scope drift (pre-delivery) | Remember content | Log turns |
|-------------------------------|:---------------------------:|:-----------------------------------:|:-------------------:|:--------------------------:|:----------------:|:---------:|
| Langfuse / Arize / Helicone   |              —              |                  —                  |          —          |             —              |        —         |     ✓     |
| Mem0 / Letta / LangMem / Zep  |              —              |           partial¹           |          —          |             —              |        ✓         |     —     |
| Portkey / litellm / OpenRouter|              —              |                  —                  |   transport only    |             —              |        —         |     —     |
| Tenacity / backoff            |              —              |                  —                  |          —          |             —              |        —         |     —     |
| LangGraph / CrewAI / AutoGen  |   max_iterations only²   |                  —                  |   max_iterations    |             —              |        ✓         |     ✓     |
| **Noos**                      |           **✓**             |                **✓**                |       **✓**         |           **✓**            |   ✓ structural   |     —     |

¹ LangMem (Python) generates procedural patterns by asking an LLM to
rewrite the system prompt. Noos does it structurally — per-cluster
correction counting, no LLM-in-the-loop, deterministic, sub-millisecond.

² `recursion_limit` / `maxIterations` count total steps regardless of
what's repeating. Noos's `RepeatedToolCallLoop` fires on the
*signature* of a retry pathology: 5+ consecutive invocations of the
same tool without interleaving. That's the shape of the $47k LangChain
incident referenced above.

Noos is not a logging layer — pair it with Langfuse / Arize / Helicone
if you want observability on top of the regulatory decisions. The value
is the real-time decision surface during the loop, not the post-hoc
record.

## Demos

Five flagship demos, each closing one loop competitors cannot. All run
canned by default (no LLM required); demos 2–4 have `ollama` and
`anthropic` live modes; demo 5 is timing-based and canned-only.

- [`regulator_tool_loop_demo`](examples/regulator_tool_loop_demo.rs) *(0.3.0, leads)*
  — agent calls `search_orders` 5× in a row with tweaked args, each
  returning an empty result. Protocol-level successful calls — retry /
  backoff crates don't flag them; `max_iterations` bounds *total*
  iteration, not consecutive-same-tool. `RepeatedToolCallLoop` fires
  structurally.
- [`regulator_correction_memory_demo`](examples/regulator_correction_memory_demo.rs)
  — 3 explicit corrections build a `CorrectionPattern`; state exports
  → JSON round-trips → imports; next-session turn on the same cluster
  fires `ProceduralWarning` pre-generation with stored example
  corrections attached. Uses `inject_corrections` helper since 0.2.2.
- [`regulator_implicit_correction_demo`](examples/regulator_implicit_correction_demo.rs) *(0.3.1-dev, new)*
  — same `ProceduralWarning` outcome as above, but WITHOUT the app
  having to emit explicit `UserCorrection` events. Three retries
  within the configured window on the same topic cluster → correction
  pattern emerges automatically. Directly addresses the adoption gap
  that "chat UIs rarely surface explicit correction signals".
- [`regulator_cost_break_demo`](examples/regulator_cost_break_demo.rs)
  — 3 retry turns with declining quality trip
  `CircuitBreak(CostCapReached)` on turn 3. Output demonstrates priority
  ordering (drift warnings on turns 1–2, circuit break dominates on 3).
- [`regulator_scope_drift_demo`](examples/regulator_scope_drift_demo.rs)
  — a refactor response adds logging / error handling / telemetry nobody
  asked for. Output shows the exact `drift_tokens` list pre-delivery.

```bash
cargo run --example regulator_tool_loop_demo
cargo run --example regulator_correction_memory_demo
cargo run --example regulator_implicit_correction_demo
cargo run --example regulator_cost_break_demo
cargo run --example regulator_scope_drift_demo
```

## Eval numbers

Tier 2.2 benchmark: mixed-cluster workload (FactQA / Debug / Refactor / Ambiguous),
regulator-enabled arm vs naive-retry baseline. **Honest reporting** — we
distinguish canned (synthetic-oracle) reproducibility numbers from
real-LLM + real-judge data points.

### Real-LLM + real-judge results

The only fair-comparison metric is: real model generating responses,
real grader scoring them, both arms running against the same model
state. Two datasets, both with `NOOS_JUDGE=anthropic` replacing the
synthetic oracle with Claude Haiku:

**Ollama phi3:mini + Haiku judge, interleaved arms (N=29 fair pairs)**

|                        | baseline | regulator |   Δ  |
|------------------------|---------:|----------:|-----:|
| total cost (tokens_out)|   25 494 |    24 118 | **−5.4 %** |
| mean quality           |    0.510 |     0.484 | −0.026 (noise) |
| queries circuit-broken |        0 |         3 | — |
| attempts               |       48 |        37 | **−23 %** |

**Anthropic claude-haiku-4-5 generator + Haiku judge (N=50)**

|                        | baseline | regulator |   Δ  |
|------------------------|---------:|----------:|-----:|
| total cost (tokens_out)|    9 100 |    10 157 | +11.6 % (regulator more expensive) |
| mean quality           |    0.570 |     0.618 | +0.048 (noise) |
| queries circuit-broken |        0 |         0 | — |
| attempts               |       58 |        64 | +10 % |

**Reading**: Quality deltas (|Δ| ≤ 0.05) sit inside the noise bar for
these sample sizes (SE(Δmean) ≈ 0.04–0.05). **No detectable quality
regression from adding the regulator** on either model. Cost direction
is model-dependent — phi3 over-generates on CPU and the regulator's
`CircuitBreak` cuts real excess; Haiku is already concise so the
regulator's overhead is slightly net-negative.

```bash
# Real Ollama + Haiku judge (requires local Ollama + ANTHROPIC_API_KEY)
NOOS_JUDGE=anthropic \
  cargo run --release --example task_eval_real_llm_regulator -- ollama

# Real Anthropic + Haiku judge (requires ANTHROPIC_API_KEY only)
NOOS_JUDGE=anthropic \
  cargo run --release --example task_eval_real_llm_regulator -- anthropic
```

### Canned (synthetic oracle, reproducibility guard only)

Used by CI to catch regressions in the harness itself. **These numbers
do NOT transfer to real LLMs** — the oracle's per-retry quality decline
triggers `QualityDeclineNoRecovery` in a way real graders don't:

|                        | baseline | regulator |   Δ  |
|------------------------|---------:|----------:|-----:|
| total cost (tokens_out)|   16 040 |    11 360 | −29.2 % |
| mean quality           |    0.598 |     0.616 | +0.018 |
| queries circuit-broken |        0 |         9 | — |

```bash
cargo run --release --example task_eval_real_llm_regulator
```

The +46 % quality-per-1k claim that historically appeared in this table
is a scoring artifact of `Cluster::canned_quality(retry)`, not a signal
on real workloads. It's preserved as a bit-for-bit CI guard so changes
to the harness surface explicitly.

### What the regulator provides on real workloads

- **Quality parity** — no detectable regression from adding the wrapper
- **Scope-drift warnings** on majority of turns (real LLMs drift more
  than canned responses)
- **Circuit-break halts** on Ambiguous-cluster retry loops when the
  model exhibits `QualityDeclineNoRecovery` — real on phi3 over 5+
  retries, rarer on Haiku
- **Procedural-memory surfacing** once `MIN_CORRECTIONS_FOR_PATTERN=3`
  threshold trips (see `regulator_correction_memory_demo`)
- **Cost-cap circuit-breaks** (not triggered in this eval — cap sized
  above typical query cost)

The regulator is infrastructure, not a quality booster. Eval confirms
it doesn't cost you measurable quality; the other value props are
qualitative surfacings that the baseline simply lacks.

## Observability

One-call metrics snapshot for Prometheus / Datadog / StatsD pipelines
— no need to call eight individual accessors:

```rust
let snap = regulator.metrics_snapshot();
for (key, value) in snap {
    metrics_client.gauge(&key, value);
}
```

All keys are `noos.` prefixed and stable across releases. Covers
confidence, logprob coverage, cumulative token spend, cost cap, tool
call / duration / failure counts, and the implicit-correction counter.
`decide()` is not sampled — it's an explicit call point — so
`metrics_snapshot()` is cheap enough to call every turn.

## Performance

Criterion benchmarks on the hot path (release build, Windows /
Ryzen-class CPU — re-run locally with `cargo bench` for your hardware):

| Operation                                | Median | Throughput |
|------------------------------------------|-------:|-----------:|
| `on_event(Cost)`                         |  20 ns |   49 M/s   |
| `on_event(Token)`                        |  44 ns |   23 M/s   |
| `on_event(ToolCall)`                     | 249 ns |   4.0 M/s  |
| `decide()` → `Continue`                  | 2.0 µs | 498 K/s    |
| `decide()` → `ScopeDriftWarn` (full)     | 3.7 µs | 268 K/s    |
| `on_event(TurnComplete)`                 | 3.9 µs | 255 K/s    |
| `on_event(TurnStart)`                    |  22 µs |  46 K/s    |
| `export` → `from_json` roundtrip         | 1.2 µs | 836 K/s    |
| **Realistic turn** (1 × start + 100 × token + complete + cost + decide) | **30 µs** | 33 K turns/s |

Interpretation: a realistic turn at 100 streamed tokens adds **~30 µs of
regulator overhead**, dominated by `TurnStart` (keyword extraction /
cluster computation). An LLM call at 200 tokens/sec runs for ~500 ms
per turn — Noos overhead is **six orders of magnitude smaller** than
the LLM latency it wraps. `decide()` on the Continue path is 2 µs, so
calling it multiple times per turn (pre-generation probe + post-delivery
probe + per-cost-event probe) remains cheap.

```bash
cargo bench --bench regulator
```

Benchmarks are not in the published crate — they live in `benches/`
for internal measurement only.

## OpenTelemetry GenAI ingestion

If your agent is already instrumented with the [OTel GenAI semantic
conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), feed
your spans to Noos without rewiring your event bus:

```rust
use noos::{regulator::otel, Regulator};

let span: serde_json::Value = get_otel_span_json();
let mut r = Regulator::for_user("alice");
for event in otel::events_from_span(&span) {
    r.on_event(event);
}
match r.decide() { /* ... */ }
```

Maps `gen_ai.user.message` → `TurnStart`, `gen_ai.assistant.message` →
`TurnComplete`, `gen_ai.usage.*` + span duration → `Cost`, and
`gen_ai.tool.message` → `ToolCall` / `ToolResult`. See
[`src/regulator/otel.rs`](src/regulator/otel.rs) for the full attribute
mapping table.

## Bindings

Cross-language bindings mirror the Rust API 1:1. The event / decision
vocabulary is identical across all three; the `.kind` string pattern
replaces Rust's exhaustive `match` where the host language lacks
data-variant enums.

### Python — `bindings/python/`

```bash
pip install noos-regulator   # once published; PyO3 abi3-py39 wheels
```

```python
from noos_regulator import Regulator, LLMEvent

r = Regulator.for_user("alice")
r.with_cost_cap(2000)
r.on_event(LLMEvent.turn_start("Refactor fetch_user"))
r.on_event(LLMEvent.turn_complete(response))
r.on_event(LLMEvent.cost(tokens_in=25, tokens_out=800, wallclock_ms=500))

d = r.decide()
if d.kind == "circuit_break":
    print(d.suggestion)
```

See [`bindings/python/README.md`](bindings/python/README.md).

### Node.js / TypeScript — `bindings/node/`

```bash
npm install noos-regulator   # once published; napi-rs native addon
```

```typescript
import { Regulator, LLMEvent } from 'noos-regulator'

const r = Regulator.forUser('alice')
r.withCostCap(2_000)
r.onEvent(LLMEvent.turnStart('Refactor fetch_user'))
r.onEvent(LLMEvent.turnComplete(response))
r.onEvent(LLMEvent.cost(25, 800, 500, 'anthropic'))

const d = r.decide()
if (d.kind === 'circuit_break') console.log(d.suggestion)
```

TypeScript `.d.ts` auto-generated from the Rust source. See
[`bindings/node/README.md`](bindings/node/README.md).

## Status (2026-04-17, crate `noos 0.3.0` on crates.io)

Path 2 MVP + two feature expansions shipped:

- **0.2.0** rebranded crate name and Rust types from `Nous*` to `Noos*`
  (crates.io name was taken).
- **0.2.1** added `#[non_exhaustive]` to `LLMEvent` / `Decision` /
  `CircuitBreakReason` so future variants don't break downstream
  exhaustive matches; `#[must_use]` on `Decision`; fixed a silent
  `LearnedState` export bug where LC `tick` + `gain_mode` were never
  flushed to the exported snapshot.
- **0.2.2** Path B — `Regulator::corrections_prelude()` /
  `inject_corrections(&str) -> String` helpers replace the 15-line
  hand-threading recipe for `ProceduralWarning`.
- **0.3.0** Path A — tool-call observation channel. New events
  `LLMEvent::ToolCall` + `LLMEvent::ToolResult`, new
  `CircuitBreakReason::RepeatedToolCallLoop`, new per-turn tool-stats
  accessors, and a fourth flagship demo. Catches the retry-loop failure
  mode that transport-retry crates and `max_iterations` don't.

442 tests passing, zero clippy warnings, zero rustdoc warnings.

### Empirically validated

- **Regulator (primary API)** — Cross-session strategy learning via
  `LearnedState` (Tier 1.1 synthetic, 3 seeds, 2σ bar). Survives
  `RegulatorState` export/import roundtrips so learned per-cluster
  strategies persist across process restarts.
- **Advanced (local Mamba/SSM inference only)** — Compensatory
  state-retention modulation via `CognitiveSession` (perplexity −1.86 %
  on emotional text, 3 runs bit-identical). Not exercised by `Regulator`;
  see [Advanced: direct cognitive-session access](#advanced-direct-cognitive-session-access).

### Shipped infrastructure with honest eval posture

- `Regulator` + four flagship demos + 50-query eval harness.
  Real-LLM + real-judge evals (see [Eval numbers](#eval-numbers))
  show quality parity with baseline (|Δ| ≤ 0.05 at N=29 and N=50, within
  noise). Canned oracle numbers are preserved as CI regression guards
  only, not as product claims.

### Measured limitations

- Metacognition abstention: infrastructure-only on simple tasks (matches
  smart baseline).
- Fatigue detection: regime-dependent (slower than rolling average on
  abrupt drops).

Full honest-scope table + full session-by-session project status in
[`CLAUDE.md`](https://github.com/Triangle-Technology/noos/blob/main/CLAUDE.md) on the
source repo (not published to crates.io — it cross-references internal
memory files that stay private).

## Advanced: direct cognitive-session access

Underneath `Regulator` runs `CognitiveSession`, a pipeline producing
continuous signals (`conservation`, `confidence`, strategy
recommendation, gain mode) plus delta-modulation output for local
Mamba/SSM inference.

Most integrations should prefer `Regulator` — the event surface is
smaller, the Decision enum is easier to branch on, and cross-session
learning survives through `RegulatorState` without touching the cognitive
layer directly. Use `CognitiveSession` directly only if you need one of:

- **Raw continuous signals** for a custom decision policy (e.g. you want
  `signals.conservation` to influence retrieval budget instead of relying
  on `CircuitBreak` to halt).
- **Delta-modulation hints** for local Mamba inference via the `candle`
  feature flag — the validated perplexity −1.86 % path.

Semantic contract for signals + closed-loop requirements:
[`docs/app-contract.md`](docs/app-contract.md). Same crate, one layer
deeper; `Regulator` and `CognitiveSession` can coexist in the same
process.

## Docs

- [`docs/regulator-guide.md`](docs/regulator-guide.md) — app integrator's
  guide: event ordering, decision handling, gotchas.
- [`docs/app-contract.md`](docs/app-contract.md) — semantic contract
  between Noos and your app.
- [`docs/regulator-design.md`](docs/regulator-design.md) — Session 15
  Path 2 design doc.
- [`principles.md`](principles.md) — 10 design principles any change must
  respect.

## License

MIT. See [`LICENSE`](LICENSE).
