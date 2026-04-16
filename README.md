# Nous

> Reliability infrastructure for Rust LLM agents — scope drift, cost
> circuit breaks, and procedural correction memory as a small event-driven
> crate.

`Regulator` sits between your agent's retry loop and your LLM. Every turn
your loop runs, you emit a handful of events — user message, LLM response,
tokens spent, quality signal when you have one. `Regulator` returns a
`Decision`: `Continue`, `ScopeDriftWarn`, `CircuitBreak`,
`ProceduralWarning`, or `LowConfidenceSpans`. Your loop branches on the
variant and keeps moving.

Nothing in Nous wraps your LLM client. There is no framework lock-in and no
runtime dependency on a specific model. The event surface is a single enum
your code owns.

## Problem

Generic agent retry loops burn cost because none of their halt predicates
watch response quality. Logging layers record the turn after it happens.
Memory stores hold interaction content but don't extract behavioural
patterns from corrections. Scope-drift (the LLM answered a *different*
question) typically reaches the user and only gets caught if the user
notices.

Nous surfaces these signals *during* the loop so the app can act before
delivery — strip drifted material, halt on cost × quality compound,
consult learned corrections before the next generation.

## Quick start

```toml
[dependencies]
nous = "0.1"
```

```rust
use nous::{Decision, LLMEvent, Regulator};

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
    Decision::ScopeDriftWarn { drift_tokens, drift_score, .. } => {
        // drift_score >= 0.50 means the response added keywords with no
        // anchor in the task. Strip, re-prompt, or annotate.
    }
    Decision::CircuitBreak { reason, suggestion } => {
        // Halt the retry loop; surface `suggestion` to the user.
    }
    Decision::ProceduralWarning { patterns } => {
        // Read `pattern.example_corrections` into the next prompt so
        // the LLM sees the user's prior pushback before generating.
    }
    Decision::LowConfidenceSpans { .. } => { /* reserved for future */ }
}
```

Full per-event contract + timing rules (pre- vs post-generation `decide()`
calls, per-query vs per-task regulator lifetime): see
[`docs/regulator-guide.md`](docs/regulator-guide.md).

## What this closes that competitors don't

|                             | Log turns | Remember content | Scope drift (pre-delivery) | Cost × quality halt | Pattern extraction from corrections |
|-----------------------------|:---------:|:----------------:|:--------------------------:|:-------------------:|:-----------------------------------:|
| Langfuse / Arize / Helicone |     ✓     |        —         |             —              |          —          |                  —                  |
| Mem0 / Letta / LangChain mem|     —     |        ✓         |             —              |          —          |                  —                  |
| Portkey / litellm / OpenRouter|   —     |        —         |             —              |   transport only    |                  —                  |
| Tenacity / backoff          |     —     |        —         |             —              |          —          |                  —                  |
| **Nous**                    |     —     |    ✓ structural  |           **✓**            |       **✓**         |                 **✓**               |

Nous is not a logging layer — pair it with Langfuse / Arize / Helicone if
you want observability on top of the regulatory decisions. The value is
the real-time decision surface during the loop, not the post-hoc record.

## Demos

Three flagship demos, each closing one loop competitors cannot. All run
canned by default (no LLM required); each has `ollama` and `anthropic`
live modes.

- [`regulator_scope_drift_demo`](examples/regulator_scope_drift_demo.rs)
  — a refactor response adds logging / error handling / telemetry nobody
  asked for. Output shows the exact `drift_tokens` list pre-delivery.
- [`regulator_cost_break_demo`](examples/regulator_cost_break_demo.rs)
  — 3 retry turns with declining quality trip
  `CircuitBreak(CostCapReached)` on turn 3. Output demonstrates priority
  ordering (drift warnings on turns 1–2, circuit break dominates on 3).
- [`regulator_correction_memory_demo`](examples/regulator_correction_memory_demo.rs)
  — 3 corrections build a `CorrectionPattern`; state exports → JSON
  round-trips → imports; next-session turn on the same cluster fires
  `ProceduralWarning` pre-generation with stored example corrections
  attached.

```bash
cargo run --example regulator_scope_drift_demo
cargo run --example regulator_cost_break_demo
cargo run --example regulator_correction_memory_demo
```

## Eval numbers

50-query mixed-cluster workload, regulator-enabled arm vs naive-retry
baseline.

### Canned (synthetic quality oracle, reproducible bit-for-bit)

|                        | baseline | regulator |   Δ  |
|------------------------|---------:|----------:|-----:|
| total cost (tokens_out)|   16 040 |    11 360 | **−29.2 %** |
| total quality          |    29.90 |     30.80 | **+0.90** |
| quality per 1k tokens  |     1.86 |      2.71 | **+0.85 (+46 %)** |
| queries circuit-broken |        0 |         9 | — |
| scope drift flagged    |        — |        41 | — |

```bash
cargo run --release --example task_eval_real_llm_regulator
```

### Live (phi3:mini via Ollama, 4h41m wallclock)

Same 50-query stream, real LLM for response text + token counts, same
synthetic quality oracle for scoring.

|                        | baseline | regulator |   Δ  |
|------------------------|---------:|----------:|-----:|
| total cost (tokens_out)|   64 436 |    12 552 | **−80.5 %** |
| total quality          |    29.90 |     30.80 | **+0.90** |
| quality per 1k tokens  |     0.46 |      2.45 | **+1.99 (+433 %)** |
| queries circuit-broken |        0 |         9 | — |
| scope drift flagged    |        — |        41 | — |

```bash
cargo run --release --example task_eval_real_llm_regulator -- ollama
```

Why the live cost gap widens: a real small model over-generates on
uncontrolled retries (baseline burns 4× the canned estimate), while
the regulator halts retry loops on `QualityDeclineNoRecovery` so cost
stays near the canned bound. The quality-per-1k-tokens ratio is the
fair cross-arm metric when the regulator cuts retries short —
`total_quality` alone under-credits the cost saved.

**Caveat**: both runs use a synthetic quality oracle. Live-LLM cost is
real, quality is not. A full publication run needs a real grader —
the harness accepts `-- anthropic` for Claude-graded runs.

## Status (2026-04-15)

Path 2 MVP complete: public API stable (`Regulator` / `LLMEvent` /
`Decision`), three flagship demos, canned-mode eval harness, 422 tests
passing, zero clippy warnings on demos.

### Empirically validated

- **Regulator (primary API)** — Cross-session strategy learning via
  `LearnedState` (Tier 1.1 synthetic, 3 seeds, 2σ bar). Survives
  `RegulatorState` export/import roundtrips so learned per-cluster
  strategies persist across process restarts.
- **Advanced (local Mamba/SSM inference only)** — Compensatory
  state-retention modulation via `CognitiveSession` (perplexity −1.86 %
  on emotional text, 3 runs bit-identical). Not exercised by `Regulator`;
  see [Advanced: direct cognitive-session access](#advanced-direct-cognitive-session-access).

### Shipped infrastructure, canned-eval numbers

- `Regulator` + three flagship demos + 50-query eval harness.
  Live-LLM numbers are pending.

### Measured limitations

- Metacognition abstention: infrastructure-only on simple tasks (matches
  smart baseline).
- Fatigue detection: regime-dependent (slower than rolling average on
  abrupt drops).

Full honest-scope table + full session-by-session project status in
[`CLAUDE.md`](https://github.com/Triangle-Technology/nous/blob/main/CLAUDE.md) on the
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
  between Nous and your app.
- [`docs/regulator-design.md`](docs/regulator-design.md) — Session 15
  Path 2 design doc.
- [`principles.md`](principles.md) — 10 design principles any change must
  respect.

## License

MIT. See [`LICENSE`](LICENSE).
