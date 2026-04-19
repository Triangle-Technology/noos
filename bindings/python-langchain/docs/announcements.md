# Outreach drafts — post-publish

Four ready-to-post drafts (Show HN, LangChain Discord, Reddit
r/LocalLLaMA, a tweet) covering the launch of `noos` +
`noos-langchain` + `@triangle-technology/noos`. All share positioning
on the same two wedges (tool-loop halt + procedural correction memory)
so copy-readers see a consistent story no matter which channel they
land on.

## Status as of 2026-04-19

All three packages live:

- `noos 0.4.0` — https://crates.io/crates/noos
- `noos 0.1.0` (Python) — https://pypi.org/project/noos/
- `noos-langchain 0.1.0` — https://pypi.org/project/noos-langchain/
- `@triangle-technology/noos 0.1.0` (npm) — https://www.npmjs.com/package/@triangle-technology/noos

The crate-root README at https://github.com/Triangle-Technology/noos
leads with real-LLM + real-judge parity numbers as the quality claim
and the S33 benchmark (20-250 ns per event, ~2 µs `decide()`, ~30 µs
per realistic turn) as the overhead claim. Posts below reference both
consistently. The `$47k LangChain tool-loop incident` anchor
(https://github.com/langchain-ai/langchain/issues/13099) is the
tool-loop framing — confirm the link still resolves before posting.

## Timing suggestion

Post all four channels within a 2-hour window so momentum compounds.
Show HN traffic peaks 08–11 am US-Pacific (22:00 – 01:00 VN). Discord
+ Reddit don't have strong time-of-day peaks; post them in the same
window for consistency. Tweet last — it should link back to the HN
post once that's up so the X reader lands on the deeper discussion.

---

## Show HN

**Title (≤80 chars, HN limit)**

```
Show HN: Noos – A pre-delivery regulator for LLM agents (Rust, 44ns/event)
```

Alternative titles, pick one:

- `Show HN: Noos – Halt LLM-agent tool-loops before they blow the budget`
- `Show HN: Noos – Event-driven circuit breaker for LangChain/LangGraph agents`

**Body**

```markdown
Hi HN — I've been shipping a reliability layer for LLM agents as a Rust
crate (`noos` 0.4.0) plus Python + Node bindings and a LangChain callback
adapter (`noos-langchain` 0.1.0). Summary of what it does and why you'd
care:

The problem: LangChain / CrewAI / AutoGen agents in production hit two
recurring failure modes that existing libs don't catch pre-delivery:

1. Tool-call retry loops. The agent calls the same tool with the same args
   repeatedly. `recursion_limit` / `max_iterations` caps total step count
   regardless of what repeats — it missed the $47k LangChain tool-loop
   incident [1].

2. Procedural amnesia. User corrects the agent on Tuesday ("don't add
   telemetry"); Wednesday the agent does it again. Content memory stores
   like Mem0 / Zep / Letta keep the CONTENT but not the RULE.

Noos observes the event stream from the agent (tokens, tool calls,
responses, cost, user corrections) and emits a pre-delivery Decision —
Continue, ScopeDriftWarn, ProceduralWarning, or CircuitBreak — that the
application acts on before the response ships. The Rust core dispatches
per event in 20–250 ns, with decide() at ~2 µs (criterion benchmarks on
my Ryzen laptop).

Concretely for a LangChain agent:

    from noos import Regulator
    from noos_langchain import NoosCallbackHandler, CircuitBreakError

    regulator = Regulator.for_user("alice").with_cost_cap(10_000)
    handler = NoosCallbackHandler(regulator, raise_on_circuit_break=True)

    executor = AgentExecutor(agent=agent, tools=tools, callbacks=[handler])

    try:
        executor.invoke({"input": "Find order 42"})
    except CircuitBreakError as e:
        # Agent halted: e.decision.reason.kind ∈ {
        #   "repeated_tool_call_loop", "cost_cap_reached",
        #   "quality_decline_no_recovery", "repeated_failure_pattern" }
        ...

Same handler works with LangGraph (`config={"callbacks": [handler]}`) and
CrewAI (pass via the LLM constructor's callbacks). Async variant for
`ainvoke` / `astream` is `AsyncNoosCallbackHandler`.

Honest caveats: quality impact on real workloads is at-parity with
unregulated agents in my evals so far (|Δmean_quality| ≤ 0.05 on
Anthropic Haiku + phi3 interleaved runs). The value is infrastructure:
halt / warn / pattern surfaces the agent's failure modes without hurting
good-case behavior. Cost savings are model-dependent (phi3 over-generates
so Noos cuts more; Haiku is concise so Noos overhead is small or slightly
net-negative).

OTel-native: if your agent already emits OpenTelemetry GenAI spans, you
can feed them directly via `LLMEvent.from_otel_span_json` — no callbacks
needed.

Code: https://github.com/Triangle-Technology/noos
crate: https://crates.io/crates/noos
PyPI: https://pypi.org/project/noos/ , https://pypi.org/project/noos-langchain/
npm:  https://www.npmjs.com/package/@triangle-technology/noos

Happy to answer anything about the design. Particularly curious if
anyone has a workload where the current signals would fail — the evals
cover synthetic + mixed-model runs but I don't have a real pilot user yet.

[1] https://github.com/langchain-ai/langchain/issues/13099
```

---

## LangChain Discord (#general or #langgraph)

Short form, conversational. Post-publish only.

```
Hey folks — just released `noos-langchain`, a callback handler that
wires a pre-delivery regulator into any LangChain / LangGraph agent to
halt tool-loops, cap cost × quality, and remember user corrections
across sessions.

```python
from noos import Regulator
from noos_langchain import NoosCallbackHandler, CircuitBreakError

regulator = Regulator.for_user("alice").with_cost_cap(10_000)
handler = NoosCallbackHandler(regulator, raise_on_circuit_break=True)

executor = AgentExecutor(agent=agent, tools=tools, callbacks=[handler])
```

Uses LC's standard BaseCallbackHandler so works with LangGraph
(`config={"callbacks": [...]}`), CrewAI (pass via LLM), and anything
else on langchain-core. AsyncNoosCallbackHandler for async flows. 44ns
per event, ~2µs per decide() — 6 orders of magnitude below an LLM call.

Would love feedback, especially from anyone who's hit the tool-loop
budget-blowout pathology in production.

PyPI:   https://pypi.org/project/noos-langchain/
GitHub: https://github.com/Triangle-Technology/noos
```

---

## Reddit r/LocalLLaMA (post type: text)

**Title**

```
I built a pre-delivery regulator for LLM agents — halts tool-loops in 44ns
```

**Body**

```markdown
Been working on a reliability layer for LLM agents — open-sourced today.

**What it is**: `noos` — a Rust crate (plus Python + Node bindings) that
observes an agent's LLM + tool event stream and emits a `Decision` before
the response ships: halt on tool-call loops, halt on cost × quality
compound, warn on scope drift, surface learned procedural patterns from
user corrections. Drops into LangChain / LangGraph / CrewAI via a standard
callback.

**Why**: two failure modes cost real money in agent deployments, and
existing tooling doesn't catch them pre-delivery:

- Tool-call retry loops. `recursion_limit` caps total steps but doesn't
  detect that the agent called the same tool with the same args 20 times
  — the $47k LangChain incident pattern.
- Procedural amnesia. User corrects the agent today; tomorrow it repeats
  the same mistake. Content memory (Mem0, Zep, Letta) stores the CONTENT
  but not the RULE.

**Honest numbers**:

- Overhead is 44ns per event, ~2µs per decide() (criterion benchmarks on
  a Ryzen laptop).
- Quality impact on real workloads (Anthropic Haiku + phi3 interleaved
  evals): at-parity with baseline (|Δmean_quality| ≤ 0.05). The value is
  infrastructure — halt/warn/pattern without hurting the good case.
- Cost direction is model-dependent. Phi3 over-generates on retries so
  Noos halts save more; Haiku is concise so overhead is near-zero or
  slightly net-positive.

**Local LLM relevance**: works with Ollama via the Anthropic-style LLM
wrappers; the regulator itself has zero LLM dependency (it's a
signal-processing layer, not a generation layer). Good fit for
budget-constrained self-hosted deployments where a runaway agent can
still burn your electricity bill or block a shared GPU.

```python
from noos import Regulator
from noos_langchain import NoosCallbackHandler, CircuitBreakError

regulator = Regulator.for_user("alice").with_cost_cap(10_000)
handler = NoosCallbackHandler(regulator, raise_on_circuit_break=True)

try:
    agent_executor.invoke(
        {"input": "Find order 42"},
        config={"callbacks": [handler]},
    )
except CircuitBreakError as e:
    print(f"Halted: {e.decision.reason.kind}")
```

Also supports OpenTelemetry GenAI spans directly (`LLMEvent.from_otel_span_json`)
so if you already have OTel tracing, just pipe the spans in.

**Links**:

- GitHub: https://github.com/Triangle-Technology/noos
- PyPI: `pip install noos noos-langchain`
- npm: `npm install @triangle-technology/noos`

Would appreciate feedback on whether the signals are well-calibrated for
local-model workflows specifically — my evals so far are synthetic +
mixed-provider, I haven't deployed this against a serious Ollama agent
in the wild yet.
```

---

## Tweet / X

Post last so it can link to the HN discussion (`<HN_LINK>` below —
swap for the real URL after the Show HN post goes live).

**Single tweet (≤280 chars)**

```
Just shipped noos-langchain 0.1.0 — drop NoosCallbackHandler into any
LangChain / LangGraph / CrewAI agent. Pre-delivery halts on tool-loops,
cost × quality regressions, scope drift. 44 ns / event. MIT.

pip install noos-langchain
<HN_LINK>
```

**Thread (3 tweets, for more reach)**

```
1/  Just shipped noos-langchain 0.1.0 — a pre-delivery regulator for
    LLM agents. Drop-in callback for LangChain / LangGraph / CrewAI.

    pip install noos noos-langchain
    https://pypi.org/project/noos-langchain/

2/  Catches the two failure modes that `recursion_limit` + Mem0/Zep
    don't:

    • Tool-call retry loops (the $47k LangChain incident pattern)
    • Procedural amnesia — agent making the same mistake user already
      corrected

    Fires a `CircuitBreak(RepeatedToolCallLoop)` at 5 consecutive
    same-tool calls, a `ProceduralWarning` at 3 corrections on the
    same cluster.

3/  Honest eval numbers: real-LLM + real-judge runs show quality
    parity with baseline (|Δ| ≤ 0.05). Infrastructure, not a quality
    booster. 20-250 ns per event, ~30 µs per realistic turn — 6
    orders of magnitude below an LLM call.

    More: <HN_LINK>
```

**LinkedIn / longer form (~1200 chars)**

Same pitch as Show HN but with a short prelude naming the team /
project context. Use when posting in an org's product feed vs a
personal dev community.
