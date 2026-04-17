# Migrating to Noos from adjacent tools

Concrete "replace your code with this" recipes for the four most
common integrations Noos replaces or augments:

1. [LangChain / CrewAI / AutoGen `recursion_limit`](#1-from-langchain--crewai--autogen-recursion_limit)
2. [Mem0 / Letta / Zep content-memory stores](#2-from-mem0--letta--zep-content-memory)
3. [tenacity / backoff / p-retry](#3-from-tenacity--backoff--p-retry)
4. [Helicone / Langfuse / Arize observability](#4-from-helicone--langfuse--arize-observability)

Each recipe is self-contained — Noos doesn't require a framework
rewrite, just an event loop instrument.

---

## 1. From LangChain / CrewAI / AutoGen `recursion_limit`

**What's wrong with the status quo**: `recursion_limit` counts total
steps regardless of what's repeating. An agent calling
`search_orders` 50 times in a row is indistinguishable from one that
called 10 different tools 5 times each. The $47k LangChain loop
incident ([reference](https://dev.to/waxell/the-47000-agent-loop-why-token-budget-alerts-arent-budget-enforcement-389i))
happened *with* `recursion_limit` set — the limit bounded TOTAL depth
but didn't detect the *signature* of a stuck loop.

**Before** (Python, LangGraph-ish):

```python
graph = StateGraph(...)
graph.add_edge("tool", "agent")
graph.add_edge("agent", "tool")
graph.recursion_limit = 25  # catches deep loops; misses same-tool pathology
```

**After** (Python via `noos-regulator`):

```python
from noos_regulator import Regulator, LLMEvent

regulator = Regulator.for_user(user_id)
regulator.with_cost_cap(10_000)  # also defends cost-blowout path

for step in agent.iter():
    if tool_call := step.get("tool_call"):
        regulator.on_event(LLMEvent.tool_call(
            tool_name=tool_call.name,
            args_json=json.dumps(tool_call.args),
        ))
        result = execute_tool(tool_call)
        regulator.on_event(LLMEvent.tool_result(
            tool_name=tool_call.name,
            success=result.ok,
            duration_ms=result.elapsed_ms,
        ))

    d = regulator.decide()
    if d.is_circuit_break():
        print(f"halt: {d.suggestion} (reason: {d.reason.kind})")
        break  # stop the agent — RepeatedToolCallLoop or cost break
```

**What changed**: `RepeatedToolCallLoop` fires on 5 consecutive
same-tool calls *by name*, regardless of whether the total step
count is high. Catches the $47k shape at loop-depth 5 instead of
loop-depth 25+.

Rust and Node are 1:1 equivalents — same events, same decision
dispatch. See `bindings/node/examples/basic.mjs` for TS.

---

## 2. From Mem0 / Letta / Zep (content memory)

**What's wrong with the status quo**: content-memory stores remember
*what the user said*, retrieved by semantic search at each turn.
They don't recognize that the user keeps pushing back the same way.
The third time a user says "no telemetry", content memory has three
records; Noos has a procedural pattern with an emergence threshold.

**Before** (Python, Mem0):

```python
from mem0 import MemoryClient
mem = MemoryClient()

# Store every message
mem.add(user_message, user_id="alice")

# Retrieve at each turn by similarity
context = mem.search(current_message, user_id="alice")
prompt = f"Context: {context}\n\n{current_message}"
```

**After** (Python via `noos-regulator`, with optional implicit
correction detection):

```python
from noos_regulator import Regulator, LLMEvent

regulator = Regulator.for_user("alice")
regulator.with_implicit_correction_window_secs(60.0)  # auto-detect retries

# Drive events normally
regulator.on_event(LLMEvent.turn_start(current_message))

# Before generating, check for a learned correction pattern
d = regulator.decide()
if d.is_procedural_warning():
    # Prepend learned examples to the prompt. 0.2.2 helper:
    prompt = regulator.inject_corrections(current_message)
else:
    prompt = current_message

response = llm.complete(prompt)
regulator.on_event(LLMEvent.turn_complete(response))
regulator.on_event(LLMEvent.cost(tokens_in, tokens_out, wallclock_ms))

# If user corrects, record it — or rely on implicit detection from the
# window-based retry gate.
regulator.on_event(LLMEvent.user_correction(correction_text, corrects_last=True))
```

**What changed**:

- **Pattern threshold** — `ProceduralWarning` only fires when
  `MIN_CORRECTIONS_FOR_PATTERN = 3` corrections have accumulated on a
  topic cluster. Below the threshold, nothing is injected — avoids
  false "rules" from single corrections.
- **Opaque clustering** — corrections are grouped by a deterministic
  top-2-keyword hash, not an embedding. Sub-millisecond retrieval,
  no vector DB.
- **Implicit detection** — `with_implicit_correction_window_secs`
  means you don't have to instrument explicit correction events. A
  fast same-cluster retry counts automatically.
- **Noos is additive to Mem0** — keep Mem0 for content retrieval,
  add Noos for the procedural layer. They don't conflict.

---

## 3. From tenacity / backoff / p-retry

**What's wrong with the status quo**: retry libraries retry
uniformly — same backoff on transient errors and on hallucinated
tool names. 90.8% of retries in a 200-task benchmark were wasted on
non-retryable errors ([reference](https://towardsdatascience.com/your-react-agent-is-wasting-90-of-its-retries-heres-how-to-stop-it/)).
Plain retry wrappers can't see response quality; they fire the same
number of retries whether quality is recovering or collapsing.

**Before** (Python, tenacity):

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=1, max=16))
def call_llm(prompt: str) -> str:
    return llm.complete(prompt).text
```

**After** (Python via `noos-regulator`, compound halt predicate):

```python
from noos_regulator import Regulator, LLMEvent

regulator = Regulator.for_user(user_id).with_cost_cap(2_000)

for attempt in range(5):
    regulator.on_event(LLMEvent.turn_start(prompt))
    response = llm.complete(prompt).text
    regulator.on_event(LLMEvent.turn_complete(response))
    regulator.on_event(LLMEvent.cost(
        tokens_in=count_tokens(prompt),
        tokens_out=count_tokens(response),
        wallclock_ms=elapsed_ms,
    ))

    # Feed quality if your grader produces one — LLM-as-judge, heuristic,
    # whatever. The compound predicate (cost × quality) NEEDS quality
    # signals to make smart halt decisions.
    if quality := grade(response):
        regulator.on_event(LLMEvent.quality_feedback(quality))

    d = regulator.decide()
    if d.is_circuit_break():
        # Halt — either CostCapReached + quality < 0.5, or
        # QualityDeclineNoRecovery — neither is recoverable.
        break
    elif response_is_good(response):
        break  # success, no retry needed
```

**What changed**:

- **Cost × quality compound** — `CostCapReached` only fires when
  cumulative tokens > cap AND mean quality < 0.5. A quality-preserving
  expensive retry is allowed; a cost-burning quality-dropping retry is
  not.
- **Quality decline detection** — `QualityDeclineNoRecovery` fires
  when recent quality trend is both downward AND below 0.5. Catches
  "agent is making things worse" without a cap.
- **Noos doesn't replace backoff** — if you want exponential backoff
  between retries, keep tenacity and add Noos at the halt-decision
  layer. They address different concerns.

---

## 4. From Helicone / Langfuse / Arize observability

**What's wrong with the status quo**: observability layers log turns
*after they happen*. Drift / loops / cost blowouts show up in the
dashboard post-mortem, not at the decision point. Noos is the
decision layer; observability is the record layer. **They pair; one
doesn't replace the other.**

**Before** (TypeScript, Langfuse):

```typescript
import { Langfuse } from 'langfuse'
const lf = new Langfuse()

const trace = lf.trace({ userId: 'alice' })
const span = trace.span({ name: 'llm-call' })

const response = await llm.complete(prompt)

span.end({ output: response })
trace.update({ output: response })
// Drift / loops only visible in dashboard, after delivery
```

**After** (TypeScript via `noos-regulator`, with Langfuse
alongside):

```typescript
import { Langfuse } from 'langfuse'
import { Regulator, LLMEvent } from 'noos-regulator'

const lf = new Langfuse()
const regulator = Regulator.forUser('alice')
regulator.withCostCap(2_000)

const trace = lf.trace({ userId: 'alice' })

regulator.onEvent(LLMEvent.turnStart(userMessage))
const response = await llm.complete(prompt)
regulator.onEvent(LLMEvent.turnComplete(response))
regulator.onEvent(LLMEvent.cost(tokensIn, tokensOut, wallclockMs))

// PRE-DELIVERY decision — act before the user sees the response
const d = regulator.decide()
if (d.kind === 'scope_drift_warn') {
  trace.event({
    name: 'noos.drift_detected',
    input: { taskTokens: d.taskTokens, driftTokens: d.driftTokens },
    level: 'WARNING',
  })
  // Option: strip drifted content, re-prompt, or deliver with warning
}
if (d.kind === 'circuit_break') {
  trace.event({
    name: 'noos.circuit_break',
    input: { reason: d.reason?.kind, suggestion: d.suggestion },
    level: 'ERROR',
  })
  // Halt
}

// Both observability AND intervention
trace.update({ output: response })

// Periodic metrics snapshot
const metrics = regulator.metricsSnapshot()
for (const [key, value] of Object.entries(metrics)) {
  trace.score({ name: key, value })
}
```

**What changed**:

- Langfuse continues doing what it does well — tracing, dashboards,
  session replay.
- Noos adds the *pre-delivery decision* that Langfuse doesn't do.
- `metrics_snapshot()` produces stable-keyed observability data you
  can pipe into any metrics system — Prometheus, Datadog, StatsD, or
  directly into Langfuse as scores.

---

## See also

- [`regulator-guide.md`](regulator-guide.md) — full event lifecycle
  + decision recipes + P10 priority explanation + 7 gotchas from
  production.
- [`app-contract.md`](app-contract.md) — Noos ↔ application
  semantic contract, signal interpretation, closed-loop
  requirements.
- Crate root `README.md` — public overview + competitor matrix +
  performance numbers.
