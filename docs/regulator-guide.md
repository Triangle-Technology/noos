# Regulator Integrator's Guide

This is the reference for engineers wiring Nous's `Regulator` into an
agent loop. If you just want to see the shape of the API, read the
[README](../README.md) or run one of the demos. If you're about to ship
a regulator integration, read this end-to-end — the gotchas section
covers every pitfall surfaced in Sessions 21–24 of development.

Scope: `src/regulator/*` (Path 2, the LLM-event-driven layer).
Companion doc for Path 1 semantics:
[`app-contract.md`](app-contract.md).

---

## 1. The shape of the contract

One `Regulator` instance serves one user. Each turn of your agent's loop
emits a small set of `LLMEvent`s. After emitting, you call `decide()`
and branch on the returned `Decision`. That's the whole API surface.

```rust
use noos::{Decision, LLMEvent, Regulator};

let mut regulator = Regulator::for_user(user_id).with_cost_cap(2_000);

regulator.on_event(LLMEvent::TurnStart { user_message });
// ... call LLM, get response + tokens_out + wallclock_ms ...
regulator.on_event(LLMEvent::TurnComplete { full_response });
regulator.on_event(LLMEvent::Cost { tokens_in, tokens_out, wallclock_ms, provider });
regulator.on_event(LLMEvent::QualityFeedback { quality, fragment_spans });

match regulator.decide() {
    Decision::Continue => { /* ship response */ }
    Decision::ScopeDriftWarn { drift_tokens, drift_score, task_tokens } => { /* ... */ }
    Decision::CircuitBreak { reason, suggestion } => { /* halt + surface suggestion */ }
    Decision::ProceduralWarning { patterns } => { /* consult before generating */ }
    Decision::LowConfidenceSpans { spans } => { /* reserved */ }
}
```

Nothing in `Regulator` wraps your LLM client. You choose when to call
it, how to call it, whether to retry. The regulator only sees the
events you emit.

---

## 2. Event lifecycle per turn

### 2.1 Required events for every turn

| Order | Event | What it does |
|-------|-------|--------------|
| 1 | `TurnStart { user_message }` | Resets per-turn state (token stats, scope tracker). Sets the current topic cluster (used by `ProceduralWarning` and `UserCorrection`). Runs the wrapped Path 1 cognitive pipeline. |
| 2 | `TurnComplete { full_response }` | Populates the scope tracker's response side so `ScopeDriftWarn` can compute. Buffers the response awaiting a `QualityFeedback` signal. |
| 3 | `Cost { tokens_in, tokens_out, wallclock_ms, provider }` | Records raw counters and feeds `normalize_cost(tokens_out, wallclock_ms)` into Path 1's `track_cost`, closing the depletion loop (see `app-contract.md` §2). |

### 2.2 Optional events

| Event | When | Purpose |
|-------|------|---------|
| `Token { token, logprob, index }` | Per-token during streaming | Populates the rolling logprob window for `confidence()`. Non-streaming callers skip this. If your provider doesn't expose logprobs, pass `0.0` — the accumulator falls back to a structural heuristic. |
| `QualityFeedback { quality, fragment_spans }` | After a grader or user signal lands | Closes the learning loop — strategy EMA, reward signal, and depletion state all update. Drains any buffered `TurnComplete` into `process_response`. Without this, strategy learning never fires. |
| `UserCorrection { correction_message, corrects_last }` | Next turn, when the user is pushing back | Records into the `CorrectionStore` keyed by the **prior turn's** topic cluster. `corrects_last = false` is treated as a new independent query and dropped — use `TurnStart` for that instead. |

### 2.3 Ordering guarantees

The regulator is forgiving about missing events — any missing event
degrades one signal gracefully rather than crashing. Three orderings
are load-bearing though:

1. `TurnStart` **must** precede everything else in the turn. It sets
   the topic cluster the rest of the turn's events attribute to.
2. `TurnComplete` **must** precede `QualityFeedback` if you want
   strategy learning to fire. The feedback handler drains the
   buffered response; without a buffered response it no-ops.
3. `UserCorrection` **must** follow `TurnStart` — the correction
   attributes to whichever cluster `TurnStart` established. Firing
   a correction with no active turn drops silently (empty cluster).

---

## 3. Decision handling recipes

`decide()` is idempotent within a turn — call it as many times as you
like. The priority order is locked (see §4). The recipes below assume
you've emitted the events from §2.

### 3.1 `Continue`

No intervention needed. Ship the response. This is the happy path —
the vast majority of turns in a healthy agent.

### 3.2 `ScopeDriftWarn { drift_tokens, drift_score, task_tokens }`

The LLM's response contains keywords with no anchor in the user's task.
`drift_score` ∈ [0, 1] where 1.0 is "completely disjoint keyword bags"
and 0.5 is the firing threshold.

Three app responses, in rising aggressiveness:

- **Accept with annotation**: pass `drift_tokens` to the UI as an
  "added beyond request" marker. Cheapest option.
- **Ask the user**: surface `"The response also covered: {drift_tokens}.
  Accept or re-prompt?"`. Best when the expansion might be welcome.
- **Auto-strip + re-prompt**: delete drifted material or re-prompt with
  "Answer only what was asked — do not add {drift_tokens}". Best for
  strict scope (refactor, tool-call responses).

Drift detection is keyword-based (set difference over
`detector::extract_topics` output). It's robust for short-to-medium
responses; long verbose-but-on-topic responses can cross the threshold
because they introduce background vocabulary. Empirical total error
rate (FPR + FNR combined) ≤ 20 % on the 10-case checkpoint in
`src/regulator/scope.rs::decision_checkpoint_fpr_on_hand_crafted_cases`.

### 3.3 `CircuitBreak { reason, suggestion }`

The agent should stop retrying on this task. `suggestion` is a
human-readable string the app can surface directly.

Two reasons fire on the current implementation:

- `CircuitBreakReason::CostCapReached { tokens_spent, tokens_cap, mean_quality_last_n }`
  — cumulative `tokens_out` crossed the cap AND recent quality is below
  `POOR_QUALITY_MEAN = 0.5`. Cost alone doesn't halt; quality alone
  doesn't halt. The compound trip is what this predicate captures.
- `CircuitBreakReason::QualityDeclineNoRecovery { turns, mean_delta }` —
  oldest-minus-newest quality across the rolling window exceeds
  `QUALITY_DECLINE_MIN_DELTA = 0.15` AND the window mean is below
  `POOR_QUALITY_MEAN`. Detects "retries are making it worse, not
  better".

A third reason (`RepeatedFailurePattern`) is declared in the `Decision`
enum for future use but isn't yet emitted by `decide()`.

Typical app response: stop the current retry loop, show the user the
suggestion, ask for clarification or mark the task abandoned.

### 3.4 `ProceduralWarning { patterns }`

This user has corrected the agent ≥ `MIN_CORRECTIONS_FOR_PATTERN = 3`
times on the current topic cluster in prior sessions. The warning fires
**pre-generation** — you get it on the turn's `decide()` call after
`TurnStart` but before you've run the LLM.

Each `CorrectionPattern` contains:

- `topic_cluster` — the hash matching the current turn's cluster.
- `pattern_name` — opaque identifier (`corrections_on_{cluster}`). No
  English regex parses the correction text into a rule; the app / LLM
  does that at generation time.
- `example_corrections` — up to 3 most-recent raw correction texts,
  newest first. These are what you pass to the LLM.
- `learned_from_turns`, `confidence` — provenance counters.

Recommended app flow:

```rust
// `LLMEvent::TurnStart` takes ownership of the user_message String,
// so clone into the event and keep the original in scope for the prompt.
let user_message: String = /* ... */;

regulator.on_event(LLMEvent::TurnStart {
    user_message: user_message.clone(),
});

// Probe BEFORE generation — see §5.2 for why timing matters here.
let prelude = if let Decision::ProceduralWarning { patterns } = regulator.decide() {
    patterns
        .iter()
        .flat_map(|p| &p.example_corrections)
        .map(|ex| format!("- {ex}"))
        .collect::<Vec<_>>()
        .join("\n")
} else {
    String::new()
};

let prompt_with_memory = if prelude.is_empty() {
    user_message
} else {
    format!(
        "User has previously corrected responses on this topic with:\n{prelude}\n\n\
         Current request: {user_message}"
    )
};

// ... call LLM with prompt_with_memory ...
```

### 3.5 `LowConfidenceSpans { spans }`

Reserved. Not emitted in the current release. Will flag specific
response fragments with low per-token logprob confidence for the app to
highlight or re-generate. Ignore for now.

---

## 4. Priority order (P10)

Multiple predicates can fire on the same turn. `decide()` returns ONE
variant, following this strict order:

```
CircuitBreak(CostCapReached)            ← highest: hard stop, cost + quality
CircuitBreak(QualityDeclineNoRecovery)  ← hard stop, quality trend
ScopeDriftWarn                          ← semantic warning
ProceduralWarning                       ← historical advisory
Continue                                ← fallthrough (no predicates fired)
```

Rationale: urgent stop signals dominate semantic warnings, which dominate
historical advisories. The `regulator_cost_break_demo` demonstrates this
live — turns 1–2 show `ScopeDriftWarn` (advisory, app continues);
turn 3 trips the cost cap and `CircuitBreak` suppresses the still-live
drift signal.

This order is locked in `Regulator::decide` and verified by the
`decide_priority_*` tests in `src/regulator/mod.rs`.

---

## 5. Regulator lifetime

The hardest design question for integrators is when to instantiate a
new `Regulator` vs when to keep the existing one. Getting this wrong
either loses cross-session learning or traps the agent in a permanent
halt state.

### 5.1 The two lifetime scopes

The regulator's internal state splits into two categories:

| Persistence | State |
|-------------|-------|
| **Task-scoped** (resets per task) | `CostAccumulator`, `ScopeTracker`, `TokenStatsAccumulator`, `pending_response` |
| **User-scoped** (persists across tasks) | `LearnedState` (Path 1 strategy EMA) and `CorrectionStore` patterns |

A single `Regulator` instance accumulates task-scoped state across
every event it receives. That's correct for a single conversation /
task but wrong for an agent loop serving 50 independent queries — once
`QualityDeclineNoRecovery` fires on one bad cluster, the regulator
halts every subsequent query.

### 5.2 Per-query reset pattern

To keep user-scoped state while resetting task-scoped state, round-trip
through `export()` / `import()`:

```rust
let snapshot = regulator.export();
regulator = Regulator::import(snapshot).with_cost_cap(COST_CAP);
```

`export()` pulls out `LearnedState` + correction patterns.
`import()` rebuilds a fresh regulator with those persistent pieces
restored and everything else zeroed. The eval harness in
[`examples/task_eval_real_llm_regulator.rs`](../examples/task_eval_real_llm_regulator.rs)
uses this pattern between every query.

Two trade-offs you inherit when using this pattern:

1. **Cost cap re-applies**: `import()` rehydrates with the default cap
   (`DEFAULT_TOKEN_CAP = 10 000`). You must re-apply
   `with_cost_cap(...)` after `import()`.
2. **Below-threshold corrections drop**: `export()` only preserves
   clusters at or above `MIN_CORRECTIONS_FOR_PATTERN = 3`. If a user
   has accumulated 2 corrections and you export-then-import, those 2
   corrections are gone. Pattern formation requires within-task
   accumulation. See `Regulator::export` doc for the documented
   trade-off rationale.

### 5.3 When to reset

- **Every distinct task / query** (retrieval-agent serving 50 queries,
  multi-user dashboard, long-running batch): reset between queries to
  keep cost/quality/scope state task-scoped.
- **Within a single conversation / multi-turn task**: don't reset.
  Cost accumulates across retries; quality decline detects "we've been
  struggling for multiple turns"; scope drift checks the latest turn.
- **Across process restarts**: `export()` to durable storage at
  checkpoint time; `import()` on startup. Correction patterns survive;
  per-turn state starts fresh. This is the path the
  `regulator_correction_memory_demo` exercises.

---

## 6. Gotchas

Ordered roughly by "impact if missed". Each is a known surface surfaced
by Sessions 21–24 of development or the test suite.

### 6.1 Task-phrasing trap (Session 21)

If your task message says `"do not add X"`, the scope tracker's
keyword extraction puts `X` in the task bag. When the LLM responds with
a paragraph containing `X`, scope-drift does NOT flag it — because `X`
is in the task.

**Fix**: use positive phrasing. `"Refactor fetch_user to be async.
Keep the database lookup logic unchanged."` avoids the trap. `"Don't
add logging, error handling, or telemetry."` walks into it.

### 6.2 `decide()` timing for `ProceduralWarning` (Session 23)

`ProceduralWarning` is lower priority than `ScopeDriftWarn`. If
`TurnComplete` has populated the scope tracker's response side when
you call `decide()`, any drift will dominate and hide the warning.

**Fix**: call `decide()` to probe for `ProceduralWarning`
**after** `TurnStart` but **before** `TurnComplete`. At that point the
scope tracker's response side is empty, `drift_score` returns `None`,
`ScopeDriftWarn` skips, and `ProceduralWarning` surfaces. You can call
`decide()` again after `TurnComplete` to check for drift on the actual
response. This is the pattern `regulator_correction_memory_demo.rs`
uses.

### 6.3 Cluster-hash stability (Session 23)

`UserCorrection` accumulates into a store keyed by the turn's topic
cluster. The cluster is `build_topic_cluster(extract_topics(user_message))` —
the top 2 alphabetical meaningful-word keywords joined with `+`.

Small message variations that share those top 2 keywords hash to the
same cluster; variations that don't won't accumulate into the same
pattern. `"Make my auth module async"`, `"Refactor auth to support
async"`, and `"Change my auth function to async"` all hash to
`async+auth`. `"Debug my async auth"` — different, because `debug`
may replace one of the top-2 keywords.

**Fix**: when designing prompts or test harnesses, verify cluster
identity empirically. A 15-line throwaway probe that prints
`regulator.export().correction_patterns` keys for a handful of
candidate messages catches misalignment before it silently breaks your
workflow.

### 6.4 Logprob availability (Session 17)

`Regulator::confidence()` has a primary path (rolling mean-NLL over
the per-turn logprob window) and a fallback path (structural heuristic
on the buffered response text — length + `?` density).

The fallback is language-neutral but has a lower ceiling (~0.70 max).
If your provider exposes logprobs and you aren't emitting
`LLMEvent::Token { logprob, .. }` per-token, you're on the fallback
unnecessarily. OpenAI and local candle expose logprobs; Anthropic (as
of 2026-04) doesn't.

**Fix**: if logprobs are available, stream them into the regulator:

```rust
for (i, tok) in stream.enumerate() {
    regulator.on_event(LLMEvent::Token {
        token: tok.text,
        logprob: tok.logprob.unwrap_or(0.0),
        index: i,
    });
}
```

Pass `0.0` when a provider intermittently omits a logprob — the
accumulator treats non-finite or non-negative values as "unavailable"
and doesn't poison the window.

### 6.5 `QualityFeedback` is load-bearing for strategy learning

Without a `QualityFeedback` event, the strategy-learning path inside
the wrapped session never fires — the `LearnedState` stays empty,
`response_strategies` accumulates nothing, and `turn.signals.strategy`
(the Path 1 recommendation surface) stays `None`. `CircuitBreak
(QualityDeclineNoRecovery)` also never fires because the rolling
quality window has no samples to compare.

**Not affected**: `ProceduralWarning`. That path counts
`LLMEvent::UserCorrection { corrects_last: true }` events only — it
doesn't look at `QualityFeedback` at all. An app that emits
corrections but never quality still builds procedural memory correctly.

**Fix**: always emit `QualityFeedback` when you have a signal. If you
only have implicit signals (user retried vs didn't), still emit
something — a conservative `0.5` neutral is better than no event at
all, because the consolidation *happens* regardless of the quality
value.

### 6.6 `with_cost_cap` after `import` (Session 24)

`Regulator::import(state)` rebuilds the cost accumulator with the
library default cap (`DEFAULT_TOKEN_CAP = 10 000`). Any prior
`with_cost_cap(N)` on the exported instance is lost.

**Fix**: re-apply `with_cost_cap(...)` after every `import`:

```rust
let snapshot = regulator.export();
regulator = Regulator::import(snapshot).with_cost_cap(COST_CAP);
//                                      ^^^^^^^^^^^^^^^^^^^^^^^^
//                                  must be re-applied each time
```

### 6.7 `!Send + !Sync`

The regulator is not thread-safe in v0.1. Multi-user server apps use
one regulator per user per task. Don't wrap in an `Arc<Mutex<_>>`
unless you're ready to own the locking discipline — the
`CognitiveSession` inside mutates state per event and is designed for
single-threaded access.

---

## 7. Persistence

`RegulatorState` is `serde`-serialisable. Path 1 `LearnedState` and
Path 2 `correction_patterns` both ride in the same envelope.

```rust
let state = regulator.export();
let json = serde_json::to_string(&state)?;
// write `json` to disk / database / session store ...

// on next process start:
let state: RegulatorState = serde_json::from_str(&json)?;
let regulator = Regulator::import(state).with_cost_cap(COST_CAP);
```

Schema-evolution policy: new fields in `RegulatorState` carry
`#[serde(default)]` so older snapshots keep loading. Verified by
[`pre_session_20_snapshot_deserialises_with_empty_patterns`](../src/regulator/state.rs)
which locks the pre-Session-20 backcompat.

`RegulatorState` contains user-derived content (correction texts) and
should be treated as PII-equivalent. Scope storage by user identity;
never ship one user's state to another. See
[`app-contract.md` §3.2](app-contract.md) for the full privacy
discussion.

---

## 8. Performance

The regulator's work is bounded and cheap:

- `decide()` — O(window size) for cost/quality/scope predicates, O(1)
  for the cluster hash lookup. Sub-millisecond on commodity hardware.
- `on_event(Cost | QualityFeedback)` — O(1) amortised (bounded
  VecDeque pushes).
- `on_event(TurnStart | TurnComplete)` — O(message length) for keyword
  extraction via `detector::extract_topics`. Still sub-millisecond on
  typical message sizes.
- `export()` / `import()` — O(correction patterns + LearnedState
  size). Serde JSON round-trip of a mature regulator is typically
  under a kilobyte.

The wrapped `CognitiveSession` (Path 1) runs the convergence loop
synchronously; its worst-case is 5 iterations with a damping alpha.
See [`CLAUDE.md`](../CLAUDE.md) §7 for the full pipeline timing.

---

## 9. See also

- [`../README.md`](../README.md) — one-page crate overview.
- [`app-contract.md`](app-contract.md) — Path 1 + Path 2 semantic
  contract.
- [`regulator-design.md`](regulator-design.md) — Session 15 design
  document (original spec + per-session implementation notes).
- [`../examples/regulator_scope_drift_demo.rs`](../examples/regulator_scope_drift_demo.rs),
  [`regulator_cost_break_demo.rs`](../examples/regulator_cost_break_demo.rs),
  [`regulator_correction_memory_demo.rs`](../examples/regulator_correction_memory_demo.rs)
  — runnable demos, each closing one loop competitors can't.
- [`../examples/task_eval_real_llm_regulator.rs`](../examples/task_eval_real_llm_regulator.rs)
  — 50-query eval harness; primary source of the efficiency numbers
  in the README.
