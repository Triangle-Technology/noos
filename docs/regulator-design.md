# Regulator Design — Path 2 (Session 15)

Design document for Nous's pivot to **Rust-native reliability infrastructure for LLM agents**. Authored 2026-04-15 after 4-agent research synthesis (SSM ecosystem / Rust LLM ecosystem / agent regulation landscape / SSM-specific tooling) converged on Path 2 with refined framing.

**This is a design doc, not a code commit.** Session 15 produces this document + architecture memo + CLAUDE.md positioning update. Implementation starts Session 16.

## 1. Problem

Current Nous (post-Phase 14):

- **Downstream signals are fine** — `body_budget`, `conservation`, `confidence`, `strategy`, `recent_quality` all have working infrastructure, documented operating envelopes, test coverage.
- **Input adapter is wrong** — signals driven by regex on user text (emotional lexicon, topic extraction, structural pattern matching). v3 diagnostic (2026-04-15) confirmed:
  - `body_budget` stuck at ~1.0 across 20-turn workload (regex doesn't produce depletion)
  - `confidence` stuck at 0.5 on gibberish (regex can't detect lexical OOD)
  - `strategy` fires only when warmup > MODERATE_MIN_COUNT (not a bug but undocumented user pitfall)

The signals themselves are valuable (research confirms: MAST NeurIPS 2025, Chroma Context Rot, Agent Drift 2601.04170, Abstention TACL 2025). What Nous reads to build them is the wrong input modality.

## 2. Core change

Rewrite the input adapter to read **LLM operation events** instead of **user text patterns**.

```
┌─────────────────────────────────────────────────────────────────┐
│                        BEFORE (Path 1)                          │
├─────────────────────────────────────────────────────────────────┤
│ user text → regex/lexicon → CognitiveState → downstream signals │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                         AFTER (Path 2)                          │
├─────────────────────────────────────────────────────────────────┤
│ LLM events (tokens, logprobs, corrections, cost, timing) →      │
│   input adapter → CognitiveState → downstream signals           │
└─────────────────────────────────────────────────────────────────┘
```

`CognitiveState` is the fixed intermediate. `CognitiveSignals` keeps its shape. Only the input pathway changes. **Tầng 2 delta modulation continues unchanged** — still reads `CognitiveState`, still outputs `DeltaModulation`, still useful for SSM/Mamba deployments as a specialized feature.

## 3. New public API surface

### 3.1 `Regulator` — the new top-level entry point

```rust
pub struct Regulator {
    session: CognitiveSession,          // wraps existing, stays for backcompat
    user_id: String,                    // for cross-session procedural memory
    token_stats: TokenStatsAccumulator, // rolling entropy window
    scope_tracker: ScopeTracker,        // task vs response token overlap
    cost_tracker: CostAccumulator,      // tokens/wallclock/call counts
    correction_memory: CorrectionStore, // per-user procedural preferences
}

impl Regulator {
    /// New regulator for a specific user. Loads persistent state if exists.
    pub fn for_user(user_id: impl Into<String>) -> Self { ... }

    /// Ingest one LLM event. Updates internal state, does not emit decision.
    pub fn on_event(&mut self, event: LLMEvent) { ... }

    /// Query current regulatory decision. Call after a turn completes.
    pub fn decide(&self) -> Decision { ... }

    /// Export procedural memory for persistence across process restarts.
    /// Includes the wrapped session's LearnedState plus correction patterns.
    pub fn export(&self) -> RegulatorState { ... }

    pub fn import(state: RegulatorState) -> Self { ... }

    /// Escape hatch: expose wrapped session for users who want raw signals.
    pub fn session(&self) -> &CognitiveSession { &self.session }
    pub fn session_mut(&mut self) -> &mut CognitiveSession { &mut self.session }
}
```

### 3.2 `LLMEvent` — input stream

```rust
pub enum LLMEvent {
    /// New turn starts. User's message.
    TurnStart {
        user_message: String,
    },

    /// One token emitted by the LLM with its logprob.
    /// For streaming API clients: emit per-token.
    /// For non-streaming: emit once at TurnComplete with aggregate.
    Token {
        token: String,
        logprob: f64,
        index: usize,
    },

    /// Turn response complete. Full text.
    TurnComplete {
        full_response: String,
    },

    /// Cost accounting. Emit after TurnComplete.
    Cost {
        tokens_in: u32,
        tokens_out: u32,
        wallclock_ms: u32,
        /// Optional provider tag for multi-provider agents.
        provider: Option<String>,
    },

    /// User corrected the last response. Used for procedural learning.
    /// `corrects_last` = true means "this user message is a correction of the
    /// previous response", not a new independent query.
    UserCorrection {
        correction_message: String,
        corrects_last: bool,
    },

    /// Low-level: ground truth about response quality if available.
    /// E.g., user gave thumbs-up/down, task completed, etc.
    QualityFeedback {
        quality: f64, // [0, 1]
        /// Optional: which fragment(s) of the response triggered the feedback.
        fragment_spans: Option<Vec<(usize, usize)>>,
    },
}
```

### 3.3 `Decision` — output

```rust
pub enum Decision {
    /// Continue normally. No intervention required.
    Continue,

    /// Stop the agent loop. Further LLM calls unlikely to help.
    CircuitBreak {
        reason: CircuitBreakReason,
        suggestion: String, // human-readable, e.g. "ask user to clarify scope"
    },

    /// Response drifted beyond task scope. App can accept or re-prompt.
    ScopeDriftWarn {
        drift_tokens: Vec<String>,
        drift_score: f64,        // [0, 1] — how far out of scope
        task_tokens: Vec<String>, // original task keywords for reference
    },

    /// Specific response fragments have low confidence. App can highlight to
    /// user for review or re-generate those spans.
    LowConfidenceSpans {
        spans: Vec<ConfidenceSpan>,
    },

    /// Apply learned procedural pattern before next generation.
    /// E.g., "user_123 + refactor cluster → don't expand to error handling".
    ProceduralWarning {
        patterns: Vec<CorrectionPattern>,
    },
}

pub enum CircuitBreakReason {
    /// Budget cap reached with response quality still poor.
    CostCapReached {
        tokens_spent: u32,
        tokens_cap: u32,
        mean_quality_last_n: f64,
    },
    /// Quality trending down across N consecutive turns, no recovery.
    QualityDeclineNoRecovery {
        turns: usize,
        mean_delta: f64,
    },
    /// Repeated failure on same topic cluster.
    RepeatedFailurePattern {
        cluster: String,
        failure_count: usize,
    },
}

pub struct ConfidenceSpan {
    pub start_char: usize,
    pub end_char: usize,
    pub confidence: f64,     // [0, 1]
    pub mean_token_logprob: f64,
}

pub struct CorrectionPattern {
    pub user_id: String,
    pub topic_cluster: String,
    pub pattern_name: String,  // e.g. "do_not_add_error_handling"
    pub learned_from_turns: usize,
    pub confidence: f64,
}
```

## 4. Usage examples

### 4.1 Minimal agent loop with Anthropic SDK

```rust
use noos::regulator::{Regulator, LLMEvent, Decision, CircuitBreakReason};

let mut regulator = Regulator::for_user("user_123");
let client = anthropic_sdk::Client::new();

let user_msg = "Refactor this function to be async.";
regulator.on_event(LLMEvent::TurnStart { user_message: user_msg.into() });

let response = client
    .messages()
    .model("claude-sonnet-4-6")
    .user(user_msg)
    .send()
    .await?;

// Stream tokens into regulator (or emit single Token at end for non-streaming)
for (i, t) in response.content_blocks().iter().enumerate() {
    regulator.on_event(LLMEvent::Token {
        token: t.text.clone(),
        logprob: t.logprob.unwrap_or(0.0),
        index: i,
    });
}
regulator.on_event(LLMEvent::TurnComplete {
    full_response: response.text(),
});
regulator.on_event(LLMEvent::Cost {
    tokens_in: response.usage.input_tokens,
    tokens_out: response.usage.output_tokens,
    wallclock_ms: response.elapsed_ms,
    provider: Some("anthropic".into()),
});

match regulator.decide() {
    Decision::Continue => send_to_user(&response.text()),
    Decision::ScopeDriftWarn { drift_tokens, drift_score, .. } if drift_score > 0.5 => {
        warn_user(&format!(
            "Response added unexpected topics: {:?}. Continue?",
            drift_tokens
        ));
    }
    Decision::CircuitBreak { reason: CircuitBreakReason::RepeatedFailurePattern { .. }, suggestion } => {
        return_error(&suggestion);
    }
    other => send_to_user_with_annotation(&response.text(), &other),
}

// If user corrects next turn:
let next_msg = "Don't add logging — I specifically said refactor only.";
regulator.on_event(LLMEvent::UserCorrection {
    correction_message: next_msg.into(),
    corrects_last: true,
});
// Regulator now has procedural pattern persisted for this user.
```

### 4.2 Local Mamba agent (keeps Tầng 2 bonus)

```rust
use noos::regulator::{Regulator, LLMEvent};
use noos::inference::mamba::CognitiveMambaModel;
use noos::inference::cognitive_model::CognitiveModel;

let mut regulator = Regulator::for_user("user_456");
let mut model = CognitiveMambaModel::from_pretrained("state-spaces/mamba-130m-hf", config)?;

let user_msg = "I'm so stressed about this deadline.";
regulator.on_event(LLMEvent::TurnStart { user_message: user_msg.into() });

// Path 2 signals drive Tầng 2 delta modulation
let delta_mod = regulator.session().current_delta_modulation(model.num_layers());

// Generate with delta modulation (unchanged Tầng 2 path — SSM bonus feature)
for i in 0..tokens.len() {
    let result = model.forward_cognitive(&[tokens[i]], i, &delta_mod)?;
    // ... emit tokens, update regulator per above
}
```

### 4.3 Backwards-compatible existing code

```rust
// Existing Path 1 users — this still works unchanged
use noos::session::CognitiveSession;

let mut session = CognitiveSession::new();
let turn = session.process_message("Hello");
let sampling = turn.sampling;
let delta = turn.delta_modulation;
// ... existing flow unchanged
```

`CognitiveSession::process_message(text)` continues to work. It's equivalent to `Regulator::on_event(TurnStart + TurnComplete)` but with text-regex-derived signals. New users use `Regulator`; old users keep their code.

## 5. Signal source migration

| Signal | Old source (regex) | New source (LLM events) | Priority |
|--------|-------------------|-------------------------|----------|
| `arousal` | `emotional.rs` English lexicon | Per-token entropy trajectory (rising entropy = confusion) + `UserCorrection` frequency | High |
| `gate_type` | `thalamic_gate.rs` text patterns | Response structure (completion vs refusal vs clarification) from `TurnComplete` | Medium |
| `body_budget` depletion | `sensory_pe` + arousal | `Cost` events (tokens, wallclock) + quality-trend | High |
| `signals.confidence` | Fixed 0.5 base | Per-fragment logprob analysis from `Token` stream | High |
| `signals.strategy` | Regex on response text | Extracted correction patterns from `UserCorrection` events | Medium |
| `signals.conservation` | `body_budget` + arousal | Unchanged formula; driven by new body_budget source | Medium |
| `signals.recent_quality` | EMA of `quality` passed to `process_response` | EMA of `QualityFeedback.quality` + inferred from corrections | Low |
| `signals.salience` | Arousal + gate | Response-entropy spike + correction frequency | Low |

**Deprecation plan**:
- `emotional.rs` regex arousal → deprecated in 0.2.0, removed in 0.3.0
- `thalamic_gate.rs` text patterns → deprecated in 0.2.0, removed in 0.3.0
- `cognition/detector.rs` keyword extraction → retained for topic clustering (backwards compat with LearnedState), but not used for cognitive-state building

## 6. Practitioner vocabulary

Public-facing documentation (README, crate docs, integration guides) uses practitioner vocabulary. Internal docs (`docs/theories.md`, `docs/brain-map.md`, module `//!` comments) retain biological citations where they inspired the mechanism.

**Translation table**:

| Internal (biological) | Public (practitioner) |
|----------------------|----------------------|
| Allostatic controller | Reliability layer |
| Body budget / resource pressure | Cost tracker / budget envelope |
| Conservation signal | Cost-overrun prevention |
| Convergence failure | Loop-spiral detection |
| Scope drift | Spec drift defense |
| Procedural memory | Repeated-mistake learning |
| Metacognitive abstention | Low-confidence fragment flagging |
| LC-NE gain mode / arousal | Output-entropy trajectory |
| Gate classification | Response-type inference |
| Cognitive signals | Regulatory signals |

**Tagline candidates for 1.0 release**:
- "Reliability infrastructure for Rust LLM agents"
- "Cost + scope + calibration sidecar for agent loops"
- "The regulation layer between your agent and your LLM"

## 7. Three flagship demos (Sessions 21-23)

Each demo closes a loop that Mem0, Letta, LangChain Memory, Langfuse, Arize, OPA demonstrably CANNOT close in one call. This is the articulation strategy (research agent 3 flagged articulation risk).

### Demo 1 — Scope drift intercept (`regulator_scope_drift_demo.rs`)

**Setup**: Agent asked to refactor a function. LLM response expands to add error handling, logging, telemetry.

**Baseline agent** (no regulator): sends entire response to user. User now has to manually identify what wasn't asked for.

**Regulator agent**: `ScopeDriftWarn` emitted with drift_tokens listing non-task additions. App can either auto-strip, ask user, or accept.

**What no competitor does**: Mem0 stores the interaction. Langfuse logs the trace. OPA checks allow/deny. **None detect semantic drift from task to response in real time.**

### Demo 2 — Cost circuit break (`regulator_cost_break_demo.rs`)

**Setup**: Agent given ambiguous task. LLM generates long response, user corrects, LLM retries, cost accumulates.

**Baseline agent**: runs until token cap, returns best-effort. Cost wasted on repeated low-quality attempts.

**Regulator agent**: After 3 turns without quality improvement + cost > threshold, `CircuitBreak { CostCapReached }` emitted with suggestion "ask user to clarify scope". Agent halts loop, returns suggestion to user.

**What no competitor does**: Portkey has provider-level circuit-breakers (e.g., retry on 429). **None halt on quality+cost compound signals.**

### Demo 3 — Procedural correction memory (`regulator_correction_memory_demo.rs`)

**Setup**: Over 5 sessions, user_123 corrects agent 3 times: "don't add docstrings to refactor output". Correction text varies across turns.

**Baseline agent (Mem0 style)**: stores each correction message verbatim. Next session: retrieves similar past corrections by semantic search. Effectiveness depends on retrieval quality.

**Regulator agent**: extracts pattern `{user: user_123, topic: refactor, rule: no_new_docstrings}`. Persists as `CorrectionPattern`. Session 6: emits `ProceduralWarning` BEFORE generation. Pattern is structural, not text-based.

**What no competitor does**: Memory systems store CONTENT. **None extract behavioral PATTERNS from corrections and apply them prospectively.** This is the clearest differentiation.

## 8. Migration path (backwards compatibility)

**0.1 → 0.2 (post-MVP)**:
- All Path 1 APIs stay callable
- `Regulator` added as new top-level surface
- `emotional.rs` regex still used by default (via `CognitiveSession::process_message`), with `#[deprecated]` attribute pointing to `Regulator`
- Tầng 2 delta modulation unchanged

**0.2 → 0.3 (post-demo validation)**:
- `Regulator` is the primary documented API
- `CognitiveSession::process_message(text)` still works but docs recommend `Regulator`
- SSM users: `Regulator::for_user(...)` + `current_delta_modulation()` hook provides Tầng 2

**0.3 → 1.0 (post-real-LLM eval)**:
- Remove `emotional.rs` regex default path (apps can implement their own via `Regulator::session_mut().model.belief.affect.arousal = _`)
- API stability guarantee

## 9. Implementation plan (Sessions 16-25)

| Session | Deliverable | Test target |
|---------|-------------|-------------|
| 16 | `src/regulator/mod.rs` skeleton: `Regulator`, `LLMEvent`, `Decision`, empty event-handling dispatch | All existing 324 tests still pass + 5 new unit tests on event dispatch |
| 17 | `TokenStatsAccumulator` + entropy-based confidence. Replaces fixed-0.5 `signals.confidence` base. | Confidence responds to low/high logprob runs; gibberish text produces low confidence |
| 18 | `ScopeTracker` + keyword-overlap scope-drift detector (MVP). Emits `ScopeDriftWarn`. | Demo: task "refactor function" vs response "add logging + error handling" → drift_score > 0.3 |
| 19 | `CostAccumulator` + circuit-break rules. Emits `CircuitBreak`. | Demo: 3-turn quality decline + cost > 1000 tokens → CircuitBreak fires |
| 20 | `CorrectionStore` + procedural pattern extraction from `UserCorrection` events. Extends `LearnedState`. | Demo: 3 similar corrections in different words → single `CorrectionPattern` extracted |
| 21 | `examples/regulator_scope_drift_demo.rs` with anthropic_sdk | End-to-end demo showing scope drift detected on real Claude output |
| 22 | `examples/regulator_cost_break_demo.rs` | End-to-end demo on real LLM |
| 23 | `examples/regulator_correction_memory_demo.rs` | End-to-end demo with persistent state across 2 runs |
| 24 | Real-LLM reliability eval: Regulator-enabled agent vs Regulator-disabled on 50-query benchmark with cost cap | Quality, cost, scope-compliance metrics; publish numbers |
| 25 | Docs: README, `app-contract.md` rewrite, `regulator-guide.md`, practitioner vocabulary migration | Public docs ready for crates.io publish |

**Total**: 10 sessions after Session 15 design. MVP end at Session 23, validated 24, shipped 25.

## 10. Explicitly out of scope (for now)

- **Multi-provider event correlation** (tracking same conversation across Claude + GPT calls) — v1 assumes single provider per Regulator instance
- **Streaming token-by-token intervention** — v1 collects all events, decides post-turn. Per-token intervention (stop mid-stream) deferred
- **Cross-user pattern aggregation** — v1 keeps each user's patterns isolated. Anonymized cross-user learning deferred
- **Hybrid-SSM-aware Tầng 2 extensions** — v1 Tầng 2 works on pure SSM only. Hybrid delta modulation (Jamba/Granite 4.0) deferred pending demand signal
- **Non-Rust bindings (Python/TS)** — v1 is Rust-native only. FFI deferred

## 11. Open questions (resolve during implementation)

1. **Token logprob availability**: Anthropic API does not expose per-token logprobs by default. How to handle? Options:
   - (a) Require log probability enablement in API call (when available)
   - (b) Fall back to confidence-from-response-structure (entropy of top-k alternatives not available)
   - (c) Document Regulator works best with local inference where logprobs are free
   → Decision owed by Session 17.

2. **Scope-drift MVP detection**: keyword overlap is naive. When does it need embeddings?
   - Acceptable for Demo 1 if drift_score > 0.3 is interpretable by user
   - Embedding-based detection deferred to v0.3 if demo feedback suggests insufficient precision

3. **Correction pattern extraction**: how to differentiate "don't do X" (rule) vs "prefer Y over X" (preference)?
   - v1 treats all as rules; preference-vs-rule deferred
   - Pattern naming: extract from correction text ("don't add X" → `no_new_X`) or require explicit tagging?

4. **RegulatorState serialization format**: JSON via serde (matches existing LearnedState) or custom binary? JSON for v0.1.

5. **Thread safety**: `Regulator` is `!Send + !Sync` in v1. Multi-user server apps use one Regulator per user per task. Async-safe `Regulator` deferred.

## 11a. Implementation notes (post-ship)

This design was the Session 15 forward-looking spec. Sessions 16-20 shipped
the infrastructure phase; the notes below record resolved open questions and
the design-vs-implementation deltas future readers should be aware of. The
design itself is not rewritten — deviations are preferred at this
boundary so the original intent is still reviewable.

### §11.1 — logprob availability (Session 17)

Resolved as **hybrid** per option (a) + (b) + (c) compromise. Primary path
uses mean-NLL over a rolling window when the provider exposes logprobs
(OpenAI, vLLM, local candle). Fallback path is a language-neutral structural
heuristic (length + `?` density, P9b-compliant). `LOGPROB_UNAVAILABLE = 0.0`
sentinel per §3.2 contract; non-finite / spurious positive values fail-open to
"unavailable". Both paths composed via `confidence_with_fallback`. See
`src/regulator/token_stats.rs` module doc.

### §11.2 — scope-drift detection (Session 18)

Resolved as **keyword-overlap with decision checkpoint**. Metric `|response \
task| / |response|`, threshold 0.5. Decision checkpoint encoded as
`decision_checkpoint_fpr_on_hand_crafted_cases` test: 10 hand-crafted
(5 drift + 5 non-drift) cases, ≤ 20% error rate bar. PASS — embedding-based
detection deferred to v0.3 until empirical evidence motivates it.

### §11.3 — correction pattern extraction (Session 20)

Resolved as **structural, opaque, language-neutral**. Rejected approach (a):
English-regex "don't X" → `no_new_X` rule extraction (P9b red flag — duplicates
cortical work). Shipped approach: `pattern_name = format!("corrections_on_{cluster}")`;
raw `example_corrections` texts stored for app / LLM to interpret at retrieval
time. Differentiation vs Mem0/Letta preserved: Nous surfaces patterns
PROACTIVELY via `Decision::ProceduralWarning` once count threshold trips,
instead of requiring semantic search per turn.

**Deviation from §3.1 and architecture plan**: `correction_patterns` lives on
`RegulatorState` (`src/regulator/state.rs`), NOT extended into
`LearnedState`. Rationale: keeps Layer-0 `types/world.rs` free of Path 2
types; avoids serde tuple-key JSON workarounds; Regulator is 1:1 with user so
`user_id` is implicit and the key can be a plain `String` (topic_cluster).
Plan intent preserved: patterns survive process restart, count ≥ 3 threshold,
cluster-identity shared with strategy learning via
`detector::build_topic_cluster` (moved public in Session 20's P3 refactor).

### §11.4 — `RegulatorState` serialization (Session 20)

JSON via serde (matches `LearnedState`). Extended schema adds
`correction_patterns` with `#[serde(default)]` so pre-Session-20 snapshots
still load. Test `pre_session_20_snapshot_deserialises_with_empty_patterns`
(in `src/regulator/state.rs`) locks the backcompat.

### §11.5 — `Decision` priority ordering (P10, Sessions 19-20)

Design doc §3.3 lists Decision variants without priority. Implementation
locks an explicit order in `Regulator::decide`:

```
1. CircuitBreak(CostCapReached)        — hard stop, cost + quality
2. CircuitBreak(QualityDeclineNoRecovery) — hard stop, quality trend
3. ScopeDriftWarn                      — semantic warning
4. ProceduralWarning                   — historical-pattern advisory
5. Continue                            — fallthrough
```

Rationale: urgent stops dominate semantic warnings dominate advisories.
Tests `decide_priority_circuit_break_dominates_scope_drift` (Session 19) and
`decide_priority_scope_drift_dominates_procedural_warning` (Session 20)
encode two adjacent-pair priority assertions. `LowConfidenceSpans` will slot
between `ProceduralWarning` and `Continue` when Session 22+ work surfaces it
(span-local advisory comes last among advisories).

### §11.6 — thread safety (resolved)

Unchanged from §11 point 5: `Regulator` remains `!Send + !Sync` in v1.
Multi-user server apps use one Regulator per user per task.

## 12. References

- Internal memos archived in the project's private `.claude/memory/`
  directory record the pivot decision, vocabulary discipline rule, and
  v3 diagnostic that surfaced the adapter gap. Those memos are the
  authoritative history; this design doc is the public distillation.
- Research agent 3 report (this session transcript) — "cognitive-signal + active intervention" market gap
- MAST NeurIPS 2025 (arxiv 2503.13657) — 14 failure modes this addresses
- Chroma Context Rot (trychroma.com research) — context degradation problem
- Agent Drift (arxiv 2601.04170) — scope drift problem named
- Abstention survey TACL 2025 — confidence calibration gap
