# Nous Principles

10 non-negotiable rules. Every change MUST comply with ALL 10.

> **Crates.io readers**: references below to `docs/brain-map.md`,
> `docs/intervention.md`, `docs/theories.md`, `docs/task-eval-design.md`
> and similar point at internal architecture notes that live in the
> source repo on GitHub but are not published to crates.io (they use
> biological framing that the public surface intentionally doesn't).
> Find them at <https://github.com/Triangle-Technology/noos/tree/main/docs> if you
> need the deep context.

**Principles at a glance:**

| # | Principle | One-line rule |
|---|-----------|---------------|
| P1 | Neuroscience Grounding | Every module has precise mechanism citation, not metaphor |
| P2 | Pure Functions by Default | No hidden mutation |
| P3 | Single Source of Truth | Search before creating; extract shared logic |
| P4 | Layered Dependencies | 7-layer hierarchy; lower never imports higher |
| P5 | Fail-Open Gracefully | Safe defaults; no `unsafe`; no `unwrap()` |
| P6 | Test the Contract | Test behavior, not implementation |
| P7 | Document Why, Update Same Session | Comments explain WHY; docs updated in session |
| P8 | Adding a Brain Module | 11-step checklist |
| **P9a** | **Compensate, Don't Amplify** (metric-scoped) | **For perplexity/state-retention: gain ≤ 1.0. For other metrics: empirical per-metric.** |
| **P9b** | **Don't Duplicate Cortical Work** (universal) | **LLM handles language, sentiment, topic, intent natively. No lexicon regex.** |
| **P10** | **Signal Ordering & Gating** | **Priority rules are as load-bearing as signal content** |

---

## P1: Neuroscience Grounding — Mechanism, Not Metaphor

**Rule**: Every cognitive module, every constant, every algorithm MUST have a neuroscience basis at the level of **precise mechanism**, not metaphor.

Three levels of grounding required:

1. **Module level** — module-level `//!` doc comment states brain analog, **mechanism paper** citation (not review), performance profile
2. **Constant level** — every threshold/rate/weight has a `///` comment citing the paper or derivation
3. **Algorithm level** — the core logic maps to a named brain mechanism describing **substrate + transformation + gating**

If there's no neuroscience justification, the code doesn't belong in Nous. Move it to the application layer.

### Mechanism vs Metaphor test

Every brain analog must pass this test. "Region X handles Y" is a metaphor. "Region X performs mechanism Z through substrate W, gated by condition G" is precise. **Only the precise version tells you what code to write.**

| Region | Metaphorical (INSUFFICIENT) | Precise (REQUIRED) |
|---|---|---|
| Amygdala | "reports arousal level" | **tags specific stimuli** with salience (output: `Vec<SalientCue>`, not scalar) |
| Hippocampus | "stores memories" | **pattern completion from partial cues** via CA3 recurrent connections |
| Basal ganglia | "selects actions" | **suppresses premature actions** via NoGo/indirect pathway (output is "don't", not "do") |
| LC-NE | "measures focus" | **sustains attention via tonic mode**; output is duration modulation, not scalar |
| Insula | "reports incompleteness" | **generates felt incompleteness** that **gets gated** by higher priorities |

A module that reads "like" its brain analog but doesn't match the precise mechanism will fail silently — produce outputs that feel cognitively plausible but don't match any computation the model understands. The model treats such outputs as noise.

### How to research precision

When adding or auditing a module:

1. **Find the mechanism paper**, not the review. Reviews state metaphors ("amygdala = fear"); mechanism papers describe substrate and transformation ("central nucleus of amygdala tags sensory input with salience in ~12ms via Pavlovian associative learning").
2. **Identify substrate**: cells, connections, transmitters. Not "the amygdala" but "central nucleus of the amygdala projecting to periaqueductal gray via GABAergic connections".
3. **Identify transformation**: input signal type → computation → output signal type. Not "detects threat" but "tags sensory inputs with salience score based on associative Pavlovian learning".
4. **Identify gating**: under what conditions this mechanism does NOT fire, or gets suppressed by other mechanisms. Almost always present in real brain, almost always omitted from metaphorical descriptions.
5. **Design Rust signature from the transformation**, not from the metaphor.

**Red flag**: if your module-level `//!` doc cites only a review paper or says "brain analog: X does Y", you likely have a metaphor. Find the mechanism paper.

**Module doc format**:
```rust
//! Thalamic Gate — intelligent early filtering controlling pipeline depth.
//!
//! Brain analog: thalamus filters ~60% of sensory input before reaching cortex.
//! Key papers: Crick 1984 (reticular complex), LeDoux 1996 (fast pathway).
//!
//! Pure function, <1ms, $0 LLM cost.
```

**Constant format** (illustrative example):
```rust
/// Convergence threshold: 1% change = perceptually insignificant (Lamme 2000).
/// Below this delta, further iterations yield no meaningful perceptual update.
pub const CONVERGENCE_EPSILON: f64 = 0.01;
```

**Why**: Constants ARE the cognitive architecture. Remove the neuroscience → it's arbitrary code. Any other AI framework can do arbitrary code.

---

## P2: Pure Functions by Default

**Rule**: Functions MUST be pure (no side effects) unless mutation is explicitly needed and documented.

Mutable functions (`&mut self`) must start their doc comment with `/// Mutable:` explaining what changes and why mutation is necessary.

```rust
// ✅ Pure — takes input, returns output, no side effects
pub fn compute_arousal(message: &str) -> ArousalResult { ... }

// ✅ Mutable — clearly documented, mutation justified
/// Mutable: updates context vector via EMA drift.
/// Requires mutation because context accumulates across turns (dlPFC working memory).
pub fn update_state(&mut self, message: &str, is_user: bool) { ... }

// ❌ Hidden mutation
pub fn classify(&self, message: &str) -> GateResult {
    self.counter += 1;  // Caller doesn't expect side effects from &self
    ...
}
```

**Why**: The convergence loop works BECAUSE each module is a pure transform. If `perceive()` had hidden side effects, the iterative settling would produce different results on each call — breaking convergence.

---

## P3: Single Source of Truth

**Rule**: Before writing ANY new function, constant, or type — search first. If similar logic exists, extract and reuse.

No duplicated logic across modules. If two modules need the same predicate, the predicate lives in ONE place.

**Current shared locations**:
- `cognition/detector.rs` — topic extraction, pattern scoring, overlap computation. Used by gate, belief_state, world_model
- `cognition/adaptive_thresholds.rs` — single source of P10 gating rules; `get_adaptive_threshold()` + per-signal `threshold_*()` helpers
- `types/memory.rs` — `MemoryAtom::is_active()` predicate. MUST be used by ALL consumers (retrieval, consolidation, sleep cycle, importance)
- `math/` — cosine similarity, clamp, softmax. Used by any module needing math

**When to extract**: the moment you write logic that looks similar to existing code. Not "next time" — now.

**Pattern consistency**: When in doubt about how to structure a new module, follow what existing modules do. Reference templates:
- **Simplest module**: `emotional.rs` — pure function, LazyLock regexes, stateful tracker with new/track/query pattern
- **Orchestrator**: `convergence.rs` — composes multiple modules, damped iteration
- **Stateful module**: `locus_coeruleus.rs` — mutable state with sync_from/sync_to_learned pattern

Rust compiler enforces types and ownership but NOT structural patterns. Consistency across modules must be maintained by discipline.

**Why**: Silent divergence kills cognitive architectures. If `is_active()` has slightly different logic in retrieval vs consolidation, atoms will appear/disappear inconsistently. Users see incoherent behavior with no obvious cause.

---

## P4: Layered Dependencies

**Rule**: Modules follow a strict dependency hierarchy. Lower layers never import higher layers.

```
Layer 0: types/        — Data structures only. No logic, no imports from other layers.
Layer 1: math/         — Pure math. Imports nothing except std.
Layer 2: cognition/detector.rs — Shared utilities. Imports Layer 0-1 only.
Layer 3: cognition/{emotional, thalamic_gate, locus_coeruleus, belief_state, hs_arousal}
         — Independent modules. Import Layer 0-2. Do NOT import each other unless documented.
Layer 4: cognition/{world_model, resource_allocator, adaptive_thresholds, signals}
         — Composed modules. Import Layer 0-3.
Layer 5: cognition/{convergence, dynamics, intervention, delta_modulation}
         — Orchestration + interventions. Import Layer 0-4.
         convergence is the ONLY module that calls perceive + gate + gain + allocate together.
```

Post-phase-13 (2026-04-14): Layer 6 (`integration.rs`) removed as orphan. Post-audit-pass-1 (2026-04-10): `prefrontal.rs` renamed to `locus_coeruleus.rs` after cortical PFC parts were deleted.

**Allowed cross-imports within Layer 3**:
- `belief_state` → `emotional` (for `compute_arousal`) ✅
- `belief_state` → `detector` (for `extract_topics`) ✅
- `thalamic_gate` → `detector` (for topic overlap) ✅
- `thalamic_gate` → `emotional` ❌ (gate doesn't need emotional internals — arousal comes via `GateContext.arousal`)
- `locus_coeruleus` → `thalamic_gate` ❌ (LC doesn't classify gates — convergence does)

**External dependencies** (Phase 4+): All go through traits. The cognitive engine NEVER imports concrete implementations of AI providers, storage backends, or embedding models.

```rust
// Future Phase 4: trait boundaries
pub trait AiProvider: Send + Sync { ... }
pub trait MemoryStore: Send + Sync { ... }
pub trait EmbeddingProvider: Send + Sync { ... }
```

**Extensibility goal**: Adding a standalone cognitive module = 1 new file + `pub mod` registration in `mod.rs`. If adding a module requires modifying more than 3 existing files (beyond `mod.rs`, `CLAUDE.md`, `brain-map.md`), the architecture may need refactoring — document why in comments.

Two module categories:
- **Standalone** (e.g., `emotional.rs`, `hs_arousal.rs`): 1 file + registration. Other modules call it, it doesn't call them.
- **Convergence-integrated** (e.g., `resource_allocator.rs`): 1 file + registration + wiring into `convergence::run_one_iteration()` + `compute_max_delta()`. This is architectural work — expected and acceptable, but must verify convergence still holds.

**Why**: Without a dependency rule, modules evolve into a tangled web. When everything imports everything, changing one module breaks ten others. Layers make change safe and predictable.

---

## P5: Fail-Open Gracefully

**Rule**: On any error, return a safe default instead of crashing. No `unsafe`. No `unwrap()`.

Three sub-rules:

1. **Fail-open defaults**: Every function that can fail returns `Option<T>` or `Result<T, E>`. Callers have documented fallback behavior.
2. **No `unsafe`**: Ever. If you need unsafe, you need a different design.
3. **No `unwrap()`**: Use `?`, `.unwrap_or()`, `.unwrap_or_default()`, or `match`. Exception: `LazyLock` regex compilation with `.expect()` (compile-time known valid).

**Documented defaults**:
```rust
// allocate_context_budget → None means "use fixed defaults"
// compute_resource_pressure(None) → 0.5 (neutral pressure)
// converge() catches panic → returns perceive-only model (1 iteration, converged=true)
// softmax(empty) → empty vec
// cosine_similarity(mismatched) → 0.0
```

**Fail-open ≠ swallow errors**: Return a safe default AND preserve error context. Silently discarding errors makes debugging impossible. Rust's type system enables this — prefer `Result<T, NousError>` over `Option<T>` for functions that can fail, so callers can `.unwrap_or_default()` while still having access to the error for logging.

```rust
// ❌ Swallows error — caller never knows what went wrong
pub fn allocate(ctx: &Ctx) -> Option<Allocation> {
    // returns None on error — no context
}

// ✅ Fail-open with context — caller can log or ignore
pub fn allocate(ctx: &Ctx) -> Result<Allocation, NousError> {
    // caller: allocate(ctx).unwrap_or_default()  — fail-open
    // caller: allocate(ctx).inspect_err(|e| warn!("{e}")).unwrap_or_default()  — fail-open + log
}
```

**Why**: Brain doesn't crash — it degrades. If the thalamus fails, cortex still processes (just without filtering). If the amygdala fails, cognition continues (just without emotional coloring). Nous must behave the same way. But the brain also has monitoring — ACC detects when something went wrong. Nous needs that too.

---

## P6: Test the Contract

**Rule**: Every cognitive module has tests. Tests verify BEHAVIOR (what the function produces given an input), not IMPLEMENTATION (how it produces it internally).

```rust
// ✅ Tests contract: "frustrated message produces high arousal and negative valence"
#[test]
fn frustrated_message_high_arousal() {
    let r = compute_arousal("I'm so frustrated, nothing works!!!");
    assert!(r.arousal >= 0.4);
    assert_eq!(r.valence, AffectValence::Negative);
}

// ❌ Tests implementation: "third regex in NEGATIVE_HIGH matches 'frustrated'"
#[test]
fn regex_index_2_matches() {
    assert!(NEGATIVE_HIGH[2].is_match("frustrated"));
}
```

**Why**: Implementation changes constantly — we refine regex patterns, adjust scoring formulas, add new keywords. The contract ("frustrated message → high arousal") must remain stable across all these changes. Testing the contract means refactoring doesn't break tests.

**Test naming**: `snake_case` describing the behavior, not the function name. `frustrated_message_high_arousal` not `test_compute_arousal_3`.

---

## P7: Document Why, Update Same Session

**Rule**: Three documentation requirements, all non-negotiable:

1. **Code comments explain WHY, not WHAT**. The code shows what. Comments explain the reasoning.
2. **`CLAUDE.md` updated in the SAME session** when adding/removing modules, changing architecture, or adding constants. Never "I'll update docs later."
3. **`docs/brain-map.md` updated** when adding a new brain theory or module.

```rust
// ❌ Comments WHAT (useless — code already says this)
// Clamp arousal to 0-1 range
let arousal = clamp(arousal, 0.0, 1.0);

// ✅ Comments WHY (valuable — explains design decision)
// Clamp to [0, 1]: unbounded arousal causes cascading amplification
// in the convergence loop via gate↔arousal feedback (CR4).
let arousal = clamp(arousal, 0.0, 1.0);
```

**Documentation map**:

| File | Purpose | Update when |
|------|---------|-------------|
| `CLAUDE.md` | Working context (auto-loaded) | Any structural change — same session |
| `principles.md` | Non-negotiable rules (this file) | Rules change (rare) |
| `docs/brain-map.md` | Module ↔ Brain ↔ Paper ↔ Constants | New module or new theory |

**No hardcoded counts**: Never write "112 tests" or "19 files" without a date. Use `(as of YYYY-MM)` or describe how to get the count (`cargo test 2>&1 | tail -3`). Hardcoded counts go stale after one session and mislead every session after.

**Why**: After 50 sessions, the only thing preventing drift is documentation. If CLAUDE.md says "19 files" but there are now 35, every future session starts with wrong assumptions.

---

## P8: Adding a Brain Module (Checklist)

When adding a new cognitive module (the most common operation in Phase 3-5), follow ALL steps:

1. **Research**: Identify the neuroscience paper, brain region, and the specific mechanism to implement.

2. **Layer check**: Determine which layer (P4) the module belongs to. If it imports from a higher layer → wrong design.

3. **Search first** (P3): Does any existing module already handle part of this? Extract shared logic to `detector.rs` or a shared utility rather than duplicating.

4. **Create file**: `cognition/{module_name}.rs`
   - Module-level `//!` doc: brain analog, paper citation, performance profile (P1)
   - All constants with `///` citation (P1)
   - Pure functions by default (P2). Document any `&mut self` with justification.

5. **Types**: Add new types to `types/` if needed. Use `#[derive(Debug, Clone, Serialize, Deserialize)]`. Optional fields for backward compatibility.

6. **Register module**: Add `pub mod {name};` to `cognition/mod.rs`.

7. **Tests**: Behavioral tests for every public function (P6). Test edge cases: empty input, extreme values, fail-open defaults.

8. **Wire into convergence** (if the module participates in the feedback loop):
   - Add call in `convergence::run_one_iteration()`
   - Add field to `compute_max_delta()` if the module produces a convergence-relevant value
   - Verify convergence still holds: run `cargo test convergence` — iterations ≤ 5, delta < 0.01

9. **Update CLAUDE.md** (P7): architecture tree, module count, test count, constants reference table (if new constants added).

10. **Update `docs/brain-map.md`** (P7): add module entry with brain region, paper, constants table.

11. **Verify**: `cargo test` (all pass), `cargo clippy` (0 warnings), `cargo build` (0 errors).

---

## P9: Compensate & Don't Duplicate — two related rules

**Rule**: Nous NEVER duplicates what the model already does (P9b), and when
intervening on state retention it does so **compensatorily**, not by
amplifying existing processing (P9a). These are conceptually distinct —
**P9b is universal**, **P9a is metric-scoped**.

The 2026-04-14 audit split the original P9 after the Tầng 4 bottleneck
finding demonstrated gain direction is metric-dependent (compensate helps
perplexity, amplify helps task accuracy). Treating the two parts as one
"universal rule" caused over-generalization — see CR5 check documentation.

---

### P9a: Gain direction for state-retention interventions (METRIC-SCOPED)

**Rule**: For interventions targeting **perplexity or next-token state
retention** (Tầng 2 delta modulation, Tầng 4 uniform scaling when the
metric is prediction quality), modulation gain MUST be ≤ 1.0. Amplifying
(gain > 1.0) hurts perplexity monotonically; compensating (gain < 1.0)
helps monotonically.

**Scope**: P9a IS METRIC-SPECIFIC. The convergent evidence is all on
perplexity / state retention. For task-accuracy interventions (e.g. the
bottleneck paper's +8.27% gain with amplify on downstream benchmarks),
gain direction is an **empirical question per metric** — not axiomatic.

**Evidence for P9a on perplexity** (three convergent streams):

- SC text injection: 378-call eval across 3 models — every brain-state
  text injection HURT quality (a long-form-quality proxy).
- Nous perplexity eval (2026-04-10, `examples/diagnose_harm.rs`): gain > 1.0
  hurts monotonically (+5.21% at 1.2, +15.7% at 1.5); gain < 1.0 helps
  monotonically (-2.87% at 0.8, -1.86% at 0.9). Controls perfect.
- CognitiveGate self-learning: alpha ≈ 0 for routine content — model's
  SSM state already encodes what Nous would compute.
- Bottleneck compound test (2026-04-13): T2 compensatory -1.86%, T4
  compensatory +1.70%, T2+T4 -0.06% — both in compensatory direction on
  perplexity, no amplification hurts less.

**Evidence against P9a universality** (metric boundary):

- Bottleneck paper (arXiv 2602.22719): uniform amplify 5× → +8.27% average
  across 5 SSMs / 6 task-accuracy benchmarks.
- Same intervention in Nous perplexity eval: uniform amplify 5×
  catastrophic (+242% on emotional text).
- **Conclusion**: metric determines direction, not intervention shape.

**Applying P9a**: Any module output or modulation hook that affects
next-token prediction (perplexity path) MUST have gain ≤ 1.0 by
construction. Safety rails (0.5 ≤ gain ≤ 2.0) are physics; cognitive
logic never reaches above 1.0 regardless.

**When P9a does NOT apply**: an intervention explicitly targeting task
accuracy, abstention calibration, or other non-perplexity metrics is NOT
bound by gain ≤ 1.0. Such interventions MUST:

1. Declare their target metric in module `//!` doc.
2. Have per-metric eval evidence supporting their chosen gain direction
   (3+ runs, fair baseline, ≥ 2σ improvement — same bar as P9a perplexity
   evidence).
3. Absent that evidence, default to compensatory direction to avoid silent
   harm to perplexity users who share the pipeline.

---

### P9b: Don't duplicate cortical work (UNIVERSAL)

**Rule**: Before building or keeping any module, verify the model doesn't
already do this natively. If it does, delete the module or move the
capability to the application layer.

**Three tests**:

1. **Does the model already do this?** Pattern recognition, language,
   reasoning, **sentiment classification, topic tracking, intent detection**
   — all native to any modern LLM.
2. **Is the module an *input* to compensatory modulation, or producing text
   for the prompt?** Text output for prompt → almost always cortical
   duplication. Internal scalar feeding Tầng 2/3/4 modulation → potentially
   valid.
3. **Could a well-crafted system prompt replace this module's output?** If
   yes → cortical duplication.

**Scope**: UNIVERSAL. P9b applies regardless of target metric. Duplicating
native capabilities wastes module capacity and introduces "two brains"
conflict (SC 378-call finding).

**Language-specific regex is a P9b red flag**: any regex matching
sentiment/intent/topic lexicons in a specific language (English "angry",
Vietnamese "bực", etc.) is explicit duplication of the LLM's native
language understanding. This rule has been under-enforced in Nous — the
2026-04-14 audit found leaked SC-era regex in 4 live modules
(`emotional.rs`, `thalamic_gate.rs`, `detector.rs`,
`resource_allocator.rs`). A 5th module (`integration.rs`) held such
patterns and was removed as orphan in phase 13 2026-04-14.

**Allowed substitutes for lexicon regex**:

- Language-agnostic structural signals: punctuation density, CAPS ratio,
  `!!!`/`...` repetition, numbered-list / heading / code-block markers,
  message length, question count.
- Hidden-state statistics (see `hs_arousal.rs`): SSM state churn measured
  in tensor space is language-neutral by construction.
- Application-layer classification (if the app genuinely needs intent or
  sentiment as a control signal, the app calls the LLM to classify, not
  Nous).

### What the model actually lacks (valid Nous territory regardless of rule)

- Persistent state across conversations (no hippocampus)
- Bounded SSM state capacity (compensatory retention via gain < 1.0 — P9a)
- Real-time self-monitoring mid-forward-pass (adaptive thresholds)
- Embodied grounding (body budget, allostasis)
- Neuromodulatory gain control (LC-NE tonic/phasic switching)

### What the model already does (P9b forbidden territory)

- Topic tracking (perfect attention beats regex)
- Emotional recognition from text (in any language)
- Strategy selection when prompted well
- Cross-context pattern recognition
- Language detection / classification

### Reverse-engineering from cortex

The wrong question is "what should Nous output?" — this produces metadata,
annotations, and generic signals that duplicate cortical work (P9b
violation).

The right question is "what specific failure mode does the model have, and
what biological mechanism fills that gap precisely?" This forces starting
from observed model deficits, not from brain-inspired design aesthetics.

**Reference**: `feedback_compensate_dont_amplify.md` (full rule history),
`docs/intervention.md` (empirical basis for P9a).

### Red flags

- Module produces text for the prompt (P9b)
- Module enumerates what's missing (coverage bias — SC SIGNALS v1 failure)
- Module's output is indistinguishable from a well-crafted system prompt (P9b)
- Module duplicates a classification the model does natively (P9b)
- Module uses language-specific lexicon regex (P9b, special case)
- Module gain direction is > 1.0 for state-retention targets (P9a)
- Module claims universal gain-direction policy without declaring its metric (P9a)

### Green flags

- Module reads internal state (SSM hidden state) rather than text
- Module provides persistent state the model lacks across time
- Module effect is measurable as improvement over no-Nous baseline (any
  metric, as long as the metric is declared)
- Module stays passive when cortex is sufficient (like gate α ≈ 0 for routine)
- Module gain direction is ≤ 1.0 for perplexity-targeting modules
- Module declares its target metric and has eval evidence for chosen direction

---

## P10: Signal Ordering & Gating

**Rule**: When multiple cognitive signals can fire on the same input, **explicit priority/gating rules are mandatory**. Signal ordering is as load-bearing as signal content. Correct content with wrong ordering fails indistinguishably from wrong content.

### Why this matters

SC SIGNALS v1 failed (6.77 vs SELECTIVE 7.20) not because the 5 signals were wrong — they were neuroscience-grounded and well-implemented. They failed because they fired **in parallel without priority**. At emotional moments (Turn 3), INCOMPLETE (insular breadth check) fired alongside emotional signals, and without gating it pushed the model to broaden when it should have narrowed.

SIGNALS v2 added ONE rule: **amygdala SALIENT fires → SUPPRESSES insular INCOMPLETE**. Matches LeDoux 1996 — low road dominates during emotional salience, executive breadth checks gated out. Result: 7.16 vs 7.09, principle vindicated.

### Rule in practice

For every cognitive module added or reviewed, answer these four questions:

1. **When does this signal fire?** — input conditions
2. **When does it NOT fire?** — suppression conditions (almost always present in real neural circuits)
3. **Which other signals does it suppress?** — dominance relations
4. **Which signals suppress it?** — gating relations

If any of these are "none" or "not specified", find out why before merging. "None" is a valid answer ONLY when justified by neurobiology.

### Nous application

Current `convergence.rs` runs 5 bidirectional connections in parallel. **This is where ordering must be audited.** Under emotional salience (high arousal), breadth-oriented signals (resource_allocator expanding budget, ignition thresholds lowering) should be gated, not averaged with salient signals.

Post-phase-13 (2026-04-14): the ignition check previously hosted by `integration.rs` was removed with that module. The P10 gating rule — "does the threshold narrow under amygdala salience?" — now lives in `adaptive_thresholds.rs`, which is the single source of gating for Nous. Callers that want the Dehaene-style ignition predicate must call `get_adaptive_threshold()` directly. No cognition module currently wires this; re-wiring is deferred until there is a caller with a measured need.

### How to document gating

In module-level `//!` doc, add a **Gating** section:

```rust
//! ## Gating
//!
//! - **Suppresses**: `insular_incompleteness` when salience > 0.6 (LeDoux 1996 low road dominance)
//! - **Suppressed by**: `basal_ganglia::hold` during action commitment (Bogacz 2007 NoGo)
//! - **Inactive when**: routine gate type (saves computation)
```

Without this, the module is incomplete even if its computation is correct.

### Reference

- `feedback_biological_precision.md` — gating corollary section
- SC methodology: C:\tmp\bai-hoc-non-cortical-signals.md
- LeDoux 1996 — amygdala low road dominance as the canonical gating example

---

## Critical Rules

### CR1: Never change a constant without understanding its brain basis
Read the citation. Understand why that value was chosen. If you change it, update the citation and document the new justification in the same commit.

### CR2: Convergence loop invariants are sacred
The 3 convergence constants (epsilon, max iterations, damping alpha — see `docs/brain-map.md` §7) create the convergence guarantee. Changing them changes the behavior of EVERY message processed by Nous. Require explicit justification with neuroscience or mathematical proof.

### CR3: EMA rates are tuned together
EMA learning rates across modules (see `docs/brain-map.md` per-module constants) balance each other. Slower rates exist where false signals are costly (threats, gate rewards). Faster rates exist where natural change is expected (context topics). Changing one rate in isolation shifts the balance between modules.

### CR4: Clamping bounds are safety rails
Every arousal, PE, confidence, and pressure value is clamped to [0, 1]. These bounds prevent cascading amplification in the convergence loop's feedback connections. Never remove a clamp without proving the feedback loop remains stable.

### CR5: Interrogate implementation before abandoning a principle

When eval data seems to contradict a principle from this file, `theories.md`, or `intervention.md`, **interrogate the implementation first**. The prior probability that "my quick implementation failed to capture the deep rule" is vastly higher than "the deep rule is wrong".

SC nearly abandoned the non-cortical axiom after 3 failed implementations (MODERATE 6.52, RICH 6.51, SIGNALS v1 6.77). The principle was correct; the implementations were metaphorical. SIGNALS v2 (7.16) vindicated the principle with a surgical fix: 2 new signals + 1 gating rule. Not a redesign.

**8-step template** (required before revising any principle):

1. **Confirm the data** — 3+ runs minimum, report mean AND dispersion
2. **Enumerate implementations** — list every version you've tried
3. **Diagnose each failure** — not "scored lower" but turn-level or token-level behavioral explanation
4. **Look at mechanism precisely** — not metaphor, not review paper. Substrate + transformation + gating
5. **Find the specific mechanism you did NOT implement** — name it exactly
6. **Surgical fix** — add the missing piece, don't redesign
7. **Re-eval with 3+ runs** — compare to prior failed attempt AND baseline
8. **Only now consider revising the principle** — prefer refinement ("holds under condition X") over abandonment

**Red flags** you're abandoning prematurely: single-run eval, "philosophical vs empirical" framing, proposing a redesign instead of an addition, unable to name the precise mechanism you should have implemented.

**Reference**: `feedback_principle_vs_data.md` in memory for full template and Nous-specific application.
