# Nous — Reliability infrastructure for Rust LLM agents

## What is Nous

A Rust crate providing a **reliability layer** for LLM agent loops: cost-overrun prevention, spec-drift defense, fragment-confidence flagging, repeated-mistake learning, plus compensatory state-retention for local Mamba/SSM inference. Evidence-first, practitioner-vocabulary positioning; biological/neuroscience framings (allostatic controller, brain subcortex) are internal mechanism references only.

**As of 2026-04-15, Nous is committed to Path 2** (per `memory/project_nous_reframe_llm_perspective_2026_04_15.md`). The core architectural pivot: input adapter rewritten from "regex on user text" → "LLM operation events (tokens, logprobs, corrections, cost)." Implementation plan: `docs/regulator-design.md` + Sessions 16-25 in `memory/project_path2_architecture_plan.md`. **Tầng 2 delta modulation preserved as first-class SSM feature.**

### What's validated (fair baseline, ≥3 seeds or bit-identical runs)

1. **Per-cluster strategy learning across sessions** (`LearnedState` + `export_learned` / `import_learned`)
   - Tier 1.1 synthetic: **0.900 vs 0.725 baseline quality, first_correct 1.67 vs 4.00** (3 seeds, 2σ bar PASS both axes)
   - Tier 1.3 multi-signal synthetic: **+2.73 total quality over smart app-level baseline** on 24-query mixed workload
   - Tier 1.4 budget sweep synthetic: **Nous wins at every tested budget cap 4.0–20.0** (no crossover)
   - **Tier 2.1 real-LLM multi-signal (2026-04-15, v3 authoritative after diagnostic pass)**: ΔNous−Smart=**+0.37** at 20 queries, budget=7.0 (narrowly >2σ significant, **below 1.0 compound bar**). Naive DirectAnswer beats both Smart (+1.45) and Nous (+1.08). Strategy learning pipeline **operational** (correct-strat rate 55.6% vs Smart 21.7%) but "correct strategy" from synthetic design is **economically wrong on weak LLM** — StepByStep costs 2.8× DirectAnswer without 2.8× quality gain on mamba-130m. Conservation never fires (design intent per Tier 1.2: needs cost + arousal + poor-outcomes, this eval is cost-only). Gibberish never abstains (real pipeline gap: confidence signal lacks lexical OOD detection). See `memory/project_finding_real_llm_multi_signal_2026_04_15.md`. Multi-signal compound currently validated on synthetic ONLY.

2. **Calibrated resource-pressure signaling** (`body_budget` + `signals.conservation` + `track_cost`)
   - Tier 1.2: operates in [0, 0.85], reaches 0.526 on sustained-stress + poor outcomes at turn 58
   - Design intent: fires only when cost AND poor outcomes both apply (not cost alone)

3. **Compensatory activation-level modulation for local Mamba inference** (Tầng 2 delta modulation)
   - Perplexity eval: **-1.86% on emotional text, 0.00% on controls** (mamba-130m, 3 runs bit-identical)
   - P9a: gain ≤ 1.0 helps or is neutral; gain > 1.0 monotonically hurts at every tầng

### What's infrastructure-only (measured limitations)

- **Metacognition** (`signals.confidence` + `signals.strategy.is_none()`): Tier 1.5 TIE F1=1.00 with smart per-cluster history baseline. Same kind of predicate expressed differently. Infrastructure-only on simple task; topic-shifting / cross-session abstention untested.
- **Fatigue detection** (`signals.recent_quality` EMA): Tier 1.7 SLOWER than 5-sample rolling avg on abrupt drops (8 vs 4 turns). Tier 1.8 Pareto split on noisy gradual. Regime-dependent; not a fast-detection tool.

### What's deferred

- **Memory atoms** (`MemoryStore` + `hybrid_recall`): trait + sync interface shipped; async wiring for real apps deferred. Blocks LoCoMo-style Tier 2 eval.

### What's owed before claiming more

- **Tier 2 real-LLM benchmark** (2026-04-15 first pull: **negative on weak model**). `examples/task_eval_real_llm_multi_signal.rs` closed the reality loop on mamba-130m. Result: multi-signal compound collapses from synthetic +2.73 to real +0.78 (within noise). Naive DirectAnswer beats both Smart baseline and Nous. Two paths forward: (a) escalate to frontier model (API integration exists in `ai/`) to test if compound is weak-model-specific; (b) diagnose conservation-never-fires + gibberish-never-abstains before re-running. Until one of these shifts the data, allostatic-controller framing remains aspiration rather than description.
- **Language-neutral arousal path** (P9b debt — design project, not cleanup): the -1.86% Emotional finding currently depends on English regex in `emotional.rs`. Baseline arousal 0.450 decomposes into +0.30 (NEGATIVE_HIGH) + 0.15 (NEGATIVE_MOD). Structural signals (punctuation / CAPS / question marks) contribute 0 on prose text. Removing the lexicon without a calibrated replacement would drop arousal to ~0 and destroy the mechanism. The proper fix is wiring `hs_arousal` into the `CognitiveSession` pipeline so arousal comes from the LLM's hidden-state churn (language-neutral by construction), but current HS-derived arousal magnitude (0.162 on the same Emotional text) is below regex (0.450) — requires calibration + architectural changes. See `memory/project_lexicon_removal_analysis_2026_04_14.md` for the three resolution paths (Path A hs_arousal wiring, Path B structural redesign, Path C accept as scoped interim).

### Aspiration (destination, not description)

Long-term direction: **reliability infrastructure for production LLM agents**, implemented via allostatic-inspired mechanisms. Public vocabulary is practitioner-pain (cost control, spec drift, loop spiral, repeated-mistake learning); internal mechanism vocabulary is biological (Barrett 2025 allostasis, Graybiel 2008 habit formation, Aston-Jones 2005 LC-NE) where it inspired the implementation.

The Barrett 2025 six-gap framework (metacognition, session memory, resource budgeting, fatigue, reward learning, state retention) remains the conceptual backbone, but expression is reframed:
- Cost overrun prevention ← resource budgeting
- Loop spiral detection ← convergence failure / fatigue
- Spec drift defense ← scope drift
- Repeated-mistake learning ← procedural / reward learning
- Low-confidence fragment flagging ← metacognitive abstention
- State-retention compensation ← Tầng 2 delta modulation (SSM only)

Becomes a finding, not aspiration, once real-LLM reliability benchmarks (Sessions 24 per `memory/project_path2_architecture_plan.md`) show Regulator-enabled agents measurably outperform baselines.

### Origin

Ported from Semantic Computer (TypeScript production app). SC proved cognitive algorithms work but found text injection HURTS quality. Nous found the same at activation level: amplifying model processing hurts, compensating model limitations helps.

### Philosophy

Data-driven. Trait-based. Zero `unsafe`. Compiles to native + WASM. **Compensate, don't amplify** — Nous fills gaps the model structurally has, never duplicates what models do well (pattern recognition, language, reasoning).

### Axiom evolution

- SC: "don't duplicate" (text injection hurts, AD-150→187)
- Nous Phase 1-6: "don't amplify, compensate" (gain > 1.0 hurts, gain < 1.0 helps)
- Nous 2026-04-13: allostatic controller as aspiration (pivot) → 2026-04-14 CR5 check: direction defensible, execution partial
- Nous 2026-04-14 post-audit: Occam-minimal identity (3 validated + 2 limitations + 1 deferred); allostatic language reserved for destination, not description

## Principles

**Before writing ANY code, read `principles.md`. Every change MUST comply with ALL 10 principles.**

| # | Principle | One-line rule |
|---|-----------|---------------|
| P1 | Neuroscience Grounding | Precise mechanism citation (substrate + transformation + gating), not metaphor |
| P2 | Pure Functions by Default | No side effects unless `&mut self` with documented justification |
| P3 | Single Source of Truth | Search before creating; extract shared logic; never duplicate |
| P4 | Layered Dependencies | Strict 7-layer hierarchy; lower never imports higher |
| P5 | Fail-Open Gracefully | Safe defaults on error; no `unsafe`; no `unwrap()` |
| P6 | Test the Contract | Test behavior not implementation; every module has tests |
| P7 | Document Why, Update Same Session | Comments explain WHY; update CLAUDE.md + brain-map.md same session |
| P8 | Adding a Brain Module | 11-step checklist for the most common operation |
| **P9a** | **Compensate, Don't Amplify** (metric-scoped) | **For perplexity/state-retention: gain ≤ 1.0. For task-accuracy: empirical per-metric.** |
| **P9b** | **Don't Duplicate Cortical Work** (universal) | **LLM handles language/sentiment/topic/intent natively. No lexicon regex.** |
| **P10** | **Signal Ordering & Gating** | **Priority rules are as load-bearing as signal content** |

P9 was split 2026-04-14 after Tầng 4 bottleneck finding showed gain direction is metric-dependent (paper's task accuracy +8.27% with amplify, Nous perplexity catastrophic with same intervention). Treating P9 as "universal" had caused over-generalization. See `principles.md §P9` and `memory/project_cr5_check_pivot_2026_04_13.md` for the CR5 interrogation.

Critical rules: CR1-CR4 protect constants and invariants. **CR5** requires interrogating implementation before abandoning a principle (8-step template in `principles.md`).

## Current Status

Phase 1-6 + Tầng 1-4 + non-cortical audit + hs_arousal readout + **Phase 7 allostatic API surface** + 4 calibration/safety bugfixes via task-eval validation + **Phase 12 audit cleanup** (P9 split P9a/P9b, VN regex + LearnedState + DMN cleanup, CR5 honest reframing, CR2 invariant tests, N-timescale framing) + **Phase 13 prune** (2026-04-14: integration.rs 557 LOC removed — zero non-test callers — plus CLAUDE.md Occam-minimal identity reframe) + **Phase 14 reality loop first pull** (2026-04-15: multi-signal compound eval on real mamba-130m — +0.78 Δ < 1.0 bar, v3 +0.37 after warmup fix; superseded single-signal Path B skeleton; fixed inherited shuffle bug; `NOUS_DIAGNOSE` env-var diagnostic mode added) + **Phase 15 Path 2 design** (2026-04-15: commit to Rust-native reliability infrastructure pivot; `docs/regulator-design.md` + `memory/project_path2_architecture_plan.md` written; no code changes this phase) + **Session 16 Regulator skeleton** (2026-04-15: `src/regulator/mod.rs` lands public API shape — `Regulator`, `LLMEvent` (6 variants), `Decision` (5 variants with supporting types), `RegulatorState`. Event dispatch minimal: `TurnStart` → `process_message`, `TurnComplete` buffers response, `QualityFeedback` drains buffer into `process_response`. `Token` / `Cost` / `UserCorrection` inert. `decide()` returns `Continue` until Sessions 18-20 wire predicates. Principle review fixed 2 violations: P2 Mutable prefix, P6 private-field peeks → observable-behavior tests + added drain-noop test)
+ **Session 17 TokenStatsAccumulator** (2026-04-15: `src/regulator/token_stats.rs` lands rolling logprob window + entropy-based confidence. R2 logprob-availability decision resolved per architecture plan recommendation — hybrid strategy: primary path uses mean-NLL over rolling window when provider exposes logprobs (OpenAI/vLLM/local candle); fallback path uses language-neutral structural heuristic (length + `?` ratio) when provider lacks logprobs (Anthropic as of 2026-04). `LOGPROB_UNAVAILABLE=0.0` sentinel per design doc convention; non-finite/positive values fail-open to unavailable. `Regulator::confidence()` composes primary + fallback via `confidence_with_fallback`. `Regulator::logprob_coverage()` surfaces per-turn availability. `TurnStart` now resets the window; `Token` feeds it. `signals.confidence` in wrapped session unchanged — plumbing deferred)
+ **Session 18 ScopeTracker** (2026-04-15: `src/regulator/scope.rs` lands keyword-overlap scope-drift detector reusing `cognition::detector::extract_topics` + `to_topic_set` + `count_topic_overlap` per P3 — matches `memory/retrieval.rs` precedent. Metric `drift = |response_kw \ task_kw| / |response_kw|` (response-centric, [0,1], `None` before both sides populated). `DRIFT_WARN_THRESHOLD = 0.5` triggers `ScopeDriftWarn`. Decision checkpoint satisfied: 10 hand-crafted cases audit passes ≤ 20% error bar)
+ **Session 19 CostAccumulator + CircuitBreak** (2026-04-15: `src/regulator/cost.rs` lands cumulative token/wallclock counters + rolling quality history + `normalize_cost` with `TOKEN_COST_WEIGHT = 0.7`. `decide()` gains two CircuitBreak predicates with explicit P10 priority: CostCapReached > QualityDeclineNoRecovery > ScopeDriftWarn > Continue. `with_cost_cap(u32)` builder)
+ **Session 20 CorrectionStore + ProceduralWarning + state.rs split** (2026-04-15: `src/regulator/correction.rs` lands structural correction-pattern extraction — MIN_CORRECTIONS_FOR_PATTERN=3, pattern_name opaque `corrections_on_{cluster}` to avoid English regex (P9b-compliant). `src/regulator/state.rs` splits RegulatorState out of mod.rs and extends with `correction_patterns: HashMap<String, CorrectionPattern>` field using `#[serde(default)]` for pre-Session-20 backcompat. `CorrectionPattern` extended with `example_corrections: Vec<String>` + marked `#[non_exhaustive]` for future field additions. `LLMEvent::UserCorrection` handler attributes correction to `current_topic_cluster` (derived via shared `detector::build_topic_cluster` — P3 move from `world_model.rs`). `decide()` priority updated: CircuitBreak×2 > ScopeDriftWarn > **ProceduralWarning (new, advisory — slots between semantic warning and Continue)** > Continue. Export replays example_corrections on import so patterns survive JSON round-trip. 410 unit + 12 integration = **422 tests pass** (+22 Session 20: 10 correction + 3 state + 9 regulator wiring). 7 clippy warnings on `--lib --tests` (all pre-existing, 0 new). **Path 2 infrastructure phase COMPLETE**; Sessions 21-23 = 3 flagship demos. Run `cargo test 2>&1 | grep "test result"` for current count.
+ **Session 21 Demo 1 — Scope drift intercept** (2026-04-15: `examples/regulator_scope_drift_demo.rs` — end-to-end event flow `TurnStart → TurnComplete → Cost → decide()` producing `Decision::ScopeDriftWarn`. Three modes: `canned` (default, deterministic, no LLM), `ollama` (via `ureq` to localhost:11434/api/chat), `anthropic` (via `ureq` to api.anthropic.com/v1/messages with `ANTHROPIC_API_KEY`). Canned path produces `drift_score=0.80` on a realistic drifted response (task keywords: async/database/fetch_user/keep/logic/lookup/refactor/unchanged; drift keywords: added/await/call/comments/counter/db/doc/duration). Live paths gracefully fall back to canned on transport failure so the demo always runs. No new deps (ureq already in Cargo.toml). Zero clippy warnings on the example; Path 1 test count unchanged at 422. Articulation: baseline Mem0/Letta/Langfuse store or log; `Regulator` emits decision PRE-delivery so the app can auto-strip, re-prompt, or accept.
+ **Session 22 Demo 2 — Cost circuit break** (2026-04-15: `examples/regulator_cost_break_demo.rs` — 3 retry turns with declining quality (0.50 → 0.35 → 0.20), 400 tokens_out each, against `Regulator::with_cost_cap(1_000)`. Turn 3 trips `Decision::CircuitBreak { CostCapReached { tokens_spent: 1200, tokens_cap: 1000, mean_quality_last_n: 0.35 } }` with suggestion "Cost cap reached with poor recent quality. Ask the user to clarify scope or abandon this task." P10 priority demonstrated in output: turns 1-2 fire `ScopeDriftWarn` (advisory, agent continues); turn 3 fires `CircuitBreak` (halt) and dominates the still-live drift signal. `QualityDeclineNoRecovery` also qualifies on turn 3 (mean_delta=0.30 ≥ 0.15, mean=0.35 < 0.5) — `CostCapReached` wins by priority, noted in module doc. Three modes match Session 21 (canned default / ollama / anthropic with graceful fallback). **P3 additional fix same session**: HTTP adapters extracted from Sessions 21 + 22 into `examples/regulator_common/mod.rs` (shared via `#[path]` attribute — Cargo doesn't auto-discover files in `examples/<subdir>/`, so the module compiles only when explicitly included). Single-source `call_ollama`/`call_anthropic` ready for Session 23 reuse. Zero clippy warnings on both demos; Path 1 test count unchanged at 422.
+ **Session 23 Demo 3 — Procedural correction memory** (2026-04-15: `examples/regulator_correction_memory_demo.rs` — the clearest Mem0/Letta differentiator per research agent 3. 3 learning turns on user_123 with varied user messages that all hash to the same cluster `async+auth` (verified empirically: "Make my auth module async" / "Refactor auth to support async" / "Change my auth function to async" all produce `build_topic_cluster → "async+auth"`). Each turn records one correction via `LLMEvent::UserCorrection { corrects_last: true }`. After turn 3, `regulator.export()` yields a `RegulatorState` with 1 `CorrectionPattern` (learned_from_turns=3, confidence=0.15, 3 example_corrections). Snapshot round-trips through `serde_json::to_string/from_str` (461 bytes) simulating process restart. After `Regulator::import(state)`, a new `TurnStart("Add async handling to my auth")` (same cluster) — and CRUCIALLY, **decide() called after TurnStart but BEFORE TurnComplete** so scope tracker's response_keywords is empty and `ScopeDriftWarn` skips — triggers `Decision::ProceduralWarning { patterns }` pre-generation. The 3 example_corrections ride along for the LLM to interpret intent. Reuses `regulator_common::{call_ollama, call_anthropic}` (Session 22 P3 extraction). Imports `MIN_CORRECTIONS_FOR_PATTERN` from `nous::regulator::correction` for the progress display. P5 compliant — serde roundtrip errors use `match` + graceful exit, no `.expect()`. Articulation (in demo take-away): "counts per-cluster corrections structurally (no embedding); fires ProceduralWarning proactively once MIN threshold trips — demonstrably absent from every content-retrieval memory system in Rust/Python LLM ecosystem as of 2026-04". **Path 2 flagship demo phase COMPLETE**; Sessions 21-23 each close a loop competitors cannot. Zero clippy warnings on all 3 demos; Path 1 test count unchanged at 422.
+ **Session 24 Tier 2.2 real-LLM regulator eval** (2026-04-15: `examples/task_eval_real_llm_regulator.rs` — 50-query stream, 4-cluster mix (18 FactQA / 10 Refactor / 9 Ambiguous / 13 Debug), 2 arms (baseline vs regulator-enabled), deterministic canned oracle. **Canned numbers (reproducible bit-for-bit)**: baseline 50 served / 86 attempts / 16_040 tokens / total_q=29.90 / q-per-1k=1.86; regulator 50 served / 68 attempts / 11_360 tokens / total_q=30.80 / q-per-1k=2.71. **Cost saved 29.2%**, **total quality +0.90** (regulator actually HIGHER because halting bad retries stops quality from decaying further), **efficiency +0.85 q/1k (+46%)**. 9 of 9 Ambiguous queries cut short by `CircuitBreak(QualityDeclineNoRecovery)` with `mean_delta=0.20` on each. 41 scope_drift_flags (mostly Refactor + some Debug drift). Per-query reset via `export()`/`import()` so `cost/scope/token-stats/quality-history` reset between queries (each query is a separate "task") while `LearnedState + CorrectionPattern` persist. `procedural_warnings = 0` on this stream — not a bug: per-query export drops below-threshold correction records per the documented `Regulator::export` trade-off, so 13 scattered Debug corrections don't reach threshold. Pattern formation path is demonstrated by Demo 3 (within-task accumulation). Three modes (canned / ollama / anthropic) with graceful canned-fallback on transport failure. Zero clippy warnings; Path 1 test count unchanged at 422. Articulation: cost efficiency story holds on synthetic workload; live-LLM numbers at scale remain Session 24b (user-executed follow-up, runnable via `-- ollama` / `-- anthropic` flags).
+ **Session 25 Docs phase — Path 2 public surface** (2026-04-15: `README.md` created at crate root (~160 lines, practitioner vocabulary throughout, honest-scope status table); `docs/regulator-guide.md` created (~320 lines app-integrator guide — event lifecycle, decision recipes, P10 priority explanation, regulator-lifetime guidance (per-task vs per-user), 7 gotchas consolidated from Sessions 21-24); `docs/app-contract.md` §8.1-8.4 added (regulator lifetime / `decide()` timing / correction-pattern persistence trade-off / `QualityFeedback` load-bearing property). Vocabulary audit complete on new public docs — grep for `allostatic|subcortical|LC-NE|cortex|amygdala|hippocamp|arousal|tầng|body.budget` across README.md + regulator-guide.md returns zero hits after two touch-ups: "Tầng 2 delta modulation" → "Compensatory state-retention modulation for local Mamba SSM inference" in README validation section; "body-budget loop" → "depletion loop (see app-contract.md §2)" in regulator-guide. Biological framing preserved in internal docs (brain-map.md, theories.md, module `//!` comments) per `memory/feedback_llm_operational_framing_2026_04_15.md` rule. Session 24 code review found + fixed 2 minor violations: stale module-docstring references (`queries_halted` → `queries_circuit_broken`, `simulate_llm` → `Cluster::canned_quality`, "cumulative cost cap" contradicting per-query reset section) plus unexplained `tokens_in: 25` constant (added WHY comment to match adjacent `wallclock_ms`). Zero clippy regression; test count unchanged at 422.
+ **Session 26 crates.io publication prep** (2026-04-16: `LICENSE` file created at crate root (MIT, 2026 Trian); `Cargo.toml` enriched with `description` (practitioner-vocabulary: "Reliability layer for Rust LLM agents: scope drift, cost circuit breaks, and procedural correction memory as event-driven Decisions"), `readme = "README.md"`, `documentation = "https://docs.rs/nous"`, `keywords = [llm, agent, regulator, reliability, circuit-breaker]`, `categories = [development-tools, rust-patterns]`, and an `include` whitelist that ships only integrator-facing paths (src/, examples/, tests/, Cargo.toml, README.md, LICENSE, principles.md, docs/regulator-{design,guide}.md, docs/app-contract.md) — excluding internal working docs (brain-map, intervention, theories, ssm-*, tang3-*, task-eval-design, eval-artifacts/, CLAUDE.md) whose biological framing and/or `memory/*.md` cross-refs stay out of the published crate. Deep pre-publication code review found + fixed 5 documentation issues: (1) README `[LICENSE](LICENSE)` link pointing at nonexistent file; (2) README competitor matrix overclaiming `Log turns ✓` for Nous (Nous isn't an observability layer — corrected to `—` with explicit "pair with Langfuse" note); (3) regulator-guide §3.4 `ProceduralWarning` recipe had a use-after-move bug (`user_message.clone()` after `user_message` moved into `LLMEvent::TurnStart`); (4) regulator-guide §3.2 statistical imprecision ("false-positive rate ≤ 20%" → "total error rate (FPR + FNR) ≤ 20%" matching the actual test assertion); (5) regulator-guide §6.5 vs app-contract §8.4 inconsistency — guide incorrectly claimed ProceduralWarning depends on QualityFeedback; fixed to match app-contract's correct claim (ProceduralWarning only needs UserCorrection events). Plus 2 Session 24 eval semantic notes added to `task_eval_real_llm_regulator.rs` module doc: (a) `total_quality` is final-retry not best-of-N — all +0.90 quality delta traces to declining-Ambiguous quality × last-retry tracking; (b) `scope_drift_flags` under-reports circuit-broken turns because P10 makes CircuitBreak dominate ScopeDriftWarn in `decide()`. Principle preamble added to `principles.md` noting that internal architecture refs (brain-map, intervention, theories) live in the source repo, not the crate. **`cargo publish --dry-run` passes clean**: 95 files packaged, 1.2 MiB (327.4 KiB compressed), no warnings beyond the expected "aborting upload due to dry run". **Pre-publication clippy cleanup** (same session): fixed all 7 pre-existing warnings — `manual_range_contains` in `resource_allocator.rs`, `len_zero` in `session.rs`, 5× `useless_vec` in `memory/retrieval.rs` (`Some(&vec![...])` → `Some(&[...])`). **Zero clippy warnings on `cargo clippy --lib --tests`**. **Post-clippy P3 audit** found 2 magic-number violations: (a) `regulator_scope_drift_demo.rs` bare `40, 180, 0` → named `CANNED_TOKENS_IN/OUT/WALLCLOCK_MS`; (b) `task_eval_real_llm_regulator.rs` bare `25, 800` → named `EVAL_TOKENS_IN/WALLCLOCK_MS`. Fixed for consistency with Demos 2+3 named-constant pattern. **Final pre-ship sweep** found 4 rustdoc warnings and fixed them all: unresolved intra-doc links `[t]` in `types/intervention.rs:336` (math notation in docstring — wrapped in backticks to escape), redundant explicit link targets in `regulator/correction.rs:21` and `regulator/state.rs:12` (`[Foo](path::Foo)` → `[Foo]` when the intra-doc resolver finds it). **Zero warnings on `cargo doc --no-deps`** — docs.rs will render cleanly. **The crate is ready to ship to crates.io** — the final `cargo publish` is deferred to a fresh session where live-LLM validation can run first (Session 27 entry point in `memory/project_nous_status.md`).
+ **Session 27 Live-LLM validation + publish deferred** (2026-04-16: validation-only session, no code changes; 422 tests unchanged; dry-run skipped as already clean from Session 26. **Phase 2 all 3 live demos against Ollama `phi3:mini` PASSED** with articulation equal-to-or-*stronger* than canned: Demo 1 scope-drift produced `drift_score=1.00` on a 1168-token live response (stronger than canned 0.80 — all 10 response keywords with zero anchor in task), single call took 777 s wallclock; Demo 2 cost-break fired `CircuitBreak(CostCapReached)` on turn 3 at cumulative 1978 real tokens vs the 1200 canned estimate (real phi3:mini over-generates heavily → cost-savings story grows on live, not shrinks); Demo 3 correction-memory roundtripped 3 corrections through 461-byte JSON snapshot and fired `Decision::ProceduralWarning { patterns }` pre-generation on the Phase-3 `TurnStart` with all 3 example_corrections attached. Phase 2 observation worth a Session 28 audit (not blocking publish): Demo 2 turn 2 had cumulative 1770/1000 tokens + mean_quality_last_n = (0.50 + 0.35)/2 = 0.425 < 0.50, so both CostCapReached predicates should trip — yet `decide()` returned `ScopeDriftWarn`. Likely Cost/QualityFeedback/decide sequencing in the demo runs `decide()` before the turn's cost+quality are both drained, so mean_quality_last_n at decide-time is still turn 1's 0.50 alone. Turn 3 fires CircuitBreak correctly, so the demo's articulation still holds. Phase 3 canned baseline re-reproduced bit-for-bit (cost saved 29.2%, q/1k delta +0.85, 9 CB hits QualityDeclineNoRecovery, 41 scope_drift_flags). **50-query live eval launched as detached Windows process PID 15048** via `powershell Start-Process -FilePath target\release\examples\task_eval_real_llm_regulator.exe -ArgumentList ollama -WindowStyle Hidden -RedirectStandardOutput eval_live_ollama.txt`; writing to `eval_live_ollama.txt` + `eval_live_ollama_err.txt` in crate root. Wallclock estimate ~24h on phi3:mini (~154 LLM calls across both arms × 10-13 min each). User explicitly chose "let it run overnight; revisit results next session". **Phase 4 publish deferred to Session 28** — wait for live numbers before shipping `0.1.0`. Session 28 entry point: `tail -30 eval_live_ollama.txt` to check for completion `Per-arm summary` block; `powershell -Command "Get-Process -Id 15048"` to check process still running; compare live 8-metric table to canned baseline in `memory/project_nous_status.md`; if live confirms or improves on canned, ship 0.1.0 with live numbers; if similar, ship with canned + live-run footnote linking to committed `eval-artifacts/` copy; if regression, iterate to 0.1.1 before publish.)

**Identity (honest — refined 2026-04-14 after 6-gap framing review)**:

Nous addresses **persistence at different timescales**. Three timescales have empirical support; two are infrastructure with measured limitations; one is deferred.

| Timescale | Mechanism | Validation | Status |
|-----------|-----------|------------|--------|
| **Within forward pass** | Tầng 2 delta modulation (compensatory gain < 1.0 on HiSPA corridor) | Perplexity eval, 3 runs bit-identical | ✅ **-1.86%** Emotional, 0.00% controls |
| **Across conversation** | Per-cluster strategy EMA in `LearnedState` (survives `export_learned`/`import_learned`) | Tier 1.1 synthetic, 3 seeds, 2σ bar | ✅ 0.900 vs 0.725 baseline, first_correct 1.67 vs 4.00 |
| **Across turns (resource)** | `body_budget` + `signals.conservation` depletion/replenishment | Tier 1.2/1.3 synthetic | ⚠ Calibrated but REQUIRES app to call `track_cost()`; signal stays ~0 in default path (design intent: fires when cost AND poor outcomes both apply) |
| **Across turns (metacognition)** | `signals.confidence` + `signals.strategy.is_none()` | Tier 1.5 abstention | ⚠ TIE F1=1.00 vs per-cluster smart baseline (infrastructure-only on simple task) |
| **Across turns (fatigue)** | `signals.recent_quality` EMA | Tier 1.7 abrupt, 1.8 gradual | ⚠ Regime-dependent: SLOWER than 5-sample rolling avg on abrupt drops (8 vs 4 turns latency); Pareto split on noisy gradual |
| **Across sessions (memory atoms)** | `MemoryStore` trait + `hybrid_recall` | — | ⏸ Deferred (async wiring) |

Reading: **3 wins + 2 measured limitations + 1 deferred**. Not "5/6 ✅".

**Framing note**: The 6-gap list previously treated "session amnesia (learned)" and "reward learning" as separate gaps, but both are validated by the **same** mechanism (per-cluster EMA in LearnedState) via the **same** Tier 1.1 eval. The timescale frame above avoids that double-counting. See `docs/intervention.md` Phase 7 table for the mapping from old 6-gap names to new timescale labels.

**Strategic pivot (2026-04-13, refined 2026-04-14)**:

Research (Barrett 2025, MAST NeurIPS 2025, DEER 2025) motivated a shift from "improve model perplexity" to "make AI system function adaptively on task metrics." Direction accepted as aspiration; execution checked via CR5 interrogation (`memory/project_cr5_check_pivot_2026_04_13.md`). 2026-04-14 audit further refined by splitting the "6 structural gaps" framing into the timescale table above — which surfaces the double-counting of session-amnesia/reward-learning and makes the measured-limitation rows explicit. See `memory/project_strategic_pivot_2026_04_13.md` for the original pivot analysis.

**Phases completed:**
- Phase 1: Types + Math + Detector (foundation)
- Phase 2: Fast Cognitive Modules (convergence loop, 0 LLM calls, <25ms)
- Phase 3: Subcortical modules (predictive coding, basal ganglia, virtual body, conversation dynamics)
- Phase 4: Kernel + AI abstraction (errors, EventBus, AiProvider trait, request builder, response parser, plugin system, pipeline executor)
- Phase 5: Memory system (MemoryStore trait, importance/forgetting, hybrid retrieval, consolidation)
- Phase 6: Adaptive thresholds (precision gain control)

**Cortical modules removed** (2026-04-09): decomposer, orchestrator, dimension_registry, dialogue_state, causal_graph, dmn_creative — these duplicated cortical work the LLM already performs. See `docs/intervention.md` for the "two brains" finding.

**Non-cortical audit** (2026-04-10 → 2026-04-11): Applied P9 (compensate don't amplify) + P10 (signal ordering & gating) across 3 passes. Pass 1: deleted 3 orphan modules + split prefrontal → locus_coeruleus. Pass 2: after SSM research synthesis, deleted belief_state cortical duplication + simplified world_model::perceive + refactored dynamics to use PE as familiarity proxy. Review cleanup: removed dead code chain (GateType::Familiar, GateRewardTracker, HabituationTracker, context_topics threading, orphan UserBeliefs/KnowledgeBeliefs/PredictedIntent), **fixed critical drift where sensory_pe was effectively constant 0.3**, updated resource_allocator + thalamic_gate module docs to honest scope framing. 291 tests pass. Perplexity eval unchanged: -1.86% emotional / 0.00% controls.

**Intervention tầng:**
- Tầng 1 (Sampling Modulation): CognitiveSampler, compute_sampling_override(), InferenceEngine — modulates output selection
- Tầng 2 (Delta Modulation): compute_delta_modulation(), CognitiveModel trait, forward_cognitive() — modulates HOW model thinks via SSM delta scaling. CR5 surgical fix 2026-04-14: `GateType::Routine` short-circuits to passthrough (addresses hs_arousal Technical regression by trusting gate classification over churn-derived arousal). Perplexity re-eval still owed.
- Tầng 3 (Architecture Integration): CognitiveGate, CognitiveMambaWithGate — learnable layer IN model that reads cognitive state FROM hidden activations and modulates downstream processing. Parameters learned end-to-end. Gate starts near-passthrough (alpha ~ 0.05). Training infrastructure: forward_train(), from_pretrained_with_gate(), AdamW optimizer on gate params only.
- Tầng 4 (Structural Compensation): BottleneckSteering — scales mixer output channels at routing bottleneck layer (Layer 20 in mamba-130m). Compensates structural information constraint (KL=813). Based on Mohan et al. 2026 (arXiv 2602.22719, +8.27% avg on task accuracy). Calibration infrastructure: variance-based channel selection (438 high / 154 mid / 176 neutral — matches paper's 435/155/178). Eval finding: amplify (5×) hurts perplexity even with calibration; compensatory (0.8×/0.9× selective) nearly neutral. Paper metric (task accuracy) ≠ Nous metric (perplexity). P9 confirmed at Tầng 4.

**Eval finding** (2026-04-09): `examples/eval_intervention.rs` — Tầng 2 delta modulation works end-to-end. Emotional stress → Phasic gain, KL=0.006, 5/24 layers. Thalamocortical feedback loop closed: gate output → `inject_gate_feedback()` → arousal evolves per-token (0.40→0.19 over 15 tokens). Loop structurally complete; needs CognitiveMambaWithGate for non-zero gate alpha.

**Perplexity eval** (2026-04-10): `examples/perplexity_eval.rs` + `examples/diagnose_harm.rs`:
- gain > 1.0 (amplify): **HURTS** — +5.21% perplexity at gain=1.2, +15.7% at gain=1.5
- gain < 1.0 (compensate): **HELPS** — -2.87% perplexity at gain=0.8, -1.86% at gain=0.9
- **Principle: Don't amplify what model already does. Compensate what model structurally lacks.**
- Tầng 2 mapping corrected: all signals push gain ≤ 1.0 (compensatory retention)
- **After correction**: Emotional -1.86% HELPS, Technical/Creative/Routine 0.00% (controls perfect)
- **First empirical evidence Nous adds value**: selective improvement on emotional text, no harm elsewhere.

**Bottleneck eval** (2026-04-13): `examples/bottleneck_eval.rs` — Tầng 4 bottleneck steering at Layer 20:
- Uniform 5× (all channels): **CATASTROPHIC** — +242% emotional, +711% creative
- Calibrated 5×/2× (variance-selected channels): still hurts (+186% emotional) — selectivity helps but 5× too aggressive for perplexity
- Calibrated compensatory (0.8×/0.9× selective): **NEARLY NEUTRAL** — +1.7% emotional, -0.36% routine
- **Key insight**: paper measured task accuracy (+8.27%), Nous measures perplexity. Amplify helps classification, hurts generation. P9 validated at Tầng 4: compensatory direction is the path for generation quality.
- Calibration infrastructure validated: variance-based channel selection produces 438/154/176 split (paper: 435/155/178)
- **Compound test** (T2 + T4): T2 alone = -1.86% emotional, T4 comp alone = +1.70%, **T2+T4 = -0.06%** — T4 cancels T2's benefit. No synergy. T4 compensatory attenuates mixer output (reduces info flow) which conflicts with T2 compensatory (retains state). **Tầng 2 delta modulation remains the only proven perplexity mechanism.** Bottleneck infrastructure ready for task-accuracy eval if added later.

**Theory wiring** (post-Phase 6): Aligned code with 8 principles from `docs/theories.md`:
- Principle 2 (Precision): convergence → `build_threshold_context()`; adaptive_thresholds → `get_adaptive_threshold()`
- Principle 4 (Affect): `body_budget` in WorldModel, affect pre-colors all thresholds
- Principle 7 (Learning): strategy detection + per-topic EMA + efference copy in `consolidate()`
- Principle 8 (Allostasis): body_budget depletes/replenishes, feeds into thresholds
- See `docs/brain-map.md` for full status (8/8 principles wired)

**Phase 7 API** (2026-04-14) — allostatic controller surface:

```rust
use noos::session::CognitiveSession;

// Create session (optionally with persisted learned state).
let mut session = CognitiveSession::new();
// or: let session = CognitiveSession::with_learned(loaded_state, 64);

// Per-turn: process user input, get allostatic signals.
let turn = session.process_message("User input");
turn.signals.conservation;    // [0,1] — invest or conserve?
turn.signals.salience;        // [0,1] — urgent/novel?
turn.signals.confidence;      // [0,1] — trust assessment?
turn.signals.strategy;        // Option<ResponseStrategy> — learned recommendation
turn.signals.gain_mode;       // Phasic/Tonic/Neutral
turn.signals.recent_quality;  // [0,1] — EMA of response quality
turn.signals.rpe;             // [-1,+1] — most recent reward PE

// Application generates response using turn.sampling (Tầng 1) +
// turn.delta_modulation (Tầng 2) if using local inference.

// Report actual cost (closes allostatic loop).
session.track_cost(0.7);       // [0,1] normalized effort

// Report quality (feeds reward learning).
session.process_response("response text", 0.85);

// Between turns: between-turn maintenance (budget replenishment).
session.idle_cycle();

// End of session: persist learned state.
let snapshot = session.export_learned();
// serialize snapshot (serde_json), store, restore next session.
```

Full demo: `cargo run --example allostatic_demo` (6 scenarios, no candle needed).

## Architecture

```
src/
├── lib.rs                         # Public API surface + re-exports
├── session.rs                     # CognitiveSession: high-level API (perceive→converge→sample→learn)
│
├── types/                         # Data structures — no logic, just shape
│   ├── mod.rs                     # Module declarations
│   ├── belief.rs                  # SharedBeliefState, AffectState, TopicBeliefs (opaque hash), Predictions (minimal)
│   ├── gate.rs                    # GateType (3: URGENT/ROUTINE/NOVEL), GateResult, GateContext, ProblemType (5)
│   ├── memory.rs                  # MemoryAtom, Synapse, AtomType (6), SynapseType (9)
│   ├── intervention.rs             # CognitiveState, InterventionDepth, SamplingOverride, LogitBias (Tier 1) + DeltaModulation, LayerTarget, ForwardResult (Tier 2)
│   └── world.rs                   # WorldModel, LearnedState, GainMode, ResponseStrategy, StrategyConfidence, DynamicsState
│
├── math/                          # Pure math — no domain knowledge
│   ├── mod.rs                     # Re-exports
│   ├── vector.rs                  # cosine_similarity(f32), clamp(f64)
│   └── softmax.rs                 # softmax with temperature (Desimone & Duncan 1995)
│
├── errors.rs                      # NousError, NousResult (P5: fail-open with context)
│
├── kernel/                        # Plugin system + pipeline infrastructure
│   ├── mod.rs                     # Module declarations
│   ├── events.rs                  # EventBus: typed pub/sub (Global Workspace broadcast, Dehaene 2001)
│   ├── plugin.rs                  # SemanticPlugin trait + PluginRegistry + PluginCapability
│   └── pipeline.rs                # Pipeline executor: sequential composition steps with event emission
│
├── ai/                            # AI provider abstraction (P4 trait boundary)
│   ├── mod.rs                     # Module declarations
│   ├── intervention.rs             # InferenceProvider + LogitIntervenor traits (Tier 1 intervention interface)
│   ├── provider.rs                # AiProvider + EmbeddingProvider traits, CompletionRequest/Response, StreamChunk
│   ├── request.rs                 # build_provider_request() for Anthropic/OpenAI/Google (P3: single function)
│   └── response.rs                # parse_sse_line() + parse_full_response() for all providers
│
├── inference/                     # In-process inference — the unified brain (Tầng 1 + 2 + 3 + 4)
│   ├── mod.rs                     # Module declarations
│   ├── bottleneck.rs              # BottleneckSteering: Tầng 4 activation scaling at routing bottleneck (candle feature flag)
│   ├── cognitive_model.rs         # CognitiveModel trait: forward_cognitive() with DeltaModulation (Tầng 2)
│   ├── cognitive_gate.rs          # CognitiveGate: learnable cognitive layer IN model (Tầng 3, candle feature flag)
│   ├── mamba.rs                   # CognitiveMambaModel + CognitiveMambaWithGate + HfTokenizer (candle feature flag)
│   ├── sampler.rs                 # CognitiveSampler: logit bias + penalties + temp + top-p + sampling
│   ├── model.rs                   # LocalModel trait — in-process model abstraction (candle impl behind feature flag)
│   ├── tokenizer.rs               # NousTokenizer trait — text ↔ token conversion
│   └── engine.rs                  # InferenceEngine: model + tokenizer + cognitive sampler = unified brain
│
├── memory/                        # Memory system (P4 trait boundary)
│   ├── mod.rs                     # Module declarations
│   ├── store.rs                   # MemoryStore trait + InMemoryStore (Eichenbaum 2004)
│   ├── importance.rs              # Ebbinghaus forgetting with interference decay (Model D, McClelland 1995 CLS)
│   ├── retrieval.rs               # Hybrid recall: vector + spreading activation + context re-ranking
│   └── consolidation.rs           # Episodic→semantic clustering + pruning (Diekelmann 2010)
│
├── cognition/                     # Non-cortical brain modules (LLM = cortex, Nous = subcortical)
│   ├── mod.rs                     # Module declarations
│   ├── detector.rs                # P3: topic extraction utility (used purely as opaque cluster hash)
│   ├── emotional.rs               # Arousal heuristic (interim scalar, honest naming) + Pavlovian threat EMA
│   ├── hs_arousal.rs              # SSM hidden state → arousal: state churn as LC unsigned PE (Grella 2024)
│   ├── signals.rs                 # CognitiveSignals: application-facing allostatic interface (Phase 7, Barrett 2025)
│   ├── thalamic_gate.rs           # Compute-saving routing (3-type: URGENT/ROUTINE/NOVEL), arousal+pressure feedback
│   ├── locus_coeruleus.rs         # LC-NE: gain mode (phasic/tonic) + arousal fast override (Aston-Jones 2005)
│   ├── belief_state.rs            # Affect update + next-turn topic cluster hash (cortical duplication deleted)
│   ├── world_model.rs             # perceive() + consolidate() + maintain() — allostatic/body-state accounting
│   ├── resource_allocator.rs      # Softmax char-budget allocation for retrieval subsystems (application infrastructure)
│   ├── convergence.rs             # Thalamocortical loop: damped belief propagation — 5 bidirectional connections (P10 gating gap)
│   ├── dynamics.rs                # Conversation regime (Murray 2014) — uses PE-based familiarity proxy (not topic regex)
│   ├── adaptive_thresholds.rs     # Universal precision control (P10 single source of gating) (Friston 2010)
│   ├── intervention.rs            # Brainstem → cortex: cognitive state → sampling override (Tầng 1)
│   └── delta_modulation.rs        # LC → cortex: cognitive state → SSM delta scaling (Tầng 2, compensatory)
│
└── regulator/                     # Path 2 external regulatory layer (Sessions 16-20, 2026-04-15)
    ├── mod.rs                     # Regulator + LLMEvent + Decision + CircuitBreakReason + ConfidenceSpan + CorrectionPattern (#[non_exhaustive], extended with example_corrections Session 20); public API evolves 16→20
    ├── correction.rs              # CorrectionStore + opaque pattern extraction (pattern_name=`corrections_on_{cluster}`, no English regex — P9b-compliant). 3 constants (MIN_CORRECTIONS_FOR_PATTERN=3, MAX_EXAMPLE_CORRECTIONS=3, MAX_CORRECTIONS_PER_CLUSTER=20). Session 20.
    ├── cost.rs                    # CostAccumulator (cumulative token/wallclock + rolling quality history) + normalize_cost (Cost → [0,1] for session.track_cost) + 8 constants. Session 19.
    ├── scope.rs                   # ScopeTracker (keyword-overlap drift reusing detector::extract_topics + to_topic_set + count_topic_overlap per P3) + DRIFT_WARN_THRESHOLD (Session 18)
    ├── state.rs                   # RegulatorState persistence envelope (split from mod.rs in Session 20). Wraps LearnedState + correction_patterns with #[serde(default)] for pre-Session-20 backcompat.
    └── token_stats.rs             # TokenStatsAccumulator (rolling logprob window + mean-NLL → confidence) + structural_confidence fallback (Session 17, R2 hybrid)
```

**Audit 2026-04-10 (non-cortical cleanup, pass 1)**:
- Deleted 3 orphan modules: `predictive_coding`, `basal_ganglia`, `virtual_body` (operated on cortical constructs, no external callers)
- Deleted `types/reasoning.rs` (types only used by deleted modules)
- Split `prefrontal.rs` → `locus_coeruleus.rs` (kept LC-NE, deleted dlPFC/mPFC/ACC cortical parts — no external callers of those)
- Removed `GoalState` type, `LearnedState.{goal, context_vector, user_topic_vector}`, `WorldModel.{somatic_state, abstract_pe}` (orphaned after deletions)
- Updated `emotional.rs` module doc with honest heuristic naming (not a claim to implement amygdala; regex scalar proxy for downstream compensatory modules)
- Added P10 gating sections to `convergence`, `integration` (later removed phase 13), `adaptive_thresholds`, `delta_modulation`

**Audit 2026-04-11 (non-cortical cleanup, pass 2)** — after SSM research synthesis (see `memory/feedback_ssm_research_2026_04.md`):
- Deleted `belief_state::update_topic_beliefs` (regex topic tracking, cortical duplication)
- Deleted `belief_state::compute_topic_pe` (topic-level prediction error, cortical)
- Deleted `belief_state::update_knowledge_beliefs` (unused data structurer)
- Deleted `TopicBeliefs.familiarity` field, `TopicBeliefs.intent_topics`, `is_topic_familiar()`, `TOPIC_PE_FAMILIAR_THRESHOLD`
- Simplified `world_model::perceive` — `sensory_pe` is now just arousal-level surprise (no topic_pe, no intent_pe)
- Refactored `dynamics.rs` — uses `sensory_pe` as familiarity proxy (low PE = familiar) instead of `topic.familiarity`. New constants: `FAMILIAR_PE_THRESHOLD=0.4`, `DEEP_FAMILIAR_PE_THRESHOLD=0.25`
- Research justification: Mamba's attention natively tracks topics across full context; regex extraction was cortical duplication. See `docs/intervention.md` "SSM-specific intervention research" section.

**Review pass (2026-04-11 after audit)** — principle compliance review found dead code chain left by pass 2:
- **Deleted `GateType::Familiar` variant** — can never be returned after cortical topic tracking removed. Gate now 3-type (URGENT/ROUTINE/NOVEL).
- **Deleted `check_familiar()`, `FAMILIAR_MIN_CONTEXT_TOPICS`, `FAMILIAR_TOPIC_OVERLAP_RATIO`** — dead after Phase B.
- **Deleted `GateContext.context_topics` field** — always empty, threaded through dead code paths.
- **Deleted `_context_topics`, `_skip_gate` params from `perceive()`** — dead parameters.
- **Deleted `context_topics` threading through `converge()`** — dead threading.
- **Deleted `threshold_familiar_overlap()` from adaptive_thresholds** — never called.
- **Deleted `GateRewardTracker`, `HabituationTracker`, `GateRewards`, `LearnedState.gate_rewards`, `GateAdjustment`** — orphaned infrastructure, never instantiated outside self-tests.
- **Fixed `resource_allocator.rs::infer_task_load`** — Familiar branch → Routine branch.
- **Fixed critical drift: `sensory_pe` was effectively constant 0.3** because `Predictions.next_arousal` was never populated. Changed to compute arousal delta (turn-over-turn surprise), no predictions needed. Matches Barrett 2017 allostasis (previous state IS the homeostatic expectation).
- **Deleted `Predictions.next_arousal`, `next_intent`, `needed_knowledge`** — all orphaned.
- **Deleted `UserBeliefs`, `KnowledgeBeliefs`, `UserEmotionalState`, `PredictedIntent`** — all orphaned type infrastructure.
- **Deleted `SharedBeliefState.user`, `SharedBeliefState.knowledge`** — orphaned fields.
- **Deleted `PREDICTION_DECAY_RATE`** constant — no longer used.
- **Updated `resource_allocator.rs` `//!` doc** — honest about being application-level infrastructure, not neural computation. The softmax mechanism is real (Desimone & Duncan), but "layers" allocated are retrieval subsystems, not brain regions.
- **Updated `thalamic_gate.rs` `//!` doc** — honest about being compute-saving pipeline routing, not cognitive thalamic relay.
- **Rewrote 2 dead Familiar tests** → replaced with arousal amplification + pressure-damped urgent tests.
- Tests: 321 (pre-pass-1) → 296 (post-pass-1) → 292 (post-pass-2) → **291** (post-review cleanup)
- Pipeline validation: perplexity eval still shows -1.86% emotional / 0.00% controls (3 runs bit-identical)
- 12 clippy warnings (down from 13 baseline, 0 new)

## Core Algorithm: Convergence Loop

The most distinctive algorithm. All messages pass through damped iterative settling with 5 bidirectional feedback connections (thalamocortical gamma cycles). Converges in 2-3 iterations, <25ms, 0 LLM calls.

Details: pipeline diagram, 5 connections, constants → `docs/brain-map.md` §7 Convergence Loop.

## Module Dependency Graph

```
convergence
├── adaptive_thresholds (build_threshold_context — Principle 2+4)
├── world_model (perceive — updates body_budget, arousal-level sensory_pe)
│   ├── belief_state (update_affect)
│   │   └── emotional (compute_arousal — interim heuristic)
│   └── detector (extract_topics — opaque cluster hash only, not cognitive)
├── thalamic_gate (classify_gate_with_feedback — receives affect-modulated arousal)
│   └── detector
├── locus_coeruleus (gain_mode, set_arousal, nudge_gain_from_confidence)
├── resource_allocator (allocate_context_budget, compute_resource_pressure)
│   └── math (clamp, softmax)
└── math (clamp)

world_model.consolidate (post-response learning)
├── detector (detect_response_strategy, extract_topics — Principle 7)
└── updates: predictions, RPE, body_budget, strategy EMA, efference copy compliance

dynamics (temporal hierarchy — regime feeds into WorldModel)
└── types/world (WorldModel, DynamicsState)

delta_modulation (Layer 3 — cognitive state → SSM delta scaling)
├── types/intervention (CognitiveState, DeltaModulation, LayerTarget)
├── types/world (GainMode)
└── math (clamp)

inference (Layer 7 — above cognition, integrates model + cognitive state)
├── cognition/intervention (compute_sampling_override — Tầng 1)
├── cognition/delta_modulation (compute_delta_modulation — Tầng 2)
├── inference/cognitive_model (CognitiveModel trait — Tầng 2 forward)
├── inference/cognitive_gate (CognitiveGate — Tầng 3 learnable layer, candle feature flag)
│   └── inference/mamba (RmsNorm — shared normalization)
├── inference/bottleneck (BottleneckSteering — Tầng 4 activation scaling, candle feature flag)
├── inference/mamba (CognitiveMambaWithGate — Tầng 3 model with embedded gate + Tầng 4 bottleneck)
│   ├── inference/cognitive_gate (CognitiveGate, CognitiveGateConfig)
│   └── inference/bottleneck (BottleneckSteering, BottleneckConfig)
├── math/softmax (softmax_f32 — P3 single source)
└── types/intervention (CognitiveState, SamplingOverride, DeltaModulation, ForwardResult)

regulator (Path 2 — external regulatory layer, Sessions 16-20 shipped)
├── session (CognitiveSession — wrapped Path 1 pipeline)
│   └── [all Path 1 dependencies inherited]
├── types/world (LearnedState — for RegulatorState.learned)
├── regulator/token_stats (TokenStatsAccumulator — per-turn rolling logprob
│   window, mean-NLL confidence, structural fallback; P9b-compliant)
├── regulator/scope (ScopeTracker — per-turn task/response keyword bags +
│   set-difference drift metric)
├── regulator/cost (CostAccumulator — cumulative token/wallclock + rolling
│   quality history + normalize_cost; drives CircuitBreak predicates in
│   decide(); feeds session.track_cost)
├── regulator/correction (CorrectionStore — per-(user, cluster) record of
│   corrections + opaque structural pattern extraction; drives
│   ProceduralWarning in decide())
├── regulator/state (RegulatorState — persistence envelope; wraps
│   LearnedState + correction_patterns with #[serde(default)] backcompat)
└── cognition/detector (extract_topics + to_topic_set + count_topic_overlap
    + build_topic_cluster — P3 shared utilities; build_topic_cluster also
    consumed by world_model for LearnedState.response_strategies keying)
   # Path 2 infrastructure phase COMPLETE. Sessions 21-23 = 3 flagship demos.
```

## Constants

All constant values, brain citations, and per-module tables → **`docs/brain-map.md`** (single source of truth, P3).

Critical rules protecting constants → `principles.md` CR1-CR4.

## Coding Conventions

### Rust-Specific
- **No `unsafe`** — ever
- **No `unwrap()`** — use `Result`, `Option`, or safe defaults
- **Pure functions preferred** — document any mutation in function doc
- **`#[derive(Debug, Clone, Serialize, Deserialize)]`** on all public types
- **`LazyLock`** for compiled regexes (lazy static, thread-safe)
- **`f64`** for all cognitive values (arousal, confidence, PE, etc.)
- **`f32`** only for embedding vectors (memory efficiency)

### Naming
- Files: `snake_case.rs`
- Types/Enums: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Module-level constants: `const` (not `static`)

### Documentation
- Every public function has `///` doc comment
- Every constant has `///` or `//` comment explaining WHY this value
- Brain citations in module-level `//!` doc comments
- Reference paper: `Author Year (description)`

### Testing
- Tests in `#[cfg(test)] mod tests` at bottom of each file
- Test names: `snake_case` describing behavior (e.g., `routine_vietnamese`)
- One assertion per concept (multiple asserts OK if testing same property)
- No mocks — pure functions make mocks unnecessary

### Error Handling
- **Fail-open pattern**: on error, return safe defaults instead of crashing
  - `allocate_context_budget()` → `Option<AllocationResult>` (None = use defaults)
  - `compute_resource_pressure(None)` → 0.5 (neutral)
  - `converge()` error → return perceive-only model
- **`thiserror`** available for future phases (kernel errors, persistence errors)

## Dependencies

| Crate | Purpose | Why this one |
|-------|---------|-------------|
| `serde` + `serde_json` | Serialization | De facto standard, needed for persistence |
| `regex` | Pattern matching | Arousal detection, gate classification, topic extraction |
| `thiserror` | Error types | NousError enum (P5 fail-open with context) |
| `tokio` | Async runtime | AI provider streaming, plugin lifecycle, pipeline execution |
| `async-trait` | Async in traits | AiProvider, EmbeddingProvider, SemanticPlugin, MemoryStore |
| `futures` | Stream utilities | Streaming response handling |
| `rand` | Random sampling | Weighted token sampling in CognitiveSampler (stochastic resonance, Benzi 1981) |
| `candle-core` (optional) | Tensor ops | In-process inference (behind `candle` feature flag) |
| `candle-nn` (optional) | Neural network layers | Mamba model implementation (behind `candle` feature flag) |
| `approx` (dev) | Float comparison | `assert_relative_eq!` in tests |

## Documentation Map

| File | Purpose | When to read |
|------|---------|-------------|
| `CLAUDE.md` | Working context (this file) | Auto-loaded every conversation |
| `principles.md` | 10 coding principles + 5 critical rules | Before writing any code |
| `docs/theories.md` | 8 computational principles — destination (what Nous should become) | Before designing new features |
| `docs/intervention.md` | Intervention architecture — compensate don't amplify, findings chain SC→Nous | Before designing AI integration or module wiring |
| `docs/brain-map.md` | Implementation status — where we are vs theories + all constants (P3 single source) | When adding modules or checking gaps |
| `README.md` | Crate root — public-facing overview, quick start, Path 2 shipped status, competitor comparison, eval numbers (Session 25) | External readers / crates.io visitors |
| `docs/app-contract.md` | Nous ↔ application semantic contract — signal meanings, body_budget, LearnedState ownership, closed-loop requirements, app-level pitfalls (Path 1 + Path 2 §8 extension) | When embedding Nous in an application or reasoning about signal interpretation |
| `docs/regulator-guide.md` | **Path 2 integrator's guide (Session 25)**. Event lifecycle, Decision handling recipes, P10 priority, regulator-lifetime rules, 7 gotchas consolidated from Sessions 21-24 | Before wiring `Regulator` into an agent loop |
| `docs/task-eval-design.md` | Task-eval methodology + tier table (1.1–1.8 shipped, 2.1 v3 shipped) + 2σ bar criteria + Tier 2 candidates | When designing new evals or validating Phase 7 claims |
| `docs/regulator-design.md` | **Path 2 design doc (Session 15)**. Regulator API, LLMEvent/Decision enums, migration plan, Sessions 16-25 implementation roadmap | Before implementing any regulator/ module code |
| `docs/tang3-roadmap.md` | Tầng 3 roadmap — functional circuit activation via subspace steering | Before designing Tầng 3 features |
| `docs/ssm-cognitive-state.md` | **Key finding**: SSM hidden state IS cognitive state. Gate reads hs, not xs. 45x better differentiation. | Before designing Tầng 4 |
| `docs/mamba-research.md` | Mamba/SSM architecture analysis: injection points, Falcon Mamba specs | Before SSM-specific implementation |

## Build & Test

```bash
cargo build              # Build library (default: no candle)
cargo build --features candle  # Build with candle (requires MinGW dlltool or MSVC)
cargo test               # Run all tests
cargo clippy             # Lint (must be 0 warnings)
cargo doc --open         # Generate docs
cargo run --example delta_modulation  # Demo Tầng 2 delta modulation
cargo run --example allostatic_demo   # Demo Phase 7 allostatic controller (no candle needed)
cargo run --example agent_simulation  # Three-way fair comparison: naive vs simple-retry (fair baseline, no Nous) vs allostatic warm-start

# Task-eval suite (synthetic harnesses validating the 6-gap claims — see docs/task-eval-design.md)
cargo run --example task_eval_synthetic        # Tier 1.1 cross-session reward learning (Nous warm 0.900 vs baseline 0.725)
cargo run --example task_eval_conservation     # Tier 1.2 conservation signal calibration
cargo run --example task_eval_multi_signal     # Tier 1.3 signal compounding (+2.73 over smart baseline at budget=12)
cargo run --example task_eval_budget_sweep     # Tier 1.4 Pareto envelope across budget caps 4-20
cargo run --example task_eval_abstention       # Tier 1.5 metacognition (Nous = baseline F1=1.00, infrastructure-only)
cargo run --example task_eval_fatigue          # Tier 1.7 abrupt fatigue (Nous slower than rolling avg, negative)
cargo run --example task_eval_fatigue_gradual  # Tier 1.8 gradual + noisy fatigue (Pareto split: Nous +q, smart -harm)
cargo run --example task_eval_diagnose         # Diagnostic for synthetic eval cluster mismatch
cargo run --example gate_classification_diagnose # Diagnostic for thalamic_gate classification on perplexity texts
cargo run --features candle --release --example task_eval_real_llm_abstention  # Tier 2 Path B skeleton (real-LLM abstention calibration, SKELETON: stubbed model signals pending wiring)

# Path 2 regulator demos (Sessions 21-23)
cargo run --example regulator_scope_drift_demo              # Demo 1 canned — ScopeDriftWarn on pre-written drifted response
cargo run --example regulator_scope_drift_demo -- ollama    # Demo 1 live via Ollama (requires `ollama serve`)
cargo run --example regulator_scope_drift_demo -- anthropic # Demo 1 live via Anthropic (requires ANTHROPIC_API_KEY)
cargo run --example regulator_cost_break_demo               # Demo 2 canned — CircuitBreak(CostCapReached) after 3 retry turns
cargo run --example regulator_cost_break_demo -- ollama     # Demo 2 live via Ollama
cargo run --example regulator_cost_break_demo -- anthropic  # Demo 2 live via Anthropic
cargo run --example regulator_correction_memory_demo        # Demo 3 canned — ProceduralWarning after 3 corrections + export/import roundtrip
cargo run --example regulator_correction_memory_demo -- ollama     # Demo 3 live via Ollama
cargo run --example regulator_correction_memory_demo -- anthropic  # Demo 3 live via Anthropic

# Path 2 real-LLM reliability eval (Tier 2.2, Session 24)
cargo run --example task_eval_real_llm_regulator            # 50-query mixed-cluster eval, canned (baseline +0.85 q/1k, cost -29.2%)
cargo run --example task_eval_real_llm_regulator -- ollama  # Live via Ollama (per-query latency × 50 = long run)
cargo run --example task_eval_real_llm_regulator -- anthropic # Live via Anthropic (requires ANTHROPIC_API_KEY)
```

## Session-End Checklist

Before ending any session that modified code or docs:
1. `cargo test` + `cargo clippy` — must pass, 0 warnings
2. Update this file (`CLAUDE.md`) if files added/removed or architecture changed
3. Update `docs/brain-map.md` if constants or modules changed
4. Update memory (`project_nous_status.md`) with what was done and what's next
