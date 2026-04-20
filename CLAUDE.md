# Noos — Reliability infrastructure for Rust LLM agents

## What is Noos

A Rust crate providing a **reliability layer** for LLM agent loops: cost-overrun prevention, spec-drift defense, fragment-confidence flagging, repeated-mistake learning, plus compensatory state-retention for local Mamba/SSM inference. Evidence-first, practitioner-vocabulary positioning; biological/neuroscience framings (allostatic controller, brain subcortex) are internal mechanism references only.

**As of 2026-04-15, Noos is committed to Path 2** (per `memory/project_nous_reframe_llm_perspective_2026_04_15.md`). The core architectural pivot: input adapter rewritten from "regex on user text" → "LLM operation events (tokens, logprobs, corrections, cost)." Implementation plan: `docs/regulator-design.md` + Sessions 16-25 in `memory/project_path2_architecture_plan.md`. **Tầng 2 delta modulation preserved as first-class SSM feature.**

### What's validated (fair baseline, ≥3 seeds or bit-identical runs)

1. **Per-cluster strategy learning across sessions** (`LearnedState` + `export_learned` / `import_learned`)
   - Tier 1.1 synthetic: **0.900 vs 0.725 baseline quality, first_correct 1.67 vs 4.00** (3 seeds, 2σ bar PASS both axes)
   - Tier 1.3 multi-signal synthetic: **+2.73 total quality over smart app-level baseline** on 24-query mixed workload
   - Tier 1.4 budget sweep synthetic: **Noos wins at every tested budget cap 4.0–20.0** (no crossover)
   - **Tier 2.1 real-LLM multi-signal (2026-04-15, v3 authoritative after diagnostic pass)**: ΔNoos−Smart=**+0.37** at 20 queries, budget=7.0 (narrowly >2σ significant, **below 1.0 compound bar**). Naive DirectAnswer beats both Smart (+1.45) and Noos (+1.08). Strategy learning pipeline **operational** (correct-strat rate 55.6% vs Smart 21.7%) but "correct strategy" from synthetic design is **economically wrong on weak LLM** — StepByStep costs 2.8× DirectAnswer without 2.8× quality gain on mamba-130m. Conservation never fires (design intent per Tier 1.2: needs cost + arousal + poor-outcomes, this eval is cost-only). Gibberish never abstains (real pipeline gap: confidence signal lacks lexical OOD detection). See `memory/project_finding_real_llm_multi_signal_2026_04_15.md`. Multi-signal compound currently validated on synthetic ONLY.

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

- **Tier 2 real-LLM benchmark** (2026-04-15 first pull: **negative on weak model**). `examples/task_eval_real_llm_multi_signal.rs` closed the reality loop on mamba-130m. Result: multi-signal compound collapses from synthetic +2.73 to real +0.78 (within noise). Naive DirectAnswer beats both Smart baseline and Noos. Two paths forward: (a) escalate to frontier model (API integration exists in `ai/`) to test if compound is weak-model-specific; (b) diagnose conservation-never-fires + gibberish-never-abstains before re-running. Until one of these shifts the data, allostatic-controller framing remains aspiration rather than description.
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

Ported from Semantic Computer (TypeScript production app). SC proved cognitive algorithms work but found text injection HURTS quality. Noos found the same at activation level: amplifying model processing hurts, compensating model limitations helps.

### Philosophy

Data-driven. Trait-based. Zero `unsafe`. Compiles to native + WASM. **Compensate, don't amplify** — Noos fills gaps the model structurally has, never duplicates what models do well (pattern recognition, language, reasoning).

### Axiom evolution

- SC: "don't duplicate" (text injection hurts, AD-150→187)
- Noos Phase 1-6: "don't amplify, compensate" (gain > 1.0 hurts, gain < 1.0 helps)
- Noos 2026-04-13: allostatic controller as aspiration (pivot) → 2026-04-14 CR5 check: direction defensible, execution partial
- Noos 2026-04-14 post-audit: Occam-minimal identity (3 validated + 2 limitations + 1 deferred); allostatic language reserved for destination, not description

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

P9 was split 2026-04-14 after Tầng 4 bottleneck finding showed gain direction is metric-dependent (paper's task accuracy +8.27% with amplify, Noos perplexity catastrophic with same intervention). Treating P9 as "universal" had caused over-generalization. See `principles.md §P9` and `memory/project_cr5_check_pivot_2026_04_13.md` for the CR5 interrogation.

Critical rules: CR1-CR4 protect constants and invariants. **CR5** requires interrogating implementation before abandoning a principle (8-step template in `principles.md`).

## Current Status

Phase 1-6 + Tầng 1-4 + non-cortical audit + hs_arousal readout + **Phase 7 allostatic API surface** + 4 calibration/safety bugfixes via task-eval validation + **Phase 12 audit cleanup** (P9 split P9a/P9b, VN regex + LearnedState + DMN cleanup, CR5 honest reframing, CR2 invariant tests, N-timescale framing) + **Phase 13 prune** (2026-04-14: integration.rs 557 LOC removed — zero non-test callers — plus CLAUDE.md Occam-minimal identity reframe) + **Phase 14 reality loop first pull** (2026-04-15: multi-signal compound eval on real mamba-130m — +0.78 Δ < 1.0 bar, v3 +0.37 after warmup fix; superseded single-signal Path B skeleton; fixed inherited shuffle bug; `NOUS_DIAGNOSE` env-var diagnostic mode added) + **Phase 15 Path 2 design** (2026-04-15: commit to Rust-native reliability infrastructure pivot; `docs/regulator-design.md` + `memory/project_path2_architecture_plan.md` written; no code changes this phase) + **Session 16 Regulator skeleton** (2026-04-15: `src/regulator/mod.rs` lands public API shape — `Regulator`, `LLMEvent` (6 variants), `Decision` (5 variants with supporting types), `RegulatorState`. Event dispatch minimal: `TurnStart` → `process_message`, `TurnComplete` buffers response, `QualityFeedback` drains buffer into `process_response`. `Token` / `Cost` / `UserCorrection` inert. `decide()` returns `Continue` until Sessions 18-20 wire predicates. Principle review fixed 2 violations: P2 Mutable prefix, P6 private-field peeks → observable-behavior tests + added drain-noop test)
+ **Session 17 TokenStatsAccumulator** (2026-04-15: `src/regulator/token_stats.rs` lands rolling logprob window + entropy-based confidence. R2 logprob-availability decision resolved per architecture plan recommendation — hybrid strategy: primary path uses mean-NLL over rolling window when provider exposes logprobs (OpenAI/vLLM/local candle); fallback path uses language-neutral structural heuristic (length + `?` ratio) when provider lacks logprobs (Anthropic as of 2026-04). `LOGPROB_UNAVAILABLE=0.0` sentinel per design doc convention; non-finite/positive values fail-open to unavailable. `Regulator::confidence()` composes primary + fallback via `confidence_with_fallback`. `Regulator::logprob_coverage()` surfaces per-turn availability. `TurnStart` now resets the window; `Token` feeds it. `signals.confidence` in wrapped session unchanged — plumbing deferred)
+ **Session 18 ScopeTracker** (2026-04-15: `src/regulator/scope.rs` lands keyword-overlap scope-drift detector reusing `cognition::detector::extract_topics` + `to_topic_set` + `count_topic_overlap` per P3 — matches `memory/retrieval.rs` precedent. Metric `drift = |response_kw \ task_kw| / |response_kw|` (response-centric, [0,1], `None` before both sides populated). `DRIFT_WARN_THRESHOLD = 0.5` triggers `ScopeDriftWarn`. Decision checkpoint satisfied: 10 hand-crafted cases audit passes ≤ 20% error bar)
+ **Session 19 CostAccumulator + CircuitBreak** (2026-04-15: `src/regulator/cost.rs` lands cumulative token/wallclock counters + rolling quality history + `normalize_cost` with `TOKEN_COST_WEIGHT = 0.7`. `decide()` gains two CircuitBreak predicates with explicit P10 priority: CostCapReached > QualityDeclineNoRecovery > ScopeDriftWarn > Continue. `with_cost_cap(u32)` builder)
+ **Session 20 CorrectionStore + ProceduralWarning + state.rs split** (2026-04-15: `src/regulator/correction.rs` lands structural correction-pattern extraction — MIN_CORRECTIONS_FOR_PATTERN=3, pattern_name opaque `corrections_on_{cluster}` to avoid English regex (P9b-compliant). `src/regulator/state.rs` splits RegulatorState out of mod.rs and extends with `correction_patterns: HashMap<String, CorrectionPattern>` field using `#[serde(default)]` for pre-Session-20 backcompat. `CorrectionPattern` extended with `example_corrections: Vec<String>` + marked `#[non_exhaustive]` for future field additions. `LLMEvent::UserCorrection` handler attributes correction to `current_topic_cluster` (derived via shared `detector::build_topic_cluster` — P3 move from `world_model.rs`). `decide()` priority updated: CircuitBreak×2 > ScopeDriftWarn > **ProceduralWarning (new, advisory — slots between semantic warning and Continue)** > Continue. Export replays example_corrections on import so patterns survive JSON round-trip. 410 unit + 12 integration = **422 tests pass** (+22 Session 20: 10 correction + 3 state + 9 regulator wiring). 7 clippy warnings on `--lib --tests` (all pre-existing, 0 new). **Path 2 infrastructure phase COMPLETE**; Sessions 21-23 = 3 flagship demos. Run `cargo test 2>&1 | grep "test result"` for current count.
+ **Session 21 Demo 1 — Scope drift intercept** (2026-04-15: `examples/regulator_scope_drift_demo.rs` — end-to-end event flow `TurnStart → TurnComplete → Cost → decide()` producing `Decision::ScopeDriftWarn`. Three modes: `canned` (default, deterministic, no LLM), `ollama` (via `ureq` to localhost:11434/api/chat), `anthropic` (via `ureq` to api.anthropic.com/v1/messages with `ANTHROPIC_API_KEY`). Canned path produces `drift_score=0.80` on a realistic drifted response (task keywords: async/database/fetch_user/keep/logic/lookup/refactor/unchanged; drift keywords: added/await/call/comments/counter/db/doc/duration). Live paths gracefully fall back to canned on transport failure so the demo always runs. No new deps (ureq already in Cargo.toml). Zero clippy warnings on the example; Path 1 test count unchanged at 422. Articulation: baseline Mem0/Letta/Langfuse store or log; `Regulator` emits decision PRE-delivery so the app can auto-strip, re-prompt, or accept.
+ **Session 22 Demo 2 — Cost circuit break** (2026-04-15: `examples/regulator_cost_break_demo.rs` — 3 retry turns with declining quality (0.50 → 0.35 → 0.20), 400 tokens_out each, against `Regulator::with_cost_cap(1_000)`. Turn 3 trips `Decision::CircuitBreak { CostCapReached { tokens_spent: 1200, tokens_cap: 1000, mean_quality_last_n: 0.35 } }` with suggestion "Cost cap reached with poor recent quality. Ask the user to clarify scope or abandon this task." P10 priority demonstrated in output: turns 1-2 fire `ScopeDriftWarn` (advisory, agent continues); turn 3 fires `CircuitBreak` (halt) and dominates the still-live drift signal. `QualityDeclineNoRecovery` also qualifies on turn 3 (mean_delta=0.30 ≥ 0.15, mean=0.35 < 0.5) — `CostCapReached` wins by priority, noted in module doc. Three modes match Session 21 (canned default / ollama / anthropic with graceful fallback). **P3 additional fix same session**: HTTP adapters extracted from Sessions 21 + 22 into `examples/regulator_common/mod.rs` (shared via `#[path]` attribute — Cargo doesn't auto-discover files in `examples/<subdir>/`, so the module compiles only when explicitly included). Single-source `call_ollama`/`call_anthropic` ready for Session 23 reuse. Zero clippy warnings on both demos; Path 1 test count unchanged at 422.
+ **Session 23 Demo 3 — Procedural correction memory** (2026-04-15: `examples/regulator_correction_memory_demo.rs` — the clearest Mem0/Letta differentiator per research agent 3. 3 learning turns on user_123 with varied user messages that all hash to the same cluster `async+auth` (verified empirically: "Make my auth module async" / "Refactor auth to support async" / "Change my auth function to async" all produce `build_topic_cluster → "async+auth"`). Each turn records one correction via `LLMEvent::UserCorrection { corrects_last: true }`. After turn 3, `regulator.export()` yields a `RegulatorState` with 1 `CorrectionPattern` (learned_from_turns=3, confidence=0.15, 3 example_corrections). Snapshot round-trips through `serde_json::to_string/from_str` (461 bytes) simulating process restart. After `Regulator::import(state)`, a new `TurnStart("Add async handling to my auth")` (same cluster) — and CRUCIALLY, **decide() called after TurnStart but BEFORE TurnComplete** so scope tracker's response_keywords is empty and `ScopeDriftWarn` skips — triggers `Decision::ProceduralWarning { patterns }` pre-generation. The 3 example_corrections ride along for the LLM to interpret intent. Reuses `regulator_common::{call_ollama, call_anthropic}` (Session 22 P3 extraction). Imports `MIN_CORRECTIONS_FOR_PATTERN` from `noos::regulator::correction` for the progress display. P5 compliant — serde roundtrip errors use `match` + graceful exit, no `.expect()`. Articulation (in demo take-away): "counts per-cluster corrections structurally (no embedding); fires ProceduralWarning proactively once MIN threshold trips — demonstrably absent from every content-retrieval memory system in Rust/Python LLM ecosystem as of 2026-04". **Path 2 flagship demo phase COMPLETE**; Sessions 21-23 each close a loop competitors cannot. Zero clippy warnings on all 3 demos; Path 1 test count unchanged at 422.
+ **Session 24 Tier 2.2 real-LLM regulator eval** (2026-04-15: `examples/task_eval_real_llm_regulator.rs` — 50-query stream, 4-cluster mix (18 FactQA / 10 Refactor / 9 Ambiguous / 13 Debug), 2 arms (baseline vs regulator-enabled), deterministic canned oracle. **Canned numbers (reproducible bit-for-bit)**: baseline 50 served / 86 attempts / 16_040 tokens / total_q=29.90 / q-per-1k=1.86; regulator 50 served / 68 attempts / 11_360 tokens / total_q=30.80 / q-per-1k=2.71. **Cost saved 29.2%**, **total quality +0.90** (regulator actually HIGHER because halting bad retries stops quality from decaying further), **efficiency +0.85 q/1k (+46%)**. 9 of 9 Ambiguous queries cut short by `CircuitBreak(QualityDeclineNoRecovery)` with `mean_delta=0.20` on each. 41 scope_drift_flags (mostly Refactor + some Debug drift). Per-query reset via `export()`/`import()` so `cost/scope/token-stats/quality-history` reset between queries (each query is a separate "task") while `LearnedState + CorrectionPattern` persist. `procedural_warnings = 0` on this stream — not a bug: per-query export drops below-threshold correction records per the documented `Regulator::export` trade-off, so 13 scattered Debug corrections don't reach threshold. Pattern formation path is demonstrated by Demo 3 (within-task accumulation). Three modes (canned / ollama / anthropic) with graceful canned-fallback on transport failure. Zero clippy warnings; Path 1 test count unchanged at 422. Articulation: cost efficiency story holds on synthetic workload; live-LLM numbers at scale remain Session 24b (user-executed follow-up, runnable via `-- ollama` / `-- anthropic` flags).
+ **Session 25 Docs phase — Path 2 public surface** (2026-04-15: `README.md` created at crate root (~160 lines, practitioner vocabulary throughout, honest-scope status table); `docs/regulator-guide.md` created (~320 lines app-integrator guide — event lifecycle, decision recipes, P10 priority explanation, regulator-lifetime guidance (per-task vs per-user), 7 gotchas consolidated from Sessions 21-24); `docs/app-contract.md` §8.1-8.4 added (regulator lifetime / `decide()` timing / correction-pattern persistence trade-off / `QualityFeedback` load-bearing property). Vocabulary audit complete on new public docs — grep for `allostatic|subcortical|LC-NE|cortex|amygdala|hippocamp|arousal|tầng|body.budget` across README.md + regulator-guide.md returns zero hits after two touch-ups: "Tầng 2 delta modulation" → "Compensatory state-retention modulation for local Mamba SSM inference" in README validation section; "body-budget loop" → "depletion loop (see app-contract.md §2)" in regulator-guide. Biological framing preserved in internal docs (brain-map.md, theories.md, module `//!` comments) per `memory/feedback_llm_operational_framing_2026_04_15.md` rule. Session 24 code review found + fixed 2 minor violations: stale module-docstring references (`queries_halted` → `queries_circuit_broken`, `simulate_llm` → `Cluster::canned_quality`, "cumulative cost cap" contradicting per-query reset section) plus unexplained `tokens_in: 25` constant (added WHY comment to match adjacent `wallclock_ms`). Zero clippy regression; test count unchanged at 422.
+ **Session 26 crates.io publication prep** (2026-04-16: `LICENSE` file created at crate root (MIT, 2026 Trian); `Cargo.toml` enriched with `description` (practitioner-vocabulary: "Reliability layer for Rust LLM agents: scope drift, cost circuit breaks, and procedural correction memory as event-driven Decisions"), `readme = "README.md"`, `documentation = "https://docs.rs/noos"`, `keywords = [llm, agent, regulator, reliability, circuit-breaker]`, `categories = [development-tools, rust-patterns]`, and an `include` whitelist that ships only integrator-facing paths (src/, examples/, tests/, Cargo.toml, README.md, LICENSE, principles.md, docs/regulator-{design,guide}.md, docs/app-contract.md) — excluding internal working docs (brain-map, intervention, theories, ssm-*, tang3-*, task-eval-design, eval-artifacts/, CLAUDE.md) whose biological framing and/or `memory/*.md` cross-refs stay out of the published crate. Deep pre-publication code review found + fixed 5 documentation issues: (1) README `[LICENSE](LICENSE)` link pointing at nonexistent file; (2) README competitor matrix overclaiming `Log turns ✓` for Noos (Noos isn't an observability layer — corrected to `—` with explicit "pair with Langfuse" note); (3) regulator-guide §3.4 `ProceduralWarning` recipe had a use-after-move bug (`user_message.clone()` after `user_message` moved into `LLMEvent::TurnStart`); (4) regulator-guide §3.2 statistical imprecision ("false-positive rate ≤ 20%" → "total error rate (FPR + FNR) ≤ 20%" matching the actual test assertion); (5) regulator-guide §6.5 vs app-contract §8.4 inconsistency — guide incorrectly claimed ProceduralWarning depends on QualityFeedback; fixed to match app-contract's correct claim (ProceduralWarning only needs UserCorrection events). Plus 2 Session 24 eval semantic notes added to `task_eval_real_llm_regulator.rs` module doc: (a) `total_quality` is final-retry not best-of-N — all +0.90 quality delta traces to declining-Ambiguous quality × last-retry tracking; (b) `scope_drift_flags` under-reports circuit-broken turns because P10 makes CircuitBreak dominate ScopeDriftWarn in `decide()`. Principle preamble added to `principles.md` noting that internal architecture refs (brain-map, intervention, theories) live in the source repo, not the crate. **`cargo publish --dry-run` passes clean**: 95 files packaged, 1.2 MiB (327.4 KiB compressed), no warnings beyond the expected "aborting upload due to dry run". **Pre-publication clippy cleanup** (same session): fixed all 7 pre-existing warnings — `manual_range_contains` in `resource_allocator.rs`, `len_zero` in `session.rs`, 5× `useless_vec` in `memory/retrieval.rs` (`Some(&vec![...])` → `Some(&[...])`). **Zero clippy warnings on `cargo clippy --lib --tests`**. **Post-clippy P3 audit** found 2 magic-number violations: (a) `regulator_scope_drift_demo.rs` bare `40, 180, 0` → named `CANNED_TOKENS_IN/OUT/WALLCLOCK_MS`; (b) `task_eval_real_llm_regulator.rs` bare `25, 800` → named `EVAL_TOKENS_IN/WALLCLOCK_MS`. Fixed for consistency with Demos 2+3 named-constant pattern. **Final pre-ship sweep** found 4 rustdoc warnings and fixed them all: unresolved intra-doc links `[t]` in `types/intervention.rs:336` (math notation in docstring — wrapped in backticks to escape), redundant explicit link targets in `regulator/correction.rs:21` and `regulator/state.rs:12` (`[Foo](path::Foo)` → `[Foo]` when the intra-doc resolver finds it). **Zero warnings on `cargo doc --no-deps`** — docs.rs will render cleanly. **The crate is ready to ship to crates.io** — the final `cargo publish` is deferred to a fresh session where live-LLM validation can run first (Session 27 entry point in `memory/project_nous_status.md`).
+ **Session 27 Live-LLM validation + publish deferred** (2026-04-16: validation-only session, no code changes; 422 tests unchanged; dry-run skipped as already clean from Session 26. **Phase 2 all 3 live demos against Ollama `phi3:mini` PASSED** with articulation equal-to-or-*stronger* than canned: Demo 1 scope-drift produced `drift_score=1.00` on a 1168-token live response (stronger than canned 0.80 — all 10 response keywords with zero anchor in task), single call took 777 s wallclock; Demo 2 cost-break fired `CircuitBreak(CostCapReached)` on turn 3 at cumulative 1978 real tokens vs the 1200 canned estimate (real phi3:mini over-generates heavily → cost-savings story grows on live, not shrinks); Demo 3 correction-memory roundtripped 3 corrections through 461-byte JSON snapshot and fired `Decision::ProceduralWarning { patterns }` pre-generation on the Phase-3 `TurnStart` with all 3 example_corrections attached. Phase 2 observation worth a Session 28 audit (not blocking publish): Demo 2 turn 2 had cumulative 1770/1000 tokens + mean_quality_last_n = (0.50 + 0.35)/2 = 0.425 < 0.50, so both CostCapReached predicates should trip — yet `decide()` returned `ScopeDriftWarn`. Likely Cost/QualityFeedback/decide sequencing in the demo runs `decide()` before the turn's cost+quality are both drained, so mean_quality_last_n at decide-time is still turn 1's 0.50 alone. Turn 3 fires CircuitBreak correctly, so the demo's articulation still holds. Phase 3 canned baseline re-reproduced bit-for-bit (cost saved 29.2%, q/1k delta +0.85, 9 CB hits QualityDeclineNoRecovery, 41 scope_drift_flags). **50-query live eval launched as detached Windows process PID 15048** via `powershell Start-Process -FilePath target\release\examples\task_eval_real_llm_regulator.exe -ArgumentList ollama -WindowStyle Hidden -RedirectStandardOutput eval_live_ollama.txt`; writing to `eval_live_ollama.txt` + `eval_live_ollama_err.txt` in crate root. Wallclock estimate ~24h on phi3:mini (~154 LLM calls across both arms × 10-13 min each). User explicitly chose "let it run overnight; revisit results next session". **Phase 4 publish deferred to Session 28** — wait for live numbers before shipping `0.1.0`. Session 28 entry point: `tail -30 eval_live_ollama.txt` to check for completion `Per-arm summary` block; `powershell -Command "Get-Process -Id 15048"` to check process still running; compare live 8-metric table to canned baseline in `memory/project_nous_status.md`; if live confirms or improves on canned, ship 0.1.0 with live numbers; if similar, ship with canned + live-run footnote linking to committed `eval-artifacts/` copy; if regression, iterate to 0.1.1 before publish.)

**Identity (honest — refined 2026-04-14 after 6-gap framing review)**:

Noos addresses **persistence at different timescales**. Three timescales have empirical support; two are infrastructure with measured limitations; one is deferred.

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
- Tầng 4 (Structural Compensation): BottleneckSteering — scales mixer output channels at routing bottleneck layer (Layer 20 in mamba-130m). Compensates structural information constraint (KL=813). Based on Mohan et al. 2026 (arXiv 2602.22719, +8.27% avg on task accuracy). Calibration infrastructure: variance-based channel selection (438 high / 154 mid / 176 neutral — matches paper's 435/155/178). Eval finding: amplify (5×) hurts perplexity even with calibration; compensatory (0.8×/0.9× selective) nearly neutral. Paper metric (task accuracy) ≠ Noos metric (perplexity). P9 confirmed at Tầng 4.

**Eval finding** (2026-04-09): `examples/eval_intervention.rs` — Tầng 2 delta modulation works end-to-end. Emotional stress → Phasic gain, KL=0.006, 5/24 layers. Thalamocortical feedback loop closed: gate output → `inject_gate_feedback()` → arousal evolves per-token (0.40→0.19 over 15 tokens). Loop structurally complete; needs CognitiveMambaWithGate for non-zero gate alpha.

**Perplexity eval** (2026-04-10): `examples/perplexity_eval.rs` + `examples/diagnose_harm.rs`:
- gain > 1.0 (amplify): **HURTS** — +5.21% perplexity at gain=1.2, +15.7% at gain=1.5
- gain < 1.0 (compensate): **HELPS** — -2.87% perplexity at gain=0.8, -1.86% at gain=0.9
- **Principle: Don't amplify what model already does. Compensate what model structurally lacks.**
- Tầng 2 mapping corrected: all signals push gain ≤ 1.0 (compensatory retention)
- **After correction**: Emotional -1.86% HELPS, Technical/Creative/Routine 0.00% (controls perfect)
- **First empirical evidence Noos adds value**: selective improvement on emotional text, no harm elsewhere.

**Bottleneck eval** (2026-04-13): `examples/bottleneck_eval.rs` — Tầng 4 bottleneck steering at Layer 20:
- Uniform 5× (all channels): **CATASTROPHIC** — +242% emotional, +711% creative
- Calibrated 5×/2× (variance-selected channels): still hurts (+186% emotional) — selectivity helps but 5× too aggressive for perplexity
- Calibrated compensatory (0.8×/0.9× selective): **NEARLY NEUTRAL** — +1.7% emotional, -0.36% routine
- **Key insight**: paper measured task accuracy (+8.27%), Noos measures perplexity. Amplify helps classification, hurts generation. P9 validated at Tầng 4: compensatory direction is the path for generation quality.
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
├── errors.rs                      # NoosError, NoosResult (P5: fail-open with context)
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
│   ├── tokenizer.rs               # NoosTokenizer trait — text ↔ token conversion
│   └── engine.rs                  # InferenceEngine: model + tokenizer + cognitive sampler = unified brain
│
├── memory/                        # Memory system (P4 trait boundary)
│   ├── mod.rs                     # Module declarations
│   ├── store.rs                   # MemoryStore trait + InMemoryStore (Eichenbaum 2004)
│   ├── importance.rs              # Ebbinghaus forgetting with interference decay (Model D, McClelland 1995 CLS)
│   ├── retrieval.rs               # Hybrid recall: vector + spreading activation + context re-ranking
│   └── consolidation.rs           # Episodic→semantic clustering + pruning (Diekelmann 2010)
│
├── cognition/                     # Non-cortical brain modules (LLM = cortex, Noos = subcortical)
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
└── regulator/                     # Path 2 external regulatory layer (Sessions 16-20 landed infra; 0.2.0-0.3.0 published 2026-04-16; Session 31 added OTel adapter + adversarial tests)
    ├── mod.rs                     # Regulator + LLMEvent (8 variants; Session-24 Token/Cost/UserCorrection + 0.3.0 ToolCall/ToolResult) + Decision (5 variants, #[non_exhaustive] + #[must_use] since 0.2.1) + CircuitBreakReason (4 variants, #[non_exhaustive]: CostCapReached/QualityDeclineNoRecovery/RepeatedFailurePattern/RepeatedToolCallLoop) + ConfidenceSpan + CorrectionPattern. Includes Path B helpers (corrections_prelude/inject_corrections since 0.2.2) and Path A tool-stats accessors (tool_total_calls/tool_counts_by_name/tool_total_duration_ms/tool_failure_count since 0.3.0). Session 32 added `with_implicit_correction_window(Duration)` builder + `implicit_corrections_count()` accessor — fast same-cluster retries auto-record corrections without explicit `UserCorrection` events. Window setting + per-process counter NOT persisted by export/import (ephemeral signal).
    ├── correction.rs              # CorrectionStore + opaque pattern extraction (pattern_name=`corrections_on_{cluster}`, no English regex — P9b-compliant). 3 constants (MIN_CORRECTIONS_FOR_PATTERN=3, MAX_EXAMPLE_CORRECTIONS=3, MAX_CORRECTIONS_PER_CLUSTER=20). Session 20.
    ├── cost.rs                    # CostAccumulator (cumulative token/wallclock + rolling quality history) + normalize_cost (Cost → [0,1] for session.track_cost) + 8 constants. Session 19.
    ├── otel.rs                    # OpenTelemetry GenAI semantic conventions adapter (Session 31). `events_from_span(&Value) -> Vec<LLMEvent>` — maps `gen_ai.user.message` → TurnStart, `gen_ai.assistant.message` → TurnComplete, `gen_ai.usage.*` + span duration → Cost, `gen_ai.tool.message` → ToolCall / ToolResult (paired when duration present). SDK-idiomatic dict-form JSON only; OTLP/JSON protobuf form callers convert upstream. 13 tests.
    ├── scope.rs                   # ScopeTracker (keyword-overlap drift reusing detector::extract_topics + to_topic_set + count_topic_overlap per P3) + DRIFT_WARN_THRESHOLD (Session 18). Session 31 added 5 adversarial tests documenting known limitations: rank-11 alphabetical truncation, synonym surface-form mismatch, verbose on-topic FP, case-insensitivity, non-English max-drift.
    ├── state.rs                   # RegulatorState persistence envelope (split from mod.rs in Session 20). Wraps LearnedState + correction_patterns with #[serde(default)] for pre-Session-20 backcompat.
    ├── token_stats.rs             # TokenStatsAccumulator (rolling logprob window + mean-NLL → confidence) + structural_confidence fallback (Session 17, R2 hybrid)
    └── tools.rs                   # ToolStatsAccumulator (per-turn tool-call + tool-result history + consecutive-same-tool loop detector). 0.3.0 Path A. TOOL_LOOP_THRESHOLD=5 drives CircuitBreak(RepeatedToolCallLoop). Reset per TurnStart. Session 31 added 6 adversarial tests: tail-run after mixed prefix, diverging args same name, over-threshold count semantics, resolved loops with short tails, alternating A/B (out-of-scope by design), exact boundary pair.

benches/                          # Criterion benchmarks (Session 33) — NOT in Cargo.toml include, internal only
└── regulator.rs                  # 9 benchmarks: per-event dispatch (Token/TurnStart/TurnComplete/Cost/ToolCall), decide() on Continue + ScopeDrift, export→import roundtrip, realistic full-turn. Run `cargo bench --bench regulator`.

bindings/python/                   # PyO3 extension module (Session 30) — NOT in Cargo.toml include
├── Cargo.toml                    # pyo3 = 0.23 with abi3-py39 + extension-module features
├── pyproject.toml                # maturin 1.7+ backend, PyPI name `noos` (renamed from `noos-regulator` on 2026-04-19 for brand consistency with Rust crate)
├── src/lib.rs                    # 5 pyclasses (Regulator / LLMEvent / Decision / CircuitBreakReason / CorrectionPattern). Session 31 added `LLMEvent.from_otel_span_json`. Session 32 added `Regulator.with_implicit_correction_window_secs` + `implicit_corrections_count`. Session 33 added `Regulator.metrics_snapshot`.
├── examples/basic.py             # 4 scenarios mirroring the Rust demos
└── tests/test_regulator.py       # 25 behavioural tests

bindings/node/                     # napi-rs 3.x Node native addon (Session 34, CI green since S36) — NOT in Cargo.toml include
├── Cargo.toml                    # napi = 3 (features = ["napi6"] — S36 bump for BigInt export), napi-derive = 3. No local build.rs; napi-build 2.3.1 is still a transitive build-dep but runs cleanly on Linux/macOS/Windows-MSVC.
├── package.json                  # @napi-rs/cli ^3.0.0 (S36 bump — 2.x's panic with napi 3.x derive macros), npm name `@triangle-technology/noos` (scoped; bare `noos` was squatted on npm), 5 prebuilt target triples
├── src/lib.rs                    # Parallel port of bindings/python: Regulator / LLMEvent / Decision / CircuitBreakReason / CorrectionPattern (napi-rs auto-generates .d.ts). `LLMEvent` struct has `#[napi(js_name = "LLMEvent")]` (S36 — napi-rs 3.x Case::Pascal would rewrite to LlmEvent). OTel helper exposed as freestanding `llmEventsFromOtelSpanJson()` (S36 — mixed `#[napi(factory)]` + plain `#[napi]` static methods in one impl block silently corrupts class registration). Includes `Regulator.metricsSnapshot` + `withImplicitCorrectionWindowSecs`.
├── examples/basic.mjs            # 4 scenarios mirroring the Rust demos
├── __test__/regulator.test.mjs   # 15 behavioural tests via `node --test` (no framework dep)
└── README.md                     # Status: "0.1.0-pre — bindings complete, pending CI binary build for npm. Windows-GNU blocked on libnode.dll; WSL / MSVC / Linux-macOS CI all work."

bindings/python-langchain/         # Pure-Python LangChain + LangGraph adapter (Session 38) — NOT in Cargo.toml include, separate pip package
├── pyproject.toml                # hatchling backend; PyPI name `noos-langchain`; deps `noos>=0.1.0` + `langchain-core>=0.3.0`; Python >=3.9
├── CHANGELOG.md                  # 0.1.0 release notes (sync + async handler, 3 consumption modes, fail-open extraction, LangGraph-via-LC-callbacks)
├── src/noos_langchain/__init__.py    # exports `NoosCallbackHandler` + `AsyncNoosCallbackHandler` + `CircuitBreakError` + `__version__`; module-level docstring with sync / async / LangGraph quick-starts
├── src/noos_langchain/callback.py    # Three-class design: `_BaseHandlerLogic` mixin holds shared state + per-hook method bodies; `NoosCallbackHandler(_BaseHandlerLogic, BaseCallbackHandler)` exposes sync hooks; `AsyncNoosCallbackHandler(_BaseHandlerLogic, AsyncCallbackHandler)` exposes async-def hooks. Both set `raise_error=True` + `run_inline=True` class attrs so `CircuitBreakError` propagates through LC's callback manager instead of being swallowed + stays on the same thread for stable UUID-based tool tracking. Maps `on_chain_start` (root) / `on_chat_model_start` / `on_llm_start` → `TurnStart`; `on_llm_new_token` (opt-in `emit_tokens`) → `Token`; `on_llm_end` → `TurnComplete` + `Cost`; `on_tool_start` / `end` / `error` → `ToolCall` + `ToolResult` (duration_ms from `monotonic_ns` deltas per run_id); `on_llm_error` no-op; `on_chain_end` / `on_chain_error` (root) clears turn flag. Three consumption modes (poll `last_decision`, `on_decision=callback`, `raise_on_circuit_break=True`).
├── src/noos_langchain/_compat.py     # Defensive payload extractors — `extract_user_message` (input/question/query/prompt dict keys + messages[-1] fallback + json.dumps fallback), `extract_token_usage` (OpenAI `token_usage` / Anthropic `usage` / modern LC `usage_metadata` / `generation_info` — 4 paths, fail-open `(0, 0)`), `extract_response_text` (generation.text → message.content → str fallback), `tool_name_from_serialized` (LC 0.3+ `name` key → legacy `id[-1]` → `unknown` default), `extract_chat_messages` (nested messages[0][-1]). None raise to caller.
├── src/noos_langchain/py.typed   # PEP 561 marker — type hints propagate to mypy / pyright
├── examples/basic_smoke.py       # 4 scenarios against fabricated LC payloads — scope drift / tool loop / cost break / procedural memory. No LLM or network.
├── examples/openai_tools_agent.py    # Full `AgentExecutor` + OpenAI tools agent with looping toy tools; demos `raise_on_circuit_break=True`. Requires `OPENAI_API_KEY`.
├── examples/anthropic_tools_agent.py # Same shape as openai_tools_agent but via `langchain-anthropic` ChatAnthropic + `create_tool_calling_agent`. First end-to-end demo hitting a real LLM provider (Claude Haiku — already validated by S36 real-judge eval). Requires `ANTHROPIC_API_KEY`.
├── examples/crewai_agent.py      # CrewAI Agent + Task + Crew demo via the LangChain LLM callback path — pass `NoosCallbackHandler` to `ChatAnthropic(callbacks=[handler])`, no separate `noos-crewai` package needed. Tool-loop detection works iff underlying LLM's tool-call events propagate through LC callbacks (True for Anthropic + OpenAI). Requires `ANTHROPIC_API_KEY`.
├── examples/langgraph_agent.py   # LangGraph `create_react_agent` demo with toy looping tool; handler wires in via `config={"callbacks": [handler]}`. Shows LG-via-LC-callbacks path. Requires `OPENAI_API_KEY`.
├── docs/announcements.md         # Post-publish outreach drafts — Show HN / LangChain Discord / Reddit r/LocalLLaMA. Consistent positioning across all three (tool-loop + procedural memory wedges, 44ns overhead, OTel-native, honest quality-parity caveats).
├── tests/test_callback.py        # 30+ behavioural tests via fabricated `SimpleNamespace` payloads — no actual LC agent runtime needed. Covers: turn boundary detection (chain/llm/chat_model paths incl. nested), token extraction for OpenAI/Anthropic/usage_metadata/no-usage shapes, scope drift end-to-end, tool lifecycle + error + orphan tool_end, `on_decision` + `raise_on_circuit_break`, regulator passthrough, `on_llm_error` no-op, `chain_error` resets turn flag, nested chain_error leaves flag unchanged, multi-turn lifecycle (chain_start → chain_end → chain_start), multi-LLM-calls-in-one-turn cost accumulation, async handler basic flow / tool-loop / raise / anthropic-style extraction (via `asyncio.run`), `emit_tokens` off/on behavior.
└── README.md                     # Public-facing pip page: quick-starts for LangChain / LangGraph / async, hook → event mapping table, migration patterns (from `recursion_limit` / `tenacity` / Langfuse), behavioural notes, persistence recipe, benchmark citation (S33 — 20-250ns per event, ~2µs decide). Leads with "overhead 6 orders of magnitude below an LLM call" framing.
```

`.github/workflows/publish.yml` extended (Session 38): `langchain-build` + `langchain-publish` jobs build pure-Python sdist+wheel via `hatch build` and `twine upload` to PyPI. Depends on `python-publish` because `noos-langchain` declares `noos>=0.1.0` in its dependency chain; PyPI needs `noos` live for users' `pip install noos-langchain` to resolve.

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

regulator (Path 2 — external regulatory layer, shipped as `noos` 0.2.0 → 0.3.0)
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
├── regulator/tools (ToolStatsAccumulator — per-turn tool-call + tool-result
│   history + consecutive-same-tool loop detector; drives
│   CircuitBreak(RepeatedToolCallLoop) in decide(); 0.3.0 Path A)
├── regulator/state (RegulatorState — persistence envelope; wraps
│   LearnedState + correction_patterns with #[serde(default)] backcompat)
└── cognition/detector (extract_topics + to_topic_set + count_topic_overlap
    + build_topic_cluster — P3 shared utilities; build_topic_cluster also
    consumed by world_model for LearnedState.response_strategies keying)
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
| `thiserror` | Error types | NoosError enum (P5 fail-open with context) |
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
| `docs/theories.md` | 8 computational principles — destination (what Noos should become) | Before designing new features |
| `docs/intervention.md` | Intervention architecture — compensate don't amplify, findings chain SC→Noos | Before designing AI integration or module wiring |
| `docs/brain-map.md` | Implementation status — where we are vs theories + all constants (P3 single source) | When adding modules or checking gaps |
| `README.md` | Crate root — public-facing overview, quick start, Path 2 shipped status, competitor comparison, eval numbers (Session 25) | External readers / crates.io visitors |
| `docs/app-contract.md` | Noos ↔ application semantic contract — signal meanings, body_budget, LearnedState ownership, closed-loop requirements, app-level pitfalls (Path 1 + Path 2 §8 extension) | When embedding Noos in an application or reasoning about signal interpretation |
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
cargo run --example agent_simulation  # Three-way fair comparison: naive vs simple-retry (fair baseline, no Noos) vs allostatic warm-start

# Task-eval suite (synthetic harnesses validating the 6-gap claims — see docs/task-eval-design.md)
cargo run --example task_eval_synthetic        # Tier 1.1 cross-session reward learning (Noos warm 0.900 vs baseline 0.725)
cargo run --example task_eval_conservation     # Tier 1.2 conservation signal calibration
cargo run --example task_eval_multi_signal     # Tier 1.3 signal compounding (+2.73 over smart baseline at budget=12)
cargo run --example task_eval_budget_sweep     # Tier 1.4 Pareto envelope across budget caps 4-20
cargo run --example task_eval_abstention       # Tier 1.5 metacognition (Noos = baseline F1=1.00, infrastructure-only)
cargo run --example task_eval_fatigue          # Tier 1.7 abrupt fatigue (Noos slower than rolling avg, negative)
cargo run --example task_eval_fatigue_gradual  # Tier 1.8 gradual + noisy fatigue (Pareto split: Noos +q, smart -harm)
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
cargo run --example regulator_implicit_correction_demo      # Demo 5 (Session 32) — ProceduralWarning from implicit retries, no explicit UserCorrection needed

# Path 2 real-LLM reliability eval (Tier 2.2, Session 24)
cargo run --example task_eval_real_llm_regulator            # 50-query canned (CI reproducibility guard only: +29.2% cost, +0.85 q/1k — synthetic oracle artifact, NOT real-workload claim)
cargo run --example task_eval_real_llm_regulator -- ollama  # Live via Ollama phi3:mini (add NOOS_JUDGE=anthropic for real grading; S36 N=29 interleaved: Δmean_q=−0.026, Δcost=−5.4%)
cargo run --example task_eval_real_llm_regulator -- anthropic # Live via Anthropic Haiku (add NOOS_JUDGE=anthropic for real grading; S36 N=50: Δmean_q=+0.048, Δcost=+11.6%)

# Offline shadow replay CLI (Session 39, 2026-04-20) — JSONL events → Regulator → JSONL decisions + summary.
# Language-agnostic; works with any pipeline that can emit JSON (D1 export, log file, eval harness output).
# Dual-use: SC dev-time eval integration (cross-repo, see `semantic-computer/eval/run-with-noos-replay.mjs`)
# + future production shadow replay + 6th flagship demo for crates.io users.
cargo run --release --example shadow_replay < events.jsonl                               # Basic
cargo run --release --example shadow_replay -- --cost-cap=10000 < events.jsonl           # With cost cap
cargo run --release --example shadow_replay -- --summary-only < events.jsonl             # Just the _summary line
```

## Recent sessions

- **Session 28 Ship 0.1.3 → 0.3.0** (2026-04-16): crate name rename `nous` → `nous-regulator` (crates.io name collision) → `noos` (final); GitHub repo `Triangle-Technology/nous` → `/noos` with auto-redirect preserved. Rust-type rename `NousError/NousResult/NousTokenizer` → `Noos*` fenced as 0.2.0 breaking. **0.2.1** QA pass: `#[non_exhaustive]` on `LLMEvent`/`Decision`/`CircuitBreakReason` + `#[must_use]` on `Decision`; fixed silent data-loss bug in `CognitiveSession::export_learned` (LC `tick` + `gain_mode` mutated on every turn but never flushed to exported `LearnedState` because `sync_to_learned` was defined but never called — roundtrip test was `0 == 0`, masking loss; test strengthened with `assert!(learned.tick > 0)`). **0.2.2** Path B: `Regulator::corrections_prelude()` + `inject_corrections(&str) -> String` helpers replacing the 15-line ProceduralWarning hand-threading recipe. **0.3.0** Path A: `LLMEvent::ToolCall` + `LLMEvent::ToolResult`, new `CircuitBreakReason::RepeatedToolCallLoop`, new `src/regulator/tools.rs` (`ToolStatsAccumulator`, `TOOL_LOOP_THRESHOLD=5`), 4 observability accessors, fourth flagship demo `examples/regulator_tool_loop_demo.rs`. P10 priority now `CircuitBreak(CostCapReached) > CircuitBreak(QualityDeclineNoRecovery) > CircuitBreak(RepeatedToolCallLoop) > ScopeDriftWarn > ProceduralWarning > Continue`. 442 tests, 0 clippy warnings, 0 rustdoc warnings. Published 0.1.3 / 0.2.0 / 0.2.1 / 0.2.2 / 0.3.0.

- **Session 29 Principle-compliance audit + cleanup** (2026-04-17, doc-only): user-requested principles.md compliance sweep on Session 28 changes. Findings: (1) P7 real violation — CLAUDE.md not updated for any of 0.1.3–0.3.0 arch changes (fixed this session). (2) P1 exemption ambiguity — `src/regulator/*` doesn't cite neuroscience because regulator is LLM-operational framing per `feedback_llm_operational_framing_2026_04_15.md`, but `principles.md` didn't document the exemption (fixed: added §P1 "Regulator exemption (2026-04-17)" subsection). (3) P2 `/// Mutable:` prefix inconsistency across codebase — swept 7 methods missing prefix (`tools.rs::{reset_turn, record_call, record_result}`, `session.rs::inject_gate_feedback`, `mamba.rs::{set_bottleneck, clear_bottleneck}` × 2 variants). (4) P10 `## Gating` section missing from regulator modules — added to all 5 (`scope.rs`, `cost.rs`, `correction.rs`, `token_stats.rs`, `tools.rs`) with Suppresses/Suppressed by/Inactive when triplet citing the priority chain. No code-behaviour changes; doc-only sweep. Per user rule, NO publish — new-version publishes now require an explicit achievement criterion from the user.

- **Session 30 Real-judge eval + Python bindings** (2026-04-17, two deliverables from the post-Session-29 objective audit ranked #1 + #2). **(A) Real-judge eval (rank 1 — "synthetic oracle is the #1 credibility gap")**: added `call_anthropic_judge(task, response) -> Result<f64>` to `examples/regulator_common/mod.rs` (claude-haiku-4-5 default; override via `NOOS_JUDGE_MODEL`; 5-band rubric; robust JSON extraction via first-`{` / last-`}` slice to tolerate LLM preamble and code-fence wrapping). Wired into `examples/task_eval_real_llm_regulator.rs::llm_call` behind `NOOS_JUDGE=anthropic` env var — unset reuses the deterministic canned oracle (backward compatible; canned numbers reproduce bit-for-bit: baseline 16_040 tok / 29.90 q, regulator 11_360 tok / 30.80 q, Δq/1k +0.85, Δcost −29.2%). Set + live mode replaces the oracle per response; per-call grader failure falls back to the oracle so one flaky grade can't invalidate a 50-query run. Module docstring, run-command block, and end-of-run `Notes:` section updated to surface the new flag. `#[allow(dead_code)]` added to the judge function because `regulator_common/mod.rs` is `#[path]`-included by 4 examples, and the 3 non-eval demos don't consume it. **(B) Python bindings (rank 2 — "Rust-only is currently a liability; Python/TS binding expands audience ~10×")**: new out-of-tree crate at `bindings/python/` (NOT in Cargo.toml `include`, stays out of the published `noos` crate). `Cargo.toml`: `pyo3 = { version = "0.23", features = ["extension-module", "abi3-py39"] }` — abi3-py39 produces ONE wheel covering CPython 3.9+ and lets the binding build without a Python interpreter present (via `PYO3_NO_PYTHON=1`). `pyproject.toml`: maturin 1.7+ backend, PyPI name `noos-regulator`, license MIT. `src/lib.rs` (~500 LOC) exposes 5 `#[pyclass]`es: `Regulator` (with `for_user` / `with_cost_cap` builder chaining via `PyRefMut<'py, Self>` + `mem::replace` swap trick / `on_event` / `decide` / `export_json` / `from_json` / 8 accessors), `LLMEvent` (frozen; 8 staticmethod factories matching every Rust variant; opaque `.kind: str` getter), `Decision` (frozen; `.kind: str` + variant-specific getters returning `Option<T>` — None when not applicable; `is_continue` / `is_scope_drift` / `is_circuit_break` / `is_procedural_warning` / `is_low_confidence` predicates), `CircuitBreakReason` (same pattern: `.kind` + per-variant getters for all 4 reason variants including the 0.3.0 `RepeatedToolCallLoop`), `CorrectionPattern` (frozen read-only data class for `Decision.patterns`). `bindings/python/README.md` covers install (`pip install noos-regulator`), quick-start with Python 3.10 `match` / pre-3.10 `if`-chain equivalents, full API summary tables, and `maturin develop --release` build-from-source instructions. `bindings/python/examples/basic.py` has 4 scenarios mirroring the 4 flagship Rust demos. `bindings/python/tests/test_regulator.py` has 15 behavioural tests (all Decision variants, all CircuitBreakReason variants, tool-stats accessors, `export_json`/`from_json` roundtrip incl. malformed-JSON `ValueError`, `inject_corrections` no-op path, explicit kind-exhaustiveness assertion on LLMEvent factories). Verified: `PYO3_NO_PYTHON=1 cargo check` clean, `cargo clippy` zero warnings, Rust main crate still 442 tests + 0 clippy. Design rationales: (i) abi3 over version-locked for distribution simplicity; (ii) flat `.kind`-dispatch over Python-enum-wrapping-Rust-data-variant because PyO3 doesn't natively round-trip data-bearing enums; (iii) JSON-string persistence over RegulatorState pyclass because JSON strings are portable + trivially testable; (iv) `unsendable` on `Regulator` — agent loops are single-threaded, Regulator owns non-Send types via its wrapped `CognitiveSession`. No `cargo publish` / no `maturin publish` — per the 2026-04-17 rule. Session-28 live-eval numbers in main README still stand; user can now re-run with `NOOS_JUDGE=anthropic` to upgrade quality from oracle to real grader before the next publish gate.

- **Session 31 README reframe + adversarial tests + OTel adapter** (2026-04-17, three deliverables from the same post-29 audit, ranked #4 / #5 / #3 respectively, all doc-and-code no publish). **(A) Rank 4 README reframe**: tagline led with "scope drift, cost circuit breaks, procedural correction memory" — burying the 2 strongest (net-new) claims. Rewrote tagline to lead with "tool-call retry loops, learns from user corrections across sessions, halts on cost × quality compound, flags scope drift." Problem section restructured into "Two failure modes cost real money in production" (tool-loop citing $47k LangChain incident, wasted-retries citing 90.8% benchmark) + "Two more corrode trust over time" (procedural amnesia vs Mem0/Letta/Zep content stores, scope drift as catch-all). Competitor matrix reordered: Tool-loop / Pattern-from-corrections / Cost×quality / Scope-drift / Remember-content / Log-turns (strongest wedge → weakest). Added footnote ¹ (LangMem Python does procedural memory generatively; Noos structural differs) + ² (`recursion_limit` counts steps regardless of what repeats; `RepeatedToolCallLoop` fires on the *signature* of the $47k pathology). Demos section reordered to match: tool_loop_demo first, correction_memory second, cost_break third, scope_drift last. Quick-start match arms also reordered so Decision handling mirrors the new priority. **(B) Rank 5 adversarial tests** — Session 30 audit flagged existing test suites as "staged" (all simple linear patterns). Added 6 tool-loop cases (tail run after mixed prefix, diverging args same name, over-threshold count preserves depth, resolved historical loops with short tails don't re-fire, alternating A/B never fires (design boundary), explicit exact-threshold boundary pair) + 5 scope cases documenting known limitations as PASSING tests (rank-11 alphabetical truncation FP, synonym surface-form FP, verbose on-topic FP, case-insensitive regression guard, non-English max-drift regression guard). Both modules' adversarial tests pass — current implementation is actually more robust than the audit suggested on tool-loop; scope tests explicitly pin the limitations as documentation. Test count: 442 → **467** (+25: 13 OTel + 6 tool-loop + 5 scope + 1 doctest). **(C) Rank 3 OTel GenAI adapter** — new module `src/regulator/otel.rs` with `events_from_span(&serde_json::Value) -> Vec<LLMEvent>` (no new deps, `serde_json` already a main-crate dep). Maps the stable subset of `gen_ai.*` semantic conventions — `gen_ai.user.message` → TurnStart, `gen_ai.assistant.message` → TurnComplete, `gen_ai.usage.input_tokens` + `gen_ai.usage.output_tokens` + span duration → Cost, `gen_ai.tool.message` with `gen_ai.tool.name` → ToolCall (and + `gen_ai.tool.duration_ms` → paired ToolResult). SDK-idiomatic dict-form JSON only; OTLP protobuf-JSON callers convert upstream. Canonical event ordering regardless of input order. Partial spans degrade gracefully (cost without messages, messages without cost). Error-handling: clock-skew negative-duration guards don't panic; missing optional fields default to 0/None per the LLMEvent forgiving-contract. 13 self-tests in the module (covers every branch). Python binding gained `LLMEvent.from_otel_span_json(span_json_str) -> list[LLMEvent]` (5 new Python tests). README.md gained an "OpenTelemetry GenAI ingestion" section with the 8-line integration example. **Verification**: 467 tests pass (429 unit + 13 OTel + 11 adversarial + 8 session + 4 memory + 2 doctests); 0 clippy warnings main crate and Python binding; 0 rustdoc warnings after 3 redundant-link-target cleanups; canned eval numbers still bit-for-bit reproducible. No publish.

- **Session 32 Implicit correction detector + 5th flagship demo** (2026-04-17, adoption-unlock for the strongest wedge). Closes the #1 adoption risk the Session 29 research agent flagged: chat UIs rarely emit explicit `LLMEvent::UserCorrection`, so the `ProceduralWarning` path and procedural-memory wedge had no way to activate without app-level instrumentation most teams don't add. **Mechanism**: new `Regulator::with_implicit_correction_window(Duration)` builder gates a two-part detector — (1) temporal proximity (TurnStart arrives within window of last TurnComplete) AND (2) topic continuity (new turn maps to same `current_topic_cluster`). Both required. When both hold, the regulator synthesises a correction record into `CorrectionStore` using the new `user_message` as the correction text. Three new `Regulator` fields: `implicit_correction_window: Option<Duration>` (opt-in, default None = no behaviour change), `last_turn_complete_at: Option<std::time::Instant>` (stamped on TurnComplete only when feature is on; `Instant` is not Serialize so NOT persisted by export/import — ephemeral signal correctly doesn't span process restarts), `implicit_corrections_count: usize` (per-process observability counter, also not persisted). New accessor `Regulator::implicit_corrections_count()`. Python binding: `Regulator.with_implicit_correction_window_secs(f64)` (rejects non-finite / non-positive with `ValueError`) + `Regulator.implicit_corrections_count()`. **Tests**: 8 new unit tests in `mod.rs` using real `Instant` + `std::thread::sleep` (50-500ms windows): default-off guard, fast-retry fires, window-expiry fails closed, different-cluster fails closed, first-turn fails closed (no prior complete), empty-cluster fails closed (build_topic_cluster returned empty), 3-retry pattern accumulation → ProceduralWarning pre-generation with 3 example_corrections, import-roundtrip resets counter. Test construction required careful message choice — top-2 alphabetical meaningful words must match across retries for cluster identity (first draft had "Fix fetch_user async bug" which sorts to "async+bug" not "async+fetch_user"; corrected in same session). **5th flagship demo**: `examples/regulator_implicit_correction_demo.rs` runs three scenarios (3 fast retries → pattern fires; retry outside 500ms window → counter stays 0; retry on different cluster → counter stays 0). Canned-only — timing + cluster signal, no LLM needed. Output explicitly shows implicit_corrections_count incrementing 1 → 2 → 3 and ProceduralWarning with 3 example_corrections on turn 4. README.md updated to list 5 flagship demos (was 4), Python tests suite grew to 23 (3 new). **Verification**: **475 tests pass** (461 unit + 8 session + 4 memory + 2 doctests), +8 over Session 31 baseline; 0 clippy warnings main crate and Python binding; 0 rustdoc warnings; canned eval reproduces. No publish.

- **Session 33 Criterion benchmarks + metrics_snapshot observability** (2026-04-17, two deliverables addressing production-credibility gaps that aren't on the original Session 29 audit but are equally blocking for adoption). **(A) Criterion benchmarks**: new `benches/regulator.rs` (NOT in Cargo.toml `include` — internal measurement only, stays out of the published crate). Adds `criterion = { version = "0.5", features = ["html_reports"] }` as a dev-dependency + `[[bench]] name = "regulator" harness = false` entry. Nine benchmarks: per-event dispatch for `Token` / `TurnStart` / `TurnComplete` / `Cost` / `ToolCall`, `decide()` on Continue + ScopeDrift paths, `export_then_import` roundtrip, realistic full-turn (1×TurnStart + 100×Token + TurnComplete + Cost + decide). **Measured numbers on Windows/Ryzen** (release build, 3s measurement windows): Cost=20ns, Token=44ns, ToolCall=249ns, decide/Continue=2.0µs, decide/ScopeDrift=3.7µs, TurnComplete=3.9µs, TurnStart=22µs, export/import=1.2µs, full realistic turn (100 tokens)=30µs. README gained a Performance section quoting these numbers with the framing "an LLM call at 200 tokens/sec runs for ~500ms per turn — Noos overhead is six orders of magnitude smaller." The numbers remove the "what's the overhead?" adoption objection with data. Three `#[must_use]` warnings on `black_box(r.decide())` fixed with `let _ =` prefix. **(B) `Regulator::metrics_snapshot() -> HashMap<String, f64>`**: one-call observability dump for Prometheus / Datadog / StatsD pipelines. 8 stable `noos.`-prefixed keys (confidence, logprob_coverage, total_tokens_out, cost_cap_tokens, tool_total_calls, tool_total_duration_ms, tool_failure_count, implicit_corrections_count). Does NOT call `decide()` — that's an explicit call point. Python binding exposes the same method returning `dict[str, float]`. 2 new unit tests (key stability contract, state-change tracking) + 2 Python tests. README gained an Observability section above Performance. **Verification**: **477 tests pass** (463 unit + 8 + 4 + 2 doctests; +2 from Session 32's 475); 0 clippy warnings main crate and Python binding; 0 rustdoc warnings. No publish.

- **Session 34 Node.js / TypeScript bindings** (2026-04-17, closes the second major audience gap after Python). TS/Node is ~25% of LLM agent prod market (LangChain.js, Vercel AI SDK, Mastra, BAML TS), a tier of users Python/Rust doesn't reach. New out-of-tree crate at `bindings/node/` (NOT in `include` whitelist; main crate unchanged). **Cargo.toml**: `napi = { version = "3", default-features = false, features = ["napi4"] }` + `napi-derive = "3"`. napi 3.x chosen specifically because it handles Windows libnode delay-loading at runtime — napi 2.x's `napi-build` requires `libnode.dll` at link time which standard Windows Node installs don't ship. No `build.rs` (napi 3 doesn't need it). **package.json**: `@napi-rs/cli` as devDep, 5 prebuilt target triples (windows-msvc / macOS x64 + arm64 / linux-gnu x64 + aarch64). npm package name `noos-regulator`. **`src/lib.rs` (~500 LOC)**: parallel port of the PyO3 binding via `#[napi]` attribute macros which auto-generate `.d.ts`. 5 classes mirrored 1:1 — `Regulator` / `LLMEvent` / `Decision` / `CircuitBreakReason` / `CorrectionPattern`. Same `.kind: string` + `Option<T> → T | null` variant pattern. Same JSON-string persistence via `exportJson` / `fromJson`. Full parity including `withImplicitCorrectionWindowSecs` (Session 32), `metricsSnapshot` (Session 33), `fromOtelSpanJson` (Session 31). Three implementation adaptations: (i) no fluent builder chaining (napi-rs can't return `&mut Self` across JS boundary) — `withCostCap` returns void, chain with separate statements; (ii) `u64` (`duration_ms` on ToolResult) exposed as `BigInt` in TS; (iii) Decision's `fragment_spans` reshaped from `(usize, usize)` tuples to `number[][]` pairs because napi-rs tuple support is weak. **Examples + tests**: `examples/basic.mjs` runs 4 scenarios mirroring Rust demos; `__test__/regulator.test.mjs` has 15 behavioural tests using Node's built-in `--test` runner (zero framework deps). **Windows-GNU build status blocker**: this machine's Rust toolchain is `x86_64-pc-windows-gnu`, for which `napi-build 2.3.1` (transitively pulled in by napi) panics "libnode.dll not found in any search path" even after `npx node-gyp install` populated `node.lib`. napi-build's GNU path requires a DLL Node-Windows doesn't ship. Attempted workarounds: switching to napi 3 (still pulls napi-build 2.3.1 transitively), setting `CARGO_CFG_NAPI_RS_CLI_VERSION` env var (no effect), removing local `build.rs` (napi's own build.rs still fails). **Decision**: ship the skeleton as documented 0.1.0-pre since (a) code is syntactically parallel to the known-working PyO3 port, (b) Linux/macOS CI or Windows-MSVC toolchain resolves immediately, (c) blocking on a local-only Windows-GNU tooling bug would waste time. `bindings/node/README.md` documents the situation upfront under "Status" and "Build from source" sections. Main crate unchanged: **463 lib tests still pass**. No publish. **Session 35 entry point**: anyone with Linux / macOS / Windows-MSVC can run `cd bindings/node && npm install && npm run build && npm test` to validate end-to-end. Alternatively wire GitHub Actions (`ubuntu-latest` runner) to build + test + npm publish.

- **Session 35 GitHub Actions CI + migration guide** (2026-04-17, infrastructure polish that converts the work of Sessions 30-34 from "locally tested" to "actually validated on every push"). **(A) CI workflow** at `.github/workflows/ci.yml` with 5 jobs on stable Rust: (i) `rust-checks` on ubuntu-latest — cargo test + clippy (with `RUSTFLAGS: -D warnings`) + rustdoc + `cargo fmt --check` (non-blocking, informational) + **canned eval reproducibility guard** that greps for the exact `cost_saved = +4680  (29.2%)` + `total_quality delta = +0.90` strings from `cargo run --release --example task_eval_real_llm_regulator` (any drift in Tier 2.2 numbers fails CI, forcing explicit acknowledgement); (ii) `rust-platforms` matrix — build + test on macos-latest + windows-latest to catch platform-specific regressions (Windows-latest on GitHub is MSVC, so `napi-build` works — different from the local Windows-GNU setup); (iii) `bench-smoke` — `cargo bench --bench regulator --no-run` to catch bench code breakage without burning minutes on sampling; (iv) **`python-binding`** — Python 3.12 + `maturin develop --release` + `pytest -v` against all 25 Python tests (first-class first validation path for the PyO3 bindings since local Python isn't installed); (v) **`node-binding`** — Node 20 + `npm install` + `npm run build` + `npm test` + `npm run example` (FIRST EVER end-to-end validation of the napi-rs Node binding — resolves the Windows-GNU libnode.dll blocker from Session 34 by running on ubuntu-latest where napi-build works). Uses `Swatinem/rust-cache@v2` with per-platform shared-keys to minimize re-compile time; `dtolnay/rust-toolchain@stable` for toolchain setup. Triggers: push to main + PR to main + `workflow_dispatch`. Also added `.github/dependabot.yml` with 5 ecosystems (cargo × 3, npm, github-actions) weekly, 3-PR cap per ecosystem, labeled appropriately. **(B) Migration guide** at `docs/migrating.md` (added to Cargo.toml `include` → ships to crates.io). Four concrete "replace your code with this" recipes with before-and-after code samples: (1) **LangChain / CrewAI / AutoGen `recursion_limit` → `RepeatedToolCallLoop`** (cites $47k LangChain incident), (2) **Mem0 / Letta / Zep content memory → Noos procedural pattern** (threshold-gated, structural, no embeddings), (3) **tenacity / backoff / p-retry → Noos cost×quality compound halt** (cites 90.8% wasted-retries benchmark), (4) **Helicone / Langfuse / Arize observability → pair Langfuse with Noos pre-delivery** (complement, not replace; includes `metrics_snapshot` piping sample). All four use Python syntax primarily because Python is the largest target audience; #4 uses TypeScript to mirror the binding work of S34. **(C) README polish**: CI + crates.io + docs.rs + license badges added above the tagline; migration-guide pointer added below the quick-start code sample. **Verification**: 477 tests unchanged; `cargo publish --dry-run` clean, **101 files** packaged (up from 95 — `docs/migrating.md` now shipping), **367 KiB** compressed (up from 327 KiB). No publish.

- **Session 35b Principle-compliance self-audit** (2026-04-17, same-day addendum triggered by user request "review all code again from this session. Review code if it against principles"). Audited every file changed in Sessions 30-35 against all 10 principles + CR1-CR5. **Findings**: 5 real violations (4 + 1 minor), 1 initially-flagged issue retracted after verification. **Fixes applied in same session**: (1) **P1 exemption preamble** — `src/regulator/otel.rs` was missing the `**Scope note (P1 / P9b)**` doc block that every other regulator submodule has. Added matching preamble explaining the LLM-operational scope (JSON lookups against OTel GenAI vocab, no Noos-side text parsing). (2) **P2 `/// Mutable:` prefix** — 3 methods on `bindings/python/src/lib.rs` (`on_event`, `with_cost_cap`, `with_implicit_correction_window_secs`) and the same 3 on `bindings/node/src/lib.rs` took `&mut self`/`PyRefMut` without the required prefix. Rewrote all 6 docstrings to start with `/// Mutable:` + justify why mutation is necessary (wrapped session state, mem::replace swap for builder shape). (3) **P3 duplication** — `call_anthropic` and `call_anthropic_judge` in `examples/regulator_common/mod.rs` both duplicated the Anthropic HTTP plumbing (key read, endpoint URL, header set, JSON parse boilerplate). Extracted `anthropic_messages_post(model, max_tokens, user_content) -> Result<Value>` helper; both callers now share it and do only their own response post-processing. (4) **P7 minor** — demo constants `WINDOW_MS` / `QUICK_WAIT_MS` / `PAST_WINDOW_WAIT_MS` in `examples/regulator_implicit_correction_demo.rs` lacked `///` WHY comments. Added one-sentence rationale to each (window length chosen to keep demo wallclock <1s while surviving CI jitter). **Retracted finding**: initially flagged `duration_ms.get_u64().1` in `bindings/node/src/lib.rs` as a possible bug (suspected `.1` returns `lossless` bool, not the u64). After reading `napi-3.8.5/src/bindgen_runtime/js_values/bigint.rs:95` directly, confirmed `get_u64() -> (bool, u64, bool)` where `.1` IS the u64 value. Code is correct; my initial claim was wrong, retracted. **Pre-existing violations NOT from this session**: `src/regulator/tools.rs` and `src/regulator/state.rs` both lack the P1 preamble (S28 and S20 work respectively). S29 audit missed these. Flagged as cleanup candidates for a future session, NOT fixed here since outside S30-35 scope. **Verification after fixes**: 477 tests pass (unchanged), 0 code clippy warnings (main + Python), 0 rustdoc warnings, `cargo publish --dry-run` still clean. **Principle status at end of S30-35**: clean for every file this session touched.

- **Session 37 Full-codebase principle-compliance sweep + NaN-safety bugfix + P1 magic-number extraction** (2026-04-19, multi-round user instruction starting "Kiểm tra toàn bộ Noos... review code if it against principles", continuing "tiếp tục tìm và cải thiện code, kiểm tra principles", then "có lỗi nào cần fix nữa không"). 7 commits across the session: [`092fd68`](https://github.com/Triangle-Technology/noos/commit/092fd68) (doc cleanup) + [`e609314`](https://github.com/Triangle-Technology/noos/commit/e609314) (NaN-safety bugfix) + [`11d07e7`](https://github.com/Triangle-Technology/noos/commit/11d07e7) (gate-feedback + allostatic constants) + [`5632205`](https://github.com/Triangle-Technology/noos/commit/5632205) (LC nudge + sustained arousal constants) + [`d3c0ab9`](https://github.com/Triangle-Technology/noos/commit/d3c0ab9) (CR4 NaN contract tests). **Audit scope**: all 10 principles + CR1-CR5 against the full codebase, not just one session's delta (S29 / S35b were scoped to recent changes). **Round 1 (doc cleanup, 092fd68)**: P1 preamble additions to `src/regulator/tools.rs` + `src/regulator/state.rs` (previously flagged by S35b as pre-existing); P2 `/// Mutable:` prefix reorder on `src/session.rs::inject_gate_feedback`; 6 `## Gating (P10)` sections added to cognition modules (`belief_state`, `dynamics`, `emotional`, `intervention`, `resource_allocator`, `world_model`). Plus 5 rustdoc warnings introduced during the Gating writes — fixed same round (broken intra-doc link to nonexistent `SamplingOverride::none`; 3 redundant explicit `(crate::path::Type)` link targets; 1 redundant target on `CorrectionPattern`). **Round 2 (e609314, LATENT BUG FOUND)**: audit for NaN-safety uncovered that `math::clamp` (used throughout `cognition/` + `memory/`) is NaN-safe via `max(min).min(max)` — absorbs NaN to min — whereas `f64::clamp` (method form, used in 7 sites in `regulator/` + `session.rs`) propagates NaN. One real exposure: `session.rs::track_cost` accepts user-supplied `cost: f64` and the prior `cost.clamp(0.0, 1.0)` would pass NaN through to `body_budget` (CR4 invariant, must stay in `[0, 1]`). Unified all 7 sites on `math::clamp` (P3 single-source + P5 fail-open). Documented the NaN-safety contract on `math::clamp` itself, added 2 tests locking NaN + infinity behavior. README stale "442 tests passing" → reproducible verification command (P7 no-hardcoded-counts). **Round 3 (11d07e7)**: P1 magic-number extraction in `src/session.rs::inject_gate_feedback` (5 bare literals: `GATE_ACTIVE_MIN_ALPHA=0.1`, `GATE_AROUSAL_PRIOR_WEIGHT=0.8`, `GATE_PASSIVE_DECAY=0.95`, `GATE_SURPRISE_MIN_DELTA=0.05`, `GATE_PE_PRIOR_WEIGHT=0.7`) and `src/cognition/world_model.rs` allostatic core (5 literals: `PE_DEPLETION_RATE=0.02`, `AROUSAL_DEPLETION_RATE=0.01`, `NATURAL_REPLENISHMENT_RATE=0.005`, `POSITIVE_RPE_REPLENISHMENT_RATE=0.05`, `NEGATIVE_RPE_DEPLETION_RATE=0.02`) — each with derivation comment (Barrett 2017 / Sterling 2012 / Schultz 1997 where applicable, EMA-window math where the paper doesn't fix a specific rate). Also fixed 2 stale `PrefrontalState` doc refs (module was split into `LocusCoeruleus` in the 2026-04-10 non-cortical audit; `src/session.rs:116` + `src/types/intervention.rs:64` still referenced the old name). **Round 4 (5632205)**: further P1 magic-number extraction in `src/cognition/convergence.rs` (`LC_RPE_NUDGE_WEIGHT=0.15`, Aston-Jones & Cohen 2005 §Task utility) and `src/cognition/signals.rs` (`SUSTAINED_AROUSAL_CONSERVATION_WEIGHT=0.2`, `BUDGET_FACTOR_MIN_DIVISOR=0.01` divide-by-zero guard). **Round 5 (d3c0ab9)**: P6 contract tests `track_cost_nan_does_not_corrupt_body_budget` + `track_cost_infinity_does_not_corrupt_body_budget` — both would have failed before round 2's fix, lock the CR4 body_budget `[0, 1]` invariant under NaN + ±inf inputs. **Round 6 (137bebd, 2nd NaN gap)**: sibling finding — `CognitiveSession::process_response(_, quality)` took user-supplied `quality: f64` and passed straight to `consolidate` without validation. NaN quality poisoned `response_rpe = quality - last_prediction`, which then propagated to `CognitiveSignals.rpe` (public field applications read). Fix mirrors the existing `CostAccumulator::record_quality` pattern — drop non-finite at boundary, silent fail-open. Finite out-of-range values still pass through (clamping happens downstream in `consolidate` body-budget math). 2 contract tests added: `process_response_nan_quality_dropped_silently` + `process_response_infinity_quality_dropped_silently` — both would fail without the guard. **Audit of all user-supplied f64 entry points complete**: (1) `Regulator::on_event(LLMEvent::Token.logprob)` guards non-finite in `on_token`; (2) `Regulator::on_event(LLMEvent::QualityFeedback.quality)` guards via `record_quality`; (3) `CognitiveSession::track_cost(cost)` uses NaN-safe `math::clamp` (fixed this session); (4) `CognitiveSession::process_response(_, quality)` drops non-finite (fixed this session); (5) `CognitiveSession::inject_gate_feedback(gate_alpha, gate_delta_gain)` — documented contract `[0,1]` + `[0.5,2.0]`, math safe for in-contract inputs, no defensive guard added (not on the typical app path). **Clean (no fixes needed)**: P3 (no cross-module duplication — 5 `make_atom` test helpers have distinct signatures per test file), P4 (types/ + math/ have no crate imports, cognition respects layer hierarchy), P5 (173 `unwrap()` calls all inside `#[cfg(test)]` blocks; production code clean; 2 documented candle-mmap `unsafe` exceptions in `inference/mamba.rs`), P6 (every cognition + regulator module has `#[cfg(test)]`; `types/*.rs` Layer 0 shapes have no logic to test), binding parity (Python + Node have identical 20-method Regulator surface + 8 LLMEvent factories + matched Decision / CircuitBreakReason / CorrectionPattern — only `from_otel_span_json` differs in shape, documented S36 fix). **Known-debt items deliberately NOT touched**: P9b lexicon regex in `emotional.rs` (NEGATIVE_HIGH/MOD + POSITIVE_HIGH/MOD English) + `detector.rs` (SEQUENCE_MARKERS English) remain as Path C scoped interim per `memory/project_lexicon_removal_analysis_2026_04_14.md`; `cognition/convergence.rs:34` TODO (explicit per-connection priority rules) + `cognition/resource_allocator.rs:392` FIXME (pressure-semantic redesign) both carry documented context and aren't closeable inside this sweep. **Verification**: **469 tests pass** (+6 from 463 baseline: 2 math::clamp NaN/infinity contract + 2 track_cost NaN/infinity contract + 2 process_response NaN/infinity contract), 0 clippy code warnings, 0 rustdoc warnings. No publish (per the 2026-04-18 refined rule — incremental doc/polish work stays local until the next real feature landing; the two NaN fixes are arguably a "breakthrough" but user has not explicitly requested a publish). **Repo HEAD**: `137bebd` at session close; 7 commits ahead of `origin/main`.

- **Session 36 First fully-green CI + TLS silent-fallback bug + real-judge reveals oracle-scoring artifact** (2026-04-17, user instruction "push CI, run judge"). **(A) CI: 5 fix commits to green** — [`12518be`](https://github.com/Triangle-Technology/noos/commit/12518be) ureq `"tls"` feature (every `call_anthropic` Err-branch silently fell back to canned; Ollama-HTTP unaffected) + `@napi-rs/cli` 2.x → 3.x (napi 3.x derive macros panic without 3.x CLI env vars) + Python CI `maturin develop` venv (maturin rejects systemwide install). [`fb3efc7`](https://github.com/Triangle-Technology/noos/commit/fb3efc7) napi feature `napi4` → `napi6` (BigInt gated by `#[cfg(feature = "napi6")]` at `bindgen_runtime/js_values.rs:43`) + strip `(factory)` from `from_otel_span_json` (factory requires Self-return). [`eee27ba`](https://github.com/Triangle-Technology/noos/commit/eee27ba) move `from_otel_span_json` out of `impl LLMEvent` into freestanding `llm_events_from_otel_span_json` (mixed `#[napi(factory)]` + plain `#[napi]` static method in one impl block silently corrupts class registration — build passes, class misses all static methods). [`b0e665a`](https://github.com/Triangle-Technology/noos/commit/b0e665a) `#[napi(js_name = "LLMEvent")]` on the struct (napi-rs 3.x parser/mod.rs:1252 runs struct names through `Case::Pascal`, converting `LLMEvent` → `LlmEvent`; tests imported `LLMEvent` → undefined → every `LLMEvent.turnStart` call threw TypeError). [`b4bf936`](https://github.com/Triangle-Technology/noos/commit/b4bf936) `implicitCorrectionExample` async keyword (await inside non-async fn). **All 6 jobs green** by run 24557893378: Rust × 3 platforms + Python binding + Node binding + bench-smoke. First time in project history. **(B) TLS silent-fallback bug — scope clarification**: Bug latent since S21 (ureq added with `default-features = false, features = ["json"]`, no TLS backend). S27 Phase 2 "live demos" and S28 "live eval" ran against **Ollama** (HTTP, no TLS needed) — those numbers are genuine. S36 was the first session to attempt `-- anthropic` mode; first judge-eval run silently canned on every 100+ "API call" (Err("Unknown Scheme"); `Err(_)` catch swaps canned response + tokens + `is_live=false`; giveaway was 7.5s wallclock). Post-fix run is the first real Anthropic-live data point. Ollama-live cost numbers stand; Ollama-live *quality* still uses the synthetic oracle unless `NOOS_JUDGE=anthropic` is set. Worth adding `eprintln!` to fallback arms for future runs. **(C) Real-judge eval first valid data point** — 50 queries × Anthropic generator + Haiku judge (~200 calls, 492.5s): **regulator arm costs MORE** than baseline (cost_saved = −1057, −11.6%), Δq/1k = −0.09 (noise), CircuitBreak fires **0 times** (canned: 9). The canned +29.2% / +0.85 q/1k delta is a scoring artifact — `Cluster::canned_quality(retry)` was designed to decline for Ambiguous cluster (triggering QualityDeclineNoRecovery); real Haiku scores responses on content, not retry index, and doesn't show systematic decline. **Ollama-live's +80.5% / +433% from S28 also used synthetic oracle for scoring**; cost numbers genuine but quality numbers synthetic. **Only judge-mode tests real response quality**. Eval classifier on real run: "≈ Regulator matches baseline on efficiency — infrastructure value (halt / warn / pattern surfaces) without measurable loss." **Verification**: 477 tests pass (unchanged), 0 clippy warnings, CI 6/6 green. Repo HEAD `b4bf936`. No publish. **Pending user call**: whether to rewrite README headline numbers from canned to real. Full memo: `memory/project_session_36_ci_green_tls_oracle_findings_2026_04_17.md`.

- **Session 38 LangChain Python adapter — first framework integration** (2026-04-19, user instruction "dùng các kỹ thuật tư duy phù hợp, kết hợp với research tìm hướng đi tiếp theo cho noos" → recommendation "Direction A adoption sprint + B-lite outreach" → "làm như recommendation"). **Strategic context**: Session 36's real-judge eval showed regulator efficiency matches baseline within noise on real LLMs (|Δmean_q| ≤ 0.05 across 3 valid data points: canned oracle 50q / Anthropic-Haiku sequential 50q / phi3-Haiku interleaved N=29). Zero real users. Zero case studies. Real-judge parity isn't a death sentence — it means Noos is an infrastructure layer (44ns/event overhead confirmed in S33 bench) not a magic efficiency button — but **distribution is the blocker**: bindings built + CI-green since S36 but not on PyPI/npm, no framework integration. First-principles + red-team + JTBD analysis pointed to LangChain Python adapter (~50× audience multiplier from "Rust LLM agents" niche → "any OTel-emitting agent" universe) as the highest-leverage deliverable that doesn't require user outreach. **Scope**: pure-Python pip package `noos-langchain` at `bindings/python-langchain/` (NOT in Cargo.toml include, main crate unchanged). **Package structure**: `pyproject.toml` (hatchling backend, PyPI `noos-langchain`, deps `noos>=0.1.0` + `langchain-core>=0.3.0`, Python ≥3.9), `LICENSE` (MIT copy), `README.md` (~6 KB — quick-start, hook→event table, behavioural notes, persistence recipe), `src/noos_langchain/__init__.py` (3 exports), `src/noos_langchain/callback.py` (~250 LOC main handler + `CircuitBreakError`), `src/noos_langchain/_compat.py` (~170 LOC defensive payload extractors), `examples/basic_smoke.py` (4 scenarios, no LLM/network), `examples/openai_tools_agent.py` (full LC agent with toy looping tools, requires OPENAI_API_KEY), `tests/test_callback.py` (19 behavioural tests, fabricated payloads via SimpleNamespace — no LC runtime needed). **`NoosCallbackHandler` design**: subclasses `langchain_core.callbacks.base.BaseCallbackHandler`; sets `raise_error=True` + `run_inline=True` at class level so `CircuitBreakError` propagates through LC manager instead of being silently logged (LC defaults both False) and stays on the same thread for stable UUID-based tool tracking in async chains. Maps LC hooks → Noos events: `on_chain_start`(root only via `parent_run_id=None`) / `on_chat_model_start` / `on_llm_start` → `LLMEvent.turn_start`; `on_llm_new_token` (opt-in `emit_tokens=True`) → `LLMEvent.token`; `on_llm_end` → `LLMEvent.turn_complete` + `LLMEvent.cost`; `on_tool_start` → `LLMEvent.tool_call`; `on_tool_end` / `on_tool_error` → `LLMEvent.tool_result` with `duration_ms` computed from `time.monotonic_ns()` deltas tracked per `run_id`. After every decision-changing event, `_update_decision` refreshes `handler.last_decision`, fires user-supplied `on_decision(decision)` callback, and raises `CircuitBreakError` if `raise_on_circuit_break=True`. **Three consumption modes** documented in README: (1) poll `handler.last_decision` after `invoke()`, (2) subscribe via `on_decision=callback` to react mid-run, (3) `raise_on_circuit_break=True` to abort immediately. **`_compat.py` payload extractors**: `extract_user_message` tries `inputs["input"|"question"|"query"|"prompt"]` → `messages[-1].content` → `json.dumps` fallback; `extract_token_usage` tries 4 paths (OpenAI `llm_output["token_usage"]["prompt_tokens"/"completion_tokens"]` → Anthropic `llm_output["usage"]["input_tokens"/"output_tokens"]` → modern LC `message.usage_metadata["input_tokens"/"output_tokens"]` → `generation_info["usage"/"token_usage"]`) with `(0, 0)` fail-open; `extract_response_text` tries `generations[0][0].text` → `message.content` → `str(response)`; `tool_name_from_serialized` tries `serialized["name"]` → `serialized["id"][-1]` → `"unknown"` fallback. None of the extractors raise. **Turn boundary semantics**: one root chain = one Noos turn; nested chains + multiple LLM calls in an `AgentExecutor` run accumulate into same turn (costs sum, final `turn_complete` wins scope-drift scoring). Documented trade-offs in README: not thread-safe (Regulator `unsendable`), tool-loop detection uses tool name only (5 consecutive calls fires `RepeatedToolCallLoop` regardless of args), `emit_tokens` is off by default (LC doesn't surface logprobs through callbacks → regulator uses structural fallback). **CI job added** at `.github/workflows/ci.yml`: `python-langchain-binding` on ubuntu-latest — builds `noos` from `bindings/python` via `maturin develop --release` into a shared venv at `$GITHUB_WORKSPACE/.venv`, installs `langchain-core` + `pytest` via `pip`, installs `noos-langchain` editable via `pip install -e . --no-deps` (workaround: `noos` isn't on PyPI yet so regular `pip install -e .` with dep-resolution would fail), runs `pytest tests/ -v`, runs `python examples/basic_smoke.py` as smoke. **Verification**: **469 Rust tests pass** (unchanged — no Rust code touched); `cargo check --lib` clean. Python validation deferred to CI (no Python locally). Main crate untouched, publish rule not triggered. **Pending user actions (all Direction A completion)**: (a) set `PYPI_API_TOKEN` + `NPM_TOKEN` secrets + `git tag v0.4.0 && git push origin v0.4.0` to publish `noos` / `noos-regulator` / `@triangle-technology/noos` via `.github/workflows/publish.yml`; (b) publish `noos-langchain` to PyPI (workflow extension needed once noos itself is on PyPI); (c) B-lite outreach — HN / r/LocalLLaMA / LangChain Discord post leading with OTel-native positioning + 44ns overhead + tool-loop halt demo. **Why LangChain first** (over LangGraph-TS or CrewAI): largest audience (~60% of the agent framework market per 2026 surveys), official callback API stable since 0.2+, LangGraph uses the same callback surface so this adapter works there transparently. CrewAI has coarser callbacks (no per-tool hook in 0.x); would need its own adapter later. **README headline numbers pivot (still pending from S36)**: not done in this session; adoption track doesn't require it but is a sequential item before public outreach since README currently headlines the canned synthetic-oracle +29.2% cost / +0.85 q/1k delta which S36 proved was a `Cluster::canned_quality(retry)` artifact.

## Session-End Checklist

Before ending any session that modified code or docs:
1. `cargo test` + `cargo clippy` — must pass, 0 warnings
2. Update this file (`CLAUDE.md`) if files added/removed or architecture changed
3. Update `docs/brain-map.md` if constants or modules changed
4. Update memory (`project_nous_status.md` or the latest session memo) with what was done and what's next
5. **Do NOT `cargo publish` / `maturin publish` / `npm publish` unless the session cleared a genuine breakthrough** (rule refined 2026-04-18, superseding the 2026-04-17 "explicit achievement gate" rule). Incremental doc/polish work stays local until the next real feature landing. Doc-only patch releases (0.4.1 style) are specifically excluded from this — they ship only when paired with a feature, not as their own trigger.
