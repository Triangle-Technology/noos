//! Tier 2.2 — Regulator vs naive-retry baseline on a mixed 50-query workload.
//!
//! Session 24 extends the Session 21-23 demos into an eval: quantify the
//! Regulator's per-query value on a stream that deliberately mixes
//! clusters where each of the three flagship Decision variants should
//! fire (scope drift, cost circuit break, procedural warning).
//!
//! ## Two arms
//!
//! 1. **Baseline (no regulator)** — naive agent. Per query: try up to
//!    [`NAIVE_RETRY_CAP`] times; if quality < [`QUALITY_RETRY_THRESHOLD`],
//!    retry; count all attempts' cost. No halt predicate, no scope
//!    tracking, no correction memory. This is the "generic backoff
//!    loop" that Portkey / tenacity / litellm ship today.
//!
//! 2. **Regulator** — same retry loop, but gated by [`Regulator::decide`].
//!    [`Decision::CircuitBreak`] halts the current query's retry loop
//!    (counted as `queries_circuit_broken`; the query itself still
//!    counts as served with whatever its last attempt produced).
//!    [`Decision::ProceduralWarning`] is logged pre-generation;
//!    [`Decision::ScopeDriftWarn`] marks the delivered turn as drifted.
//!    A cost cap applies **per-query**, reset between queries — see
//!    "Per-query regulator lifetime" below.
//!
//! ## Per-query regulator lifetime
//!
//! A single `Regulator` accumulates cost + quality history across every
//! event it receives. For a 50-query stream each query is a SEPARATE
//! task, so the eval resets per-query state (cost / scope / token
//! stats / quality window) between queries via
//! [`Regulator::export`] → [`Regulator::import`] round-trips. This
//! preserves the persistent state — `LearnedState` strategy EMA and
//! `CorrectionPattern` memory — while clearing cost-break / scope-drift
//! counters so they reflect only the current task.
//!
//! The alternative ("one Regulator across all 50 queries") trips
//! `QualityDeclineNoRecovery` on the first Ambiguous query and halts
//! every subsequent query regardless of cluster. That's the right
//! behaviour for a SINGLE task but the wrong frame for this eval.
//!
//! ## Primary metric: quality per unit cost
//!
//! Both arms serve all 50 queries. The regulator may halt retry loops
//! early via `CircuitBreak` when cost + quality compound triggers —
//! that's the cost-saving path. `quality / cost` captures efficiency.
//!
//! | Metric | What it measures |
//! |--------|------------------|
//! | `total_cost` | Sum of `tokens_out` across every LLM call (incl. retries) |
//! | `total_quality` | Sum of quality scores across all 50 served queries |
//! | `quality_per_1k_tokens` | `total_quality / (total_cost / 1000)` — efficiency |
//! | `circuit_broken` | Regulator-only: queries where `CircuitBreak` cut the retry loop short |
//! | `scope_drift_flags` | Regulator-only: turns where `ScopeDriftWarn` fired on delivery |
//! | `procedural_warnings` | Regulator-only: `ProceduralWarning` fires pre-generation |
//!
//! ### Semantic notes (read before quoting numbers)
//!
//! 1. **`total_quality` sums FINAL-retry quality, not best-of-N**. If an
//!    app picks the best attempt across retries, both arms would record
//!    the same quality on clusters whose first-attempt quality is the
//!    max (e.g., Ambiguous first attempt = 0.35, all later retries
//!    lower). In that case the quality delta collapses. With last-retry
//!    tracking plus declining Ambiguous quality, the regulator's halt
//!    at retry 3 records a HIGHER final quality (0.15) than the
//!    baseline's full retry 5 (0.05) — all of the reported +0.90
//!    quality delta is this effect on 9 Ambiguous queries.
//! 2. **`scope_drift_flags` under-reports circuit-broken turns**. P10
//!    priority makes `CircuitBreak` dominate `ScopeDriftWarn` in
//!    [`Regulator::decide`], so circuit-broken turns that would have
//!    flagged drift don't show up in this metric. Ambiguous canned
//!    responses drift against their tasks, but the 9 circuit-broken
//!    Ambiguous turns aren't counted here. The reported 41 is faithful
//!    to `decide() == ScopeDriftWarn`; the underlying drift rate is
//!    somewhat higher.
//!
//! ## Run
//!
//! ```bash
//! # Canned, deterministic (default — numbers reproducible):
//! cargo run --release --example task_eval_real_llm_regulator
//!
//! # Live — Ollama / Anthropic (per-query latency × 50 = 1-5 minute runs):
//! cargo run --release --example task_eval_real_llm_regulator -- ollama
//! cargo run --release --example task_eval_real_llm_regulator -- anthropic
//! ```
//!
//! ## What this session delivers
//!
//! Infrastructure + canned-mode numbers. Live-mode runs require either
//! local Ollama or an Anthropic API key + budget; a full-scale publication
//! run is a user-executed follow-up. The **canned numbers this session
//! reports are reproducible bit-for-bit** on any machine — the
//! quality oracle ([`Cluster::canned_quality`]) is a pure function of
//! `(cluster, retry)` and uses no randomness.

use std::env;
use std::time::Instant;

use noos::{CircuitBreakReason, Decision, LLMEvent, Regulator};

#[path = "regulator_common/mod.rs"]
mod regulator_common;
use regulator_common::{call_anthropic, call_ollama};

// ── Query taxonomy ────────────────────────────────────────────────

/// The four clusters in the stream. Each is designed to exercise a
/// specific Regulator decision path:
///
/// - `FactQA` — short, high-quality answers. Neither arm struggles.
///   Establishes the "easy queries aren't penalized" baseline.
/// - `Refactor` — LLM tends to over-expand (add docstrings, logging,
///   error handling). Exercises `ScopeDriftWarn`.
/// - `Ambiguous` — user intent unclear; responses stay low quality
///   across retries. Exercises `CircuitBreak(CostCapReached)` +
///   `CircuitBreak(QualityDeclineNoRecovery)`.
/// - `Debug` — mediocre first responses with correction follow-ups.
///   Accumulates into a `CorrectionPattern`; after threshold, later
///   Debug turns trigger `ProceduralWarning`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cluster {
    FactQA,
    Refactor,
    Ambiguous,
    Debug,
}

impl Cluster {
    /// Templated query with a numeric index so each query is distinct
    /// at the string level. The top-2 keywords (after stop-word filter)
    /// stay stable across indices so `build_topic_cluster` returns the
    /// same hash for same-variant queries — required for
    /// `CorrectionPattern` accumulation on the Debug cluster.
    fn user_query(self, idx: usize) -> String {
        match self {
            Self::FactQA => format!("What is the capital city of country number {idx}?"),
            Self::Refactor => format!("Refactor my auth handler number {idx} to async"),
            Self::Ambiguous => format!("Improve widget {idx} somehow"),
            Self::Debug => format!("Debug my parser module number {idx}"),
        }
    }

    /// Canned LLM response used by the default mode. Shapes differ per
    /// cluster so the drift / quality / procedural paths get exercised.
    fn canned_response(self, idx: usize, retry: usize) -> String {
        match self {
            Self::FactQA => format!("The capital is city_{idx}."),
            Self::Refactor => format!(
                "async fn auth_{idx}() {{ /* ... */ }}\n\
                 /// Added tracing, error handling, and timeout for resilience.\n\
                 tracing::info!(...); db.timeout(...).await??;"
            ),
            Self::Ambiguous => format!("One interpretation of widget {idx} (attempt {retry}): ... vague output ..."),
            Self::Debug => format!("// attempt {retry} for parser {idx}\nassert!(tokens_valid());"),
        }
    }

    /// Canned quality score. `Ambiguous` quality decays per retry to
    /// trigger `QualityDeclineNoRecovery` on the regulator arm.
    fn canned_quality(self, retry: usize) -> f64 {
        match self {
            Self::FactQA => 0.85,
            Self::Refactor => 0.70, // drifts but answers the ask
            Self::Ambiguous => (0.35 - 0.10 * retry as f64).max(0.05),
            Self::Debug => 0.55, // mediocre — triggers user correction
        }
    }

    /// Canned per-call `tokens_out`. Ambiguous is expensive to punish
    /// the baseline's retry behaviour; FactQA is cheap.
    fn canned_tokens_out(self) -> u32 {
        match self {
            Self::FactQA => 40,
            Self::Refactor => 180,
            Self::Ambiguous => 260,
            Self::Debug => 140,
        }
    }

    /// Whether this cluster produces a follow-up
    /// [`LLMEvent::UserCorrection`]. Only `Debug` does — this is what
    /// builds the procedural pattern.
    fn produces_correction(self) -> bool {
        matches!(self, Self::Debug)
    }

    fn correction_text(self, idx: usize) -> String {
        match self {
            Self::Debug => format!("not quite — try a different angle for parser {idx}"),
            _ => String::new(),
        }
    }
}

// ── Stream generation (deterministic, mixed) ──────────────────────

/// 50 queries, interleaved. Ratio: 18 FactQA / 10 Refactor / 9 Ambiguous /
/// 13 Debug (verified by counting the PATTERN literal — rebalance the
/// array and update this comment together).
///
/// **Procedural memory caveat**: the 13 Debug queries each emit one
/// `UserCorrection`, but the per-query `export()`/`import()` reset
/// DROPS below-threshold correction records per the documented
/// trade-off (see `Regulator::export` docs). Corrections therefore
/// never cross `MIN_CORRECTIONS_FOR_PATTERN = 3` in this stream
/// structure. `procedural_warnings` reports 0 on the canned run; this
/// is faithful to the intended usage — pattern formation requires
/// within-task correction accumulation, not cross-task. See
/// `regulator_correction_memory_demo.rs` for the path that does
/// exercise pattern formation.
fn generate_stream() -> Vec<(Cluster, usize)> {
    use Cluster::*;
    const PATTERN: [Cluster; 50] = [
        FactQA, Debug, Refactor, FactQA, Ambiguous,
        Debug, FactQA, Debug, Refactor, FactQA,
        Ambiguous, Debug, FactQA, Refactor, Debug,
        FactQA, Ambiguous, Debug, FactQA, Refactor,
        Debug, FactQA, Ambiguous, Refactor, FactQA,
        Debug, FactQA, Ambiguous, Debug, Refactor,
        FactQA, Debug, Refactor, FactQA, Ambiguous,
        Debug, FactQA, Ambiguous, Refactor, FactQA,
        Debug, Refactor, FactQA, Ambiguous, FactQA,
        Refactor, FactQA, Ambiguous, Debug, FactQA,
    ];
    PATTERN.iter().enumerate().map(|(i, &c)| (c, i)).collect()
}

// ── Arm config ────────────────────────────────────────────────────

/// Maximum retry attempts per query for BOTH arms. Set high enough
/// that baseline keeps burning cost on Ambiguous clusters after quality
/// has already bottomed out — that's the gap the regulator closes.
const NAIVE_RETRY_CAP: usize = 5;

/// Quality floor below which a retry is attempted.
const QUALITY_RETRY_THRESHOLD: f64 = 0.4;

/// Input tokens per LLM call. Roughly constant across the eval stream
/// (one-sentence user prompts). Named constant for P3 consistency with
/// the per-demo `PER_TURN_TOKENS_IN` pattern in Sessions 22-23.
const EVAL_TOKENS_IN: u32 = 25;

/// Wallclock estimate per LLM call. Kept under `cost::TYPICAL_TURN_WALLCLOCK_MS`
/// (10 000) so `normalize_cost` stays token-dominated.
const EVAL_WALLCLOCK_MS: u32 = 800;

/// Per-query token cap for the regulator arm. Reset between queries
/// (per-query regulator lifetime). 1000 matches the Session 19 plan
/// target: three 260-token Ambiguous retries total 780 — under cap,
/// so `CircuitBreak(CostCapReached)` doesn't fire on this workload
/// but `CircuitBreak(QualityDeclineNoRecovery)` does once the rolling
/// quality trend crosses `QUALITY_DECLINE_MIN_DELTA`.
const COST_CAP: u32 = 1_000;

// ── Metrics ───────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
struct RunMetrics {
    /// Arm label for the report row.
    label: &'static str,
    /// Queries served (should equal stream length for both arms).
    queries_served: usize,
    /// Regulator-only: queries where the retry loop was cut short by
    /// `CircuitBreak`. Subset of `queries_served`.
    queries_circuit_broken: usize,
    /// Sum of retry attempts across all served queries.
    total_attempts: usize,
    /// Sum of `tokens_out` across every LLM call including retries.
    total_cost: u32,
    /// Sum of final-attempt quality per served query.
    total_quality: f64,
    /// Regulator-only: turns where `ScopeDriftWarn` fired on delivery.
    scope_drift_flags: usize,
    /// Regulator-only: turns where `ProceduralWarning` fired pre-generation.
    procedural_warnings: usize,
    /// Regulator-only: which CircuitBreak reasons fired (for report).
    circuit_break_reasons: Vec<CircuitBreakReason>,
}

impl RunMetrics {
    fn mean_quality(&self) -> f64 {
        if self.queries_served == 0 {
            0.0
        } else {
            self.total_quality / self.queries_served as f64
        }
    }

    fn quality_per_1k_tokens(&self) -> f64 {
        if self.total_cost == 0 {
            0.0
        } else {
            self.total_quality * 1_000.0 / self.total_cost as f64
        }
    }
}

// ── LLM call dispatcher ───────────────────────────────────────────

/// Returns `(response_text, quality, tokens_out)`. Quality is
/// synthetic (oracle) even in live mode — a real integration supplies
/// quality via `QualityFeedback` from a grader or user signal. For
/// this eval, the canned oracle based on cluster + retry provides a
/// fair cross-arm comparison.
fn llm_call(
    mode: &str,
    cluster: Cluster,
    idx: usize,
    retry: usize,
) -> (String, f64, u32) {
    let quality = cluster.canned_quality(retry);
    match mode {
        "canned" => (
            cluster.canned_response(idx, retry),
            quality,
            cluster.canned_tokens_out(),
        ),
        "ollama" => match call_ollama(&cluster.user_query(idx)) {
            Ok((text, _ti, tokens_out, _wc)) => (text, quality, tokens_out),
            Err(_) => (
                cluster.canned_response(idx, retry),
                quality,
                cluster.canned_tokens_out(),
            ),
        },
        "anthropic" => match call_anthropic(&cluster.user_query(idx)) {
            Ok((text, _ti, tokens_out, _wc)) => (text, quality, tokens_out),
            Err(_) => (
                cluster.canned_response(idx, retry),
                quality,
                cluster.canned_tokens_out(),
            ),
        },
        _ => unreachable!("mode validated in main"),
    }
}

// ── Arm 1: Baseline (no regulator) ────────────────────────────────

fn run_baseline(stream: &[(Cluster, usize)], mode: &str) -> RunMetrics {
    let mut m = RunMetrics {
        label: "baseline",
        ..Default::default()
    };
    for &(cluster, idx) in stream {
        let outcome = run_retry_loop(mode, cluster, idx, &mut m, None);
        m.queries_served += 1;
        m.total_attempts += outcome.attempts;
        m.total_quality += outcome.final_quality;
    }
    m
}

// ── Arm 2: Regulator-enabled ──────────────────────────────────────

fn run_regulator(stream: &[(Cluster, usize)], mode: &str) -> RunMetrics {
    let mut m = RunMetrics {
        label: "regulator",
        ..Default::default()
    };
    let mut regulator = Regulator::for_user("eval_user").with_cost_cap(COST_CAP);

    for &(cluster, idx) in stream {
        // Per-query reset via export/import roundtrip. Preserves
        // LearnedState + CorrectionPattern (durable); clears
        // CostAccumulator + ScopeTracker + TokenStatsAccumulator +
        // pending_response (per-task). `with_cost_cap` re-applies the
        // cap because `import` rehydrates with `DEFAULT_TOKEN_CAP`.
        let snapshot = regulator.export();
        regulator = Regulator::import(snapshot).with_cost_cap(COST_CAP);

        regulator.on_event(LLMEvent::TurnStart {
            user_message: cluster.user_query(idx),
        });

        // Pre-generation probe: `ProceduralWarning` fires ONLY when
        // `TurnComplete` hasn't populated scope-tracker response
        // keywords yet (Session 23 design). Record the warning here.
        if matches!(regulator.decide(), Decision::ProceduralWarning { .. }) {
            m.procedural_warnings += 1;
        }

        let outcome = run_retry_loop(mode, cluster, idx, &mut m, Some(&mut regulator));

        if outcome.circuit_broken {
            m.queries_circuit_broken += 1;
        }

        // Post-delivery probe: `ScopeDriftWarn` fires on the DELIVERED
        // response. Separate `decide()` call from the pre-generation
        // probe above.
        if matches!(regulator.decide(), Decision::ScopeDriftWarn { .. }) {
            m.scope_drift_flags += 1;
        }

        m.queries_served += 1;
        m.total_attempts += outcome.attempts;
        m.total_quality += outcome.final_quality;

        // Follow-up correction (Debug cluster only).
        if cluster.produces_correction() {
            regulator.on_event(LLMEvent::UserCorrection {
                correction_message: cluster.correction_text(idx),
                corrects_last: true,
            });
        }
    }
    m
}

// ── Shared retry loop ─────────────────────────────────────────────

/// Outcome of one query's retry loop.
struct LoopOutcome {
    final_quality: f64,
    attempts: usize,
    /// True if the regulator arm halted the loop via `CircuitBreak`
    /// before the quality threshold was reached or the retry cap was
    /// exhausted.
    circuit_broken: bool,
}

/// Both arms use the same retry logic — the only difference is the
/// regulator arm feeds cost + quality events into [`Regulator`]
/// between attempts and halts on `CircuitBreak`.
fn run_retry_loop(
    mode: &str,
    cluster: Cluster,
    idx: usize,
    m: &mut RunMetrics,
    mut regulator: Option<&mut Regulator>,
) -> LoopOutcome {
    let mut final_quality = 0.0;
    let mut attempts = 0;

    for retry in 0..NAIVE_RETRY_CAP {
        attempts = retry + 1;
        let (response, quality, tokens_out) = llm_call(mode, cluster, idx, retry);
        m.total_cost += tokens_out;
        final_quality = quality;

        if let Some(reg) = regulator.as_deref_mut() {
            reg.on_event(LLMEvent::TurnComplete {
                full_response: response.clone(),
            });
            reg.on_event(LLMEvent::Cost {
                tokens_in: EVAL_TOKENS_IN,
                tokens_out,
                wallclock_ms: EVAL_WALLCLOCK_MS,
                provider: Some(mode.to_string()),
            });
            reg.on_event(LLMEvent::QualityFeedback {
                quality,
                fragment_spans: None,
            });

            // Check circuit break after this attempt.
            if let Decision::CircuitBreak { reason, .. } = reg.decide() {
                m.circuit_break_reasons.push(reason);
                return LoopOutcome {
                    final_quality,
                    attempts,
                    circuit_broken: true,
                };
            }
        }

        if quality >= QUALITY_RETRY_THRESHOLD {
            break;
        }
    }

    LoopOutcome {
        final_quality,
        attempts,
        circuit_broken: false,
    }
}

// ── Reporting ─────────────────────────────────────────────────────

fn print_header(mode: &str, stream_len: usize) {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Tier 2.2 — Regulator vs naive-retry baseline (Session 24)      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    println!("Mode:        {mode}");
    println!("Stream:      {stream_len} queries, 4-cluster mix");
    println!("Retry cap:   {NAIVE_RETRY_CAP} attempts/query");
    println!("Quality bar: retry if < {QUALITY_RETRY_THRESHOLD:.2}");
    println!("Cost cap:    {COST_CAP} tokens (regulator arm only)\n");
}

fn print_row(m: &RunMetrics) {
    println!(
        "  {:<12}  {:>6} {:>6} {:>8} {:>5} {:>8.3} {:>8.2} {:>8.2}",
        m.label,
        m.queries_served,
        m.queries_circuit_broken,
        m.total_attempts,
        m.total_cost,
        m.mean_quality(),
        m.total_quality,
        m.quality_per_1k_tokens(),
    );
}

fn print_deltas(baseline: &RunMetrics, regulator: &RunMetrics) {
    let cost_saved = baseline.total_cost as i64 - regulator.total_cost as i64;
    let quality_delta = regulator.total_quality - baseline.total_quality;
    let efficiency_delta =
        regulator.quality_per_1k_tokens() - baseline.quality_per_1k_tokens();

    println!("Primary comparison:");
    println!(
        "  cost_saved = {:+}  ({:.1}%)",
        cost_saved,
        100.0 * cost_saved as f64 / baseline.total_cost as f64
    );
    println!(
        "  total_quality delta = {:+.2}  (regulator - baseline)",
        quality_delta
    );
    println!(
        "  quality_per_1k_tokens delta = {:+.2}  (regulator - baseline)",
        efficiency_delta
    );
    println!();

    if efficiency_delta >= 0.5 {
        println!(
            "  ✓ Regulator is more cost-efficient by {:+.2} quality per 1k tokens.",
            efficiency_delta
        );
    } else if efficiency_delta >= 0.1 {
        println!(
            "  ≈ Regulator edges baseline on efficiency ({:+.2}) — real but narrow.",
            efficiency_delta
        );
    } else if efficiency_delta.abs() < 0.1 {
        println!(
            "  ≈ Regulator matches baseline on efficiency ({:+.2}) — infrastructure value",
            efficiency_delta
        );
        println!("    (halt / warn / pattern surfaces) without measurable loss.");
    } else {
        println!(
            "  ⚠ Regulator UNDERPERFORMS baseline on efficiency ({:+.2}) — investigate.",
            efficiency_delta
        );
    }
}

fn print_regulator_diagnostics(m: &RunMetrics) {
    if m.label != "regulator" {
        return;
    }
    println!();
    println!("Regulator diagnostics:");
    println!("  scope_drift_flags:     {}", m.scope_drift_flags);
    println!("  procedural_warnings:   {}", m.procedural_warnings);
    println!("  queries_circuit_broken: {}", m.queries_circuit_broken);
    if !m.circuit_break_reasons.is_empty() {
        println!("  circuit_break reasons:");
        for reason in &m.circuit_break_reasons {
            match reason {
                CircuitBreakReason::CostCapReached {
                    tokens_spent,
                    tokens_cap,
                    mean_quality_last_n,
                } => {
                    println!(
                        "    CostCapReached  tokens_spent={tokens_spent}  cap={tokens_cap}  \
                         mean_quality_last_n={mean_quality_last_n:.2}"
                    );
                }
                CircuitBreakReason::QualityDeclineNoRecovery { turns, mean_delta } => {
                    println!(
                        "    QualityDeclineNoRecovery  turns={turns}  mean_delta={mean_delta:.2}"
                    );
                }
                CircuitBreakReason::RepeatedFailurePattern {
                    cluster,
                    failure_count,
                } => {
                    println!(
                        "    RepeatedFailurePattern  cluster={cluster}  count={failure_count}"
                    );
                }
                other => {
                    // Future CircuitBreakReason variants — render generically.
                    println!("    {other:?}");
                }
            }
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────

fn main() {
    let mode = env::args().nth(1).unwrap_or_else(|| "canned".into());
    match mode.as_str() {
        "canned" | "ollama" | "anthropic" => {}
        other => {
            eprintln!("Unknown mode {other:?}. Use canned | ollama | anthropic.");
            std::process::exit(2);
        }
    }

    let stream = generate_stream();
    print_header(&mode, stream.len());

    let t0 = Instant::now();
    let baseline = run_baseline(&stream, &mode);
    let regulator = run_regulator(&stream, &mode);
    let wallclock_s = t0.elapsed().as_secs_f64();

    println!("Per-arm summary (wallclock {wallclock_s:.1}s):");
    println!(
        "  {:<12}  {:>6} {:>6} {:>8} {:>5} {:>8} {:>8} {:>8}",
        "arm", "served", "CB_hits", "attempts", "cost", "mean_q", "total_q", "q/1k"
    );
    println!("  {}", "─".repeat(80));
    print_row(&baseline);
    print_row(&regulator);
    println!();

    print_deltas(&baseline, &regulator);
    print_regulator_diagnostics(&regulator);

    println!();
    println!("Notes:");
    println!("  • Canned mode uses a deterministic quality oracle; numbers are");
    println!("    reproducible bit-for-bit on any machine.");
    println!("  • Live modes (ollama / anthropic) call real LLMs for response text");
    println!("    + tokens_out, but STILL use the synthetic quality oracle. A full");
    println!("    publication run needs a real quality grader; that's Session 24b.");
    println!("  • Quality-per-1k-tokens is the fair cross-arm metric when the");
    println!("    regulator cuts retry loops short — total_quality alone wouldn't");
    println!("    weight cost saved against the quality preserved.");
    println!("  • `procedural_warnings` = 0 is EXPECTED in this eval structure:");
    println!("    the per-query export/import roundtrip drops below-threshold");
    println!("    correction records (known trade-off per `Regulator::export`");
    println!("    docs). Pattern formation requires within-task accumulation;");
    println!("    see `regulator_correction_memory_demo.rs` for that path.");
}
