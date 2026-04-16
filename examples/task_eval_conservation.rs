//! Tier 1.2 conservation eval — does `signals.conservation` help apps stay
//! within budget while preserving quality?
//!
//! Run: `cargo run --example task_eval_conservation`
//!
//! ## What this tests
//!
//! Claim: the `conservation` signal integrates body_budget depletion, resource
//! pressure, and sustained arousal — so apps that read it can stop/downshift
//! expensive work earlier than a cost-only counter would, preserving aggregate
//! quality under a fixed budget.
//!
//! ## Setup
//!
//! - Mixed stream of 20 queries (stressful and benign, deterministic seed).
//! - Budget cap = 10.0 effort units. Agents that exceed stop serving new queries.
//! - Two modes per query: "full" (cost 0.8, quality 0.9) or "shallow"
//!   (cost 0.2, quality 0.5). Shallow represents a cheaper fallback path.
//! - Three agents:
//!   1. **Always-full** (reference): uses full mode until budget can't afford it.
//!   2. **Cost-threshold** (FAIR BASELINE — no Nous): switches to shallow when
//!      cumulative cost exceeds budget / 2. Uses only self-tracked cost.
//!   3. **Nous-conservation**: switches to shallow when `signals.conservation`
//!      exceeds 0.5. Uses Nous's combined signal.
//!
//! ## Where Nous should win
//!
//! On stressful inputs, `perceive` depletes body_budget via arousal — the
//! cost-only tracker misses this. Nous's conservation responds; the agent
//! switches to shallow earlier on the stress-heavy suffix of the stream,
//! leaving more budget for later queries.
//!
//! ## Where Nous should NOT win
//!
//! On a pure cost-driven stream with no stress variation, conservation
//! reduces to "budget-below-threshold" and is roughly equivalent to the
//! cost-only tracker. Honest comparison: report both cases.

use noos::session::CognitiveSession;
use noos::types::intervention::CognitiveSignals;

// ─── Query stream ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum QueryKind {
    BenignCheap,
    BenignExpensive,
    StressfulCheap,
    StressfulExpensive,
}

impl QueryKind {
    fn is_stressful(self) -> bool {
        matches!(self, Self::StressfulCheap | Self::StressfulExpensive)
    }

    fn text(self, idx: usize) -> String {
        match self {
            Self::BenignCheap => format!("What is the capital of country {idx}?"),
            Self::BenignExpensive => format!(
                "Explain the full derivation of algorithm {idx} step by step \
                 with all intermediate calculations and proofs."
            ),
            Self::StressfulCheap => format!(
                "HELP!!! Everything is broken!!! The system {idx} is failing!!!"
            ),
            Self::StressfulExpensive => format!(
                "Urgent: please debug this complex failure in system {idx}!!! \
                 I am extremely worried and the whole team is counting on me!!!"
            ),
        }
    }
}

/// Generate a deterministic 20-query stream mixing all 4 kinds.
/// Fixed seed so baseline vs nous compare on the same sequence.
fn generate_stream() -> Vec<(QueryKind, String)> {
    // Deterministic interleaving that puts half the stress at the end,
    // exercising the "late stress spike" case where cost-only underestimates.
    let pattern = [
        QueryKind::BenignCheap,
        QueryKind::BenignExpensive,
        QueryKind::BenignCheap,
        QueryKind::BenignExpensive,
        QueryKind::BenignCheap,
        QueryKind::BenignExpensive,
        QueryKind::BenignCheap,
        QueryKind::BenignExpensive,
        QueryKind::BenignCheap,
        QueryKind::BenignExpensive,
        QueryKind::StressfulCheap,
        QueryKind::StressfulExpensive,
        QueryKind::StressfulCheap,
        QueryKind::StressfulExpensive,
        QueryKind::StressfulCheap,
        QueryKind::StressfulExpensive,
        QueryKind::StressfulCheap,
        QueryKind::StressfulExpensive,
        QueryKind::StressfulCheap,
        QueryKind::StressfulExpensive,
    ];
    pattern
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, k.text(i)))
        .collect()
}

// ─── Processing modes ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Full,
    Shallow,
}

fn cost_for(kind: QueryKind, mode: Mode) -> f64 {
    let base = match kind {
        QueryKind::BenignCheap | QueryKind::StressfulCheap => 0.3,
        QueryKind::BenignExpensive | QueryKind::StressfulExpensive => 0.8,
    };
    match mode {
        Mode::Full => base,
        Mode::Shallow => 0.2,
    }
}

fn quality_for(kind: QueryKind, mode: Mode) -> f64 {
    match mode {
        Mode::Full => match kind {
            QueryKind::BenignCheap => 0.85,
            QueryKind::BenignExpensive => 0.90,
            QueryKind::StressfulCheap => 0.80,
            QueryKind::StressfulExpensive => 0.90,
        },
        Mode::Shallow => match kind {
            QueryKind::BenignCheap => 0.55,
            QueryKind::BenignExpensive => 0.45,
            QueryKind::StressfulCheap => 0.50,
            QueryKind::StressfulExpensive => 0.40,
        },
    }
}

// ─── Agents ───────────────────────────────────────────────────────────────

const BUDGET_CAP: f64 = 10.0;

#[derive(Debug, Default, Clone)]
struct RunResult {
    queries_served: usize,
    queries_skipped: usize,
    total_cost: f64,
    total_quality: f64,
    mode_switches_to_shallow: usize,
}

impl RunResult {
    fn avg_quality(&self) -> f64 {
        if self.queries_served == 0 {
            0.0
        } else {
            self.total_quality / self.queries_served as f64
        }
    }
}

/// Always-full agent (reference low bar): uses full mode until it would exceed budget.
fn run_always_full(stream: &[(QueryKind, String)]) -> RunResult {
    let mut r = RunResult::default();
    for (kind, _text) in stream {
        let cost = cost_for(*kind, Mode::Full);
        if r.total_cost + cost > BUDGET_CAP {
            r.queries_skipped += 1;
            continue;
        }
        r.total_cost += cost;
        r.total_quality += quality_for(*kind, Mode::Full);
        r.queries_served += 1;
    }
    r
}

/// Cost-threshold agent (FAIR baseline, no Nous): switches to shallow when
/// cumulative cost crosses budget / 2.
fn run_cost_threshold(stream: &[(QueryKind, String)]) -> RunResult {
    let mut r = RunResult::default();
    let threshold = BUDGET_CAP / 2.0;
    let mut in_shallow = false;
    for (kind, _text) in stream {
        if !in_shallow && r.total_cost >= threshold {
            in_shallow = true;
            r.mode_switches_to_shallow += 1;
        }
        let mode = if in_shallow { Mode::Shallow } else { Mode::Full };
        let cost = cost_for(*kind, mode);
        if r.total_cost + cost > BUDGET_CAP {
            r.queries_skipped += 1;
            continue;
        }
        r.total_cost += cost;
        r.total_quality += quality_for(*kind, mode);
        r.queries_served += 1;
    }
    r
}

/// Nous-conservation agent: uses `signals.conservation` to decide mode.
/// Calls `process_message` to update Nous's internal state, then reads the
/// signal and acts. Reports cost back via `track_cost` to close the loop.
/// Threshold at which Nous-conservation agent switches to shallow mode.
///
/// Calibrated 2026-04-14 against observed signal range: after the sustained
/// fix, `signals.conservation` tops out around 0.30 over 60 sustained-stress
/// turns (body_budget doesn't drop far enough below the adaptive
/// `threshold_body_budget_conservation` base 0.30 to contribute a large
/// `budget_factor` on realistic session lengths). Using 0.5 as documented
/// in app-contract.md would never trigger. Using 0.2 matches the signal's
/// actual mid-range and exercises the switch. Broader calibration gap
/// between documented 0.5 threshold and observed mid-range is still owed.
const NOUS_CONSERVATION_THRESHOLD: f64 = 0.2;

fn run_nous_conservation(stream: &[(QueryKind, String)]) -> RunResult {
    let mut session = CognitiveSession::new();
    let mut r = RunResult::default();
    let mut in_shallow = false;

    for (kind, text) in stream {
        let turn = session.process_message(text);

        // Decide mode from signals.conservation. Above threshold → shallow.
        // Hysteresis: once in shallow, stay (don't flip-flop within a run).
        if !in_shallow && turn.signals.conservation > NOUS_CONSERVATION_THRESHOLD {
            in_shallow = true;
            r.mode_switches_to_shallow += 1;
        }
        let mode = if in_shallow { Mode::Shallow } else { Mode::Full };

        let cost = cost_for(*kind, mode);
        if r.total_cost + cost > BUDGET_CAP {
            r.queries_skipped += 1;
            // Still close the loop so conservation reflects reality.
            session.track_cost(0.0);
            continue;
        }

        let quality = quality_for(*kind, mode);
        r.total_cost += cost;
        r.total_quality += quality;
        r.queries_served += 1;

        // Close the allostatic loop: report cost normalized to [0, 1].
        // Full cost 0.8 → track_cost(0.8); shallow cost 0.2 → track_cost(0.2).
        session.track_cost(cost);
        let response = match mode {
            Mode::Full => "Detailed response covering all aspects thoroughly.",
            Mode::Shallow => "Short answer.",
        };
        session.process_response(response, quality);
    }

    r
}

// ─── Diagnostic pass: trace conservation signal per turn ──────────────────

#[derive(Debug, Clone)]
struct Trace {
    signals: CognitiveSignals,
    body_budget: f64,
    sustained: f64,
    arousal: f64,
}

/// Trace that uses REPORTED quality — for seeing the "typical app" signal
/// trajectory (costs reported, responses succeed at their normal quality).
fn trace_conservation(stream: &[(QueryKind, String)]) -> Vec<Trace> {
    let mut session = CognitiveSession::new();
    let mut traces = Vec::with_capacity(stream.len());
    for (kind, text) in stream {
        let turn = session.process_message(text);
        let model = session.world_model();
        traces.push(Trace {
            signals: turn.signals.clone(),
            body_budget: model.body_budget,
            sustained: model.belief.affect.sustained,
            arousal: model.belief.affect.arousal,
        });
        let cost = cost_for(*kind, Mode::Full);
        session.track_cost(cost);
        session.process_response("ack", quality_for(*kind, Mode::Full));
    }
    traces
}

/// Trace that simulates "struggling through hard content" — low reported
/// quality so RPE doesn't replenish body_budget. Needed for sensitivity
/// check: when apps report high quality, body_budget stays stable even
/// under cost (RPE-driven replenishment). The conservation signal is
/// designed to fire when cost + poor outcomes BOTH apply.
fn trace_conservation_struggling(stream: &[(QueryKind, String)]) -> Vec<Trace> {
    let mut session = CognitiveSession::new();
    let mut traces = Vec::with_capacity(stream.len());
    for (kind, text) in stream {
        let turn = session.process_message(text);
        let model = session.world_model();
        traces.push(Trace {
            signals: turn.signals.clone(),
            body_budget: model.body_budget,
            sustained: model.belief.affect.sustained,
            arousal: model.belief.affect.arousal,
        });
        let cost = cost_for(*kind, Mode::Full);
        session.track_cost(cost);
        // Low quality (0.35): simulates app struggling — responses unsatisfactory.
        // RPE stays negative → no budget replenishment, gradual decline.
        session.process_response("ack", 0.35);
    }
    traces
}

// ─── Reporting ────────────────────────────────────────────────────────────

fn print_row(name: &str, r: &RunResult) {
    println!(
        "  {:<28} served={:>2} skipped={:>2} switches={:>2} cost={:>5.2} avg_q={:.3} total_q={:>5.2}",
        name,
        r.queries_served,
        r.queries_skipped,
        r.mode_switches_to_shallow,
        r.total_cost,
        r.avg_quality(),
        r.total_quality
    );
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  task_eval_conservation — Tier 1.2 conservation signal eval  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("Tests whether `signals.conservation` helps an app stay within");
    println!("budget while preserving aggregate quality. Synthetic task, fair");
    println!("comparison — cost-only baseline is the bar Nous must beat.\n");

    let stream = generate_stream();
    let total_stressful = stream.iter().filter(|(k, _)| k.is_stressful()).count();
    println!(
        "Stream: {} queries ({} stressful, budget cap = {:.1})\n",
        stream.len(),
        total_stressful,
        BUDGET_CAP
    );

    let always_full = run_always_full(&stream);
    let cost_threshold = run_cost_threshold(&stream);
    let nous = run_nous_conservation(&stream);

    println!("Per-condition summary:");
    println!(
        "  {:<28} {:>10} {:>10} {:>10} {:>6} {:>7} {:>8}",
        "condition", "served", "skipped", "switches", "cost", "avg_q", "total_q"
    );
    println!("  {}", "─".repeat(90));
    print_row("always-full (reference)", &always_full);
    print_row("cost-threshold (no Nous)", &cost_threshold);
    print_row("nous-conservation", &nous);

    // Quick dispersion check: since the stream + simulator are deterministic,
    // repeated runs produce bit-identical output. The 3-seed requirement from
    // CR5 only applies when there's stochastic variation — this harness uses
    // a fixed pattern, so one run IS the sample.
    println!("\n(Deterministic stream — seeds would produce identical runs; 3-seed");
    println!("requirement applies when stochasticity exists.)");

    // Pass/fail against the primary metric: total quality delivered within budget.
    println!("\nComparison (higher total_quality = better):");
    let cost_q = cost_threshold.total_quality;
    let nous_q = nous.total_quality;
    let delta = nous_q - cost_q;
    if delta > 0.5 {
        println!(
            "  ✓ Nous-conservation beats cost-threshold by {:+.2} total quality.",
            delta
        );
    } else if delta > 0.05 {
        println!(
            "  ≈ Nous-conservation edges cost-threshold by {:+.2} total quality —",
            delta
        );
        println!("    within noise territory for a synthetic task. Real benchmarks needed.");
    } else if delta.abs() <= 0.05 {
        println!(
            "  ≈ Nous-conservation matches cost-threshold ({:+.2}) — no discernible advantage",
            delta
        );
        println!("    on this stream. Expected: conservation ≈ cost-below-threshold when");
        println!("    stress depletion is not the dominant signal.");
    } else {
        println!(
            "  ⚠ Nous-conservation UNDERPERFORMS cost-threshold by {:+.2} — investigate.",
            delta
        );
    }

    println!("\nDiagnostic: state trace (first 8 turns):");
    let traces = trace_conservation(&stream);
    println!(
        "  {:<4} {:<20} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "turn", "kind", "cons", "sal", "bud", "sust", "aro"
    );
    for (i, t) in traces.iter().take(8).enumerate() {
        println!(
            "  {:<4} {:<20} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            i + 1,
            format!("{:?}", stream[i].0),
            t.signals.conservation,
            t.signals.salience,
            t.body_budget,
            t.sustained,
            t.arousal,
        );
    }
    println!("\nTrace for turns 11-20 (stressful suffix):");
    for (i, t) in traces.iter().skip(10).enumerate() {
        println!(
            "  {:<4} {:<20} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            i + 11,
            format!("{:?}", stream[i + 10].0),
            t.signals.conservation,
            t.signals.salience,
            t.body_budget,
            t.sustained,
            t.arousal,
        );
    }

    // ── Sensitivity check: does conservation EVER cross 0.5 in this regime? ──
    println!("\nSensitivity check — sustained-stress run (60 turns, all stressful):");
    let long_stream: Vec<(QueryKind, String)> = (0..60)
        .map(|i| {
            let kind = if i % 2 == 0 {
                QueryKind::StressfulExpensive
            } else {
                QueryKind::StressfulCheap
            };
            (kind, kind.text(i))
        })
        .collect();
    // Use the "struggling" variant: low reported quality prevents RPE-driven
    // body_budget replenishment that would mask cost depletion. This matches
    // the real "should conserve" scenario: costs accumulating AND responses
    // not meeting the bar.
    let long_traces = trace_conservation_struggling(&long_stream);
    let max_cons = long_traces
        .iter()
        .map(|t| t.signals.conservation)
        .fold(0.0_f64, |a, b| a.max(b));
    let first_crossing = long_traces
        .iter()
        .position(|t| t.signals.conservation > 0.5);
    let min_bud = long_traces
        .iter()
        .map(|t| t.body_budget)
        .fold(1.0_f64, |a, b| a.min(b));
    let min_sust = long_traces
        .iter()
        .map(|t| t.sustained)
        .fold(1.0_f64, |a, b| a.min(b));
    println!(
        "  Max conservation over 60 stressful turns: {:.3}",
        max_cons
    );
    println!("  Min body_budget over 60 turns:            {:.3}", min_bud);
    println!("  Min sustained over 60 turns:              {:.3}", min_sust);
    match first_crossing {
        Some(t) => println!("  First crossing conservation > 0.5 at turn: {}", t + 1),
        None => println!(
            "  ⚠ Conservation NEVER crossed 0.5 over 60 turns of sustained stress.\n    \
               Signal may be tuned for even longer depletion horizons, OR the\n    \
               body_budget depletion rate is too low relative to the adaptive\n    \
               threshold. Follow-up: inspect adaptive_thresholds / COST_DEPLETION_RATE."
        ),
    }

    println!("\nNotes:");
    println!("  • Synthetic benchmark — behavior illustration, not a claim validation");
    println!("    for real LLM use. Tier 2 benchmarks (real agent streams) apply.");
    println!("  • Conservation signal's value over cost-only tracking is clearest");
    println!("    when stress depletes budget independently of reported cost.");
    println!("  • If nous == cost-only on this synthetic stream, it means stress");
    println!("    depletion didn't dominate — a real finding, report it honestly.");
}
