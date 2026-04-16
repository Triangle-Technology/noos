//! Tier 1.3 multi-signal eval — does Nous compound on a mixed workload?
//!
//! Run: `cargo run --example task_eval_multi_signal`
//!
//! Tiers 1.1 and 1.2 each validated ONE signal against ONE baseline:
//! - Tier 1.1 (`task_eval_synthetic.rs`): cross-session reward learning vs simple retry.
//! - Tier 1.2 (`task_eval_conservation.rs`): conservation signal vs cost-threshold.
//!
//! Tier 1.3 asks the harder question: **when a workload needs multiple
//! signals, does Nous compound their value vs a smart baseline that uses
//! equivalent app-level tracking?** A weak result here ("Nous matches smart
//! baseline") would mean the allostatic claim is architecturally fine but
//! doesn't beat a competent engineer writing app-level state.
//!
//! ## Setup
//!
//! Stream: 24 queries mixing 5 categories (3 pre-trained + 2 novel), varying
//! stress (some emotionally loaded), varying cost, some with ambiguous-format
//! responses (so safe-detection vs naive-detection may matter).
//!
//! Budget cap: 12.0 effort units. Over-budget → query skipped.
//!
//! ## Three agents compared
//!
//! 1. **Naive** (reference): single strategy (DirectAnswer), no adaptation,
//!    serves until budget runs out. Shows the floor.
//! 2. **Smart baseline** (FAIR BASELINE, no Nous): tracks cost, per-category
//!    strategy memory (last-used + rotate on low quality), switches to
//!    shallow when cost > budget/2. This is what a competent engineer builds
//!    without Nous — the bar Nous-full must beat to claim allostatic value.
//! 3. **Nous-full pipeline**: warm-started `CognitiveSession` pre-trained on
//!    categories 1-3, uses `signals.strategy` for learned recommendations,
//!    `signals.conservation > 0.2` to switch to shallow mode, closed loop
//!    via `track_cost` + `process_response`.
//!
//! ## What would a positive result look like
//!
//! Nous-full total_quality > smart_baseline total_quality by ≥ ~1.0 on a
//! 24-query stream (equivalent of 1+ free "expensive-success" slots gained).
//! If Nous matches smart baseline within noise, the allostatic claim is
//! infrastructure-only — fine for building on, but not by itself a value
//! story for LLM applications.

use nous_regulator::session::CognitiveSession;
use nous_regulator::types::world::{LearnedState, ResponseStrategy};
use std::collections::BTreeMap;

// ─── Categories, strategies, query generation ─────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Category {
    // Pre-trained categories (Nous has prior LearnedState for these).
    Debug,
    Lookup,
    Clarify,
    // Novel categories (Nous has no prior data — must discover, same as baseline).
    Analyze,
    Summarize,
}

impl Category {
    fn user_query(self, idx: usize) -> String {
        match self {
            Self::Debug => format!("Help me debug this numerical issue {idx}."),
            Self::Lookup => format!("What is the default port for service {idx}?"),
            Self::Clarify => format!("Make system {idx} better somehow."),
            Self::Analyze => format!("Analyze the tradeoffs in approach {idx}."),
            Self::Summarize => format!("Summarize the key results of experiment {idx}."),
        }
    }

    fn correct_strategy(self) -> AppStrategy {
        match self {
            Self::Debug => AppStrategy::StepByStep,
            Self::Lookup => AppStrategy::DirectAnswer,
            Self::Clarify => AppStrategy::AskClarifying,
            Self::Analyze => AppStrategy::StepByStep,
            Self::Summarize => AppStrategy::DirectAnswer,
        }
    }

    /// Whether queries of this category tend to produce stressful tone.
    fn is_stressful(self) -> bool {
        matches!(self, Self::Debug | Self::Clarify)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppStrategy {
    DirectAnswer,
    AskClarifying,
    StepByStep,
}

impl AppStrategy {
    fn next(self) -> Self {
        match self {
            Self::DirectAnswer => Self::AskClarifying,
            Self::AskClarifying => Self::StepByStep,
            Self::StepByStep => Self::DirectAnswer,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Full,
    Shallow,
}

/// Simulated LLM returning (response_text, quality, cost).
///
/// Quality depends on (strategy, category) match AND mode (shallow takes a
/// quality hit). Some responses use ambiguous format to stress safe-detection.
fn simulate_llm(strategy: AppStrategy, category: Category, mode: Mode) -> (String, f64, f64) {
    let correct = category.correct_strategy();
    let quality_full = if strategy == correct { 0.90 } else {
        match strategy {
            AppStrategy::AskClarifying => 0.35,
            AppStrategy::StepByStep => 0.30,
            AppStrategy::DirectAnswer => 0.20,
        }
    };
    let (quality, cost) = match mode {
        Mode::Full => (quality_full, match strategy {
            AppStrategy::DirectAnswer => 0.2,
            AppStrategy::AskClarifying => 0.35,
            AppStrategy::StepByStep => 0.70,
        }),
        Mode::Shallow => (quality_full * 0.55, 0.15),
    };

    // Response text: well-formatted for Debug+StepByStep (multi-line),
    // single-line for Analyze+StepByStep (ambiguous to safe-detection —
    // exercises the brittleness-fix path). Lookup+DirectAnswer is short.
    let text = match (category, strategy, mode) {
        (Category::Debug, AppStrategy::StepByStep, Mode::Full) => {
            "1. First, check the loop bounds.\n\
             2. Then, verify the index.\n\
             3. Next, trace a small example.\n\
             4. Finally, add assertions."
                .to_string()
        }
        (Category::Analyze, AppStrategy::StepByStep, Mode::Full) => {
            // Single-line numbered — ambiguous format (regression case from
            // phase 2). Safe detection should return None here → consolidate
            // doesn't record this as StepByStep → Nous must still succeed
            // at the TASK even if strategy-learning rejects the turn.
            "1. First tradeoff. 2. Second tradeoff. 3. Third tradeoff.".to_string()
        }
        (_, AppStrategy::AskClarifying, _) => {
            "What exactly do you need? What have you tried so far?".to_string()
        }
        (_, AppStrategy::DirectAnswer, _) => {
            format!("Short answer for {:?}.", category)
        }
        (_, AppStrategy::StepByStep, _) => {
            "1. First step.\n2. Then more.\n3. Finally done.".to_string()
        }
    };

    (text, quality, cost)
}

// ─── Stream generation (deterministic, mixed) ─────────────────────────────

fn generate_stream() -> Vec<(Category, String)> {
    // Deliberate ordering: interleaves pre-trained + novel + stressful.
    // Total 24 queries (5 × ~5 each).
    let pattern = [
        Category::Debug,        // pre-trained, stressful, StepByStep
        Category::Lookup,       // pre-trained, DirectAnswer
        Category::Analyze,      // novel, StepByStep with ambiguous format
        Category::Clarify,      // pre-trained, stressful, AskClarifying
        Category::Summarize,    // novel, DirectAnswer
        Category::Debug,
        Category::Lookup,
        Category::Analyze,
        Category::Clarify,
        Category::Summarize,
        Category::Debug,
        Category::Lookup,
        Category::Analyze,
        Category::Clarify,
        Category::Summarize,
        Category::Debug,
        Category::Lookup,
        Category::Analyze,
        Category::Clarify,
        Category::Summarize,
        Category::Debug,
        Category::Lookup,
        Category::Analyze,
        Category::Summarize,
    ];
    pattern
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, c.user_query(i)))
        .collect()
}

// ─── Pre-training for warm-start (Nous only) ──────────────────────────────

fn train_prior_session() -> LearnedState {
    let mut session = CognitiveSession::new();
    // Train only the "pre-trained" categories. Novel ones are deliberately absent.
    let trained = [Category::Debug, Category::Lookup, Category::Clarify];
    for _round in 0..6 {
        for &cat in &trained {
            let _ = session.process_message(&cat.user_query(999));
            let (resp, quality, cost) = simulate_llm(cat.correct_strategy(), cat, Mode::Full);
            session.track_cost(cost);
            session.process_response(&resp, quality);
        }
    }
    session.export_learned()
}

// ─── Metrics ──────────────────────────────────────────────────────────────

const BUDGET_CAP: f64 = 12.0;

#[derive(Debug, Default, Clone)]
struct RunResult {
    queries_served: usize,
    queries_skipped: usize,
    total_cost: f64,
    total_quality: f64,
    mode_switches_to_shallow: usize,
    correct_strategy_used: usize,
    correct_on_novel: BTreeMap<Category, (usize, usize)>, // (correct, attempts)
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

fn record_outcome(
    r: &mut RunResult,
    cat: Category,
    strategy: AppStrategy,
    quality: f64,
    cost: f64,
) {
    r.total_cost += cost;
    r.total_quality += quality;
    r.queries_served += 1;
    if strategy == cat.correct_strategy() {
        r.correct_strategy_used += 1;
    }
    if matches!(cat, Category::Analyze | Category::Summarize) {
        let entry = r.correct_on_novel.entry(cat).or_insert((0, 0));
        entry.1 += 1;
        if strategy == cat.correct_strategy() {
            entry.0 += 1;
        }
    }
}

// ─── Agent 1: Naive (single strategy, no adaptation) ─────────────────────

fn run_naive(stream: &[(Category, String)]) -> RunResult {
    let mut r = RunResult::default();
    let strategy = AppStrategy::DirectAnswer;
    for (cat, _) in stream {
        let (_resp, quality, cost) = simulate_llm(strategy, *cat, Mode::Full);
        if r.total_cost + cost > BUDGET_CAP {
            r.queries_skipped += 1;
            continue;
        }
        record_outcome(&mut r, *cat, strategy, quality, cost);
    }
    r
}

// ─── Agent 2: Smart baseline (no Nous) ────────────────────────────────────
//
// Everything a competent engineer would build WITHOUT Nous:
// - Per-category strategy memory (remembers last successful strategy).
// - Rotation on low quality.
// - Cost tracking + conservation (shallow mode once spending crosses budget/2).

const COST_CONSERVATION_FRACTION: f64 = 0.5;
const QUALITY_RETRY_THRESHOLD: f64 = 0.5;

fn run_smart_baseline(stream: &[(Category, String)]) -> RunResult {
    let mut r = RunResult::default();
    let mut per_category_strategy: BTreeMap<Category, AppStrategy> = BTreeMap::new();
    let mut per_category_last_quality: BTreeMap<Category, f64> = BTreeMap::new();

    for (cat, _) in stream {
        // Select strategy: start with DirectAnswer, remember last-used per category.
        let current_strategy = per_category_strategy
            .entry(*cat)
            .or_insert(AppStrategy::DirectAnswer);

        // If last attempt on this category was low quality, rotate.
        if let Some(&q) = per_category_last_quality.get(cat) {
            if q < QUALITY_RETRY_THRESHOLD {
                *current_strategy = current_strategy.next();
            }
        }
        let strategy = *current_strategy;

        // Cost-aware mode: shallow once we've spent half budget.
        let mode = if r.total_cost >= BUDGET_CAP * COST_CONSERVATION_FRACTION {
            if r.mode_switches_to_shallow == 0 {
                r.mode_switches_to_shallow = 1;
            }
            Mode::Shallow
        } else {
            Mode::Full
        };

        let (_resp, quality, cost) = simulate_llm(strategy, *cat, mode);
        if r.total_cost + cost > BUDGET_CAP {
            r.queries_skipped += 1;
            continue;
        }
        per_category_last_quality.insert(*cat, quality);
        record_outcome(&mut r, *cat, strategy, quality, cost);
    }
    r
}

// ─── Agent 3: Nous full pipeline (warm-started + conservation-aware) ─────

fn map_rec(r: ResponseStrategy, fallback: AppStrategy) -> AppStrategy {
    match r {
        ResponseStrategy::StepByStep => AppStrategy::StepByStep,
        ResponseStrategy::ClarifyFirst => AppStrategy::AskClarifying,
        ResponseStrategy::DirectAnswer => AppStrategy::DirectAnswer,
        _ => fallback,
    }
}

/// Switch threshold for Nous-conservation (calibrated in phase 6 to match
/// the signal's operating range under realistic mixed workload).
const NOUS_CONSERVATION_THRESHOLD: f64 = 0.2;

fn run_nous_full(stream: &[(Category, String)], training: LearnedState) -> RunResult {
    let mut session = CognitiveSession::with_learned(training, 64);
    let mut r = RunResult::default();
    let mut per_category_strategy: BTreeMap<Category, AppStrategy> = BTreeMap::new();
    let mut per_category_last_quality: BTreeMap<Category, f64> = BTreeMap::new();
    let mut in_shallow = false;

    for (cat, text) in stream {
        let turn = session.process_message(text);

        // Conservation-driven mode selection.
        if !in_shallow && turn.signals.conservation > NOUS_CONSERVATION_THRESHOLD {
            in_shallow = true;
            r.mode_switches_to_shallow += 1;
        }
        let mode = if in_shallow { Mode::Shallow } else { Mode::Full };

        // Strategy selection: prefer Nous's learned recommendation (cross-session).
        // Fall back to per-category app-level tracking + retry.
        let current = per_category_strategy
            .entry(*cat)
            .or_insert(AppStrategy::DirectAnswer);
        let strategy = if let Some(rec) = turn.signals.strategy {
            map_rec(rec, *current)
        } else if let Some(&q) = per_category_last_quality.get(cat) {
            if q < QUALITY_RETRY_THRESHOLD {
                current.next()
            } else {
                *current
            }
        } else {
            *current
        };
        *current = strategy;

        let (resp, quality, cost) = simulate_llm(strategy, *cat, mode);
        if r.total_cost + cost > BUDGET_CAP {
            r.queries_skipped += 1;
            // Close the loop even on skip so signal reflects reality.
            session.track_cost(0.0);
            continue;
        }
        per_category_last_quality.insert(*cat, quality);
        record_outcome(&mut r, *cat, strategy, quality, cost);

        // Close the allostatic loop: cost + quality both reported so that
        // conservation, reward learning, and safe-detection can all update.
        session.track_cost(cost);
        session.process_response(&resp, quality);
    }
    r
}

// ─── Reporting ────────────────────────────────────────────────────────────

fn print_row(name: &str, r: &RunResult) {
    println!(
        "  {:<22} served={:>2} skipped={:>2} switches={:>2} cost={:>5.2}  avg_q={:.3}  total_q={:>6.2}  correct={:>2}/{}",
        name,
        r.queries_served,
        r.queries_skipped,
        r.mode_switches_to_shallow,
        r.total_cost,
        r.avg_quality(),
        r.total_quality,
        r.correct_strategy_used,
        r.queries_served,
    );
}

fn print_novel_breakdown(name: &str, r: &RunResult) {
    if r.correct_on_novel.is_empty() {
        return;
    }
    println!("  {} — novel-category correct-strategy rate:", name);
    for (cat, (correct, total)) in &r.correct_on_novel {
        println!(
            "    {:<12}  {}/{} correct",
            format!("{:?}", cat),
            correct,
            total
        );
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  task_eval_multi_signal — Tier 1.3 signal integration test   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("Asks whether Nous compounds reward learning + conservation +");
    println!("safe detection on a mixed workload, vs a smart app-level baseline.\n");

    let stream = generate_stream();
    let total_stressful = stream.iter().filter(|(c, _)| c.is_stressful()).count();
    println!(
        "Stream: {} queries (5 categories, {} stressful, budget={:.1})\n",
        stream.len(),
        total_stressful,
        BUDGET_CAP
    );

    let training = train_prior_session();
    println!(
        "Training exported LearnedState: {} strategy clusters, tick={}\n",
        training.response_strategies.len(),
        training.tick
    );

    let naive = run_naive(&stream);
    let smart = run_smart_baseline(&stream);
    let nous = run_nous_full(&stream, training);

    println!("Per-condition summary:");
    println!(
        "  {:<22} {:>12} {:>12} {:>12} {:>6} {:>7} {:>9} {:>10}",
        "agent", "served", "skipped", "switches", "cost", "avg_q", "total_q", "correct/n"
    );
    println!("  {}", "─".repeat(100));
    print_row("naive (reference)", &naive);
    print_row("smart baseline (no Nous)", &smart);
    print_row("nous-full pipeline", &nous);

    println!();
    print_novel_breakdown("smart baseline", &smart);
    print_novel_breakdown("nous-full", &nous);

    println!("\nPrimary metric (total_quality within budget):");
    let smart_q = smart.total_quality;
    let nous_q = nous.total_quality;
    let delta = nous_q - smart_q;
    if delta >= 1.0 {
        println!(
            "  ✓ Nous-full beats smart baseline by {:+.2} total quality.",
            delta
        );
        println!("    Signal compounding produces meaningful compound benefit on this workload.");
    } else if delta >= 0.2 {
        println!(
            "  ≈ Nous-full edges smart baseline by {:+.2} total quality —",
            delta
        );
        println!(
            "    real but narrow. Real benchmarks needed to confirm this holds beyond synthetic."
        );
    } else if delta.abs() <= 0.2 {
        println!(
            "  ≈ Nous-full matches smart baseline ({:+.2}) — signals don't compound",
            delta
        );
        println!(
            "    meaningfully on this workload. Allostatic claim is infrastructure-only here."
        );
    } else {
        println!(
            "  ⚠ Nous-full UNDERPERFORMS smart baseline by {:+.2} — investigate.",
            delta
        );
    }

    println!("\nSecondary metric (correct-strategy rate overall):");
    let smart_rate = if smart.queries_served > 0 {
        smart.correct_strategy_used as f64 / smart.queries_served as f64
    } else {
        0.0
    };
    let nous_rate = if nous.queries_served > 0 {
        nous.correct_strategy_used as f64 / nous.queries_served as f64
    } else {
        0.0
    };
    println!(
        "  smart={:.1}%  nous={:.1}%  delta={:+.1}%",
        smart_rate * 100.0,
        nous_rate * 100.0,
        (nous_rate - smart_rate) * 100.0
    );

    println!("\nNotes:");
    println!("  • Synthetic task — behavior illustration, not real-LLM validation.");
    println!("  • Smart baseline deliberately uses competent app-level tracking.");
    println!("    Nous's advantage must come from signal quality, not baseline weakness.");
    println!("  • Novel categories (Analyze, Summarize) test whether Nous adapts");
    println!("    without prior training — cold-session behavior within a stream.");
}
