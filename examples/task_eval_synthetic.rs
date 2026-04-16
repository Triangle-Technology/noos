//! Synthetic task-eval harness (Tier 1.1 from docs/task-eval-design.md).
//!
//! Run: `cargo run --example task_eval_synthetic`
//!
//! Extends the single-category `agent_simulation.rs` to a MULTI-CATEGORY,
//! MULTI-SEED comparison that reports mean AND standard deviation — the
//! statistical bar set by CR5 step 1.
//!
//! ## What this harness tests
//!
//! Claim: Nous's cross-session LearnedState lets warm-start agents skip the
//! discovery phase that simple-retry agents must repeat for each category.
//!
//! Setup:
//! - 3 task categories, each with one correct strategy.
//! - A training session pre-exposes Nous to all categories → exported LearnedState.
//! - Eval: K-turn sequence with category order shuffled by seed.
//! - Baseline: simple quality-threshold retry, no Nous.
//! - Nous warm: imports training LearnedState, uses `signals.strategy`.
//! - ≥3 seeds. Report per-seed and aggregated mean ± stddev.
//!
//! ## What this harness is (and isn't)
//!
//! - IS: a skeleton demonstrating the fair-comparison shape. Runs quickly,
//!   no external deps, no candle.
//! - IS NOT: a validation of Nous. The task is synthetic and deterministic;
//!   the simulated LLM returns hand-authored qualities. A positive result
//!   here shows the harness works; a negative result shows Nous's
//!   cross-session story fails even in the favorable synthetic case.
//! - For real validation, Tier 2 benchmarks (LoCoMo, MetaMedQA) apply.
//!
//! ## Interpreting results
//!
//! See `docs/task-eval-design.md` §5 for what would count as "claim validated".
//! Headline: Nous warm must beat simple retry by ≥2 stddev on time-to-first-correct
//! to be anything more than noise on this synthetic task.

use nous_regulator::session::CognitiveSession;
use nous_regulator::types::world::{LearnedState, ResponseStrategy};
use std::collections::BTreeMap;

// ─── Task categories and strategies ───────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Category {
    NumericalDebug,
    QuickLookup,
    AmbiguousRequest,
}

impl Category {
    fn all() -> [Category; 3] {
        [
            Category::NumericalDebug,
            Category::QuickLookup,
            Category::AmbiguousRequest,
        ]
    }

    fn user_query(self) -> &'static str {
        match self {
            Category::NumericalDebug => "Help me debug this numerical issue.",
            Category::QuickLookup => "What is the default port for Postgres?",
            Category::AmbiguousRequest => "Make this better somehow.",
        }
    }

    fn correct_strategy(self) -> AppStrategy {
        match self {
            Category::NumericalDebug => AppStrategy::StepByStep,
            Category::QuickLookup => AppStrategy::DirectAnswer,
            Category::AmbiguousRequest => AppStrategy::AskClarifying,
        }
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

/// Deterministic simulated LLM. Quality depends on (strategy, category) match.
fn simulate_llm(strategy: AppStrategy, category: Category) -> (String, f64) {
    let correct = category.correct_strategy();
    let quality = if strategy == correct {
        0.90 // right approach
    } else {
        match (strategy, correct) {
            // Near-misses yield partial quality.
            (AppStrategy::AskClarifying, _) => 0.30,
            (AppStrategy::StepByStep, _) => 0.25,
            (AppStrategy::DirectAnswer, _) => 0.15,
        }
    };
    // Response formats matter: Nous's `detect_response_strategy` uses regex
    // anchors that require specific shapes (multi-line numbered for StepByStep,
    // ≥2 question marks for ClarifyFirst). These responses are chosen to
    // unambiguously trigger the intended detection — a REAL app running against
    // real LLMs should prompt its model toward these formats when it wants
    // Nous to recognize the strategy correctly. See `detect_response_strategy`
    // in src/cognition/detector.rs for exact patterns.
    let text = match strategy {
        AppStrategy::DirectAnswer => "Short answer.".to_string(),
        AppStrategy::AskClarifying => {
            "What exactly isn't working? What error do you see?".to_string()
        }
        AppStrategy::StepByStep => "1. First, check the bounds.\n\
             2. Then, verify the index.\n\
             3. Next, trace a small example.\n\
             4. Finally, add assertions."
            .to_string(),
    };
    (text, quality)
}

fn strategy_cost(s: AppStrategy) -> f64 {
    match s {
        AppStrategy::DirectAnswer => 0.1,
        AppStrategy::AskClarifying => 0.3,
        AppStrategy::StepByStep => 0.8,
    }
}

// ─── Task sequence generation (deterministic from seed) ───────────────────

/// xorshift64 for reproducible, dependency-free sequence shuffling.
fn rand_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a shuffled K-turn sequence drawing from all categories roughly evenly.
fn generate_sequence(seed: u64, turns: usize) -> Vec<Category> {
    let cats = Category::all();
    let mut seq: Vec<Category> = (0..turns).map(|i| cats[i % cats.len()]).collect();
    // Fisher-Yates shuffle with seeded xorshift.
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    for i in (1..seq.len()).rev() {
        let j = (rand_u64(&mut state) as usize) % (i + 1);
        seq.swap(i, j);
    }
    seq
}

// ─── Metrics ──────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
struct EvalMetrics {
    avg_quality: f64,
    total_cost: f64,
    /// Per-category: turn index (within this run) of first correct choice. Missing = never.
    time_to_first_correct: BTreeMap<Category, Option<usize>>,
}

impl EvalMetrics {
    /// Mean turn-index of first correct strategy across categories. Lower is better.
    /// Categories that never saw the correct strategy count as `penalty` (typically 2*n_turns)
    /// so stddev stays finite while still ranking "never" worse than any real turn index.
    fn first_correct_summary(&self, penalty: f64) -> f64 {
        let vals: Vec<f64> = self
            .time_to_first_correct
            .values()
            .map(|o| o.map(|t| t as f64).unwrap_or(penalty))
            .collect();
        if vals.is_empty() {
            penalty
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
    }
}

fn record_correct(
    metrics: &mut EvalMetrics,
    cat: Category,
    strategy: AppStrategy,
    turn_idx: usize,
) {
    let entry = metrics.time_to_first_correct.entry(cat).or_insert(None);
    if entry.is_none() && strategy == cat.correct_strategy() {
        *entry = Some(turn_idx);
    }
}

// ─── Baseline: simple-retry, no Nous ──────────────────────────────────────

const QUALITY_THRESHOLD_FOR_RETRY: f64 = 0.5;

/// Per-category strategy memory at application level (no Nous).
/// Models a plain app that remembers "what worked last time" per category,
/// which is a stronger baseline than a single strategy for all.
fn run_baseline(sequence: &[Category]) -> EvalMetrics {
    let mut metrics = EvalMetrics::default();
    let mut per_category: BTreeMap<Category, AppStrategy> = BTreeMap::new();
    let mut qualities = Vec::new();

    for (turn_idx, &category) in sequence.iter().enumerate() {
        // Start with app-memory choice, else DirectAnswer. Rotate on prior low quality.
        let entry = per_category
            .entry(category)
            .or_insert(AppStrategy::DirectAnswer);
        let strategy = *entry;

        let (_resp, quality) = simulate_llm(strategy, category);
        qualities.push(quality);
        metrics.total_cost += strategy_cost(strategy);
        record_correct(&mut metrics, category, strategy, turn_idx);

        // App decides next attempt on this category: if quality was low, rotate.
        if quality < QUALITY_THRESHOLD_FOR_RETRY {
            *per_category.get_mut(&category).expect("entry inserted") = strategy.next();
        }
    }

    metrics.avg_quality = qualities.iter().sum::<f64>() / qualities.len() as f64;
    for cat in Category::all() {
        metrics.time_to_first_correct.entry(cat).or_insert(None);
    }
    metrics
}

// ─── Nous warm-start: imports LearnedState from training ─────────────────

fn train_prior_session() -> LearnedState {
    let mut session = CognitiveSession::new();
    // Train by seeing correct strategy for every category, multiple turns each.
    for _round in 0..6 {
        for cat in Category::all() {
            let _ = session.process_message(cat.user_query());
            let (resp, quality) = simulate_llm(cat.correct_strategy(), cat);
            session.track_cost(strategy_cost(cat.correct_strategy()));
            session.process_response(&resp, quality);
        }
    }
    session.export_learned()
}

fn map_rec(r: ResponseStrategy, fallback: AppStrategy) -> AppStrategy {
    match r {
        ResponseStrategy::StepByStep => AppStrategy::StepByStep,
        ResponseStrategy::ClarifyFirst => AppStrategy::AskClarifying,
        ResponseStrategy::DirectAnswer => AppStrategy::DirectAnswer,
        _ => fallback,
    }
}

fn run_nous_warm(sequence: &[Category], training: LearnedState) -> EvalMetrics {
    let mut session = CognitiveSession::with_learned(training, 64);
    let mut metrics = EvalMetrics::default();
    let mut qualities = Vec::new();
    let mut per_category_last_quality: BTreeMap<Category, f64> = BTreeMap::new();
    let mut per_category_choice: BTreeMap<Category, AppStrategy> = BTreeMap::new();

    for (turn_idx, &category) in sequence.iter().enumerate() {
        let turn = session.process_message(category.user_query());

        let current = per_category_choice
            .entry(category)
            .or_insert(AppStrategy::DirectAnswer);

        // Prefer Nous's learned recommendation; fall back to per-category retry.
        let strategy = if let Some(rec) = turn.signals.strategy {
            map_rec(rec, *current)
        } else if let Some(&q) = per_category_last_quality.get(&category) {
            if q < QUALITY_THRESHOLD_FOR_RETRY {
                current.next()
            } else {
                *current
            }
        } else {
            *current
        };
        *current = strategy;

        let (resp, quality) = simulate_llm(strategy, category);
        qualities.push(quality);
        metrics.total_cost += strategy_cost(strategy);
        per_category_last_quality.insert(category, quality);
        record_correct(&mut metrics, category, strategy, turn_idx);

        // Close the allostatic loop so warm session keeps refining.
        session.track_cost(strategy_cost(strategy));
        session.process_response(&resp, quality);
    }

    metrics.avg_quality = qualities.iter().sum::<f64>() / qualities.len() as f64;
    for cat in Category::all() {
        metrics.time_to_first_correct.entry(cat).or_insert(None);
    }
    metrics
}

// ─── Aggregation & reporting (mean + stddev) ──────────────────────────────

#[derive(Debug)]
struct AggregatedMetrics {
    avg_quality_mean: f64,
    avg_quality_std: f64,
    total_cost_mean: f64,
    total_cost_std: f64,
    first_correct_mean: f64,
    first_correct_std: f64,
}

fn aggregate(runs: &[EvalMetrics], penalty: f64) -> AggregatedMetrics {
    let n = runs.len() as f64;
    let mean_of = |f: &dyn Fn(&EvalMetrics) -> f64| runs.iter().map(f).sum::<f64>() / n;
    let std_of = |mean: f64, f: &dyn Fn(&EvalMetrics) -> f64| {
        let var = runs
            .iter()
            .map(|r| {
                let d = f(r) - mean;
                d * d
            })
            .sum::<f64>()
            / n.max(1.0);
        var.sqrt()
    };
    let q: &dyn Fn(&EvalMetrics) -> f64 = &|r| r.avg_quality;
    let c: &dyn Fn(&EvalMetrics) -> f64 = &|r| r.total_cost;
    let f: &dyn Fn(&EvalMetrics) -> f64 = &|r| r.first_correct_summary(penalty);
    let quality_mean = mean_of(q);
    let cost_mean = mean_of(c);
    let first_mean = mean_of(f);
    AggregatedMetrics {
        avg_quality_mean: quality_mean,
        avg_quality_std: std_of(quality_mean, q),
        total_cost_mean: cost_mean,
        total_cost_std: std_of(cost_mean, c),
        first_correct_mean: first_mean,
        first_correct_std: std_of(first_mean, f),
    }
}

fn print_aggregated(label: &str, a: &AggregatedMetrics) {
    println!(
        "  {:<28} quality={:.3}±{:.3}   cost={:.2}±{:.2}   first_correct={:.2}±{:.2}",
        label,
        a.avg_quality_mean,
        a.avg_quality_std,
        a.total_cost_mean,
        a.total_cost_std,
        a.first_correct_mean,
        a.first_correct_std
    );
}

// ─── Main ─────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  task_eval_synthetic — multi-category, multi-seed harness   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("Tier 1.1 from docs/task-eval-design.md.");
    println!("Synthetic task — skeleton, not a validation. See doc §2 bar.\n");

    let turns = 12; // 4 per category in expectation
    let seeds: &[u64] = &[42, 7, 1337]; // ≥3 seeds (CR5 step 1)
    // Penalty for "never saw correct strategy this run": 2× turns, clearly worse
    // than any real turn index while keeping stddev finite.
    let never_penalty: f64 = (turns * 2) as f64;

    let training = train_prior_session();
    println!(
        "Training exported LearnedState: {} strategy entries, tick={}",
        training.response_strategies.len(),
        training.tick
    );
    println!("Eval: {} turns per run × {} seeds\n", turns, seeds.len());

    let mut baseline_runs = Vec::new();
    let mut nous_runs = Vec::new();

    println!("Per-seed results:");
    println!(
        "  {:<8} {:<28} {:>8} {:>8} {:>14}",
        "seed", "condition", "quality", "cost", "first_correct"
    );
    println!("  {}", "─".repeat(72));

    for &seed in seeds {
        let sequence = generate_sequence(seed, turns);

        let baseline = run_baseline(&sequence);
        let nous = run_nous_warm(&sequence, training.clone());

        println!(
            "  {:<8} {:<28} {:>8.3} {:>8.2} {:>14.2}",
            seed,
            "baseline (simple retry)",
            baseline.avg_quality,
            baseline.total_cost,
            baseline.first_correct_summary(never_penalty)
        );
        println!(
            "  {:<8} {:<28} {:>8.3} {:>8.2} {:>14.2}",
            seed,
            "nous warm",
            nous.avg_quality,
            nous.total_cost,
            nous.first_correct_summary(never_penalty)
        );

        baseline_runs.push(baseline);
        nous_runs.push(nous);
    }

    let baseline_agg = aggregate(&baseline_runs, never_penalty);
    let nous_agg = aggregate(&nous_runs, never_penalty);

    println!("\nAggregated (mean ± stddev):");
    print_aggregated("baseline (simple retry)", &baseline_agg);
    print_aggregated("nous warm", &nous_agg);

    // Report comparison with the 2-stddev bar from task-eval-design.md §5.
    println!("\n2-stddev bar check (Nous must exceed baseline by ≥2 baseline stddev):");
    let quality_delta = nous_agg.avg_quality_mean - baseline_agg.avg_quality_mean;
    let first_correct_delta = baseline_agg.first_correct_mean - nous_agg.first_correct_mean;

    let quality_two_std = 2.0 * baseline_agg.avg_quality_std;
    let first_correct_two_std = 2.0 * baseline_agg.first_correct_std;

    println!(
        "  avg_quality:   delta = {:+.3}   threshold (2σ of baseline) = {:.3}  -> {}",
        quality_delta,
        quality_two_std,
        if quality_delta > quality_two_std {
            "PASS"
        } else {
            "NOT VALIDATED"
        }
    );
    println!(
        "  first_correct: delta = {:+.2}   threshold (2σ of baseline) = {:.2}  -> {}",
        first_correct_delta,
        first_correct_two_std,
        if first_correct_delta > first_correct_two_std {
            "PASS"
        } else {
            "NOT VALIDATED"
        }
    );

    println!("\nNotes:");
    println!("  - This is a synthetic task. PASS here = harness behaves, not 'Nous works'.");
    println!("  - 3 seeds is the MINIMUM per CR5. Real claims want more.");
    println!("  - Tier 2 benchmarks (LoCoMo, MetaMedQA) are what publishable claims require.");
    println!("  - If Nous FAILS on this synthetic favorable case, the cross-session");
    println!("    claim is not ready — don't ship the claim.");
}
