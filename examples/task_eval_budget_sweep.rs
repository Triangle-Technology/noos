//! Budget sweep — characterizes the Pareto envelope where Nous-full beats
//! the smart baseline (vs where it doesn't).
//!
//! Run: `cargo run --example task_eval_budget_sweep`
//!
//! Tier 1.3 (`task_eval_multi_signal.rs`) showed Nous-full beats smart
//! baseline by +2.73 total quality at budget=12.0, but that Nous costs more
//! per quality unit (Pareto trade-off). This sweep tests:
//!
//! - At what budget does Nous's quality advantage actually flip?
//! - In which regime should apps use Nous-full vs simple cost-tracking?
//!
//! Same 24-query mixed workload as Tier 1.3. Three agents, varying the
//! budget cap from tight (4.0) to generous (20.0). Reports total_quality
//! and quality-per-cost for each agent at each budget.
//!
//! ## What "winning" means at each budget
//!
//! - Tight budget: agents must skip queries. The one that picks the right
//!   strategy on the queries it CAN serve wins on total_quality.
//! - Loose budget: agents serve all queries. Quality differences come from
//!   strategy selection and mode choice.
//! - Mid budget: tension between conservation (preserve budget for later)
//!   and quality (use full mode now). The interesting regime.

use nous_regulator::session::CognitiveSession;
use nous_regulator::types::world::{LearnedState, ResponseStrategy};
use std::collections::BTreeMap;

// ─── Categories (same as multi_signal) ────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Category {
    Debug,
    Lookup,
    Clarify,
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

fn simulate_llm(strategy: AppStrategy, category: Category, mode: Mode) -> (String, f64, f64) {
    let correct = category.correct_strategy();
    let quality_full = if strategy == correct {
        0.90
    } else {
        match strategy {
            AppStrategy::AskClarifying => 0.35,
            AppStrategy::StepByStep => 0.30,
            AppStrategy::DirectAnswer => 0.20,
        }
    };
    let (quality, cost) = match mode {
        Mode::Full => (
            quality_full,
            match strategy {
                AppStrategy::DirectAnswer => 0.2,
                AppStrategy::AskClarifying => 0.35,
                AppStrategy::StepByStep => 0.70,
            },
        ),
        Mode::Shallow => (quality_full * 0.55, 0.15),
    };

    let text = match (category, strategy) {
        (Category::Debug, AppStrategy::StepByStep) => {
            "1. First, check the loop bounds.\n\
             2. Then, verify the index.\n\
             3. Next, trace a small example.\n\
             4. Finally, add assertions."
                .to_string()
        }
        (_, AppStrategy::AskClarifying) => {
            "What exactly do you need? What have you tried so far?".to_string()
        }
        (_, AppStrategy::DirectAnswer) => format!("Short answer for {:?}.", category),
        (_, AppStrategy::StepByStep) => {
            "1. First step.\n2. Then more.\n3. Finally done.".to_string()
        }
    };
    (text, quality, cost)
}

fn generate_stream() -> Vec<(Category, String)> {
    let pattern = [
        Category::Debug, Category::Lookup, Category::Analyze, Category::Clarify, Category::Summarize,
        Category::Debug, Category::Lookup, Category::Analyze, Category::Clarify, Category::Summarize,
        Category::Debug, Category::Lookup, Category::Analyze, Category::Clarify, Category::Summarize,
        Category::Debug, Category::Lookup, Category::Analyze, Category::Clarify, Category::Summarize,
        Category::Debug, Category::Lookup, Category::Analyze, Category::Summarize,
    ];
    pattern.iter().enumerate().map(|(i, &c)| (c, c.user_query(i))).collect()
}

fn train_prior_session() -> LearnedState {
    let mut session = CognitiveSession::new();
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

// ─── Agents (parameterized over budget) ───────────────────────────────────

#[derive(Debug, Default, Clone)]
struct RunResult {
    served: usize,
    skipped: usize,
    cost: f64,
    quality: f64,
    switches: usize,
}

impl RunResult {
    fn quality_per_cost(&self) -> f64 {
        if self.cost > 1e-9 {
            self.quality / self.cost
        } else {
            0.0
        }
    }
}

const QUALITY_RETRY_THRESHOLD: f64 = 0.5;
const COST_CONSERVATION_FRACTION: f64 = 0.5;
const NOUS_CONSERVATION_THRESHOLD: f64 = 0.2;

fn run_smart_baseline(stream: &[(Category, String)], budget: f64) -> RunResult {
    let mut r = RunResult::default();
    let mut per_cat_strat: BTreeMap<Category, AppStrategy> = BTreeMap::new();
    let mut per_cat_q: BTreeMap<Category, f64> = BTreeMap::new();
    for (cat, _) in stream {
        let cur = per_cat_strat.entry(*cat).or_insert(AppStrategy::DirectAnswer);
        if let Some(&q) = per_cat_q.get(cat) {
            if q < QUALITY_RETRY_THRESHOLD {
                *cur = cur.next();
            }
        }
        let strategy = *cur;
        let mode = if r.cost >= budget * COST_CONSERVATION_FRACTION {
            if r.switches == 0 {
                r.switches = 1;
            }
            Mode::Shallow
        } else {
            Mode::Full
        };
        let (_resp, q, c) = simulate_llm(strategy, *cat, mode);
        if r.cost + c > budget {
            r.skipped += 1;
            continue;
        }
        per_cat_q.insert(*cat, q);
        r.cost += c;
        r.quality += q;
        r.served += 1;
    }
    r
}

fn map_rec(r: ResponseStrategy, fallback: AppStrategy) -> AppStrategy {
    match r {
        ResponseStrategy::StepByStep => AppStrategy::StepByStep,
        ResponseStrategy::ClarifyFirst => AppStrategy::AskClarifying,
        ResponseStrategy::DirectAnswer => AppStrategy::DirectAnswer,
        _ => fallback,
    }
}

fn run_nous_full(stream: &[(Category, String)], budget: f64, training: LearnedState) -> RunResult {
    let mut session = CognitiveSession::with_learned(training, 64);
    let mut r = RunResult::default();
    let mut per_cat_strat: BTreeMap<Category, AppStrategy> = BTreeMap::new();
    let mut per_cat_q: BTreeMap<Category, f64> = BTreeMap::new();
    let mut in_shallow = false;
    for (cat, text) in stream {
        let turn = session.process_message(text);
        if !in_shallow && turn.signals.conservation > NOUS_CONSERVATION_THRESHOLD {
            in_shallow = true;
            r.switches += 1;
        }
        let mode = if in_shallow { Mode::Shallow } else { Mode::Full };

        let cur = per_cat_strat.entry(*cat).or_insert(AppStrategy::DirectAnswer);
        let strategy = if let Some(rec) = turn.signals.strategy {
            map_rec(rec, *cur)
        } else if let Some(&q) = per_cat_q.get(cat) {
            if q < QUALITY_RETRY_THRESHOLD {
                cur.next()
            } else {
                *cur
            }
        } else {
            *cur
        };
        *cur = strategy;

        let (resp, q, c) = simulate_llm(strategy, *cat, mode);
        if r.cost + c > budget {
            r.skipped += 1;
            session.track_cost(0.0);
            continue;
        }
        per_cat_q.insert(*cat, q);
        r.cost += c;
        r.quality += q;
        r.served += 1;
        session.track_cost(c);
        session.process_response(&resp, q);
    }
    r
}

// ─── Sweep ────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  task_eval_budget_sweep — Pareto envelope characterization   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("Same 24-query mixed workload as Tier 1.3, varied budget cap.");
    println!("Compare smart-baseline (no Nous) vs nous-full at each budget.\n");

    let stream = generate_stream();
    let training = train_prior_session();

    let budgets: &[f64] = &[4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0];

    println!(
        "  {:>6} | {:<30} | {:<30} | {:>9}",
        "budget", "smart baseline", "nous-full", "delta_q"
    );
    println!(
        "  {:>6} | {:>5} {:>4} {:>5} {:>6} {:>5} | {:>5} {:>4} {:>5} {:>6} {:>5} | {:>9}",
        "", "served", "skip", "cost", "tot_q", "q/c", "served", "skip", "cost", "tot_q", "q/c", ""
    );
    println!("  {}", "─".repeat(110));

    let mut crossover: Option<f64> = None;
    let mut prev_winner: Option<&'static str> = None;

    for &budget in budgets {
        let smart = run_smart_baseline(&stream, budget);
        let nous = run_nous_full(&stream, budget, training.clone());

        let delta_q = nous.quality - smart.quality;
        let winner = if delta_q > 0.05 {
            "nous"
        } else if delta_q < -0.05 {
            "smart"
        } else {
            "tie"
        };

        if let Some(prev) = prev_winner {
            if prev != winner && crossover.is_none() && winner != "tie" && prev != "tie" {
                crossover = Some(budget);
            }
        }
        prev_winner = Some(winner);

        let marker = if delta_q > 0.5 {
            "+++"
        } else if delta_q > 0.05 {
            "+"
        } else if delta_q < -0.5 {
            "---"
        } else if delta_q < -0.05 {
            "-"
        } else {
            "~"
        };

        println!(
            "  {:>6.1} | {:>5} {:>4} {:>5.2} {:>6.2} {:>5.2} | {:>5} {:>4} {:>5.2} {:>6.2} {:>5.2} | {:>+6.2} {}",
            budget,
            smart.served, smart.skipped, smart.cost, smart.quality, smart.quality_per_cost(),
            nous.served, nous.skipped, nous.cost, nous.quality, nous.quality_per_cost(),
            delta_q, marker
        );
    }

    println!("\nLegend:  +++ Nous wins big (>0.5)  + Nous edge  ~ tie  - smart edge  --- smart wins big");
    println!();
    match crossover {
        Some(b) => println!(
            "  Crossover detected near budget = {:.1} (winner flipped). \n  \
              Apps with budget below this should consider smart baseline; above, Nous-full.",
            b
        ),
        None => println!(
            "  No crossover detected in tested range. One agent dominates throughout —\n  \
              read the table to see which and by how much."
        ),
    }

    println!("\nNotes:");
    println!("  • Synthetic task — illustrates regime sensitivity, not absolute calibration.");
    println!("  • Per-cost efficiency (q/c) often favors smart baseline because it");
    println!("    conserves earlier; per-query absolute quality often favors Nous because");
    println!("    it stays in full mode while there's budget.");
    println!("  • At very tight budgets, both agents skip queries — winner determined by");
    println!("    which strategies they pick on the queries they CAN serve.");
}
