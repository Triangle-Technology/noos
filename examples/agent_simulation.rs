//! Agent Simulation — fair three-way comparison.
//!
//! Run: `cargo run --example agent_simulation`
//!
//! Simulates three agents on the same numerical-debugging task:
//!
//! 1. **Naive** — uses one strategy, never adapts. Reference low bar showing
//!    what "zero adaptation" looks like.
//! 2. **Simple retry** — app-level quality-threshold rotation. No Nous.
//!    Pure `if quality < 0.5: next_strategy` logic. THIS IS THE FAIR BASELINE:
//!    what a reasonable application does without Nous.
//! 3. **Allostatic warm** — Nous CognitiveSession with imported LearnedState
//!    from a prior "training" session. Demonstrates cross-session value:
//!    learned recommendation from a prior session skips the discovery phase.
//!
//! ## What this demo is honest about
//!
//! Within a single cold session, Nous's signal-driven rotation reduces to the
//! same logic as (2). The allostatic claim is about persistence across
//! sessions — exported LearnedState lets a new session start pre-calibrated.
//! Without an export/import step, there is no cross-session story, and the
//! "allostatic agent" advantage disappears. This demo shows that explicitly.
//!
//! ## What this demo does NOT claim
//!
//! This is a synthetic task with a deterministic simulated LLM. Real-world
//! value requires task-level eval on benchmarks like LoCoMo or multi-turn
//! adaptation (see `docs/task-eval-design.md`). Treat this file as a
//! behavior illustration, not a measurement.

use nous_regulator::session::CognitiveSession;
use nous_regulator::types::world::{LearnedState, ResponseStrategy};

/// Simulated application strategies.
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
            Self::StepByStep => Self::DirectAnswer, // cycle
        }
    }

    /// Only StepByStep succeeds on numerical debugging.
    fn succeeds_on_numerical_debugging(self) -> bool {
        matches!(self, Self::StepByStep)
    }
}

/// Deterministic simulated LLM: quality is a function of strategy-task fit.
fn simulate_llm(strategy: AppStrategy) -> (String, f64) {
    match strategy {
        AppStrategy::DirectAnswer => (
            "The answer is 42.".to_string(),
            0.15, // Wrong — user wanted actual debugging.
        ),
        AppStrategy::AskClarifying => (
            "What exactly isn't working? What error do you see?".to_string(),
            0.30, // Partial — clarifying delays, user wants action.
        ),
        AppStrategy::StepByStep => (
            "1. First, check the loop bounds carefully.\n\
             2. Then, verify the index starts at the correct value.\n\
             3. Next, trace through a small example by hand.\n\
             4. Finally, add assertions to catch the off-by-one."
                .to_string(),
            0.90, // Right approach — debugging is step-driven.
        ),
    }
}

fn strategy_cost(strategy: AppStrategy) -> f64 {
    match strategy {
        AppStrategy::DirectAnswer => 0.1,
        AppStrategy::AskClarifying => 0.3,
        AppStrategy::StepByStep => 0.8,
    }
}

#[derive(Debug)]
struct AgentResult {
    strategies_tried: Vec<AppStrategy>,
    qualities: Vec<f64>,
    total_cost: f64,
    first_success_turn: Option<usize>,
}

impl AgentResult {
    fn avg_quality(&self) -> f64 {
        if self.qualities.is_empty() {
            0.0
        } else {
            self.qualities.iter().sum::<f64>() / self.qualities.len() as f64
        }
    }
}

// ─── Agent 1: Naive (reference — zero adaptation) ─────────────────────────

fn run_naive_agent(turns: usize) -> AgentResult {
    let mut qualities = Vec::new();
    let mut strategies_tried = Vec::new();
    let mut total_cost = 0.0;
    let mut first_success_turn: Option<usize> = None;
    let strategy = AppStrategy::DirectAnswer;

    for turn_num in 0..turns {
        strategies_tried.push(strategy);
        let (_response, quality) = simulate_llm(strategy);
        qualities.push(quality);
        total_cost += strategy_cost(strategy);

        if strategy.succeeds_on_numerical_debugging() && first_success_turn.is_none() {
            first_success_turn = Some(turn_num);
        }
    }

    AgentResult {
        strategies_tried,
        qualities,
        total_cost,
        first_success_turn,
    }
}

// ─── Agent 2: Simple retry (fair baseline — NO Nous) ──────────────────────
//
// What a minimally-competent application does without any Nous: on low
// quality, rotate to the next strategy. This is the bar Nous must beat to
// claim allostatic value.

const QUALITY_THRESHOLD_FOR_RETRY: f64 = 0.5;

fn run_simple_retry_agent(turns: usize) -> AgentResult {
    let mut qualities = Vec::new();
    let mut strategies_tried = Vec::new();
    let mut total_cost = 0.0;
    let mut first_success_turn: Option<usize> = None;
    let mut current_strategy = AppStrategy::DirectAnswer;
    let mut last_quality: Option<f64> = None;

    for turn_num in 0..turns {
        // App-level retry logic. No Nous.
        if let Some(q) = last_quality {
            if q < QUALITY_THRESHOLD_FOR_RETRY {
                current_strategy = current_strategy.next();
            }
        }

        strategies_tried.push(current_strategy);
        let (_response, quality) = simulate_llm(current_strategy);
        qualities.push(quality);
        total_cost += strategy_cost(current_strategy);
        last_quality = Some(quality);

        if current_strategy.succeeds_on_numerical_debugging() && first_success_turn.is_none() {
            first_success_turn = Some(turn_num);
        }
    }

    AgentResult {
        strategies_tried,
        qualities,
        total_cost,
        first_success_turn,
    }
}

// ─── Prior-session training (produces LearnedState for warm-start) ────────
//
// Simulates a previous session where the user asked similar questions and
// StepByStep succeeded. The app knows the task is numerical-debugging and
// chose StepByStep; Nous observed success and built reward learning. At the
// end, we export LearnedState for the next session to import.

fn train_prior_session() -> LearnedState {
    let mut session = CognitiveSession::new();

    // Same user query shape used in the eval below (matches cluster hash).
    // Multiple turns with StepByStep succeeding → reward learning accumulates.
    for _ in 0..10 {
        let _turn = session.process_message("Help me debug this numerical issue.");
        let (response, quality) = simulate_llm(AppStrategy::StepByStep);
        session.track_cost(strategy_cost(AppStrategy::StepByStep));
        session.process_response(&response, quality);
    }

    session.export_learned()
}

// ─── Agent 3: Allostatic warm-start (Nous with imported learning) ─────────
//
// This is where Nous's genuine cross-session value should appear: the prior
// session's exported LearnedState, imported into a new session, lets Nous
// recommend StepByStep on turn 1 instead of discovering it.

fn run_allostatic_warm_agent(turns: usize, prior_learned: LearnedState) -> AgentResult {
    let mut session = CognitiveSession::with_learned(prior_learned, 64);
    let mut qualities = Vec::new();
    let mut strategies_tried = Vec::new();
    let mut total_cost = 0.0;
    let mut first_success_turn: Option<usize> = None;
    let mut current_strategy = AppStrategy::DirectAnswer;
    let mut last_quality: Option<f64> = None;

    for turn_num in 0..turns {
        let turn = session.process_message("Help me debug this numerical issue.");

        // Prefer Nous's learned recommendation. Falls back to app-level retry
        // (same logic as simple_retry agent) when no recommendation yet.
        if let Some(recommended) = turn.signals.strategy {
            current_strategy = map_recommendation(recommended, current_strategy);
        } else if let Some(q) = last_quality {
            if q < QUALITY_THRESHOLD_FOR_RETRY {
                current_strategy = current_strategy.next();
            }
        }

        strategies_tried.push(current_strategy);
        let (response, quality) = simulate_llm(current_strategy);
        qualities.push(quality);
        total_cost += strategy_cost(current_strategy);
        last_quality = Some(quality);

        // Close the allostatic loop so learning continues this session too.
        session.track_cost(strategy_cost(current_strategy));
        session.process_response(&response, quality);

        if current_strategy.succeeds_on_numerical_debugging() && first_success_turn.is_none() {
            first_success_turn = Some(turn_num);
        }
    }

    AgentResult {
        strategies_tried,
        qualities,
        total_cost,
        first_success_turn,
    }
}

fn map_recommendation(r: ResponseStrategy, fallback: AppStrategy) -> AppStrategy {
    match r {
        ResponseStrategy::StepByStep => AppStrategy::StepByStep,
        ResponseStrategy::ClarifyFirst => AppStrategy::AskClarifying,
        ResponseStrategy::DirectAnswer => AppStrategy::DirectAnswer,
        _ => fallback,
    }
}

// ─── Main harness ─────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Agent Simulation — three-way comparison                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("Task: Numerical debugging (only StepByStep strategy succeeds).");
    println!("Each agent gets 10 attempts on the same persistent problem.\n");
    println!("This demo compares Nous against a FAIR baseline (simple retry)");
    println!("— not a straw man. Nous's claim to value rides on beating");
    println!("simple retry, not on beating zero-adaptation.\n");

    let turns = 10;

    // Train a prior session to give agent 3 something to import.
    let prior_learned = train_prior_session();
    println!(
        "Prior session exported LearnedState: {} strategy entries, tick={}\n",
        prior_learned.response_strategies.len(),
        prior_learned.tick
    );

    let naive = run_naive_agent(turns);
    let simple_retry = run_simple_retry_agent(turns);
    let allostatic_warm = run_allostatic_warm_agent(turns, prior_learned);

    print_agent_trace("Naive (reference)", &naive);
    println!();
    print_agent_trace("Simple retry (fair baseline, no Nous)", &simple_retry);
    println!();
    print_agent_trace("Allostatic warm (Nous + imported LearnedState)", &allostatic_warm);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!(
        "  {:<40} {:>14} {:>12} {:>10}",
        "Agent", "First success", "Avg quality", "Total cost"
    );
    println!("  {}", "─".repeat(80));
    print_summary_row("Naive", &naive);
    print_summary_row("Simple retry", &simple_retry);
    print_summary_row("Allostatic warm", &allostatic_warm);

    println!("\n  Interpretation:");

    // Compare allostatic to simple retry — the only fair comparison.
    let retry_success = simple_retry.first_success_turn;
    let allostatic_success = allostatic_warm.first_success_turn;

    match (retry_success, allostatic_success) {
        (Some(r), Some(a)) if a < r => println!(
            "  ✓ Allostatic (warm) succeeded on turn {}, simple retry took until turn {}.\n    \
               Cross-session learning shaved {} turns via imported recommendation.",
            a + 1,
            r + 1,
            r - a
        ),
        (Some(r), Some(a)) if a == r => println!(
            "  ≈ Allostatic (warm) matched simple retry (both turn {}).\n    \
               Imported LearnedState didn't shorten discovery on this run.",
            a + 1
        ),
        (Some(_), None) => println!(
            "  ⚠ Allostatic failed where simple retry succeeded — investigate."
        ),
        (None, Some(_)) => println!(
            "  ✓ Allostatic succeeded where simple retry did not."
        ),
        _ => println!("  Neither mechanism converged in 10 turns."),
    }

    let retry_avg = simple_retry.avg_quality();
    let allostatic_avg = allostatic_warm.avg_quality();
    if retry_avg > 0.0 {
        let ratio = allostatic_avg / retry_avg;
        if ratio > 1.1 {
            println!(
                "  ✓ Allostatic avg quality {:.2}x simple-retry baseline.",
                ratio
            );
        } else if ratio < 0.9 {
            println!(
                "  ⚠ Allostatic avg quality {:.2}x simple-retry — below baseline.",
                ratio
            );
        } else {
            println!(
                "  ≈ Allostatic avg quality {:.2}x simple-retry — within noise on this run.",
                ratio
            );
        }
    }

    println!("\n  Honesty notes:");
    println!("  • Comparing to 'naive' alone overstates Nous's value — naive does no");
    println!("    adaptation at all. Simple retry is the fair bar.");
    println!("  • Cold-session allostatic would behave roughly like simple retry.");
    println!("    Nous's allostatic value story is CROSS-session, not intra-session.");
    println!("  • This is a synthetic, deterministic task. Real validation needs");
    println!("    task-level eval on a real benchmark. See docs/task-eval-design.md.");
}

fn print_agent_trace(name: &str, result: &AgentResult) {
    println!("── {} ──", name);
    println!("  {:<4} {:<16} {:>8}", "Turn", "Strategy", "Quality");
    for (i, (s, q)) in result
        .strategies_tried
        .iter()
        .zip(result.qualities.iter())
        .enumerate()
    {
        let marker = if *q > 0.7 { " ✓" } else { "" };
        println!("  {:<4} {:<16} {:>8.2}{}", i + 1, format!("{:?}", s), q, marker);
    }
}

fn print_summary_row(name: &str, result: &AgentResult) {
    let success = match result.first_success_turn {
        Some(t) => format!("turn {}", t + 1),
        None => "never".to_string(),
    };
    println!(
        "  {:<40} {:>14} {:>12.3} {:>10.2}",
        name,
        success,
        result.avg_quality(),
        result.total_cost
    );
}
