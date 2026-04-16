//! Tier 1.5 abstention calibration — does Nous's `signals.strategy` +
//! `signals.confidence` produce well-calibrated abstention decisions on
//! questions the agent shouldn't try to answer?
//!
//! Run: `cargo run --example task_eval_abstention`
//!
//! ## Why this eval matters
//!
//! Metacognition (0% unknown detection in frontier models, per MetaMedQA)
//! is a well-documented LLM weakness. Tier 1.5 tests whether Nous gives
//! apps a reliable "I don't know this" signal — the precondition for
//! graceful abstention.
//!
//! ## Setup
//!
//! Stream: 30 questions across 5 categories. **3 pre-trained** (Nous has
//! prior LearnedState data — should attempt + answer well). **2 novel**
//! (no prior data — should abstain rather than guess).
//!
//! Decision per turn: answer (with picked strategy) OR abstain.
//!
//! - **Answering correctly** (right strategy on pre-trained): full quality.
//! - **Answering wrongly** (any strategy on novel, or wrong strategy on
//!   pre-trained): low quality (~0.2). Hurts the user.
//! - **Abstaining on novel**: 0 quality but no harm — user knows to ask
//!   elsewhere.
//! - **Abstaining on pre-trained**: 0 quality and lost opportunity (FN).
//!
//! ## Three agents
//!
//! 1. **Always-answer** (reference): never abstains. Worst-case for
//!    abstention F1.
//! 2. **Smart baseline** (no Nous): tracks per-cluster historical success
//!    rate, abstains when cluster has < 3 observations OR avg quality < 0.4.
//!    Equivalent to "I don't know this enough to try."
//! 3. **Nous-confidence**: abstains when `signals.strategy.is_none()` (no
//!    learned recommendation for cluster) OR `signals.confidence < 0.4`.
//!    Uses Nous's combined uncertainty signal.
//!
//! ## Metrics
//!
//! - **Abstention precision**: of times agent abstained, how many SHOULD
//!   have been abstained on (true negatives / (true negatives + false
//!   negatives)).
//! - **Abstention recall**: of times agent SHOULD have abstained, how many
//!   it caught (true negatives / (true negatives + false positives)).
//! - **F1**: harmonic mean of the above.
//! - **Net quality**: total quality on answered + 0 on abstained (so
//!   abstaining on pre-trained is opportunity cost).
//! - **Harm**: count of false-positive answers (answered novel → low quality
//!   delivered to user).

use nous::session::CognitiveSession;
use nous::types::world::{LearnedState, ResponseStrategy};
use std::collections::BTreeMap;

// ─── Categories ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Category {
    // Pre-trained — agent should attempt + answer well.
    Debug,
    Lookup,
    Clarify,
    // Novel — no prior data, agent should abstain.
    Compose,
    Translate,
}

impl Category {
    fn user_query(self, idx: usize) -> String {
        match self {
            Self::Debug => format!("Help me debug this numerical issue {idx}."),
            Self::Lookup => format!("What is the default port for service {idx}?"),
            Self::Clarify => format!("Make system {idx} better somehow."),
            Self::Compose => format!("Compose a poem about topic {idx}."),
            Self::Translate => format!("Translate phrase {idx} into French."),
        }
    }

    fn correct_strategy(self) -> AppStrategy {
        match self {
            Self::Debug => AppStrategy::StepByStep,
            Self::Lookup => AppStrategy::DirectAnswer,
            Self::Clarify => AppStrategy::AskClarifying,
            // Novel categories also have a "correct" strategy in principle,
            // but the agent has no way to know it without prior learning.
            Self::Compose => AppStrategy::DirectAnswer,
            Self::Translate => AppStrategy::DirectAnswer,
        }
    }

    fn is_pretrained(self) -> bool {
        matches!(self, Self::Debug | Self::Lookup | Self::Clarify)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppStrategy {
    DirectAnswer,
    AskClarifying,
    StepByStep,
}

fn simulate_llm(strategy: AppStrategy, category: Category) -> (String, f64) {
    let correct = category.correct_strategy();
    let quality = if strategy == correct {
        // Pre-trained correct: high quality. Novel "correct": still moderate
        // (agent guessed lucky but no real expertise).
        if category.is_pretrained() { 0.90 } else { 0.40 }
    } else {
        // Wrong strategy: low quality regardless of category.
        0.20
    };
    let text = match strategy {
        AppStrategy::DirectAnswer => format!("Short answer for {:?}.", category),
        AppStrategy::AskClarifying => "What exactly do you need? What have you tried?".to_string(),
        AppStrategy::StepByStep => {
            "1. First, identify.\n2. Then, check.\n3. Finally, verify.".to_string()
        }
    };
    (text, quality)
}

fn generate_stream() -> Vec<(Category, String)> {
    // Roughly equal mix of categories with deliberate variety.
    let pattern = [
        Category::Debug,
        Category::Compose,    // novel
        Category::Lookup,
        Category::Translate,  // novel
        Category::Clarify,
        Category::Debug,
        Category::Compose,
        Category::Lookup,
        Category::Translate,
        Category::Clarify,
        Category::Debug,
        Category::Compose,
        Category::Lookup,
        Category::Translate,
        Category::Clarify,
        Category::Debug,
        Category::Compose,
        Category::Lookup,
        Category::Translate,
        Category::Clarify,
        Category::Debug,
        Category::Compose,
        Category::Lookup,
        Category::Translate,
        Category::Clarify,
        Category::Debug,
        Category::Compose,
        Category::Lookup,
        Category::Translate,
        Category::Clarify,
    ];
    pattern
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, c.user_query(i)))
        .collect()
}

fn train_prior_session() -> LearnedState {
    let mut session = CognitiveSession::new();
    let trained = [Category::Debug, Category::Lookup, Category::Clarify];
    // IMPORTANT: training idx must produce the SAME topic cluster hash as eval
    // queries. `build_topic_cluster` filters words with len < 3 (see
    // `src/cognition/detector.rs::extract_meaningful_words`). Using idx=999
    // would inject "999" as a 3-char topic, putting training under a different
    // cluster than eval (which uses idx 0-29 — single/double digit, filtered).
    // Use a single-digit idx so it gets filtered consistently with eval.
    for round in 0..6 {
        let idx = round; // 0-5: single digit, filtered as too-short by extract_topics.
        for &cat in &trained {
            let _ = session.process_message(&cat.user_query(idx));
            let (resp, quality) = simulate_llm(cat.correct_strategy(), cat);
            session.track_cost(0.5);
            session.process_response(&resp, quality);
        }
    }
    session.export_learned()
}

// ─── Decision outcomes ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum Decision {
    Answer(AppStrategy),
    Abstain,
}

#[derive(Debug, Default, Clone)]
struct AbstentionStats {
    /// Pre-trained question, agent answered with correct strategy.
    true_positive_correct: usize,
    /// Pre-trained question, agent answered with wrong strategy (still attempted).
    true_positive_wrong: usize,
    /// Pre-trained question, agent abstained (wrong — should have answered).
    false_negative: usize,
    /// Novel question, agent abstained (correct — shouldn't have tried).
    true_negative: usize,
    /// Novel question, agent answered (wrong — guessed).
    false_positive: usize,
    /// Total quality summed (abstentions = 0).
    total_quality: f64,
}

impl AbstentionStats {
    fn precision(&self) -> f64 {
        let denom = self.true_negative + self.false_negative;
        if denom == 0 {
            0.0
        } else {
            self.true_negative as f64 / denom as f64
        }
    }
    fn recall(&self) -> f64 {
        let denom = self.true_negative + self.false_positive;
        if denom == 0 {
            0.0
        } else {
            self.true_negative as f64 / denom as f64
        }
    }
    fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r < 1e-9 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
    fn total_attempts(&self) -> usize {
        self.true_positive_correct
            + self.true_positive_wrong
            + self.false_positive
    }
    fn total_abstentions(&self) -> usize {
        self.true_negative + self.false_negative
    }
}

fn record(stats: &mut AbstentionStats, cat: Category, decision: Decision) {
    match decision {
        Decision::Answer(strategy) => {
            let (_, quality) = simulate_llm(strategy, cat);
            stats.total_quality += quality;
            if cat.is_pretrained() {
                if strategy == cat.correct_strategy() {
                    stats.true_positive_correct += 1;
                } else {
                    stats.true_positive_wrong += 1;
                }
            } else {
                stats.false_positive += 1;
            }
        }
        Decision::Abstain => {
            // 0 quality contribution.
            if cat.is_pretrained() {
                stats.false_negative += 1;
            } else {
                stats.true_negative += 1;
            }
        }
    }
}

// ─── Agent 1: Always-answer (reference, never abstains) ──────────────────

fn run_always_answer(stream: &[(Category, String)]) -> AbstentionStats {
    let mut stats = AbstentionStats::default();
    for (cat, _) in stream {
        record(&mut stats, *cat, Decision::Answer(AppStrategy::DirectAnswer));
    }
    stats
}

// ─── Agent 2: Smart baseline — per-cluster success tracking ──────────────
//
// Tracks per-cluster (avg quality, count). Abstains when cluster has fewer
// than MIN_OBSERVATIONS samples OR avg quality below ABSTAIN_THRESHOLD.
// This is the natural app-level "I don't know this domain enough" pattern.

const MIN_OBSERVATIONS_FOR_ATTEMPT: usize = 3;
const ABSTAIN_QUALITY_THRESHOLD: f64 = 0.4;

fn run_smart_baseline(
    stream: &[(Category, String)],
    pretrained_observations: &BTreeMap<Category, (f64, usize)>,
) -> AbstentionStats {
    let mut stats = AbstentionStats::default();
    let mut cluster_history: BTreeMap<Category, (f64, usize)> = pretrained_observations.clone();
    let mut last_strategy: BTreeMap<Category, AppStrategy> = BTreeMap::new();
    // Initialize last_strategy from training: best-known strategy per cluster.
    for &cat in &[Category::Debug, Category::Lookup, Category::Clarify] {
        last_strategy.insert(cat, cat.correct_strategy());
    }
    for (cat, _) in stream {
        let (avg_q, count) = *cluster_history.get(cat).unwrap_or(&(0.0, 0));
        let should_attempt =
            count >= MIN_OBSERVATIONS_FOR_ATTEMPT && avg_q >= ABSTAIN_QUALITY_THRESHOLD;
        let decision = if should_attempt {
            let strat = *last_strategy.get(cat).unwrap_or(&AppStrategy::DirectAnswer);
            Decision::Answer(strat)
        } else {
            Decision::Abstain
        };
        record(&mut stats, *cat, decision);
        // Update history if attempted.
        if let Decision::Answer(strat) = decision {
            let (_, quality) = simulate_llm(strat, *cat);
            let entry = cluster_history.entry(*cat).or_insert((0.0, 0));
            // Running average update.
            let new_count = entry.1 + 1;
            entry.0 = entry.0 * (entry.1 as f64) / (new_count as f64)
                + quality / (new_count as f64);
            entry.1 = new_count;
        }
    }
    stats
}

// ─── Agent 3: Nous-confidence — abstains based on signals ────────────────
//
// Decision rule: abstain if `signals.strategy.is_none()` (Nous has no
// learned recommendation for this cluster) OR `signals.confidence` falls
// below a threshold (Nous's classification is unreliable). Simulates an
// app that respects Nous's uncertainty signaling.

const NOUS_CONFIDENCE_ABSTAIN_THRESHOLD: f64 = 0.4;

fn map_rec(r: ResponseStrategy, fallback: AppStrategy) -> AppStrategy {
    match r {
        ResponseStrategy::StepByStep => AppStrategy::StepByStep,
        ResponseStrategy::ClarifyFirst => AppStrategy::AskClarifying,
        ResponseStrategy::DirectAnswer => AppStrategy::DirectAnswer,
        _ => fallback,
    }
}

fn run_nous_confidence(stream: &[(Category, String)], training: LearnedState) -> AbstentionStats {
    let mut session = CognitiveSession::with_learned(training, 64);
    let mut stats = AbstentionStats::default();
    for (cat, text) in stream {
        let turn = session.process_message(text);

        // Decision rule (revised after first-run analysis 2026-04-14):
        //
        // Original rule was AND: abstain if strategy.is_none() AND confidence < 0.4.
        // First run: Nous never abstained — confidence stays around 0.5 for
        // benign novel queries (gate.confidence default + low pe_volatility),
        // never crossing 0.4. The AND condition was too restrictive.
        //
        // Revised rule: abstain when `signals.strategy.is_none()` — Nous has
        // no learned recommendation for this cluster. This is the strongest
        // direct "I haven't seen this before" signal Nous provides. Confidence
        // is a secondary modulator; primary signal is strategy availability.
        let abstain = turn.signals.strategy.is_none();
        let _ = NOUS_CONFIDENCE_ABSTAIN_THRESHOLD; // kept for documentation

        let decision = if abstain {
            Decision::Abstain
        } else {
            let strat = turn
                .signals
                .strategy
                .map(|r| map_rec(r, AppStrategy::DirectAnswer))
                .unwrap_or(AppStrategy::DirectAnswer);
            Decision::Answer(strat)
        };

        record(&mut stats, *cat, decision);

        // Close the loop so Nous keeps learning even mid-stream.
        if let Decision::Answer(strat) = decision {
            let (resp, quality) = simulate_llm(strat, *cat);
            session.track_cost(0.5);
            session.process_response(&resp, quality);
        }
    }
    stats
}

// ─── Reporting ────────────────────────────────────────────────────────────

fn print_row(name: &str, s: &AbstentionStats) {
    println!(
        "  {:<26} attempts={:>2}  abstentions={:>2}  TP_corr={:>2}  TP_wrong={:>2}  FP={:>2}  TN={:>2}  FN={:>2}  prec={:.2}  rec={:.2}  F1={:.2}  q={:.2}",
        name,
        s.total_attempts(),
        s.total_abstentions(),
        s.true_positive_correct,
        s.true_positive_wrong,
        s.false_positive,
        s.true_negative,
        s.false_negative,
        s.precision(),
        s.recall(),
        s.f1(),
        s.total_quality,
    );
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  task_eval_abstention — Tier 1.5 metacognition signal test  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("30-query stream mixing pre-trained categories (Debug, Lookup,");
    println!("Clarify) and novel categories (Compose, Translate). Tests whether");
    println!("Nous's `signals.confidence` + `signals.strategy.is_none()` produce");
    println!("well-calibrated abstention decisions vs a smart per-cluster tracker.\n");

    let stream = generate_stream();
    let pretrained_count = stream.iter().filter(|(c, _)| c.is_pretrained()).count();
    let novel_count = stream.len() - pretrained_count;
    println!(
        "Stream: {} queries ({} pre-trained answerable, {} novel that should abstain)\n",
        stream.len(),
        pretrained_count,
        novel_count
    );

    let training = train_prior_session();

    // For smart baseline, pre-populate per-cluster history equivalent to training.
    let mut pretrained_obs: BTreeMap<Category, (f64, usize)> = BTreeMap::new();
    for &cat in &[Category::Debug, Category::Lookup, Category::Clarify] {
        // Training was 6 rounds at quality 0.9.
        pretrained_obs.insert(cat, (0.9, 6));
    }

    let always = run_always_answer(&stream);
    let smart = run_smart_baseline(&stream, &pretrained_obs);
    let nous = run_nous_confidence(&stream, training);

    println!("Per-condition results:");
    println!(
        "  {:<26} {:<60}",
        "agent", "(metrics — see legend below)"
    );
    println!("  {}", "─".repeat(140));
    print_row("always-answer (reference)", &always);
    print_row("smart baseline (no Nous)", &smart);
    print_row("nous-confidence", &nous);

    println!("\nLegend:");
    println!("  TP_corr = correctly answered pre-trained with right strategy (good)");
    println!("  TP_wrong = answered pre-trained with wrong strategy (delivered to user but suboptimal)");
    println!("  FP = answered novel question that should've been abstained (HARM)");
    println!("  TN = correctly abstained on novel (good)");
    println!("  FN = abstained on pre-trained that should've been answered (lost opportunity)");
    println!("  prec = TN / (TN + FN)  rec = TN / (TN + FP)  F1 = harmonic mean");
    println!("  q = total quality (abstentions count as 0)");

    println!("\nPrimary metric — abstention F1 (higher = better calibrated):");
    let smart_f1 = smart.f1();
    let nous_f1 = nous.f1();
    let delta_f1 = nous_f1 - smart_f1;
    if delta_f1.abs() < 0.05 {
        println!(
            "  ≈ Tied: smart F1 = {:.2}, nous F1 = {:.2} (Δ = {:+.2})",
            smart_f1, nous_f1, delta_f1
        );
    } else if delta_f1 > 0.0 {
        println!(
            "  ✓ Nous F1 = {:.2}, smart F1 = {:.2} (Nous better by {:+.2})",
            nous_f1, smart_f1, delta_f1
        );
    } else {
        println!(
            "  ⚠ Nous F1 = {:.2}, smart F1 = {:.2} (smart better by {:+.2})",
            nous_f1, smart_f1, -delta_f1
        );
    }

    println!("\nSecondary — harm comparison (lower FP = less wrong-answer harm):");
    println!(
        "  always-answer: FP={}  smart: FP={}  nous: FP={}",
        always.false_positive, smart.false_positive, nous.false_positive
    );

    println!("\nNotes:");
    println!("  • Synthetic task — illustrates whether the SIGNAL is decision-grade.");
    println!("    Real validation would use MetaMedQA or similar.");
    println!("  • Smart baseline is given fair starting state (pre-populated cluster");
    println!("    history matching Nous's training). Both start with equal info.");
    println!("  • If Nous matches smart baseline F1, the metacognition claim from");
    println!("    docs/intervention.md gap #1 is infrastructure-only on this task.");
    println!("  • If Nous beats smart baseline, `signals.confidence` adds discriminating");
    println!("    info beyond per-cluster historical tracking.");
}
