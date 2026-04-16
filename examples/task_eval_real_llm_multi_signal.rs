//! Tier 2 — Real-LLM multi-signal compound eval (pivot from Path B skeleton).
//!
//! Run: `cargo run --features candle --release --example task_eval_real_llm_multi_signal`
//!
//! ## What this eval tests
//!
//! Does Nous's MULTI-SIGNAL compounding (conservation + strategy + confidence +
//! recent_quality) beat a smart app-level baseline on a REAL LLM (mamba-130m)?
//!
//! Tier 1.3 synthetic showed +2.73 total quality compound win. Tier 1.4
//! synthetic showed Nous wins at every budget. This eval asks: does the
//! compound transfer to real-model quality signals — or do the signals only
//! compound on deterministic synthetic quality?
//!
//! ## Why this pivots from `task_eval_real_llm_abstention.rs`
//!
//! The Path B skeleton tests ONE signal (confidence) on single-turn
//! abstention — Nous's weakest signal on Tier 1.5 (TIED smart baseline
//! F1=1.00). This eval tests the STRONGEST signal pattern (multi-signal
//! compound under budget pressure) from Tier 1.3/1.4 where Nous wins.
//!
//! Per 8-framework re-analysis 2026-04-15: Tier 2 should test Nous's point
//! of maximum differentiation, not the weakest. Path B skeleton kept as
//! historical reference; this file supersedes it as the primary Tier 2 target.
//!
//! ## Design
//!
//! - Stream: 20 queries across 5 categories (3 pre-trained + 2 novel) plus
//!   4 gibberish queries where abstention is correct.
//! - Budget cap: 7.0 effort units (tight enough that mode-switch matters).
//! - Quality oracle: mamba-130m perplexity of expected continuation →
//!   mapped to `quality = clamp((6.0 − avg_cross_entropy) / 4.5, 0, 1)`.
//! - Warmup: uses ACTUAL eval prompts (fixes Path B cluster-identity bug
//!   from `docs/app-contract.md` §1.4).
//!
//! ## Three agents
//!
//! 1. **Naive**: fixed DirectAnswer strategy, Full mode, no abstention.
//! 2. **Smart baseline (no Nous)**: per-category strategy memory with
//!    rotate-on-low-quality + cost-threshold shallow mode + count-based
//!    abstention on unfamiliar clusters (< 3 observations). Everything a
//!    competent engineer would build WITHOUT Nous.
//! 3. **Nous-full**: warm-started `CognitiveSession`, uses
//!    `signals.strategy` for learned recommendations,
//!    `signals.conservation > 0.2` to switch to shallow,
//!    `signals.confidence < 0.4` to abstain, closed loop via
//!    `track_cost` + `process_response`.
//!
//! ## Pre-registered decision (from 8-framework re-analysis 2026-04-15)
//!
//! | Outcome | Action |
//! |---------|--------|
//! | Nous total_q ≥ smart + 1.0 | Multi-signal compound confirmed on real LLM. Elevate Tier 1.3/1.4 story to "validated on real model." |
//! | \|Nous − smart\| < 1.0 | Compound is synthetic artifact on real LLM. Downgrade Tier 1.3/1.4 to "synthetic-only". Honest. |
//! | Nous < smart − 1.0 | Nous actively hurts on real LLM. Investigate signal calibration before any public claim. |
//!
//! ## Honest caveats
//!
//! - mamba-130m is a weak model (~130M params). Quality oracle is noisy.
//!   Weak-model results DO NOT automatically generalize to frontier models.
//! - 20 queries × 3 seeds is a minimum. Real benchmark needs ≥100 queries.
//! - Strategy discrimination may be weak on mamba-130m regardless of Nous
//!   (Direct prompts may dominate all categories). If so, Nous's win/loss
//!   comes from conservation + confidence, not strategy.
//! - Quality→[0,1] mapping has calibration choices documented in `quality_from_ce`.
//!
//! ## See also
//!
//! - `examples/task_eval_multi_signal.rs` — Tier 1.3 synthetic predecessor
//! - `examples/task_eval_budget_sweep.rs` — Tier 1.4 budget sweep
//! - `examples/perplexity_eval.rs` — mamba-130m integration pattern
//! - `examples/task_eval_real_llm_abstention.rs` — Path B skeleton (single-signal)
//! - `docs/task-eval-design.md §8` — original Path B design doc

#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("Requires `candle` feature:");
    eprintln!("  cargo run --features candle --release --example task_eval_real_llm_multi_signal");
}

#[cfg(feature = "candle")]
use nous::inference::cognitive_model::CognitiveModel;
#[cfg(feature = "candle")]
use nous::inference::mamba::{CognitiveMambaModel, HfTokenizer, MambaConfig};
#[cfg(feature = "candle")]
use nous::inference::model::LocalModel;
#[cfg(feature = "candle")]
use nous::inference::tokenizer::NousTokenizer;
#[cfg(feature = "candle")]
use nous::math::softmax::softmax_f32;
#[cfg(feature = "candle")]
use nous::session::CognitiveSession;
#[cfg(feature = "candle")]
use nous::types::world::{LearnedState, ResponseStrategy};
#[cfg(feature = "candle")]
use std::collections::BTreeMap;

// ── Budget + thresholds ───────────────────────────────────────────────────

#[cfg(feature = "candle")]
const BUDGET_CAP: f64 = 7.0;
#[cfg(feature = "candle")]
const NOUS_CONSERVATION_THRESHOLD: f64 = 0.2;
#[cfg(feature = "candle")]
const NOUS_CONFIDENCE_THRESHOLD: f64 = 0.4;
#[cfg(feature = "candle")]
const QUALITY_RETRY_THRESHOLD: f64 = 0.4;
#[cfg(feature = "candle")]
const COST_CONSERVATION_FRACTION: f64 = 0.5;
#[cfg(feature = "candle")]
const SMART_ABSTAIN_MIN_COUNT: usize = 3;

// ── Types ─────────────────────────────────────────────────────────────────

#[cfg(feature = "candle")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Category {
    // Pre-trained (Nous has warm LearnedState for these).
    Recall,    // short factual — correct=Direct
    Reason,    // simple step-by-step — correct=StepByStep
    Clarify,   // ambiguous request — correct=AskClarifying
    // Novel (cold, no warmup data).
    Compose,   // creative short completion — correct=Direct
    Explain,   // elaborative — correct=StepByStep
    // Abstention-correct.
    Gibberish, // nonsense prompts — correct action = abstain
}

#[cfg(feature = "candle")]
impl Category {
    fn base_question(self, idx: usize) -> &'static str {
        match (self, idx % 4) {
            (Category::Recall, 0) => "The capital of France is",
            (Category::Recall, 1) => "The color of healthy grass is",
            (Category::Recall, 2) => "The opposite of hot is",
            (Category::Recall, _) => "One plus one equals",
            (Category::Reason, 0) => "All dogs are mammals and Rex is a dog so Rex is a",
            (Category::Reason, 1) => "Rain makes ground wet. It is raining now. The ground is",
            (Category::Reason, 2) => "A square has four equal sides and four",
            (Category::Reason, _) => "If it is warmer than the freezing point, ice will",
            (Category::Clarify, 0) => "Can you please make this better for the team",
            (Category::Clarify, 1) => "This thing here really needs some improvement soon",
            (Category::Clarify, 2) => "Please go and fix the issue that came up",
            (Category::Clarify, _) => "Could you review that document we talked about",
            (Category::Compose, 0) => "The sunset painted the sky with",
            (Category::Compose, 1) => "A small child laughed and ran through",
            (Category::Compose, 2) => "The old tree stood tall and quiet against",
            (Category::Compose, _) => "In the early morning the river flowed past",
            (Category::Explain, 0) => "The idea of democracy as a system of government",
            (Category::Explain, 1) => "Photosynthesis in green plants involves",
            (Category::Explain, 2) => "The second law of thermodynamics states that",
            (Category::Explain, _) => "Supply and demand in an economy refers to",
            (Category::Gibberish, 0) => "The florbnal of a quizfax is typically",
            (Category::Gibberish, 1) => "A zintravian pollard measures",
            (Category::Gibberish, 2) => "The klarbish of the grendlemarp equals",
            (Category::Gibberish, _) => "To properly calibrate a voomvax one must first",
        }
    }

    /// Expected continuation used to score mamba's perplexity. Short, natural.
    fn expected_answer(self, idx: usize) -> &'static str {
        match (self, idx % 4) {
            (Category::Recall, 0) => " Paris.",
            (Category::Recall, 1) => " green.",
            (Category::Recall, 2) => " cold.",
            (Category::Recall, _) => " two.",
            (Category::Reason, 0) => " mammal.",
            (Category::Reason, 1) => " wet.",
            (Category::Reason, 2) => " corners.",
            (Category::Reason, _) => " melt.",
            (Category::Clarify, _) => " please specify.",
            (Category::Compose, _) => " soft colors.",
            (Category::Explain, _) => " a complex process.",
            (Category::Gibberish, _) => " unknown.",
        }
    }

    fn should_abstain(self) -> bool {
        matches!(self, Category::Gibberish)
    }

    fn correct_strategy(self) -> AppStrategy {
        match self {
            Category::Recall => AppStrategy::DirectAnswer,
            Category::Reason => AppStrategy::StepByStep,
            Category::Clarify => AppStrategy::AskClarifying,
            Category::Compose => AppStrategy::DirectAnswer,
            Category::Explain => AppStrategy::StepByStep,
            Category::Gibberish => AppStrategy::DirectAnswer, // abstaining is what counts
        }
    }

    fn is_pretrained(self) -> bool {
        matches!(self, Category::Recall | Category::Reason | Category::Clarify)
    }
}

#[cfg(feature = "candle")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppStrategy {
    DirectAnswer,
    AskClarifying,
    StepByStep,
}

#[cfg(feature = "candle")]
impl AppStrategy {
    fn next(self) -> Self {
        match self {
            Self::DirectAnswer => Self::AskClarifying,
            Self::AskClarifying => Self::StepByStep,
            Self::StepByStep => Self::DirectAnswer,
        }
    }

    /// Cost per query. Mirrors Tier 1.3 cost calibration.
    fn base_cost(self) -> f64 {
        match self {
            Self::DirectAnswer => 0.25,
            Self::AskClarifying => 0.35,
            Self::StepByStep => 0.70,
        }
    }

    /// Primer text prepended to the base question. Drives quality via
    /// which prompt style mamba-130m finds natural for each completion.
    fn primer_after(self) -> &'static str {
        match self {
            Self::DirectAnswer => "",
            Self::AskClarifying => " To clarify, do you mean",
            Self::StepByStep => " Let me think step by step. First,",
        }
    }

    /// Response text used in closing the Nous loop (`process_response`).
    /// Must match `detect_response_strategy` format so LearnedState records
    /// the intended strategy. Formats taken from Tier 1.1 / Tier 1.3 fixes.
    fn canonical_response(self) -> &'static str {
        match self {
            Self::DirectAnswer => "Short direct answer.",
            Self::AskClarifying => "What exactly do you need? What have you tried so far?",
            Self::StepByStep => "1. First, identify the core issue.\n2. Then, examine each part.\n3. Finally, combine results.",
        }
    }
}

#[cfg(feature = "candle")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Full,
    Shallow,
}

#[cfg(feature = "candle")]
impl Mode {
    fn cost_multiplier(self) -> f64 {
        match self {
            Self::Full => 1.0,
            Self::Shallow => 0.5,
        }
    }

    /// Shallow mode truncates the prompt primer → weaker context → higher ppl.
    fn quality_multiplier(self) -> f64 {
        match self {
            Self::Full => 1.0,
            Self::Shallow => 0.75,
        }
    }
}

// ── Dataset ───────────────────────────────────────────────────────────────

#[cfg(feature = "candle")]
#[derive(Debug, Clone, Copy)]
struct Query {
    category: Category,
    idx: usize,
}

#[cfg(feature = "candle")]
fn build_dataset() -> Vec<Query> {
    // 20 queries total.
    //
    // Pre-trained: 4 Recall + 3 Reason + 3 Clarify = 10
    // Novel:       3 Compose + 3 Explain          = 6
    // Abstention:  4 Gibberish                    = 4
    //
    // Ordering mixes all three tiers to avoid agent-warmup bias.
    let plan = [
        (Category::Recall, 0),
        (Category::Reason, 0),
        (Category::Gibberish, 0),
        (Category::Compose, 0),
        (Category::Clarify, 0),
        (Category::Explain, 0),
        (Category::Recall, 1),
        (Category::Gibberish, 1),
        (Category::Reason, 1),
        (Category::Compose, 1),
        (Category::Explain, 1),
        (Category::Clarify, 1),
        (Category::Recall, 2),
        (Category::Gibberish, 2),
        (Category::Reason, 2),
        (Category::Compose, 2),
        (Category::Explain, 2),
        (Category::Clarify, 2),
        (Category::Recall, 3),
        (Category::Gibberish, 3),
    ];
    plan.iter().map(|&(c, i)| Query { category: c, idx: i }).collect()
}

// ── Quality oracle (mamba-130m perplexity) ────────────────────────────────

/// Map average cross-entropy → quality in [0, 1].
///
/// Calibration (mamba-130m on short completions):
/// - ce ≈ 2.0 (easy factual): quality ≈ 0.89
/// - ce ≈ 4.0 (noisy continuation): quality ≈ 0.44
/// - ce ≈ 6.0 (high perplexity / nonsense): quality ≈ 0.0
///
/// Clamped to [0, 1].
#[cfg(feature = "candle")]
fn quality_from_ce(ce: f64) -> f64 {
    ((6.0 - ce) / 4.5).clamp(0.0, 1.0)
}

/// Compute average cross-entropy of `tokens[i+1..]` predicted from `tokens[..i]`.
/// Reset cache at start. Mirrors `compute_avg_cross_entropy` in perplexity_eval.rs.
#[cfg(feature = "candle")]
fn compute_ce(model: &mut CognitiveMambaModel, tokens: &[u32]) -> f64 {
    if tokens.len() < 2 {
        return 0.0;
    }
    model.reset_cache();
    let mut total_ce = 0.0;
    let mut count = 0;
    for i in 0..tokens.len() - 1 {
        let input_token = tokens[i];
        let target_token = tokens[i + 1] as usize;
        let logits = match model.forward(&[input_token], i) {
            Ok(l) => l,
            Err(_) => continue,
        };
        if target_token >= logits.len() {
            continue;
        }
        let probs = softmax_f32(&logits);
        let target_prob = probs[target_token] as f64;
        let ce = -(target_prob.max(1e-10)).ln();
        total_ce += ce;
        count += 1;
    }
    if count == 0 { 0.0 } else { total_ce / count as f64 }
}

/// Real-LLM quality for (query, chosen_strategy, mode).
///
/// Builds prompt = base_question + strategy_primer + expected_answer,
/// computes avg CE on just the answer tokens, maps to [0, 1].
#[cfg(feature = "candle")]
fn measure_quality(
    model: &mut CognitiveMambaModel,
    tokenizer: &HfTokenizer,
    query: Query,
    strategy: AppStrategy,
    mode: Mode,
) -> f64 {
    let base = query.category.base_question(query.idx);
    let primer = strategy.primer_after();
    let answer = query.category.expected_answer(query.idx);
    let prompt = format!("{base}{primer}{answer}");

    let tokens = match tokenizer.encode(&prompt, false) {
        Ok(t) if t.len() >= 3 => t,
        _ => return 0.0,
    };

    let ce = compute_ce(model, &tokens);
    let raw_quality = quality_from_ce(ce);
    (raw_quality * mode.quality_multiplier()).clamp(0.0, 1.0)
}

// ── Metrics ───────────────────────────────────────────────────────────────

#[cfg(feature = "candle")]
#[derive(Debug, Default, Clone)]
struct RunResult {
    queries_served: usize,
    queries_abstained: usize,
    queries_skipped: usize,
    total_cost: f64,
    total_quality: f64,
    mode_switches_to_shallow: usize,
    correct_strategy_used: usize,
    abstention_tp: usize, // abstained + should-abstain
    abstention_fp: usize, // abstained + shouldn't-abstain (lost answer)
    abstention_fn: usize, // answered + should-abstain (hallucination)
    hallucinations_q_sum: f64, // summed quality on hallucinated gibberish
}

#[cfg(feature = "candle")]
impl RunResult {
    fn avg_quality(&self) -> f64 {
        if self.queries_served == 0 { 0.0 } else { self.total_quality / self.queries_served as f64 }
    }

    fn abstention_f1(&self) -> f64 {
        let tp = self.abstention_tp as f64;
        let fp = self.abstention_fp as f64;
        let fn_ = self.abstention_fn as f64;
        let p = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let r = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 }
    }
}

// ── Warmup (Nous LearnedState) ────────────────────────────────────────────

/// Train a LearnedState using the ACTUAL eval prompts so cluster hashes match.
///
/// Each pre-trained (category, idx) pair gets `WARMUP_ROUNDS` observations.
///
/// CRITICAL: `WARMUP_ROUNDS` must be ≥ `MODERATE_MIN_COUNT` (5 per
/// `types/world.rs`) for `get_recommended_strategy` to return Some on that
/// cluster. 2026-04-15 diagnostic pass found v1/v2 runs used rounds=2 —
/// below the threshold — so `turn.signals.strategy` returned None for
/// every eval turn, silently disabling Nous's cross-session reward learning.
/// v3 bumps to 6 (matches Tier 1.3 `train_prior_session` pattern).
///
/// Uses `canonical_response` text so `detect_response_strategy_safe` records
/// the right strategy per cluster. No mamba calls — strategy learning works
/// from text alone.
#[cfg(feature = "candle")]
const WARMUP_ROUNDS: usize = 6;

#[cfg(feature = "candle")]
fn train_warmup() -> LearnedState {
    let mut session = CognitiveSession::new();
    for _round in 0..WARMUP_ROUNDS {
        for cat in [Category::Recall, Category::Reason, Category::Clarify] {
            for idx in 0..4 {
                let prompt = cat.base_question(idx);
                let _turn = session.process_message(prompt);
                let strategy = cat.correct_strategy();
                session.track_cost(strategy.base_cost());
                session.process_response(strategy.canonical_response(), 0.85);
            }
        }
    }
    session.export_learned()
}

// ── Agent runners ─────────────────────────────────────────────────────────

#[cfg(feature = "candle")]
fn run_naive(
    model: &mut CognitiveMambaModel,
    tokenizer: &HfTokenizer,
    stream: &[Query],
) -> RunResult {
    let mut r = RunResult::default();
    let strategy = AppStrategy::DirectAnswer;
    let mode = Mode::Full;
    for q in stream {
        let cost = strategy.base_cost() * mode.cost_multiplier();
        if r.total_cost + cost > BUDGET_CAP {
            r.queries_skipped += 1;
            continue;
        }
        // Naive never abstains. If should_abstain, it hallucinates.
        let quality = measure_quality(model, tokenizer, *q, strategy, mode);
        record_answer(&mut r, *q, strategy, mode, cost, quality);
    }
    r
}

#[cfg(feature = "candle")]
fn run_smart_baseline(
    model: &mut CognitiveMambaModel,
    tokenizer: &HfTokenizer,
    stream: &[Query],
) -> RunResult {
    let mut r = RunResult::default();
    let mut per_cat_strategy: BTreeMap<Category, AppStrategy> = BTreeMap::new();
    let mut per_cat_last_q: BTreeMap<Category, f64> = BTreeMap::new();
    // Cluster key = (category, idx) — gives smart baseline the cluster granularity
    // it would naturally build (per-prompt history).
    let mut cluster_stats: BTreeMap<(Category, usize), (usize, f64)> = BTreeMap::new();

    for q in stream {
        // Shallow mode once we've spent > half budget.
        let mode = if r.total_cost >= BUDGET_CAP * COST_CONSERVATION_FRACTION {
            if r.mode_switches_to_shallow == 0 {
                r.mode_switches_to_shallow = 1;
            }
            Mode::Shallow
        } else {
            Mode::Full
        };

        // Strategy: per-category memory + rotate on low quality.
        let current = per_cat_strategy.entry(q.category).or_insert(AppStrategy::DirectAnswer);
        if let Some(&last_q) = per_cat_last_q.get(&q.category) {
            if last_q < QUALITY_RETRY_THRESHOLD {
                *current = current.next();
            }
        }
        let strategy = *current;

        // Count-based abstention: abstain when cluster count < 3 AND average
        // quality is low. On fresh clusters or clusters with poor history.
        let (count, avg_q) = cluster_stats
            .get(&(q.category, q.idx))
            .copied()
            .unwrap_or((0, 0.0));
        let abstain = count >= SMART_ABSTAIN_MIN_COUNT && avg_q < 0.35;

        if abstain {
            record_abstain(&mut r, *q);
            continue;
        }

        let cost = strategy.base_cost() * mode.cost_multiplier();
        if r.total_cost + cost > BUDGET_CAP {
            r.queries_skipped += 1;
            continue;
        }

        let quality = measure_quality(model, tokenizer, *q, strategy, mode);
        per_cat_last_q.insert(q.category, quality);
        let entry = cluster_stats.entry((q.category, q.idx)).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 = (entry.1 * (entry.0 - 1) as f64 + quality) / entry.0 as f64;

        record_answer(&mut r, *q, strategy, mode, cost, quality);
    }
    r
}

#[cfg(feature = "candle")]
fn run_nous_full(
    model: &mut CognitiveMambaModel,
    tokenizer: &HfTokenizer,
    stream: &[Query],
    warm: LearnedState,
) -> RunResult {
    let mut session = CognitiveSession::with_learned(warm, 24);
    let mut r = RunResult::default();
    let mut per_cat_strategy: BTreeMap<Category, AppStrategy> = BTreeMap::new();
    let mut per_cat_last_q: BTreeMap<Category, f64> = BTreeMap::new();
    let mut in_shallow = false;

    // Diagnostic mode: per-turn signal trajectory logging. Enabled by
    // `NOUS_DIAGNOSE=1` env var. Used to understand why conservation never
    // fires (mode_sw=0) and why confidence never drops on gibberish (absF1=0)
    // in the base eval run.
    let diagnose = std::env::var("NOUS_DIAGNOSE").is_ok();
    if diagnose {
        println!("\n  ── NOUS_DIAGNOSE: per-turn signal trajectory ──");
        println!("  {:>3} {:<10} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>3} {:<8} {:<12} dec",
            "t", "cat", "body_b", "arous", "cons", "salnc", "conf", "recQ", "costsum",
            "str", "gate", "strategy");
    }

    for (turn_idx, q) in stream.iter().enumerate() {
        let prompt = q.category.base_question(q.idx);
        let turn = session.process_message(prompt);

        // Conservation-driven mode.
        if !in_shallow && turn.signals.conservation > NOUS_CONSERVATION_THRESHOLD {
            in_shallow = true;
            r.mode_switches_to_shallow += 1;
        }
        let mode = if in_shallow { Mode::Shallow } else { Mode::Full };

        // Confidence-driven abstention: when confidence is low AND no prior strategy.
        let abstain = turn.signals.confidence < NOUS_CONFIDENCE_THRESHOLD
            && turn.signals.strategy.is_none();

        if abstain {
            if diagnose {
                print_diagnose_row(
                    turn_idx, q, &turn, r.total_cost, None, None, "abstain",
                );
            }
            record_abstain(&mut r, *q);
            // Close the loop — zero cost, no response to learn from.
            session.track_cost(0.0);
            continue;
        }

        // Strategy: prefer Nous's learned recommendation (cross-session).
        // Fall back to per-category rotation.
        let current = per_cat_strategy.entry(q.category).or_insert(AppStrategy::DirectAnswer);
        let strategy = if let Some(rec) = turn.signals.strategy {
            map_rec(rec)
        } else if let Some(&last_q) = per_cat_last_q.get(&q.category) {
            if last_q < QUALITY_RETRY_THRESHOLD {
                current.next()
            } else {
                *current
            }
        } else {
            *current
        };
        *current = strategy;

        let cost = strategy.base_cost() * mode.cost_multiplier();
        if r.total_cost + cost > BUDGET_CAP {
            if diagnose {
                print_diagnose_row(
                    turn_idx, q, &turn, r.total_cost, Some(strategy), Some(mode), "skip(bud)",
                );
            }
            r.queries_skipped += 1;
            session.track_cost(0.0);
            continue;
        }

        let quality = measure_quality(model, tokenizer, *q, strategy, mode);
        per_cat_last_q.insert(q.category, quality);

        if diagnose {
            let dec = format!("q={:.3}", quality);
            print_diagnose_row(
                turn_idx, q, &turn, r.total_cost + cost, Some(strategy), Some(mode), &dec,
            );
        }

        record_answer(&mut r, *q, strategy, mode, cost, quality);

        // Close the allostatic loop.
        session.track_cost(cost);
        session.process_response(strategy.canonical_response(), quality);
    }

    if diagnose {
        println!("  ── end trajectory ──\n");
    }
    r
}

/// Print one diagnostic row of per-turn Nous-full state.
#[cfg(feature = "candle")]
fn print_diagnose_row(
    turn_idx: usize,
    q: &Query,
    turn: &nous::session::TurnResult,
    cost_accum: f64,
    strategy: Option<AppStrategy>,
    _mode: Option<Mode>,
    decision: &str,
) {
    let cat_short = match q.category {
        Category::Recall => "Recall",
        Category::Reason => "Reason",
        Category::Clarify => "Clarify",
        Category::Compose => "Compose",
        Category::Explain => "Explain",
        Category::Gibberish => "Gibberish",
    };
    let strategy_str = match turn.signals.strategy {
        Some(ResponseStrategy::DirectAnswer) => "Direct",
        Some(ResponseStrategy::StepByStep) => "Step",
        Some(ResponseStrategy::ClarifyFirst) => "Clarify",
        Some(ResponseStrategy::StructuredAnalysis) => "Struct",
        Some(ResponseStrategy::ExecuteTask) => "Exec",
        None => "None",
    };
    let gate_str = format!("{:?}", turn.gate_type);
    let has_str = if strategy.is_some() { "Y" } else { "N" };
    println!(
        "  {:>3} {:<10} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.2} {:>3} {:<8} {:<12} {}",
        turn_idx,
        cat_short,
        turn.body_budget,
        turn.arousal,
        turn.signals.conservation,
        turn.signals.salience,
        turn.signals.confidence,
        turn.signals.recent_quality,
        cost_accum,
        has_str,
        gate_str,
        strategy_str,
        decision,
    );
}

// ── Shared scoring helpers ────────────────────────────────────────────────

#[cfg(feature = "candle")]
fn record_answer(
    r: &mut RunResult,
    q: Query,
    strategy: AppStrategy,
    _mode: Mode,
    cost: f64,
    quality: f64,
) {
    r.total_cost += cost;
    r.total_quality += quality;
    r.queries_served += 1;
    if strategy == q.category.correct_strategy() && !q.category.should_abstain() {
        r.correct_strategy_used += 1;
    }
    if q.category.should_abstain() {
        r.abstention_fn += 1;
        r.hallucinations_q_sum += quality;
    }
}

#[cfg(feature = "candle")]
fn record_abstain(r: &mut RunResult, q: Query) {
    // Abstention: no cost, no quality, no query_served count.
    r.queries_abstained += 1;
    if q.category.should_abstain() {
        r.abstention_tp += 1;
    } else {
        r.abstention_fp += 1;
    }
}

#[cfg(feature = "candle")]
fn map_rec(r: ResponseStrategy) -> AppStrategy {
    match r {
        ResponseStrategy::StepByStep | ResponseStrategy::StructuredAnalysis => {
            AppStrategy::StepByStep
        }
        ResponseStrategy::ClarifyFirst => AppStrategy::AskClarifying,
        ResponseStrategy::DirectAnswer | ResponseStrategy::ExecuteTask => {
            AppStrategy::DirectAnswer
        }
    }
}

// ── Reporting ─────────────────────────────────────────────────────────────

#[cfg(feature = "candle")]
fn print_row(name: &str, r: &RunResult) {
    println!(
        "  {:<22} served={:>2} abst={:>2} skip={:>2} mod_sw={:>1} cost={:>5.2}  avg_q={:.3}  total_q={:>6.2}  correct={:>2}/{}  absF1={:.3}  hal={}",
        name,
        r.queries_served,
        r.queries_abstained,
        r.queries_skipped,
        r.mode_switches_to_shallow,
        r.total_cost,
        r.avg_quality(),
        r.total_quality,
        r.correct_strategy_used,
        r.queries_served,
        r.abstention_f1(),
        r.abstention_fn,
    );
}

#[cfg(feature = "candle")]
fn summarize_metric(name: &str, results: &[RunResult], extract: impl Fn(&RunResult) -> f64) -> (f64, f64) {
    let values: Vec<f64> = results.iter().map(extract).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let stddev = var.sqrt();
    println!("  {:<22} {:.3} ± {:.3}", name, mean, stddev);
    (mean, stddev)
}

// ── Main ──────────────────────────────────────────────────────────────────

#[cfg(feature = "candle")]
fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  task_eval_real_llm_multi_signal — Tier 2 multi-signal compound   ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    println!("Loading mamba-130m-hf (this may take ~15s)...");
    let model_id = "state-spaces/mamba-130m-hf";
    let config = MambaConfig::mamba_130m();

    let tokenizer = match HfTokenizer::from_pretrained(model_id) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Tokenizer load failed: {e}");
            return;
        }
    };
    let mut model = match CognitiveMambaModel::from_pretrained(model_id, config) {
        Ok(m) => {
            println!("  model ready: {} layers.\n", m.num_layers());
            m
        }
        Err(e) => {
            eprintln!("Model load failed: {e}");
            return;
        }
    };

    let dataset = build_dataset();
    let total_answerable = dataset.iter().filter(|q| !q.category.should_abstain()).count();
    let total_gibberish = dataset.iter().filter(|q| q.category.should_abstain()).count();
    let total_novel = dataset
        .iter()
        .filter(|q| !q.category.is_pretrained() && !q.category.should_abstain())
        .count();
    println!(
        "Stream: {} queries ({} answerable [{} pre-trained + {} novel], {} gibberish)",
        dataset.len(),
        total_answerable,
        total_answerable - total_novel,
        total_novel,
        total_gibberish,
    );
    println!("Budget cap: {:.1} effort units\n", BUDGET_CAP);

    let seeds: [u64; 3] = [42, 7, 1337];
    let mut naive_results: Vec<RunResult> = Vec::new();
    let mut smart_results: Vec<RunResult> = Vec::new();
    let mut nous_results: Vec<RunResult> = Vec::new();

    for &seed in &seeds {
        println!("─── Seed {seed} ───");

        let warm = train_warmup();
        if seed == seeds[0] {
            println!(
                "  warmup LearnedState: {} strategy clusters, tick={}",
                warm.response_strategies.len(),
                warm.tick,
            );
        }

        let ordered = shuffle_seeded(&dataset, seed);

        let naive = run_naive(&mut model, &tokenizer, &ordered);
        let smart = run_smart_baseline(&mut model, &tokenizer, &ordered);
        let nous = run_nous_full(&mut model, &tokenizer, &ordered, warm);

        print_row("naive (ref)", &naive);
        print_row("smart baseline", &smart);
        print_row("nous-full", &nous);
        println!();

        naive_results.push(naive);
        smart_results.push(smart);
        nous_results.push(nous);
    }

    // ── Aggregated metrics ──
    println!("═══ Aggregated (mean ± stddev across 3 seeds) ═══\n");

    println!("Total quality (primary metric):");
    let _ = summarize_metric("naive (ref)", &naive_results, |r| r.total_quality);
    let (smart_tq, _) = summarize_metric("smart baseline", &smart_results, |r| r.total_quality);
    let (nous_tq, nous_tq_sd) = summarize_metric("nous-full", &nous_results, |r| r.total_quality);

    println!("\nAvg quality per served query:");
    let _ = summarize_metric("naive (ref)", &naive_results, |r| r.avg_quality());
    let _ = summarize_metric("smart baseline", &smart_results, |r| r.avg_quality());
    let _ = summarize_metric("nous-full", &nous_results, |r| r.avg_quality());

    println!("\nAbstention F1:");
    let _ = summarize_metric("naive (ref)", &naive_results, |r| r.abstention_f1());
    let (smart_f1, _) = summarize_metric("smart baseline", &smart_results, |r| r.abstention_f1());
    let (nous_f1, _) = summarize_metric("nous-full", &nous_results, |r| r.abstention_f1());

    println!("\nCorrect-strategy rate (on served answerable queries):");
    let correct_rate = |r: &RunResult| {
        if r.queries_served > 0 {
            r.correct_strategy_used as f64 / r.queries_served as f64
        } else {
            0.0
        }
    };
    let _ = summarize_metric("naive (ref)", &naive_results, correct_rate);
    let _ = summarize_metric("smart baseline", &smart_results, correct_rate);
    let _ = summarize_metric("nous-full", &nous_results, correct_rate);

    // ── Decision tree ──
    println!("\n═══ Pre-registered decision (primary = total_quality) ═══");
    let delta_tq = nous_tq - smart_tq;
    let bar = 1.0;
    println!("  Δtotal_q (Nous − Smart) = {:+.2}  (bar = ±{bar:.1})", delta_tq);
    if delta_tq >= bar {
        println!("  → Outcome: MULTI-SIGNAL COMPOUND CONFIRMED on real LLM.");
        println!("    Update `docs/intervention.md` + `docs/task-eval-design.md` with real-LLM result.");
    } else if delta_tq.abs() < bar {
        println!("  → Outcome: COMPOUND IS SYNTHETIC ARTIFACT on real LLM.");
        println!("    Tier 1.3/1.4 synthetic wins don't transfer. Downgrade claim to synthetic-only.");
    } else {
        println!("  → Outcome: NOUS HURTS on real LLM (Δ ≤ −{bar:.1}).");
        println!("    Investigate: strategy mapping? conservation threshold? warmup clusters?");
    }

    println!("\n═══ Secondary — abstention F1 (matches Tier 1.5 TIE question) ═══");
    let delta_f1 = nous_f1 - smart_f1;
    println!("  ΔF1 (Nous − Smart) = {:+.3}", delta_f1);
    if delta_f1 >= 0.05 {
        println!("  → Nous-confidence separates from smart baseline on real LLM.");
    } else if delta_f1.abs() < 0.05 {
        println!("  → Confidence signal = per-cluster history on real LLM too (Tier 1.5 TIE transfers).");
    } else {
        println!("  → Nous confidence HURTS abstention on real LLM. Investigate threshold.");
    }

    // ── Honest caveats ──
    println!("\n═══ Honest caveats (always printed per task-eval-design.md §7) ═══");
    println!("  - mamba-130m is a weak model; results do not automatically generalize.");
    println!("  - 20 queries × 3 seeds is the minimum. Publishable benchmark needs ≥100.");
    println!("  - nous-full total_q stddev = {:.3}. If Δ < 2×stddev, outcome is noise.", nous_tq_sd);
    println!("  - Cluster-identity fixed: warmup uses eval prompts, not placeholders.");
    println!("  - Strategy primer effect on mamba-130m may be weak; watch correct-strategy rate.");
}

// ── Utility ───────────────────────────────────────────────────────────────

#[cfg(feature = "candle")]
fn shuffle_seeded(items: &[Query], seed: u64) -> Vec<Query> {
    // Splitmix64-style finalizer. Required because the naive LCG
    // `(seed * A + i) * B = seed*A*B + i*B` collapses to sort-by-i for any
    // seed — A*B is a constant offset, so the relative ordering is `i*B`
    // which is monotonic. That produces identical shuffle across seeds.
    // Discovered 2026-04-15 after first run showed stddev=0 across 3 seeds.
    let mut out: Vec<(u64, Query)> = items
        .iter()
        .enumerate()
        .map(|(i, q)| {
            let mut x = seed.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^= x >> 31;
            (x, *q)
        })
        .collect();
    out.sort_by_key(|(m, _)| *m);
    out.into_iter().map(|(_, q)| q).collect()
}
