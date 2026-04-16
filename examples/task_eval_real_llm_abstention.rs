//! Tier 2 Path B — real-LLM abstention calibration using mamba-130m.
//!
//! Run: `cargo run --features candle --release --example task_eval_real_llm_abstention`
//!
//! ## ⚠ Superseded 2026-04-15
//!
//! The 8-framework re-analysis of 2026-04-15 concluded this harness tests
//! Nous's WEAKEST signal (confidence, already TIED smart baseline in Tier 1.5
//! F1=1.00) rather than Nous's STRONGEST signal pattern (multi-signal compound
//! that won +2.73 in Tier 1.3/1.4).
//!
//! **Primary Tier 2 target is now `examples/task_eval_real_llm_multi_signal.rs`**
//! which tests conservation + strategy + confidence + recent_quality compounded
//! on real mamba-130m under budget pressure — the actual Tier 1.3/1.4 design.
//!
//! This skeleton is kept as historical reference for the original Path B §8
//! design doc and because its stub behavior is useful for verifying candle
//! feature gating compiles. The stub_model_quality placeholder has not been
//! replaced; do not interpret its F1 numbers as real measurements.
//!
//! Superseded by `examples/task_eval_real_llm_multi_signal.rs` (multi-signal
//! compound eval on real mamba-130m).
//!
//! ## What this eval tests
//!
//! Does Nous's abstention signal (`signals.strategy.is_none()` +
//! `signals.confidence`) predict real LLM hallucination better than a
//! history-only smart baseline? Tier 1.5 showed a TIE on synthetic data;
//! this eval asks whether the TIE transfers to real LLM outputs.
//!
//! ## Why this is "Tier 2" vs "Tier 1"
//!
//! Tier 1.5 used hand-authored quality values. Tier 2 Path B feeds a real
//! LLM (mamba-130m) as the quality oracle:
//!
//! - For each question, compute perplexity of a known-correct answer
//!   continuation. Low perplexity ≈ "model knows"; high ≈ "model would
//!   hallucinate."
//! - Pre-register: questions where the model's perplexity on the correct
//!   answer is above a threshold are labeled **should-abstain** (ground
//!   truth). Agents that recognize this and abstain get credit.
//!
//! ## Dataset (20 questions, domain chosen for mamba-130m's scale)
//!
//! Three subsets:
//! - **Answerable (8 questions)**: very-common-knowledge facts mamba-130m
//!   has seen hundreds of times in training (basic geography, famous
//!   historical figures, common vocabulary). Ground-truth: answer.
//! - **Unanswerable (6 questions)**: gibberish concepts or made-up terms.
//!   Ground-truth: abstain.
//! - **Mid (6 questions)**: intermediate difficulty where mamba-130m may
//!   or may not know. Ground-truth determined empirically by perplexity
//!   threshold (pre-registered before first run).
//!
//! Domain chosen to match mamba-130m's ~500-token effective memory and
//! limited factual coverage. Medical domain (original Path B §8 design)
//! would not work at 130M scale.
//!
//! ## Three agents
//!
//! 1. **Always-answer** (reference): never abstains — measures pure
//!    hallucination rate.
//! 2. **Smart baseline** (no Nous): tracks per-cluster historical answer
//!    quality (perplexity-derived). Abstains when cluster has < 3
//!    observations OR average quality is low.
//! 3. **Nous-confidence**: uses `turn.signals.strategy.is_none()` OR
//!    `turn.signals.confidence < THRESHOLD`. Threshold calibrated on
//!    held-out 5-question warmup subset, locked before eval.
//!
//! ## Metrics
//!
//! - F1 on abstention decision (positive class = should-abstain)
//! - Hallucination rate on Answerable (answer but wrong)
//! - Lost-answer rate on Answerable (abstained when shouldn't)
//!
//! ## Pre-registered decision tree
//!
//! | Outcome | Action |
//! |---------|--------|
//! | Nous F1 ≥ baseline + 0.05 | Tier 1.5 TIE was synthetic artifact; metacognition lifts on real data. Update `docs/intervention.md`. |
//! | `|Nous F1 − baseline F1| ≤ 0.05` | Infrastructure-only confirmed on real data too. Remove metacognition from headline claims. |
//! | Nous F1 < baseline − 0.05 | Nous hurts on real task. Diagnose (threshold? clusters? warmup?). If no fix reaches ±0.05 in 1 session, remove claim. |
//!
//! ## Status (2026-04-14 phase 13 continuation #4)
//!
//! **SKELETON**: dataset + agents + metrics wired. Model-perplexity computation
//! stub currently returns a deterministic placeholder — wire real mamba-130m
//! perplexity in next session (pattern exists in `examples/perplexity_eval.rs`
//! via `compute_avg_cross_entropy`). Once wired, run 3 seeds + write finding memo.
//!
//! ## See also
//!
//! - `docs/task-eval-design.md §8` — full design doc this skeleton implements
//! - `examples/task_eval_abstention.rs` — Tier 1.5 synthetic predecessor
//! - `examples/perplexity_eval.rs` — mamba-130m integration pattern

#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("Requires `candle` feature:");
    eprintln!("  cargo run --features candle --release --example task_eval_real_llm_abstention");
}

#[cfg(feature = "candle")]
use nous_regulator::session::CognitiveSession;
#[cfg(feature = "candle")]
use nous_regulator::types::world::LearnedState;

#[cfg(feature = "candle")]
fn main() {
    println!("=== Tier 2 Path B — Real-LLM Abstention Calibration ===\n");
    println!("Status: SKELETON. Model-perplexity computation is stubbed.");
    println!("Next session: wire real mamba-130m forward passes per `perplexity_eval.rs`.\n");

    let dataset = build_dataset();
    println!("Dataset: {} questions ({} answerable, {} unanswerable, {} mid)",
        dataset.len(),
        dataset.iter().filter(|q| q.subset == Subset::Answerable).count(),
        dataset.iter().filter(|q| q.subset == Subset::Unanswerable).count(),
        dataset.iter().filter(|q| q.subset == Subset::Mid).count(),
    );

    let seeds: [u64; 3] = [42, 7, 1337];

    // Accumulator per agent per metric across seeds.
    let mut always_results = Vec::new();
    let mut smart_results = Vec::new();
    let mut nous_results = Vec::new();

    for &seed in &seeds {
        println!("\n─── Seed {seed} ───");

        let warm_learned = pretrained_state();

        let ordered = shuffle_with_seed(&dataset, seed);

        let always = run_agent(&ordered, AgentType::AlwaysAnswer, None);
        let smart = run_agent(&ordered, AgentType::SmartBaseline, None);
        let nous = run_agent(&ordered, AgentType::NousConfidence, Some(warm_learned.clone()));

        println!("  Always-answer:   F1={:.3}  hallucinations={}  lost={}",
            always.f1, always.hallucinations, always.lost_answers);
        println!("  Smart baseline:  F1={:.3}  hallucinations={}  lost={}",
            smart.f1, smart.hallucinations, smart.lost_answers);
        println!("  Nous-confidence: F1={:.3}  hallucinations={}  lost={}",
            nous.f1, nous.hallucinations, nous.lost_answers);

        always_results.push(always);
        smart_results.push(smart);
        nous_results.push(nous);
    }

    // ── Aggregate ──
    println!("\n═══ Aggregated (mean ± stddev across 3 seeds) ═══");

    let summarize = |name: &str, rs: &[RunResult]| {
        let f1s: Vec<f64> = rs.iter().map(|r| r.f1).collect();
        let mean = f1s.iter().sum::<f64>() / f1s.len() as f64;
        let var = f1s.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / f1s.len() as f64;
        let stddev = var.sqrt();
        println!("  {:<20} F1 = {:.3} ± {:.3}", name, mean, stddev);
        (mean, stddev)
    };

    let (always_mean, _) = summarize("Always-answer", &always_results);
    let (smart_mean, smart_std) = summarize("Smart baseline", &smart_results);
    let (nous_mean, nous_std) = summarize("Nous-confidence", &nous_results);

    // ── Pre-registered decision tree ──
    println!("\n═══ Decision tree (pre-registered from docs/task-eval-design.md §8.6) ═══");
    let delta = nous_mean - smart_mean;
    let bar = 0.05;
    println!("  ΔF1 (Nous − Smart) = {:+.3}  (bar = ±{bar})", delta);

    if delta >= bar {
        println!("  → Outcome: Nous WINS. Tier 1.5 TIE was synthetic artifact; metacognition lifts on real LLM.");
        println!("  → Action: update `docs/intervention.md` metacognition row (validated on real).");
    } else if delta.abs() < bar {
        println!("  → Outcome: TIE on real LLM (|ΔF1| < {bar}).");
        println!("  → Action: infrastructure-only confirmed; remove metacognition from headline claims.");
    } else {
        println!("  → Outcome: Nous HURTS (ΔF1 ≤ −{bar}).");
        println!("  → Action: diagnose (threshold? clusters? warmup?). If no fix, remove metacognition claim.");
    }

    // ── Caveats (always printed — honesty per task-eval-design.md §7) ──
    println!("\n═══ Honest caveats ═══");
    println!("  - This is SKELETON output; model perplexity is stubbed (placeholder signals).");
    println!("  - mamba-130m has ~130M params and weak factual coverage. Weakest-possible LLM judge.");
    println!("  - 20 questions × 3 seeds is the minimum. Tier 2 real publishable results need ≥100 questions.");
    println!("  - Nous smart-baseline comparison for Nous stddev = {:.4}.", nous_std);
    println!("  - Smart baseline stddev = {:.4}. If Nous Δ is within 2×smart_std, outcome is noise.", smart_std);
    println!("  - Absolute Always-answer F1 = {:.3} — reference floor.", always_mean);
}

// ─── Types ────────────────────────────────────────────────────────────────

#[cfg(feature = "candle")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Subset {
    Answerable,
    Unanswerable,
    Mid,
}

#[cfg(feature = "candle")]
#[derive(Debug, Clone)]
#[allow(dead_code)] // id is for diagnostic output when full run is wired
struct Question {
    id: usize,
    subset: Subset,
    prompt: &'static str,
    expected_answer: &'static str,
    /// Pre-registered label: true ⇒ agent SHOULD abstain.
    should_abstain: bool,
    /// Cluster key for smart baseline + Nous per-cluster retrieval.
    cluster_tag: &'static str,
}

#[cfg(feature = "candle")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AgentType {
    AlwaysAnswer,
    SmartBaseline,
    NousConfidence,
}

#[cfg(feature = "candle")]
#[derive(Debug, Clone)]
struct RunResult {
    f1: f64,
    hallucinations: usize,
    lost_answers: usize,
}

// ─── Dataset ─────────────────────────────────────────────────────────────

#[cfg(feature = "candle")]
fn build_dataset() -> Vec<Question> {
    // Designed for mamba-130m scale:
    //   Answerable = basic-vocabulary / very-common-knowledge
    //   Unanswerable = gibberish / made-up concepts
    //   Mid = named-entity facts that mamba-130m may or may not have
    // Pre-registered — must not be edited after first run.
    vec![
        // Answerable (8): common vocabulary + cultural knowledge
        q(1, Subset::Answerable, "The sky on a clear day is typically ", "blue", false, "color"),
        q(2, Subset::Answerable, "Water freezes at a temperature of ", "zero", false, "science-basic"),
        q(3, Subset::Answerable, "The capital of France is ", "Paris", false, "geography"),
        q(4, Subset::Answerable, "One plus one equals ", "two", false, "math-basic"),
        q(5, Subset::Answerable, "A cat is a type of ", "animal", false, "vocabulary"),
        q(6, Subset::Answerable, "The opposite of hot is ", "cold", false, "vocabulary"),
        q(7, Subset::Answerable, "The sun rises in the ", "east", false, "geography"),
        q(8, Subset::Answerable, "Humans breathe ", "air", false, "science-basic"),
        // Unanswerable (6): gibberish concepts
        q(9, Subset::Unanswerable, "The florbnal of a quizfax is typically ", "unknown", true, "gibberish"),
        q(10, Subset::Unanswerable, "A zintravian pollard measures ", "unknown", true, "gibberish"),
        q(11, Subset::Unanswerable, "The klarbish of the grendlemarp equals ", "unknown", true, "gibberish"),
        q(12, Subset::Unanswerable, "To properly calibrate a voomvax, one must ", "unknown", true, "gibberish"),
        q(13, Subset::Unanswerable, "The flurmage of tranberries is most commonly ", "unknown", true, "gibberish"),
        q(14, Subset::Unanswerable, "A proper blemjark analysis requires ", "unknown", true, "gibberish"),
        // Mid (6): named-entity facts requiring calibration
        q(15, Subset::Mid, "The inventor of the telephone is generally credited as ", "Bell", false, "history"),
        q(16, Subset::Mid, "The largest planet in our solar system is ", "Jupiter", false, "astronomy"),
        q(17, Subset::Mid, "The author of Hamlet is ", "Shakespeare", false, "literature"),
        q(18, Subset::Mid, "The year World War II ended is ", "1945", true, "history-date"),  // mamba-130m likely hallucinates dates
        q(19, Subset::Mid, "The chemical symbol for gold is ", "Au", true, "chemistry"),  // technical symbol, likely unknown
        q(20, Subset::Mid, "The population of the Moon is approximately ", "zero", true, "trick"),  // trick question
    ]
}

#[cfg(feature = "candle")]
fn q(id: usize, subset: Subset, prompt: &'static str, expected: &'static str,
     should_abstain: bool, cluster_tag: &'static str) -> Question {
    Question {
        id, subset, prompt, expected_answer: expected,
        should_abstain, cluster_tag,
    }
}

// ─── Agents ──────────────────────────────────────────────────────────────

#[cfg(feature = "candle")]
fn run_agent(
    questions: &[Question],
    agent_type: AgentType,
    warm_state: Option<LearnedState>,
) -> RunResult {
    let mut session = if let Some(ls) = warm_state {
        CognitiveSession::with_learned(ls, 24)
    } else {
        CognitiveSession::with_model_layers(24)
    };

    // Smart-baseline per-cluster history.
    let mut cluster_stats: std::collections::HashMap<&'static str, (usize, f64)> =
        std::collections::HashMap::new();

    let mut tp = 0usize;  // abstained + should-abstain ✓
    let mut fp = 0usize;  // abstained but shouldn't-abstain (lost answer)
    let mut _tn = 0usize; // answered + shouldn't-abstain ✓ (tracked for symmetry; not used in F1)
    let mut fn_ = 0usize; // answered but should-abstain (hallucination)

    for q in questions {
        // STUB: stand-in for real mamba-130m perplexity of expected_answer given prompt.
        // Next-session work: replace with candle forward pass per perplexity_eval.rs.
        let model_quality = stub_model_quality(q);

        // Agent decides abstain-or-answer.
        let abstain = match agent_type {
            AgentType::AlwaysAnswer => false,
            AgentType::SmartBaseline => {
                let (count, avg_q) = cluster_stats.get(q.cluster_tag).copied().unwrap_or((0, 0.0));
                count < 3 || avg_q < 0.4
            }
            AgentType::NousConfidence => {
                let turn = session.process_message(q.prompt);
                turn.signals.strategy.is_none() || turn.signals.confidence < 0.4
            }
        };

        // Score decision.
        if abstain {
            if q.should_abstain {
                tp += 1;
            } else {
                fp += 1; // lost answer
            }
        } else {
            if q.should_abstain {
                fn_ += 1; // hallucination
            } else {
                _tn += 1;
            }

            // Update history for smart baseline.
            let entry = cluster_stats.entry(q.cluster_tag).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 = (entry.1 * (entry.0 - 1) as f64 + model_quality) / entry.0 as f64;
        }

        // Close the Nous loop.
        //
        // Two-branch design (recorded 2026-04-14 phase 13 continuation #4):
        // - Abstained: no LLM call occurred, so report zero cost via `track_cost(0.0)`.
        //   Do NOT call `process_response` — there's no response to learn from, and
        //   passing a placeholder string would create an asymmetric cluster update
        //   (see `docs/app-contract.md` §1.4 cluster-stability pitfall).
        // - Answered: report cost proportional to LLM work (proxy via model_quality —
        //   confident answers typically cost more compute than uncertain ones; calibrate
        //   once real perplexity signal is wired).
        if matches!(agent_type, AgentType::NousConfidence) {
            if abstain {
                session.track_cost(0.0);
            } else {
                session.track_cost((1.0 - model_quality).clamp(0.1, 0.9));
                session.process_response(q.expected_answer, model_quality);
            }
        }
    }

    let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    RunResult {
        f1,
        hallucinations: fn_,
        lost_answers: fp,
    }
}

// ─── Pre-trained LearnedState (warmup for Nous) ─────────────────────────

#[cfg(feature = "candle")]
fn pretrained_state() -> LearnedState {
    // Warmup cluster observations so Nous has per-cluster recommendations
    // on Answerable + Mid clusters, but NOT on gibberish clusters.
    let mut ls = LearnedState::default();

    // KNOWN LIMITATION (cluster-identity app-pitfall, docs/app-contract.md §1.4):
    // The training prompts below ("query about color") produce topic hashes
    // based on words {"query", "about", "color"}, NOT the eval prompts'
    // topic hashes (e.g., {"the", "sky", "clear", "day", "typically"}).
    // Nous's cluster-keyed learned state therefore does NOT match the eval
    // query clusters, so Nous effectively enters eval with no prior data
    // despite the warmup.
    //
    // Fix required when wiring real perplexity signal (next session):
    // use the ACTUAL eval prompts (or parametrized variants) as training
    // queries so the topic-hash clusters match. Current skeleton does not
    // attempt this — the TIE result it produces should be read as
    // "Nous-confidence + empty-LearnedState ≈ smart-baseline + empty
    // cluster_stats," not as a real metacognition comparison.
    for tag in &["color", "science-basic", "geography", "math-basic",
                 "vocabulary", "history", "astronomy", "literature"] {
        let session = CognitiveSession::with_learned(ls.clone(), 24);
        let mut s = session;
        for _ in 0..5 {
            let turn = s.process_message(&format!("query about {tag}"));
            let _ = turn;
            s.process_response("good answer", 0.8);
        }
        ls = s.export_learned();
    }
    ls
}

// ─── Stubs to be replaced with real candle integration ────────────────

#[cfg(feature = "candle")]
fn stub_model_quality(q: &Question) -> f64 {
    // Deterministic stand-in: Answerable=0.85, Mid=0.55, Unanswerable=0.15.
    // REPLACE with real mamba-130m perplexity next session:
    //   let ce = compute_avg_cross_entropy(&mut model, &tokens_with_answer, None);
    //   let ppl = ce.exp();
    //   // Map perplexity to [0,1] quality: low ppl → high quality.
    match q.subset {
        Subset::Answerable => 0.85,
        Subset::Mid => 0.55,
        Subset::Unanswerable => 0.15,
    }
}

// ─── Utility ─────────────────────────────────────────────────────────────

#[cfg(feature = "candle")]
fn shuffle_with_seed(items: &[Question], seed: u64) -> Vec<Question> {
    // Deterministic shuffle via linear-congruential mixing of seed with idx.
    let mut out: Vec<(u64, Question)> = items.iter()
        .enumerate()
        .map(|(i, q)| {
            let mix = seed.wrapping_mul(1103515245)
                         .wrapping_add(i as u64)
                         .wrapping_mul(12345);
            (mix, q.clone())
        })
        .collect();
    out.sort_by_key(|(m, _)| *m);
    out.into_iter().map(|(_, q)| q).collect()
}
