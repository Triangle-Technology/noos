//! Tier 1.8 gradual-degradation fatigue eval — does Nous's smooth signal win
//! the regime that Tier 1.7's abrupt-degradation gave to direct quality
//! monitoring?
//!
//! Run: `cargo run --example task_eval_fatigue_gradual`
//!
//! ## Why this eval matters
//!
//! Tier 1.7 (`task_eval_fatigue.rs`) tested abrupt quality drop and Nous's
//! combined signal lost (8 vs 4 turn detection latency, 8 vs 4 harm count).
//! Honest finding for that regime. But Nous's smoothing is by-design — the
//! tradeoff should pay off when:
//!
//! - Quality is noisy turn-to-turn (so rolling avg false-positives)
//! - Degradation is gradual rather than abrupt
//!
//! Tier 1.8 tests EXACTLY that regime to see whether the design tradeoff
//! has a regime where it pays.
//!
//! ## Setup
//!
//! - 60-turn stream of similar queries.
//! - **Phase 1 (turns 1-30)**: noisy quality around mean 0.85, std 0.15
//!   (individual values swing 0.55-1.0 typically; can dip lower).
//! - **Phase 2 (turns 31-60)**: linear decay of mean from 0.85 down to 0.35
//!   over 30 turns. Same noise std 0.15.
//!
//! Deterministic xorshift PRNG for noise — re-runs are bit-identical.
//!
//! ## Three agents (same as Tier 1.7, retuned for noise)
//!
//! 1. **Naive** (reference): never adapts.
//! 2. **Smart baseline**: 5-sample rolling avg, abstain when avg < 0.55.
//! 3. **Nous-fatigue**: conservation > 0.30 OR recent_quality < 0.45 (with
//!    3-turn warmup), same calibration as Tier 1.7.
//!
//! ## Hypotheses
//!
//! - **H1**: Smart baseline false-positives in Phase 1 (noise dips trigger
//!   premature abstain). Nous's smoothing avoids this.
//! - **H2**: In Phase 2, both agents eventually detect the trend; difference
//!   is in WHEN (smart can be earlier, but its early signal in Phase 1 hurt
//!   total quality if it abstained too soon).
//! - **H3**: Net total quality favors whichever agent had right balance:
//!   resist Phase 1 noise + react to Phase 2 trend.

use noos::session::CognitiveSession;

const STREAM_LEN: usize = 60;
const PHASE_BOUNDARY: usize = 30;
const PHASE1_MEAN: f64 = 0.85;
const PHASE2_END_MEAN: f64 = 0.35;
const NOISE_STD: f64 = 0.15;
const HARM_THRESHOLD: f64 = 0.5;
const NOISE_SEED: u64 = 0xCAFE_BABE;

/// xorshift64 — deterministic, dependency-free PRNG.
fn rand_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Convert xorshift output to uniform [0, 1).
fn rand_unit(state: &mut u64) -> f64 {
    (rand_u64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Box-Muller transform for normal(0, 1).
fn rand_normal(state: &mut u64) -> f64 {
    let u1 = rand_unit(state).max(1e-10); // avoid log(0)
    let u2 = rand_unit(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Pre-compute the noisy quality stream (deterministic from NOISE_SEED).
fn generate_qualities() -> Vec<f64> {
    let mut state = NOISE_SEED;
    let mut out = Vec::with_capacity(STREAM_LEN);
    for turn in 0..STREAM_LEN {
        let mean = if turn < PHASE_BOUNDARY {
            PHASE1_MEAN
        } else {
            // Linear decay from PHASE1_MEAN at turn=PHASE_BOUNDARY to
            // PHASE2_END_MEAN at turn=STREAM_LEN-1.
            let progress = (turn - PHASE_BOUNDARY) as f64
                / (STREAM_LEN - 1 - PHASE_BOUNDARY) as f64;
            PHASE1_MEAN + (PHASE2_END_MEAN - PHASE1_MEAN) * progress
        };
        let noisy = mean + NOISE_STD * rand_normal(&mut state);
        out.push(noisy.clamp(0.0, 1.0));
    }
    out
}

fn simulate_response(turn_idx: usize) -> String {
    format!(
        "1. First, analyze the issue at turn {}.\n\
         2. Then, evaluate the options.\n\
         3. Next, select the appropriate action.\n\
         4. Finally, verify the result.",
        turn_idx
    )
}

#[derive(Debug, Default, Clone)]
struct RunResult {
    served: usize,
    abstained: usize,
    total_quality: f64,
    harm_count: usize,
    first_abstain_turn: Option<usize>,
    /// Did the agent abstain BEFORE phase 2 begins? (false positive territory)
    false_positive_phase1: bool,
}

fn run_naive(qualities: &[f64]) -> RunResult {
    let mut r = RunResult::default();
    for (turn, &quality) in qualities.iter().enumerate() {
        let _ = turn;
        r.served += 1;
        r.total_quality += quality;
        if quality < HARM_THRESHOLD {
            r.harm_count += 1;
        }
    }
    r
}

const ROLLING_WINDOW: usize = 5;
const SMART_QUALITY_THRESHOLD: f64 = 0.55;

fn run_smart_baseline(qualities: &[f64]) -> RunResult {
    let mut r = RunResult::default();
    let mut window: Vec<f64> = Vec::with_capacity(ROLLING_WINDOW);
    let mut abstaining = false;
    for (turn, &quality) in qualities.iter().enumerate() {
        if !abstaining && window.len() == ROLLING_WINDOW {
            let avg = window.iter().sum::<f64>() / window.len() as f64;
            if avg < SMART_QUALITY_THRESHOLD {
                abstaining = true;
                r.first_abstain_turn = Some(turn);
                if turn < PHASE_BOUNDARY {
                    r.false_positive_phase1 = true;
                }
            }
        }
        if abstaining {
            r.abstained += 1;
        } else {
            r.served += 1;
            r.total_quality += quality;
            if quality < HARM_THRESHOLD {
                r.harm_count += 1;
            }
            window.push(quality);
            if window.len() > ROLLING_WINDOW {
                window.remove(0);
            }
        }
    }
    r
}

const NOUS_CONSERVATION_THRESHOLD: f64 = 0.30;
const NOUS_RECENT_QUALITY_THRESHOLD: f64 = 0.45;
const QUALITY_WARMUP_TURNS: usize = 3;
const TURN_COST: f64 = 0.5;

fn run_nous_fatigue(qualities: &[f64]) -> RunResult {
    let mut session = CognitiveSession::new();
    let mut r = RunResult::default();
    let mut abstaining = false;
    for (turn, &quality) in qualities.iter().enumerate() {
        let user_msg = format!("Help me handle situation {}.", turn);
        let signals = session.process_message(&user_msg).signals;

        if !abstaining {
            let quality_armed = r.served >= QUALITY_WARMUP_TURNS;
            let should_abstain = signals.conservation > NOUS_CONSERVATION_THRESHOLD
                || (quality_armed
                    && signals.recent_quality < NOUS_RECENT_QUALITY_THRESHOLD);
            if should_abstain {
                abstaining = true;
                r.first_abstain_turn = Some(turn);
                if turn < PHASE_BOUNDARY {
                    r.false_positive_phase1 = true;
                }
            }
        }

        if abstaining {
            r.abstained += 1;
            session.track_cost(0.0);
        } else {
            r.served += 1;
            r.total_quality += quality;
            if quality < HARM_THRESHOLD {
                r.harm_count += 1;
            }
            let resp = simulate_response(turn);
            session.track_cost(TURN_COST);
            session.process_response(&resp, quality);
        }
    }
    r
}

fn print_row(name: &str, r: &RunResult) {
    let first_abstain = match r.first_abstain_turn {
        Some(t) => format!("turn {}", t + 1),
        None => "never".to_string(),
    };
    let fp_marker = if r.false_positive_phase1 {
        " (FP-Phase1)"
    } else {
        ""
    };
    println!(
        "  {:<28}  served={:>2}  abstained={:>2}  harm={:>2}  total_q={:>5.2}  first_abstain={}{}",
        name,
        r.served,
        r.abstained,
        r.harm_count,
        r.total_quality,
        first_abstain,
        fp_marker
    );
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  task_eval_fatigue_gradual — Tier 1.8 noisy gradual decay   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!(
        "Stream: {} turns. Mean quality {:.2} for first {} turns,",
        STREAM_LEN, PHASE1_MEAN, PHASE_BOUNDARY
    );
    println!(
        "linear decay to {:.2} over remaining {} turns. Noise std {:.2}.\n",
        PHASE2_END_MEAN,
        STREAM_LEN - PHASE_BOUNDARY,
        NOISE_STD
    );
    println!("Tests whether Nous's smooth signal resists Phase-1 noise");
    println!("false-positives that may trip a tight rolling-avg baseline.\n");

    let qualities = generate_qualities();

    // Show stream summary.
    let phase1_avg = qualities[..PHASE_BOUNDARY].iter().sum::<f64>() / PHASE_BOUNDARY as f64;
    let phase2_avg = qualities[PHASE_BOUNDARY..].iter().sum::<f64>()
        / (STREAM_LEN - PHASE_BOUNDARY) as f64;
    let phase1_min = qualities[..PHASE_BOUNDARY]
        .iter()
        .fold(1.0_f64, |a, b| a.min(*b));
    let phase2_min = qualities[PHASE_BOUNDARY..]
        .iter()
        .fold(1.0_f64, |a, b| a.min(*b));
    println!(
        "  Phase 1 actual: avg={:.2}, min={:.2}",
        phase1_avg, phase1_min
    );
    println!(
        "  Phase 2 actual: avg={:.2}, min={:.2}\n",
        phase2_avg, phase2_min
    );

    let naive = run_naive(&qualities);
    let smart = run_smart_baseline(&qualities);
    let nous = run_nous_fatigue(&qualities);

    println!("Per-condition results:");
    print_row("naive (reference)", &naive);
    print_row("smart baseline (rolling avg)", &smart);
    print_row("nous-fatigue", &nous);

    println!("\nFalse-positive (FP-Phase1) check:");
    println!(
        "  smart: {}  nous: {}",
        if smart.false_positive_phase1 {
            "FP — abstained in Phase 1"
        } else {
            "OK — held through Phase 1"
        },
        if nous.false_positive_phase1 {
            "FP — abstained in Phase 1"
        } else {
            "OK — held through Phase 1"
        }
    );

    println!("\nPrimary metric (total quality served — higher = better):");
    let smart_q = smart.total_quality;
    let nous_q = nous.total_quality;
    let delta = nous_q - smart_q;
    if delta > 1.0 {
        println!(
            "  ✓ Nous-fatigue beats smart baseline by {:+.2} total quality on noisy gradual.",
            delta
        );
        println!("    Nous's smoothing pays off in this regime.");
    } else if delta > 0.05 {
        println!(
            "  ≈ Nous-fatigue edges smart baseline by {:+.2} — narrow win.",
            delta
        );
    } else if delta.abs() <= 0.05 {
        println!(
            "  ≈ Tied: smart={:.2}, nous={:.2} (Δ={:+.2}). Smoothing didn't help here.",
            smart_q, nous_q, delta
        );
    } else {
        println!(
            "  ⚠ Smart baseline beats Nous by {:+.2} even on noisy gradual.",
            -delta
        );
        println!("    Either baseline already noise-robust enough, OR Nous's");
        println!("    calibration is wrong for this regime too.");
    }

    println!("\nSecondary — harm count (lower = less bad-quality delivered):");
    println!(
        "  naive={}  smart={}  nous={}",
        naive.harm_count, smart.harm_count, nous.harm_count
    );

    println!("\nNotes:");
    println!("  • Synthetic noisy gradual — illustrates regime sensitivity.");
    println!("  • If Nous wins here while losing Tier 1.7 (abrupt), gap #4");
    println!("    becomes regime-dependent rather than negative.");
    println!("  • If Nous loses both regimes, the conservation+recent_quality");
    println!("    combination needs a faster complementary signal (window-based)");
    println!("    to be useful for fatigue detection at all.");
}
