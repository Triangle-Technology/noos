//! Tier 1.7 fatigue eval — does Nous's combined signal detect late-onset
//! quality degradation faster than a rolling-quality-average baseline?
//!
//! Run: `cargo run --example task_eval_fatigue`
//!
//! ## Why this eval matters
//!
//! `docs/intervention.md` lists "context rot / fatigue (all 18 frontier
//! models degrade with length, none detect it — Chroma 2025)" as gap #4.
//! Nous's claim: `signals.conservation` + `recent_quality` + `rpe` together
//! signal "things are degrading" earlier than any single rolling metric.
//!
//! Tier 1.7 tests this by simulating an abrupt quality drop mid-stream
//! (turn 25 of 50) and measuring how quickly each agent recognizes the
//! degradation and switches behavior.
//!
//! ## Setup
//!
//! - 50-turn stream of similar queries (same category, same strategy).
//! - **Phase 1 (turns 1-25)**: model produces normal quality (0.85).
//! - **Phase 2 (turns 26-50)**: model degrades to low quality (0.40).
//!   Simulates context rot, attention dilution, model fatigue, etc.
//!
//! Both Nous and the smart baseline have access to per-turn quality via
//! their normal feedback channels. The question: does Nous's compound
//! signal cross its decision threshold in FEWER turns post-degradation than
//! the smart baseline's rolling average crosses its threshold?
//!
//! ## Three agents
//!
//! 1. **Naive** (reference): never adapts. Serves all 50, gets late-phase
//!    low quality without recognizing it. Floor for "delivered harm."
//!
//! 2. **Smart baseline (rolling avg)**: tracks last-N quality scores,
//!    switches to abstain when rolling avg drops below threshold. This
//!    is what an engineer would write — direct quality monitoring.
//!
//! 3. **Nous-fatigue**: monitors `signals.conservation` (which integrates
//!    body_budget depletion + sustained arousal + resource pressure) AND
//!    `recent_quality` together. Switches to abstain when either exceeds
//!    its threshold.
//!
//! ## Metrics
//!
//! - **Detection latency**: turns from degradation onset (turn 26) to first
//!   decision to abstain. Lower = better.
//! - **Harm count**: low-quality responses (q < 0.5) delivered to user.
//!   Lower = better.
//! - **Total quality**: sum of quality served (abstentions = 0).
//!   Higher = better but only if abstention timing is right.

use nous_regulator::session::CognitiveSession;

const STREAM_LEN: usize = 50;
const DEGRADATION_TURN: usize = 25; // turns 0..24 high quality, 25..49 low
const HIGH_QUALITY: f64 = 0.85;
const LOW_QUALITY: f64 = 0.40;
const HARM_THRESHOLD: f64 = 0.5;

fn simulate_llm(turn_idx: usize) -> (String, f64) {
    let quality = if turn_idx < DEGRADATION_TURN {
        HIGH_QUALITY
    } else {
        LOW_QUALITY
    };
    // Realistic StepByStep response (matches detect_response_strategy).
    let text = format!(
        "1. First, analyze the issue at turn {}.\n\
         2. Then, evaluate the options.\n\
         3. Next, select the appropriate action.\n\
         4. Finally, verify the result.",
        turn_idx
    );
    (text, quality)
}

#[derive(Debug, Default, Clone)]
struct RunResult {
    served: usize,
    abstained: usize,
    total_quality: f64,
    harm_count: usize,
    first_abstain_turn: Option<usize>,
}

impl RunResult {
    /// Detection latency: turns from degradation onset (turn 25) to first abstain.
    /// Returns None if agent never abstained.
    fn detection_latency(&self) -> Option<usize> {
        self.first_abstain_turn
            .filter(|&t| t >= DEGRADATION_TURN)
            .map(|t| t - DEGRADATION_TURN)
    }
}

// ─── Agent 1: Naive (never adapts) ────────────────────────────────────────

fn run_naive() -> RunResult {
    let mut r = RunResult::default();
    for turn in 0..STREAM_LEN {
        let (_resp, quality) = simulate_llm(turn);
        r.served += 1;
        r.total_quality += quality;
        if quality < HARM_THRESHOLD {
            r.harm_count += 1;
        }
    }
    r
}

// ─── Agent 2: Smart baseline (rolling-quality average) ────────────────────
//
// Maintains a window of recent qualities. When the window's average drops
// below threshold, switches to abstain mode (and stays there — hysteresis).
// This is the canonical "monitor your output quality" pattern.

const ROLLING_WINDOW: usize = 5;
const SMART_QUALITY_THRESHOLD: f64 = 0.55;

fn run_smart_baseline() -> RunResult {
    let mut r = RunResult::default();
    let mut window: Vec<f64> = Vec::with_capacity(ROLLING_WINDOW);
    let mut abstaining = false;
    for turn in 0..STREAM_LEN {
        if !abstaining && window.len() == ROLLING_WINDOW {
            let avg = window.iter().sum::<f64>() / window.len() as f64;
            if avg < SMART_QUALITY_THRESHOLD {
                abstaining = true;
                r.first_abstain_turn = Some(turn);
            }
        }
        if abstaining {
            r.abstained += 1;
        } else {
            let (_resp, quality) = simulate_llm(turn);
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

// ─── Agent 3: Nous-fatigue (combined conservation + recent_quality) ──────

// Calibration note (fixed 2026-04-14 after first-run bug):
// `signals.recent_quality` defaults to 0.5 (the EMA initial value); using
// threshold 0.55 caused immediate-turn-1 abstain. Setting threshold to 0.45
// keeps the signal "armed" through normal operation (recent_quality stays
// near 0.5 or higher when responses succeed) and only fires when sustained
// low quality pulls the EMA down below 0.45.
//
// Conservation default starts near 0.045 (resource_pressure baseline floor);
// 0.30 threshold gives meaningful headroom for sustained-stress accumulation.
const NOUS_CONSERVATION_THRESHOLD: f64 = 0.30;
const NOUS_RECENT_QUALITY_THRESHOLD: f64 = 0.45;
const TURN_COST: f64 = 0.5;
/// Minimum turns of feedback before quality-based decisions are valid.
/// Avoids triggering on the EMA initial state before real data arrives.
const QUALITY_WARMUP_TURNS: usize = 3;

fn run_nous_fatigue() -> RunResult {
    let mut session = CognitiveSession::new();
    let mut r = RunResult::default();
    let mut abstaining = false;
    for turn in 0..STREAM_LEN {
        let user_msg = format!("Help me handle situation {}.", turn);
        let signals = session.process_message(&user_msg).signals;

        if !abstaining {
            // Quality-based abstain only valid AFTER warmup — recent_quality
            // EMA starts at 0.5 default, so early turns would falsely trigger
            // a < 0.55 threshold (first-run bug). Conservation signal is valid
            // immediately because it starts at the resource_pressure floor.
            let quality_armed = r.served >= QUALITY_WARMUP_TURNS;
            let should_abstain = signals.conservation > NOUS_CONSERVATION_THRESHOLD
                || (quality_armed
                    && signals.recent_quality < NOUS_RECENT_QUALITY_THRESHOLD);
            if should_abstain {
                abstaining = true;
                r.first_abstain_turn = Some(turn);
            }
        }

        if abstaining {
            r.abstained += 1;
            session.track_cost(0.0);
        } else {
            let (resp, quality) = simulate_llm(turn);
            r.served += 1;
            r.total_quality += quality;
            if quality < HARM_THRESHOLD {
                r.harm_count += 1;
            }
            session.track_cost(TURN_COST);
            session.process_response(&resp, quality);
        }
    }
    r
}

// ─── Reporting ────────────────────────────────────────────────────────────

fn print_row(name: &str, r: &RunResult) {
    let detection = match r.detection_latency() {
        Some(d) => format!("{} turns", d),
        None => "never".to_string(),
    };
    let first_abstain = match r.first_abstain_turn {
        Some(t) => format!("turn {}", t + 1),
        None => "never".to_string(),
    };
    println!(
        "  {:<28}  served={:>2}  abstained={:>2}  harm={:>2}  total_q={:>5.2}  first_abstain={}  detection_latency={}",
        name,
        r.served,
        r.abstained,
        r.harm_count,
        r.total_quality,
        first_abstain,
        detection
    );
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  task_eval_fatigue — Tier 1.7 late-onset degradation test    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!(
        "Stream: {} turns. Quality drops from {:.2} to {:.2} at turn {}.",
        STREAM_LEN,
        HIGH_QUALITY,
        LOW_QUALITY,
        DEGRADATION_TURN + 1
    );
    println!("Tests whether Nous detects the degradation in fewer turns than");
    println!("a rolling-quality-avg baseline (window={}, threshold={:.2}).\n",
        ROLLING_WINDOW, SMART_QUALITY_THRESHOLD);

    let naive = run_naive();
    let smart = run_smart_baseline();
    let nous = run_nous_fatigue();

    println!("Per-condition results:");
    print_row("naive (reference)", &naive);
    print_row("smart baseline (rolling avg)", &smart);
    print_row("nous-fatigue", &nous);

    println!("\nPrimary metric — detection latency (lower = caught faster):");
    let smart_latency = smart.detection_latency();
    let nous_latency = nous.detection_latency();
    match (smart_latency, nous_latency) {
        (Some(s), Some(n)) if n + 2 <= s => println!(
            "  ✓ Nous detected in {} turns; smart baseline took {} turns. Nous {} turns earlier.",
            n, s, s - n
        ),
        (Some(s), Some(n)) if n <= s + 1 && s <= n + 1 => println!(
            "  ≈ Tied (or near): smart {} turns, nous {} turns post-degradation.",
            s, n
        ),
        (Some(s), Some(n)) => println!(
            "  ⚠ Smart detected faster: {} vs {} turns.",
            s, n
        ),
        (None, Some(n)) => println!(
            "  ✓ Nous detected in {} turns; smart baseline never detected.",
            n
        ),
        (Some(s), None) => println!(
            "  ⚠ Smart detected in {} turns; nous never detected.",
            s
        ),
        (None, None) => println!("  Neither agent detected the degradation."),
    }

    println!("\nSecondary — harm count (low-quality responses delivered to user):");
    println!(
        "  naive={}  smart={}  nous={}  (lower = less harm)",
        naive.harm_count, smart.harm_count, nous.harm_count
    );

    println!("\nNotes:");
    println!("  • Synthetic late-onset degradation — illustrates whether Nous's");
    println!("    combined signal is sensitive enough for fatigue detection.");
    println!("  • Both Nous and smart baseline see the same quality stream; the");
    println!("    test is signal sensitivity, not information access.");
    println!("  • If Nous detects ≥2 turns earlier than rolling avg, the");
    println!("    combined-signal claim has measurable support on this task.");
    println!("  • If Nous matches or lags rolling avg, the conservation+recent_quality");
    println!("    combination is no faster than a direct quality monitor.");
}
