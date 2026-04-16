//! # Regulator demo 2 — Cost circuit break
//!
//! Session 22 flagship demo. End-to-end event flow showing
//! [`Decision::CircuitBreak`] firing on the
//! [`CircuitBreakReason::CostCapReached`] predicate after three retry
//! turns on an ambiguous task where quality never recovers.
//!
//! ## Scenario
//!
//! A user asks for help optimising a slow SQL query. Each agent retry
//! misses the point; quality drops 0.50 → 0.35 → 0.20. Token cost
//! accumulates at ~400 tokens per turn against a
//! [`Regulator::with_cost_cap(1_000)`](noos::Regulator::with_cost_cap).
//! After turn 3, total cost (~1200) passes the cap AND the rolling
//! mean quality is well below [`POOR_QUALITY_MEAN`] (0.50) — the
//! cost+quality compound predicate fires and the loop halts.
//!
//! ## Run
//!
//! ```bash
//! # Canned demo (deterministic, no LLM required — default):
//! cargo run --example regulator_cost_break_demo
//!
//! # Live — Ollama (requires `ollama serve` + `ollama pull phi3:mini`):
//! cargo run --example regulator_cost_break_demo -- ollama
//!
//! # Live — Anthropic (requires ANTHROPIC_API_KEY env var):
//! cargo run --example regulator_cost_break_demo -- anthropic
//! ```
//!
//! Optional env overrides match Session 21's Demo 1:
//! `NOUS_OLLAMA_URL`, `NOUS_OLLAMA_MODEL`, `NOUS_ANTHROPIC_MODEL`.
//!
//! ## What this closes that competitors cannot
//!
//! - Portkey / litellm / OpenRouter: provider-level circuit-breakers
//!   fire on **transport** failures (4xx / 5xx / rate limit). None halt
//!   on `quality × cost` **compound** signals.
//! - Langfuse / Arize / Helicone: log per-turn cost and quality, report
//!   over dashboards post-hoc. No in-loop halt decision.
//! - Generic retry loops (tenacity, backoff): blind to quality — they
//!   retry on transport failure regardless of whether retries are
//!   actually converging.
//!
//! `Regulator::decide()` returns `CircuitBreak(CostCapReached)` *during*
//! the agent loop so the app can stop burning cost before the
//! conventional retry ceiling. The suggestion string is actionable:
//! "ask the user to clarify scope or abandon this task."
//!
//! ## P10 priority note
//!
//! Early turns may also show [`Decision::ScopeDriftWarn`] (task and
//! response share only a few keywords). The regulator's priority order
//! is `CircuitBreak(CostCapReached) > CircuitBreak(QualityDecline) >
//! ScopeDriftWarn > ProceduralWarning > Continue` — so once the cost
//! cap trips on turn 3, the halt signal dominates any still-live
//! advisory. [`CircuitBreakReason::QualityDeclineNoRecovery`] would
//! also qualify on this turn (mean_delta = 0.30 ≥ 0.15 threshold, mean
//! = 0.35 < 0.5); `CostCapReached` wins by priority.

use std::env;

use noos::{CircuitBreakReason, Decision, LLMEvent, Regulator};

#[path = "regulator_common/mod.rs"]
mod regulator_common;
use regulator_common::{call_anthropic, call_ollama};

/// Cost cap for the demo. Chosen from the architecture plan's Session
/// 19 test target (1_000 tokens) so that three 400-token turns trip it
/// between turns 2 and 3 — short enough to run interactively, long
/// enough that a single turn doesn't.
const COST_CAP: u32 = 1_000;

/// Per-turn output-token count reported via [`LLMEvent::Cost`].
///
/// 400 ≈ three to four paragraphs of prose. Three such turns land
/// cumulative tokens at 1200 — comfortably above [`COST_CAP`] without
/// requiring the demo to run absurdly long turns to trigger the cap.
const PER_TURN_TOKENS_OUT: u32 = 400;

/// Typical per-turn input-token count (the user's prompt is short).
const PER_TURN_TOKENS_IN: u32 = 25;

/// Per-turn wallclock estimate for the canned path. 1.8 s is realistic
/// for a small-to-mid model; `normalize_cost` weights tokens 0.7 and
/// wallclock 0.3 so the canned wallclock doesn't dominate depletion.
const PER_TURN_WALLCLOCK_MS: u32 = 1_800;

/// Synthetic quality trajectory for the canned demo: declines monotonically
/// so [`QualityDeclineNoRecovery`] would also be eligible on turn 3,
/// demonstrating the P10 priority ordering noted in the module docs.
///
/// Values chosen so mean-of-3 = 0.35 (< [`POOR_QUALITY_MEAN`] = 0.5)
/// and first-minus-last = 0.30 (≥ [`QUALITY_DECLINE_MIN_DELTA`] = 0.15),
/// i.e. both circuit-break predicates qualify simultaneously.
const QUALITIES: &[f64] = &[0.50, 0.35, 0.20];

/// Three user prompts and their canned LLM responses. The responses
/// are realistic-if-unhelpful retries — the kind of output that would
/// genuinely earn the declining quality scores. Used by the canned
/// mode and as fallback when live LLM calls fail.
const TURNS: &[(&str, &str)] = &[
    (
        "Help me optimize this slow PostgreSQL query with a JOIN.",
        "Try adding an index on the JOIN column to help optimize your query.",
    ),
    (
        "That didn't work. Keep the same PostgreSQL JOIN but try another angle.",
        "Consider a composite index on user_id and created_at for the JOIN.",
    ),
    (
        "Still too slow. Rewrite the PostgreSQL query if needed.",
        "Rewrite it as a subquery with DISTINCT to bypass the JOIN entirely.",
    ),
];

fn main() {
    let mode = env::args().nth(1).unwrap_or_else(|| "canned".into());

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Regulator demo 2 — Cost circuit break                         ║");
    println!("║  Session 22 / Path 2 flagship                                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("Mode:      {mode}");
    println!("Cost cap:  {COST_CAP} tokens  (via Regulator::with_cost_cap)");
    println!(
        "Per turn:  ~{PER_TURN_TOKENS_OUT} tokens_out, ~{PER_TURN_WALLCLOCK_MS} ms wallclock"
    );
    println!("Quality:   declining trajectory {QUALITIES:?}\n");

    let mut regulator = Regulator::for_user("demo_user").with_cost_cap(COST_CAP);
    let mut halted = false;

    for (i, ((user_msg, canned_resp), &quality)) in
        TURNS.iter().zip(QUALITIES.iter()).enumerate()
    {
        let turn_num = i + 1;

        let (response, tokens_in, tokens_out, wallclock_ms, provider_tag) = match mode.as_str() {
            "canned" => canned_turn(canned_resp, "canned"),
            "ollama" => match call_ollama(user_msg) {
                Ok((r, ti, to, wc)) => (r, ti, to, wc, "ollama"),
                Err(e) => {
                    eprintln!("⚠  Turn {turn_num} Ollama call failed: {e}");
                    eprintln!("   Falling back to canned response for this turn.\n");
                    canned_turn(canned_resp, "canned-fallback")
                }
            },
            "anthropic" => match call_anthropic(user_msg) {
                Ok((r, ti, to, wc)) => (r, ti, to, wc, "anthropic"),
                Err(e) => {
                    eprintln!("⚠  Turn {turn_num} Anthropic call failed: {e}");
                    eprintln!("   Falling back to canned response for this turn.\n");
                    canned_turn(canned_resp, "canned-fallback")
                }
            },
            other => {
                eprintln!("Unknown mode {other:?}. Use `canned`, `ollama`, or `anthropic`.");
                std::process::exit(2);
            }
        };

        regulator.on_event(LLMEvent::TurnStart {
            user_message: (*user_msg).into(),
        });
        regulator.on_event(LLMEvent::TurnComplete {
            full_response: response.clone(),
        });
        regulator.on_event(LLMEvent::Cost {
            tokens_in,
            tokens_out,
            wallclock_ms,
            provider: Some(provider_tag.into()),
        });
        regulator.on_event(LLMEvent::QualityFeedback {
            quality,
            fragment_spans: None,
        });

        println!("──── Turn {turn_num} ────");
        println!("User:     {user_msg}");
        println!("LLM:      {response}");
        println!("Quality:  {quality:.2}  (user feedback)");
        println!(
            "Cost:     +{tokens_out} tokens  →  cumulative {}/{}",
            regulator.total_tokens_out(),
            regulator.cost_cap_tokens()
        );

        match regulator.decide() {
            Decision::Continue => {
                println!("Decision: Continue\n");
            }
            Decision::ScopeDriftWarn {
                drift_score,
                drift_tokens,
                ..
            } => {
                println!(
                    "Decision: ScopeDriftWarn (drift_score {drift_score:.2}, \
                     {} non-task keyword(s)) — advisory, app continues\n",
                    drift_tokens.len()
                );
            }
            Decision::CircuitBreak {
                reason,
                suggestion,
            } => {
                print_circuit_break(&reason, &suggestion, turn_num);
                halted = true;
                break;
            }
            Decision::ProceduralWarning { patterns } => {
                // Not expected on this demo (no repeated corrections on
                // one cluster). Handled for completeness.
                println!(
                    "Decision: ProceduralWarning ({} pattern(s)) — advisory, app continues\n",
                    patterns.len()
                );
            }
            Decision::LowConfidenceSpans { spans } => {
                println!(
                    "Decision: LowConfidenceSpans ({} span(s)) — advisory, app continues\n",
                    spans.len()
                );
            }
            other => {
                // Future Decision variants — keep the demo running rather
                // than panic if the enum grows.
                println!("Decision: {other:?} — not exercised by this demo\n");
            }
        }
    }

    println!();
    println!("──── Take-away ────");
    if halted {
        println!("Regulator halted the loop after a cost × quality compound signal.");
        println!("Baseline agent (no regulator): keeps retrying inside a generic backoff loop,");
        println!("burning tokens because none of its predicates watch response quality.");
        println!("Portkey / litellm / OpenRouter trip on transport errors only (4xx / 5xx /");
        println!("rate-limit) — not on \"three low-quality turns in a row\". Langfuse / Arize");
        println!("surface the same data, but only after the loop has run.");
    } else {
        println!("(Loop ran to completion without a circuit break on this run.)");
        println!("(Try the default `canned` mode to see CircuitBreak(CostCapReached) fire");
        println!(" with the plan-target values.)");
    }
}

/// Render the CircuitBreak in a form the reader can cross-check against
/// the `cost.rs` module doc thresholds.
fn print_circuit_break(reason: &CircuitBreakReason, suggestion: &str, turn_num: usize) {
    println!("Decision: [CircuitBreak]  ←  HALT");
    match reason {
        CircuitBreakReason::CostCapReached {
            tokens_spent,
            tokens_cap,
            mean_quality_last_n,
        } => {
            println!("  reason:              CostCapReached");
            println!("  tokens_spent:        {tokens_spent}");
            println!("  tokens_cap:          {tokens_cap}");
            println!(
                "  mean_quality_last_n: {mean_quality_last_n:.2}  \
                 (threshold POOR_QUALITY_MEAN = 0.50)"
            );
        }
        CircuitBreakReason::QualityDeclineNoRecovery { turns, mean_delta } => {
            println!("  reason:              QualityDeclineNoRecovery");
            println!("  turns:               {turns}");
            println!(
                "  mean_delta:          {mean_delta:.2}  \
                 (threshold QUALITY_DECLINE_MIN_DELTA = 0.15)"
            );
        }
        CircuitBreakReason::RepeatedFailurePattern {
            cluster,
            failure_count,
        } => {
            println!("  reason:              RepeatedFailurePattern");
            println!("  cluster:             {cluster}");
            println!("  failure_count:       {failure_count}");
        }
        other => {
            // Future CircuitBreakReason variants — render generically.
            println!("  reason:              {other:?}");
        }
    }
    println!("  suggestion:          {suggestion}");
    println!();
    println!(
        "Turn {turn_num}: agent halts. The app surfaces the suggestion and \
         stops retrying."
    );
}

// ── Canned fallback ──────────────────────────────────────────────────

/// Single source (P3) for the canned per-turn tuple, parameterised by
/// the specific response text and provider tag the caller wants printed.
/// Used by the `canned` mode and by both live-mode transport-failure
/// fallbacks. Token counts come from [`PER_TURN_TOKENS_IN`] /
/// [`PER_TURN_TOKENS_OUT`] / [`PER_TURN_WALLCLOCK_MS`] so any change to
/// the demo budget envelope updates in one place.
fn canned_turn(
    response: &str,
    provider_tag: &'static str,
) -> (String, u32, u32, u32, &'static str) {
    (
        response.to_string(),
        PER_TURN_TOKENS_IN,
        PER_TURN_TOKENS_OUT,
        PER_TURN_WALLCLOCK_MS,
        provider_tag,
    )
}

// HTTP adapters live in `regulator_common/mod.rs` (P3 single source for
// Sessions 21-23 demos).
