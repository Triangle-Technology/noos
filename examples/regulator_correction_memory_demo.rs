//! # Regulator demo 3 — Procedural correction memory
//!
//! Session 23 flagship demo, and the clearest differentiator from
//! Mem0 / Letta / LangChain memory per research agent 3. Shows:
//!
//! 1. A user corrects the agent 3 times on the same topic cluster.
//! 2. [`Regulator::export`](nous_regulator::Regulator::export) snapshots the
//!    learned [`CorrectionPattern`](nous_regulator::CorrectionPattern) to a
//!    serialisable [`RegulatorState`](nous_regulator::RegulatorState).
//! 3. The snapshot round-trips through `serde_json` — simulating a
//!    process restart / next session.
//! 4. [`Regulator::import`](nous_regulator::Regulator::import) restores the
//!    pattern; on the next [`LLMEvent::TurnStart`] in the same cluster,
//!    [`Decision::ProceduralWarning`] fires **before** generation.
//!
//! ## Scenario
//!
//! User `user_123` sends four small refactor asks across four sessions
//! (Sessions 1-3 in Phase 1, Session 4 post-restart in Phase 3). All
//! four messages hash to the same topic cluster `async+auth` via
//! [`build_topic_cluster`](nous_regulator::cognition). After the first three
//! sessions, the user has corrected the agent three times about adding
//! docstrings. On session 4, before the LLM even generates, the
//! regulator surfaces the learned rule with `example_corrections`
//! attached.
//!
//! ## Run
//!
//! ```bash
//! # Canned (deterministic, no LLM required — default):
//! cargo run --example regulator_correction_memory_demo
//!
//! # Live — Ollama / Anthropic share the same Sessions 21-22 plumbing:
//! cargo run --example regulator_correction_memory_demo -- ollama
//! cargo run --example regulator_correction_memory_demo -- anthropic
//! ```
//!
//! ## What this closes that competitors cannot
//!
//! - **Mem0 / Letta / LangChain memory**: store each correction message
//!   verbatim; on every turn run a semantic-search query to retrieve
//!   "similar past corrections". Effectiveness depends on embedding
//!   quality, similarity threshold, recall tuning. **No pattern
//!   extraction** — content is preserved but the behavioral rule is
//!   not explicit.
//! - **Nous**: structurally counts per-cluster corrections. Once
//!   [`MIN_CORRECTIONS_FOR_PATTERN`](nous_regulator::CorrectionStore) is reached,
//!   a [`CorrectionPattern`] is exposed proactively via
//!   [`Decision::ProceduralWarning`] — no retrieval query, no semantic
//!   similarity threshold. Raw example texts ride along so the app /
//!   LLM can read the intent at generation time.
//!
//! This is the "extract behavioral patterns, not just store content"
//! differentiation. The pattern is language-neutral (P9b): no English
//! regex parses correction text into a rule — the app / LLM does that
//! interpretation by reading `example_corrections`.

use std::env;

use nous_regulator::regulator::correction::MIN_CORRECTIONS_FOR_PATTERN as MIN_FOR_PATTERN;
use nous_regulator::{Decision, LLMEvent, Regulator, RegulatorState};

#[path = "regulator_common/mod.rs"]
mod regulator_common;
use regulator_common::{call_anthropic, call_ollama};

/// User identity threaded through Phases 1-3. Matches the Session 23
/// scenario in `docs/regulator-design.md` §7 Demo 3 ("user_123").
const USER_ID: &str = "user_123";

/// Four user messages. Each extracts to a top-2 keyword bag that
/// hashes to the same cluster `async+auth` via `build_topic_cluster`
/// (verified empirically before writing the demo).
///
/// Messages 0-2 are the **learning** turns; message 3 is the
/// **post-import** turn that should trigger `ProceduralWarning`.
const USER_MESSAGES: &[&str] = &[
    "Make my auth module async",
    "Refactor auth to support async",
    "Change my auth function to async",
    "Add async handling to my auth",
];

/// Canned LLM responses used by the default mode. Each includes a
/// docstring the user will complain about. Live modes generate real
/// responses per turn (quality varies by model).
const CANNED_RESPONSES: &[&str] = &[
    "async fn auth() { /* ... */ }\n/// This function authenticates requests.",
    "pub async fn auth() -> Result<User, AuthError> { /* ... */ }\n/// Authenticates the incoming user.",
    "async fn auth_validate() { /* ... */ }\n/// Params: token (&str). Returns Session.",
];

/// Correction messages the user sends across Phase 1 turns. Recorded
/// under the active cluster via `LLMEvent::UserCorrection { corrects_last:
/// true }`. Note how each phrases the same rule differently — this is
/// the input Nous extracts a STRUCTURAL pattern from, without parsing
/// any of them for English "don't X" constructs.
const CORRECTIONS: &[&str] = &[
    "Don't add docstrings to the refactor output please",
    "Skip the doc comments — just show the code",
    "No new docstrings. I want minimal diffs this time",
];

/// Canned per-turn cost accounting. Kept modest so the cost-cap
/// predicate does NOT fire — this demo is about procedural memory, not
/// circuit breaks.
const PER_TURN_TOKENS_IN: u32 = 25;
const PER_TURN_TOKENS_OUT: u32 = 120;
const PER_TURN_WALLCLOCK_MS: u32 = 900;

fn main() {
    let mode = env::args().nth(1).unwrap_or_else(|| "canned".into());

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Regulator demo 3 — Procedural correction memory               ║");
    println!("║  Session 23 / Path 2 flagship (clearest Mem0/Letta contrast)    ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("Mode: {mode}");
    println!("User: {USER_ID}\n");

    // ── Phase 1: Learn the pattern ────────────────────────────────
    println!("══════ Phase 1: Learning ({} corrections incoming) ══════\n", CORRECTIONS.len());

    let mut regulator = Regulator::for_user(USER_ID);

    for i in 0..CORRECTIONS.len() {
        let turn_num = i + 1;
        let user_msg = USER_MESSAGES[i];
        let correction = CORRECTIONS[i];

        let (response, tokens_in, tokens_out, wallclock_ms, provider_tag) =
            match mode.as_str() {
                "canned" => canned_turn(CANNED_RESPONSES[i], "canned"),
                "ollama" => match call_ollama(user_msg) {
                    Ok((r, ti, to, wc)) => (r, ti, to, wc, "ollama"),
                    Err(e) => {
                        eprintln!("⚠  Turn {turn_num} Ollama failed: {e}; fallback\n");
                        canned_turn(CANNED_RESPONSES[i], "canned-fallback")
                    }
                },
                "anthropic" => match call_anthropic(user_msg) {
                    Ok((r, ti, to, wc)) => (r, ti, to, wc, "anthropic"),
                    Err(e) => {
                        eprintln!("⚠  Turn {turn_num} Anthropic failed: {e}; fallback\n");
                        canned_turn(CANNED_RESPONSES[i], "canned-fallback")
                    }
                },
                other => {
                    eprintln!("Unknown mode {other:?}. Use canned|ollama|anthropic.");
                    std::process::exit(2);
                }
            };

        // Full per-turn event cycle.
        regulator.on_event(LLMEvent::TurnStart {
            user_message: user_msg.into(),
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
        regulator.on_event(LLMEvent::UserCorrection {
            correction_message: correction.into(),
            corrects_last: true,
        });

        println!("── Turn {turn_num} ──");
        println!("User:       {user_msg}");
        println!("LLM:        {response}");
        println!("User corrects: {correction}");

        // Read-back: below threshold `correction_patterns` is empty;
        // at/above threshold it contains a `CorrectionPattern` whose
        // `learned_from_turns` equals the count we've recorded so far.
        let snapshot = regulator.export();
        let threshold_reached = !snapshot.correction_patterns.is_empty();
        println!(
            "Corrections so far on this cluster: {turn_num} / {MIN_FOR_PATTERN} (threshold: {})",
            if threshold_reached { "REACHED" } else { "not yet" }
        );
        println!();
    }

    // ── Phase 2: Persist + simulate process restart ───────────────
    println!("══════ Phase 2: Persist → restart ══════\n");

    let state: RegulatorState = regulator.export();
    println!(
        "Exported state has {} correction pattern(s) for {USER_ID}:",
        state.correction_patterns.len()
    );
    for (cluster, pattern) in &state.correction_patterns {
        print_pattern(cluster, pattern, "  ");
    }

    // P5: handle serde errors gracefully (no `.expect` on runtime-valid
    // data; reserved for compile-time-valid cases per principle).
    let json = match serde_json::to_string(&state) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Could not serialise RegulatorState: {e}");
            std::process::exit(1);
        }
    };
    println!("\nJSON snapshot: {} bytes.", json.len());
    println!("(A real app persists this to disk / database / session store.)\n");

    // Simulate a fresh process: drop the old regulator, deserialise,
    // rebuild via `Regulator::import`.
    drop(regulator);
    let restored_state: RegulatorState = match serde_json::from_str(&json) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Could not deserialise RegulatorState: {e}");
            std::process::exit(1);
        }
    };
    let mut regulator = Regulator::import(restored_state);
    println!("Regulator restored from snapshot.\n");

    // ── Phase 3: Next-session turn fires ProceduralWarning ────────
    println!("══════ Phase 3: Next session — ProceduralWarning pre-generation ══════\n");

    let next_msg = USER_MESSAGES[3];
    println!("User (new session): {next_msg}");

    // IMPORTANT: call `decide()` AFTER `TurnStart` but BEFORE any
    // `TurnComplete`. Scope tracker has task keywords only — no
    // response yet — so `drift_score` returns None and
    // `ScopeDriftWarn` skips. ProceduralWarning can then surface.
    regulator.on_event(LLMEvent::TurnStart {
        user_message: next_msg.into(),
    });

    match regulator.decide() {
        Decision::ProceduralWarning { patterns } => {
            println!(
                "Regulator (pre-generation): [ProceduralWarning]  — {} pattern(s) apply",
                patterns.len()
            );
            for pattern in &patterns {
                print_pattern(&pattern.topic_cluster, pattern, "  ");
            }
            println!();
            println!("The app / LLM can now read these examples BEFORE generating,");
            println!("avoiding the same class of mistake this user has corrected repeatedly.");
        }
        Decision::Continue => {
            println!("(Unexpected) Regulator returned Continue.");
            println!("Check whether the new message hashes to the same cluster as Phase 1.");
        }
        other => {
            println!("(Unexpected) Regulator returned {other:?}");
        }
    }

    // ── Take-away ─────────────────────────────────────────────────
    println!();
    println!("══════ Take-away ══════");
    println!("Baseline (Mem0 / Letta / LangChain memory):");
    println!("  • store every correction message verbatim;");
    println!("  • on every new turn, run a semantic-search query to retrieve");
    println!("    similar past corrections;");
    println!("  • effectiveness depends on embedding quality + similarity threshold.");
    println!();
    println!("Regulator:");
    println!("  • counts per-cluster corrections structurally (no embedding);");
    println!("  • fires ProceduralWarning proactively once MIN threshold trips;");
    println!("  • raw example_corrections ride along so the LLM can read intent;");
    println!("  • P9b-compliant: pattern_name is opaque (`corrections_on_{{cluster}}`),");
    println!("    no English regex parses the correction text for rule extraction.");
    println!();
    println!("This is the \"extract behavioral patterns, not just store content\"");
    println!("differentiation — demonstrably absent from every content-retrieval");
    println!("memory system in the Rust or Python LLM ecosystem as of 2026-04.");
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Single source (P3) for the canned per-turn tuple, parameterised by
/// the specific response text + provider tag. Same pattern as the
/// Session 22 demo — token / wallclock numbers come from module
/// constants so the whole demo's budget envelope changes in one place.
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

/// Print a `CorrectionPattern` in a reader-friendly shape. Used by
/// both Phase 2 (exported state preview) and Phase 3 (post-warning
/// pattern display) so the two displays are consistent.
fn print_pattern(cluster: &str, pattern: &nous_regulator::CorrectionPattern, indent: &str) {
    println!("{indent}cluster:            {cluster}");
    println!("{indent}  pattern_name:     {}", pattern.pattern_name);
    println!(
        "{indent}  learned_from_turns: {}",
        pattern.learned_from_turns
    );
    println!("{indent}  confidence:       {:.2}", pattern.confidence);
    println!(
        "{indent}  example_corrections ({}):",
        pattern.example_corrections.len()
    );
    for ex in &pattern.example_corrections {
        println!("{indent}    • {ex}");
    }
}
