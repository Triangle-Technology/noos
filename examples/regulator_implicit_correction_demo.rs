//! Demo 5 — Implicit correction detection (Session 32).
//!
//! Most chat UIs do NOT emit an explicit
//! [`LLMEvent::UserCorrection`] when the user re-asks. They just
//! submit another message. Without implicit detection, the
//! [`Decision::ProceduralWarning`] path never activates for those
//! apps — the strongest wedge of the regulator (procedural memory
//! from corrections) stays inert.
//!
//! This demo shows the implicit correction detector closing that gap:
//! when a retry arrives within a configured window AND maps to the
//! same topic cluster, Noos synthesises a correction record against
//! that cluster. After the threshold trips, a pre-generation
//! `decide()` call fires `ProceduralWarning` with learned examples
//! attached.
//!
//! ## What it shows
//!
//! Three scenarios, each run with
//! `Regulator::with_implicit_correction_window(500ms)`:
//!
//! 1. **Three fast same-cluster retries** → implicit counter = 3,
//!    `ProceduralWarning` fires with 3 `example_corrections` on the
//!    4th turn.
//! 2. **Retry after the window expires** → counter stays at 0
//!    (temporal gate fails closed).
//! 3. **Retry on a different topic cluster within the window** →
//!    counter stays at 0 (topic gate fails closed).
//!
//! Both gates are required — temporal proximity AND topic continuity.
//!
//! ## Run
//!
//! ```bash
//! cargo run --example regulator_implicit_correction_demo
//! ```
//!
//! The demo is canned-only (no LLM required) because implicit
//! correction is a timing + cluster signal, not a model signal —
//! swapping in live LLM responses wouldn't change what the feature
//! surfaces.

use std::thread::sleep;
use std::time::Duration;

use noos::{Decision, LLMEvent, Regulator};

/// Implicit-correction window used by all three scenarios. Short
/// enough (500 ms) to keep the demo wall-clock under one second, long
/// enough that `thread::sleep` jitter on a loaded CI runner doesn't
/// accidentally expire the window in scenarios 1 and 3.
const WINDOW_MS: u64 = 500;

/// Pause between `TurnComplete` and the next `TurnStart` in the
/// "fires on fast retry" + "different cluster" scenarios. Well under
/// WINDOW_MS so the temporal gate passes.
const QUICK_WAIT_MS: u64 = 20;

/// Pause in the "window-expired" scenario. Just over WINDOW_MS so the
/// temporal gate fails closed, with enough margin that normal OS
/// scheduling jitter doesn't accidentally land inside the window.
const PAST_WINDOW_WAIT_MS: u64 = 600;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Demo 5 — Implicit correction (Session 32)                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    println!(
        "Regulator window: {} ms\nAll three scenarios use the same `Regulator::for_user('alice')\n.with_implicit_correction_window(500ms)` builder.\n",
        WINDOW_MS
    );

    scenario_1_fast_retries_build_pattern();
    scenario_2_window_expiry_fails_closed();
    scenario_3_different_cluster_fails_closed();

    println!("\n── Takeaway ───────────────────────────────────────────────────");
    println!("Implicit correction detection is the adoption unlock for");
    println!("procedural memory: apps don't have to wire explicit");
    println!("`LLMEvent::UserCorrection` events — the regulator detects retries");
    println!("structurally from timing + topic continuity, then feeds the");
    println!("correction store which drives ProceduralWarning pre-generation.");
    println!("Compared to content-memory stores (Mem0, Letta, Zep) which");
    println!("require semantic search at retrieval time, the pattern fires");
    println!("BEFORE generation — the app can use `inject_corrections` to");
    println!("prepend learned examples to the next LLM prompt.");
}

fn scenario_1_fast_retries_build_pattern() {
    println!("── Scenario 1 — three fast same-cluster retries ───────────────");
    let mut r = Regulator::for_user("alice")
        .with_implicit_correction_window(Duration::from_millis(WINDOW_MS));

    // Turn 1 — no prior complete, no implicit correction. Sets the
    // topic cluster to "async+fetch_user" (top-2 alphabetical of
    // meaningful words async, fetch_user, refactor).
    println!("  turn 1: 'Refactor fetch_user to be async'");
    r.on_event(LLMEvent::TurnStart {
        user_message: "Refactor fetch_user to be async".into(),
    });
    r.on_event(LLMEvent::TurnComplete {
        full_response: "(unsatisfactory response 1)".into(),
    });

    // Turn 2 — retry within window, same cluster. Synthetic
    // correction recorded. Message constructed to preserve top-2
    // = {async, fetch_user}: {fix, fetch_user, async, refactoring}
    // sorts alphabetically (async, fetch_user, fix, refactoring).
    sleep(Duration::from_millis(QUICK_WAIT_MS));
    println!("  turn 2: 'Fix the fetch_user async refactoring'  [+20ms]");
    r.on_event(LLMEvent::TurnStart {
        user_message: "Fix the fetch_user async refactoring".into(),
    });
    println!(
        "    → implicit_corrections_count = {}",
        r.implicit_corrections_count()
    );
    r.on_event(LLMEvent::TurnComplete {
        full_response: "(unsatisfactory response 2)".into(),
    });

    // Turn 3 — another retry, same cluster. {make, fetch_user, async,
    // properly}.
    sleep(Duration::from_millis(QUICK_WAIT_MS));
    println!("  turn 3: 'Make fetch_user async properly'        [+20ms]");
    r.on_event(LLMEvent::TurnStart {
        user_message: "Make fetch_user async properly".into(),
    });
    println!(
        "    → implicit_corrections_count = {}",
        r.implicit_corrections_count()
    );
    r.on_event(LLMEvent::TurnComplete {
        full_response: "(unsatisfactory response 3)".into(),
    });

    // Turn 4 — the 4th start fires the 3rd implicit correction. Count
    // on cluster "async+fetch_user" hits 3 = MIN_CORRECTIONS_FOR_PATTERN.
    // `decide()` BEFORE TurnComplete (pre-generation) fires the warning.
    sleep(Duration::from_millis(QUICK_WAIT_MS));
    println!("  turn 4: 'Update fetch_user to async version'    [+20ms]");
    r.on_event(LLMEvent::TurnStart {
        user_message: "Update fetch_user to async version".into(),
    });
    println!(
        "    → implicit_corrections_count = {}",
        r.implicit_corrections_count()
    );

    match r.decide() {
        Decision::ProceduralWarning { patterns } => {
            println!("    → decide() = ProceduralWarning");
            for p in &patterns {
                println!(
                    "      pattern: {} (learned_from_turns={}, confidence={:.2})",
                    p.pattern_name, p.learned_from_turns, p.confidence
                );
                for (i, ex) in p.example_corrections.iter().enumerate() {
                    println!("        example[{}]: {:?}", i, ex);
                }
            }
        }
        other => println!("    → decide() = {other:?}  [unexpected — pattern should have fired]"),
    }
    println!();
}

fn scenario_2_window_expiry_fails_closed() {
    println!("── Scenario 2 — retry outside the window ──────────────────────");
    let mut r = Regulator::for_user("bob")
        .with_implicit_correction_window(Duration::from_millis(WINDOW_MS));

    println!("  turn 1: 'Refactor fetch_user to be async'");
    r.on_event(LLMEvent::TurnStart {
        user_message: "Refactor fetch_user to be async".into(),
    });
    r.on_event(LLMEvent::TurnComplete {
        full_response: "resp".into(),
    });

    println!("  waiting {PAST_WINDOW_WAIT_MS}ms (past the {WINDOW_MS}ms window)...");
    sleep(Duration::from_millis(PAST_WINDOW_WAIT_MS));

    println!("  turn 2: 'Fix the fetch_user async refactoring'  [too late]");
    r.on_event(LLMEvent::TurnStart {
        user_message: "Fix the fetch_user async refactoring".into(),
    });
    println!(
        "    → implicit_corrections_count = {}  (temporal gate failed closed)",
        r.implicit_corrections_count()
    );
    println!();
}

fn scenario_3_different_cluster_fails_closed() {
    println!("── Scenario 3 — fast retry, different topic cluster ───────────");
    let mut r = Regulator::for_user("carol")
        .with_implicit_correction_window(Duration::from_millis(WINDOW_MS));

    println!("  turn 1: 'Refactor fetch_user to be async'  [cluster: async+fetch_user]");
    r.on_event(LLMEvent::TurnStart {
        user_message: "Refactor fetch_user to be async".into(),
    });
    r.on_event(LLMEvent::TurnComplete {
        full_response: "resp".into(),
    });

    sleep(Duration::from_millis(QUICK_WAIT_MS));
    // Different cluster: {explain, tokio, channels, scheduling} →
    // top-2 = {channels, explain} → cluster "channels+explain"
    println!(
        "  turn 2: 'Explain tokio channels and scheduling'  [cluster: channels+explain]"
    );
    r.on_event(LLMEvent::TurnStart {
        user_message: "Explain tokio channels and scheduling".into(),
    });
    println!(
        "    → implicit_corrections_count = {}  (topic gate failed closed)",
        r.implicit_corrections_count()
    );
    println!();
}
