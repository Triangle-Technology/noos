//! # Regulator demo 1 — Scope drift intercept
//!
//! Session 21 flagship demo. End-to-end event flow into [`Regulator`] showing
//! how [`Decision::ScopeDriftWarn`] surfaces non-task keywords *before* the
//! response reaches the user.
//!
//! ## Scenario
//!
//! A user asks an LLM to refactor a small function to be async. A typical
//! over-eager LLM response adds logging, error handling, and telemetry —
//! none of which were requested. The regulator catches the drift; a
//! baseline Mem0-style agent would pass the expanded response through
//! and leave the user to notice.
//!
//! ## Run
//!
//! ```bash
//! # Canned demo (deterministic, no LLM required — default):
//! cargo run --example regulator_scope_drift_demo
//!
//! # Live — Ollama (requires `ollama serve` + `ollama pull phi3:mini`):
//! cargo run --example regulator_scope_drift_demo -- ollama
//!
//! # Live — Anthropic (requires ANTHROPIC_API_KEY env var):
//! cargo run --example regulator_scope_drift_demo -- anthropic
//! ```
//!
//! Optional env overrides:
//!
//! - `NOUS_OLLAMA_URL` (default `http://localhost:11434/api/chat`)
//! - `NOUS_OLLAMA_MODEL` (default `phi3:mini`)
//! - `NOUS_ANTHROPIC_MODEL` (default `claude-haiku-4-5-20251001`)
//!
//! ## What this closes that competitors cannot
//!
//! - Mem0 / Letta / LangChain memory: **store** interaction content and
//!   retrieve it post-hoc via semantic search. None detect drift from task
//!   to response in real time.
//! - Langfuse / Arize: **log** the turn after the fact for later analysis.
//!   No pre-delivery regulatory decision.
//! - Portkey / litellm: provider-level circuit-breakers (rate limits,
//!   retries on 5xx). No semantic scope enforcement.
//!
//! `Regulator::decide()` returns `ScopeDriftWarn` *before* the response is
//! delivered. The app can auto-strip, ask the user, or accept — the
//! decision surfaces the signal instead of waiting for the user to notice.

use std::env;

use nous_regulator::{Decision, LLMEvent, Regulator};

#[path = "regulator_common/mod.rs"]
mod regulator_common;
use regulator_common::{call_anthropic, call_ollama};

/// User task. Kept short and unambiguous: task keyword bag is `{async,
/// database, fetch_user, function, logic, refactor, unchanged}` (after
/// [`extract_topics`] stop-word + min-length-3 filter). A faithful
/// refactor stays inside this vocabulary.
///
/// [`extract_topics`]: nous_regulator::cognition
const TASK: &str =
    "Refactor fetch_user to be async. Keep the database lookup logic unchanged.";

/// Pre-written drifted response used by the canned demo path. Shape
/// deliberately mirrors what a verbose, over-helpful LLM typically
/// produces when asked to "refactor": the async version is present, but
/// it's wrapped in logging, error handling, and telemetry nobody asked
/// for.
///
/// Against [`TASK`] the scope tracker produces `drift_score = 1.0` in
/// the Session 18 plan case (disjoint bags); this real-ish response
/// keeps `async` and a couple of task keywords so the score lands in
/// the "clearly flagged, not pathological" range.
const CANNED_DRIFTED_RESPONSE: &str = "\
Here is the async version plus some improvements for observability:

- Added tracing::info! at entry for request logging
- Wrapped the database call in a tokio timeout for resilience
- Propagated errors via Result<User, FetchError>
- Emitted a prometheus counter for fetch frequency
- Added doc comments for the new async signature

```rust
async fn fetch_user(id: u64) -> Result<User, FetchError> {
    tracing::info!(user_id = id, \"fetching user\");
    let row = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        db.query_row(id),
    )
    .await
    .map_err(|_| FetchError::Timeout)?
    .map_err(FetchError::Database)?;
    metrics::counter!(\"db.fetch_user\", 1);
    Ok(User::from_row(row))
}
```
";

/// Canned per-turn cost accounting (matches the named-constant pattern
/// in Demos 2 and 3). Values are coarse estimates — this demo's value
/// is the drift decision, not calibrated billing.
const CANNED_TOKENS_IN: u32 = 40;
const CANNED_TOKENS_OUT: u32 = 180;
const CANNED_WALLCLOCK_MS: u32 = 0;

fn main() {
    let mode = env::args().nth(1).unwrap_or_else(|| "canned".into());

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Regulator demo 1 — Scope drift intercept                      ║");
    println!("║  Session 21 / Path 2 flagship                                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("Mode:  {mode}");
    println!("Task:  {TASK}\n");

    let (response, tokens_in, tokens_out, wallclock_ms, provider_tag) = match mode.as_str() {
        "canned" => canned_turn("canned"),
        "ollama" => match call_ollama(TASK) {
            Ok((r, ti, to, wc)) => (r, ti, to, wc, "ollama"),
            Err(e) => {
                eprintln!("⚠  Ollama call failed: {e}");
                eprintln!("   Falling back to canned demo so the drift detection still runs.\n");
                canned_turn("canned-fallback")
            }
        },
        "anthropic" => match call_anthropic(TASK) {
            Ok((r, ti, to, wc)) => (r, ti, to, wc, "anthropic"),
            Err(e) => {
                eprintln!("⚠  Anthropic call failed: {e}");
                eprintln!("   Falling back to canned demo so the drift detection still runs.\n");
                canned_turn("canned-fallback")
            }
        },
        other => {
            eprintln!("Unknown mode {other:?}. Use `canned`, `ollama`, or `anthropic`.");
            std::process::exit(2);
        }
    };

    // ── Baseline view ─────────────────────────────────────────────────
    println!("──── Baseline agent (Mem0 / Langfuse / OPA style) ────");
    println!("[Response passes straight through to the user]\n");
    println!("{response}");
    println!("\n[Agent logs the turn and moves on. The user has to notice the scope");
    println!(" expansion themselves.]\n");

    // ── Regulator view ────────────────────────────────────────────────
    let mut regulator = Regulator::for_user("demo_user");
    regulator.on_event(LLMEvent::TurnStart {
        user_message: TASK.into(),
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

    println!("──── Regulator agent ────");
    match regulator.decide() {
        Decision::ScopeDriftWarn {
            drift_tokens,
            drift_score,
            task_tokens,
        } => {
            println!("[Decision::ScopeDriftWarn]");
            println!("  drift_score:  {drift_score:.2}  (threshold 0.50)");
            println!("  task_tokens:  {task_tokens:?}");
            println!("  drift_tokens: {drift_tokens:?}");
            println!();
            println!(
                "{} response keyword(s) have no anchor in the task.",
                drift_tokens.len()
            );
            println!("Under a baseline agent these would reach the user unannotated;");
            println!("the regulator surfaces them before delivery so the app can:");
            println!("  • auto-strip the added material and re-prompt,");
            println!("  • ask the user whether the expansion is welcome, or");
            println!("  • accept with an inline 'added beyond request' annotation.");
        }
        Decision::Continue => {
            println!("[Decision::Continue]");
            println!("Response keywords align with task — no drift flagged.");
            println!("(On this run the model happened to stay on-task. Try `canned`");
            println!(" mode or a chattier model to see the drift path fire.)");
        }
        Decision::CircuitBreak {
            reason,
            suggestion,
        } => {
            println!("[Decision::CircuitBreak] ({reason:?})");
            println!("{suggestion}");
        }
        Decision::ProceduralWarning { patterns } => {
            println!("[Decision::ProceduralWarning] ({} pattern(s))", patterns.len());
            println!("(Unexpected for this single-turn demo — patterns require ≥ 3");
            println!(" prior corrections on this cluster.)");
        }
        Decision::LowConfidenceSpans { spans } => {
            println!("[Decision::LowConfidenceSpans] ({} span(s))", spans.len());
            println!("(Reserved for a future session that wires per-span logprobs.)");
        }
    }

    // Extra diagnostics so the reader can see the cost loop also closed.
    println!();
    println!("──── Turn accounting ────");
    println!(
        "  tokens_in={tokens_in}  tokens_out={tokens_out}  wallclock_ms={wallclock_ms}  \
         provider={provider_tag}"
    );
    println!(
        "  regulator.total_tokens_out() = {}",
        regulator.total_tokens_out()
    );
    println!(
        "  regulator.cost_cap_tokens()  = {}",
        regulator.cost_cap_tokens()
    );
    println!(
        "  regulator.confidence()       = {:.2}  (fallback: no logprobs wired for this demo)",
        regulator.confidence()
    );
}

// ── Canned fallback ──────────────────────────────────────────────────

/// Single source (P3) for the canned turn tuple, parameterised only by the
/// `provider_tag` the caller wants printed. Used by the default `canned`
/// mode and by both live-mode transport-failure fallbacks. Token counts
/// are coarse estimates suitable for demonstration of the cost loop —
/// the demo's value is the drift decision, not calibrated billing.
fn canned_turn(provider_tag: &'static str) -> (String, u32, u32, u32, &'static str) {
    (
        CANNED_DRIFTED_RESPONSE.to_string(),
        CANNED_TOKENS_IN,
        CANNED_TOKENS_OUT,
        CANNED_WALLCLOCK_MS,
        provider_tag,
    )
}

// HTTP adapters live in `regulator_common/mod.rs` (P3 single source for
// Sessions 21-23 demos).
