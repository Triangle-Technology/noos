//! Shared HTTP adapters for the regulator demos (Sessions 21-23).
//!
//! **P3 compliance**: Demos 1, 2, and 3 all need the same "one-shot
//! non-streaming chat completion" call against either Ollama or
//! Anthropic. Sessions 21 and 22 each grew a local copy of
//! `call_ollama` + `call_anthropic`; this module exists to give Session
//! 23 (and any future regulator demo) a single source.
//!
//! ## Include pattern
//!
//! Cargo auto-discovers top-level `examples/*.rs` as binaries but does
//! NOT auto-discover `examples/<subdir>/`. Bringing this module into a
//! demo therefore uses a `#[path]` attribute so Cargo doesn't try to
//! compile `regulator_common` as a standalone binary:
//!
//! ```ignore
//! #[path = "regulator_common/mod.rs"]
//! mod regulator_common;
//! use regulator_common::{call_ollama, call_anthropic};
//! ```
//!
//! ## Env overrides
//!
//! - `NOUS_OLLAMA_URL` (default `http://localhost:11434/api/chat`)
//! - `NOUS_OLLAMA_MODEL` (default `phi3:mini`)
//! - `NOUS_ANTHROPIC_MODEL` (default `claude-haiku-4-5-20251001`)
//! - `NOUS_JUDGE_MODEL` (default `claude-haiku-4-5-20251001`)
//!
//! The Anthropic call reads `ANTHROPIC_API_KEY` (required). When the
//! key is unset the call returns a clear error rather than panicking,
//! so demos can gracefully fall back to their canned path (P5).
//!
//! [`call_anthropic_judge`] shares the same key and fallback contract.

use std::env;
use std::time::Instant;

use serde_json::{json, Value};

/// Tuple returned by both adapters: `(response_text, tokens_in,
/// tokens_out, wallclock_ms)`. Matches the shape demos feed into
/// [`noos::LLMEvent::Cost`].
pub type TurnTuple = (String, u32, u32, u32);

/// Call Ollama's `/api/chat` endpoint with `stream: false`. Returns the
/// assistant text plus token / wallclock accounting.
///
/// Silently consumes token-count fields when Ollama doesn't populate
/// them (`prompt_eval_count` / `eval_count`) — pre-v0.1.30 builds omit
/// the fields for cached prefixes. A zero counter there is harmless for
/// demo accounting; the cost-cap predicate is dominated by the
/// `tokens_out` accumulator over multiple turns anyway.
pub fn call_ollama(user_msg: &str) -> Result<TurnTuple, String> {
    let url = env::var("NOUS_OLLAMA_URL")
        .unwrap_or_else(|_| "http://localhost:11434/api/chat".into());
    let model = env::var("NOUS_OLLAMA_MODEL").unwrap_or_else(|_| "phi3:mini".into());

    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": user_msg}],
        "stream": false
    });

    let t0 = Instant::now();
    let resp = ureq::post(&url)
        .send_json(&body)
        .map_err(|e| format!("HTTP request failed: {e}"))?;
    let data: Value = resp
        .into_json()
        .map_err(|e| format!("JSON parse failed: {e}"))?;

    let content = data["message"]["content"]
        .as_str()
        .ok_or("response missing message.content")?
        .to_string();
    let tokens_in = data["prompt_eval_count"].as_u64().unwrap_or(0) as u32;
    let tokens_out = data["eval_count"].as_u64().unwrap_or(0) as u32;
    let wallclock_ms = t0.elapsed().as_millis() as u32;

    Ok((content, tokens_in, tokens_out, wallclock_ms))
}

/// One-shot POST to Anthropic's Messages API (`/v1/messages`), single
/// user-message turn. Shared by [`call_anthropic`] and
/// [`call_anthropic_judge`] (P3: single source of the HTTP + header
/// plumbing).
///
/// Requires `ANTHROPIC_API_KEY`. Returns the parsed response JSON —
/// callers extract whichever fields they need (`call_anthropic`
/// wants `content[].text` + `usage.*` for cost accounting;
/// `call_anthropic_judge` wants only a score field inside the
/// response text).
///
/// `#[allow(dead_code)]`: this module is `#[path]`-included by four
/// examples; not every caller uses every helper, and the compiler
/// warns per includer.
#[allow(dead_code)]
fn anthropic_messages_post(
    model: &str,
    max_tokens: u32,
    user_content: &str,
) -> Result<Value, String> {
    let key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| "ANTHROPIC_API_KEY env var unset".to_string())?;
    let body = json!({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_content}]
    });
    let resp = ureq::post("https://api.anthropic.com/v1/messages")
        .set("x-api-key", &key)
        .set("anthropic-version", "2023-06-01")
        .set("content-type", "application/json")
        .send_json(&body)
        .map_err(|e| format!("Anthropic HTTP request failed: {e}"))?;
    resp.into_json::<Value>()
        .map_err(|e| format!("Anthropic JSON parse failed: {e}"))
}

/// Call Anthropic's Messages API (`/v1/messages`) for a single
/// non-streaming turn. Requires `ANTHROPIC_API_KEY`.
///
/// Anthropic returns `content: [{type: "text", text: "..."}]`; the
/// text blocks are concatenated into a single string so the caller's
/// scope / drift / cost pipeline treats the turn uniformly.
pub fn call_anthropic(user_msg: &str) -> Result<TurnTuple, String> {
    let model = env::var("NOUS_ANTHROPIC_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".into());

    let t0 = Instant::now();
    let data = anthropic_messages_post(&model, 512, user_msg)?;

    let content = data["content"]
        .as_array()
        .and_then(|blocks| {
            let joined: String = blocks
                .iter()
                .filter_map(|b| b["text"].as_str())
                .collect::<Vec<_>>()
                .join("");
            if joined.is_empty() {
                None
            } else {
                Some(joined)
            }
        })
        .ok_or_else(|| format!("response missing content[].text: {data}"))?;

    let tokens_in = data["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32;
    let tokens_out = data["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32;
    let wallclock_ms = t0.elapsed().as_millis() as u32;

    Ok((content, tokens_in, tokens_out, wallclock_ms))
}

/// LLM-as-judge quality grader (Claude).
///
/// Rates how well `response` answers `task` on a 0.0–1.0 scale. Prompt
/// asks the judge for a JSON object `{"score": float, "reason":
/// string}`; only `score` is consumed. The reason field is there to
/// nudge the model toward thoughtful scoring (chain-of-thought effect)
/// without parser burden on the caller.
///
/// ## Why this exists (2026-04-17)
///
/// The Tier 2.2 eval (`examples/task_eval_real_llm_regulator.rs`) ships
/// a synthetic quality oracle so the canned numbers are
/// bit-reproducible. Live-mode runs historically used the same oracle,
/// so `total_quality` deltas were driven by the harness structure, not
/// by real grading. This adapter closes that loop: set
/// `NOOS_JUDGE=anthropic` and the eval's `llm_call` swaps in this
/// grader per response.
///
/// ## Contract
///
/// - Returns `f64` in `[0, 1]`, clamped. Non-parseable responses fall
///   back to `Err` (caller decides — the eval falls back to the canned
///   oracle so one flaky grader call doesn't invalidate the run).
/// - Requires `ANTHROPIC_API_KEY`. Without it, returns an error and
///   the caller falls back (P5 fail-open).
/// - Cheap model by default (`claude-haiku-4-5`); override via
///   `NOOS_JUDGE_MODEL` for a stronger grader.
///
/// The rubric is deliberately short — long rubrics bias small judges
/// toward middle-score hedging. The grader sees only the task and the
/// response; it does not see which arm produced the response.
///
/// `#[allow(dead_code)]`: this module is `#[path]`-included by four
/// examples (Sessions 21-24); only the Tier 2.2 eval
/// (`task_eval_real_llm_regulator`) consumes the judge, so the other
/// three demos compile with this function seen as unused.
#[allow(dead_code)]
pub fn call_anthropic_judge(task: &str, response: &str) -> Result<f64, String> {
    let model = env::var("NOUS_JUDGE_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".into());

    let prompt = format!(
        "You are grading a response to a task.\n\n\
        Task: {task}\n\n\
        Response: {response}\n\n\
        Rate the response on a 0.0–1.0 scale:\n\
        - 1.0 = fully answers the task, on-topic, correct, concise\n\
        - 0.7 = answers the task but drifts, adds unasked content, or minor errors\n\
        - 0.4 = partial answer, significant drift or vagueness\n\
        - 0.1 = mostly off-topic or wrong\n\
        - 0.0 = empty, refusal, or pure drift\n\n\
        Respond with ONLY a JSON object, no prose before or after:\n\
        {{\"score\": <float>, \"reason\": \"<one short sentence>\"}}"
    );

    let data = anthropic_messages_post(&model, 128, &prompt)?;

    let raw = data["content"]
        .as_array()
        .and_then(|blocks| blocks.iter().find_map(|b| b["text"].as_str()))
        .ok_or_else(|| format!("judge response missing content[].text: {data}"))?
        .trim();

    // Claude sometimes wraps JSON in ```json fences or adds a line
    // before it. Find the first `{` and last `}` and parse that slice.
    let start = raw.find('{').ok_or("judge response has no '{'")?;
    let end = raw.rfind('}').ok_or("judge response has no '}'")?;
    if end <= start {
        return Err(format!("judge response braces out of order: {raw}"));
    }
    let json_slice = &raw[start..=end];

    let parsed: Value = serde_json::from_str(json_slice)
        .map_err(|e| format!("judge JSON parse failed on slice {json_slice:?}: {e}"))?;
    let score = parsed["score"]
        .as_f64()
        .ok_or_else(|| format!("judge response missing numeric score: {parsed}"))?;

    Ok(score.clamp(0.0, 1.0))
}
