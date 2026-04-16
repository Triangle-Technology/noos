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
//!
//! The Anthropic call reads `ANTHROPIC_API_KEY` (required). When the
//! key is unset the call returns a clear error rather than panicking,
//! so demos can gracefully fall back to their canned path (P5).

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

/// Call Anthropic's Messages API (`/v1/messages`) for a single
/// non-streaming turn. Requires `ANTHROPIC_API_KEY`.
///
/// Anthropic returns `content: [{type: "text", text: "..."}]`; the
/// text blocks are concatenated into a single string so the caller's
/// scope / drift / cost pipeline treats the turn uniformly.
pub fn call_anthropic(user_msg: &str) -> Result<TurnTuple, String> {
    let key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| "ANTHROPIC_API_KEY env var unset".to_string())?;
    let model = env::var("NOUS_ANTHROPIC_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".into());

    let body = json!({
        "model": model,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": user_msg}]
    });

    let t0 = Instant::now();
    let resp = ureq::post("https://api.anthropic.com/v1/messages")
        .set("x-api-key", &key)
        .set("anthropic-version", "2023-06-01")
        .set("content-type", "application/json")
        .send_json(&body)
        .map_err(|e| format!("HTTP request failed: {e}"))?;
    let data: Value = resp
        .into_json()
        .map_err(|e| format!("JSON parse failed: {e}"))?;

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
