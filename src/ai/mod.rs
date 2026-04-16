//! AI provider abstraction (Phase 4).
//!
//! Trait-based interface to external LLM services (Anthropic, OpenAI, Google,
//! Ollama, etc.). Request builders, response parsers, streaming support.
//!
//! ## Relationship to `CognitiveSession`
//!
//! This module is **infrastructure**, not part of the core cognitive flow.
//! `CognitiveSession` does not call `AiProvider` — the session produces
//! `SamplingOverride` and `DeltaModulation` signals that applications apply
//! to whichever model backend they use.
//!
//! Use this module when building Nous-powered applications that need:
//! - A normalized interface across multiple LLM providers
//! - Streaming token handling
//! - Logit-level intervention via `LogitIntervenor` (Tầng 1)
//!
//! For purely sync cognitive processing (no LLM calls from inside Nous),
//! `CognitiveSession` + `CognitiveSignals` is sufficient.

pub mod intervention;
pub mod provider;
pub mod request;
pub mod response;
