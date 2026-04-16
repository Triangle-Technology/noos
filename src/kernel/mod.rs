//! Kernel — plugin system + pipeline infrastructure (Phase 4).
//!
//! Provides a typed event bus, pluggable extension points, and sequential
//! pipeline execution for applications building richer flows on top of Nous.
//!
//! ## Relationship to `CognitiveSession`
//!
//! This module is **infrastructure**, not part of the core cognitive flow.
//! `CognitiveSession` does not use it. Applications that need to:
//! - Chain multiple processing steps (tokenization → retrieval → generation)
//! - Broadcast cognitive events (Global Workspace analog, Dehaene 2001)
//! - Load pluggable processors (custom detectors, external enrichment)
//!
//! ...can build those on top of Nous using these types. For simple use
//! cases, `CognitiveSession::process_message()` alone is sufficient.
//!
//! Brain analog: Global Workspace broadcast (Dehaene & Changeux 2011) for
//! `events`; modular plug-in architecture for `plugin`.

pub mod events;
pub mod plugin;
pub mod pipeline;
