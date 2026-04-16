//! Integration tests — sync memory API usage from external consumer perspective.
//!
//! Validates that applications can:
//! 1. Create MemoryAtom instances via public types
//! 2. Call hybrid_recall on pre-loaded atoms (sync, no async runtime needed)
//! 3. Store atoms through the MemoryStore trait
//!
//! Nous memory is intentionally sync-retrievable: applications own async I/O
//! (loading from database/disk), Nous owns cognitive computation over loaded
//! atoms. This separation keeps CognitiveSession sync while supporting
//! persistent memory backends.

use nous_regulator::{
    hybrid_recall, ActivationSource, AtomSource, AtomType, InMemoryStore, MemoryAtom,
    MemoryStore, RecallOptions, Synapse, SynapseType,
};
use std::collections::HashMap;

fn make_atom(id: &str, content: &str, topics: Vec<String>) -> MemoryAtom {
    MemoryAtom {
        id: id.into(),
        content: content.into(),
        embedding: None,
        atom_type: AtomType::Episodic,
        source: AtomSource::default(),
        importance: 0.7,
        access_count: 0,
        last_accessed_at: 0.0,
        created_at: 0.0,
        topics,
        domain: None,
        consolidated_from: None,
        is_consolidated: false,
        parent_id: None,
        depth: None,
        label: None,
        child_ids: None,
        superseded: false,
        suppressed: false,
        dormant: false,
        tags: vec![],
        encoding_context: None,
        retrieval_reward: None,
        reconsolidation_count: None,
        arousal: None,
        valence: None,
        epoch: None,
        crystallized: false,
    }
}

#[test]
fn hybrid_recall_finds_topic_match() {
    // Application loads its atoms (sync in Nous; async I/O is app's concern).
    let atoms = vec![
        make_atom("a1", "Rust async basics", vec!["rust".into(), "async".into()]),
        make_atom("a2", "Python type hints", vec!["python".into(), "types".into()]),
        make_atom("a3", "Rust ownership rules", vec!["rust".into(), "ownership".into()]),
    ];
    let synapses = HashMap::new();

    let results = hybrid_recall(
        &atoms,
        None, // No query embedding — topic-only match
        "How does rust ownership work?",
        &synapses,
        &RecallOptions::default(),
    );

    // Rust-related atoms should activate.
    assert!(
        !results.is_empty(),
        "Topic-matching query should retrieve at least one atom"
    );
    assert!(
        results.iter().any(|a| a.atom.id == "a3"),
        "Ownership query should retrieve the ownership atom"
    );
}

#[test]
fn activation_sources_distinguish_retrieval_paths() {
    // Vector search vs graph spreading vs convergent — distinct enum values.
    // Just verify the enum is usable from outside.
    let v = ActivationSource::Vector;
    let g = ActivationSource::Graph;
    let c = ActivationSource::Convergent;
    assert_ne!(v, g);
    assert_ne!(g, c);
}

#[tokio::test]
async fn in_memory_store_round_trip() {
    // MemoryStore trait is async (applications providing persistent backends
    // will naturally be async); the default InMemoryStore works for tests
    // and lightweight in-process use.
    let mut store = InMemoryStore::new();

    let atom = make_atom("a1", "Test content", vec!["test".into()]);
    store.store_atom(atom.clone()).await.unwrap();

    let retrieved = store.get_atom("a1").await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().content, "Test content");

    // Synapses link atoms by ID.
    let atom2 = make_atom("a2", "Related", vec!["test".into()]);
    store.store_atom(atom2).await.unwrap();

    let syn = Synapse {
        id: "s1".into(),
        source: "a1".into(),
        target: "a2".into(),
        synapse_type: SynapseType::RelatesTo,
        strength: 0.6,
        created_at: 0.0,
        last_accessed_at: None,
        access_count: None,
    };
    store.store_synapse(syn).await.unwrap();

    let syns = store.get_synapses_for("a1").await.unwrap();
    assert_eq!(syns.len(), 1);
    assert_eq!(syns[0].target, "a2");

    assert_eq!(store.atom_count(), 2);
}

#[test]
fn recall_options_default_is_usable() {
    // Apps should be able to start with defaults and tweak.
    let mut opts = RecallOptions::default();
    assert_eq!(opts.top_k, 10);
    assert_eq!(opts.graph_depth, 2);

    // Override one field.
    opts.top_k = 5;
    assert_eq!(opts.top_k, 5);
}
