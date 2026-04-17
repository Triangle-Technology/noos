//! Criterion benchmarks for the regulator hot path.
//!
//! Measures per-event overhead + full-turn overhead so integrators
//! can answer "how much does Noos add to my agent loop?" with data
//! rather than intuition.
//!
//! Run:
//!
//! ```bash
//! cargo bench --bench regulator
//! ```
//!
//! Criterion writes HTML reports to `target/criterion/`. The console
//! shows mean / std-dev / throughput per benchmark. Each bench runs
//! for ~5 seconds of wall clock with statistical sampling — numbers
//! are more stable than `Instant::now()` measurements.
//!
//! ## Scope
//!
//! - **Per-event dispatch** for each `LLMEvent` variant (hot path:
//!   `Token` for streaming clients).
//! - **`decide()`** — the primary call. Measured with a realistic
//!   prior state (several events drained).
//! - **Persistence roundtrip** — `export_json` → `from_json` latency.
//! - **Full realistic turn** — one `TurnStart`, 100 streamed `Token`s,
//!   one `TurnComplete`, one `Cost`, one `decide()`. Best mirror of
//!   what an app pays per turn.
//!
//! Benches are NOT in the crates.io `include` whitelist, so this file
//! never ships to integrators. It's a project-internal measurement
//! tool.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use noos::{LLMEvent, Regulator};

// ── Helpers ────────────────────────────────────────────────────────

/// Construct a regulator with a short prior history so `decide()` has
/// realistic state to compute against. Without prior events, most
/// checks short-circuit on empty accumulators.
fn primed_regulator() -> Regulator {
    let mut r = Regulator::for_user("bench_user").with_cost_cap(10_000);
    r.on_event(LLMEvent::TurnStart {
        user_message: "Refactor fetch_user to be async and add proper error handling".into(),
    });
    r.on_event(LLMEvent::TurnComplete {
        full_response: "async fn fetch_user() { /* reasonable response */ }".into(),
    });
    r.on_event(LLMEvent::Cost {
        tokens_in: 25,
        tokens_out: 200,
        wallclock_ms: 400,
        provider: Some("bench".into()),
    });
    r
}

// ── Per-event benchmarks ───────────────────────────────────────────

fn bench_event_turn_start(c: &mut Criterion) {
    let mut group = c.benchmark_group("event/turn_start");
    group.throughput(Throughput::Elements(1));
    group.bench_function("dispatch", |b| {
        b.iter_batched(
            || Regulator::for_user("u"),
            |mut r| {
                r.on_event(black_box(LLMEvent::TurnStart {
                    user_message: "Refactor fetch_user to be async".into(),
                }));
                r
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_event_token(c: &mut Criterion) {
    // `Token` is the hottest event for streaming clients — an LLM
    // emitting 200 tokens/sec means ~5ms per token budget; the
    // regulator should add far less than that.
    let mut group = c.benchmark_group("event/token");
    group.throughput(Throughput::Elements(1));
    group.bench_function("dispatch", |b| {
        let mut r = Regulator::for_user("u");
        r.on_event(LLMEvent::TurnStart {
            user_message: "prompt".into(),
        });
        let mut idx = 0usize;
        b.iter(|| {
            r.on_event(black_box(LLMEvent::Token {
                token: "hello".into(),
                logprob: -1.5,
                index: idx,
            }));
            idx += 1;
        });
    });
    group.finish();
}

fn bench_event_turn_complete(c: &mut Criterion) {
    let mut group = c.benchmark_group("event/turn_complete");
    group.throughput(Throughput::Elements(1));
    group.bench_function("dispatch", |b| {
        b.iter_batched(
            || {
                let mut r = Regulator::for_user("u");
                r.on_event(LLMEvent::TurnStart {
                    user_message: "Refactor fetch_user to be async".into(),
                });
                r
            },
            |mut r| {
                r.on_event(black_box(LLMEvent::TurnComplete {
                    full_response: "async fn fetch_user() { /* response text */ }".into(),
                }));
                r
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_event_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("event/cost");
    group.throughput(Throughput::Elements(1));
    group.bench_function("dispatch", |b| {
        let mut r = primed_regulator();
        b.iter(|| {
            r.on_event(black_box(LLMEvent::Cost {
                tokens_in: 25,
                tokens_out: 200,
                wallclock_ms: 400,
                provider: None,
            }));
        });
    });
    group.finish();
}

fn bench_event_tool_call(c: &mut Criterion) {
    let mut group = c.benchmark_group("event/tool_call");
    group.throughput(Throughput::Elements(1));
    group.bench_function("dispatch", |b| {
        let mut r = Regulator::for_user("u");
        r.on_event(LLMEvent::TurnStart {
            user_message: "search".into(),
        });
        b.iter(|| {
            r.on_event(black_box(LLMEvent::ToolCall {
                tool_name: "search_orders".into(),
                args_json: Some("{\"id\":42}".into()),
            }));
        });
    });
    group.finish();
}

// ── decide() benchmark ─────────────────────────────────────────────

fn bench_decide_continue(c: &mut Criterion) {
    // Realistic "Continue" path — drained state, no threshold
    // triggers. Represents the 95th percentile of in-production calls
    // where no concern warrants intervention.
    let mut group = c.benchmark_group("decide/continue");
    group.throughput(Throughput::Elements(1));
    group.bench_function("hot_path", |b| {
        let r = primed_regulator();
        b.iter(|| {
            let _ = black_box(r.decide());
        });
    });
    group.finish();
}

fn bench_decide_scope_drift(c: &mut Criterion) {
    // Response drifted — decide() must extract task + response
    // keyword bags and compute the set-difference ratio.
    let mut group = c.benchmark_group("decide/scope_drift");
    group.throughput(Throughput::Elements(1));
    group.bench_function("full_extraction", |b| {
        let mut r = Regulator::for_user("u");
        r.on_event(LLMEvent::TurnStart {
            user_message: "Refactor fetch_user to be async. Keep the database lookup unchanged."
                .into(),
        });
        r.on_event(LLMEvent::TurnComplete {
            full_response: "added counter timing retry cache wrapper handler middleware logger queue"
                .into(),
        });
        r.on_event(LLMEvent::Cost {
            tokens_in: 40,
            tokens_out: 180,
            wallclock_ms: 500,
            provider: None,
        });
        b.iter(|| {
            let _ = black_box(r.decide());
        });
    });
    group.finish();
}

// ── Persistence benchmark ──────────────────────────────────────────

fn bench_export_import_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistence");
    group.throughput(Throughput::Elements(1));
    group.bench_function("export_then_import", |b| {
        let r = primed_regulator();
        b.iter(|| {
            let state = r.export();
            let json = serde_json::to_string(&state).expect("serialise");
            let parsed: noos::RegulatorState =
                serde_json::from_str(&json).expect("deserialise");
            black_box(Regulator::import(parsed));
        });
    });
    group.finish();
}

// ── Full-turn benchmark ────────────────────────────────────────────

fn bench_full_turn_realistic(c: &mut Criterion) {
    // Most informative single number for integrators: how much does
    // Noos add per turn?
    //
    // Shape: 1 TurnStart, 100 Token events (typical 50-500 for short
    // responses), 1 TurnComplete, 1 Cost, 1 decide(). This is the
    // per-turn cost an agent loop pays regardless of whether the
    // Decision ends up being Continue or actionable.
    let mut group = c.benchmark_group("full_turn/realistic");
    group.throughput(Throughput::Elements(1));
    group.bench_function("100_tokens", |b| {
        b.iter_batched(
            || Regulator::for_user("u").with_cost_cap(10_000),
            |mut r| {
                r.on_event(LLMEvent::TurnStart {
                    user_message: "Refactor fetch_user to be async".into(),
                });
                for i in 0..100 {
                    r.on_event(LLMEvent::Token {
                        token: "word".into(),
                        logprob: -1.2,
                        index: i,
                    });
                }
                r.on_event(LLMEvent::TurnComplete {
                    full_response: "async fn fetch_user() { /* response */ }".into(),
                });
                r.on_event(LLMEvent::Cost {
                    tokens_in: 25,
                    tokens_out: 100,
                    wallclock_ms: 500,
                    provider: None,
                });
                let _ = black_box(r.decide());
                r
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

// ── Harness ────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_event_turn_start,
    bench_event_token,
    bench_event_turn_complete,
    bench_event_cost,
    bench_event_tool_call,
    bench_decide_continue,
    bench_decide_scope_drift,
    bench_export_import_roundtrip,
    bench_full_turn_realistic,
);
criterion_main!(benches);
