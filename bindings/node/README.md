# noos-regulator (Node.js / TypeScript)

> Node.js native bindings for [Noos](https://crates.io/crates/noos), a
> reliability layer for LLM agents: scope drift, cost circuit breaks,
> procedural correction memory, and tool-call loop detection as
> event-driven decisions.

Built via [napi-rs 3.x](https://napi.rs/). TypeScript definitions
auto-generated from the Rust source at build time.

## Status

**0.1.0-pre** — bindings code complete, pending CI binary builds for
publication to npm. Build from source instructions below cover
Linux / macOS / Windows-MSVC today.

## Install (once published)

```bash
npm install noos-regulator
# or
pnpm add noos-regulator
# or
yarn add noos-regulator
```

## Quick start

```typescript
import { Regulator, LLMEvent } from 'noos-regulator'

const r = Regulator.forUser('alice')
r.withCostCap(2_000)

r.onEvent(LLMEvent.turnStart('Refactor fetch_user to be async'))

// ... call your LLM ...

r.onEvent(LLMEvent.turnComplete(responseText))
r.onEvent(LLMEvent.cost(25, 800, 500, 'anthropic'))

const decision = r.decide()
switch (decision.kind) {
  case 'continue':
    /* deliver response */
    break
  case 'scope_drift_warn':
    console.log(`drift ${decision.driftScore?.toFixed(2)}: ${decision.driftTokens}`)
    break
  case 'circuit_break':
    console.log(`halt: ${decision.suggestion} (${decision.reason?.kind})`)
    break
  case 'procedural_warning':
    decision.patterns?.forEach((p) =>
      console.log(`learned: ${p.patternName}`, p.exampleCorrections)
    )
    break
}
```

## API summary

| Class | Purpose |
|-------|---------|
| `Regulator` | Main API. `forUser(id)`, `withCostCap(n)`, `withImplicitCorrectionWindowSecs(s)`, `onEvent(e)`, `decide()`, `exportJson()`, `fromJson(s)` + accessors. |
| `LLMEvent` | Event factories. `turnStart`, `token`, `turnComplete`, `cost`, `qualityFeedback`, `userCorrection`, `toolCall`, `toolResult`, `fromOtelSpanJson`. |
| `Decision` | Output of `decide()`. `.kind` + variant-specific getters returning `T \| null`. |
| `CircuitBreakReason` | Nested on `Decision.reason` when `kind === 'circuit_break'`. |
| `CorrectionPattern` | Items of `Decision.patterns` when `kind === 'procedural_warning'`. |

### Decision variants

| `.kind` | Available getters |
|---------|-------------------|
| `'continue'` | — |
| `'scope_drift_warn'` | `driftScore`, `driftTokens`, `taskTokens` |
| `'circuit_break'` | `reason` (a `CircuitBreakReason`), `suggestion` |
| `'procedural_warning'` | `patterns` (an array of `CorrectionPattern`) |
| `'low_confidence_spans'` | reserved |

### `CircuitBreakReason` variants

| `.kind` | Available getters |
|---------|-------------------|
| `'cost_cap_reached'` | `tokensSpent`, `tokensCap`, `meanQualityLastN` |
| `'quality_decline_no_recovery'` | `turns`, `meanDelta` |
| `'repeated_failure_pattern'` | `cluster`, `failureCount` |
| `'repeated_tool_call_loop'` | `toolName`, `consecutiveCount` |

## Persistence

```typescript
// Save
const snapshot: string = r.exportJson()
await fs.writeFile('regulator.json', snapshot, 'utf-8')

// Restore
const snapshot = await fs.readFile('regulator.json', 'utf-8')
const r = Regulator.fromJson(snapshot)
```

Cross-session learning (strategy EMA + correction patterns at
threshold) survives the roundtrip. Malformed JSON throws.

## Observability

One-call metrics snapshot for Prometheus / Datadog / StatsD:

```typescript
const snap = r.metricsSnapshot()
for (const [key, value] of Object.entries(snap)) {
  metricsClient.gauge(key, value)
}
```

All keys are `noos.` prefixed (confidence, logprob_coverage,
total_tokens_out, cost_cap_tokens, tool_*, implicit_corrections_count).

## OpenTelemetry GenAI ingestion

If your agent is already instrumented with the OTel GenAI semantic
conventions, feed spans to Noos without wiring a separate event bus:

```typescript
import { readFile } from 'node:fs/promises'
import { Regulator, LLMEvent } from 'noos-regulator'

const spanJson = await readFile('span.json', 'utf-8')
const r = Regulator.forUser('alice')
for (const event of LLMEvent.fromOtelSpanJson(spanJson)) {
  r.onEvent(event)
}
const d = r.decide()
```

## Build from source

Requires: Rust toolchain (1.75+), Node.js 18+.

```bash
cd bindings/node
npm install
npm run build
# Produces an `index.js` + `index.d.ts` + `.node` native addon.

npm test       # 15 behavioural tests via `node --test`
npm run example # Runs examples/basic.mjs
```

**Windows-GNU note**: `napi-build` 2.3.1 requires `libnode.dll` for the
GNU ABI path, which standard Windows Node installs don't ship. Work
around with WSL, MSVC Rust toolchain (`rustup default
stable-x86_64-pc-windows-msvc`), or CI running ubuntu/macos. Linux and
macOS builds work out of the box.

## License

MIT. Same as the Rust crate.
