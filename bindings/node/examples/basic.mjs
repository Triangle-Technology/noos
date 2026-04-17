// Basic Node.js smoke-test of the bindings.
//
// Runs the same four scenarios as the Rust flagship demos — all
// canned, no LLM required — and prints the Decision for each step.
//
// Run (after `npm run build`):
//
//     node examples/basic.mjs

import { Regulator, LLMEvent } from '../index.js'

function scopeDriftExample() {
  console.log('── scope drift ────────────────────────────────')
  const r = Regulator.forUser('alice')

  r.onEvent(LLMEvent.turnStart(
    'Refactor fetch_user to be async. Keep the database lookup logic unchanged.'
  ))
  r.onEvent(LLMEvent.turnComplete(
    'async function fetchUser(id) { ' +
    '  await asyncio.sleep(0); // added await call for non-blocking ' +
    '  let counter = 0; // added counter tracking requests ' +
    '  return await db.lookup(id, { duration: timeout }) ' +
    '}'
  ))
  r.onEvent(LLMEvent.cost(40, 180, 0, 'canned'))

  const d = r.decide()
  console.log(`  decision.kind = ${d.kind}`)
  if (d.isScopeDrift()) {
    console.log(`  driftScore    = ${d.driftScore?.toFixed(2)}`)
    console.log(`  driftTokens   = ${JSON.stringify(d.driftTokens)}`)
  }
  console.log()
}

function costBreakExample() {
  console.log('── cost circuit break ─────────────────────────')
  const r = Regulator.forUser('bob')
  r.withCostCap(1000)

  for (let i = 0; i < 3; i++) {
    const quality = [0.50, 0.35, 0.20][i]
    r.onEvent(LLMEvent.turnStart(`Optimize this SQL query attempt ${i + 1}`))
    r.onEvent(LLMEvent.turnComplete(`attempt ${i + 1} response...`))
    r.onEvent(LLMEvent.cost(25, 400, 0, 'canned'))
    r.onEvent(LLMEvent.qualityFeedback(quality, null))
    const d = r.decide()
    console.log(
      `  turn ${i + 1}: quality=${quality.toFixed(2)} total_out=${r.totalTokensOut()} decision=${d.kind}`
    )
    if (d.isCircuitBreak()) {
      console.log(`    reason.kind = ${d.reason.kind}`)
      console.log(`    suggestion  = ${d.suggestion}`)
      break
    }
  }
  console.log()
}

function implicitCorrectionExample() {
  console.log('── implicit correction (timing-based) ─────────')
  const r = Regulator.forUser('carol')
  r.withImplicitCorrectionWindowSecs(0.5)

  // Turn 1
  r.onEvent(LLMEvent.turnStart('Refactor fetch_user to be async'))
  r.onEvent(LLMEvent.turnComplete('(unsatisfactory response 1)'))

  // Fast retry on same cluster — synthetic correction recorded
  await new Promise((resolve) => setTimeout(resolve, 20))
  r.onEvent(LLMEvent.turnStart('Fix the fetch_user async refactoring'))
  console.log(`  after fast retry: implicit_corrections = ${r.implicitCorrectionsCount()}`)

  // Check metrics snapshot for observability integration
  console.log(`  metrics snapshot:`)
  const metrics = r.metricsSnapshot()
  for (const [key, value] of Object.entries(metrics)) {
    console.log(`    ${key}: ${value}`)
  }
  console.log()
}

function toolLoopExample() {
  console.log('── tool-call loop detection ───────────────────')
  const r = Regulator.forUser('dave')
  r.onEvent(LLMEvent.turnStart('Find the user with id 42'))

  for (let i = 0; i < 5; i++) {
    r.onEvent(LLMEvent.toolCall('search_orders', `{"attempt": ${i}}`))
    r.onEvent(LLMEvent.toolResult('search_orders', true, 120n, null))
  }

  const d = r.decide()
  console.log(`  tool_total_calls = ${r.toolTotalCalls()}`)
  console.log(`  decision.kind    = ${d.kind}`)
  if (d.isCircuitBreak()) {
    console.log(`  reason.kind             = ${d.reason.kind}`)
    console.log(`  reason.toolName         = ${d.reason.toolName}`)
    console.log(`  reason.consecutiveCount = ${d.reason.consecutiveCount}`)
  }
  console.log()
}

// Run — Node top-level await is supported in .mjs since v14.
scopeDriftExample()
costBreakExample()
await implicitCorrectionExample()
toolLoopExample()
