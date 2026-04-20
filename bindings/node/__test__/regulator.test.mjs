// Behavioral tests for the Node.js bindings.
//
// Mirrors the Python `test_regulator.py` suite: construct events,
// fire decide() at the right moments, check kind + variant-specific
// attributes. Run with Node's built-in test runner:
//
//     node --test __test__/

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { Regulator, LLMEvent, llmEventsFromOtelSpanJson } from '../index.js'

test('empty regulator returns continue', () => {
  const r = Regulator.forUser('alice')
  const d = r.decide()
  assert.equal(d.kind, 'continue')
  assert.ok(d.isContinue())
  assert.equal(d.driftScore, null)
  assert.equal(d.reason, null)
  assert.equal(d.patterns, null)
})

test('scope drift fires on drifted response', () => {
  const r = Regulator.forUser('alice')
  r.onEvent(LLMEvent.turnStart(
    'Refactor fetch_user to be async. Keep the database lookup logic unchanged.'
  ))
  r.onEvent(LLMEvent.turnComplete(
    'added counter timing retry cache wrapper handler middleware logger queue'
  ))
  r.onEvent(LLMEvent.cost(40, 180, 0, 'test'))
  const d = r.decide()
  assert.equal(d.kind, 'scope_drift_warn')
  assert.ok(d.isScopeDrift())
  assert.ok(d.driftScore >= 0.5)
  assert.ok(Array.isArray(d.driftTokens))
  assert.ok(d.driftTokens.length > 0)
})

test('cost break fires when cap and quality both trip', () => {
  const r = Regulator.forUser('bob')
  r.withCostCap(1000)
  for (let i = 0; i < 3; i++) {
    const q = [0.5, 0.35, 0.2][i]
    r.onEvent(LLMEvent.turnStart(`attempt ${i + 1}`))
    r.onEvent(LLMEvent.turnComplete(`r${i + 1}`))
    r.onEvent(LLMEvent.cost(25, 400, 0, 'test'))
    r.onEvent(LLMEvent.qualityFeedback(q, null))
  }
  const d = r.decide()
  assert.equal(d.kind, 'circuit_break')
  assert.ok(d.isCircuitBreak())
  assert.ok(['cost_cap_reached', 'quality_decline_no_recovery'].includes(d.reason.kind))
  assert.ok(typeof d.suggestion === 'string' && d.suggestion.length > 0)
})

test('tool loop detection fires on 5 consecutive same tool', () => {
  const r = Regulator.forUser('dave')
  r.onEvent(LLMEvent.turnStart('find user'))
  for (let i = 0; i < 5; i++) {
    r.onEvent(LLMEvent.toolCall('search_orders', `{"i": ${i}}`))
    r.onEvent(LLMEvent.toolResult('search_orders', true, 100n, null))
  }
  assert.equal(r.toolTotalCalls(), 5)
  const d = r.decide()
  assert.equal(d.kind, 'circuit_break')
  assert.equal(d.reason.kind, 'repeated_tool_call_loop')
  assert.equal(d.reason.toolName, 'search_orders')
  assert.ok(d.reason.consecutiveCount >= 5)
})

test('tool stats accessors', () => {
  const r = Regulator.forUser('eve')
  r.onEvent(LLMEvent.turnStart('task'))
  r.onEvent(LLMEvent.toolCall('a', null))
  r.onEvent(LLMEvent.toolResult('a', true, 50n, null))
  r.onEvent(LLMEvent.toolCall('b', null))
  r.onEvent(LLMEvent.toolResult('b', false, 30n, 'boom'))
  r.onEvent(LLMEvent.toolCall('a', null))
  r.onEvent(LLMEvent.toolResult('a', true, 20n, null))

  assert.equal(r.toolTotalCalls(), 3)
  const counts = r.toolCountsByName()
  assert.equal(counts.a, 2)
  assert.equal(counts.b, 1)
  assert.equal(r.toolTotalDurationMs(), 100n)
  assert.equal(r.toolFailureCount(), 1)
})

test('export + from_json roundtrip preserves type', () => {
  const r = Regulator.forUser('frank')
  r.onEvent(LLMEvent.turnStart('hello'))
  r.onEvent(LLMEvent.turnComplete('world'))
  r.onEvent(LLMEvent.cost(10, 20, 5, 'test'))

  const snapshot = r.exportJson()
  assert.equal(typeof snapshot, 'string')
  const parsed = JSON.parse(snapshot)
  assert.equal(typeof parsed, 'object')

  const r2 = Regulator.fromJson(snapshot)
  assert.equal(r2.userId, 'frank')
})

test('from_json throws on malformed JSON', () => {
  assert.throws(() => Regulator.fromJson('not json at all'))
})

test('inject_corrections is a no-op when no pattern', () => {
  const r = Regulator.forUser('grace')
  r.onEvent(LLMEvent.turnStart('something'))
  const prompt = 'Do the thing'
  assert.equal(r.injectCorrections(prompt), prompt)
})

test('LLMEvent factory kinds are exhaustive', () => {
  const evts = [
    LLMEvent.turnStart('x'),
    LLMEvent.token('t', 0.0, 0),
    LLMEvent.turnComplete('r'),
    LLMEvent.cost(1, 2, 3, null),
    LLMEvent.qualityFeedback(0.5, null),
    LLMEvent.userCorrection('fix it', true),
    LLMEvent.toolCall('search', null),
    LLMEvent.toolResult('search', true, 100n, null),
  ]
  const kinds = new Set(evts.map((e) => e.kind))
  const expected = new Set([
    'turn_start', 'token', 'turn_complete', 'cost',
    'quality_feedback', 'user_correction', 'tool_call', 'tool_result',
  ])
  assert.deepEqual(kinds, expected)
})

test('metrics snapshot exposes stable keys', () => {
  const r = Regulator.forUser('m')
  r.withCostCap(5000)
  const snap = r.metricsSnapshot()
  const expectedKeys = [
    'noos.confidence',
    'noos.logprob_coverage',
    'noos.total_tokens_out',
    'noos.cost_cap_tokens',
    'noos.tool_total_calls',
    'noos.tool_total_duration_ms',
    'noos.tool_failure_count',
    'noos.implicit_corrections_count',
    'noos.scope_drift_threshold',
  ]
  for (const key of expectedKeys) {
    assert.ok(key in snap, `missing metric key ${key}`)
    assert.equal(typeof snap[key], 'number')
  }
  assert.equal(snap['noos.cost_cap_tokens'], 5000)
  assert.equal(snap['noos.scope_drift_threshold'], 0.5)
})

test('scope drift threshold builder accepts in range', () => {
  const r = Regulator.forUser('u')
  r.withScopeDriftThreshold(0.7)
  assert.equal(r.scopeDriftThreshold(), 0.7)
})

test('scope drift threshold builder rejects invalid', () => {
  const r = Regulator.forUser('u')
  assert.throws(() => r.withScopeDriftThreshold(-0.1))
  assert.throws(() => r.withScopeDriftThreshold(1.1))
  assert.throws(() => r.withScopeDriftThreshold(Number.NaN))
  assert.throws(() => r.withScopeDriftThreshold(Number.POSITIVE_INFINITY))
})

test('implicit correction off by default', () => {
  const r = Regulator.forUser('u')
  r.onEvent(LLMEvent.turnStart('Refactor fetch_user to be async'))
  r.onEvent(LLMEvent.turnComplete('resp'))
  r.onEvent(LLMEvent.turnStart('Fix the fetch_user async refactoring'))
  assert.equal(r.implicitCorrectionsCount(), 0)
})

test('implicit correction fires on fast same-cluster retry', async () => {
  const r = Regulator.forUser('u')
  r.withImplicitCorrectionWindowSecs(0.5)
  r.onEvent(LLMEvent.turnStart('Refactor fetch_user to be async'))
  r.onEvent(LLMEvent.turnComplete('(unsatisfactory)'))
  await new Promise((resolve) => setTimeout(resolve, 20))
  r.onEvent(LLMEvent.turnStart('Fix the fetch_user async refactoring'))
  assert.equal(r.implicitCorrectionsCount(), 1)
})

test('with_implicit_correction_window rejects non-positive', () => {
  const r = Regulator.forUser('u')
  assert.throws(() => r.withImplicitCorrectionWindowSecs(0))
  assert.throws(() => r.withImplicitCorrectionWindowSecs(-1))
  assert.throws(() => r.withImplicitCorrectionWindowSecs(NaN))
})

test('OTel span JSON → LLMEvent array', () => {
  const span = {
    attributes: {
      'gen_ai.system': 'anthropic',
      'gen_ai.usage.input_tokens': 25,
      'gen_ai.usage.output_tokens': 800,
    },
    events: [
      { name: 'gen_ai.user.message', attributes: { content: 'find order 42' } },
      { name: 'gen_ai.assistant.message', attributes: { content: 'found' } },
    ],
    start_time_unix_nano: 1_000_000_000,
    end_time_unix_nano: 1_500_000_000,
  }
  const events = llmEventsFromOtelSpanJson(JSON.stringify(span))
  assert.equal(events.length, 3)
  const kinds = events.map((e) => e.kind)
  assert.deepEqual(kinds, ['turn_start', 'turn_complete', 'cost'])
})

test('OTel empty span → empty array', () => {
  assert.deepEqual(llmEventsFromOtelSpanJson('{}'), [])
})

test('OTel malformed JSON throws', () => {
  assert.throws(() => llmEventsFromOtelSpanJson('not json'))
})
