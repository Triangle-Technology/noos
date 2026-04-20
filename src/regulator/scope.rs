//! Scope tracker — detects when an LLM response drifts beyond the
//! keywords of the user's task.
//!
//! **Scope note (P1 / P9b)**: like the other `regulator` sub-modules,
//! this is an I/O adapter, not a cognitive module. Keyword extraction
//! reuses [`cognition::detector::extract_topics`](crate::cognition::detector::extract_topics)
//! per P3 — that utility strips stop-words and short tokens and returns
//! a small sorted set of "meaningful words". Here those sets are
//! treated as opaque keyword bags, not cognitive topic models. The only
//! computation performed is a set-difference ratio — no sentiment
//! lexicon, no topic reasoning. P1 applies to the wrapped
//! [`CognitiveSession`](crate::session::CognitiveSession); P9b is
//! satisfied by construction.
//!
//! ## Gating (P10)
//!
//! This module produces the [`Decision::ScopeDriftWarn`] signal via
//! [`Regulator::decide`].
//!
//! - **Suppresses**: [`Decision::ProceduralWarning`], [`Decision::Continue`].
//! - **Suppressed by**: every [`Decision::CircuitBreak`] variant
//!   ([`CircuitBreakReason::CostCapReached`],
//!   [`CircuitBreakReason::QualityDeclineNoRecovery`],
//!   [`CircuitBreakReason::RepeatedToolCallLoop`]) — urgent stop signals
//!   dominate semantic warnings.
//! - **Inactive when**: [`ScopeTracker::drift_score`] returns `None`
//!   (either the task or the response keyword bag is empty — no
//!   baseline to measure drift against).
//!
//! [`Decision::ScopeDriftWarn`]: super::Decision::ScopeDriftWarn
//! [`Decision::ProceduralWarning`]: super::Decision::ProceduralWarning
//! [`Decision::Continue`]: super::Decision::Continue
//! [`Decision::CircuitBreak`]: super::Decision::CircuitBreak
//! [`CircuitBreakReason::CostCapReached`]: super::CircuitBreakReason::CostCapReached
//! [`CircuitBreakReason::QualityDeclineNoRecovery`]: super::CircuitBreakReason::QualityDeclineNoRecovery
//! [`CircuitBreakReason::RepeatedToolCallLoop`]: super::CircuitBreakReason::RepeatedToolCallLoop
//! [`Regulator::decide`]: super::Regulator::decide
//!
//! ## Drift metric (MVP)
//!
//! ```text
//! drift_score = |response_keywords \ task_keywords| / |response_keywords|
//! ```
//!
//! - `0.0` = every meaningful word in the response also appears in the
//!   task (minimal expansion).
//! - `1.0` = the response's keyword bag is disjoint from the task's
//!   (the Session 18 plan example: task "refactor function" vs
//!   response "add logging + error handling").
//! - Returns `None` when either keyword bag is empty (no baseline).
//!
//! The metric is response-centric: it asks *"how much of the response
//! has no anchor in the task"*. It will flag verbose or pedagogical
//! responses even when they're on-topic (discussing background
//! concepts), which is the expected false-positive regime the plan's
//! Session 18 decision checkpoint measures. Embedding-based detection
//! is deferred until the 10-case false-positive audit runs.
//!
//! ## Reuse-vs-roll decision for `detector` (Session 18)
//!
//! Reused, three functions:
//!
//! - [`extract_topics`](crate::cognition::detector::extract_topics) —
//!   keyword extraction (stop-word filter, min-length 3, top-10
//!   alphabetical). Already used by `memory/retrieval.rs`.
//! - [`to_topic_set`](crate::cognition::detector::to_topic_set) —
//!   lowercased `HashSet<String>` builder for O(1) membership lookup.
//! - [`count_topic_overlap`](crate::cognition::detector::count_topic_overlap) —
//!   intersection size between a token list and a topic set.
//!
//! Rolling any of these inline would duplicate the same lowercasing +
//! stop-word logic the crate already ships in one place (P3 violation
//! precedent: the Session 18 first-pass did this and was corrected
//! before Session 19). The only scope-specific logic below is the
//! set-difference *subtraction* (`|response| − overlap`) and the
//! `drift_tokens` filter — neither has a pre-existing utility.
//!
//! If the Session 18 decision checkpoint surfaces a specific regime
//! where `extract_topics` is wrong for scope-drift (e.g. alphabetical
//! sort clipping important long-tail keywords after position 10),
//! revisit then — not preemptively.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::cognition::detector;

// ── Constants ──────────────────────────────────────────────────────────

/// Drift score at or above which [`Regulator::decide`](super::Regulator::decide)
/// emits [`Decision::ScopeDriftWarn`](super::Decision::ScopeDriftWarn).
///
/// Set at 0.5 — the plan test target ("refactor function" vs "add
/// logging + error handling") produces `drift_score = 1.0` (disjoint
/// keyword bags), so this threshold fires unambiguously on the target
/// case while leaving headroom for on-topic-but-verbose responses in
/// the [0.3, 0.5) band. The Session 18 decision checkpoint measures
/// false-positive rate on 10 hand-crafted cases; if FPR > 20%, this
/// constant (or the metric itself) is the first knob to revisit.
pub const DRIFT_WARN_THRESHOLD: f64 = 0.5;

/// Minimum length for morphological prefix matching in [`task_anchored`].
///
/// At 4, `user` (len 4) can prefix-match `users` (morphological variant)
/// but `ai` (2) cannot match `aim` (spurious). Short tech terms in
/// [`detector::extract_meaningful_words`] are kept in the keyword bag
/// (via [`detector::extract_meaningful_words`]'s `TECH_TERMS` carve-out)
/// but NOT eligible for prefix expansion — they match exactly only.
const MIN_STEM_PREFIX_LEN: usize = 4;

/// True when `response_token` shares a morphological root with any
/// `task_token`, either by exact lowercased membership in `task_set`
/// or by bidirectional prefix match (the shorter of the pair is a
/// prefix of the longer, both ≥ [`MIN_STEM_PREFIX_LEN`]).
///
/// Catches the common English morphological class the 2026-04-20
/// empirical Gemini smoke surfaced — `async` (task) vs `asynchronous`
/// (response), `await` vs `awaited`, `configure` vs `configured`. Does
/// NOT resolve synonyms (`make async` ≠ `non-blocking`) nor rank-11
/// alphabetical truncation — those remain documented limitations
/// addressed by the adversarial tests.
///
/// O(|task_tokens|) per response token; with `extract_topics`' top-10
/// cap this is ≤ 10 comparisons — negligible versus the µs-scale cost
/// of a `Regulator::decide` call.
fn task_anchored(
    response_token: &str,
    task_set: &HashSet<String>,
    task_tokens: &[String],
) -> bool {
    if task_set.contains(response_token) {
        return true;
    }
    if response_token.len() < MIN_STEM_PREFIX_LEN {
        return false;
    }
    for task_token in task_tokens {
        if task_token.len() < MIN_STEM_PREFIX_LEN {
            continue;
        }
        let (shorter, longer) = if task_token.len() <= response_token.len() {
            (task_token.as_str(), response_token)
        } else {
            (response_token, task_token.as_str())
        };
        if longer.starts_with(shorter) {
            return true;
        }
    }
    false
}

// ── ScopeTracker ───────────────────────────────────────────────────────

/// Per-turn scope state: task keywords (from the user's message) and
/// response keywords (from the LLM's output).
///
/// Lifecycle is per-turn: [`set_task`](Self::set_task) clears any
/// previous response before loading the new task keywords, and
/// [`set_response`](Self::set_response) is called once the LLM output
/// settles. [`drift_score`](Self::drift_score) returns `None` while
/// either side is empty, so an in-progress turn never triggers a false
/// warning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScopeTracker {
    /// Keywords extracted from the user's task message (opaque, in the
    /// `detector::extract_topics` sense: lowercased, stop-words
    /// filtered, min-length 3, top-10 alphabetical).
    task_keywords: Vec<String>,
    /// Keywords extracted from the LLM's response.
    response_keywords: Vec<String>,
}

impl ScopeTracker {
    /// Construct an empty tracker. Equivalent to `Self::default()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mutable: set the task keywords from the user's message and clear
    /// any stale response keywords from a previous turn. Requires
    /// mutation because scope state is per-turn — a fresh task must
    /// reset the baseline.
    pub fn set_task(&mut self, user_message: &str) {
        self.task_keywords = detector::extract_topics(user_message);
        self.response_keywords.clear();
    }

    /// Mutable: set the response keywords. Requires mutation because
    /// the tracker accumulates per-turn evidence before drift can be
    /// computed.
    pub fn set_response(&mut self, full_response: &str) {
        self.response_keywords = detector::extract_topics(full_response);
    }

    /// Keywords extracted from the most recent `set_task` call. Empty
    /// before any task is set.
    pub fn task_tokens(&self) -> &[String] {
        &self.task_keywords
    }

    /// Keywords extracted from the most recent `set_response` call.
    /// Empty before any response is set or after `set_task` resets the
    /// turn.
    pub fn response_tokens(&self) -> &[String] {
        &self.response_keywords
    }

    /// Drift score in `[0, 1]` — fraction of response keywords that do
    /// not share a morphological root with any task keyword. Returns
    /// `None` when either bag is empty (no baseline for comparison).
    ///
    /// Anchoring is two-tier: exact lowercased membership in the task
    /// set OR bidirectional prefix match via [`task_anchored`] (both
    /// tokens ≥ [`MIN_STEM_PREFIX_LEN`]). The prefix tier absorbs the
    /// common English morphological class (`async` ↔ `asynchronous`,
    /// `await` ↔ `awaited`, `configure` ↔ `configured`) that the
    /// 2026-04-20 empirical Gemini smoke flagged as a false-positive
    /// source on technical prompts.
    ///
    /// See module docs for the metric rationale and the remaining
    /// known false-positive regimes (synonyms, rank-11 truncation,
    /// verbose on-topic).
    pub fn drift_score(&self) -> Option<f64> {
        if self.task_keywords.is_empty() || self.response_keywords.is_empty() {
            return None;
        }
        let task_set = detector::to_topic_set(&self.task_keywords);
        let non_task = self
            .response_keywords
            .iter()
            .filter(|r| !task_anchored(&r.to_lowercase(), &task_set, &self.task_keywords))
            .count();
        Some(non_task as f64 / self.response_keywords.len() as f64)
    }

    /// Response keywords that are NOT morphologically anchored in the
    /// task — the concrete "drift" set surfaced in
    /// [`Decision::ScopeDriftWarn`](super::Decision::ScopeDriftWarn).
    ///
    /// Empty when no response is set or when every response keyword is
    /// anchored in the task (exactly or via prefix). Ordering mirrors
    /// [`response_tokens`](Self::response_tokens) (alphabetical).
    pub fn drift_tokens(&self) -> Vec<String> {
        let task_set = detector::to_topic_set(&self.task_keywords);
        self.response_keywords
            .iter()
            .filter(|k| !task_anchored(&k.to_lowercase(), &task_set, &self.task_keywords))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tracker_has_no_drift_score() {
        let tracker = ScopeTracker::new();
        assert!(tracker.drift_score().is_none());
        assert!(tracker.task_tokens().is_empty());
        assert!(tracker.response_tokens().is_empty());
        assert!(tracker.drift_tokens().is_empty());
    }

    #[test]
    fn task_only_has_no_drift_score() {
        // Without a response, there's nothing to compare against —
        // the tracker returns None rather than claiming zero drift.
        let mut tracker = ScopeTracker::new();
        tracker.set_task("refactor this function to be async");
        assert!(!tracker.task_tokens().is_empty());
        assert!(tracker.drift_score().is_none());
    }

    #[test]
    fn response_only_has_no_drift_score() {
        // Without a task, there's no baseline — same None contract.
        let mut tracker = ScopeTracker::new();
        tracker.set_response("Here is the refactored async function.");
        assert!(!tracker.response_tokens().is_empty());
        assert!(tracker.drift_score().is_none());
    }

    #[test]
    fn plan_example_flags_high_drift() {
        // Session 18 test target from the architecture plan:
        // task "refactor function" vs response "add logging + error
        // handling" → drift_score > 0.3. With disjoint keyword bags we
        // expect the maximum score 1.0.
        let mut tracker = ScopeTracker::new();
        tracker.set_task("refactor this function to be async");
        tracker.set_response("add logging and error handling");
        let drift = tracker.drift_score().expect("both sides populated");
        assert!(
            drift > 0.3,
            "plan test target must produce drift > 0.3 (got {drift})"
        );
        // Additionally: at or above the warning threshold — the
        // Regulator-level test confirms this triggers ScopeDriftWarn.
        assert!(drift >= DRIFT_WARN_THRESHOLD);
    }

    #[test]
    fn on_task_response_keeps_drift_low() {
        // Response that echoes and confirms the task keywords should
        // sit well below the warning threshold. We intentionally
        // constrain the response vocabulary so the metric behaves
        // predictably; verbose on-task responses drift higher and are
        // the known false-positive regime measured by the decision
        // checkpoint.
        let mut tracker = ScopeTracker::new();
        tracker.set_task("refactor the async function");
        tracker.set_response("refactor async function");
        let drift = tracker.drift_score().expect("both sides populated");
        assert!(
            drift < DRIFT_WARN_THRESHOLD,
            "minimal on-task response should stay under warning threshold (got {drift})"
        );
    }

    #[test]
    fn drift_tokens_only_contains_non_task_keywords() {
        let mut tracker = ScopeTracker::new();
        tracker.set_task("refactor the async function");
        tracker.set_response("add logging telemetry for the async function");
        let drift_tokens = tracker.drift_tokens();
        // Task keywords: {refactor, async, function} (stop-words "the"
        // filtered, min-len 3). Drift keywords should NOT include any
        // task keyword.
        for token in tracker.task_tokens() {
            assert!(
                !drift_tokens.contains(token),
                "drift_tokens must not contain task keyword {token:?}"
            );
        }
        // Drift set should be non-empty on this input (added logging /
        // telemetry / add / for).
        assert!(!drift_tokens.is_empty());
    }

    #[test]
    fn set_task_resets_previous_response() {
        // A new task invalidates the previous turn's response — the
        // tracker must clear it so drift isn't computed against stale
        // data.
        let mut tracker = ScopeTracker::new();
        tracker.set_task("refactor the async function");
        tracker.set_response("add logging and error handling");
        assert!(tracker.drift_score().is_some());

        // New turn starts — response cleared, only task present.
        tracker.set_task("explain tokio runtime");
        assert!(tracker.response_tokens().is_empty());
        assert!(
            tracker.drift_score().is_none(),
            "new task must clear stale response"
        );
    }

    #[test]
    fn drift_score_bounded_zero_to_one() {
        // Defensive: whatever the inputs, the metric must stay in
        // [0, 1]. Otherwise downstream threshold logic breaks.
        let mut tracker = ScopeTracker::new();
        tracker.set_task("task keyword");
        tracker.set_response("completely different response content");
        let drift = tracker.drift_score().expect("both sides populated");
        assert!((0.0..=1.0).contains(&drift));
    }

    /// Design checkpoint: false-positive-rate audit on 10 hand-crafted
    /// cases.
    ///
    /// Does keyword-overlap produce believable drift scores on real text?
    /// If total error rate (FPR + FNR) > 20% on 10 hand-crafted test cases,
    /// reconsider embedding-based detection.
    ///
    /// This test encodes the audit. Each case has a manually-assigned
    /// ground-truth label (`should_flag`) and the test asserts that
    /// the keyword-overlap metric mis-classifies no more than 2 of 10
    /// (≤ 20% total error rate, which bounds FPR + FNR above the
    /// specific 20% FPR figure).
    #[test]
    fn decision_checkpoint_fpr_on_hand_crafted_cases() {
        // Each case: (task, response, expected-to-flag).
        // Drift-expected (should flag):
        //   D1 — plan target: refactor vs logging/errors
        //   D2 — unrelated recipe
        //   D3 — JS answer to SQL question
        //   D4 — architecture musings in a bug-fix request
        //   D5 — cake recipe when asked about docker
        // Non-drift (should NOT flag):
        //   N1 — minimal on-task echo
        //   N2 — on-task with light restatement
        //   N3 — explain-tokio → tokio-focused answer
        //   N4 — fix-error → fix-error answer
        //   N5 — explain-jwt → jwt-focused answer
        let cases: &[(&str, &str, bool)] = &[
            (
                "refactor this function to be async",
                "add logging and error handling",
                true, // D1: plan target
            ),
            (
                "explain tokio runtime",
                "here is a recipe for chocolate cake with frosting",
                true, // D2
            ),
            (
                "help me with SQL queries",
                "JavaScript frameworks overview: React, Vue, Angular",
                true, // D3
            ),
            (
                "fix the authentication bug",
                "my thoughts on microservice architecture patterns",
                true, // D4
            ),
            (
                "explain docker containers",
                "chocolate cake baking instructions with butter",
                true, // D5
            ),
            (
                "refactor async function",
                "refactor async function",
                false, // N1: identical → 0 drift
            ),
            (
                "refactor the async function",
                "refactored async function returned",
                false, // N2: stemming would help but on-task overall
            ),
            (
                "tokio async runtime rust",
                "tokio async runtime rust futures scheduling",
                false, // N3: response extends task keywords
            ),
            (
                "fix error authentication rust",
                "fix error authentication rust verify",
                false, // N4: response stays within task vocabulary
            ),
            (
                "jwt token format explain",
                "jwt token format explain signature",
                false, // N5: minimal extension
            ),
        ];

        let mut mis_classifications = 0usize;
        let mut report = String::new();
        for (task, response, should_flag) in cases {
            let mut tracker = ScopeTracker::new();
            tracker.set_task(task);
            tracker.set_response(response);
            let drift = tracker.drift_score().unwrap_or(0.0);
            let flagged = drift >= DRIFT_WARN_THRESHOLD;
            let correct = flagged == *should_flag;
            if !correct {
                mis_classifications += 1;
            }
            report.push_str(&format!(
                "  [{}] task={:?} resp={:?} drift={:.2} flag={} expected={} {}\n",
                if correct { "✓" } else { "✗" },
                task,
                response,
                drift,
                flagged,
                should_flag,
                if correct { "" } else { "← MIS" },
            ));
        }

        let total = cases.len();
        let error_rate = mis_classifications as f64 / total as f64;
        assert!(
            error_rate <= 0.2,
            "decision checkpoint failed: {mis_classifications}/{total} \
             mis-classified ({:.0}% error rate, bar ≤ 20%). Report:\n{report}",
            error_rate * 100.0
        );
    }

    // ── Adversarial tests (Session 31) ─────────────────────────────
    //
    // Session 30 audit flagged the keyword-bag metric as under-tested
    // against known failure modes. These tests document the metric's
    // behaviour on those shapes so future contributors (and skeptics)
    // can see exactly what is and isn't caught.

    #[test]
    fn adversarial_rank_11_keyword_truncation_causes_false_positive() {
        // KNOWN LIMITATION — documented, not a bug.
        //
        // `extract_topics` sorts meaningful words alphabetically then
        // truncates to top 10. On long task prompts with >10 meaningful
        // words, alphabetically-late keywords get dropped. A response
        // that reuses ONLY those dropped keywords appears as 100% drift
        // even though it's on-topic.
        //
        // Task meaningful words (alphabetical):
        //   async, authentication, cases, error, handling, module,
        //   operations, proper, refactor, support, timeout, user, yaml
        // Top-10 kept: async...support
        // Truncated:   timeout, user, yaml
        //
        // Response reuses only {users, configure, timeout, via, yaml}
        // → overlap 0 against task top-10 → drift=1.0.
        let mut t = ScopeTracker::new();
        t.set_task(
            "refactor user authentication module support async \
             operations proper error handling timeout cases yaml",
        );
        t.set_response("users should configure timeout via yaml");
        let drift = t.drift_score().expect("both sides populated");
        assert!(
            drift >= DRIFT_WARN_THRESHOLD,
            "rank-11 truncation gotcha documented: on-topic response \
             scored drift={drift}. Embedding-based fallback for \
             borderline cases is tracked as a post-0.3.0 follow-up."
        );
    }

    #[test]
    fn adversarial_synonyms_flag_as_drift() {
        // KNOWN LIMITATION — keyword-bag cannot recognize semantic
        // equivalence across surface forms. "make async" ~ "non-blocking"
        // but share zero tokens → drift=1.0.
        //
        // The README's competitor matrix explicitly notes this:
        // "Scope drift" is Noos's weakest wedge versus Phoenix/Arize
        // embedding-based drift. This test pins the known behaviour
        // so any future swap (e.g., an optional embedding-based path)
        // starts with a clear baseline.
        let mut t = ScopeTracker::new();
        t.set_task("make this function async");
        t.set_response("convert to non-blocking code");
        let drift = t.drift_score().expect("both sides populated");
        assert!(
            drift >= DRIFT_WARN_THRESHOLD,
            "synonym-equivalent response flags as drift={drift} — \
             expected for surface-form metric"
        );
    }

    #[test]
    fn adversarial_verbose_on_topic_can_overwhelm_task_keywords() {
        // KNOWN LIMITATION — a verbose on-topic explanation expands
        // the response keyword bag with peripheral vocabulary. If
        // alphabetical truncation of the RESPONSE drops every
        // task-matching word, drift can score high on a correct
        // answer.
        //
        // Task: {explain, runtime, tokio}
        // Response (after extraction + alphabetical top-10): typically
        // {applications, async, asynchronous, building, channels,
        // event-driven, io, network, networking, operations} — none
        // of which are in the task set. Drift → 1.0.
        //
        // This is the false-positive regime the module docs call out
        // as "on-topic but verbose." Callers with latency budget to
        // spare can pair Noos's keyword check with an embedding-based
        // drift check on borderline-high scores.
        let mut t = ScopeTracker::new();
        t.set_task("explain tokio runtime");
        t.set_response(
            "Tokio is an async runtime for Rust that provides \
             event-driven networking, timers, scheduled tasks, IO \
             operations, channels, synchronization primitives, and \
             utilities for building asynchronous network applications \
             at scale",
        );
        let drift = t.drift_score().expect("both sides populated");
        assert!(
            drift > 0.5,
            "verbose-on-topic false-positive documented: drift={drift}"
        );
    }

    #[test]
    fn case_insensitive_matching_holds() {
        // Regression guard: `extract_topics` lowercases, so identical
        // text with different case must score as zero drift.
        let mut t = ScopeTracker::new();
        t.set_task("REFACTOR async FUNCTION");
        t.set_response("refactor ASYNC function");
        let drift = t.drift_score().expect("both sides populated");
        assert!(
            drift < 0.1,
            "case-insensitive matching must treat identical text as \
             zero drift (got {drift})"
        );
    }

    // ── Morphological prefix matching (2026-04-20 Gemini smoke fix) ─────

    #[test]
    fn morphological_prefix_match_anchors_async_and_asynchronous() {
        // 2026-04-20 empirical smoke reproducer: Gemini answered
        // "Explain async/await in JavaScript" with response using
        // "asynchronous" + "awaited" — surface-form different from
        // "async" + "await" in the task but same morphological root.
        // Before fix: both counted as drift, bumping score to 0.8.
        // After fix: prefix match anchors them, drift reduced.
        let mut t = ScopeTracker::new();
        t.set_task("Explain what async/await does in JavaScript. Two sentences maximum.");
        t.set_response(
            "async/await provides a cleaner, more synchronous-looking syntax for \
             writing asynchronous code in JavaScript, making it easier to manage \
             operations that take time without blocking. An async function \
             implicitly returns a Promise, and await pauses execution until \
             awaited Promise settles.",
        );
        let drift = t.drift_score().expect("both sides populated");
        // With morphological matching `async` anchors `asynchronous`,
        // `await` anchors `awaited`. Drift score drops but stays above
        // zero because the response adds genuinely new vocabulary
        // (cleaner, blocking, code, etc). Regression guard: after-fix
        // score MUST be lower than the pre-fix 0.8 baseline.
        assert!(
            drift < 0.8,
            "prefix-match fix must reduce drift below pre-fix baseline 0.8 (got {drift})"
        );
    }

    #[test]
    fn morphological_prefix_match_bidirectional() {
        // Task uses the longer form, response uses the shorter form —
        // the same prefix-match logic must fire in both directions.
        let mut t = ScopeTracker::new();
        t.set_task("asynchronous programming patterns");
        t.set_response("async code patterns");
        let drift = t.drift_score().expect("both sides populated");
        // With bidirectional matching `async` (response) shares prefix
        // with `asynchronous` (task) → anchored. `code` is new → drift.
        // `patterns` exact match. Drift = 1 / 3 ≈ 0.33 < 0.5 threshold.
        assert!(
            drift < DRIFT_WARN_THRESHOLD,
            "bidirectional prefix match should keep this below threshold (got {drift})"
        );
    }

    #[test]
    fn morphological_prefix_match_respects_min_length() {
        // Guard against over-matching on short tokens: a 3-letter task
        // word must NOT anchor an unrelated response word that happens
        // to share those 3 letters as prefix. `MIN_STEM_PREFIX_LEN = 4`
        // enforces this.
        let mut t = ScopeTracker::new();
        t.set_task("fix bug"); // extract: [bug, fix] — both len 3
        t.set_response("fixture bugs arguments"); // extract: [arguments, bugs, fixture]
        // `fix` (len 3) is NOT eligible for prefix matching. `fixture`
        // must NOT falsely anchor to `fix`. Only exact match.
        // - arguments: no exact, no prefix match eligible → drift
        // - bugs: no exact (task has `bug`), prefix? `bug` len 3 NOT
        //   eligible for prefix. `bugs` (4) vs `bug` (3) — both directions
        //   require len ≥ 4. `bug` can't qualify. → drift
        // - fixture: same story → drift
        // Drift = 3/3 = 1.0.
        let drift = t.drift_score().expect("both sides populated");
        assert!(
            drift >= DRIFT_WARN_THRESHOLD,
            "short-token drift must not be masked by prefix match (got {drift})"
        );
    }

    #[test]
    fn morphological_prefix_match_does_not_mask_real_drift() {
        // Truly disjoint vocabularies must still flag. The prefix match
        // anchors morphological variants, not arbitrary prefix overlaps
        // with unrelated content.
        let mut t = ScopeTracker::new();
        t.set_task("explain tokio runtime");
        t.set_response("chocolate cake baking instructions");
        // No task token is prefix of any response token (or vice-versa).
        // Must still flag as drift.
        let drift = t.drift_score().expect("both sides populated");
        assert!(
            drift >= DRIFT_WARN_THRESHOLD,
            "fully disjoint vocabulary must still flag drift (got {drift})"
        );
    }

    #[test]
    fn morphological_prefix_match_drift_tokens_excludes_anchored() {
        // Sanity: `drift_tokens` must reflect the same anchoring logic
        // as `drift_score` — a response token that prefix-matches a
        // task token should NOT appear in the drift token list.
        let mut t = ScopeTracker::new();
        t.set_task("refactor async function");
        t.set_response("refactored async function completely");
        let drift_tokens = t.drift_tokens();
        // `refactored` prefix-matches `refactor` → anchored, NOT in drift.
        // `completely` (10) — no task prefix → drift.
        assert!(
            !drift_tokens.contains(&"refactored".to_string()),
            "`refactored` should be morphologically anchored to `refactor` \
             (drift_tokens = {drift_tokens:?})"
        );
        assert!(
            drift_tokens.contains(&"completely".to_string()),
            "`completely` has no task anchor and MUST appear in drift_tokens \
             (got {drift_tokens:?})"
        );
    }

    #[test]
    fn non_english_response_flags_drift() {
        // Regression guard: Unicode word splitting must still
        // produce keywords, and a disjoint-vocabulary response must
        // flag as high drift. The existing English-only stop-word
        // list does NOT cover other languages, so non-English text
        // produces a large keyword bag with zero overlap on an
        // English task — drift → 1.0, correctly reading as
        // "off-topic for the task's language and content".
        let mut t = ScopeTracker::new();
        t.set_task("explain tokio runtime");
        t.set_response("dịch chương trình sang tiếng Việt hoàn toàn");
        let drift = t.drift_score().expect("both sides populated");
        assert!(
            drift >= DRIFT_WARN_THRESHOLD,
            "fully non-overlapping vocabulary must flag high drift \
             (got {drift})"
        );
    }
}
