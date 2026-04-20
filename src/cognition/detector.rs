//! Shared detection utilities — pattern scoring, topic extraction, overlap.
//!
//! P3: single source of truth for topic/pattern operations used by
//! thalamic_gate, belief_state, world_model, retrieval, emotional valence,
//! and more.

use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

/// A group of regex patterns with an associated weight.
pub struct PatternGroup {
    pub patterns: Vec<Regex>,
    pub weight: f64,
}

/// Score input against pattern groups. Returns 0.0-1.0 (capped).
pub fn score_patterns(input: &str, groups: &[PatternGroup]) -> f64 {
    let mut score = 0.0;
    for group in groups {
        for pattern in &group.patterns {
            if pattern.is_match(input) {
                score += group.weight;
                break; // One match per group is enough
            }
        }
    }
    score.min(1.0)
}

// ── Stop words ──────────────────────────────────────────────────────

static STOP_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "has", "his", "how", "its", "let",
        "may", "new", "now", "old", "see", "way", "who", "did", "got", "get",
        "him", "yet", "say", "she", "too", "use", "own", "why", "try", "ran",
        "run", "set", "put", "add", "big", "end", "far", "few", "saw", "men",
        "two", "ask", "ago", "per", "any",
        "this", "that", "with", "from", "have", "been", "will", "would", "could",
        "should", "about", "their", "there", "these", "those", "which", "where",
        "when", "what", "into", "also", "more", "most", "some", "such", "than",
        "then", "them", "they", "very", "just", "only", "does", "each", "other",
        "being", "were", "here", "both", "between", "through", "during", "before",
        "after", "above", "below", "under", "over", "again", "further", "once",
    ]
    .into_iter()
    .collect()
});

static TECH_TERMS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "ai", "ml", "go", "ci", "cd", "k8s", "s3", "ec2", "ui", "ux",
        "db", "os", "ip", "vm", "io", "rx", "dl", "lr", "qa",
    ]
    .into_iter()
    .collect()
});

static WORD_SPLITTER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\s,;:!?\.\(\)\[\]\{\}/\\]+").expect("valid regex")
});

/// Extract meaningful words from text, filtering stop words.
///
/// `min_length`: minimum character count (words shorter than this are excluded
/// unless they're known tech terms).
pub fn extract_meaningful_words(text: &str, min_length: usize) -> HashSet<String> {
    let lower = text.to_lowercase();
    let mut result = HashSet::new();

    for word in WORD_SPLITTER.split(&lower) {
        let word = word.trim_matches(|c: char| !c.is_alphanumeric());
        if word.is_empty() {
            continue;
        }
        if STOP_WORDS.contains(word) {
            continue;
        }
        if word.len() >= min_length || TECH_TERMS.contains(word) {
            result.insert(word.to_string());
        }
    }
    result
}

/// Extract topics from text — meaningful words with min length 3,
/// deduplicated, sorted alphabetically, truncated to top 10.
///
/// **Stability note**: callers that use the output as a lookup key
/// (e.g. [`build_topic_cluster`], `LearnedState.response_strategies`
/// keying, `correction_patterns` keying) depend on the alphabetical
/// ordering AND the top-10 cap to keep cluster identity stable across
/// code changes. Do not switch this to a different ranking without
/// migrating all persisted state that keys off it.
///
/// For contexts where identity stability does NOT matter — e.g.
/// keyword-overlap comparison in [`regulator::scope`] — prefer
/// [`extract_topics_ranked`], which uses frequency + length ranking
/// and a configurable cap.
///
/// [`regulator::scope`]: crate::regulator::scope
pub fn extract_topics(text: &str) -> Vec<String> {
    let words = extract_meaningful_words(text, 3);
    let mut topics: Vec<String> = words.into_iter().collect();
    topics.sort();
    topics.truncate(10);
    topics
}

/// Extract topics from text with importance-weighted ranking, truncated
/// to `limit` keywords.
///
/// Ranking primary-to-tertiary:
///
/// 1. **Frequency DESC** — words repeated across the text are more
///    content-bearing than one-off mentions.
/// 2. **Length DESC** — when frequencies tie (common in short prompts
///    where every meaningful word appears once), longer words carry
///    more specificity than short ones. `authentication` outranks
///    `user`.
/// 3. **Alphabetical ASC** — deterministic tiebreak so repeated calls
///    on the same input produce byte-identical output.
///
/// Compared to [`extract_topics`], this function:
///
/// - Accepts a configurable `limit` (scope-drift uses 20 to absorb
///   longer prompts that alphabetical top-10 would truncate).
/// - Surfaces keywords that are frequency- or length-salient even
///   when alphabetically late (the 2026-04-20 empirical Gemini smoke
///   showed `timeout` + `yaml` getting dropped at rank-11 alphabetical
///   while being the exact words the response reused).
///
/// Does NOT change [`extract_topics`] — cluster identity + persisted
/// state stay stable. Use this function only in comparison contexts
/// (scope drift, similarity scoring) where the ordering is internal.
///
/// Same token definition as [`extract_meaningful_words`] (stop-word
/// filter, min-length 3 unless the token is in the internal
/// `TECH_TERMS` carve-out).
pub fn extract_topics_ranked(text: &str, limit: usize) -> Vec<String> {
    use std::collections::HashMap;

    let lower = text.to_lowercase();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for word in WORD_SPLITTER.split(&lower) {
        let word = word.trim_matches(|c: char| !c.is_alphanumeric());
        if word.is_empty() {
            continue;
        }
        if STOP_WORDS.contains(word) {
            continue;
        }
        if word.len() >= 3 || TECH_TERMS.contains(word) {
            *counts.entry(word.to_string()).or_insert(0) += 1;
        }
    }

    let mut ranked: Vec<(String, usize)> = counts.into_iter().collect();
    // Sort key: (-freq, -len, word) — frequency and length descending
    // (negated usize via Reverse), word ascending for determinism.
    ranked.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| b.0.len().cmp(&a.0.len()))
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(limit);
    ranked.into_iter().map(|(word, _)| word).collect()
}

/// Convert topic list to a lowercase set for O(1) comparison.
pub fn to_topic_set(topics: &[String]) -> HashSet<String> {
    topics.iter().map(|t| t.to_lowercase()).collect()
}

/// Count how many topics appear in the topic set.
pub fn count_topic_overlap(topics: &[String], topic_set: &HashSet<String>) -> usize {
    topics
        .iter()
        .filter(|t| topic_set.contains(&t.to_lowercase()))
        .count()
}

/// Topic overlap ratio: intersection / min(|a|, |b|).
pub fn topic_overlap_ratio(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let set_b = to_topic_set(b);
    let overlap = count_topic_overlap(a, &set_b);
    let min_len = a.len().min(b.len());
    if min_len == 0 {
        return 0.0;
    }
    overlap as f64 / min_len as f64
}

/// Build a topic-cluster key from the top-2 meaningful topics (sorted,
/// joined by '+').
///
/// P3 single source of cluster identity: used by `world_model` to key
/// `LearnedState.response_strategies` AND by `regulator::correction` to
/// key procedural patterns. Both paths derive clusters the same way so
/// strategy learning + correction patterns agree on what counts as
/// "the same topic".
///
/// Filters topics with length < 3 (matching `extract_topics`
/// convention), lowercases for case-insensitive match, sorts
/// alphabetically, takes the top 2, joins with `+`. Empty input or
/// all-short-tokens → empty string (caller contract: empty cluster
/// means "no identifiable topic, skip clustering").
pub fn build_topic_cluster(topics: &[String]) -> String {
    let mut sorted: Vec<String> = topics
        .iter()
        .filter(|t| t.len() >= 3)
        .map(|t| t.to_lowercase())
        .collect();
    sorted.sort();
    sorted.truncate(2);
    sorted.join("+")
}

/// Compute question ratio in content (number of '?' / number of sentences).
pub fn compute_question_ratio(content: &str) -> f64 {
    let questions = content.matches('?').count();
    // Estimate sentence count by periods, exclamations, questions, or newlines
    let sentences = content
        .chars()
        .filter(|c| *c == '.' || *c == '!' || *c == '?' || *c == '\n')
        .count()
        .max(1);
    questions as f64 / sentences as f64
}

// P9b note (2026-04-14): `is_vietnamese()` removed — it was a language-
// detection heuristic that duplicated cortical capability (LLM detects
// language trivially). The function had no non-test consumers.

/// Response strategy detected from LLM output content (back-compat wrapper).
///
/// Returns a strategy for every input — defaulting to `DirectAnswer` when no
/// positive pattern matches. Suitable for display/logging/telemetry where
/// "I don't know" isn't useful. **For reward learning prefer
/// `detect_response_strategy_safe`** — see below.
///
/// Priority: clarify-first > step-by-step > structured-analysis > execute-task > direct-answer.
pub fn detect_response_strategy(content: &str) -> crate::types::world::ResponseStrategy {
    detect_response_strategy_safe(content)
        .unwrap_or(crate::types::world::ResponseStrategy::DirectAnswer)
}

/// Response strategy detected from LLM output content, with ambiguity signal.
///
/// Returns `Some(strategy)` when a branch CLEARLY matches, `None` when no
/// branch has enough evidence — including the case where content doesn't
/// positively look like a direct answer either (e.g. two-paragraph medium-length
/// narrative prose, which could be any of several strategies).
///
/// Why this exists: the binary `detect_response_strategy` defaults to
/// `DirectAnswer` when no other branch fires. In reward-learning contexts this
/// is silent poisoning — any formatted-unusually LLM response that the detector
/// doesn't recognize gets recorded as "DirectAnswer succeeded/failed" even
/// though the actual strategy was something else. `task_eval_synthetic.rs`
/// phase 2 (2026-04-14) surfaced this: single-line numbered lists
/// ("1. First. 2. Then.") fail STEP_PATTERN (needs newline-anchored steps),
/// single-question clarifications fail the `questions >= 2` predicate, and
/// both silently become DirectAnswer in `learned.response_strategies`.
///
/// `consolidate` in `world_model.rs` uses this function to skip learning
/// when the strategy is ambiguous, avoiding the poisoning.
pub fn detect_response_strategy_safe(
    content: &str,
) -> Option<crate::types::world::ResponseStrategy> {
    use crate::types::world::ResponseStrategy;

    let len = content.len();
    let questions = content.matches('?').count();

    // Clarify-first: 2+ questions in short response (clear signal).
    if questions >= 2 && len < 800 {
        return Some(ResponseStrategy::ClarifyFirst);
    }

    // Step-by-step: newline-anchored numbered steps + sequence markers.
    static STEP_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?m)^\s*\d+[\.\)]\s").expect("valid regex")
    });
    // Case-insensitive: capitalized sentence starts ("First,", "Then,") are the
    // natural way humans write — matching only lowercase silently misdetected
    // StepByStep as DirectAnswer (allostatic_demo 2026-04-14 finding).
    // P9b: English-only sequence markers (interim). Language-agnostic
    // STEP_PATTERN (numbered list newlines) already provides the primary
    // signal; these are secondary disambiguation.
    static SEQUENCE_MARKERS: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)\b(first|then|next|finally|step\s*\d)\b").expect("valid regex")
    });
    let step_count = STEP_PATTERN.find_iter(content).count();
    let seq_count = SEQUENCE_MARKERS.find_iter(content).count();
    if step_count >= 2 && seq_count >= 2 {
        return Some(ResponseStrategy::StepByStep);
    }

    // Structured-analysis: headings or table.
    static HEADING_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?m)^#{1,4}\s").expect("valid regex")
    });
    let heading_count = HEADING_PATTERN.find_iter(content).count();
    if heading_count >= 3 || (content.contains('|') && content.lines().count() > 3) {
        return Some(ResponseStrategy::StructuredAnalysis);
    }

    // Execute-task: high code ratio.
    static CODE_BLOCK: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"```[\s\S]*?```").expect("valid regex")
    });
    let code_len: usize = CODE_BLOCK
        .find_iter(content)
        .map(|m| m.as_str().len())
        .sum();
    let code_count = CODE_BLOCK.find_iter(content).count();
    if (len > 0 && code_len as f64 / len as f64 >= 0.4) || code_count >= 3 {
        return Some(ResponseStrategy::ExecuteTask);
    }

    // Positive DirectAnswer detection: short, no questions, no numbered
    // steps, no headings, no code. This is a real detection, not a default
    // fallthrough — short direct responses are a legitimate strategy choice
    // that reward learning SHOULD pick up.
    //
    // Criteria:
    // - Length under 200 chars (~1-2 short sentences). Responses longer than
    //   this tend to be multi-paragraph narrative/analysis, which is ambiguous
    //   without structural markers — better to return None than guess.
    // - No question marks (not clarifying).
    // - No numbered items (not step-by-step).
    // - No headings (not structured).
    // - No code blocks (not execute-task).
    const DIRECT_ANSWER_MAX_LEN: usize = 200;
    if len <= DIRECT_ANSWER_MAX_LEN
        && len > 0
        && questions == 0
        && step_count == 0
        && heading_count == 0
        && code_count == 0
    {
        return Some(ResponseStrategy::DirectAnswer);
    }

    // Ambiguous — none of the positive branches fire strongly enough.
    // Reward learning should skip this response rather than silently attribute
    // it to DirectAnswer.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_topics_ranked_prefers_frequency() {
        // Words repeated should outrank one-off mentions regardless of
        // alphabetical position.
        let topics = extract_topics_ranked(
            "async async async code code pattern runtime tokio handler",
            5,
        );
        // Top-2 should be the two most-frequent: `async` (3×), `code` (2×).
        assert_eq!(topics[0], "async");
        assert_eq!(topics[1], "code");
    }

    #[test]
    fn extract_topics_ranked_length_tiebreaker() {
        // All words freq=1 — length DESC decides order. Longer words
        // surface first because they're typically more specific.
        let topics = extract_topics_ranked("refactor user authentication", 10);
        assert_eq!(topics[0], "authentication"); // len 14
        assert_eq!(topics[1], "refactor"); // len 8
        assert_eq!(topics[2], "user"); // len 4
    }

    #[test]
    fn extract_topics_ranked_alphabetical_final_tiebreaker() {
        // Freq=1, same length — alphabetical decides. This is the
        // determinism guarantee.
        let topics = extract_topics_ranked("zebra apple", 10);
        assert_eq!(topics[0], "apple");
        assert_eq!(topics[1], "zebra");
    }

    #[test]
    fn extract_topics_ranked_honours_limit() {
        let topics = extract_topics_ranked(
            "one two three four five six seven eight nine ten eleven",
            3,
        );
        assert_eq!(topics.len(), 3);
    }

    #[test]
    fn extract_topics_ranked_dedupes_within_text() {
        // Repeated words should be counted but returned once.
        let topics = extract_topics_ranked("async async async", 10);
        assert_eq!(topics.len(), 1);
        assert_eq!(topics[0], "async");
    }

    #[test]
    fn extract_topics_ranked_keeps_tech_terms_under_min_length() {
        // Short tech terms (e.g. `ai`, `io`) bypass the min-length
        // filter, matching `extract_topics` behaviour.
        let topics = extract_topics_ranked("use ai and ml for QA", 10);
        assert!(topics.contains(&"ai".to_string()));
        assert!(topics.contains(&"ml".to_string()));
        assert!(topics.contains(&"qa".to_string()));
    }

    #[test]
    fn extract_topics_filters_stop_words() {
        let topics = extract_topics("the quick brown fox jumps over the lazy dog");
        assert!(!topics.contains(&"the".to_string()));
        assert!(topics.contains(&"quick".to_string()));
        assert!(topics.contains(&"brown".to_string()));
        assert!(topics.contains(&"jumps".to_string()));
    }

    #[test]
    fn extract_topics_keeps_tech_terms() {
        let _topics = extract_topics("use ai and ml for QA");
        let words = extract_meaningful_words("use ai and ml for QA", 3);
        assert!(words.contains("ai"));
        assert!(words.contains("ml"));
        assert!(words.contains("qa"));
    }

    #[test]
    fn topic_overlap_identical() {
        let a = vec!["rust".into(), "async".into()];
        let b = vec!["rust".into(), "async".into()];
        assert!((topic_overlap_ratio(&a, &b) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn topic_overlap_no_match() {
        let a = vec!["rust".into()];
        let b = vec!["python".into()];
        assert!((topic_overlap_ratio(&a, &b)).abs() < f64::EPSILON);
    }

    #[test]
    fn topic_overlap_partial() {
        let a = vec!["rust".into(), "async".into(), "tokio".into()];
        let b = vec!["rust".into(), "python".into()];
        let ratio = topic_overlap_ratio(&a, &b);
        assert!((ratio - 0.5).abs() < f64::EPSILON); // 1 overlap / min(3,2)=2
    }

    #[test]
    fn topic_overlap_empty() {
        let a: Vec<String> = vec![];
        let b = vec!["rust".into()];
        assert_eq!(topic_overlap_ratio(&a, &b), 0.0);
    }

    #[test]
    fn score_patterns_basic() {
        let groups = vec![PatternGroup {
            patterns: vec![Regex::new(r"\berror\b").unwrap()],
            weight: 0.5,
        }];
        assert!((score_patterns("got an error", &groups) - 0.5).abs() < f64::EPSILON);
        assert!((score_patterns("all good", &groups)).abs() < f64::EPSILON);
    }

    #[test]
    fn score_patterns_caps_at_one() {
        let groups = vec![
            PatternGroup {
                patterns: vec![Regex::new(r"a").unwrap()],
                weight: 0.7,
            },
            PatternGroup {
                patterns: vec![Regex::new(r"b").unwrap()],
                weight: 0.7,
            },
        ];
        assert!((score_patterns("a b", &groups) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn question_ratio() {
        assert!((compute_question_ratio("What? How?") - 1.0).abs() < f64::EPSILON);
        assert!((compute_question_ratio("Hello world.")).abs() < f64::EPSILON);
    }

    #[test]
    fn detect_strategy_clarify() {
        use crate::types::world::ResponseStrategy;
        let content = "What version are you using? What error do you see?";
        assert_eq!(detect_response_strategy(content), ResponseStrategy::ClarifyFirst);
    }

    #[test]
    fn detect_strategy_step_by_step_capitalized() {
        use crate::types::world::ResponseStrategy;
        // Natural writing with capitalized sentence starts — must be detected.
        // Regression test for case-sensitivity bug (2026-04-14 allostatic_demo).
        let content =
            "1. First, identify the issue.\n\
             2. Then, check the logs.\n\
             3. Next, apply the fix.\n\
             4. Finally, verify it works.";
        assert_eq!(
            detect_response_strategy(content),
            ResponseStrategy::StepByStep,
            "Capitalized sequence markers must be detected (case-insensitive)"
        );
    }

    #[test]
    fn detect_strategy_step_by_step_lowercase() {
        use crate::types::world::ResponseStrategy;
        // Lowercase should still work (no regression).
        let content =
            "1. first identify the issue\n\
             2. then check the logs\n\
             3. next apply the fix";
        assert_eq!(
            detect_response_strategy(content),
            ResponseStrategy::StepByStep,
        );
    }

    // ─── `detect_response_strategy_safe` tests (Option-returning variant) ───

    #[test]
    fn safe_positive_matches_clarify() {
        use crate::types::world::ResponseStrategy;
        let content = "What version? What error?";
        assert_eq!(
            detect_response_strategy_safe(content),
            Some(ResponseStrategy::ClarifyFirst)
        );
    }

    #[test]
    fn safe_positive_matches_step_by_step() {
        use crate::types::world::ResponseStrategy;
        let content =
            "1. First, identify.\n\
             2. Then, check.\n\
             3. Finally, verify.";
        assert_eq!(
            detect_response_strategy_safe(content),
            Some(ResponseStrategy::StepByStep)
        );
    }

    #[test]
    fn safe_positive_matches_short_direct_answer() {
        use crate::types::world::ResponseStrategy;
        // Clear DirectAnswer — short, no questions, no lists.
        let content = "The default port for Postgres is 5432.";
        assert_eq!(
            detect_response_strategy_safe(content),
            Some(ResponseStrategy::DirectAnswer)
        );
    }

    #[test]
    fn safe_ambiguous_single_line_numbered_returns_none() {
        // Regression case that motivated this function: single-line
        // numbered list ("1. First. 2. Then.") — sequence markers present
        // but step_count=1 (only line-start 1.). Existing code silently
        // returned DirectAnswer. Safe version should return None to avoid
        // poisoning reward learning.
        let content = "1. First. 2. Then. 3. Finally.";
        assert_eq!(
            detect_response_strategy_safe(content),
            None,
            "Single-line numbered list is ambiguous — must return None"
        );
    }

    #[test]
    fn safe_ambiguous_single_question_clarifying_returns_none() {
        // Regression case: short response with exactly 1 question mark.
        // Old detector classified this as DirectAnswer (wrong — it's
        // clearly a clarifying question). Safe version must return None.
        let content = "What do you mean exactly?";
        assert_eq!(
            detect_response_strategy_safe(content),
            None,
            "Single-question clarifying response is ambiguous — must return None"
        );
    }

    #[test]
    fn safe_ambiguous_medium_prose_returns_none() {
        // Medium-length narrative prose with no clear structural markers.
        // Could be any of several strategies — the detector shouldn't
        // guess and poison learning.
        let content = "The system architecture evolved over several iterations. \
                       Initial prototypes focused on correctness rather than performance, \
                       and subsequent versions refined the data pipeline while preserving \
                       the original semantic guarantees. Feedback from early adopters \
                       informed the subsequent redesign.";
        assert_eq!(
            detect_response_strategy_safe(content),
            None,
            "Medium narrative prose must not be misclassified as DirectAnswer"
        );
    }

    #[test]
    fn safe_empty_returns_none() {
        // Empty content: no positive detection possible.
        assert_eq!(detect_response_strategy_safe(""), None);
    }

    #[test]
    fn backcompat_wrapper_defaults_to_direct_answer() {
        use crate::types::world::ResponseStrategy;
        // Ambiguous input: safe returns None, back-compat wrapper returns DirectAnswer.
        let content = "1. First. 2. Then. 3. Finally.";
        assert_eq!(detect_response_strategy_safe(content), None);
        assert_eq!(
            detect_response_strategy(content),
            ResponseStrategy::DirectAnswer,
            "Back-compat wrapper defaults to DirectAnswer (unchanged behavior for callers)"
        );
    }
}
