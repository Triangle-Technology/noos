//! Arousal Heuristic — regex-based scalar arousal proxy for downstream modulation.
//!
//! ## Honest naming (P1 — mechanism vs metaphor)
//!
//! This module is a **heuristic**, not a claim to implement amygdala function.
//! Real amygdala tags specific stimuli with salience in ~12ms via Pavlovian
//! associative learning (LeDoux 1996) — it does not produce a scalar arousal
//! level. This module returns a scalar because downstream compensatory modules
//! (adaptive_thresholds, delta_modulation, locus_coeruleus) consume a scalar.
//!
//! The long-term replacement is an SSM-readout layer that derives arousal
//! from the model's own hidden state (see `docs/ssm-cognitive-state.md` —
//! hs differentiates content 45x better than the residual stream).
//! Until then, this regex heuristic produces a "good enough" signal to feed
//! compensatory downstream, which is where the empirical value is (proven by
//! the -1.86% perplexity improvement on emotional text).
//!
//! ## Why keep it as interim
//!
//! - Downstream modules need SOME arousal input to drive compensatory behavior
//! - Model-derived arousal (SSM readout) is the correct endpoint but not yet built
//! - Regex patterns are coarse but have measurable signal on stress-laden text
//!
//! ## What this module does NOT claim
//!
//! - It does NOT implement amygdala low-road processing
//! - It does NOT produce a biologically faithful arousal signal
//! - It does NOT duplicate model's native emotion detection (which is better)
//!
//! ## What this module DOES provide
//!
//! - A scalar input for compensatory downstream modulation
//!
//! ## P9b status (2026-04-14 phase 13 continuation — scoped exception)
//!
//! The English-language lexicon regex here is a P9b violation
//! (duplicates LLM native sentiment classification), but it is a
//! **scoped, documented exception rather than a cleanup debt**.
//!
//! Why the exception persists: the perplexity-validated **-1.86% Emotional**
//! mechanism depends on this arousal path. Decomposition (phase 13 data):
//!
//! - Exclamations, CAPS, question marks, `...`/`!!!`: 0 contribution on the
//!   emotional eval text (prose without emphatic punctuation)
//! - NEGATIVE_HIGH lexicon match ("terrible"): +0.30
//! - NEGATIVE_MOD lexicon match ("stressed"/"anxious"/"deadline"/...): +0.15
//! - Total: 0.45, which crosses `threshold_delta_arousal_emergency` (0.55
//!   with volatility-lowering) and triggers compensatory `gain_factor < 1.0`.
//!
//! Remove the lexicon without a calibrated replacement and arousal drops to
//! ~0 on emotional prose. The compensatory modulation never fires.
//! Emotional % goes from -1.86% to +0.00%. The mechanism dies.
//!
//! The Vietnamese lexicon regex was removed 2026-04-14 phase 12 because the
//! perplexity eval corpus is English (VN removal didn't affect measurement)
//! and the VN patterns had no active eval support. English regex has
//! active eval support — the -1.86% finding.
//!
//! **Proper resolution paths**:
//!
//! - **Path A (medium-term target)**: wire `hs_arousal` into
//!   `CognitiveSession::process_message`. Arousal from SSM hidden-state
//!   churn is language-neutral by construction. Blocker: current HS-derived
//!   arousal (0.162 on Emotional text) is ~3× weaker than regex (0.450);
//!   needs calibration + architectural change to give session access to
//!   model hidden states.
//! - **Path B (compromise)**: engineer a structural signal that reaches
//!   ~0.45 on formal emotional prose. Uncertain — emphatic punctuation
//!   covers only a subset of emotional registers.
//! - **Path C (current state)**: accept the lexicon as a scoped English-
//!   specific interim. Non-English inputs produce arousal ~0 on prose
//!   (the -1.86% finding is English-specific, not universal).
//!
//! All functions are pure (<1ms, $0 LLM cost).
//!
//! ## Gating (P10)
//!
//! Scalar producer (`compute_arousal`) consumed by every downstream
//! signal. Not a competing decision itself — priority competition
//! happens in `adaptive_thresholds` where arousal narrows/expands
//! thresholds.
//!
//! - **Fires when**: called from [`belief_state::update_affect`](super::belief_state::update_affect)
//!   on every `perceive()` call.
//! - **Inactive when**: never — empty or non-English prose returns a
//!   zero/near-zero arousal, but the function still runs.
//! - **Suppresses**: nothing directly. High arousal flowing into
//!   [`adaptive_thresholds`](super::adaptive_thresholds) is what
//!   narrows breadth-oriented thresholds (LeDoux 1996 amygdala low-road
//!   dominance — see §P10 SIGNALS v2 example).
//! - **Suppressed by**: nothing upstream. No gate skips arousal
//!   computation for a given message.

use regex::Regex;
use std::sync::LazyLock;

use crate::math::clamp;
use crate::types::belief::AffectValence;

// ── Result Types ───────────────────────────────────────────────────────

/// Result of arousal computation.
#[derive(Debug, Clone)]
pub struct ArousalResult {
    /// 0.0-1.0, amygdala activation level.
    pub arousal: f64,
    /// Emotional polarity.
    pub valence: AffectValence,
}

// ── Pattern Definitions ────────────────────────────────────────────────

// P9b note: English lexicon regex is a SCOPED EXCEPTION, not pending cleanup.
// See module-level doc §"P9b status" for decomposition data showing the -1.86%
// Emotional mechanism depends on these patterns + why removal without a
// calibrated replacement destroys the measurement.

static NEGATIVE_HIGH: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"(?i)\b(frustrat|angry|furious|hate|terrible|awful|worst|disaster|catastroph|panic|terrif)\w*\b").expect("valid regex"),
    ]
});

static NEGATIVE_MOD: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"(?i)\b(confus|stuck|struggl|annoying|difficult|weird|broken|fail|stress|anxious|worried|concern|nervous|overwhelm|upset|lost|wrong)\w*\b").expect("valid regex"),
        Regex::new(r"(?i)\b(hours?|days?|forever|all day|all night|so long|too long|ages)\b.*\b(nothing|no luck|can'?t|won'?t|doesn'?t|don'?t|not work)\w*\b").expect("valid regex"),
        Regex::new(r"(?i)\b(nothing|no luck|can'?t|won'?t|doesn'?t|don'?t|not)\b.*\b(work|help|fix|solve|progress)\w*\b").expect("valid regex"),
        Regex::new(r"(?i)\b(deadline|pressure|rush|hurry|urgent|emergency|crisis)\w*\b").expect("valid regex"),
    ]
});

static POSITIVE_HIGH: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"(?i)\b(amazing|awesome|perfect|excellent|brilliant|fantastic|incredible)\b").expect("valid regex"),
        Regex::new(r"(?i)\b(finally|eureka|figured it out|nailed it|breakthrough)\b").expect("valid regex"),
    ]
});

static POSITIVE_MOD: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"(?i)\b(good|great|nice|helpful|works|working|better|improved|solved)\b").expect("valid regex"),
        Regex::new(r"(?i)\b(thanks|thank you|appreciate)\b").expect("valid regex"),
    ]
});

static NEGATION_NEAR_NEGATIVE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(not|isn'?t|wasn'?t|aren'?t|weren'?t|haven'?t|hasn'?t|couldn'?t|shouldn'?t|wouldn'?t|never)\s+(?:\w+\s+){0,2}(frustrat|angry|furious|hate|terrible|awful|worst|disaster|catastroph|confus|stuck|struggl|annoying|difficult|weird|broken|fail)\w*").expect("valid regex")
});

// ── Core Function ──────────────────────────────────────────────────────

/// Compute emotional arousal from message content.
///
/// Pure function, <1ms, $0 LLM cost. Returns arousal level and valence.
pub fn compute_arousal(message: &str) -> ArousalResult {
    let mut arousal = 0.0_f64;
    let mut positive_score = 0.0_f64;
    let mut negative_score = 0.0_f64;

    // Check for proximal negation (inverts valence)
    let has_negation = NEGATION_NEAR_NEGATIVE.is_match(message);

    // 1. Exclamation marks
    let excl_count = message.matches('!').count();
    if excl_count >= 3 {
        arousal += 0.2;
    } else {
        arousal += excl_count as f64 * 0.05;
    }

    // 2. CAPS ratio
    let alpha_chars: Vec<char> = message.chars().filter(|c| c.is_alphabetic()).collect();
    let upper_count = alpha_chars.iter().filter(|c| c.is_uppercase()).count();
    if alpha_chars.len() >= 8 {
        let caps_ratio = upper_count as f64 / alpha_chars.len() as f64;
        if caps_ratio >= 0.5 {
            arousal += 0.2;
        } else if caps_ratio >= 0.3 {
            arousal += 0.1;
        }
    }

    // 3. Negative patterns (high intensity)
    let neg_high = NEGATIVE_HIGH.iter().any(|p| p.is_match(message));
    if neg_high && !has_negation {
        negative_score += 0.4;
        arousal += 0.3;
    }

    // 4. Negative patterns (moderate intensity)
    let neg_mod = NEGATIVE_MOD.iter().any(|p| p.is_match(message));
    if neg_mod && !has_negation {
        negative_score += 0.2;
        arousal += 0.15;
    }

    // 5. Positive patterns (high intensity)
    let pos_high = POSITIVE_HIGH.iter().any(|p| p.is_match(message));
    if pos_high {
        positive_score += 0.4;
        arousal += 0.3;
    }

    // 6. Positive patterns (moderate intensity)
    let pos_mod = POSITIVE_MOD.iter().any(|p| p.is_match(message));
    if pos_mod {
        positive_score += 0.2;
        arousal += 0.1;
    }

    // 7. Multiple question marks (curiosity/frustration signal)
    let question_count = message.matches('?').count();
    if question_count >= 3 {
        arousal += 0.1;
    }

    // 8. Repeated punctuation (... or !!!)
    if message.contains("...") || message.contains("!!!") {
        arousal += 0.05;
    }

    // Determine valence
    let valence = if negative_score > positive_score {
        AffectValence::Negative
    } else if positive_score > negative_score {
        AffectValence::Positive
    } else {
        AffectValence::Neutral
    };

    ArousalResult {
        arousal: clamp(arousal, 0.0, 1.0),
        valence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_message_low_arousal() {
        let r = compute_arousal("Can you help me with this?");
        assert!(r.arousal < 0.3);
        assert_eq!(r.valence, AffectValence::Neutral);
    }

    #[test]
    fn frustrated_message_high_arousal() {
        let r = compute_arousal("I'm so frustrated, nothing works!!!");
        assert!(r.arousal >= 0.4);
        assert_eq!(r.valence, AffectValence::Negative);
    }

    #[test]
    fn excited_message_positive() {
        let r = compute_arousal("This is amazing! Finally figured it out!");
        assert!(r.arousal >= 0.3);
        assert_eq!(r.valence, AffectValence::Positive);
    }

    #[test]
    fn caps_boost_arousal() {
        let r = compute_arousal("WHY IS THIS NOT WORKING");
        assert!(r.arousal >= 0.2);
    }

    #[test]
    fn negation_inverts_negative() {
        let r = compute_arousal("I'm not frustrated at all");
        // Negation should prevent negative scoring
        assert_eq!(r.valence, AffectValence::Neutral);
    }

    #[test]
    fn arousal_clamped() {
        // Even with all signals firing, should stay <= 1.0
        let r = compute_arousal("FRUSTRATED!!! TERRIBLE!!! EVERYTHING IS BROKEN!!! HATE THIS!!! WHY???");
        assert!(r.arousal <= 1.0);
    }
}
