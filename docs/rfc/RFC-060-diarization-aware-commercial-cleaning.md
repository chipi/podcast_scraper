# RFC-060: Diarization-Aware Commercial Detection & Cleaning

- **Status**: Draft
- **Authors**: Architecture Review
- **Stakeholders**: Core Pipeline, Summarization, GIL/KG Consumers
- **Related PRDs**:
  - `docs/prd/PRD-020-audio-speaker-diarization.md`
  - `docs/prd/PRD-005-episode-summarization.md`
- **Related RFCs**:
  - `docs/rfc/RFC-058-audio-speaker-diarization.md` (provides diarization signals)
  - `docs/rfc/RFC-059-speaker-detection-refactor-test-audio.md` (§3: commercial fixtures)
  - `docs/rfc/RFC-042-hybrid-summarization-pipeline.md` (hybrid cleaning triggers)
- **Related Issues**:
  - Issue #109: Commercial segments in test fixtures
  - Issue #477: LLM call consolidation / cleaning cost investigation
  - Issue #482: Audio-based speaker diarization
  - Issue #483: Speaker detection refactor & test audio improvements
  - Issue #486: Phase 1 — expanded patterns + positional heuristics (unblocked)
  - Issue #488: Phase 2 — diarization-enhanced signals (blocked by #482, #486)
- **Related Documents**:
  - `src/podcast_scraper/preprocessing/core.py` (current cleaning pipeline)
  - `src/podcast_scraper/cleaning/hybrid.py` (hybrid LLM trigger logic)
  - `docs/wip/transcript-cleaning-cost-quality-eval-plan.md` (deleted — content absorbed into this RFC)

## Abstract

This RFC proposes a multi-signal commercial detection system that combines text pattern matching, positional heuristics, and optional LLM verification to identify and remove host-read sponsor segments from podcast transcripts before summarization. Current detection relies on four hardcoded phrases and fails when hosts paraphrase ads or weave sponsor messages naturally into conversation.

**Phase 1 (expanded patterns + positional heuristics) benefits all transcription providers** — local Whisper, OpenAI API, Gemini, Mistral — because it operates on transcript text, not audio. Diarization signals (RFC-058) are a future enhancement that can boost accuracy for locally-transcribed episodes.

**Architecture Alignment:** Builds on top of the existing cleaning pipeline (`clean_for_summarization()`) and hybrid cleaning strategy (`pattern` → `hybrid` → `llm`). Diarization signals are a future input to the pattern-based stage, not a replacement for the existing architecture.

**Evaluation Alignment:** The cleaning cost/quality tradeoffs explored in Issue #477 (previously tracked in `docs/wip/transcript-cleaning-cost-quality-eval-plan.md`) are directly relevant. This RFC's Phase 1 improvements should be validated using the evolving eval tooling to measure cost, quality, and stability before/after — the eval infrastructure is being developed in parallel and the specific tooling will be determined as it matures.

## Problem Statement

### The Core Quality Problem

Commercial content leaking into summaries and downstream features (GIL quotes, KG topics) fundamentally breaks the pipeline's core value proposition. Users want the *substance* of a podcast conversation, not ads.

### Why Current Detection Fails

The current `remove_sponsor_blocks()` in `preprocessing/core.py` uses **four hardcoded English phrases**:

```python
for phrase in [
    "this episode is brought to you by",
    "today's episode is sponsored by",
    "today's episode is sponsored by",  # curly apostrophe duplicate
    "our sponsors today are",
]:
```

For each match, it deletes until the next `\n\n` or 2000 characters. This approach has critical gaps:

1. **Host-read ads are invisible**: In most podcasts, the host reads ads in their own voice with no speaker change, no jingle, no "Ad:" label. The text flows naturally from content → ad → content. If the host doesn't use one of the four exact phrases, the ad passes through undetected.

2. **Paraphrased sponsor messages**: Hosts commonly say things like "I want to tell you about...", "Quick shout-out to our friends at...", "Before we continue, let me mention..." — none of which match the four phrases.

3. **Woven-in ads**: Some hosts integrate sponsors into conversation: "...speaking of building great products, that reminds me of what Stripe does..." — indistinguishable from content by pattern matching alone.

4. **Boundary detection is crude**: The "delete until next blank line or 2000 chars" heuristic either under-removes (stops too early at a paragraph break mid-ad) or over-removes (nukes 2000 chars of actual content when no blank line follows).

5. **No positional awareness**: Ads cluster at predictable positions (post-intro, mid-episode, pre-outro). The current code doesn't use position to weight detection confidence.

6. **Duplicate code**: `remove_sponsor_blocks` exists in both `preprocessing/core.py` and `providers/ml/summarizer.py` with slightly different implementations.

### Impact

- **Summaries** contain sponsor plugs and calls-to-action mixed with genuine insights
- **GIL quotes** may attribute commercial language to speakers as if it were their analytical insight
- **KG topics** may include brand names and commercial products as podcast themes
- **Eval benchmarks** are artificially clean because test fixtures contain no commercials (#109)

**Use Cases:**

1. **Host-read mid-roll**: Lenny Rachitsky seamlessly transitions from discussing product strategy to reading a Stripe ad, then back — no pause, no jingle, same speaker
2. **Conversational sponsor integration**: Host weaves "speaking of X, our sponsor Y does exactly that..." into discussion flow
3. **Multi-sponsor episodes**: Three different sponsor reads at different positions, each with unique phrasing

## Goals

### Phase 1 Goals (unblocked — core value)

1. **Detect host-read sponsor segments** with measurably better recall than the current four-phrase approach, maintaining >= 90% precision
2. **Combine text patterns + positional heuristics**: These two signals work for all providers (local and cloud) without any audio dependency
3. **Precise boundary detection**: Identify ad start and end points accurately, not just "delete 2000 chars"
4. **Consolidate duplicate code**: Single `CommercialDetector` implementation replaces duplicated `remove_sponsor_blocks`
5. **Measurable improvement**: Before/after comparison validated via the evolving eval tooling

### Future Goals (nice-to-have, after diarization)

6. **Diarization-enhanced detection**: When diarization data is available (RFC-058), use "who + when" signals to boost confidence for host-read ads in locally-transcribed episodes
7. **Enhanced LLM verification**: Leverage speaker context in hybrid cleaning prompts

## Constraints & Assumptions

**Constraints:**

- Must not remove genuine content (precision >= 90% — false positives are worse than false negatives)
- Must work without diarization (pattern-only fallback for `diarize=false`)
- Must not add significant latency to the cleaning pipeline (< 100ms for pattern+position; LLM pass is already budgeted in hybrid strategy)
- Must be backward compatible — `clean_for_summarization()` API unchanged

**Assumptions:**

- Host-read ads are the dominant ad format in interview/conversation podcasts
- Ads cluster at predictable temporal positions (0-15% intro, 40-60% mid-roll, 85-100% outro)
- Brand names and promotional URLs are strong signals when combined with positional context
- Diarization, when available, reliably identifies who is speaking (< 15% DER per RFC-058)

## Design & Implementation

### 1. Multi-Signal Detection Architecture

The enhanced commercial detector combines four signal layers, each independent but increasingly powerful when combined:

```text
Signal Layer 1: Text Patterns (current, expanded)
    ↓ candidates
Signal Layer 2: Positional Heuristics (new)
    ↓ weighted candidates
Signal Layer 3: Diarization Context (new, optional)
    ↓ high-confidence candidates
Signal Layer 4: LLM Verification (existing hybrid path)
    ↓ verified removals
```

Each layer produces a **confidence score** per candidate segment. Final removal decision uses a configurable threshold.

### 2. Signal Layer 1: Expanded Text Patterns

Replace the four hardcoded phrases with a comprehensive, categorized pattern system:

```python
@dataclass
class SponsorPattern:
    pattern: re.Pattern
    category: str          # "intro", "body", "cta", "outro"
    confidence: float      # base confidence for this pattern
    boundary_hint: str     # "block_start", "block_end", "inline"

SPONSOR_PATTERNS: List[SponsorPattern] = [
    # Block starters (high confidence — these almost always begin an ad)
    SponsorPattern(
        re.compile(r"this episode is (?:brought to you|sponsored|powered) by", re.I),
        "intro", 0.9, "block_start",
    ),
    SponsorPattern(
        re.compile(r"today'?s (?:episode|show|podcast) is (?:brought|sponsored|supported)", re.I),
        "intro", 0.9, "block_start",
    ),
    SponsorPattern(
        re.compile(r"(?:a )?(?:quick )?(?:word|message|shout.?out) from (?:our|today'?s) sponsor", re.I),
        "intro", 0.85, "block_start",
    ),
    SponsorPattern(
        re.compile(r"(?:let me|I want to) tell you about", re.I),
        "intro", 0.6, "block_start",
    ),
    SponsorPattern(
        re.compile(r"before we (?:continue|get back|go on|dive in)", re.I),
        "intro", 0.5, "block_start",
    ),

    # Body signals (medium confidence — need corroboration)
    SponsorPattern(
        re.compile(r"(?:visit|go to|head to|check out) \S+\.com(?:/\S+)?", re.I),
        "cta", 0.5, "inline",
    ),
    SponsorPattern(
        re.compile(r"use (?:code|promo|coupon) [A-Z0-9]+", re.I),
        "cta", 0.7, "inline",
    ),
    SponsorPattern(
        re.compile(r"(?:free trial|sign up|get started) (?:at|today)", re.I),
        "cta", 0.5, "inline",
    ),

    # Block enders (signal end of ad)
    SponsorPattern(
        re.compile(r"(?:now )?(?:back to|let'?s get back to|returning to) (?:the|our)", re.I),
        "outro", 0.7, "block_end",
    ),
    SponsorPattern(
        re.compile(r"(?:welcome back|we'?re back|okay,? so)", re.I),
        "outro", 0.5, "block_end",
    ),
    SponsorPattern(
        re.compile(r"thanks (?:again )?to (?:our )?(?:friends|partners|sponsor)", re.I),
        "outro", 0.85, "block_end",
    ),
]

BRAND_NAMES: Set[str] = {
    "stripe", "figma", "notion", "linear", "vanta", "miro",
    "zapier", "hubspot", "squarespace", "shopify", "mailchimp",
    "convertkit", "airtable", "wix", "justworks", "lenny",
    # expanded dynamically from episode metadata if available
}
```

**Boundary detection** improves on "delete to next `\n\n`":

```python
def detect_sponsor_boundaries(text: str, match_start: int) -> Tuple[int, int]:
    """Find precise start and end of a sponsor block.

    Searches backward from match for block_start pattern,
    and forward for block_end pattern or topic change.
    Falls back to paragraph boundaries if no patterns found.
    """
```

Strategy:
- **Start**: Scan backward from match for a `block_start` pattern (up to ~500 chars). If found, use that position. Otherwise, use start of current paragraph.
- **End**: Scan forward for a `block_end` pattern (up to ~2000 chars). If found, use that position. Otherwise, use end of current paragraph or next speaker turn that doesn't contain sponsor signals.

### 3. Signal Layer 2: Positional Heuristics

Ads cluster at predictable positions in podcast episodes:

```python
POSITION_WINDOWS = {
    "pre_roll":  (0.00, 0.15),   # first 15% — opening sponsor
    "mid_roll":  (0.35, 0.65),   # middle 30% — mid-roll break
    "post_roll": (0.80, 1.00),   # last 20% — closing sponsor/outro
}

def position_confidence_boost(
    match_position: int,
    total_length: int,
) -> float:
    """Return confidence boost based on position in transcript.

    Matches in known ad-break positions get a boost.
    Matches in conversation-heavy positions (15-35%, 65-80%)
    get a penalty (more likely genuine content).
    """
    relative_pos = match_position / total_length
    for window_name, (start, end) in POSITION_WINDOWS.items():
        if start <= relative_pos <= end:
            return 0.15  # boost
    return -0.10  # penalty for non-typical ad position
```

### 4. Signal Layer 3: Diarization Context (Future Enhancement)

> **Note:** This layer is a nice-to-have future enhancement, not a Phase 1 requirement. It becomes available once RFC-058 (diarization) is implemented. The core value of this RFC — expanded patterns + positional heuristics — is delivered without this layer.

When diarization data is available (RFC-058), it could provide additional signals:

```python
def diarization_sponsor_signals(
    candidate_start: int,
    candidate_end: int,
    diarized_segments: List[DiarizedSegment],
    host_speaker_id: str,
) -> DiarizationSignals:
    """Extract diarization-based signals for a sponsor candidate.

    Returns:
        DiarizationSignals with:
        - is_host_monologue: bool — candidate is single-speaker (host),
          no guest speech. Ads are almost always host-only.
        - monologue_duration_s: float — how long the host speaks
          uninterrupted. Long host monologues in mid-episode are
          unusual in interviews → likely ad.
        - topic_discontinuity: bool — different speakers before and
          after the candidate, but same speaker during. Suggests
          insertion of non-conversational content.
        - surrounding_turn_gap_s: float — if the host had a turn
          before the candidate, how much silence preceded the ad?
          Sponsors often follow a micro-pause.
    """
```

**Key diarization signals for sponsor detection:**

| Signal | What it tells us | Confidence impact |
| --- | --- | --- |
| **Host monologue in mid-episode** | Interviews have rapid turn-taking; a 30-90s host monologue in the middle is anomalous → likely ad | +0.20 |
| **Topic discontinuity** | Conversation topic before and after candidate is connected, but candidate is unrelated → insertion | +0.15 |
| **Single speaker during candidate** | Ads are always single-speaker (host reads alone) — if guest speaks during candidate, it's not an ad | -0.30 (disqualify) |
| **Duration matches ad length** | Sponsor reads are typically 30-90 seconds; candidate segment duration in this range → corroborates | +0.10 |

### 4a. Cloud API Transcription Considerations

The expanded pattern + positional detection (Layers 1-2) operates on transcript text and is **provider-agnostic** — it works identically for transcripts from local Whisper, OpenAI API, Gemini, and Mistral. This is a key advantage: Phase 1 improvements immediately benefit all users regardless of transcription backend.

For cloud API providers, some additional strategies are worth exploring:

- **Prompt-based cleaning at summarization time**: Rather than a separate cleaning pass, the summarization prompt can instruct the LLM to "ignore sponsor reads, advertisements, and promotional content." This avoids extra API calls but keeps noisy text in the context window, which can still affect output quality. The tradeoff between prompt-only vs. structural cleaning should be validated empirically (see Issue #477).
- **Cloud provider-specific features**: Some APIs (e.g., Gemini) may offer content categorization or labeling that could flag non-conversational segments. Worth monitoring as these APIs evolve, but not a dependency.
- **Hybrid approach for cloud**: Pattern+position cleaning first (cheap, local), then let the summarization prompt handle any residual commercial content. This gives two layers of defense without extra API calls for cleaning.

The evaluation framework (Issue #477) should compare these approaches on the same dataset to find the right default for each provider tier.

### 5. Signal Layer 4: LLM Verification (Existing Hybrid Path)

The existing hybrid cleaning strategy already calls `provider.clean_transcript()` when pattern-based cleaning is insufficient. This RFC enhances the LLM prompt to leverage diarization context:

```python
SPONSOR_DETECTION_PROMPT = """
Analyze the following transcript segment and determine if it contains
a commercial/sponsor message. The segment is spoken by {speaker_name}
at position {relative_position:.0%} of the episode.

Context before: "{context_before}"
Segment: "{candidate_text}"
Context after: "{context_after}"

Is this a sponsor/commercial message? Reply with:
- YES: if this is clearly a sponsor read or advertisement
- PARTIAL: if this contains some sponsor content mixed with real content
- NO: if this is genuine conversation content

For PARTIAL, indicate where the sponsor content starts and ends.
"""
```

### 6. Consolidated Detection Pipeline

```python
class CommercialDetector:
    """Multi-signal commercial segment detection.

    Combines text patterns, positional heuristics, diarization context,
    and optional LLM verification to identify sponsor segments.
    """

    def __init__(
        self,
        diarization_result: Optional[DiarizationResult] = None,
        host_speaker_id: Optional[str] = None,
        confidence_threshold: float = 0.65,
    ):
        self._diarization = diarization_result
        self._host_speaker_id = host_speaker_id
        self._threshold = confidence_threshold

    def detect(self, text: str) -> List[CommercialCandidate]:
        """Detect commercial segments in transcript text.

        Returns candidates with confidence scores and boundaries.
        Only candidates above threshold are returned.
        """
        # Layer 1: Text pattern scan
        candidates = self._scan_patterns(text)

        # Layer 2: Position-based confidence adjustment
        for c in candidates:
            c.confidence += position_confidence_boost(c.start, len(text))

        # Layer 3: Diarization signals (if available)
        if self._diarization and self._host_speaker_id:
            for c in candidates:
                signals = diarization_sponsor_signals(
                    c.start, c.end,
                    self._diarization.segments,
                    self._host_speaker_id,
                )
                c.confidence += self._apply_diarization_signals(signals)
                # Disqualify if guest speaks during candidate
                if not signals.is_host_monologue:
                    c.confidence = 0.0

        # Filter by threshold
        return [c for c in candidates if c.confidence >= self._threshold]

    def remove(self, text: str) -> str:
        """Detect and remove commercial segments from text."""
        candidates = self.detect(text)
        # Sort by position descending to avoid offset shifts
        for c in sorted(candidates, key=lambda x: x.start, reverse=True):
            text = text[:c.start] + text[c.end:]
        return text
```

### 7. Integration into Cleaning Pipeline

Replace `remove_sponsor_blocks()` in `clean_for_summarization()`:

```python
def clean_for_summarization(
    text: str,
    diarization_result: Optional[DiarizationResult] = None,
    host_speaker_id: Optional[str] = None,
) -> str:
    text = strip_credits(text)
    text = strip_garbage_lines(text)
    text = clean_transcript(text, ...)

    # Enhanced sponsor detection (replaces old remove_sponsor_blocks)
    detector = CommercialDetector(
        diarization_result=diarization_result,
        host_speaker_id=host_speaker_id,
    )
    text = detector.remove(text)

    text = remove_outro_blocks(text)
    text = remove_summarization_artifacts(text)
    return text.strip()
```

**Backward compatibility**: When `diarization_result=None` (default), the detector operates in pattern+position mode only — strictly more capable than the current four-phrase approach, no regression.

### 8. Module Structure

```text
src/podcast_scraper/
└── cleaning/
    ├── base.py                    # existing
    ├── pattern_based.py           # existing — uses CommercialDetector
    ├── hybrid.py                  # existing — enhanced LLM prompt
    ├── llm_based.py               # existing
    └── commercial/                # NEW
        ├── __init__.py
        ├── detector.py            # CommercialDetector class
        ├── patterns.py            # SPONSOR_PATTERNS, BRAND_NAMES
        ├── positions.py           # POSITION_WINDOWS, position scoring
        └── diarization_signals.py # diarization-aware signal extraction
```

### 9. Eliminating Duplicate Code

Current state: `remove_sponsor_blocks` exists in both `preprocessing/core.py` and `providers/ml/summarizer.py`. Also, `SPONSOR_BLOCK_PATTERNS` is defined in `summarizer.py` but unused.

Resolution:
- Delete `remove_sponsor_blocks` from both locations
- Delete unused `SPONSOR_BLOCK_PATTERNS` from `summarizer.py`
- All sponsor detection goes through `CommercialDetector` in `cleaning/commercial/detector.py`
- `clean_for_summarization()` calls the detector
- `summarizer.py` profile `cleaning_v4` calls `clean_for_summarization()` (no separate sponsor removal)

## Key Decisions

1. **Confidence-scored candidates, not binary detection**
   - **Decision**: Each candidate gets a float confidence from combined signals; threshold determines removal
   - **Rationale**: Enables tuning precision vs. recall; makes the system auditable (log scores for debugging); allows future ML scoring model

2. **Diarization enhances but doesn't replace pattern matching**
   - **Decision**: Pattern matching is the primary detection method; diarization provides confidence boosts/penalties
   - **Rationale**: Must work without diarization (fallback); patterns catch ads even without audio; diarization alone can't distinguish ad-reading from a genuine monologue

3. **Precision over recall**
   - **Decision**: Default threshold (0.65) is tuned for >= 90% precision even at cost of some recall
   - **Rationale**: Removing genuine content is worse than leaving an ad in. Summarization prompts provide a second line of defense ("ignore sponsorships")

4. **Brand name list is extensible**
   - **Decision**: Start with a curated set; allow expansion from episode metadata and config
   - **Rationale**: Different podcasts have different sponsors; a fixed list can't cover all cases. Users can add brands via config.

## Alternatives Considered

1. **LLM-only detection (send full transcript to LLM)**
   - **Pros**: Most accurate; understands context perfectly
   - **Cons**: Expensive (full transcript per episode); slow; requires API access
   - **Why Rejected**: Too expensive for default pipeline. Keep as hybrid fallback for uncertain candidates.

2. **Audio-level jingle/music detection**
   - **Pros**: Many ads have intro/outro music or jingles
   - **Cons**: Requires audio analysis ML; many host-read ads have no jingle; adds heavy dependency
   - **Why Rejected**: Doesn't solve the main problem (host-read ads with no audio cue). Could be a future enhancement.

3. **Classifier trained on labeled ad segments**
   - **Pros**: Could learn subtle patterns human rules miss
   - **Cons**: Requires labeled training data; podcast ad styles vary widely; overfitting risk
   - **Why Rejected**: Insufficient labeled data currently. The confidence-scoring approach can evolve into a learned model later if data accumulates.

## Testing Strategy

**Test Coverage:**

- **Unit tests**: Pattern matching, boundary detection, position scoring, confidence calculation
- **Integration tests**: Full `CommercialDetector.remove()` on enhanced fixture transcripts (#109)
- **Regression tests**: Current cleaning behavior preserved when `CommercialDetector` replaces `remove_sponsor_blocks`
- **Future**: Unit tests with diarization mocks (Phase 2, after RFC-058)

**Test Fixtures (depends on #109):**

The enhanced test fixtures (#109) are essential — they provide:
- Host-read sponsor blocks (using host speaker label, no `Ad:` marker)
- Mid-roll "pre-recorded" ads (using `Ad:` label)
- Varied phrasing across episodes (not just the four hardcoded phrases)
- Brand names from `BRAND_NAMES` set

**Evaluation:**

Commercial detection improvements must be measurable. The key metrics are:

- **Sponsor recall**: % of known sponsor segments removed
- **Sponsor precision**: % of removed content that was actually sponsor
- **Content preservation**: % of genuine content retained
- **Brand leakage**: any `BRAND_NAMES` appearing in summaries
- **Cost**: cleaning calls and token usage (relevant for hybrid/LLM cleaning)

The specific eval tooling and scripts are being developed in parallel (Issue #477). This RFC does not prescribe a specific evaluation tool — it defines *what* needs to be measured. The eval infrastructure should support before/after comparison on the same dataset to validate that Phase 1 is strictly better than the current four-phrase approach.

**Test Organization:**

```text
tests/
├── unit/podcast_scraper/
│   └── cleaning/
│       └── commercial/           # NEW
│           ├── test_detector.py
│           ├── test_patterns.py
│           └── test_positions.py
├── integration/
│   └── cleaning/
│       └── test_commercial_cleaning.py  # NEW — full pipeline on fixtures
```

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1** (unblocked — can start now):
  - Expanded pattern library + positional heuristics (no diarization needed)
  - Drop-in replacement for current `remove_sponsor_blocks`
  - Deduplicate sponsor removal code (consolidate `core.py` + `summarizer.py`)
  - Benefits all providers (local Whisper, OpenAI, Gemini, Mistral)
  - Validate improvement using the evolving eval tooling
- **Phase 2** (blocked by RFC-058 diarization):
  - Diarization-aware signals — confidence boost for locally-diarized transcripts
  - Enhanced LLM verification prompt with speaker context
  - Update hybrid cleaning trigger thresholds
- **Phase 3** (ongoing):
  - Benchmark on real podcast episodes using eval tooling
  - Tune confidence thresholds based on empirical results

**Monitoring:**

- Log sponsor detection confidence scores per candidate
- Track removal count and total characters removed per episode
- Brand leakage check: scan summaries for `BRAND_NAMES` matches

**Success Criteria:**

1. Pattern+position mode (Phase 1) is strictly better than current four-phrase approach, validated via eval tooling
2. >= 90% precision (no genuine content removed) on benchmark set
3. Zero brand name leakage in summaries on benchmark set
4. All existing cleaning tests pass
5. Code deduplication complete (single `CommercialDetector`, no duplicate `remove_sponsor_blocks`)

## Relationship to Other RFCs

This RFC (RFC-060) completes the **connected value chain** from audio processing to clean output:

```text
RFC-059 §2-3 (#109, #111)   Realistic test fixtures with commercials + unique voices
        ↓
RFC-060 Phase 1 (this)       Expanded patterns + positional detection (all providers)
        ↓
Clean summaries              Core pipeline output quality (immediate win)

RFC-058 (#482)               Diarization: who is speaking when (future)
        ↓
RFC-060 Phase 2 (this)       Diarization-enhanced detection (local Whisper only)
        ↓
Even cleaner summaries       Catches host-read ads that patterns miss
```

**Phase 1 is independent of diarization** — it delivers value to all users immediately.

1. **RFC-059** (Test Audio): Provides the test fixtures that validate detection works
2. **RFC-058** (Diarization): Future enhancement — provides "who + when" signals for Phase 2
3. **RFC-042** (Hybrid Summarization): Existing hybrid cleaning strategy that this RFC enhances
4. **Issue #477**: Cleaning cost/quality evaluation that validates the improvement

## Benefits

1. **Core output quality**: Summaries free of commercial content
2. **Downstream accuracy**: GIL quotes and KG topics not contaminated by ad copy
3. **Graceful degradation**: Works without diarization (pattern+position), better with it
4. **Measurable**: Before/after metrics via `eval_cleaning.py`
5. **Consolidation**: Single detection system replaces duplicate code

## Migration Path

1. **Phase 1** (unblocked): Replace `remove_sponsor_blocks` in `core.py` with `CommercialDetector` (pattern+position mode). Delete duplicate from `summarizer.py` and unused `SPONSOR_BLOCK_PATTERNS`. Zero API change to `clean_for_summarization()`. Strictly better detection for all providers.
2. **Phase 2** (after RFC-058): Add optional `diarization_result` parameter to `clean_for_summarization()`. Callers that have diarization data pass it in; others get pattern-only (Phase 1 behavior).

## Open Questions

1. Should `BRAND_NAMES` be configurable per-feed (some podcasts have recurring sponsors)?
2. What confidence threshold balances precision and recall best? Start at 0.65, tune with `eval_cleaning.py`.
3. Should the LLM verification step run for all uncertain candidates or only when hybrid mode is active?
4. Should removed sponsor segments be preserved in a separate `.sponsors.txt` file for auditing?
5. How to handle non-English sponsor segments? Current patterns are English-only.

## References

- **Source Code**: `src/podcast_scraper/preprocessing/core.py` (current `remove_sponsor_blocks`)
- **Source Code**: `src/podcast_scraper/cleaning/hybrid.py` (hybrid trigger logic)
- **Source Code**: `src/podcast_scraper/providers/ml/summarizer.py` (duplicate sponsor removal + unused patterns)
- **Related RFC**: `docs/rfc/RFC-058-audio-speaker-diarization.md`
- **Related RFC**: `docs/rfc/RFC-059-speaker-detection-refactor-test-audio.md`
- **Related Issue**: #109 (commercial segments in fixtures)
- **Related Issue**: #477 (cleaning cost/quality evaluation)
- **Related Issue**: #482 (diarization)
