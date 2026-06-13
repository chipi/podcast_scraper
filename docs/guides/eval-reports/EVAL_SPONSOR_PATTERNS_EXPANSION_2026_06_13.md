# Eval: SPONSOR_PATTERNS expansion for host-read native ads (#986)

**Date:** 2026-06-13
**Ticket:** [#986](https://github.com/chipi/podcast_scraper/issues/986)
**Companion:** [EVAL_FIXTURES_V2_TIER1_TUNING_2026_06_08.md](EVAL_FIXTURES_V2_TIER1_TUNING_2026_06_08.md)
(Sub-task B documented the coverage gap; threshold sweep on real-prod
returned 2-6% content recall regardless of tuning)

## TL;DR

`SPONSOR_PATTERNS` extended from 13 entries to 19 (six new). Measured against
the 54-episode `my-manual-run-10` real-prod sample:

- **Original 13 patterns** → 92 total hits across 54 episodes
- **New 6 patterns** → 290 additional hits
- **+315% additional coverage** on the same corpus, no test regressions

The new patterns target the host-read native-ad shape that #904 identified
as the structural gap — production-credit outros, subscription pitches,
spoken URLs — not template phrases like "brought to you by".

## Method

1. Sampled all 54 transcripts under
   `.test_outputs/manual/my-manual-run-10/run_20260421-190016_2606de6d/transcripts/`.
   Corpus is heavily FT-Unhedged-dominant (most episodes share a similar
   outro/production-credit template).
2. Scanned for ad-shaped phrases not currently caught by the existing
   `SPONSOR_PATTERNS`. Manually inspected matches to filter out false
   positives (`deal` for M&A, `trial` for court cases, `code` for
   programming, etc.).
3. Identified six distinct native-ad patterns appearing across the
   corpus with high enough specificity to encode safely.
4. Added them to `src/podcast_scraper/cleaning/commercial/patterns.py`
   with appropriate `confidence` and `boundary_hint`.
5. Re-ran per-pattern hit count to measure coverage delta.

## New patterns (with rationale)

| Pattern (regex shape) | Category | Boundary | Confidence | Hits / 54 eps |
| --- | --- | --- | ---: | ---: |
| `is produced by <Name>` | outro | block_end | 0.80 | 48 |
| `(our )?executive producer is/are` | outro | block_end | 0.85 | 47 |
| `special thanks to` | outro | block_end | 0.70 | 49 |
| `(premium )?subscribers can get/access/...` | cta | inline | 0.80 | 47 |
| `(N[- ]day )?free trial( is available)?` | cta | inline | 0.70 | 49 |
| `<domain>.com slash <name>` (spoken URL) | cta | inline | 0.70 | 50 |

The last one — "X.com slash Y" — is the spoken form of a URL with a path.
The existing `(?:visit\|go to\|head to\|check out) \S+\.com(?:/\S+)?`
pattern only catches "go to ft.com" but ASR (Whisper, Deepgram) transcribes
the slash as the word "slash". Real podcasts spell URLs out verbally; the
new pattern picks up that form.

## Why I stopped at 6

Two reasons:

1. **Corpus bias.** The 54-episode sample is FT-Unhedged-dominant. Adding
   patterns derived from this corpus that don't appear in other shows
   would silently over-fit. The six patterns above each correspond to a
   widely-used pattern across podcast shows (production credits, exec
   producer line, subscription pitches, free trials) so they should
   generalise. More aggressive patterns ("listeners, we'll be back",
   "speaking of X") are show-specific.
2. **False-positive risk.** Hand-scanned 247 ad-indicator phrases from
   first 40 episodes that the current pattern set missed. Of those, the
   majority were legitimate uses ("deal" = M&A, "trial" = court case,
   "premium" = risk premium, "subscribe" = subscriber metaphor,
   "discount" = financial discount). Tight patterns avoid lopping off
   real content.

## Coverage lift — per pattern per episode

```text
Per-pattern hit count across 54 prod episodes:
hits | conf | category | pattern
   0   0.90  intro     this episode is (?:brought to you|sponsored|powered) by
   0   0.90  intro     today's (?:episode|show|podcast) is (?:brought|sponsored|supported)
   0   0.85  intro     (?:a )?(?:quick )?(?:word|message|shout.?out) from (?:our|today's) sponsor
   2   0.60  intro     (?:let me|I want to) tell you about
   0   0.50  intro     before we (?:continue|get back|go on|dive in)
   0   0.85  intro     our sponsors today are
  49   0.50  cta       (?:visit|go to|head to|check out) \S+\.com(?:/\S+)?
   0   0.70  cta       use (?:code|promo|coupon) [A-Z0-9]+
   1   0.50  cta       (?:free trial|sign up|get started) (?:at|today)
  33   0.70  outro     (?:now )?(?:back to|let's get back to|returning to) (?:the|our)
   7   0.50  outro     (?:welcome back to (?:the|our) (?:show|episode|podcast)|we're back...)
   0   0.85  outro     (?:thanks(?: again)?|thank you) to (?:our )?(?:friends|partners|sponsor)
  48   0.80  outro    [NEW] is produced by <Name>
  47   0.85  outro    [NEW] (our )?executive producer is/are
  49   0.70  outro    [NEW] special thanks to
  47   0.80  cta      [NEW] (premium )?subscribers can get/access/read
  49   0.70  cta      [NEW] (N-day )?free trial( is available)?
  50   0.70  cta      [NEW] <domain>.com slash <name>  (spoken URL)
```

The original "template" patterns (top 6) all return **0 hits on real
prod** — they're tuned to synthetic v2 fixtures that use the template
shape, not real podcasts. Six of the original 13 are now demonstrably
v2-only patterns. They stay because (a) v2 fixtures still need them and
(b) future template-style shows may emerge.

## Acceptance per #986

- [x] Real-prod sample catalogued (54 episodes)
- [x] New patterns derived from sample with provisional confidence scores
- [x] Each new pattern justified by hit count + rationale (above)
- [x] No regressions in 74 cleaning + commercial unit tests
- [ ] v3 fixtures contribution — deferred to the next fixture batch
      (#921); the patterns above are now documented for any future
      v3-fixture sponsor-block generation

## What's NOT in this expansion

- **Per-show ad signature**: shows like NPR, Pivot, Marketplace have
  distinct sponsor copy. Catching those would need either per-show
  curation or LLM-based detection (the `cleaning_v3`/`v4` LLM passes
  effectively do this already for production episodes — pattern matching
  is the first-line filter, not the only one).
- **Mid-roll explicit "stay tuned"** — captured indirectly by the existing
  `(?:back to|returning to)` pattern but could be tightened.
- **Threshold re-sweep on real prod**: the prior
  `_HIGH_CONFIDENCE_FOR_LARGE_BLOCK` and `_INLINE_STANDALONE_CONFIDENCE`
  defaults (0.85 / 0.70) were verified Pareto-optimal at v2 fixture scale
  and held against the 2-6% coverage band. With the new patterns boosting
  hit count by 315%, a tighter threshold sweep on real prod could squeeze
  more out — filing as a follow-up if cleaning quality degrades observably.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python <<'PY'
import sys
sys.path.insert(0, 'src')
from podcast_scraper.cleaning.commercial.patterns import SPONSOR_PATTERNS
from pathlib import Path
import collections

root = Path('.test_outputs/manual/my-manual-run-10/run_20260421-190016_2606de6d/transcripts')
hits = collections.Counter()
for f in sorted(root.glob('*.txt')):
    text = f.read_text(encoding='utf-8', errors='ignore')
    for pat in SPONSOR_PATTERNS:
        n = sum(1 for _ in pat.pattern.finditer(text))
        if n:
            hits[pat.pattern.pattern] += n
for pat in SPONSOR_PATTERNS:
    print(f'{hits.get(pat.pattern.pattern, 0):>5}  {pat.pattern.pattern[:80]}')
PY
```
