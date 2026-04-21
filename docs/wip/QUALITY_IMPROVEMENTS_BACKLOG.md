# Quality Improvements Backlog

Running list of quality issues observed in production runs that we defer
fixing until after the next larger run (100+ episodes) confirms the
signal. Each entry: problem → fix → priority / effort.

Source runs:

- `2026-04-21 10-feed cloud_balanced run` (`my-manual-run-10` in FUTURE).
  10 episodes, post-#646, post-CLI-profile-routing-fix.
  Results: 118 insights / 100 topics / 142 entities across 10 feeds,
  median 12 ins · 10 topics · ~14 entities per episode.

## Findings from 10-feed run

### 1. Quotes per insight too low (1.4 vs 3–5 target)

**Observed.** Aggregate 167 quotes / 118 insights = **1.42 per insight**;
range 1.08–2.70 across feeds. The outlier (2.70) was the fallback-to-staged
case; mega-bundled + prefilled path consistently sits at 1.08–1.50.

**Root cause (likely).** Gemini's `extract_quotes` prompt asks for "all
short verbatim quotes" with 3 example slots, but models return 1–2 by
default. `max_output_tokens=512` is tight but probably not binding.

**Fix.** Strengthen prompt to **explicitly require 3–5 distinct quotes
from different parts of the transcript**; raise `max_output_tokens` to
1024 as insurance; relax QA (0.3) / NLI (0.5) floors slightly if still
short. Code: `src/podcast_scraper/providers/gemini/gemini_provider.py:1749`
(and the equivalent in each cloud provider — audit all 6).

**Priority / effort.** Medium priority, low effort (~15 LOC + re-run).
Wait for next production run to confirm the 1.4 baseline is stable.

### 2. Topic lengths trend long; cross-episode cluster potential is ~zero

**Observed.** Median topic is 3 words / 22 chars. 21% are 4–6 words.
Worst offenders read like sentence fragments:

- "leadership in the age of AI"
- "language models vs enterprise platforms"
- "Impact of war on civilians"
- "Hope and resilience in Iran"
- "Mushtaba Khamenei succession"

Cross-episode duplicates: **1 of 99** unique topic strings shared across
episodes (`us-iran conflict` appears 2×). Exact-match clustering would
produce ~99 singleton clusters.

**Root cause.** KG v3 prompt produces noun phrases but doesn't cap length
or disallow preposition-heavy / comparison / event-framing phrasings.
Proper-noun compounds like "Mushtaba Khamenei succession" can never
cluster (entity + concept mixed).

**Fix.** Tighten the KG v3 prompt:

- Hard cap at 2–3 words.
- Disallow prepositions ("in", "of", "for", "vs") at start / middle.
- Prefer canonical concept names over event framings.
- Require the topic be reusable across episodes (generic concepts, not
  episode-specific events).

Pair with semantic clustering (check `topic_clusters.json` is running on
this corpus before the prompt change, or after).

**Priority / effort.** High priority (blocks clustering quality),
low–medium effort (~30 LOC in `kg/prompts` + re-run). After next run.

### 3. Grounding < 100 % on some feeds

**Observed.** WSJ 12/12, NPR 12/12, most feeds 11–12/12 grounded.
**`omnycontent.com_c9fdce2d` 8/12 grounded** — worst. `acast` and
`flightcast` at 11/12.

**Root cause (hypothesis).** For some insights the LLM-generated text
doesn't find a supporting verbatim span in the transcript even with
retry. Could be paraphrase-heavy insights, or the transcript is missing
the supporting passage (transcription error).

**Fix.** Not yet clear. Investigate the 4 ungrounded insights in the
omnycontent episode specifically — are they "inference" insights without
direct evidence, or is the QA model missing a real match?

**Priority / effort.** Low (95 % grounding is fine at scale). Defer
until next run; look for systematic pattern across 100+ episodes.

### 4. `simplecast.com_2e104dc7 / run_20260421-172501` interrupted

**Observed.** Transcript + segments present, no metadata/gi/kg/bridge.

**Root cause.** Interrupted mid-pipeline (not a code bug).

**Fix.** Not a code issue. If it recurs on clean runs, investigate.

**Priority / effort.** Noise. Skip.

## Findings from NEXT 100+-episode run — APPEND HERE

*(Add new sections as we learn more.)*

## Ideas & solution paths (not yet scoped)

Larger architectural directions we've raised but not yet turned into
concrete work. Promote into the Findings section once we have data /
decide to act.

- **Per-stage provider routing.** Today one provider runs summary +
  GIL + KG. Research (#632, cross-dataset baseline) showed different
  providers win different stages (Gemini tops GI insight coverage,
  qwen3.5 tops KG locally). A routing layer would let `cloud_quality`
  pick the best per stage. Currently deferred as "future work" in
  `cloud_quality.yaml` comments.

- **Quote deduplication across clusters.** Multi-quote extraction per
  insight is per-insight-local. A corpus-level dedupe pass could
  surface the same quote supporting multiple related insights (better
  evidence linking, smaller total quote count). Pairs with insight
  clustering (#599) and topic alignment.

- **Topic-label canonicalisation.** Given the ~99 / 100 unique topic
  strings in the 10-feed run, a canonicalisation step (either prompt-
  time or post-KG) that maps paraphrases to a canonical label could
  turn useless singleton clusters into useful cross-episode ones. Pairs
  with KG v3 prompt tightening (Finding 2).

- **Ground-truth probes.** Right now the only quality signal is
  coverage % against silver and model-vs-silver ROUGE. A small set of
  labelled "do these insights / topics / entities appear in the
  transcript" probes, run on each re-ingestion, would give a cheap
  direct accuracy check.

- **Voxtral for transcription.** Closed #641 as "not a win" on
  hallucination risk. If mistral ships anti-loop mitigations (or if we
  can add chunking as the workaround from #286), revisit — 6× cheaper
  than whisper-1 would be a big cost win.

- **Audio preprocessing preset variants by genre.** `speech_optimal_v1`
  is tuned for conversational speech. If we start ingesting music-heavy
  podcasts or news interviews with heavy background noise, we may want
  `speech_news_v1` / `interview_v1` etc. Today profile picks one preset
  globally.

## Related references

- `docs/wip/POST_REINGESTION_PLAN.md` — 6-step validation plan when the
  next production corpus lands.
- `docs/wip/BACKEND_2_6_CLOSEOUT.md` — 2.6 epic handoff, what shipped
  and what's deferred to 2.7.
- `docs/wip/VIEWER_609_SCOPING.md` — viewer UI slices for insight
  cluster + explore-expansion integration.
- `docs/guides/AI_PROVIDER_COMPARISON_GUIDE.md` — real-episode numbers
  per provider (what the 10-feed run was expected to hit).
- `docs/guides/REAL_EPISODE_VALIDATION.md` — when / how to build a
  harness; ties back to the "real episodes before push" CLAUDE.md rule.
