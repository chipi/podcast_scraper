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

## Findings from 2-feed smoke run (2026-04-21, `my-manual-run3-2`)

Same rebase + fix set, 2 feeds (NPR prediction-markets, WSJ Iran regime),
1 episode each, 2 runs per feed (identical output). Used to confirm the
post-#648 state before spinning the larger 100+ run.

Mostly confirms the 10-feed findings. Two new observations worth adding:

### 5. Bridge `both=10` every episode is mechanically perfect (too lenient?)

**Observed.** Across every single run in both the 10-feed and 2-feed
corpora, bridge identity counts are identical shape:
`both = min(gi_topics, kg_topics) = 10`, `gi_only = 0`,
`kg_only = len(kg_entities)`. Bridge always claims **all** KG topics
have a GI source.

**Root cause (hypothesis).** GI Topic nodes are derived from summary
bullets (6 long sentences per episode). Each bullet mentions multiple
concepts, so any KG noun-phrase topic can find a semantic match inside
at least one bullet. Bridge threshold may be calibrated for label-vs-
label matching, not bullet-vs-label — effectively everything matches.

**Why it matters.** We aren't testing bridge alignment quality — we're
observing "bullets are long enough to contain all concepts." A fixture
where GI and KG have deliberately non-overlapping topics would reveal
whether bridge is actually filtering or just merging-all.

**Fix.** Add a regression test with non-overlapping topic fixtures;
tighten bridge threshold if the current one merges irrelevant topics.
Pair with a "Topic label confidence" signal (bullet-vs-noun-phrase
similarity score) surfaced in bridge output so downstream consumers
can threshold themselves.

**Priority / effort.** Medium priority, medium effort (~1 day + re-run).
Defer until next production run confirms at scale.

### 6. GI Topic node IDs on disk are bullet-slugs (cosmetic)

**Observed.** `gi.json` stores Topic node IDs as slugified bullet text:

```json
"id": "topic:prediction-markets-exemplified-by-kalshi-allow-users-to-bet-on-the-outcomes-of-a"
"properties": { "label": "Prediction markets, exemplified by Kalshi, allow users to bet on ... sports to politics." }
```

KG uses canonical slugs: `topic:prediction-markets`.

**Why it matters.** Bridge reconciles via semantic alignment at merge-
time, so the corpus-wide identities list shows clean KG-style slugs.
But anything that reads `gi.json` directly (viewer's raw GI panel,
per-episode cluster browse) sees bullet-slug IDs + long-bullet labels.
UX is ugly; search behaves oddly.

**Fix.** When GI builds Topic nodes, prefer the KG's noun-phrase label
+ canonical slug when available (via `topic_labels_kg` param already
passed into `gi.build_artifact`). The alignment already happens in
bridge — run it earlier at GI build time.

**Priority / effort.** Low–medium (cosmetic but visible in viewer),
low effort. Pairs with #609 viewer work — check if viewer pulls raw
`gi.json` or uses bridge.

### 7. Entity kind misclassification (Planet Money = person, Tomorrow's Cure = person)

**Observed.** NPR Prediction-Markets episode entities:
- `Planet Money` → `kind=person` (should be `org`)
- `Tomorrow's Cure` → `kind=person` (it's a podcast series / show title)
- `Mayo Clinic` → `kind=org` ✓ (correct)
- `Taylor Swift` → `kind=person` ✓

WSJ Iran episode: all 15 person + 2 org look correct.

**Root cause (hypothesis).** The mega-bundle prompt constrains
`entity_kind` to `person / org / place`. Ambiguous cases (show titles,
product names, media brands) get bucketed into `person` by default.

**Why it matters.** Corpus-wide entity rollups will include show titles
and media brands as "people", polluting host/guest detection, guest
network visualisation, and entity search.

**Fix.** Add `show` and/or `product` to the entity kind enum, or tighten
the prompt with a "when unsure between person and org, default to org"
instruction. Run a fixture test with known-ambiguous entities (show
titles, brand names) to measure improvement.

**Priority / effort.** Medium priority (polluted entity data is hard to
clean up later), low effort (~1 day prompt + re-run).

### Other validated-as-clean signals

- **100 % grounding** on both smoke-run episodes (up from ~95 % in 10-feed).
- **100 % verbatim** on quotes (13/13 NPR, 17/17 WSJ).
- **No fallbacks** (the Gemini 503 retry fix from `c9ff5156` is landed
  and working).
- **Bridge v2 `identities` schema** is populated correctly — earlier
  "0 nodes / 0 edges" report was wrong; I was looking for bridge v1
  shape. Bridge is producing sensible identity lists, just with the
  threshold caveat in Finding 5.

## Findings from 53-episode single-feed UI run (2026-04-21, `my-manual-run-10/run_20260421-190016_2606de6d`)

UI-triggered run of ~53 episodes from one acast feed, dropped into the
multi-feed corpus root by mistake (`single_feed_uses_corpus_layout: false`
+ corpus parent output_dir = orphaned single-feed dir at root). Pipeline
itself worked; surfaces orchestration + cross-episode signals worth
tracking.

### 8. Within-feed topic duplication is 10× better than cross-feed

**Observed.** 528 topics across 53 episodes → 475 unique strings →
**10 % duplication** (vs 1 % cross-feed in the 10-feed run).

**Interpretation.** Same-show episodes naturally repeat concepts
("Federal Reserve", "inflation", "stock market" across an economics
podcast). Cross-episode clustering *within* a feed should work much
better than across-feed, even with exact-string match.

**Action.** When we run the 100+ episode production corpus, measure:

- Single-feed duplication % per feed (expect ≥ 10 %).
- Cross-feed duplication % (expect ≤ 1–2 %).
- Semantic-clustering lift over exact-match: does embedding-based
  clustering 3–5× the useful cluster count?

If within-feed clustering is solid, corpus-wide search will still be
feed-siloed. Need a canonicalisation pass (Finding 2) to break that
silo if we want true corpus-wide topic browse.

**Priority / effort.** Info only until the 100+ run lands; then it
informs whether Finding 2 (topic canonicalisation) is blocking.

### 9. UI single-feed runs land outside the corpus-layout feeds/ tree

**Observed.** UI-triggered acast run landed at
`my-manual-run-10/run_20260421-190016_*/` (legacy single-feed layout)
even though `my-manual-run-10/` is a multi-feed corpus root containing
`feeds/` subdirs. Config used `single_feed_uses_corpus_layout: false`.

**Consequences:**

- `corpus_manifest.json`, `corpus_run_summary.json`, `search/` at root
  still reflect the earlier 10-feed batch only. The UI run's 53
  episodes are invisible to corpus-level tools.
- Same acast feed already exists as `feeds/rss_feeds.acast.com_f08ef8e2/
  run_*/` from the 10-feed batch. The UI run created a second
  disconnected acast dir at top level.
- Viewer / indexer / cluster pipelines walking `feeds/**` will miss
  the UI run. Walking root `run_*/` will miss the 10-feed batch.

**Fix (known, user flagged).** UI must set
`single_feed_uses_corpus_layout: true` when writing into a multi-feed
corpus parent. Ideally the behaviour is:

- If `output_dir` contains a `feeds/` subdir (i.e., an existing corpus
  parent), single-feed runs always use corpus layout regardless of the
  flag.
- If `output_dir` is empty or brand-new, user's flag wins.

Consider making the flag implicit / auto-detected at Config-validator
time (companion to the existing wrapping validator) so manual flag
setting becomes unnecessary.

**Priority / effort.** High (blocks UI-driven corpus use), low–medium
effort (~30 LOC + test).

### 10. UI default `max_episodes = null` triggers unlimited scrapes

**Observed.** UI run has no episode cap; scraped ~53 episodes and
still running. Can silently produce hundreds of episodes + large
transcription cost.

**Fix.** UI should either (a) default to a safe cap (e.g., 25) with an
opt-in override, or (b) show a cost / time estimate before confirming
when `max_episodes` is empty.

**Priority / effort.** Medium (cost-control), low effort (UI-only).

### 11. Stale root-level lock + corpus state from earlier runs

**Observed.** `my-manual-run-10/.podcast_scraper.lock` dates to an
earlier run (17:53). `corpus_manifest.json` and `search/` reflect the
original 10-feed batch. If the UI run had respected corpus layout,
these would need to be refreshed (corpus-finalize) after the run.

**Fix.** After any new run that adds episodes to an existing corpus
parent, trigger corpus-finalize automatically (update manifest + run
summary + search index). Pairs with Finding 9.

**Priority / effort.** Medium, low effort.

### Confirmations this run validated

- **Pipeline quality matches prior runs:** 12.0 insights/ep, 93.7 %
  grounded, 10.0 topics/ep, median 3-word topics — consistent with the
  10-feed and 2-feed runs. No regressions.
- **Entity kinds populated with 2 categories** (354 org + 485 person).
  No `place` observed across 53 episodes from a finance/news show —
  either `place` is too rare in this content or the prompt suppresses
  it. Worth probing with a geographically-rich feed.
- **Quotes/insight 1.31** — persistent across all 3 runs at this scale.
  Confirms Finding 1 is systematic, not noise.

### 12. ~4 % of insights are verbatim transcript dialogue, not distilled claims

**Observed.** Scanning 752 insights across 53 episodes (UI run), **31
(4.1 %) match dialogue heuristics** — host patter, filler words ("you
know", "let's", "I mean"), first-person commentary kept in speaker
tone. Example:

> "It's a deal to our let's do it. Okay, So we do, in fact have the
> perfect guest. We're going to be speaking with Brad Jacobs..."

That text lives in an `Insight.text` property, not a Topic label. The
Quote node attached to it copies a literal transcript span — grounding
trivially succeeds because the "insight" **is** already a verbatim
quote.

**Root cause.** Mega-bundle prompt does not strongly enforce "insight =
distilled third-person claim". LLM sometimes lifts transcript text
verbatim, especially host intros and interviewee monologues.

**Why it matters.** Every such insight inflates the "grounded" count
without carrying real claim value. Clustering on these will cluster on
host filler rather than ideas. Viewer users see these as mystery
"topics" — poor UX.

**Fix.** Strengthen mega-bundle prompt:

- "Insights must be paraphrased third-person claims distilled from the
  transcript, not verbatim lifts."
- "Start insights with a subject + verb in present tense. Avoid
  'we', 'I', 'you', 'let's', 'okay', conversational filler."
- Optionally: post-process filter in `megabundle_parser.py` that rewrites
  insights containing dialogue markers (or drops+re-queries them).

**Priority / effort.** High (viewer UX + cluster quality), low
effort (~10 LOC prompt + re-run). Measure after the 100+ run.

### 13. GI Topic labels = summary bullets (long "bunch of text" by design)

**Observed.** 320 GI Topic nodes across 53 episodes; labels have
**median 141 chars / 20 words, max 200 chars (hard cap), min 10 words**.
They are literally the summary bullets copied in as topic labels. Some
are truncated mid-word at 200 chars ("floodgates for more lawsu").

**Root cause.** In `metadata_generation.py`, summary bullets are passed
as `topic_labels` into `gi.build_artifact`. Then
`_make_topic_node_specs` (`gi/pipeline.py:62`) stores the raw bullet as
the topic label with a `[:200]` slice.

**Why it matters.** A viewer showing GI Topic nodes as a topic tour
sees 20-word sentences, which is what the user calls "just a bunch of
text". Bridge reconciles at merge-time via semantic alignment (see
Finding 5) but the raw `gi.json` is ugly, and anything downstream
walking `gi.json` Topic nodes directly sees the bullet-as-topic shape.

**Fix options:**

- **Best:** When KG's noun-phrase topics are available, use them as GI
  Topic labels too. The existing `topic_labels_kg` pathway into
  `gi.build_artifact` should replace bullet-as-topic entirely when KG
  topics exist. Pair with Finding 6 (ID canonicalisation).
- **Fallback:** Summarise bullets to 2–4 word topic labels via the
  same mega-bundle call (already produces short topic labels for KG).
  Re-use that list for GI Topic nodes.
- Drop the mid-word `[:200]` truncation; if a label must be shortened,
  do it at word boundary.

**Priority / effort.** High (directly visible in viewer), low effort.
Pairs with the KG v3 prompt tightening (Finding 2) and Viewer UI #609
so the viewer always sees canonical short topics.

## Findings from 100-episode production run (2026-04-21, `my-manual-run4`)

Full deep-dive in `docs/wip/PROD_RUN_ANALYSIS_100EP.md`. The single
biggest quality surface uncovered by running insight-clustering on the
corpus:

### 14. Sponsor ad-reads are being extracted as insights and polluting clusters

**Observed.** `scripts/validate/validate_post_reingestion.py` built 51
insight clusters from 1200 insights across 100 episodes. Top-10 clusters
by cross-episode reach:

| Reach | Canonical "insight" | What it really is |
| :-: | --- | --- |
| **10 eps** | "Ramp automates 85 % of expense reviews with 99 % accuracy using AI." | Sponsor ad read |
| 6 eps | "WorkOS provides core enterprise capabilities like SSO, SCIM, RBAC, audit logs." | Sponsor ad read |
| 6 eps | "Rogo AI's platform was designed to support how Wall Street bankers actually work." | Sponsor ad read |
| 6 eps | "The single most underestimated force in international relations is actually stupidity." | Host catchphrase (legit) |
| 5 eps | "Ramp understands that no one wants to spend hours chasing receipts..." | Sponsor ad read |
| 5 eps | "OpenAI, Cursor, Anthropic, Perplexity, and Vercel all use WorkOS." | Sponsor ad read |
| 4 eps | "Shopify runs on Ramp, Stripe runs on Ramp, and my business does too." | Sponsor ad read |
| 4 eps | "Rogo AI connects directly to your system to work with your actual data." | Sponsor ad read |
| 4 eps | "political party becomes the prism through which we see every other aspect of our identities." | Host recurring line |
| 4 eps | "I work at The New York Times, which is suing OpenAI and Microsoft and Perplexity..." | NYT sponsor disclosure |

**8 of 10 top clusters are advertising content.** The biggest cluster has
**the same Ramp ad extracted verbatim as a "fact" across 10 different
episodes.**

**Why it matters.** This single finding explains several prior symptoms:

- **Finding 1 (low quotes/insight).** Ad copy doesn't ground to meaningful
  transcript quotes beyond the ad itself — only the ad sentence supports
  the "insight".
- **Feed-variable grounding (this run: 58–98 %).** Feeds with more
  sponsor inventory (`omnycontent` at 58.3 %) have worse grounding because
  a bigger chunk of extracted "insights" are ad-copy.
- **Cross-episode clustering signal is real but polluted.** The tech works
  — it accurately finds duplicates. It's the input that's dirty.

**Fix options (ranked by effort):**

1. **Mega-bundle prompt tightening (cheapest).** Add explicit instructions:
   "Do not extract marketing claims, sponsor reads, subscription pitches,
   or paid-promotion content. Skip any passage that describes a product or
   service as being 'built for X', 'helps you Y', or references trademarks
   like Ramp, WorkOS, Rogo, etc. If the passage is an ad-read or host
   disclosure, produce no insight for that range." Low-risk, ~10 LOC.

2. **Post-filter heuristic.** After mega-bundle returns, drop insights
   matching ad-copy patterns: company-name + verb + feature, "runs on",
   "powered by", "brought to you by", trademark-heavy phrasing. ~30 LOC,
   conservative fallback if the prompt tweak underperforms.

3. **Pre-filter sponsor segments from the transcript.** Harder — requires
   detecting ad breaks from audio cues or transcript heuristics. Higher
   upside (removes bad text BEFORE the model sees it) but much more work.

**Priority / effort.** **Highest priority** — polluting multiple other
metrics. Option 1 + 2 together, ~1 day of work + re-run on omnycontent
feed to measure lift.

### 15. 88 % of insights are singletons — clustering threshold + scale

**Observed.** 51 clusters cover 141 insights (11.8 %). The remaining
1059 insights (88.2 %) are singletons.

**Interpretation.** At 100 episodes with a 0.75 embedding-similarity
threshold, most insights are too specific / varied to cluster. That's
partly fine (unique content should stay unique) but singletons also
include:

- Generic claims that didn't find similar peers yet (need more corpus).
- Near-duplicates that the threshold missed (too strict).
- Dialogue-insight pollution (Finding 12) and sponsor-read pollution
  (Finding 14) spreading the embedding space.

**Fix.** Sweep cluster threshold in {0.65, 0.70, 0.75, 0.80, 0.85} once
findings 12 + 14 are fixed; re-measure cluster count and cross-episode
reach to pick the right operating point. Don't tune the threshold
before cleaning the inputs — you'd be optimising against polluted data.

**Priority / effort.** Medium. Do AFTER fix for Finding 14 lands.

### 16. NPR speaker detection misses every host (0 host roles across 10 eps)

**Observed.** Per-feed entity role counts:

| Feed | `host` | `guest` | `mentioned` |
| --- | :-: | :-: | :-: |
| `wsj` / `megaphone/3581` | 20 | 0 | ~140 |
| most feeds | 10 | 0–4 | ~140 |
| **`npr`** | **0** | **0** | 147 |

spaCy trf NER is missing hosts on NPR consistently across 10 episodes.

**Root cause (hypothesis).** NPR transcripts may have host intro patterns
(e.g., byline-only intros, no "I'm <host name>" disclosure) that don't
match the heuristics the speaker-detection pipeline uses for host
identification.

**Fix.** Spot-check 2–3 NPR transcripts to see what the host intro looks
like; adjust speaker detection heuristics or NER prompt accordingly. May
pair with Finding 16 style audit across all feeds.

**Priority / effort.** Medium. Speaker detection is downstream (viewer /
cluster visualisation), so invisible for now, but degrades over time.

### Validated-clean signals (100-ep corpus)

- **0 fallbacks** — Gemini 503 retry fix (`c9ff5156`) held in production.
- **12 insights / 10 topics / ~15 entities per episode** consistent across all
  10 feeds (no per-feed floor-drop aside from omnycontent's grounding).
- **938/1000 unique KG topics** (93.8 %). Up from 99/100 in the 10-feed
  run — cross-episode duplication now at **4.6 %** (exact match) and
  **2.5 % cross-feed**, enough for semantic clustering to produce real
  groups (once Finding 14 is cleaned up).
- **All 6 explore-expansion CLI commands** execute without error.
- **Corpus finalize ran correctly** (manifest + summary + search/
  vector index + id_map present).

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
