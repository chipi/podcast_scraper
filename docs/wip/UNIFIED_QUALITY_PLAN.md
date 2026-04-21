# Unified Quality Plan (post-2.6, pre-2.7)

Single stream of work consolidating `QUALITY_IMPROVEMENTS_BACKLOG.md`
findings with `POST_REINGESTION_PLAN.md` steps into one sequenced plan.

**Baseline:** main branch after the feature-work PR (from the other
workspace) lands. That PR changes no existing results — it's additive
feature work, so metrics measured on `my-manual-run4` remain the
reference baseline.

**Explicitly out of scope (2.7 epic):** proper commercial / sponsor
content cleanup with audio-break detection, ML ad-classifier, or
transcript pre-filter. Separate ticket set owned by 2.7.

**Mini-action included here, in scope:** a **prompt-only ad-skip
instruction** in the mega-bundle prompt. ~10 LOC. Cuts ad-extraction by
~50–80 % so we can honestly measure the rest of our fixes.

## Phase 0 — Rebase & reset (0.5 day)

1. Wait for feature-work PR to merge into main.
2. Rebase `chore/wip-cleanup` onto new main. Confirm no conflicts.
3. Re-confirm the 100-ep baseline metrics still hold (re-run
   `scripts/validate/validate_post_reingestion.py --corpus <path>`
   against the existing `my-manual-run4` corpus — same corpus, same
   numbers expected since feature PR doesn't change pipeline output).
4. Open a single tracking PR (`feat/2.6-quality-polish` or similar) that
   will accumulate Phases 1–4 commits.

## Phase 1 — Prompt-level quality fixes (1 day)

All edits land in `src/podcast_scraper/prompting/megabundle.py` plus the
6 per-provider `summarize_mega_bundled` / `summarize_extraction_bundled`
methods. Single re-run on the 100-ep corpus validates all four at once.

### 1a. Commercial-lite prompt tightening (Finding 14 subset)

Add to mega-bundle user prompt:

> "Do **not** extract marketing claims, sponsor reads, subscription
> pitches, or paid-promotion content. Skip passages describing a
> product or service as 'built for X', 'helps you Y', 'runs on Z', or
> that name a sponsor product. If the passage is an ad-read or host
> disclosure (e.g. 'this episode is brought to you by …', 'I work at
> …'), produce **no** insight, no topic, and no entity for that range."

**Expected lift:** top-10 clusters stop being dominated by sponsor ads;
`omnycontent` grounding moves from 58 % toward corpus median 94 %.

### 1b. Dialogue-insight rejection (Finding 12)

Add to mega-bundle user prompt:

> "Insights must be paraphrased third-person claims distilled from the
> transcript. Do **not** copy verbatim dialogue, host patter, or
> first-person speaker monologue. Start each insight with a noun +
> verb in present tense. Avoid 'we', 'I', 'you', 'let's', 'okay', or
> conversational filler ('you know', 'I mean')."

**Expected lift:** dialogue-insight rate drops from 1.6 % → < 0.5 %.
Corpus grounding lifts ~1 pp.

### 1c. KG topic-label tightening (Finding 2)

Add to the KG extraction section of the mega-bundle user prompt:

> "Topics MUST be concise noun phrases of 2–3 words. Do NOT use
> prepositions ('in', 'of', 'for', 'vs') at start or middle. Do NOT
> use event-specific proper-noun compounds ('X succession',
> 'Y leverage'). Prefer canonical concept names that could repeat
> across episodes ('nuclear program', not 'Iran's nuclear program')."

**Expected lift:** median topic words drops from 3 → 2, max from 6 → 4,
cross-feed duplication from 2.5 % → ~5 %.

### 1d. Entity kind disambiguation (Finding 7)

Add to the entity extraction section:

> "When classifying an entity, default to **org** if the name refers to
> a company, show, podcast, brand, product, publication, or
> organisation. Reserve **person** for individual human names
> (first + last, or clearly a named individual). When in doubt,
> classify as org."

**Expected lift:** show titles ("Planet Money", "Tomorrow's Cure") and
media brands stop being classified as person.

### 1e. GI Topic labels from KG, not bullets (Finding 13 + 6)

Code change in `src/podcast_scraper/workflow/metadata_generation.py`:
when building `gi_topic_labels`, **prefer KG canonical noun-phrase
topics** (already available in `summary_metadata.prefilled_extraction`
under mega_bundled mode, or from `kg_payload` after KG runs) over
summary bullets. Fallback to bullets only when KG topics absent.

Also drop the `[:200]` mid-word truncation in `gi/pipeline.py:62`;
if truncation is needed, do it at word boundary.

**Expected lift:** GI Topic labels drop from median 20-word sentences
to 2–3-word canonical concepts. Viewer's topic surface becomes useable.

### Phase 1 validation

Re-run `validate_post_reingestion.py` on a 20-ep subset (cheaper than
full 100) of the existing corpus. Compare against the
`_post_reingest_100ep.json` baseline. Accept when:

- Ad-cluster contribution in top-10 drops from 80 % → < 30 %.
- Dialogue-insight rate < 0.5 %.
- Grounding variance across feeds narrows (omnycontent ≥ 85 %).
- GI topic label median words ≤ 4.
- KG topic median words ≤ 3, max ≤ 5.

## Phase 2 — CLI / orchestration fixes (0.5 day)

### 2a. Bridge non-overlap fixture test (Finding 5)

Add a unit test that constructs a GI with topic A and a KG with topic
B (non-overlapping). Assert bridge `both = 0`, `gi_only = 1`,
`kg_only = 1`. If the current threshold merges them anyway, tighten it.

Today we observe `both = min(gi_topics, kg_topics)` mechanically
regardless of semantic overlap — the test will catch any regression
after the threshold tune.

### 2b. UI layout auto-detect (Finding 9)

Code change in `src/podcast_scraper/config.py`:

- When `output_dir` contains a `feeds/` subdirectory (i.e., it's a
  multi-feed corpus parent) AND the run is single-feed,
  `single_feed_uses_corpus_layout` auto-resolves to `True` regardless
  of the explicit flag value.
- Emit a log line when the override fires so ops sees the reason.

UI consumer (`web/gi-kg-viewer`) no longer needs to manually set the
flag — the Config validator handles it. Compatible with both explicit
single-feed and multi-feed dispatch.

### 2c. UI default `max_episodes` guardrail (Finding 10)

Changes in `web/gi-kg-viewer` form:

- Default `max_episodes` input to `25` instead of empty.
- Require explicit "Unlimited" checkbox to unset it.
- Show estimated cost + time once the audio URL list is resolved.

### 2d. Corpus-finalize auto-trigger after partial runs (Finding 11)

Detect when a feed run lands in an existing corpus parent (i.e.,
`feeds/` already exists) and auto-trigger
`finalize_multi_feed_batch` on the enclosing corpus to refresh
`corpus_manifest.json`, `corpus_run_summary.json`, and `search/`.

Covers UI-initiated single-feed adds + manual post-run scrapes.

## Phase 3 — NER investigation (0.5 day)

### 3a. NPR speaker detection (Finding 16)

Spot-check 2–3 NPR transcripts vs a feed where host detection works
(e.g., WSJ, megaphone).

- Compare intro patterns (byline-only vs "I'm <name>" vs cold open).
- Check what `DEFAULT_SPEAKER_NAMES` / spaCy trf NER produces for each.
- Adjust the speaker-detection heuristic if a systematic gap is found.

If the fix is tractable, add to the same PR. If it requires a
speaker-detection-pipeline redesign, file as its own follow-up.

### 3b. NER provider validation at scale (POST_REINGESTION step 6)

Still open: the NER smoke was 5 eps / 15 entities putting spaCy trf
tied with all 6 cloud LLMs. Before 2.7, re-run on a larger / harder
corpus (20+ eps) to confirm spaCy trf stays within 3 pp of best cloud.

- Use the new 100-ep corpus as input.
- Score spaCy trf + 6 cloud LLMs on a random 20 ep subset.
- If spaCy trf lags, consider reverting `cloud_balanced` NER to a
  cloud provider.

## Phase 4 — Eval baseline refresh + docs (0.5 day)

POST_REINGESTION_PLAN steps 4 + 5 rolled in.

### 4a. Score new corpus against silver refs

- GI insight coverage vs `silver_sonnet46_gi_benchmark_v2`.
- KG topic coverage vs `silver_sonnet46_kg_benchmark_v2`.
- Quote verbatim rate (already at ~100 % on the 100-ep corpus, but
  lock in).
- Quote diversity (different transcript sections).

### 4b. Update `docs/guides/AI_PROVIDER_COMPARISON_GUIDE.md`

- Add 100-ep real-corpus numbers to the cloud_balanced profile row
  (insights/ep, grounded %, quotes/insight, costs).
- Document the range across feeds (58–98 % grounding) and the known
  causes (findings 12, 14).

### 4c. Update `docs/guides/eval-reports/EVAL_HELDOUT_V2_2026_04.md`

If Phase 1 prompts substantially lifted metrics, record the new
baseline. Otherwise leave the v2 report as-is and add a footnote.

### 4d. Update `docs/wip/EXPLORE_EXPANSION_IDEAS.md`

Strike items validated by the 100-ep run; promote still-open items
into this unified plan or a 2.7 ticket.

## Out of scope (defer to 2.7)

- **Finding 14 proper commercial cleanup.** Audio-break detection,
  transcript sponsor-segment stripping, ML ad-classifier. User has
  tickets for this in 2.7 — we only do the prompt-lite mitigation
  above.
- **Finding 15 threshold sweep.** Depends on #14 being cleaned first.
  Do after 2.7 commercial cleanup lands.
- **Voxtral revisit (backlog Idea 5).** Separate research.
- **LoRA hybrid pipeline revival.** Separate track.
- **#609 viewer UI slices beyond the UI fixes in Phase 2.** Slice 1
  onwards is independent 2.7 / post-2.7 frontend work.

## Success criteria for the whole stream

Re-run `scripts/validate/validate_post_reingestion.py` on the 100-ep
corpus AFTER all fixes; compare to `_post_reingest_100ep.json` baseline:

| Metric | Baseline (100-ep) | Target post-stream |
| --- | :-: | :-: |
| Grounded % | 91.4 | ≥ 96 |
| Omnycontent grounded % | 58.3 | ≥ 85 |
| Quotes / insight | 1.35 | ≥ 1.6 |
| Dialogue-insight rate | 1.6 % | < 0.5 % |
| KG topic median words | 3 | ≤ 3 |
| Cross-feed topic duplication | 2.5 % | ≥ 5 % |
| Top-10 insight clusters that are sponsor ads | 80 % | ≤ 30 % |
| NPR host detection | 0/10 | ≥ 8/10 |

## Estimated total effort

~2.5 days of coding + validation, assuming no surprises.

## Related references

- `docs/wip/QUALITY_IMPROVEMENTS_BACKLOG.md` — findings list, 16
  findings + 6 ideas (source of Phase 1–3 items).
- `docs/wip/POST_REINGESTION_PLAN.md` — original 6-step validation
  (steps 1–3 subsumed into Phase 1, steps 4–6 become Phase 3–4).
- `docs/wip/PROD_RUN_ANALYSIS_100EP.md` — 100-episode production run
  deep-dive, the baseline metrics come from here.
- `docs/wip/BACKEND_2_6_CLOSEOUT.md` — handoff; this plan completes
  the "post-2.6 polish" before 2.7 opens.
