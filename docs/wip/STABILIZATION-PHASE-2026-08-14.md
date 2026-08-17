# Stabilization phase — 2026-08-14

> **Superseded as a plan — kept as evidence.** All work items from this phase are now tracked
> in **[epic #1657 — corpus integrity](https://github.com/chipi/podcast_scraper/issues/1657)**
> and its child issues. This document remains the *measured record* behind them: the S3
> per-episode audit, the S6 leverage decision and its two gates, and the S7 verdict on the
> first real enrichment run. Read it for evidence, not for what to do next.
>
> The root cause found after S7 — speaker detection silently skipped for every episode over
> 25 MB — is **#1646**, and it is why the S3 audit came back clean while the corpus was
> already damaged: S3 measured structure and never opened a GI file.

**Status:** OPEN — ingestion deliberately paused. Planning moved to #1657 on 2026-08-14.

A hold on corpus growth to fix what we found while growing it. Ingestion resumes only
when the exit criteria below are met.

The rationale, stated precisely: **enrichment is corpus-wide and re-runnable, so its
correctness does not depend on when it runs. Per-episode artifacts are not.** Summaries,
insights and KG nodes are minted at ingest time by whatever the pipeline is that day. If
a per-episode defect exists, every episode ingested before it is found carries it. So the
pause protects the per-episode layer, not the enrichment layer.

**That rationale has been vindicated, not retired.** S3 audited the per-episode layer and
found it healthy — but S3 measured *structure* (bullet counts, KG density, GI↔KG bridging)
and never opened a GI file. The verdict below did, and found a per-episode defect of exactly
the shape the pause exists to catch: episodes ingested on 2026-08-12/13 carry insights that
no surface will ever show. The pause stays.

---

## Entry state (frozen 2026-08-14 ~07:00Z)

| | |
| --- | --- |
| Corpus | **662 episodes**, 14 feeds |
| Running | `1ebba1af` — The Pragmatic Engineer, off=17 n=16 |
| Queued | `0260ddba` — The Pragmatic Engineer, off=33 n=16 |
| Queued | `6d9d9c6f` — `corpus_enrichment`, `--only <7 deterministic>` (diagnostic) |
| Cancelled | six new-feed windows — see "Resume" below |

The two Pragmatic Engineer windows are the tail of the previous batch (17+16=33, 33+16=49,
landing the show at its 50 target). They were left to run: cancelling mid-batch would have
left the show at a ragged offset for no benefit.

### Cancelled, to be re-queued verbatim on resume

Cancelled via `POST /api/jobs/{id}/cancel` while `queued` — a clean dequeue, no SIGTERM.
Re-queueing costs nothing: `skip_existing` is GUID-keyed corpus-wide, so a re-issued job
is idempotent against whatever already landed.

| Show | Feed URL | offset | max_episodes |
| --- | --- | --- | --- |
| Odd Lots | `https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/8a94442e-5a74-4fa2-8b8d-ae27003a8d6b/982f5071-765c-403d-969d-ae27003a8d83/podcast.rss` | 0 | 25 |
| Odd Lots | *(same)* | 25 | 25 |
| Conversations with Tyler | `https://rss.libsyn.com/shows/137081/destinations/850607.xml` | 0 | 25 |
| Conversations with Tyler | *(same)* | 25 | 25 |
| In Our Time | `https://podcasts.files.bbci.co.uk/b006qykl.rss` | 0 | 15 |
| In Our Time | *(same)* | 15 | 15 |

All with `skip_existing=true`, `episode_order=newest`. Window sizes follow the audio-minutes
rule from **G-opt** (`window ≈ 1400 / median_minutes`): Odd Lots 50m→25, CwT 55m→25,
In Our Time 54m→15.

---

## Exit criteria — what "stable" means

Ingestion resumes when **all** of these hold. Each must be evidenced by a command and its
output, not by assertion.

| # | Criterion | Status 2026-08-14 13:00Z |
| --- | --- | --- |
| 1 | **Enrichment produces data on the real corpus.** A named, non-empty enricher set runs, and `GET /api/corpus/enrichments` shows artifacts for it. (H2) | **MET** — by the operator-block route, **not** the royal route |
| 2 | **An empty enricher set cannot silently succeed.** The no-op path is loud, with a test. | **NOT MET** — no code exists |
| 3 | **The per-episode layer has been audited** on a sample spanning several feeds. | **MET but insufficient** — S3 measured structure, not attribution; see criterion 5 |
| 4 | **No unexplained job failure** in the current queue, understood line-by-line. | **PARTIAL** — 1 of 7 explained |
| 5 | **Insight attribution loss is diagnosed and bounded.** *(added 2026-08-14)* The share of GI insights dropped as `surfaceable: false` is understood at cause, the affected runs are enumerated, and the loss rate on newly ingested episodes is back to the ≤2 % baseline observed before 2026-08-12. | **NOT MET** |

Explicitly **not** exit criteria: H1 (audio archive) is blocked on operator-supplied
secrets and must not gate the phase; H4 is recorded-only by operator decision.

**H1 is confirmed non-blocking for the attribution work.** Transcripts are retained with
working diarization — the Latent Space episode that lost 100 % of its insights has a
96 KB transcript with 4 distinct `SPEAKER_NN` labels across 120 turns. Re-deriving
attribution needs the transcript and an LLM pass, not the discarded audio.

---

## Work items

### S1 — fix enrichment the royal route — `[NOT STARTED — now P1.3]`

Status corrected 2026-08-14: this was marked `[IN PROGRESS]`, but no code was ever written
(`git log --since=2026-08-13 -- src/ tests/` returns nothing). Confirmed still live in prod —
`run_id=6b7159ef` reports `profile: null`.

Homework **H2**. Advisor engaged on the design. Working hypothesis of the causal chain:

| # | Location | What happens |
| --- | --- | --- |
| 1 | `config.py:4097` | `profile_name = data.pop("profile", None)` — name is known here |
| 2 | `config.py:4173` | `profile_dict.pop("profile", None)` — discarded, because `Config` is `extra="forbid"` with no `profile` field |
| 3 | `orchestration.py:1791` | `profile = getattr(cfg, "profile", None) or block.get("profile")` → both empty |
| 4 | `profile_sets.py:159` | `enricher_set_for_profile(None)` → empty set → zero enrichers, reported as success |

Constraints on the fix:
- `profile_sets.py` stays the source of truth. The `enrichment:` YAML block is advisory
  documentation mirroring it, **not** an input to resolution.
- The profile must survive **both** invocation paths, one of which crosses a **subprocess
  boundary** (`_maybe_spawn_enrichment_after_pipeline` spawns a child). An in-memory
  attribute alone does not survive that.
- Royal route only — no shim, no hardcoded profile.

### S2 — verify enrichment on the real corpus — `[DONE 2026-08-14 — superseded by S7]`

The queued `6d9d9c6f` (`--only <7 deterministic>`) is a *diagnostic*, not the fix: it
bypasses profile resolution by naming enrichers explicitly. Its value is proving the
executor and the enrichers themselves work. Do not read a green result from it as evidence
that S1 is fixed — those are different code paths.

It was cancelled and never ran. `d08408f0` did the verification instead, via the operator
`enrichment.enrichers:` block — a third route, and the same caveat applies with full force:
**`6b7159ef` succeeding is not evidence that S1 is fixed.** Findings in S7 below.

### S3 — per-episode quality audit — `[DONE 2026-08-14 — exit criterion 3 met]`

62 episodes sampled across **all 14 feeds** (5 per feed where available), reading
`/api/corpus/episodes/detail`.

**Result: the per-episode layer is healthy.** This is the layer the ingestion pause exists
to protect, and it turns out not to need protecting.

| Feed | n | bullets/ep | median bullet chars | KG nodes/ep | GI↔KG bridged | unlinked GI |
| --- | --- | --- | --- | --- | --- | --- |
| Invest Like the Best | 5 | 10.2 | 186 | 24.8 | 18.6 | 0.2 |
| Lenny's Podcast | 5 | 11.0 | 201 | 15.4 | 12.2 | 0.0 |
| Latent Space | 5 | 9.2 | 191 | 22.6 | 14.6 | 0.6 |
| The a16z Show | 5 | 8.8 | 188 | 25.4 | 14.8 | 0.0 |
| The Pragmatic Engineer | 5 | 8.4 | 227 | 12.6 | 10.0 | 0.0 |
| Hard Fork | 5 | 8.0 | 228 | 21.2 | 16.6 | 0.0 |
| No Priors | 5 | 8.0 | 206 | 22.6 | 15.2 | 0.4 |
| The Daily | 5 | 8.0 | 195 | 22.2 | 12.6 | 0.0 |
| Unhedged | 5 | 8.0 | 191 | 20.6 | 13.4 | 0.0 |
| The Journal. | 5 | 8.0 | 180 | 18.4 | 14.0 | 0.2 |
| NVIDIA AI Podcast | 5 | 8.0 | 184 | 20.0 | 13.4 | 0.0 |
| Planet Money | 5 | 6.0 | 170 | 22.8 | 12.6 | 0.8 |
| Dwarkesh / Ideas of India | 1 each | 8.0 | ~206 | 18–26 | 11–14 | 0–1 |

Corpus-wide totals across the 62:

| Check | Result |
| --- | --- |
| Ad / sponsor copy in summaries | **1 / 62** — Lenny's Podcast, matched `use code` |
| Empty summary | 1 / 62 |
| Zero `cil_digest_topics` | **0 / 62** (exactly 5 topics per episode everywhere) |
| Missing `bridge_partition` | **0 / 62** |

Full GI/KG coverage independently confirmed via `/api/corpus/coverage`: `with_gi=662`,
`with_kg=662`, `with_both=662`, `with_neither=0`.

**Reading:** unlinked GI (`gi_only`) is 0.0–0.8 per episode against 10–18 bridged — the
GI↔KG join is doing its job. Ad contamination at 1.6 % is low for a corpus where most feeds
are ad-supported. Summary substance (6–11 bullets, 170–228 chars each) is consistent across
very different show formats.

**Conclusion that matters for sequencing:** the defect surface is entirely in the
**corpus-level enrichment layer**, not in per-episode artifacts. The pause was justified on
the theory that a per-episode defect would be baked into every episode ingested before it was
found — that theory tested clean, which lowers the risk of resuming ingestion.

#### Not covered

- **Semantic correctness.** This audit measures structure and contamination, not whether
  insights are *true* or faithful to the transcript. That remains F1qa, untouched.
- **Sampling.** 5 episodes per feed, evenly stepped — not a random sample, and not powered to
  find rare defects. A 1-in-62 finding rate means anything rarer than ~2 % is invisible here.
- **The 2 episodes missing `summary_short`** (index shows 660 of 662) were not identified
  individually.
- The single empty-summary episode was counted but not diagnosed.

### S4 — homework items worth doing in this window — `[SUPERSEDED by "Revised priorities"]`

Kept for the reasoning; the ordering below is no longer current. **D6 moved up to P1.4**
(a dry key is the leading suspect for the attribution loss, and Phase 2 spends budget
re-deriving). **H3 moved down to Phase 3.** H1 stays operator-blocked but is confirmed
**not** a blocker for the attribution fix — transcripts are retained with working diarization.

| Item | Why now |
| --- | --- |
| **D6** — every budget signal measures spend, none measures headroom | Cap was just raised to $10 and ~130 episodes are about to be queued against it |
| **H1** — audio archive built, documented, switched off | **Blocked on operator**: two GitHub secrets. Every episode ingested meanwhile discards audio, the one artifact we cannot regenerate |
| **H3** — post-ingest passes architecturally inconsistent | Directly adjacent to S1; whatever S1 touches informs it |

### S5 — reviews — `[TODO]`

Advisor / reviewer passes on whatever S1 changes before it merges.

---

## S6 — enrichment leverage decision (2026-08-14)

Operator direction: do not spend the 662-episode run confirming breakage. Decide
enable / disable / fix per enricher first, so the run demonstrates improvement.

### Verdict

**The leverage is operational, not algorithmic.** The cross-show value the product promises
already exists — `search/topic_clusters.json` has 166/269 clusters spanning ≥2 feeds — while
every enrichment run since 2026-08-08 has been a 3 ms no-op and 557 of 662 episodes have no
enrichment at all. Getting the *structurally sound* enrichers to cover 662/662 beats
repairing the starved ones.

Re-keying co-occurrence, which earlier looked like the headline fix, was **measured and
rejected**: the alias map covers only 24.2 % of distinct topic ids, and canonicalising lifts
recurrence from 4.24 % to 7.80 % — still 92.2 % singletons. See H5.

### Decision table

| Enricher | Decision | Effort | Justification |
| --- | --- | --- | --- |
| `insight_sentiment` | **ENABLE** | S | No cross-episode keying; structurally immune to the granularity problem |
| `insight_density` | **ENABLE** | S | Gate A passed 105/105 |
| `guest_coappearance` | **ENABLE** | S | Person ids converge where topic ids don't; Gate B passed |
| `temporal_velocity` | **FIX (config) then ENABLE** | S | `window_months: 24` so the window covers 2025-02 onward; consume `content_series`, **not** the `velocity` scalar |
| `topic_cooccurrence_corpus` | **DISABLE** | — | 1 of 4,216 pairs backed by ≥2 episodes; starved by input, not buggy |
| `topic_theme_clusters` | **DISABLE for now** | M | Emits ≤1 cluster at current gating; needs canonical ids, not knob changes |
| `grounding_rate` | **DISABLE** | — | Tautological — always 1.0 |

### Pre-run gates — both executed, both PASS

**Gate A — `insight_density` timing.** Threshold fixed in advance: PASS ≥ 0.9, FAIL < 0.5.

```
episodes flagged insight_density=True: 105
read ok: 105   errors: {}
has_timing TRUE : 105 / 105 (100.0%)
```

Sample payload is real, not the fallback: `{"has_timing": true, "duration_seconds": 3213.0,
"counts": {"early": 9, "mid": 8, "late": 16, "unknown": 0}, "total_insights": 33}`.

**Gate B — `guest_coappearance` viability.** Threshold: named ratio ≥ 0.5 AND ≥ 20 persons in
≥2 episodes. Result: **847 total persons**; of the top 50, **50/50 named (100 %)** and **20 in
≥2 episodes**.

**Caveat that must not be dropped:** the 100 % named ratio is measured over the **top 50 of
847**, ranked by insight count — a population biased toward well-resolved speakers. The
corpus-wide named ratio is **not** measured, and `/api/corpus/persons/top` caps `limit` at 50
(`corpus_persons.py:177`), so it cannot be measured through that route.

**Tempering the value claim:** the recurring names are overwhelmingly *hosts* — Katie Martin
+ Robert Armstrong (Unhedged), Kevin Roose + Casey Newton (Hard Fork), Sarah Guo + Elad Gil
(No Priors), Ryan Knutson + Jessica Mendoza (The Journal.). So the enricher will largely
rediscover "who hosts what". The genuinely valuable cross-show signal is thin at the top:
**Sam Altman (2 episodes)** and **Ben Horowitz (3)** are the only clear cross-show guests in
the top 50. Real, but modest.

**Data-quality observation:** `person:andreessen-horowitz` — an *organisation* — is the
top-ranked Person with 54 episodes and 723 insights. Org-as-person will distort any
co-appearance community centred on it.

### What NOT to do

- **Do not lower `min_pair_episode_count` to 1.** Every pair qualifies, `involved` exceeds
  `_MAX_LINKAGE_TOPICS = 400`, average-linkage degrades to all-singletons, singletons are
  dropped at output — `cluster_count: 0` with `status: ok`. Output-shaped silence.
- **Do not patch `grounding_rate`'s ratio.** Any denominator reachable through `SUPPORTED_BY`
  *is* the numerator. A real rate needs upstream attribution of ungrounded insights to
  speakers — a different feature.
- **Do not lower the 0.75 clustering threshold as a quick win.** Correction to an earlier
  assumption recorded here: **it is not a config knob.** `model_registry.py:1676-1680` states
  no Config field exposes it; it is a function default in `search/topic_clusters.py`. False
  merges land precisely on the product's core claim, so any sweep belongs offline with a
  manual false-merge spot-check.
- **Do not read the `temporal_velocity` scalar as "trending."** For singleton topics it is a
  two-valued function {0.0, 6.0} and nothing normalises by episodes-per-month, so it ranks by
  "published recently" against a backfill-shaped denominator.

### Still unresolved

- Which leg of the no-op is live in prod — missing `--profile` at spawn vs no
  `enrichment.enrichers:` block in the corpus `viewer_operator.yaml`. Both produce the
  observed `profile: null` + empty `per_enricher`.
- Why the 2026-08-08 run covered exactly ~12 episodes × 9 feeds.
- ~~Whether `insight_sentiment`'s sidecars exist on disk.~~ **CLOSED 2026-08-14** — they do.
  `GET /api/corpus/episode/enrichments/insight_sentiment` returns 200 with a real payload for
  an episode whose `enrichments_available` lists only `insight_density`. Confirmed
  reporting-layer gap, not an enricher failure.
- Corpus-wide named-person ratio (see Gate B caveat).

---

## S7 — verdict on the first real enrichment run (2026-08-14)

Job `d08408f0` ran unattended at 10:16:52Z–10:17:08Z (exit 0, `run_id=6b7159ef`,
`duration_ms=6905`, 678 bundles). It is the first run in the corpus's life to produce
enrichment data. **Verdict: the enrichers do what they were designed to do; two of the four
produce output with little product value; and the data exposed a per-episode defect upstream
of enrichment entirely.**

Note `profile: null` in both `/api/enrichment/run-summary` and `/api/enrichment/status`.
The enricher set came from the `enrichment.enrichers:` block in the corpus
`viewer_operator.yaml`, **not** from profile resolution. **S1 is still unfixed** — criterion 1
is met by workaround.

### Per-enricher

| Enricher | Verdict | Evidence |
| --- | --- | --- |
| `insight_sentiment` | **Sound — keep** | Sample `total_insights` sums to **497**, exactly the surfaceable Insight count in the same episodes' GI. No drift. Polarity pos 243 / neu 84 / neg 170 |
| `insight_density` | **Sound — keep** | Sample sums to **470** = 497 surfaceable − 27 with no `SUPPORTED_BY` edge to a timed Quote (`insight_density.py:108`). Fully explained |
| `temporal_velocity` | **Ship `content_series` only; suppress the scalar** | `velocity_last_over_6mo` over 5,918 topics: **5,632 = 0.0, 256 = exactly 6.0**, 30 anything else. `content_series` is real: `person:donald-trump` 34 weeks / 80 mentions |
| `guest_coappearance` | **Revisit the ENABLE decision** | 277 pairs, **267 (96.4 %) have `episode_count: 1`**. All **6 communities are single-show host groups**; **zero** cross-show communities |
| `topic_cooccurrence_corpus` | **Stale artifact served as live** | Listed by `/api/corpus/enrichments` (1.5 MB) but absent from `6b7159ef`'s `per_enricher`. It is the 2026-08-10 output; nothing in the API says so |

Both `insight_*` enrichers expose a field named `total_insights` that counts different
things (470 vs 497 over the same episodes). Correct, but it invites a false comparison.

`guest_coappearance` was ENABLE'd on Gate B, which passed on the **top 50 of 847** persons.
The full artifact contradicts the inference drawn from that sample: it rediscovers "who hosts
what" (Roose+Newton, Martin+Armstrong, Barbaro+Kitroeff, Guo+Gil) and surfaces no cross-show
structure at all. `person:brandon` — a bare first name — is a live person id.

`temporal_velocity` also carries the unfixed granularity problem: **5,621 of 5,918 topics
(95 %) appear in exactly one episode**, which is what inflates the artifact to 43 MB on disk
/ 23.5 MB per GET.

### The defect the data exposed — insight attribution loss

Across 48 sampled episodes (every 14th of 678, newest-first), GI produced **645 Insight
nodes**; **148 (22.9 %) were dropped as `surfaceable: false`**, 132 of them carrying
`speaker_voice_type: "unidentified"`. **Nine of 48 episodes (18.75 %) have insights but zero
surfaceable ones** — they contribute nothing to any surface.

The filter is correct and deliberate (`_loaders.py:95-113`): an unattributed stance has
nobody holding it. The failure is upstream. On the Latent Space episode GI extracted **29
insights and named Person nodes (Sarah Sachs, Simon Last)**, then attributed all 29 to
`SPEAKER_00 / unidentified`. The people were found; the insights were never bound to them.
On the a16z episode the only Person node is `person:andreessen-horowitz` — an organisation,
the same org-as-person defect S6 flagged.

**It is per-run damage, not a feed property and not a permanent defect.** The same feed is
clean on one day and lossy on another:

| Feed | 08-05 | 08-11 | 08-12 | 08-13 |
| --- | --- | --- | --- | --- |
| The Daily | 0 % | 0 % | **20 %** | 0 % |
| No Priors | 3 % | 0 % | **22.6 %** | 0 % |
| Hard Fork | 0 % | — | **13.5 %** | 0 % |
| NVIDIA AI Podcast | — | — | **66.7 %** | 0 % |
| Unhedged | — | 0 % | 0 % | **70 %** |
| Planet Money | — | — | 20 % | **57.1 %** |
| Latent Space | — | — | — | **100 %** |
| Lenny's / Invest Like the Best | — | — | 0 % *(ILTB)* | 0 % *(Lenny's)* |

Corpus-wide by ingest day: 08-05 **1.7 %**, 08-11 **0 %**, 08-12 **20.0 %**, 08-13 **35.7 %**.
Some runs on the bad days are clean, so the unit of damage is the **run**, not the day.

**Leading hypothesis, not yet verified:** speaker detection degraded softly during the
08-12/08-13 bulk ingestion. Two jobs failed loudly in that window — `97180b23` (08-12,
`RuntimeError: cannot schedule new futures after interpreter shutdown`) and `8645ecd0`
(08-13, `ProviderRuntimeError … OpenAI speaker detection failed: no budget/credit left on
this key`) — and the episodes that *succeeded* alongside them may carry the silent version of
the same failure. If so this is a T7-shaped bug: the run reports success while the cause goes
unreported.

### Also observed

- **The Pragmatic Engineer produced 2 insights across 2 episodes** where Hard Fork averages
  20/episode. A different failure, undiagnosed — and that is the feed we just ingested 16
  more of.
- `GET /api/corpus/episodes` caps its page at 200 rows regardless of `limit` (declared
  `le=1000`, `corpus_library.py:327`); pagination is by `next_cursor`, and `offset` is
  silently ignored.

### Not covered by this verdict

- **Semantic correctness** — untouched. Counts and linkage only; never whether an insight is
  true or a sentiment label matches its text. Still F1qa.
- **Sample is 48 of 678**, systematic (every 14th, newest-first), not random. Per-feed and
  per-day cells are 1–6 episodes; the percentages above are indicative, not estimates.
  Latent Space's 100 % rests on **2 episodes**.
- **Why attribution fails per run** — the hypothesis above is unverified. The speaker
  detection and diarization stages have not been read.
- **`guest_coappearance` correctness** — its *value* was judged, not whether its pair
  counting is right.
- The 6 unexplained job failures remain unread.
- Corpus-wide surfaceable ratio; the effect of the 23.5 MB payload on any client.

---

## Revised priorities (2026-08-14) — `[SUPERSEDED by #1657]`

> This section was the first cut at re-ordering the work. It has been **replaced by the nine
> slices in [#1657](https://github.com/chipi/podcast_scraper/issues/1657)**, which carry the
> full homework inventory, the definition of done, and the deploy/repair sequence. The phases
> below are kept because their *reasoning* is still the reasoning — P0.1 is what found #1646 —
> but the slices in the epic are what gets executed.

**The reordering principle** (unchanged, and now the epic's): an enrichment layer computing
correctly over inputs that are 23 % destroyed is not a working system. Attribution outranks
everything, including S1.

**What P0.1 found:** `_check_episode_size_skip` (`workflow/stages/processing.py:717-765`)
skips speaker detection for any episode whose audio exceeds 25 MB — an OpenAI Whisper upload
limit, applied while transcribing with Deepgram, to a stage that reads only the episode title
and description. 488 of 678 episodes (72 %). Full analysis in #1646.

### Phase 0 — diagnose before scoping *(nothing below is scopeable until this lands)*

| | Item | Output |
| --- | --- | --- |
| P0.1 | **Diagnose the attribution loss at cause.** Read the speaker-detection / diarization → GI binding path. Answer: why does a run with 4 clean `SPEAKER_NN` labels and named Person nodes bind zero insights to a person? | A named cause, not a correlation |
| P0.2 | **Enumerate the blast radius.** Every run and episode in the corpus with `surfaceable` loss above the ≤2 % baseline. Not a sample — the full 678 | A re-derivation work-list |
| P0.3 | **Read the 6 unexplained job failures** (`/api/jobs/{id}/log`). Confirm or kill the "soft degradation alongside the loud failures" hypothesis | Criterion 4 closed |
| P0.4 | **Freeze today's numbers as a checked-in baseline** — per-feed/per-run surfaceable ratio, the four enrichers' record counts, the 48-episode sample | The comparison the Phase 2 re-run is measured against |

P0.4 is not bookkeeping. Without a baseline captured **before** any fix, the validation
re-run has nothing to compare to and "it looks better" becomes the acceptance test.

### Phase 1 — fix, each with unit + integration + client coverage

| | Item | Why here |
| --- | --- | --- |
| P1.1 | **The attribution fix** — scope set by P0.1 | The defect that destroys 23 % of the corpus's insights |
| P1.2 | **Loud empty-enricher-set** (exit criterion 2) | Small, and it is what makes every later "the fix worked" claim verifiable |
| P1.3 | **S1 — the profile royal route** | Now that criterion 1 is met by workaround, this is correctness-of-route, not availability-of-data |
| P1.4 | **D6 — budget headroom signal** | Promoted to Phase 1: a key running dry is the leading suspect for P0.1, and Phase 2 spends LLM budget re-deriving. Fixing without it risks reproducing the defect during validation |
| P1.5 | **`temporal_velocity`: suppress the scalar, expose `content_series`** | Ship-blocking for anything consuming "trending" |
| P1.6 | **`guest_coappearance`: revisit ENABLE** | Decide on the full artifact, not the top-50 sample |
| P1.7 | **Stale-artifact honesty** — `/api/corpus/enrichments` must distinguish "produced by the last run" from "left over from 2026-08-10" | Silent staleness is how the 3 ms no-op stayed invisible for a month |

"Client" coverage means the consumer surface, not just the API: an enricher whose output no
client reads is not validated by a green test.

### Phase 2 — validate by re-running

| | Item |
| --- | --- |
| P2.1 | Re-derive attribution for the P0.2 work-list (transcripts are retained; no audio needed) |
| P2.2 | **Full-corpus enrichment re-run** over all 678, compared against the P0.4 baseline |
| P2.3 | Only then: resume ingestion — re-queue the six cancelled windows verbatim |

### Phase 3 — deferred until Phases 0–2 close

H6 (incrementality — but see the design note below), H3 (post-ingest passes),
H7 (auto-spawn), the 43 MB artifact size, the 200-row page cap, H1 (still operator-blocked).

H6 is **downgraded on evidence**: the doc warned about a 30 s `wait_for` cap, but the real
678-episode pass took **6.9 s**. It is waste, not a timeout risk.

### Design note — incrementality must not block re-validation

Enrichment should be incremental *and* must keep a full-corpus re-run. Two constraints that
are easy to get wrong:

1. **The staleness key must include the input, not just the enricher.** `envelope.py` already
   persists `computed_at`, `enricher_version` and `schema_version`. Keying on those alone is a
   trap: fix speaker attribution upstream, re-run enrichment, and every episode is "unchanged
   at the same enricher version" — all 678 skipped, the fix invisible. The key must be
   `f(input GI identity, enricher_version, schema_version)` so an upstream fix invalidates
   downstream work **without anyone remembering to force it**.
2. **`--force` / `--all` is the override, not the mechanism.** If correctness depends on an
   operator passing a flag, it will be wrong the first time someone forgets. The flag exists
   for Phase 2 and for re-running at an unchanged version — not to compensate for a key that
   does not notice its inputs changed.

---

## Log

- **07:00Z** — six new-feed windows cancelled; phase opened; advisor engaged on S1.
- **07:30Z** — enrichment job `6d9d9c6f` cancelled so it could not auto-fire before fixes.
- **08:00Z** — leverage decision recorded; Gates A and B executed, both PASS.
- **08:18Z** — enrichment job `d08408f0` queued explicitly (the auto-chain does not fire, H7).
- **10:16–10:17Z** — `d08408f0` ran unattended, exit 0. First enrichment data in the corpus's
  life: `insight_sentiment` 6840 records, `insight_density` 6187, `temporal_velocity` 5918,
  `guest_coappearance` 277. All four `circuit_state: closed`.
- **10:29Z** — S3 filed; session ended; machine restarted.
- **~12:00Z** — corpus confirmed at **678 episodes** (the two Pragmatic Engineer tail windows
  landed), `with_both=678`, `with_neither=0`. Queue idle: 71 succeeded / 15 cancelled /
  7 failed, nothing running or queued.
- **~13:00Z** — S7 verdict filed. Attribution loss found (22.9 % of GI insights dropped,
  per-run not per-feed). Phase re-scoped; exit criterion 5 added; priorities reordered into
  Phases 0–3.
