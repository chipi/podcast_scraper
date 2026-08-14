# Stabilization phase — 2026-08-14

**Status:** OPEN — ingestion deliberately paused.

A hold on corpus growth to fix what we found while growing it. Ingestion resumes only
when the exit criteria below are met.

The rationale, stated precisely: **enrichment is corpus-wide and re-runnable, so its
correctness does not depend on when it runs. Per-episode artifacts are not.** Summaries,
insights and KG nodes are minted at ingest time by whatever the pipeline is that day. If
a per-episode defect exists, every episode ingested before it is found carries it. So the
pause protects the per-episode layer, not the enrichment layer.

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

1. **Enrichment produces data on the real corpus.** A named, non-empty enricher set runs,
   and `GET /api/corpus/enrichments` shows artifacts for it. (H2)
2. **An empty enricher set cannot silently succeed.** The no-op path is loud — the failure
   mode that hid this for the corpus's entire life is closed, with a test.
3. **The per-episode layer has been audited** on a sample spanning several feeds: summary
   substance, insight/KG density, GI↔KG bridging, and ad/sponsor contamination. Findings
   filed in the homework doc.
4. **No unexplained job failure** in the current queue. `failed_total` is understood
   line-by-line, not just counted.

Explicitly **not** exit criteria: H1 (audio archive) is blocked on operator-supplied
secrets and must not gate the phase; H4 is recorded-only by operator decision.

---

## Work items

### S1 — fix enrichment the royal route — `[IN PROGRESS]`

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

### S2 — verify enrichment on the real corpus — `[BLOCKED on S1]`

The queued `6d9d9c6f` (`--only <7 deterministic>`) is a *diagnostic*, not the fix: it
bypasses profile resolution by naming enrichers explicitly. Its value is proving the
executor and the enrichers themselves work. Do not read a green result from it as evidence
that S1 is fixed — those are different code paths.

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

### S4 — homework items worth doing in this window — `[TODO]`

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
- Whether `insight_sentiment`'s sidecars exist on disk. `enrichments_available` lists only
  `insight_density`, which reads as a reporting-layer gap rather than an enricher failure —
  untraced either way.
- Corpus-wide named-person ratio (see Gate B caveat).

---

## Log

- **07:00Z** — six new-feed windows cancelled; phase opened; advisor engaged on S1.
- **07:30Z** — enrichment job `6d9d9c6f` cancelled so it could not auto-fire before fixes.
- **08:00Z** — leverage decision recorded; Gates A and B executed, both PASS.
