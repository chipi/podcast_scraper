# Incremental rollout — consolidated follow-ups (F1–F15)

**Status:** living document — updated as feeds and episodes are onboarded.
**Reconstructed locally 2026-08-12** on the homelab from the operator's handover digest.
The original lives in the prod repo (`ai-ml-improvements` worktree); if the two diverge,
the prod copy is the source of record for F1–F15 and this copy is the source of record
for the rollout log in the final section.

Legend: `[OPEN]` needs work · `[DONE]` resolved · `[INFO]` by-design / no action.

---

## A. Provenance / metadata labels

### A1 (= F2 + F12) — Deepgram output mislabeled as Whisper — `[OPEN, medium]`

Episode metadata records `transcript_source="whisper_transcription"` and
`whisper_model="nova-3"` even when **Deepgram** did the transcription.

Metadata-only; no viewer or API consumer reads these fields. Harms:

1. Misleads provenance audits — already caused one false "wrong profile" scare.
2. Latent: only bites in a mixed Whisper + Deepgram corpus, where you can't target a
   single engine for reprocessing.
3. Analytics misattribution.

Does **not** affect UI, extraction quality, routing, or cost.

**Fix:** provider-neutral field names, or populate from `config_snapshot.ml_providers`
(which already holds `"deepgram"`). Note `transcript_source` is a `Literal` enum, so a
true rename touches both the schema and every reader.

### A2 (= F13) — `key_quotes` / `named_entities` always empty — `[OPEN, low]`

`summary.key_quotes` and `summary.named_entities` are always `0`. The real quotes and
entities live in the search index (`doc_type` `quote` / `kg_entity`) and the KG artifact,
not the summary block.

**Decide:** remove the dead sub-fields, or populate them.

---

## B. Discovery / add semantics

### B1 (= F4 + F15) — offset counts RSS position, not corpus absence — `[OPEN docs / INFO]`

`episode_offset` counts position in the RSS feed, which is **not** the same as "not in the
corpus". Separately, the discovery script's corpus-guid set (read from
`search/metadata.json`) lags the live corpus. So a "contiguous-N-new" window sometimes
includes one already-present episode and the feed adds N-1.

Observed: WSJ / NVIDIA +4 of 5, Flightcast fewer.

`skip_existing` (corpus-wide, D7) is the exact guard — no duplicate, no error. **Benign.**

**Optional fix:** re-read corpus guids immediately before each feed, or widen the window
and cap on adds. Either way, document the offset semantics.

### B2 — `[WITHDRAWN 2026-08-12]` "Latent Space feed serves only 2 items" — measurement error

**Raised and disproved within the same session. Recorded so nobody re-raises it.**

The claim was that the Flightcast/Latent Space feed served only 2 `<item>` elements and was
therefore permanently capped at 13 episodes. **This was false.** The feed serves **219
items** and is healthy.

**Cause of the error:** discovery counted items with `grep -c "<item>"`, which counts matching
**lines**, not occurrences. The Latent Space feed is 13.2 MB of XML minified onto very few
lines, so it reported 2. The other eight feeds serve multi-line XML where line count happens
to equal occurrence count, so they were unaffected — which made a tooling bug look like a
feed-specific defect. Correct form: `grep -o "<item>" | wc -l`.

**Lesson worth keeping:** when one item in a survey looks catastrophically different from the
others, suspect the measurement before the subject. The "dead feed" story was coherent,
matched an existing known issue (B1's Flightcast note), and was entirely wrong.

**Latent Space's `+0` in the +50 batch** therefore reverts to the original explanation:
`episode_offset=0` targeted the newest 10 episodes, which were already among its 13, and
`skip_existing` skipped them. It can reach any reasonable target using a real offset.

---

## C. Outcome / index signals (cosmetic)

### C1 (= F1) — skips tally as `failed`, not `skipped` — `[OPEN, low but affects EXIT gates]`

A clean $0 all-skip run reports `{failed: 1}`, which fails the Step-0/Step-1 EXIT criteria.
No functional harm. The skip-existing path (including the D7 corpus-layout branch) never
sets `status="skipped"` — only the exception path does
(`episode_processor.py:1655` / `:2644`).

**Fix:** set `status="skipped"` on both skip branches, and add a test.

> **Operational note (2026-08-12):** this mislabel does **not** propagate to job-level
> status. The WSJ re-run below was a complete all-skip and still returned
> `status=succeeded`. Watchers keyed on job status are safe; only per-episode outcome
> tallies are affected.

### C2 (= F3) — `reindex_recommended: true` after an all-skip reindex — `[OPEN, low]`

Run-summary mtime advances even though the index didn't change.

**Fix:** base staleness on the episode / fingerprint **set**, not mtime.

### C3 (= F6) — first post-D8 reindex re-embeds the whole corpus — `[INFO, by-design]`

The fingerprints file didn't exist yet, so the first reindex after the D8 deploy re-embeds
everything once; every reindex after that skips unchanged episodes. Not a bug — note it in
ops so a post-deploy ~10 min reindex isn't misread as a regression.

---

## D. LiteLLM prod gateway (Option-A / ADR-142 follow-ups)

### D1 (= F7) — podcast-prod key had no budget cap — `[DONE]`

Set `max_budget=$25` via `/key/update` (verified).

**Residual code TODO — still open:** `deploy-litellm.yml:213` `/key/generate` should mint
the key *with* a budget so a fresh box gets one automatically, rather than relying on a
manual `/key/update` after the fact.

### D2 (= F8) — orphan gateway key — `[OPEN, hygiene]`

`proj-podcast-prod=b404b602` on the prod gateway is unused. Delete via `/key/delete` once
settled.

### D3 (= F9) — ops-card reads prod spend from the wrong gateway — `[OPEN, medium]`

`server/routes/llm_gateway.py` reads prod LLM spend `{box="prod"}` scraped **from
homelab**. Now that prod has its own gateway, point the card at the local gateway's
metrics.

### D4 (= F10) — `litellm_api_base` override is not durable — `[OPEN, medium]`

The override lives **only** in the box's `viewer_operator.yaml`. It survives deploys, but a
DR or volume rebuild loses it and silently reverts to the homelab gateway.

**Fix:** document in the DR runbook, or drive it from a deploy-managed env
(`config.py:2097`).

### D5 (= F11) — provisioning idempotency gap — `[OPEN, medium]`

The prod gateway never had the app key until this session; the app "worked" only by way of
homelab. Add a `deploy-prod` post-check asserting the pipeline's key authenticates against
the **configured** `litellm_api_base`.

---

## E. Deploy

### E1 (= F5) — `deploy-all-prod.yml` unvalidated — `[OPEN, low]`

The one-trigger orchestrator has never been dispatched. Needs
`secrets.DEPLOY_ORCHESTRATOR_PAT` plus one live run. Until then, deploy via the three
individual workflows.

---

## F. Quality QA

### F1qa (= F14) — no semantic-correctness audit — `[OPEN, optional]`

Quality has been assessed on structure and samples (strong), but there is no
insight-by-insight grounding check against the transcript.

**If a grounding gate is wanted:** spot-check N insights per episode, or use the
insight-node confidence scores.

---

## Status summary

15 items → **13 distinct** (F2+F12 merged, F4+F15 merged). B2 was raised 2026-08-12 and
**withdrawn the same session** as a measurement error — it is not open work.

| State | Count | Items |
| --- | --- | --- |
| `[DONE]` | 1 | D1 (residual code TODO still open) |
| `[INFO]` | 2 | C3, part of B1 |
| `[WITHDRAWN]` | 1 | B2 |
| `[OPEN]` | 10 | A1, A2, B1(docs), C1, C2, D2, D3, D4, D5, E1, F1qa |

**Highest-value OPEN:** D3 / D4 / D5 (gateway durability + correctness), A1 (label bug),
C1 (skip mislabel breaks EXIT gates).

### Found during the 2026-08-12 homelab session, not in the original F1–F15

| Item | Where | Nature |
| --- | --- | --- |
| **B1 evidence** | Rollout log | B1's under-count did **not** reproduce 5/5 when discovery ran per-feed; suggests the fix target and that the pending WSJ/NVIDIA +4 investigation can close |
| **Cost blindness** | Rollout log | `prod:4001` unreachable from `tag:homelab-host` — cost cannot be measured from the box that now runs ingestion. Companion to D3/D4 |
| **Enricher artifacts** | Rollout log | Only `topic_cooccurrence_corpus` appears under `/api/corpus/enrichments`; the enrichers the corpus growth is meant to feed are unaccounted for |
| **ACL grant** | Rollout log | `tag:homelab-host → tag:prod:443` added and applied; README corrected |
| **B2 (withdrawn)** | §B | Kept as a record of a disproved claim, not as work |

---

## Rollout log — Step-2 volume batch (+10/feed → ~250)

Run from the **homelab** against the prod operator API. Add-only; `skip_existing=true`
throughout, so every entry is idempotent and re-runnable.

### Infrastructure change made to enable this batch

**2026-08-12 — tailnet ACL: `tag:homelab-host` → `tag:prod:443`.**

Homelab previously held a single grant to prod, `:8099` (the ADR-145 delivery-worker outbox
seam). Every operator-API call on `:443` was dropped by the tailnet packet filter — all
ports timed out rather than refusing, including `:22`, the documented signature of "no
incoming grant" (`policy.hujson:194`).

Diagnosis evidence: `:8099` answered (HTTP 404 — TCP connects) while `:443`, `:80`, `:22`
and `:8000` all returned `curl (28) Operation timed out`, from `100.87.33.61` to
`100.124.111.115`.

Change: `"dst": ["tag:prod:8099"]` → `["tag:prod:443,8099"]`, committed as `fd337a17` and
pushed directly to `main` (branch protection bypassed, at operator direction), which
triggered `.github/workflows/tailscale-acl.yml` in `apply` mode (run `31565516573`,
success). Verified afterwards by the live path, not the workflow status:
`corpus_stats_http=200`.

Rationale for making it durable rather than running from an admin device: the operator
confirmed homelab will drive ingestion on a standing basis over the coming months.

Also corrected in the same commit: `tailscale/README.md` documented the pre-ADR-128
`tofu apply` flow, which has applied nothing since 2026-07-28. Following it leads an agent
to edit the policy, run `tofu apply`, and see no change on the live tailnet.

### Feed results

Baseline at batch start: **180 episodes / 9 feeds.**

Handover feed IDs resolved to real show names via `GET /api/corpus/feeds` (2026-08-12):
`Acast-6478` = **Unhedged**, `Simplecast-l2i9` = **Hard Fork**, `Flightcast` = **Latent
Space**, `NPR` = **Planet Money**.

| # | Show (handover ID) | Offset | Job ID | Status | Corpus | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | The Journal. (WSJ-Journal) | 23 | `5758bed6-0f06-4032-9fba-ba503bdee8e1` | succeeded | 180 → 180 | +0 |
| 2 | NVIDIA AI Podcast (NVIDIA-AI) | 17 | `cb45fa31-33d8-4b77-905b-da9e5d24d851` | succeeded | 180 → 190 | **+10** |
| 3 | Invest Like the Best | 19 | `e9a3ae29-0cde-46c0-9869-4af20dc0e603` | succeeded | 190 → 200 | **+10** |
| 4 | Unhedged (Acast-6478) | 21 | `5d20b36c-a51a-4607-a3b1-4d2770213e79` | succeeded | 200 → 210 | **+10** |
| 5 | The Daily | 25 | `5f1b34a0-5bd3-4705-b06f-76c41ce6750f` | succeeded | 210 → 220 | **+10** |
| 6 | Hard Fork (Simplecast-l2i9) | 19 | `15412156-0ec3-400b-86b8-a2f17b55d150` | succeeded | 220 → 230 | **+10** |
| 7 | Latent Space (Flightcast) | 0 | `f363523d-069b-4697-8587-8d9c9b9b1077` | succeeded | 230 → 230 | +0 |

**BATCH COMPLETE — 180 → 230 (+50). 7/7 succeeded, 0 failed, 0 stale, 0 cancelled.**

### Final per-feed state (verified via `GET /api/corpus/feeds`)

| Show | Before | After | Δ |
| --- | --- | --- | --- |
| No Priors | 27 | 27 | — (not in this batch) |
| Planet Money | 28 | 28 | — (not in this batch) |
| The Journal. | 26 | 26 | +0 (batch-2 already landed) |
| NVIDIA AI Podcast | 16 | 26 | **+10** |
| Invest Like the Best | 15 | 25 | **+10** |
| Unhedged | 17 | 27 | **+10** |
| The Daily | 21 | 31 | **+10** |
| Hard Fork | 17 | 27 | **+10** |
| Latent Space | 13 | 13 | +0 |
| **Total** | **180** | **230** | **+50** |

Arithmetic checks out against `catalog_episode_count` at both ends.

**Expected landing zone: ~235–245**, not 250 — baseline 180 plus six productive feeds at
+8/+10 each (see B1). WSJ contributes ~0 because its batch-2 had already landed before this
session picked up.

### Observations from this batch

- **WSJ +0 is correct, not a fault.** Batch-2 for WSJ completed before handoff; the corpus
  was already 180 when this session started, and `skip_existing` skipped all ten. This is
  also the datapoint showing C1's tally mislabel doesn't reach job-level status.
- **The B1 under-count did not reproduce once — 5 for 5.** This is the most actionable finding
  of the batch. B1 records NVIDIA at +4 of 5 in batch-1, and predicts feeds landing +8/+9
  rather than +10. In this batch **every productive feed delivered a full +10**: NVIDIA,
  Invest Like the Best, Unhedged, The Daily, Hard Fork.

  **Hypothesis (not proven):** batch-1 ran discovery **once up front for all feeds**, so the
  corpus-guid set went stale as earlier feeds added episodes — later feeds in the same batch
  then computed windows against an out-of-date set and short-added. This batch triggered each
  feed **separately, after the previous one finished**, so discovery was effectively fresh
  every time.

  If that holds, it has two consequences worth acting on:
  1. B1's proposed fix ("re-read corpus guids right before each feed") targets the **real
     cause**, and is worth building.
  2. The short adds in batch-1 were never a feed-specific property — so no per-feed
     investigation is needed for the WSJ/NVIDIA +4 under-counts listed as pending work.
     **That pending item can likely be closed.**

  **Not proven:** this is 5 observations from a batch whose sequencing differed from batch-1
  in exactly the suspected variable, but nobody has read the discovery code to confirm the
  mechanism. Someone with SSH should verify before closing the item.

- **Latent Space +0 was predicted and is correct.** It ran at `episode_offset=0` — the only
  feed not given a real offset — which targets the *newest* 10 episodes. It already held 13,
  so the whole window was already present and `skip_existing` skipped it. The broken
  feed-guid mapping B1 flags did not need to be invoked to explain this. **To actually grow
  Latent Space, it needs a non-zero offset** (its back-catalog), not a retry at 0.
- **Possible over-add in the pre-handoff WSJ run — unresolved.** The handover digest has
  NPR ending at 169 with WSJ then in-flight; the corpus measured 180 before this session's
  WSJ job, implying the original run went 169 → 180 = **+11**, above its own
  `max_episodes=10`. Two readings, not distinguishable without SSH: the digest's
  intermediate numbers are approximate, or something added beyond the cap. Noted because it
  runs *opposite* to the B1 under-count, which is the direction everything else drifts.
  Not treated as blocking.

### Pass-3 — RSS discovery, and the Latent Space cap (2026-08-12)

Target: ~500 episodes. Discovery run **from the homelab without SSH**, by fetching each RSS
directly and counting `<item>` elements — this is a viable substitute for `discover10.py` and
needs no box access.

| Feed | In corpus | RSS items available |
| --- | --- | --- |
| The Daily | 31 | 2944 |
| Invest Like the Best | 25 | 592 |
| Planet Money | 28 | 355 |
| Unhedged | 27 | 330 |
| NVIDIA AI Podcast | 26 | 306 |
| The Journal. | 26 | 300 |
| Hard Fork | 27 | 209 |
| No Priors | 27 | 173 |
| Latent Space | 13 | 219 |

**Every feed has ample back-catalog.** The smallest (No Priors, 173) still supports more than
triple the target.

> **Correction:** the first run of this table reported Latent Space at **2** items and
> concluded the feed was dead. That was a measurement error — see the withdrawn **B2**. The
> count above (219) is correct. Counting must use `grep -o "<item>" | wc -l`; `grep -c`
> counts lines and undercounts minified feeds catastrophically.

**Target: ~500 across all 9 feeds**, i.e. **55 per feed**. Invest Like the Best was already
launched at +36 (to 61) before the correction landed and is left to overshoot slightly —
harmless.

| Feed | Now | Target | Deficit | Offset |
| --- | --- | --- | --- | --- |
| Invest Like the Best | 25 | 61 | +36 | 25 (running) |
| Latent Space | 13 | 55 | +42 | 13 |
| The Journal. | 26 | 55 | +29 | 26 |
| NVIDIA AI Podcast | 26 | 55 | +29 | 26 |
| No Priors | 27 | 55 | +28 | 27 |
| Unhedged | 27 | 55 | +28 | 27 |
| Hard Fork | 27 | 55 | +28 | 27 |
| Planet Money | 28 | 55 | +27 | 28 |
| The Daily | 31 | 55 | +24 | 31 |
| **Total** | **230** | | **+271** | **= 501** |

Run as **one job per feed** covering its full deficit, rather than 27 separate 10-windows —
fewer jobs to supervise, and `skip_existing` keeps a retry cheap if one fails.

**Throughput — plan for ~18 hours, not an afternoon.** Invest Like the Best added roughly 15
episodes in its first hour, i.e. **~4 min/episode**, putting the full +271 push on the order of
**18 hours**. The +50 batch took ~3 hours; this one is 5.4× the volume. Rough estimate,
extrapolated from one feed's partial progress — expect variance, since episode length drives
transcription time and The Daily (~20 min/episode) should run far faster than Invest Like the
Best (~90 min/episode).

**Operational note:** a job of this size outlives a 1-hour monitor. Use a persistent watcher,
or the job will silently outrun its supervision and look like it stalled.

### Quality assessment after the +50 batch (2026-08-12)

Requested check on whether quality degraded. **No degradation detected.**

Verified:

- **Coverage is complete:** `total_episodes: 230, with_gi: 230, with_kg: 230, with_both: 230,
  with_neither: 0`.
- **Index agrees:** `episode_title: 230`, `summary_short: 230`, 24 961 vectors.
- **Per-episode density sits inside the batch-1 baseline** — insights 17.6/ep (baseline 6–31),
  KG nodes 22.7/ep (baseline ~20–29), quotes 26/ep.
- **Spot-check** of a newly-ingested The Daily episode: thesis-level `summary_title`,
  substantive bullets carrying named entities and hard figures, and
  `bridge_partition: {gi_only: 0, kg_only: 13, both: 13, total: 26}` — every GI node has a KG
  counterpart.
- **Unplanned benefit:** the batch extended the corpus *backward* — a new month appeared
  (2025-11, 6 episodes) and Feb–May thickened. Useful for `temporal_velocity`.

NOT verified:

- **The new 50 cannot be isolated** from corpus-wide averages — no per-episode stats endpoint
  and no pre-batch index snapshot. In-band averages are evidence, not proof.
- **No semantic grounding check** (F1qa / F14 still open).
- **Only one corpus-level enricher artifact exists:** `topic_cooccurrence_corpus` v1.1.0.
  `topic_perspectives`, disagreement, `guest_coappearance`, `temporal_velocity` and
  `topic_similarity` do **not** appear under `/api/corpus/enrichments`. They may be
  per-episode sidecars rather than corpus artifacts — undetermined from this endpoint.
  **Worth resolving**, since the entire expansion rationale is feeding those enrichers.
- **Cost is unobtainable from the homelab.** `prod:4001` returns `000` (dropped — the ACL
  grants `tag:prod:4001` to `autogroup:admin` only, not `tag:homelab-host`) and
  `/api/corpus/runs` is `404`. Credentials do not help; the packets never arrive. Fixing it
  means adding `4001` to the homelab grant, exactly as `443` was added above.

### Offsets used

The handover's **fixed** offsets, not live-derived ones — `discover10.py` needs SSH, which
this session does not have. `skip_existing` absorbs the drift; per B1 this is exactly why
some feeds land +8/+9 rather than +10.

### Stop criteria in force

- `status=failed` → stop, report `job_id` + `error_reason`.
- A `+0` from a feed that should produce a delta → stop and report. That indicates a
  wholly stale offset window rather than a pipeline fault, and it's worth a decision rather
  than burning ~30 min per no-op job.
- `+8` / `+9`, or a thin Flightcast, are **not** anomalies (B1). Keep going.
