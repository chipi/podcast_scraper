# HANDOVER — ingest the 10 remaining Batch A feeds, 10 episodes each

**Date:** 2026-08-29 · **Plan of record:** [ONBOARDING-SHOWS-FOR-ENRICHER-VALUE.md](ONBOARDING-SHOWS-FOR-ENRICHER-VALUE.md)
§5f (list) / §5g (protocol) / §5i (thresholds) / §5j (current state) · **Expansion vehicle:** `#630`

> ## STATUS 2026-09-02 — this job is BEING RUN NOW. The document is live, not superseded.
>
> **The unit of work is 10 episodes × 10 feeds = 100 episodes.** It is *not* "add ten feeds" —
> a feed appearing in the corpus does not mean this job is done, and an earlier version of this
> banner made exactly that mistake and told readers not to re-run. Wrong: a 100-episode pass is
> repeatable and is being repeated.
>
> **In flight:** ten jobs queued 2026-09-01 22:41, `max_episodes=10` each, serial. Feed 1
> (Conversations with Tyler) `succeeded`; In Our Time running since 00:52:56Z; 8 queued. Zero
> errors, all costs $0.00 (DGX Whisper + vLLM).
>
> **Passes so far**, from the jobs registry — this has been run more than once:
>
> | Launched | Shape |
> | --- | --- |
> | 2026-08-14 06:32 | 6 jobs, max=15/25 — all cancelled |
> | 2026-08-30 00:07 → 06:45 | ~13 × `max=1` — the §5g Phase-1 smoke |
> | 2026-08-30 08:33 | **10 × max=10 — first full 100-episode pass** |
> | 2026-08-30 19:28 → 21:20 | ~30 × `max=1` — debugging the silent failures fixed in `e8c6f35e` |
> | 2026-08-31 08:55 | 10 × max=10 — 8 cancelled, 2 succeeded |
> | 2026-08-31 09:05 | **10 × max=10 — second full pass** |
> | 2026-09-01 22:41 | **10 × max=10 — third full pass, IN FLIGHT** |
>
> **Consequence worth an operator's eye:** because these runs use `episode_selection=unprocessed`,
> each pass takes 10 *new* episodes rather than re-taking the same newest 10. So the Batch A feeds
> stand at **20 episodes each** and this pass takes them to **30** — roughly 300 episodes from the
> ten feeds, not 100. Each individual pass is exactly the intended 100 episodes of work; the
> accumulation is across passes. Whether to stop at 30/feed is a depth decision (§5g Phase 3), not
> something this document should assume.
>
> **Two instructions below are WRONG as written.** They are corrected in place and marked
> ✗ WRONG / ✓ CORRECT — the reasoning is kept because both mistakes are easy to repeat:
>
> 1. **Step 1's `PUT /api/feeds` merge is unnecessary** for adding feeds you intend to ingest by
>    URL, and the whole-list-replace hazard it warns about is real but avoidable — see Step 1.
> 2. **`episode_order=newest` is not the control that matters.** The control is
>    **`episode_selection`**, a per-request parameter added 2026-09-01 (`998d5312`) that did not
>    exist when this was written — see Step 2.
>
> **The mechanics here are superseded by [docs/guides/INGESTION_RUNBOOK.md](../guides/INGESTION_RUNBOOK.md)**,
> which is the canonical reference for nightly-vs-backfill selection. What is still live in this
> document is the **assessment** half: the §5i gates in Step 3 and the §5g buckets in Step 4,
> which have not been applied to any of the ten feeds yet.

---

## The goal, and only the goal

Ingest **10 episodes** for each of the **10 feeds** below — 100 episodes of work — then measure
each against the §5i thresholds and write the verdict into the plan doc's §5j.

**This is a repeatable pass, not a one-off.** It has run three times (see the banner). Steps 1–2
below describe how; the corrections on them still apply. **Step 3–4 have never been done for any
Batch A feed** — no feed has a §5i grade or a §5g bucket — so the measurement half is the part
that keeps getting skipped.

Do **not** start Batch B — §5f gates it on Batch A being measured, and it is not measured yet. Do
**not** re-litigate the pre-existing feeds; §5g records that as an operator decision.

---

## Where the corpus stands

Verified live **2026-09-02** via `GET /api/corpus/feeds?path=/app/output`:

**24 feeds, 966 episodes** (rising as the pass runs).

| Cohort | Episodes each |
| --- | --- |
| The nine originals | 62–70 |
| Probe group 1 — a16z / Lenny's / Pragmatic Engineer | 51–71 |
| Probe group 1 — Dwarkesh / Ideas of India | 10 (still at probe depth) |
| **The ten Batch A feeds** | **20** (EconTalk 19, Ground Truths 21) → **30** as this pass lands |

The 20 is the residue of two prior 100-episode passes, not a single deep ingest — see the pass
table in the banner.

Carried over from 2026-08-29 and still true:

- **No feed trips any §5i gate.** All five probe shows are **DEEPEN** on the pipeline axis.
- The ad-contamination question is **settled**: 0 real hits in 24 episodes across the three
  heaviest ad-load feeds. Do not re-run it as a discovery test.

---

## The ten feeds

Every URL below was re-fetched on 2026-08-29 — all live, all publishing within the last two
weeks, item count equal to enclosure count (every entry carries audio). `#` is the §5f Batch A
row number. **All ten match §5f's 2026-08-13 verification plus two weeks of new episodes.**

| # | Show | Domain | Items | Newest | RSS |
| --- | --- | --- | --- | --- | --- |
| 4 | Conversations with Tyler | ideas/econ | 298 | 08-19 | `https://rss.libsyn.com/shows/137081/destinations/850607.xml` |
| 6 | In Our Time (BBC) | history/science/philosophy | 1105 | 08-27 | `https://podcasts.files.bbci.co.uk/b006qykl.rss` |
| 7 | The Rest Is History | history, dialogic | 718 | 08-26 | `https://feeds.megaphone.fm/GLT4787413333` |
| 8 | Empire: World History | history, dialogic | 403 | 08-26 | `https://feeds.megaphone.fm/empirepodcast` |
| 9 | ChinaTalk | geo: China, tech policy | 560 | 08-24 | `https://feeds.megaphone.fm/CHTAL4990341033` |
| 10 | Sinica Podcast | geo: China | 558 | 08-18 | `https://rss.art19.com/sinica` |
| 12 | Odd Lots | finance, dialogic | 1263 | 08-28 | `https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/8a94442e-5a74-4fa2-8b8d-ae27003a8d6b/982f5071-765c-403d-969d-ae27003a8d83/podcast.rss` |
| 13 | EconTalk | economics | 1064 | 08-24 | `https://feeds.simplecast.com/wgl4xEgL` |
| 14 | Ground Truths | biotech, medicine×AI | 94 | 08-23 | `https://api.substack.com/feed/podcast/587835/s/119690.rss` |
| 15 | The Long Run | biotech | 206 | 08-25 | `https://feeds.soundcloud.com/users/soundcloud:users:317770704/sounds.rss` |

### Nothing blocks queueing — but two notes

1. **Backfill caution (§5g).** In Our Time (1105), Odd Lots (1263) and EconTalk (1064) carry
   large, partly-dated archives. **The guard is `episode_selection`, not `episode_order`** — see
   the banner on Step 2. `episode_order` defaults to `newest` and needs no flag; what must never
   happen is `episode_selection: unprocessed` set corpus-wide, which turns the nightly into a
   back-catalog crawler.
2. **Count RSS items by occurrence, not by line.** An earlier draft of this handover flagged
   Ground Truths as a broken 1-episode feed and In Our Time as halved. Both were artifacts of
   `grep -c '<item'`, which counts matching *lines*: Substack serves its entire feed on one line
   with no newlines, so 94 episodes counted as 1. Use:

   ```bash
   curl -sSL "$RSS" > /tmp/f.xml
   grep -o '<item[ >]'   /tmp/f.xml | wc -l   # items
   grep -o '<enclosure ' /tmp/f.xml | wc -l   # should match
   ```

   Cross-check with `https://itunes.apple.com/search?term=<show>&entity=podcast` — its
   `trackCount` and `feedUrl` are independent of your fetch, and that is what settled it here
   (94, and the same URL §5f already had).

---

## Prerequisites

| What | Value |
| --- | --- |
| Prod host | `prod-podcast.tail6d0ed4.ts.net` (Tailscale; resolve with `scripts/ops/resolve_prod_tailnet_host.sh`) |
| Operator auth | header `X-Operator-Key: <key>`; the key is on this machine at `~/podcast_operator_api_key.txt` — **never paste its value into a doc, log, or commit** |
| Corpus path | `path=/app/output` (in-container root) on every call |

`GET /api/corpus/*` is open; `GET|PUT /api/feeds` and `POST /api/jobs` are operator-gated
(`src/podcast_scraper/server/app_operator_guard.py`).

---

## Step 1 — add the ten feeds to `feeds.spec.yaml` — ✗ never was a prerequisite

> Already done once (all ten are in `feeds.spec.yaml`; the corpus reads 24 feeds), and it is a
> one-time step — not part of each 100-episode pass.
>
> **✗ WRONG as a prerequisite.** A raw URL passed as `feed=` to `POST /api/jobs` is used verbatim
> without consulting the spec, so ingestion never needed this step — the note at the end of this
> section already said so and should have been the headline. The spec entry matters only for
> *whole-batch* runs (and for the nightly, which is exactly why the runbook's corpus-wide
> selection trap bites).
>
> **✓ The hazard below is real** and worth keeping: `PUT /api/feeds` replaces the whole list, so
> a naive "add my ten" call would have wiped the fourteen that were already there.

**`PUT /api/feeds` REPLACES THE WHOLE LIST. It does not append.** Sending only the ten new feeds
would wipe the existing fourteen and orphan 765 episodes from the batch spec. Read → merge →
write, and check the count both before and after.

```bash
B=https://prod-podcast.tail6d0ed4.ts.net
KEY=$(tr -d '\n\r' < ~/podcast_operator_api_key.txt)

# 1. snapshot the current spec (expect 14)
curl -fsS -H "X-Operator-Key: $KEY" "$B/api/feeds?path=/app/output" > /tmp/feeds-before.json
jq '.feeds | length' /tmp/feeds-before.json

# 2. merge: existing + the ten new URLs, order preserved, deduped server-side
jq '{feeds: (.feeds + [
  "https://rss.libsyn.com/shows/137081/destinations/850607.xml",
  "https://podcasts.files.bbci.co.uk/b006qykl.rss",
  "https://feeds.megaphone.fm/GLT4787413333",
  "https://feeds.megaphone.fm/empirepodcast",
  "https://feeds.megaphone.fm/CHTAL4990341033",
  "https://rss.art19.com/sinica",
  "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/8a94442e-5a74-4fa2-8b8d-ae27003a8d6b/982f5071-765c-403d-969d-ae27003a8d83/podcast.rss",
  "https://feeds.simplecast.com/wgl4xEgL",
  "https://api.substack.com/feed/podcast/587835/s/119690.rss",
  "https://feeds.soundcloud.com/users/soundcloud:users:317770704/sounds.rss"
])}' /tmp/feeds-before.json > /tmp/feeds-after.json
jq '.feeds | length' /tmp/feeds-after.json   # MUST be 24

# 3. write it back
curl -fsS -X PUT -H "X-Operator-Key: $KEY" -H 'Content-Type: application/json' \
  --data @/tmp/feeds-after.json "$B/api/feeds?path=/app/output" | jq '.feeds | length'
```

`PUT` is an atomic whole-file write (`atomic_write_text`), capped at 5000 entries, and dedupes on
URL. Keep `/tmp/feeds-before.json` until the batch is done — it is your rollback.

> A raw URL passed as `feed=` to `POST /api/jobs` is used **verbatim** without consulting the spec
> (`_resolve_feed_url`, `routes/jobs.py:96`), so ingestion would work without Step 1. Do Step 1
> anyway: a feed absent from `feeds.spec.yaml` is skipped by every future whole-batch run.

**New since 2026-08-28 (`#1872`)** — a spec entry may be an object with a per-feed `profile:`,
not just a URL string, and the named profile's registry + YAML layers are resolved *under* that
entry's own overrides (`rss/feeds_spec.py:101,238`). That is the mechanism for routing **one**
feed through a different deployment profile; a batch run otherwise applies one profile to every
feed. Plain URL strings (as used above) remain valid and inherit the corpus profile — start
there, and only pin a profile if a specific feed needs one.

---

## Step 2 — ingest, one feed at a time — ✗ THE SELECTION FLAG HERE IS THE WRONG ONE

> **✗ WRONG:** this section says to pass `episode_order=newest` and calls it the guard against
> walking the back-catalog of In Our Time (1105 items), Odd Lots (1263) and EconTalk (1064).
> **It is not that guard, and it was never needed:** `episode_order` already defaults to
> `newest` at every layer — `cli.py:634` (argparse default), `config.py:741` (field default), and
> no profile overrides it. Passing it changes nothing.
>
> **✓ CORRECT — the parameter that actually decides is `episode_selection`**, added per-request
> on 2026-09-01 (`998d5312`), after this document was written:
>
> - **positional** (the default, and what the nightly must keep) — `max_episodes` counts feed
>   POSITIONS, so `skip_existing` drops what is on disk and the back-catalog is unreachable *by
>   construction*.
> - **`unprocessed`** (what a deliberate backfill wants) — already-ingested episodes are dropped
>   by guid FIRST, so the cap counts **episodes of work**, not positions in a feed that moves.
>
> The 2026-09-01 run used `episode_selection=unprocessed` and its log states the effect exactly:
>
> ```
> episode_selection=unprocessed: 20 feed item(s) already ingested and dropped BEFORE the
>   limit; 278 candidate(s) remain. This is what makes --max-episodes mean 'N episodes of
>   work' rather than 'N positions in a feed that moves'.
> ```
>
> That is why the run ingested Conversations with Tyler feed positions **21–28**, not 1–10 — the
> newest 20 were already on disk. Correct behaviour, and it makes the archive worry moot: with
> `unprocessed` the limit applies after the guid filter, so a 1263-item feed still yields exactly
> 10 new episodes off the newest end.
>
> **⚠ Never set `unprocessed` corpus-wide in `viewer_operator.yaml`** — it converts every nightly
> into a back-catalog crawler. Set it per request. Never combine it with an offset. Both traps
> are written up in [INGESTION_RUNBOOK.md](../guides/INGESTION_RUNBOOK.md) §1, which is canonical
> for this and supersedes the command below.

**§5g says serial, one show at a time** — the pipeline runs one job at a time regardless, and
serial execution keeps each result attributable. Wait for `succeeded` before starting the next.

```bash
RSS="https://rss.libsyn.com/shows/137081/destinations/850607.xml"   # one at a time
curl -fsS -X POST -H "X-Operator-Key: $KEY" \
  --get --data-urlencode "path=/app/output" \
  --data-urlencode "feed=$RSS" \
  --data-urlencode "skip_existing=true" \
  --data-urlencode "max_episodes=10" \
  --data-urlencode "episode_selection=unprocessed" \
  "$B/api/jobs" | jq
# episode_order is omitted deliberately — it already defaults to newest.
# episode_selection=unprocessed is per-request; the nightly is unaffected.
# -> 202 {"job_id": "...", "status": "queued|running", "queue_position": n}
```

All parameters are **query** parameters, not a JSON body (`routes/jobs.py:166`). `skip_existing`,
`max_episodes`, `episode_offset` and `episode_order` apply **only** when `feed=` is given; without
it you get a whole-batch run over all 24 feeds, which is not what you want.

There is also an optional **`profile=`** query param (`#1872`, 2026-08-28) that overrides both the
feed's pinned profile and the corpus operator YAML for **this run only** — nothing is persisted.
Omit it unless a specific feed needs a different profile; the default path is what produced the
existing 765 episodes.

Watch it:

```bash
curl -fsS -H "X-Operator-Key: $KEY" "$B/api/jobs?path=/app/output" \
  | jq -r '.jobs[:5][] | "\(.job_id[0:8]) \(.status) \(.created_at)"'
curl -fsS -H "X-Operator-Key: $KEY" "$B/api/jobs/$JOB_ID/log" | tail -40
```

### Deviation from §5g Phase 1 — and why it is deliberate

§5g's protocol is **1 episode → assess → deepen**. The operator has asked for **10 straight**.
That is a considered change, not an oversight: the 1-episode phase existed to catch ad
contamination and cost blowups cheaply, and §5i + §5j have since settled the ad question (0 real
hits in 24 ad-heavy episodes) and the pipeline has run 765 episodes without a threshold failure.
The residual risk is **cost**, not quality — so watch spend rather than re-running a smoke phase.

**Per-run soft cap is $10** (§5g). At the §5g planning figure of ≲$0.30/episode, 10 episodes is
~$3/feed and ~$30 for the batch. **That figure has never been verified** — see "Not verified"
below. Check actual spend after the first feed completes and stop the batch if it is materially
above it.

---

## Step 3 — measure each feed — ← **THE PART THAT KEEPS GETTING SKIPPED**

This is the exact recipe that produced §5j's table; it is proven against prod, not proposed.

```bash
B=https://prod-podcast.tail6d0ed4.ts.net; P=/app/output
curl -fsS "$B/api/corpus/feeds?path=$P" > /tmp/allfeeds.json

FID=$(jq -r --arg t "Conversations with Tyler" \
      '.feeds[]|select(.display_title|test($t))|.feed_id' /tmp/allfeeds.json)

curl -fsSG --data-urlencode "path=$P" --data-urlencode "feed_id=$FID" \
     --data-urlencode "limit=4" "$B/api/corpus/episodes" \
  | jq -r '.items[].metadata_relative_path' \
  | while read -r rel; do
      curl -fsSG --data-urlencode "path=$P" --data-urlencode "metadata_relpath=$rel" \
           "$B/api/corpus/episodes/detail" \
        | jq -r '"\(.episode_title[0:40])\tbullets=\(.summary_bullets|length)\tpartition=\(.bridge_partition)"'
    done
```

Response-shape gotchas that cost time: `/api/corpus/episodes` returns **`items`**, not
`episodes`; `/api/corpus/episodes/detail` takes **`metadata_relpath`** (URL-encode it — the paths
contain spaces and `&`).

### The gates (§5i — evidence-based, replaces the §5g bands)

| Signal | INVESTIGATE | Why |
| --- | --- | --- |
| KG nodes/ep (`bridge_partition.total`) | **< 18** | observed floor across 14 feeds is 19.5 |
| `bridge_partition.both` | **< 8** | **the primary signal** — widest real spread (12.0–19.5) |
| `gi_only` | **> 2** | observed max 0.8; a rise means GI output the graph cannot corroborate |
| any ad marker in `summary_bullets` | **any** | a single hit is a **pipeline defect**, not a content verdict |
| bullets/ep | *not a gate* | pipeline constant (§5i Finding 3) — record only |

4 episodes per feed is a **gate check, not a ranking**. §5i's own caveat holds: that sample is
too small to say feed X beats feed Y, and §5j saw feeds move ±3 on `both` between samples.

---

## Step 4 — record the verdict — ← **ALSO SKIPPED; no Batch A feed has a bucket after 3 passes**

Per feed, append to **§5j** of the plan doc: episodes ingested, KG/ep, `both`, `gi_only`, ad-marker
count, modelled cost, and the bucket. §5g's four buckets, and the distinction matters:

- **DEEPEN** — content clears the bar, processing is clean. Go to 10; revisit for 20/50.
- **PARK** — content clears the bar, processing does not. Stay put, log the **pipeline defect** in
  `INCREMENTAL-ROLLOUT-FOLLOWUPS-2026-08-11.md`, revisit after the fix.
- **DROP** — content does not clear the editorial bar. The only content-driven exit.
- **BLOCKED** — structurally not ingestible (over the §5h 2-hour ceiling, non-English, dead feed).

**A bad probe result is not a reason to drop a show** (§5g Phase 3). Parking a show because our
cleaning stage mangles it, and dropping it as "low quality", produce the same corpus today and
opposite outcomes in three months.

Findings land in **two** places and confusing them loses information: content verdicts and depth
decisions go in the plan doc; pipeline defects the probe exposes go in the rollout-followups doc,
because those are reusable across all 24 feeds.

**A dropped show with no recorded reason will be re-proposed by someone in six months.** Probe
group 1 has no recorded cost, and that is exactly why the $10 cap cannot be reasoned about now.

---

## Two open decisions that are NOT part of this job

Flagging so they are not silently absorbed:

1. **Dwarkesh and Ideas of India are stuck at 10 episodes.** Both are DEEPEN on the evidence
   (§5j) but nobody made the depth call. Needs an operator decision, not an agent's. Still true
   on 2026-09-02 — every other feed has moved and these two have not.
2. **Batch B (§5f, 10 feeds) stays closed** until this batch is measured.

Two observations from watching the 2026-09-01 run that are **not explained** and want an owner:

3. **KG `node_count` was exactly 29 on all seven episodes measured** in the Conversations with
   Tyler job — identical, not merely close. `prod_dgx_full.yaml:51` sets `kg_max_entities: 15`,
   so 29 is not that cap directly. A constant across seven different episodes is not a property
   of content, and §5i promoted graph structure to the **primary** quality signal — a saturated
   constant cannot discriminate. **I did not establish the cause.**
4. **18 of 36 warnings in that job were `insight_salvage`**, all identical: `model returned 30
   insights for a ceiling of 25; keeping the first 25. The prompt is not constraining the count.`
   Salvage works, but roughly a fifth of each episode's insights are discarded by **arrival
   order** rather than quality.

---

## Not verified — do not treat these as done

- **Cost per episode.** No probe-group-1 cost was ever recorded, and it is not recoverable from
  the corpus API. The ≲$0.30/ep figure in §5g is a **planning assumption**, never a measurement.
  The `podcast_pipeline_run_cost_usd_total` metric is the source — read it before and after the
  first feed.
- **Episode duration for the ten pending feeds.** Last measured 2026-08-13 (§5h). A show that
  drifted over the 2-hour ceiling would be BLOCKED, and nobody has re-checked.
- **Licensing / bridge constraints** for any pending feed. §7 has listed this as open since
  2026-07 and nothing has closed it.
- **Superseded 2026-09-02:** the original note here said the `POST /api/jobs` call had never
  been executed. It has now — the 2026-09-01 batch ran it ten times against prod on `git_sha
  e8c6f35`, and `episode_selection` (which did not exist when this was written) is the parameter
  that governed the result. Source re-read after rebasing to `e555bc92`.
- The original wording, kept for the record: the parameters were read from
  `routes/jobs.py:166` ff. and `build_pipeline_argv` (`server/jobs.py:349`), and the auth,
  `GET /api/feeds`, `GET /api/corpus/*` and the whole Step 3 recipe **were** run live against prod.
  Step 1's `PUT` was **not** run either; it is read from `routes/feeds.py:104`.
- **Ad detection is string-matching** on summary text (§5i caveat). It cannot see a sponsor
  segment summarised into neutral prose. A clean result is good evidence, not proof.
