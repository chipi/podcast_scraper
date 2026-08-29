# HANDOVER — ingest the 10 remaining Batch A feeds, 10 episodes each

**Date:** 2026-08-29 · **Plan of record:** [ONBOARDING-SHOWS-FOR-ENRICHER-VALUE.md](ONBOARDING-SHOWS-FOR-ENRICHER-VALUE.md)
§5f (list) / §5g (protocol) / §5i (thresholds) / §5j (current state) · **Expansion vehicle:** `#630`

---

## The goal, and only the goal

Ingest **10 newest episodes** for each of the **10 feeds** in the table below, then measure each
one against the §5i thresholds and write the verdict back into the plan doc's §5j.

That is the whole job. Do **not** start Batch B — §5f gates it on Batch A being measured. Do
**not** re-litigate the existing 14 feeds; §5g records that as an operator decision.

---

## Where the corpus stands

Verified live 2026-08-29 (see §5j for the full table and the commands):

- **14 feeds, 765 episodes.** `feeds.spec.yaml` carries the same 14 — spec and corpus in sync.
- The nine originals plus **probe group 1** (§5g): a16z 71, Lenny's 53, Pragmatic Engineer 51,
  Dwarkesh 10, Ideas of India 10.
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
   large, partly-dated archives. Always pass `episode_order=newest` — never let a run walk the
   back-catalog.
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

## Step 1 — add the ten feeds to `feeds.spec.yaml`

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

## Step 2 — ingest, one feed at a time

**§5g says serial, one show at a time** — the pipeline runs one job at a time regardless, and
serial execution keeps each result attributable. Wait for `succeeded` before starting the next.

```bash
RSS="https://rss.libsyn.com/shows/137081/destinations/850607.xml"   # one at a time
curl -fsS -X POST -H "X-Operator-Key: $KEY" \
  --get --data-urlencode "path=/app/output" \
  --data-urlencode "feed=$RSS" \
  --data-urlencode "skip_existing=true" \
  --data-urlencode "max_episodes=10" \
  --data-urlencode "episode_order=newest" \
  "$B/api/jobs" | jq
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

## Step 3 — measure each feed

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

## Step 4 — record the verdict

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
   (§5j) but nobody made the depth call. Needs an operator decision, not an agent's.
2. **Batch B (§5f, 10 feeds) stays closed** until this batch is measured.

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
- **The `POST /api/jobs` call in Step 2 has not been executed** — the parameters are read from
  `routes/jobs.py:166` ff. and `build_pipeline_argv` (`server/jobs.py:349`), **re-read after
  rebasing onto `origin/main` at `5dbe32bf` (2026-08-28)**, and the auth,
  `GET /api/feeds`, `GET /api/corpus/*` and the whole Step 3 recipe **were** run live against prod.
  Step 1's `PUT` was **not** run either; it is read from `routes/feeds.py:104`.
- **Ad detection is string-matching** on summary text (§5i caveat). It cannot see a sponsor
  segment summarised into neutral prose. A clean result is good evidence, not proof.
