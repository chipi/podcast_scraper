# Handover for the prod agent: after `feat/player-1.0.4` is deployed

Written 2026-10-10, for whoever deploys this branch to prod later. **Every prod-changing step
needs the operator's go, one by one.** Prod fixes are text-file fixes: no DGX, no relabel, no full
re-index (repo `CLAUDE.md`). Do not run analysis inside the prod containers. To inspect data, copy
the files off read-only and analyse them locally, then delete the copy.

Branch summary for the operator: `docs/wip/HANDOVER-player-1.0.4-2026-10-10.md`.

## State of prod when this was written

- Images: `sha-3e06151` (contains `2ca8dac75`).
- Corpus ledger: version `2.7.21`; 0024–0027 applied. Read from
  `/app/output/upgrade_ledger.json`, read-only, 2026-10-10.
- This branch adds one migration, **0028** (→ 2.7.22). After the deploy it is the only pending
  one.
- Re-check both before starting: someone may have deployed or migrated since.

## Before the deploy

1. The branch is merged to `main` and CI is green at the merge commit (Python application, viewer,
   player, stack test). Red CI is not advisory.
2. Images are built for that SHA (stack test publishes `sha-<short>` for all five).
3. The queue is empty. The nightly starts at 23:00 UTC, and a deploy kills any running job.
4. Dispatch the prod deploy for that SHA; the operator clicks the prod gate. Never approve it
   yourself.

## After the deploy, in this order

### 1. Confirm the deploy

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/podcast_prod_operator deploy@prod-podcast \
  'docker ps --format "{{.Names}} {{.Image}}" | grep -i api'
```

Every api container shows the new `sha-…`. The smoke tests are green.

### 2. Migration 0028: time the transcript passages already in the search index

**Why.** The indexer timed transcript chunks only from `metadata.summary.timestamps`. The pipeline
does not write that field, so every transcript passage in prod's index sits at 0 ms. The deploy
fixes the indexer for new ingests. 0028 times the existing rows in place from the timed transcript
on disk (`*.segments.json`): no re-index, no model, no DGX.

**What it writes.**
- `start_time` / `end_time` on rows of the `segments` and `segments_nonen` tables, via a
  partial-column `merge_insert` on `id`, then `compact()`.
- The matching `timestamp_*_ms` in `search/metadata.json`.
- An undo record, written first: `search/transcript_chunk_times_undo.json`.
- It writes a row only when the row's stored text equals the transcript slice it maps to. A
  transcript changed since indexing is left alone.

**Run it in `compose-api-1`** (as 0024–0027 were). Like every `upgrade run`, it takes a ~13 GB
corpus snapshot.

```bash
docker exec compose-api-1 python -m podcast_scraper.cli upgrade status --corpus-dir /app/output
docker exec compose-api-1 python -m podcast_scraper.cli upgrade run --to 2.7.22 --dry-run --corpus-dir /app/output
```

**Read the dry-run line before asking for the go.** It reads:

> `N transcript row(s) at 0 ms; T timeable from segments on disk; left alone: a no segments,
> b transcript changed since indexing, c no overlapping segment, d no char span
> (s of the rows are source-language text)`

What to expect:
- `N` is roughly every transcript row in the index.
- `a` (no segments) is small: episodes with a plain downloaded transcript and no timed one (e.g.
  riverside bc34ff99 "AI Sychophancy with Steve Rathje").
- `b` (transcript changed since indexing) is small. **If it is large, stop and report**: the
  index and the transcripts disagree, and timing those rows would be guesswork.
- `d` (no char span) is about 0. If it is large, the sidecar is not the one written with this
  index; stop.
- These expectations come from a copy of the e2e corpus (284 of 284 timed). **0028 has never run
  on prod data.**

**Apply** (with the operator's go), then verify:

```bash
docker exec compose-api-1 python -m podcast_scraper.cli upgrade run --to 2.7.22 --corpus-dir /app/output
docker exec compose-api-1 python -m podcast_scraper.cli upgrade verify --corpus-dir /app/output
```

**Spot-check.** A Close Listening MCP `search_corpus` for a sentence you know from an episode:
transcript hits now carry a non-zero `timestamp_start_ms`.

**Undo** (restores every written row and sidecar entry):

```bash
docker exec compose-api-1 python -c "from pathlib import Path; from podcast_scraper.upgrade.migrations.m0028_transcript_chunk_times import undo; print(undo(Path('/app/output')))"
```

### 3. Duplicate playback milestones (beta dashboard)

**Why.** Before `6c6dcd104` a listener who reopened an episode past 25% re-reported 25 with a new
timestamp, and each re-report was kept. The dashboard counts rows per milestone, so they read as
more listeners. The deploy stops new repeats. The rows already written stay until this pass runs.

**Tool:** `scripts/ops/dedupe_playback_milestones.py`.
- It keeps the first record per user, episode and milestone; every other line is kept byte for
  byte.
- It backs each file up beside itself and writes receipts.
- `--undo` restores and keeps lines appended since the apply.
- It is not in the image (`scripts/` is not copied), so copy it in.
- `player-api-1` mounts app data read-write at `/app/appdata`.

```bash
scp -o IdentitiesOnly=yes -i ~/.ssh/podcast_prod_operator scripts/ops/dedupe_playback_milestones.py deploy@prod-podcast:/tmp/
ssh -o IdentitiesOnly=yes -i ~/.ssh/podcast_prod_operator deploy@prod-podcast '
  docker cp /tmp/dedupe_playback_milestones.py player-api-1:/tmp/ &&
  docker exec player-api-1 python /tmp/dedupe_playback_milestones.py --appdata /app/appdata'   # dry-run
```

Read `{"mode": "dry-run", "users": …, "files": …, "dropped": …}`, then ask for the go and run the
same with `--apply`. Exit 1 means a repeat was left or a file changed mid-write; re-run. Undo:
`--undo`. Remove the copies from `/tmp` afterwards.

Expected effect: the dashboard's per-milestone counts drop by the number dropped. The real
listener counts were never higher than the deduplicated numbers.

### 4. Moments now matches the Brief: check it

**Why.** Moments used only named-speaker insights, while the Brief falls back to unnamed ones.
On a copy of prod (2,958 episodes), 289 episodes showed insights and no moments. They now
agree on all but 2, and neither of those 2 has a time on disk. Every "▶ Moments" door reads the
new `has_moments`.

Check, signed in as a smoke account (`project_prod_smoke_test_accounts` in Claude memory):
- An episode where naming named nobody now has moments, with no speaker on them. Example: anchor.fm
  5a072801 "#10 Michael Montgomery: Psychophobia…". `GET /api/app/episodes/{slug}` has
  `has_moments: true`; `GET /api/app/episodes/{slug}/moments` is non-empty.
- These episodes still have no moments and must show **no** door (`has_moments: false`):
  - "AI Sychophancy with Steve Rathje": no timestamps on disk;
  - "How a Sardine Gets From the Ocean…": only quoteless insights;
  - "Dylan Patel - Inside the Trillion-Dollar AI Buildout", "Why Apple Is Heading Into a Pivotal
    Moment", "Rory Johnston on Why His $200 Oil Prediction…": every insight tier 0.
- Side effect to expect: in 55 episodes whose only "named" insight had no quote, the Brief now
  shows the grounded unnamed insights as well.

### 5. Search changes: check them

- **Speaker / topic scope** (operator viewer, MCP). These now go into the query. The **first**
  speaker or topic search after each ingest builds an index of every insight's speakers and
  topics, about 3 s at prod size; later ones are fast. It is deliberately not in the cache
  warmer, because a CPU-bound warmer starved searches into 504s on 2026-10-06. A slow first
  speaker search is expected; a slow second one is not.
- **The Brief's search** finds other word forms ("invested" finds "investing") and passages with
  all but one of three or more remembered words, and marks what matched. Try both in the app.

### 6. The first nightly after the deploy

- The index update for new episodes writes timed transcript rows. Check a new episode's transcript
  hit in `search_corpus`: its `timestamp_start_ms` is non-zero without 0028 having touched it.
- No new errors in the api log around `episode_has_moments` or `_tier_scope`.

## Not covered here

- **Speaker naming (#2304)** is someone else's: 230 episodes where naming resolved nobody, 3 where
  the value gate dropped everything, and 1 transcript without timestamps. Moments no longer
  depends on naming. Do not fix naming as part of this deploy.
- The phone build (TestFlight / Play) is the mobile session's.
- 0028 and the milestone pass have been tested locally, not on prod data. Read every dry-run.
