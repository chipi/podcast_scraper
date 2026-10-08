# Handover — prod steps after PR #2298 (player beta feedback + Android blank-render fix)

For the agent deploying PR #2298. Each step that changes prod needs the operator's go, one by
one. Background: `docs/wip/player-memory-analysis-2026-10-08.md`.

## Order (operator, 2026-10-08)

1. Publish **1.0.3** to TestFlight and Play internal. Confirm the Android build is actually
   available on Play: a tester device's Play page offers 1.0.3 as an update. The Console saying
   "published" is not enough — on 2026-10-07 the build was published but not installable for about
   half an hour.
2. Deploy the server (this PR) — **only after step 1 is confirmed**. The deploy changes what 1.0.2
   shows (below), so it must not go out while there is no 1.0.3 to update to.
3. `0024`, then `0025` (step 2 below), each with the operator's OK.
4. Set the released version to **1.0.3** in the operator viewer. 1.0.2 then shows the update
   prompt; this needs the deploy, because #2296 (`9f8a8605b`) is not in `sha-0f63257`.
5. The operator tells testers to update.

**Accepted risks** (operator: "nobody will go and update just like that"):

- Between 1 and 2, a 1.0.3 install talks to the old server, which lacks `/whats-new`,
  `/recommended` and `/podcasts/suggested`. Those Home sections and the guided start's shows step
  fail until the deploy.
- After 2, 1.0.2 shows Your Week's two new sections ("You listened to", "You saved") with raw text
  keys as headings (`home.yourWeekSection.listened_this_week`). It also stops showing new episodes
  from follows: they moved to What's new, which 1.0.2 lacks. This lasts until the tester updates.

## 0. Prod as of 2026-10-08 (read-only check)

- Images `sha-0f63257`; corpus at **2.7.17**, migrations through `0023` applied
  (`upgrade status` in `compose-api-1`).
- `0024_shared_removed_speaker_prefixes` (2.7.18, #2294) is **not** in that image. It has its own
  review — dry-run and read its frozen set name by name, then apply — **before** `0025` (strict id
  order).
- 96 GB free on `/`. Every `upgrade run` takes a whole-corpus snapshot (13 GB on 2026-10-03).

## 1. The server half: check after the deploy

The episode detail now serves the player-size copy, plus a separate thumbnail for cards:

```bash
# any episode slug; a session token as in the smoke specs
curl -s -H "Authorization: Bearer $TOKEN" https://closelistening.app/api/app/episodes/<slug> \
  | jq '{artwork_url, artwork_thumb_url}'
# expect  artwork_url  …&size=medium   and   artwork_thumb_url  …&size=thumb
```

Until step 2 runs, `size=medium` falls back to the original — the API mounts the corpus read-only
and cannot make the copy. **Cards are fixed immediately**: their 320px thumbnails already exist
(m0013). The player hero keeps the original until step 2.

## 2. `m0025_artwork_medium` — the 1024px player copies (needs the operator's OK)

Writes `corpus-art/derived/medium/<sha>.jpg` (≤1024px, never upscaled) for every stored cover. It
is derived data only: no artifact text changes, no LLM, no fetch. 1,318 stored images on
2026-10-08; the total size of the copies has not been measured — read it off the dry-run host
with `du` after apply.

```bash
docker exec compose-api-1 python -m podcast_scraper.cli upgrade status --corpus-dir /app/output
# after 0024 is applied:
docker exec compose-api-1 python -m podcast_scraper.cli upgrade run --to 2.7.19 --dry-run --corpus-dir /app/output
#   expect: "would write ~1318 medium copy(ies) of 1318 stored image(s); 0 undecodable"
docker exec compose-api-1 python -m podcast_scraper.cli upgrade run --to 2.7.19 --corpus-dir /app/output
docker exec compose-api-1 python -m podcast_scraper.cli upgrade verify --corpus-dir /app/output
```

Then confirm what the player is served:

```bash
REF='<an artwork ref from step 1>'
for s in medium large; do curl -s -o /dev/null -w "$s %{size_download}\n" \
  "https://closelistening.app/api/app/artwork?ref=$REF&size=$s"; done
# medium must be smaller than large for a >1024px original
```

**Undo:** delete `/app/output/.podcast_scraper/corpus-art/derived/medium/`. The API falls back to
the originals, which is today's behaviour. Undecodable images are listed in
`artwork_medium_failed.jsonl` and keep being served as originals.

## 2b. `m0026_missing_covers_stored` — covers that were only a feed-host URL (needs the operator's OK)

One show ("The China-Global South Podcast") has no stored cover. Its 3000x3000 PNG is 11.9 MB and
the writer's cap was 8 MB (now 32 MB), so phones download the original for a 116 px tile.
**FETCHES**, like 0021: it downloads each missing image once, writes the original plus its thumb
and medium copies, and records `image_local_relpath` on the affected episodes. Each metadata
rewrite is backed up and receipted; `undo` restores them.

```bash
docker exec compose-api-1 python -m podcast_scraper.cli upgrade run --to 2.7.20 --dry-run --corpus-dir /app/output
#   expect: 1 cover to fetch (the libsyn PNG) for that show's episodes
docker exec compose-api-1 python -m podcast_scraper.cli upgrade run --to 2.7.20 --corpus-dir /app/output
docker exec compose-api-1 python -m podcast_scraper.cli upgrade verify --corpus-dir /app/output
```

An unfetchable image is recorded in `missing_covers_failed.jsonl` and keeps its remote URL.

## 2c. After the deploy: re-measure

- `curl` `/your-week` with a token: about 3 s on `sha-0f63257`; `f5ab549a9` should bring it to
  0.2–0.6 s.
- Open an episode twice: `/episodes/{slug}/related` is now cached, so the second open should not
  take 2.4 s.
- `make perf-android` on the release-tier app, and `make perf-ios` (1.0.2 and 1.0.3 walks), to
  compare with `docs/wip/perf-scan-2026-10-08.md`.

## 3. The phone half needs a store build (1.0.3), not the deploy

The Android fix — no backdrop blur, the paging changes and Copy debug info — lives in the web
bundle **baked into the native apps**. A server deploy does not reach the testers' phones. It ships
when 1.0.3 is built for Play internal and TestFlight. 1.0.3 also carries new native code (the
`AppProcess.memoryInfo` plugin, Android and iOS), so it must be a full native build, not a web-only
update.

## 4. Afterwards: confirm with the tester

On the Pixel 8 that went black: Settings → **Copy debug info** gives the model, WebView version,
free/total RAM and the app's memory. Then Episode notes → toggle Key points → open "More like this"
and scroll it, the case that reproduced 3 of 3 on 1.0.2.

## Not done here (for the operator to decide)

- Shows with no stored cover still load their feed's remote image, up to 3000px (one on Discover's
  trending rail). Fixing it needs a migration that downloads and stores those covers. Not built.
- Several endpoints return everything for the client to page: the entity cards' episodes,
  `/your-week`, `/favorites`, `/collections`, `/resurfacing`, `/playback`, `/podcasts`. Images are
  paged; these JSON payloads are not. Planned for the performance scan.
