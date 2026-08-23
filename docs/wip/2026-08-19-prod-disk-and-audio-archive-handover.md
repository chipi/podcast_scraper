# Prod disk pressure and the audio archive — handover

**For the agent with SSH.** Written 2026-08-19 after the reprocess incident. Everything below is
read from the repo at `hotfix/cost-containment-and-scope-gate`; measurements are cited to their
source. Where I could not verify something from here it says so.

## TL;DR

Audio is ~43 GB of a 48 GB corpus, spread over three directories, and **nothing anywhere evicts
it**. Two of those directories hold the SAME bytes twice because prod runs the default
`corpus_media_link_mode: copy`. The remote archive exists, is not enabled, and would move only
one of the three directories off the box. The cheapest large win is not the archive.

## 1. Where the bytes are

Measured 2026-08-18 by `inspect-prod-corpus` `disk_usage`, recorded in commit `0ff7435d`:

| Directory | Size | What it is |
| --- | --- | --- |
| `media/` | **30.5 GB** | episode audio copied in for viewer playback |
| `.podcast_scraper/audio-cache/` | **12.6 GB** | content-addressed GUID cache (#947) |
| `.tmp_media/` | **3.4 GB** | staging scratch |
| everything else | ~1.5 GB | transcripts, metadata, GI/KG, artwork, search index |

That was **before** the two incident runs. Those added 127 re-processed episodes with
`--no-transcript-cache`, so every one was re-downloaded and re-copied.

## 2. Why it only grows

Three independent reasons, all verifiable in the repo:

1. **There is no eviction. Anywhere.** `src/podcast_scraper/utils/audio_cache.py` exposes
   `store`, `lookup_by_guid`, `fetch_into`, `copy_into`, `store_via` — and no `prune`, `evict`,
   or retention of any kind. The cache is append-only by construction. There is no
   `audio_cache_max_bytes`, no TTL, no LRU.
2. **Audio is stored TWICE.** `persist_episode_media` defaults to `True` and
   `corpus_media_link_mode` defaults to `"copy"` (config.py:3982-3999). `cloud_balanced.yaml`
   — prod's profile — sets neither, so prod runs both defaults. The config's own docstring says
   `hardlink`/`symlink` "halv[es] on-disk audio footprint when the cache is on the same
   filesystem". We are paying that penalty.
3. **A reprocess re-downloads everything.** `--no-transcript-cache` is correct for a repair (the
   cache key cannot distinguish a damaged transcript from a good one), but it means every repair
   run adds a fresh copy of every episode it touches.

## 3. The four levers, cheapest first

I have ordered these by bytes-recovered per unit of risk, **not** by how interesting they are.
The archive is third, not first.

### Lever 0 — `.tmp_media/` (3.4 GB, zero risk, do it now)

Staging scratch. Nothing references it after a run completes. Safe to delete outright when no
pipeline container is running (both are stopped as of 12:18z). This is free.

### Lever 1 — stop storing audio twice (up to ~12 GB going forward, low risk)

Set `corpus_media_link_mode: hardlink`. Caveats that must be checked ON THE BOX:

- Hardlinking requires `media/` and the audio-cache to be on the **same filesystem**. The
  measurement shows the cache at `<corpus>/.podcast_scraper/audio-cache`, i.e. inside the corpus,
  so this is likely satisfied — but confirm with `stat -c %d` on one file from each.
- It falls back to a copy silently when linking is unavailable, so this cannot break anything;
  it can only fail to help.
- **It is not retroactive.** Existing duplicates stay until something de-duplicates them. A
  hardlink-dedupe pass over existing pairs is a separate job (`jdupes -L` or equivalent) — worth
  doing, but verify the two files really are byte-identical first; `media/` preserves the source
  extension and the cache may hold a different container.

### Lever 2 — the remote archive (#1679) (~12.6 GB off the box, medium effort)

`audio_storage_backend: remote` makes `resolve_backend` return an rclone backend **instead of**
the local one (`utils/audio_cache.py:171-190` — it is exclusive, not additive). So enabling it
moves the audio-cache off the box.

**What it does NOT do, and this is the part to be clear about:**

- It does **not** touch `media/` — the 30.5 GB, the biggest directory. That is governed by
  `persist_episode_media` / `corpus_media_link_mode`, not by the storage backend.
- It is **not retroactive**. From `docs/recipes/prod-audio-archive.md`: enabling it stores audio
  for episodes ingested *from that point on*. Existing audio stays local until something moves it.
- So "turn on the archive" alone recovers approximately nothing today. It changes the trajectory,
  not the current state.

**What is actually missing to enable it** (from #1679, verified by reading the repo):

| Piece | State |
| --- | --- |
| Storage Box in IaC | conditional on `var.audio_storage_box_type != ""`; whether it is set is outside the repo |
| rclone creds reaching prod | **nothing delivers them** — no `RCLONE_CONFIG_*` in `deploy-prod.yml`, cloud-init, or `compose/*` |
| `audio_storage_backend: remote` | **no profile sets it** |
| `archive backfill` implementation | present (`src/podcast_scraper/archive/backfill.py`) |
| a way to run it against prod | **none** — CLI only, no workflow |

Step 2 is the substantive one and does not exist today.

### Lever 3 — retention policy (unbounded win, needs a decision)

There is no answer today to "how long do we keep raw audio locally". Until there is, every lever
above only delays the next alert. The decision is genuinely the operator's: keeping audio enables
future re-transcription (better ASR, new diarization) without re-fetching; not keeping it means
depending on publishers still serving those episodes.

## 4. The decay clock — why the archive is time-sensitive anyway

`archive backfill` recovers audio **from the publisher**, so it only works while the episode is
still served. Every day the archive stays unwired, some episodes roll off and their audio becomes
permanently unrecoverable — and with it, any future re-transcription of those episodes.

Prior measurement on the local corpus: 9/9 damaged episodes were still served by their feeds
(archives 71-2950 items). The equivalent check on prod's work-list has not been done.

This is the one item in the backlog where **the cost of waiting is not delay, it is irreversible
loss**. It argues for enabling the archive even though it recovers no disk today.

## 5. Do NOT delete these

From `0ff7435d`, learned the hard way while fixing the backup:

- `.podcast_scraper/corpus-art/` — artwork referenced by metadata's `image_local_relpath`.
- `.podcast_scraper/audio-archive-provenance.jsonl` — records which audio is a RE-FETCH rather
  than the original bytes. #1631 calls this "load-bearing rather than decorative": a later WER
  comparison against a dynamic-ad re-encode is silently wrong without it.

Both live under `.podcast_scraper/` alongside `audio-cache/`. **Prune the child, never the
parent.** The backup's first exclude pass got this wrong.

## 6. What to check on the box

```bash
# where the bytes actually are, now (post-incident)
du -sh /var/lib/docker/volumes/*corpus_data*/_data/{media,.tmp_media,.podcast_scraper} 2>/dev/null
du -sh /var/lib/docker/volumes/*corpus_data*/_data/.podcast_scraper/audio-cache

# same filesystem? (decides whether hardlink mode can work)
stat -c '%d %n' <one file in media/> <one file in audio-cache/>

# are media/ and cache entries actually duplicate bytes?
md5sum <a media file> <its cache counterpart>

# free space and the trend
df -h /
```

## 7. Interaction with the pending repair — read before restarting anything

- `.viewer/jobs.paused` is SET and must stay set until the hotfix image is deployed. There is no
  working cost protection in the running image.
- The 32-episode repair is **still entirely undone** — forensics showed zero overlap between the
  work-list and the 127 episodes processed. See
  `docs/incidents/INCIDENT-2026-08-18-unattended-reprocess-orphan-spend.md`.
- The repair needs audio. Without the archive it re-fetches from live feeds, which works while
  episodes are still served — and adds *more* disk. **Disk headroom is now a precondition for the
  repair, not an unrelated concern.**
- A backfill is hours of rate-limited downloading. Do not overlap it with the nightly corpus
  backup (05:37 UTC), a reprocess, or a deploy.

## 8. Open questions I could not answer from here

1. Does the Hetzner Storage Box already exist, or does provisioning still need doing?
2. Is `audio_cache_in_corpus` actually True in prod? The measurement implies yes; the profile
   does not say.
3. Are `media/` and the cache byte-identical per episode, or different containers? This decides
   whether a dedupe pass is safe.
4. Current free space and the daily trend — how much time is there before this is urgent again?
5. Do we backfill everything, or only what a reprocess would plausibly need? Everything is safer
   given the decay clock, but costs storage.
6. Should `reprocess-prod.yml` **assert** the archive is configured rather than mentioning it in
   a comment? Today a reprocess with the archive off silently re-fetches from feeds, which works
   until it doesn't — and the failure surfaces as missing episodes, not as a clear cause.

## 9. Related

- #1679 — audio archive built but not enabled (the substantive issue)
- #1199 — remote storage backend; #1631 — `archive backfill`
- `docs/recipes/prod-audio-archive.md` — the operator recipe
- `INCIDENT-2026-08-05-prod-disk-image-pileup.md` — the LAST disk incident on this box, from
  unpruned Docker images. Same shape: something accumulated for days with no eviction and no
  alert until the disk was nearly full. Worth reading for what did and did not catch it.
- `INCIDENT-2026-08-18-unattended-reprocess-orphan-spend.md` — why there is 15 GB of new audio
