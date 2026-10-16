# Audio timing on hosts that insert ads (RFC-129, #2359)

## Why this exists

Some hosts insert ads into an episode when it is fetched and change them within hours, so the audio
a listener plays is shifted from our transcript, differently for different listeners. The player
fixes this itself: it fingerprints short ranges of the stream it plays and lines them up with the
episode's **audio reference** (fingerprints of the audio we transcribed). This recipe produces what
the player needs on the server side:

- **references** for every episode (`<stem>.audioref.json` next to the transcript), served at
  `GET /api/app/episodes/{slug}/audio-reference`. New episodes get theirs at the end of their
  ingest run; existing ones are backfilled once from the audio archive.
- **server time maps** (`<corpus>/audio_time_maps.json`), the fallback for a player that cannot
  measure its stream (a web page on a host that blocks cross-origin reads, a variable-bitrate
  file). Rebuilt daily.

Design and measurements: [RFC-129](../rfc/RFC-129-audio-sync-for-dynamic-ads.md). No episode is
hidden for drifting; no DGX, no model: CPU, ffmpeg, rclone.

## Quick reference

All commands run in the pipeline image, on the live corpus (`/app/output`), on the prod host:

```bash
cd /srv/podcast-scraper
C="docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.prod.yml -f compose/docker-compose.vps-prod.yml"
$C run --rm --no-deps pipeline-llm python -m podcast_scraper.audio_sync <command> --corpus /app/output ...
```

| Command | Reads | Writes (only with `--write`) |
| --- | --- | --- |
| `drift-check` | each episode's served length (a header read per episode) | `audio_timing_drift.json` |
| `build-references --rclone-remote audioarchive` | the archived original, only where provenance says byte-identical | `<stem>.audioref.json` per episode |
| `build-maps --rclone-remote audioarchive` | the archived original and the stream as served now, for drifted episodes | `audio_time_maps.json` |

Every command without `--write` only reports. All writes are atomic.

## After the deploy that ships RFC-129

1. **Dry run of the reference backfill.** It should list mostly `would_write` and some
   `skip_no_exact_original` (episodes whose archive copy is a later re-download):

   ```bash
   $C run --rm --no-deps pipeline-llm python -m podcast_scraper.audio_sync build-references \
     --corpus /app/output --rclone-remote audioarchive --limit 20
   ```

2. **Backfill all references**, detached. It pulls each original from the archive (about 2,600
   files, about 156 GiB), so it runs for hours. It is resumable: an episode that already has a
   reference is skipped.

   ```bash
   $C run -d --no-deps --name audio-refs pipeline-llm python -m podcast_scraper.audio_sync \
     build-references --corpus /app/output --rclone-remote audioarchive --write
   docker logs -f audio-refs        # one line per episode; a summary line at the end
   ```

3. **Drift check, then server maps:**

   ```bash
   $C run --rm --no-deps pipeline-llm python -m podcast_scraper.audio_sync drift-check \
     --corpus /app/output --write
   $C run --rm --no-deps pipeline-llm python -m podcast_scraper.audio_sync build-maps \
     --corpus /app/output --rclone-remote audioarchive --write
   ```

4. **Check:**
   - `curl` the api (signed in) for a drifted episode's
     `/api/app/episodes/<slug>/audio-reference`: 200 with `checkpoints`.
   - `/api/app/episodes/<slug>` carries `audio_time_map`.
   - On a phone, play that episode: the transcript highlight stays on the voice across an ad
     break, and a Moment from it starts on its words.

5. **Daily:** steps 3 (drift check and maps). New episodes get references at ingest; re-run step 2
   only to fill gaps (it skips what exists).

## Verification

- `find /srv/podcast-scraper/corpus -name '*.audioref.json' | wc -l` grows during the backfill.
- `python3 -c "import json; print(len(json.load(open('/srv/podcast-scraper/corpus/audio_time_maps.json'))))"`
  is the number of mapped episodes.
- Pipeline logs after an ingest run: `audio references (<run>): {'written': N, ...}`.

## Troubleshooting

- **`skip_no_exact_original`:** the archive holds only a re-download (provenance
  `byte_identical_to_transcribed_audio: false`), or no audio. No reference is possible; the
  player falls back to the server map, if any.
- **`skip_poor_match` (build-maps):** under half of the original's talk was found in the stream
  as served now (the host serves another edit, or the stream failed). That episode gets no map.
- **`skip_stream_unreadable`:** the host did not answer within 15 minutes; the next daily run
  retries.
- **Undo:** delete `audio_time_maps.json` (maps), or the `*.audioref.json` files (references).
  The api and the player treat a missing file as "nothing measured".
