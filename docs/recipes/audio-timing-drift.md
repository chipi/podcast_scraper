# Audio timing on hosts that insert ads (RFC-129, #2359)

## Why this exists

Some hosts insert ads into an episode when it is fetched and change them within hours, so the audio
a listener plays is shifted from our transcript, differently for different listeners. The player
fixes this itself: it fingerprints short ranges of the stream it plays and lines them up with the
episode's **audio reference** (fingerprints of the audio we transcribed), served at
`GET /api/app/episodes/{slug}/audio-reference` from `<stem>.audioref.json` next to the transcript.

- **New episodes:** the pipeline writes the reference at the end of the ingest run, from the audio
  it downloaded, before the local copy goes to the archive and is deleted. Nothing to run.
- **Existing episodes:** one backfill after the deploy that ships RFC-129, from the audio archive.
  The archive is a one-time source here, never part of a routine job.

Design and measurements: [RFC-129](../rfc/RFC-129-audio-sync-for-dynamic-ads.md). No episode is
hidden for drifting; no DGX, no model: CPU, ffmpeg, rclone (backfill only).

## Quick reference

All commands run in the pipeline image, on the live corpus (`/app/output`), on the prod host:

```bash
cd <deploy-dir>
C="<the deployment's docker compose command for the control-plane stack>"
$C run --rm --no-deps pipeline-llm python -m podcast_scraper.audio_sync <command> --corpus /app/output ...
```

`--env-file` is required: Compose looks for `.env` next to the first compose file (`compose/`),
not in the directory you run it from, and without it fails with "required variable
PODCAST_DOCKER_PROJECT_DIR is missing a value" (prod, 2026-10-17).

| Command | When | Reads | Writes (only with `--write`) |
| --- | --- | --- | --- |
| `build-references --rclone-remote audioarchive` | once, after the deploy | the archived original, only where provenance says byte-identical | `<stem>.audioref.json` per episode |
| `drift-check` | manual, to diagnose | each episode's served length | `audio_timing_drift.json` |
| `build-maps` | manual, to diagnose | each listed episode's reference and its stream as served now | `audio_time_maps.json` |

Every command without `--write` only reports. All writes are atomic.

## After the deploy that ships RFC-129 (one-off)

1. **Dry run of the reference backfill.** It should list mostly `would_write` and some
   `skip_no_exact_original` (episodes whose archive copy is a later re-download):

   ```bash
   $C run --rm --no-deps pipeline-llm python -m podcast_scraper.audio_sync build-references \
     --corpus /app/output --rclone-remote audioarchive --limit 20
   ```

2. **Backfill all references**, detached. It pulls each original from the archive (about 2,600
   files, about 156 GiB) into a temporary folder that is deleted after each episode, so it runs for
   hours without filling the disk. It is resumable: an episode that already has a reference is
   skipped. The references themselves take about 2.6 MB per hour of audio (version 2 keeps a long
   window at every checkpoint): about 7 GB for the corpus, so check `df -h /` has that free.

   ```bash
   $C run -d --no-deps --name audio-refs pipeline-llm python -m podcast_scraper.audio_sync \
     build-references --corpus /app/output --rclone-remote audioarchive --write
   docker logs -f audio-refs        # one line per episode; a summary line at the end
   ```

3. **Check:**
   - `curl` the api (signed in) for an episode's `/api/app/episodes/<slug>/audio-reference`: 200
     with `checkpoints`.
   - On a phone, play an episode from a host that inserts ads: the transcript highlight stays on the
     voice across an ad break, and a Moment from it starts on its words.

Nothing runs daily: references are made once per episode, and each player measures its own stream.
A reference is format version 2 (`"version": 2`, long windows). The player still syncs with a
version 1 reference, but downloads far more of the stream after a changed break; rebuild one with
`build-references --force --path <metadata path>`.

## Diagnosis (manual)

- `drift-check --write` lists the episodes whose served length differs from what we transcribed.
- `build-maps --write` then writes a server time map for them, from their references: the player's
  fallback when it cannot measure its stream. A map is a snapshot of the stream at that moment; the
  ads change within hours.

## Verification

- `find <deploy-dir>/corpus -name '*.audioref.json' | wc -l` grows during the backfill.
- Pipeline logs after an ingest run: `audio references (<run>): {'written': N, ...}`.

## Troubleshooting

- **`skip_no_exact_original`:** the archive holds only a re-download (provenance
  `byte_identical_to_transcribed_audio: false`), or no audio. No reference is possible; the
  player falls back to the server map, if any.
- **`skip_no_reference` (build-maps):** the episode has no reference yet; run the backfill.
- **`skip_poor_match` (build-maps):** under half of the reference's checkpoints were found in the
  stream as served now (the host serves another edit, or the stream failed). That episode gets no
  map.
- **`skip_stream_unreadable`:** the host did not answer within 15 minutes.
- **Undo:** delete `audio_time_maps.json` (maps), or the `*.audioref.json` files (references).
  The api and the player treat a missing file as "nothing measured".
