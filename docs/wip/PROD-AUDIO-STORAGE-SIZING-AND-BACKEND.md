# Prod audio storage — sizing, cost, and write-path backend

**Status:** Active (plan) · **Date:** 2026-07-17 · Part of the Goal-1 prod go-live work.
**Tracking issue:** [#1199](https://github.com/chipi/podcast_scraper/issues/1199)
— configurable local-vs-remote object-storage backend (audio archive primary use case).

## Why this exists

Prod (`prod-podcast`) currently stores **no audio** — the pipeline transcribes,
then discards the media. Reprocessing (new diarization models, re-transcription,
bitrate experiments) therefore requires **re-downloading** every episode, which
is what we did to build the local `prod-v2` corpus. Going forward we want to
**persist audio** so reprocessing is a local read, not a re-fetch (and so we're
resilient to feeds pruning old episodes — dynamic-ad-insertion feeds like acast
mutate/expire).

This note sizes that storage, prices it, and picks the write-path backend.

## Ground-truth sizing (measured, not estimated)

Measured from the real re-downloaded audio in
`podcast_scraper-FUTURE/.test_outputs/manual/prod-v2/corpus` (90 episodes, 9
feeds) and cross-checked against prod corpus metadata (`episode.duration_seconds`
+ the `size=` param embedded in `content.media_url`):

| metric | value |
| --- | --- |
| audio per episode (mean) | **48 MB** (median 46, min 1.2, max 152) |
| duration per episode (mean) | **45 min** (~128 kbps MP3 ≈ 0.96 MB/min) |
| current prod corpus | **100 unique episodes** (dedup by `content.media_id` sha256; 110 metadata files incl. re-runs), 10 feeds |
| current audio footprint | **~4.8 GB** (100 × 48 MB; measured 90-ep sample = 4.23 GB) |

**Planning constant: 48 MB/episode** (use 50 MB for headroom).

### Growth projection

10 feeds today, mixed cadence (daily news → weekly interview). Bands:

| cadence assumption | new eps/mo | audio/mo | audio/yr |
| --- | --- | --- | --- |
| light (weekly-only) | ~43 | ~2 GB | ~25 GB |
| **mid (mixed)** | **~130** | **~6.2 GB** | **~75 GB** |
| heavy (all-daily) | ~300 | ~14 GB | ~170 GB |

Plus one-off backfill: deepening all 10 feeds to ~100 eps each = 1,000 eps ≈
**48 GB**.

## Cost (verified July 2026 — Hetzner)

Hetzner raised Cloud Volume prices on 2026-04-01 (€0.044 → **€0.0572/GB/mo**),
which widens the gap vs the flat-rate Storage Box.

| option | product | rate | 100 GB | 500 GB | 1 TB |
| --- | --- | --- | --- | --- | --- |
| Block Volume | Hetzner Cloud Volume | €0.0572/GB/mo | €5.72/mo | €28.60/mo | €57.20/mo |
| **Object store** | **Hetzner Storage Box BX11** | flat 1 TB | — | — | **€3.20/mo** |
| Object store | Hetzner Storage Box BX21 | flat 5 TB | — | — | €11.40/mo (5 TB) |

*(Prices +VAT; Storage Box includes unlimited traffic, hourly billing capped
monthly, no setup fee, no minimum term.)*

**A 1 TB Storage Box (€3.20/mo) holds ~21,000 episodes (≈ 13 years of
mid-cadence growth) and is ~18× cheaper than a 1 TB Volume.** Audio is a *cold*
archive (reprocess-only, never hot-served), so it does not need block-device
performance.

## Capability gap (why this needs a decision, not just a bucket)

The pipeline has **no native remote-storage backend** — grep of `src/` for
`boto3|s3|rclone|sftp|fsspec|smb` is empty. Audio persistence is **local
filesystem only**:

- `audio_cache_enabled` / `audio_cache_dir` (default `.cache/audio`, kept
  external to the corpus so corpus backups stay lean)
- `audio_cache_in_corpus` + `corpus_media_link_mode` (`copy` | `hardlink` |
  `symlink`) — places audio under `corpus/media/`

Storage Box speaks **SFTP/SSH, Samba/CIFS, WebDAV, rsync, BorgBackup** (not S3).
Hetzner **Object Storage** (a separate product) is S3-compatible if we prefer
`boto3`.

Corpus durability is already handled separately: `backup-corpus-prod.yml` tars
`/srv/podcast-scraper/corpus` and uploads it as GitHub Release snapshots on the
backup repo (7 daily + 4 weekly). That's fine for the 116 MB corpus but a poor
fit for GBs of audio (GH release asset limits + churn), so audio needs its own
path.

## Write-path options

Operator has green-lit **evolving the write path** if it means better Storage
Box support (2026-07-17). So we are not limited to a host mount.

| # | approach | code change | pros | cons |
| --- | --- | --- | --- | --- |
| A | **Mount** the Storage Box (SSHFS / rclone-mount / CIFS) → pipeline writes to the mount path | none | zero code; go-live today | host-mount fragility; network on the write path; a stalled mount can hang the pipeline |
| B1 | **Native SFTP backend** (paramiko) for the audio cache / media placement | medium | no host mount; direct to Storage Box; testable | new dep (paramiko); we own ret/timeout logic |
| **B2** | **rclone-backed backend** (shell out to `rclone copy`) | medium | one backend covers Storage Box **and** S3/Object Storage; battle-tested retries; no host mount | rclone binary dependency on the host |
| B3 | **S3 backend** (boto3) → Hetzner **Object Storage** | medium | S3 semantics; native `boto3` | Object Storage is per-GB (pricier than Storage Box); another product to run |
| C | **Decoupled sync** — pipeline writes audio to local cache; a post-run `rclone`/`rsync` ships it to the box; reprocess pulls back | low–medium | keeps hot path local; box is pure archive | audio not durable until the sync runs; two-step |

## Recommendation

1. **Storage: Hetzner Storage Box BX11 (1 TB, €3.20/mo)** for the audio archive.
   18× cheaper than a Volume, ample for a decade, cold-archive fit.
2. **Write path: evolve to a pluggable remote audio backend, `rclone`-backed
   (B2).** One backend serves Storage Box now and S3/Object Storage later if we
   ever move; avoids host-mount fragility. If we need go-live *before* the code
   lands, mount (A) is the stopgap — but treat it as temporary.
3. **Dedup by `content.media_id` (sha256)** so re-runs don't re-store the same
   audio — the metadata already carries it.
4. **Corpus stays as-is** (hot, 116 MB, already snapshot-backed). The boot-disk
   risk for the *corpus* is mitigated by `backup-corpus-prod.yml`; a dedicated
   corpus Volume is optional, not urgent.

## Open decisions (operator)

- [ ] Storage Box (SFTP/cheap/flat) vs Object Storage (S3/boto3/per-GB) — this
      note recommends Storage Box.
- [ ] Backend now (B2) vs mount-stopgap (A) for the first live cut.
- [ ] Retention: keep **all** audio forever (dedup'd) vs a rolling window
      (e.g. keep N months, re-fetchable on demand).
- [ ] Whether to promote the backend design to an RFC/ADR before implementing
      (new dep + a storage boundary = design-change territory).

## Implementation status (#1199)

**Write path — DONE** (on `production`):

- `utils/storage_backend.py` — `StorageBackend` interface + `LocalStorageBackend`
  (byte-compatible with the existing #947 layout) + `RcloneStorageBackend`
  (rclone-backed; transport chosen in `rclone config`, credentials never in our
  config). Injectable runner so CI drives rclone with an in-memory fake — no real
  remote, no binary in CI.
- Config: `audio_storage_backend: local|remote` + `audio_remote_rclone_remote` /
  `audio_remote_base_path` / `audio_remote_rclone_bin`. A `remote` backend with no
  remote name **fails loud at config load** (the Deepgram trap, #1195); a missing
  rclone binary fails loud at construction. Per-episode upload/download is
  best-effort (ERROR-logged, never silent).
- `audio_cache.resolve_backend` / `fetch_into` / `store_via` wired into the
  download path in `episode_processor.py`: archive on download, fetch-back on
  cache-hit (so the "avoid re-fetch" loop closes, not write-only). `local`
  default → **zero behaviour change**.
- Keyed by the sharded GUID digest (`sha256/aa/bb/<digest><ext>`), dedupe by
  existence. Tests: `test_storage_backend.py`, `test_audio_cache.py`
  (backend layer). Full lint/type/policy green.

## Read-back access (the two use cases)

Two flows, both reusing the backend + the archive-aware download path:

- **Flow A — pull audio to laptop: DONE.** `podcast-scraper archive pull` resolves
  episode→guid→key from corpus metadata (`episode.guid` is stored) and downloads
  via the backend with human names (`<feed>/<NNNN> - <title>.<ext>`); selectors
  `--all/--feed/--episode/--since`, `--dry-run`, `--force`; source `--rclone-remote`
  or `--local-root`. Standalone (no full pipeline config). Also enables
  reprocess-on-laptop (set `audio_storage_backend=remote`).
- **Flow B — one-off reprocess in prod: pipeline half DONE.** `--reprocess-existing-only`
  re-runs episode processing, which calls `download_media_for_transcription` →
  `audio_cache.fetch_into` → pulls each episode's audio from the remote before
  re-diarize/re-transcribe. No feed re-fetch. Remaining plumbing below.

## Landed (build pieces 2 & 3)

- **rclone in the pipeline image** — `docker/pipeline/Dockerfile` installs the
  `rclone` binary (harmless when unused; local backend is the default).
- **`reprocess-prod.yml`** — confirm-gated (`PROD_REPROCESS`) `workflow_dispatch`
  that joins the tailnet, SSHes to prod, and runs
  `--reprocess-existing-only --reprocess-source <…>` reading audio from the remote
  archive. Plus the manual `docker exec` recipe.
- **Recipe** — `docs/recipes/prod-audio-archive.md`: provisioning, the
  **password + env-var** rclone injection (one obscured-password secret, no mount,
  no compose change), and both read-back flows end to end.

## Landed — provisioning as code

- **`infra/terraform/storage_box.tf`** — `hcloud_storage_box "audio_archive"`,
  gated on `audio_storage_box_type` (empty = not provisioned, so PROD is
  unaffected until opt-in). Cloud-native, TF-managed (provider 1.62 has it GA);
  SFTP enabled, weekly snapshot, delete-protected. Outputs `audio_storage_box_server`
  + `_username` for the rclone config. `tofu validate` green. (BX11 = current
  offering: 1 TB / ~€3.20/mo; bx11/bx21/bx31/bx41 map to `storage_box_type`.)

## Remaining (operator action)

- **`tofu apply`** with `audio_storage_box_type = "bx11"` +
  `TF_VAR_audio_storage_box_password` (a per-instance apply — operator's call).
- `rclone obscure` that password → GH secret `PROD_RCLONE_STORAGEBOX_PASS`; add
  the `RCLONE_CONFIG_HETZNERBOX_*` lines (host/user from `tofu output`) to the
  `deploy-prod.yml` `.env` staging; set `audio_storage_backend: remote` in the
  prod profile. All in the recipe.
- Corpus-media hardlink source is local-only today (remote → falls back to copy);
  revisit if remote reprocess needs the link optimisation.
- Optional RFC/ADR to ratify the storage boundary + the rclone system dep.
- Tier-2 sizing guard so the footprint projection stays honest as feeds grow.
