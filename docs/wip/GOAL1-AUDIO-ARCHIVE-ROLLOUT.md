# Audio-archive Storage Box — rollout / transition plan (#1199)

How we take the audio-archive remote storage from "built, gated off" to "prod is
archiving audio to the Storage Box, and the new commands work." Sequenced **after
the DR drill is green** (it depends on the same prod box + backup safety net).

## What's built (and what's off)

- **Provisioning:** `infra/terraform/storage_box.tf` — `hcloud_storage_box`
  "podcast-audio-archive", **gated on `var.audio_storage_box_type`** (empty default
  ⇒ provisions **nothing**; prod has none yet). Types bx11/bx21/bx31/bx41. Access =
  SFTP (server/username/password); rclone uses the same password obscured.
- **Configurable backend:** `config.audio_storage_backend: "local" | "remote"`.
  `local` = filesystem (`audio_cache_dir` / `audio_cache_in_corpus`); `remote` =
  rclone via `audio_remote_rclone_remote` (e.g. `hetzner-box`). A misconfigured
  remote **fails loud at load**.
- **New commands:** `podcast_scraper archive pull --corpus <root> --dest <dir>`
  (fetch archived episode audio; selectors + `--dry-run`; only `pull` today) and
  the **`reprocess-prod.yml`** workflow (one-off prod reprocessing).

## Data-at-risk / backup status (the question to answer first)

| Data | Backed up? | Notes |
|---|---|---|
| **Corpus** (transcripts, GI/`.gi.json`, derivatives) | ✅ daily | `backup-corpus-prod.yml` tars `/srv/podcast-scraper/corpus` → private GH Release. This is the irreplaceable data. |
| **Audio archive** (Storage Box) | ❌ **not covered** | Separate remote store; the corpus backup does not touch it. Not provisioned yet, so nothing to lose *today*. |

**Is the archived audio irreplaceable?** Mostly no — audio is re-downloadable from
the source feeds ([[project_transcript_vs_audio_hosting]]: audio is bridge-only).
BUT the archive exists *precisely because* feeds drift / episodes vanish (dynamic
ad-insertion, deletions). So archived audio is **best-effort-irreplaceable**: fine
to lose while feeds are live, a real loss once a feed goes dark. → Decide a Storage
Box backup/snapshot policy in step 6 before it becomes the only copy.

## Rollout sequence (after DR-drill green)

0. **Gate:** DR drill green + corpus backup/restore verified (Phase 0) — the box is
   safe to change.
1. 🧑 **Provision the Storage Box** — set `TF_VAR_audio_storage_box_password`
   (high-entropy ≥32 chars, secret, never commit) + `audio_storage_box_type=bx11`
   (+ location). 🤖 pre-review `tofu plan`: MUST be a pure `+ hcloud_storage_box`
   add, **no** change/replace to `hcloud_server.prod` (same abort rule as Phase 4.1).
2. 🤝 **Verify connectivity** — configure the rclone remote (`RCLONE_CONFIG_<NAME>_*`
   from the SFTP creds) and `rclone lsd <remote>:` succeeds; a test put/get round-trips.
3. 🤝 **Backfill existing audio** → Storage Box, then **verify integrity**
   (per-file checksum / size compare). Keep the local copy until verified.
4. 🤝 **Flip the backend** — set `audio_storage_backend=remote` +
   `audio_remote_rclone_remote=<NAME>` in the prod profile. Fails loud if the remote
   is misconfigured, so it won't silently write audio nowhere.
5. 🤝 **Validate end-to-end** — a pipeline run writes new episode audio to the
   archive; `archive pull --corpus … --dest …` retrieves it (try `--dry-run` first);
   `reprocess-prod` uses archived audio for a one-off reprocess.
6. 🧑 **Archive backup policy** — decide + implement: Storage Box has Hetzner
   snapshot/sub-account options; OR accept "re-download from feeds" as the recovery
   path (document it). Do this **before** the archive is the only copy of anything.
7. 🤝 Only after 3–5 verified: prune the redundant local audio (if desired).

## New-command runbook (post-rollout)

- **Pull archived audio to the laptop:**
  `podcast_scraper archive pull --corpus <corpus_root> --dest ./audio [selectors] [--dry-run]`
  (backend from the profile; `--dry-run` previews the episode set).
- **One-off prod reprocess:** trigger `reprocess-prod.yml` (typed-confirm gate;
  `type: choice` profile; `environment: prod` + secret preflight).

## Where this sits in the go-live plan

A **follow-on phase after go-live** (post-Phase-6), not a blocker for orrery. It
touches only the podcast_scraper tenant's storage, not the shared edge. Prereqs:
DR drill green + corpus backup verified. Fold in as "Phase 7 — audio archive
transition" once Phase 5/6 land.
