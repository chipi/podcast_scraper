# Audio archive + cold storage — plan

**Status:** planning → execution starting at Phase 0
**Date:** 2026-08-19
**Owner:** Marko + Claude (prod agent)
**Trigger:** disk-low alert on prod after the 2026-08-18 unattended reprocess re-downloaded 127 episodes.
**Epic:** #1788 (umbrella for this initiative — master plan mirrored there).
**Issues:** #1786 (stop double-save), #1787 (per-run offload+evict), #1679 (enable the archive backend — existing epic), #1778 (disk-low escalation), #1657/#1655 (corpus enrichment repair).

---

## The model (agreed — this is the load-bearing decision)

Audio is a **purely-internal archive**, used only for **offline analysis** and **reproducible reprocessing**. It is **never served** to users — the player is bridge-only and always refers back to the original source; hosting/serving audio would breach T&Cs. Consequences:

- **Cold storage = the only long-term home** for audio.
- **Local audio = a disposable working copy** that exists only while a run is actively processing it. At rest, local audio → **~0**.
- Analysis/reprocessing later **pulls bytes back down from cold** on demand.
- "Just re-download from the publisher when needed" is a **trap**: dynamic ad-insertion re-encodes the file, so a re-fetch returns *different bytes* and silently corrupts WER/consistency comparisons against existing derivations. `audio-archive-provenance.jsonl` exists to flag this. The cold archive is what makes reprocessing **reproducible**.

There is **no serving latency requirement**, so there is **no hot-cache window and no retention TTL** to design — local audio is simply evicted after each run.

---

## What the data says (measured on prod, 2026-08-19, read-only)

| Thing | Finding |
|---|---|
| Disk `/` | 150G, 70% used, 44G free. The `<10%` alert (#1778) fired at run-peak; `--rm` cleanup freed it. |
| Audio total | **69 GB** = `media/` 52.7G + `audio-cache/` 12.6G + `.tmp_media/` 3.7G |
| Double-save | **Byte-identical confirmed** — `md5(media) == md5(audio-cache)`; cache filename is the content SHA256 |
| Dedup headroom | total audio 65.3G vs unique-by-size 43.7G → **~21.6 GB provably duplicate** |
| Archive | **Not wired** — no `rclone`, no `RCLONE_*` in `.env`, no `audio_storage_backend`, Storage Box unconfirmed |
| Tooling | `hardlink` (util-linux) is installed on the box |

**Three root causes (all verified in the repo):**
- (a) `audio_cache.py` — **no eviction anywhere** (append-only by construction).
- (b) **double-store** — `persist_episode_media=True` + `corpus_media_link_mode=copy`, un-overridden in `cloud_balanced.yaml`.
- (c) reprocess `--no-transcript-cache` **re-downloads** every touched episode.

---

## Phased plan

### Phase 0 — Immediate relief (on-box, no code, ~25 GB) — #1786
- Delete `.tmp_media/*` (3.7G scratch; nothing running).
- Retroactive content-hash **hardlink-dedup** of `media/` ↔ `audio-cache/` + cross-run dupes (~21.6G) via the installed `hardlink`. Zero data loss (hardlinks preserve every path).
- Result: 70% → ~54% used. Buys headroom (a precondition for the still-undone 32-ep repair).
- **Gate:** dry-run counts first, then operator go (on-box destructive-ish).

### Phase 1 — Stop the bleeding (app config) — #1786
- Set `corpus_media_link_mode: hardlink` in the prod profile. Future runs stop double-storing (same-fs confirmed). Rides with the cost-cap hotfix deploy.

### Phase 2 — Cold-storage backend — #1679 (existing epic)
- Provision Hetzner Storage Box (IaC `var.audio_storage_box_type`).
- **Deliver rclone creds to prod** (`RCLONE_CONFIG_*` via ADR-115 secret staging) — the substantive missing piece; nothing delivers them today.
- Install `rclone` in the pipeline image; set `audio_storage_backend: remote` in the prod profile.

### Phase 3 + 5 — Per-run offload + local eviction — #1787
- New **end-of-run stage** in `orchestration.py` (next to `enrich`/`reindex`), **per `run_dir` (feed-run)** granularity:
  1. rclone the run's audio to cold
  2. **delete all local audio** for that run (`media/` + `audio-cache/`); keep transcripts/derivations + provenance
  3. **idempotent/resumable** — an orphan sweep evicts local audio already in cold (covers crashed runs).
     ON DEMAND since 2026-08-21 (`archive sweep` / sweep-prod-audio.yml), NOT start-of-run: on the run
     path it cost one rclone round trip per episode across the whole corpus before the run applied its
     `--reprocess-episode-ids` work-list, so a one-episode repair stalled ~16 minutes in silence.
  4. optional disk-watermark floor (pause at ~85%)
- Retention dissolves: local = evict fully; cold = keep indefinitely (cheap).

### Phase 4 — Backfill (decay clock) — #1679
- Wire `archive backfill` (present in `src/podcast_scraper/archive/backfill.py`, CLI-only) as a **prod workflow**. Archive at-risk existing audio before episodes roll off feeds — this is the one item where waiting = **irreversible loss**.
- Schedule off the 05:37 UTC backup / reprocess / deploy windows.

### Cross-cutting (related, tracked elsewhere)
- 32-ep repair still 100% undone (zero overlap with the 127) — gated on disk headroom (Phase 0) + hotfix. See the incident doc / #1657.
- 127 reprocessed episodes lost enrichment (19% of corpus) — re-enrichment pass under #1657/#1655.
- Cost-cap hotfix (`hotfix/cost-containment-and-scope-gate`) must deploy before the queue resumes (#1757). `jobs.paused` stays SET until then.

---

## Guardrails (do NOT delete)
When evicting/dedup'ing, touch **audio files only**. Never:
- `.podcast_scraper/audio-archive-provenance.jsonl` — records re-fetch vs original bytes (load-bearing for reproducible reprocessing)
- `.podcast_scraper/corpus-art/` — artwork referenced by metadata `image_local_relpath`

Prune the child, never the parent (the corpus backup got this wrong once — commit 0ff7435d).

---

## What we need from the operator (external deps)
- **Hetzner Storage Box** — does one exist? If not, create it (or authorize `infra-apply` + confirm `audio_storage_box_type`).
- **rclone creds** (SFTP/WebDAV user+pass or key) → staged into prod via ADR-115.
- **Go for Phase 0** on-box ops (after dry-run).
- Branch strategy: app/infra on feature branches → `main`; `production` stays the prod-state checkpoint.

## Open decisions
- Phase 2: Storage Box as **primary** archive (local is thin/disposable) — agreed. Confirm the rclone remote type (SFTP vs WebDAV vs S3-compatible).
- Whether to add the optional disk-watermark floor in Phase 3.
