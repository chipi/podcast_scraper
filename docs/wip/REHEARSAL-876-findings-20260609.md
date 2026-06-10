# #876 re-diarization — rehearsal findings (2026-06-09)

Dry-run of `docs/wip/RUNBOOK-876-corpus-rediarization.md` against a local copy of
the prod corpus (`.test_outputs/manual/prod-v2/corpus`, 100 episodes, built
2026-05-05). Treated as a rehearsal for the real prod migration. Two real
issues surfaced; one fixed in-tree, one needs a scoping decision before any full
run.

## Corpus facts (source of truth used)

- 110 episode metadata files = **99 `whisper_transcription` + 11 `direct_download`**.
- 2 run-dirs per feed (the cross-run dupes memory already flagged).
- Whisper episodes span **2026-01-28 → 2026-05-05**, ~**11 per feed** across 9
  feeds (the 10th feed is all direct-download).
- DGX health gate passed: Whisper large-v3 (`:8000`) + pyannote 3.1 (`:8001`).
- Backup taken: `~/corpus_backup_20260609-185035.tar.gz` (745 MB).

## Finding 1 — FIXED: per-feed Config rebuild drops DGX routing fields (ADR-096)

**Symptom.** First pilot failed config validation on **every** feed:
`transcription_fallback_provider is required when transcription_provider is
tailnet_dgx_whisper (ADR-096)`. Run exited **rc=0** despite all 10 feeds erroring
(a second, latent issue — total config failure reports success).

**Root cause.** `cli._build_config` materialises a per-feed `Config` from an
**explicit allowlist** payload. The ADR-096 / #814 / #926 DGX routing fields were
added to the `Config` model + profile but never added to the "no argparse flag,
carry from `--config` YAML" loop (cli.py ~4165, the #646 profile-completeness
audit). So the per-feed rebuild kept `transcription_provider=tailnet_dgx_whisper`
(it has an argparse flag) but dropped:

- `transcription_fallback_provider` → ADR-096 contract fails
- `diarization_provider` (`tailnet_dgx`) → would have **silently degraded
  diarization to in-process pyannote** even if the contract had passed
- `dgx_diarize_port`, `dgx_diarize_model`, `dgx_whisper_model`

The single-shot `Config.model_validate(profile)` path was fine — only the
per-feed rebuild (the path `--feeds-spec` / `make redo-diarization` uses) was
lossy. That's why stack-tests using a single resolved config didn't catch it.

**Fix.** Added the 5 fields to the carry-over loop in `cli._build_config`.
After the fix: 0 validation errors; ownership line confirms
`transcription=tailnetdgxwhispertranscription(api)` **and**
`diarization=tailnetdgxwhispertranscription(api)`.

**Follow-up worth filing:** a fully-failed multi-feed run returning rc=0 is its
own bug — caller can't tell a no-op from success.

## Finding 2 — OPEN (needs decision): runbook full-run has no episode bound

`make redo-diarization` runs the CLI with `--skip-existing --reprocess-source
whisper_transcription` and **no `--max-episodes`, no date filter,
`transcribe_missing=True`**. With this profile `max_episodes` resolves to
**None**.

The corpus was built ~2026-05-05 with a **per-feed cap of ~10** (newest). The
live feeds now carry **199-300+ episodes each**. So the full run as written
would, per feed, fetch and DGX-transcribe **every** available episode:

- existing 99 whisper episodes → re-diarized (the #876 goal) ✓
- existing 11 direct-download → skipped ✓
- **every feed episode published since 2026-05-05 AND every older un-ingested
  episode → newly ingested + DGX-transcribed** ✗ unintended

Estimated blast radius: **~2,000-3,000 episode cascades** instead of 99.
Confirmed empirically: the 3-episode pilot selected the **3 newest feed
episodes** (`no transcript; downloading media`) — i.e. brand-new episodes, not
the existing whisper set. `--max-episodes N` truncates to the newest N of the
*live* feed before `--reprocess-source` matching, so it does not target the
on-disk episodes either.

### Scoping options for the existing-99 re-diarization

| Option | Mechanism | Trade-off |
| --- | --- | --- |
| A. Date-scoped | add `--until 2026-05-06` + per-feed `--max-episodes ~11` | Closest to "re-diarize what's there"; minor risk of feed-order drift including/excluding a few |
| B. Re-diarize + backfill | `--until <today>` + raised cap (20-30/feed) | Merges #876 with the parked cap-raise; bigger DGX cost; moves toward the ~200 goal |
| C. Strict existing-only | new tooling: enumerate on-disk GUIDs, reprocess exactly those, no feed-driven new ingest | Safest semantics; requires a code change (new flag/mode) |

Pilot also does **not** exercise Bug3 (re-diarization shifting offsets vs
*existing* GI) because it processed new episodes with fresh, aligned GI. A
faithful pilot must target an existing whisper episode — which depends on the
scoping decision above.

## Resolution (2026-06-09)

- **Finding 1** fixed in `cli._build_config` (carry-over loop). Post-fix pilot
  ran clean against DGX (ownership line confirms DGX for transcription AND
  diarization).
- **Finding 2** resolved by building a **strict existing-only migration mode**
  (issue #946, branch `feat/946-existing-only-rediarization`): new
  `--reprocess-existing-only` flag + `collect_existing_guids` helper +
  `prepare_episodes_from_feed` filter + `make migrate-diarization`. The episode
  set is defined entirely by on-disk GUIDs; new feed items are dropped; caps are
  ignored. Operator framing: a migration of existing data while ingestion is
  paused — the corpus never grows.
- **Validated end-to-end** (CLI `--dry-run` on the npr feed): live feed now has
  **355 items**; migration mode kept the **10 on-disk episodes**, dropped 345
  new, 0 rolled off → "Episodes to process: 10 of 355". Without the flag that is
  345 unintended ingestions on a single feed.
- Tests: 17 scraping (4 new migration-mode + 3 helper) + 57 parser green;
  config/cli regression surfaces green; flake8/black/mypy clean.

## Status

- Tooling complete + validated on `feat/946-existing-only-rediarization` (uncommitted).
- The actual #876 migration run (`make migrate-diarization` against the real
  corpus through DGX) is the next operational step — gated on operator go.
- Backup retained: `~/corpus_backup_20260609-185035.tar.gz`.
