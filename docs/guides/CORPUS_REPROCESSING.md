# Corpus reprocessing runbook

**How to rebuild an existing corpus's artifacts (diarization, cleaning, GI, KG,
enrichment) by re-running our real, profile-driven pipeline — never a bespoke
script.**

## Purpose & the one rule

You have a corpus on disk and you want to regenerate some or all of its derived
artifacts — because the code improved (better diarization, new cleaning like the
`#1188` cross-promo removal, a GI/KG schema change), or because the existing
artifacts are wrong (mislabeled speakers). Reprocessing does that **through the
same pipeline that produced the corpus**, scoped to the episodes already on disk.

> **The rule: reprocess only via the profile-driven `podcast_scraper.cli`.**
> Everything — transcription provider, diarization provider, speaker-detection NER,
> summary/GI/KG models — is decided by the **profile** (`--config <profile>.yaml`).
> Do **not** hand-assemble stages in a script or point tools at the corpus directly;
> that reproduces combinations that don't exist in production (e.g. spaCy NER when the
> profile says Gemini) and quietly diverges from how the pipeline actually runs. If a
> capability is missing, add it to the pipeline (a `pipeline_stage` mode), don't fork it.

## Which mode do you want?

| Goal | Mode | Re-ASR? | Re-diarize? | Command |
| --- | --- | --- | --- | --- |
| **Fix speakers / full rebuild** (correct diarization + named screenplays, current ASR, re-clean, re-extract) | full reprocess | yes | yes | `make migrate-diarization` |
| **Re-name speakers only** (re-resolve names on the FROZEN diarization; re-render + cascade GI/KG) | `--pipeline-stage relabel_only` | no | no | see [Re-name / re-diarize only](#re-name-re-diarize-only) |
| **Re-diarize only** (fresh diarization aligned to the existing ASR; no re-transcribe; re-name + cascade) | `--pipeline-stage rediarize_only` | no | yes | see [Re-name / re-diarize only](#re-name-re-diarize-only) |
| **Re-extract only** (reuse transcript + diarization; re-run cleaning + GI + KG on the existing base) | `--pipeline-stage rederive_only` | no | no | see [Re-extract only](#re-extract-only) |
| **Enrich gaps** (fill missing corpus-level enrichments) | `cli enrich` | no | no | `make enrich CORPUS=<corpus>` |

Key fact that drives the choice: the three stages of the transcript — **ASR text**,
**diarization** (which voice), and **naming** (who) — are now **decoupled** for
reprocessing. `relabel_only` re-resolves names on the frozen diarization; `rediarize_only`
re-diarizes the audio and aligns the fresh voices to the existing ASR (no re-transcribe);
a **full** reprocess re-transcribes and re-diarizes together. Pick the narrowest stage that
covers what actually changed — each freezes everything below it, so a single-variable
reprocess is a clean before/after gate. (Note: `rediarize_only` needs the source audio and
does not apply to direct-download transcript feeds, which never went through ASR.)

---

## Full reprocess — `make migrate-diarization`

Re-runs, per on-disk episode, the **full cascade**: transcribe → diarize →
screenplay → clean → GI → KG → bridge → search index, then re-derives corpus-wide
`SPOKEN_BY` edges. Everything is profile-driven.

```bash
make migrate-diarization \
  CORPUS_DIR=<corpus> \
  PROFILE=config/profiles/cloud_with_dgx_primary.yaml
```

> ⚠️ **`migrate-diarization`, NOT `redo-diarization`.** They differ by one flag with a
> huge consequence:
>
> | target | `--reprocess-existing-only`? | what it processes |
> | --- | --- | --- |
> | `make migrate-diarization` | **yes** | the corpus's on-disk GUIDs — **correct** |
> | `make redo-diarization` | no | scrapes the **live feed** and processes the newest items (a 583-episode feed → wrong episodes) |
>
> Always confirm the log line reads `Existing-only re-diarization: kept N, dropped …
> new feed item(s)`. If you see episodes downloading by title from the feed, stop.

The two things that silently break a re-diarization:

> ⚠️ **Clear `.cache/transcripts` first** (`rm -rf .cache/transcripts`). The transcript
> cache is keyed by audio hash and stores the *already-formatted* (post-diarization)
> screenplay. A warm cache short-circuits transcribe→diarize→format, so the run reuses
> the **old** diarization and re-diarization becomes a silent no-op (`Transcript cache
> hit … transcribe_sec=0.0`). Clearing the dir is reliable; `transcript_cache_enabled:
> false` in the profile does **not** always take effect through the CLI merge.
>
> ⚠️ **Keep the machine awake for DGX runs** (`caffeinate -i …`, mains power). If it
> sleeps mid-run the tailnet drops and every DGX diarize POST fails with `Connection
> reset by peer`, falling back to slow in-process pyannote.

### Full-reprocess procedure

1. **Health gate** (abort if a required service is down). For DGX profiles, check the
   Whisper (`:8000`) and pyannote (`:8001`) endpoints on the tailnet host before starting.
2. **Backup** — the reprocess **overwrites** transcripts/diarization/GI/KG/index:

   ```bash
   tar -czf "$HOME/corpus_backup_$(date +%Y%m%d-%H%M%S).tar.gz" \
     -C "$(dirname "$CORPUS_DIR")" "$(basename "$CORPUS_DIR")"
   ```

3. **Pilot 2–3 episodes on a COPY** before the full run.
   > ⚠️ **`--max-episodes` is ignored under `--reprocess-existing-only`** — it processes
   > *all* on-disk GUIDs. To pilot a subset, **trim the corpus copy** to the episodes you
   > want (delete the other `metadata/*.metadata.json` + `transcripts/*`), then run
   > existing-only; the GUID scan picks up exactly what remains.

   Pilot acceptance: ≥2 distinct **named** `Name:` markers on multi-speaker episodes;
   `diarization_num_speakers` matches the known cast; GI `Quote` nodes carry
   `timestamp_start_ms`; `SPOKEN_BY` edges present after `enrich-edges`; no offset-guard
   warnings.
4. **Verify offsets** (`make verify-gil-offsets-strict CORPUS_DIR=<pilot>`). Re-diarization
   shifts char offsets; GI is rebuilt against the new ad-free base so quotes re-derive
   exactly. A mismatch means GI was not rebuilt against the new transcript — investigate,
   do not proceed.
5. **Full run** (`make migrate-diarization`), watch the scoping log line and DGX
   fallback breadcrumbs. Budget ~6–7 min/episode for large-v3 + pyannote.
6. **Post-run**: offsets clean, spot-check ~5 episodes (named screenplay + `SPOKEN_BY` +
   KG entities), episode count unchanged, a vector search returns sensible results.
7. **Rollback** if needed: `rm -rf "$CORPUS_DIR" && tar -xzf <backup> -C "$(dirname "$CORPUS_DIR")"`.

---

## Re-name / re-diarize only

When the **ASR text is correct** but the speaker labels are not, you no longer need a full
reprocess:

- **`--pipeline-stage relabel_only`** — freeze the diarization (the `SPEAKER_NN` clustering on
  disk) and re-resolve only the *names* on it, then re-render the screenplay and cascade GI/KG.
  Use it after a change to the naming/roster logic. No audio, no ASR, no re-diarize.
- **`--pipeline-stage rediarize_only`** — freeze the ASR text and re-diarize the **audio** with the
  profile's diarizer (e.g. DGX pyannote `community-1`), align the fresh voices to the existing
  transcript, then re-name and cascade. Use it to test a different diarizer without paying for ASR.
  Needs the source audio; does **not** apply to direct-download transcript feeds (they never had
  ASR to align to).

```bash
.venv/bin/python -m podcast_scraper.cli \
  --config <profile>.yaml \
  --feeds-spec <corpus>/feeds.spec.yaml \
  --output-dir <corpus> \
  --pipeline-stage relabel_only   # or rediarize_only
```

Each stage freezes everything below it, so the run is a clean single-variable before/after gate
against the prior corpus.

---

## Re-extract only

When the speakers are already correct and you only changed a **downstream** stage
(GI), re-derive from the transcript already on disk with `gi-repair`:

```bash
# episode ids, one per line (blank lines and # comments ignored)
printf '%s\n' substack:post:207850718 > /tmp/ids.txt

.venv/bin/python -m podcast_scraper.cli gi-repair \
  --output-dir <corpus> \
  --config <profile>.yaml \
  --episode-ids /tmp/ids.txt \
  --force-healthy \
  --litellm-api-base http://<gateway>:4001/v1   # override the profile's pin if needed
```

Rewrites the SAME `gi.json` in place (diffable, no new run dir, no index split-brain),
calls no ASR provider, and writes a JSONL audit trail. Omit `--episode-ids` to sweep every
legacy-placeholder artifact instead. `--force-healthy` is required to overwrite an artifact
that is *not* a placeholder — that refusal is the safety property of the sweep, so it is
opt-in and logs a WARNING per artifact. Requesting an id that matches nothing is a
**failure** (non-zero exit), not a quiet no-op.

> **`--pipeline-stage rederive_only` was a silent no-op until 2026-09-01 — it now works.**
> It coerces `transcribe_missing=false` (correct: it must never call an ASR provider), but the
> only other exit from `process_episode_download` was the `if cfg.transcribe_missing and
> temp_dir:` gate, so no processing job was produced and the run exited **0** having re-derived
> nothing. Verified broken on 2026-08-16 and again on 2026-08-31.
>
> It now resolves the on-disk transcript directly (`_resolve_existing_transcript_for_enrich`)
> and queues the cascade, with no audio, no temp dir, and no ASR credential required. An
> episode with no transcript on disk is a loud failure, not a quiet skip. Guarded by
> `tests/unit/workflow/test_rederive_only_reuses_transcripts.py`.
>
> `rederive_only` re-runs cleaning + GI + KG. It does **not** re-run diarization or naming — use
> `rediarize_only` / `relabel_only` for those. `gi-repair` below remains the narrower tool when
> you want GI only, in place, with a diffable audit trail.

`gi-repair` re-derives **GI only** — not summary, not KG. If the *names* are wrong use
`relabel_only`; if the *diarization* is wrong use `rediarize_only`; only a wrong ASR
transcript needs the full reprocess.

---

## Enrich gaps

To (re)build only corpus-level enrichments (topic clusters, co-appearance, etc.):

```bash
make enrich CORPUS=<corpus> [WITH_ML=1] [PROFILE=<profile>.yaml]
make enrich-relational-edges CORPUS_DIR=<corpus>   # re-derive SPOKEN_BY
```

---

## Notes

- This is a **data operation**, not a code change — run it from `main` as a tracked
  operation, not bundled into a feature PR.
- If the corpus feeds eval, record quality-vs-baseline before/after.
- Profiles: DGX diarization → `cloud_with_dgx_primary.yaml`; cloud-only → `cloud_balanced.yaml`
  (Deepgram diarization, Gemini everything-else, `gemini-2.5-flash-lite`).
