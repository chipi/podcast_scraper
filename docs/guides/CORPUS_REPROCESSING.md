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

## Reading a running reprocess — what the signals actually mean

Added 2026-09-24 after a repair where every one of these was misread at least once,
costing hours. Each entry is a thing that looks like evidence and is not.

### The corpus lock file's existence means NOTHING

`filelock` never unlinks `.podcast_scraper.lock` on release — it drops the `flock` and
removes the `.holder` sidecar. So the file is present, 0 bytes, after **every run that has
ever succeeded**. It was read as "locked" three times over 2026-09-22..24, diagnosed as
stale, and hand-removed on prod. It was never stale.

```bash
# WRONG — always true after the first run, tells you nothing
[ -e corpus/.podcast_scraper.lock ] && echo "locked"

# RIGHT — probes the flock, which is the only authority
docker exec compose-api-1 python3 -c "
from podcast_scraper.utils.corpus_lock import corpus_lock_state
import json; print(json.dumps(corpus_lock_state('/app/output'), indent=2))"
```

`held: true` means a live process holds it; the `holder` names pid/hostname/start. A
genuinely contended lock is normal — wait, don't delete.

### A work-list OVERRIDES `--skip-existing`; you do not need to delete anything

`--reprocess-episode-ids` forces each listed episode past `skip_existing` on its own
(`_force_reprocess_for_source`). A plan to "delete the artifacts so the re-ingest sees
them as new" is both unnecessary **and self-defeating**: on the
`--reprocess-existing-only` path (which `--reprocess-episode-ids` implies) the episode set
is built FROM on-disk metadata, so deleting a guid's metadata removes it from the
selectable universe and the run logs *"none of the N listed episode(s) are in this feed's
corpus"* and exits 0.

Confirm the override fired — this line per episode is the proof:

```text
[1] [#925] forcing re-transcription + diarization (reprocess-source=None): <path>
```

If it is absent, skip-existing won and the run is a no-op regardless of exit code.

### A `full` run writes a NEW run dir; `relabel_only` writes back into the old one

`_existing_metadata_path_for_reprocess` returns a path only for the
never-transcribe stages (`relabel_only`, `rediarize_only`, `retranscript_only`) and
`rederive_only`. A `full` run creates `run_<ts>/` fresh, and newest-run-per-episode makes
it win for every reader — which is why repair-first / cleanup-after is safe and you never
need to mutate the damaged records in place.

### Per-episode resolution: one line decides whether the run is doing what you asked

```text
reprocess: [N] (on-disk M) 'Title' -> <file>.txt                     resolved from its own record
reprocess: [N] (on-disk M) 'Title' -> <file>.txt [ADOPTED from run_X] its own run had none
reprocess: [N] (on-disk M) 'Title' -> no transcript on disk — will be transcribed by this run
reprocess: [N] (on-disk M) 'Title' -> NO OWN TRANSCRIPT ... SKIPPED   only for relabel/rediarize/retranscript
```

A reprocess OVERWRITES the transcript it picks. Before 2026-09-23 a missing transcript
fell through to a `{idx} - *.txt` glob, and since an on-disk idx is unique only inside one
`run_*` dir, that handed 10 serving episodes **another episode's** transcript with GI/KG
rebuilt from the wrong words. It now refuses instead.

### The processing-loop line states its own verdict — believe it, not the raw counts

```text
INFO  Processing loop: WORKING — 2 running, 3 queued; longest in flight 412s ...
ERROR Processing loop: STUCK — every job is accounted for and nothing is in flight ...
```

`WORKING` at INFO is ordinary waiting and is **not** a problem, however long it runs.
Only `STUCK` at ERROR is. The pre-2026-09-24 version printed raw counts and was misread in
both directions — 366 lines of normal waiting counted as wedges (producing a "17.3h of
dead time" figure that was wrong by >2x), and a real wedge called healthy for an hour.

### `docker logs`, never the `logs/reprocess-*.log` file

The file is written through the GitHub runner's ssh session. Cancel the run (or hit the
6h cap) and the file stops growing **while the container keeps working**. A frozen file is
not a stalled job.

```bash
docker logs --since 10m podcast-reprocess-<run-id> | tail -40
```

### The GitHub runner is an observer, not the job

`ubuntu-latest` is hard-capped at 6h of job execution; `timeout_minutes` cannot raise it.
The remote container is launched with its own detached watchdog and **outlives a cancelled
run**. So: a `completed/cancelled` run does not mean the work stopped, and killing the run
does not release the corpus lock. Stop the container explicitly.

The inverse misreads the same surface and is worse, because it reads as good news:
`completed/success` on *a* reprocess run does not mean *your* batch finished. Successive
batches are separate runs, and a watcher started against an earlier one keeps reporting that
one's ending forever. On 2026-09-24 a watcher's `BATCH ENDED: completed/success` was taken
as the 284-episode job finishing in three minutes; it belonged to a batch that had ended at
06:31, while the real job was 40 minutes in and on its third feed.

The container name carries the run id, so the check costs nothing — do it before believing
any status line:

```bash
docker ps --filter name=podcast-reprocess --format '{{.Names}}  {{.Status}}'
# podcast-reprocess-35979952564  Up 40 minutes (healthy)
#                   ^^^^^^^^^^^ must be the run id you are reading about
```

A running container outranks every status surface. GitHub reports on the *runner*; only the
container reports on the *work*.

### Prod questions go to the PROD MCP only

`mcp__claude_ai_Close_Listening__*` reads the live corpus. The repo's `.mcp.json` server
(`make serve-mcp`) reads your **local working tree** and will answer `feeds: []` /
`not_found` / `ImportError` about a prod episode — an empty answer that looks like a
finding. See AGENTS.md § "MCP servers — PROD vs DEV".

### The search index is NOT rebuilt by a reprocess

LanceDB writes by merge-upsert and never deletes, and the pipeline does not drop stale
chunks. New content is indexed, but superseded chunks from the old (wrong) artifacts
survive until a rebuild:

```text
reindex-prod.yml  mode=rebuild  with_clusters=true
```

`with_clusters` matters when GI was regenerated — `search/topic_clusters.json` derives
from it. Shares the `prod-corpus` lock, so it cannot overlap a reprocess.

**Check it, do not assume it in either direction.** This heading is the safe default, not a
law: `maybe_index_corpus` is gated on `vector_search` / `skip_auto_vector_index` and NOT on
`pipeline_stage`, so a `relabel_only` run is not excluded by its stage, and `_finalize_pipeline`
does call it against the corpus finalize dir. Whether it actually runs for YOUR batch depends
on the layout the run resolves (`_corpus_finalize_dir_for`) and on where finalize happens in a
multi-feed spec. Measured on the 2026-09-24 284-episode multi-feed batch: three feeds closed
with `vector_index_sec: 0.0` each, `topic-clusters: skipped (missing LanceDB index at
<run-local path>)`, and the corpus index untouched since 09:11:16 — two minutes BEFORE that
batch started. So for that shape, no.

The two cheap measurements that settle it, rather than reasoning about the code:

```bash
# 1. has the serving index been written since the batch began?
docker exec compose-api-1 stat -c %y /app/output/search/lance_index
# 2. did any feed spend time indexing?
docker logs <container> | grep -o '"vector_index_sec": [0-9.]*' | sort -u
```

The index has **no run-dir column**, so staleness cannot be tested by path. Test it by CONTENT
— pull each episode's rows from the `segments` table and check the chunk text still appears in
that episode's current on-disk transcript. A chunk the repair did not regenerate survives
holding the wrong episode's words and will not be found there. Run it from `compose-api-1`
(`/app/output/search/lance_index`, tables `aux` / `insights` / `segments`); the image has
pyarrow but **no pandas**, so use `to_arrow()`, and do not install anything to answer this.

Done that way on the 10 repaired episodes: 361 indexed chunks, 0 stale. Their index was
refreshed — the general rule above did not apply to them, which is exactly why it gets
measured rather than assumed.

### Enrichments are a separate pass

`insight_sentiment` / `insight_density` come from the RFC-088 enricher executor, not the
pipeline. A fresh run dir has none until a re-enrich runs.
