# Repair plan — the 10 episodes corrupted by the 2026-09-23 reprocess

Execute AFTER `e2dedbd17` is deployed. Every figure here was measured on prod this session;
anything unmeasured says so.

**This is revision 2. Revision 1 planned to delete 79 artifacts first. That was wrong twice over
and must not be resurrected — see "The route that was rejected" at the bottom.**

## What happened

A `relabel_only` batch ran with a resolver that, when an episode had no transcript of its own, fell
back to globbing `{idx} - *.txt` across the feed root. An on-disk idx is unique only inside one
`run_*` dir, so 16 episodes were handed a **different episode's** transcript and 10 reached the
serving corpus with `gi.json`/`kg.json` rebuilt from the wrong words — "Pax Silica: Inside the Trump
Administration's Tech Strategy" published Intel's CEO Lip-Bu Tan as a speaker, taken from
`0010 - Re-engineering the Semiconductor Supply Chain`. Fixed in `21bce7aa3` + `e2dedbd17`.

## Facts

| fact | measured |
| --- | --- |
| damaged serving episodes | **10** of 2002 |
| never transcribed at all (no vouched `.txt` + `.segments.json` for their guid, any run of their feed) | **9** of 10 |
| audio fetchable (HTTP 206, real Content-Range) | **10 of 10**, ~610 MB |
| **still present in their live feed** | **10 of 10** (guid found in feed XML; 179–355 items per feed) |

Feed membership is a hard gate, not a nicety: an aged-out episode is rebuilt by
`_synthesize_feed_item`, which has no enclosure, so `media_url` is `None` and scheduling bails at
`episode_processor.py:555-558` — **before** the audio-archive lookup. All 10 pass, so all 10 are
repairable by a full re-ingest.

## No deletion is required

`--reprocess-episode-ids` **overrides `--skip-existing` per episode**.
`_force_reprocess_for_source` (`episode_processor.py:3964-3989`) returns True for any episode whose
guid or episode_id is in the list — its docstring: *"forced back through download+transcribe,
overriding `--skip-existing` for it alone"*. Identity matching covers both guid and `substack:post:N`
forms. The direct-download path has the same override at `:4047-4058`.

So a `pipeline_stage=full` work-list dispatch, **with the corrupt artifacts left in place**, selects
the 10, forces them past skip-existing, downloads audio and runs ASR.

A full run writes to a **fresh** run dir: `_existing_metadata_path_for_reprocess` returns a path
only for `relabel_only` / `rediarize_only` / `retranscript_only` / `rederive_only`
(`metadata_generation.py:3690-3702`), and `full` means no `--pipeline-stage` flag
(`reprocess-prod.yml:347-351`). Newest-run-per-episode then makes the new copy win for every reader.
That is what makes repair-first / cleanup-after safe.

## Sequence

### 1. Preconditions
- `e2dedbd17` in the running image. Pass `image_sha` **explicitly** to `deploy-all-prod`; blank
  resolves "newest published", which may not be this commit.
- No reprocess container. **Both halves of the original wording here were wrong.** Corrected in
  place rather than quietly deleted, because the same false claim also reached a runbook:
  - "`corpus/.podcast_scraper.lock` absent" is not a reachable precondition. `filelock` never
    unlinks the file on release — it drops the `flock` and removes the `.holder` sidecar — so a
    0-byte lock file sits there after every run that has ever succeeded. Probe the `flock`
    instead (`corpus_lock_state` in `utils/corpus_lock.py`); the file's existence means nothing.
  - "The lock does **not** release when a GitHub run is cancelled (verified twice)" is false. It
    does release. What survives a cancelled run is the **container**, which still holds the flock
    because it is still working — cancelling the runner does not stop it. Stop the container to
    release the lock. The "verified twice" backed three unnecessary hand-removals of a file that
    was never stale.
- `corpus/.viewer/jobs.paused` present.
- Profile `prod_dgx_full` → `tailnet_dgx_whisper`. Self-hosted ASR, so the failure mode is DGX
  availability, not billing.

### 2. Backup
Run `backup-corpus-prod.yml` — a full corpus backup, which `reprocess-prod.yml:9-10` already
instructs for any reprocess. No bespoke 79-file tarball: it was needed only by the deletion route.

### 3. Probe ONE episode, artifacts untouched
Nobody has traced whether the serving layer picks up a repaired episode. Do one first.

Use **"What's Going On With Lettuce?"** — 27 MB, the fastest signal.

```
workflow: reprocess-prod.yml
  confirm              = PROD_REPROCESS
  selection            = episode_ids_worklist
  episode_ids          = 49ed7d0e-8542-11f1-a28b-2f96b6aae640
  pipeline_stage       = full          # NOT relabel_only; these need ASR
  use_transcript_cache = false
  profile              = config/profiles/prod_dgx_full.yaml
  litellm_api_base     = http://<DGX_IP>:4001/v1
  timeout_minutes      = 240
  cost_cap_usd         = (set a small cap; the selection gate prices before spending, and a
                          refusal costs nothing — scraping.py:764-771)
```

Watch for, in order:
- `reprocess work-list: restricting this run to 1 of N on-disk episodes` (`scraping.py:677-682`)
- the forced-re-transcription line (`episode_processor.py:501-506`) — if absent, skip-existing won
  and the run will no-op
- the per-episode resolution line (`scraping.py:746`) naming a real file

Accept only if ALL hold:
1. A **new** `run_*` dir holds `<stem>.txt` AND `<stem>.segments.json` for this episode.
2. The new record's `transcript_file_path` names **this** episode.
3. The transcript's opening words are about lettuce/produce — a **content** check. The resolver
   vouches filenames and sidecars, never words.
4. The app/API shows the episode with its own speakers.
5. `existing_transcript_path_in_corpus` now returns a `.txt`, not a `.metadata.json`.

If 1–3 pass and 4 fails: **stop.** The artifacts are right and the serving layer is the problem — a
different fix, and continuing would produce 9 more invisible repairs.

Per-episode full-pipeline wall-clock is **not measured**. The 4.7 min/episode figure is
`relabel_only`, which does no ASR and no diarization. This run supplies the first real number; do
not quote a batch ETA before it.

### 4. The remaining 9
Same inputs, work-list of the other 9 ids:

```
bfa86990-175c-11f1-805b-b37d2c0211e7   Powering the AI Inference Wave with EPRI's Ben Sooter
0abbdc7e-4cb3-11f1-bb0f-9b67bad2c789   Pax Silica: Inside the Trump Administration's Tech Strategy
20105fae-4c96-11f1-a903-0b1789dff273   Amex Global Business Travel
2ffbf42c-503f-4b51-9462-af3c5626c900   When designing a GAME, what's in a name?
substack:post:208082176                Inside the Model Factory — Eiso Kant
substack:post:202758604                Red-Teaming after Mythos — Zico Kolter & Matt Fredrikson
substack:post:202359797                The Professor of Outputmaxxing — Anjney Midha
substack:post:202058620                The Self-Driving Lab — Joseph Krause
51dac028-86da-11f1-be2e-638d60f86eab   Can't Get an IMAX Ticket?
```

### 5. Rebuild the search index — NOT optional
LanceDB writes by merge-upsert and **never deletes**; only `reindex-prod.yml mode=rebuild` drops and
rebuilds (`reindex-prod.yml:5-9,22-24`). The pipeline itself does not index. Without this step the
corrupt episodes' wrong-content chunks stay searchable after every other step succeeds.

Run `reindex-prod.yml` with `mode=rebuild`, `with_clusters=true` — `search/topic_clusters.json` was
derived from GI built on the wrong words (`reindex-prod.yml:43-49`). It shares the `prod-corpus`
lock, so it cannot overlap a reprocess.

### 6. Re-enrich the 10
`insight_sentiment` / `insight_density` come from the RFC-088 enricher executor
(`enrichment/enrichers/__init__.py`), dispatched separately — a pipeline run does not produce them,
so the new run dirs have none. Either re-enrich the 10 or accept the gap until the next scheduled
pass, explicitly.

### 7. Clean up the superseded corrupt artifacts (optional, last)
Only now, when selection no longer needs them and newest-run already resolves to the repair. 79
files listed by the generator in this session. Harmless to leave — they are superseded copies.

### 8. Verify
`speaker_coherence_report.py --newest-run-only` + the pairing audit. The 10 should appear in neither.

## Rollback
Restore from the step-2 corpus backup. The repair adds new run dirs rather than mutating old ones,
so the cheaper rollback is to delete the new run dirs, which returns newest-run to the old records.

## Accepted trade-off
The wrong pages keep serving until each episode's repair lands. Revision 1 claimed "absent beats
wrong" — with this code that option does not exist at any price: deleting the metadata forfeits
selectability (below), and there is no way to blank a page without deleting the metadata, since the
catalog row and speakers come from it. Wrong-then-correct is the only executable sequence.

## The route that was rejected, and why

Revision 1 said: delete the 79 artifacts to clear the skip-existing presence marker, then re-ingest.
Both halves were wrong.

1. **Unnecessary.** `--reprocess-episode-ids` already overrides skip-existing per episode
   (above). The "proof" that it did not was a test of `existing_transcript_path_in_corpus` in
   isolation — a different code path from the skip decision for a work-list member.
2. **Self-defeating.** On the `--reprocess-existing-only` path (implied by
   `--reprocess-episode-ids`, `config.py:6350-6351`) the episode set is built FROM on-disk metadata
   via `_on_disk_guid_index` (`scraping.py:604`, `:371-378`), then intersected with the work-list
   (`:648-652`). Deleting a guid's metadata removes it from the selectable universe: the run logs
   "none of the N listed episode(s) are in this feed's corpus" and **returns `[]` with exit 0**
   (`scraping.py:669-676`). The same file is both the skip marker and the selection record.
3. **Its own verification would have passed in that failure state** — step 3 checked
   `existing_transcript_path_in_corpus` → None, which is exactly what a deleted record returns.

Also wrong in revision 1: the claim that deletion would move the serving count 2002 → 1992. API
caches key on stamp-file mtimes (`perf_cache.py:139-156`), which a raw `rm` does not bump.

## Not verified
- Whether a `full` run regenerates `bridge.json` / `context.json` specifically.
- Whether the audio archive holds any of the 10 (irrelevant while all 10 are in-feed).
- Per-episode full-pipeline duration, hence no batch ETA.
- Whether any OTHER run dir holds metadata for these guids (moot under repair-first; would matter if
  anyone resurrects the deletion route).

## Outcome — executed 2026-09-24, all 10 repaired

Ran on `sha-e2dedbd` via `--reprocess-episode-ids`, no deletions. Verified **by episode id**, never
by topical search: a topical query ranks a different episode about the same guest above the
repaired one and looks exactly like a failure (observed searching "Pax Silica" — a different Jacob
Helberg episode won).

Per-episode checks, all 10 green: the serving view resolves it; its `transcript_file_path` NAMES
this episode (the corruption test); transcript + `.segments.json` on disk; the transcript's words
are this episode's by title-token overlap; `gi.json`/`kg.json` post-date the repair.

```text
FULLY VERIFIED: 10 of 10
corpus-wide mismatched-transcript count: 0 (was 10)
```

Still outstanding, and NOT done by this repair:

- **The search index.** LanceDB merge-upserts and never deletes, and the pipeline does not drop
  stale chunks, so the wrong-transcript chunks are still queryable. Needs `reindex-prod.yml`
  `mode=rebuild with_clusters=true` — separate workflow, after the big batch.
- **Re-enrichment of the 10** after that rebuild.
- The four items under "Not verified" above remain unverified; the repair route made them moot
  rather than answering them.
