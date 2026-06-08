# Runbook — corpus-wide SPOKEN_BY reprocess (#876)

**Goal:** raise `SPOKEN_BY` coverage from ~10/110 episodes (only the pre-diarized
`direct_download` ones) toward full, by re-diarizing the ~90
`whisper_transcription` episodes and re-deriving the relational edges.

**Unblocked by:** #875 (`gi/speakers.py` now attributes the new diarization's
*named* screenplay markers directly — see `tests/unit/search/test_enrich_edges_cli.py::test_run_emits_spoken_by_for_named_diarized_transcript`).

## Why a full re-run (not reprocess-from-transcripts)

`SPOKEN_BY` needs **speaker labels in the transcript**. The whisper episodes'
existing transcripts are plain text (Whisper emits no speaker labels), so
`make reprocess-corpus-from-transcripts` alone cannot add them. The labels come
from **diarization**, which needs the audio → the episodes must be re-run with
diarization on.

## Procedure

Steps 1–3 are now one command (#925):

```bash
make redo-diarization CORPUS_DIR=<corpus> PROFILE=config/profiles/local_dgx_full.yaml
```

`make redo-diarization` (`Makefile`) runs the pipeline with `--skip-existing
--reprocess-source whisper_transcription`, which force-reprocesses **only** the
`whisper_transcription` episodes (the `direct_download` ones are left untouched),
re-transcribing + re-diarizing them under the supplied **PROFILE** (diarization is
driven entirely by the profile's `diarize: true` — `local.yaml` / `local_dgx_full.yaml`),
then runs `enrich-edges` to re-derive corpus-wide `SPOKEN_BY`. Run it **on a
diarization-capable host** (e.g. the DGX) so pyannote rides the GPU.

Expanded, the same steps are:

1. **Identify the targets** — the `whisper_transcription` episodes (`--reprocess-source`
   selects them by `content.transcript_source`; `direct_download` skipped).
2. **Re-run the pipeline with diarization** on those episodes (`diarize: true` from
   the profile). This re-transcribes + diarizes, writing the *named* screenplay
   transcript + diarized segments; the build emits `SPOKEN_BY` during GI build.
3. **Re-derive relational edges** corpus-wide (idempotent):
   `make enrich-edges CORPUS_DIR=<corpus>` (also run automatically by `redo-diarization`).
4. **Validate coverage** with the CIL queries:
   - `positions_of(person)` / `who_said(topic)` return results across episodes, not
     just the pre-diarized 10.
   - Spot-check a recurring guest resolves to one `person:{slug}` aggregating quotes
     from all their episodes (this is the #909 payoff).

## Notes

- Re-transcription is redundant work (we already have the text) but the pipeline
  couples transcribe+diarize; a diarize-only reprocess would avoid it but is not
  built (out of scope — one-time cost). If the corpus grows this becomes worth it.
- Promote this file to `docs/guides/` once the reprocess has actually been run and
  the steps are confirmed against prod.
