# Task — finish the `pipeline_stage=relabel_only` option properly

**Status:** Open. Created 2026-07-24. Do this **while the 100-episode batch runs**
(dead time). We added a real pipeline option; it must meet the same bar as every
other option before we lean on it.

## What it is

A new `pipeline_stage` mode that **re-resolves speaker names on a corpus's existing
frozen `SPEAKER_NN` diarization** through the *real* pipeline resolver
(`apply_diarization_to_result` → `resolve_speaker_roster`), profile-driven, with
**no re-ASR and no re-diarize** (GPU-free, no audio). It fixes the June bad-relabel
rot (ad readers crowned as hosts) without a full reprocess.

Validated core (2026-07-24): the real resolver on prod-v2 Hard Fork ep 0001 names
`SPEAKER_06→Kevin Roose`, `SPEAKER_08→Casey Newton`, leaves ad readers unnamed
(does NOT crown "Amy Lawrence"). See `scratchpad/relabel_ep1_demo.py`.

## Done so far

- `apply_diarization_to_result(..., precomputed_diarization=DiarizationResult)` —
  skips the cache/provider (audio) path, uses the supplied diarization
  (`providers/ml/diarization/pipeline.py`). flake8 + mypy clean.

## The proper-job checklist (same standard as other options)

- [ ] **Config enum** — add `"relabel_only"` to the `pipeline_stage` `Literal`
      (`config.py:3273`) and the CLI `choices` (`cli.py:1835`).
- [ ] **Coercion** — extend `_coerce_pipeline_stage_before` (`config.py:3853`):
      `transcribe_missing=false`, reuse on-disk transcripts, force the existing
      episodes through the resolver (no diarize/provider), like `enrich_only` does
      for transcription. Log the coercion once, matching the other modes.
- [ ] **Episode gating** — the `relabel_only` branch in `episode_processor.py`:
      load the on-disk transcript + `.segments.json`, reconstruct a
      `DiarizationResult` from the `SPEAKER_NN` segments, call
      `apply_diarization_to_result(..., precomputed_diarization=...)`, re-render the
      screenplay, re-save transcript + segments + `.adfree`, then cascade GI/KG/edges.
      Skip download/transcribe/provider-diarize.
- [ ] **Config template** — options are invokable from a **config file**, so
      `pipeline_stage: relabel_only` must appear/round-trip in the config template
      and any profile schema (check `config/profiles/*` + the profile template +
      `test_profile_yaml_registry_drift`). Realign the template.
- [ ] **Tests** — same coverage as other options:
      - unit: coercion sets the right flags for `relabel_only`;
      - unit: `apply_diarization_to_result` honours `precomputed_diarization` (no
        provider call, correct roster);
      - integration: `relabel_only` on a small fixture corpus re-resolves names +
        rewrites the screenplay, no audio touched.
- [ ] **Docs** — add the `relabel_only` row/section to
      `docs/guides/CORPUS_REPROCESSING.md`; document the config field in
      `docs/api/CONFIGURATION.md`; add the flag to `docs/api/CLI.md`'s reprocess list.
- [ ] **`relabel_corpus.py` retirement** — once `relabel_only` ships profile-bound,
      the non-profile spaCy `scripts/backfill/relabel_corpus.py` is superseded;
      migrate its one eval consumer (`scripts/eval/build_crossshow_dataset.py`) and
      delete it.

## Why it matters (operator, 2026-07-24)

"We just added an option to the CLI. We need to document it, have proper test
coverage… these options can be invoked from a config, they need to be in the
template. There's a whole lot of realigning if we're adding an option." Do it
right, while the batch runs.
