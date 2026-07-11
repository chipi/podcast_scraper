# Diarization tuning eval — runbook

**Goal:** stop pyannote over-segmenting (splitting ~2–4 real speakers into 10–20 `SPEAKER_NN`
labels), by tuning `clustering_threshold` × `max_speakers` against ground-truth fixtures until the
detected speaker count matches, then adopting the winning params as defaults.

## Why (the diagnosis)

The "100+ unknown guests" on the real corpus was **diarization over-segmentation**, not 100 real
un-named people. One episode produced **15 labels for ~2 real speakers** (two labels held 75% of
the audio; the other 13 were 6–122 s fragments). The provider only exposed `min/max_speakers`, not
pyannote's **clustering threshold** — the actual over/under-segmentation knob (higher threshold →
merges more → fewer speakers).

## The pieces (all committed, GPU-free)

1. **The knob** — `config.diarization_clustering_threshold` (None = model default), applied in
   `PyAnnoteDiarizationProvider` via `_apply_clustering_threshold` (overrides only the clustering
   block, preserves the rest). `diarization_max_speakers` (default 20) is the other lever.
2. **Ground truth** — every **v3** fixture has `tests/fixtures/transcripts/v3/<name>.groundtruth.json`
   with `expected_diarized_voices` (humans + ad voices = what a correct diarizer should detect),
   `type` (monologue/interview/panel), `has_commercial`, `failure_modes`. Regenerate with
   `python tests/fixtures/scripts/make_groundtruth.py` (`--check` in CI). **v1/v2 are dead — use v3.**
3. **The sweep** — `scripts/eval/score/diarization_tuning_sweep_v1.py`: sweeps the grid, scores
   `detected == expected` per fixture, reports per-combo match-rate + misses + best combo.

The eval set (45 fixtures): 34 interview, 9 monologue, **1 true 3-voice panel (p05_e04)**, **30
with a mid-roll ad voice**. The panel + ad cases are the **guardrails** — they stop the sweep from
"winning" with a degenerate always-merge-to-2 (which would break real panels and misplace ad reads).

## Running it

```bash
# GPU-free sanity check (loads fixtures + grid, no diarization):
python scripts/eval/score/diarization_tuning_sweep_v1.py --dry-run

# Round 1 — full sweep on the DGX GPU (15 combos × 45 fixtures ≈ minutes on GPU, hours on CPU):
#   switch the shared GPU first:  gpu-mode-swap.sh research   (NEVER `code` — that's coder-next)
python scripts/eval/score/diarization_tuning_sweep_v1.py --device cuda \
    --out data/eval/runs/diarization_tuning_v1
# -> prints BEST: thr=<x> max_speakers=<n> -> M/45 (P%); results.json has the full grid.
```

Local CPU/MPS works but is slow (pyannote-on-MPS falls back to CPU for its ops); use the DGX for
the sweep. Default model is the non-gated `speaker-diarization-community-1` (no HF token needed).

## Rounds

- **Round 1 (fixtures):** sweep until a param combo hits **100% count-match** across all 45 v3
  fixtures (incl. the panel + ad cases). Record the winning `clustering_threshold` / `max_speakers`.
- **Round 2 (real episodes):** apply the winning params to the real over-segmented corpus episodes
  (audio is under `.test_outputs/manual/prod-v2/corpus/**/media/*.mp3`), confirm the 15→~2–4 count
  drop, then set the winners as the `config` defaults.

## Gotchas (learned the hard way)

- **Always read `tests/fixtures/FIXTURES_VERSION` first.** v1/v2 are deprecated; analysing them
  wastes a whole pass. The sidecars live only under v3.
- `expected_diarized_voices` **includes the ad voice** — a 2-host episode with a mid-roll ad has
  expected = 3. The roster's `voice_type=commercial` then parks the ad; the diarizer still detects it.
- The `high_person_density` failure-mode tag is about **person-entity** density (many people
  *discussed*), **not** speaking voices — those fixtures still have ≤2 speakers.
