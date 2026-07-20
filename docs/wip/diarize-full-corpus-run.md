# Run diarization across the full prod-v2 corpus

**Status:** queued — not started
**Owner:** operator (Marko)
**Related:** #1170 diarization eval; branch `feat/enrichment-surfaces`

## Intent

Run the winning diarization pipeline (**community-1**, beats pyannote 3.1 on
full-45 fixtures: count 40 vs 32, DER 7.1% vs 10.8% — see
`project_diarization_findings_1170` memory) across every episode currently in
the prod-v2 corpus, so every episode ships with a real speaker-count / RTTM
and downstream enrichers (host/guest swap #1169, Speaker/Quote surfaces in
the viewer, person-quality scoring) can consume it uniformly.

## Prep checklist (do these before the run)

- [ ] Confirm `gpu-mode-swap.sh research` slot is free on DGX
      (`~/agentic-ai-homelab/infra/dgx/bin/gpu-mode-swap.sh status`).
- [ ] Confirm pyannote community-1 weights are cached at
      `/root/.cache/torch/pyannote` on DGX (see
      `reference_pyannote_cache_offline`).
- [ ] Snapshot prod-v2 episode list + audio hosts before the run so a
      partial batch is resumable without re-diarizing what's already done.
- [ ] Decide idempotency key — enricher SHA + model tag so a re-run only
      touches episodes whose diarization is missing or on an older tag.

## Success criteria

- Every prod-v2 episode has a `diar_rttm` artifact + speaker-count column.
- Enrichment surfaces in the viewer (Speaker nodes, host/guest partition)
  render for episodes on this corpus.
- Full-corpus DER + count deltas logged into the #1170 eval report as a
  new tier alongside curated-8 / full-45.

## Open questions

- Real-10 (Step 2c) is still deferred pending real-audio RTTM labeling —
  should the full-corpus run block on that, or ship in parallel?
- Should host/guest swap #1169 gate on this, or land first with best-effort
  labels?

## Notes

- Do NOT trigger from this project directory — GPU work runs on DGX only.
- Never use `gpu-mode-swap.sh code` (coder-next is operator's IDE).
- Estimated cost: ~N episodes × ~0.5x realtime on GB10 with community-1;
  size the sweep window before kicking off.
