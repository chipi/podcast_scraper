# ADR-146: Corpus reprocess methodology — single-variable validation, reprocess-once economics, judge-panel parity gate

- **Status**: Accepted
- **Date**: 2026-08-02
- **Authors**: Marko Dragoljevic
- **Related ADRs**: [ADR-123](ADR-123-quality-gate-transcription-failover.md) (failover), [ADR-124](ADR-124-model-governance-registry-sanctioned.md) (registry), [ADR-134](ADR-134-provider-specific-speaker-labeling.md) / [ADR-137](ADR-137-llm-host-guest-role-on-voice-resolution.md) / [ADR-140](ADR-140-versioned-labeling-profiles.md) (labeling), [ADR-135](ADR-135-v2.4-gi-route-and-tag-and-kg-voice-node.md) (v2.4 schema), [ADR-058](ADR-058-additive-pyannote-diarization-with-separate-extra.md) (diarization)
- **Related issues**: #1335 (v2.2), #1355 (v2.4), #1189 (fixture gate), #630 (expansion)

## Context & Problem Statement

The corpus is reprocessed **v2 → v3** and scaled from ~90 episodes toward 500–1000
(10k horizon). A 500–1000-episode run is expensive and slow (large-v3 ≈ 7.8× realtime →
1000 eps ≈ 4 GPU-days; 10k ≈ 40 days). Two failure modes had to be designed out:

1. **Un-attributable quality deltas.** If several producing variables (ASR, diarization,
   naming, GI/KG schema, the LLM) change at once, a downstream quality change cannot be
   traced to its cause.
2. **Double reprocessing.** Any change to the stored artifact shape, the input text, or a
   producing model that lands *after* the scale run forces the entire corpus to be
   rebuilt a second time.

This methodology governs how every corpus-producing change is validated and sequenced. It
was applied across v2.1–v2.4 (shipped) and governs the v2.5 LLM swap and the scale run.

## Decision

**1 — Single-variable validation.** Each corpus version changes exactly **one** producing
variable, then the whole cascade re-runs and that version is compared against its immediate
predecessor. The acceptance stick is **parity with the prior version, not ultimate truth** —
the previous corpus is the already-accepted baseline, so each step need only be "not worse
than what we had." (Same discipline as the deepgram freeze arbiter in v2.2.)

The applied ladder:

| Version | Single variable | State |
| --- | --- | --- |
| v2.1 | speaker naming (on frozen deepgram diarization) | done |
| v2.2 | diarization → pyannote community-1 (DGX-local) | MERGED #1335 |
| v2.3 | ASR → faster-whisper turbo + failover (ADR-123) | MERGED (in #1355) |
| v2.4 | GI/KG schema (route-and-tag #1191, naming-4) | MERGED #1355 / ADR-135 |
| v2.5 | LLM (Gemini → DGX-local) | current front |

**2 — Reprocess-once economics.** Anything that changes (a) the stored artifact shape
(KG/GI schema), (b) the input text (cleaning), or (c) a producing model (ASR / diarization /
LLM) must be **locked before** the scale run. The "next cut" is therefore not "the most
issues" — it is **everything that would otherwise force a second full rebuild**. The scale
run reprocesses the frozen combination **once**.

**3 — Cost-aware measurement.** Transcription (OpenAI Whisper, ~$0.50/ep) is the only
expensive layer; everything downstream is free (DGX) or cents (cloud LLM). So:

- **Deterministic layers are the primary verdict, no baseline re-run needed** — transcript
  WER vs the prior version and **speaker-roster parity** (names + roles) are the star signals.
- **Noisy layers (GI/KG/summary) get a cheap noise floor** — re-run only the downstream
  cascade on the prior version's existing transcript (`rediarize_only`: reuse the paid
  transcript, re-diarize/re-name/re-enrich). The real signal is then
  (vX − vX−1) **above** the run-to-run noise (vX−1 − vX−1′).

**4 — Judge-panel parity is the LLM-swap ship gate (judge-panel ONLY).** For a producing-LLM
swap (v2.5), the ship gate is a **cross-vendor judge panel** scoring the new output vs the
prior-LLM baseline; the swap ships only at statistical parity. The panel must be
**disjoint-vendor** (silver + judge from a vendor NOT in the candidate cohort — else a
same-vendor style boost inflates the score), **scalar mode** (pairwise showed same-vendor
bias and worse rank-correlation vs cloud), and the score parser must strip reasoning
(`</think>`) blocks before extracting digits. **Human ground-truth (golden fixtures #1189)
is NOT part of the parity gate** — it is the separate reprocess *acceptance* gate that
guards the scale run against regressions.

## Rationale

- Attributability is only free if you change one thing at a time; bundling is cheaper
  per-run but destroys the ability to say *why* quality moved.
- Parity-vs-prior (not absolute truth) is the correct stick because the prior corpus is
  already in production and accepted — the question at each step is only "did this
  regress," which a cheap deterministic comparison answers.
- Locking artifact-shape/input/model before the scale run is the difference between one
  4-GPU-day run and two.
- Judge-panel-only (vs adding human-GT to the gate) keeps the LLM-swap gate scalable and
  un-blocked on the #1189 fixture ladder; the vendor-disjoint + scalar constraints are what
  make the panel trustworthy (see the trust-matrix finding in `autoresearch/JUDGING.md`).

## Alternatives Considered

1. **Bundle multiple variables per version.** Rejected — a GI delta folded into the ASR step
   is un-attributable (ASR? or schema?). Cheaper per-run, but you lose the diagnosis.
2. **Absolute-truth (human-GT) acceptance at every step.** Rejected — bottlenecks every
   single-variable step on human labelling; parity-vs-prior is sufficient and cheap.
3. **Re-transcribe to establish each baseline.** Rejected — transcription is the only
   expensive layer; the prior corpus on disk already *is* the baseline.
4. **Judge-panel + human-GT as the LLM-swap gate.** Rejected for v2.5 (2026-08-02) — couples
   the swap gate to the unfrozen #1189 ladder; judge-panel-only is the decided ship signal,
   #1189 stays the separate reprocess-acceptance gate.
5. **Lean cut (defer some schema epics).** Rejected — reprocessing 1000 eps twice is not
   worth deferring; full Bucket-A "born correct" was chosen (2026-07-20).

## Consequences

- **Positive**: every quality delta is attributable; the corpus is rebuilt once; measurement
  cost is bounded to the deterministic layers + a cheap noise floor.
- **Negative**: more sequential versions (v2.1→v2.5) than a single bundled rebuild.
- **Neutral**: the ship gate for an LLM swap depends on a correctly-configured judge panel
  (disjoint-vendor, scalar, `</think>`-stripped) — a standing operational requirement.

## Implementation Notes

- **Acceptance harness**: `scripts/backfill/relabel_corpus.py` (`--llm none` = deterministic,
  no GPU) over the frozen prod-v2 corpus.
- **Judge panel**: `autoresearch/` (`JUDGING.md`, `PER_MODEL_OPTIMAL_PARAMS.md`,
  `bundled_prompt_tuning/`).
- **Profiles**: `config/profiles/prod_dgx_*.yaml`, `cloud_with_dgx_primary.yaml` select the
  producing models per version.
- **LLM serving is external** (DGX / homelab) and its topology is operationally transient —
  the methodology depends on *a* served candidate set, not a fixed host.

## References

- [ADR-123: quality-gate transcription failover](ADR-123-quality-gate-transcription-failover.md)
- [ADR-135: v2.4 GI route-and-tag + KG voice node](ADR-135-v2.4-gi-route-and-tag-and-kg-voice-node.md)
- `autoresearch/JUDGING.md` — judge-panel trust methodology (scalar > pairwise; vendor disjointness)
- Issues: #1335 (v2.2), #1355 (v2.4), #1189 (fixture acceptance gate), #630 (expansion vehicle)
