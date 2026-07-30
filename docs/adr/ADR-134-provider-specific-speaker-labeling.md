# ADR-134: Provider-specific speaker-labeling strategies

- **Status**: Accepted
- **Date**: 2026-07-25
- **Authors**: Marko Dragoljevic
- **Related RFCs**: —
- **Related PRDs**: — (#876 speaker quality, #1170 diarization, #1190 corpus reprocess)

## Context & Problem Statement

Speaker labeling (mapping diarized `SPEAKER_NN` clusters to real names) is **downstream of, and
coupled to, the diarizer's clustering footprint**. Different diarizers cluster the same audio
differently: Deepgram (cloud) produces coarse clusters — it tends to merge a show's cold-open
montage into the host's own cluster; pyannote **community-1** (self-hosted on the DGX) produces
finer clusters — it splits each host into their own cluster, splits some guests across two clusters,
and leaves recurring ad/promo readers as their own clusters (sometimes with one stray content turn
merged in).

The speaker-labeling heuristics (host-candidate selection, montage detection, ad/recorded-voice
detection, name canonicalization gating) were tuned — implicitly — to Deepgram's footprint. They
lived inside functions that read as provider-agnostic but silently assumed "the opening voices are
the hosts" and "a cluster is textually pure." When the corpus reprocess arc switched the diarizer to
community-1 (a single-variable v2.2 gate), those hidden assumptions broke: dominant hosts published
under ASR-garbled names ("Kevin Russo"/"Casey Noonan"), a recurring ad reader was named as a host,
and split guests produced duplicate names ("Adam Rodman" + "Dr. Adam Rodman"). A cascade + single-
variable pilot caught this.

The project is **local-first**: community-1 on the DGX is the production diarizer, not Deepgram.

## Decision

There is **no single diarizer-agnostic labeling heuristic**. Labeling that is sensitive to cluster
shape is made **provider-specific**: a `DiarizationLabelingStrategy` is selected by
`diarization_provider`, layered over a **shared core** of provider-independent primitives and
invariants. Deliberate overfitting to a diarizer's clustering is acceptable **when it is explicit
and contained inside that provider's strategy** — never smuggled into a shared "generic" function.

- **Shared core (provider-independent):** the naming invariants (N1: a guest with a host-like name
  must not steal the host identity; "a wrong label is worse than an unnamed voice"), the naming
  precedence (self-intro > host-introduction > LLM > pool/forced), and the reusable primitives
  (`_canonicalize_to_known_host`, `_surname_token`, self-intro extraction, the same-person predicate,
  known-hosts/detected-guests plumbing, the LLM speaker path).
- **Provider-specific strategy (cluster-shape footprint):** host-candidate *eligibility*, montage /
  split handling, ad-voice / recorded-voice detection tuning, and the *gating* of canonicalization.

`pyannote_community1` is the product strategy and receives the real investment; `deepgram` becomes a
legacy strategy, frozen at its current (v2.1.x-validated) behavior.

## Rationale

- The coupling is real, not incidental: labeling reads signals (who self-introduces, talk patterns,
  textual recurrence) whose *cluster attribution* the diarizer decides. An abstraction that works
  equally well across coarse and fine clustering does not exist for the ambiguous cases.
- Explicit provider strategies convert the failure mode ("silent overfit to one diarizer") into a
  maintainable structure: each provider's quirks are named, tested against that provider's real
  output, and cannot regress the other.
- Contained overfitting respects the operator's standing anti-overfit rule — the objection was never
  to provider-coupling, it was to provider-coupling **disguised as generic logic**.

## Alternatives Considered

1. **One diarizer-agnostic heuristic set (status quo).** Rejected: it is what broke. The clustering
   differences are structural; forcing one rule set makes it fragile to any diarizer swap and hides
   the coupling.
2. **Full fork of ALL labeling per provider.** Rejected: most labeling (invariants, precedence, name
   primitives, LLM path) is genuinely provider-independent; duplicating it invites divergence and
   double-maintenance. Only the cluster-shape-sensitive layer is forked.
3. **Tune the diarizer / reconsider community-1.** Rejected as off-topic: local (community-1) is the
   product requirement; diarizer quality is a separate, already-answered question. The task is to
   make labeling work on community-1's real output.

## Consequences

- **Positive:** provider-coupling is explicit and contained; community-1 (the product) gets a
  strategy fitted to its real clustering; Deepgram stays working, frozen; each strategy is testable
  against its own diarizer's output; future diarizers add a strategy without touching the others.
- **Negative:** a small selection/plumbing layer is added; two strategies to maintain (but Deepgram
  is legacy/frozen, so effectively one active).
- **Neutral:** the shared/strategy boundary is a judgment line that must be kept honest — a rule that
  depends on cluster shape belongs in the strategy, not the core.

## Implementation Notes

- **Module:** `src/podcast_scraper/providers/ml/diarization/roster.py` (+ `boilerplate.py`,
  `speaker_detectors/hosts.py`).
- **Pattern:** strategy selected by `diarization_provider`; shared primitives + per-provider
  cluster-shape strategy. community-1 strategy (this ADR's motivating case): turn-level
  `recorded_voices` (a merged stray turn cannot hide an ad reader), host-candidate *eligibility*
  hardened against cold-open fragments/ad/recorded voices, a known-host unique-first-name snap gated
  to the eligibility set (garbled dominant hosts), and the symmetric same-person dedup for split
  guests.

## References

- #876 (speaker quality), #1170 (diarization), #1190 (corpus reprocess v2.2 = community-1)
- ADR-110 (LLM speaker resolution), #1188 (recurring-passage / recorded-voice detection)
