# Batch plan — diarization follow-ups (one PR on `feat/diarization-followups`)

Now that diarization has landed (#895/#901/#908), this batch closes the
speaker-attribution loop end to end and then validates the whole graph. All
items ship in **one PR** on `feat/diarization-followups`.

## Items

| # | Item | Issue | Type | Status |
| - | ---- | ----- | ---- | ------ |
| 1 | Full speaker-ID — **approach A**: unify `gi/speakers.py` on the diarizer's already-named `speaker_label` (reuse N-capable `map_speakers_to_names`); fix panels/multi-guest | #875 | implement | ready |
| 2 | Reprocess the 90 `whisper_transcription` episodes → corpus-wide `SPOKEN_BY` | #876 | implement | depends on #1 |
| 3 | Cross-episode person identity — *land + surface* the payoff of #1/#2 (validate corpus-wide `person_profile` / `who_said` / `positions_of`) | #909 | implement + validate | depends on #1/#2 |
| 4 | Coverage debt (B-tier tests) | — | tests | ready (see `COVERAGE-DEBT-deepgram-diarization-pr908.md`) |
| 5 | **Graph validation + all-tests focus** | — | validation | final phase (protocol below) |

## Approach decisions (locked)

- **#875 = approach A** (early-naming via diarized `speaker_label`, not the
  cluster-as-signal/late-naming re-architecture).

## Final phase — graph validation protocol (blind notes)

After items 1–4 land:

1. Operator does a full graph + test review and **does not share their gaps yet**.
2. Agent runs the **first few rounds of graph/test validation independently**.
3. Operator then shares their observed gaps (incl. issues noticed in 2.6.1).
4. **Compare notes** — the union drives the fix list.

This keeps the agent's testing unbiased by the operator's prior observations.

### Round-1 outcome (2026-06-07) — methodology correction

Round-1 was run against the GI **stub** path (`model_version="test"`) — which we
do NOT use for validation. Its findings (F1 "empty person profile", F2
"mentioned vs speaking divergence") were **stub artifacts, discarded**. The real
finding: **no current validation data carries the speaker layer.**

- `viewer-validation-corpus/v2` = synthetic **artifacts only** (no transcripts);
  its generator emits `Topic/Insight/Quote/Episode` + `MENTIONS/HAS_INSIGHT/HAS_QUOTE`
  — **zero `SPOKEN_BY` / `Person`**. So it cannot validate #875/#876/#909 or the
  viewer Person surfaces.
- The v2 **audio/transcript** family (`tests/fixtures/{audio,transcripts}/v2`, 32 eps)
  DOES carry named speakers (#111) + commercials (#109, 21/32) — but emitting
  `SPOKEN_BY` through it is the ML/diarization pipeline (heavy, gated).

### Next step after rebase — `viewer-validation-corpus/v3`

Build a **v3** synthetic generator that adds the speaker layer (deterministic, no
ML), so the person/speaker graph is validatable in ci-fast:

- `Person` nodes + `SPOKEN_BY` edges from named speaker turns.
- A **recurring guest** across ≥2 episodes (one `person:{slug}`) → #909.
- A **panel** episode (≥3 named speakers) → #875.
- A **mentioned-only** person (KG entity, no `SPOKEN_BY`) → models the F2
  distinction deliberately so we can decide on it with real-shaped data.
- Keep v2 content (commercials etc.). Bump to v3 (don't mutate v2 — keeps the
  N-1 corpus-compat tests stable).

Then: re-run graph/CIL/viewer validation against v3 → finally the
v2-audio→diarization e2e as the realistic proof (option B).

## Decided AGAINST (from the early identity-vision audit)

The five `docs/wip/rfc_*` / `product_ux_identity_vision` drafts (~1.5-month-old
"super-early thinking") were audited against the shipped system: ~70–80% already
delivered (CIL resolver, `person_profile`, all viewer surfaces via PRD-033
#882–#890, diarization). The surviving high-value thread is **#909**. Explicitly
**not** pursued (over-engineered for a host+guest corpus):

- Speaker-clusters-as-signal → late-naming / resolve-in-CIL re-architecture.
- Conversation graph / `RESPONDS_TO` edges (trivial in host↔guest).
- CIL confidence/provenance "single source of truth" consolidation.
- Voice-based linking for *unnamed* speakers (expensive ~5%).
- Conversation View UI — deferred unless concrete user pull.

→ The five draft files in `docs/wip/` are superseded by this audit + #909 and can
be deleted once the operator confirms.

## Out of this batch / inputs pending

- Operator's concrete **2.6.1 graph issues** — shared in the final-phase protocol,
  not before.
