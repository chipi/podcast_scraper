# Flightcast / interview-panel labeling — what's fixed and what's an accepted limitation

- **Status**: Analysis + limitations record (naming-4). Follow-up levers noted.
- **Date**: 2026-07-29
- **Context**: flightcast (Latent Space-style interview podcasts) held the largest block of remaining
  `unknown` labeling defects after the naming-4 pass — 15 voices / 18,521s, single talks running
  500–5,000s unattributed. This records the three failure modes, which are fixed, and which are
  accepted limitations of the *naming* layer (to be revisited elsewhere).

## FIXED in naming-4

**First-name-only group intro (the big recovery).** The host introduces guests by bare first name —
*"we're here with **akshat** of moto… together with **vibu**"* — of people who **are** in metadata
(`Akshat Bubna`, `Vibhu`). The cue matched but binding required a surname, so they stayed unnamed.
naming-4 binds a bare first name when it **uniquely** matches one stated person (cue path only;
`first_name_only_intro` flag). Validated: flightcast 0005 SPEAKER_01 (2,036s) UNNAMED → **Akshat
Bubna**.

**Merged-cluster host mislabel.** See below — the wrong name is now suppressed.

## ACCEPTED LIMITATION #1 — feed-metadata mismatch (not a naming defect)

flightcast 0002 metadata lists `[Eiso Kant, Andrej Karpathy, Peng Ming…]`, but the host actually
introduces *"Vaisal Khan from FullSight… and Dibu."* The feed metadata names **mentioned/other
people, not the episode's speakers**, so #876's closed-list rule literally cannot bind the real
names — they are not in the candidate list. The 5,056s dominant voice is unnameable without
inventing a name.

- **Why not fixed here:** binding a name not in the stated list is exactly the #876 failure the whole
  design prevents. This is a **feed-data quality** problem, not a binding bug.
- **Future lever (separate thinking):** a "speaker names spoken in the intro but absent from
  metadata" signal — either surfaced as a data-quality flag, or a *guarded* extract-from-intro path
  that adds intro-introduced names to the candidate set. Both need their own design; deferred.

## ACCEPTED LIMITATION #2 — diarization merge (guest recovery needs re-diarization)

flightcast 0011: *"…take turns introducing yourselves. Yeah, **I'm Lucas and I'm Axel.**"* — both
guests' self-introductions landed in the **host's** diarization cluster (SPEAKER_00). Consequences:

- **Host mislabel — FIXED (naming-4).** `extract_self_introduced_host` grabbed the first self-intro
  and named the host "Lucas". naming-4 detects a cluster whose distinct self-intros map to **2+
  different stated people** as a MERGE and **suppresses its name** (`suppress_merged_speaker_clusters`
  flag) — a wrong name removed (#876: no name > wrong name). Validated: SPEAKER_00 no longer "Lucas".
- **Guest recovery — NOT fixable in naming.** Lukas' and Axel's self-intros are in the *wrong
  cluster*; their own clusters (SPEAKER_01 1,704s, SPEAKER_02 1,123s) carry no self-intro, so no
  naming heuristic has the evidence to bind them.
- **Future lever (different layer):** a **`rediarize_only`** pass / better diarization to split the
  merged intro turns back onto the guest clusters — a diarization-stability item, not a naming one.

## Net

The naming-4 flightcast pass recovers the clean first-name-intro guests and stops painting a guest's
name onto the host. The residual (metadata mismatch + diarization merge) is **outside the naming
layer** and recorded here for the feed-data-quality and diarization work streams.
