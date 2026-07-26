# ADR-128: Provider-agnostic speaker-name recovery from episode metadata

- **Status**: Accepted
- **Date**: 2026-07-26
- **Authors**: Marko Dragoljevic
- **Related RFCs**: —
- **Related PRDs**: — (#876 speaker quality, #1190 corpus reprocess, #1178/#1179 ASR lock)
- **Related ADRs**: [ADR-126](ADR-126-provider-specific-speaker-labeling.md) (shared core vs
  per-provider strategy), [ADR-123](ADR-123-quality-gate-transcription-failover.md) (coverage gate)

## Context & Problem Statement

Every ASR mistranscribes proper nouns some of the time — they are rare, out-of-vocabulary tokens
that carry no sentence context. This is **not specific to any one model**:

- OpenAI Whisper (the v2.2 transcripts) rendered Kevin Roose as **"Kevin Russo"** and Casey Newton
  as **"Casey Noon"**.
- DGX-local turbo (the v2.3 transcripts) rendered them **"Kevin Roos"** / **"Casey Noon"**, and
  guests like David Duvenaud as **"David Duvino"**, Sebastian Malaby as **"Sebastian Maliby"**.

Because our speaker naming reads names **out of the transcript** (self-introductions, host
introductions), a mistranscribed name is published as the speaker label. Measured on the v2.3 turbo
corpus, only 75% of published names were exactly correct; ~20% more were mangled-but-fixable and ~5%
were unrecoverable (mostly garbage, not real names).

**The correct spelling is almost always already in the episode metadata.** Across the 90-episode
corpus: **100%** of episodes have a title AND a description, and **93%** name at least one guest in
them, spelled correctly ("My guest today is Brian Chesky…", title "Brian Chesky – AI Founder Mode").
The feed blurb likewise states the hosts ("journalists Kevin Roose and Casey Newton").

We **already** exploit half of this: `_canonicalize_to_known_host` snaps a mangled name to a
configured **host** (exact/near first name + soundex/edit-distance surname). It lives in the shared
core and runs for **every** provider — which is exactly why it fixed OpenAI Whisper's own "Kevin
Russo" → "Kevin Roose" on the Deepgram/community-1 v2.2 corpus. The gap is that we do **not** apply
the same matching to **guests** against the metadata-stated guest names.

Two defects, both provider-agnostic:

1. **No guest-metadata canonicalization.** Host manglings recover; guest manglings do not, even when
   the correct spelling sits in `metadata_named` / `detected_guests`.
2. **`known_hosts` is empty on the `full`/transcription path for org-authored feeds.** The
   transcription consumers (`transcription.py`) DO set `job.feed_hosts = cached_hosts` before the
   roster runs — so the anchor plumbing is intact. The defect is upstream: `cached_hosts` itself is
   empty. The host detector on this profile is `gemini.detect_hosts`, which **short-circuits on the
   RSS author tag** — when `feed_authors` is present it returns `set(feed_authors)` verbatim and
   never reads the description. Every one of the corpus's feeds is *org*-authored (`The New York
   Times`, `NPR`, `The Wall Street Journal…`, `Financial Times…`), so it returns the org, which
   `is_network_or_org_author` then strips → `cached_hosts` empty. Meanwhile the descriptions DO name
   the hosts ("Each week, journalists Kevin Roose and Casey Newton explore…"). The `relabel_only` /
   `rediarize_only` paths avoid this because they read hosts from sibling metadata via the
   **deterministic** `detect_hosts_from_feed`, which reads the description statement first.
   (Evidence: `cached_hosts` / `tried.known_hosts` empty on **90/90** v2.3 episodes; all 9 feeds
   org-authored; the deterministic parser recovers real hosts for 6/9 of them.) Nothing to do with
   which ASR ran.

   > **Correction.** An earlier revision of this ADR (and the reverted commit 263b1430) claimed the
   > cause was an un-wired `feed_hosts` argument on the full path. A Fable-5 review showed that fix
   > was **dead code** — `transcription.py` already overwrites `job.feed_hosts` from `cached_hosts`
   > before it is read. The real cause is the empty `cached_hosts` above.

## Decision

Speaker-name recovery from metadata is a **provider-agnostic, shared-core capability**, not a
per-ASR or per-diarizer patch. Three changes, all in (or feeding) the shared naming core — none
scoped to turbo:

1. **Metadata name extraction is a shared input, unconditional of provider.** `known_hosts` (feed
   blurb) and the stated guests (`metadata_named` / `detected_guests`, from title + description) are
   read the same way for every ASR and every diarizer. They already are; this ADR records it as a
   contract, not a scenario-specific step.

2. **Extend canonicalization symmetrically to guests.** Add a guest counterpart to the existing host
   snap: a mangled published name that does not match a `known_host` is snapped to a **stated guest**
   (`metadata_named` ∪ `detected_guests`) by the same fuzzy rule (exact-or-near first name + surname
   within soundex OR a small edit distance). It runs in the shared core after all naming paths
   converge (near the final plausibility gate, ADR-126), for **every provider** — so the v2.2
   Deepgram/community-1 corpus benefits from it too, not only turbo. Guards (added after the Fable-5
   review, see Invariants): an exact match to a stated ref is never re-snapped; a non-host voice is
   never snapped onto a known-host's spelling (preserves the host-candidate gate); corroborated refs
   are matched before the un-corroborated `metadata_named` subjects.

3. **Fix the empty-`cached_hosts` cause: deterministic host fallback.** When the LLM detector yields
   no host after org-author stripping, `detect_feed_hosts_and_patterns` falls back to the
   deterministic `detect_hosts_from_feed`, which reads the host statement out of the **description**
   first — the exact function the `relabel_only` / `rediarize_only` paths already use via sibling
   metadata. This populates `cached_hosts`, which the transcription path already threads onto the
   roster. One place, provider-agnostic; anchors the roster for every `full` reprocess AND first-pass
   ingest, independent of ASR or diarizer. (The earlier tuple-threading approach was reverted as dead
   code — see the correction note in Context.)

## Invariants (unchanged — this ADR must not weaken them)

- **Never author a name.** A name is only ever snapped to one **stated** in the episode's own
  metadata (host blurb, title, description). We do not invent, and we do not snap to a global name
  list. A mangling too far from any stated name (edit distance beyond threshold, first name
  unrecognisable) **stays unrecovered** — a raw `SPEAKER_NN` or the mangled token, per "a wrong
  label is worse than an unnamed voice" (#876).
- **One name, one voice.** A stated name already bound to one voice is not reused for another.
- **Ambiguity abstains.** If a mangled name is near two different stated people, keep it unresolved.
- **Exact matches are never re-snapped.** A published name that already equals a stated ref is left
  alone (else two stated people sharing a first name could move an exact match onto a near-ref).
- **The host-candidate gate is preserved.** A non-host voice is never snapped onto a known-host's
  spelling — a guest self-introducing "Kevin Ross" is not painted host "Kevin Roose" merely because
  the host voice was left unnamed. Host-identity canonicalization stays gated to host-candidate
  voices; a genuinely mangled co-host already carries `role == "host"` and still snaps.

### Accepted bounded risk

The guest snap matches a mangled name against `metadata_named`, which includes people the episode is
*about* but not necessarily in the room (e.g. a lawsuit defendant named in the show notes). A voice
that self-introduces as a genuinely different person whose name is fuzzy-close to such a subject
(e.g. "Sam Alton" → "Sam Altman") could be mislabelled. We accept this bounded tail because
`metadata_named` is **required** for real recoveries (e.g. "David Duvenaud" is present only there,
not in the corroborated `detected_guests`). It is mitigated by matching corroborated refs first and
by the respell-only fuzzy bound; measured incidence on the real corpus was **zero** (see Evidence).

## Non-Goals

- **Not an ASR change.** We do not switch off turbo, re-transcribe, or tune the model. This is
  downstream / relabel-cheap.
- **Not turbo-specific.** Nothing here keys on `dgx_whisper_model` or the transcription provider.
- **Not name invention.** Guests with no metadata reference anywhere (a name that appears only,
  mangled, in the transcript) are **not** recoverable and are out of scope — that residual is the
  ASR's true, separate cost.
- **Does not close turbo's coverage gap.** Turbo also *surfaces fewer* names than OpenAI (weaker
  self-intro/greeting transcription → some voices never get a candidate name at all: 148 vs 183
  published). That is a distinct ASR-quality issue, tracked separately, not addressed here.

## Evidence

Measured on the real v2.3 turbo corpus (90 episodes), re-resolving from the frozen turbo diarization.
Two separate levers, measured separately (an earlier revision reported a single "70%→88%" number
against a name-in-refs oracle that was blind to wrong-voice attribution — that oracle is not used
here).

**Lever 1 — host anchor (deterministic fallback).** The pipeline's `cached_hosts` was empty on
**90/90** episodes (every feed org-authored → the LLM detector's org short-circuit stripped to
nothing). The deterministic `detect_hosts_from_feed` recovers real hosts from the description for
**6 of the 9 feeds** (Hard Fork → Kevin Roose + Casey Newton; Unhedged → Katie Martin + Robert
Armstrong; Invest Like the Best → Patrick O'Shaughnessy; No Priors → Elad Gil + Sarah Guo; The Daily
→ Michael Barbaro et al.; The Journal → Jessica Mendoza + Ryan Knutson). The other 3 (NVIDIA AI
Podcast, Planet Money, Latent Space) name no host in their blurb and correctly stay empty — no
invention.

**Lever 2 — guest snap, measured attribution-aware.** Isolating the snap (host anchor ON in both
arms, only `_recover_stated_names` toggled) across all 90 episodes, and classifying every name it
changes as a **respell** (same person, surname edit/soundex/stem-close — a mistranscription
corrected) or a **swap** (a different person — the accepted-risk failure mode):

| snap effect (90 episodes) | count |
| --- | ---: |
| names changed by the snap | 10 |
| — respell (same person, correctly re-spelled) | 10 |
| — cross-person swap (mislabel) | 0 |

Examples: `Sebastian Maliby → Sebastian Mallaby`, `Nick Allardyce → Nick Allardice`, `RJ Skirinj →
RJ Scaringe`, `David Duvino → David Duvenaud`. The snap is conservative (10 changes corpus-wide) and
made **zero** cross-person mislabels — the accepted `metadata_named` tail risk did not materialise.

Host recovery is directly proven: re-resolving a Hard Fork turbo episode with `known_hosts` present
snapped "Kevin Roos"→"Kevin Roose", "Casey Noon"→"Casey Newton", and the guest "David Duvino"→"David
Duvenaud".

## Consequences

- **The whole corpus improves, not just turbo.** Applied provider-agnostically, the guest snap also
  repairs any names OpenAI Whisper mangled on the v2.2 Deepgram/community-1 corpus, and the host
  fallback fixes any org-authored feed on any provider.
- **Cheap and reversible.** Both levers are shared-core naming logic + a feed-detection fallback; the
  guest snap runs under `relabel_only` (no audio, no ASR), so the existing corpus is repaired without
  re-transcription. The host fallback takes effect on the next `full` reprocess.
- **The deepgram frozen base stays stable.** Per ADR-126, the guest snap is a shared-core tightening;
  the full 154-test diarization suite (incl. the golden-fixture arbiter) is green, and the snap is
  reference-bounded (it may only re-spell to a stated name, never invent or demote).
- **Not yet validated end-to-end on a live `full` run.** Both levers are proven by unit tests +
  deterministic re-resolution of the frozen corpus; a live `full --reprocess` (which re-transcribes
  and calls the LLM) has not been run. That is the remaining validation gap.
