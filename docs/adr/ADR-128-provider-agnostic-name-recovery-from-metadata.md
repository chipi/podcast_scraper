# ADR-128: Provider-agnostic speaker-name recovery from episode metadata

- **Status**: Accepted
- **Date**: 2026-07-26
- **Authors**: Marko Dragoljevic
- **Related RFCs**: —
- **Related PRDs**: — (#876 speaker quality, #1190 corpus reprocess, #1178/#1179 ASR lock)
- **Related ADRs**: [ADR-126](ADR-126-provider-specific-speaker-labeling.md) (shared core vs
  per-provider strategy), [ADR-123](ADR-123-transcription-coverage-failover.md) (coverage gate)

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
2. **`known_hosts` is wiped by `full --reprocess-existing-only`.** That path takes audio from the
   #947 cache without re-fetching the RSS, so the new run's metadata `feed` block is written empty at
   roster time → `hosts_from_feed_statement` returns nothing → host canonicalization is silently
   disabled corpus-wide. (Observed: v2.2 `relabel_only` had `known_hosts` on 60/90 episodes; v2.3
   `full` had it on 0/90.) This has nothing to do with which ASR ran.

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
   Deepgram/community-1 corpus benefits from it too, not only turbo.

3. **Fix the `known_hosts`-wipe at its cause.** A reprocess must preserve the feed hosts. `full
   --reprocess-existing-only` reads the feed block (title/description/authors) from the **on-disk
   episode metadata** it is reprocessing (the same file `relabel_only` already reads) instead of
   writing an empty block, so host canonicalization is available at roster time regardless of RSS
   re-fetch. Provider-agnostic — it fixes any `full` reprocess.

## Invariants (unchanged — this ADR must not weaken them)

- **Never author a name.** A name is only ever snapped to one **stated** in the episode's own
  metadata (host blurb, title, description). We do not invent, and we do not snap to a global name
  list. A mangling too far from any stated name (edit distance beyond threshold, first name
  unrecognisable) **stays unrecovered** — a raw `SPEAKER_NN` or the mangled token, per "a wrong
  label is worse than an unnamed voice" (#876).
- **One name, one voice.** A stated name already bound to one voice is not reused for another.
- **Ambiguity abstains.** If a mangled name is near two different stated people, keep it unresolved.

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

Measured on the v2.3 turbo corpus (148 published names) against the episode metadata:

| bucket | count | share |
| --- | ---: | ---: |
| already exactly correct | 111 | 75% |
| mangled, **recoverable** by metadata snap (hosts + guests) | 30 | 20% |
| unrecoverable (mostly garbage: ad testimonials, opener leaks, botched possessives) | 7 | 5% |

→ **~95% correct achievable** downstream, no re-transcription. Host recovery is proven (re-resolving
a Hard Fork turbo episode with `known_hosts` present snapped "Kevin Roos"→"Kevin Roose",
"Casey Noon"→"Casey Newton" instantly); guest recovery is the symmetric half this ADR adds.

## Consequences

- **The whole corpus improves, not just turbo.** Applied provider-agnostically, the guest snap also
  repairs any names OpenAI Whisper mangled on the v2.2 Deepgram/community-1 corpus.
- **Roster parity stops being ASR-confounded.** The v2.3-vs-v2.2 speaker comparison (31% identical
  raw) was dominated by the `known_hosts` wipe + missing guest snap; with these fixed it reflects the
  true residual ASR gap, which is small.
- **Cheap and reversible.** All three changes are shared-core naming logic + a reprocess metadata
  read; they run under `relabel_only` (no audio, no ASR), so the existing corpus is repaired without
  re-transcription.
- **The deepgram frozen base must stay stable.** Per ADR-126, the guest snap is a shared-core
  tightening; validate with the 90-episode deepgram relabel arbiter (it may only *add* correct names,
  never demote a real one — snapping is reference-bounded).
