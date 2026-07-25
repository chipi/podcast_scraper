# Follow-up: diarization cluster artifacts behind the naming fabrications (#1330)

Status: **naming-layer containment landed; the diarization-layer cause is deferred.** Tracked here
per repo convention (local follow-up, not a GH issue) so the cause-fix isn't lost.

## What the naming layer now contains (landed, #1330)

Re-running the real corpus surfaced two "fabricated person" classes. Both are **diarization
artifacts** the naming layer now contains at cause-in-its-own-layer, verified by deterministic
replay on the real episodes:

1. **Cold-open montage cluster.** Diarization merges a show's opening montage — several hosts'
   garbled self-intros in one breath ("I'm Kevin Russo… I'm Casey Noon…") — into a single short
   `SPEAKER_NN`. Naming it after the first intro fabricated an extra person ("Kevin Russo") while
   the real Kevin Roose was correctly named on his own long cluster.
   - Fix (`_self_intros_by_voice` + `distinct_self_introductions`): a voice that introduces itself
     as ≥2 distinct people is refused a self-intro name **when short** (`talk_time <
     MONTAGE_CLIP_MAX_TALK_S`, tied to the intro window). A *long* dominant voice that merely
     absorbed a merged clip keeps its name, resolved from its own leading self-intro.

2. **Detected-guest name forced onto a bumper.** The forced one-name-one-voice path painted a
   still-"spare" detected guest ("Robert Pape") onto a 30s "We'll be right back" bumper, when the
   same person was already named under a title ("Professor Pape").
   - Fix (`_name_guest_voices` + `_surname_token`): a detected-guest name is spare only if the same
     person isn't already on the roster. Only **honorific-form** roster names ("Professor Pape")
     claim their surname, so a genuinely distinct same-surname guest is not suppressed.

Both fixes fail toward **unnamed**, never toward a wrong label — the codebase's stated invariant.

## The actual cause (diarization layer, deferred)

The naming layer cannot see that the montage clip and the real host are the same person, nor split
a merged cold-open back apart. The root defects live one layer down:

- **Merge:** a cold-open montage (and sometimes a co-host's intro clip) is merged into one cluster,
  or into a real speaker's cluster — the montage-detection and talk-time gate are containment, not a
  cure. The clip's talk time is still attributed to whoever owns the merged cluster.
- **Split:** conversely, one physical speaker can be split across two `SPEAKER_NN`, over-counting
  `num_speakers` and dividing talk-time / talk-share.

Everything downstream that keys on the voice id inherits the error: `num_speakers`, talk stats,
ordered turns, ad/cameo gating.

## The cause-fix (a diarization-layer change)

Normalise clusters before naming, when the evidence says two clusters are one speaker (or one
cluster is a montage of several): embedding-centroid distance (the embedding-evidence backend
already loads per-voice embeddings), identical/near self-intro strings, temporal exclusivity +
turn-taking. Blast radius is why this is **not** in the naming PR: it rewrites voice ids, so it must
run before (or atomically with) everything that consumes them, with its own tests and its own
before/after corpus measure. It belongs on its own branch, keyed to **v2.2** (community-1
diarization is the version that changes this behaviour — measure there).

## Entry criteria before attempting

- A Tier-2 matrix row (ADR-095) reproducing a montage-merge and a split, with **synthetic** names
  (never a real episode — the never-commit-real-episodes rule).
- Confirmation from the v2.2 community-1 sweep of how often merges/splits actually occur; if rare,
  the naming-layer containment is sufficient and this stays deferred.

## Related

- #1330 (naming-layer containment, landed), #876 (partial-naming), #1226 (the one-name-one-claim
  guard the containment must not break), v2.2 (community-1 diarization).
