# Host seat without a name: the question-rate signal (#2366)

Measured 2026-10-15 on the prod snapshot `snapshot-prod-20261015` and the naming gold sets.
Nothing is built. This note is the "design note with the candidate signal measured" step of #2366.

## How big the gap is

The newest run of each served episode, 2,500 episodes with speaker diagnostics:

- **479 have no named host** in the served record.
- **The current roster seats a host in 258 of them** (`roster_replay.py --repool`, deterministic
  rungs only, no LLM step: 142 by self-introduction, 137 by known hosts, 4 other). These are stale
  records that a relabel would change. Their correctness has NOT been read.
- **221 still have no host on current code.** 94 have no host pool at all; about 100 have a pool
  (the right names, e.g. The Rest Is History's two hosts) but no voice gets matched to it. 28 have
  unpunctuated text. Only 16 have a single dominant voice, so this is mostly not a diarization
  merge.

By feed, the 221 are led by The a16z Show (50), The Rest Is History (18), MLST (15), The Open
Africa Podcast, BizNews Radio, Latent Space (12 each) and Philosophy For Our Times (11).

## The signal

Candidates are the voices holding at least 10% of the episode's talk, in a punctuated transcript
(at least 20 full stops). The host is the candidate whose question rate ("?" per 1,000 words) is
at least twice that of every other candidate. Otherwise the signal picks nobody. No names, no
language-specific words.

Frozen on dev_v1, then scored once on the held-out sets. Episodes with gold host(s) and guest(s):

| Signal | dev_v1 | val_v1 (held out) | val_v2 (held out) |
| --- | --- | --- | --- |
| question rate, 2x margin | 47 / 48 = 0.98 | 177 / 186 = 0.95 | 56 / 64 = 0.88 |
| most questions (no margin) | 66 / 75 = 0.88 | 329 / 399 = 0.82 | 92 / 123 = 0.75 |
| first to speak | 53 / 75 = 0.71 | 278 / 399 = 0.70 | 89 / 123 = 0.72 |
| last to speak | 54 / 75 = 0.72 | 318 / 399 = 0.80 | 91 / 123 = 0.74 |
| most talk | 5 / 75 = 0.07 | 51 / 399 = 0.13 | 12 / 123 = 0.10 |

The 2x rule picks a voice in about half the episodes (48 of 75, 186 of 399, 64 of 123).

The one dev_v1 error is an MLST guest holding 85% of the talk with many rhetorical questions. No
guard was added for it: tuning on a single dev case is not allowed, and the val errors were not
read.

## What is not known

- **Precision on the gap population itself.** The gold sets score every episode, and most of them
  already get their host from a name rung. The 221 may be harder: panels, co-hosted shows,
  storytellers.
- **What a seat buys.** A host seat with no name changes the role, not the published name. It
  names someone only where the pool has exactly one host, or with a later name rung.
- **The 258 newly seated hosts are unread.**

## Options (operator's decision)

1. Build it as the lowest host rung: it seats a host only when no other rung did, and names it
   only from a one-host pool. Then replay with every changed voice read, gates, mutants. At val_v2's
   0.88, roughly 1 seat in 8 would be wrong.
2. Leave it out and keep the class on the "an LLM would solve better" list
   (`NAMING_KNOWN_ISSUES.md`): who leads the conversation is a question of meaning.
3. Relabel first. 258 of the 479 change on current code, and the real gap is then re-measured on
   records the current code wrote.
