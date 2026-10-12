# Naming ladder — known issues and limits

Companion to [the naming ladder](NAMING_LADDER.md) (what each rung does) and the
[naming gate runbook](NAMING_GATE_RUNBOOK.md) (how a change is judged). This page lists what the
ladder still gets wrong, how big each class is, why a ladder rule does not fix it, and what would.
Measured on 2026-10-12, at the end of the #2304 naming arc (branch `fix/naming-coverage-2304`).

## How these were measured

- **The failure atlas** (`scripts/measure/naming_atlas.py`) lists every gold-labelled voice the
  ladder gets wrong, with where the right name was available and the rung that dropped it.
- **The whole-corpus replay** (`scripts/measure/roster_replay.py --shard K/N --trace-out`, then
  `scripts/measure/naming_replay_compare.py`) gives absolute coverage on every served episode, and
  every changed voice with its own words.
- **The coverage scanner** (`scripts/measure/naming_coverage.py`) counts published episodes and
  quotes with no named voice.

Numbers below are the validation set (`val_v1`, 500 episodes) unless marked dev (`dev_v1`, 101)
or prod (2,616 served episodes, replay).

| Outcome on val | Voices |
| --- | --- |
| correct name | 643 |
| missing name | 411 |
| wrong name | 100 (43 of them swaps) |
| spurious name | 17 |
| role error | 18 |

Of the missing names, 140 were stated or proposed somewhere (feed hosts, metadata, the model,
a self-introduction) but not placed, 33 were only in the title, and 238 were nowhere in metadata,
title or transcript.

## Known issues

### K1. Two voices carry each other's names (swaps)

- **Size:** 43 val / 12 dev voices. 38 of 55 come from the model's per-voice verdict
  (`llm_resolution`) on co-host pairs and two-guest panels (Kevin Roose / Casey Newton, Sarah Guo /
  Elad Gil, The Daily's two reporters).
- **Why no rule:** address evidence (who says whose first name) was measured over every named pair
  in both gold sets (411 correct, 19 swapped). A "swap when address says so" rule breaks more
  correct pairs than it fixes at every threshold (>= 2 and 3x: 8 broken, 5 fixed; >= 4 and 3x: 4
  broken, 1 fixed): diarization bleed puts the other person's greeting in a cluster. Unnaming
  model-named voices instead loses more correct names than wrong ones in every evidence bucket
  (worst bucket: 26 right, 23 wrong).
- **What would fix it:** voice identity across a feed's episodes (the host's voice recurs), or a
  change to the model's naming input. Issue #2333.

### K2. A host's second diarization cluster stays unnamed

- **Size:** about 27 val voices (Capitalisn't, Complex Systems, Peter Attia, Dwarkesh, ChinaTalk,
  Odd Lots). The host's name is on the main cluster; the gold also labels the split-off one.
- **Why no rule:** the ladder publishes one name per voice on purpose (`_recover_stated_names`'s
  over-split exception). Telling that a second cluster is the same person needs the voice, not the
  text. Issue #2333.

### K3. Two hosts merged into one cluster (The Rest Is History)

- **Size:** 30 prod episodes with no named voice; the merged voice holds 80-85% of the talk and
  says "so true, Tom" to itself.
- **Why no rule:** either name on the merged voice is half wrong; it stays unnamed by design
  (scenario `two_hosts_merged_into_one_voice`). Needs re-diarization. Issue #2333; parked by the
  operator.

### K4. The model's feed-host verdict on a lecture guest's voice (Dwarkesh)

- **Size:** 7 prod episodes (Sarah Paine lectures, Jensen Huang).
- **Why no rule:** dropping the model's host name on the voice that owns the talk fixed Dwarkesh
  but changed 54 voices and stripped real hosts on host-dominant shows (Eric Topol, Lenny, Robert
  Armstrong, Shane Parrish). Per-feed talk-share history is not available for the shows that need
  it. Issue #2333.

### K5. Publisher transcripts that were never diarized

- **Size:** 96 prod episodes (Explaining Brazil 45, In Moscow's Shadows 39, The Flip 8, Korea
  Deconstructed 3, Every 1): `transcript_source = direct_download`, empty roster.
- **Why no rule:** there is no voice to name, even where the text says "Hello, my name is
  Gustavo Ribeiro". Re-transcribe with diarization, publish an episode-level name, or accept as a
  limit. Issue #2332; parked by the operator.

### K6. The name is nowhere

- **Size:** 238 val voices: no feed host, metadata, model verdict, self-introduction, title, or
  even the surname in the transcript.
- **Why no rule:** there is nothing to place. Only host discovery (feed-level evidence, RSS author
  tags) or the model could supply a name.

### K7. Title-only guests that are never addressed

- **Size:** after F2/F2c/F2d, of the 44 dev+val title-only voices: 9 never have their name said,
  12 only by first name, and 5 are deliberately not named because nobody addresses them by first
  name (Alan Waxman, Josh Kushner, Michael Klarman, Branko Milanovic, Varun Mohan).
- **Why:** a full name said on air also matches products and segments ("Project Astra", "Cosmic
  Queries", "Surge AI"); requiring first-name ADDRESS is what keeps those out. Issue #2329.

### K8. A metadata-stated guest who owns the talk (parked rule)

- **Candidate:** one spare metadata person, and exactly one unnamed non-host voice with more than
  half the talk, so that voice is the person. Gold (dev+val): 27 right, 1 wrong (BRAVE,
  guest-hosted); at 15% share it was 32 right, 14 wrong (hosts' second clusters).
- **Why parked:** it narrows a design that was built and reverted (`roster.py`, "NOT DONE: ...
  drawing its candidate from `metadata_named`"): it admitted people never in the room, including
  a man who died in 1956, and `test_a_stated_name_we_failed_to_place_is_our_DEFECT` pins that.
  If picked up: require the surname spoken in the episode, read every new name for "did this
  person speak", keep that test binding.

### K9. An unpublishable self-introduction with no alternative holds the host seat

- **Example:** Trivium China, "I'm your host, Tribune co-founder, Andrew Polk" read as "Tribune".
- **Why:** dropping unpublishable self-introductions was tried; it lost Kenny Malone ("Hi, it's
  Kenny" snaps to the host) and took the guest role from about 12 real guest voices. Only a
  replacement by the voice's own publishable introduction ships. Issue #2331.

### K10. Edge intro clips typed as ads

- **Size:** 23 dev+val voices, 0.2-2.4% of the talk (mostly Capitalisn't's recorded opener "I'm
  Bethany McLean... And I'm Luigi Zingales").
- **Why left:** the hosts' real voices carry their names; naming a 0.2% clip serves no listener.

### K11. Near-even two-voice interviews and the ownership guard

- **Example:** The Pragmatic Engineer at 51-52% / 48% (Janvi Kalra, Steve Yegge): the host seat
  "owns the conversation" and the forced host name is refused.
- **Why left:** letting a seat that performs the host role keep the name while it owns the talk
  was replayed: all 8 voices it named host were guests or merged clusters (Stewart Butterfield as
  Lenny, Garry Tan as Anish Acharya). The guard stays.

### K12. All-lowercase transcripts

- **Size:** 42 of 2,602 served episodes (Colombia Calling, The Rest Is History, Empire, FUTURES).
- **Why:** capitalisation is the only name signal left in a self-introduction; names stated
  elsewhere are still read case-blind (scenario `lowercase_transcript_with_stated_names`).

## Changes tried and rejected (so they are not tried again blind)

| Change | What the replay showed |
| --- | --- |
| Refuse every unstated introduced name | lost 59 names, many correct |
| Edit-distance host snap (each part <= 2) | merged Kevin Ross into Kevin Roose, Daniel Smith into Daniela Smyth |
| Drop the model's host name on a dominant voice (F3) | 54 voices changed, real hosts stripped |
| Ownership guard off for host-performing seats | 8 of 8 changed voices wrong |
| Swap pairs by address evidence | more correct pairs broken than fixed |
| Drop unpublishable self-introductions | lost Kenny Malone, ~12 guest roles |
| Title vouched by a full name said on air alone | named "Project Astra", "Cosmic Queries"; lost 4 correct names |

## Measurement caveats

- **The validation set was read while tuning on 2026-10-12.** The failure atlas was built on
  `val_v1` and its failures were read to rank the classes above and to shape F2d and the parked
  K8 threshold. That breaks the runbook's rule (score val once, never read it while tuning), so
  val results for changes made after the atlas (`2bed7f4c3`, and K8 if it is ever picked up) are
  optimistic by an unknown amount. Every change was also replayed on all served episodes with
  every changed voice read, which does not depend on val. A fresh validation set is the fix.
- **The replay cannot run the model.** It uses the model's stored verdicts, so it is a floor on
  what production does, not a prediction.
- **The snapshot moves under reprocessing.** Compare two replays only when both ran on the same
  snapshot (side by side); the dev atlas moved from 26 to 27 wrong names between morning and
  afternoon on 2026-10-12 with identical code, because backfill batches reprocessed episodes.
