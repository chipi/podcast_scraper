# Naming Phase C — handover (2026-10-13)

Branch `fix/naming-phase-c-2304`, commit `35733cc1a` (#2304). The operator decided to move on to
S5 / K8 / F7 with the 7 known regressions below written down, not fixed.

## What shipped on the branch

| Change | Live defect it fixes (nightly 2026-10-13) |
| --- | --- |
| A. A feed-stated host stays in the host pool even when an ad-shaped intro clip says his name | Macro Musings: "I am your host, David Beckworth" typed as an ad, pool emptied, guest seated as host |
| B. `plausible_person_names()` drops groups / institutions from guest and metadata names | BizNews: "NTT's Japanese owners" painted onto a voice as "Japanese owners" |
| C. Opener rule: with a host already seated, the opener takes a further seat only if its first 3 turns name the show or a stated person by full name | Exposed by A: In Our Time / Curious Cases guests carrying the merged "This BBC podcast is supported by ads" tag seated as host; Capitalisn't guest cold open seated and named "Luigi Zingales" |
| D. A lone unnamed host seat with two unclaimed stated hosts is the one it does not keep addressing | Hard Fork: one host seat, two names |

Operator principle behind A (2026-10-13): the episode/show metadata is the centre of authority; a
stated host heard in an ad or intro clip is confirmation, never a reason to drop him.

## Numbers

Prod corpus, 2,632 episodes, `roster_replay.py --repool --signatures`, main and branch replayed
back to back (one episode, MLST "Alexander Mattick", was reprocessed by backfill batch 15 between
the runs and is excluded):

- publish no named voice 222 -> 211; named voices 5,014 -> 5,028; episodes with a named host
  2,235 -> 2,249; episodes losing every name 0.
- 49 episodes change; every changed voice read: **42 better, 7 worse**.

Gold gate (`naming_gate.py --repool --signatures`):

| Set | Better | Worse | Note |
| --- | --- | --- | --- |
| dev_v1 (101) | 1 | 0 | Amjad Masad named |
| val_v1 (500) | 5 | 1 | the worse case is a label error the operator accepted (below) |
| val_v2 (150, scored once) | 1 | 0 | one wrong name now unnamed (wrong 31 -> 30, missing 96 -> 97); correct 224 = 224 |

val_v1 label correction (operator, option 1): case v328, The Rest Is History "A Murderous Affair:
Death at Mayerling (Part 2)", SPEAKER_00. Labelled "host, no name" because "each voice addresses
both Tom and Dominic"; that is true of SPEAKER_01 (Tom 3, Dominic 2) but SPEAKER_00 addresses
Dominic 6 times and Tom 0, so Tom Holland is right. `labels_v1` is unchanged; the correction is in
`.test_outputs/naming_gold/val_v1/label_corrections.json` (local, real data, never committed).

## The 7 known regressions (NOT fixed)

| # | Episode | Voice | Main | Branch | Why | Idea, untested |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Capitalisn't — How Capitalism Became Global ft. Sven Beckert | SPEAKER_02 (44%, lectures, addresses "Luigi") | guest | host (step 4 count-fill) | Fix A put Bethany McLean back in the pool (cap 1 -> 2); the LLM put Sven's name on SPEAKER_03 (34%, asks host questions), so step 4 saw no unplaced guest and seated the dominant talker | Distrust an LLM guest name on a voice that asks most of the questions. A talk-share guard ("out-talks every seated host") was tried and REJECTED: it removed 11 real second hosts (Hard Fork 4, The Rest Is Politics 5, Capitalisn't 2) |
| 2 | The a16z Show — The Infrastructure Behind the Machine Age | SPEAKER_03 | Raghu Raghuram (guest, forced) | unnamed guest | Main's forced name only worked because SPEAKER_00 (a guest) was wrongly seated as host; with it back among the guests there are 2 candidates for 1 name and the forced rule declines | Probably correct to decline; whether SPEAKER_03 is Raghu is unverified |
| 3 | Curious Cases — Phantom Pain | SPEAKER_13 (whole episode; "First of all, Tamar, can you explain…") | host | guest | The host's cluster carries the merged BBC ad tag and names neither the show nor a guest in its first 3 turns, so rule C refuses it, same as the guest-carrying-the-tag cases | A host asks the stated guests questions by first name across the episode; measured on the 34 openers it does NOT separate cleanly (In Our Time guests address each other too: Cybernetics "Jacob" 2x, Roman Arena "Kathleen" 2x) |
| 4 | Curious Cases — Going Viral | SPEAKER_11 ("What is a virus? … Martha, were you going to add on that?") | host | guest | as 3 | as 3 |
| 5 | In Our Time — Melvyn Bragg meets Misha Glenny | SPEAKER_03 ("So, Misha, how does it change now?") | host | guest | as 3; both people are pool hosts in this special | as 3 |
| 6 | In Our Time — Welcoming Misha Glenny | SPEAKER_04 ("why do you think you're the right person for this role?") | host | unknown | as 3 | as 3 |
| 7 | Curious Cases — The Diamond Throwdown | SPEAKER_04 (0–557 s of 2,074) | host | guest | as 3 | unclear whether this voice is a host at all |

Root cause shared by 3–6: when the network ad tag is diarized into a voice, the opener test sees
the tag, not the person. A cleaner fix is upstream: skip the feed's recurring ad-tag turn when
choosing the opener (`recurring_text` already exists in the pipeline but `_opening_voice` does not
read it), so the opener is whoever speaks first after the tag.

## Replay kit (prod, `compose-api-1:/tmp/nc`, real data — delete when done)

- `run4.sh <roster> <hosts> <out>` (4 shards of the committed `roster_replay.py`), `coverage_cmp.py`,
  `review_changes.py`, `naming_gate.py`, `trace_ep.py <jsonl> <episode substring>`,
  `opener_probe.py` (the 34-opener probe).
- Variants: `roster_main.py` (main), `roster_c5.py` (= this commit before Black), replays
  `r_main.jsonl` / `r_c5.jsonl`.
- Diffs MUST be replayed back to back: the backfill reprocesses episodes between runs.

## Still to clean up

Real-data copies: prod `/tmp/nc_kit.tgz`, prod `compose-api-1:/tmp/nc` (gold sets, replays),
local `scratchpad/nc_kit*`. Each delete needs the operator's per-path OK.
