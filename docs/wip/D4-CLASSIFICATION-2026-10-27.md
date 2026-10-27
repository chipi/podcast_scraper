# #2094 D4: the 43 served candidates, sorted (2026-10-27)

D4, as prod measured it over 3006 served episodes on 2026-10-27: a person-named voice holding
under 5% of the talk while a voice holding 40% or more is unnamed. List:
`~markodragoljevic/prod-naming-20261027/d4_candidates.json` (43 episodes). The question for the
naming arc: is the small voice's name another person's (a wrong name, rank 1), or is the name
right and the big voice simply unnamed (hidden insights, a lower rank)?

Read-only; no corpus written. Scripts in the session scratchpad (`rm27/d4_vs_0034.py`,
`d4_three.py`, `d4_trace.py`).

## Result

| Bucket | Episodes | How decided |
|---|---|---|
| Name kept by 0034's review as the small voice's own | 40 | in `upgrade/lists/0034_talk_floor_names_removed.json`; the shipped verdicts say `CORRECT` for every one of their small named voices: 46 voices: basis self_intro 25, addressed 8, undecidable 5, continuity 3, clip_own_person 3, host_read 2 (counted per voice, rm27/d4_vs_0034.py; 6 episodes have two) |
| Kept by the live pipeline's host exemption, name confirmed | 1 | `https://iono.fm/e/1732633`: trace `talk_floor kept_host` (share 0.050), both night-1 readers `self_intro` segment 176 |
| Kept by the host exemption, undecidable | 1 | `631e269e-b6a1-4241-8992-8a9d97cb83ff` SPEAKER_00: trace `kept_host` (share 0.021), name from `llm_resolution`; both night-1 readers `UNDECIDABLE`; the role (host) is doubtful against the episode title, the name is not decided by the text |
| Not read | 1 | `ce2dc3a9-d06c-43f3-918e-b4ec014a82ff` (two host names at 2.1% and 2.9%): ingested after snapshot-prod-20261025, not in the night-1 sample; its files are in snapshot-prod-20261027, read in the final validation pass |

**Wrong names found: 0** of the 42 read. So none joins 0037. The 5 undecidable-kept voices of
0034 and `631e269e` keep a name the text does not decide; the operator's rule (a wrong name is
worse than a hidden one) would take such names off only on evidence that they are someone else's,
which none of them has.

## What D4 then measures

For the 41 episodes whose small voice carries its own name, the big unnamed voice is the loss:
hidden insights, not a wrong person. Whether it is the same person split across two voices
(a short clip or intro named, the main voice not) or someone else is not decided here. That is a
naming-coverage question (a name ADDED to a voice), and it goes through its own change record if
pursued; it is not part of 0037.

## Not covered

- `ce2dc3a9-d06c-43f3-918e-b4ec014a82ff` is unread (above).
- The 0034 verdicts were read as shipped; they were not re-judged here.
- The big voices of the 41 episodes were not read.
- The host exemption (`_apply_talk_floor`: seated host + LLM-called host + no third-person
  mention) is how `631e269e` keeps an undecided name; whether the exemption is too wide is not
  measured here.
