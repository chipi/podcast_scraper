---
id: 0033_hand_removed_names_refused
class: full
status: open
issue: none (the operator opens issues; found by prod's upgrade verify, 2026-10-19)
touches:
  - src/podcast_scraper/speaker_detectors/hosts.py
  - src/podcast_scraper/upgrade/migrations/m0015_unpublishable_speaker_names_removed.py
  - src/podcast_scraper/upgrade/migrations/m0033_hand_removed_names_refused.py
---

# 0033 — a name a migration removed by hand cannot come back through a reprocess

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Plan:
[INGESTION-ARC-2026-10-19.md](../INGESTION-ARC-2026-10-19.md), Stage 1. A wrong name is worse than a
hidden insight (operator, 2026-10-19), so this goes before the hidden-insight work.

## 1. Intake

- **Found:** prod's `upgrade verify` after 0031's apply (2026-10-19 08:59Z) exits 1. One of its two
  FAIL lines is `0015_unpublishable_speaker_names_removed: 1 episode(s) still carry a refused name`,
  for "Ep176_ David Roblin on Building" (run 20261014-073935). The same line was on prod's
  2026-10-17 verify, and it is identical before and after 0031 on the clone (`lost/clone30L`,
  `lost/gateworkB6c`). So it predates 0031.
- **0015** removed unpublishable speaker names on 2026-10-03. Its `REMOVE` dict is a hand list of
  junk that no gate rule catches safely: "Before Gene", "As Colin", "Super Willing To Be", "Moral",
  "Generation", "Pan-African". It is operator-approved after reading every published host and guest.
- **The episode:** it was reprocessed on 2026-10-14, after 0015 ran. A migration runs once (the
  ledger records 0015 as applied), so 0015 never sees a later run.
- **0015's plan on the clone** (read-only): 1 name ("Generation"), 1 episode;
  `segments_relabelled=6110`, `quotes_unattributed=109`, `insights_unattributed=58`,
  `insights_surface_to_connect=58`, `gi_persons_deleted=1`, `kg_nodes_deleted=1`,
  `kg_edges_deleted=2`, `bridge_identities_removed=1`, `roster_entries=1`.
- **Related, not this change:** the misspelled-name class and the wrong-person rules (Stage 2).

## 2. Outcome and baseline

- **The outcome:** no insight, quote or segment is published under a name 0015 removed by hand,
  now or after any reprocess.
- **Baseline** (clone 10-16 + 0030, after 0031):
  - published under a hand-removed name: 1 episode, 58 insights, 109 quotes, 6,110 segments, all
    "Generation";
  - 0015's verify: FAIL, 1 episode.
- **Prod:** the same episode and line (prod's verify, 2026-10-19 08:59Z). Prod's counts not read yet.

## 3. Diagnosis by loss

- **Cause, measured:** the hand list lives only in the migration; the pipeline's publish gate
  (`speaker_detectors.hosts.is_publishable_speaker_name`) never reads it. Against that gate today
  (`scratchpad/lost/gatecheck15.py`):

  | Name | published with person shape | published, loose |
  | --- | --- | --- |
  | Before Gene, As Colin, Super Willing To Be | no | yes |
  | Moral, Generation, Pan-African | **yes** | yes |

  So any reprocess can publish "Moral", "Generation" or "Pan-African" again; one already did.
- **Loss:** the 58 insights under "Generation" are a wrong name, the worst class. The other names
  publish nowhere today (0015's verify finds 1 episode).
- **The path, read from the episode (clone, `run_20261014-073935`):** the roster entry is
  `{"id": "guest", "name": "Generation", "role": "guest", "voices": ["SPEAKER_00"], "source":
  "self_intro"}`. The self-introduction detector read segment 93, SPEAKER_00: "I'm Generation X, I
  suppose, without giving my age completely away." as "I'm <name>". The episode's stated guest,
  David Roblin (`source: "episode_metadata"`), stays `placed: false`, no voice. There is no LLM
  resolution in the inputs.
- **So it is a wrong person, not only junk.** SPEAKER_00 is very likely David Roblin: the stated
  guest, who talks about himself as the guest ("I'm a Welshman", "as I'm recruiting someone"). The
  host's "David Roblin, welcome to The Long Run." (segment 29) is diarized onto SPEAKER_00 too, so it
  cannot be read as an address. Two defects:
  1. the self-introduction detector accepts an ordinary word ("Generation" from "Generation X")
     as a name;
  2. the publish gate does not refuse 0015's hand-removed names, so nothing downstream catches it.
- **The repair's choice, for step 4 and the reviewer:** remove the name (the voice becomes unnamed,
  its 58 insights hidden), or name the voice David Roblin. Naming him is better for the listener but
  is a decision from evidence, so it goes the way 0031's entries went: a reviewer reads the episode
  and writes a verdict, and the migration applies the verdict. A rule that seats the stated guest
  here is Stage 2's work, not this change's.

## 4. Prediction

Dated: 2026-10-19

**Clone:** `lost/gateworkB6c` (10-16 snapshot + 0030 + 0031, 0031's measured after-state: S1 14,841;
S1n 7; S1c 0; S2 26,351; S3 7,710; S10a 178; S10b 2; S12 24; A1 357; insights 116,595; 2,685
served episodes).

**The episode** (E), "Ep176_ David Roblin on Building" (`run_20261014-073935`), read from its
files: 62 insights, all shown, routing `surface` (58 under "Generation", 3 under "Luke
Timmerman", 1 with no speaker); quotes: 109 on the "Generation" person, 5 on Luke Timmerman. The
roster places "Generation" (guest, `self_intro`) on SPEAKER_00 and leaves David Roblin
(`episode_metadata`) unplaced.

**The cause, reproduced** (`extract_self_introduced_host`, today's code): "I'm Generation X, I
suppose, …" gives `'Generation'`. `is_plausible_mononym` and `is_publishable_speaker_name` both pass
it; they also pass "Millennial", "Boomer" and "Moral".

**Two repairs for E.** The acceptance (step 5) picks one, from a fresh reviewer's verdict on
SPEAKER_00:

| Number (clone) | Before | U: unname SPEAKER_00 | N: name it David Roblin |
| --- | --- | --- | --- |
| insights published under a hand-removed name | 58 | 0 | 0 |
| quotes published under it | 109 | 0 | 0 |
| episodes 0015's verify fails on | 1 | 0 | 0 |
| S1 | 14,841 | 14,899 (+58) | 14,841 |
| S2 | 26,351 | 26,409 (+58) | 26,351 |
| S1n, S1c, S3, S10a, S10b, S12, A1 | as above | unchanged | unchanged |
| insights shown under "David Roblin" | 0 | 0 | 58 |
| episodes written | | 1 | 1 |

- U removes the name the way 0015 does. 0015's own plan for E on the clone: 6,110 segments
  relabelled; 109 quotes and 58 insights unattributed; 1 GI person, 1 KG node and 2 KG edges
  deleted; 1 bridge identity and 1 roster entry removed. E's 4 other insights stay shown, so S10a
  does not move.
- N renames SPEAKER_00 voice by voice (m0025's rewrite, as 0031 does) from "Generation" to "David
  Roblin", guest, and places the metadata entry on that voice. The same 6,110 segments and 109
  quotes change, and the 58 insights stay shown under the new name.

**Two rules for future runs.** They write no corpus file, so no number above moves because of them:
1. The publish gate refuses 0015's hand-removed names. The clone has 1 episode carrying one (E).
2. The one-word self-introduction rule refuses a word that is a label, not a name ("Generation" in
   "Generation X"). The population it must leave alone: the clone's 93 placed one-word
   `self_intro` names, 70 distinct (`scripts/measure/self_intro_names_census.py`). Predicted: it
   refuses exactly 1 of the 93, E's "Generation", and keeps the other 92.

**Seen in the census, outside this change:** "Kanye's" (1 episode, a16z "What Makes a Great
Entrepreneur?", 48 insights; the transcript reads "I'm Kanye's engineer", a retold call; the
stated guest Ben Horowitz is placed on no voice). The operator, 2026-10-19: it stays out of 0033
and goes to Stage 2 as a category (possessives on every extraction path; the stated guest seated
when the self-introduction is refused). 0033's self-introduction rule does not refuse it, so the
census prediction above stands at 1 of 93.

## 5. Acceptance

_TODO_

## 6. Build

_TODO_

## 7. Measure

_TODO_

## 8. Verdict

_TODO_

## 9. Operator review

_TODO_

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Not covered

_TODO_
