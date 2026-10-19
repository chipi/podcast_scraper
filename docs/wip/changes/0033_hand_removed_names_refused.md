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

Dated: YYYY-MM-DD

_TODO_

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
