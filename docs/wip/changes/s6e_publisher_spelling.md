---
id: s6e_publisher_spelling
class: full
status: landed
issue: none (the person-identity design S6; operator decision, 2026-11-01)
touches:
  - src/podcast_scraper/identity/ledger.py
  - src/podcast_scraper/identity/guards.py
---

# s6e_publisher_spelling — the publisher's own text is the reference spelling of a person

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Names are kept out of this
record; the measurements are in eval-data `data/eval/name-consolidation/d2i9/`.

## 1. Intake

- **Operator decision, 2026-11-01:** "I would want spelling that's coming from official title,
  description, whatever, to win. And that's a point of reference, if they exist there." This closes
  the S6 result's open decision (eval-data `s6/RESULT-2026-10-31.md`: "linking the show text to the
  group by a fuzzy name match ... an open decision").
- **S6a's rule today** (`ledger._add_display_spellings`): publisher text picks a MEMBER, but only one
  it names exactly (folded name or minted id, `guards.EpisodeEvidence.publisher_named`). The shown
  spelling is always a member's own. A group whose every episode misspells the person cannot show
  the publisher's spelling.
- **The publisher fields** (`guards._publisher_names`, the diagnostics' `tried`): `known_hosts` (the
  feed), `metadata_named` (every name the episode's metadata states: title and description,
  `diarization/pipeline.py`), and `detected_guests`.
- **Found by 0041:** a host's group holds two ASR misspellings; the feed names the real spelling, and
  0025's rule on the respelled episode wants it.
- The repo's rules this keeps: titles stay in a display name (m0017); a one-word name never beats a
  full one (m0017, 0041).

## 2. Baseline

Snapshot-prod-20261030, the ledger in memory: 475 person groups. Truth set (s2 truth, g0 rows, the
pairs whose canonical is a group): train 12 wrong, held-out 9 wrong, under the served rule.

## 3. Diagnosis

The rule takes a member's name, so the publisher's text is evidence for a choice among misspellings,
never the spelling itself.

## 4. Prediction

Dated: 2026-11-01

The change, in `ledger._add_display_spellings`, with the publisher names each episode states kept on
`EpisodeEvidence`:

- A publisher name of an episode holding a group member counts for the group when it is:
  - a member's name, folded; or
  - a full name that the roster's own predicates (`roster._same_person`, which needs one surname;
    `_same_person_on_one_episode`, the episode-scope one) match to a member's full name.
- It never counts when it is exactly the name of another person present in that episode.
- The shown spelling is the publisher's own text, voted by the number of episodes stating it; ties
  go to the longer, then the lexical, spelling.
- A bare publisher name never beats a full one; titles stay.
- With no publisher name, the rule is unchanged.
- `display_evidence`: `{"rule": "publisher_text", "episodes": n}`.

Predicted on the snapshot (`d2i9/spelling_publisher_text.py`, the prototype the code must equal):

- **11 of 475 groups change** (10 after 4b's tightening).
  - All but one read as the show's own spelling (public figures).
  - 1 gains its title, as the publisher writes it.
- **Truth set:** held-out 9 → 9 wrong; train 12 → 13 wrong. The one row is a person two shows' texts
  spell differently ("C." as a middle initial or not). Both spellings are publisher text; the readers
  recorded one show's. Not a wrong name; stated.
- **Joins:** unchanged; only the spelling moves.
- **Must not move:** joins, canonical ids, members, unjoined people; groups with no publisher name.
- The prototype's output and the built ledger's `display_name` must be equal for every group (step 7).
- **Then 0041 writes these spellings;** its gate is rerun.

### 4b. Rows added after the fresh read (the rule tightened before any code)

- **Only this episode's nodes of the group are matched,** not every spelling in the group: a
  publisher name could otherwise match a member from another episode. The prototype then drops one
  group of the 11.
- **"Full" counts a name without its leading titles**: a title plus a surname is one name, so
  "Professor X" cannot fuzzy-match every member with that surname.
- **Two publisher names that both fit a member, neither exactly, give no vote** for that episode. It
  could be two people, e.g. a relative with a near-identical name. An exact name settles the choice.
  Measured: 0 episodes dropped this way on the snapshot.
- **`detected_guests` alone** (NER over the metadata, not the publisher's own list) counts only on
  an exact match.
- Measured again (`d2i9/spelling_publisher_text.py`):
  - **10 of 475 groups change**; 0 of them only in case or accents;
  - the truth set is as before: held-out 9 → 9, train 12 → 13 (the same two-show row).
- **Must not move (added):**
  - `display_evidence` of every unchanged group;
  - groups whose only match is a surname-only or title-only form.
- **The fixpoint (step 7):** build the ledger, apply 0041, rebuild; `display_name` and
  `display_evidence` identical.
- **`display_evidence` has no reader** outside the ledger and its tests (searched `src` and `web`).
- **Step 8:** the 10 against each episode's publisher text itself; every tie between two publisher
  spellings.

## 5. Acceptance

Fresh reader, 2026-11-01, a sonnet reviewer subagent with read-only tools, before any code: "not
complete"; two wrong-person holes and five rows. All are taken in 4b, the rule tightened and
remeasured.

## Not covered

- A person whom no publisher text names keeps the member rule.
- Two shows' publisher texts that disagree: the episode count decides; both are references.
