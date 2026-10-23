---
id: 0034_talk_floor_names_removed
class: full
status: built, not applied
issue: none (the operator opens issues; naming arc, 2026-10-23)
touches:
  - src/podcast_scraper/upgrade/migrations/m0034_talk_floor_names_removed.py
  - pyproject.toml (package data upgrade/lists/*.json)
---

# 0034 — the talk floor on served episodes: a name off a voice that barely speaks

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md). Plan:
podcast-scraper-eval-data `docs/wip/NAMING-GATES-2026-10-22.md`, the coding plan, item C7. A wrong
name is worse than a hidden insight (operator, 2026-10-19).

## 1. Intake

- **The pipeline change.** New episodes get the talk floor (`speaker_name_min_talk_share` 0.05 on
  `prod_dgx_full`, naming ladder doc v26).
- **Why served episodes need a migration.** They were written before the floor, and 88% of
  two-voice episodes were named by older code that the pipeline will not re-run. Text only: a
  voice's talk share comes from its segments on disk.
- **T3 is not part of this change.** The introduction-cue rule was measured beside the floor, but
  the operator decided on 2026-10-23 to ship the floor only. On the runs the corpus serves, T3's
  name extraction put phrases such as "York Times" on voices: 23 fires on the measured splits, 12
  right.

## 2. Outcome and baseline

- **The outcome:** no name is served on a voice holding under 5% of its episode's talk, unless a
  review reads the name as that voice's own.
- **Baseline, 10-17 snapshot copy (`lost/gateworkB6c`, read only).**
  - `find_changes` lists 510 voices in 357 served episodes.
  - On the four labelled segments, 92 names are under the floor:
    - 84 bad: 81 ad / promo / clip, including 4 host-read ads, and 3 spurious;
    - 1 wrong;
    - 6 right and 1 role error. These are short host intros / outros and fragments; none is an
      ad read.

  Source: eval repo NAMING-RETHINK, "Talk floor 5%: host-read ads".
- **Prod:** not counted yet. The list is made on a fresh prod snapshot (section 5).

## 3. Diagnosis

- **The detector reproduces the measured rule on the labelled episodes.** Of the 92 measured
  names, 87 are listed.
- **The 5 not listed:** their labelled run is no longer the run the corpus serves; the detector
  reads the served one.
- **The 21 listed beyond the measurement:** voices the labels do not score (low-confidence labels,
  mostly clips, promos and unknowns).

## 4. Design

- **A frozen, reviewed list.**
  - `find_changes` runs the floor over a corpus copy.
  - `write_list` freezes it in `upgrade/lists/0034_talk_floor_names_removed.json`.
  - The migration applies that list and nothing else.
- **The review.** The reviewer reads each listed voice against the name the floor would remove,
  using the verdicts' own meaning:
  - `WRONG_PERSON` (the name is not this voice's) takes it off;
  - `CORRECT` / `CORRECT_MISSPELLED` keeps it, for example a host's short intro;
  - a listed voice without a verdict fails `verify`, and without a verdicts file nothing is
    written.
- **The rewrite.** It goes through 0029's voice-by-voice rewrite with 0031's derived surfaces:
  transcript prefixes and offsets, turns, diagnostics, context, ad maps, metadata counts, GI, KG
  and bridge. It is all or none per episode, and an episode whose stored name or role moved on is
  refused untouched.
- **The list is bound.** The verdicts header's `list_sha256` is the hash of the WHOLE list
  (every field of every entry), the list file is part of the migration's digest (`list_file`),
  and the receipts header records the list's hash. An entry edited after the review refuses the
  load.
- **No verdicts file:** `plan` prints one refusal line; `apply` raises and the upgrade run stops
  there with nothing written (the runner's contract, as 0031's).
- **Undo:** from `.podcast_scraper/upgrade-backups/0034/`, while each file is still as written.
- **Not in the registry yet.** An empty list with no verdicts file would fail `verify` on every
  upgrade run.
- **Tests** (`tests/unit/podcast_scraper/upgrade/test_m0034_talk_floor_names_removed.py`):
  - the floor rule;
  - list freeze and load;
  - the detector on a corpus;
  - apply, verify and undo, with the name off every surface the fixture writes (metadata,
    segments, transcript, turns, diagnostics, GI persons and quotes, KG);
  - an entry edited after the review refuses the load;
  - plan without a verdicts file is a refusal line;
  - the list file is part of the digest;
  - a `CORRECT` verdict keeping the name;
  - no verdicts file;
  - a moved-on voice refused;
  - not registered.

## 5. Still to do before apply

1. Make the list on a fresh prod snapshot copy (`--find <copy>`).
2. Do the full read audit: read every listed voice's stretch, and judge the downstream LLM impact
   by identity.
3. Have a fresh reviewer write the verdicts file.
4. Produce the clone outcome table: before and after on the snapshot copy, every surface.
5. Register the migration, with its version.
6. Get the operator's recorded go.
7. Apply with the corpus-migration skill.
