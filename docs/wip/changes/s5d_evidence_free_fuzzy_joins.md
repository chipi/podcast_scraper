---
id: s5d_evidence_free_fuzzy_joins
class: full
status: landed
issue: none (the person-identity design D2; operator decision 2026-11-01, "B plus a reviewed allow-list")
touches:
  - src/podcast_scraper/kg/entity_clusters.py
  - src/podcast_scraper/identity/person_match.py
  - src/podcast_scraper/identity/guards.py
  - src/podcast_scraper/identity/ledger.py
  - src/podcast_scraper/upgrade/lists/s5d_reviewed_fuzzy_joins.json
---

# s5d_evidence_free_fuzzy_joins — two one-episode mentions are not joined on a near spelling alone

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Names are kept out of this
record; ids are used. The measurements and the reads are in eval-data
`data/eval/name-consolidation/d2i9/`.

## 1. Intake

**Reading list:**
- the person-identity design, D2 (plan V3, eval-data `docs/wip/NAME-CONSOLIDATION-ARC-V3-2026-10-30.md`): step 9, join
  reads.
- [s5c_scoped_names_not_pooled.md](s5c_scoped_names_not_pooled.md): the same mechanism, a match-pool
  rule in `_JoinTest._matches` (its 4c).
- [0041_joined_names_respelled.md](0041_joined_names_respelled.md): 0041 respells joined names from
  this ledger.
- #2352 and #2353 (the naming gaps, D37): checked; neither covers these joins.
- The reads, in eval-data `d2i9/`:
  - `fuzzy_joins.py` and its output `fuzzy_no_evidence.jsonl`;
  - `read_pairs.py` → `read_pairs_76.txt`;
  - `read_pairs_59.py` → `read_pairs_59.txt`, with the verdicts in `read_pairs_59_verdicts.md`;
  - `build_allow.py`.

**What the join read found** (step 9, snapshot-prod-20261030, the D2 ledger on `clone-pre0038a`):
- **135 fuzzy person joins with no evidence:**
  - **fuzzy:** the names are not equal after normalisation and not a hyphen or initial variant;
  - **no evidence:** neither id is a voice in any of its episodes, and publisher text names neither.
    The publisher test is `publisher_named`, which counts all three `PUBLISHER_KEYS`, including
    `detected_guests` (NER over the metadata).
- **76 of them join two ids that each occur in one episode.** The read of each (`read_pairs_76.txt`:
  show, both episode titles, the KG's description of each person) found:
  - **68 one person,** spelled two ways (ASR or transliteration);
  - **8 two people** who share a word and a few letters, each in a two-member group.
- **The other 59** (one side in two or more episodes; `read_pairs_59.txt`) were read too:
  - **2 two people,** each in a two-member group;
  - **1 unclear:** a node named by the surname alone, described as one person, under an id minted
    for a relative;
  - **56 one person.**
- **Each of the 10 wrong joins** puts one person's mentions under another person's name. They have
  been on prod since D1. On this ledger 0041 would also write the wrong spelling into the mention's
  episode files (section 4).
- **Operator decision, 2026-11-01:** "B plus a reviewed allow-list".
  - A fuzzy join between two evidence-free one-episode ids is refused, unless a human read listed
    the pair as one person.
  - **Rejected:** a narrower string-only rule (both differing words are names elsewhere in the
    corpus). It caught about half of the wrong joins and split about 9 right ones.

## 2. Baseline

Two baseline commands, both on `clone-pre0038a`:
- `d2i9/fuzzy_joins.py`: 135 fuzzy joins with no voice and no publisher evidence, written for
  reading. 76 of them are one-episode on both sides (`read_pairs.py`: "76 pairs").
- `d2i9/ledger_counts.py`:
  `{"person_groups": 372, "org_groups": 100, "joins": 658, "person_joins_full": 365,
  "person_joins_scoped": 184, "org_joins": 109, "blocked_joins": 9}`.

0041's plan on this ledger (`d2i9/s5d_0041_effect.py`): **573 episodes, 2,864 files, 1 fold.**

## 3. Diagnosis

The cluster rule (`cluster_person` mode) accepts a near spelling as a match. For two mentions in one
episode each, the spelling is the only thing joining them. No guard looks at that: the join guards
fire only inside a shared episode, and these pairs share none.

A one-episode rule also does not hold over time. Once either id of a wrong pair gains a second
episode, the rule no longer covers the pair and it would merge again. That is why the reviewed
two-people verdicts are kept as well (rule 2 below).

## 4. Prediction

Dated: 2026-11-01T20:35 (revised after the first fresh read, section 5, before any code)

**The switch.** `build_entity_canonical_map` takes a new keyword, `pair_rule`, which defaults to
None (today's behaviour). Only the ledger build (`identity/ledger.build_ledger`) passes one, so the
other callers are unchanged. `pair_rule(cand, member)` returns one of three verdicts:
- `None`: the edge counts as today;
- `"ignore"`: the edge is not evidence (rule 1);
- `"apart"`: the candidate may not join this cluster (rule 2).

**Rule 1 (an ignored edge),** per edge in `_JoinTest._matches`. The edge between candidate `a` and
member `b` is ignored when all of these hold:
1. both are persons, and neither id is episode-scoped (s5c covers those);
2. each has at most one episode (`EntityCandidate.freq <= 1`; an id with no `episode_id` counts as
   having none);
3. neither is a voice (`voice_of[id]` non-empty) nor publisher-named (`publisher_named`) in any of
   its episodes;
4. the names are a near spelling only. A new public helper,
   `identity.person_match.near_spelling_only(a, b, language)`, is true when the names are neither
   `_equal_after_normalisation` nor `_hyphen_or_initial_variant` under that language's vocabulary.
   The language is the one the match itself uses (`_pair_language(a, b)`). An equal or
   hyphen/initial edge is never ignored;
5. the two ids are not in one group of the list's `groups`.

A candidate joins a cluster when **any** edge to it matches and is not ignored. A cluster where every
matching edge is ignored is not joined, and nothing is written to `blocked_joins`: a near spelling
alone is not evidence of one person.

**Rule 2 (a reviewed two-people pair),** in `_JoinTest.__call__`, checked first, against the full
cluster (not the pool that `no_one_word_bridge` or s5c narrow). When a member of the cluster and the
candidate are a pair in the list's `apart`, and the candidate's name matches that member's
(`_match`), the join is refused:
- **It is a blocked join.** A row `[candidate_id, member_id, "reviewed_apart",
  episode_id]` is written to `blocked_joins`, with the candidate's first episode (sorted).
- The candidate goes on to the next eligible cluster, as for any refusal.
- **For the 8 pairs both rules cover,** rule 2 is evaluated first, so each gets a row.
- The pair stays two people as the corpus grows, whatever the episode counts or evidence. Clusters in
  this build never merge, so no third id can put the two into one group either.

Rule 1 writes no row. It does not refuse a join; it says a near spelling alone is not evidence.

The pair rule is built in `identity/guards.reviewed_pair_rule(evidence, reviewed)` (the ledger's
evidence, and the list). The ledger build passes it to `build_entity_canonical_map(pair_rule=)`.

**The list,** `upgrade/lists/s5d_reviewed_fuzzy_joins.json`, ids only:
- `groups`: 64 groups, 132 ids, the 68 one-person pairs grouped by canonical. Rule 1 is
  pair-agnostic inside a group, so a build edge between two members of a group (not member to
  canonical) passes too.
- `apart`: these 10 pairs:
  - `person:alex-goldman` / `person:alex-goldmark`
  - `person:chip-wilson` / `person:chris-wilson`
  - `person:david-albert` / `person:david-hilbert`
  - `person:john-gledson` / `person:john-gladstone`
  - `person:john-terry` / `person:john-kerry`
  - `person:nick-warner` / `person:senator-warner`
  - `person:rick-perry` / `person:richard-perry`
  - `person:rose-kennedy` / `person:rosemary-kennedy`
  - `person:sergei-karalyov` / `person:sergei-karaganov`
  - `person:wisena-williams` / `person:wilson-williams`

Each entry carries the read it rests on.

**The future:**
- a new evidence-free pair of one-episode ids stays two names;
- it joins once a person gains a second episode, a voice or publisher text, or a later read adds it
  to `groups`;
- a reviewed two-people pair never merges.

**Predicted on the snapshot.** The ledger is rebuilt on `clone-pre0038a` with its served ledger kept,
as in D2's deploy; the before and after scoreboards come from `ledger_counts.py`:

| | before | after |
|---|---|---|
| joins | 658 | **648** |
| person groups | 372 | **362** |
| full-name person joins | 365 | **355** |
| scoped person joins | 184 | 184 |
| org joins | 109 | 109 |
| org groups | 100 | 100 |
| `blocked_joins` rows | 9 | **19** (10 `reviewed_apart`) |
| evidence-free fuzzy joins (`fuzzy_joins.py`) | 135 | **125** (76 → 68; 59 → 57) |

- **Joins lost: exactly the 10 pairs above.** Gained 0, moved 0. Of the 20 ids, 0 join any other
  cluster.
  - Pre-code evidence: `d2i9/s5d_order_replay.py` lists, for each of the 20 ids, every person candidate
    that shares a show with it and that the build's own `_match` accepts. Its output:
    "ids: 20 matches other than the partner: 0".
  - So a refused id has no other cluster to go to, and no third id matches it.
  - All 20 are non-scoped (`is_scoped_person_id` false for each, in the same output), so all 10 lost
    joins are full-name joins (365 → 355).
  - Any other difference is listed.
- The 68 listed pairs and the 57 other evidence-free joins are still joined.
- Canonical ids and `display_name` of every surviving group are unchanged.
- **0041,** on the new ledger. The figures come from 0041's own plan on the served ledger, and on
  that ledger with the 10 groups deleted (`d2i9/s5d_0041_effect_apart.py`; rule 1's 8 alone:
  `s5d_0041_effect.py`). That is a dissolution, not a rebuild; the replay above is why the two agree.
  - **episodes 573 → 564** (566 with rule 1's 8 alone);
  - **files 2,864 → 2,833** (2,840 with rule 1's 8 alone);
  - folds 1 → 1;
  - **10 episodes differ,** exactly the ten that each hold one id of a pair. In 9 of them 0041 no
    longer writes the other person's spelling and the episode drops out of the plan. The tenth is
    still written for its other renames, minus the wrong one;
  - 0041 respells only a joined name (`episode_renames`). The 20 ids are unjoined, so they keep
    their own spelling. No fold can newly apply: the two ids of each pair share no episode.
- All four D2 gates are rerun, since the code digest changes. Their scoreboard figures other than
  0041's change by nothing beyond the rows above.

**The rule version** stays `cluster-person-no-bridge-guarded/3`, as for s5c (4b there): this is part
of D2's `/3`, which is not on prod yet.

### 4c. A revision after the first measurement (section 7)

Dated: 2026-11-01T21:40

**What the first measurement showed:**
- **2 joins lost that section 4 did not predict:** `person:enoch-karangwana` →
  `person:enoch-godongwana`, and `person:yu-shang` → `person:yushan`.
- **1 `blocked_joins` row more than predicted:** `person:yu-shang` refused against `person:yu-shun`
  by `mention_cooccurrence`.

**The edges** (`d2i9/s5d_4c_check.py`, the build's own `_match` and pool, against the served
groups):
- **`person:enoch-karangwana`:**
  - it does not match the canonical directly;
  - it matches only `person:enoch-gaurangwana` and `person:enoch-gorongwana`, both one-episode and
    evidence-free, so rule 1 ignored those edges;
  - with the group listed, the rule returns None for both.
- **`person:yu-shang`:**
  - its pool is the one two-word member, `person:yu-shan` (one episode, no evidence), which rule 1
    ignores;
  - the voiced canonical is a one-word name, so `no_one_word_bridge` keeps it out of a two-word
    candidate's pool.

**Why the prediction missed them.**
- `fuzzy_joins.py` judged each member against its group's canonical, and counted evidence when
  either end had it. Both canonicals have a voice and publisher text, so neither member was among
  the 135.
- The build tests the edge a member actually joined through. Both joined through another
  one-episode, evidence-free near spelling, and rule 1 now ignores that edge.
- The first reader's replay used the same member-to-canonical view.

**The read of the two:**
- `person:enoch-karangwana` is one person with the group. Every description is the same office on
  the same show, and its episode's publisher text names the canonical's spelling. **A right join
  lost.**
- `person:yu-shang` is unclear. Its one episode also holds the group's voiced host and a guest of a
  near name, and the descriptions disagree. **Left split:** a wrong name is worse than a hidden
  one (operator, 2026-10-19).
  - The new row is its order effect: freed from the group, it tried the guest's group, and the
    existing guard refused it there.
  - The group's `display_name` changes from the spelling of the member that left to the
    remaining member's full spelling. That spelling is equal after normalisation to the publisher's
    one-word form (spacing removed); the departed one is not.

**The change:** one group added to the list's `groups`, all four ids of `person:enoch-godongwana`'s
group. Its basis is in the file under `groups_added`. No code changes. The list then holds 65 groups,
136 ids, and 10 `apart` pairs.
- **All four, not just the canonical,** because the restored edge runs between two of its one-episode
  members. Rule 1's group test is pair-agnostic inside a group (condition 5).
- No `apart` entry involves these ids, so rule 2 cannot interfere.

**What the miss means for section 4.**
- Section 4's populations came from `fuzzy_joins.py`'s member-to-canonical view, so they missed any
  member that joined through an evidence-free edge to a non-canonical member of a group with
  evidence. Those populations are: the 135 / 76 / 59, the read splits of section 1, the row
  "evidence-free fuzzy joins 135 → 125", the "57 other evidence-free joins still joined", the first
  reader's replay, and the spot-check denominators.
- **The measurement settles the population.** Rule 1 changes a join only by losing it. Section 7's
  first measurement lists every lost join, 12, and gained 0 and moved 0. So the joins rule 1 and
  rule 2 touch on this snapshot are exactly those 12: the 10 pairs plus the 2 above, and all 12 have
  now been read.
- The row "evidence-free fuzzy joins" is withdrawn as a gating number. The lost-join list replaces
  it.

**Predicted** (the same rebuild and the same `ledger_counts.py`):

| | 4's prediction | 4c |
|---|---|---|
| joins | 648 | **647** |
| person groups | 362 | **362** |
| full-name person joins | 355 | **354** |
| scoped person joins | 184 | 184 |
| org joins / groups | 109 / 100 | 109 / 100 |
| `blocked_joins` rows | 19 | **20** |

- **Lost: the 10 pairs, and `person:yu-shang`.** Gained 0, moved 0.
- **`display_name` changed: 1,** `person:yushan`'s group. No surviving canonical changes.
- **0041: episodes 573 → 563, files 2,864 → 2,830.**
  - Computed with `d2i9/s5d_4c_check.py`: 0041's own plan on the first measured ledger, with
    `person:enoch-karangwana` put back in its group (4c's ledger, simulated).
  - The same script gives the first measured ledger 562 / 2,827. `person:enoch-karangwana`'s episode is respelled
    again.
  - The 0041 gate rerun on the real rebuild must give 563 / 2,830.

### 4d. Wrong joins the evidence-free read could not see (step 9's wider read)

Dated: 2026-11-01T23:20

**The reads** (eval-data `d2i9/`), each on D2's measured ledger (`s5d_measure_4c.out`):
- **The cards** (`card_read.py`): every card whose shown name changes against prod's, 151.
- **The near-spelling joins with any evidence** (`near_only_members.py`, `near_pairs_ranked.py`):
  - 276 pairs, edge view, none of them in the earlier reads (`near_pairs_ranked.txt`), each read
    from that file's names and flags; the short forms (a surname, a title) are set aside;
  - the 11 unclear ones read with context (`read_group_cards.py`).
- **Every bare-name join** (`scoped_joins_read.py`): 184 scoped ids in 84 groups.

**Found two people, 16 ids.** Each verdict rests on both sides' show, episode and KG description,
side by side in `d2i9/s5d_4d_evidence.out`, which also covers the 3 replay pairs and the one gained
join. The scoped id of `person:zoran-mamdani`'s group is the least certain: its description fits an
academic namesake better than the group's politician. Unclear goes to split, as for
`person:yu-shang`.
- **Full-name pairs, 5.** Each has one side voiced or publisher-named, so rule 1 never applied:
  - `person:stephen-king` / `person:stephen-hawking`. The card shows the wrong person's name,
    since the publisher's text names the other;
  - `person:mike-jones` / `person:mike-johnson`;
  - `person:thomas-ahle` / `person:thomas-hales`;
  - `person:michael-ovitz` / `person:michael-moritz`;
  - `person:mark-cockleberg` / `person:mark-zuckerberg`.
- **Bare names joined to another person's full name, 11 scoped ids** in the groups of:
  - `person:christopher-alexander`, `person:eva-leila`, `person:harriet-ryan` (2),
    `person:geetha-godwin`, `person:henry-i`, `person:rajmohan-gandhi`, `person:thomas-cromwell`,
    `person:zoran-mamdani` (with `person:zoran-mandani`), `person:steve-pierce`,
    `person:peter-pavel`.
  - Read time shows a bare name in its group's spelling, so these put a wrong full name on the page.
- **The rest:** the other 271 respelling pairs, 173 bare names, and the remaining cards were read
  as one person. Some cards show the worse of two equal spellings; that is a display tie, not a
  join.

**The change:** 20 pairs added to `apart`, 30 in all. No code changes.
- The 12 scoped-id pairs and the 5 full-name pairs above.
- **3 that the order replay found** (`d2i9/s5d_order_replay_4d.py`):
  - freed, the scoped id of `person:henry-i`'s group matches `person:david-i` and `person:roger-i`;
  - the scoped id of `person:zoran-mamdani`'s group matches `person:mayor-mamdani`;
  - each is another person, so each is an apart pair.

**Order effects** (`s5d_order_replay_4d.out`, each of the 32 ids against every candidate on its
shows):
- **One new join:** the scoped id of `person:rajmohan-gandhi`'s group matches
  `person:mohandas-gandhi`, a single id today.
  - **Read:** both descriptions name one person, the independence leader (`s5d_4d_evidence.out`).
  - **Visit order:** both have 1 episode, and the scoped id's name sorts before the full name's. So
    it is visited first: refused against `person:rajmohan-gandhi`'s cluster (3 episodes, visited
    earlier), it forms its own cluster, which `person:mohandas-gandhi` then joins. A full-name
    candidate may match a scoped member; s5c narrows the pool only for a scoped candidate.
- No other id has a match outside its pairs.

**Predicted** (the same rebuild and `ledger_counts.py`):

| | 4c measured | 4d |
|---|---|---|
| joins | 647 | **632** (−16 lost, +1 gained) |
| person groups | 362 | **350** (13 dissolve, 1 new) |
| full-name person joins | 354 | **349** |
| scoped person joins | 184 | **174** (−11, +1) |
| org joins / groups | 109 / 100 | 109 / 100 |
| `blocked_joins` rows | 20 | **39** |

- **The 19 new rows** are each pair that meets in the visit order: 5 full-name pairs, plus 14 for
  the scoped ids:
  - 1 each for the scoped ids of the groups of `person:christopher-alexander`, `person:eva-leila`,
    `person:geetha-godwin`, `person:rajmohan-gandhi`, `person:thomas-cromwell`, `person:steve-pierce`
    and `person:peter-pavel`;
  - 2 for the two scoped ids of `person:harriet-ryan`'s group;
  - 3 for the scoped id of `person:henry-i`'s group (against `person:henry-i`, `person:david-i` and
    `person:roger-i`);
  - 2 for the scoped id of `person:zoran-mamdani`'s group (against `person:zoran-mamdani` and
    `person:mayor-mamdani`).
  - `person:zoran-mandani` joins the canonical's cluster before it can meet that scoped id, so that
    pair writes no row.
- **The 13 groups that dissolve:** `christopher-alexander`, `eva-leila`, `harriet-ryan`,
  `geetha-godwin`, `henry-i`, `rajmohan-gandhi`, `thomas-cromwell`, `steve-pierce`, `peter-pavel`,
  `stephen-hawking`, `mike-johnson`, `thomas-hales`, `michael-moritz`.
- `zoran-mamdani` and `mark-zuckerberg` keep two members each.
- **Lost:** 4c's 11 plus these 16 ids. **Gained:** 1, `person:mohandas-gandhi` with the scoped id
  above. **Moved:** 0.
- **0041: episodes 563 → 556, files 2,830 → 2,805.**
  - Computed with `d2i9/s5d_4d_evidence.py`, 0041's own plan on 4c's measured ledger. The 16 ids
    are taken out, the new group is added, and every `display_name` is recomputed by the build's
    own `_add_display_spellings`.
  - The simulated ledger has 350 person groups, as predicted. The only `display_name` change is the
    new group's.
  - Bare names are never respelled (0041's placeholder rule), so the drop comes from the 5
    full-name ids.
  - The gate rerun on the real rebuild must give 556 / 2,805.

**Not covered by 4d:**
- **New wrong bare-name joins.** At least 6% of the 184 read wrong (11).
  - The count is a lower bound: some bare ids have no description and could not be judged.
  - The two large groups of one show's co-hosts, `person:yu-shun` and `person:yushan`, hold bare ids
    whose described roles disagree. They are not read for collisions of one name.
  - Until voice evidence (1C), a new bare name can still join a namesake's full name on its show.
    Only a later read catches it.
- **Cards where the publisher's own spelling loses** (`card_read.out`): one person, but the
  publisher's text names a spelling the group's fuzzy match does not reach. Examples:
  `person:wicknell-shevaya`, `person:john-glenn-hill`, `person:sofia-du`, and `person:jeff-glow`
  (whose person also has a separate group, `person:jeff-guo`). That is a display defect and a missed
  join, not a wrong merge; it is outside this change.
- **`person:stephen-king`** itself holds two people of one name (an economist and an author). The
  rule compares names, so it cannot split them.

### 4e. The reviewed list holds every read verdict, and the report's open pair

Dated: 2026-11-02T03:10

**The report.** `scripts/measure/near_spelling_joins.py` (`identity/near_joins`) lists every ledger
member whose only link into its group is a near spelling and that no read has reviewed. It is the
step to run after each ledger rebuild until voice evidence (1C) decides these joins.
- **Its first run** on the clone: 451 open pairs. 280 had been read as one person by name this
  session; 170 were bare names and title forms of the same person, read in
  `d2i9/near_joins_unread.txt`.
- **The one-person verdicts go to `groups`**, merged by union-find: 65 → 329 groups, 790 ids.
  - `groups` acts only on rule 1's pairs, so the ledger does not change. Measured
    (`scratchpad lst_check.py`): joins 632 → 632, id map identical, `display_name` and evidence
    identical, `blocked_joins` 39 → 39.
  - No apart pair sits inside one group.
- **One pair stayed open,** `person:john-fisher` with the bare id
  `person:unresolved-fisher-f8ad589c-6503-4d8d-81b2-4a64dce7f60f`, which has no description. Read
  from its episodes:
  - the bare id is a voiced speaker in one show's bookstore episode;
  - the full id's two episodes are two people of one name, a businessman and a sixteenth-century
    bishop.
  - So the bare id is another person, and the pair goes to `apart` (31 pairs).

**Predicted** (the ledger rebuilt on the clone):
- joins 632 → **631**, scoped joins 174 → **173**;
- person groups 350 → **349** (the two-member group dissolves);
- `blocked_joins` 39 → **40** (one `reviewed_apart` row);
- 0041's plan **unchanged**: bare names are never respelled, and the full id keeps its own name;
- the report after: **0 open pairs**.

Not covered: `person:john-fisher` and `person:stephen-king` each hold two people of one name; a name
rule cannot split them.

## 5. Acceptance

**First fresh reader:** 2026-11-01, a sonnet reviewer subagent with read-only tools, one pass before
any code.
Dated: 2026-11-01T20:05
- **Verdict:** "Not complete".
- **Its findings,** each taken in section 4's revision:
  - integers for every scoreboard row;
  - 0041's counts as integers;
  - the refused pairs listed by id;
  - the reading list and the baseline commands;
  - the rule stated per edge;
  - the match's own language for the near-spelling test;
  - an id with no episode;
  - the evidence definition, `detected_guests` included;
  - an explicit switch.
- **Its replay of the visit order,** on the clone's ledger with the real `_match`: 148 members keep an
  allowed edge, and exactly the 8 rule-1 pairs lose every edge.
- **Its spot check:** 20 of the 68 one-person verdicts and all 8 two-people verdicts; it disagreed with
  none.
- **Rule 2 and the 2 pairs from the 59-pair read** were added after this read. They need the second
  reader.

**Second fresh reader:** 2026-11-01, a sonnet reviewer subagent with read-only tools, one pass before
any code.
Dated: 2026-11-01T20:55
- **Verdict:** "Not complete".
- **Its blockers,** each taken above:
  - rule 2's order effects: the replay;
  - rule 2's `blocked_joins` rows and the order of the two rules;
  - `guards.py` in `touches`;
  - rule 2 checked against the full cluster, and the "third name" sentence reworded;
  - the rule-1-only 0041 counts, and the dissolution stated;
  - the scoped check;
  - #20 named as thin.
- **What it confirmed:** the 0041 integers, both 59-pair two-people verdicts, 10 of the 56 one-person
  verdicts, the list counts and the public-repo rule.
- Every row is taken, as in s5c's 4b.

**Third fresh reader (4c):** 2026-11-01, a sonnet reviewer subagent with read-only tools, one pass
over 4c before the second measurement.
Dated: 2026-11-01T22:05
- **Verdict:** "Not complete".
- **Its findings,** each taken in 4c:
  - the two members' real edges, shown by the build's own test;
  - 0041's 4c integers;
  - the list's new totals;
  - all four ids in the added group;
  - `person:yu-shang` and the opposite defaults of the two unclear cases in Not covered;
  - section 4's populations named as resting on the member-to-canonical view, with the lost-join
    list as the measure that settles them;
  - the evidence for the `display_name` change.
- **What it confirmed:** the arithmetic, the list's integrity, and the public-repo rule.

**Fourth fresh reader (4d):** 2026-11-01, a sonnet reviewer subagent with read-only tools, one pass
over 4d before its measurement.
Dated: 2026-11-02T00:10
- **Verdict:** "Not complete".
- **Its findings,** each taken in 4d:
  - person names in prose, replaced by ids;
  - 0041's prediction recomputed with the build's own `display_name`;
  - the gained join read;
  - every verdict's both-sides evidence put in one file;
  - the visit-order frequencies shown;
  - the bare-name rate stated as a lower bound, with the unread co-host groups named;
  - the cards whose publisher spelling loses;
  - the list's `apart_basis`.
- **What it confirmed:** the 4d integers, the visit-order rows (the `person:zoran-mandani` no-row
  claim in either order), and the list's integrity.

## 6. Build

**Tests, written first** (`tests/unit/podcast_scraper/identity/test_guards.py::TestReviewedPairRule`,
and one in `test_ledger.py`). Before the code: 12 failed, 56 passed.
- two one-episode near spellings are not joined;
- a reviewed group, a second episode, a voice or publisher text lets them join;
- an equal name and a middle-initial variant are never ignored;
- an id with no episode counts as at most one;
- any edge that counts admits the candidate;
- a reviewed apart pair is refused and recorded, also when a third member matches both;
- without a pair rule the build is unchanged;
- the shipped list loads, and its ids are disjoint.

One existing ledger test (the canonical tie) gained a second episode for its person, since two
one-episode near spellings no longer join at all.

**Code:**
- `kg/entity_clusters`: `build_entity_canonical_map(pair_rule=)`; `_JoinTest._matches` drops
  ignored edges from the pool; `_JoinTest._apart` handles rule 2 and writes its rows.
- `identity/guards`: `GUARD_REVIEWED_APART`, `REVIEWED_LIST`, `load_reviewed`,
  `reviewed_pair_rule`.
- `identity/person_match.near_spelling_only`.
- `identity/ledger.build_ledger` passes the rule.

**A defect found by the existing tests and fixed in the build:** `near_spelling_only` tested
equality before the language's folds only. A German pair equal under the folds counted as a near
spelling; it now tests both.

**Mutants** (`replay-proven-fix` `mutate.py`, 11): killed 11 of 11.
- They covered: the pool filter, rule 2 off, its row, a check of the first member only, the episode
  limit at 2 and at exactly 1, the voice test, the publisher test, the group test, the fold
  equality, and the hyphen/initial test.
- The last one survived the first run. The middle-initial test was added, and then it was killed.

**Areas run:** identity and KG unit tests, `946 passed`. Black, isort, flake8, codespell and
`make type` ("Success: no issues found in 2414 source files") pass.

## 7. Measure

**The first measurement:** the ledger rebuilt read-only on `clone-pre0038a`, with the served ledger
kept (`d2i9/s5d_measure.py` → `s5d_measure.out`). It found 12 lost joins, 2 of them unpredicted,
and 20 rows. That led to 4c.

**The second measurement,** after 4c (`s5d_measure_4c.out`, plus `ledger_counts.py`):

| | before | 4c predicted | after |
|---|---|---|---|
| joins | 658 | 647 | **647** |
| person groups | 372 | 362 | **362** |
| full-name person joins | 365 | 354 | **354** |
| scoped person joins | 184 | 184 | **184** |
| org joins / groups | 109 / 100 | 109 / 100 | **109 / 100** |
| `blocked_joins` rows | 9 | 20 | **20** (10 `reviewed_apart`) |

- **Lost: 11,** the 10 apart pairs and `person:yu-shang`. Gained 0, moved 0.
- **Surviving canonicals changed: 0.** `display_name` changed: 1, `person:yushan`'s group.
**The third measurement,** after 4d (`s5d_measure_4d.out`, plus `ledger_counts.py`, against the
4c ledger the clone then served):

| | 4c measured | 4d predicted | 4d measured |
|---|---|---|---|
| joins | 647 | 632 | **632** |
| person groups | 362 | 350 | **350** |
| full-name person joins | 354 | 349 | **349** |
| scoped person joins | 184 | 174 | **174** |
| org joins / groups | 109 / 100 | 109 / 100 | **109 / 100** |
| `blocked_joins` rows | 20 | 39 | **39** (29 `reviewed_apart`) |

- **Lost: 15. Moved: 1. Gained: 0.**
  - The id-map diff counts the scoped id of `person:rajmohan-gandhi`'s former group as moved, to
    `person:mohandas-gandhi`. 4d's prediction counted that as one lost and one gained; it is the
    same change.
  - The 15 lost are exactly 4d's other 15 ids.
- **New canonicals: 1,** `person:mohandas-gandhi`. **`display_name` changed: 0** in surviving
  groups.
- **0041's figure** (556 episodes, 2,805 files predicted) is measured by the 0041 gate rerun,
  recorded with the D2 gates.

## Not covered

- **`person:yu-shang`** (4c): unclear, **left split.** Its one episode holds the group's voiced host
  and a guest of a near name, and the descriptions disagree. It joins once it gains a voice,
  publisher text, a second episode, or a later read adds it to `groups`.
- **The two unclear cases go in opposite directions,** and the reason is the default. The rule
  splits only what rests on a near spelling alone. The 59-pair case joins its group through an edge
  rule 1 does not touch (the group's id has two or more episodes), so the rule keeps it joined; no
  read found it wrong. `person:yu-shang`'s only edge is a near spelling, so the rule splits it, and
  the read could not show it right.
- **The unclear pair of the 59-pair read** (a surname-only node under a relative's id): left joined;
  the node's description is the person it joins.
- **Fuzzy joins with voice or publisher evidence:** not refused. That evidence is what rule 1 asks for.
- **The list is a snapshot read,** with two consequences:
  - A `groups` entry only ever lets a join through; it never forces one. A wrong entry would keep a
    wrong join.
  - Each verdict rests on one reader's judgment against KG descriptions, which are LLM-written.
    - The two fresh readers' spot checks agreed with every verdict they checked: 20 of the 68, all 8,
      10 of the 56, and both of the 59-pair read's two-people verdicts.
    - Three rest on thin evidence: #44 and #62 of the 76, and #20 of the 59, whose descriptions name
      different cities.
- **The 56 multi-episode joins judged one person:** each side was read on one of its episodes only.
- **1C** (voice evidence for scoped names) is deferred by the operator.
