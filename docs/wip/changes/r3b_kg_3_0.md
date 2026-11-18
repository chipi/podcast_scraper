---
id: r3b_kg_3_0
class: full
status: open
issue: none (the R3b epic, GI 4.0 and KG 3.0; tracked in the private plan)
touches:
  - src/podcast_scraper/upgrade/migrations/m0051_kg_3_0.py
  - src/podcast_scraper/upgrade/registry.py
  - src/podcast_scraper/kg/v3_file.py
  - src/podcast_scraper/kg/io.py
  - src/podcast_scraper/kg/pipeline.py
---

# R3b (KG half) — every served podcast KG file rewritten from KG 2.1 to KG 3.0, and the writer switched

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Builder: the database
session (the operator, 2026-11-17: R3b is the database session's). The migration lives in public's
framework and lands through public. The GI half has its own record in Common
(`docs/wip/changes/r3b_gi_4_0.md`); this one follows its decisions where they apply.

## 1. Intake

- Common `docs/wip/platform-plan/epics/E04-r3b-podcast-corpus-ontology-v3.md`: R3b's goal, scope
  (KG 3.0: podcast and episode map onto source and item with `HAS_ITEM`, hosts and guests become
  people with roles, `PUBLISHED_BY` from the publisher registry, podcast ids never change), the
  writer switch shipped with the migration, the freeze order, the exit check.
- `docs/architecture/corpus/ontology-v3.md`: KG 3.0's nodes, edges, roles; "Reading v2 artifacts as
  v3" (the mapping `kg/as_v3.py` implements; "making the files strictly v3 is R3b's migration, which
  decides what each extra property becomes").
- `docs/architecture/kg/kg.v3.schema.json`: KG 3.0, strict; `Person.properties.respelled_from`
  allowed since 470268707 (R3b's request: the identity ledger reads it).
- `src/podcast_scraper/kg/as_v3.py`: the lossless view this migration writes, minus what only a view
  carries (`adapted_from`) and the root `episode_id` the schema does not allow.
- ADR-178 (ontology v3 in two halves), ADR-181 (type-specific fields).
- The GI half's record and its operator decisions (served files only; step 3 as a risk ranking).

## 2. Outcome and baseline

A format migration: nothing a listener or reader is shown moves; every served KG file becomes 3.0.

Copy: `/work/closelistening/snapshots/gate-0049d_published_names_corrected` (snapshot 2026-11-16
with 0046-0049; restated 2026-11-18; one episode re-counted on `gate-0049e` with the GI half).

`scripts/measure/r3b_kg_census.py` and `r3b_kg_validate.py` (Common checkout, read-only):

| | served |
| --- | --- |
| KG files, all `schema_version` 2.1, all with `extraction.transcript_ref` | 3,376 |
| nodes: `Episode` / `Podcast` / `Person` / `Organization` / `Object` / `Topic` | 3,376 / 3,376 / 19,048 / 20,456 / 8,284 / 33,512 |
| edges: `HAS_EPISODE` / `HOSTS` / `GUESTS_ON` / `MENTIONS` (all `from`/`to`) | 3,376 / 3,262 / 2,337 / 81,086 |
| `Person` nodes with `respelled_from` | 722 |
| adapted files failing KG 3.0 (schema of 470268707) | 3,376, each only on the root `episode_id` |

Scope (the operator, 2026-11-17, for R3b): served files only.

## 3. Diagnosis by loss

Operator accepted (2026-11-17, R3b): a ranked list of risks to served answers instead of a loss
ranking.

| # | Risk | Reach (served) | Guard |
| --- | --- | --- | --- |
| 1 | A reader still reads a KG 2.x name | every surface over 3,376 files | every public KG reader goes through `kg_as_v3` (R3a); step 7 replays the served surfaces old-on-2.1 against new-on-3.0 |
| 2 | `respelled_from` dropped and the identity ledger loses its evidence | 722 people | kept; the schema allows it; the ledger rebuilt with 0 joins changed |
| 3 | `HOSTS`/`GUESTS_ON` as `CONTRIBUTES_TO` changes a host/guest answer | 5,599 edges | the role carried on the edge; the host and guest routes replayed |
| 4 | The writer keeps writing 2.1 | every new episode | the writer switch ships here; the step-7 pipeline run validates a new file as 3.0 |
| 5 | A later migration writes 2.x names into a 3.0 file | the episodes it touches | to measure: the KG-writing migrations (the identity rewrites) |

## 4. Prediction

Dated: 2026-11-18, on `gate-0049f_published_names_corrected` (snapshot 2026-11-16 with 0046-0049,
0049 at its final 640 entries), restated after step 5 with every row.

**Population.** Reached: 3,376 served KG files. Not reached: the unserved KG files (older runs; they
stay 2.1, byte-identical); every GI, segments, metadata and transcript file (0 changed outside the
`.kg.json` suffix).

**Root.** `schema_version` 3.0, `item_id` and `content_type: podcast_episode` on 3,376;
`episode_id` dropped on 3,376; `extraction.transcript_ref` to `document_ref` with `variant: adfree`
on 3,376 (every episode `en`; 0 non-English); the 121 `document_ref`s that do not resolve to a file
on the copy are copied unchanged.

**Nodes.** 3,376 `Episode` to `ContentItem` (`title`, `content_type`, `source_id` = the `Podcast`
node id, `published_at` from `publish_date` on 3,376, `feed_id` kept on 3,376, `podcast_id` dropped
on 3,376 where it equals `feed_id` on 3,376). 3,376 `Podcast` to `Source` (`source_kind: podcast`),
ids unchanged (T-8). `Person` 19,038 (roles: `mentioned` 12,544, `host` 3,486, `guest` 3,008;
`respelled_from` kept on 723), `Organization` 20,456 (`mentioned`), `Object` 8,284 (`mentioned`),
`Topic` 33,512: counts, ids, labels and descriptions unchanged (longest 586 characters).

**Edges.** 3,376 `HAS_EPISODE` to `HAS_ITEM`; 3,262 `HOSTS` and 2,337 `GUESTS_ON` (all to the
`Podcast` node) to `CONTRIBUTES_TO` with role `host`/`guest`; 81,088 `MENTIONS` unchanged; every
edge written with `from`/`to`.

**The registry.** `config/publishers.yaml` ships empty, so `PUBLISHED_BY` 0 on the copy; 0
`Organization` nodes re-id'd or merged (20,456 ids unchanged); ADR-180 rule 6 as amended (an
identical slug is one organisation) changes no id. The writer's rule, with a scenario test: when the
registry lists a podcast's `Source` id, its KG gets `PUBLISHED_BY` to that `org:` id.

**Strict validation.** 3,376 of 3,376 valid against `kg.v3.schema.json` (measured: the adapted
files fail only on the root `episode_id`, which the migration drops); `adapted_from` absent on 3,376.

**Must not move** (old code on 2.1 against new code on 3.0, every served episode, 3,376): the
relational graph's answers (public's in-memory graph), the CIL routes, `node-episodes` per node, the
digest's linkable topics, the enrichers that read KG, the search index rows that read KG; the
identity ledger rebuilt with 0 joins changed (the join count before and after stated in step 7).

**The writers after this change** (none may write a 2.x name into a 3.0 file): the KG writer
(`kg/pipeline.py`) writes 3.0 and validates strictly; Common's identity family (`respell.py`,
`promote_placeholders.py`) and public's KG-walking migrations (m0006, m0008) write 3.0 on a 3.0
file: a respell and a promote run on the 3.0 copy leave 3,376 files valid.

**Verify and undo.** `upgrade verify` over every applied migration reads the same on the 3.0 copy as
on the 2.1 copy. Undo: restored 3,376, refused 0, byte-identical; a second apply writes 0; an older
migration's undo is refused on a file this one rewrote (the framework's sha check).

## 5. Acceptance

Section 4 is restated with these rows before step 6.

**Fresh reviewer (advisor, 2026-11-18, minutes 9), rows added in full:**

1. The edge target rule: the epic says hosts and guests become `PARTICIPATES_IN`; ontology-v3 and
   `as_v3.py` say `CONTRIBUTES_TO`. Predict how many `HOSTS`/`GUESTS_ON` point at the `Podcast`
   node and how many at the `Episode` node; fix the epic or the record before code.
2. Strict-schema rows: `podcast_id` dropped on 3,376 (equal to `feed_id` on N); `publish_date` to
   `published_at` on 3,376; description length; `role` values by count; `label` kept; every edge
   written with `from`/`to`.
3. `variant` populations, non-English episodes, `document_ref` resolving to a file.
4. Registry rows: `PUBLISHED_BY` 0 with ADR-180 rule 6 as amended; 0 `Organization` nodes re-id'd
   or merged against `config/publishers.yaml`, 20,456 ids unchanged; the writer's rule for
   `PUBLISHED_BY` when the registry maps a podcast, with a scenario test.
5. Writers after the apply: Common's identity family (`respell.py`, `promote_placeholders.py`) and
   public's KG-walking migrations (m0006, m0008) write KG files; each writes 3.0 after the apply; a
   respell or promote run on the 3.0 copy leaves 3,376 files valid; `adapted_from` absent on 3,376.
6. `upgrade verify` over every applied migration on the 3.0 copy; undo of a pre-R3b KG migration
   refused on 3.0; the derived-surface census (D17); the writer-switch run's new episode valid.
7. Must-not-move populations: the identity ledger's join count; a count per served surface; the
   search-index rows and the bridge/context digests that read KG, named or stated out of scope.
8. Step 1 misses: ADR-180 as amended, `config/publishers.yaml`, `kg/pipeline.py`, m0006/m0008,
   `identity/bare_name_scope.py`, the GI half's two reviews, ingestion's 0049 record, DB-ERA-SWEEP.
9. Spot checks for step 8: one of the 722 `respelled_from` people; one `.adfree.txt` episode and
   one non-English episode (0 on this copy, below); one episode with `feed_id` != `podcast_id` (0,
   below); one `HOSTS` edge to the `Episode` node (0, below); a `the-daily` episode's KG `Source` id
   equal to its GI `Source` id; one `Organization` whose slug matches a registry id (scenario-test
   fixture); the writer-switch run's new file; one unserved 2.1 file unchanged; the longest
   description.
10. Not verified by the reviewer: whether `person_node` closes its properties like
    `organization_node` (then `respelled_from` must be explicit; 470268707 made it explicit).

**Measured for these rows** (`scripts/measure/r3b_kg_detail.py` in the Common checkout, on
`gate-0049d_published_names_corrected`):

- Row 1: all 3,262 `HOSTS` and all 2,337 `GUESTS_ON` point at the `Podcast` node, 0 at an
  `Episode`. So `CONTRIBUTES_TO` (Person to Source, with the role) is right, as ontology-v3 and the
  adapter say; the epic's "`PARTICIPATES_IN`" is corrected in Common's plan (database session).
- Row 2: `feed_id` equals `podcast_id` on 3,376 of 3,376; roles: Person `mentioned` 12,535, `host`
  3,506, `guest` 3,007; Organization `mentioned` 20,456; Object `mentioned` 8,284; the longest
  description is 586 characters.
- Row 3: `variant` `adfree` on 3,376; every episode's metadata `episode.language` is `en` (0
  non-English); `transcript_ref` resolves to a file under the run folder for 3,255, not for 121 (the
  data as it is; R3b copies the reference unchanged; E06's importer meets the same 121).

## 6. Build

This branch: 0051 (`upgrade/migrations/m0051_kg_3_0.py`, registered in `upgrade/registry.py`), and
the KG writer writes KG 3.0 (`kg/io.write_artifact` converts with `kg/v3_file.to_v3_file` and
validates strictly). Common's side (Common `feat/news-release`, pushed): its KG readers through
`kg_read_view`, its KG editors through the edit view (2.x names in, 3.0 out), its concept-Topics as
valid KG 3.0 Topics (`topic:concept-<slug>`), and four test fixtures that held invalid KG. Common's
full suite against this branch: 8,803 passed.

## 7. Measure

On a copy of `gate-0049f_published_names_corrected` with 0050 applied (the GI half,
`r3b_gi_4_0.md` section 7, holds the shared rows: undo, second apply, gold, served surfaces, derived
surfaces), 2026-11-18.

| Check | Result |
| --- | --- |
| Change gate 0051 (report `r3b_kg_3_0.gate.json`, plan sha256 `52c1d9ca…`) | class full; 3,376 `.kg.json` written; **gating moves: none** |
| 0051's counters against section 4 | files 3,376; `CONTRIBUTES_TO` 5,599 (the 3,262 `HOSTS` and 2,337 `GUESTS_ON`, all to the `Podcast` node); `PUBLISHED_BY` 0 (the registry lists no podcast); refused 0: **each equal to the prediction** |
| `upgrade verify`, every applied migration | 48 ok; failing 0041 and 0049 before and after alike; **verify regressions: none** |
| Undo (0051, then 0050) and second apply | restored 3,376, refused 0; the copy byte-identical to before both (156,354 files); a second apply writes 0 files |
| Strict validation after respell and promote | 3,376 of 3,376 valid KG 3.0 |
| Served surfaces and derived surfaces | as in the GI half: every difference explained; index rows (684,131) and bridges identical |

**Builder's verdict: GO.** The deciding number: verify regressions 0, with every counter equal to
its prediction.

Not covered at this step: step 8's fresh reviewer; the operator's review (step 9).
