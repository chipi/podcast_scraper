# 0036: step-5 acceptance, addendum B (the REBUILD case)

<!-- markdownlint-disable MD029 -- the criterion numbers are identifiers the record cites -->

Written by a fresh reviewer (not the builder), 2026-10-26T10:54 UTC, after the record's block
"Dated: 2026-10-26T10:53" (15 episodes whose bridge is `{}`; design restated) and from
[the first acceptance](0036_bridge_identities_restored.acceptance.md) and
`builders/bridge_builder.py` (`build_bridge`, `_collect_identity`). Ids only; no names.

Scope: only the 15 episodes whose bridge has no `identities` list, on the base of the first
acceptance (`snapshot-prod-20261025` with 0034 and 0035 applied). The 19 append-only episodes stay
under the first acceptance unchanged.

## Which first-acceptance criteria apply to the rebuild, and how

| First | In the rebuild |
|---|---|
| 1 | Applies: the baseline is re-derived; the 15 empty bridges and the 74 person gaps in them (of 115 and 103 publishable) are counted by the committed `bridge_gaps.py`. |
| 2 | Replaced by R1 to R4: for these 15 the unit is the identity count of the whole file, not a diff by id. The 102 / 34 totals still hold in sum (74 rebuilt-person identities plus 28 appended). |
| 3 | Replaced by R2: "only added elements" is false for a rebuild; the file is replaced. The census bar (R9) stays: 34 files changed in all, 0 other. |
| 4, 5 | Apply to the rebuild: publishable gaps 0 after; the unpublishable person left out is absent from the rebuilt file and still listed as a gap (R5). |
| 6 | Applies unchanged (scoreboard equal to the integer). |
| 7 | Replaced by R2 (equality with `build_bridge`, not with `_collect_identity` alone). |
| 9 | Applies and is stricter: `fuzzy_reconcile=False`, so no `fuzzy_merges` key in any of the 15 (R6). |
| 10, 11 | Apply: the 74 rebuilt persons are inside the 102 read; the 25-sample must include at least 5 from the 15 rebuilt files. |
| 12, 13 | Extended by R7 (topic readers). |
| 14 to 16 | Extended by R8 (new mutations). |
| 17, 18 | Apply; `emitted_at` rule in R3 makes them satisfiable. |
| 19 | Applies; verify FAILs on an empty-bridge file (R10). |
| 20 to 24 | Apply unchanged. |

## Pass bars (integers)

R1. **Rebuilt count: exactly 15 files**, each of which was `{}` in the base (no `identities`,
`episode_id` or `schema_version`) and has an `identities` list after. FAIL on 14 or 16. Identities
in the 15 rebuilt files: **490 = 74 person + 56 org + 271 topic + 89 object**, counted by a script
of the reviewer's from the after-files by id prefix; FAIL on any component off by one. Persons
left out by the publishable filter: **1**.

R2. **Field-for-field equality, 15 of 15.** For each of the 15, with `gi` and `kg` read from the
episode's own files and `expected = build_bridge(episode_id, gi, kg, fuzzy_reconcile=False)` with
every `person:` identity that `is_publishable_speaker_name(name, language=<episode language>)`
(default `require_person_shape=True`) refuses removed, the after-file equals `expected` in every
field except `emitted_at`: `schema_version`, `episode_id`, the `identities` list (order, every key,
nested `sources`, `aliases`) compared with `json.dumps(x, sort_keys=False, ensure_ascii=False)`
per identity. 15 of 15; one difference fails. No `fuzzy_merges` key.

R3. **`emitted_at` must be deterministic.** `build_bridge` defaults it to the wall clock, so a
rebuild that omits the argument differs on every run. The migration must pass a value computed
from the episode's files or a constant of the migration, never the clock. Check: (a) in each of
the 15 after-files `emitted_at` matches `YYYY-MM-DDTHH:MM:SSZ`; (b) apply, undo, apply: the 15
files have the same sha256 on both applies (the first acceptance's criterion 18); (c) a second
apply writes 0 files because each file now has an `identities` list, and the corpus sha256 census
is identical to the first apply's. The comparison in R2 skips `emitted_at` because the expected
value is built with its own clock; the stability checks (b) and (c) are what prove it is not the
clock. FAIL if (b) differs in one of the 15.

R4. **`episode_id` and `schema_version` present in 15 of 15**, `episode_id` equal to the episode's
id as the pipeline's own bridge writer states it, `schema_version == "1.0"` as `build_bridge`
emits.

R5. **Unpublishable: 1 left out of the rebuilt files** and 0 of the 12 first-acceptance
unpublishable persons added anywhere. The after-state of `bridge_gaps.py` still lists the same 12
keys (the 1 left out is among them, because GI or KG still hold the node). Publishable gaps after:
0 in 0 episodes.

R6. **No fuzzy pass:** 0 files with `fuzzy_merges`; the build ran with no embedding model loaded
(the test passes a stub embedder that fails if called).

R7. **Reader effect for the 15 (topic readers).** Using the app's own `cil_queries` on the clone,
before and after, for the 15 episodes:
- `topic_person_ids` (the query "which people spoke on this topic") and the topic timeline: the
  number of the 15 episodes that return at least one topic row after and none before, stated as an
  integer predicted by the builder before the run from the count of `topic:` identities with
  `sources.gi` true (these readers read `sources.gi` only). Expected 15 of 15; an episode whose GI
  holds no topic is named by id and counted, and the bar is "15 minus that count", not fewer.
- Topic identities reachable through those readers for the 15: before 0, after = the number of
  `topic:` identities with `sources.gi` true (at most 271), equal to an independent walk of the GI.
- `person_topic_ids` for the restored GI persons equals the keys of their `person_profile` topics.
- Org and object identities (56, 89) are reachable only through the node-episodes listing and
  `_bridge_all_ids`: both counts are stated, 56 and 89 rows gained.

R8. **Mutations (every row killed, 0 survivors, baseline 0 failures on unmutated code):**

| New rule | Test | Mutation that must turn it red |
|---|---|---|
| empty-bridge branch | a bridge `{}` with a GI and a KG: file becomes the `build_bridge` result, with org, topic and object identities | the empty-bridge branch removed (falls to append-only: file stays without org/topic/object and `episode_id`) |
| empty-bridge detection | a bridge `{"identities": []}` and a bridge with no `identities` key are both treated by the stated rule; a non-empty bridge is never rebuilt wholesale | detection widened to any bridge; narrowed to the `{}` literal |
| filter in the rebuild | an unpublishable `person:` node in GI or KG of an empty-bridge episode: not in the rebuilt file; org, topic and object identities unaffected | filter not applied in the rebuild path; filter applied to all prefixes |
| no fuzzy | stub embedder with similarity 1.0 over two near-named persons: both stay, no `fuzzy_merges` | `fuzzy_reconcile=True` (the `build_bridge` default) |
| deterministic `emitted_at` | two applies a second apart yield the same sha256 | `emitted_at` omitted, so the clock is used |
| skip reasons | empty bridge and missing GI and KG; unreadable GI: skipped with a counted reason, not written | guard removed |

R9. **Census:** exactly 34 changed files, all `*.bridge.json`; the 15 rebuilt are in the 34; 0
other file touched; undo restores the 15 to the `{}` bytes of the base (byte-identical sha256).

R10. **Verify:** FAIL on the base (the 15 empty bridges are reported with a count of 15 episodes,
74 person gaps); PASS after; FAIL if one rebuilt file is replaced by `{}` again (1 episode). The
verify and the migration call the same publishable function object.

## Not covered

1. **Why the 15 became empty.** The record says the snapshot's 0025 backups show each of the 15
held 21 to 30 identities before 0025 (0017 and 0020 backups for two of them) and `{}` after. That
shows 0025 as the last writer, not the line of code that emptied them. No bar here proves the
cause; this addendum does not diagnose it.
2. **Whether the current shared rewrite can still write `{}`.** Reading
`m0025_one_person_one_entry.py` lines 1180 to 1227: for a bridge that loads as a dict it always
assigns `bridge["identities"] = out`, and keeps every non-person identity, so it cannot produce a
file with no `identities` key from a dict. A bridge that fails to load (`_load` returns None) is
not written. I found no existing test that pins this: a search of `test_m0025*.py` for a bridge
with org or topic identities returned nothing, and `test_m0036_*.py` has no empty-bridge case yet.
Required before the build is accepted: a test that runs the shared rewrite on a bridge holding
org, topic and object identities plus one person it removes, and asserts the file still has
`identities`, `episode_id` and `schema_version` and all non-person identities; with its mutation
(non-person identities dropped). Until it exists, "it cannot write `{}` again" is a reading, not a
proof. This also covers a code path I have not read: whatever older 0025 version wrote the 15.
3. The original contents of the 15 bridges (21 to 30 identities each) are not compared to the
rebuilt 490: the rebuild is the builder's output today, so a fuzzy merge or an alias the old file
carried is lost; R2 proves equality with the builder, not with the history. Count of old versus
new identities per episode is not a bar.
4. Fuzzy merges the pipeline would have made with an embedding model: not run.
5. Org, topic and object identities of the 15 are not read by a human; only counted (R1) and
checked against the builder (R2).
6. Episodes ingested since the snapshot; prod's count of empty bridges (the 15 is the clone's);
prod is a step-10 plan diff under the first acceptance's criterion 22.
7. The language field of the 15 episodes is not audited (first acceptance, Not covered 6).
8. Search-index rows, entity-resolver output and other surfaces built from the bridge are not
rebuilt or re-run for the 15 (first acceptance, criterion 23 lists them).
9. Superseded `run_*` bridges are not counted among the 15.
