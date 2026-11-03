# Ingestion arc, branch `feat/ingestion-arc` (draft, 2026-11-03)

One long branch from main `e1206af1`. It deploys only after the infra and project-structure work
lands, so it takes **only work that changes nothing about deployment**: no compose, Docker,
workflow, deploy-contract or package-layout change, and nothing that counts as finished only once
it runs on prod or the DGX. Code here ships with a later deploy; it adds no step to that deploy's
runbook unless an item says so.

Sources: `HANDOVER-NAMING-2026-11-03.md` (§2, §4), the deleted `INGESTION-ARC-2026-10-19.md`
(`git show 775b09c8^:docs/wip/INGESTION-ARC-2026-10-19.md`), eval-data
`docs/wip/NAME-CONSOLIDATION-ARC-V3-2026-10-30.md`, open GH issues, and the ingestion session's
pointers (2026-11-03). Every issue state below is from `gh issue view` on 2026-11-03.

**Status: draft.** The order and the cut are proposals. The operator has not ranked them.

## How the method applies, per tier

[CORPUS_CHANGE_METHOD.md](../guides/CORPUS_CHANGE_METHOD.md) v0.9 governs every item that changes
what the corpus serves. Per tier (from the ingestion session's D1/D2 practice, 2026-11-03):

| Tier | Record | Measured by | Outcome file / prod hook |
| --- | --- | --- | --- |
| A, read-time fix (A2, A6) | none | repro test first; before/after of the served output on a snapshot | no |
| A, migration code (A3, A4) | yes: `upgrade/` is `content_code`; the gate decides full or light | tests, plus the gate on a clone | only if a migration is built |
| B, forward rule | yes, steps 1–5 before code (hook) | gold S7/S8 (design on dev_v1, val_v2 once) and an old-vs-new replay over a snapshot's inputs | no |
| L, ledger rule | yes, as a full change, though the hook does not force it (`identity/` is not `content_code`) | prediction from the planned code (record s6h), ledger rebuilt on a clone, diffed per group, respell writes counted | no; the respell runs at the next refresh |
| M, instruments | the guide's step 0 ("if a gate is missing, building it is the change") | tests | no |
| C, tooling | per item | tests | no |

Measuring needs a prod snapshot clone, which is not on this machine (see below). The gold cases and
labels are in eval-data (`data/eval/references/silver/naming_speakers_v1/<set>/cases/` and
`labels_v1/`); use those labels, not an older local copy. D1/D2's absolute S7 counts were scored on
labels older than eval-data `1019b1c` (2026-10-24) and were not re-measured; any baseline here is
taken fresh on the current labels.

## Tier A: code, tests and docs; no corpus change, no deploy step

| # | Item | Source | Notes |
| --- | --- | --- | --- |
| A0 | The operator's working rules for a multi-step arc, written into `AGENTS.md` § "Running a multi-step arc" | operator, 2026-11-03 | docs only; first task |
| A1 | Set the 15 rule records to `landed` (d1, s5a–s5d, s5o, s6a–s6h); drop the handover's stale "#2405 waits" line | ingestion session | docs only |
| A2 | `person_profile` empty for `person:robert-caro`, `person:anita-anand` | handover §2.1 | repro first; read-time fix in `server/cil_queries` (ladder rung 1) |
| A3 | m0023 `_write_text_with_backup` creates backup dirs without `match_corpus_owner` | handover §2.2 | test + fix; changes every migration's code digest. 0031–0042 are landed, so no outcome goes stale; the record says so |
| A4 | `write_with_backup` keeps the first backup, so a twice-written file undoes to its pre-first-run state | handover §4 | same module as A3 |
| A5 | A test that checks each verdict row's `line_sha256` against the served segment | handover §4 | test only |
| A6 | CIL readers read each feed's latest run only, so a fix in an older run's bridge is invisible | handover §4 | read-time |
| A7 | #2253 `gi inspect` slices `.adfree.txt` spans from `.txt` | GH, open | tool fix |
| A8 | #2131 `parse_webvtt` / `parse_srt` never unescape entities | GH, open | forward only; stored text stays as is |
| A9 | Mixed-voice majority is measured in seconds: into `CORPUS_CHANGE_METHOD.md` | handover §4 | docs only |
| A10 | Review budgets (method v0.9 §4a) are not enforced | handover §4 | tooling |
| A11 | #2180 203 integration tests carry no `integration` marker | GH, open | test files only, no workflow edit |
| A12 | eval-data: measuring scripts skip episodes with an absolute transcript path; recount D1's gold moves with the corrected script | ingestion session | eval-data repo, its own branch |

## Tier B: naming rules for new episodes

Each one changes what the corpus serves, so each starts with a change record
(`docs/wip/changes/`, steps 1–5 before code) and is measured on a snapshot clone. Stored episodes
stay as they are unless a migration is built; **a migration adds an apply step to the deploy**, so
any migration here is built and gated on the branch, and its apply waits for the operator.

| # | Item | Source |
| --- | --- | --- |
| B5 | A name in the middle of an episode title is a stated spelling | #2363, open |
| B6 | Patreon supporter shout-outs published as guests | #2364, open |
| B7 | An eponymous host treated as the show | #2136, open |
| B8 | A house-ad read published as a guest | #2123, open |
| B9 | Rosters wrong beyond a missing name | #2367, open |
| B10 | Recurrence-scan blind spots | handover §4 (record 0034, round 6) |
| B11 | The hardfork-guest-in-record golden failure: cause not established | handover §4 |

## Tier L: ledger rules, which rewrite stored files on prod after a deploy, without a migration

0041 is applied, so on prod the ledger refresh after every `enrich` and `upgrade run` respells the
episode files to the current group spellings (`identity/refresh_guard.py`). A deployed change to the
ledger's join or display rules, or to a reviewed list, therefore rewrites stored files at the next
refresh. Each is a corpus change: change record, prediction, and a clone gate counting what the
respell writes (the near-spelling joins report, plus the ledger change-log's respell counts).
**The deploy that carries any of these changes stored data.** Whether that fits "changes nothing
about deployment" is the operator's call.

| # | Item | Source |
| --- | --- | --- |
| L1 | Roster same-person rules unified or justified (RFC-131 Phase 5), if they change what the ledger reads | #2403, open |
| L2 | Feed-host spelling rule: a feed's stated host matched by first name and host role (`dara-obrey`, `luisa-lemi`, `john-glenn-hill`) | handover §2.3 |
| L3 | `known_hosts` as mixed evidence, fixed at the pipeline (`_recurrent_hosts_from_disk`); a ledger change if it feeds the ledger | handover §2.4 |
| L4 | Strict-tier losses D1 → D2: `sarah-gonzalez`, `erika-barris`, `david-sanger` | handover §2.5 |
| L5 | Check #2056 (duplicate people from spelling variants) against D2; close it or name what is left | GH, open |

## Tier M: measuring instruments (from the 2026-10-19 plan's Stage 2; D1/D2 did not do them)

| # | Item | Notes |
| --- | --- | --- |
| M1 | Gold refresh: re-cut the stale gold cases on a new baseline, and draw a new unseen validation set | D1/D2 scored on `labels_v1` |
| M2 | Split S7: same person misspelled vs another person (`gold_insights` counts an ASR spelling of the right person as wrong) | eval-data |
| M3 | The loss-ranking script: `origin/wip/loss-ranking`, 85 commits ahead of main, never merged, not read | decide: rebase, salvage or drop |
| M4 | V3 S7, roster vs voices: `quality/name_report.py` measures R5 (`listed_not_heard_entries`); the operator's decision on how a listed-but-unheard name shows was never asked, so no read-time fix exists | needs an operator decision before code |

Measuring tiers L, B and M locally needs a snapshot copy; none is on this machine. Source: the
backup repo, `scripts/ops/corpus_snapshot/download_and_verify_snapshot.sh --tag
snapshot-prod-20261030` (about 10 GB; the one D1/D2 were measured on), then `upgrade run --to
2.7.36 --no-snapshot` to bring it to D2's state, or a newer daily snapshot. A working copy is about
8.3 GB, one base plus a gate copy per run. D2's gate chain: eval-data
`data/eval/name-consolidation/d2out/scripts/chain5.sh`.

## Tier C: tooling for later migrations (built here, used on prod later)

| # | Item | Source |
| --- | --- | --- |
| C1 | The code digest covers a migration and most of the package; narrow it | handover §4 |
| C2 | Re-run an applied migration for named episodes (today: calling the class in the api container) | handover §4 |
| C3 | A naming migration that re-renders text keeps the stored ad cut; recompute it or end with 0030's resync | handover §4 |
| C4 | A gate row "wrong-person voices before / after" | handover §4 |

## Not in this arc, and why

| Item | Why not |
| --- | --- |
| Diarizer merges promo/ad/clip into a host voice | upstream diarization, DGX |
| #2187, #2251, #2353, #2370, #2284, #2292 | ASR and translation runs on the DGX |
| 0035's reviewer points, 0032's 91 voices, the near-spelling 0-pairs check, `robert-carrow` trace | need prod's served data or a deploy |
| Bare names pooled again (#1801), cross-episode voice identity (#2333), shadow-eval round 2 | wait on the voice-fingerprint shadow validation |
| 12 KG-only persons left out of the bridge | needs the operator's approval first |
| CI interpreter pin (ubuntu-24.04 / Python 3.11.8) | workflow change |
| V3 S9's remaining parts (a daily name report, the R7 alert, promoting a shadow namer) | scheduling and alerting on prod; the ledger-after-ingest step, the ledger state gauge and the regression tests are done |
| #2407, #2408 (index and MCP memory) | no owner recorded; the operator decides whether they belong here |
| Triage-fleet escalations, mobile, player, translation epic issues | not ingestion |

## Not covered by this draft

- About 200 open issues were read by title only. Only the ones above were checked one by one.
- `origin/wip/loss-ranking` was not read by anyone since 2026-10-25.
- Eval-data's 2026-10-22 naming notes and `MIGRATION_ARC_PLAN` / `MIGRATION_INVENTORY` were not
  re-read; parts may still be open.
- No item has a size estimate.
- The snapshot download (about 10 GB, plus working copies) was not tried. Free disk: 1.2 TB
  (`df -h /work`, 2026-11-03).
- No naming split is fully unread (eval-data `naming_speakers_v1/README.md`): val_v1, val_v2 and
  val_v3 were read; val_v6 was scored at the 2026-10-25 release gate and 7 of its voices were
  panel-read; val_v4 and val_v5's read state is not checked. A tier-B or tier-L rule needs a fresh
  split (seeded, excluding every existing cases folder, labelled blind and audited), or val_v6
  once, totals only, with that caveat in the record. No unseen feed exists.
