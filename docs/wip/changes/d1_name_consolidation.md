---
id: d1_name_consolidation
class: full
status: open
issue: none (RFC-131; plan in podcast-scraper-eval-data docs/wip/NAME-CONSOLIDATION-ARC-V3-2026-10-30.md)
touches:
  - src/podcast_scraper/gi/pipeline.py
  - src/podcast_scraper/providers/ml/diarization/roster.py
  - src/podcast_scraper/speaker_detectors/*.py
  - src/podcast_scraper/upgrade/cli_handlers.py
---

# d1_name_consolidation — the name-consolidation arc's first deploy: one home for identity, raw labels never shown, guards and the one-word rule inside the join

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Covers the code of
RFC-131 steps S1b (one home, the identity ledger), S3 (raw labels never shown) and S4 (guards
inside the join, the one-word rule), on branch `feat/name-consolidation`. Migration N11 (raw labels
out of the files) has its own record. Person names are kept out of this public record; the pairs
and names are in the eval-data repo under `data/eval/name-consolidation/`, cited by path.

## 1. Intake

- RFC-131 (person identity, one home), ADR-166 (the ledger; no joins without it), ADR-167 (guards
  inside the join): the design this change implements.
- The arc plan V3 (eval-data `docs/wip/NAME-CONSOLIDATION-ARC-V3-2026-10-30.md`): steps, measures
  R1–R7, the operator's decisions.
- #852 (the cross-episode map), #904 (its predicate), #2056 (one-token drift), #1978 (no regex on
  placeholder ids, superseded for two shapes by S3): the code this replaces or extends.
- #2352 (open: near-identical-name merge, among three non-English defects): the class S4 refuses.
- #2403 (Phase 5: the roster's same-person rules): out of this change, next.
- #2197 (Person nodes that are not one person), #2219 (entity registry by evidence): related, not
  addressed here.
- #2333 (cross-episode voice identity): not used; no voice embeddings in this arc (plan V3).
- #2069 (reversible role migrations, a ledger): the ledger here is for identity joins only.
- Measured before this record: the S2 truth set (192 pairs, two blind readers, controls 10/10),
  G0 (the deployed map scored on it), S3's prod report runs, S4's corpus-wide export and replay.

## 2. Outcome and baseline

Own numbers (definitions in `quality/name_report.py` and eval-data `s4/s4_score.py`):

- **R1** raw labels on shown surfaces. Prod, image sha-48e3945, corpus token 1793330465.0899494:
  insights 66 in 21 episodes, transcript segments 222,461 in 732 episodes (report c57479f).
- **R3b** wrong merges on the truth set: S1 train 1 (P159), S3 train 1 (P013, residual). Held-out
  0. (G0, eval-data `g0/g0_score_a89e53f.json`.)
- **R3a** truth-set splits: S1 held-out 12, S1 train 10 (G0 ids, S4 export).
- **Corpus-wide person joins:** 760 (export `s4/s4_export_a89e53f.json.gz`, sha256
  31f0e745…a54a18, prod run 2026-10-30 21:32Z, same corpus token).
- **Scoreboard (S1, S1n, S1c, S2, S3, S10a/b, S12, A1): not yet measured for this change.** It is
  taken before and after on the step-7 snapshot clone. Until then this section has no scoreboard
  baseline.

## 3. Diagnosis by loss

- Raw labels shown: R1 above, one cause (no predicate on the display path), counted per surface by
  the report.
- Wrong merges: the corpus-wide export's 760 joins replayed and read join by join (eval-data
  `s4/RESULT-2026-10-30.md`). Causes by count of wrong joins: a chain through a bare surname or a
  title-stripped one-word name (14 of the 18 joins the no-bridge rule removes; 3 false splits, 1
  read as one person by the operator), a fuzzy one-word vs one-word join (16 of 23 by my reading;
  the operator read one of those as one person), two people in one episode with no voice (guard c:
  3 right, 5 false splits), one person's two spellings on two voices (guard a: 1 false split, by
  the operator's reading).

## 4. Prediction

MISSED: no prediction was written before the code. S1b, S3 and S4 were built and measured (S3's
prod report, S4's replay) before this record existed; the method's hook and pre-commit check did not
run on this worktree (the hook points at a main checkout without `scripts/hooks/`, and no git hook
is installed). A prediction written now would describe known numbers, not predict them.

Operator accepted: 2026-10-31 steps 4–5 missed for S1b, S3 and S4; steps 7–8 in full on a fresh
snapshot clone.

What is known before step 7 (measured, not predicted), and what must not move:

- R1 shown surfaces 0 (prod report c71876a).
- Person joins 760 → 720; 40 removed, 11 moved, 0 added (S4 replay, gate PASS).
- Truth set: S1 train wrong 1 → 0; S1 held-out joined 21 → 19, train 35 → 34; S3 residual 1.
- Must not move: everything outside person joins and raw-label display (orgs' joins, insights'
  visibility except the 66 raw-label insights becoming unattributed, quotes, transcripts' text).

## 5. Acceptance

MISSED: see section 4. Operator accepted: 2026-10-31.

## 6. Build

Commits on `feat/name-consolidation` (S1b … S4): 042c8d6af and the S1b series, 1802035eb (S3),
a89e53fe2 (report), 855a12da6 (guards), a41530cbb (one-word rule, no-bridge linkage), 774de1a28
(non-Latin names, report reads the ledger), c5bc10c77 (ledger undo), and the ledger step (below).
- **The ledger step on every corpus-writing path** (plan V3: "the last stage of every
  corpus-writing path"), added 2026-10-31 after the D1 inventory found it unwired: only the
  `identity-ledger` command wrote the ledger, so episodes ingested after the deploy would have had
  no cross-episode joins until a hand-run build. `identity.ledger.refresh_ledger` runs after a
  successful `enrich` run (the post-pipeline chain, #1653, queues one coalesced run after every
  ingest and reprocess batch, so one build per batch, not per feed), after an `upgrade run` that
  applied a migration, and after an `enrich-edges` pass that rewrote artifacts. A failed build
  leaves the previous ledger and fails the command with exit 4 (never silent). `enrich
  --no-identity-ledger` opts out. Tests: `test_ledger_refresh.py` 11 passed; 6 mutants killed (each
  hook off, the upgrade hook on a dry run, enrich-edges always rebuilding, a swallowed failure).
  It writes only `identity_ledger.json` and its change log and previous copy; no content file.

## 7. Measure

_TODO_: the write rehearsal on a fresh prod snapshot clone (scoreboard before/after, full
`upgrade verify`, gold gates S6/S7, the ledger write and restore, the api against the ledger).

## 8. Verdict

_TODO_

## 9. Operator review

_TODO_

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

- The method's gates did not run on this worktree, and I did not check the method when S3 and S4
  started. The change: until the hooks run on this worktree, `quality.change_gate.check_edit` is
  run by hand before each content-code edit and each commit (it refused 0038's paths until the
  record existed, and allowed them after).
- Missed once more, 2026-10-31: `upgrade/cli_handlers.py` (the ledger hook) was edited before this
  record listed it; the hand check run before the commit refused it, and it was added under
  `touches` before the commit.

## Not covered

_TODO_
