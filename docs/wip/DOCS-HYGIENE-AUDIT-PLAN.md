# Docs-hygiene audit — WIP tree (executed 2026-08-02)

Executed the WIP-tree hygiene pass. Principle enforced: **permanent docs (ADR/RFC/PRD/release/
guide/api) and code/tests/README must never reference `docs/wip/` docs** — WIP is ephemeral and
gets deleted, so a permanent artifact that points at it rots. WIP↔WIP references are fine (a WIP
set travels together). Scope: the WIP tree (~160 docs); the ADR/RFC/PRD/guides trees themselves
were only touched to remove their WIP references.

## Outcome

- **29 WIP docs deleted** (work shipped/closed, verified vs closed issues/commits):
  - 18 with zero references anywhere (first pass).
  - 11 freed by removing their references from permanent docs + code (see below), then deleted.
- **Permanent-doc + code references to WIP removed** (this is what "released" the 11):
  - `docs/api/PLATFORM_API.md`, `docs/prd/PRD-037-discovery.md`, `docs/prd/PRD-035-learning-platform.md`,
    `docs/releases/RELEASE_v2.6.1.md`, `docs/adr/ADR-135-*.md`,
    `docs/guides/eval-reports/EVAL_AUTORESEARCH_JUDGE_TRUST_MATRIX_2026_07.md`,
    `docs/rfc/RFC-098/100/101-*.md`.
  - Code/tests/README: `src/podcast_scraper/net/__init__.py`, `config.py`, `utils/runtime_env.py`,
    `tests/conftest.py`, `tests/integration/eval/test_v3_fixtures.py`, `web/learning-player/README.md`.
  - Dead WIP↔WIP links the deletions left were cleaned in `WIP_README.md`, `RFC-088-…AUDIT`,
    `INFRA-HARDENING-PLAN.md`, `SPEAKER-PIPELINE-SUBSYSTEM-AUDIT.md`.
- Strict docs build green throughout (`make docs` → `MAKE_DOCS_EXIT=0`); zero dangling refs remain.

## HELD gaps — RESOLVED (promoted, then freed)

Both held clusters turned out to be *already-promoted* content whose WIP archaeology just hadn't been
cleaned. Resolved this pass:

1. **The 3 ontology specs** (REVIEW / V2 / ROUND3) — RFC-097 labelled ROUND3 "the live design," but
   the live design had **already shipped** into the permanent `docs/architecture/corpus/ontology.md`
   (RFC-097 Completed, #1036 CLOSED, chunks 1–9 shipped). RFC-097's "the live design" label was
   stale. → RFC-097's WIP-spec references repointed to `docs/architecture/corpus/ontology.md`; the 3
   archaeology specs **deleted** (history in git). `SPEC_KG_GI_ONTOLOGY_V3_WISHLIST` (deferred future
   ideas) stays as a pure WIP doc, no longer referenced by RFC-097.
2. **`POST_RFC097_DEV_PROD_REMOVAL.md`** — a decision record a **frozen** eval artifact
   (`data/eval/runs/_PRE_FIX_NOTE.md`) cites. → **Promoted** to
   `docs/guides/DEV_PROD_ENV_DETECT_REMOVAL.md` (added to the mkdocs nav). The frozen note keeps its
   old `docs/wip/…` path (never edit `data/eval/runs/` — `feedback_never_mutate_historical_artifacts`);
   that stale link lives outside the docs build and is the one accepted exception.

Net: **32 WIP docs deleted** (18 + 11 + 3 ontology), **1 promoted** to guides; zero permanent artifact
now references `docs/wip/`.

## NOT covered (equal weight)

- The **ADR/RFC/PRD/guides trees themselves** were not audited for their own staleness — only their
  WIP references were removed. A separate pass is needed to classify those ~376 docs.
- **WIP↔permanent references FROM wip docs** (a wip doc citing an ADR/RFC as its target) were not
  enumerated here; that reverse map is the next hygiene sub-task if we want to find which WIP notes
  are ready to promote.
- The remaining ~130 WIP docs were not re-classified beyond the original DONE/KEEP audit.
  → **Pass 2 started this, 2026-09-27. See below.**

## Pass 2 — 2026-09-27 (the ~130 re-classification)

116 docs re-classified against git history, source, and issue state rather than their own
status headers. **8 deleted** (app/UI + CI-test domain, every open item verified landed,
zero referrers), **1 promoted** — `OPERATOR-SMOKE-TEST-PLAN.md` →
`docs/guides/OPERATOR_SMOKE_TEST.md`, because `deploy-operator.yml` cited it and a workflow
must not depend on a doc that gets deleted when its arc ends.

**Three lessons this pass, which are the reusable part:**

1. **The referrer sweep must not filter by file extension.** `REDESIGN-PHASE3-HANDS.md` was
   deleted and restored: `web/learning-player/src/theme/directions.css` cites it as the
   provenance of shipped theme numbers, and the first sweep's `--include` list had no `.css`.
   A doc that live code cites for *why a value is what it is* is not stale, however finished
   its arc looks.
2. **Three permanent→WIP references had crept back**, so the 2026-08-02 claim of zero is no
   longer true. `PRD-046:164`, and the two the promotion above fixed. The invariant needs a
   guard, not another manual pass.
3. **Index rows go stale in the direction that flatters the doc.** `WIP_README` described
   `PLAN-storyline-theme-rename` as "Proposed, not started" when all six of its stages had
   shipped and #1603 was closed. Trusting the index would have kept a finished plan forever.

**NOT done in pass 2 (equal weight):** ~23 backend-domain docs classified but not acted on; four
docs deletable only after an edit elsewhere; two promote-candidates. **Pass 3 resolved all six of
the latter** — see below. The backend set remains untouched by design.

## Pass 3 — 2026-09-27 (the sharper test, applied)

Pass 2's criterion was "delete only if every open item landed", which is nearly unfalsifiable —
almost every doc lists one open item, so almost everything survived (8 of 116). The test used
here, after an advisor review:

> **Name the concrete next action whose executor will open THIS file, and the section they will
> read.** Only that yields KEEP. **Tracking content can never satisfy it** — status belongs in
> the issue tracker, and a stale status table is worse than none because agents act on it.

Two supporting rules: a doc cited by live code is **frozen** until the citer is repointed, and a
doc holding a decision or validated negative is **harvested then deleted**, never simply kept.

**Result: 119 → 91 files.** Eight promoted into permanent homes, eight retired into the issue
tracker, twelve deleted outright.

**Promoted — these had stopped being WIP and nobody had moved them:**

| Was | Now | Why it could not just be deleted |
| --- | --- | --- |
| `OPERATOR-SMOKE-TEST-PLAN` | `docs/guides/OPERATOR_SMOKE_TEST.md` | `deploy-operator.yml:392` cites it |
| `2026-08-13-e2e-on-intel-mac` | `docs/guides/E2E_ON_INTEL_MAC.md` | `run-local-stack.sh:19` **and `.gitignore:347`** |
| `2026-09-11-share-card-design` | `docs/uxs/UXS-017-share-cards.md` | `UXS-014:141` called it the SSOT; `og/card.py:3` cites it |
| `CORPUS-V4-FIXTURE-LADDER` | `docs/architecture/TEST_CORPUS_FIXTURE_LADDER.md` | **19 files** cite it — 14 e2e specs, a schema, a fixture, a unit test, both e2e READMEs |
| `MCP-E2E-GUIDE` | folded into `docs/guides/MCP_SERVER_GUIDE.md` | the guide pointed at it for a procedure it did not contain |
| `nightly-test-time-analysis` | inlined into both conftests | they cited it for *why* the sleep fixtures exist |
| `SYNTHETIC-CORPUS-FULL-FIDELITY-PLAN` | inlined into the fixture README | the README said "Full recipe:" and stopped |
| `PLAYER-CURATION-DELIVERY-MOAT-ARCH` | "Why this shape" in `PRD-046` | `PRD-046:164` cited it as the design doc |
| `REDESIGN-PHASE3-HANDS` | inlined into `directions.css` | the CSS cited it as provenance for shipped theme numbers |

**Retired into the issue tracker** — nine comments on #1923, #1938, #1849, #1483, #1570, #1169,
#2156 and two on #1596, carrying the content that would otherwise be re-derived.

**Two findings worth more than the file count:**

1. **Half the directory was load-bearing.** Of the first 14 docs tested, 7 were frozen by a
   citer. `docs/wip/` is contractually ephemeral and was in practice infrastructure — which is
   why passes 1 and 2 both stalled at a fraction of their target. The lever was promotion, not
   deletion.
2. **A code audit has a shelf life of weeks.** `DEAD-CODE-AUDIT` was 22 days old and **5 of its
   10 ranked items were already wrong** — three "unreachable" routes had become redirects, so
   deleting them would have broken bookmarked URLs; the nav bug it called live was fixed by
   #2013; a named component no longer existed. Posting that list verbatim, as the analysis
   proposed, would have published a worklist that damages the app. Past its shelf life an audit
   is not a backlog, it is misinformation.

**NOT done in pass 3 (equal weight):**

- **~82 backend-domain docs** (pipeline, corpus, enrichment, DGX, prod infra, observability,
  security-ops — 16,580 lines) classified but deliberately untouched: they are a parallel
  thread's working set and several of their open items are only verifiable in
  `agentic-ai-homelab`. **27 of them cite only closed issues**, so that is where the next large
  reduction sits — but it is that thread's call, not this one's.
- **6 app docs from the last two weeks** are inside the operator's protected window.
- `manual-test-plan-gi-kg.md` is still a promote-candidate — the only written copy of the manual
  GI/KG validation procedure.
- The two original NOT-covered items (the ~376 permanent docs, the reverse reference map) remain
  untouched, and the permanent→WIP invariant **still has no mechanical guard** — it has now
  decayed twice and been repaired by hand three times.
