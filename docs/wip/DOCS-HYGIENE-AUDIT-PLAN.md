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

**NOT done in pass 2 (equal weight):**

- **~23 backend-domain docs** (pipeline, corpus, enrichment, DGX, prod infra) were classified
  but NOT acted on — they belong to a parallel thread working from another worktree, and
  several of their open items live in `agentic-ai-homelab`, which this pass could not verify.
- **4 docs are deletable only after an edit elsewhere**: `nightly-test-time-analysis.md`
  (cited from `tests/e2e/conftest.py` + `tests/integration/conftest.py`), `MCP-E2E-GUIDE.md`
  (cited from `docs/guides/MCP_SERVER_GUIDE.md`), `SYNTHETIC-CORPUS-FULL-FIDELITY-PLAN.md`
  (cited from the fixture README), `PLAYER-CURATION-DELIVERY-MOAT-ARCH.md` (`PRD-046:164`).
- **2 are promote-candidates, not deletes**: `manual-test-plan-gi-kg.md` and
  `2026-08-13-e2e-on-intel-mac.md` are the only written copies of live procedures.
- The other two NOT-covered items above (the ~376 permanent docs, the reverse reference map)
  remain untouched.
