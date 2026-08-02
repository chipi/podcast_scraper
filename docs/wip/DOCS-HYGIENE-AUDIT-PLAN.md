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
