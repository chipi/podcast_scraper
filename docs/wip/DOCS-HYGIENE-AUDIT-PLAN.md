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

## HELD — 4 docs NOT deleted (need a decision), and why

These are the ones a permanent artifact still depends on — i.e. real gaps, not hygiene:

1. **The 3 ontology specs** — `SPEC_KG_GI_ONTOLOGY_REVIEW_2026-06-20.md`, `…_V2_…`, `…_V2_ROUND3_…` —
   are referenced by **`docs/rfc/RFC-097-unified-kg-gi-ontology-v2.md`**, which literally labels
   ROUND3 *"round-3 spec, the live design."* **The live ontology design lives in a WIP doc that a
   permanent RFC points to.** This is the gap: promote the ROUND3 spec's live content **into**
   RFC-097 (or a permanent `docs/spec/`), then the 3 WIP archaeology docs are free to delete.
   (`project_ontology_v2_handoff` memory agrees ROUND3 was the live spec.)
2. **`POST_RFC097_DEV_PROD_REMOVAL.md`** — freed from all code refs, BUT still referenced by
   **`data/eval/runs/_PRE_FIX_NOTE.md`**, a **frozen eval-run artifact** (`feedback_never_mutate_
   historical_artifacts` — never edit `data/eval/runs/`). Options: (a) leave POST_RFC097 as a
   permanent-ish design record (move it to `docs/adr/` or `docs/guides/`), or (b) operator OK to
   touch the frozen note. Recommend (a) — promote it out of WIP, since a frozen artifact will
   forever cite it.

## Gaps this surfaced (next-priority signal)

The two held clusters are the actionable gaps: **the KG/GI ontology "live design" was never promoted
out of WIP into RFC-097**, and **the dev/prod-removal reconciliation** is a design record a frozen
eval note depends on but which lives in WIP. Both should be *promoted* (WIP→permanent), not deleted.

## NOT covered (equal weight)

- The **ADR/RFC/PRD/guides trees themselves** were not audited for their own staleness — only their
  WIP references were removed. A separate pass is needed to classify those ~376 docs.
- **WIP↔permanent references FROM wip docs** (a wip doc citing an ADR/RFC as its target) were not
  enumerated here; that reverse map is the next hygiene sub-task if we want to find which WIP notes
  are ready to promote.
- The remaining ~130 WIP docs were not re-classified beyond the original DONE/KEEP audit.
