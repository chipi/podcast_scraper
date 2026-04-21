# Release Playbook

**Status:** v3 (standing plan — human and AI executable)  
**Audience:** Maintainers cutting a release; AI agents decomposing work into tasks.

This document defines **what "done" means** before a public tag and the **order** to get there.
**Git mechanics and per-step detail** live in the
[Development Guide — Release checklist](DEVELOPMENT_GUIDE.md#release-checklist). Use **both**:
this playbook for **policy, phase order, and commands**; the Development Guide for edge cases and
explanations.

**Ordering note:** This playbook intentionally places the **version bump late** (Phase 6) — after
validation and docs are green — so the tagged commit already contains everything. The Development
Guide lists the bump earlier for historical reasons; **follow this order** when using both docs.

**Related**

- [ADR-031: Mandatory Pre-Release Validation](../adr/ADR-031-mandatory-pre-release-validation.md)
  — target: a single `make pre-release` gate (script not implemented yet; see Phase 3).
- [Releases index](../releases/index.md) — published `RELEASE_vX.Y.Z.md` files and history.

---

## Git tags and version fields

Annotated Git tags use the form **`vX.Y.Z`** (e.g. `v2.5.0`). Package metadata uses **`X.Y.Z`
without the `v`**:

- `pyproject.toml` → `[project].version` (e.g. `2.6.0`)
- `src/podcast_scraper/__init__.py` → `__version__` (same `X.Y.Z`)

**Rule:** The numeric **X.Y.Z** must match everywhere; the **git tag** adds **`v`** in front.

---

## Release types (semver)

| Type | Typical bump | Eval + profiles | Phases to run |
| ---- | ------------ | --------------- | ------------- |
| **Major** | X.0.0 | **Mandatory** | All (1 -- 7) |
| **Minor** | 0.X.0 | **Recommended**; document omission | All; Phase 2 can be "carry forward" note |
| **Patch** | 0.0.X | Optional | 1, 3, 4, 5 (light), 6, 7; skip 2 unless provider defaults changed |

---

## Dual validation tracks (quality vs cost)

Releases that change providers, default models, or quality-sensitive pipeline behavior are
defensible on **two** axes:

1. **Evaluation (quality)** — [Experiment Guide](EXPERIMENT_GUIDE.md), baselines under `data/eval/`,
   [Evaluation reports](eval-reports/index.md).
2. **Performance (resource / wall time)** — [Performance Profile Guide](PERFORMANCE_PROFILE_GUIDE.md),
   `make profile-freeze` / `make profile-diff`, artifacts under `data/profiles/`, companion
   `*.stage_truth.json`, [Performance reports](performance-reports/index.md).

---

## Phases

Complete phases in order so the **tagged commit** already includes green docs and finalized notes.

---

### Phase 1 — Scope and version `[manual]`

**Skip for:** Never skip; always start here.

**Inputs:** Release intent (feature set, provider changes, bug fixes).

**Steps:**

- Decide **semver** bump (see table above).
- For **major** releases, list the **prod-intended provider/model matrix**: which
  `config/profiles/freeze/*.yaml` or `config/acceptance/*.yaml` configs, which `DATASET_ID`
  labels, which reference host.
- Run **`make release-docs-prep`** — this stubs `docs/releases/RELEASE_vX.Y.Z.md` (from
  `pyproject.toml` version) and regenerates architecture diagram SVGs.

**Outputs:** Scope description (in PR description or top of draft `RELEASE_vX.Y.Z.md`); provider
matrix table for major.

**Gate:** Scope agreed (PR / issue / team decision).

---

### Phase 2 — Quality and performance validation `[manual]`

**Skip for:** Patch unless provider defaults changed.

**Inputs:** Provider matrix from Phase 1; clean working tree on release branch.

**Steps — eval (quality):**

```bash
# Per Experiment Guide workflow: dataset, baseline, experiment run
make experiment-run CONFIG=data/eval/configs/<release_eval>.yaml
# Publish or link eval report under docs/guides/eval-reports/
```

**Steps — profiles (resource cost):**

```bash
make profile-freeze VERSION=vX.Y.Z \
  PIPELINE_CONFIG=config/profiles/freeze/<preset>.yaml \
  DATASET_ID=<label>

# Optional: compare to previous release
make profile-diff FROM=vPREVIOUS TO=vX.Y.Z
```

- Commit `data/profiles/<version>.yaml` and `<version>.stage_truth.json`.
- Update [performance reports](performance-reports/index.md) if publishing a written snapshot.

**If skipping for minor:** Add a subsection to release notes: "Eval/profiles unchanged from
vX.Y.(Z-1)."

**Outputs:** Eval run artifacts under `data/eval/`; profile YAML + `stage_truth.json` under
`data/profiles/`; optional report pages.

**Gate:** Eval meets baseline / quality policy; profile shows no unexpected regression (advisory or
threshold via `regression_rules.yaml` when mature).

---

### Phase 3 — Engineering gates `[partial — CI covers PR; full gate is manual]`

**Skip for:** Never skip.

**Inputs:** All code and doc changes on release branch.

**Steps — ADR-031 one-shot (recommended before tag):**

```bash
make pre-release   # runs scripts/pre_release_check.py then make ci
```

`pre_release_check` verifies `pyproject.toml` / `__init__.py` version match, and that
`docs/releases/RELEASE_vX.Y.Z.md` plus `docs/releases/index.md` reference the current version.
Then **`make ci`** runs the full gate (format-check, lint, type, security, complexity, docstrings,
spelling, tests, coverage-enforce, docs, build — includes `make build`).

**Optional extra (not in `make ci`):**

```bash
make quality       # radon, vulture, interrogate, codespell — resolve or document
```

**Outputs:** Green exit codes; `dist/` or `.build/dist/` artifacts.

**Gate:** `make pre-release` exits 0; no unresolved lint/type/coverage failures.

---

### Phase 4 — Documentation build, links, and stale content `[manual]`

**Skip for:** Never skip (even patch — docs may have broken links from other merges).

**Inputs:** `git diff main..HEAD --name-only` (or `vPREVIOUS..HEAD`) to identify what changed.

**Steps — automated checks:**

```bash
make fix-md          # auto-fix markdown
make lint-markdown   # verify clean
make docs            # MkDocs strict build — fix any warnings/errors
```

**Steps — diagram refresh (if not done in Phase 1):**

```bash
make release-docs-prep   # regenerate docs/architecture/diagrams/*.svg
git add docs/architecture/diagrams/*.svg
```

**Steps — stale prose sweep (diff-driven):**

Use the diff to map **changed source** to **docs that may need updating**:

| Changed area | Check these docs |
| ------------ | ---------------- |
| `src/podcast_scraper/config.py`, CLI flags | [CONFIGURATION](../api/CONFIGURATION.md), [CLI](../api/CLI.md) |
| `src/podcast_scraper/service.py`, public API | [SERVICE](../api/SERVICE.md), [CORE](../api/CORE.md) |
| Server routes (`server/routes/`) | [Server Guide](SERVER_GUIDE.md), relevant RFCs |
| Breaking behavior | [MIGRATION_GUIDE](../api/MIGRATION_GUIDE.md), [VERSIONING](../api/VERSIONING.md) |
| Pipeline / workflow modules | [ARCHITECTURE](../architecture/ARCHITECTURE.md), [Platform blueprint](../architecture/PLATFORM_ARCHITECTURE_BLUEPRINT.md) |
| Viewer (`web/gi-kg-viewer/`) | `web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md`, UXS docs in `docs/uxs/` |
| `mkdocs.yml` or new/moved pages | Verify nav entries; `make docs` catches missing pages |

**Agent hint:** Run `git diff --name-only vPREVIOUS..HEAD -- src/` to get the file list, then match
against the left column above to determine which docs to review.

**Steps — broken links:**

Internal links are caught by `make docs` (strict). For external link checking, see **Roadmap**
below.

**Outputs:** Clean `make docs` exit; updated prose where the sweep found drift.

**Gate:** `make docs` exits 0; `make lint-markdown` exits 0.

---

### Phase 5 — Comprehensive release notes `[manual]`

**Skip for:** Never skip, but **depth** varies: major gets full spotlights; patch gets a short
paragraph.

**Inputs:** Draft `docs/releases/RELEASE_vX.Y.Z.md` created by `make release-docs-prep` in
Phase 1; recent releases for reference (template:
[RELEASE_v2.6.0](../releases/RELEASE_v2.6.0.md)).

**Steps:**

- Edit `docs/releases/RELEASE_vX.Y.Z.md`:
  - Title, date, release type.
  - Summary (1--3 paragraphs).
  - Spotlight sections (major/minor): tables with links to PRD/RFC/guides.
  - Upgrade / migration notes (link to `MIGRATION_GUIDE` anchors if breaking).
  - Eval/profile references or "carried forward" note from Phase 2.
  - GitHub compare link: `https://github.com/chipi/podcast_scraper/compare/vPREVIOUS...vX.Y.Z`.
- Update `docs/releases/index.md`: Latest Release blurb and table row.

**Agent hint:** Start from the stub file; copy the **section headings** from the most recent
`RELEASE_v*.md` and fill. Do not invent a new layout.

**Outputs:** Completed `RELEASE_vX.Y.Z.md`; updated `index.md`.

**Gate:** `make docs` still passes after edits; `make lint-markdown` clean.

---

### Phase 6 — Version bump and commit `[manual]`

**Skip for:** Never skip.

**Inputs:** Agreed version from Phase 1; all prior phases green.

**Steps:**

```bash
# Bump version (no v prefix) — updates pyproject.toml + __init__.py in one step
make bump VERSION=X.Y.Z
# Optional: ALLOW_DIRTY=1 if you must bump with uncommitted files; FORCE_TAG=1 if tag vX.Y.Z exists

# Stage and commit everything
git add pyproject.toml src/podcast_scraper/__init__.py \
  docs/releases/ docs/architecture/diagrams/*.svg \
  data/profiles/*.yaml data/profiles/*.stage_truth.json
git commit -m "chore: release vX.Y.Z"
```

**Outputs:** Single release commit on the branch.

**Gate:** `git diff --cached` shows only release-related changes; version strings match.

---

### Phase 7 — Tag and GitHub Release `[manual]`

**Skip for:** Never skip.

**Inputs:** Release commit from Phase 6; push access.

**Steps:**

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin <branch>
git push origin vX.Y.Z
```

Then publish GitHub Release:

```bash
# Via gh CLI (preferred for agent automation):
gh release create vX.Y.Z --title "vX.Y.Z" --notes-file docs/releases/RELEASE_vX.Y.Z.md
```

Optional post-release dev version bump (Development Guide section 8).

**Outputs:** Git tag; GitHub Release with notes.

**Gate:** Tag visible on remote; GitHub Release page exists.

---

## Task template (for decomposition)

Use this shape when splitting phases into issues, PR subtasks, or agent steps:

| Field | Content |
| ----- | ------- |
| **ID** | Short label (e.g. `REL-docs-01`) |
| **Phase** | Which phase above |
| **Objective** | One sentence |
| **Inputs** | Branch, version, host for profiles, configs |
| **Steps** | Bullets; prefer `make <target>` from the Makefile |
| **Outputs** | Paths or URLs |
| **Gate** | Pass condition |
| **Automation** | `manual` / `partial` / `ci` / `planned` |

---

## Automation status summary

| Phase | Status today | Target |
| ----- | ------------ | ------ |
| 1. Scope | `manual` | `manual` (human decision) |
| 2. Eval + profiles | `manual` | `partial` — `make profile-freeze` is scripted; eval run is scripted; publishing report is manual |
| 3. Engineering gates | `partial` — `make pre-release` = check script + `make ci`; optional `make quality` | `ci` — extend script (changelog file, stricter rules) per ADR-031 evolution |
| 4. Docs + links | `partial` — `make docs` + `make lint-markdown` catch structural issues; stale prose is manual | `partial` + external link checker (planned) |
| 5. Release notes | `manual` | `partial` — `make release-docs-prep` stubs the file; content is manual |
| 6. Version bump | `partial` — **`make bump VERSION=X.Y.Z`** | `planned` — integrate into `pre_release_check` dry-run |
| 7. Tag + GH release | `manual` | `ci` — tag-triggered workflow or `gh release create` |

---

## Roadmap (things to build)

Items that would close gaps between the current manual workflow and full automation.

| ID | Item | Status | Notes |
| -- | ---- | ------ | ----- |
| **R-01** | **`make pre-release`** — `scripts/pre_release_check.py` + `make ci` | **Done** | Version + release notes file + index; `make ci` covers tests/docs/build |
| **R-02** | **`make bump VERSION=X.Y.Z`** — `scripts/tools/bump_version.py` | **Done** | Dirty tree / existing tag guards; follow-up: optional `make quality` inside pre-release |
| **R-03** | **External link checker** — periodic or pre-release job (e.g. `lychee` or `linkchecker`) against the built site to catch 404s on external URLs that `make docs --strict` does not verify. | Small | Phase 4 link coverage |
| **R-04** | **Tag-triggered GitHub Release workflow** — `.github/workflows/release.yml`: on `push` of `v*` tag, build, optionally publish to PyPI or registry, create GitHub Release from `docs/releases/RELEASE_vX.Y.Z.md`. | Medium | Phase 7 CI automation |
| **R-05** | **Diff-to-docs mapper** — lightweight script or agent skill: given `git diff --name-only`, output the list of docs from the Phase 4 sweep table that should be reviewed. Input: ref range. Output: file list + "review needed" flag. | Small | Phase 4 agent execution |
| **R-06** | **Eval + profile matrix runner** — wrapper that runs `make experiment-run` and `make profile-freeze` for each row in a provider matrix config (YAML list of presets). Produces a summary table. | Medium | Phase 2 scaling for major releases |

---

## Changelog

| Version | Change |
| ------- | ------ |
| 3 | R-01/R-02 implemented: `make pre-release`, `make bump`; Phase 3/6 updated; roadmap table status column. |
| 2 | Commands per phase; inputs/outputs/gate on every phase; skip conditions by release type; automation status table; diff-driven doc sweep; ordering reconciled with Development Guide; roadmap for planned tooling. |
| 1 | Initial lean playbook: phases, major/minor eval-profile policy, ADR-031 interim gates, tag `v` + semver alignment. |
