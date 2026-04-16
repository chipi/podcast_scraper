# WIP — Release process automation follow-up (R-03 -- R-06)

**Status:** Backlog / design notes  
**Parent:** [Release Playbook](../guides/RELEASE_PLAYBOOK.md) **Roadmap**. **R-01** (`make pre-release`,
`scripts/pre_release_check.py`) and **R-02** (`make bump`, `scripts/tools/bump_version.py`) are
implemented; this file tracks **R-03 -- R-06** only.

This note tracks the **remaining** automation ideas from the playbook roadmap so they are not lost
and can be turned into issues or PRs later.

---

## R-03 — External link checker

**Goal:** Catch broken **external** URLs that `mkdocs build --strict` does not validate (internal
links are covered by MkDocs).

**Sketch:**

- Run against the **built** site (`.build/site/` after `make docs`) or against markdown sources.
- Tool candidates: `lychee`, `markdown-link-check`, or `linkchecker` on HTML output.
- **Gate:** Optional pre-release enhancement or a **weekly** workflow to avoid rate limits.
- **Deliverable:** Makefile target `make docs-linkcheck` (or `make pre-release` hook behind a flag).

---

## R-04 — Tag-triggered GitHub Release workflow

**Goal:** On `push` of tag `v*`, CI builds artifacts and creates a **GitHub Release** using
`docs/releases/RELEASE_vX.Y.Z.md` as the body.

**Sketch:**

- `.github/workflows/release.yml` — `on.push.tags: [ 'v*' ]`.
- Steps: checkout, setup Python, `pip install build`, `make build` (or `python -m build`), attach
  wheel/sdist to release (optional PyPI publish behind secret).
- `gh release create` with `--notes-file docs/releases/RELEASE_vX.Y.Z.md` (parse tag to version).
- **Constraints:** Must not publish if `RELEASE_vX.Y.Z.md` is missing (fail the job).

---

## R-05 — Diff-to-docs mapper

**Goal:** Given `git diff --name-only <base>..HEAD`, output which **docs** from the Release Playbook
Phase 4 sweep table should be reviewed.

**Sketch:**

- Script: `scripts/tools/release_doc_sweep_hints.py --base v2.6.0` or `--base main`.
- Map prefixes (`src/podcast_scraper/config.py` → CONFIGURATION.md, CLI.md, etc.) using the same
  table as the playbook (single source of truth: either YAML data file imported by doc generator, or
  script + playbook link to script output).
- **Agent use:** stdout list of paths; exit 0 always unless `--strict` and unmapped paths touched.

---

## R-06 — Eval + profile matrix runner

**Goal:** For **major** releases, run `make experiment-run` and `make profile-freeze` for each row in
a small **matrix config** (YAML list of `CONFIG`, `DATASET_ID`, `VERSION` suffix).

**Sketch:**

- `config/release/provider_matrix.example.yaml` — list of presets.
- `scripts/tools/run_release_benchmark_matrix.sh` or Python driver that loops and writes a summary
  Markdown table (append to WIP or paste into release notes).
- **Secrets:** Document which jobs need API keys vs fixture-only rows.

---

## References

- [ADR-031](../adr/ADR-031-mandatory-pre-release-validation.md)
- [Release Playbook](../guides/RELEASE_PLAYBOOK.md)
