# Technical Debt Registry

This document tracks recognised technical debt across the project -- items that
work correctly today but have a known path to a better solution when the time
is right.

Each entry records the current state (how we cope), the ideal state (what
"done" looks like), the two or three realistic options, and a rough priority
and trigger for when to revisit.

---

## TD-001: CodeQL `py/path-injection` false positives

**Tracking:** [#538](https://github.com/chipi/podcast_scraper/issues/538)

**Status:** Managed (dismissal process in place)

**Current state:**

CodeQL flags every filesystem call (`os.walk`, `os.path.isdir`, `open`, etc.)
that receives a value derived from a FastAPI query parameter, even when the
value has been sanitised via `os.path.normpath` + `str.startswith` in a shared
helper function.  CodeQL's taint-tracking state machine requires the guard to
appear inline in the same function as the filesystem call; our architecture
performs sanitisation in shared helpers (`resolve_corpus_path_param`,
`resolved_corpus_root_str`, `normpath_if_under_root`,
`safe_relpath_under_corpus_root`).

As of April 2026, approximately 60 alerts have been dismissed as false
positives.  The full inventory and dismissal process are documented in
[`docs/ci/CODEQL_DISMISSALS.md`](../ci/CODEQL_DISMISSALS.md).

**Why this is debt, not a bug:**

- Zero security risk -- the sanitisers are correct and thoroughly tested.
- Low operational cost -- dismissing a new alert takes ~30 seconds (one
  `gh api` call + one table row), and the agent `.cursorrules` rule 16
  automates classification.
- But the codebase carries a growing list of dismissed alerts, and every new
  server route that touches the filesystem will produce more.

**Options to eliminate it:**

| Option | Approach | Pros | Cons |
| --- | --- | --- | --- |
| A | Custom CodeQL query pack | Write a `.ql` model extension that teaches CodeQL's taint tracker that `resolve_corpus_path_param` and friends are sanitisers (`isSanitizer` override). All ~60 alerts disappear at source; no dismissals needed. | Niche skill (CodeQL QL language); pack must be maintained as CodeQL evolves; benefits only this repo. |
| B | Inline sanitisation | Restructure every route handler to do `normpath + startswith` inline before any filesystem call. CodeQL sees the guard and stops flagging. | Duplicates the same 3-line guard in every handler; worse code quality for the sake of a static analysis tool; ongoing maintenance burden on every new route. |
| C | Wait for CodeQL improvement | GitHub has discussed cross-function sanitiser support in CodeQL issues. If/when they ship it, the false positives disappear with no code changes. | No timeline; may never land. |

**Recommendation:** Option C (wait) as the default posture.  Option A is the
right investment if we ever do a dedicated security hardening sprint or if the
alert volume becomes annoying.  Option B is not recommended -- it makes the
code worse.

**Priority:** Low

**Trigger to revisit:**

- CodeQL ships cross-function sanitiser modelling (Option C resolves itself)
- A new, genuinely dangerous alert type appears that requires deeper CodeQL
  investigation (justifies learning the QL language for Option A)
- Alert count exceeds ~100 and the dismissal table becomes unwieldy
- A security audit or compliance review requires zero open/dismissed alerts

---

<!-- Add new TD-NNN entries above this line, following the same template. -->
