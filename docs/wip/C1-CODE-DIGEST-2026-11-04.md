# C1: the migration code digest covers most of the package (design note, 2026-11-04)

Arc item C1 (`INGESTION-ARC-2026-11-03.md`, tier C). Status: **measured, not built**; the operator
decides between the options below.

## What happens today

- `upgrade/outcome_hook.code_digest` hashes the migration module and every `podcast_scraper` module
  it reaches by import (`module_closure`), parent packages included, plus its verdicts and list
  files. The gate records it; `build_outcome` copies it into the outcome; the landing check
  (`quality/change_gate`) compares it with the tree while the record is open; prod's outcome hook
  (`check_static`) refuses an apply whose installed code has another digest.
- For 0036 to 0043 the closure is **610 of the package's 778 modules**. An edit to any of them makes
  an open migration's outcome stale. On 2026-11-04 tier L's `identity/` edits and a variable rename
  each made 0043's outcome stale; it was re-gated three times with no change in its plan.

## Options measured

| Option | Closure for 0043 | Effect |
| --- | --- | --- |
| today (`closure-v1`) | 610 modules | every edit in most of the package re-gates every open migration |
| parent `__init__` bytes only, their imports not followed (`closure-v2`) | **603** | almost nothing: the breadth is the migrations' own explicit imports (roster, gi, kg) |
| modules whose code ran during the gate's apply on the clone, plus the migration's direct imports | not measured | narrow and evidence-based; misses code a prod-only input would reach |
| keep v1, re-gate only at the end of a branch | 610 | today's practice made explicit; costs one gate run per open migration per branch |

## What any change needs

- **Versioned:** outcomes record `digest_scheme`; one without it is `closure-v1`. The hook and the
  landing check compute the digest by the outcome's own scheme, so prod's existing outcomes keep
  validating and none is rebuilt.
- The prod hook's refusal (`the installed code is not the code that was measured`) stays the
  guard; a narrower scheme narrows what that guard sees.

## Recommendation

Keep `closure-v1` and make "re-gate open migrations after the branch's last package change" a
step of the branch's runbook (already in the arc's deploy inventory). The executed-code option is
the only one that narrows, and it weakens the prod guard for inputs the clone does not have:
that is a safety trade the operator decides, with a measurement of its closure first.
