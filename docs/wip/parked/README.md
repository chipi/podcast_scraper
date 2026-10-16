# Parked work, kept as patches

Two branches were deleted from GitHub on 2026-10-16 (operator decision). What they held that
main did not is kept here as patches. **Neither patch is applied.** They are a record, and a
starting point if the idea comes back. Both were cut from main as it stood at their base commit,
so expect conflicts if you apply one.

## `cleaned-base-llm-selection-PARKED.patch`

From `feat/cleaned-base-llm-selection` at `4758c43ad` (2026-10-05). Base: `2915387fb`
(2026-10-04). 44 files, +2,495 / −571.

What it is: LLM transcript cleaning by **selection** (#2276). The model picks which sentences to
drop, or names ad blocks by both ends, and never writes transcript text. One shared prompt for
every provider, and a "cleaned base" every consumer reads. The core is
`src/podcast_scraper/cleaning/span_selection.py` and `workflow/adfree_transcript.py`, plus about
460 lines of tests.

Why it is parked: it was held back from #2285 on purpose (only the naming half shipped there).
On 2026-10-16 the operator kept it parked because mobile's rule-based mid-roll cutting
(#2168, `50e3ca642`) covers the same ground without an LLM.

Note: the patch also contains that branch's naming and roster commits. Those are on main already
(the #2285 squash `201e0b584`), so take main's side for those files.

## `naming-stated-guest-binding-PARKED.patch`

From `feat/naming-stated-guest-binding` at `0bc374d51`. Base: `33cf9db36` (2026-10-03). Only the
`src/` changes: the parked stated-guest binding experiment and a narrowed third-person surname
check (`roster.py`, `hosts.py`, `resolution.py`).

The rest of that branch is on main now:

- `scripts/measure/trace_rung_precision.py`
- the `--names-new` / `--names-old` flags in `scripts/measure/naming_gate.py`
- `docs/wip/NAMING_DECISION_TRACE.md`
- the branch's scoreboard, as `docs/wip/NAMING_SCOREBOARD_stated-guest-binding-2026-10.md`
  (a different document from main's `NAMING_SCOREBOARD.md`)

Its 7 measured and rejected changes are in `NAMING_KNOWN_ISSUES` under "tried and rejected"
(`d0361c721`).
