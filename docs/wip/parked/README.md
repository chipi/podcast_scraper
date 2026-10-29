# Parked work, kept as patches

A branch was deleted from GitHub on 2026-10-16. What it held that
main did not is kept here as a patch. **The patch is not applied.** It is a record, and a
starting point if the idea comes back. It was cut from main as it stood at its base commit,
so expect conflicts if you apply it.

## `cleaned-base-llm-selection-PARKED.patch`

From `feat/cleaned-base-llm-selection` at `4758c43ad` (2026-10-05). Base: `2915387fb`
(2026-10-04). 44 files, +2,495 / −571.

What it is: LLM transcript cleaning by **selection** (#2276). The model picks which sentences to
drop, or names ad blocks by both ends, and never writes transcript text. One shared prompt for
every provider, and a "cleaned base" every consumer reads. The core is
`src/podcast_scraper/cleaning/span_selection.py` and `workflow/adfree_transcript.py`, plus about
460 lines of tests.

Why it is parked: it was held back from #2285 on purpose (only the naming half shipped there).
On 2026-10-16 it was kept parked because mobile's rule-based mid-roll cutting
(#2168, `50e3ca642`) covers the same ground without an LLM.

Note: the patch also contains that branch's naming and roster commits. Those are on main already
(the #2285 squash `201e0b584`), so take main's side for those files.
