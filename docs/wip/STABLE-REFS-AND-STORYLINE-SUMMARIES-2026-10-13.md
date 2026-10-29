# Stable references (#2345) and storyline summaries (#2344) — plan, 2026-10-13

Branch `feat/2344-2345-refs-and-storyline-summaries`, from main e6cc51be2. One PR.

## Decisions (operator, 2026-10-13)

| # | Decision |
| --- | --- |
| 1 | Theme (`tc:`) and storyline (`thc:`) ids are kept across re-clustering: a new cluster inherits a previous id when the member Jaccard is at least 0.5, one-to-one. Each artifact publishes `id_history`: successors (renamed or absorbed → the cluster that took its members) and gone ids. |
| 2 | The repair runs in the board-summarizer sidecar's loop, before the summaries, with no LLM call. |
| 3 | Re-keyed or flagged: interests (follows), "Less like this", favorites, collection/board items, notes, learning paths and their history, short-listen progress, highlight graph refs. Episode references are flagged only. Append-only analytics logs are left alone. |
| 4 | A reference with a successor is re-keyed silently. One that is truly gone stays, muted, "No longer available", with a remove action. |
| 5 | Every storyline gets a summary (all 99 on prod): consistency. A short thread (2–4 episodes) says what it is and what each episode adds; a longer one also how it developed and where the voices differ. |
| 6 | Storyline prompt: at most 3 sentences, 80 words. |
| 7 | Board summaries: the sweep drops summaries of deleted boards; the fingerprint counts only episodes that still exist. |
| 8 | Order: #2345 first, then #2344. |
| 9 | Backfill: #2344's sweep fills storylines on its own under a per-sweep cap, plus a one-shot command (dry-run, then apply). #2345: a one-time re-key of existing user state (dry-run → apply → undo, with receipts); it repairs only renames it has history for, older broken references are flagged. |
| 10 | Storyline durability (minimum size, stickier labels) is a separate issue later. |

## Prod numbers behind them (read-only, 2026-10-13)

- 99 storylines. Episodes spanned: 2 → 18 storylines, 3 → 13, 4 → 12, 5+ → 56 (largest 89).
- Id churn over 3 days of snapshots: Jaccard ≥ 0.5 keeps 80 of 251 renamed themes and 3 of 10 renamed storylines.

## Steps

1. `search/stable_cluster_ids.py`: inherit ids from the previous artifact, publish `id_history`; wired into the storyline enricher and the theme clustering.
2. `server/reference_integrity.py`: successors + existence → re-key or flag every holder in (3); per-user receipts; the sidecar runs it each tick; `scripts/ops/rekey_user_references.py` for the one-time backfill (dry-run / apply / undo).
3. The player shows flagged references muted with a remove action.
4. Storyline summaries in the sidecar (prompts under `prompts/shared/summarization/storyline_*`), stored in the player data dir, served on the storyline page and at the top of its learning path; orphan sweep; one-shot backfill command.
5. Board-summary fixes (7).

## Status (2026-10-13)

Done on the branch, each with tests: steps 1–5. Commits: stable ids (2512ba7d3), the reference
pass, sidecar step and backfill command (7020abf0b), the player's "No longer available"
(67193c205), storyline summaries and the board fixes (0b28f8da5). The post-deploy order is §7 of
POST-DEPLOY-BOARDS-AND-DGX-GATEWAY (not in this repository).

Measured on prod (read-only, 2026-10-14): 6 accounts, 3 of them hold 18 follows that point at
nothing — 13 themes and 5 storylines renamed by re-clustering. No saved storylines, and no reference
to the one id m0010 changed (`person:peter-attia-md`).

Closed on 2026-10-14:

- A storyline is saved (favorite, board, note) by its own `thc:` id, once its card has loaded; an
  older save by its anchor topic is re-keyed to the storyline that topic belongs to now, or flagged.
- Playwright: `storyline-summary-and-gone-references.spec.ts` — the summary on the storyline page
  (by anchor and by id) and atop its Learn path, the save by id, a gone follow muted and
  unfollowable.

Not covered:

- Person ids are re-keyed where the #852 entity map merges spellings. m0010 changed one id
  (`person:peter-attia-md`), which no prod account references; nothing to repair.
- Decision 10 (storyline durability) is not filed: issues are opened only on the operator's word.
