# Digest & Library UX — alignment decisions

**Status:** Decided (feeds implementation in repo)  
**Spec:** [DIGEST-LIBRARY-UX-IMPROVEMENTS.md](./DIGEST-LIBRARY-UX-IMPROVEMENTS.md)  
**GitHub:** [#610](https://github.com/chipi/podcast_scraper/issues/610)

## §2.3 Similarity (topic band hit rows)

**Decision:** Show the **semantic tier label** (“Strong match” / “Good match” / “Weak match”) **inline on the row**, **right-aligned** in the hit meta area (per UX spec). Put the **raw numeric score** on a native **`title`** on that same element: `Similarity: 0.87` (and keep the existing row-level hover/title for other catalog fields unchanged).

**Rationale:** The WIP assumed a visible mono pill; current code often exposed score only inside a long native `title`. Inline text matches the spec’s “plain text label” and keeps precision available without implying false precision in the primary label.

## §2.4 CIL cluster pill colour (kg vs Quote)

**Decision:** Implement **`kg`** Tailwind treatment for **Digest Recent** CIL pills only: `bg-kg/15`, `border-kg`, text **`text-surface-foreground`** (readable on both themes). **Episode rail** and any other `CilTopicPillsRow` usage keep the existing **Quote-linked** amber chrome via a prop default so we do not change non-Digest surfaces.

**Rationale:** `text-kg-foreground` / `--ps-kg-foreground` do not exist in the theme today; `surface-foreground` on a light `kg` tint meets contrast without expanding the token set.

## §3.2 Feed filter threshold

**Decision:** Keep **`15`** feeds as the cutoff for showing the client-side “Filter feeds…” input, as a named constant in `LibraryView.vue` (and note in UXS tunables when docs are updated).

## §4 Recency dot (24h)

**Decision:** Parse **`publish_date`** as local **`YYYY-MM-DD`** at **00:00 local**. Show the dot when `Date.now() - localMidnight` is in **`[0, 86_400_000)`** (strict rolling 24 hours from the start of the listed publish **calendar day**). Hover title uses whole hours since that midnight. Full “same calendar day as today” is a subset edge case when the listed day is “today”.
