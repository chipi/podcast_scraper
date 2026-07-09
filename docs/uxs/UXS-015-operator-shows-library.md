# UXS-015: Operator Shows Library (shows-first browse)

**Surface:** `web/gi-kg-viewer` — `library` tab, **Shows** mode
**Inherits:** [UXS-001](UXS-001-gi-kg-viewer.md) (operator shared design system — tokens, type, density),
[UXS-003](UXS-003-corpus-library.md) (episode-first Library — episode-row contract)
**PRD/RFC:** [PRD-044](../prd/PRD-044-operator-shows-library.md) · [RFC-104](../rfc/RFC-104-operator-shows-library.md)

This UXS owns the **static visual contract** (layout grid, card appearance, tokens, states, a11y).
Behavioral rules (fetch strategy, state machine, cross-link) live in RFC-104.

## Scope

Two views inside the Library tab's **Shows** mode:

1. **Shows grid** — the corpus's shows as cover cards.
2. **Show detail** — one show's header + its episode list.
Plus the **mode toggle** (Shows | Episodes) at the tab head. Reuses `PodcastCover` and the UXS-003
episode-row; introduces no new tokens.

## Mode toggle

- Segmented control at the Library tab head, left-aligned, above the content: `[ Shows | Episodes ]`.
- Tokens: inherits the shared segmented-control style (`bg-overlay`, active `bg-overlay-2 text-primary`,
  inactive `text-muted`), `text-xs font-semibold`, `rounded`, `h-7`. Matches the Digest/Library chip bar.
- `data-testid="library-mode-shows"` / `library-mode-episodes`; `aria-pressed` on the active segment.

## Shows grid (ShowsView)

**Layout**

- Responsive grid: `grid gap-3` at `grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5`
  (dense, operator-tool density per UXS-001 — smaller than the consumer's editorial grid).
- **Show card** (`shows-card-{feed_id}`): vertical stack —
  - `PodcastCover` at `size-class="h-full w-full aspect-square"`, `rounded-lg`, `object-cover`,
    `bg-elevated` placeholder; initials fallback when no art.
  - Title: `text-sm font-semibold leading-snug line-clamp-2 text-surface-foreground`.
  - Meta: `text-[11px] text-muted` — "N episodes" (`episode_count`).
  - Optional description: `text-[11px] text-muted line-clamp-2` (only if present; never pushes layout).
- Card container: `rounded-lg border border-default bg-overlay p-2 hover:bg-overlay-2`,
  `role="button" tabindex="0"`, visible `focus-visible:ring-2 ring-primary` (parity with library rows).

**States**

- Loading: skeleton cards (`animate-pulse bg-overlay` squares) or the shared spinner — match UXS-003.
- Error: `text-xs text-danger` inline message + retry affordance (shared pattern).
- Empty (0 feeds): centered `text-sm text-muted` — "No shows in this corpus." (`shows-grid-empty`).

## Show detail (ShowDetailView)

**Header** (`show-detail`)

- Row: large `PodcastCover` `h-20 w-20 sm:h-24 sm:w-24 rounded-xl object-cover shrink-0` + a text column.
- Title: `text-xl font-extrabold leading-tight tracking-tight text-surface-foreground`.
- Sub-line: `text-xs text-muted` — "N episodes" · optional RSS link (`rss_url`, `text-primary
  hover:underline`, opens in new tab, `rel="noopener"`).
- Description: `text-sm text-muted leading-relaxed`, `line-clamp-3`; when longer than the clamp, a
  "Show more/less" toggle (`text-xs text-primary`) mirroring the consumer PodcastView 180-char clamp.
- **Back to shows**: a top-left `‹ Shows` button (`show-detail-back`), `text-xs text-muted
  hover:text-surface-foreground`, keyboard-focusable; returns to the grid (replace-in-panel).

**Episode list** (`show-detail-episode-{i}`)

- Reuses the **UXS-003 episode row** verbatim (cover `h-9 w-9`, recency dot, title, publish date,
  summary line, topic pills, GI/KG badges), same classes + `data-library-episode-row` semantics, so a
  Show-detail episode is visually and behaviorally identical to a flat-Library episode.
- Newest-first; "Load more" button (`show-detail-load-more`) when `next_cursor` present.
- Empty (show has 0 episodes): `text-xs text-muted` — "No episodes." (`show-detail-empty`).

## Cross-surface flow (visual)

- Grid card → detail: in-panel replace (no modal, no new backdrop — one-surface rule, operator memory /
  UXS-014). Back returns to grid at prior scroll.
- Episode row → graph: click routes through the existing `focusEpisode` path; the Library tab yields to
  the Graph tab (existing transition), so no new visual affordance is introduced here.

## Accessibility

- Grid + rows: `role="button"`, `tabindex="0"`, Enter/Space activate, visible focus ring; card
  `aria-label` = `"{title}, {episode_count} episodes"`.
- Cover images: `alt` = `"Cover for {title}"`; decorative fallback initials are `aria-hidden` with the
  label carried by the card `aria-label`.
- Mode toggle: `aria-pressed`; arrow-key movement optional (parity with existing segmented controls).
- Contrast: all text meets UXS-001 AA targets on `bg-overlay` / `bg-elevated`.

## Review checklist

- [ ] Uses only UXS-001 tokens; no bespoke colors/shadows.
- [ ] `PodcastCover` reused (no new image component); graceful art fallback, never a broken `<img>`.
- [ ] Episode row is byte-for-byte the UXS-003 row (classes + testids), not a re-styled copy.
- [ ] Loading / error / empty present on grid **and** detail.
- [ ] Replace-in-panel (no modal / second backdrop) for grid ↔ detail.
- [ ] Focus rings + keyboard activation on every card/row/toggle.
- [ ] `data-testid`s match RFC-104 §Testing (`shows-grid`, `shows-card-{id}`, `show-detail`,
      `show-detail-back`, `show-detail-episode-{i}`, `library-mode-shows|episodes`).

## Related

UXS-001 (tokens), UXS-003 (episode row), UXS-011/012 (consumer parallel for reference only — the
operator uses UXS-001 density, **not** the consumer editorial system).
