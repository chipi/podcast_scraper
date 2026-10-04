# Token vocabulary cross-map (operator viewer ↔ consumer app)

The repository has **two independent design systems** on purpose — the operator GI/KG viewer
([UXS-001](UXS-001-gi-kg-viewer.md), `--ps-*`) and the consumer learning app
([UXS-011](UXS-011-consumer-learning-app.md), `--lp-*`). Different audiences, different identity;
neither should borrow the other's values.

They do share **names**. The same Tailwind class (`text-topic`, `bg-accent`, …) compiles in both
apps but means a different hue — and, for `accent`, a different *rule* about where it may be spent.
This page exists so that nobody (human or agent) working across both directories carries one
system's meaning into the other. It maps; it does not unify.

Values are the **dark defaults** (the baseline of both systems). The viewer also ships a light
theme; the consumer app repaints through visual directions (`src/theme/directions.css`).

## Shared names

| Concept | Operator viewer (`--ps-*`) | Consumer app (`--lp-*`) |
| --- | --- | --- |
| `accent` | **Alias of `primary`** (`#4c90f0`), used freely across shell chrome — `tailwind.config.js` maps both names to `--ps-primary`. | **Per-show adaptive**, derived from the artwork and contrast-clamped, default `#efa843`. Spent **only** on things a finger can act on: focus rings, `aria-selected` state, `.lp-fav` hover/pressed (#2013). Enforced by `accent-discipline.test.ts`. |
| `topic` | Teal `#7cd0d4` — a KG identity hue. | Cool grey `#a8b0c6` — deliberately quiet; knowledge hues separate by temperature and value, not saturation. |
| `person` | Peach `#ffb37a`. | Warm grey `#ccc7bb`. |
| `grounded` | **Alias of `gi`** (`#7dd3a0`) — grounding is the GI layer's colour. | Its own sage `#9fb8a4`; no `gi`/`kg` tokens exist in the consumer app. |
| `warning` | Orange `#ec9a3c`. | Amber `#efa843` — the same value as the brand default, so a warning and the default accent share a hue; meaning is carried by placement and text. |
| `theme` | **Topics discussed together** (co-occurrence) — teal `#7dd3c0`. | **Topics that mean the same thing** (similarity) — lilac `#a9a3c6`. **Same word, different data** — see *Topic groupings* below. |

## Topic groupings — the same word means different data

Both apps show two ways of grouping topics, and the API feeds both. Each app's **visible** names
are the source of truth; its CSS token names follow its visible names; the **API names lag** and
match neither app exactly (the wire still calls a storyline a "theme cluster", `thc:`).

| Grouping | API / data | Consumer app shows | Viewer shows | CSS |
| --- | --- | --- | --- | --- |
| Topics **discussed together** (co-occurrence) | `storylines`, `thc:` ids — `/api/app/storylines`, `storylinesDoc` | **Storyline** | **Theme** (Details block, "Theme regions" legend, show rail, "Theme landscape") | `--lp-storyline` · `--ps-theme` |
| Topics that **mean the same thing** (similarity) | topic clusters, `tc:` ids — `/api/app/themes` (`AppInterestCluster`), `topicClustersDoc` | **Theme** | **Topic cluster** | `--lp-theme` · viewer uses `kg` (UXS-001) |
| A single topic | `topic` | Topic | Topic | `--lp-topic` · `--ps-topic` |

So **`--ps-theme` and `--lp-theme` are the same name for different data.** Code, copy or a design
decision moved between the apps must be translated by grouping, not by word: the viewer's
"Theme" is the consumer app's "Storyline", and the consumer app's "Theme" is the viewer's
"Topic cluster". Whether the two apps *should* share words is a product decision about visible
labels; this page records the current state, it does not change it.

## Names only one system has

| Only in the operator viewer | Only in the consumer app |
| --- | --- |
| `primary` / `primary-foreground`; the `-foreground` pairs for `success`, `warning`, `danger`, `gi`, `kg`; `gi`, `kg` (identity colours, marked frozen in UXS-001); `related-topic`; `graph-canvas` | `storyline`; the four `insight-*` type marks (aliases onto `topic` / `grounded` / `warning` / `person`); `brand-default` |

There is **no `primary` token in the consumer app** and **no `storyline` / `insight-*` token in the
viewer**. A class written for one app that names one of these will not resolve in the other.
`theme` resolves in both — to different data (above).

## Rules of thumb

- **Never copy a value across.** If a hue looks right in the other app, that is a coincidence of
  two palettes, not a shared token.
- **`accent` is the dangerous one.** In the viewer it is ordinary chrome; in the consumer app it is
  a scarce signal guarded by a test. Code moved from the viewer into the consumer app that uses
  `accent` for a label or a badge will fail `accent-discipline.test.ts`, correctly.
- **`theme` is a false friend.** It compiles in both apps and names different groupings: the
  viewer's co-occurrence grouping, the consumer app's similarity grouping. Translate by grouping.
- **Concept names agree; meanings mostly agree.** `topic`, `person` and `grounded` mark the same
  knowledge-layer objects in both apps — only the paint differs. `accent` and `theme` are the
  exceptions.

## Keeping this page true

Both columns are checked indirectly: each design system's own tables are tested against its
`tokens.css` (`web/learning-player/src/__checks__/uxs-token-tables.test.ts` and
`web/gi-kg-viewer/src/__checks__/uxs-token-tables.test.ts`, #2280). This page itself is not — when
a value on either side changes, update it in the same change.
