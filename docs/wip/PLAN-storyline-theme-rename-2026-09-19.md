# Plan — finishing the Storyline / Theme rename

**Status:** proposed, not started. Decision needed on scope (A–D below).

## The finding that reframes this

This is not a rename to *start*. It is one already **~75% done and stalled**, and the uneven
middle is worse than either end state.

| Layer | `storyline` (correct) | `theme_cluster` (wrong) | Done |
| --- | --- | --- | --- |
| Player (`web/learning-player/src`) | 591 | 47 | 93% |
| Backend (`src/podcast_scraper`) | 193 | 126 | 60% |
| Operator viewer (`web/gi-kg-viewer/src`) | 8 | 10 | 44% |

The clearest single illustration, in one file:

```python
# src/podcast_scraper/server/routes/app_discover.py
@router.get("/theme-clusters", response_model=AppStorylinesResponse)
def top_storylines(...)
```

The function and the response model were corrected. The route path and the module behind it were
not. Nobody decided that — it is what piecemeal correction looks like, and #1603 keeps being
reopened because of it.

## What is actually inverted

| Wire prefix | Backend name | Built from | Reader-facing name |
| --- | --- | --- | --- |
| `thc:` | "theme cluster" | **co-occurrence** — topics discussed together | **Storyline** |
| `tc:` | "topic cluster" | **vector similarity** — topics that mean similar things | **Theme** |

So `search/theme_clusters.py` serves storylines and `search/topic_clusters.py` serves themes; the
route `/theme-clusters` returns storylines and `/clusters` returns themes.

## Four separable pieces, very different costs

| # | Piece | Size | Cost | Deadline | Value |
| --- | --- | --- | --- | --- | --- |
| **A** | Internal modules + symbols | 183 in src/web, 144 tests, 71 docs | mechanical; no data touched | none | **high** — this is what people misread |
| **B** | API field + route names | 6 schema fields, 2 routes, 97 client refs | coordinated across 2 in-repo clients | none | medium — aligns the contract |
| **C** | Wire prefixes `thc:` / `tc:` | 1 mint site each, 4 fixture occurrences | cheap **now** | **launch** | **low** once A lands |
| **D** | Artifact filename `topic_theme_clusters.json` | 16 code refs | forces a prod re-enrichment | none | low |

### The deadline applies to the piece that matters least

I previously framed this as "mechanical while pre-launch, expensive after". That is true only of
**C** — and C is the *least* valuable piece. Once A has landed, `thc:` is an opaque token that
appears in one documented place; nobody reads it and nothing depends on its spelling. A and B carry
the value and have **no deadline at all**.

Correcting that framing is the main output of this analysis.

### Why D is not worth it

Renaming the artifact file means every existing corpus — including prod — needs re-enrichment or a
compat read. The repo's standing rule forbids compat shims (forward-only), so it would be a real
prod operation for zero user-visible gain. The filename is invisible to readers of the code once
the module around it is named correctly.

## Measured facts behind the costs

- **Mint sites are singular.** `thc:` at `enrichment/enrichers/topic_theme_clusters.py:456`
  (`f"thc:{slug}"`); `tc:` at `search/topic_clusters.py:512` (`f"tc:{tc_slug}"`). One line each.
- **Persisted user state does not parse prefixes.** The per-user interests/collections stores keep
  whatever token the client sends; no prefix logic, so no data migration.
- **Fixture data is trivial.** 4 occurrences of `thc:` across two committed corpus JSONs
  (3.8 KB + 5.5 KB). No search-index rows carry the id.
- **Prefix MATCHING is small and already centralised-ish**: `utils/interests.ts` (the boundary),
  plus `FollowedInterests.vue`, `LibraryView.vue`, `DiscoveryList.vue`.
- **API fields have exactly two consumers**, both in this repo: player (34 refs), viewer (63).
  The MCP server consumes none.

## Recommendation

**Do A. Then B. Skip C and D.**

That finishes the rename where it is read, leaves the wire format alone, and touches no corpus.
If C is ever wanted it stays available at the same price until launch — but after A it is hard to
argue for.

## Staged execution

Each stage is independently shippable and independently verifiable. Do not batch them.

### Stage A1 — the storyline half of the backend (`thc:`)

| From | To |
| --- | --- |
| `search/theme_clusters.py` | `search/storylines.py` |
| `consumer_theme_cluster_map()` | `storyline_map_by_topic()` |
| `top_theme_clusters_by_member_count()` | `top_storylines_by_member_count()` |
| `DEFAULT_MIN_THEME_MEMBERS` | `DEFAULT_MIN_STORYLINE_MEMBERS` |
| `THEME_CLUSTERS_REL`, `_THEME_CLUSTERS_REL` | `STORYLINES_REL` (value unchanged — see D) |
| `cluster_anchor()`, `_anchor_topic_id()` | unchanged (already neutral) |

`git mv` + symbol rename. The module's docstring keeps one line recording that the artifact it
reads is still named `topic_theme_clusters.json` and why (D deliberately skipped).

**Verify:** `pytest tests/unit/podcast_scraper/search tests/unit/podcast_scraper/server
tests/integration/server` + `mypy`. No behaviour changes, so any diff in test *outcomes* is a bug
in the rename.

### Stage A2 — the theme half of the backend (`tc:`)

`search/topic_clusters.py` → `search/themes.py`; `top_clusters_by_member_count()` →
`top_themes_by_member_count()`; `InterestCluster` → `Theme` where it means `tc:`.

Held separate from A1 because the two halves are exactly what gets confused — doing them in one
commit makes the diff unreviewable for the one property that matters.

### Stage A3 — the clients

Player (47) and viewer (10) internal symbols. No API change yet, so each client stage is
independently revertable.

### Stage B1 — API fields

`theme_cluster_id` / `theme_cluster_label` / `theme_cluster_size` → `storyline_id` / `_label` /
`_size`; `FeedSignalTheme` → `FeedSignalStoryline`; `dominant_themes` → `dominant_storylines`.
Server + both clients in ONE commit — a split would ship a broken contract between them.

### Stage B2 — routes

`/api/app/theme-clusters` → `/api/app/storylines`; `/api/app/clusters` → `/api/app/themes`.
Update `HTTP_API.md` in the same commit.

### Stage A4 — docs

UXS-013 §Vocabulary rewritten from "here is the inversion to watch for" to "here are the names",
retaining a short historical note. Close the naming half of #1603.

## Risks, and what to do about them

| Risk | Mitigation |
| --- | --- |
| A half-applied rename is worse than none — exactly today's state | Stages are ordered so each leaves the tree coherent; never merge a stage that renames a producer without its consumers |
| `theme` is a real English word; blind `sed` will hit prose and unrelated code (`--lp-theme`, `theme/tokens.css`, `DESIGN_DIRECTION` themes) | Rename by SYMBOL, never by substring. The design-token `theme` family is unrelated and must not move |
| The viewer is the least-converted layer (44%) and has its own tests | Its stage runs last and its suite (`ci-ui-full`) gates it |
| Fixture corpora embed `thc:` | Untouched — C is skipped |
| A stage looks green because a test was renamed alongside the code it guards | Per stage, assert the test COUNT is unchanged, and diff the list of test ids before/after |

## Explicitly NOT in this plan

- **C (wire prefixes)** — deferred indefinitely; available until launch if ever wanted.
- **D (artifact filename)** — declined; costs a prod re-enrichment for no reader-visible gain.
- **Where semantic clusters (`tc:` / Theme) belong in the product** — an open product question
  (#1603, #1595). This plan renames the thing; it does not decide where it lives.
- **The `"theme"` / `"similar"` i18n pair and the Knowledge Panel lead-in** — user-visible copy,
  tracked on #1603, not gated on this.
