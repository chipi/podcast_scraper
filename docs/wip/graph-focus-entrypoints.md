# WIP: Graph focus id entry points (Risk 4 follow-up)

**Status:** Draft checklist (not indexed). **Purpose:** align every “open graph” path with
the **node id vocabulary** the merged GI+KG graph actually contains (`topic:`, `tc:`,
episode-scoped slugs from bullets, search hit payloads, and so on).

## Phase 0 (landed)

- **Code:** `web/gi-kg-viewer/src/utils/cilGraphFocus.ts` — maps a `CilDigestTopicPill` +
  optional episode id to `graphNavigation.requestFocusNode` (primary `topic:…`, fallback
  episode id, optional `pendingFocusCameraIncludeRawIds` for `tc:…` when
  `in_topic_cluster`). Used from **Digest Recent** (`DigestView.vue`) and **Episode rail**
  canonical topic pills (`EpisodeDetailPanel.vue`).
- **Tests:** `web/gi-kg-viewer/src/utils/cilGraphFocus.test.ts` (Vitest).
- **Docs:** [Development Guide — Viewer v2](../guides/DEVELOPMENT_GUIDE.md#viewer-v2-rfc-062-489) (bullet **CIL pill to graph focus**).

## Why this exists

CIL and digest **pills** unify **bridge** `topic:` ids with optional RFC-075
**`topic_cluster_compound_id`** (`tc:`) for the viewer. The **cytoscape merge** still
contains multiple node kinds and legacy paths (for example bullet-derived `topic:` slugs).
If one surface passes the wrong token, focus can **miss** or highlight the **wrong**
vertex.

## Entry surfaces to audit (code + UX)

| Surface | Expected focus token | Notes |
| ------- | -------------------- | ----- |
| Digest — CIL chip | `topic:` and/or `tc:` from `cil_digest_topics[]` | Server builds pills; viewer must pass compound when clustered. |
| Digest — topic band / semantic row | Topic ids from digest API / GI load | Confirm same slugging as graph merge. |
| Search — open graph from hit | Hit payload (`lifted`, anchors, …) | Transcript lift vs insight hit may differ. |
| Library — Episode rail CIL | Same pill shape as digest detail | List rows omit pills; detail only. |
| Graph — double-tap expand (RFC-076) | Canonical `node_id` from graph | Already constrained by API contract. |
| Person / org drill-ins | `person:` / `org:` | Less overlap with `tc:` but worth a row in tests. |

## Suggested engineering outcomes (remaining)

1. ~~**Single helper** for CIL pills~~ — done for Digest + Episode rail (`cilGraphFocus.ts`).
2. **Extend or reuse** the same contract for other surfaces (topic band rows, explore,
   any future chips) and optionally dedupe SearchPanel’s `topic_cluster` camera logic
   with one shared primitive.
3. **Playwright** assertion that clustered pill passes camera ids (optional: spy on
   store or assert zoom behaviour if stable in mocks).
4. Optional: **dev-only log** when focus id is not found in the loaded graph (guarded so
   production builds stay quiet).

## When to promote out of `docs/wip/`

After the audit is done, fold the **contract** into [UXS-002 Corpus
Digest](../uxs/UXS-002-corpus-digest.md) / [UXS-003 Corpus
Library](../uxs/UXS-003-corpus-library.md) or the [Development Guide — GI / KG
viewer](../guides/DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) and keep this file as
history or delete it.

## GitHub issue

Tracked as [#596](https://github.com/chipi/podcast_scraper/issues/596) (body template:
`docs/wip/github-issue-graph-focus.md`).
