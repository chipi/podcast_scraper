Mirror of [chipi/podcast_scraper#596](https://github.com/chipi/podcast_scraper/issues/596) for
grep-friendly copy; edit the GitHub issue for canonical discussion.

## Motivation

The merged GI+KG graph uses several id shapes (`topic:…` leaves, RFC-075 `tc:…` compound
parents, episode ids, search hit anchors). **CIL digest pills** carry both `topic_id` and
optional `topic_cluster_compound_id`, but different UI surfaces historically called
`graphNavigation.requestFocusNode` inconsistently. Wrong or incomplete arguments produce
empty focus, wrong selection, or correct selection with poor framing.

**Re-pipeline / enrichment:** canonical ids and cluster membership can change when
artifacts are rebuilt. That is a separate lifecycle concern; see [RFC-072 operational
note](https://github.com/chipi/podcast_scraper/blob/main/docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md#operational-note-re-pipeline-enrichment-and-read-path-stance).

## What landed (Phase 0)

- Shared helper: `web/gi-kg-viewer/src/utils/cilGraphFocus.ts`
- Wired from **Digest Recent** CIL pills and **Episode rail** canonical topic pills so
  clustered pills pass `tc:…` into `pendingFocusCameraIncludeRawIds` (same contract as
  Search transcript/KG hits that pass a compound for camera bbox).
- Vitest: `web/gi-kg-viewer/src/utils/cilGraphFocus.test.ts`
- Maintainer checklist and follow-ups: `docs/wip/graph-focus-entrypoints.md`
- Developer note: [Development Guide — Viewer v2](https://github.com/chipi/podcast_scraper/blob/main/docs/guides/DEVELOPMENT_GUIDE.md#viewer-v2-rfc-062-489) (bullet **CIL pill to graph focus**).

## Proposed follow-up (pick up later)

1. **Audit** remaining “open graph” paths in `docs/wip/graph-focus-entrypoints.md` (topic
   band rows, Explore, any new chips). For each, record which id is primary vs camera-only.
2. **Optional dedupe:** factor SearchPanel’s `topic_cluster` → `cameraIncludeRawIds` into
   the same module or a sibling helper so Search and CIL cannot drift.
3. **Playwright (optional):** one assertion per major surface if we can observe stable
   behaviour under mocks (otherwise skip).
4. **Dev-only diagnostics (optional):** when `resolveCyNodeId` misses after focus request,
   log primary + extras once in `import.meta.env.DEV`.

## Acceptance

- [ ] Remaining entry surfaces documented or refactored through `cilGraphFocus` (or a
      single documented module family).
- [ ] No regression: Digest CIL pill + Episode rail pill + Search “go graph” still work
      on fixture corpora.
