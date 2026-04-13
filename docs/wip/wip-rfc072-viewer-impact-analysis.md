# RFC-072 Impact on GI/KG Viewer v2 — Plumbing Analysis

**Date:** 2026-04-12
**Context:** RFC-072 (Canonical Identity Layer & Cross-Layer Bridge) changes ID
schemes and node types across GIL and KG artifacts. This note maps the immediate
impact on the existing GI/KG viewer (RFC-062) — what breaks, what needs updating,
and what new wiring is needed. Use-case features (Position Tracker, Guest Brief)
are out of scope here; this is plumbing only.

---

## Impact Area 1: Node Type Vocabulary

GIL renames `Speaker` to `Person`. KG removes the `Entity` wrapper — nodes use
`person:` / `org:` IDs with `kind` property instead of `type: "Entity"` with
`entity_kind`.

| File | What breaks | What to do |
| --- | --- | --- |
| `web/gi-kg-viewer/src/utils/visualGroup.ts` | Checks `n.type === 'Entity'` and splits on `entity_kind`. Post-migration, GIL emits `type: "Person"` and KG nodes no longer use `type: "Entity"` with `entity_kind`. | Add `Person` as a recognized type (map to `Entity_person` visual group, or introduce a new `Person` group). Handle KG nodes that use `kind: "person"` / `kind: "org"` instead of `entity_kind`. |
| `web/gi-kg-viewer/src/utils/colors.ts` | Has `Speaker` in `graphNodeTypeStyles` and `graphNodeTypesOrdered`. Post-migration, GIL emits `Person` not `Speaker`. | Add `Person` style (can reuse `Speaker` colors or give it the `Entity_person` purple). Decide whether `Speaker` stays as a legacy fallback or is removed. |
| `web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts` | Has selector `node[type = "Speaker"]`. | Add `node[type = "Person"]` selector. |
| `web/gi-kg-viewer/src/stores/graphFilters.ts` | Filter toggles use `RawGraphNode.type` keys from the artifact. `Speaker` disappears, `Person` appears. | Update any hardcoded type lists. |

**Design decision needed:** Unify `Speaker` (GIL) and `Entity_person` (KG) into a
single `Person` visual group? RFC-072 says they are the same thing
(`person:lex-fridman`). The viewer currently shows them as two different colors
(green for Speaker, purple for Entity\_person). Post-CIL, they are the same node
in the bridge — showing them as two colors in a merged graph would be confusing.

---

## Impact Area 2: ID Prefix Handling

`speaker:` becomes `person:`. `entity:person:` becomes `person:`.
`entity:organization:` becomes `org:`.

| File | What breaks |
| --- | --- |
| `web/gi-kg-viewer/src/utils/parsing.ts` — `entityDisplayNameFromId()` | Regex `^entity:(?:person\|organization):(.+)$` will no longer match. New IDs are `person:slug` and `org:slug`. |
| `web/gi-kg-viewer/src/utils/parsing.ts` — `nodeLabel()` / `fullPrimaryNodeLabel()` | Falls back to `entityDisplayNameFromId` for label extraction. Will return empty string for new ID formats, so nodes may show raw IDs instead of humanized names. |
| `web/gi-kg-viewer/src/utils/searchFocus.ts` — `resolveCyNodeId()` | Candidate list includes `k:kg:${raw}` variants for old KG IDs. New `person:` / `org:` IDs need different resolution candidates. |
| `web/gi-kg-viewer/src/utils/mergeGiKg.ts` — `entityCanonicalKey()` | Dedup checks `DEDUP_TYPES = Set(['Entity', 'Topic'])`. Post-migration, `Entity` type may not exist. `Person` and `Organization` need to be in the dedup set. |
| `web/gi-kg-viewer/src/utils/mergeGiKg.ts` — GI+KG merge | Prefixes `g:` / `k:` to all node IDs. With CIL, `person:lex-fridman` appears in both `gi.json` and `kg.json`, producing `g:person:lex-fridman` and `k:person:lex-fridman`. Current `deduplicateEntities` merges by `name` property — this still works but is fragile. Cleaner path: dedup by canonical ID prefix (`person:`, `org:`, `topic:`) before falling back to name matching. |

---

## Impact Area 3: Search Result Display

Search hits carry `speaker_id` in metadata (for quotes) and `source_id` for graph
focus. These change from `speaker:slug` to `person:slug`, and from
`entity:person:slug` to `person:slug`.

| File | What breaks |
| --- | --- |
| `web/gi-kg-viewer/src/components/search/ResultCard.vue` | Displays `q.speaker_id` raw. Currently shows `speaker:sam-altman`; will show `person:sam-altman`. Not broken, but display string changes. |
| `web/gi-kg-viewer/src/stores/explore.ts` | `ExploreTopSpeaker` interface has `speaker_id: string`. Field name stays but value changes from `speaker:slug` to `person:slug`. Filtering still works. |
| `web/gi-kg-viewer/src/components/explore/ExplorePanel.vue` | Displays `sp.name \|\| sp.speaker_id` — fallback to raw ID will show `person:slug` instead of `speaker:slug`. |
| `web/gi-kg-viewer/src/utils/searchFocus.ts` | `graphNodeIdFromSearchHit` returns `metadata.source_id`. For KG entities, `source_id` was `entity:person:slug`; now it is `person:slug`. The `resolveCyNodeId` candidate list needs updating. |

---

## Impact Area 4: Bridge Artifact — New Capability

`bridge.json` per episode is a new artifact. The viewer currently has no awareness
of it.

**Immediate plumbing (not use cases, just wiring):**

1. **Artifact discovery** — `GET /api/artifacts` (`routes/artifacts.py`) globs for
   `*.gi.json` and `*.kg.json`. It does not discover `bridge.json`. Add
   `**/bridge.json` to the glob (or a separate endpoint).

2. **Corpus catalog** — `corpus_catalog.py` checks for sibling `.gi.json` /
   `.kg.json` from metadata. Add `has_bridge` flag to episode detail responses.

3. **Graph merge** — `mergeGiKg.ts` currently deduplicates Entity/Topic nodes by
   name. With `bridge.json` available, the merge could use canonical IDs from the
   bridge instead of fuzzy name matching. This is a cleaner, more reliable dedup.

4. **Node detail panel** — `GraphNodeRailPanel.vue` / `NodeDetail.vue` could show
   "appears in GI / KG / both" from `bridge.json` `sources` field. Small UI
   addition, high diagnostic value.

---

## Impact Area 5: Explore Panel (GI-only)

The explore endpoint (`GET /api/explore`) scans `gi.json` files. It returns
`speaker_id` fields with old `speaker:` prefix format.

| File | What to update |
| --- | --- |
| `src/podcast_scraper/gi/explore.py` (server-side) | Will emit `person:` IDs instead of `speaker:` after migration. |
| `web/gi-kg-viewer/src/stores/explore.ts` | `ExploreTopSpeaker.speaker_id` — field name is `speaker_id` but value becomes `person:slug`. Consider renaming the interface field to `person_id` for clarity, or keep as-is since it is a display concern. |

---

## Concrete Task List (Plumbing Only)

### Must fix (viewer breaks without these)

1. `parsing.ts` — update `entityDisplayNameFromId` regex to match `person:(.+)` and
   `org:(.+)` in addition to (or instead of) the old `entity:person:` /
   `entity:organization:` patterns.
2. `visualGroup.ts` — handle `type: "Person"` from GIL (map to visual group). Handle
   KG nodes that no longer have `type: "Entity"` with `entity_kind`.
3. `colors.ts` — add `Person` to `graphNodeTypeStyles` and `graphNodeTypesOrdered`.
   Decide on color unification with `Entity_person`.
4. `cyGraphStylesheet.ts` — add `node[type = "Person"]` selector.
5. `mergeGiKg.ts` — update `DEDUP_TYPES` and `entityCanonicalKey` to handle `Person`
   / `Organization` types (or dedup by canonical ID prefix).
6. `searchFocus.ts` — update `resolveCyNodeId` candidates for new ID formats.

### Should fix (viewer works but displays oddly)

7. `ResultCard.vue` — humanize `person:slug` display in quote attribution.
8. `ExplorePanel.vue` — same for speaker display fallback.
9. `explore.ts` — consider renaming `speaker_id` to `person_id` in the interface.

### New plumbing (bridge awareness)

10. Server: add `bridge.json` to artifact discovery or a new endpoint.
11. `mergeGiKg.ts` — optionally use bridge canonical IDs for dedup instead of name
    matching.
12. Episode detail: add `has_bridge` flag.

---

## References

- RFC-072: `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md`
- RFC-062: `docs/rfc/RFC-062-gi-kg-viewer-v2.md`
- Viewer source: `web/gi-kg-viewer/src/`
- Server source: `src/podcast_scraper/server/`
