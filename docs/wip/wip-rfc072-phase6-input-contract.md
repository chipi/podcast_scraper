# RFC-072 — Phase 6 analysis input contract (WIP)

**Purpose:** Stable, versioned description of what a **future Phase 6 analysis layer**
(contradiction detection, stance comparison, position-change over time) should consume,
built on top of RFC-072 Phases 1–5. This is a **contract note**, not an implementation
commitment.

**Related:** [RFC-072](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md),
[out-of-scope backlog](wip-rfc072-out-of-scope-backlog.md),
[Semantic Search Guide — lift](../guides/SEMANTIC_SEARCH_GUIDE.md#chunk-to-insight-lift-and-offset-verification-rfc-072--528),
[GIL / KG / CIL cross-layer](../guides/GIL_KG_CIL_CROSS_LAYER.md).

---

## 1. Core artifacts (per episode)

| Artifact | Role for Phase 6 |
| -------- | ---------------- |
| `*.gi.json` | Typed **Insight** / **Quote** / **Person** / **Topic** graph; **`position_hint`** on insights when available; evidence via **SUPPORTED_BY**, **ABOUT**, **SPOKEN_BY**. |
| `*.bridge.json` | **`identities`** rows: canonical **`id`** + **`display_name`** for stable human-facing labels. |
| `bridge.json` (optional corpus registry) | Future corpus-level alias/topic merge (see backlog); not required for per-episode jobs. |

---

## 2. HTTP surfaces (read paths)

Analysis jobs may call (or batch equivalent offline):

- **CIL / GI query APIs** — cross-episode arcs already shaped for “who said what about topic X”
  (see `cil_queries` and SERVER_GUIDE CIL section).
- **GET `/api/search`** — semantic hits; **transcript** rows carry optional **`lifted`**:

```json
{
  "insight": {
    "id": "string",
    "text": "string",
    "grounded": true,
    "insight_type": "string | null",
    "position_hint": 0.42
  },
  "speaker": { "id": "person:…", "display_name": "string" },
  "topic": { "id": "topic:…", "display_name": "string" },
  "quote": {
    "timestamp_start_ms": 0,
    "timestamp_end_ms": 0
  }
}
```

- **`lift_stats`** on the same response — coverage signal for monitoring (`transcript_hits_returned`,
  `lift_applied`).

Optional corpus file **`cil_lift_overrides.json`** (see Semantic Search Guide) normalises ids
and char alignment for lift; analysers should treat **`lifted.speaker.id`** / **`topic.id`**
as **already alias-resolved** for display and join keys when overrides are in use.

---

## 3. Observability hooks (minimum)

Before shipping heavy ML analysis:

1. **Lift coverage** — from **`lift_stats`** and logs (per corpus, per query in logs if needed).
2. **Offset gate** — `verify-gil-chunk-offsets` verdict + overlap rate (CI or periodic on real corpora).
3. **`position_hint` density** — share of insights with non-null hint (existing GI quality tooling
  can be extended).

---

## 4. Eval / golden data (starter)

See **`tests/fixtures/cil_phase6_golden/README.md`** — placeholder for small frozen JSON bundles
(insight pairs + expected human labels) once Phase 6 RFC defines tasks.

---

## Revision history

| Date | Change |
| ---- | ------ |
| 2026-04-13 | Initial Phase 6 input contract + pointers to lift_stats and overrides |
