# RFC-072 — Consolidated out-of-scope and deferred work (from GitHub #524–#528)

**Purpose:** Single place for everything the **phase tracking issues** explicitly or implicitly
**exclude**, plus a short **gap read** toward the **full RFC-072 vision** (flagship use cases,
lift, and follow-on analysis). **Not** a commitment order; use for planning and RFC hygiene.

**Source issues (implementation tracking in RFC-072 front matter):**

- [#524](https://github.com/chipi/podcast_scraper/issues/524) — Phase 1 (slugifier)
- [#525](https://github.com/chipi/podcast_scraper/issues/525) — Phase 2 (ontology + viewer compat + GIL v1.1 fields)
- [#526](https://github.com/chipi/podcast_scraper/issues/526) — Phase 3 (`bridge.json` + viewer awareness)
- [#527](https://github.com/chipi/podcast_scraper/issues/527) — Phase 4 (cross-layer **API** only)
- [#528](https://github.com/chipi/podcast_scraper/issues/528) — Phase 5 (search **lift**; gated on offset verification)

---

## A. Explicit "not in this issue" (verbatim themes)

Only **#527** and **#528** have a **What is NOT in this issue** section. Combined themes:

1. **Phase 5 (semantic search lift)** — called out from #527 as separate (#528); #528 is the home issue.
2. **Phase 6 (analysis layer)** — separate RFC (contradiction / stance / position-change **detection**
   beyond chronological arcs); see RFC-072 Known Limitation §5 and roadmap Phase 6.
3. **Viewer UI for Position Tracker / Guest Brief** — #527 states API-only; dedicated viewer/UX work is
   **future** (PRD/RFC/UXS as needed).
4. **Corpus-level alias registry** — deferred per RFC-072 Known Limitations (name variation mitigation).
5. **Topic deduplication** — deferred per RFC-072 Known Limitations (topic slug fragmentation).
6. **FAISS index changes or rebuild** — out of scope for #528 (lift is query-time on existing index).
7. **Viewer UI for displaying lifted search results** — out of scope for #528 (separate issue).

---

## B. Implicit boundaries from Phase 1–3 issue bodies (no "NOT" section)

These are **in-scope exclusions** stated as **Scope** / **Why standalone**, not a titled out-of-scope block.

### #524 (Phase 1)

- **No** `gi.json` / `kg.json` schema or ID-prefix changes yet — slugifier module + tests + wiring to
  **replace local slug logic** only.
- **No** viewer changes.

### #525 (Phase 2)

- **Not** Phase 3 (`bridge.json`), Phase 4 (query API), or Phase 5 (search lift) — those are later issues.
- Viewer work here is **compatibility** for new IDs/types, not new flagship **product** surfaces for
  Position Tracker / Guest Brief (those remain deferred per #527).

### #526 (Phase 3)

- **Additive** bridge artifact and plumbing; does not replace the **GIL/KG separation** (ADR-052).
- Optional viewer improvements (dedup via bridge, node detail "appears in GI/KG") — **not** the full
  Position Tracker / Guest Brief **experience** (still deferred per #527).

---

## C. Deduplicated master list (planning backlog)

Unique deferred / excluded themes across #524–#528 and aligned RFC-072 **Known Limitations**:

| # | Theme | Where it shows up |
| - | ----- | ----------------- |
| 1 | Semantic search **lift** (Phase 5) | #527 excludes; #528 owns; blocked on **char offset alignment** (Known Limitation §1) |
| 2 | **Viewer** for **lifted** transcript hits | #528 excludes |
| 3 | **Viewer** for **Position Tracker / Guest Brief** (consume Phase 4 API) | #527 excludes |
| 4 | **Phase 6 analysis layer** (contradiction, stance, automated position-change) | #527, #528; RFC-072 §5 Known Limitation |
| 5 | **Corpus-level alias registry** (merge `person:*` variants) | #527; Known Limitation §2 |
| 6 | **Topic deduplication** / canonical topic merge across slugs | #527; Known Limitation §3 |
| 7 | **FAISS rebuild** / index schema change for lift | #528 excludes (not required for RFC design) |
| 8 | **Offset mapping layer** (if Quote vs chunk spaces diverge) | Known Limitation §1; prerequisite to safe lift |
| 9 | **`position_hint` gaps** when duration or quotes missing | Known Limitation §4 (quality, not a separate issue) |

---

## D. Gap assessment toward the **full RFC-072 vision**

**Vision anchor:** RFC-072 **Vision** + **Flagship use cases** (Position Tracker, Guest Brief,
follow-the-thread, controversy-oriented queries) + **Section 6** search lift + **Phase 6** interpretation.

| Pillar | Vision intent | Typical gap (today) | Direction |
| ------ | ------------- | ------------------- | --------- |
| **Identity** | Stable `person:` / `org:` / `topic:` | Name/topic fragmentation | Extraction quality; later **alias registry** (#5) |
| **Join** | Auditable cross-layer join | Per-episode bridge only | **Corpus-level** alias/topic merge (#5–6) optional Postgres consumer later |
| **Query (data)** | Cross-episode arcs, briefs, timelines | **HTTP API** may land without **SPA** surfaces | Viewer/UXS/PRD issues for CIL panels (#3) |
| **Search UX** | Chunks **read as** grounded Insights | **Lift** + **viewer** for `lifted` | #528 + follow-up UI (#2); **offset gate** (#1, #8) |
| **Wow factor** | Automated contradiction / shift | Not in RFC-072 | **Phase 6 RFC** + eval (#4) |
| **Trust** | Correct attribution | Offset / normalisation drift | **verify-gil-chunk-offsets** + mapping if needed (#8) |

**Ordering heuristic (from RFC + issues):**

1. Phases **1 → 3** establish IDs, artifacts, bridge.
2. **Offset verification** (Step 1 of #528 / Known Limitation §1) before treating lift as production-safe.
3. Phase **4** API can ship **without** viewer; product value grows when **#3** is addressed.
4. **#5–6** improve recall of cross-episode queries; not blockers for first vertical slice.
5. **#4** is a **separate** product/RFC track once bridge + queries are exercised on real corpora.

---

## E. Related canonical docs

- RFC-072: `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` (Known Limitations, roadmap)
- Semantic search lift + offset gate: `docs/guides/SEMANTIC_SEARCH_GUIDE.md` (chunk-to-Insight lift section)
- Cross-layer map: `docs/guides/GIL_KG_CIL_CROSS_LAYER.md`

---

## Revision history

| Date | Change |
| ---- | ------ |
| 2026-04-15 | Initial consolidation from GitHub #524–#528 bodies + RFC-072 Known Limitations |
