# RFC-113: Graph-Aware PKM Export (Obsidian)

- **Status**: Draft
- **Authors**: Marko, Claude (Opus 4.8), advisor (Fable 5)
- **Stakeholders**: Core team, platform users (PKM/second-brain), Consumer App
- **Related PRDs**:
  - `docs/prd/PRD-046-delivery-and-curation.md` (the arc this extends; the export was its named next arc)
  - `docs/prd/PRD-041-consolidation.md` (the personal corpus this serializes)
- **Related RFCs**:
  - `docs/rfc/RFC-111-curation-surfaces.md` (§3 froze the highlight↔entity serialization shape for this)
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` (canonical ids → wikilink targets)
  - `docs/rfc/RFC-114-personal-corpus.md` (**Phase 1**: the corpus + the revision counter this exports over)

## Abstract

The podcast-Readwise category (Snipd, Podwise) is a **feeder**: it exports **flat** highlights into
the user's PKM, where the user does the connecting. closelistening builds the connected graph. This
RFC exports it **as a graph**: each highlight becomes a Markdown note that **wikilinks** to
`[[Entity]]`, `[[Guest]]`, `[[Topic]]`, `[[Episode]]` notes, so the user's **Obsidian** vault mirrors
their personal knowledge graph — not a flat dump. v1 is **Obsidian-only**, a **one-way incremental
pull** driven by RFC-114 Phase 1's **revision counter + change log** (so it can express *deletions*,
not just additions). Notion is explicitly a **separate later slice** (different mechanism — OAuth
push). The highlight payload shape is already frozen (RFC-111 §3); this RFC adds the **vault
contract** (layout, filenames, frontmatter, link syntax, tombstones) + the emitter.

## Problem Statement

Users live in Obsidian. Our value is trapped on our surface (or exportable only as flat Markdown,
PRD-040). Competitors feed vaults with disconnected atoms; we can emit the *connections* because every
highlight carries canonical entity ids (#1419) over the shared KG (RFC-072). The gaps: (1) an emitter
that turns that into a linked note graph; (2) an **incremental** mechanism that handles deletions and
retroactive membership (a favorite joining an old episode) — which a naive "changed-since-timestamp"
cannot; (3) stable filenames that survive label renames and entity merges.

## Goals

1. **Graph-carrying export**: highlight notes + entity/guest/topic/episode notes, wikilinked.
2. **Obsidian first** (Markdown + `[[wikilinks]]`). Notion is a named later slice, not v1.
3. **Incremental, one-way pull** over RFC-114's change log: emit adds **and** tombstones; deterministic,
   re-runnable, no duplication. A **full re-export** is always valid (id-keyed overwrite) as the
   fallback + the v1-if-114-slips path.
4. **Reuse the frozen RFC-111 §3 shape** for the highlight payload; define the **vault contract** here.
5. **Extractive, no LLM** (D6).

## Constraints & Assumptions

- **Depends on RFC-114 Phase 1** — the export scopes to `experienced ∪ saved` (labelled) and its
  cursor **is** the `corpus_revision`; its deletions come from the change log. This RFC does **not**
  invent its own per-surface corpus derivation (that's the drift RFC-114 exists to kill).
- **One-way v1** — platform → vault. Conflict-free: we own the emitted namespace (`closelistening/`);
  user edits outside it are untouched.
- **Bridge-only audio** — notes carry transcript quotes + deep-links, never audio.

## Design & Implementation

### 1. Vault contract (the public artifact — appendix-level detail)

**Layout** (all under a `closelistening/` root so re-export never touches the user's own notes):
- `closelistening/Highlights/<highlight_id>.md`
- `closelistening/People/<person_id>.md`, `Topics/<topic_id>.md`, `Episodes/<episode_slug>.md`

**Filenames are canonical ids, never display labels** (resolves the earlier draft's contradiction).
Obsidian resolves `[[wikilinks]]` by filename, so id-keyed names survive **label renames** and
**alias merges** (RFC-072 KL2 is future — labels *will* move). Human-readable labels live in
frontmatter `aliases:` and in the link's display text.

**Link syntax**: `[[person_ab12|Ada Lovelace]]` (id target, label shown).

**Highlight note** (frontmatter + body):
```markdown
---
id: h_1a2b3c
episode: acquired-nvidia
t_ms: 3921000
entities: [person_ab12, topic_scaling]
source: user            # or "auto" (GI editor's-pick)
aliases: ["“The bottleneck was never compute…”"]
---
> “The bottleneck was never compute; it was our willingness to throw away a working model.”
— [[Episodes/acquired-nvidia|NVIDIA: The Machine…]] · [▶ 1:05:21](https://…/player/acquired-nvidia?t=3921)
Discusses [[People/person_ab12|Jensen Huang]] · [[Topics/topic_scaling|Scaling Laws]]
```
**Entity note** is thin (id, label, source) — the graph emerges from backlinks, not duplicated body.

### 2. The emitter + incremental cursor

- `GET /api/app/export?format=obsidian&since=<revision>` → a bundle (zip of Markdown + a `manifest.json`).
- **`since` = RFC-114 `corpus_revision`.** The emitter reads `GET /api/app/corpus/changes?since=` →
  the adds + **tombstones**; it emits/overwrites notes for adds (id-keyed ⇒ idempotent) and records
  removed ids in the manifest so the client deletes those vault files.
- **`manifest.json`**: `{ from_revision, to_revision, written: [paths], removed: [paths] }` — the
  client applies adds then deletions, giving a vault that tracks the corpus exactly.
- **Full export** = `since=0`: emits the whole `closelistening/` tree, valid any time (the fallback,
  and the v1 shape if 114's change log isn't ready — ship full-only, add the delta when the counter is).
- **Entity-id merges** (RFC-072 KL2, future): a merge emits a tombstone for the losing id + rewrites
  referring highlight notes in the next delta — same tombstone primitive, no special case.

### 3. Notion (separate later slice — NOT v1)

Notion is architecturally different: a zip of Markdown can't carry Notion relations, and Notion has
**no client-supplied stable page ids**, so idempotency needs a persisted `entity_id → notion_page_id`
map + an **OAuth push integration** into the user's workspace. That is a different delivery mechanism
(push, not pull) with its own auth + state — a separate RFC/slice, explicitly out of this v1.

## Key Decisions

1. **Wikilinked, id-keyed entity notes** — the differentiator; survives renames + merges (labels in
   frontmatter/display only).
2. **Incremental over RFC-114's change log (adds + tombstones)** — deletions are expressible; full
   re-export is the always-valid fallback.
3. **Obsidian-only v1; Notion is a separate push slice** — not "same shape, second target".
4. **Own a `closelistening/` namespace** — one-way, conflict-free with the user's own notes.

## Alternatives Considered

- **Flat Markdown export (PRD-040 / Snipd shape).** The thing we differentiate against; kept as the
  existing simple export, not this.
- **Timestamp cursor.** Rejected — can't express deletions or retroactive membership (RFC-114 §Alts).
- **Live Obsidian plugin (Snipd-style).** Heavier + platform-specific; the pull/bundle works for any
  vault. A plugin can wrap this API later.
- **Two-way sync in v1.** Rejected — vault-side conflict resolution is a large separate arc.

## Testing Strategy

- Unit: highlight-note render (id-keyed links, deep-link, frontmatter, aliases); entity-note dedup;
  the `since` delta emits adds **and** tombstones; a deleted highlight → a `removed` manifest entry;
  a label rename → same file, updated alias (no new file).
- Integration: export a fixture corpus → a valid vault; re-export at the same revision → empty delta;
  a changed/removed highlight → exactly one written/removed path; full export (`since=0`) round-trips.
- No-audio assertion on every emitted note (bridge-only).

## Rollout & Monitoring

- Ship Obsidian export behind a flag (full-export first if RFC-114's change log trails; delta when
  ready). Read-only from the platform side; disabling the endpoint is the rollback.
- Monitor: exports, delta sizes, % highlights with `entities` (span highlights can be ref-less per
  RFC-111 §3 — those notes are flat; carry that metric so we know the flat-fraction we're shipping).

## Open Questions

- Frontmatter schema stability if two-way sync ever lands.
- Do we export `saved` (favorited-unheard) episodes' entities, or `experienced` only? (Default:
  `experienced`; `saved` behind a toggle, labelled.)
- Orgs: the bridge carries MENTIONS_ORG but `AppEntityRef` is person/topic only — orgs absent from the
  vault v1; confirm intended.

## References

- RFC-111 §3 (frozen highlight shape), RFC-114 Phase 1 (corpus + revision counter), RFC-072 (canonical
  ids + KL2 merge), PRD-046 (arc), PRD-041 (personal corpus).
