# ADR-136: Per-episode context digest (`.context.json`) — a reprocess-free consolidated content surface

- **Status**: Accepted
- **Date**: 2026-07-28
- **Authors**: Marko Dragoljevic
- **Related issues**: v2.4 arc (GI/KG optimizations), #1220 (KG Voice node), ADR-135 (route-and-tag + Voice)
- **Related work**: labeling-output `exposed` metric (36af801e); sidecar-completeness directive
- **Design source**: this ADR (operator proposal, 2026-07-28)

## Context & Problem Statement

Downstream consumers — search, **future translation**, and other batch processes — need a single,
per-episode surface that answers **"what is this episode about?"** without re-parsing the graph or
reprocessing the episode. Today that content is spread across three artifacts, each with a different
charter, and none is a flat, denormalized digest:

| Artifact | Charter | Holds |
| --- | --- | --- |
| `.manifest.json` | **How it was processed** (provenance, RFC-109/ADR-132) | per-stage what-ran, quality metrics, cost, versions (incl. `naming.exposed`) |
| `.metadata.json` | **What the episode IS** (product record, ADR-133) | feed/episode facts, `speakers`, `detected_hosts/guests`, `normalized_entities` |
| `.gi.json` / `.kg.json` | The **normalized graph** | Person / Organization / Topic / Insight / Quote nodes + edges |

The gap was proven concretely: extracting the clean named-vs-Voice speaker rate required opening
`.gi.json` and walking `SPOKEN_BY` edges, because no consolidated per-episode content record existed
(Javier hit the same wall extracting from sidecars). The manifest is the wrong home — it is
**provenance** ("how it was made"), and folding denormalized content into it muddies both concerns.

## Decision

Introduce a new per-episode sidecar, **`.context.json`**: a flat, denormalized, reprocess-free
digest of the episode's content, written **after GI/KG complete** by rolling up what those stages
already produced. It is a **cache/view** — the graph artifacts remain the source of truth; the
digest is a convenience surface for consumers that must not re-derive it.

### Separation of concerns (why a new file, not an extension)

- **Provenance** (`.manifest.json`) stays "how it was processed" — this is where **stage-output
  metrics** go (the `exposed` metric; GI/KG counts). NOT content.
- **Content digest** (`.context.json`) is "what the episode is about" — denormalized for reuse.
  A dedicated single-purpose file gives search/translation a clean, independently-versioned contract
  and lets corpus-level aggregates roll up from the per-episode digests.

Extending `.metadata.json` was considered and rejected: it already carries partial entity data
(`normalized_entities`, `detected_hosts/guests`), but it is the product record, and growing it into a
mixed content-digest bag blurs its charter and its schema. A separate file keeps each artifact honest.

## Scope — Phase 1 (this ADR): consolidate ONLY what we already have (zero new LLM)

Phase 1 is a **deterministic denormalization** of existing outputs — no new model calls, no new cost,
so it can be produced for every episode (and backfilled by a migration over existing corpora). The
field set below is **illustrative**, not frozen — the principle is "everything relevant we already
produce, in one flat place":

- **Basic** (from `.metadata.json` / feed): title, show/feed, published date, duration, `episode_id`,
  language, hosts, guests.
- **Summary** (from the summary stage, already generated): the episode summary text.
- **Entities — denormalized from the GI/KG graph** (no new extraction):
  - `people` — `Person` nodes, **excluding bare-speaker `Voice` nodes** (#1220): resolved humans only.
  - `companies` — `Organization` nodes.
  - `topics` — `Topic` nodes.
  - `voices` — the unresolved-speaker split **with the specific labels** (not just a count), reusing
    the labeling `exposed` shape so consumers get the actual context:
    `{"total": N, "unknown": ["SPEAKER_03", …], "unidentified": ["SPEAKER_04", …]}`
    — `unknown` = a real person we FAILED to name (defect), `unidentified` = nobody ever names them.
    This states the clean speaker picture directly, closing the gap that motivated this ADR.
- **Provenance pointer**: source artifact `schema_version`s (gi/kg) + a `context` `schema_version`, so
  a consumer knows what the digest was built from and whether it is stale.

## Non-goals / deferred to Phase 2+ (need NEW extraction — designed separately, costed)

These require a new LLM pass and are explicitly **out of Phase 1**:

- **glossary / terminology** (e.g. `SFT → Supervised Fine-Tuning`) — highest value for **translation**
  (a term map + do-not-translate list); the natural first Phase-2 addition.
- **concepts** as a type distinct from `topics` (today they overlap `Topic`).
- **jokes**, and any other net-new derived content.

Each Phase-2 addition is a new extraction stage with its own cost, gate, and manifest provenance —
not smuggled into the free Phase-1 rollup.

## Translation forward-look (why this shape)

Phase 1 already gives translation the two things it most needs from context: the **verbatim entity
list** (people/companies that must not be translated) and the summary for global context. Phase 2's
glossary completes the "translation context pack." Building the digest now, entity-first, is what
makes a future translation pass tractable without reprocessing.

## Consequences

- **+** One flat, reprocess-free surface for search/translation/other consumers; the clean
  speaker picture is finally readable per-episode without walking the graph.
- **+** Deterministic + free in Phase 1 → backfillable across the existing corpus by migration.
- **−** One more per-episode file to write and version; it is a denormalized cache, so it can drift
  if the graph is regenerated without rebuilding the digest — mitigated by the provenance pointer and
  by writing it in the same run, right after KG.
- **Contract**: consumers treat `.context.json` as read-only cache; the graph remains source of truth.

## Alternatives considered

1. **Extend `.metadata.json`** — rejected (mixes product-record and content-digest concerns).
2. **Put content in `.manifest.json`** — rejected (manifest is provenance, not content).
3. **Corpus-level only** — rejected (inherently per-episode; corpus aggregates roll up from per-episode
   digests).

## Resolved decisions (2026-07-28)

- **Write point**: a **new post-KG consolidation step** — runs after KG finishes and rolls up
  gi/kg/metadata/summary. Independently re-runnable; the backfill reuses the same code path. (Not
  folded into the KG stage tail — keeps the digest decoupled from KG.)
- **Backfill**: yes — a deterministic migration that builds `.context.json` from the existing
  `.gi.json`/`.kg.json`/`.metadata.json` of the **latest `prod-v2.3-turbo` corpus** (no reprocess,
  no LLM). Historical corpora are out of scope; the current turbo corpus is the target.
- **`voices` shape**: the **split with specific labels** (see the entities list above), not a bare
  count — `{total, unknown[], unidentified[]}` — so the digest carries the actual unresolved-speaker
  context, reusing the `naming.exposed` classification.

## Still open (non-blocking, settle during implementation)

- Final field names (illustrative here, per "don't get hung up on specifics").
- Whether `people` entries carry role (host/guest) inline or stay a flat name list.
