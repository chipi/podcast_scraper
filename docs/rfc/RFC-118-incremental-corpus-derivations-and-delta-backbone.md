# RFC-118: Incremental corpus derivations — the shared delta backbone

- **Status**: Draft — design only, no code landed. Supersedes the "incremental is a non-goal" premise of [RFC-088 §"Non-goals"](RFC-088-enrichment-layer-architecture.md).
- **Authors**: Marko Dragoljevic (chipi), Claude
- **Stakeholders**: Operator (sign-off), pipeline + enrichment + search maintainers
- **Related RFCs/ADRs**: [RFC-088](RFC-088-enrichment-layer-architecture.md) (enrichment layer; its incremental non-goal is the thing this reverses), [ADR-104](../adr/ADR-104-enrichment-layer-boundary-vs-kg-direct.md) (enrichment boundary)
- **Related documents**: `docs/wip/ARCH-CORPUS-DERIVATIONS-REFACTOR.md` (the prior two-mode analysis this formalises)

> Written after a single-episode reprocess on prod (2026-08-23) drove a **full-corpus** enrichment
> pass whose `topic_consensus` (DeBERTa NLI, pairwise) ran ~28 minutes and **timed out**
> (`run_summary` `efdca585`, `topic_consensus: timeout 1720768ms records=0`, run status `failed`).
> One changed episode should never cost 28 minutes of cross-episode NLI. The reindexer already
> solved this class of problem years ago; this RFC generalises its mechanism into a
> pipeline-wide seam.

---

## 1. Context

### 1.1 What went wrong

A one-episode repair correctly enqueued corpus enrichment (E4 fixed the enqueue). But the
corpus-scope enrichers re-derived over the **whole 678-episode corpus**:

| Enricher | Scope | This run |
| --- | --- | --- |
| `topic_consensus` (pairwise NLI) | corpus | **timeout, 0 records, ~28 min** |
| `topic_similarity` (pairwise cosine) | corpus | ok, 5,820 records, 8.8 MB |
| `topic_cooccurrence_corpus` | corpus | ok, 27,554 records |
| `temporal_velocity` | corpus | ok, 5,213 records |
| `grounding_rate`, `guest_coappearance`, `topic_theme_clusters` | corpus | ok |
| `insight_density`, `insight_sentiment` | episode | per-episode |

7 of 9 enrichers are corpus-scope. The executor discovers all bundles and passes the full
`all_bundles` to each (`enrichment/executor.py`; `enrichers/topic_similarity.py:58`
`for b in all_bundles`). The only optimisation today is a per-**enricher** staleness gate
(`enrichment/staleness.py` docstring: *"staleness decisions select enrichers, never episodes;
a 16-episode ingest triggers a full-corpus pass"*). There is no per-**episode** delta.

### 1.2 Why the old decision expired

RFC-088 declared incremental/delta enrichment a **non-goal** because "deterministic enrichers
complete in seconds." That was true — until the pairwise **ML** enrichers landed.
`topic_consensus` is O(pairs) NLI at ~ms/pair and `topic_similarity` is O(topics²) cosine;
those two are the entire wall-clock. The premise behind the non-goal is dead, so the non-goal
dies with it (see §8, RFC-088 amendment).

### 1.3 The reference we already own

The **reindexer** is the complete pattern this RFC copies:

- **Incremental:** a per-episode content fingerprint sidecar (`episode_fingerprints.json`) +
  `_fingerprint_skip_unchanged()` (`search/two_tier_indexer.py`); only changed episodes
  re-embed. Delta scope flows from `finalize` → `maybe_index_corpus()`
  (`workflow/orchestration.py`) → `index_corpus(rebuild=False)`.
- **Full rebuild:** `POST /api/index/rebuild?rebuild=true` (`routes/index_rebuild.py`) / CLI
  `--rebuild` → MVCC-clear + re-embed all, behind a per-corpus mutex (`CorpusRebuildGate`).
- **Staleness indicator:** `compute_index_staleness()` (`server/index_staleness.py`) →
  `reindex_recommended` + typed `reindex_reasons` (6 reasons), surfaced on
  `GET /api/index/stats`.

Enrichment has fragments (a `POST /api/jobs/enrichment` enqueue, `/status`, `/health`,
per-enricher re-enable; MCP `enrichment_*` tools) but **none** of the incremental delta, the
freshness indicator, or a first-class full-re-derive lever.

---

## 2. Decision

Build a **shared delta backbone** and make every corpus derivation consume it.

**The orchestrator owns the fingerprint.** It computes per-episode content fingerprints once
per run, produces a single `CorpusDelta`, and distributes it to every consumer in the chain —
indexing, enrichment, clustering, and future derivations. Consumers stop computing their own
notion of "what changed." This is a standardised seam: new derivations piggyback on one delta
definition instead of each reinventing staleness.

Two operating modes, everywhere:

1. **Incremental (default, the 1-episode repair path):** each derivation processes only the
   delta. Work is proportional to what changed, not to corpus size.
2. **Explicit full re-derive (operator-invoked):** a first-class lever per derivation, exposed
   on the operator UI and MCP, mirroring `POST /api/index/rebuild`. Used after a model/threshold
   change, or when the freshness indicator says the corpus has drifted.

Operator decision (2026-08-23): **share** the fingerprint (orchestrator-owned backbone, not
per-derivation sidecars) and surface staleness **per-enricher** with a rolled-up corpus flag.

---

## 3. The `CorpusDelta` contract (backbone)

Lives at a shared/orchestrator level — **not** inside enrichment.

```python
@dataclass(frozen=True)
class CorpusDelta:
    """What changed this run, plus the full corpus for cross-episode consumers.

    changed_ids : episode_ids whose content fingerprint differs from the prior
                  derivation run (or are new).
    removed_ids : episode_ids present in prior output but absent now.
    all_bundles : the FULL corpus — a pairwise consumer needs the (n-k) unchanged
                  episodes to form the k×(n-k) new pairs.
    """
    changed_ids: frozenset[str]
    removed_ids: frozenset[str]
    all_bundles: list[EpisodeArtifactBundle]
```

**Fingerprint source of truth.** Promote the reindexer's `_episode_fingerprint` into an
orchestrator-level function over each episode's derivation inputs (gi + kg content hash,
mtime-immune — reuse `staleness.input_fingerprint`). The orchestrator writes/reads one
per-episode fingerprint manifest and computes the delta:

- `changed_ids` = episodes whose current fp ≠ stored fp, plus episodes absent from the manifest.
- `removed_ids` = manifest episodes absent from the current bundle set.

**Reindex retrofit.** `maybe_index_corpus()` currently derives its own fingerprints. It is
retrofitted to consume the orchestrator's `CorpusDelta` (the index keeps its embedding-level
skip as an implementation detail, but "what changed" comes from the backbone). One definition
of changed, pipeline-wide.

**Distribution.** `finalize` computes `CorpusDelta` once and passes it to every consumer:
index, enrichment, clustering. `force=True` (explicit full) constructs a delta with
`changed_ids = ALL, removed_ids = ∅` and instructs consumers to ignore prior caches.

---

## 4. Enrichment as a consumer (WS-1)

### 4.1 Interface

```python
class EnricherManifest(...):
    supports_incremental: bool = False   # declared, not duck-typed — executor reports delta vs full

class Enricher(Protocol):
    async def enrich(self, *, bundle, corpus_root, all_bundles, config, ctx) -> EnricherResult: ...

    # optional — presence gated by supports_incremental
    async def enrich_incremental(
        self, *, delta: CorpusDelta, prior_output: dict[str, Any] | None,
        corpus_root: Path, config: dict[str, Any], ctx: RunContext,
    ) -> EnricherResult:
        """Recompute only what delta touches, merged into prior_output. Returns the
        SAME `data` shape as enrich() (the merged full result). MUST be
        output-identical to a full enrich() over the same corpus (§7 gate)."""
```

The executor (not the enricher) loads `prior_output` and passes the backbone's `delta`, so the
enricher stays a pure `(delta, prior) → data` function — trivially unit-testable side by side
with `enrich()`.

`enrich()` and `enrich_incremental()` share a private `_merge(candidate_pairs, prior, changed)`
kernel. **Full == delta with an empty cache** (`_merge(all_pairs, prior={}, changed=ALL)`), so
the two paths cannot diverge in scoring/filtering logic — only in what they reuse.

### 4.2 Per-enricher strategy (all 7 corpus-scope)

| Enricher | Class | Strategy |
| --- | --- | --- |
| `topic_consensus` | pairwise ML (the 28-min bill) | **delta-merge**, raw-score pair cache. **Do first.** |
| `topic_similarity` | pairwise embedding | **delta-merge**, vector cache (re-embed only changed topics). |
| `topic_cooccurrence_corpus` | additive aggregation | **skip-gate** (cheap full); counter-delta only if measured hot |
| `temporal_velocity` | window aggregation | **skip-gate** |
| `grounding_rate` | additive aggregation | **skip-gate** |
| `guest_coappearance` | pair counts + union-find rollup | **skip-gate** |
| `topic_theme_clusters` | clustering (capped linkage) | **skip-gate** on topic-set fingerprint |

Rule: an enricher needs **delta-merge** iff its cost is superlinear in corpus size **and**
decomposable by episode-pair (the 2 ML ones). Everything else gets a **skip-gate** only (full
recompute is seconds). No counter-deltas for aggregations in v1 — premature, and each adds
reconciliation surface for no measured win.

### 4.3 Pairwise cache (the hard case)

**`topic_consensus`** — cache **raw** scores (`cosine`, `contradiction`), not post-threshold
rows, keyed by `(insight_a_id, insight_b_id, model_version)` with the endpoint `episode_id`s
stored. A pair is valid iff **neither** endpoint episode is in `changed_ids ∪ removed_ids`.
Recompute only pairs with a changed endpoint; reuse the `(n-k)×(n-k)` block; re-apply the
threshold from cache (so a threshold change re-filters cheaply without re-scoring). For k=1 the
NLI bill drops from ~28 min to ~1/n of it.

**`topic_similarity`** — cache the **vectors**; re-embed only new/changed-label topics; the
O(topics²) cosine over cached vectors is cheap. Ship vector-cache first, measure, add
neighbour-list delta only if still slow.

**Invalidation:** any bump to `model_version` / manifest `version` discards the cache (the
explicit-full path). `topic_id`s are deterministic label slugs — the delta path fail-safes to
recompute (never stale-reuse) if a cached endpoint id no longer resolves.

---

## 5. Global surface + MCP (WS-3)

Mirror the reindex reference for enrichment, and expose triggers on MCP for **both** (reindex
is HTTP-only today):

- **Staleness indicator:** `compute_enrichment_staleness()` mirroring
  `compute_index_staleness()` → **per-enricher** freshness rows + a rolled-up
  `reenrich_recommended` corpus flag; typed reasons (`enricher_version_changed`,
  `gate_metrics_changed`, `corpus_artifacts_newer`, `last_run_failed_or_timed_out`). Surfaced
  on a new `GET /api/enrichment/stats` the operator UI polls.
- **Full-re-derive lever:** a first-class force/full flag on the enrichment job endpoint
  (alongside the existing `/api/corpus/topic-clusters/rebuild`), behind a per-corpus mutex like
  `CorpusRebuildGate`.
- **Operator UI:** one widget, two rows — `reindex_recommended` and `reenrich_recommended`,
  each with reasons and a global lever button (same pattern as the existing rebuild button).
- **MCP tools (new):** `reindex`, `reenrich` (write-gated like `enrichment_cancel` /
  `enrichment_re_enable`), and `corpus_status` reporting index + enrichment freshness. MCP and
  UI drive the identical endpoints.

---

## 6. Orchestrator alignment (WS-2, folded into the backbone)

`finalize` computes one `CorpusDelta` and threads it through **every** derivation:
index (retrofit), enrich (§4), cluster (C1 skip-gate). A single-episode add ⇒ all derivations
scoped to the delta, end to end. This is the "pipeline + orchestrator fully aligned for
incremental" requirement; it is not a separate workstream but the backbone's distribution edge.

---

## 7. Correctness gate — non-negotiable

A delta path that silently diverges from full is worse than slow-but-correct: it survives
review. Before **any** delta enricher merges:

- **Reconciliation test (Tier-2):** on a corpus fixture, run each `supports_incremental`
  enricher **both ways** — full `enrich()` vs `enrich_incremental()` with a synthetic
  1-episode delta against a prior built from the other n−1 — and assert **byte-identical**
  `data` (canonical sort). This is the property that makes delta safe to ship.
- **Slug-stability guard:** in the delta path, a reused cache endpoint whose `topic_id` no
  longer resolves to the same label is treated as changed (fail-safe recompute).
- **Periodic prod reconciliation:** schedule an explicit `--force` full re-derive on a cadence
  and diff against the incrementally-maintained output; alert on drift. The production analogue
  of the unit test — catches slug drift and cache corruption a fixture can't.

---

## 8. Phasing

| Phase | Task | Content |
| --- | --- | --- |
| **PR0 (foundational)** | #71 | `CorpusDelta` + orchestrator fingerprint authority + reindex retrofit + reconciliation harness + delta/full run-stats. No enricher behaviour change yet. |
| **PR1** | #67 | `topic_consensus` delta-merge + raw-score cache. Kills the 28-min failure. Gated by PR0's reconciliation test. |
| **PR2** | #67 | `topic_similarity` vector cache. |
| **PR3 (parallel)** | #69 | `compute_enrichment_staleness()`, `GET /api/enrichment/stats`, full-re-derive lever, operator UI widget, MCP `reindex`/`reenrich`/`corpus_status`. |
| **PR4 (deferred)** | #67 | counter-deltas for the aggregations — only if measured hot. |
| **Doc** | #70 | RFC-088 amendment (below) + strike the "delta-assign is later-only" line in the wip analysis. |

The whole set lands as one rebuild cycle (no interim half-measure — the stack rebuilds anyway).

## 9. Non-goals

- **Delta-clustering.** `topic_theme_clusters` / topic clustering stay skip-or-full; incremental
  linkage is out of scope.
- **Counter-deltas for the 5 deterministic aggregations** in v1 (skip-gate suffices; revisit on
  measured evidence).
- **Streaming/online enrichment.** This is batch delta, triggered per run.

## 10. Risks

- **Cache/full divergence** — mitigated by §7 (the reconciliation test is the gate, not a nicety).
- **Fingerprint coupling** — one backbone means one bug can mis-scope every derivation; the
  reconciliation cadence (§7) is the backstop, and `force` full is always the escape hatch.
- **Slug non-determinism** — fail-safe recompute (§4.3, §7) prevents stale reuse.

## 11. RFC-088 amendment

RFC-088's "incremental/delta updates are a non-goal for the initial release" is **invalidated**
for the pairwise ML enrichers (`topic_similarity`, `topic_consensus`), whose cost is not
"seconds." Incremental corpus enrichment is now a goal, delivered via the shared delta backbone
described here. The deterministic enrichers remain full-recompute behind a skip-gate, consistent
with the original reasoning.
