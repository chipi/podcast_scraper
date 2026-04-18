# RFC-073: Enrichment Layer Architecture

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Core team
- **Target Release**: TBD
- **Related PRDs**:
  - `docs/prd/PRD-026-topic-entity-view.md` -- first read-side consumer of
    enrichment outputs (topic co-occurrence, temporal velocity)
  - `docs/prd/PRD-027-enriched-search.md` -- query-time enrichment extending
    semantic search (Phase 4 extension of this RFC)
  - `docs/prd/PRD-017-grounded-insight-layer.md`
  - `docs/prd/PRD-019-knowledge-graph-layer.md`
- **Related ADRs**:
  - `docs/adr/ADR-004-flat-filesystem-archive-layout.md`
  - `docs/adr/ADR-020-protocol-based-provider-discovery.md`
  - `docs/adr/ADR-024-unified-provider-pattern.md`
  - `docs/adr/ADR-026-per-capability-provider-selection.md`
  - `docs/adr/ADR-051-per-episode-json-artifacts-with-logical-union.md`
  - `docs/adr/ADR-052-separate-gil-and-kg-artifact-layers.md`
- **Related RFCs**:
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md`
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md`
  - `docs/rfc/RFC-055-knowledge-graph-layer-core.md`
  - `docs/rfc/RFC-061-semantic-corpus-search.md`
  (Future analysis layer consumes `nli_contradiction` enricher output as the base
  layer for query-time analysis.)
- **Related Documents**:
  - `docs/architecture/gi/ontology.md`
  - `docs/architecture/kg/ontology.md`

---

## Abstract

The pipeline today produces three core artifact layers per episode: GIL (`*.gi.json`),
KG (`*.kg.json`), and the bridge (`*.bridge.json` -- RFC-072). These are the main
dish -- grounded, verbatim, trustworthy. They do not change.

This RFC introduces an **Enrichment Layer** -- a fourth artifact tier that sits on top
of the core layers, adds derived signals, computed annotations, and cross-episode
intelligence, and is always optional. Enrichers are the salt and pepper: they sharpen
what is already there, make it more navigable and more useful, but never alter the
underlying ingredients. A missing or failed enricher degrades the experience gracefully;
it never corrupts the core.

**Clarification — "do not change":** enrichers must never **write** core artifacts.
Operators and automation can still **replace** `*.gi.json`, `*.kg.json`, and
`*.bridge.json` by re-running the core pipeline. After such a rebuild, canonical ids,
cluster membership, and sibling paths may differ from an earlier run; enrichment outputs
under `metadata/enrichments/` (or corpus-scope enrichments) may be **stale** until
enrichers run again. Read paths (CIL HTTP, corpus library, viewer) always reflect **what
is on disk now**. See [RFC-072 — Operational note: re-pipeline, enrichment, and read-path
stance](RFC-072-canonical-identity-layer-cross-layer-bridge.md#operational-note-re-pipeline-enrichment-and-read-path-stance).

The RFC defines what an enricher is, what it is not, the protocol every enricher must
implement, the output directory structure, the registry mechanism, and two concrete
enricher implementations that demonstrate the pattern. These enrichers directly serve
the use cases defined in PRD-026 (Topic Entity View) and PRD-027 (Enriched Search).

**Filesystem alignment note:** This RFC follows the existing flat layout defined in
ADR-004 and ADR-051. Core artifacts are sibling files in `metadata/` sharing a common
stem (e.g. `0001 - episode_title.gi.json`). Enrichment outputs live in a dedicated
`metadata/enrichments/` subdirectory. There are no per-episode directories.

---

## Problem Statement

The core artifact layers (GIL, KG, bridge) are production-ready after RFC-072. They
answer the question "what happened in this episode and who said what." But several
high-value signals are derivable from the core artifacts that the core pipeline does not
and should not produce:

- Which topics in the corpus are semantically related to each other, even if they never
  appeared in the same episode?
- How fast is a topic growing or declining across the corpus over time?
- What is the grounding quality rate for a given speaker across all their appearances?
- Do two Insights from different speakers on the same topic contradict each other?

These signals are **derived** -- they require reading existing artifacts, not re-running
extraction. They are **additive** -- they annotate what exists without replacing it.
They are **optional** -- the core use cases (search, browse, transcript access) work
without them. And they are **heterogeneous** -- some require only arithmetic, some
require embeddings, some require ML models, some require LLM calls.

There is currently no architectural home for this class of computation. Without a
defined pattern, enrichers will either be bolted onto the core pipeline (coupling that
should not exist) or built ad hoc per feature (no reuse, no consistency, no testability).

This RFC defines that architectural home.

---

## What an Enricher Is

An enricher is a **self-contained computation unit** that:

1. **Reads** one or more core artifacts (`*.gi.json`, `*.kg.json`, `*.bridge.json`,
   transcript, or corpus-level index) as input.
2. **Computes** a derived signal, annotation, or aggregation.
3. **Writes** its output to a dedicated file in `metadata/enrichments/`
   (episode-scope) or `enrichments/` under the corpus root (corpus-scope).
4. **Declares** its inputs, outputs, version, and dependencies in a manifest.
5. **Never modifies** core artifacts. Ever.
6. **Never blocks** the core pipeline. Enrichers run after core artifacts are written,
   in a separate enrichment pass.
7. **Fails gracefully** -- a failed enricher logs its failure and is skipped. The
   absence of an enrichment output is a valid state, not an error condition for
   downstream consumers.

---

## What an Enricher Is NOT

This section is as important as the definition above.

**An enricher is not a pipeline stage.** The core pipeline stages (transcription,
speaker detection, summarisation, GIL extraction, KG extraction, bridge emission) are
sequential, blocking, and required. Enrichers are none of these things. If you find
yourself wanting an enricher to be blocking or required, that signal belongs in the core
pipeline, not the enrichment layer.

**An enricher is not a replacement for better extraction.** If a signal should be
present in `*.gi.json` or `*.kg.json` -- if it is part of the grounding contract or
the entity model -- it belongs in the core pipeline as an extraction improvement, not
in an enricher. Enrichers operate on what exists; they do not patch what should have
been extracted correctly in the first place.

**An enricher is not a data source.** Enrichers derive signals from the corpus.
They do not ingest new external data (that is a provider / content adapter concern),
they do not call external APIs at enrichment time (exception: LLM enrichers, which
are explicitly flagged as such and require opt-in config), and they do not introduce
facts that are not traceable to the core artifacts. LLM tier outputs are synthetic
narration over derived/core inputs; they carry `derived: true` and non-authoritative
UX, and should include citation IDs back into core artifacts where possible.

**An enricher is not authoritative.** Enricher outputs are always marked with
`derived: true` and carry their own confidence and provenance. They are never presented
to the user with the same trust weight as GIL-grounded Insights. A GIL Insight is
backed by a verbatim quote. An enricher annotation is backed by a computation over
GIL Insights. These are different epistemic claims and the system must keep them
distinct.

**An enricher is not a feature.** An enricher produces data. A feature consumes that
data and presents it to the user. The Topic Entity View (a feature) may consume
topic velocity enricher output (data), but the enricher does not know or care about
the Topic Entity View. This separation keeps enrichers testable, reusable, and
composable.

---

## Enricher Protocol

Every enricher implements the following Python protocol. The protocol uses PEP 544
structural typing with `@runtime_checkable`, consistent with the existing provider
protocols (ADR-020). Unlike the multi-capability provider pattern (ADR-024), enrichers
are single-method, stateless units with no lifecycle (no `initialize` / `cleanup`).
This justifies a simpler module-attribute discovery rather than per-capability factory
functions.

```python
# podcast_scraper/enrichment/protocol.py

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable


class EnricherScope(Enum):
    EPISODE = "episode"    # runs once per episode, reads episode artifacts
    CORPUS = "corpus"      # runs once per corpus, reads all episode artifacts


class EnricherTier(Enum):
    DETERMINISTIC = "deterministic"  # pure arithmetic / graph traversal, no model
    EMBEDDING = "embedding"          # uses vector embeddings (FAISS)
    ML = "ml"                        # uses a local ML model (NLI, classifier)
    LLM = "llm"                      # calls an LLM API -- requires explicit opt-in


@dataclass(frozen=True)
class EpisodeArtifactBundle:
    """Resolved paths to core artifacts for one episode."""

    metadata_path: Path       # e.g. metadata/0001 - ep.metadata.json
    gi_path: Path | None      # e.g. metadata/0001 - ep.gi.json
    kg_path: Path | None      # e.g. metadata/0001 - ep.kg.json
    bridge_path: Path | None  # e.g. metadata/0001 - ep.bridge.json
    episode_id: str           # raw GUID or sha256:... (no "episode:" prefix)
    stem: str                 # shared filename stem, e.g. "0001 - ep"


@dataclass(frozen=True)
class EnricherManifest:
    id: str                   # e.g. "topic_cooccurrence"
    version: str              # semver e.g. "1.0.0"
    scope: EnricherScope
    tier: EnricherTier
    reads: list[str]          # artifact suffixes consumed e.g. [".kg.json", ".bridge.json"]
    writes: str               # output filename e.g. "topic_cooccurrence.json"
    description: str
    requires_opt_in: bool     # True for LLM tier enrichers


@runtime_checkable
class Enricher(Protocol):

    @property
    def manifest(self) -> EnricherManifest:
        """Declare inputs, outputs, tier, and scope."""
        ...

    def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,  # set for EPISODE scope
        corpus_root: Path,                     # output root (contains metadata/)
        all_bundles: list[EpisodeArtifactBundle] | None,  # set for CORPUS scope
        config: dict,                          # enricher-specific config from YAML
    ) -> dict:
        """
        Compute enrichment. Return the output dict to be written.
        Must not raise -- catch exceptions internally and return
        a result with status: "failed" and error details.
        Never modify any core artifact file.

        Note: the runner wraps each call in try/except as a safety
        net, but enrichers should still handle their own errors.
        """
        ...
```

**Key design choices in the protocol:**

- **`EpisodeArtifactBundle`** replaces raw directory paths. The runner resolves
  artifact paths using existing `discover_metadata_files` and `_determine_*_path`
  helpers, so enrichers never guess at filename stems or directory layout.
- **`episode_id`** in the bundle is the raw identifier (GUID or `sha256:...`),
  matching the root-level `episode_id` field in `*.gi.json`, `*.kg.json`, and
  `*.bridge.json`. The `episode:` prefix is only used on Episode graph node IDs,
  not at the artifact root level.
- **Keyword-only arguments** prevent positional confusion between episode and corpus
  scope parameters.

**Output contract -- every enricher output MUST include:**

Episode-scope enricher (has `episode_id`):

```json
{
  "enricher_id": "topic_cooccurrence",
  "enricher_version": "1.0.0",
  "schema_version": "1.0",
  "episode_id": "my-podcast-guid-abc123",
  "derived": true,
  "computed_at": "2026-04-13T10:00:00Z",
  "status": "ok",
  "data": { }
}
```

Corpus-scope enricher (`episode_id` omitted -- spans all episodes):

```json
{
  "enricher_id": "temporal_velocity",
  "enricher_version": "1.0.0",
  "schema_version": "1.0",
  "derived": true,
  "computed_at": "2026-04-13T10:00:00Z",
  "status": "ok",
  "data": { }
}
```

If the enricher fails (episode-scope example; corpus-scope omits `episode_id`):

```json
{
  "enricher_id": "topic_cooccurrence",
  "enricher_version": "1.0.0",
  "schema_version": "1.0",
  "episode_id": "my-podcast-guid-abc123",
  "derived": true,
  "computed_at": "2026-04-13T10:00:00Z",
  "status": "failed",
  "error": "KeyError: 'nodes' -- artifact may be malformed",
  "data": null
}
```

The `derived: true` field is non-negotiable. Every consumer of enricher output must
be able to distinguish derived annotations from grounded core artifacts. This is the
enrichment layer's equivalent of the GIL grounding contract.

A JSON schema for the enricher output envelope will be added alongside the first
enricher implementation, following the precedent of `gi.schema.json` and
`kg.schema.json`.

---

## Directory Structure

The enrichment layer follows the existing flat archive layout (ADR-004, ADR-051).
Core artifacts are sibling files in `metadata/` sharing a common stem derived from
`build_whisper_output_name`. Enrichment outputs live in a dedicated `enrichments/`
subdirectory under `metadata/` (episode-scope) or under the corpus root
(corpus-scope).

**Single-feed layout:**

```text
output/                                    # corpus root (SERVE_OUTPUT_DIR)
  transcripts/
    0001 - episode_title.txt               # transcript
    0001 - episode_title.segments.json     # optional segment timestamps
  metadata/
    0001 - episode_title.metadata.json     # core -- never modified
    0001 - episode_title.gi.json           # core -- never modified
    0001 - episode_title.kg.json           # core -- never modified
    0001 - episode_title.bridge.json       # core -- never modified
    enrichments/                           # episode-scope enricher outputs
      0001 - episode_title.topic_cooccurrence.json
      0001 - episode_title.insight_density.json
      ...
  enrichments/                             # corpus-scope enricher outputs
    temporal_velocity.json
    grounding_rate.json
    topic_similarity.json
    guest_coappearance.json
    ...
```

**Multi-feed layout** (each feed under `feeds/rss_{host}_{hash}/`):

```text
output/
  feeds/
    rss_example_com_a1b2/
      metadata/
        0001 - ep.metadata.json
        0001 - ep.gi.json
        ...
        enrichments/
          0001 - ep.topic_cooccurrence.json
      transcripts/
        ...
    rss_other_com_c3d4/
      ...
  enrichments/                             # corpus-scope (spans all feeds)
    temporal_velocity.json
    ...
```

**Naming convention for episode-scope enrichments:**

Episode-scope enrichment files use the episode stem + enricher output name:
`{stem}.{enricher_writes}`. For example, if the stem is `0001 - episode_title`
and the enricher writes `topic_cooccurrence.json`, the output file is
`0001 - episode_title.topic_cooccurrence.json`.

**Invariants:**

- Core artifacts are never in the `enrichments/` directory.
- Enrichment artifacts are never outside the `enrichments/` directory.
- Episode-scope enrichers write to `metadata/enrichments/`.
- Corpus-scope enrichers write to `{corpus_root}/enrichments/`.
- `discover_metadata_files` (existing) is unaffected -- it globs `*.metadata.json`
  under `metadata/`, not under `metadata/enrichments/`.
- In multi-feed layouts, episode-scope enrichments stay co-located with their feed's
  `metadata/enrichments/` (the runner writes to `bundle.metadata_path.parent /
  "enrichments"`). Corpus-scope enrichments always write to the top-level
  `{corpus_root}/enrichments/`, spanning all feeds.

---

## Enricher Registry

Enrichers are registered via a YAML config block and discovered at runtime. No
hardcoded enricher lists in the pipeline.

```yaml
# config.yaml

enrichment:
  enabled: true
  pass: after_core              # always -- enrichers never run inline
  enrichers:
    # --- deterministic (on by default) ---
    - id: topic_cooccurrence
      enabled: true
    - id: topic_cooccurrence_corpus
      enabled: true
    - id: temporal_velocity
      enabled: true
    - id: grounding_rate
      enabled: true
    - id: guest_coappearance
      enabled: true
    - id: insight_density
      enabled: true
    # --- embedding / ML (off by default) ---
    - id: topic_similarity
      enabled: false            # embedding tier -- opt-in
    - id: nli_contradiction
      enabled: false            # ML tier -- opt-in
    # --- LLM (off, double opt-in) ---
    - id: llm_position_narration
      enabled: false            # LLM tier -- requires opt_in: true
      # opt_in: true            # uncomment to activate
```

**Config integration:** The `enrichment` block maps to a new `EnrichmentConfig`
Pydantic model nested in the existing `Config` class, following the same pattern as
other config sections. Enricher IDs are validated against `_BUILTIN_ENRICHERS` keys.
Invalid IDs are config errors, not silent skips.

**Registry implementation:**

```python
# podcast_scraper/enrichment/registry.py

import logging
from importlib import import_module
from typing import Iterator

from .protocol import Enricher

logger = logging.getLogger(__name__)

_BUILTIN_ENRICHERS: dict[str, str] = {
    "topic_cooccurrence":
        "podcast_scraper.enrichment.builtin.topic_cooccurrence",
    "topic_cooccurrence_corpus":
        "podcast_scraper.enrichment.builtin.topic_cooccurrence_corpus",
    "temporal_velocity":
        "podcast_scraper.enrichment.builtin.temporal_velocity",
    "nli_contradiction":
        "podcast_scraper.enrichment.builtin.nli_contradiction",
    "grounding_rate":
        "podcast_scraper.enrichment.builtin.grounding_rate",
    "guest_coappearance":
        "podcast_scraper.enrichment.builtin.guest_coappearance",
    "insight_density":
        "podcast_scraper.enrichment.builtin.insight_density",
}


def load_enricher(enricher_id: str) -> Enricher:
    module_path = _BUILTIN_ENRICHERS[enricher_id]
    module = import_module(module_path)
    return module.enricher_instance


def resolve_enrichers(config: dict) -> Iterator[Enricher]:
    for entry in config.get("enrichment", {}).get("enrichers", []):
        if not entry.get("enabled", False):
            continue
        enricher = load_enricher(entry["id"])
        if enricher.manifest.requires_opt_in and not entry.get("opt_in", False):
            logger.warning(
                "Enricher %s requires opt_in -- skipped", enricher.manifest.id
            )
            continue
        yield enricher
```

**Pipeline integration -- two-phase enrichment pass:**

The enrichment pass runs in two explicit phases after all core artifacts are written.
It is invoked **once at the end of `run_pipeline`**, after all episodes have been
processed and all core artifacts (metadata, GIL, KG, bridge) exist on disk. It is
gated by `enrichment.enabled` in config -- when `false` or absent, the call is a
no-op.

Episode-scope enrichers run first (once per episode). Corpus-scope enrichers run
second (once per corpus). Enrichers within a phase have no guaranteed order and must
not depend on each other's output. A corpus-scope enricher **may** read episode-scope
enricher output (since Phase 1 completes before Phase 2 starts).

```python
# podcast_scraper/enrichment/enrichment_pass.py

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from podcast_scraper.search.corpus_scope import discover_metadata_files
from podcast_scraper.builders.rfc072_artifact_paths import bridge_json_path_adjacent_to_metadata
from podcast_scraper.workflow.metadata_generation import (
    _determine_gi_path,
    _determine_kg_path,
)

from .protocol import EpisodeArtifactBundle, EnricherScope
from .registry import resolve_enrichers

logger = logging.getLogger(__name__)


def _build_bundles(corpus_root: Path) -> list[EpisodeArtifactBundle]:
    """Build artifact bundles using existing discovery helpers."""
    bundles = []
    for meta_path in discover_metadata_files(corpus_root):
        meta_s = str(meta_path)
        gi = Path(_determine_gi_path(meta_s))
        kg = Path(_determine_kg_path(meta_s))
        bridge = Path(bridge_json_path_adjacent_to_metadata(meta_s))
        stem = meta_path.stem
        for suffix in (".metadata",):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

        episode_id = ""
        if bridge.is_file():
            doc = json.loads(bridge.read_text())
            episode_id = doc.get("episode_id", "")

        bundles.append(
            EpisodeArtifactBundle(
                metadata_path=meta_path,
                gi_path=gi if gi.is_file() else None,
                kg_path=kg if kg.is_file() else None,
                bridge_path=bridge if bridge.is_file() else None,
                episode_id=episode_id,
                stem=stem,
            )
        )
    return bundles


def _failure_envelope(
    enricher, error: str, episode_id: str | None = None,
) -> dict:
    envelope: dict = {
        "enricher_id": enricher.manifest.id,
        "enricher_version": enricher.manifest.version,
        "schema_version": "1.0",
        "derived": True,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "error": error,
        "data": None,
    }
    if episode_id:
        envelope["episode_id"] = episode_id
    return envelope


def run_enrichment_pass(corpus_root: Path, config: dict) -> None:
    """
    Run all enabled enrichers for the corpus.
    Phase 1: episode-scope enrichers (once per episode).
    Phase 2: corpus-scope enrichers (once for the whole corpus).
    Never raises -- failures are logged and written to enricher output.
    """
    bundles = _build_bundles(corpus_root)
    enrichers = list(resolve_enrichers(config))

    episode_enrichers = [
        e for e in enrichers if e.manifest.scope == EnricherScope.EPISODE
    ]
    corpus_enrichers = [
        e for e in enrichers if e.manifest.scope == EnricherScope.CORPUS
    ]

    # --- Phase 1: episode-scope ---
    for bundle in bundles:
        enrichments_dir = bundle.metadata_path.parent / "enrichments"
        enrichments_dir.mkdir(exist_ok=True)
        for enricher in episode_enrichers:
            fname = f"{bundle.stem}.{enricher.manifest.writes}"
            output_path = enrichments_dir / fname
            try:
                result = enricher.enrich(
                    bundle=bundle,
                    corpus_root=corpus_root,
                    all_bundles=None,
                    config=config,
                )
            except Exception as exc:
                logger.warning(
                    "Enricher %s failed for %s: %s",
                    enricher.manifest.id, bundle.stem, exc,
                )
                result = _failure_envelope(
                    enricher, str(exc), episode_id=bundle.episode_id,
                )
            output_path.write_text(json.dumps(result, indent=2))

    # --- Phase 2: corpus-scope ---
    corpus_enrichments_dir = corpus_root / "enrichments"
    corpus_enrichments_dir.mkdir(exist_ok=True)
    for enricher in corpus_enrichers:
        output_path = corpus_enrichments_dir / enricher.manifest.writes
        try:
            result = enricher.enrich(
                bundle=None,
                corpus_root=corpus_root,
                all_bundles=bundles,
                config=config,
            )
        except Exception as exc:
            logger.warning(
                "Enricher %s (corpus) failed: %s",
                enricher.manifest.id, exc,
            )
            result = _failure_envelope(enricher, str(exc))
        output_path.write_text(json.dumps(result, indent=2))
```

---

## Enricher Tiers

Enrichers are classified by what they require to run. This classification is declared
in the manifest and drives config defaults -- deterministic enrichers are on by default,
LLM enrichers require explicit opt-in.

| Tier | Requires | Default | Examples |
|---|---|---|---|
| `deterministic` | Nothing beyond Python stdlib | On | topic co-occurrence, temporal velocity, grounding rate |
| `embedding` | FAISS index (already present) | Off | topic similarity, semantic deduplication |
| `ml` | Local ML model via Ollama or HuggingFace | Off | NLI contradiction, sentiment |
| `llm` | LLM API call -- explicit opt-in only | Off, opt-in | position narration, profile synthesis |

LLM tier enrichers are the only ones that incur API cost and carry hallucination risk.
They require both `enabled: true` and `opt_in: true` in config. The `requires_opt_in`
flag in the manifest enforces this at the registry level -- a misconfigured LLM enricher
that lacks explicit opt-in is skipped with a WARNING log, never run by accident. No
output file is written for skipped enrichers; consumers treat a missing enrichment file
as "not configured," not as a failure.

---

## Example Enrichers

### Enricher 1: Topic Co-occurrence (`topic_cooccurrence`)

**What it does:** For a given episode, computes which topic pairs appear together.
Across the corpus, these co-occurrence counts provide the data that the Topic Entity
View (PRD-026) renders as `RELATED_TO`-style connections. The KG ontology reserves
`RELATED_TO` as an edge type but the v1 builder does not emit it; this enricher
provides the signal without mutating `*.kg.json`.

**Why it matters:** PRD-026's Related Topics section (FR6) needs to know which topics
cluster together. A topic with zero related-topic signals is an island -- a user
browsing `topic:ai-safety` has no way to discover that `topic:ai-regulation` is
closely associated in the corpus. Co-occurrence is the cheapest possible signal for
this -- no model, no embeddings, pure counting.

**Tier:** `deterministic`
**Scope:** `episode` (per-episode co-occurrence) + `corpus` (aggregated counts,
separate enricher `topic_cooccurrence_corpus`)
**Reads:** `*.bridge.json`
**Writes:** `topic_cooccurrence.json`

```python
# podcast_scraper/enrichment/builtin/topic_cooccurrence.py

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from ..protocol import (
    Enricher,
    EnricherManifest,
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
)


class TopicCooccurrenceEnricher:

    @property
    def manifest(self) -> EnricherManifest:
        return EnricherManifest(
            id="topic_cooccurrence",
            version="1.0.0",
            scope=EnricherScope.EPISODE,
            tier=EnricherTier.DETERMINISTIC,
            reads=[".bridge.json"],
            writes="topic_cooccurrence.json",
            description=(
                "Computes which topic pairs co-occur in this episode. "
                "Corpus-level aggregation (separate enricher) provides "
                "data for RELATED_TO-style connections in the viewer."
            ),
            requires_opt_in=False,
        )

    def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict,
    ) -> dict:
        assert bundle is not None
        if bundle.bridge_path is None or not bundle.bridge_path.is_file():
            return {
                "enricher_id": self.manifest.id,
                "enricher_version": self.manifest.version,
                "schema_version": "1.0",
                "episode_id": bundle.episode_id,
                "derived": True,
                "computed_at": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "error": "bridge artifact not found",
                "data": None,
            }

        bridge = json.loads(bundle.bridge_path.read_text())
        episode_id = bridge.get("episode_id", bundle.episode_id)

        topics = [
            i for i in bridge.get("identities", [])
            if i["type"] == "topic"
        ]
        topic_ids = [t["id"] for t in topics]
        pairs = [
            {"topic_a": a, "topic_b": b, "count": 1}
            for a, b in combinations(sorted(topic_ids), 2)
        ]

        return {
            "enricher_id": self.manifest.id,
            "enricher_version": self.manifest.version,
            "schema_version": "1.0",
            "episode_id": episode_id,
            "derived": True,
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "status": "ok",
            "data": {
                "topic_count": len(topic_ids),
                "pair_count": len(pairs),
                "pairs": pairs,
            },
        }


enricher_instance = TopicCooccurrenceEnricher()
```

**Corpus aggregation** (corpus-scope pass, separate enricher):

The corpus-scope `topic_cooccurrence_corpus` enricher (registered separately in
`_BUILTIN_ENRICHERS`) scans all per-episode `*.topic_cooccurrence.json` files under
`metadata/enrichments/`, sums pair counts, and writes a ranked list of topic pairs to
`{corpus_root}/enrichments/topic_cooccurrence.json`. This output provides the data
that the Topic Entity View renders as topic-to-topic connections -- it does **not**
inject edges into `*.kg.json` (immutability invariant).

---

### Enricher 2: Temporal Velocity (`temporal_velocity`)

**What it does:** Computes how fast each topic is growing or declining in the corpus
by counting topic appearances per time window (monthly by default) and computing a
trend signal (acceleration, stable, declining).

**Why it matters:** PRD-026's Timeline section (FR3) needs more than a list of
episodes -- it needs a *trend*. A topic that appeared in 2 episodes in 2023 and 12
episodes in Q1 2026 is accelerating. A topic that peaked in 2024 and has not appeared
since is declining. This signal requires no model -- it is pure arithmetic over KG
MENTIONS edges and episode publish dates. But it is the difference between a flat
timeline and a living corpus map.

**Tier:** `deterministic`
**Scope:** `corpus`
**Reads:** `*.kg.json` + `*.bridge.json` (all episodes via `all_bundles`)
**Writes:** `temporal_velocity.json` (under `{corpus_root}/enrichments/`)

```python
# podcast_scraper/enrichment/builtin/temporal_velocity.py

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..protocol import (
    Enricher,
    EnricherManifest,
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
)


class TemporalVelocityEnricher:

    @property
    def manifest(self) -> EnricherManifest:
        return EnricherManifest(
            id="temporal_velocity",
            version="1.0.0",
            scope=EnricherScope.CORPUS,
            tier=EnricherTier.DETERMINISTIC,
            reads=[".kg.json", ".bridge.json"],
            writes="temporal_velocity.json",
            description=(
                "Computes monthly topic mention counts across corpus "
                "and derives a trend signal (accelerating / stable / "
                "declining) per topic."
            ),
            requires_opt_in=False,
        )

    def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict,
    ) -> dict:
        assert all_bundles is not None
        counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for ep in all_bundles:
            if ep.bridge_path is None or ep.kg_path is None:
                continue
            if not ep.bridge_path.is_file() or not ep.kg_path.is_file():
                continue

            bridge = json.loads(ep.bridge_path.read_text())
            kg = json.loads(ep.kg_path.read_text())

            episode_node = next(
                (n for n in kg.get("nodes", [])
                 if n.get("type") == "Episode"),
                None,
            )
            if not episode_node:
                continue
            publish_date = (
                episode_node.get("properties", {}).get("publish_date")
            )
            if not publish_date:
                continue

            year_month = publish_date[:7]  # "2026-04"

            for identity in bridge.get("identities", []):
                if (
                    identity["type"] == "topic"
                    and identity.get("sources", {}).get("kg")
                ):
                    counts[identity["id"]][year_month] += 1

        topics = []
        for topic_id, monthly in counts.items():
            sorted_months = sorted(monthly.keys())
            monthly_counts = [monthly[m] for m in sorted_months]
            trend = self._trend(monthly_counts)
            topics.append({
                "topic_id": topic_id,
                "monthly_counts": dict(
                    zip(sorted_months, monthly_counts)
                ),
                "total_episodes": sum(monthly_counts),
                "trend": trend,
            })

        topics.sort(key=lambda t: t["total_episodes"], reverse=True)

        return {
            "enricher_id": self.manifest.id,
            "enricher_version": self.manifest.version,
            "schema_version": "1.0",
            "derived": True,
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "status": "ok",
            "data": {
                "topic_count": len(topics),
                "topics": topics,
            },
        }

    def _trend(self, counts: list[int]) -> str:
        if len(counts) < 6:
            return "insufficient_data"
        recent = sum(counts[-3:])
        previous = sum(counts[-6:-3])
        if previous == 0:
            return "accelerating" if recent > 0 else "stable"
        ratio = recent / previous
        if ratio > 1.5:
            return "accelerating"
        if ratio < 0.5:
            return "declining"
        return "stable"


enricher_instance = TemporalVelocityEnricher()
```

**Output shape:**

```json
{
  "enricher_id": "temporal_velocity",
  "enricher_version": "1.0.0",
  "schema_version": "1.0",
  "derived": true,
  "computed_at": "2026-04-13T10:00:00Z",
  "status": "ok",
  "data": {
    "topic_count": 42,
    "topics": [
      {
        "topic_id": "topic:ai-regulation",
        "monthly_counts": {
          "2024-01": 2,
          "2024-06": 4,
          "2025-01": 6,
          "2025-06": 9,
          "2026-01": 11,
          "2026-03": 14
        },
        "total_episodes": 46,
        "trend": "accelerating"
      }
    ]
  }
}
```

---

## Enricher Catalogue

The following enrichers are initial candidates. The two above are fully specified. The
remainder are described at design level -- implementation follows the same protocol
pattern.

| ID | Tier | Scope | Reads | Default | What it computes |
| --- | --- | --- | --- | --- | --- |
| `topic_cooccurrence` | deterministic | episode | bridge | On | Topic pair co-occurrence counts per episode |
| `topic_cooccurrence_corpus` | deterministic | corpus | episode enrichments | On | Aggregated co-occurrence -- data for RELATED_TO-style viewer connections |
| `temporal_velocity` | deterministic | corpus | kg + bridge | On | Monthly topic mention counts + trend signal |
| `grounding_rate` | deterministic | corpus | gi + bridge | On | % grounded Insights per person across corpus |
| `guest_coappearance` | deterministic | corpus | kg + bridge | On | Which persons appear together across episodes |
| `insight_density` | deterministic | episode | gi | On | Insight count per episode segment (early/mid/late) |
| `topic_similarity` | embedding | corpus | FAISS index | Off | Cosine similarity between topic embeddings |
| `nli_contradiction` | ml | corpus | gi + bridge | Off | NLI-scored contradiction pairs across Insights on shared topics |

The first six are pure arithmetic -- no model, no external dependency, trivially
testable. They ship enabled by default. The last two require opt-in.

**`nli_contradiction` scope clarification:** This enricher produces **candidate
contradiction pairs** -- each record contains `topic_id`, `person_a_id`,
`person_b_id`, `insight_a_id`, `insight_b_id`, and a `contradiction_score`
(0.0 -- 1.0). It does **not** generate human-readable position summaries or
rankings.
A future analysis layer may add query-time refinement (summaries, ranking). The
enricher output is a **base layer** that can surface raw contradiction pairs
(with Insight text) without further LLM refinement.

---

## Key Decisions

1. **Enrichers never modify core artifacts**
   - **Decision**: Hard invariant. `*.gi.json`, `*.kg.json`, `*.bridge.json` are
     immutable after the core pipeline writes them.
   - **Rationale**: The grounding contract is the system's trust foundation. An
     enricher that annotates a GIL Insight in place blurs the line between grounded
     extraction and derived computation. The `enrichments/` directory is the explicit
     boundary.

2. **Enrichers are always non-blocking**
   - **Decision**: Enrichers run in a separate pass after all core artifacts are
     written. They never gate the core pipeline.
   - **Rationale**: Core pipeline reliability must not depend on enricher availability.
     A corpus is fully usable -- searchable, browsable, queryable -- with zero
     enrichers. Enrichers add value; they do not enable basic function.

3. **`derived: true` is non-negotiable on all enricher output**
   - **Decision**: Every enricher output file carries `derived: true` at the root.
   - **Rationale**: Downstream consumers (API, UI, LLM prompts) must be able to
     distinguish grounded core facts from derived computations. This is the enrichment
     layer's equivalent of `grounded: true/false` in GIL.

4. **LLM tier requires explicit double opt-in**
   - **Decision**: `enabled: true` AND `opt_in: true` both required in config.
     Registry enforces this -- misconfigured LLM enrichers are skipped with a
     WARNING log, never run by accident.
   - **Rationale**: LLM enrichers incur API cost and carry hallucination risk. They
     should never run by accident or by default. The double opt-in makes the intent
     explicit.

5. **Enrichers are not features**
   - **Decision**: Enrichers produce data files. Features (UI views, API endpoints)
     consume enricher output. An enricher has no knowledge of which features consume
     it.
   - **Rationale**: Keeps enrichers testable in isolation, reusable across features,
     and composable. The Topic Entity View (PRD-026) consumes temporal velocity
     output. A future controversy radar feature could also consume temporal velocity
     output. The enricher does not know or care about either.

6. **Only registered enrichers may read artifact paths**
   - **Decision**: No network access except for declared LLM tier enrichers. The
     registry is the only entry point for enricher execution.
   - **Rationale**: Prevents unregistered code from accessing corpus data. LLM tier
     enrichers are the only exception (they need provider API access) and require
     explicit double opt-in.

7. **Two-phase execution: episode first, then corpus**
   - **Decision**: The enrichment pass runs in two explicit phases. Phase 1 runs all
     episode-scope enrichers (once per episode). Phase 2 runs all corpus-scope
     enrichers (once for the whole corpus). Enrichers within a phase have no
     guaranteed order and must not depend on each other's output. A corpus-scope
     enricher may read episode-scope enricher output (Phase 1 completes before
     Phase 2 starts).
   - **Rationale**: Corpus aggregation enrichers (e.g. `topic_cooccurrence_corpus`)
     depend on per-episode enricher output. A flat list with no ordering would
     produce wrong or partial results.

8. **Outputs are overwritten on re-run; initial release uses full corpus recompute**
   - **Decision**: Running the enrichment pass overwrites existing enricher output
     files unconditionally. Corpus-scope enrichers recompute from scratch across all
     episodes. Incremental/delta updates are a non-goal for the initial release.
   - **Rationale**: Full recompute is simple and correct. For typical corpus sizes
     (hundreds of episodes), deterministic enrichers complete in seconds. Incremental
     updates add complexity that is not justified until corpus sizes reach thousands
     of episodes; that optimization is deferred to a future RFC.

---

## Alternatives Considered

1. **Enrichers as pipeline stages (blocking)**
   - **Description**: Run enrichers inline in the core pipeline after each episode.
   - **Pros**: Simpler execution model, single pass.
   - **Cons**: A slow or failing enricher delays or breaks core pipeline execution.
     Couples optional computation to required computation. Violates the
     salt-and-pepper principle -- the dish should not depend on the seasoning.
   - **Why Rejected**: Non-blocking enrichment pass is the correct separation.

2. **Enrichers writing back into core artifacts**
   - **Description**: An enricher adds fields to `*.gi.json` or `*.kg.json` directly.
   - **Pros**: One artifact per episode, simpler for consumers.
   - **Cons**: Destroys the immutability of core artifacts. Makes provenance
     ambiguous -- which fields came from extraction, which from enrichment? Breaks
     the grounding contract. Creates versioning nightmares.
   - **Why Rejected**: The `enrichments/` directory boundary is non-negotiable.

3. **Single combined enrichment artifact per episode**
   - **Description**: All enrichers write to a single `enrichments.json` rather than
     one file per enricher.
   - **Pros**: Fewer files per episode.
   - **Cons**: A failed enricher can corrupt the entire combined artifact. Cannot
     independently version, enable, or disable individual enrichers. Cannot run
     enrichers in parallel without merge conflicts.
   - **Why Rejected**: One file per enricher is cleaner, independently versioned,
     and parallelism-safe.

4. **Per-episode directories (layout migration)**
   - **Description**: Introduce per-episode directories
     (`episode_abc/gi.json`, etc.) and place `enrichments/` inside each.
   - **Pros**: Clean per-episode encapsulation; enrichments are co-located.
   - **Cons**: Requires migrating the entire flat layout (ADR-004), changing
     `discover_metadata_files`, `corpus_catalog`, `_determine_*_path`, all server
     globs, and every test fixture. Massive scope for a non-core concern.
   - **Why Rejected**: Enrichments fit within the existing flat layout using
     `metadata/enrichments/` without any migration.

---

## Testing Strategy

**Unit tests:**

- Each enricher is unit-tested with synthetic fixture artifacts in
  `tests/unit/enrichment/test_{enricher_id}.py`.
- Tests cover: correct output shape, `derived: true` present, `status: ok` on valid
  input, `status: failed` on malformed input (never raises).
- `EpisodeArtifactBundle` construction and enricher manifest completeness tested
  separately in `tests/unit/enrichment/test_protocol.py`.

**Integration tests:**

- `tests/integration/enrichment/test_enrichment_pass.py` runs the full two-phase
  enrichment pass against eval corpus episodes and asserts:
  `metadata/enrichments/` directory created, configured episode-scope enrichers
  produce output files per episode, `{corpus_root}/enrichments/` directory created,
  corpus-scope enrichers produce output files, failed enrichers produce
  `status: failed` output (not exceptions).

**Corpus-scope enricher tests:**

- Synthetic multi-episode corpus with known topic distributions using the flat
  `metadata/` layout.
- Assert temporal velocity trend signals match expected values.
- Assert topic co-occurrence counts are correct.

**Success criteria:**

1. All configured deterministic enrichers produce output for every episode in the eval
   corpus.
2. No enricher failure propagates an exception to the core pipeline.
3. All enricher outputs carry `derived: true`.
4. Core artifacts (`*.gi.json`, `*.kg.json`, `*.bridge.json`) are byte-identical
   before and after the enrichment pass.
5. LLM tier enricher is skipped (with WARNING log) when `opt_in: true` is absent,
   regardless of `enabled: true`.

---

## Rollout

**Phase 1 -- Protocol and registry:**
Implement `protocol.py` (including `EpisodeArtifactBundle`), `registry.py`,
`enrichment_pass.py` with two-phase execution. No enrichers yet.
Wire enrichment pass into `workflow.orchestration` as a no-op when no enrichers are
configured.

**Phase 2 -- Deterministic enrichers:**
Implement and ship `topic_cooccurrence`, `topic_cooccurrence_corpus`,
`temporal_velocity`, `grounding_rate`, `guest_coappearance`, `insight_density`.
All enabled by default. These are the salt and pepper -- always on, zero cost,
immediate value. These enrichers directly serve PRD-026 (Topic Entity View) and
provide the grounding data for PRD-027 (Enriched Search).

**Phase 3 -- Embedding and ML enrichers (opt-in):**
Implement `topic_similarity` and `nli_contradiction`. Off by default, opt-in.
Document in enricher guide.

**Phase 4 -- LLM and query-time enrichers (separate RFC):**
Position narration, profile synthesis, search enrichment. Double opt-in.
Separate RFC defining LLM enricher prompt contract and output schema. This phase also
covers the **QueryEnricher protocol** required by PRD-027 (Enriched Search) -- a
request-time enricher that operates on FAISS results rather than writing files.
Ideas for future LLM enrichers include search synthesis and profile summarisation.

**Phase numbering disambiguation:** "RFC-073 Phase 4" (this section) refers to the
**QueryEnricher protocol** extension for request-time LLM enrichment. "RFC-072
Phase 4" refers to **CIL query patterns** (cross-layer joins, `position_arc` and
`person_profile` queries). These are distinct capabilities in different RFCs that
happen to share the same phase number. PRD-027 (Enriched Search) depends on RFC-073
Phase 4; PRD-028/029 (Position Tracker/Person Profile) depend on RFC-072 Phase 4.

---

## Server and Viewer Consumption (Scope)

This RFC defines the **write side** of the enrichment layer: protocol, registry,
execution, and output format. The **read side** -- how the FastAPI server discovers
and serves enrichment files, and how the GI/KG viewer consumes them -- is defined by
the consuming PRDs:

- **PRD-026 (Topic Entity View)** is the first read-side consumer. It requires API
  endpoints to serve `topic_cooccurrence`, `temporal_velocity`, and `grounding_rate`
  enrichment outputs. It drives the server changes listed below.
- **PRD-027 (Enriched Search)** extends semantic search with query-time LLM
  enrichment. It requires a **QueryEnricher protocol** (Phase 4 extension of this
  RFC) that operates at request time rather than writing files.

Likely server integration points (informational, not normative):

- `artifacts.py` route globs (`**/*.gi.json`, etc.) would need extension for
  `metadata/enrichments/*.json` and `enrichments/*.json`.
- `corpus_catalog.py` could add `has_enrichments` / enrichment file list to
  `CatalogEpisodeRow`.
- The viewer's `useArtifactsStore` would fetch enrichment data via new or extended
  API routes.

Until the PRD-026 server routes are implemented, enrichment files are produced on
disk and available for manual inspection or script consumption.

---

## Benefits

1. **Clean separation**: Derived signals live in `enrichments/`, grounded facts live
   in core artifacts. The boundary is explicit, enforced, and permanent.
2. **Composable**: Each enricher is independently testable, versioned, and deployable.
   Features consume whichever enrichers they need without coupling to others.
3. **Scalable complexity**: Start with six trivial deterministic enrichers. Add ML
   and LLM tiers progressively as the corpus and use cases demand. No architectural
   change required.
4. **Extensible**: New enrichers register via the same registry and protocol. Adding
   a new enricher requires no changes to the runner, config schema, or existing
   enrichers.
5. **Trust-preserving**: `derived: true` on all enricher output means downstream
   consumers -- including LLM prompts -- always know what is grounded fact vs
   computed signal. This is the enrichment layer's equivalent of the GIL grounding
   contract.
6. **Layout-compatible**: Enrichments fit within the existing flat archive layout
   (ADR-004, ADR-051) without requiring a filesystem migration. Existing discovery
   helpers (`discover_metadata_files`, `_determine_*_path`) are unaffected.

---

## Open Questions

The following are deliberately deferred and do not block implementation of Phases 1-2:

1. **CLI subcommand for standalone enrichment.** Should there be a `make enrich` /
   `python -m podcast_scraper.cli enrich --output-dir ...` that runs the enrichment
   pass independently of the full pipeline? Useful for re-enriching an existing
   corpus without re-running extraction. Likely yes for Phase 2.

2. **Parallelism within a phase.** Episode-scope enrichers are embarrassingly
   parallel across episodes (each writes to a different file). Should the runner use
   a thread pool? For deterministic enrichers the overhead is negligible; for
   embedding/ML enrichers on large corpora it would matter. Deferred until Phase 3.

3. **`.gitignore` for enrichment outputs.** Test fixtures and eval corpus outputs
   may generate `enrichments/` directories. Should these be gitignored globally or
   per-fixture? Follow existing pattern for `metadata/` test outputs.

4. **Enricher output cleanup.** If an enricher is disabled after previously running,
   its output files remain on disk. Should the runner delete stale enrichment files,
   or leave cleanup to the user? Default: leave them (safe, no data loss).

---

## References

- `docs/prd/PRD-026-topic-entity-view.md` -- first read-side consumer of enrichment
  outputs (topic co-occurrence, temporal velocity, grounding rate)
- `docs/prd/PRD-027-enriched-search.md` -- query-time enrichment extending semantic
  search (Phase 4 extension)
- `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` -- bridge artifact
  that enrichers read
- `docs/architecture/gi/ontology.md` -- GIL grounding contract
- `docs/architecture/gi/gi.schema.json` -- GIL JSON schema
- `docs/architecture/kg/ontology.md` -- KG ontology (`RELATED_TO` reserved, not
  emitted by v1 builder)
- `docs/architecture/kg/kg.schema.json` -- KG JSON schema
- `docs/adr/ADR-004-flat-filesystem-archive-layout.md` -- flat layout that enrichments
  extend
- `docs/adr/ADR-020-protocol-based-provider-discovery.md` -- protocol pattern
  (enricher protocol follows same PEP 544 approach)
- `docs/adr/ADR-024-unified-provider-pattern.md` -- unified provider pattern
  (enrichers diverge: simpler single-method units)
- `docs/adr/ADR-026-per-capability-provider-selection.md` -- per-capability config
- `docs/adr/ADR-051-per-episode-json-artifacts-with-logical-union.md` -- sibling
  artifact naming convention (`*.gi.json`, `*.kg.json`, `*.bridge.json`)
- `docs/adr/ADR-052-separate-gil-and-kg-artifact-layers.md` -- layer separation
  rationale that the enrichment boundary extends
- `src/podcast_scraper/search/corpus_scope.py` -- `discover_metadata_files` used by
  enrichment pass for episode discovery
- `src/podcast_scraper/workflow/metadata_generation.py` -- `_determine_*_path` helpers
  used by enrichment pass for artifact resolution
