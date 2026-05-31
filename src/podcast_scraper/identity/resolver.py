"""File-local entity resolver — freeform text → canonical CIL id.

Maps a query/mention string (e.g. ``"Sam Altman"``, ``"OpenAI"``,
``"AI regulation"``) to a canonical identity id (``person:…`` / ``org:…`` /
``topic:…``) by matching against a corpus-wide registry of canonical entities.

The registry is aggregated from the per-episode GI and KG artifacts, reusing the
identity-collection and fuzzy-reconciliation machinery already proven in
``builders/bridge_builder.py`` (the per-episode bridge). The resolver is
**conservative**: it returns ``None`` rather than guess when no confident match
exists, because a false positive produces a misleading downstream signal (e.g.
the KG-proximity seed in RFC-091).

Resolution is layered cheap→fuzzy:

1. exact canonical id (``text`` already looks like ``type:slug`` and is known)
2. exact slug (``slugify(text)`` → ``person:/org:/topic:{slug}``)
3. exact display-name / alias (normalized inverted-index lookup)
4. fuzzy embedding match (cosine ≥ threshold over registry display names)

See ``docs/rfc/RFC-090-hybrid-retrieval.md`` (entity filters) and
``docs/rfc/RFC-091-kg-proximity-signal.md`` (traversal seed) for consumers, and
issue #849 for the prerequisite tracking.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..builders.bridge_builder import (
    collect_identity,
    fuzzy_reconcile,
    strip_layer_prefixes,
)
from .slugify import slugify

logger = logging.getLogger(__name__)

CIL_PREFIXES = ("person:", "org:", "topic:")
_TYPE_PRECEDENCE = ("person", "org", "topic")
DEFAULT_FUZZY_THRESHOLD = 0.75
DEFAULT_EMBED_MODEL_ID = "minilm-l6"


def _norm(text: str) -> str:
    """Normalize a name for exact lookup: lowercase, collapse whitespace."""
    return " ".join(str(text).lower().split())


@dataclass
class ResolveResult:
    """A resolved match with provenance for debugging / eval."""

    id: str
    score: float
    method: str  # "exact_id" | "slug" | "alias" | "fuzzy"


@dataclass
class EntityRegistry:
    """Corpus-wide canonical-entity registry with exact + fuzzy lookup."""

    records: Dict[str, Dict[str, Any]]  # canonical id -> {id, type, display_name, aliases, freq}
    name_index: Dict[str, str]  # normalized display-name/alias -> canonical id
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD
    _fuzzy_ids: List[str] = field(default_factory=list)
    _fuzzy_matrix: Any = None  # np.ndarray (N, dim) of L2-normalized display-name embeddings
    _model_id: str = DEFAULT_EMBED_MODEL_ID
    _embedder: Any = None

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def from_identities(
        cls,
        identities: Dict[str, Dict[str, Any]],
        freq: Counter,
        *,
        embedder: Any = None,
        model_id: str = DEFAULT_EMBED_MODEL_ID,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    ) -> "EntityRegistry":
        records: Dict[str, Dict[str, Any]] = {}
        name_index: Dict[str, str] = {}

        for cid, rec in identities.items():
            display = str(rec.get("display_name", "") or "")
            aliases = [a for a in (rec.get("aliases") or []) if isinstance(a, str)]
            record = {
                "id": cid,
                "type": str(rec.get("type", "")),
                "display_name": display,
                "aliases": aliases,
                "freq": int(freq.get(cid, 0)),
            }
            records[cid] = record

            # Inverted index over display name + aliases (skip id-shaped aliases,
            # which bridge fuzzy-merge adds purely for traceability).
            names = [display, *aliases]
            for name in names:
                if not name or name.startswith(CIL_PREFIXES):
                    continue
                key = _norm(name)
                if not key:
                    continue
                existing = name_index.get(key)
                if existing is None or record["freq"] > records[existing]["freq"]:
                    name_index[key] = cid

        registry = cls(
            records=records,
            name_index=name_index,
            fuzzy_threshold=fuzzy_threshold,
            _model_id=model_id,
            _embedder=embedder,
        )
        registry._build_fuzzy_index()
        return registry

    def _build_fuzzy_index(self) -> None:
        """Precompute L2-normalized embeddings of display names for fuzzy match.

        Degrades gracefully (fuzzy disabled) if the embedding model or numpy is
        unavailable — mirrors ``bridge_builder`` behavior.
        """
        ids: List[str] = []
        names: List[str] = []
        for cid, rec in self.records.items():
            display = rec.get("display_name") or ""
            if display:
                ids.append(cid)
                names.append(display)
        if not names:
            return
        try:
            import numpy as np

            if self._embedder is not None:
                matrix = np.asarray(
                    self._embedder.encode(names, normalize_embeddings=True), dtype=float
                )
            else:
                from ..providers.ml.embedding_loader import encode

                matrix = np.asarray(
                    encode(names, self._model_id, normalize=True, return_numpy=True),
                    dtype=float,
                )
        except Exception as exc:  # noqa: BLE001 — fuzzy is best-effort
            logger.debug("Fuzzy index unavailable; exact-only resolution: %s", exc)
            return
        self._fuzzy_ids = ids
        self._fuzzy_matrix = matrix

    def _fuzzy_lookup(self, text: str) -> Optional[Tuple[str, float]]:
        if self._fuzzy_matrix is None or not self._fuzzy_ids:
            return None
        try:
            import numpy as np

            from ..providers.ml.embedding_loader import encode

            if self._embedder is not None:
                qvec = np.asarray(
                    self._embedder.encode([text], normalize_embeddings=True), dtype=float
                )[0]
            else:
                qvec = np.asarray(
                    encode(text, self._model_id, normalize=True, return_numpy=True), dtype=float
                )
            sims = self._fuzzy_matrix @ qvec
            best_idx = int(sims.argmax())
            best_score = float(sims[best_idx])
        except Exception as exc:  # noqa: BLE001
            logger.debug("Fuzzy lookup failed: %s", exc)
            return None
        if best_score >= self.fuzzy_threshold:
            return self._fuzzy_ids[best_idx], best_score
        return None


def _iter_loaded(corpus_dir: Path):
    """Yield ``(source, artifact_dict)`` for every GI and KG artifact under *corpus_dir*."""
    from ..gi.corpus import load_gi_artifacts
    from ..gi.explore import scan_artifact_paths as scan_gi_paths
    from ..kg.corpus import load_kg_artifacts, scan_kg_artifact_paths

    for _path, data in load_kg_artifacts(scan_kg_artifact_paths(corpus_dir)):
        yield "kg", data
    for _path, data in load_gi_artifacts(scan_gi_paths(corpus_dir)):
        yield "gi", data


def build_entity_registry(
    corpus_dir: Path | str,
    *,
    embedder: Any = None,
    model_id: str = DEFAULT_EMBED_MODEL_ID,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    do_fuzzy_reconcile: bool = True,
) -> EntityRegistry:
    """Aggregate a corpus-wide canonical-entity registry from GI + KG artifacts.

    Builds directly from artifacts (does not require ``bridge.json`` to have been
    generated). Reuses ``bridge_builder.collect_identity`` to fold every
    ``person:``/``org:``/``topic:`` node into one identity map, then runs
    ``fuzzy_reconcile`` once at corpus scope to merge single-layer near-duplicates.
    """
    corpus_dir = Path(corpus_dir)
    identities: Dict[str, Dict[str, Any]] = {}
    freq: Counter = Counter()

    for source, data in _iter_loaded(corpus_dir):
        for node in data.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            nid_raw = node.get("id")
            if nid_raw is None:
                continue
            sid = strip_layer_prefixes(str(nid_raw))
            if not sid.startswith(CIL_PREFIXES):
                continue
            collect_identity(identities, node, source=source)
            freq[sid] += 1

    if do_fuzzy_reconcile and identities:
        try:
            fuzzy_reconcile(identities, threshold=fuzzy_threshold, embedder=embedder)
        except Exception as exc:  # noqa: BLE001 — reconciliation is best-effort
            logger.debug("Corpus fuzzy reconciliation skipped: %s", exc)

    return EntityRegistry.from_identities(
        identities,
        freq,
        embedder=embedder,
        model_id=model_id,
        fuzzy_threshold=fuzzy_threshold,
    )


class EntityResolver:
    """Resolve freeform text to a canonical CIL id against a corpus registry."""

    def __init__(self, registry: EntityRegistry):
        self.registry = registry

    @classmethod
    def from_corpus(
        cls,
        corpus_dir: Path | str,
        *,
        embedder: Any = None,
        model_id: str = DEFAULT_EMBED_MODEL_ID,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    ) -> "EntityResolver":
        return cls(
            build_entity_registry(
                corpus_dir,
                embedder=embedder,
                model_id=model_id,
                fuzzy_threshold=fuzzy_threshold,
            )
        )

    def resolve(self, text: str) -> Optional[str]:
        """Return the canonical id for *text*, or ``None`` (conservative)."""
        result = self.resolve_detail(text)
        return result.id if result else None

    def resolve_detail(self, text: str) -> Optional[ResolveResult]:
        """Resolve with provenance: ``(id, score, method)`` or ``None``."""
        t = (text or "").strip()
        if not t:
            return None
        records = self.registry.records

        # 1. Exact canonical id.
        if t.startswith(CIL_PREFIXES) and t in records:
            return ResolveResult(t, 1.0, "exact_id")

        # 2. Exact slug → typed candidate (precedence person > org > topic).
        try:
            slug = slugify(t)
        except ValueError:
            slug = ""
        if slug:
            for typ in _TYPE_PRECEDENCE:
                cid = f"{typ}:{slug}"
                if cid in records:
                    return ResolveResult(cid, 1.0, "slug")

        # 3. Exact display-name / alias.
        alias_id = self.registry.name_index.get(_norm(t))
        if alias_id is not None:
            return ResolveResult(alias_id, 1.0, "alias")

        # 4. Fuzzy embedding match.
        fuzzy = self.registry._fuzzy_lookup(t)
        if fuzzy is not None:
            return ResolveResult(fuzzy[0], fuzzy[1], "fuzzy")

        return None


# Process-level registry cache (mirrors search/corpus_graph + embedding_loader):
# building a registry scans the whole corpus and embeds every display name, so it
# must not be rebuilt per query. Keyed by resolved corpus path + threshold + model.
_entity_registries: Dict[Tuple[str, float, str], EntityRegistry] = {}
_entity_registries_lock = threading.Lock()


def get_entity_resolver(
    corpus_dir: Path | str,
    *,
    embedder: Any = None,
    model_id: str = DEFAULT_EMBED_MODEL_ID,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> EntityResolver:
    """Return an ``EntityResolver`` backed by a process-cached corpus registry."""
    key = (str(Path(corpus_dir).resolve()), float(fuzzy_threshold), str(model_id))
    with _entity_registries_lock:
        if key not in _entity_registries:
            _entity_registries[key] = build_entity_registry(
                corpus_dir,
                embedder=embedder,
                model_id=model_id,
                fuzzy_threshold=fuzzy_threshold,
            )
        return EntityResolver(_entity_registries[key])


def clear_entity_resolver_cache() -> None:
    """Clear the registry cache (tests / after corpus re-index)."""
    with _entity_registries_lock:
        _entity_registries.clear()
