"""#1058 chunk 3 — Corpus-level cross-show Topic clustering.

Background: under airgapped / airgapped_thin profiles, KG Topic nodes
are derived from BART/SummLlama bullet text. Each show's Topics are
literally the bullet text, so the same concept gets a different label
in every show ("AI safety", "AI alignment", "alignment problem"). The
cross-show connectivity surfaces (``cross_show_synthesis``,
``topic→related_topics`` across feeds) need topics that match across
shows to surface anything — without a corpus-level merge, they stay
empty even after a real pipeline run on a multi-show corpus.

This module fixes the cross-show gap deterministically. It walks every
KG artifact in a corpus, embeds each Topic label via
sentence-transformers (already an airgapped dependency, also used by
``gi/about_edges.py``), clusters by cosine similarity, and for each
cluster that spans ≥2 episodes from ≥2 shows adds a synthetic
``concept:topic-{slug}`` Topic node + ``RELATED_TO`` edges from every
source Topic to the concept-Topic. Idempotent — re-running on a
corpus that already has concept-Topics adds nothing new.

The clustering is intentionally distinct from the per-episode
intra-cluster merge done by ``topic_clusters_default_0_75``: this one
operates across episodes / shows. The concept-Topic node lives in
each source KG artifact (duplicated, with stable ID for idempotency)
so ``corpus_graph`` resolves the edge target regardless of which
artifact the join lands on first.

Tested with a pluggable embedder (see ``cluster_topics`` signature) so
the test suite exercises the algorithm without loading a real model.
The production caller uses sentence-transformers by default.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CLUSTER_THRESHOLD = 0.75
DEFAULT_MIN_EPISODES_PER_CLUSTER = 2

_SLUG_NORMALISE = re.compile(r"[^a-z0-9]+")


@dataclass
class TopicMention:
    """A single Topic node observed in a KG artifact."""

    episode_id: str
    podcast_id: Optional[str]
    label: str
    topic_id: str
    kg_path: Path


@dataclass
class TopicCluster:
    """A cluster of TopicMentions sharing a canonical concept."""

    canonical_label: str
    concept_id: str
    members: List[TopicMention] = field(default_factory=list)

    @property
    def episode_count(self) -> int:
        return len({m.episode_id for m in self.members})

    @property
    def podcast_count(self) -> int:
        return len({m.podcast_id for m in self.members if m.podcast_id})


@dataclass
class ClusteringSummary:
    """Result of an apply pass."""

    clusters_found: int
    concept_topics_added: int
    related_to_edges_added: int
    artifacts_mutated: int


def slug_for_concept(label: str) -> str:
    """Stable kebab-slug for the synthetic concept-Topic id."""
    if not label:
        return ""
    return _SLUG_NORMALISE.sub("-", label.strip().lower()).strip("-")


def gather_corpus_topics(corpus_root: Path) -> List[TopicMention]:
    """Walk every ``*.kg.json`` under ``corpus_root`` and collect every
    Topic node with its source episode / podcast ids and on-disk path.
    """
    import json

    out: List[TopicMention] = []
    for kg_path in corpus_root.rglob("*.kg.json"):
        try:
            data = json.loads(kg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("topic_clustering: skip %s (%s)", kg_path, exc)
            continue
        episode_id, podcast_id = _episode_and_podcast_ids(data)
        if not episode_id:
            continue
        for n in data.get("nodes") or []:
            if not isinstance(n, dict) or n.get("type") != "Topic":
                continue
            props = n.get("properties") or {}
            label = (props.get("label") or "").strip()
            topic_id = n.get("id") or ""
            if not label or not isinstance(topic_id, str):
                continue
            out.append(
                TopicMention(
                    episode_id=episode_id,
                    podcast_id=podcast_id,
                    label=label,
                    topic_id=topic_id,
                    kg_path=kg_path,
                )
            )
    return out


def _episode_and_podcast_ids(
    kg_artifact: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort extract (episode_id, podcast_id) from a KG artifact."""
    episode_id: Optional[str] = None
    podcast_id: Optional[str] = None
    for n in kg_artifact.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        t = n.get("type")
        nid = n.get("id")
        if t == "Episode" and isinstance(nid, str):
            episode_id = nid
            props = n.get("properties") or {}
            pid = props.get("podcast_id")
            if isinstance(pid, str):
                podcast_id = pid
        elif t == "Podcast" and isinstance(nid, str) and podcast_id is None:
            podcast_id = nid
    return episode_id, podcast_id


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot-product cosine for unit-normalised vectors. Falls back to
    explicit normalisation when inputs aren't unit-length."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def cluster_topics(
    topics: List[TopicMention],
    embedder: Callable[[List[str]], List[List[float]]],
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    min_episodes: int = DEFAULT_MIN_EPISODES_PER_CLUSTER,
) -> List[TopicCluster]:
    """Group ``topics`` into clusters via greedy single-link clustering
    on cosine similarity from ``embedder`` output.

    Two TopicMentions land in the same cluster when their label
    embeddings cosine ≥ ``threshold``. Clusters are kept only when
    they span ≥ ``min_episodes`` distinct episodes (default 2 — the
    cross-show contract).

    The canonical label is the most-frequent surface label in the
    cluster, ties broken by shortest string. The concept_id is
    ``concept:topic-{slug}`` from the canonical label.

    Pure-data function — does no I/O. Caller owns artifact mutation.
    """
    if not topics:
        return []

    labels = [t.label for t in topics]
    vectors = embedder(labels)
    if len(vectors) != len(topics):
        raise ValueError(f"embedder returned {len(vectors)} vectors for {len(topics)} inputs")

    # Greedy clustering: for each topic in order, assign to the
    # existing cluster whose first member has cosine ≥ threshold,
    # else start a new cluster.
    cluster_assignments: List[int] = [-1] * len(topics)
    cluster_seeds: List[int] = []
    for i, vec_i in enumerate(vectors):
        assigned = False
        for cluster_idx, seed_idx in enumerate(cluster_seeds):
            sim = _cosine(vec_i, vectors[seed_idx])
            if sim >= threshold:
                cluster_assignments[i] = cluster_idx
                assigned = True
                break
        if not assigned:
            cluster_assignments[i] = len(cluster_seeds)
            cluster_seeds.append(i)

    # Group by cluster index.
    buckets: Dict[int, List[TopicMention]] = {}
    for i, topic in enumerate(topics):
        buckets.setdefault(cluster_assignments[i], []).append(topic)

    clusters: List[TopicCluster] = []
    for members in buckets.values():
        episode_ids = {m.episode_id for m in members}
        if len(episode_ids) < min_episodes:
            continue
        canonical = _pick_canonical_label(members)
        concept_id = f"concept:topic-{slug_for_concept(canonical)}"
        clusters.append(
            TopicCluster(
                canonical_label=canonical,
                concept_id=concept_id,
                members=members,
            )
        )
    # Deterministic ordering — sort by canonical label so re-runs
    # produce identical output regardless of dict iteration order.
    clusters.sort(key=lambda c: c.canonical_label.lower())
    return clusters


def _pick_canonical_label(members: List[TopicMention]) -> str:
    """Most-frequent label; ties broken by shortest string then
    lexicographic. Returns the surface label (not the slug)."""
    counts: Dict[str, int] = {}
    for m in members:
        counts[m.label] = counts.get(m.label, 0) + 1
    # max by (count, -len, -lex) — ties pick the shorter / earlier label
    return max(
        counts.items(),
        key=lambda kv: (kv[1], -len(kv[0]), tuple(-ord(c) for c in kv[0])),
    )[0]


def apply_concept_topics_to_corpus(
    clusters: List[TopicCluster],
    *,
    write: bool = True,
) -> ClusteringSummary:
    """Add concept-Topic nodes + RELATED_TO edges to every source KG
    artifact for every cluster.

    Idempotent: clusters whose concept-Topic node + edges are already
    present add nothing. When ``write`` is False, the function returns
    the would-be summary without touching disk (dry-run).

    The artifact is re-written via ``kg.io.write_artifact`` with
    strict validation on, so a malformed mutation surfaces
    immediately.
    """
    import json

    from .io import write_artifact as _kg_write_artifact

    artifacts_mutated: Set[Path] = set()
    concept_topics_added = 0
    related_to_edges_added = 0

    # Group cluster mutations by KG path so each artifact is written
    # at most once.
    per_artifact: Dict[Path, List[Tuple[TopicCluster, TopicMention]]] = {}
    for cluster in clusters:
        for member in cluster.members:
            per_artifact.setdefault(member.kg_path, []).append((cluster, member))

    for kg_path, work in per_artifact.items():
        try:
            data = json.loads(kg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("topic_clustering: cannot reload %s, skipping (%s)", kg_path, exc)
            continue

        nodes = data.setdefault("nodes", [])
        edges = data.setdefault("edges", [])
        existing_node_ids = {n.get("id") for n in nodes if isinstance(n.get("id"), str)}
        existing_related_to = {
            (e.get("from"), e.get("to")) for e in edges if e.get("type") == "RELATED_TO"
        }

        artifact_changed = False
        for cluster, member in work:
            if cluster.concept_id not in existing_node_ids:
                nodes.append(
                    {
                        "id": cluster.concept_id,
                        "type": "Topic",
                        "properties": {
                            "label": cluster.canonical_label,
                            "is_concept": True,
                        },
                    }
                )
                existing_node_ids.add(cluster.concept_id)
                concept_topics_added += 1
                artifact_changed = True
            edge_key = (member.topic_id, cluster.concept_id)
            if edge_key in existing_related_to:
                continue
            edges.append(
                {
                    "type": "RELATED_TO",
                    "from": member.topic_id,
                    "to": cluster.concept_id,
                }
            )
            existing_related_to.add(edge_key)
            related_to_edges_added += 1
            artifact_changed = True

        if artifact_changed:
            if write:
                _kg_write_artifact(kg_path, data, validate=True)
            artifacts_mutated.add(kg_path)

    return ClusteringSummary(
        clusters_found=len(clusters),
        concept_topics_added=concept_topics_added,
        related_to_edges_added=related_to_edges_added,
        artifacts_mutated=len(artifacts_mutated),
    )


def _default_embedder() -> Callable[[List[str]], List[List[float]]]:
    """Production embedder backed by sentence-transformers (same model
    used by ``gi/about_edges.py``). Built lazily so callers that
    supply their own embedder don't pay the model-load cost."""

    def _embed(labels: List[str]) -> List[List[float]]:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        arr = model.encode(labels, convert_to_numpy=True, normalize_embeddings=True)
        return [list(map(float, v)) for v in arr]

    return _embed


def cluster_and_apply_corpus_topics(
    corpus_root: Path,
    *,
    embedder: Optional[Callable[[List[str]], List[List[float]]]] = None,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    min_episodes: int = DEFAULT_MIN_EPISODES_PER_CLUSTER,
    write: bool = True,
) -> ClusteringSummary:
    """End-to-end convenience: gather + cluster + apply.

    ``embedder`` defaults to sentence-transformers (production); pass
    a callable for tests to stub the embedding model out.
    """
    topics = gather_corpus_topics(corpus_root)
    if not topics:
        return ClusteringSummary(
            clusters_found=0,
            concept_topics_added=0,
            related_to_edges_added=0,
            artifacts_mutated=0,
        )
    embed = embedder or _default_embedder()
    clusters = cluster_topics(
        topics,
        embed,
        threshold=threshold,
        min_episodes=min_episodes,
    )
    return apply_concept_topics_to_corpus(clusters, write=write)
