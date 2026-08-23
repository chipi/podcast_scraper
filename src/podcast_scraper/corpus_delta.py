"""Shared per-run corpus delta — the backbone every corpus derivation consumes (RFC-118).

The orchestrator computes ONE per-episode content fingerprint set per run, diffs it
against the manifest persisted at the corpus root, and hands the resulting
:class:`CorpusDelta` to every corpus-level derivation (vector index, topic clusters,
corpus enrichment). Consumers stop inventing their own notion of "what changed" —
one definition, pipeline-wide.

Fingerprints hash the *derivation inputs* of an episode: the ``gi.json`` and
``kg.json`` content. They are content hashes (mtime-immune) and schema-versioned, so
a change to the fingerprint recipe invalidates every stored fingerprint at once.
The vector index keeps its own embedding-level fingerprint (rows + model id) as an
implementation detail — the backbone answers "did this episode's derived content
change", not "does this exact embedding exist".

Manifest lifecycle: ``compute_corpus_delta`` reads the manifest but never writes it.
The orchestrator persists via ``write_fingerprint_manifest`` only after the
synchronous derivations succeeded — a failed finalize leaves the manifest untouched,
so the next run sees the same episodes as changed and re-derives (fail-safe,
never stale-skip).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle

logger = logging.getLogger(__name__)

# Bump to invalidate every stored fingerprint (recipe change ⇒ full re-derive).
FINGERPRINT_SCHEMA_VERSION = 1

MANIFEST_FILENAME = "derivation_fingerprints.json"

# Marker hashed in place of a missing artifact so present-with-empty-content and
# absent stay distinguishable.
_ABSENT = b"\x00absent\x00"


@dataclass(frozen=True)
class CorpusDelta:
    """What changed this run, plus the full corpus for cross-episode consumers.

    changed_ids : episode_ids whose content fingerprint differs from the prior
                  derivation run (or are new to the manifest).
    removed_ids : episode_ids present in the prior manifest but absent now.
    all_bundles : the FULL corpus — a pairwise consumer needs the unchanged
                  episodes too (k changed × (n−k) unchanged pairs).
    forced      : True for an explicit full re-derive; consumers must ignore
                  prior caches.
    fingerprints: the fresh {episode_id: fingerprint} map this delta was computed
                  from; the orchestrator persists it after derivations succeed.
    """

    changed_ids: frozenset[str]
    removed_ids: frozenset[str]
    all_bundles: List[EpisodeArtifactBundle]
    forced: bool = False
    fingerprints: Dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    @property
    def is_empty(self) -> bool:
        """True when nothing changed — consumers with a current output may skip."""
        return not self.changed_ids and not self.removed_ids and not self.forced

    def summary(self) -> Dict[str, object]:
        """Compact loggable summary (counts, not id lists — corpora are large)."""
        return {
            "changed": len(self.changed_ids),
            "removed": len(self.removed_ids),
            "total": len(self.all_bundles),
            "forced": self.forced,
        }

    def changed_metadata_relpaths(self, corpus_root: Path) -> List[str]:
        """Corpus-root-relative metadata paths of the changed episodes.

        The vector index iterates metadata files, not episode ids — and its id
        vocabulary (``episode.episode_id``) differs from the bundles' guid-or-stem
        fallback. Paths are unambiguous in both, so they are the cross-process
        currency for the reindex retrofit.
        """
        root = Path(corpus_root).resolve()
        out: List[str] = []
        for b in self.all_bundles:
            if b.episode_id not in self.changed_ids:
                continue
            try:
                out.append(b.metadata_path.resolve().relative_to(root).as_posix())
            except ValueError:
                # A bundle outside the corpus root cannot be expressed; the index
                # then sees it as "changed by default" (absent set entries are only
                # ever used to count drift, never to skip).
                continue
        return sorted(out)


def episode_derivation_fingerprint(bundle: EpisodeArtifactBundle) -> str:
    """Content hash of one episode's derivation inputs (gi + kg), mtime-immune."""
    h = hashlib.sha256()
    h.update(f"v{FINGERPRINT_SCHEMA_VERSION}".encode("utf-8"))
    for path in (bundle.gi_path, bundle.kg_path):
        h.update(b"\x1e")
        if path is None:
            h.update(_ABSENT)
            continue
        try:
            h.update(path.read_bytes())
        except OSError:
            # Unreadable counts as absent → fingerprint changes when it reappears.
            h.update(_ABSENT)
    return h.hexdigest()


def manifest_path(corpus_root: Path) -> Path:
    """The fingerprint manifest sidecar at the corpus root."""
    return Path(corpus_root) / MANIFEST_FILENAME


def load_fingerprint_manifest(corpus_root: Path) -> Dict[str, str]:
    """Load ``{episode_id: fingerprint}``; ``{}`` when absent, unreadable, or
    written by a different fingerprint schema (recipe change ⇒ everything changed)."""
    try:
        data = json.loads(manifest_path(corpus_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("schema") != FINGERPRINT_SCHEMA_VERSION:
        return {}
    fps = data.get("fingerprints")
    return {str(k): str(v) for k, v in fps.items()} if isinstance(fps, dict) else {}


def write_fingerprint_manifest(corpus_root: Path, fingerprints: Dict[str, str]) -> None:
    """Atomically persist the manifest (temp + rename). Non-fatal on failure."""
    p = manifest_path(corpus_root)
    try:
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(
            json.dumps(
                {"schema": FINGERPRINT_SCHEMA_VERSION, "fingerprints": fingerprints},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        tmp.replace(p)
    except OSError as exc:  # a manifest hiccup must never fail a finished run
        logger.warning("corpus_delta: could not write %s: %s", p, exc)


def compute_corpus_delta(
    corpus_root: Path,
    bundles: Optional[List[EpisodeArtifactBundle]] = None,
    *,
    force: bool = False,
) -> CorpusDelta:
    """Fingerprint the corpus and diff against the persisted manifest.

    Args:
        corpus_root: Corpus root directory (single-feed root or multi-feed parent).
        bundles: Pre-discovered bundles; discovered from disk when ``None``.
        force: Explicit full re-derive — every episode is treated as changed and
            consumers must ignore prior caches. ``removed_ids`` stays accurate.

    Returns:
        The delta, carrying the fresh fingerprint map for later persistence.
    """
    corpus_root = Path(corpus_root)
    if bundles is None:
        from podcast_scraper.enrichment.paths import discover_episode_bundles

        bundles = discover_episode_bundles(corpus_root)

    fresh: Dict[str, str] = {}
    for b in bundles:
        if b.episode_id in fresh:
            logger.debug(
                "corpus_delta: duplicate episode_id %r (stem=%r) — last bundle wins",
                b.episode_id,
                b.stem,
            )
        fresh[b.episode_id] = episode_derivation_fingerprint(b)

    stored = load_fingerprint_manifest(corpus_root)
    removed = frozenset(stored) - frozenset(fresh)
    if force:
        changed = frozenset(fresh)
    else:
        changed = frozenset(eid for eid, fp in fresh.items() if stored.get(eid) != fp)
    return CorpusDelta(
        changed_ids=changed,
        removed_ids=frozenset(removed),
        all_bundles=list(bundles),
        forced=force,
        fingerprints=fresh,
    )
