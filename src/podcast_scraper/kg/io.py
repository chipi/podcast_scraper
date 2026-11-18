"""KG artifact I/O: read and write per-episode kg.json files."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, cast, Dict, Optional

from podcast_scraper.utils.atomic_io import write_json_atomic

from .schema import validate_artifact

# model_version values ``kg.pipeline.build_artifact`` stamps when NO extractor produced a graph.
_FAILED_EXTRACTIONS = frozenset({"provider:extraction_failed", "no_extractor"})
_GRAPH_NODE_TYPES = frozenset({"Topic", "Entity", "Person", "Organization", "Object"})


def is_failed_extraction(payload: Dict[str, Any]) -> bool:
    """True when the artifact records that extraction was attempted and produced nothing."""
    mv = str((payload.get("extraction") or {}).get("model_version") or "")
    return mv in _FAILED_EXTRACTIONS


def previous_artifact_to_keep(path: Path, new_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The artifact already at ``path`` when ``new_payload`` would only destroy it, else None.

    A re-run (``relabel_only`` / ``rederive_only``) rewrites kg.json in place. When its extraction
    fails, writing the result replaces a real graph with an empty one. Prod, 2026-09-29: two
    relabel runs turned working NVFP4 graphs (10+ topics each) into ``extraction_failed`` and
    reported ``succeeded``. The previous graph is only kept when it came from a provider extractor
    and has subject nodes — ``topic_labels`` graphs are fabricated (ADR-156), and an empty graph
    is the honest answer over those.
    """
    if not is_failed_extraction(new_payload):
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        old = read_artifact(p, validate=False)
    except (OSError, ValueError):
        return None
    mv = str((old.get("extraction") or {}).get("model_version") or "")
    if not mv.startswith("provider:") or is_failed_extraction(old):
        return None
    if not any(n.get("type") in _GRAPH_NODE_TYPES for n in old.get("nodes") or []):
        return None
    return old


def write_artifact(path: Path, payload: Dict[str, Any], validate: bool = True) -> None:
    """Write a KG artifact to path (e.g. episode.kg.json).

    ATOMIC: a failed or interrupted write leaves the PREVIOUS artifact intact. Same reasoning as
    ``gi.io.write_artifact`` — see ``utils.atomic_io``.

    Args:
        path: Output file path.
        payload: Dict with schema_version, episode_id, extraction, nodes, edges.
        validate: If True, run validation before writing.
    """
    # KG 3.0 from R3b on (docs/wip/changes/r3b_kg_3_0.md): every KG file is written as KG 3.0. A
    # writer may still build in 2.x names; the file is converted, given PUBLISHED_BY when the
    # publisher registry lists its source, and validated strictly against kg.v3.schema.json. A file
    # that does not validate is not written (never warn and write).
    from podcast_scraper.kg.v3_file import to_v3_file

    v3 = to_v3_file(payload, language=_episode_language(Path(path)), registry=_registry())
    if validate:
        validate_artifact(v3, strict=False)
    validate_artifact(v3, strict=True)
    write_json_atomic(
        Path(path),
        v3,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )


@lru_cache(maxsize=1)
def _registry() -> Any:
    from podcast_scraper.publishers import load_registry

    return load_registry()


def _episode_language(path: Path) -> Optional[str]:
    """The episode's ``episode.language`` from the metadata beside a ``.kg.json`` file."""
    if not path.name.endswith(".kg.json"):
        return None
    meta = path.with_name(path.name[: -len(".kg.json")] + ".metadata.json")
    try:
        doc = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    lang = (doc.get("episode") or {}).get("language") if isinstance(doc, dict) else None
    return str(lang) if lang else None


def read_artifact(
    path: Path,
    *,
    validate: bool = True,
    strict: bool = False,
) -> Dict[str, Any]:
    """Read a KG artifact from path.

    Args:
        path: Path to .kg.json file.
        validate: If True, run minimal (and optional strict JSON Schema) validation.
        strict: Passed to ``validate_artifact`` when validate is True.

    Returns:
        Parsed artifact dict.
    """
    with open(path, encoding="utf-8") as f:
        data = cast(Dict[str, Any], json.load(f))
    if validate:
        validate_artifact(data, strict=strict)
    return data
