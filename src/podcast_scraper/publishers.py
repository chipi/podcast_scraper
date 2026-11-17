"""The publisher registry: publishers are organisations across media (ADR-180 rule 6, T-12).

Held in public configuration (``config/publishers.yaml``). Each publisher has an organisation id
(``org:{slug}``, the KG organisation node, so a publisher that also publishes a podcast is one
organisation across both), a name, its web domains, external identifiers kept side by side and
never merged into one key, and the sources it publishes, per content type. A source points at its
publisher through ``PUBLISHED_BY`` (KG 3.0), drawn from this file only: organisation nodes found
by NER are never joined to a publisher automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

#: The registry beside the other public configuration.
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "publishers.yaml"


class PublisherRegistryError(ValueError):
    """The registry file breaks a rule the loader enforces."""


@dataclass(frozen=True)
class Publisher:
    id: str
    name: str
    domains: Tuple[str, ...] = ()
    external_ids: Mapping[str, str] = field(default_factory=dict)
    #: content type -> the source ids it publishes under that type.
    sources: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def served(self, host: Optional[str] = None) -> Dict[str, str]:
        """``{id, name, domain}`` as a surface shows it: the matched domain, else the first."""
        domain = _host(host) if host and _host(host) in self.domains else None
        return {"id": self.id, "name": self.name, "domain": domain or next(iter(self.domains), "")}


def _host(value: str) -> str:
    host = value.strip().lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


@dataclass(frozen=True)
class Registry:
    publishers: Tuple[Publisher, ...] = ()

    def by_id(self, publisher_id: str) -> Optional[Publisher]:
        return next((p for p in self.publishers if p.id == publisher_id), None)

    def for_source(self, content_type: str, source_id: str) -> Optional[Publisher]:
        """The publisher of ``source_id`` under ``content_type``, or ``None``."""
        return next(
            (p for p in self.publishers if source_id in p.sources.get(content_type, ())), None
        )

    def for_domain(self, host: str) -> Optional[Publisher]:
        """The publisher whose domain is ``host``, or a parent of it (``news.bbc.co.uk``)."""
        want = _host(host)
        for p in self.publishers:
            for domain in p.domains:
                if want == domain or want.endswith("." + domain):
                    return p
        return None


def _publisher(raw: Any, where: str) -> Publisher:
    if not isinstance(raw, Mapping):
        raise PublisherRegistryError(f"{where}: an entry is not a mapping")
    pid, name = str(raw.get("id") or ""), str(raw.get("name") or "").strip()
    if not pid.startswith("org:") or len(pid) <= len("org:"):
        raise PublisherRegistryError(f"{where}: id {pid!r} is not an organisation id (org:...)")
    if not name:
        raise PublisherRegistryError(f"{where}: {pid} has no name")
    sources = raw.get("sources") or {}
    if not isinstance(sources, Mapping):
        raise PublisherRegistryError(f"{where}: {pid} sources is not a mapping")
    return Publisher(
        id=pid,
        name=name,
        domains=tuple(_host(str(d)) for d in raw.get("domains") or ()),
        external_ids={str(k): str(v) for k, v in (raw.get("external_ids") or {}).items()},
        sources={str(t): tuple(str(s) for s in ids or ()) for t, ids in sources.items()},
    )


def validate(publishers: List[Publisher]) -> None:
    """One id, one domain and one source per publisher: each claimed twice is refused."""
    seen: Dict[Tuple[str, str], str] = {}
    for p in publishers:
        keys = [("id", p.id)] + [("domain", d) for d in p.domains]
        keys += [(f"source {t}", s) for t, ids in p.sources.items() for s in ids]
        for key in keys:
            if key in seen:
                raise PublisherRegistryError(
                    f"{key[0]} {key[1]!r} is claimed by both {seen[key]} and {p.id}"
                )
            seen[key] = p.id


def load_registry(path: Path = DEFAULT_PATH) -> Registry:
    """The registry at ``path``; a missing file is an empty registry."""
    if not path.is_file():
        return Registry()
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = doc.get("publishers") if isinstance(doc, Mapping) else None
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise PublisherRegistryError(f"{path}: publishers is not a list")
    publishers = [_publisher(r, str(path)) for r in raw]
    validate(publishers)
    return Registry(tuple(publishers))
