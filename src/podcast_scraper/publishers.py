"""The publisher registry: publishers are organisations across media (ADR-180 rule 6, T-12).

Held in public configuration (``config/publishers.yaml``). Each publisher has an organisation id
(``org:{slug}``, a KG ``Organization`` node, so a publisher that also publishes a podcast is one
organisation across both), a name, its web domains, external identifiers kept side by side
and never merged into one key, and the KG ``Source`` ids it publishes (``podcast:...`` or
``source:...``, one list for every type). A source points at its publisher through
``PUBLISHED_BY`` (KG 3.0), drawn from this file only.

A registry id shares the ``org:{slug}`` namespace with the organisation nodes NER writes, so an NER
organisation with the identical slug is the same organisation (the operator, 2026-11-18; ADR-180
rule 6). Nothing else joins them: never a fuzzy or partial name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

#: Registry ids: KG ``Organization`` nodes, the namespace NER's organisations use too.
ID_PREFIX = "org:"
_SOURCE_PREFIXES = ("podcast:", "source:")

#: The registry beside the other public configuration.
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "publishers.yaml"


class PublisherRegistryError(ValueError):
    """The registry file breaks a rule the loader enforces."""


@dataclass(frozen=True)
class Publisher:
    """One publisher: an organisation, its domains, external ids and the sources it publishes."""

    id: str
    name: str
    domains: Tuple[str, ...] = ()
    external_ids: Mapping[str, str] = field(default_factory=dict)
    #: The KG ``Source`` ids it publishes (``PUBLISHED_BY``), any type.
    sources: Tuple[str, ...] = ()

    def served(self, host: Optional[str] = None) -> Dict[str, str]:
        """``{id, name, domain}`` as a surface shows it: the matched domain, else the first."""
        domain = _host(host) if host and _host(host) in self.domains else None
        return {"id": self.id, "name": self.name, "domain": domain or next(iter(self.domains), "")}


def _host(value: str) -> str:
    host = value.strip().lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


@dataclass(frozen=True)
class Registry:
    """The loaded registry; lookups by id, by source id and by domain."""

    publishers: Tuple[Publisher, ...] = ()

    def by_id(self, publisher_id: str) -> Optional[Publisher]:
        """The publisher with ``publisher_id``, or ``None``."""
        return next((p for p in self.publishers if p.id == publisher_id), None)

    def for_source(self, source_id: str) -> Optional[Publisher]:
        """The publisher of the KG ``Source`` ``source_id``, or ``None``."""
        return next((p for p in self.publishers if source_id in p.sources), None)

    def for_domain(self, host: str) -> Optional[Publisher]:
        """The publisher of a news item's canonical URL host: a listed domain matches itself and
        its subdomains (``m.bbc.com``), the longest match wins, and no match is no publisher,
        never a guess."""
        want = _host(host)
        best: Optional[Tuple[int, Publisher]] = None
        for p in self.publishers:
            for domain in p.domains:
                if (want == domain or want.endswith("." + domain)) and (
                    best is None or len(domain) > best[0]
                ):
                    best = (len(domain), p)
        return best[1] if best else None


def _publisher(raw: Any, where: str) -> Publisher:
    if not isinstance(raw, Mapping):
        raise PublisherRegistryError(f"{where}: an entry is not a mapping")
    pid, name = str(raw.get("id") or ""), str(raw.get("name") or "").strip()
    if not pid.startswith(ID_PREFIX) or len(pid) <= len(ID_PREFIX):
        raise PublisherRegistryError(f"{where}: id {pid!r} is not a publisher id ({ID_PREFIX}...)")
    if not name:
        raise PublisherRegistryError(f"{where}: {pid} has no name")
    sources = raw.get("sources") or []
    if not isinstance(sources, list):
        raise PublisherRegistryError(f"{where}: {pid} sources is not a list of Source ids")
    for sid in sources:
        if not str(sid).startswith(_SOURCE_PREFIXES):
            raise PublisherRegistryError(f"{where}: {pid} source {sid!r} is not a KG Source id")
    return Publisher(
        id=pid,
        name=name,
        domains=tuple(_host(str(d)) for d in raw.get("domains") or ()),
        external_ids={str(k): str(v) for k, v in (raw.get("external_ids") or {}).items()},
        sources=tuple(str(s) for s in sources),
    )


def validate(publishers: List[Publisher]) -> None:
    """An id, a domain, a source or an external id claimed by two publishers is refused."""
    seen: Dict[Tuple[str, str], str] = {}
    for p in publishers:
        keys = [("id", p.id)] + [("domain", d) for d in p.domains]
        keys += [("source", s) for s in p.sources]
        keys += [(f"external id {k}", v) for k, v in p.external_ids.items()]
        for key in keys:
            if key in seen:
                raise PublisherRegistryError(
                    f"{key[0]} {key[1]!r} is claimed by both {seen[key]} and {p.id}"
                )
            seen[key] = p.id


#: Names another registry file (a test corpus's invented publishers); unset, the shipped one.
ENV_PATH = "PODCAST_PUBLISHERS_FILE"


def registry_path() -> Path:
    """The registry file in use: ``$PODCAST_PUBLISHERS_FILE`` when set, else the shipped one."""
    override = os.environ.get(ENV_PATH, "").strip()
    return Path(override) if override else DEFAULT_PATH


def load_registry(path: Optional[Path] = None) -> Registry:
    """The registry at ``path`` (default :func:`registry_path`); a missing file is empty."""
    path = path if path is not None else registry_path()
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
