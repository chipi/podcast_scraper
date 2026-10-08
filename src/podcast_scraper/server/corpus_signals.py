"""Corpus-scope enrichment envelopes, read and filtered for one entity (person or topic).

Shared by the consumer plane and the operator plane: the operator ``/api/corpus/entity-signals``
and the app's entity card read the same filtered projection, so it lives in the platform rather
than in either app (ADR-158).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from podcast_scraper import perf_cache

#: Files in ``enrichments/`` that are run bookkeeping, not an enricher's envelope.
_SUMMARY_FILES = {"run_summary.json"}


def parse_envelope(path: Path) -> dict[str, Any] | None:
    """Parsed envelope dict for an OK enricher, or ``None`` (absent / unparsable / not OK)."""
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("status") not in (None, "ok"):  # tolerate envelopes that omit status
        return None
    return parsed


# #1927 — grounding_rate is NOT here any more. It was per-Person and returned exactly 1.0 for all
# 689 people, because an insight is grounded exactly when a supporting quote exists and the quote
# carries the speaker: ungrounded insights have no speaker, so total == grounded for everyone.
# The metric is per-EPISODE now, so there is nothing to project onto a person card.
_PERSON_ENRICHERS = {"guest_coappearance", "topic_consensus"}
_TOPIC_ENRICHERS = {"temporal_velocity", "topic_similarity", "topic_cooccurrence_corpus"}
_ID_PREFIX_RE = re.compile(r"^(?:g:|k:|kg:)+")


_ENRICHER_SIGNALS_NS = "app_corpus_signals"


def read_corpus_signals(root: Path, wanted: set[str]) -> dict[str, Any]:
    """``{enricher_id: envelope data}`` for the *wanted* corpus enrichers that ran OK.

    Filename-first: the enrichment filename **is** the enricher id for every current enricher, so we
    skip the glob's non-wanted files *before* parsing them. That is the fix for the old behaviour of
    JSON-parsing the multi-MB ``topic_cooccurrence_corpus.json`` on a trending-topics request
    just to discard it — now only the wanted files are read. The envelope's ``enricher_id`` keys
    the result, so discovery semantics match the app's ``/corpus/enrichment``.
    """
    enrich_dir = root / "enrichments"
    out: dict[str, Any] = {}
    if not enrich_dir.is_dir():
        return out
    for path in sorted(enrich_dir.glob("*.json")):
        if path.name in _SUMMARY_FILES:
            continue
        if path.stem not in wanted:  # filename-first — do not parse files we would only discard
            continue
        parsed = parse_envelope(path)
        if parsed is None or parsed.get("data") is None:
            continue
        enricher_id = str(parsed.get("enricher_id") or path.stem)
        if enricher_id in wanted:
            out[enricher_id] = parsed["data"]
    return out


def corpus_signals(root: Path, wanted: set[str]) -> dict[str, Any]:
    """:func:`read_corpus_signals`, cached by corpus mtime (bumps on ingest).

    Keyed by ``(root, wanted)`` so trending-topics and entity-signals keep separate warmed subsets;
    the parsed envelopes are held once per ingest instead of re-read+re-parsed on every request.
    """
    # root is the platform corpus (corpus_root_or_503) or a _resolve_corpus-validated ?path.
    # codeql[py/path-injection] -- root validated by corpus_root_or_503 / _resolve_corpus (Type 1).
    resolved_root = str(Path(root).resolve())
    key = f"{resolved_root}::{','.join(sorted(wanted))}"
    signals: dict[str, Any] = perf_cache.get_or_compute(
        _ENRICHER_SIGNALS_NS,
        key,
        perf_cache.corpus_mtime(root),
        lambda: read_corpus_signals(root, wanted),
    )
    return signals


def norm_entity_id(value: Any) -> str:
    """Drop graph id prefixes (``g:`` / ``k:`` / ``kg:``) so ids compare like the client norm."""
    return _ID_PREFIX_RE.sub("", str(value or ""))


def filtered_entity_signals(root: Path, kind: str, id: str) -> dict[str, Any]:
    """Corpus enrichment signals filtered to ONE person/topic (the entity-card projection).

    Every corpus-scope enricher list is pre-filtered to the rows that touch the focused entity, so a
    caller reads a few KB instead of the whole (up to ~25 MB) corpus payload. Shared by the consumer
    ``/api/app/corpus/entity-signals`` (single platform corpus) and the operator
    ``/api/corpus/entity-signals`` (``?path=``-scoped viewer) — same filter, different root.
    """
    self_id = norm_entity_id(id)
    raw = corpus_signals(root, _PERSON_ENRICHERS if kind == "person" else _TOPIC_ENRICHERS)
    out: dict[str, Any] = {}

    def _hit(*ids: Any) -> bool:
        return any(norm_entity_id(i) == self_id for i in ids)

    def _filtered(enricher: str, list_key: str, keep: Any) -> None:
        env = raw.get(enricher)
        if not isinstance(env, dict):
            return
        items_any = env.get(list_key)
        if not isinstance(items_any, list):
            return
        kept = [it for it in items_any if isinstance(it, dict) and keep(it)]
        if kept:
            out[enricher] = {list_key: kept}

    if kind == "person":
        # grounding_rate deliberately absent — see _PERSON_ENRICHERS (#1927).
        _filtered(
            "guest_coappearance",
            "pairs",
            lambda r: _hit(r.get("person_a_id"), r.get("person_b_id")),
        )
        _filtered(
            "topic_consensus",
            "consensus",
            lambda r: _hit(r.get("person_a_id"), r.get("person_b_id")),
        )
    else:
        _filtered("temporal_velocity", "topics", lambda r: _hit(r.get("topic_id")))
        _filtered("topic_similarity", "topics", lambda r: _hit(r.get("topic_id")))
        _filtered(
            "topic_cooccurrence_corpus",
            "pairs",
            lambda r: _hit(r.get("topic_a_id"), r.get("topic_b_id")),
        )

    return out


__all__ = [
    "corpus_signals",
    "filtered_entity_signals",
    "norm_entity_id",
    "parse_envelope",
    "read_corpus_signals",
]
