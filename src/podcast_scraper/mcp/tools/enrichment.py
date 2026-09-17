"""Enrichment + speaker MCP tools — signals the pipeline persists but MCP never exposed.

- corpus/episode enrichment envelopes (RFC-088): the ``enrichments/`` signals the web
  ``/api/*/enrichment`` routes serve (sentiment, density, consensus, similarity, …).
- episode speaker roster / talk-share: the ``.speakers.diagnostics.json`` the pipeline
  writes next to each transcript — who spoke, talk-share %, host/guest, unattributed share.
  This has NO HTTP route at all, so it is genuinely new read capability.

All read-only file reads under the corpus dir; mirror the route logic so MCP and the web
UI report the same numbers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List

from ..context import CorpusContext

_ENRICHER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_]+$")
_METADATA_SUFFIX = ".metadata.json"


def _envelope_data(path: Path) -> Any:
    """The ``data`` block of an enrichment envelope JSON, or None."""
    import json

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return parsed.get("data") if isinstance(parsed, dict) else None


def corpus_enrichment_signals(ctx: CorpusContext) -> Dict[str, Any]:
    """Corpus-scope enrichment signals (RFC-088) — the ``enrichments/`` aggregates.

    One call for the corpus-wide envelopes: topic_similarity, topic_consensus,
    temporal_velocity, grounding_rate, guest_coappearance, topic_cooccurrence, etc. Same
    data as ``GET /api/corpus/enrichment``. Use as a capability probe + corpus aggregates.
    """
    enrich_dir = Path(ctx.corpus_dir) / "enrichments"
    signals: Dict[str, Any] = {}
    if enrich_dir.is_dir():
        for path in sorted(enrich_dir.glob("*.json")):
            enricher_id = path.stem
            if not _ENRICHER_ID_PATTERN.match(enricher_id):
                continue
            data = _envelope_data(path)
            if data is not None:
                signals[enricher_id] = data
    note = "" if signals else "no corpus enrichments/ dir (enrichment stage not run?)"
    return {"scope": "corpus", "signals": signals, "note": note}


def episode_enrichment_signals(ctx: CorpusContext, metadata_path: str) -> Dict[str, Any]:
    """Per-episode enrichment signals (RFC-088) — sentiment, density, co-occurrence.

    ``metadata_path`` from a ``list_episodes`` / ``search_corpus`` hit. Episode-scope
    envelopes live at ``<metadata_dir>/enrichments/<stem>.<enricher_id>.json``. Same data as
    ``GET /api/app/episodes/{slug}/enrichment``. Supplements ``episode_detail`` with
    pacing/sentiment context.
    """
    import glob as globmod

    meta_path = Path(ctx.corpus_dir) / metadata_path
    enrich_dir = meta_path.parent / "enrichments"
    signals: Dict[str, Any] = {}
    if enrich_dir.is_dir() and meta_path.name.endswith(_METADATA_SUFFIX):
        stem = meta_path.name[: -len(_METADATA_SUFFIX)]
        for p in sorted(globmod.glob(globmod.escape(str(enrich_dir / stem)) + ".*.json")):
            path = Path(p)
            enricher_id = path.name[len(stem) + 1 : -len(".json")]
            if not _ENRICHER_ID_PATTERN.match(enricher_id):
                continue
            data = _envelope_data(path)
            if data is not None:
                signals[enricher_id] = data
    note = "" if signals else "no per-episode enrichments for this metadata_path"
    return {"scope": "episode", "episode": metadata_path, "signals": signals, "note": note}


def _roster_from_record(content: Any, diag: Any) -> List[Dict[str, Any]]:
    """Who is on the episode, from the SPEAKER RECORD, with talk time from diagnostics (#2075).

    ``content.speakers`` is the one record every surface is written from. Each entry says whether a
    voice was matched to that person (``placed``); talk time is summed over the diagnostics voices
    the entry names. Older records carry no voice ids, so a voice is matched by its resolved name.
    A person only named (``placed: false``) is listed — marked — with no talk time: they are on the
    record, and nothing here may present them as someone who spoke.
    """
    voices = (
        [v for v in ((diag or {}).get("voices") or []) if isinstance(v, dict)]
        if isinstance(diag, dict)
        else []
    )
    total = sum(float(v.get("talk_time_s") or 0.0) for v in voices) or 0.0
    out: List[Dict[str, Any]] = []
    entries = (content or {}).get("speakers") if isinstance(content, dict) else None
    for sp in entries or []:
        if not isinstance(sp, dict) or not sp.get("name"):
            continue
        placed = sp.get("placed")
        ids = [str(v) for v in (sp.get("voices") or [])]
        if placed is False:
            mine: List[Dict[str, Any]] = []
        elif ids:
            mine = [v for v in voices if str(v.get("voice")) in ids]
        else:
            mine = [
                v for v in voices if v.get("resolved_name") == sp.get("name") and v.get("named")
            ]
        talk = sum(float(v.get("talk_time_s") or 0.0) for v in mine)
        out.append(
            {
                "name": sp.get("name"),
                "role": sp.get("role"),
                "placed": placed,
                "voices": ids or [str(v.get("voice")) for v in mine],
                "source": sp.get("source"),
                "talk_time_s": round(talk, 1),
                "talk_share": round(talk / total, 4) if total else None,
            }
        )
    return out


def episode_speaker_roster(ctx: CorpusContext, metadata_path: str) -> Dict[str, Any]:
    """Speaker roster + talk-share for one episode — who spoke, %, host/guest.

    ``speakers`` is the episode's speaker RECORD (``content.speakers``, #2075): every person on the
    episode, whether a voice was matched to them (``placed``), their role and talk share. People
    only named — by the feed, the show notes, the pre-listening guess — are listed with
    ``placed: false`` and no talk time; they did not speak as far as the audio shows.

    ``diagnostics`` is the explanation of how the names were chosen (per-voice method, voice types,
    exposed metrics), read from ``.speakers.diagnostics.json``. Its ``tried`` block is omitted: it
    holds the raw candidate lists the roster was given, including the pre-listening guess, and
    returning it here put a second, contradictory answer to "who spoke" in front of the caller.
    Distinct from ``who_said_about_topic`` / ``person_positions`` (knowledge-graph person queries).
    """
    from ...server.app_content_source import transcript_corpus_relpath, transcript_relpath
    from ...server.app_corpus_access import load_json_artifact

    root = Path(ctx.corpus_dir)
    meta = load_json_artifact(root, metadata_path)
    content = (meta or {}).get("content") if isinstance(meta, dict) else None
    transcript_rel = transcript_relpath(content) if isinstance(content, dict) else None
    if not transcript_rel:
        return {
            "episode": metadata_path,
            "speakers": _roster_from_record(content, None),
            "diagnostics": None,
            "note": "no transcript for episode",
        }
    transcript_corpus_rel = transcript_corpus_relpath(metadata_path, transcript_rel)
    base, _ = os.path.splitext(transcript_corpus_rel)
    diag = load_json_artifact(root, base + ".speakers.diagnostics.json")
    note = (
        "" if diag else "no persisted speaker diagnostics (diarization off or pre-diagnostics run)"
    )
    explanation = (
        {k: v for k, v in diag.items() if k != "tried"} if isinstance(diag, dict) else diag
    )
    return {
        "episode": metadata_path,
        "speakers": _roster_from_record(content, diag),
        "diagnostics": explanation,
        "note": note,
    }
