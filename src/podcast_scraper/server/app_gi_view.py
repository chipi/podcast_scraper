"""Project a ``*.gi.json`` artifact to the consumer insights shape (#1068).

Pure functions over the parsed GIL artifact dict (nodes + edges, RFC-049/097) — no
HTTP, no disk. Defensive: malformed nodes/edges are skipped rather than raising, so
varied real-corpus artifacts never break the endpoint.
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.server.schemas import AppInsight, AppQuote


def _opt_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _opt_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _opt_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _person_label(person_id: Any) -> str | None:
    """``person:jane-doe`` -> ``jane-doe`` (best-effort, no KG name lookup here)."""
    if isinstance(person_id, str) and person_id.strip():
        pid = person_id.strip()
        return pid.split(":", 1)[1] if ":" in pid else pid
    return None


def insights_from_gi(artifact: Any, *, limit: int | None = None) -> list[AppInsight]:
    """Return surfaceable insights from a GI artifact dict, ranked for display.

    ADR-133/#1191: insights are sorted by ``salience`` descending (the route-and-tag ranking) so a
    surface can take the first N; ``routing_tag == "drop"`` insights are excluded (belt-and-
    suspenders with the value gate). ``limit`` caps the result to the top-N after sorting (e.g.
    ``gi_surface_default_limit``); ``None`` returns all. Ties and pre-3.1 artifacts (no
    ``salience``) fall back to extraction order, so the projection is unchanged for old corpora.
    """
    if not isinstance(artifact, dict):
        return []
    nodes = artifact.get("nodes")
    if not isinstance(nodes, list):
        return []
    edges = artifact.get("edges")
    edges = edges if isinstance(edges, list) else []

    quotes: dict[Any, dict] = {}
    for node in nodes:
        if isinstance(node, dict) and node.get("type") == "Quote":
            props = node.get("properties")
            quotes[node.get("id")] = props if isinstance(props, dict) else {}

    supported: dict[Any, list[Any]] = {}  # insight_id -> [quote_id]
    spoken_by: dict[Any, Any] = {}  # quote_id -> person_id
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        etype, frm, to = edge.get("type"), edge.get("from"), edge.get("to")
        if etype == "SUPPORTED_BY":
            supported.setdefault(frm, []).append(to)
        elif etype == "SPOKEN_BY":
            spoken_by[frm] = to

    out: list[AppInsight] = []
    for node in nodes:
        if not isinstance(node, dict) or node.get("type") != "Insight":
            continue
        props = node.get("properties")
        props = props if isinstance(props, dict) else {}
        text = _opt_str(props.get("text"))
        if text is None:
            continue

        # AN UNATTRIBUTED STANCE IS NOT A STANCE. It is a floating opinion that nobody holds and
        # nobody can disagree with, so it does not belong on a surface — whatever the classifier
        # called it. GI marks these `surfaceable: False` when the speaking voice is not a named
        # person (an advertisement, a voice we failed to name, or the vox-pop of a narrated piece
        # that nobody names).
        #
        # They stay in the artifact: a FACT is still a fact, and the corpus needs them for CONNECT —
        # story threads across episodes never needed a speaker. This gate is about what we PUBLISH
        # as somebody's insight, not about what we keep.
        if props.get("surfaceable") is False:
            continue
        # ADR-133/#1191: a `drop`-tagged insight (FILLER) is not published on any surface.
        if _opt_str(props.get("routing_tag")) == "drop":
            continue
        insight_id = node.get("id")

        quote_models: list[AppQuote] = []
        for quote_id in supported.get(insight_id, []):
            qp = quotes.get(quote_id)
            if qp is None:
                continue
            qtext = _opt_str(qp.get("text"))
            if qtext is None:
                continue
            speaker = (
                _opt_str(qp.get("speaker_name"))
                or _opt_str(qp.get("speaker_id"))
                or _person_label(spoken_by.get(quote_id))
            )
            quote_models.append(
                AppQuote(
                    text=qtext,
                    speaker=speaker,
                    char_start=_opt_int(qp.get("char_start")),
                    char_end=_opt_int(qp.get("char_end")),
                    start_ms=_opt_int(qp.get("timestamp_start_ms")),
                    end_ms=_opt_int(qp.get("timestamp_end_ms")),
                )
            )

        grounded_prop = props.get("grounded")
        grounded = bool(grounded_prop) if isinstance(grounded_prop, bool) else bool(quote_models)
        out.append(
            AppInsight(
                id=str(insight_id) if insight_id is not None else "",
                text=text,
                grounded=grounded,
                insight_type=_opt_str(props.get("insight_type")),
                confidence=_opt_float(props.get("confidence")),
                position_hint=_opt_str(props.get("position_hint")),
                salience=_opt_float(props.get("salience")),
                rank=_opt_int(props.get("rank")),
                routing_tag=_opt_str(props.get("routing_tag")),
                tier=_opt_int(props.get("tier")),
                quotes=quote_models,
            )
        )

    # ADR-133/#1191: rank for display. Stable sort by salience desc keeps extraction order for ties
    # and for pre-3.1 artifacts (salience None -> 0.0), so old corpora project identically.
    out.sort(key=lambda ins: ins.salience if ins.salience is not None else 0.0, reverse=True)
    if limit is not None and limit >= 0:
        out = out[:limit]
    return out
