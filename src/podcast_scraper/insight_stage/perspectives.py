"""GI insight nodes projected to the app's insight cards, ranked for display (moved verbatim from
``server.relational_view``, E04 S3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from podcast_scraper.server.schemas import AppInsight, AppQuote


def _node_to_app_insight(
    node: dict[str, Any], slug_by_episode: dict[str, str] | None = None
) -> AppInsight:
    """Project a GI Insight node to AppInsight (grounded).

    ``slug_by_episode`` maps episode id -> slug so a perspective insight carries its source
    episode + the supporting quote's start moment (#2032); both are annotated onto the node by
    ``topic_perspectives`` (``_episode_id`` / ``_quote_start_ms``) and are absent elsewhere. The
    supporting quote itself (``_quote_text`` / ``_quote_end_ms``, 2026-10-11) becomes the insight's
    one quote, so a quote card shows the transcript's words, not the paraphrase.
    """
    props = node.get("properties") or {}
    text = props.get("text") or props.get("title") or ""
    conf = props.get("confidence")
    itype = props.get("insight_type") or props.get("type")
    phint = props.get("position_hint")
    sal = props.get("salience")
    rnk = props.get("rank")
    rtag = props.get("routing_tag")
    tier = props.get("tier")
    ep_id = node.get("_episode_id")
    start_ms = node.get("_quote_start_ms")
    lang = node.get("_quote_translated_from")
    end_ms = node.get("_quote_end_ms")
    qtext = node.get("_quote_text")
    quotes = (
        [
            AppQuote(
                text=qtext,
                speaker=None,
                start_ms=start_ms if isinstance(start_ms, int) else None,
                end_ms=end_ms if isinstance(end_ms, int) else None,
            )
        ]
        if isinstance(qtext, str) and qtext.strip()
        else []
    )
    episode_slug = (slug_by_episode or {}).get(str(ep_id)) if ep_id else None
    return AppInsight(
        id=str(node.get("id") or ""),
        text=str(text),
        grounded=True,
        insight_type=str(itype) if isinstance(itype, str) and itype.strip() else None,
        confidence=float(conf) if isinstance(conf, (int, float)) else None,
        position_hint=str(phint) if phint is not None else None,
        salience=float(sal) if isinstance(sal, (int, float)) else None,
        rank=int(rnk) if isinstance(rnk, int) else None,
        routing_tag=str(rtag) if isinstance(rtag, str) and rtag.strip() else None,
        tier=int(tier) if isinstance(tier, int) else None,
        episode_slug=episode_slug,
        start_ms=start_ms if isinstance(start_ms, int) else None,
        quotes=quotes,
        translated_from=lang if isinstance(lang, str) and lang else None,
    )


def _rank_for_display(insights: list[AppInsight]) -> list[AppInsight]:
    """ADR-135/#1191: drop `drop`-tagged, sort by salience desc (stable for ties / pre-3.1)."""
    kept = [i for i in insights if i.routing_tag != "drop"]
    kept.sort(key=lambda i: i.salience if i.salience is not None else 0.0, reverse=True)
    return kept


def speaker_perspectives(
    root: Path,
    topic_ids: Sequence[str],
    *,
    keep_episode_ids: set[str] | None,
    slug_by_episode: dict[str, str],
) -> list[dict[str, Any]]:
    """The topics' grounded insights grouped by speaker, each group's insights projected and ranked
    for display (``topics_perspectives``' groups, ``insights`` as :class:`AppInsight`)."""
    from podcast_scraper.server.cil_queries import topics_perspectives

    groups = topics_perspectives(str(root), str(root), topic_ids, keep_episode_ids=keep_episode_ids)
    return [
        {
            **g,
            "insights": _rank_for_display(
                [_node_to_app_insight(n, slug_by_episode) for n in g["insights"]]
            ),
        }
        for g in groups
    ]
