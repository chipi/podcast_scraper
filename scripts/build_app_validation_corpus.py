#!/usr/bin/env python3
"""Build a committed, deterministic validation corpus for the CONSUMER app.

This is the consumer "Learning Player" analogue of
``scripts/build_synthetic_validation_corpus.py`` (the viewer's Tier-3 corpus
generator). It deterministically constructs a schema-shaped corpus from the
checked-in text fixtures — ``tests/fixtures/rss/*.xml`` +
``tests/fixtures/transcripts/<FIXTURES_VERSION>/*.txt`` — with **no pipeline and
no ML**. The output is committed under ``tests/fixtures/app-validation-corpus``
so the app's Playwright e2e runs against a stable, version-pinned corpus the
consumer API can serve directly (no generate-at-setup step).

It REUSES the viewer generator's proven construction helpers (``build_gi``,
``build_kg``, ``parse_diarized_segments``, ``format_screenplay_with_offsets``,
``read_transcript_excerpts``, ``slug``, ``stable_feed_id``,
``parse_rss_feed_metadata``) so the GI/KG artifacts can't drift from what the
backend readers expect — exactly the same artifact shapes the viewer corpus
uses (``app_gi_view`` / ``app_kg_view`` read them defensively).

It differs from the viewer corpus only in the *on-disk layout* and the *extra
fields the consumer readers require*, derived from studying the consumer readers:

* Run-dir layout
  ``feeds/<show>/run_<tag>/metadata/<ep>.{metadata,gi,kg,bridge}.json`` and
  ``feeds/<show>/run_<tag>/transcripts/<ep>.{txt,segments.json}`` — what
  ``corpus_catalog`` / ``corpus_scope.discover_metadata_files`` expect. The
  ``.bridge.json`` sidecar (topic identities mirroring the GI) is what
  ``iter_cil_episode_bundles`` pairs with the GI so the #1146 perspectives
  surface reads real speaker-attributed insights (see ``_bridge_from_gi``).
* ``metadata.content.transcript_file_path`` stored **run-relative**
  (``transcripts/<ep>.txt``) — ``app_content_source.transcript_corpus_relpath``
  joins it onto the run dir.
* RAW canonical ``<ep>.segments.json`` (NOT ``.adfree.segments.json``) — the
  consumer player streams the original audio, so ``segments_view`` prefers raw.
* ``metadata.content.media_url`` (+ ``media_type``) — ``app_audio_bridge`` /
  the audio-source route read these for direct playback.
* ``search/topic_clusters.json`` with multi-member clusters — the Profile
  interests picker (``top_clusters_by_member_count``) and discover ranking read
  it.

Usage::

    python scripts/build_app_validation_corpus.py \\
        [--rss-dir tests/fixtures/rss] \\
        [--transcripts-dir tests/fixtures/transcripts/<version>] \\
        [--output tests/fixtures/app-validation-corpus] \\
        [--max-feeds 3] [--max-episodes-per-feed 2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Reuse the viewer generator's construction helpers verbatim (do NOT modify that
# script's output). Importing as a module keeps GI/KG artifact shapes identical to
# the viewer corpus, so the backend readers can't drift between the two.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from build_synthetic_validation_corpus import (  # noqa: E402
    build_gi,
    build_kg,
    format_screenplay_with_offsets,
    parse_diarized_segments,
    parse_rss_feed_metadata,
    slug,
)

# Three clearly-distinct consumer shows (canonical RSS per podcast). Chosen so the
# app surfaces (Home/What's-new, Player insights+entities, Profile interests, Queue)
# all have real data. Mirrors the show names the app e2e specs reference.
APP_SHOWS: list[tuple[str, str]] = [
    ("p05_investing", "p05"),  # Long Horizon Notes — investing
    ("p02_software", "p02"),  # Practical Systems — software/SRE
    ("p03_scuba", "p03"),  # Below the Surface — scuba diving
    # #1148: promoted into the wired set so the risk-management ↔ systems-thinking
    # overlap web spans all 9 shows (perspectives / clusters / discovery / recs).
    ("p04_photo", "p04"),  # Frame & Light — photography
    ("p01_mtb", "p01"),  # Singletrack Sessions — mountain biking
    ("p07_sustainability", "p07"),  # The Long View — sustainability
    # These three carry the detection-shape edge cases (low-grounding, NER-evading
    # host, cross-show recurring guests). Their RSS fixtures are themed for other
    # shows, so SHOW_META_OVERRIDE restores the v3 identity below.
    ("p06_edge_cases", "p06"),  # The Drift — low-grounding dialogue
    ("p08_solar", "p08"),  # Public Hour — NPR-shape / zero_host_ner
    ("p09_biohacking", "p09"),  # Cross-Show — recurring-guest web
]

# p06/p08/p09 RSS fixtures are themed for other shows (edge_cases / solar /
# biohacking); override the display metadata to the v3 show identity when wiring
# them as consumer shows (#1148).
SHOW_META_OVERRIDE: dict[str, dict[str, str]] = {
    "p06": {
        "display_title": "The Drift",
        "description": "Dialogue-heavy, meandering long-form interviews.",
    },
    "p08": {
        "display_title": "Public Hour",
        "description": "NPR-shape public-radio conversations.",
    },
    "p09": {
        "display_title": "Cross-Show",
        "description": "Recurring guests from other shows revisit their positions.",
    },
}


def _publish_date_for(ep_label: str, gt_dir: Path) -> str | None:
    """Publish date from the v3 ground truth (CORPUS_EPOCH + publish_offset_days).

    Returns the authored ``publish_date`` (``YYYY-MM-DD``) when the episode set
    ``publish_offset_days`` (#1148 time spread), else ``None`` so the caller
    falls back to the legacy single-month scheme. This is what gives
    ``temporal_velocity`` a real multi-month signal.
    """
    gt = gt_dir / f"{ep_label}.json"
    if not gt.is_file():
        return None
    try:
        pd = json.loads(gt.read_text(encoding="utf-8")).get("publish_date")
    except (OSError, ValueError):
        return None
    return pd if isinstance(pd, str) else None


def _canonicalize_persons_in(doc: dict[str, Any]) -> None:
    """Rewrite one artifact's ``person:speaker-NN`` ids to name-based ids, in place."""
    idmap: dict[str, str] = {}
    for n in doc.get("nodes", []):
        if n.get("type") != "Person":
            continue
        name = str((n.get("properties") or {}).get("name") or "").strip()
        if name:
            new_id = f"person:{slug(name)}"
            idmap[str(n.get("id"))] = new_id
            n["id"] = new_id
    for e in doc.get("edges", []):
        if e.get("from") in idmap:
            e["from"] = idmap[e["from"]]
        if e.get("to") in idmap:
            e["to"] = idmap[e["to"]]
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for n in doc.get("nodes", []):
        if n.get("type") == "Person" and str(n.get("id")) in seen:
            continue
        if n.get("type") == "Person":
            seen.add(str(n.get("id")))
        kept.append(n)
    doc["nodes"] = kept


def _stamp_publish_date(publish_iso: str, *docs: dict[str, Any]) -> None:
    """Set each artifact's Episode node ``properties.publish_date`` (#1148).

    temporal_velocity / trending read ``publish_date(kg)`` from the Episode node;
    the deterministic builder left it unset, so every topic fell outside the
    date window and the temporal signal was flat. Stamping the authored date
    (the #1148 varied 2024→now schedule) lights it up.
    """
    for doc in docs:
        for n in doc.get("nodes", []):
            if n.get("type") == "Episode":
                n.setdefault("properties", {})["publish_date"] = publish_iso


def _bridge_from_gi(gi: dict[str, Any], episode_id: str) -> dict[str, Any]:
    """Minimal CIL bridge asserting every Topic the GI has an insight about.

    ``topic_perspectives`` (the #1146 perspectives surface) walks sibling
    ``.bridge.json`` files via ``iter_cil_episode_bundles`` and only considers
    an episode when the bridge's identities intersect the target topic. The app
    corpus never wrote bridges, so perspectives read empty even though the GI
    carried the Insight→ABOUT→Topic chains. Identities mirror the GI's Topic
    nodes so the bridge can't drift from what the insights actually reference.
    """
    identities: list[dict[str, str]] = []
    seen: set[str] = set()
    for n in gi.get("nodes", []):
        if n.get("type") != "Topic":
            continue
        tid = str(n.get("id") or "")
        if not tid or tid in seen:
            continue
        seen.add(tid)
        props = n.get("properties") or {}
        name = str(props.get("name") or props.get("label") or props.get("title") or "").strip()
        if not name:
            name = tid.split(":", 1)[-1].replace("-", " ").title()
        identities.append({"id": tid, "display_name": name})
    return {"schema_version": "1", "episode_id": episode_id, "identities": identities}


def _load_pod_guests(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Map ``pod_id -> {guest_key: canonical_name}`` from the v3 manifest.

    The authored claims reference guests by key (``"Daniel"``); the GI uses
    canonical ``person:<name-slug>`` ids. This bridges the two.
    """
    out: dict[str, dict[str, str]] = {}
    if not manifest_path.is_file():
        return out
    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    for pod in doc.get("podcasts", []):
        pid = str(pod.get("pod_id") or "")
        if pid:
            out[pid] = {
                str(g.get("key")): str(g.get("canonical_name") or "")
                for g in pod.get("guests", [])
                if g.get("key")
            }
    return out


def _inject_authored_claims(
    gi: dict[str, Any], ep_label: str, gt_dir: Path, name_by_key: dict[str, str]
) -> None:
    """Inject authored topic/contradiction claims into the GI as real Insights.

    The deterministic extractor takes ``sentences[:3]`` — the greetings — so the
    engineered claims (perspectives, the diversify-vs-concentrate opposition)
    never become Insights. This adds each authored claim as an Insight with the
    full chain the enrichers read: SUPPORTED_BY→Quote→SPOKEN_BY→Person and
    ABOUT→Topic. Makes multi-perspective (#1146) + nli/disagreement (#1144) work
    on the real corpus (#1148).
    """
    gt = gt_dir / f"{ep_label}.json"
    if not gt.is_file():
        return
    try:
        doc = json.loads(gt.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    claims: list[tuple[str, str, str, bool]] = []
    for tc in doc.get("topic_claims", []):
        claims.append(
            (
                str(tc.get("speaker", "")),
                str(tc.get("topic_id", "")),
                str(tc.get("claim", "")),
                bool(tc.get("grounded", True)),
            )
        )
    for cc in doc.get("contradiction_claims", []):
        tid = str(cc.get("topic_id", ""))
        claims.append((str(cc.get("speaker_a", "")), tid, str(cc.get("claim_a", "")), True))
        claims.append((str(cc.get("speaker_b", "")), tid, str(cc.get("claim_b", "")), True))
    if not claims:
        return
    nodes = gi.setdefault("nodes", [])
    edges = gi.setdefault("edges", [])
    ids = {str(n.get("id")) for n in nodes}
    for i, (key, tid, text, grounded) in enumerate(claims):
        name = name_by_key.get(key)
        if not (name and tid and text):
            continue
        pid = f"person:{slug(name)}"
        if pid not in ids:
            nodes.append({"id": pid, "type": "Person", "properties": {"name": name}})
            ids.add(pid)
        if tid not in ids:
            label = tid.replace("topic:", "").replace("-", " ")
            nodes.append({"id": tid, "type": "Topic", "properties": {"name": label}})
            ids.add(tid)
        iid, qid = f"insight:authored-{ep_label}-{i}", f"quote:authored-{ep_label}-{i}"
        nodes.append(
            {
                "id": iid,
                "type": "Insight",
                "properties": {
                    "text": text,
                    "grounded": grounded,
                    "insight_type": "claim",
                    "position_hint": 0.5,
                },
            }
        )
        nodes.append({"id": qid, "type": "Quote", "properties": {"text": text}})
        edges.append({"from": iid, "to": qid, "type": "SUPPORTED_BY"})
        edges.append({"from": qid, "to": pid, "type": "SPOKEN_BY"})
        edges.append({"from": iid, "to": tid, "type": "ABOUT"})
        edges.append({"from": iid, "to": pid, "type": "MENTIONS_PERSON"})


def _vary_grounding(gi: dict[str, Any], ep_label: str, gt_dir: Path) -> None:
    """Make low-grounding episodes actually ungrounded, so grounding_rate discriminates.

    The builder marks every Insight ``grounded: True`` → a flat corpus rate of
    1.0 with no signal. For episodes tagged ``low_grounding_dialogue`` (the p06
    "Drift" show), flip alternate Insights to ``grounded: False`` so the enricher
    sees real variation (#1148).
    """
    gt = gt_dir / f"{ep_label}.json"
    if not gt.is_file():
        return
    try:
        modes = json.loads(gt.read_text(encoding="utf-8")).get("failure_modes", [])
    except (OSError, ValueError):
        return
    if "low_grounding_dialogue" not in modes:
        return
    insights = [n for n in gi.get("nodes", []) if n.get("type") == "Insight"]
    for i, node in enumerate(insights):
        if i % 2 == 1:  # alternate → ~half ungrounded
            node.setdefault("properties", {})["grounded"] = False


def _canonicalize_persons(*docs: dict[str, Any]) -> None:
    """Canonicalize Person ids across artifacts so cross-episode enrichers work.

    The deterministic builder assigns raw diarization ids (``person:speaker-02``)
    that differ per episode, so cross-episode enrichers (guest_coappearance,
    grounding_rate — which filter speaker-NN by design) see no signal. Mapping
    each Person to ``person:<name-slug>`` makes the same guest identical across
    episodes and canonical for the enrichers (#1148 corpus evolution).
    """
    for doc in docs:
        _canonicalize_persons_in(doc)


# Stable, sortable run tag per show (single run per feed — the catalog keeps only
# the lexicographically-greatest run_* per feed dir).
_RUN_TAG = "run_20260101_000000"

# A tiny silent MP3 (data URI) so ``content.media_url`` is a real, directly-playable
# enclosure for the audio-source route — no network, no rehosting. (ID3-less 1-frame
# MPEG is enough for the contract; the player never decodes it in e2e.)
_SILENT_MP3_DATA_URI = (
    "data:audio/mpeg;base64,"
    "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA"
    "//tQxAADB8AhSmxhIIEVCSiJrDCQBTcu3UrAIwUdkRgQbFAZC1CQEwTJ9mjRv"
    "BA"
)

# Cross-show umbrella topics so ``search/topic_clusters.json`` has multi-member
# clusters (the interests picker only surfaces multi-member clusters; singletons
# are hidden). Each umbrella spans 2+ shows.
CROSS_CUTTING_TOPICS: dict[str, list[str]] = {
    "p05": ["personal finance", "risk management"],
    "p02": ["systems thinking", "risk management"],
    "p03": ["safety practices", "risk management"],
    # #1148: risk management + systems thinking recur across all 9 wired shows so
    # they form real multi-member clusters (not singletons the picker hides).
    "p04": ["visual craft", "systems thinking"],
    "p01": ["endurance sport", "risk management"],
    "p07": ["systems thinking", "risk management"],
    "p06": ["long-form", "systems thinking"],
    "p08": ["public radio", "risk management"],
    "p09": ["systems thinking", "risk management"],
}
# Shared umbrellas injected into every show so clusters are genuinely multi-member.
SHARED_UMBRELLAS: list[str] = ["lifelong learning", "expert interviews"]


def _episode_subtitle(transcript_path: Path, fallback: str) -> str:
    """Human episode title from the transcript's ``## <subtitle>`` header line.

    The screenplay fixtures carry the real episode title on a ``## ...`` line
    (the ``# ...`` line is the show). Using it gives the catalog crisp,
    spec-referenceable episode titles instead of a generic "Show — Episode N".
    """
    try:
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("## "):
                return s[3:].strip() or fallback
    except OSError:
        pass
    return fallback


def _raw_canonical_segments(offset_segs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Player ``segments.json`` rows from the diarized offset segments.

    ``segments_view.to_contract_segments`` reads ``{id, start, end, text, speaker_label}``.
    We emit the RAW canonical segments (consumer plays the original audio timeline).
    """
    out: list[dict[str, Any]] = []
    for i, seg in enumerate(offset_segs):
        out.append(
            {
                "id": i,
                "start": float(seg.get("start") or 0.0),
                "end": float(seg.get("end") or 0.0),
                "text": str(seg.get("text") or "").strip(),
                "speaker_label": str(seg.get("speaker_label") or "SPEAKER"),
            }
        )
    return out


def _stable_episode_id(show_dir: str, ep_label: str) -> str:
    """Deterministic episode id (stable across runs; no uuid clock dependence)."""
    return "ep-" + hashlib.sha256(f"{show_dir}:{ep_label}".encode()).hexdigest()[:16]


def _clean_insight_quote_excerpts(
    diar_segments: list[dict[str, Any]],
    topics: list[str],
) -> dict[str, list[str]]:
    """Build ``{topics, insights, quotes}`` from DIARIZED utterances, not the raw file.

    ``read_transcript_excerpts`` (viewer helper) reads the raw screenplay, so its first
    "insight" is the markdown header block (``# show … ## title … Host: …``) — ugly on the
    Player insights surface. The diarized segments already drop that header and the speaker
    prefixes, so picking clean utterance sentences from them yields readable, grounded
    insights/quotes. Topics come from the umbrella list we injected (caller-supplied).
    """
    sentences: list[str] = []
    for seg in diar_segments:
        txt = str(seg.get("text") or "").strip()
        # Skip ad-read / sponsor utterances and very short interjections; keep substantive
        # sentences so insights read as real takeaways.
        if not txt or len(txt) < 40:
            continue
        low = txt.lower()
        if "sponsor" in low or "use code pod" in low or "brought to you by" in low:
            continue
        sentences.append(txt if len(txt) <= 200 else txt[:197].rstrip() + "…")
        if len(sentences) >= 6:
            break
    insights = sentences[:3]
    quotes = sentences[3:6] or sentences[:3]
    return {"topics": topics, "insights": insights, "quotes": quotes}


def _enrich_kg_with_people(kg: dict[str, Any], roster: list[dict[str, str]]) -> None:
    """Add Person nodes (+ episode→person edges) to a ``build_kg`` artifact in place.

    The viewer ``build_kg`` emits only Topic/Entity nodes, so the consumer entity-card
    *people* surface (``app_kg_view.entities_from_kg`` reads Person nodes from the KG)
    would be empty. The real people are in the diarized roster (host/guest), so we add a
    ``Person`` node per roster speaker and a ``MENTIONS`` edge from the episode node —
    keeping the artifact shape identical to a real reprocessed KG (which carries people).
    """
    nodes = kg.get("nodes")
    edges = kg.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return
    ep_node_id = next(
        (n.get("id") for n in nodes if isinstance(n, dict) and n.get("type") == "Episode"),
        None,
    )
    # Sponsor-read pseudo-speakers (e.g. ``Ad:`` lines) are not people — exclude them
    # so the entity-card people surface mirrors a real diarized roster (host + guests).
    _NON_PERSON_LABELS = {"ad", "ads", "sponsor", "announcer", "narrator", "promo"}
    existing = {n.get("id") for n in nodes if isinstance(n, dict)}
    for i, sp in enumerate(roster):
        name = str(sp.get("name") or "").strip()
        if not name or name.lower() in _NON_PERSON_LABELS:
            continue
        pid = f"person:{slug(name)}"
        if pid in existing:
            continue
        existing.add(pid)
        role = str(sp.get("role") or "").strip()
        props: dict[str, Any] = {"name": name}
        if role:
            props["role"] = role
        nodes.append({"id": pid, "type": "Person", "properties": props})
        if ep_node_id:
            edges.append({"type": "MENTIONS", "from": ep_node_id, "to": pid})


# A fixed timestamp keeps the synthesized corpus byte-stable across rebuilds.
_ENRICH_COMPUTED_AT = "2026-01-01T00:00:00Z"


def _enrichment_envelope(enricher_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """An RFC-088 enricher output envelope (the shape the consumer read surface parses)."""
    return {
        "computed_at": _ENRICH_COMPUTED_AT,
        "enricher_id": enricher_id,
        "enricher_version": "1.0",
        "schema_version": "1.0",
        "status": "ok",
        "data": data,
        "error": None,
        "error_class": None,
        "retry_count": 0,
        "circuit_state": None,
        "duration_ms": 0,
        "records_written": len(next(iter(data.values()), [])) if data else 0,
    }


_VADER: Any = None


def _insight_sentiment_envelope(gi: dict[str, Any], episode_id: str) -> dict[str, Any]:
    """An ``insight_sentiment`` envelope: per-Insight VADER compound + neg/neu/pos label.

    Mirrors the real deterministic enricher exactly (same lexicon, same ±0.05 thresholds) so the CIL
    timeline join reads the fixture identically to a live run.
    """
    global _VADER
    if _VADER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _VADER = SentimentIntensityAnalyzer()
    gi_nodes = (gi.get("data") or gi).get("nodes", []) if isinstance(gi, dict) else []
    insights: list[dict[str, Any]] = []
    counts = {"negative": 0, "neutral": 0, "positive": 0}
    for n in gi_nodes:
        if not (isinstance(n, dict) and n.get("type") == "Insight" and n.get("id")):
            continue
        text = str((n.get("properties") or {}).get("text") or "").strip()
        if not text:
            continue
        c = round(float(_VADER.polarity_scores(text)["compound"]), 4)
        label = "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
        counts[label] += 1
        insights.append({"insight_id": str(n["id"]), "compound": c, "label": label})
    return _enrichment_envelope(
        "insight_sentiment",
        {
            "episode_id": episode_id,
            "counts": counts,
            "total_insights": len(insights),
            "insights": insights,
        },
    )


def _insight_density_envelope(gi: dict[str, Any], episode_id: str) -> dict[str, Any]:
    """An ``insight_density`` envelope: this episode's insights bucketed early/mid/late.

    Round-robins the GI's Insight nodes across the three thirds (no timing) so the
    consumer episode-enrichment read surface has a real episode-scope signal.
    """
    gi_nodes = (gi.get("data") or gi).get("nodes", []) if isinstance(gi, dict) else []
    insight_ids = [
        str(n.get("id"))
        for n in gi_nodes
        if isinstance(n, dict) and str(n.get("type")) == "Insight" and n.get("id")
    ]
    segs = ["early", "mid", "late"]
    insight_segments = [
        {"insight_id": iid, "segment": segs[i % 3]} for i, iid in enumerate(insight_ids)
    ]
    counts = {"early": 0, "mid": 0, "late": 0, "unknown": 0}
    for seg in insight_segments:
        counts[seg["segment"]] += 1
    return _enrichment_envelope(
        "insight_density",
        {
            "insight_segments": insight_segments,
            "episode_id": episode_id,
            "duration_seconds": 1800.0,
            "has_timing": False,
            "counts": counts,
            "total_insights": len(insight_segments),
        },
    )


def _velocity_last_over_6mo(series: list[int]) -> float:
    """Mirror the real enricher: last-month count over the trailing 6-month mean (1.0 = flat)."""
    if not series:
        return 0.0
    six = series[-6:]
    avg = sum(six) / len(six)
    return round(series[-1] / avg, 4) if avg else 0.0


def _iso_week(publish: str) -> str:
    """``YYYY-MM-DD`` → ISO year-week ``YYYY-Www`` (for the content_series axis)."""
    iso = datetime.fromisoformat(publish[:10]).isocalendar()
    return f"{iso.year:04d}-W{iso.week:02d}"


def _tally_episode_content(
    kg: dict[str, Any],
    topic_ids: list[str],
    week: str,
    month: str,
    episode_id: str,
    *,
    counts: dict[str, int],
    months: dict[str, list[str]],
    episodes: dict[str, list[str]],
    weeks: dict[str, list[str]],
    person_weeks: dict[str, list[str]],
) -> None:
    """Fold one episode's Topic + Person mentions into the corpus-scope accumulators (in place)."""
    for tid in topic_ids:
        counts[tid] = counts.get(tid, 0) + 1
        months.setdefault(tid, []).append(month)
        episodes.setdefault(tid, []).append(episode_id)
        weeks.setdefault(tid, []).append(week)
    for node in kg.get("nodes", []):
        if isinstance(node, dict) and node.get("type") == "Person" and node.get("id"):
            person_weeks.setdefault(str(node["id"]), []).append(week)


def _content_series_data(
    topic_weeks: dict[str, list[str]], person_weeks: dict[str, list[str]]
) -> dict[str, Any]:
    """The RFC-103 now-free content_series (per-topic + per-person weekly counts).

    Matches the ``temporal_velocity`` enricher's ``content_series`` shape, authored
    deterministically from the corpus's own ISO weeks so the momentum layer works over the fixture.
    """
    all_weeks = sorted(
        {w for weeks in (*topic_weeks.values(), *person_weeks.values()) for w in weeks}
    )

    def _rows(by_id: dict[str, list[str]], id_key: str, lab_key: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for eid, weeks in sorted(by_id.items()):
            counts: dict[str, int] = {}
            for w in weeks:
                counts[w] = counts.get(w, 0) + 1
            rows.append(
                {
                    id_key: eid,
                    lab_key: eid.split(":", 1)[-1].replace("-", " "),
                    "weekly_counts": dict(sorted(counts.items())),
                    "total": len(weeks),
                }
            )
        rows.sort(key=lambda r: (-int(r["total"]), str(r[id_key])))
        return rows

    return {
        "window_weeks": all_weeks,
        "topics": _rows(topic_weeks, "topic_id", "topic_label"),
        "persons": _rows(person_weeks, "person_id", "person_label"),
    }


def _temporal_velocity_data(topic_months: dict[str, list[str]]) -> dict[str, Any]:
    """Author the canonical ``temporal_velocity`` payload from per-topic publish months.

    ``window_months`` is the corpus's own sorted month axis; per topic we bucket its episodes'
    publish months into ``monthly_counts`` and collapse the trend to ``velocity_last_over_6mo``
    (the shape the consumer Trending surface + the real enricher share). Corpus-anchored, so the
    signal is deterministic regardless of wall-clock ``now`` (the real enricher anchors its window
    to the current calendar month — unstable for a committed fixture).
    """
    window_months = sorted({m for months in topic_months.values() for m in months})
    topics: list[dict[str, Any]] = []
    for tid, months in sorted(topic_months.items()):
        counts = {m: months.count(m) for m in set(months)}
        series = [counts.get(m, 0) for m in window_months]
        topics.append(
            {
                "topic_id": tid,
                "topic_label": tid.replace("topic:", "").replace("-", " "),
                "monthly_counts": {m: counts.get(m, 0) for m in window_months},
                "velocity_last_over_6mo": _velocity_last_over_6mo(series),
                "total": sum(series),
            }
        )
    topics.sort(key=lambda r: (-r["velocity_last_over_6mo"], -r["total"], r["topic_id"]))
    return {"window_months": window_months, "topics": topics}


def _theme_clusters_data(topic_episodes: dict[str, list[str]]) -> dict[str, Any]:
    """Author one theme cluster ("storyline") for the consumer Storylines surface.

    The cross-domain *managing risk* story: risk management, systems thinking, and safety
    practices co-occur across shows (the transcripts frame risk as "systems of correlated bets" in
    investing, software, racing, and diving). Members carry ``lift_to_cluster`` (anchor = highest)
    + their episode ids, matching the enricher output the consumer readers parse. Deterministic,
    like ``search/topic_clusters.json``.
    """
    members_spec = [
        ("topic:risk-management", "risk management", 2.6),
        ("topic:systems-thinking", "systems thinking", 2.1),
        ("topic:safety-practices", "safety practices", 1.7),
    ]
    members = [
        {
            "topic_id": tid,
            "label": label,
            "lift_to_cluster": lift,
            "episode_ids": sorted(set(topic_episodes.get(tid, []))),
        }
        for tid, label, lift in members_spec
        if topic_episodes.get(tid)  # only topics that actually appear in the corpus
    ]
    clusters: list[dict[str, Any]] = []
    if len(members) >= 2:
        clusters.append(
            {
                "cluster_type": "theme",
                "canonical_label": "Managing risk across domains",
                "graph_compound_parent_id": "thc:managing-risk",
                "member_count": len(members),
                "members": members,
            }
        )
    n_members = sum(c["member_count"] for c in clusters)
    return {
        "schema_version": "1",
        "method": "cooccurrence_lift",
        "min_pair_episode_count": 2,
        "merge_threshold": 2.0,
        "episode_count": len({e for eps in topic_episodes.values() for e in eps}),
        "topic_count": len(topic_episodes),
        "cluster_count": len(clusters),
        "singletons": max(0, len(topic_episodes) - n_members),
        "clusters": clusters,
    }


def _accumulate_topic_persons(
    ep_persons: list[tuple[str, str]],
    topic_ids: list[str],
    *,
    topic_persons: dict[str, list[tuple[str, str]]],
) -> None:
    """Fold one episode's persons × topics into the topic_consensus per-Topic person set (in place).

    Within an episode every rostered person is associated with every episode topic (topics are
    episode-level in this fixture), which gives topic_consensus its cross-Person pairs per Topic.
    """
    for tid in topic_ids:
        for pid, name in ep_persons:
            topic_persons.setdefault(tid, []).append((pid, name))


def _episode_persons(kg: dict[str, Any]) -> list[tuple[str, str]]:
    """``[(person_id, person_name)]`` from a KG's canonicalised Person nodes."""
    out: list[tuple[str, str]] = []
    for node in kg.get("nodes", []):
        if isinstance(node, dict) and node.get("type") == "Person" and node.get("id"):
            pid = str(node["id"])
            name = str((node.get("properties") or {}).get("name") or pid)
            out.append((pid, name))
    return out


def _topic_consensus_data(
    topic_persons: dict[str, list[tuple[str, str]]],
    *,
    cos_threshold: float = 0.70,
    max_rows: int = 12,
) -> dict[str, Any]:
    """Author the ``topic_consensus`` payload (ADR-108 composite) — cross-Person corroboration.

    Deterministic: for each topic with ≥2 distinct persons, emit one corroborating pair between the
    two earliest-seen persons. Claim texts are templated on the topic label (clean synthetic
    corroboration) so the viewer's Consensus surfaces read as genuine agreement. Mirrors the
    composite enricher's output: ``consensus_score`` = ``cosine`` (a stable per-pair hash in
    ``[cos_threshold, 0.95]``) plus a low ``contradiction``. Highest first, capped at ``max_rows``.
    """
    consensus: list[dict[str, Any]] = []
    for tid, entries in sorted(topic_persons.items()):
        by_person: dict[str, str] = {}  # person_id -> name, first-seen order preserved
        for pid, name in entries:
            by_person.setdefault(pid, name)
        persons = list(by_person.items())
        if len(persons) < 2:
            continue
        (pid_a, name_a), (pid_b, name_b) = persons[0], persons[1]
        label = tid.replace("topic:", "").replace("-", " ")
        txt_a = (
            f"With {label}, the durable wins come from managing the whole system, not one-off bets."
        )
        txt_b = (
            f"Agreed — {label} rewards a systems view; the edge is in the process, "
            "not any single call."
        )
        h = int(hashlib.sha256(f"{tid}|{pid_a}|{pid_b}".encode()).hexdigest(), 16)
        cosine = round(cos_threshold + (h % 1000) / 1000.0 * (0.95 - cos_threshold), 6)
        contradiction = round((h % 137) / 137.0 * 0.15, 6)  # low (< contra_threshold 0.5)
        consensus.append(
            {
                "topic_id": tid,
                "person_a_id": pid_a,
                "person_a_name": name_a,
                "person_b_id": pid_b,
                "person_b_name": name_b,
                "insight_a_id": f"insight:consensus:{slug(tid)}:{slug(pid_a)}",
                "insight_a_text": txt_a,
                "insight_b_id": f"insight:consensus:{slug(tid)}:{slug(pid_b)}",
                "insight_b_text": txt_b,
                "consensus_score": cosine,
                "cosine": cosine,
                "contradiction": contradiction,
                "model_id": "all-MiniLM-L6-v2+deberta-v3-small",
                "model_version": "v2",
            }
        )
    consensus.sort(key=lambda r: (-r["consensus_score"], r["topic_id"]))
    consensus = consensus[:max_rows]
    return {
        "model_id": "all-MiniLM-L6-v2+deberta-v3-small",
        "model_version": "v2",
        "cos_threshold": cos_threshold,
        "contra_threshold": 0.5,
        "pairs_scored": len(consensus),
        "consensus": consensus,
    }


def _topic_similarity_data(
    topic_episodes: dict[str, list[str]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Author the ``topic_similarity`` payload (embedding-cosine neighbours) deterministically.

    The real enricher embeds topic labels + ranks by cosine (needs ML, so it runs separately
    ``--with-ml`` and is absent from the committed no-ML fixture). To give the surface real data we
    proxy similarity with a **shared-episode Jaccard** over the corpus's own topics — topics
    discussed in the same episodes are "similar". Emits the exact shape the viewer
    ``EnrichmentEdgesPanel`` + consumer "Similar topics" read: ``{topic_count, top_k,
    missing_topic_ids, topics[{topic_id, topic_label, top_k[{...similarity}]}]}``.
    """

    def _label(tid: str) -> str:
        return tid.replace("topic:", "").replace("-", " ")

    topics = sorted(topic_episodes)
    ep_sets = {t: set(topic_episodes.get(t, [])) for t in topics}
    out_topics: list[dict[str, Any]] = []
    missing: list[str] = []
    for t in topics:
        a = ep_sets[t]
        scored: list[tuple[str, float]] = []
        for u in topics:
            if u == t:
                continue
            union = a | ep_sets[u]
            if not union:
                continue
            jac = len(a & ep_sets[u]) / len(union)
            if jac > 0:
                scored.append((u, jac))
        scored.sort(key=lambda x: (-x[1], x[0]))
        nbrs = scored[:top_k]
        if not nbrs:
            missing.append(t)
            continue
        out_topics.append(
            {
                "topic_id": t,
                "topic_label": _label(t),
                # Map Jaccard [0,1] into a plausible cosine band [0.55, 0.95], deterministic.
                "top_k": [
                    {
                        "topic_id": u,
                        "topic_label": _label(u),
                        "similarity": round(0.55 + 0.40 * s, 6),
                    }
                    for u, s in nbrs
                ],
            }
        )
    # ``topics`` first so the fixture envelope's ``records_written`` (len of the first data value)
    # counts topics; ``sort_keys=True`` on write makes the on-disk key order canonical anyway.
    return {
        "topics": out_topics,
        "topic_count": len(out_topics),
        "top_k": top_k,
        "missing_topic_ids": missing,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rss-dir", type=Path, default=Path("tests/fixtures/rss"))
    default_version = Path("tests/fixtures/FIXTURES_VERSION").read_text(encoding="utf-8").strip()
    p.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path("tests/fixtures/transcripts") / default_version,
    )
    p.add_argument("--output", type=Path, default=Path("tests/fixtures/app-validation-corpus"))
    p.add_argument("--max-feeds", type=int, default=9)
    p.add_argument("--max-episodes-per-feed", type=int, default=4)
    args = p.parse_args()

    if not args.rss_dir.is_dir():
        sys.exit(f"--rss-dir does not exist: {args.rss_dir}")
    if not args.transcripts_dir.is_dir():
        sys.exit(f"--transcripts-dir does not exist: {args.transcripts_dir}")

    version = default_version
    out = (args.output / version).resolve()
    out.mkdir(parents=True, exist_ok=True)

    shows = APP_SHOWS[: args.max_feeds]
    # v3 ground-truth dir (sibling of the transcripts dir) — carries the authored
    # publish dates (#1148 varied 2024→now schedule).
    gt_dir = args.transcripts_dir.parent.parent / "ground-truth" / version / "ground_truth"
    pod_guests = _load_pod_guests(gt_dir.parent / "manifest.json")  # key→name for claim injection
    episode_index: list[dict[str, Any]] = []  # for the regenerate-summary print
    corpus_topic_counts: dict[str, int] = {}  # → corpus-scope temporal_velocity envelope
    # topic_id → publish months (YYYY-MM) of the episodes it appears in. The
    # temporal_velocity signal is authored from the corpus's OWN date axis (not
    # wall-clock now), so the fixture's trending signal is deterministic + stable.
    corpus_topic_months: dict[str, list[str]] = {}
    # topic_id → episode ids (for the co-occurrence theme-cluster members).
    corpus_topic_episodes: dict[str, list[str]] = {}
    # topic_id / person_id → ISO weeks (for the RFC-103 now-free content_series).
    corpus_topic_weeks: dict[str, list[str]] = {}
    corpus_person_weeks: dict[str, list[str]] = {}
    # ADR-108 topic_consensus: topic_id → [(person_id, name)], authored deterministically from the
    # episode's own persons × topics (like content_series).
    corpus_topic_persons: dict[str, list[tuple[str, str]]] = {}

    for show_rss_stem, show_dir in shows:
        rss_path = args.rss_dir / f"{show_rss_stem}.xml"
        if not rss_path.is_file():
            print(f"  warn: missing RSS {rss_path}; skipping show {show_dir}")
            continue
        feed_meta = parse_rss_feed_metadata(rss_path)
        feed_meta.update(SHOW_META_OVERRIDE.get(show_dir, {}))
        # The app keys feeds by a readable feed id (the show dir), not the sha feed id,
        # so the on-disk feeds/<show_dir>/ path is human-readable in specs/debugging.
        feed_id = show_dir
        feed_title = feed_meta["display_title"]

        transcripts = sorted(args.transcripts_dir.glob(f"{show_dir}_e[0-9]*.txt"))
        transcripts = [t for t in transcripts if "_multi_" not in t.stem and "_fast" not in t.stem]
        transcripts = transcripts[: args.max_episodes_per_feed]

        run_meta_dir = out / "feeds" / show_dir / _RUN_TAG / "metadata"
        run_tr_dir = out / "feeds" / show_dir / _RUN_TAG / "transcripts"
        run_meta_dir.mkdir(parents=True, exist_ok=True)
        run_tr_dir.mkdir(parents=True, exist_ok=True)

        for ei, transcript_path in enumerate(transcripts):
            ep_label = transcript_path.stem  # e.g. p05_e01
            episode_id = _stable_episode_id(show_dir, ep_label)
            episode_title = _episode_subtitle(transcript_path, f"{feed_title} — Episode {ei + 1}")
            # Deterministic publish dates, newest-first by show then episode so the
            # app's "What's new" (recency) order is stable. Show 0 ep 0 is newest.
            # Prefer the authored v3 date (#1148 varied 2024→now schedule, unique
            # per episode); fall back to the legacy single-month scheme.
            day = 28 - (shows.index((show_rss_stem, show_dir)) * len(transcripts) + ei)
            publish = _publish_date_for(ep_label, gt_dir) or f"2026-01-{day:02d}"

            diar_segments, roster = parse_diarized_segments(transcript_path)
            raw_text, offset_segs = format_screenplay_with_offsets(diar_segments)

            # Topics: the show's lead topic + the shared cross-show umbrellas. These are
            # curated, clean labels (the viewer helper's raw transcript-phrase extraction
            # yields conversational junk like "welcome back to" on these chatty fixtures,
            # which would look broken on the entity/topic surfaces), and they give the
            # topic_clusters the cross-show overlap the interests picker needs.
            topics = CROSS_CUTTING_TOPICS.get(show_dir, []) + SHARED_UMBRELLAS
            # Insights/quotes from CLEAN diarized utterances (not the raw header block).
            excerpts = _clean_insight_quote_excerpts(diar_segments, topics)

            # Artifact relpaths (corpus-root-relative). The catalog derives gi/kg
            # as siblings of the metadata file; transcript_file_path is RUN-relative.
            base = f"feeds/{show_dir}/{_RUN_TAG}/metadata/{ep_label}"
            metadata_rel = f"{base}.metadata.json"
            transcript_run_rel = f"transcripts/{ep_label}.txt"

            gi = build_gi(
                episode_id,
                feed_id,
                episode_title,
                publish + "T12:00:00",
                excerpts,
                quote_segments=offset_segs,
                roster=roster,
                transcript_ref=transcript_run_rel,
                metadata_relative_path=metadata_rel,
            )
            kg = build_kg(episode_id, episode_title, excerpts, metadata_relative_path=metadata_rel)
            # The viewer build_kg emits no Person nodes; add the diarized roster so the
            # consumer entity-card people surface has real data (host/guest).
            _enrich_kg_with_people(kg, roster)
            # #1148: canonicalize Person ids (speaker-NN → name-slug) so the
            # cross-episode enrichers (guest_coappearance / grounding_rate) work,
            # and stamp the Episode publish_date so temporal_velocity sees it.
            _canonicalize_persons(gi, kg)
            _stamp_publish_date(publish + "T12:00:00", gi, kg)
            _vary_grounding(gi, ep_label, gt_dir)  # #1148: real grounding variation
            # #1148: surface the authored claims (perspectives + opposition) the
            # naive sentences[:3] extractor drops.
            _inject_authored_claims(
                gi, ep_label, gt_dir, pod_guests.get(ep_label.split("_")[0], {})
            )

            (run_meta_dir / f"{ep_label}.gi.json").write_text(
                json.dumps(gi, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (run_meta_dir / f"{ep_label}.kg.json").write_text(
                json.dumps(kg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            # CIL bridge — unblocks the #1146 perspectives surface (see
            # _bridge_from_gi). Sibling of the .gi.json so iter_cil_episode_bundles
            # pairs them; identities mirror the GI's (post-injection) topics.
            (run_meta_dir / f"{ep_label}.bridge.json").write_text(
                json.dumps(_bridge_from_gi(gi, episode_id), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            # Episode-scope enrichment envelope (RFC-088) so the consumer enrichment read surface
            # (GET /api/app/episodes/{slug}/enrichment, P3 #1121) has real data in the e2e corpus.
            # insight_density (topic_cooccurrence was dropped — trivial at episode scope).
            topic_ids = [f"topic:{slug(lbl)}" for lbl in topics]
            enrich_dir = run_meta_dir / "enrichments"
            enrich_dir.mkdir(parents=True, exist_ok=True)
            (enrich_dir / f"{ep_label}.insight_density.json").write_text(
                json.dumps(_insight_density_envelope(gi, episode_id), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            # ADR-108 conversation-timeline colour: per-Insight VADER sentiment sidecar (the CIL
            # timeline/arc queries join it by insight_id).
            (enrich_dir / f"{ep_label}.insight_sentiment.json").write_text(
                json.dumps(_insight_sentiment_envelope(gi, episode_id), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            _tally_episode_content(
                kg,
                topic_ids,
                _iso_week(publish),
                publish[:7],
                episode_id,
                counts=corpus_topic_counts,
                months=corpus_topic_months,
                episodes=corpus_topic_episodes,
                weeks=corpus_topic_weeks,
                person_weeks=corpus_person_weeks,
            )
            # ADR-108: fold this episode's persons × topics into the topic_consensus accumulator.
            _accumulate_topic_persons(
                _episode_persons(kg),
                topic_ids,
                topic_persons=corpus_topic_persons,
            )

            # Transcript text + RAW canonical segments (player contract).
            (run_tr_dir / f"{ep_label}.txt").write_text(raw_text, encoding="utf-8")
            (run_tr_dir / f"{ep_label}.segments.json").write_text(
                json.dumps(_raw_canonical_segments(offset_segs), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            # Consumer-shaped metadata (the nested schema corpus_catalog reads:
            # feed/episode/summary/content). Includes content.media_url for audio.
            bullets = excerpts["insights"][:3] or [f"Key point {n + 1}" for n in range(3)]
            summary_body = excerpts["insights"][0] if excerpts["insights"] else episode_title
            metadata_doc = {
                "schema_version": "1",
                "feed": {
                    "feed_id": feed_id,
                    "title": feed_title,
                    "url": feed_meta["rss_url"],
                    "description": feed_meta["description"],
                },
                "episode": {
                    "episode_id": episode_id,
                    "title": episode_title,
                    "published_date": publish + "T00:00:00",
                    "duration_seconds": 1800,
                },
                "summary": {
                    "title": episode_title,
                    "bullets": bullets,
                    "raw_text": summary_body,
                },
                "content": {
                    "transcript_file_path": transcript_run_rel,
                    "transcript_source": "synthetic-app-validation-corpus",
                    "speakers": roster,
                    "diarization_num_speakers": len(roster),
                    "media_url": _SILENT_MP3_DATA_URI,
                    "media_type": "audio/mpeg",
                    "media_id": f"sha256:{slug(ep_label)}",
                },
            }
            (run_meta_dir / f"{ep_label}.metadata.json").write_text(
                json.dumps(metadata_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            episode_index.append(
                {
                    "show": show_dir,
                    "feed_title": feed_title,
                    "episode_id": episode_id,
                    "title": episode_title,
                    "publish_date": publish,
                    "first_line": (offset_segs[0]["text"] if offset_segs else ""),
                }
            )
            print(f"  {show_dir}/{ep_label}: {episode_title!r} ({len(offset_segs)} segs)")

    # --- search/topic_clusters.json -----------------------------------------
    # Real clusters group DISTINCT topic ids under one themed parent. We build two
    # multi-member clusters so the interests picker (top_clusters_by_member_count)
    # surfaces them and entity-card siblings (consumer_cluster_siblings) resolve:
    #
    #   tc:lifelong-learning  — the shared "lifelong learning" + "expert interviews"
    #                           umbrellas (cross-show themes), 2 distinct members.
    #   tc:show-themes         — each show's distinct lead topic (personal-finance,
    #                           systems-thinking, safety-practices), one per show.
    #
    # Members carry distinct topic ids + their own label, matching the v2 schema
    # the readers expect (graph_compound_parent_id / cil_alias_target_topic_id /
    # member_count / members[{topic_id,label}]).
    def _cluster(parent_slug: str, canonical: str, labels: list[str]) -> dict[str, Any] | None:
        members = []
        seen: set[str] = set()
        for lbl in labels:
            tid = f"topic:{slug(lbl)}"
            if tid in seen:
                continue
            seen.add(tid)
            members.append({"topic_id": tid, "label": lbl})
        if len(members) < 2:
            return None
        return {
            "canonical_label": canonical,
            "cil_alias_target_topic_id": members[0]["topic_id"],
            "graph_compound_parent_id": f"tc:{parent_slug}",
            "member_count": len(members),
            "members": members,
        }

    clusters: list[dict[str, Any]] = []
    cross_theme = _cluster("lifelong-learning", "Lifelong Learning", SHARED_UMBRELLAS)
    if cross_theme is not None:
        clusters.append(cross_theme)
    show_lead_labels = [lbl for _stem, sdir in shows for lbl in CROSS_CUTTING_TOPICS.get(sdir, [])]
    show_themes = _cluster("show-themes", "Show Themes", show_lead_labels)
    if show_themes is not None:
        clusters.append(show_themes)

    search_dir = out / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    (search_dir / "topic_clusters.json").write_text(
        json.dumps(
            {
                "schema_version": "2",
                "model": "synthetic-app-validation-corpus",
                "threshold": 0.75,
                "clusters": clusters,
                "singletons": 0,
                "topic_count": sum(len(c["members"]) for c in clusters),
                "cluster_count": len(clusters),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # --- corpus-scope enrichment envelopes (RFC-088) ------------------------
    corpus_enrich_dir = out / "enrichments"
    corpus_enrich_dir.mkdir(parents=True, exist_ok=True)

    # temporal_velocity — the consumer Trending surface (GET /api/app/corpus/enrichment)
    # reads ``data.topics[]`` with ``window_months`` + per-topic ``monthly_counts`` /
    # ``velocity_last_over_6mo`` / ``total`` (NOT the ``{trending}`` shape). We author it in
    # that canonical shape over the corpus's OWN month axis, and mirror the real enricher's
    # ``velocity_last_over_6mo`` = last-month count / trailing-6-month mean (1.0 = flat, >1
    # rising). Deterministic: derived from the authored per-episode publish dates (the #1148
    # offset schedule makes risk-management rise, per the corpus gold's ``heating_up``).
    tv = _temporal_velocity_data(corpus_topic_months)
    # RFC-103 Phase 1: the durable, now-free content_series the momentum layer + trending endpoints
    # read (per-topic + per-person weekly counts). Same shape the enricher emits in production.
    tv["content_series"] = _content_series_data(corpus_topic_weeks, corpus_person_weeks)
    (corpus_enrich_dir / "temporal_velocity.json").write_text(
        json.dumps(_enrichment_envelope("temporal_velocity", tv), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # topic_theme_clusters — the consumer Storylines surface (theme clusters = topics
    # discussed together). Authored deterministically (like search/topic_clusters.json) from the
    # cross-domain "managing risk" storyline the transcripts co-author (risk management framed as
    # "systems of correlated bets" across investing/software/racing/diving → risk-management +
    # systems-thinking + safety-practices co-occur). Members carry the shape the consumer readers
    # expect (graph_compound_parent_id thc:… / canonical_label / member_count / members[{topic_id,
    # label, lift_to_cluster, episode_ids}]).
    theme = _theme_clusters_data(corpus_topic_episodes)
    (corpus_enrich_dir / "topic_theme_clusters.json").write_text(
        json.dumps(_enrichment_envelope("topic_theme_clusters", theme), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    # topic_consensus — the reimagined NLI enricher (ADR-108). Authored deterministically from the
    # corpus's own persons × topics so the operator viewer's Consensus edges + Person "Consensus"
    # surfaces render + pivot on the fixture. (Per-person / per-topic stance over time is a
    # read-time CIL query — conversation-arc / position-arc — not an enricher artifact.)
    consensus = _topic_consensus_data(corpus_topic_persons)
    (corpus_enrich_dir / "topic_consensus.json").write_text(
        json.dumps(_enrichment_envelope("topic_consensus", consensus), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    # topic_similarity — the operator EnrichmentEdges "similar" surface + consumer "Similar topics".
    # The real enricher is embedding-cosine (ML, runs separately --with-ml); the fixture authors a
    # no-ML shared-episode-Jaccard proxy so the surface renders + pivots on committed data.
    similarity = _topic_similarity_data(corpus_topic_episodes)
    (corpus_enrich_dir / "topic_similarity.json").write_text(
        json.dumps(_enrichment_envelope("topic_similarity", similarity), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    total_size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\napp validation corpus written to {out}")
    print(f"  shows: {len(shows)}  episodes: {len(episode_index)}  clusters: {len(clusters)}")
    print(f"  total size: {total_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
