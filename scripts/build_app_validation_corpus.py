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

* Run-dir layout ``feeds/<show>/run_<tag>/metadata/<ep>.{metadata,gi,kg}.json``
  and ``feeds/<show>/run_<tag>/transcripts/<ep>.{txt,segments.json}`` — what
  ``corpus_catalog`` / ``corpus_scope.discover_metadata_files`` expect.
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
    episode_index: list[dict[str, Any]] = []  # for the regenerate-summary print
    corpus_topic_counts: dict[str, int] = {}  # → corpus-scope temporal_velocity envelope

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

            (run_meta_dir / f"{ep_label}.gi.json").write_text(
                json.dumps(gi, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (run_meta_dir / f"{ep_label}.kg.json").write_text(
                json.dumps(kg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
            for tid in topic_ids:
                corpus_topic_counts[tid] = corpus_topic_counts.get(tid, 0) + 1

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

    # --- corpus-scope enrichment envelope (RFC-088) -------------------------
    # temporal_velocity: a corpus-wide signal the consumer surface reads via
    # GET /api/app/corpus/enrichment (P3 #1121). Rank topics by cross-episode prevalence.
    trending = [
        {"topic_id": tid, "episodes": n, "velocity": round(n / max(len(episode_index), 1), 3)}
        for tid, n in sorted(corpus_topic_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    corpus_enrich_dir = out / "enrichments"
    corpus_enrich_dir.mkdir(parents=True, exist_ok=True)
    (corpus_enrich_dir / "temporal_velocity.json").write_text(
        json.dumps(
            _enrichment_envelope("temporal_velocity", {"trending": trending}),
            indent=2,
            sort_keys=True,
        )
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
