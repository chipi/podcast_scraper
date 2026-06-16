"""Corpus-level diarization / speaker-attribution quality metrics (#876).

Validates a *produced* corpus directory against the quality properties the speaker-attribution
bugs violated, so a local full-corpus re-diarization run can be checked automatically instead
of by manual spot-check (RUNBOOK-876 Step 6). Each property maps to a bug we hit:

- guest name painted on host turns        → low share of un-attributed quotes
- host = network tag ("Colossus")         → no network/org name in ``content.speakers``
- partial naming (guest left SPEAKER_xx)  → multi-speaker episodes have ≥2 named speakers
- ``num_speakers`` dropped                 → ``diarization_num_speakers`` present
- #545 offset mismatch drops attribution  → quotes carry ``speaker_id`` + timestamp

Pure file scan (no DB). Mirrors ``gi.quality_metrics`` conventions: ``compute_*`` +
``enforce_*`` returning a metrics dict / (passed, failures).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..search.corpus_scope import discover_metadata_files
from ..speaker_detectors.hosts import has_org_markers, is_known_network

# Default enforcement thresholds.
MIN_QUOTE_ATTRIBUTION_RATE = 0.90  # quotes whose speaker_id is a real person:<slug>
MIN_QUOTE_TIMESTAMP_RATE = 0.85  # quotes carrying timestamp_start_ms
MIN_SPOKEN_BY_COVERAGE = 0.85  # SPOKEN_BY edges per quote (Quote -> Person attribution)
# multi-speaker episode = diarization_num_speakers >= 2 OR >= 2 distinct quote speakers


@dataclass
class EpisodeDiarMetrics:
    """Per-episode diarization-quality counters."""

    episode: str
    diarized: bool = False
    direct_download: bool = False  # transcript downloaded, never audio → no diarization possible
    num_speakers: Optional[int] = None
    quotes_total: int = 0
    quotes_attributed: int = 0  # speaker_id is a non-empty person:<slug>
    quotes_with_timestamp: int = 0
    spoken_by_edges: int = 0  # SPOKEN_BY edges in this episode's gi.json (Quote -> Person)
    distinct_speakers: int = 0
    content_speaker_names: List[str] = field(default_factory=list)
    network_org_speaker_names: List[str] = field(default_factory=list)
    multispeaker: bool = False
    named_speaker_count: int = 0  # distinct named (non SPEAKER_xx / non person:speaker-) speakers


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _find(obj: Any, key: str) -> Any:
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k == key:
                    return v
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def _is_named_speaker(speaker_id: str) -> bool:
    """True when a quote ``speaker_id`` names a real person, not a raw diarization label.

    Real ids look like ``person:brian-chesky``; a voice the roster could not name slugs to
    ``person:speaker-02`` (or stays ``SPEAKER_02``) — those are *attributed but unnamed*.
    """
    s = (speaker_id or "").strip().lower()
    if not s:
        return False
    slug = s.split(":", 1)[1] if ":" in s else s
    return not slug.startswith("speaker")


def _episode_metrics(gi_path: Path, meta: Optional[dict]) -> EpisodeDiarMetrics:
    em = EpisodeDiarMetrics(episode=gi_path.name)
    gi = _load_json(gi_path) or {}
    quotes = [n for n in gi.get("nodes", []) if isinstance(n, dict) and n.get("type") == "Quote"]
    em.spoken_by_edges = sum(
        1
        for e in gi.get("edges", [])
        if isinstance(e, dict) and "SPOKEN" in str(e.get("type", "")).upper()
    )
    speaker_ids: set[str] = set()
    named_ids: set[str] = set()
    for q in quotes:
        props = q.get("properties", {}) if isinstance(q.get("properties"), dict) else {}
        sid = str(props.get("speaker_id") or "").strip()
        em.quotes_total += 1
        if sid:
            em.quotes_attributed += 1
            speaker_ids.add(sid)
            if _is_named_speaker(sid):
                named_ids.add(sid)
        if props.get("timestamp_start_ms") is not None:
            em.quotes_with_timestamp += 1
    em.distinct_speakers = len(speaker_ids)
    em.named_speaker_count = len(named_ids)

    if meta is not None:
        em.direct_download = str(_find(meta, "transcript_source") or "").strip().lower() == (
            "direct_download"
        )
        ns = _find(meta, "diarization_num_speakers")
        em.num_speakers = ns if isinstance(ns, int) else None
        speakers = _find(meta, "speakers")
        if isinstance(speakers, list):
            for sp in speakers:
                name = (sp.get("name") if isinstance(sp, dict) else None) or ""
                if name:
                    em.content_speaker_names.append(name)
                    # ``content.speakers`` come from the diarized roster / transcript — a TRUSTED
                    # person source where a lone token is a real first-name speaker (Neeraj,
                    # Sarah), NOT a network. So flag only explicit org markers OR a name on the
                    # known-network list (catches the "Pushkin" bumper leak, which has no
                    # generic markers) — not the bare mononym rule used for RSS author tags (#876).
                    if has_org_markers(name) or is_known_network(name):
                        em.network_org_speaker_names.append(name)

    em.diarized = em.distinct_speakers > 0 or bool(em.num_speakers)
    em.multispeaker = (em.num_speakers or 0) >= 2 or em.distinct_speakers >= 2
    return em


def _gi_for_metadata(meta_path: Path) -> Optional[Path]:
    """The ``.gi.json`` sibling of a ``.metadata.json`` file (same episode stem)."""
    name = meta_path.name
    for suffix in (".metadata.json", ".metadata.yaml", ".metadata.yml"):
        if name.endswith(suffix):
            gi = meta_path.with_name(name[: -len(suffix)] + ".gi.json")
            return gi if gi.exists() else None
    return None


def compute_diarization_quality_metrics(corpus_root: Path) -> Dict[str, Any]:
    """Walk a corpus and aggregate per-episode diarization-quality metrics."""
    # ``discover_metadata_files`` resolves the latest run per feed, so a re-diarized corpus is
    # scored on its NEW run only (the superseded un-diarized run is excluded).
    per_episode: List[EpisodeDiarMetrics] = []
    for meta_path in discover_metadata_files(Path(corpus_root)):
        gi_path = _gi_for_metadata(meta_path)
        if gi_path is None:
            continue
        per_episode.append(_episode_metrics(gi_path, _load_json(meta_path)))

    # Score over every episode that produced quotes — an episode with quotes but zero
    # attributed speakers is the #545 failure mode (don't silently drop it from the denominator).
    with_quotes = [e for e in per_episode if e.quotes_total > 0]
    # Attribution/diarization checks only apply where diarization was POSSIBLE. A
    # ``direct_download`` episode has a downloaded transcript and never any audio, so its quotes
    # carry no diarized ``speaker_id`` by design — including it in the attribution denominator
    # mislabels an unavoidable absence as the #545 drift bug (#876).
    diarizable = [e for e in with_quotes if not e.direct_download]
    q_total = sum(e.quotes_total for e in diarizable)
    q_attr = sum(e.quotes_attributed for e in diarizable)
    q_ts = sum(e.quotes_with_timestamp for e in diarizable)
    q_spoken = sum(e.spoken_by_edges for e in diarizable)
    multispeaker = [e for e in diarizable if e.multispeaker]

    return {
        "episodes_total": len(per_episode),
        "episodes_with_quotes": len(with_quotes),
        "episodes_direct_download": sum(1 for e in with_quotes if e.direct_download),
        "episodes_diarizable": len(diarizable),
        "quotes_total": q_total,
        "quote_attribution_rate": (q_attr / q_total) if q_total else 1.0,
        "quote_timestamp_rate": (q_ts / q_total) if q_total else 1.0,
        "spoken_by_coverage": (q_spoken / q_total) if q_total else 1.0,
        "episodes_unattributed": sum(
            1 for e in diarizable if e.quotes_attributed == 0
        ),  # diarizable quotes present, 0 attributed (#545)
        "episodes_with_network_speaker": sum(1 for e in diarizable if e.network_org_speaker_names),
        "episodes_missing_num_speakers": sum(1 for e in diarizable if e.num_speakers is None),
        "multispeaker_episodes": len(multispeaker),
        # The #876 partial-naming bug: a diarized voice the roster could NOT name, so a quote is
        # attributed to an unnamed ``person:speaker-xx``. Enforced signal: NAMED voices are a
        # strict minority of the attributed voices (``named*2 < distinct``) — a systematic naming
        # failure. A legit panel that names the host + main guest but leaves a couple of
        # background voices anonymous keeps named in the majority, so it no longer false-fails.
        "episodes_majority_unnamed": sum(
            1 for e in diarizable if e.named_speaker_count * 2 < e.distinct_speakers
        ),
        # Informational only (NOT enforced): ANY unnamed attributed voice — the strict
        # "every voice named" bar, which over-fires on legit multi-speaker panels.
        "episodes_with_unnamed_speaker": sum(
            1 for e in diarizable if e.distinct_speakers > e.named_speaker_count
        ),
        # Informational only (NOT enforced): multi-speaker episodes whose extracted quotes
        # happen to come from <2 distinct named people (usually a guest-dominated interview).
        "multispeaker_undernamed_episodes": sum(
            1 for e in multispeaker if e.named_speaker_count < 2
        ),
        "per_episode": [e.__dict__ for e in per_episode],
    }


def enforce_diarization_thresholds(
    metrics: Dict[str, Any],
    *,
    min_attribution_rate: float = MIN_QUOTE_ATTRIBUTION_RATE,
    min_timestamp_rate: float = MIN_QUOTE_TIMESTAMP_RATE,
    min_spoken_by_coverage: float = MIN_SPOKEN_BY_COVERAGE,
    require_num_speakers: bool = False,
) -> Tuple[bool, List[str]]:
    """Return (passed, failures). ``require_num_speakers`` is opt-in (a known propagation gap)."""
    failures: List[str] = []
    if metrics["quote_attribution_rate"] < min_attribution_rate:
        failures.append(
            f"quote_attribution_rate {metrics['quote_attribution_rate']:.3f} "
            f"< {min_attribution_rate}"
        )
    if metrics["quote_timestamp_rate"] < min_timestamp_rate:
        failures.append(
            f"quote_timestamp_rate {metrics['quote_timestamp_rate']:.3f} < {min_timestamp_rate}"
        )
    if metrics["spoken_by_coverage"] < min_spoken_by_coverage:
        failures.append(
            f"spoken_by_coverage {metrics['spoken_by_coverage']:.3f} < {min_spoken_by_coverage}"
        )
    if metrics["episodes_unattributed"] > 0:
        failures.append(
            f"{metrics['episodes_unattributed']} episode(s) have quotes but 0 attributed "
            f"speakers (#545 offset drift)"
        )
    if metrics["episodes_with_network_speaker"] > 0:
        failures.append(
            f"{metrics['episodes_with_network_speaker']} episode(s) have a network/org "
            f"speaker name in content.speakers"
        )
    if metrics.get("episodes_majority_unnamed", 0) > 0:
        failures.append(
            f"{metrics['episodes_majority_unnamed']} episode(s) attribute the majority of their "
            f"voices to unnamed speakers (person:speaker-xx) — a systematic naming failure the "
            f"roster could not resolve (#876 partial naming)"
        )
    if require_num_speakers and metrics["episodes_missing_num_speakers"] > 0:
        failures.append(
            f"{metrics['episodes_missing_num_speakers']} diarized episode(s) missing "
            f"diarization_num_speakers"
        )
    return (len(failures) == 0, failures)
