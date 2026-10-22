"""Cross-episode voice identity: the last rung of the naming ladder (design stage 3).

docs/wip/VOICE-IDENTITY-DESIGN-2026-10-12.md. Each diarized voice carries a fingerprint (the
diarizer's per-speaker embedding, stored as ``voice_embeddings`` in the speakers diagnostics
sidecar). This rung compares the episode's voices with the voices of the SAME feed's other
episodes:

1. The feed's other episodes' voices (>= ``min_talk`` seconds, same embedding model) are grouped
   by fingerprint, longest talkers first: a voice joins the group whose centroid it matches best
   at >= ``threshold``, else starts one.
2. A group is named by vote: the served name most of its named host/guest members carry, when
   that name comes from >= 2 episodes AND from more than half of the named members. A minority of
   wrong or swapped served names cannot decide it.
3. In this episode, every voice (>= 10 s) whose best group matches at >= ``threshold`` takes that
   group's name and majority role, overriding the name the text gave. The name is then removed
   from any other voice of the episode that carries it and is clearly NOT that person (similarity
   to the group below ``not_same``).

Measured (podcast-scraper-eval-data, autoresearch/voice_embedding_match, rule ``cluster-all``):
on val_v4, 50 labelled episodes of 10 feeds never used to shape it, pre-registered, wrong names
23 -> 17 (p = 0.031) and correct names 70 -> 77 (p = 0.014); every changed voice was read and none
went from a correct name to a wrong one. Never across feeds; the fingerprints are never served.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PARTICIPANT_ROLES = ("host", "guest")
SOURCE = "voice_identity"


@dataclasses.dataclass(frozen=True)
class HistoryVoice:
    """One voice of another episode of the feed: its served name, role, talk and fingerprint."""

    episode_id: str
    name: Optional[str]
    role: Optional[str]
    talk: float
    vec: Tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class VoiceGroup:
    """Voices of the feed that sound alike, with the name their vote chose."""

    person: str  # lowercased name the vote chose
    display: str  # its most common spelling
    role: str
    centroid: Tuple[float, ...]


def _unit(vec: Sequence[float]) -> Tuple[float, ...]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return tuple(x / norm for x in vec)


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _key(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def voices_from_diagnostics(
    diag: Mapping[str, Any], episode_id: str, model_name: Optional[str]
) -> List[HistoryVoice]:
    """One episode's served voices with their fingerprints (only from ``model_name``)."""
    sd = diag.get("speaker_diagnostics", diag)
    block = sd.get("voice_embeddings") or {}
    if model_name and block.get("model_name") and block.get("model_name") != model_name:
        return []
    by_voice = block.get("by_voice") or {}
    out = []
    for v in sd.get("voices") or []:
        vec = by_voice.get(v.get("voice"))
        if not vec:
            continue
        out.append(
            HistoryVoice(
                episode_id=episode_id,
                name=v.get("resolved_name") if v.get("named") else None,
                role=v.get("role"),
                talk=float(v.get("talk_time_s") or 0.0),
                vec=_unit(vec),
            )
        )
    return out


def feed_history(
    feed_dir: Path, exclude_episode_id: str, model_name: Optional[str]
) -> List[HistoryVoice]:
    """The feed's OTHER episodes' voices, newest run per episode, read from the feed workspace."""
    from ....search.corpus_scope import dedupe_metadata_paths_newest_run_per_episode

    metas = sorted(feed_dir.glob("run_*/metadata/*.metadata.json"))
    corpus_root = feed_dir.parent.parent if feed_dir.parent.name == "feeds" else feed_dir
    out: List[HistoryVoice] = []
    for meta in dedupe_metadata_paths_newest_run_per_episode(corpus_root, metas):
        meta = Path(meta)
        try:
            doc = json.loads(meta.read_text(encoding="utf-8"))
            eid = str((doc.get("episode") or {}).get("episode_id") or "")
            if not eid or eid == exclude_episode_id:
                continue
            rel = str((doc.get("content") or {}).get("transcript_file_path") or "")
            diag = meta.parent.parent / rel.replace(".txt", ".speakers.diagnostics.json")
            if rel and diag.is_file():
                out += voices_from_diagnostics(
                    json.loads(diag.read_text(encoding="utf-8")), eid, model_name
                )
        except (OSError, ValueError) as exc:
            logger.debug("voice identity: unreadable sibling %s (%s); skipped", meta, exc)
    return out


def build_groups(
    history: Sequence[HistoryVoice], threshold: float, min_talk: float
) -> List[VoiceGroup]:
    """The feed's voices grouped by fingerprint; only groups whose vote names a person."""
    pool = sorted((v for v in history if v.talk >= min_talk), key=lambda v: -v.talk)
    groups: List[Dict[str, Any]] = []
    for v in pool:
        best = max(groups, key=lambda g: _cos(v.vec, g["c"]), default=None)
        if best is not None and _cos(v.vec, best["c"]) >= threshold:
            best["m"].append(v)
            best["s"] = [a + b for a, b in zip(best["s"], v.vec)]
            best["c"] = _unit(best["s"])
        else:
            groups.append({"m": [v], "s": list(v.vec), "c": v.vec})
    out = []
    for g in groups:
        named = [v for v in g["m"] if v.name and v.role in PARTICIPANT_ROLES]
        votes = collections.Counter(_key(v.name) for v in named)
        if not votes:
            continue
        person, n = votes.most_common(1)[0]
        mine = [v for v in named if _key(v.name) == person]
        if len({v.episode_id for v in mine}) >= 2 and n * 2 > len(named):
            out.append(
                VoiceGroup(
                    person=person,
                    display=collections.Counter(v.name for v in mine).most_common(1)[0][0],
                    role=collections.Counter(v.role for v in mine).most_common(1)[0][0],
                    centroid=tuple(g["c"]),
                )
            )
    return out


def decide(
    voices: Mapping[str, Tuple[Optional[str], Optional[str], float, Optional[Sequence[float]]]],
    groups: Sequence[VoiceGroup],
    threshold: float,
    not_same: float = 0.5,
    min_voice_talk: float = 10.0,
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """``{voice: (name or None, role)}`` for the voices this rung changes.

    ``voices`` maps each voice to (served name or None, role, talk seconds, fingerprint or None).
    """
    vecs = {v: _unit(x[3]) for v, x in voices.items() if x[3]}
    owner: Dict[str, List[Tuple[str, VoiceGroup]]] = collections.defaultdict(list)
    for v, (_, _, talk, _) in voices.items():
        if v not in vecs or talk < min_voice_talk or not groups:
            continue
        g = max(groups, key=lambda g: _cos(vecs[v], g.centroid))
        if _cos(vecs[v], g.centroid) >= threshold:
            owner[g.person].append((v, g))
    out: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for person, members in owner.items():
        for v, g in members:
            if _key(voices[v][0]) != person:
                out[v] = (g.display, g.role)
        mine = {v for v, _ in members}
        # Two groups can win the same name; as measured, the LAST of them is the one compared.
        g = [x for x in groups if x.person == person][-1]
        for other, (name, role, _, _) in voices.items():
            if other in mine or _key(name) != person:
                continue
            sim = _cos(vecs[other], g.centroid) if other in vecs else 1.0
            if sim < not_same:
                out[other] = (None, role)
    return out


def talk_by_voice(diarization: Any) -> Dict[str, float]:
    """Seconds each diarized voice speaks in this episode."""
    talk: Dict[str, float] = collections.defaultdict(float)
    for seg in getattr(diarization, "segments", None) or []:
        speaker = getattr(seg, "speaker", None)
        if speaker is not None:
            talk[speaker] += max(0.0, float(seg.end) - float(seg.start))
    return dict(talk)


def apply_to_roster(
    roster: Any,
    diarization: Any,
    feed_dir: Optional[Path],
    episode_id: Optional[str],
    threshold: float,
    trace: Any = None,
    min_talk: float = 60.0,
) -> Any:
    """The roster after this rung. Abstains (returns it unchanged) without an episode id, a feed
    workspace, fingerprints for this episode, or any voted group."""
    vectors = dict(getattr(diarization, "speaker_embeddings", None) or {})
    if not vectors or not feed_dir or not episode_id:
        return roster
    model = getattr(diarization, "speaker_embeddings_model", "") or getattr(
        diarization, "model_name", ""
    )
    groups = build_groups(
        feed_history(Path(feed_dir), episode_id, model or None), threshold, min_talk
    )
    if not groups:
        return roster
    talk = talk_by_voice(diarization)
    voices = {
        v: (r.name if r.named else None, r.role, talk.get(v, 0.0), vectors.get(v))
        for v, r in roster.by_voice.items()
    }
    changes = decide(voices, groups, threshold)
    if not changes:
        return roster
    by_voice = dict(roster.by_voice)
    for v, (name, role) in changes.items():
        old = by_voice[v]
        if name:
            by_voice[v] = dataclasses.replace(
                old, name=name, role=role or old.role, named=True, source=SOURCE, forced=False
            )
        else:
            by_voice[v] = dataclasses.replace(old, name=v, named=False, source=SOURCE)
        if trace is not None:
            trace.voice(
                v,
                SOURCE,
                "named" if name else "removed",
                name=name or (old.name if old.named else None),
            )
    logger.info(
        "voice identity: %d voice(s) changed by feed fingerprints (%d groups)",
        len(changes),
        len(groups),
    )
    return dataclasses.replace(roster, by_voice=by_voice)
