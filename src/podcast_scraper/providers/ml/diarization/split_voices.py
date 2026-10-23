"""Merge one person split into two diarized voices, by their fingerprints (``merge_split_voices``).

pyannote sometimes gives one person two voices in an episode (a host's intro and outro recorded
apart, a guest on a second line). That costs a name twice: the smaller voice stays unnamed, or a
rule names it wrongly (podcast-scraper-eval-data NAMING-RETHINK, M5: 12 of 68 unnamed Gate 1
voices were a second voice of an already-named person). The diarizer already returns one
fingerprint per voice (``DiarizationResult.speaker_embeddings``); two voices of one episode whose
fingerprints are this similar are taken to be one person, and the smaller is folded into the one
that talks more, BEFORE alignment and naming, so every later step sees one voice.

The report lists every pair's similarity, merged or not, so the threshold can be chosen from a
run without re-diarizing. Off by default; the threshold is a measured choice, not a guess baked in.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import Any, Dict, List, Sequence, Tuple

DEFAULT_THRESHOLD = 0.8
#: A voice under this many seconds is not considered: its fingerprint is too noisy to trust.
MIN_TALK_S = 10.0


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def talk_by_voice(diarization: Any) -> Dict[str, float]:
    """Seconds each diarized voice speaks."""
    talk: Dict[str, float] = {}
    for seg in getattr(diarization, "segments", None) or []:
        talk[seg.speaker] = talk.get(seg.speaker, 0.0) + max(0.0, float(seg.end) - float(seg.start))
    return talk


def merge(
    diarization: Any, threshold: float = DEFAULT_THRESHOLD, min_talk_s: float = MIN_TALK_S
) -> Tuple[Any, Dict[str, Any]]:
    """``(diarization, report)``: the diarization with split voices merged, and what was seen.

    Voices whose fingerprints are at least ``threshold`` similar (cosine) are grouped (a voice
    joins every group it matches, so A~B and B~C put A, B and C together); each group becomes the
    voice of it that talks most. Abstains (unchanged) without fingerprints.
    """
    vecs: Dict[str, List[float]] = dict(getattr(diarization, "speaker_embeddings", None) or {})
    talk = talk_by_voice(diarization)
    report: Dict[str, Any] = {"threshold": threshold, "min_talk_s": min_talk_s, "merged": []}
    voices = sorted(v for v in vecs if talk.get(v, 0.0) >= min_talk_s)
    if len(voices) < 2:
        report["abstained"] = "fewer than two fingerprinted voices"
        return diarization, report
    pairs = []
    parent = {v: v for v in voices}

    def root(v: str) -> str:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b in itertools.combinations(voices, 2):
        sim = round(_cos(vecs[a], vecs[b]), 4)
        pairs.append({"a": a, "b": b, "similarity": sim})
        if sim >= threshold:
            parent[root(a)] = root(b)
    report["pairs"] = pairs
    groups: Dict[str, List[str]] = {}
    for v in voices:
        groups.setdefault(root(v), []).append(v)
    into: Dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        keep = max(members, key=lambda v: (talk.get(v, 0.0), v))
        for v in members:
            if v != keep:
                into[v] = keep
                report["merged"].append(
                    {
                        "voice": v,
                        "into": keep,
                        "similarity": round(_cos(vecs[v], vecs[keep]), 4),
                        "talk_s": round(talk.get(v, 0.0), 1),
                    }
                )
    if not into:
        return diarization, report
    segments = [
        dataclasses.replace(s, speaker=into.get(s.speaker, s.speaker)) for s in diarization.segments
    ]
    kept = {v: vec for v, vec in vecs.items() if v not in into}
    merged = dataclasses.replace(
        diarization,
        segments=segments,
        num_speakers=len({s.speaker for s in segments}),
        speaker_embeddings=kept,
    )
    return merged, report
