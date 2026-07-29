"""Provider-specific speaker-labeling strategies (ADR-134).

Speaker labeling is coupled to the diarizer's clustering footprint. Deepgram (cloud) clusters
coarsely — it tends to merge a show's cold-open montage into the host's own cluster; pyannote
community-1 (self-hosted, the product diarizer) clusters finely — it splits each host into their own
cluster, splits some guests across two clusters, and isolates recurring ad/promo readers (sometimes
with one stray content turn merged in).

There is no single diarizer-agnostic heuristic for the cluster-shape-sensitive decisions
(host-candidate eligibility, ad/recorded detection, canonicalization gating). This module holds them
as strategies selected by ``diarization_provider``, over the shared primitives in ``roster.py`` /
``boilerplate.py``. Overfitting to a diarizer is fine HERE, where it is explicit and contained — the
sin ADR-134 fixes is provider-coupling disguised as generic logic. Deepgram is the legacy/frozen
strategy; community-1 is the product strategy and gets the real investment.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Sequence, Set, Tuple

from ....speaker_detectors.boilerplate import (
    RECORDED_MAX_SHARE,
    RECORDED_MIN_REPEATED_FRACTION,
    recorded_voices,
    repeated_fraction,
)

_WORD = re.compile(r"[a-z0-9']+")


def _recorded_voices_robust(
    ordered_turns: Sequence[Tuple[str, str]],
    talk: Dict[str, float],
    shingles: Set[str],
) -> Set[str]:
    """community-1 variant of :func:`recorded_voices`: a voice is a replayed script iff it barely
    speaks (``share < RECORDED_MAX_SHARE``) AND, after DROPPING its single longest turn WHEN that
    turn is itself non-recurring, the rest of its text is mostly recurring script.

    community-1 merges one stray content turn into an ad reader's cluster (an "I'm Amy Lawrence…"
    promo picked up a 38-word interview sentence), dragging the whole-cluster ``repeated_fraction``
    below the bar and hiding the ad. Dropping that single non-recurring turn recovers the ad; a PURE
    ad reader's longest turn is itself recurring, so it is kept and the reader is still caught.
    """
    if not shingles or not ordered_turns:
        return set()
    total = sum(talk.values()) or 1.0
    by_voice: Dict[str, list] = {}
    for spk, text in ordered_turns:
        by_voice.setdefault(spk, []).append(text or "")
    out: Set[str] = set()
    for spk, turns in by_voice.items():
        if talk.get(spk, 0.0) / total >= RECORDED_MAX_SHARE:
            continue
        by_len = sorted(turns, key=lambda t: len(_WORD.findall(t.lower())), reverse=True)
        kept = by_len
        if (
            len(by_len) > 1
            and repeated_fraction(by_len[0], shingles) < RECORDED_MIN_REPEATED_FRACTION
        ):
            kept = by_len[1:]  # drop the merged stray content turn
        if repeated_fraction(" ".join(kept), shingles) >= RECORDED_MIN_REPEATED_FRACTION:
            out.add(spk)
    return out


class DiarizationLabelingStrategy:
    """Base strategy = Deepgram / coarse clustering (legacy, frozen at the v2.1.x behavior).

    Subclasses override only the cluster-shape-sensitive hooks. Everything else stays in the shared
    resolver.
    """

    name = "deepgram"

    def recorded_voices(
        self,
        ordered_turns: Optional[Sequence[Tuple[str, str]]],
        voice_texts: Optional[Dict[str, str]],
        talk: Dict[str, float],
        shingles: Set[str],
    ) -> Set[str]:
        """Voices whose turns are recurring feed boilerplate (ads/promos), not people (ADR-134)."""
        # Coarse clustering keeps ad clusters textually pure, so the whole-cluster test is enough.
        return recorded_voices(voice_texts or {}, talk, shingles)

    def host_candidate_voices(
        self,
        *,
        first_start: Dict[str, float],
        talk: Dict[str, float],
        known_hosts: Sequence[str],
        conv_guests: Set[str],
        montage_suppressed: Set[str],
        cameo_floor: float,
    ) -> Set[str]:
        """Voices eligible to be named as hosts. Base/deepgram: the first ``len(known_hosts)``
        speakers, since the hosts open the show (frozen v2.1.x behavior)."""
        # Deepgram/legacy: the hosts open the show, so the first ``len(known_hosts)`` speakers (ads
        # already excluded from ``first_start``) ARE the host candidates. Frozen v2.1.x behavior.
        return set(sorted(first_start, key=lambda v: first_start[v])[: len(known_hosts)])

    def snap_extra(self, name: str, known_hosts: Sequence[str]) -> Optional[str]:
        """Provider-specific fallback to resolve a garbled host name; base has none (ADR-134)."""
        # Deepgram/legacy: no fallback beyond the shared surname canonicalization.
        return None


def _unique_first_name_host(name: str, known_hosts: Sequence[str]) -> Optional[str]:
    """The known host whose FIRST name uniquely matches this name's first token, else None. First
    names are short and ASR-stable; inside the host-eligibility gate a garbled surname
    ("Casey Noonan" for "Casey Newton") that fails the surname tests still resolves by first name —
    exactly one configured host with that first name. Uniqueness abstains on same-first-name hosts.
    """
    toks = (name or "").split()
    if len(toks) < 2:
        return None
    first = toks[0].strip(".,'’").lower()
    matches = [h for h in known_hosts if h.split() and h.split()[0].strip(".,'’").lower() == first]
    return matches[0] if len(matches) == 1 else None


class Community1LabelingStrategy(DiarizationLabelingStrategy):
    """pyannote community-1 (fine clustering) — the product strategy (ADR-134)."""

    name = "pyannote_community1"

    def recorded_voices(
        self,
        ordered_turns: Optional[Sequence[Tuple[str, str]]],
        voice_texts: Optional[Dict[str, str]],
        talk: Dict[str, float],
        shingles: Set[str],
    ) -> Set[str]:
        """Recorded (ad/promo) voices — community-1's robust per-turn variant (ADR-134)."""
        return _recorded_voices_robust(ordered_turns or [], talk, shingles)

    def host_candidate_voices(
        self,
        *,
        first_start: Dict[str, float],
        talk: Dict[str, float],
        known_hosts: Sequence[str],
        conv_guests: Set[str],
        montage_suppressed: Set[str],
        cameo_floor: float,
    ) -> Set[str]:
        """Host-eligible voices under fine clustering: the first real speakers, excluding
        conversational guests, montage clips, and sub-cameo fragments (ADR-134)."""
        # community-1 splits hosts into their own clusters and leaves cold-open ad/promo/cameo
        # fragments as their own first-speaking clusters, so "first to speak" no longer means
        # "host". Restrict host slots to PLAUSIBLE PEOPLE — exclude conversational guests, montage
        # clips, and sub-cameo fragments — then take the first ``len(known_hosts)`` of those by
        # first-speak. This is person-vs-noise use of talk time, never host-vs-guest ranking (the
        # dominant talker is often the guest — N1); the real hosts are the first REAL speakers.
        eligible = [
            v
            for v in first_start
            if v not in conv_guests
            and v not in montage_suppressed
            and talk.get(v, 0.0) >= cameo_floor
        ]
        return set(sorted(eligible, key=lambda v: first_start[v])[: len(known_hosts)])

    def snap_extra(self, name: str, known_hosts: Sequence[str]) -> Optional[str]:
        """Resolve a garbled host surname by a unique first-name match, inside the host gate."""
        # Inside the host-eligibility gate, a garbled surname that fails the shared canonicalization
        # ("Casey Noonan" for "Casey Newton") still resolves by a unique first-name match.
        return _unique_first_name_host(name, known_hosts)


_DEEPGRAM = DiarizationLabelingStrategy()
_COMMUNITY1 = Community1LabelingStrategy()


def labeling_strategy_for(diarization_provider: Optional[str]) -> DiarizationLabelingStrategy:
    """Select the labeling strategy for a diarization provider. The self-hosted pyannote path
    (``tailnet_dgx``, community-1) gets the fine-clustering strategy; everything else is legacy."""
    p = (diarization_provider or "").lower()
    if "tailnet_dgx" in p or "pyannote" in p or "community" in p:
        return _COMMUNITY1
    return _DEEPGRAM
