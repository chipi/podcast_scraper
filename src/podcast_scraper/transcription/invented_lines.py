"""Lines Whisper writes that nobody said: subtitle credits and video sign-offs (#2187).

Whisper learnt from subtitled video, so over music, silence or speech it cannot follow it can emit
the credit line that ended its training subtitles. On the V.6b real feeds (2026-10-08) the French
episode opened with nine segments of "Sous-titrage Société Radio-Canada" while the diarizer heard
the guest talking, and the Spanish one ended on "Gracias por ver el video." Neither the loop check
nor the confidence floor catches them: each is said once per segment, at avg_logprob -0.1 to -0.5.

A segment is dropped only when its WHOLE text is one of these lines (give or take one stray token
of at most two characters at either end). A sentence that merely
contains such words (a sentence that mentions subtitles or says thanks for watching) is speech
and stays.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Mapping

#: Whole-segment lines, any language: Whisper produces them regardless of the episode's language.
INVENTED_LINES = (
    # Observed on our own runs (V.6b real feeds, 2026-10-08).
    "Sous-titrage Société Radio-Canada",
    "Sous-titrage ST' 501",
    "Gracias por ver el video.",
    # Subtitle-community credits.
    "Subtitles by the Amara.org community",
    "Sous-titres réalisés par la communauté d'Amara.org",
    "Subtítulos realizados por la comunidad de Amara.org",
    "Sottotitoli creati dalla comunità Amara.org",
    "Untertitel der Amara.org-Community",
    "Legendas pela comunidade Amara.org",
    "Sottotitoli e revisione a cura di QTSS",
    # Video sign-offs.
    "Thank you for watching.",
    "Thanks for watching!",
    "Merci d'avoir regardé cette vidéo.",
    "Gracias por ver.",
    "Grazie per la visione.",
    "Vielen Dank fürs Zuschauen.",
    "Obrigado por assistir.",
)

#: German broadcaster credits carry a year: "Untertitelung des ZDF, 2020",
#: "Untertitel im Auftrag des ZDF für funk, 2017".
_ZDF_CREDIT = re.compile(
    r"^untertitel(?:ung)? (?:des|im auftrag des) zdf(?: fur funk)?(?: \d{4})?$"
)


def _norm(text: str) -> str:
    """Lowercase, accents and punctuation removed, whitespace collapsed."""
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^\w]+", " ", folded).split())


_INVENTED = frozenset(_norm(line) for line in INVENTED_LINES)


#: A stray token this short beside an invented line does not make it speech: Rádio Novelo's
#: closing music came out twice as "A Sous-titrage Société Radio-Canada" (2026-10-09). One such
#: token, at either end; anything longer is a sentence around the words, which stays.
_STRAY_TOKEN_MAX_CHARS = 2


def _is_invented_norm(norm: str) -> bool:
    return norm in _INVENTED or bool(_ZDF_CREDIT.match(norm))


def is_invented_line(text: Any) -> bool:
    """True when ``text`` is, in its entirety, a line Whisper invents — allowing one stray token
    of at most two characters at either end."""
    norm = _norm(str(text or ""))
    if not norm:
        return False
    if _is_invented_norm(norm):
        return True
    toks = norm.split()
    if len(toks) < 2:
        return False
    if len(toks[0]) <= _STRAY_TOKEN_MAX_CHARS and _is_invented_norm(" ".join(toks[1:])):
        return True
    return len(toks[-1]) <= _STRAY_TOKEN_MAX_CHARS and _is_invented_norm(" ".join(toks[:-1]))


def drop_invented_lines(result: Dict[str, Any]) -> Dict[str, Any]:
    """Remove whole invented segments from an ASR result; return a new dict when any were removed.

    The removed segments are recorded on the result as ``asr_invented_lines`` (``start``, ``end``,
    ``text``) and ``text`` is rebuilt from the remaining segments. With nothing to remove the input
    is returned unchanged. Their time is then uncovered, so where the diarizer heard speech there
    it is reported as untranscribed (and recovered, when recovery is on) like any other gap.
    """
    segments = result.get("segments") or []
    removed: List[Mapping[str, Any]] = [
        s for s in segments if isinstance(s, Mapping) and is_invented_line(s.get("text"))
    ]
    if not removed:
        return result
    kept = [s for s in segments if not (isinstance(s, Mapping) and is_invented_line(s.get("text")))]
    out = dict(result)
    out["segments"] = kept
    out["text"] = " ".join(
        str(s.get("text", "")).strip() for s in kept if isinstance(s, Mapping) and s.get("text")
    )
    # Added to, not replaced: a repaired window is a fresh decode and is checked again.
    out["asr_invented_lines"] = list(result.get("asr_invented_lines") or []) + [
        {"start": s.get("start"), "end": s.get("end"), "text": str(s.get("text", "")).strip()}
        for s in removed
    ]
    return out
