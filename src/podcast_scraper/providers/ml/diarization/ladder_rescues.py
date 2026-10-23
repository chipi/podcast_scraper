"""Two rescues after the ladder (``ladder_rescue_rules``): names the guards refused or never placed.

Both are ports of the rules measured in podcast-scraper-eval-data (``autoresearch/naming_gate1/
variants/e1d_third_person_introduced_strict.py`` and ``split_rule.py`` with ``strict=True``);
the operator decided to ship them behind one flag and watch their fires after deploy
(``docs/wip/NAMING-GATES-2026-10-22.md``, decision 7). Change them only with a new measurement.

- **E1d, an introduced name the third-person guard refused** (source ``e1d_rescue``). A voice still
  unnamed after the ladder, whose LLM name was refused as spoken in the third person, takes that
  name when no other voice holds it AND the conversation introduces it to this voice: the voice
  says "I'm X" / "I am X" / "my name is X", or another voice welcomes or thanks X. Tuning: of the
  names the guard refused, those with either cue were right 19 times, wrong 0.
- **S30s, one host split into two voices** (source ``s30_split_merge``). In an episode with exactly
  two speaking voices (>= 150 words each), one named N as host and the other unnamed, the LLM
  naming nobody but N, no stated name other than N at all, and the unnamed voice holding >= 30%
  of the two voices' words: the unnamed voice is N, host. Tuning: S30 +10 right, 0 wrong; the
  strict form exists because held-out case x130 put the host's name on a stated guest nobody
  introduces.

Held out together: 9 fires, 8 right; the 1 wrong was plain S30's, which this module does not run.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

E1D_SOURCE = "e1d_rescue"
S30_SOURCE = "s30_split_merge"
SPLIT_MIN_WORDS = 150
SPLIT_MIN_SHARE = 0.30
_INTRO = (
    r"\b(?:welcome|joining (?:me|us)|joined by|my guest(?: today)? is|please welcome|"
    r"here with|talking (?:to|with)|thanks?,?)\s+(?:\w+\s+){0,3}"
)


def _lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def introduced(name: str, voice: str, turns: Sequence[Tuple[str, str]]) -> Tuple[bool, bool]:
    """(self-introduction by ``voice``, introduction by another voice) of ``name`` in ``turns``."""
    toks = [re.escape(t) for t in _lower(name).split() if len(t) > 2]
    if not toks:
        return False, False
    pat = "(?:" + "|".join(toks) + ")"
    own = " ".join(t for v, t in turns if v == voice).lower()
    others = " ".join(t for v, t in turns if v != voice).lower()
    selfintro = bool(re.search(rf"\b(?:i'm|i am|my name is)\s+(?:\w+\s+){{0,2}}{pat}\b", own))
    intro = bool(re.search(_INTRO + pat + r"\b", others))
    return selfintro, intro


def rescue_introduced(
    roster: Any,
    verdicts: Sequence[Mapping[str, Any]],
    turns: Sequence[Tuple[str, str]],
    trace: Any = None,
) -> Any:
    """E1d: the roster with each introduced, third-person-refused name back on its voice."""
    refused = [v for v in verdicts or () if v.get("outcome") == "third_person"]
    if not refused:
        return roster
    held = {_lower(r.name) for r in roster.by_voice.values() if r.named}
    by_voice = dict(roster.by_voice)
    for v in refused:
        voice, name = v.get("voice"), v.get("matched") or v.get("name")
        cur = by_voice.get(voice) if voice else None
        if not voice or cur is None or cur.named or not name or _lower(name) in held:
            continue
        selfintro, intro = introduced(name, voice, turns)
        if not (selfintro or intro):
            continue
        role = v.get("role") if v.get("role") in ("host", "guest") else cur.role
        by_voice[voice] = dataclasses.replace(
            cur, name=name, role=role, named=True, source=E1D_SOURCE, forced=False
        )
        held.add(_lower(name))
        if trace is not None:
            trace.voice(
                voice,
                E1D_SOURCE,
                "named",
                name=name,
                cue="self_intro" if selfintro else "introduced",
            )
    return dataclasses.replace(roster, by_voice=by_voice)


def _norm(s: Optional[str]) -> str:
    return re.sub(r",.*$", "", (s or "").strip().lower()).replace("dr. ", "")


def _same(a: Optional[str], b: Optional[str]) -> bool:
    x, y = _norm(a), _norm(b)
    if not x or not y:
        return False
    return x == y or (x.split()[-1] == y.split()[-1] and x.split()[0][:3] == y.split()[0][:3])


def split_merge(
    named: Mapping[str, Tuple[Optional[str], Optional[str]]],
    turns: Sequence[Tuple[str, str]],
    verdicts: Optional[Sequence[Mapping[str, Any]]],
    stated: Sequence[str],
) -> Optional[Tuple[str, str]]:
    """S30s: ``(voice, name)`` to name, or None (the strict rule, as measured)."""
    if not verdicts:
        return None
    words: Dict[str, int] = {}
    for v, t in turns:
        words[v] = words.get(v, 0) + len((t or "").split())
    speaking = [v for v, n in words.items() if n >= SPLIT_MIN_WORDS]
    if len(speaking) != 2:
        return None
    have = [v for v in speaking if (named.get(v) or (None, None))[0]]
    lack = [v for v in speaking if not (named.get(v) or (None, None))[0]]
    if len(have) != 1 or len(lack) != 1 or (named.get(have[0]) or (None, None))[1] != "host":
        return None
    n, u = str(named[have[0]][0]), lack[0]
    for x in verdicts:
        said = x.get("matched") or x.get("name")
        if said and not _same(said, n):
            return None
    if any(not _same(s, n) for s in stated):
        return None
    if words[u] / (words[u] + words[have[0]]) < SPLIT_MIN_SHARE:
        return None
    return u, n


def merge_split_host(
    roster: Any,
    turns: Sequence[Tuple[str, str]],
    verdicts: Optional[Sequence[Mapping[str, Any]]],
    stated: Sequence[str],
    trace: Any = None,
) -> Any:
    """S30s: the roster with the unnamed half of a split host named as the host."""
    named = {v: (r.name if r.named else None, r.role) for v, r in roster.by_voice.items()}
    hit = split_merge(named, turns, verdicts, stated)
    if not hit:
        return roster
    voice, name = hit
    by_voice = dict(roster.by_voice)
    by_voice[voice] = dataclasses.replace(
        by_voice[voice], name=name, role="host", named=True, source=S30_SOURCE, forced=False
    )
    if trace is not None:
        trace.voice(voice, S30_SOURCE, "named", name=name)
    return dataclasses.replace(roster, by_voice=by_voice)


def apply(
    roster: Any,
    llm_report: Mapping[str, Any],
    turns: List[Tuple[str, str]],
    stated: Sequence[str],
    trace: Any = None,
) -> Any:
    """Both rescues in the measured arm's order: S30s first, then E1d.

    (``e1d_s30s.py`` installed E1d's wrapper, then S30s's wrapper around it, so S30s ran first.)
    """
    verdicts = llm_report.get("verdicts") or []
    roster = merge_split_host(roster, turns, verdicts, stated, trace)
    return rescue_introduced(roster, verdicts, turns, trace)
