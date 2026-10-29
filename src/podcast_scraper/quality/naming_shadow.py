"""The naming shadow report: what the side namers and the new ladder rules did on served episodes.

Read only, from the speaker-diagnostics sidecars the pipeline writes (coding plan C9,
podcast-scraper-eval-data ``docs/wip/NAMING-GATES-2026-10-22.md``). Three tables:

- **Side namers.** Per namer that wrote a sidecar key (``transcript_naming``, the LLM pass, staged
  or bundled; ``voice_identity``, the fingerprint rung): episodes, calls that answered, and per
  voice agree / adds / removes / changes against the ladder's published name. Every non-agree
  voice is listed, so a reader can judge it; the counts alone say nothing about who is right.
  An episode where the namer did not answer (the LLM call failed; the fingerprint rung abstained,
  e.g. no fingerprints or no voted group yet) is counted under ``not_answered`` by reason and
  contributes no outcome: its "agree" rows only repeat the ladder. A voice a namer never judged is
  ``not_judged``, never ``agree`` or ``removes``: for the LLM pass a voice it was not asked about
  (under 40 words, not in ``voices_asked``); for the fingerprint rung a voice without a
  fingerprint, or an unnamed one under ``voice_identity.MIN_VOICE_TALK_S`` (``decide`` names
  neither; it still checks a named short voice for removal).
- **Talk floor.** Every name the floor took off (the decision trace's ``talk_floor`` /
  ``removed`` step), by the role the LLM pass gave that voice. The role is a PROXY for whether
  the name was the voice's own (an ``ad`` / ``promo`` voice losing a name is what the floor is
  for; a ``host`` losing one is a loss to read), not a label.
- **Rescues.** Every voice published by E1d (``e1d_rescue``) or S30s (``s30_split_merge``), with
  the LLM pass's name for the same voice beside it.

Episodes written before these keys existed are counted as ``without``, never as agreement.

``episodes_shadowed`` lists every episode that has a side namer's key: its run (relative to the
corpus root), its processing timestamp from the metadata, and per namer its outcome counts or why
it did not answer. A daily snapshot diffs these ids against the previous one to see last night's
episodes; the tables above only count an all-agree or a not-answered episode.

Every listed row carries the episode's feed and title and ``insights``: how many of the episode's GI
insights sit on that voice (placed as S7 places them, ``gold_insights.place``: the first
``SUPPORTED_BY`` quote, checked by its text), so the rows a reader should read first are the ones
whose name is on the most insights. ``None`` when the episode has no GI file.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..upgrade.corpus_selection import select_served_artifacts
from ..upgrade.migrations.m0025_one_person_one_entry import _load, _run_relative

NAMERS = ("transcript_naming", "voice_identity")
OUTCOMES = ("agree", "adds", "removes", "changes")
#: A voice the LLM pass was not asked about; counted apart from the four outcomes.
NOT_JUDGED = "not_judged"
#: What the floor table shows when the LLM pass gave no answer for the voice.
NO_LLM_ANSWER = "no LLM answer"
RESCUE_SOURCES = ("e1d_rescue", "s30_split_merge")
#: The result key each namer's per-voice entry carries its own name under.
_RESULT_KEY = {"transcript_naming": "model_result", "voice_identity": "fingerprint_result"}


def insights_by_voice(meta_path: Path) -> Optional[Dict[str, int]]:
    """GI insights per raw voice, each placed by its first quote (``gold_insights.place``)."""
    from .gold_insights import _load as _load_json, _Texts, place

    gi = _load_json(Path(str(meta_path)[: -len(".metadata.json")] + ".gi.json"))
    if not isinstance(gi, dict):
        return None
    nodes = {str(n.get("id")): n for n in gi.get("nodes") or [] if isinstance(n, dict)}
    first: Dict[str, str] = {}
    for e in gi.get("edges") or []:
        if isinstance(e, dict) and e.get("type") == "SUPPORTED_BY":
            first.setdefault(str(e.get("from")), str(e.get("to")))
    texts, run = _Texts(), Path(meta_path).parent.parent
    out: Counter = Counter()
    root = run.parent.parent.parent  # <root>/feeds/<feed>/<run>
    for iid, node in nodes.items():
        q = nodes.get(first.get(iid, ""))
        if node.get("type") != "Insight" or q is None:
            continue
        reason, voice, _label = place(q, root, run, texts)
        if reason in ("ok", "relocated") and voice:
            out[voice] += 1
    return dict(out)


def _episode_view(meta_path: Path) -> Dict[str, Any]:
    meta = _load(meta_path) or {}
    return {
        "feed": str((meta.get("feed") or {}).get("title") or ""),
        "title": str((meta.get("episode") or {}).get("title") or ""),
    }


def served_diagnostics(root: Path) -> Iterable[Tuple[str, Path, Dict[str, Any]]]:
    """``(episode id, metadata path, diagnostics)`` for every served episode with a sidecar."""
    for meta_path in select_served_artifacts(Path(root), ".metadata.json")[0]:
        meta = _load(meta_path)
        if not isinstance(meta, dict):
            continue
        eid = str((meta.get("episode") or {}).get("episode_id") or "")
        rel = _run_relative(
            (meta.get("content") or {}).get("transcript_file_path"), meta_path.parent.parent
        )
        if not eid or not rel.endswith(".txt"):
            continue
        diag = _load(meta_path.parent.parent / rel.replace(".txt", ".speakers.diagnostics.json"))
        if isinstance(diag, dict):
            yield eid, meta_path, diag


def _model_view(diag: Mapping[str, Any], voice: str) -> Dict[str, Any]:
    """The LLM pass's raw answer for one voice (``{}`` when it did not run or name it)."""
    entry = ((diag.get("transcript_naming") or {}).get("voices") or {}).get(voice) or {}
    model = entry.get("model")
    return model if isinstance(model, dict) else {}


def not_answered(block: Mapping[str, Any], namer: str) -> Optional[str]:
    """Why a namer gave no answer on this episode, or None when it answered."""
    raw = block.get("report")
    report: Mapping[str, Any] = raw if isinstance(raw, dict) else {}
    if namer == "voice_identity":
        return str(report["abstained"]) if report.get("abstained") else None
    if block.get("ok") is False:
        return str(report.get("error") or report.get("skipped") or "call gave no answer")
    return None


def _rung_judged(entry: Mapping[str, Any]) -> bool:
    """Whether the fingerprint rung could change this voice (see the module doc)."""
    from ..providers.ml.diarization.voice_identity import MIN_VOICE_TALK_S

    match = entry.get("match")
    if not isinstance(match, dict):
        return False  # no fingerprint for this voice
    named = bool((entry.get("ladder") or {}).get("name"))
    return named or float(match.get("talk_s") or 0.0) >= MIN_VOICE_TALK_S


def namer_rows(eid: str, diag: Mapping[str, Any], namer: str) -> List[Dict[str, Any]]:
    """One row per voice a namer compared with the ladder (empty when it wrote nothing or did not
    answer). A voice the LLM pass was not asked about is ``not_judged``."""
    block = diag.get(namer)
    if not isinstance(block, dict) or not_answered(block, namer):
        return []
    asked = block.get("voices_asked")
    asked_set = {str(v) for v in asked} if isinstance(asked, list) else None
    out: List[Dict[str, Any]] = []
    for voice, entry in (block.get("voices") or {}).items():
        if not isinstance(entry, dict) or entry.get("outcome") not in OUTCOMES:
            continue
        outcome = entry["outcome"]
        if namer == "transcript_naming" and asked_set is not None and voice not in asked_set:
            outcome = NOT_JUDGED
        if namer == "voice_identity" and not _rung_judged(entry):
            outcome = NOT_JUDGED
        result = entry.get(_RESULT_KEY[namer]) or {}
        out.append(
            {
                "episode": eid,
                "voice": voice,
                "outcome": outcome,
                "ladder": (entry.get("ladder") or {}).get("name"),
                "namer": result.get("name"),
                "namer_role": result.get("role"),
            }
        )
    return out


def floor_rows(eid: str, diag: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every name the talk floor took off, with the LLM pass's role for that voice."""
    trace = diag.get("decision_trace") or {}
    out: List[Dict[str, Any]] = []
    for voice, steps in (trace.get("voices") or {}).items():
        for step in steps or []:
            if step.get("rung") == "talk_floor" and step.get("decision") == "removed":
                model = _model_view(diag, voice)
                out.append(
                    {
                        "episode": eid,
                        "voice": voice,
                        "name": step.get("name"),
                        "talk_share": step.get("talk_share"),
                        "llm_role": (
                            (model.get("role") or NO_LLM_ANSWER) if model else NO_LLM_ANSWER
                        ),
                        "llm_name": model.get("name") if model else None,
                    }
                )
    return out


def rescue_rows(eid: str, diag: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every voice a ladder rescue published, with the LLM pass's name for it."""
    out: List[Dict[str, Any]] = []
    for v in diag.get("voices") or []:
        if v.get("source") in RESCUE_SOURCES and v.get("named"):
            model = _model_view(diag, str(v.get("voice")))
            out.append(
                {
                    "episode": eid,
                    "voice": v.get("voice"),
                    "rule": v.get("source"),
                    "name": v.get("resolved_name"),
                    "role": v.get("role"),
                    "llm_name": model.get("name") if model else None,
                }
            )
    return out


def episode_row(root: Path, eid: str, meta_path: Path, per_namer: Dict[str, Any]) -> Dict[str, Any]:
    """One shadowed episode: where it lives and when it was processed, with each namer's result.

    ``run`` is relative to the corpus root, so the row reads the same on any host; ``processed``
    is the metadata's own processing timestamp, never a file time."""
    meta = _load(meta_path) or {}
    run = Path(meta_path).parent.parent
    try:
        rel = run.relative_to(root)
    except ValueError:
        rel = run.resolve().relative_to(Path(root).resolve())
    return {
        "episode": eid,
        "feed": str((meta.get("feed") or {}).get("title") or ""),
        "title": str((meta.get("episode") or {}).get("title") or ""),
        "run": rel.as_posix(),
        "processed": (meta.get("processing") or {}).get("processing_timestamp"),
        "namers": per_namer,
    }


def report(root: Path) -> Dict[str, Any]:
    """The three tables over every served episode of ``root``, plus ``episodes_shadowed``: one row
    per episode that has a side namer's key, so a reader can tell which episodes are new since an
    earlier report and what each namer did on each (an all-agree episode or a not-answered one
    is otherwise only a count)."""
    episodes = 0
    episode_rows: List[Dict[str, Any]] = []
    namers: Dict[str, Dict[str, Any]] = {
        n: {
            "episodes": 0,
            "answered": 0,
            "not_answered": Counter(),
            "outcomes": Counter(),
            "differences": [],
        }
        for n in NAMERS
    }
    floor: List[Dict[str, Any]] = []
    rescues: List[Dict[str, Any]] = []
    for eid, meta_path, diag in served_diagnostics(root):
        episodes += 1
        listed: List[Dict[str, Any]] = []
        per_namer: Dict[str, Any] = {}
        for n in NAMERS:
            if not isinstance(diag.get(n), dict):
                continue
            t = namers[n]
            t["episodes"] += 1
            reason = not_answered(diag[n], n)
            if reason:
                t["not_answered"][reason] += 1
                per_namer[n] = {"answered": False, "reason": reason}
                continue
            t["answered"] += 1
            mine: Counter = Counter()
            for r in namer_rows(eid, diag, n):
                t["outcomes"][r["outcome"]] += 1
                mine[r["outcome"]] += 1
                if r["outcome"] not in ("agree", NOT_JUDGED):
                    t["differences"].append(r)
                    listed.append(r)
            per_namer[n] = {
                "answered": True,
                "outcomes": {o: mine.get(o, 0) for o in (*OUTCOMES, NOT_JUDGED)},
            }
        if per_namer:
            episode_rows.append(episode_row(root, eid, meta_path, per_namer))
        ep_floor, ep_rescues = floor_rows(eid, diag), rescue_rows(eid, diag)
        floor.extend(ep_floor)
        rescues.extend(ep_rescues)
        listed += ep_floor + ep_rescues
        if listed:
            counts, view = insights_by_voice(meta_path), _episode_view(meta_path)
            for r in listed:
                r.update(view, insights=None if counts is None else counts.get(str(r["voice"]), 0))
    for t in namers.values():
        t["outcomes"] = {o: t["outcomes"].get(o, 0) for o in (*OUTCOMES, NOT_JUDGED)}
        t["not_answered"] = dict(t["not_answered"].most_common())
        t["without"] = episodes - t["episodes"]
    return {
        "episodes": episodes,
        "namers": namers,
        "floor": {
            "removed": len(floor),
            "by_llm_role": dict(Counter(str(r["llm_role"]) for r in floor).most_common()),
            "rows": floor,
        },
        "rescues": {
            "fires": len(rescues),
            "by_rule": dict(Counter(str(r["rule"]) for r in rescues).most_common()),
            "llm_agrees": sum(
                1 for r in rescues if (r["llm_name"] or "").lower() == (r["name"] or "").lower()
            ),
            "rows": rescues,
        },
        "episodes_shadowed": episode_rows,
    }


def _where(r: Mapping[str, Any]) -> str:
    ins = r.get("insights")
    return (
        f" | {'no GI' if ins is None else f'{ins} insights'} | {r.get('feed')} — {r.get('title')}"
    )


def render(rep: Mapping[str, Any], limit: Optional[int] = None) -> str:
    """The report as plain text tables (``limit`` caps each listed table's rows)."""
    lines = [f"served episodes with a speaker sidecar: {rep['episodes']}", ""]
    for n, t in rep["namers"].items():
        o = t["outcomes"]
        lines.append(
            f"{n}: {t['episodes']} episodes ({t['without']} without), {t['answered']} answered; "
            + ", ".join(f"{k} {o[k]}" for k in (*OUTCOMES, NOT_JUDGED))
        )
        if t["not_answered"]:
            lines.append(f"  not answered: {t['not_answered']}")
        for r in t["differences"][:limit]:
            lines.append(
                f"  {r['outcome']:<8} {r['episode']} {r['voice']}: "
                f"ladder {r['ladder']!r} -> {n} {r['namer']!r} ({r['namer_role']})" + _where(r)
            )
        lines.append("")
    f = rep["floor"]
    lines.append(
        f"talk floor: {f['removed']} names removed; by the LLM pass's role {f['by_llm_role']}"
    )
    for r in f["rows"][:limit]:
        lines.append(
            f"  {r['episode']} {r['voice']}: {r['name']!r} at {r['talk_share']} "
            f"(LLM: {r['llm_role']}, {r['llm_name']!r})" + _where(r)
        )
    lines.append("")
    s = rep["rescues"]
    lines.append(
        f"rescues: {s['fires']} fires {s['by_rule']}; the LLM pass names the same person "
        f"{s['llm_agrees']} times"
    )
    for r in s["rows"][:limit]:
        lines.append(
            f"  {r['rule']:<16} {r['episode']} {r['voice']}: {r['name']!r} ({r['role']}); "
            f"LLM {r['llm_name']!r}" + _where(r)
        )
    return "\n".join(lines) + "\n"
