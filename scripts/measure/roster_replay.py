#!/usr/bin/env python3
"""Replay the speaker roster over STORED episodes, old code vs new code. Read-only.

Every naming change of 2026-10-01/02 was decided with this: re-run production's own roster over
each stored episode, fed exactly what the diarization pipeline handed it — voice texts and turns
from ``.segments.json``, the ad intervals, the feed's recurring script, the shared voice cleaning,
and the LLM's per-voice names/roles as stored in ``.speakers.diagnostics.json`` — so no LLM is
called and nothing is written. Replay fidelity measured 124/125 voices on fresh episodes.

Two code VARIANTS are compared. A variant is a set of module source files that replace the
installed ones while that variant runs (the roster, the host/name rules, the resolution guards,
the ad signatures). Take the old side from git and the new side from the working tree:

    git show HEAD~1:src/podcast_scraper/providers/ml/diarization/roster.py > /tmp/old_roster.py
    python scripts/measure/roster_replay.py --corpus /app/output \\
        --old roster=/tmp/old_roster.py \\
        --new roster=src/podcast_scraper/providers/ml/diarization/roster.py \\
        --signatures > changes.jsonl

Output: one JSON line per voice whose published name changed (kind, feed, episode, voice, talk
share, old name, new name, old/new voice type, opening words), then a summary line.

Rule of use: read every changed voice. A replay that only reports counts proves nothing.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

_PKG = "podcast_scraper"
#: Short names for the modules a variant may replace.
MODULES = {
    "roster": f"{_PKG}.providers.ml.diarization.roster",
    "hosts": f"{_PKG}.speaker_detectors.hosts",
    "resolution": f"{_PKG}.speaker_detectors.resolution",
    "ad_signatures": f"{_PKG}.providers.ml.diarization.ad_signatures",
}


def load_episode(meta: Path) -> Dict[str, Any]:
    """The stored artifacts of one episode: metadata, segments, speaker diagnostics."""
    doc = json.loads(meta.read_text(encoding="utf-8"))
    rel = str((doc.get("content") or {}).get("transcript_file_path") or "")
    run = meta.parent.parent
    segs = json.loads((run / rel.replace(".txt", ".segments.json")).read_text(encoding="utf-8"))
    segs = segs if isinstance(segs, list) else segs.get("segments", [])
    diag_path = run / rel.replace(".txt", ".speakers.diagnostics.json")
    diag = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.is_file() else {}
    return {"meta": doc, "segs": segs, "diag": diag, "feed_dir": run.parent, "meta_path": str(meta)}


#: Per feed dir: (metadata path, speaker diagnostics) of every newest-run episode. Filled by
#: :func:`index_siblings`; a variant that reads the feed's history (``host_copresence``) gets the
#: OTHER episodes of the same feed, never the one being replayed.
_SIBLINGS: Dict[str, List[Any]] = {}

#: ``--repool``: rebuild each episode's host pool with the VARIANT's own ``hosts`` module instead of
#: replaying the pool stored in its diagnostics. What it recomputes and what it cannot see is in
#: :func:`repool`.
REPOOL = False
#: ``--roles``: also report a voice whose NAME is unchanged but whose host/guest ROLE flipped.
ROLES = False


def index_siblings(corpus: Path) -> None:
    """Index every newest-run episode's diagnostics by feed dir (for feed-history variants)."""
    from podcast_scraper.search.corpus_scope import (
        dedupe_metadata_paths_newest_run_per_episode,
        discover_metadata_files,
    )

    _SIBLINGS.clear()
    for meta in dedupe_metadata_paths_newest_run_per_episode(
        corpus, discover_metadata_files(corpus)
    ):
        meta = Path(meta)
        try:
            doc = json.loads(meta.read_text(encoding="utf-8"))
            rel = str((doc.get("content") or {}).get("transcript_file_path") or "")
            run = meta.parent.parent
            diag_path = run / rel.replace(".txt", ".speakers.diagnostics.json")
            diag = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.is_file() else {}
        except (OSError, ValueError, KeyError):
            continue
        _SIBLINGS.setdefault(str(run.parent), []).append((str(meta), diag))


def repool(hosts_mod: Any, ep: Dict[str, Any], tried: Dict[str, Any]) -> List[str]:
    """This episode's host pool, rebuilt from its stored metadata with *hosts_mod*.

    SEEN: the feed's title / description / author tags and the episode's title / description are on
    the metadata, so the feed-level detector (``detect_hosts_from_feed``) and the episode-level
    composer (``compose_episode_hosts``, when the variant has one) run exactly as production would.

    NOT SEEN: config ``known_hosts``, the recurrence scan over sibling transcripts, and the
    episode's own ``<itunes:author>`` byline are not stored. Whatever the STORED pool holds beyond
    the variant's feed-level result (``residue``) stands in for them. For a variant with a composer,
    residue names the episode's metadata also states as participants (``metadata_named`` /
    ``detected_guests``) are treated as the byline (dropped when the description names the host);
    the rest as config/recurrence (kept, person-checked). A variant without a composer gets
    ``feed_level + residue`` — the stored pool refreshed by its own detector.
    """
    from podcast_scraper.kg.speaker_coherence import same_person

    feed = ep["meta"].get("feed") or {}
    episode = ep["meta"].get("episode") or {}
    stored = [str(n) for n in (tried.get("known_hosts") or []) if n]
    feed_level = sorted(
        hosts_mod.detect_hosts_from_feed(
            feed.get("title"), feed.get("description"), feed.get("authors") or []
        )
    )
    residue = [n for n in stored if not any(same_person(n, h) for h in feed_level)]
    if not hasattr(hosts_mod, "compose_episode_hosts"):
        return list(dict.fromkeys(feed_level + residue))
    participants = list(tried.get("metadata_named") or []) + list(
        tried.get("detected_guests") or []
    )
    byline = [n for n in residue if any(same_person(n, p) for p in participants)]
    more = [n for n in residue if n not in byline]
    extra: Dict[str, Any] = {}
    if "episode_people" in inspect.signature(hosts_mod.compose_episode_hosts).parameters:
        extra["episode_people"] = participants
    return list(
        hosts_mod.compose_episode_hosts(
            feed_level,
            byline,
            episode_title=episode.get("title"),
            episode_description=episode.get("description"),
            feed_title=feed.get("title"),
            more=more,
            **extra,
        )
    )


def roster_inputs(ep: Dict[str, Any], roster: Any, ad_signatures: Any = None) -> Dict[str, Any]:
    """What ``providers/ml/diarization/pipeline.py`` hands ``resolve_speaker_roster``."""
    from podcast_scraper.providers.ml.diarization import pipeline as P
    from podcast_scraper.providers.ml.diarization.base import (
        DiarizationResult,
        DiarizationSegment,
    )

    segs = [s for s in ep["segs"] if s.get("speaker")]
    turns = [(str(s["speaker"]), str(s.get("text") or "")) for s in segs]
    chunks: Dict[str, List[str]] = {}
    for v, t in turns:
        chunks.setdefault(v, []).append(t)
    voice_texts = {v: " ".join(x) for v, x in chunks.items()}
    dz = DiarizationResult(
        segments=[
            DiarizationSegment(float(s.get("start") or 0), float(s.get("end") or 0), s["speaker"])
            for s in segs
        ],
        num_speakers=len(voice_texts),
    )
    ad_intervals = P._ad_intervals(ep["segs"])
    # The production reader only needs ``output_dir``; a namespace stands in for the Config.
    feed_cfg: Any = SimpleNamespace(output_dir=str(ep["feed_dir"]))
    recurring = P._feed_recurring_text(feed_cfg)
    extra = {"ad_signatures": ad_signatures} if ad_signatures is not None else {}
    cleaning = roster.classify_voices(
        dz,
        ad_intervals,
        voice_texts=voice_texts,
        ordered_turns=turns,
        recurring_text=recurring,
        diarization_provider="tailnet_dgx",
        cameo_max_talk_s=20.0,
        **extra,
    )
    voices = ep["diag"].get("voices") or []
    llm = [v for v in voices if v.get("source") == "llm_resolution"]
    tried = ep["diag"].get("tried") or {}
    episode = ep["meta"].get("episode") or {}
    hosts_mod = getattr(roster, "_replay_hosts", None)
    known_hosts = (
        repool(hosts_mod, ep, tried)
        if REPOOL and hosts_mod is not None
        else list(tried.get("known_hosts") or [])
    )
    return dict(
        diarization=dz,
        transcript_text=" ".join(t for _, t in turns),
        detected_guests=tried.get("detected_guests") or [],
        known_hosts=known_hosts,
        voice_texts=voice_texts,
        ordered_turns=turns,
        ad_intervals=ad_intervals,
        metadata_named=tried.get("metadata_named") or [],
        llm_voice_names={v["voice"]: v["resolved_name"] for v in llm if v.get("resolved_name")}
        or None,
        llm_voice_roles={v["voice"]: v["role"] for v in llm if v.get("role")} or None,
        cleaning=cleaning,
        recurring_text=recurring,
        diarization_provider="tailnet_dgx",
        episode_text=" ".join(x for x in (episode.get("title"), episode.get("description")) if x)
        or None,
        **_optional_inputs(roster, ep, tried, known_hosts),
    )


def _optional_inputs(
    roster: Any, ep: Dict[str, Any], tried: Dict[str, Any], known_hosts: List[str]
) -> Dict[str, Any]:
    """Inputs only some roster variants accept — passed only when the variant's signature has them,
    so an OLD variant replays exactly as before."""
    params = inspect.signature(roster.resolve_speaker_roster).parameters
    out: Dict[str, Any] = {}
    if "feed_title" in params:
        out["feed_title"] = (ep["meta"].get("feed") or {}).get("title")
    if "host_copresence" in params and hasattr(roster, "host_copresence_from_diagnostics"):
        siblings = [
            d for m, d in _SIBLINGS.get(str(ep["feed_dir"]), []) if m != ep.get("meta_path")
        ]
        out["host_copresence"] = roster.host_copresence_from_diagnostics(siblings, known_hosts)
    return out


def replay(roster: Any, ep: Dict[str, Any], ad_signatures: Any = None, trace: Any = None) -> Any:
    kw = roster_inputs(ep, roster, ad_signatures)
    dz, text = kw.pop("diarization"), kw.pop("transcript_text")
    if trace is not None and "trace" in inspect.signature(roster.resolve_speaker_roster).parameters:
        kw["trace"] = trace
    return roster.resolve_speaker_roster(dz, text, **kw)


def _role_dict(role: Any) -> Dict[str, Any]:
    return {
        "name": role.name,
        "role": role.role,
        "named": role.named,
        "source": role.source,
        "voice_type": getattr(role, "voice_type", None),
    }


def write_traces(
    episodes: Iterable[Dict[str, Any]], variant: Dict[str, Any], out: Path, signatures: Any = None
) -> Dict[str, int]:
    """Replay each episode with a decision trace (#2276); write ``{episode, roster, trace}`` JSONL.

    Each episode is ALSO replayed without the trace and the two rosters compared: the trace is a
    pure observer, so ``traced_roster_differs``, ``traced_only_error`` (an exception only the
    traced pass raised) and ``trace_degraded`` (a recorder swallowed an error) must all stay 0 —
    the phase-1 gate of #2276. ``trace_bytes_max`` bounds what the trace adds to a sidecar.
    """
    from podcast_scraper.providers.ml.diarization.naming_trace import NamingTrace

    if "trace" not in inspect.signature(variant["roster"].resolve_speaker_roster).parameters:
        # Without a trace parameter both passes run the same code: the gate would pass vacuously.
        raise SystemExit("--trace-out: this roster variant takes no `trace`; nothing to measure")
    counts: Counter = Counter()
    with out.open("w", encoding="utf-8") as fh:
        for ep in episodes:
            trace = NamingTrace()
            try:
                plain = replay(variant["roster"], ep, signatures)
            except Exception as exc:  # noqa: BLE001 — one bad episode must not stop the run
                counts["replay_error"] += 1
                counts[f"error:{type(exc).__name__}"] += 1
                continue
            try:
                traced = replay(variant["roster"], ep, signatures, trace)
            except Exception as exc:  # noqa: BLE001 — only the trace broke it
                counts["traced_only_error"] += 1
                counts[f"traced_error:{type(exc).__name__}"] += 1
                continue
            counts["episodes"] += 1
            if traced.by_voice != plain.by_voice:
                counts["traced_roster_differs"] += 1
            if trace.degraded:
                counts["trace_degraded"] += 1
            counts["trace_bytes_max"] = max(
                counts["trace_bytes_max"], len(json.dumps(trace.to_dict(), ensure_ascii=False))
            )
            rec = {
                "meta_path": ep.get("meta_path"),
                "feed": (ep["meta"].get("feed") or {}).get("title"),
                "episode": (ep["meta"].get("episode") or {}).get("title"),
                "roster": {v: _role_dict(r) for v, r in traced.by_voice.items()},
                "decision_trace": trace.to_dict(),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return dict(counts)


def _exec_module(name: str, path: Path) -> types.ModuleType:
    import importlib
    import importlib.util

    base = importlib.import_module(name) if name in sys.modules or _importable(name) else None
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    # The installed module's path, so package-relative resource lookups resolve as in production.
    mod.__file__ = getattr(base, "__file__", str(path))
    sys.modules[name] = mod  # dataclasses resolve their module during exec
    spec.loader.exec_module(mod)
    return mod


def _importable(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def load_variant(files: Dict[str, Path]) -> Dict[str, types.ModuleType]:
    """Build the modules of one variant. Dependencies first, so the roster binds to them.

    The installed modules are restored afterwards; the returned roster keeps references to the
    variant's own dependency modules, captured at its import time.
    """
    saved = {MODULES[k]: sys.modules.get(MODULES[k]) for k in MODULES}
    built: Dict[str, types.ModuleType] = {}
    try:
        for key in ("ad_signatures", "hosts", "resolution", "roster"):
            if key in files:
                built[key] = _exec_module(MODULES[key], files[key])
            elif key == "roster":
                built[key] = _exec_module(
                    MODULES[key], Path(str(sys.modules[MODULES[key]].__file__))
                )
            elif key == "ad_signatures" and _importable(MODULES[key]):
                built[key] = sys.modules.get(MODULES[key]) or __import__(
                    MODULES[key], fromlist=["*"]
                )
    finally:
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod
    # The roster remembers its variant's host-rule module, so ``--repool`` rebuilds pools with it.
    if "roster" in built:
        built["roster"]._replay_hosts = built.get("hosts") or sys.modules.get(MODULES["hosts"])
    return built


def compare(
    episodes: Iterable[Dict[str, Any]],
    old: Dict[str, types.ModuleType],
    new: Dict[str, types.ModuleType],
    *,
    signatures_old: Any = None,
    signatures_new: Any = None,
) -> Iterable[Dict[str, Any]]:
    """Yield one record per voice whose published name changed, then a summary record."""
    counts: Counter = Counter()
    for ep in episodes:
        try:
            a = replay(old["roster"], ep, signatures_old)
            b = replay(new["roster"], ep, signatures_new)
        except Exception as exc:  # noqa: BLE001 — one bad episode must not stop the replay
            counts["replay_error"] += 1
            counts[f"error:{type(exc).__name__}"] += 1
            continue
        counts["episodes"] += 1
        segs = [s for s in ep["segs"] if s.get("speaker")]
        total = sum(s["end"] - s["start"] for s in segs) or 1.0
        for v in sorted(set(a.by_voice) | set(b.by_voice)):
            x, y = a.by_voice.get(v), b.by_voice.get(v)
            xn = x.name if x and x.named else None
            yn = y.name if y and y.named else None
            xr, yr = getattr(x, "role", None), getattr(y, "role", None)
            if xn == yn:
                if not (ROLES and xn and xr != yr and {xr, yr} <= {"host", "guest"}):
                    continue
                kind = f"rerole:{xr}->{yr}"
            else:
                kind = "gained" if not xn else ("lost" if not yn else "renamed")
            counts[kind] += 1
            talk = sum(s["end"] - s["start"] for s in segs if s.get("speaker") == v)
            yield {
                "kind": kind,
                "feed": (ep["meta"].get("feed") or {}).get("title"),
                "episode": (ep["meta"].get("episode") or {}).get("title"),
                "voice": v,
                "share": round(talk / total, 3),
                "old": xn,
                "new": yn,
                "old_type": getattr(x, "voice_type", None),
                "new_type": getattr(y, "voice_type", None),
                "opening": " ".join(str(s.get("text") or "") for s in segs if s["speaker"] == v)[
                    :120
                ],
            }
    yield {"summary": dict(counts)}


def corpus_episodes(corpus: Path) -> Iterable[Dict[str, Any]]:
    from podcast_scraper.search.corpus_scope import (
        dedupe_metadata_paths_newest_run_per_episode,
        discover_metadata_files,
    )

    for meta in dedupe_metadata_paths_newest_run_per_episode(
        corpus, discover_metadata_files(corpus)
    ):
        try:
            yield load_episode(Path(meta))
        except (OSError, ValueError, KeyError):
            continue


def _variant_arg(pairs: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for pair in pairs or []:
        key, _, path = pair.partition("=")
        if key not in MODULES or not path:
            raise SystemExit(f"bad variant file {pair!r}; use one of {sorted(MODULES)}=PATH")
        out[key] = Path(path)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--old", nargs="*", default=[], metavar="MODULE=PATH")
    ap.add_argument("--new", nargs="*", default=[], metavar="MODULE=PATH")
    ap.add_argument(
        "--signatures",
        action="store_true",
        help="build cross-show ad signatures in memory from the corpus (each side with its own "
        "ad_signatures module) and pass them to classify_voices",
    )
    ap.add_argument(
        "--repool",
        action="store_true",
        help="rebuild each episode's host pool from its stored metadata with each variant's own "
        "hosts module (see repool()) instead of replaying the stored pool",
    )
    ap.add_argument(
        "--roles",
        action="store_true",
        help="also report voices whose name is unchanged but whose host/guest role flipped",
    )
    ap.add_argument(
        "--trace-out",
        type=Path,
        help="instead of comparing: replay the --new variant with a per-voice decision trace "
        "(#2276) and write {episode, roster, decision_trace} JSONL here; also checks that the "
        "traced roster equals an untraced one",
    )
    args = ap.parse_args(argv)
    global REPOOL, ROLES
    REPOOL = bool(args.repool)
    ROLES = bool(args.roles)
    index_siblings(args.corpus)
    new = load_variant(_variant_arg(args.new))
    if args.trace_out:
        sig = _signatures(new.get("ad_signatures"), args.corpus) if args.signatures else None
        summary = write_traces(corpus_episodes(args.corpus), new, args.trace_out, sig)
        print(json.dumps({"summary": summary}, ensure_ascii=False))
        failed = ("traced_roster_differs", "traced_only_error", "trace_degraded")
        return 1 if any(summary.get(k) for k in failed) else 0
    old = load_variant(_variant_arg(args.old))
    sig_old = sig_new = None
    if args.signatures:
        sig_old = _signatures(old.get("ad_signatures"), args.corpus)
        sig_new = _signatures(new.get("ad_signatures"), args.corpus)
    for rec in compare(
        corpus_episodes(args.corpus), old, new, signatures_old=sig_old, signatures_new=sig_new
    ):
        print(json.dumps(rec, ensure_ascii=False))
    return 0


def _signatures(module: Optional[types.ModuleType], corpus: Path) -> Any:
    if module is None:
        return None
    doc = module.build(module._corpus_episodes(corpus))
    return module.AdSignatures(frozenset(doc["recurring"]), frozenset(doc["ad_languages"]))


if __name__ == "__main__":
    sys.exit(main())
