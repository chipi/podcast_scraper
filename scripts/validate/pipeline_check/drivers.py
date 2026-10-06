"""Drive the pipeline's deterministic stages over a fixture corpus, one variant at a time.

Every stage here calls the REAL pipeline function of whichever checkout is on ``sys.path`` — no
re-implementation — and returns a JSON-able record of what it decided. Two checkouts run the same
drivers on the same input, so their records are directly comparable.

ONE TOOL, ANY REF. A base ref may predate a function or a parameter (``main`` before the
multilingual work has no ``language`` parameter anywhere). :func:`call` passes only the keyword
arguments the target accepts, and a stage whose function does not exist on this ref is recorded
as ``{"absent": "<what>"}`` rather than failing — the comparator treats "absent on base, present
on candidate" as new behaviour, never as a match.

LLM-backed steps are not driven: their output is not deterministic, and what this tool checks is
what the pipeline CHOOSES (which vocabulary, which pattern, which gate), which the recorder sees
whether or not a model is called afterwards.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

#: Stages, in pipeline order. Each takes ``(episode, variant)`` and returns a JSON-able record.
STAGES = (
    "language",
    "gate",
    "hosts",
    "naming",
    "ad_regions",
    "transcript",
    "translation",
    "search_routing",
)


def _import(module: str, attr: str) -> Optional[Any]:
    try:
        return getattr(importlib.import_module(module), attr)
    except Exception:  # noqa: BLE001 - absent on this ref
        return None


def first_available(*candidates: Tuple[str, str]) -> Optional[Any]:
    """The first ``(module, attr)`` that exists on this ref — functions move between modules."""
    for module, attr in candidates:
        found = _import(module, attr)
        if found is not None:
            return found
    return None


def call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call *fn* with only the keyword arguments it accepts on this ref."""
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


def text_digest(text: str) -> Dict[str, Any]:
    return {"chars": len(text), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]}


# --- variants ------------------------------------------------------------------------------------


def parse_locale(spec: str) -> Dict[str, Optional[str]]:
    """A locale variant: ``en-US`` (the feed declares it), ``""`` (declares nothing), or
    ``override:en`` (declares nothing; an operator override says ``en``)."""
    if spec.startswith("override:"):
        return {"declared": None, "override": spec.split(":", 1)[1] or None}
    return {"declared": spec or None, "override": None}


def make_config(output_dir: str, variant: Dict[str, Any]) -> Any:
    """A deterministic run config carrying the variant, on whatever fields this ref has.

    The locale lands where the candidate reads it (``feed_declared_language`` /
    ``language_override``). A base ref without those fields gets no locale at all — which IS its
    behaviour: before the multilingual work the feed's tag was never read.
    """
    config = importlib.import_module("podcast_scraper.config")
    fields = set(getattr(config.Config, "model_fields", {}))
    kwargs: Dict[str, Any] = {
        "rss": "https://example.com/feed.xml",
        "output_dir": output_dir,
        "speaker_resolution_llm": False,
    }
    locale = variant.get("locale") or {}
    if locale.get("declared") is not None and "feed_declared_language" in fields:
        kwargs["feed_declared_language"] = locale["declared"]
    if locale.get("override") is not None and "language_override" in fields:
        kwargs["language_override"] = locale["override"]
    for key, value in (variant.get("overrides") or {}).items():
        if key in fields:
            kwargs[key] = value
    if variant.get("profile"):
        kwargs["profile"] = variant["profile"]
    return config.Config(**{k: v for k, v in kwargs.items() if k in fields or k == "profile"})


# --- the corpus ----------------------------------------------------------------------------------


def load_episodes(corpus: Path, feeds: Optional[List[str]]) -> List[Dict[str, Any]]:
    """Every served episode of *feeds* (all feeds when None), with what the drivers need."""
    out: List[Dict[str, Any]] = []
    for meta_path in sorted(corpus.glob("feeds/*/run_*/metadata/*.metadata.json")):
        feed_id = meta_path.parts[-4]
        if feeds and feed_id not in feeds:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_root = meta_path.parent.parent
        rel = str((meta.get("content") or {}).get("transcript_file_path") or "")
        seg_path = (
            run_root / (rel[: -len(".txt")] + ".segments.json") if rel.endswith(".txt") else None
        )
        out.append(
            {
                "key": meta_path.name[: -len(".metadata.json")],
                "feed_id": feed_id,
                "meta": meta,
                "run_root": run_root,
                "transcript_rel": rel,
                "segments_path": seg_path if seg_path and seg_path.is_file() else None,
            }
        )
    return out


# --- stages --------------------------------------------------------------------------------------


def stage_language(ep: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
    """What language the run resolves, and what the transcriber would be asked for."""
    resolve = _import("podcast_scraper.languages", "resolve_config_language")
    transcribe = _import("podcast_scraper.languages", "transcription_language")
    if resolve is None and transcribe is None:
        return {"absent": "language resolution", "config_language": getattr(cfg, "language", None)}
    out: Dict[str, Any] = {}
    if resolve is not None:
        raw, lang, source = resolve(cfg)
        out.update({"raw": raw, "language": lang, "source": source})
    if transcribe is not None:
        out["transcription_language"] = transcribe(cfg)
    return out


def stage_gate(ep: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
    """Would the episode be refused before download, and why."""
    gate = first_available(
        ("podcast_scraper.workflow.episode_processor", "_unsupported_language_skip_reason"),
    )
    if gate is None:
        return {"absent": "language gate", "refused": False}
    try:
        reason = call(gate, cfg)
    except TypeError:
        return {"absent": "language gate (signature)", "refused": False}
    return {"refused": bool(reason), "reason": reason}


def stage_hosts(ep: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
    """Hosts from the feed's own metadata (statement + author tags)."""
    detect = _import("podcast_scraper.speaker_detectors.hosts", "detect_hosts_from_feed")
    if detect is None:
        return {"absent": "detect_hosts_from_feed"}
    feed = ep["meta"].get("feed") or {}
    authors = feed.get("authors") if isinstance(feed.get("authors"), list) else None
    declared = getattr(cfg, "feed_declared_language", None)
    hosts = call(
        detect,
        feed.get("title"),
        feed.get("description"),
        authors,
        None,
        language=declared if declared is not None else "en",
    )
    return {"hosts": sorted(hosts)}


def _anonymize(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Any]:
    base = importlib.import_module("podcast_scraper.providers.ml.diarization.base")
    voice_of: Dict[str, str] = {}
    asr: List[Dict[str, Any]] = []
    turns = []
    for i, seg in enumerate(segments):
        label = str(seg.get("speaker_label") or "")
        voice_of.setdefault(label, f"SPEAKER_{len(voice_of):02d}")
        start, end = float(seg.get("start") or 0.0), float(seg.get("end") or 0.0)
        asr.append({"id": i, "start": start, "end": end, "text": str(seg.get("text") or "")})
        turns.append(base.DiarizationSegment(start=start, end=end, speaker=voice_of[label]))
    return asr, base.DiarizationResult(
        segments=turns, num_speakers=len(voice_of), model_name="pipeline-check-replay"
    )


def stage_naming(ep: Dict[str, Any], cfg: Any, feed_hosts: Optional[List[str]]) -> Dict[str, Any]:
    """Who each anonymous voice is: the fixture's names stripped back to SPEAKER_NN, re-resolved."""
    apply = _import(
        "podcast_scraper.providers.ml.diarization.pipeline", "apply_diarization_to_result"
    )
    if apply is None or ep["segments_path"] is None:
        return {"absent": "apply_diarization_to_result or segments"}
    raw = json.loads(ep["segments_path"].read_text(encoding="utf-8"))
    asr, diarization = _anonymize([s for s in raw if isinstance(s, dict)])
    episode = ep["meta"].get("episode") or {}
    out = call(
        apply,
        {"segments": asr, "text": " ".join(s["text"] for s in asr)},
        audio_path=f"/nonexistent/{ep['key']}.mp3",
        cfg=cfg,
        detected_speaker_names=None,
        precomputed_diarization=diarization,
        feed_hosts=feed_hosts,
        episode_title=episode.get("title") or ep["key"],
        episode_description=episode.get("description"),
        detection_ran=True,
    )
    diag = out.get("speaker_diagnostics") or {}
    voices = sorted(
        (
            {k: v.get(k) for k in ("voice", "resolved_name", "role", "named", "source", "reason")}
            for v in diag.get("voices") or []
            if isinstance(v, dict)
        ),
        key=lambda v: str(v.get("voice")),
    )
    labels = sorted(
        {
            str(s.get("speaker_label") or "")
            for s in out.get("segments") or []
            if isinstance(s, dict)
        }
    )
    return {"labels": labels, "voices": voices}


def stage_ad_regions(ep: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
    """The in-memory ad excision GI and KG fall back to."""
    excise = _import("podcast_scraper.gi.ad_regions", "excise_ad_regions")
    path = ep["run_root"] / ep["transcript_rel"] if ep["transcript_rel"] else None
    if excise is None or path is None or not path.is_file():
        return {"absent": "excise_ad_regions or transcript"}
    text = path.read_text(encoding="utf-8")
    kept, _segs, meta = excise(text)
    return {
        "kept": text_digest(kept),
        "removed_chars": getattr(meta, "chars_removed", None),
        "preroll_end": getattr(meta, "preroll_cut_end", None),
        "postroll_start": getattr(meta, "postroll_cut_start", None),
    }


def stage_transcript(ep: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
    """Which transcript body GI/KG read, and what it contains."""
    load = first_available(
        ("podcast_scraper.workflow.transcript_resolution", "load_processing_transcript"),
        ("podcast_scraper.workflow.adfree_transcript", "load_processing_transcript"),
    )
    if load is None or not ep["transcript_rel"]:
        return {"absent": "load_processing_transcript"}
    try:
        loaded = load(str(ep["run_root"]), ep["transcript_rel"])
    except Exception as exc:  # noqa: BLE001 - a refusal is a decision too
        return {"raised": type(exc).__name__}
    return {
        "ref": loaded.transcript_ref,
        "is_adfree": loaded.is_adfree,
        "text": text_digest(loaded.text or ""),
        "segments": len(loaded.segments or []),
    }


def stage_translation(ep: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
    """Would this episode be translated, and is its analysis blocked."""
    decide = _import("podcast_scraper.workflow.translation_stage", "decide_translation")
    blocked = _import("podcast_scraper.workflow.translation_stage", "analysis_blocked_reason")
    if decide is None:
        return {"absent": "translation stage", "translated": False, "analysis_blocked": None}
    outcome = call(decide, cfg, transcript_relpath=ep["transcript_rel"])
    reason = (
        call(
            blocked,
            cfg,
            transcript_relpath=ep["transcript_rel"],
            effective_output_dir=str(ep["run_root"]),
        )
        if blocked is not None
        else None
    )
    return {
        "translated": getattr(outcome, "status", None) not in (None, "skipped"),
        "status": getattr(outcome, "status", None),
        "reason": getattr(outcome, "reason", None),
        "analysis_blocked": reason,
    }


def stage_search_routing(ep: Dict[str, Any], cfg: Any, language: Optional[str]) -> Dict[str, Any]:
    """Which index tier this episode's transcript chunks are written to."""
    backend = _import("podcast_scraper.search.backends.lancedb_backend", "LanceDBBackend")
    split = getattr(backend, "_split_segments_by_language", None) if backend else None
    if split is None:
        return {"absent": "language routing", "tier": "segments"}
    doc_cls = importlib.import_module("podcast_scraper.search.backend").SegmentDocument
    doc = doc_cls(
        id=f"{ep['key']}_chunk_0",
        text="x",
        show_id=ep["feed_id"],
        episode_id=ep["key"],
        start_time=0.0,
        end_time=1.0,
        language=language,
    )
    english, non_english = split([doc])
    return {"tier": "segments" if english else "segments_nonen"}


def run_stages(
    ep: Dict[str, Any], variant: Dict[str, Any], set_stage: Callable[[str], None]
) -> Dict[str, Any]:
    """Every stage for one episode under one variant; ``set_stage`` labels the recorder."""
    record: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config(tmp, variant)
        for name in STAGES:
            set_stage(name)
            try:
                if name == "language":
                    record[name] = stage_language(ep, cfg)
                elif name == "gate":
                    record[name] = stage_gate(ep, cfg)
                elif name == "hosts":
                    record[name] = stage_hosts(ep, cfg)
                elif name == "naming":
                    record[name] = stage_naming(ep, cfg, (record.get("hosts") or {}).get("hosts"))
                elif name == "ad_regions":
                    record[name] = stage_ad_regions(ep, cfg)
                elif name == "transcript":
                    record[name] = stage_transcript(ep, cfg)
                elif name == "translation":
                    record[name] = stage_translation(ep, cfg)
                elif name == "search_routing":
                    record[name] = stage_search_routing(
                        ep, cfg, (record.get("language") or {}).get("language")
                    )
            except (
                Exception
            ) as exc:  # noqa: BLE001 - an exception is a result to compare, not a crash
                record[name] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    set_stage("")
    return record
