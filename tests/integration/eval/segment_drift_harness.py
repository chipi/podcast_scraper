"""Fixture harness for transcript segment-time drift (#1173, AC1).

Transcribes a v3 fixture mp3 with word timestamps and measures turn-boundary drift against the
RTTM ground truth (see :mod:`podcast_scraper.evaluation.segment_time_drift`). Two uses:

* ``regenerate_cache`` — transcribe a subset once (local ``large-v3``, the prod model) and write a
  compact words-cache so the regression test needs no model. large-v3 word timestamps (~50 ms) are
  what make the strict AC2 bound genuinely hold on the clean fixtures; a small model (base.en,
  ~320 ms word granularity) inflates the measured p95 above AC2 as a model artifact. Run manually::

      python -m tests.integration.eval.segment_drift_harness --regen

* ``measure_from_cache`` — recompute drift from the cache (no whisper), used by the CI test.

The source-turn parsing reuses ``transcripts_to_mp3.parse_segments`` — the exact splitter that
generated the RTTM — so turn *i* here is turn *i* there.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from podcast_scraper.evaluation.segment_time_drift import (
    DriftResult,
    measure_boundary_drift,
    normalize_key,
    refined_and_segment_timelines,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
V3_DIR = PROJECT_ROOT / "tests" / "fixtures" / "transcripts" / "v3"
AUDIO_DIR = PROJECT_ROOT / "tests" / "fixtures" / "audio" / "v3"
CACHE_PATH = V3_DIR / "segment_drift_cache.json"

# A varied subset (podcasts p01–p09, 82 s–8 min, incl. the asr_garble long case). Kept small so
# the cache stays a light committed fixture and regeneration is minutes, not an hour.
DEFAULT_SUBSET: Tuple[str, ...] = (
    "p01_e04",
    "p01_e01",
    "p02_e01",
    "p03_e01",
    "p06_e02",
    "p07_e02",
    "p08_e01",
    "p09_e01",
)


def _load_parse_segments():
    """Import ``parse_segments`` / ``host_for_file`` from the fixture generator script."""
    script = PROJECT_ROOT / "tests" / "fixtures" / "scripts" / "transcripts_to_mp3.py"
    spec = importlib.util.spec_from_file_location("_t2mp3", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.parse_segments, mod.host_for_file


def source_words(stem: str) -> List[Tuple[str, int, bool]]:
    """``(key, turn_index, is_turn_start)`` in speaking order for a fixture's source transcript."""
    parse_segments, host_for_file = _load_parse_segments()
    raw = (V3_DIR / f"{stem}.txt").read_text(encoding="utf-8").strip()
    turns = parse_segments(raw, host_name=host_for_file(stem))
    out: List[Tuple[str, int, bool]] = []
    import re

    for turn_index, (_speaker, text) in enumerate(turns):
        first = True
        for token in re.findall(r"[0-9a-zA-Z']+", text):
            key = normalize_key(token)
            if not key:
                continue
            out.append((key, turn_index, first))
            first = False
    return out


def rttm_onsets(stem: str) -> List[float]:
    onsets: List[float] = []
    for line in (V3_DIR / f"{stem}.rttm").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0] == "SPEAKER":
            onsets.append(float(parts[3]))
    return onsets


def transcribe_segments(stem: str, model_name: str = "large-v3") -> List[Dict]:
    """Transcribe a fixture mp3 with word timestamps → ``[{start, words:[[key, ms], ...]}]``."""
    import whisper  # local import: only the regen path needs the model

    model = whisper.load_model(model_name)
    result = model.transcribe(
        str(AUDIO_DIR / f"{stem}.mp3"), word_timestamps=True, verbose=False, fp16=False
    )
    segments: List[Dict] = []
    for seg in result["segments"]:
        words = [
            [normalize_key(w.get("word", "")), round(float(w["start"]) * 1000)]
            for w in (seg.get("words") or [])
            if w.get("start") is not None and normalize_key(w.get("word", ""))
        ]
        segments.append({"start": round(float(seg["start"]) * 1000), "words": words})
    return segments


def _segments_seconds(segments: List[Dict]) -> List[Dict]:
    """Cache stores ms ints; convert back to the seconds shape the metric functions expect."""
    return [
        {
            "start": seg["start"] / 1000.0,
            "words": [[k, ms / 1000.0] for k, ms in seg["words"]],
        }
        for seg in segments
    ]


def measure_from_cache(cache: Dict[str, List[Dict]]) -> Dict[str, Tuple[DriftResult, DriftResult]]:
    """Return ``{stem: (refined_result, unrefined_result)}`` from a words-cache (no whisper)."""
    out: Dict[str, Tuple[DriftResult, DriftResult]] = {}
    for stem, segments in cache.items():
        refined, unrefined = refined_and_segment_timelines(_segments_seconds(segments))
        src = source_words(stem)
        onsets = rttm_onsets(stem)
        out[stem] = (
            measure_boundary_drift(src, refined, onsets, min_anchor=3),
            measure_boundary_drift(src, unrefined, onsets, min_anchor=3),
        )
    return out


def load_cache() -> Dict[str, List[Dict]]:
    return json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}


def regenerate_cache(stems: Tuple[str, ...] = DEFAULT_SUBSET, model_name: str = "large-v3") -> None:
    cache = {stem: transcribe_segments(stem, model_name) for stem in stems}
    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    print(f"wrote {CACHE_PATH} ({len(cache)} fixtures)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--regen", action="store_true", help="transcribe the subset and write the cache"
    )
    ap.add_argument("--model", default="large-v3")
    args = ap.parse_args()
    if args.regen:
        regenerate_cache(model_name=args.model)
        return 0
    from podcast_scraper.evaluation.segment_time_drift import pool_drift

    measured = measure_from_cache(load_cache())
    refined = pool_drift([r for r, _ in measured.values()])
    unrefined = pool_drift([u for _, u in measured.values()])
    print(f"refined   : {refined}")
    print(f"unrefined : {unrefined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
