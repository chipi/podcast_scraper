"""#952 — faster-whisper vs openai-whisper engine-drift eval on real podcasts.

Both engines are live on the DGX (Speaches :8000 + whisper-openai :8002), so
the prediction step is cheap. Silver reference is Deepgram nova-3 — third
independent engine; not gold-grade but sufficient for the go/no-go signal
on the 0.5pp threshold the ticket asks about.

What it does, per episode:

1. POST audio to faster-whisper Speaches at ``WHISPER_FASTER_URL``.
   Capture transcript + word-level timestamps + wall-time.
2. POST audio to openai-whisper wrapper at ``WHISPER_OPENAI_URL``.
   Capture same.
3. Call Deepgram nova-3 with ``DEEPGRAM_API_KEY``. Capture same.
4. Score:
   - WER pairwise (faster vs Deepgram, openai vs Deepgram, faster vs openai)
   - Hallucination heuristic count (segments < 2s alone; known hallucination
     phrases like "Thanks for watching", "Subscribe", "♪♪♪")
   - Timestamp delta: per-word offset distribution (median absolute diff)

Usage::

    WHISPER_FASTER_URL=http://dgx-llm-1.tail6d0ed4.ts.net:8000/v1/audio/transcriptions \\
    WHISPER_OPENAI_URL=http://dgx-llm-1.tail6d0ed4.ts.net:8002/v1/audio/transcriptions \\
    DEEPGRAM_API_KEY=... \\
    python scripts/eval/score/whisper_engine_drift_v1.py \\
        --audio tests/fixtures/audio/v1/p01_e01.mp3 ... \\
        --output data/eval/runs/whisper_engine_drift_v1/predictions

    # Score after predictions are captured:
    python scripts/eval/score/whisper_engine_drift_v1.py \\
        --score data/eval/runs/whisper_engine_drift_v1/predictions \\
        --output data/eval/runs/whisper_engine_drift_v1/scores.json

The transcribe step writes per-(engine, episode) JSON to
``<output>/<episode_id>.<engine>.json``. The score step reads those.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# WER (Levenshtein over normalized tokens). Mirrors whisper_dgx_vs_cloud_v1.py.

_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize(text: str) -> List[str]:
    return _NORMALIZE_RE.sub(" ", text.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r = _normalize(ref)
    h = _normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    nr, nh = len(r), len(h)
    dp = [[0] * (nh + 1) for _ in range(nr + 1)]
    for i in range(nr + 1):
        dp[i][0] = i
    for j in range(nh + 1):
        dp[0][j] = j
    for i in range(1, nr + 1):
        for j in range(1, nh + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[nr][nh] / nr


# ---------------------------------------------------------------------------
# Hallucination heuristic. Conservative — only flags well-known faster-whisper
# silence-hallucination phrases. False positives bias toward 0; false
# negatives bias higher. Good enough for a count-level signal.

# Phrase patterns scored as a single hit when they appear, regardless of how
# many times the phrase repeats inside the text. The original ``count()``-based
# scoring over-weighted music-intro symbols (``♪♪♪♪`` would score 4) and
# repetition loops; we only care whether the engine produced the hallucination
# *signature*, not how florid it was about it.
_HALLUCINATION_PATTERNS = (
    re.compile(r"thanks?\s+(?:you\s+)?for\s+watching", re.IGNORECASE),
    re.compile(r"subscribe\s+to\s+my\s+channel", re.IGNORECASE),
    re.compile(r"please\s+subscribe", re.IGNORECASE),
    re.compile(r"like\s+and\s+subscribe", re.IGNORECASE),
    re.compile(r"don'?t\s+forget\s+to\s+subscribe", re.IGNORECASE),
    re.compile(r"[♪♫]{2,}"),  # ``♪♪`` or more — single ♪ is a legit music note mention
    re.compile(r"\[music\]", re.IGNORECASE),
    re.compile(r"\[applause\]", re.IGNORECASE),
    re.compile(r"(\b\w+\b)(?:\s+\1){3,}", re.IGNORECASE),  # any word repeated 4+ times
)


def hallucination_count(segments: List[Dict[str, Any]], text: str) -> int:
    count = sum(1 for pat in _HALLUCINATION_PATTERNS if pat.search(text))
    # Add: very short segments at the very end (silence padding artifact)
    if segments:
        tail = segments[-1]
        dur = float(tail.get("end", 0) or 0) - float(tail.get("start", 0) or 0)
        if dur < 1.0 and len(str(tail.get("text", "")).split()) <= 4:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Backends. Each returns a dict:
#   {"text": str, "segments": [{start, end, text, words: [...]}],
#    "words": [{start, end, word}], "wall_time_sec": float}


def transcribe_faster_whisper(audio_path: Path, url: str, timeout: float = 1800) -> Dict[str, Any]:
    """POST audio to Speaches /v1/audio/transcriptions with verbose_json."""
    import requests

    t0 = time.perf_counter()
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "application/octet-stream")}
        data = {
            "model": "Systran/faster-whisper-large-v3",
            "response_format": "verbose_json",
            "language": "en",
        }
        r = requests.post(url, files=files, data=data, timeout=timeout)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    payload = r.json()
    segments = payload.get("segments") or []
    # Flatten words: Speaches returns segment.words[] when timestamp_granularities=word,
    # but verbose_json default may not include them. Try both.
    words: List[Dict[str, Any]] = []
    for seg in segments:
        for w in seg.get("words", []) or []:
            words.append({"start": w.get("start"), "end": w.get("end"), "word": w.get("word")})
    return {
        "text": str(payload.get("text") or "").strip(),
        "segments": segments,
        "words": words,
        "wall_time_sec": round(elapsed, 3),
    }


def transcribe_openai_whisper(audio_path: Path, url: str, timeout: float = 1800) -> Dict[str, Any]:
    """POST audio to the whisper-openai wrapper at :8002.

    Same multipart shape as Speaches but the wrapper response format is
    openai-whisper's native dict ({"text", "segments": [{start, end, text}]}).
    Word-level timestamps require ``word_timestamps=true`` in the wrapper if
    it supports it (this varies by wrapper revision).
    """
    import requests

    t0 = time.perf_counter()
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "application/octet-stream")}
        data = {
            "model": "large-v3",
            "response_format": "verbose_json",
            "language": "en",
        }
        r = requests.post(url, files=files, data=data, timeout=timeout)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    payload = r.json()
    segments = payload.get("segments") or []
    words: List[Dict[str, Any]] = []
    for seg in segments:
        for w in seg.get("words", []) or []:
            words.append({"start": w.get("start"), "end": w.get("end"), "word": w.get("word")})
    return {
        "text": str(payload.get("text") or "").strip(),
        "segments": segments,
        "words": words,
        "wall_time_sec": round(elapsed, 3),
    }


def transcribe_deepgram(audio_path: Path, api_key: str, timeout: float = 600) -> Dict[str, Any]:
    """Silver reference — Deepgram nova-3 via the Listen API.

    Uses the same SDK shape as ``providers/deepgram/deepgram_provider.py``
    (auto-generated SDK v6.1.x: ``client.listen.v1.media.transcribe_file``).
    """
    from deepgram import DeepgramClient

    client = DeepgramClient(api_key=api_key)
    t0 = time.perf_counter()
    with open(audio_path, "rb") as f:
        kwargs: Dict[str, Any] = {
            "request": f.read(),
            "model": "nova-3",
            "smart_format": True,
            "punctuate": True,
            "utterances": True,
            "language": "en",
        }
    response = client.listen.v1.media.transcribe_file(**kwargs)
    elapsed = time.perf_counter() - t0

    # Normalize response → plain dict (SDK object has .dict() or dict-like access)
    if hasattr(response, "dict"):
        payload = response.dict()
    elif hasattr(response, "to_dict"):
        payload = response.to_dict()
    elif isinstance(response, dict):
        payload = response
    else:
        # Some auto-gen SDKs use pydantic — try model_dump
        try:
            payload = response.model_dump()
        except Exception:
            payload = dict(response) if hasattr(response, "keys") else {"results": response}

    results = payload.get("results", {}) or {}
    channels = results.get("channels", []) or []
    if not channels:
        return {"text": "", "segments": [], "words": [], "wall_time_sec": round(elapsed, 3)}
    alt = (channels[0].get("alternatives") or [{}])[0]
    text = str(alt.get("transcript") or "").strip()
    words_raw = alt.get("words") or []
    words = [
        {"start": w.get("start"), "end": w.get("end"), "word": w.get("word")} for w in words_raw
    ]
    # Build pseudo-segments from utterances if present, else single segment.
    utterances = results.get("utterances") or []
    segments: List[Dict[str, Any]] = []
    if utterances:
        for u in utterances:
            segments.append(
                {
                    "start": u.get("start"),
                    "end": u.get("end"),
                    "text": str(u.get("transcript") or "").strip(),
                }
            )
    else:
        segments.append({"start": 0, "end": elapsed, "text": text})
    return {
        "text": text,
        "segments": segments,
        "words": words,
        "wall_time_sec": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Scoring


def _median_abs_word_offset(
    ref_words: List[Dict[str, Any]], hyp_words: List[Dict[str, Any]]
) -> float:
    """Crude alignment — match words by lowercase-stripped form in order.

    Returns the median absolute start-time offset across matched pairs, or
    ``-1.0`` if no pairs found.
    """
    if not ref_words or not hyp_words:
        return -1.0

    def norm(w: Dict[str, Any]) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(w.get("word") or "").lower())

    offsets: List[float] = []
    j = 0
    for rw in ref_words:
        rwn = norm(rw)
        if not rwn:
            continue
        # Sliding window match in hyp
        while j < len(hyp_words) and norm(hyp_words[j]) != rwn:
            j += 1
        if j < len(hyp_words):
            rs = rw.get("start")
            hs = hyp_words[j].get("start")
            if isinstance(rs, (int, float)) and isinstance(hs, (int, float)):
                offsets.append(abs(float(hs) - float(rs)))
            j += 1
    if not offsets:
        return -1.0
    offsets.sort()
    return round(offsets[len(offsets) // 2], 3)


def score_runs(predictions_dir: Path) -> Dict[str, Any]:
    """Read per-(episode, engine) JSON files and compute pairwise metrics."""
    by_episode: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for p in predictions_dir.glob("*.json"):
        # Filename: <episode_id>.<engine>.json
        parts = p.stem.rsplit(".", 1)
        if len(parts) != 2:
            continue
        ep, engine = parts
        with open(p, "r", encoding="utf-8") as f:
            by_episode.setdefault(ep, {})[engine] = json.load(f)

    per_episode: List[Dict[str, Any]] = []
    for ep, engines in sorted(by_episode.items()):
        row: Dict[str, Any] = {"episode": ep}
        for eng, data in engines.items():
            row[f"{eng}_wall_sec"] = data.get("wall_time_sec")
            row[f"{eng}_word_count"] = len(_normalize(data.get("text") or ""))
            row[f"{eng}_hallucinations"] = hallucination_count(
                data.get("segments") or [], data.get("text") or ""
            )
        # Pairwise WER (silver = deepgram)
        if "deepgram" in engines:
            ref = engines["deepgram"].get("text") or ""
            if "faster" in engines:
                row["wer_faster_vs_deepgram"] = round(
                    wer(ref, engines["faster"].get("text") or ""), 4
                )
            if "openai" in engines:
                row["wer_openai_vs_deepgram"] = round(
                    wer(ref, engines["openai"].get("text") or ""), 4
                )
        if "faster" in engines and "openai" in engines:
            row["wer_faster_vs_openai"] = round(
                wer(engines["openai"].get("text") or "", engines["faster"].get("text") or ""), 4
            )
            # Timestamp drift (faster vs openai, word-level)
            row["ts_median_abs_offset_sec"] = _median_abs_word_offset(
                engines["openai"].get("words") or [],
                engines["faster"].get("words") or [],
            )
        per_episode.append(row)

    # Aggregate
    def _mean(key: str) -> float:
        vals = [r[key] for r in per_episode if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else -1.0

    return {
        "per_episode": per_episode,
        "aggregate": {
            "mean_wer_faster_vs_deepgram": _mean("wer_faster_vs_deepgram"),
            "mean_wer_openai_vs_deepgram": _mean("wer_openai_vs_deepgram"),
            "mean_wer_faster_vs_openai": _mean("wer_faster_vs_openai"),
            "mean_hallucinations_faster": _mean("faster_hallucinations"),
            "mean_hallucinations_openai": _mean("openai_hallucinations"),
            "mean_hallucinations_deepgram": _mean("deepgram_hallucinations"),
            "mean_ts_offset_faster_vs_openai_sec": _mean("ts_median_abs_offset_sec"),
        },
        "episode_count": len(per_episode),
    }


# ---------------------------------------------------------------------------
# CLI


def _episode_id(audio_path: Path) -> str:
    """Stable episode id from filename. Strip non-id chars; truncate to 60."""
    stem = audio_path.stem
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)[:60].rstrip("_")
    return safe or "ep_unknown"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Engine-drift bench (#952): faster-whisper vs openai-whisper vs Deepgram silver"
    )
    ap.add_argument("--audio", nargs="+", help="Audio file paths to transcribe")
    ap.add_argument(
        "--engines",
        nargs="+",
        default=["faster", "openai", "deepgram"],
        choices=["faster", "openai", "deepgram"],
        help="Which engines to run (default: all 3)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/runs/whisper_engine_drift_v1/predictions"),
        help="Output directory for per-(episode, engine) JSON",
    )
    ap.add_argument(
        "--score",
        type=Path,
        help="Path to predictions dir. If passed, skip transcription and just score.",
    )
    args = ap.parse_args()

    if args.score:
        scores = score_runs(args.score)
        out_json = json.dumps(scores, indent=2)
        # In score mode ``--output`` defaults to the predictions directory
        # (shared default for transcribe mode). That's a *directory*, not a
        # file — writing scores there with ``args.output.write_text`` would
        # raise ``IsADirectoryError``. Default to <score-dir>/../scores.json
        # so the docstring example works without an explicit ``--output``.
        if args.output and args.output.is_dir():
            target = args.output.parent / "scores.json"
        elif args.output:
            target = args.output
        else:
            target = args.score.parent / "scores.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(out_json)
        print(f"wrote scores: {target}")
        print(out_json)
        return 0

    if not args.audio:
        print("--audio required for the transcribe step", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    faster_url = os.environ.get("WHISPER_FASTER_URL")
    openai_url = os.environ.get("WHISPER_OPENAI_URL")
    dg_key = os.environ.get("DEEPGRAM_API_KEY")

    if "faster" in args.engines and not faster_url:
        print("WHISPER_FASTER_URL unset", file=sys.stderr)
        return 1
    if "openai" in args.engines and not openai_url:
        print("WHISPER_OPENAI_URL unset", file=sys.stderr)
        return 1
    if "deepgram" in args.engines and not dg_key:
        print("DEEPGRAM_API_KEY unset", file=sys.stderr)
        return 1

    for audio in args.audio:
        ap_path = Path(audio).resolve()
        if not ap_path.is_file():
            print(f"missing audio: {ap_path}", file=sys.stderr)
            continue
        ep_id = _episode_id(ap_path)
        for engine in args.engines:
            out_file = args.output / f"{ep_id}.{engine}.json"
            if out_file.exists():
                print(f"skip {ep_id}/{engine} (exists)")
                continue
            print(f"running {ep_id}/{engine} ...", flush=True)
            try:
                if engine == "faster":
                    result = transcribe_faster_whisper(ap_path, faster_url or "")
                elif engine == "openai":
                    result = transcribe_openai_whisper(ap_path, openai_url or "")
                else:
                    result = transcribe_deepgram(ap_path, dg_key or "")
                result["episode_id"] = ep_id
                result["audio_path"] = str(ap_path)
                result["engine"] = engine
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                wall = result.get("wall_time_sec")
                wc = len(_normalize(result.get("text") or ""))
                print(f"  OK {ep_id}/{engine}: wall={wall}s words={wc}")
            except Exception as exc:
                print(f"  FAIL {ep_id}/{engine}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
