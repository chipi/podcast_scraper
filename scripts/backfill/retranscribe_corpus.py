#!/usr/bin/env python3
"""#1173 — re-transcribe a corpus's audio, timeline-preserving, without touching the corpus.

Runs the **transcription stage only**. For each episode it preprocesses the original media with
the timeline-preserving preprocessor (silence removal OFF — see #1173), transcribes it through a
real provider, and writes the new segments to an output directory. **The source corpus is only
ever read**, never written to: re-running the expensive downstream stages (cleaning, insights, KG)
is a separate, explicit step.

Why this exists: transcripts built before #1173 were produced from silence-stripped audio, so
their timestamps run progressively early against the original media (-20 s on a 25-minute episode,
-162 s on a long one). Re-transcribing on a timeline-preserving preprocess is the ground-truth way
to repair them — and, on a self-hosted provider, it costs GPU time rather than API spend.

It doubles as a load test of a self-hosted transcription service (per-episode realtime factor and
failure counts are reported).

Usage
-----

    # See what would run (no transcription, no GPU time):
    python scripts/backfill/retranscribe_corpus.py <corpus_root> --out <dir>

    # Actually transcribe, on the DGX whisper service:
    python scripts/backfill/retranscribe_corpus.py <corpus_root> --out <dir> --apply \
        --provider dgx --host dgx-llm-1 --port 8000

    # Resume an interrupted run (episodes already written are skipped):
    ... --apply            # resume is the default; pass --force to redo them

Outputs (per episode, in ``--out``)
-----------------------------------
- ``<episode>.segments.json`` — the new segments (start/end/text), word-refined where the
  provider supports it
- ``<episode>.json`` — a record: audio duration, preprocessed duration, **timeline_error**
  (must be ~0), transcript span, realtime factor, and the *old* corpus transcript's span for
  comparison
- ``_summary.json`` — the aggregate, written at the end

Exit codes
----------
- 0 — dry-run complete, or every episode transcribed successfully.
- 1 — one or more episodes failed (see ``_summary.json``).
- 2 — script/input error (bad path, no audio found, provider unreachable).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Audio whose duration changes under preprocessing would defeat the whole point (#1173).
_MAX_TIMELINE_ERROR_SEC = 1.0


def _duration(path: Path) -> float:
    """Media duration in seconds (0.0 if ffprobe cannot read it)."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def _discover_episodes(root: Path) -> list[Path]:
    """Every unique episode's media file under a corpus root (newest run wins per episode)."""
    candidates = sorted(root.glob("feeds/*/run_*/media/*.mp3"))
    if not candidates:  # single-feed layout, or a plain directory of audio
        candidates = sorted(root.glob("**/*.mp3"))
    newest: dict[str, Path] = {}
    for media in candidates:
        newest[media.stem] = media  # sorted ascending -> the later run overwrites
    return sorted(newest.values())


def _old_transcript_span(media: Path) -> Optional[float]:
    """End of the last segment in the corpus's existing transcript for this episode, if present.

    Purely informational: it is what lets the report show the drift the re-transcription repaired.
    """
    run_dir = media.parent.parent
    for candidate in run_dir.rglob("*.segments.json"):
        if media.stem[:40] not in candidate.stem:
            continue
        try:
            raw = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        segments = raw if isinstance(raw, list) else raw.get("segments", [])
        if segments:
            try:
                return float(segments[-1]["end"])
            except (KeyError, TypeError, ValueError):
                continue
    return None


def _build_provider(args: argparse.Namespace) -> Any:
    """Build the transcription provider named by ``--provider``."""
    from podcast_scraper import config as config_module

    if args.provider == "dgx":
        from podcast_scraper.providers.tailnet_dgx.whisper_provider import (
            TailnetDgxWhisperTranscriptionProvider,
        )

        cfg = config_module.Config.model_validate(
            {
                "rss_url": "https://example.invalid/feed.xml",
                "transcription_provider": "tailnet_dgx_whisper",
                "dgx_tailnet_host": args.host,
                "dgx_whisper_port": args.port,
                "dgx_whisper_model": args.model,
                # Required by ADR-096, but never exercised: we call the DGX path directly, so a
                # DGX failure is reported as a failure rather than silently billed to the cloud.
                "transcription_fallback_provider": "openai",
            }
        )
        provider = TailnetDgxWhisperTranscriptionProvider(cfg)
        provider.initialize()
        return provider

    raise SystemExit(f"unsupported provider: {args.provider}")


def _transcribe(provider: Any, audio: Path, timeout_sec: float) -> tuple[str, list[dict], float]:
    """Transcribe via the provider's own code path (so the run exercises shipping code)."""
    return provider._transcribe_dgx(str(audio), None, timeout_sec=timeout_sec)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "corpus_root", type=Path, help="Corpus root (read-only) or a folder of audio"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Directory to write new transcripts to"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually transcribe (default: dry run)"
    )
    parser.add_argument("--provider", default="dgx", choices=["dgx"], help="Transcription backend")
    parser.add_argument("--host", default="dgx-llm-1", help="Self-hosted provider host")
    parser.add_argument("--port", type=int, default=8000, help="Self-hosted provider port")
    parser.add_argument("--model", default="Systran/faster-whisper-large-v3", help="Model id")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N episodes")
    parser.add_argument("--force", action="store_true", help="Re-do episodes already in --out")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Per-episode timeout (s)")
    args = parser.parse_args(argv)

    if not args.corpus_root.exists():
        print(f"error: no such corpus root: {args.corpus_root}", file=sys.stderr)
        return 2

    episodes = _discover_episodes(args.corpus_root)
    if args.limit:
        episodes = episodes[: args.limit]
    if not episodes:
        print(f"error: no .mp3 media found under {args.corpus_root}", file=sys.stderr)
        return 2

    total_audio = sum(_duration(ep) for ep in episodes)
    print(f"{len(episodes)} episodes, {total_audio / 3600:.1f} h of audio")

    if not args.apply:
        print("\nDRY RUN — nothing transcribed. Re-run with --apply to start.\n")
        for episode in episodes[:10]:
            print(f"  would transcribe: {episode.stem[:70]}")
        if len(episodes) > 10:
            print(f"  ... and {len(episodes) - 10} more")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "_work"
    work.mkdir(exist_ok=True)

    from podcast_scraper.preprocessing.audio.ffmpeg_processor import FFmpegAudioPreprocessor

    preprocessor = FFmpegAudioPreprocessor(
        sample_rate=16000,
        mp3_bitrate_kbps=32,
        target_loudness=-16,
        silence_removal=False,  # the #1173 invariant: duration in == duration out
    )
    provider = _build_provider(args)

    records: list[dict] = []
    failures = 0
    started = time.time()

    for index, media in enumerate(episodes, start=1):
        record_path = args.out / f"{media.stem}.json"
        if record_path.exists() and not args.force:
            print(f"[{index}/{len(episodes)}] skip (done)  {media.stem[:60]}", flush=True)
            records.append(json.loads(record_path.read_text()))
            continue

        record: dict[str, Any] = {"episode": media.stem, "media": str(media)}
        episode_started = time.time()
        preprocessed = work / f"{media.stem}.mp3"
        try:
            audio_sec = _duration(media)
            ok, _ = preprocessor.preprocess(str(media), str(preprocessed))
            if not ok:
                raise RuntimeError("preprocessing failed")
            preprocessed_sec = _duration(preprocessed)
            timeline_error = abs(audio_sec - preprocessed_sec)
            record.update(
                audio_duration=round(audio_sec, 1),
                preprocessed_duration=round(preprocessed_sec, 1),
                timeline_error=round(timeline_error, 2),
            )
            if timeline_error > _MAX_TIMELINE_ERROR_SEC:
                raise RuntimeError(
                    f"preprocessing changed the timeline by {timeline_error:.1f}s "
                    f"— transcript timestamps would not match the audio (#1173)"
                )

            text, segments, elapsed = _transcribe(provider, preprocessed, args.timeout)
            record.update(
                ok=True,
                n_segments=len(segments),
                chars=len(text),
                transcribe_seconds=round(elapsed, 1),
                realtime_factor=round(audio_sec / elapsed, 1) if elapsed else None,
                transcript_span=round(float(segments[-1]["end"]), 1) if segments else 0.0,
                old_transcript_span=_old_transcript_span(media),
            )
            (args.out / f"{media.stem}.segments.json").write_text(json.dumps(segments))
        except Exception as exc:  # noqa: BLE001 — one bad episode must not end the run
            failures += 1
            record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            preprocessed.unlink(missing_ok=True)  # keep disk bounded across a long run

        record["wall_seconds"] = round(time.time() - episode_started, 1)
        record_path.write_text(json.dumps(record, indent=2))
        records.append(record)

        if record.get("ok"):
            print(
                f"[{index}/{len(episodes)}] ok   {media.stem[:45]:45} "
                f"audio={record['audio_duration']:7.1f}s "
                f"drift_err={record['timeline_error']:4.2f}s "
                f"span={record['transcript_span']:7.1f}s "
                f"rtf={record['realtime_factor']}x",
                flush=True,
            )
        else:
            reason = str(record.get("error", ""))[:70]
            print(
                f"[{index}/{len(episodes)}] FAIL {media.stem[:45]:45} {reason}",
                flush=True,
            )

    done = [r for r in records if r.get("ok")]
    summary = {
        "episodes": len(records),
        "succeeded": len(done),
        "failed": failures,
        "wall_hours": round((time.time() - started) / 3600, 2),
        "worst_timeline_error_sec": max((r.get("timeline_error", 0.0) for r in done), default=0.0),
        "median_realtime_factor": (
            sorted(r["realtime_factor"] for r in done if r.get("realtime_factor"))[len(done) // 2]
            if done
            else None
        ),
    }
    (args.out / "_summary.json").write_text(
        json.dumps({"summary": summary, "episodes": records}, indent=2)
    )

    print(f"\n{summary['succeeded']}/{summary['episodes']} ok, {failures} failed")
    print(f"worst timeline error: {summary['worst_timeline_error_sec']}s (must be ~0)")
    print(f"median realtime factor: {summary['median_realtime_factor']}x")
    print(f"summary: {args.out / '_summary.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
