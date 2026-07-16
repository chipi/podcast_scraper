#!/usr/bin/env python3
"""#1173 — diarize a corpus on the self-hosted pyannote service and merge speakers into transcripts.

Second stage of the corpus repair, after ``retranscribe_corpus.py``. Diarizes each episode's
original audio with **pyannote v4 (speaker-diarization-community-1) on the DGX** — no commercial
diarization API is involved, so the whole rebuild stays on our own hardware — then attaches a
speaker to every transcript segment and renders the screenplay transcript.

It composes the shipping code (``TailnetDgxDiarizationProvider`` →
``align_segments_to_speakers`` → ``format_diarized_screenplay_from_segments``) rather than
reimplementing the merge, so what runs here is what runs in the pipeline.

Timeline note (#1173): diarization runs on the **original** media, and the new transcripts were
produced from a duration-preserving preprocess, so both live on the same timeline and the
overlap-based alignment is valid. That was *not* true of the pre-#1173 corpus, whose transcripts
were on a silence-stripped (shorter) timeline while diarization was not.

Usage
-----

    # Dry run (no GPU time):
    python scripts/backfill/rediarize_corpus.py <corpus_root> --transcripts <dir> --out <dir>

    # Diarize + merge, on the DGX pyannote service:
    python scripts/backfill/rediarize_corpus.py <corpus_root> \
        --transcripts .test_outputs/retranscribe-v2 --out .test_outputs/rediarize-v2 \
        --apply --host dgx-llm-1 --port 8001

Outputs (per episode, in ``--out``)
-----------------------------------
- ``<episode>.segments.json`` — the new segments, each carrying ``speaker`` / ``speaker_label``
- ``<episode>.txt``           — the screenplay transcript ("Speaker 1: ...")
- ``<episode>.diar.json``     — the raw speaker turns (start/end/speaker) and speaker count
- ``_summary.json``           — the aggregate

Exit codes
----------
- 0 — dry-run complete, or every episode diarized successfully.
- 1 — one or more episodes failed.
- 2 — script/input error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional


def _discover(corpus_root: Path, transcripts: Path) -> list[tuple[Path, Path]]:
    """Pair each new transcript's segments file with its source media."""
    media_by_stem: dict[str, Path] = {}
    for source in sorted(corpus_root.glob("feeds/*/run_*/media/*.mp3")):
        media_by_stem[source.stem] = source

    pairs: list[tuple[Path, Path]] = []
    for segments_file in sorted(transcripts.glob("*.segments.json")):
        stem = segments_file.name[: -len(".segments.json")]
        match = media_by_stem.get(stem)
        if match is not None:
            pairs.append((match, segments_file))
    return pairs


def _build_cfg(host: str, port: int, model: str, known_hosts: Optional[list[str]] = None) -> Any:
    """The roster config for the rebuild.

    ``known_hosts`` matters more than it looks. Without it the roster has no host names at all, and
    the self-introduction it would otherwise fall back on carries the ASR's spelling — "Kevin Roos",
    "Kevin Russo", "Casey Noon" — so one host becomes three people, none of them spelled correctly.
    Supplying the configured names lets the roster snap the ASR's guess back onto the real person.
    """
    from podcast_scraper import config as config_module

    return config_module.Config.model_validate(
        {
            "rss_url": "https://example.invalid/feed.xml",
            "diarization_provider": "tailnet_dgx",
            "dgx_tailnet_host": host,
            "dgx_diarize_port": port,
            "dgx_diarize_model": model,
            "known_hosts": list(known_hosts or []),
        }
    )


def _preflight(host: str, port: int, expect_model: str) -> Optional[str]:
    """Confirm the self-hosted service is up and serving the model we expect.

    The diarization provider falls back to in-process pyannote when the DGX is unreachable, which
    would silently mix a different (locally-installed) model into the corpus. Checking once, up
    front, means the run either uses the intended self-hosted v4 model for every episode or does
    not start. Returns an error string, or None when healthy.
    """
    import httpx

    url = f"http://{host}:{port}/v1/models"
    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        served = {str(m.get("id")) for m in response.json().get("data", [])}
    except Exception as exc:  # noqa: BLE001
        return f"cannot reach the diarization service at {url}: {exc}"
    if expect_model and expect_model not in served:
        return f"{url} serves {sorted(served)}, expected {expect_model!r}"
    return None


def _detected_guests(corpus_root: Path, stem: str, guests_json: Optional[Path] = None) -> list[str]:
    """The episode's guest names, for the roster to name voices with.

    ``--guests-json`` (from ``build_clean_guests.py``) takes precedence, and on a corpus built
    before the corroboration gate it is REQUIRED: the names recorded in the metadata sidecars came
    straight from the LLM speaker detector, which returned Elon Musk (the man suing OpenAI) and Sam
    Altman as *speakers* of an episode neither appears on. Reusing them would feed the rebuilt
    roster the very names the rebuild exists to remove (#1188).

    Falls back to the corpus sidecar, which is correct for a corpus built after the gate.
    """
    if guests_json is not None:
        try:
            data = json.loads(guests_json.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        entry = data.get(stem) or {}
        return [str(n) for n in (entry.get("guests") or []) if n]

    for sidecar in corpus_root.glob(f"feeds/*/run_*/metadata/{stem}.metadata.json"):
        try:
            meta = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        names = meta.get("detected_guests") or meta.get("detected_speaker_names") or []
        return [str(n) for n in names if n]
    return []


def _diarize_episode(cfg: Any, media: Path, segments_file: Path, args: argparse.Namespace) -> dict:
    """Diarize one episode and merge resolved speakers into its transcript segments.

    Calls the pipeline's own ``apply_diarization_to_result`` rather than re-implementing the
    merge, so the voices go through the real roster resolution (host = opening voice, #1169) and
    come out with *names* where they can be resolved. Emitting raw ``SPEAKER_NN`` here would push
    unresolved placeholders into every quote's ``speaker_id`` (#1167).
    """
    from podcast_scraper.providers.ml.diarization.formatting import (
        format_diarized_screenplay_from_segments,
    )
    from podcast_scraper.providers.ml.diarization.pipeline import apply_diarization_to_result

    stem = media.stem
    segments = json.loads(segments_file.read_text())
    transcript_text = " ".join(str(s.get("text", "")) for s in segments)

    enriched = apply_diarization_to_result(
        {"segments": segments, "text": transcript_text},
        str(media),
        cfg,
        _detected_guests(args.corpus_root, stem, getattr(args, "guests_json", None)),
        cache_dir=None,
    )
    merged = enriched.get("segments") or []
    diagnostics = enriched.get("speaker_diagnostics") or {}

    labelled = sum(1 for s in merged if s.get("speaker_label"))
    if not labelled:
        raise RuntimeError("diarization produced no speaker labels (silent audio, or a no-op)")

    transcript = format_diarized_screenplay_from_segments(merged)
    (args.out / f"{stem}.segments.json").write_text(json.dumps(merged))
    (args.out / f"{stem}.txt").write_text(transcript, encoding="utf-8")
    (args.out / f"{stem}.diar.json").write_text(json.dumps(diagnostics))

    voices = {str(s.get("speaker_label")) for s in merged if s.get("speaker_label")}
    named = {v for v in voices if not v.lower().startswith("speaker_")}
    return {
        "ok": True,
        "voices": len(voices),
        "named_voices": len(named),
        "names": sorted(named),
        "segments": len(merged),
        "transcript_chars": len(transcript),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("corpus_root", type=Path, help="Corpus root (read-only) — source of audio")
    parser.add_argument(
        "--transcripts", type=Path, required=True, help="Dir of new *.segments.json to merge into"
    )
    parser.add_argument("--out", type=Path, required=True, help="Directory to write results to")
    parser.add_argument(
        "--known-hosts",
        nargs="*",
        default=[],
        help="Configured host names, so the roster can snap the ASR's spelling back onto them.",
    )
    parser.add_argument(
        "--guests-json",
        type=Path,
        default=None,
        help=(
            "Corroborated guests from build_clean_guests.py. REQUIRED for a corpus built before "
            "the corroboration gate — its stored guest names include people the episode only "
            "talks ABOUT (#1188)."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Actually diarize (default: dry run)")
    parser.add_argument("--host", default="dgx-llm-1", help="DGX host")
    parser.add_argument("--port", type=int, default=8001, help="DGX pyannote port")
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-community-1",
        help="pyannote v4 model (self-hosted; no commercial API)",
    )
    parser.add_argument(
        "--expect-model",
        default="pyannote/speaker-diarization-community-1",
        help=(
            "Fail an episode whose result came from a different model than this — guards against "
            "the provider silently falling back to in-process pyannote. Pass '' to allow it."
        ),
    )
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Re-do episodes already in --out")
    args = parser.parse_args(argv)

    if not args.corpus_root.exists():
        print(f"error: no such corpus root: {args.corpus_root}", file=sys.stderr)
        return 2
    if not args.transcripts.exists():
        print(f"error: no such transcripts dir: {args.transcripts}", file=sys.stderr)
        return 2

    pairs = _discover(args.corpus_root, args.transcripts)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print("error: no (media, segments) pairs found — run retranscribe first", file=sys.stderr)
        return 2

    print(f"{len(pairs)} episodes to diarize on {args.model}")
    if not args.apply:
        print("\nDRY RUN — nothing diarized. Re-run with --apply.\n")
        for media, _ in pairs[:10]:
            print(f"  would diarize: {media.stem[:70]}")
        if len(pairs) > 10:
            print(f"  ... and {len(pairs) - 10} more")
        return 0

    problem = _preflight(args.host, args.port, args.expect_model)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = _build_cfg(args.host, args.port, args.model, args.known_hosts)

    records: list[dict] = []
    failures = 0
    started = time.time()

    for index, (media, segments_file) in enumerate(pairs, start=1):
        stem = media.stem
        out_segments = args.out / f"{stem}.segments.json"
        if out_segments.exists() and not args.force:
            print(f"[{index}/{len(pairs)}] skip (done)  {stem[:55]}", flush=True)
            continue

        record: dict[str, Any] = {"episode": stem}
        episode_started = time.time()
        try:
            record.update(_diarize_episode(cfg, media, segments_file, args))
        except Exception as exc:  # noqa: BLE001 — one bad episode must not end the run
            failures += 1
            record.update(ok=False, error=f"{type(exc).__name__}: {exc}")

        record["wall_seconds"] = round(time.time() - episode_started, 1)
        records.append(record)

        if record.get("ok"):
            names = ", ".join(record["names"][:3]) or "-"
            print(
                f"[{index}/{len(pairs)}] ok   {stem[:42]:42} "
                f"voices={record['voices']:2} named={record['named_voices']:2} "
                f"[{names[:34]:34}] ({record['wall_seconds']:.0f}s)",
                flush=True,
            )
        else:
            print(
                f"[{index}/{len(pairs)}] FAIL {stem[:45]:45} {str(record.get('error'))[:60]}",
                flush=True,
            )

    done = [r for r in records if r.get("ok")]
    summary = {
        "episodes": len(records),
        "succeeded": len(done),
        "failed": failures,
        "wall_hours": round((time.time() - started) / 3600, 2),
        "model": args.model,
    }
    (args.out / "_summary.json").write_text(
        json.dumps({"summary": summary, "episodes": records}, indent=2)
    )
    print(f"\n{summary['succeeded']}/{summary['episodes']} ok, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
