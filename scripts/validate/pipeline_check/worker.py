"""One checkout, one corpus, many variants -> one JSON file. Run as a subprocess by :mod:`.cli`.

``PYTHONPATH`` points at the checkout under test (its ``src``) and at this tool, so the pipeline
code is the ref's own and the tool is the same on both sides.

    python -m pipeline_check.worker --corpus DIR --feeds p01 --locales '["en", "en-US"]' --out F
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from . import drivers, recorder

#: Packages whose decision points are traced.
TRACED_PACKAGES = ("podcast_scraper",)

#: Modules the recorder imports to discover decision points. Importing the whole package would
#: pull in optional ML stacks; these are the ones that hold language-keyed choices, and a module
#: that does not exist on a ref is simply skipped (and named in the output).
DISCOVERY_MODULES = (
    "podcast_scraper.languages",
    "podcast_scraper.languages_guard",
    "podcast_scraper.speaker_detectors",
    "podcast_scraper.providers.ml.diarization.pipeline",
    "podcast_scraper.providers.ml.diarization.roster",
    "podcast_scraper.providers.ml.diarization.turns",
    "podcast_scraper.gi.filters",
    "podcast_scraper.gi.ad_regions",
    "podcast_scraper.gi.speakers",
    "podcast_scraper.workflow.transcript_resolution",
    "podcast_scraper.workflow.adfree_transcript",
    "podcast_scraper.workflow.translation_stage",
    "podcast_scraper.workflow.episode_processor",
    "podcast_scraper.translation.artifacts",
    "podcast_scraper.transcription.punctuation",
    "podcast_scraper.search.indexer",
    "podcast_scraper.search.segments",
    "podcast_scraper.search.backends.lancedb_backend",
)


def _normalise_seen(events: List[recorder.Event]) -> Dict[str, Any]:
    """Every string value a decision received, normalised by THIS ref's own normaliser.

    The comparator runs outside the pipeline, so it is handed the answers rather than a copy of
    the rules: what "en_GB" means is whatever the code under test says it means. A ref with no
    normaliser gets lower-case + primary subtag.
    """
    try:
        from podcast_scraper.languages import normalize_language_tag as norm
    except Exception:  # noqa: BLE001 - absent on a base ref

        def norm(tag: Any) -> Any:
            text = str(tag or "").strip().lower().replace("_", "-")
            return text.split("-")[0] or None

    seen: Set[str] = set()
    for ev in events:
        values = ev.key.values() if isinstance(ev.key, dict) else [ev.key]
        seen.update(v for v in values if isinstance(v, str))
    return {v: norm(v) for v in sorted(seen)}


def run_real(args: argparse.Namespace, rec: recorder.Recorder) -> Dict[str, Any]:
    """One real run of the pipeline's own CLI, in this process, with the recorder installed.

    The transcript cache is shared between every run of a comparison, so ASR runs once per
    episode and every side reads the same transcript: the Whisper side of non-determinism is
    removed rather than measured. Embeddings are off (``vector_search: false``) because the
    local ML stack is not reliable on every machine; search is out of this mode's scope.
    """
    import yaml

    from podcast_scraper import cli as pipeline_cli

    from . import artifacts

    run_dir: Path = args.run_dir
    out = run_dir / "out"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "profile": args.profile,
        "max_episodes": args.max_episodes,
        "transcript_cache_enabled": True,
        "transcript_cache_dir": str(args.transcript_cache),
        "vector_search": False,
        # Never write to a real audio archive: the profile under test may point at prod's rclone
        # remote, and a check must not upload anything anywhere.
        "audio_storage_backend": "local",
        **json.loads(args.overrides),
    }
    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    rec.variant = "real"
    rec.stage = "pipeline"
    code = pipeline_cli.main([args.rss, "--config", str(cfg_path), "--output-dir", str(out)])
    return {
        "exit_code": code,
        "artifacts": artifacts.collect(out) if out.exists() else {"files": [], "episodes": {}},
    }


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--real",
        action="store_true",
        help="run the real pipeline on --rss instead of driving fixtures",
    )
    ap.add_argument("--rss", default="")
    ap.add_argument("--max-episodes", type=int, default=1)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--transcript-cache", type=Path, default=None)
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--feeds", default="", help="comma-separated feed ids (default: all)")
    ap.add_argument(
        "--locales",
        default='["en"]',
        help='JSON list of locale variants; "" = the feed declares none',
    )
    ap.add_argument("--profile", default="")
    ap.add_argument("--overrides", default="{}", help="JSON object of config overrides")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rec = recorder.install(DISCOVERY_MODULES)
    import podcast_scraper

    if args.real:
        real = run_real(args, rec)
        payload_real = {
            "normalised": _normalise_seen(rec.events),
            "code": str(Path(podcast_scraper.__file__).parents[1]),
            "real": real,
            "events": [e.as_dict() for e in rec.events],
            "decision_points": {"maps": sorted(rec.maps), "functions": sorted(rec.functions)},
            "import_time_copies": recorder.import_time_copies(TRACED_PACKAGES),
            "skipped_modules": rec.skipped_modules,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload_real, indent=1, sort_keys=True, default=str), encoding="utf-8"
        )
        print(
            f"pipeline-check worker (real): exit {real['exit_code']}, "
            f"{len(real['artifacts']['episodes'])} episode(s), {len(rec.events)} decisions "
            f"-> {args.out}"
        )
        return 0

    if args.corpus is None:
        ap.error("--corpus is required without --real")
    feeds = [f for f in args.feeds.split(",") if f] or None
    # A JSON list, not a comma string: the empty locale ("declares nothing") must survive.
    locales = json.loads(args.locales)
    episodes = drivers.load_episodes(args.corpus, feeds)
    overrides = json.loads(args.overrides)

    results: Dict[str, Dict[str, Any]] = {}
    for spec in locales:
        variant = {
            "locale": drivers.parse_locale(spec),
            "profile": args.profile or None,
            "overrides": overrides,
        }
        rec.variant = spec
        per_episode: Dict[str, Any] = {}
        for ep in episodes:
            per_episode[ep["key"]] = drivers.run_stages(
                ep, variant, lambda s: setattr(rec, "stage", s)
            )
        results[spec] = per_episode

    payload = {
        "normalised": _normalise_seen(rec.events),
        "code": str(Path(podcast_scraper.__file__).parents[1]),
        "episodes": [ep["key"] for ep in episodes],
        "variants": locales,
        "results": results,
        "events": [e.as_dict() for e in rec.events],
        "decision_points": {"maps": sorted(rec.maps), "functions": sorted(rec.functions)},
        "import_time_copies": recorder.import_time_copies(TRACED_PACKAGES),
        "skipped_modules": rec.skipped_modules,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(
        f"pipeline-check worker: {len(episodes)} episode(s) x {len(locales)} variant(s), "
        f"{len(rec.events)} decisions -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
