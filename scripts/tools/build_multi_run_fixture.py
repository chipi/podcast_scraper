"""Build a multi-run corpus fixture for v2.6.1 hotfix tests (#822).

Produces a corpus directory matching the shape that prod actually has after
multiple pipeline runs — needed because existing fixtures
(``tests/fixtures/viewer-validation-corpus/``) model a single clean run and
miss the bugs in #818 / #820 / #821 / #823 that only surface across runs.

Schema mirrors prod metadata files (top-level keys: feed, episode, content,
processing, summary, grounded_insights, knowledge_graph). Per-run dir layout
mirrors prod (run_<timestamp>_<id>/metadata/, run.json, run_manifest.json,
metrics.json). corpus_manifest.json + corpus_run_summary.json at top level.

Deterministic: same args → byte-identical output (modulo absolute timestamps
which are generated from a fixed seed-derived base).

Usage:
    python scripts/tools/build_multi_run_fixture.py \\
        --output tests/fixtures/multi-run-corpus \\
        --feeds 3 \\
        --probe-episodes 1 \\
        --middle-episodes 5 \\
        --latest-episodes 5 \\
        --overlap 3

The "overlap" arg controls how many of the latest-run episodes share GUIDs
with middle-run episodes — that's the ``skip_existing`` scenario that makes
cumulative-unique != last-run-sum != cumulative-naive-add.

Per-feed unique episodes with defaults:
    probe + middle + (latest - overlap) = 1 + 5 + (5 - 3) = 8 unique
    Across 3 feeds → 24 unique cumulative; 15 in last-run sum; 33 metadata
    files on disk. Three different numbers exposing the three bug classes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

# Stable seed-derived base time so the fixture is reproducible.
BASE_TIME = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ts(offset_hours: int) -> dt.datetime:
    return BASE_TIME + dt.timedelta(hours=offset_hours)


def _iso(ts: dt.datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S%z").replace("+0000", "+00:00")


def _run_dir_name(ts: dt.datetime, run_short: str) -> str:
    return f"run_{ts.strftime('%Y%m%d-%H%M%S')}_{run_short}"


def _metadata_filename(idx: int, title: str, ts: dt.datetime, run_short: str) -> str:
    safe_title = title.replace(" ", "_").replace("/", "_")[:40]
    return f"{idx:04d} - {safe_title}_{ts.strftime('%Y%m%d-%H%M%S')}_{run_short}.metadata.json"


def _build_metadata(
    *,
    feed_id: str,
    feed_title: str,
    feed_url: str,
    feed_image_local: str,
    episode_guid: str,
    episode_title: str,
    episode_idx: int,
    publish_offset_days: int,
    duration_seconds: int,
    run_dir_name: str,
    processing_ts: dt.datetime,
    transcript_relpath: str,
) -> dict:
    """Build a metadata.json dict matching the prod schema closely enough."""
    return {
        "feed": {
            "title": feed_title,
            "url": feed_url,
            "feed_id": feed_id,
            "description": f"Fixture feed {feed_title} for v2.6.1 hotfix tests.",
            "language": "en",
            "authors": ["Fixture Author"],
            "image_url": f"https://example.invalid/{feed_title.replace(' ', '_')}.jpg",
            "image_local_relpath": feed_image_local,
        },
        "episode": {
            "title": episode_title,
            "description": f"Synthetic fixture episode {episode_title}.",
            "published_date": _iso(_ts(24 * 60 + publish_offset_days * 24)),
            "guid": episode_guid,
            "link": f"https://example.invalid/episode/{episode_guid}",
            "duration_seconds": duration_seconds,
            "episode_number": episode_idx,
            "image_url": None,
            "image_local_relpath": None,
        },
        "content": {
            "transcript_urls": [],
            "media_url": f"https://example.invalid/media/{episode_guid}.mp3",
            "media_id": f"sha256:{_sha256(episode_guid)}",
            "media_type": "audio/mpeg",
            "transcript_file_path": transcript_relpath,
            "transcript_source": "whisper_transcription",
            "whisper_model": None,
            "speakers": [],
        },
        "processing": {
            "processing_timestamp": _iso(processing_ts),
            "output_directory": (
                f"/app/output/feeds/{feed_id_to_dir(feed_id, feed_url)}/{run_dir_name}"
            ),
            "run_id": None,
            "config_snapshot": {
                "ml_providers": {
                    "transcription": {"provider": "openai"},
                    "speaker_detection": {"provider": "gemini"},
                    "summarization": {"provider": "gemini"},
                },
                "language": "en",
                "max_episodes": 10,
                "episode_order": "newest",
                "episode_offset": 0,
                "episode_since": None,
                "episode_until": None,
                "auto_speakers": True,
            },
            "schema_version": "1.0.0",
            "stage_timings": {
                "download_media_time": 3.5,
                "transcribe_time": 60.0,
                "extract_names_time": 0.8,
                "cleaning_time": 0.05,
                "summarize_time": 5.0,
                "total_processing_time": 70.0,
            },
        },
        "summary": {
            "generated_at": _iso(processing_ts),
            "word_count": 5000,
            "title": f"Summary: {episode_title}",
            "bullets": [
                f"Bullet 1 for {episode_title}",
                f"Bullet 2 for {episode_title}",
                f"Bullet 3 for {episode_title}",
            ],
            "key_quotes": None,
            "named_entities": None,
            "timestamps": None,
            "schema_status": "valid",
        },
        "grounded_insights": {
            "artifact_path": f"metadata/{episode_guid}.gi.json",
            "insight_count": 10,
            "generated_at": _iso(processing_ts),
            "schema_version": "2.0",
        },
        "knowledge_graph": {
            "artifact_path": f"metadata/{episode_guid}.kg.json",
            "node_count": 20,
            "edge_count": 25,
            "generated_at": _iso(processing_ts),
            "schema_version": "1.2",
        },
    }


def feed_id_to_dir(feed_id: str, feed_url: str) -> str:
    """Compute the prod-style stable feed directory name from a feed URL."""
    short = _sha256(feed_url)[:8]
    host = feed_url.split("/")[2].replace(".", "_") if "://" in feed_url else "fixture"
    return f"rss_{host}_{short}"


def _build_run_json(*, run_id: str, created_at: dt.datetime) -> dict:
    return {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "created_at": _iso(created_at),
        "index_file": "index.json",
        "run_manifest_file": "run_manifest.json",
        "manifest": {
            "run_id": run_id,
            "created_at": _iso(created_at),
            "created_by": "fixture-generator",
            "git_commit_sha": None,
            "git_branch": None,
            "git_dirty": False,
            "config_sha256": _sha256(run_id),
        },
    }


def _build_metrics_json(*, episodes_scraped: int) -> dict:
    """Per-run metrics — deliberately omits *_cost_usd to reproduce #823."""
    return {
        "schema_version": "1.0.0",
        "run_duration_seconds": float(episodes_scraped * 70),
        "episodes_scraped_total": episodes_scraped,
        "episodes_skipped_total": 0,
        "errors_total": 0,
        "transcripts_transcribed": episodes_scraped,
        "episodes_summarized": episodes_scraped,
        "metadata_files_generated": episodes_scraped,
        "gi_artifacts_generated": episodes_scraped,
        "gi_failures": 0,
        "kg_artifacts_generated": episodes_scraped,
        "kg_failures": 0,
        # NOTE: deliberately omit estimated_cost_usd / *_cost_usd fields here.
        # The cost-rollup aggregator (#823) must detect "field missing"
        # vs "field present but zero" and surface the gap rather than
        # silently rolling up as $0.00.
    }


def _build_corpus_manifest(*, feeds_summary: list[dict], total_run_count: int) -> dict:
    """Corpus-level manifest with ALL-ZERO cost rollup — reproduces #823."""
    return {
        "corpus_parent": "/app/output",
        "cost_rollup": {
            "by_stage": {
                "llm_bundled_clean_summary_cost_usd": 0.0,
                "llm_cleaning_cost_usd": 0.0,
                "llm_gi_cost_usd": 0.0,
                "llm_kg_cost_usd": 0.0,
                "llm_speaker_detection_cost_usd": 0.0,
                "llm_summarization_cost_usd": 0.0,
                "llm_transcription_cost_usd": 0.0,
            },
            "metrics_files_missing_cost_fields": 0,
            "run_count": total_run_count,
            "total_cost_usd": 0.0,
            "total_llm_cost_usd": 0.0,
            "total_transcription_cost_usd": 0.0,
        },
        "feeds": feeds_summary,
    }


def _build_run_summary(*, latest_run_per_feed: list[dict], total_run_count: int) -> dict:
    """Corpus run summary that ONLY reflects the latest run per feed.

    This is the staleness pattern that causes #820 and #821 — Dashboard
    widget + ``catalog_episode_count`` read this file and get only the
    latest-run count per feed, not the cumulative-unique total.
    """
    return {
        "corpus_parent": "/app/output",
        "batch_incidents": {
            "episode_incidents_unique": {"hard": 0, "policy": 0, "soft": 0},
            "episodes_documented_skips_unique": 0,
            "episodes_other_incidents_unique": 0,
            "feed_incidents_unique": {"hard": 0, "policy": 0, "soft": 0},
            "lines_in_window": 0,
            "log_path": "/app/output/corpus_incidents.jsonl",
            "window_end_offset_bytes": 0,
            "window_start_offset_bytes": 0,
        },
        "cost_rollup": {
            "by_stage": {
                "llm_bundled_clean_summary_cost_usd": 0.0,
                "llm_cleaning_cost_usd": 0.0,
                "llm_gi_cost_usd": 0.0,
                "llm_kg_cost_usd": 0.0,
                "llm_speaker_detection_cost_usd": 0.0,
                "llm_summarization_cost_usd": 0.0,
                "llm_transcription_cost_usd": 0.0,
            },
            "run_count": total_run_count,
            "total_cost_usd": 0.0,
        },
        "feeds": latest_run_per_feed,
    }


def build_fixture(
    output_dir: Path,
    *,
    n_feeds: int,
    probe_episodes: int,
    middle_episodes: int,
    latest_episodes: int,
    overlap: int,
) -> dict:
    """Build the fixture; return a summary dict for tests to assert against."""
    if overlap > min(middle_episodes, latest_episodes):
        raise ValueError("overlap cannot exceed min(middle_episodes, latest_episodes)")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    feeds_root = output_dir / "feeds"
    feeds_root.mkdir()

    feeds_manifest: list[dict] = []
    feeds_run_summary: list[dict] = []
    total_run_count = 0
    cumulative_unique_per_feed: list[int] = []
    last_run_episodes_per_feed: list[int] = []

    for feed_idx in range(n_feeds):
        feed_title = f"Fixture Feed {chr(ord('A') + feed_idx)}"
        feed_url = f"https://feeds.invalid/feed_{chr(ord('a') + feed_idx)}.rss"
        feed_id = f"sha256:{_sha256(feed_url)}"
        feed_dir_name = feed_id_to_dir(feed_id, feed_url)
        feed_dir = feeds_root / feed_dir_name
        feed_dir.mkdir()

        # Generate stable episode pool for this feed.
        # Probe run: ep_001 only.
        # Middle run: ep_001..ep_{middle_episodes}.
        # Latest run: ep_{middle - overlap + 1}..ep_{middle + (latest - overlap)}.
        # That gives:
        #   probe: 1 ep
        #   middle: middle_episodes eps (ep_001..ep_M)
        #   latest: latest_episodes eps starting at ep_{M - overlap + 1}
        # Cumulative unique = max(probe, middle) ep_index + (latest_episodes - overlap)
        #                   = middle_episodes + (latest_episodes - overlap)
        unique_ep_count = middle_episodes + (latest_episodes - overlap)
        cumulative_unique_per_feed.append(unique_ep_count)
        last_run_episodes_per_feed.append(latest_episodes)

        def _ep_guid(feed_idx: int, ep_num: int) -> str:
            return f"fixture-feed-{feed_idx}-ep-{ep_num:03d}"

        def _ep_title(ep_num: int) -> str:
            return f"Episode {ep_num} of {feed_title}"

        # --- Probe run (1 episode) ---
        probe_run_id = f"probe-{feed_idx}"
        probe_ts = _ts(feed_idx * 100 + 0)
        probe_dir_name = _run_dir_name(probe_ts, "probe000")
        _write_run(
            feed_dir / probe_dir_name,
            feed_idx=feed_idx,
            feed_id=feed_id,
            feed_title=feed_title,
            feed_url=feed_url,
            run_ts=probe_ts,
            run_short="probe000",
            run_id=probe_run_id,
            ep_nums=list(range(1, probe_episodes + 1)),
            ep_title_fn=_ep_title,
            ep_guid_fn=lambda ep: _ep_guid(feed_idx, ep),
        )
        total_run_count += 1

        # --- Middle run (middle_episodes episodes, ep_001..ep_M) ---
        middle_run_id = f"middle-{feed_idx}"
        middle_ts = _ts(feed_idx * 100 + 24)
        middle_dir_name = _run_dir_name(middle_ts, "middle00")
        _write_run(
            feed_dir / middle_dir_name,
            feed_idx=feed_idx,
            feed_id=feed_id,
            feed_title=feed_title,
            feed_url=feed_url,
            run_ts=middle_ts,
            run_short="middle00",
            run_id=middle_run_id,
            ep_nums=list(range(1, middle_episodes + 1)),
            ep_title_fn=_ep_title,
            ep_guid_fn=lambda ep: _ep_guid(feed_idx, ep),
        )
        total_run_count += 1

        # --- Latest run (latest_episodes episodes, with `overlap` reused) ---
        latest_run_id = f"latest-{feed_idx}"
        latest_ts = _ts(feed_idx * 100 + 48)
        latest_dir_name = _run_dir_name(latest_ts, "latest00")
        latest_start = middle_episodes - overlap + 1
        latest_eps = list(range(latest_start, latest_start + latest_episodes))
        _write_run(
            feed_dir / latest_dir_name,
            feed_idx=feed_idx,
            feed_id=feed_id,
            feed_title=feed_title,
            feed_url=feed_url,
            run_ts=latest_ts,
            run_short="latest00",
            run_id=latest_run_id,
            ep_nums=latest_eps,
            ep_title_fn=_ep_title,
            ep_guid_fn=lambda ep: _ep_guid(feed_idx, ep),
        )
        total_run_count += 1

        feeds_manifest.append(
            {
                "episodes_processed": latest_episodes,
                "error": None,
                "feed_url": feed_url,
                "last_run_finished_at": _iso(latest_ts),
                "ok": True,
                "stable_feed_dir": feed_dir_name,
            }
        )
        feeds_run_summary.append(
            {
                "episodes_processed": latest_episodes,
                "error": None,
                "feed_url": feed_url,
                "finished_at": _iso(latest_ts),
                "ok": True,
            }
        )

    # Top-level corpus_manifest.json + corpus_run_summary.json
    (output_dir / "corpus_manifest.json").write_text(
        json.dumps(
            _build_corpus_manifest(feeds_summary=feeds_manifest, total_run_count=total_run_count),
            indent=2,
        )
    )
    (output_dir / "corpus_run_summary.json").write_text(
        json.dumps(
            _build_run_summary(
                latest_run_per_feed=feeds_run_summary,
                total_run_count=total_run_count,
            ),
            indent=2,
        )
    )

    return {
        "n_feeds": n_feeds,
        "total_metadata_files": n_feeds * (probe_episodes + middle_episodes + latest_episodes),
        "cumulative_unique_per_feed": cumulative_unique_per_feed,
        "cumulative_unique_total": sum(cumulative_unique_per_feed),
        "last_run_episodes_per_feed": last_run_episodes_per_feed,
        "last_run_sum_total": sum(last_run_episodes_per_feed),
        "total_run_count": total_run_count,
    }


def _write_run(
    run_dir: Path,
    *,
    feed_idx: int,
    feed_id: str,
    feed_title: str,
    feed_url: str,
    run_ts: dt.datetime,
    run_short: str,
    run_id: str,
    ep_nums: list[int],
    ep_title_fn,
    ep_guid_fn,
) -> None:
    """Write a single run directory (metadata/, run.json, metrics.json)."""
    run_dir.mkdir()
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir()
    transcripts_dir = run_dir / "transcripts"
    transcripts_dir.mkdir()

    feed_dir_name = feed_id_to_dir(feed_id, feed_url)

    for ep_idx, ep_num in enumerate(ep_nums, start=1):
        episode_title = ep_title_fn(ep_num)
        episode_guid = ep_guid_fn(ep_num)
        transcript_relpath = (
            f"transcripts/{ep_idx:04d} - {episode_title.replace(' ', '_')[:40]}_"
            f"{run_ts.strftime('%Y%m%d-%H%M%S')}_{run_short}.txt"
        )
        metadata_name = _metadata_filename(ep_idx, episode_title, run_ts, run_short)

        meta = _build_metadata(
            feed_id=feed_id,
            feed_title=feed_title,
            feed_url=feed_url,
            feed_image_local=f".podcast_scraper/corpus-art/{feed_dir_name}.jpg",
            episode_guid=episode_guid,
            episode_title=episode_title,
            episode_idx=ep_num,
            publish_offset_days=feed_idx * 10 + ep_num,
            duration_seconds=1800 + ep_num * 60,
            run_dir_name=run_dir.name,
            processing_ts=run_ts + dt.timedelta(minutes=ep_idx * 6),
            transcript_relpath=transcript_relpath,
        )
        (metadata_dir / metadata_name).write_text(json.dumps(meta, indent=2))

        # Synthetic sibling artifacts so cataloging picks them up.
        gi_doc = {
            "schema_version": "2.0",
            "model_version": "fixture",
            "prompt_version": "fixture-v1",
            "episode_id": episode_guid,
            "nodes": [],
            "edges": [],
            "insights": [{"text": f"insight-{i}", "topic": "fixture"} for i in range(3)],
        }
        kg_doc = {
            "schema_version": "1.2",
            "model_version": "fixture",
            "prompt_version": "fixture-v1",
            "episode_id": episode_guid,
            "nodes": [{"id": f"topic:{i}", "label": f"topic {i}"} for i in range(5)],
            "edges": [],
        }
        bridge_doc = {
            "episode_id": episode_guid,
            "both": [],
            "gi_only": [],
            "kg_only": [],
            "total": 0,
        }
        gi_path = metadata_dir / metadata_name.replace(".metadata.json", ".gi.json")
        kg_path = metadata_dir / metadata_name.replace(".metadata.json", ".kg.json")
        bridge_path = metadata_dir / metadata_name.replace(".metadata.json", ".bridge.json")
        gi_path.write_text(json.dumps(gi_doc, indent=2))
        kg_path.write_text(json.dumps(kg_doc, indent=2))
        bridge_path.write_text(json.dumps(bridge_doc, indent=2))

        # Synthetic transcript so transcripts_dir is realistic.
        (transcripts_dir / Path(transcript_relpath).name).write_text(
            f"[fixture transcript for {episode_title}]\n"
        )

    (run_dir / "run.json").write_text(
        json.dumps(_build_run_json(run_id=run_id, created_at=run_ts), indent=2)
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"schema_version": "1.0.0", "run_id": run_id}, indent=2)
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(_build_metrics_json(episodes_scraped=len(ep_nums)), indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a multi-run corpus fixture for v2.6.1 hotfix tests."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/multi-run-corpus"),
        help="Output directory (will be deleted + recreated).",
    )
    parser.add_argument("--feeds", type=int, default=3)
    parser.add_argument("--probe-episodes", type=int, default=1)
    parser.add_argument("--middle-episodes", type=int, default=5)
    parser.add_argument("--latest-episodes", type=int, default=5)
    parser.add_argument(
        "--overlap",
        type=int,
        default=3,
        help="How many latest-run episodes share GUIDs with the middle run.",
    )
    args = parser.parse_args()

    summary = build_fixture(
        args.output,
        n_feeds=args.feeds,
        probe_episodes=args.probe_episodes,
        middle_episodes=args.middle_episodes,
        latest_episodes=args.latest_episodes,
        overlap=args.overlap,
    )

    print(f"Fixture written to {args.output}")
    print(f"  feeds: {summary['n_feeds']}")
    print(f"  total metadata files: {summary['total_metadata_files']}")
    print(f"  cumulative unique episodes: {summary['cumulative_unique_total']}")
    print(f"  last-run-sum count: {summary['last_run_sum_total']}")
    print(f"  total run count: {summary['total_run_count']}")
    print()
    print("Three different numbers — that's the point. Tests assert each.")


if __name__ == "__main__":
    main()
