"""``archive pull`` — fetch archived episode audio from the storage backend (#1199).

The #947 raw-audio archive is keyed by ``sha256(guid)``; the corpus metadata
carries ``episode.guid``, so this resolves episode -> key from the corpus and
downloads with human-meaningful names (``<feed>/<NNNN> - <title>.<ext>``) via the
same backend the pipeline writes with — local filesystem or an rclone remote
(Hetzner Storage Box / S3). Selectors scope the pull; ``--dry-run`` previews.

Standalone by design: the backend is built from explicit args (``--rclone-remote``
or ``--local-root``), not the full pipeline config, so it runs the same on a
laptop or the prod box.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List

_SAFE = re.compile(r"[^A-Za-z0-9._ -]+")
_SIZE_RE = re.compile(r"[?&]size=(\d+)")


def _safe(name: str, fallback: str = "untitled") -> str:
    cleaned = _SAFE.sub("", str(name or "")).strip().strip(".") or fallback
    return cleaned[:120]


def _ext_from(media_url: str, media_type: str) -> str:
    for cand in (os.path.splitext((media_url or "").split("?")[0])[1], ""):
        if (
            cand
            and 1 < len(cand) <= 5
            and cand.lower()
            in (
                ".mp3",
                ".m4a",
                ".aac",
                ".ogg",
                ".opus",
                ".wav",
                ".flac",
                ".mp4",
            )
        ):
            return cand.lower()
    if "mp4" in (media_type or "") or "m4a" in (media_type or ""):
        return ".m4a"
    return ".mp3"


def _iter_corpus_episodes(corpus_dir: str) -> List[Dict[str, Any]]:
    """Unique episodes under ``corpus_dir`` (dedup by guid, keep the latest run)."""
    pattern = os.path.join(corpus_dir, "feeds", "*", "run_*", "metadata", "*.metadata.json")
    by_guid: Dict[str, Dict[str, Any]] = {}
    # Sort so a later run (lexically greater ``run_<ts>``) wins the dedup.
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        ep = doc.get("episode", {}) or {}
        guid = ep.get("guid")
        if not guid:
            continue
        feed = doc.get("feed", {}) or {}
        content = doc.get("content", {}) or {}
        by_guid[guid] = {
            "guid": guid,
            "episode_id": ep.get("episode_id") or guid,
            "title": ep.get("title") or "",
            "episode_number": ep.get("episode_number"),
            "published_date": ep.get("published_date") or "",
            "feed_title": feed.get("title") or feed.get("feed_id") or "feed",
            "feed_id": feed.get("feed_id") or "",
            "media_url": content.get("media_url") or "",
            "media_type": content.get("media_type") or "",
        }
    return list(by_guid.values())


def _select(episodes: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    out = episodes
    if getattr(args, "feed", None):
        needle = args.feed.lower()
        out = [e for e in out if needle in (e["feed_title"] + " " + e["feed_id"]).lower()]
    if getattr(args, "episode", None):
        out = [e for e in out if args.episode in (e["episode_id"], e["guid"])]
    if getattr(args, "since", None):
        out = [e for e in out if (e["published_date"] or "")[:10] >= args.since]
    return out


def _dest_path(dest_dir: str, ep: Dict[str, Any], idx: int) -> str:
    num = ep.get("episode_number")
    prefix = f"{int(num):04d}" if isinstance(num, int) else f"{idx:04d}"
    fname = f"{prefix} - {_safe(ep['title'])}{_ext_from(ep['media_url'], ep['media_type'])}"
    return os.path.join(dest_dir, _safe(ep["feed_title"], "feed"), fname)


def _backend_from_args(args: argparse.Namespace):
    from ..utils.storage_backend import LocalStorageBackend, RcloneStorageBackend

    if getattr(args, "rclone_remote", None):
        return RcloneStorageBackend(
            remote=args.rclone_remote,
            base_path=getattr(args, "base_path", "") or "",
            rclone_bin=getattr(args, "rclone_bin", "rclone") or "rclone",
        )
    if getattr(args, "local_root", None):
        from pathlib import Path

        return LocalStorageBackend(Path(args.local_root))
    raise SystemExit("archive: provide a backend — --rclone-remote NAME or --local-root PATH")


def run_archive(args: argparse.Namespace) -> int:
    """Dispatch ``archive <subcommand>``."""
    sub = getattr(args, "archive_subcommand", None)
    if sub == "backfill":
        return _run_backfill(args)
    if sub != "pull":
        print(f"archive: unsupported subcommand {sub!r} (expected 'pull' or 'backfill')")
        return 2
    return _run_pull(args)


def _run_backfill(args: argparse.Namespace) -> int:
    """Execute ``archive backfill``: store audio the archive is missing (#1631).

    Exit code is 0 whenever the pass completed, including when episodes rolled off — those are
    a normal, reported outcome, not a failure. A non-zero code is reserved for episodes that
    failed for a retryable reason, so a scheduled re-run has something honest to key on.
    """
    from . import backfill as bf

    episodes = _select(_iter_corpus_episodes(args.corpus), args)
    if not episodes:
        print("archive backfill: no matching episodes in corpus")
        return 0

    backend = _backend_from_args(args)

    if args.dry_run:
        print(bf.format_dry_run(bf.plan_backfill(episodes, backend)))
        return 0

    limiter = bf.HostRateLimiter(getattr(args, "rate_limit", bf.DEFAULT_RATE_LIMIT_S))
    report = bf.BackfillReport()
    for ep in episodes:
        outcome = bf.backfill_episode(
            ep,
            backend,
            corpus_dir=args.corpus,
            force=bool(getattr(args, "force", False)),
            timeout_s=int(getattr(args, "timeout", bf.DEFAULT_TIMEOUT_S)),
            limiter=limiter,
        )
        report.outcomes.append(outcome)
        # Stream progress: a several-hundred-episode pass must be interruptible with the
        # operator knowing exactly how far it got, not silent until the summary.
        print(f"  {outcome.outcome:<15} [{outcome.feed_title[:28]}] {outcome.title[:50]}")
    print(bf.format_result(report))
    return 1 if report.counts().get(bf.FETCH_FAILED, 0) else 0


def _run_pull(args: argparse.Namespace) -> int:
    """Execute ``archive pull``: select episodes, then fetch each from the backend."""
    from ..utils import audio_cache

    episodes = _select(_iter_corpus_episodes(args.corpus), args)
    if not episodes:
        print("archive pull: no matching episodes in corpus")
        return 0

    if args.dry_run:
        total = sum(int(m.group(1)) for e in episodes if (m := _SIZE_RE.search(e["media_url"])))
        print(f"archive pull (dry-run): {len(episodes)} episode(s)")
        if total:
            print(f"  estimated size (from feed enclosure hints): {total / 1e9:.2f} GB")
        for i, e in enumerate(episodes):
            print(f"  - {_dest_path(args.dest, e, i)}  (guid={e['guid']})")
        return 0

    backend = _backend_from_args(args)
    ok = miss = skipped = 0
    for i, e in enumerate(episodes):
        dest = _dest_path(args.dest, e, i)
        if not args.force and os.path.isfile(dest) and os.path.getsize(dest) > 0:
            skipped += 1
            continue
        if audio_cache.fetch_into(backend, e["guid"], dest):
            ok += 1
            print(f"  pulled {dest}")
        else:
            miss += 1
            print(f"  MISS   {e['title'][:60]} (guid={e['guid']}) — not in archive")
    print(f"archive pull: {ok} pulled, {skipped} skipped (exists), {miss} not-in-archive")
    return 0 if miss == 0 else 1


def _add_archive_source_args(p: argparse.ArgumentParser, *, label: str) -> None:
    """The backend selector, shared so ``pull`` and ``backfill`` name it identically."""
    src = p.add_argument_group(f"archive {label} (one required)")
    src.add_argument(
        "--rclone-remote", dest="rclone_remote", help="rclone remote name (remote archive)."
    )
    src.add_argument(
        "--base-path",
        dest="base_path",
        default="podcast-audio-archive",
        help="Base path under the rclone remote.",
    )
    src.add_argument("--rclone-bin", dest="rclone_bin", default="rclone", help="rclone binary.")
    src.add_argument("--local-root", dest="local_root", help="Local archive root (local backend).")


def _add_archive_selector_args(p: argparse.ArgumentParser) -> None:
    """Episode selectors, identical across subcommands so muscle memory transfers."""
    sel = p.add_argument_group("selectors (default: all)")
    sel.add_argument("--all", action="store_true", help="Every episode (default).")
    sel.add_argument("--feed", help="Only episodes whose feed title/id contains this.")
    sel.add_argument("--episode", help="Only this episode_id or guid.")
    sel.add_argument("--since", help="Only episodes published on/after YYYY-MM-DD.")


def parse_archive_argv(argv: List[str]) -> argparse.Namespace:
    """Parse ``archive <subcommand> ...`` — ``pull`` (read) and ``backfill`` (write)."""
    parser = argparse.ArgumentParser(prog="podcast_scraper archive")
    sub = parser.add_subparsers(dest="archive_subcommand", required=True)

    pull = sub.add_parser("pull", help="Download archived episode audio to a local directory")
    pull.add_argument("--corpus", required=True, help="Corpus root (parent of feeds/).")
    pull.add_argument("--dest", required=True, help="Local output directory for pulled audio.")
    _add_archive_source_args(pull, label="source")
    _add_archive_selector_args(pull)
    pull.add_argument(
        "--dry-run", action="store_true", help="List what would be pulled + est. size."
    )
    pull.add_argument(
        "--force", action="store_true", help="Re-download even if the dest file exists."
    )

    back = sub.add_parser(
        "backfill",
        help="Fetch missing episode audio from publishers and store it in the archive",
        description=(
            "Recover audio for episodes ingested before the cache persisted (#1631). Reads the "
            "enclosure URL from corpus metadata, skips anything already archived, and stores "
            "the rest under the same sha256(guid) key the pipeline writes. Idempotent: safe to "
            "interrupt and re-run. Episodes that have aged out of a publisher's feed window are "
            "reported as 'rolled_off' — a normal outcome, not a failure. Recovered audio is "
            "re-encoded by dynamic-ad feeds and is NOT byte-identical to the audio that "
            "produced the existing transcripts; recovered entries are stamped accordingly."
        ),
    )
    back.add_argument("--corpus", required=True, help="Corpus root (parent of feeds/).")
    _add_archive_source_args(back, label="destination")
    _add_archive_selector_args(back)
    back.add_argument(
        "--dry-run",
        action="store_true",
        help="Report per feed what would be fetched, with an estimated size. Fetches nothing.",
    )
    back.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch and overwrite even when the archive already holds the episode.",
    )
    back.add_argument(
        "--rate-limit",
        dest="rate_limit",
        type=float,
        default=None,
        help=(
            "Minimum seconds between requests to the SAME host (default 1.0). Backfill walks "
            "hundreds of episodes concentrated on a few CDNs; do not set this to 0 against a "
            "live publisher."
        ),
    )
    back.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-episode download timeout in seconds (default 300).",
    )

    ns = parser.parse_args(argv)
    if getattr(ns, "archive_subcommand", None) == "backfill":
        from .backfill import DEFAULT_RATE_LIMIT_S, DEFAULT_TIMEOUT_S

        if ns.rate_limit is None:
            ns.rate_limit = DEFAULT_RATE_LIMIT_S
        if ns.timeout is None:
            ns.timeout = DEFAULT_TIMEOUT_S
    ns.command = "archive"
    return ns
