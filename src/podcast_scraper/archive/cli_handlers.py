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

from .offload import sweep_corpus

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
    if sub == "sweep":
        return _run_sweep(args)
    if sub != "pull":
        print(f"archive: unsupported subcommand {sub!r} (expected 'pull', 'backfill' or 'sweep')")
        return 2
    return _run_pull(args)


def _run_sweep(args: argparse.Namespace) -> int:
    """Execute ``archive sweep``: reclaim local audio already confirmed in cold.

    This ran at the START OF EVERY PIPELINE RUN until 2026-08-21, where it walked the whole
    corpus — one rclone round trip per episode — before the run had even applied its
    ``--reprocess-episode-ids`` work-list. A one-episode repair paid a 678-episode cost and sat
    silent for ~16 minutes. Reclaiming stranded audio is maintenance; it is not a precondition
    of processing an episode, so it lives here now and runs when an operator asks for it.

    Exit 0 whenever the pass completed. Keeping a file is a normal, reported outcome (not in
    cold, size mismatch, unknowable size) — the guards are what make this safe to run at all,
    so tripping one is not a failure.
    """
    backend = _backend_from_args(args)
    report = sweep_corpus(
        args.corpus,
        backend,
        dry_run=bool(args.dry_run),
        on_progress=lambda line: print(f"  {line}", flush=True),
    )
    print(("archive sweep (dry-run): " if args.dry_run else "archive sweep: ") + report.summary())
    if report.kept_not_in_cold:
        print(
            f"  {report.kept_not_in_cold} episode(s) KEPT — not confirmed in cold. Run "
            "`archive backfill` to put them there, then sweep again."
        )
    return 0


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
    # Index local originals once (run media/ + audio-cache) — this is what makes backfill
    # "move what we already have" instead of re-downloading it.
    local_lookup = bf.build_local_lookup(args.corpus)

    if args.dry_run:
        print(bf.format_dry_run(bf.plan_backfill(episodes, backend, local_lookup=local_lookup)))
        # Preview the cleanup: what local media/ the reconcile would then reclaim (already in cold).
        print("  " + sweep_corpus(args.corpus, backend, dry_run=True).summary())
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
            local_lookup=local_lookup,
            max_attempts=int(getattr(args, "max_retries", bf.DEFAULT_MAX_RETRIES)),
        )
        report.outcomes.append(outcome)
        # Stream progress: a several-hundred-episode pass must be interruptible with the
        # operator knowing exactly how far it got, not silent until the summary.
        print(f"  {outcome.outcome:<15} [{outcome.feed_title[:28]}] {outcome.title[:50]}")
    print(bf.format_result(report))
    # Cleanup sweep: reclaim local media/ now confirmed in cold — the just-harvested copies AND
    # lingering garbage from earlier runs. Corpus-wide but safe (confirmed-in-cold + size-guard).
    print("  " + sweep_corpus(args.corpus, backend, dry_run=False).summary())
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
    """Parse ``archive <subcommand>`` — ``pull`` (read), ``backfill`` + ``sweep`` (write)."""
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
            "Reconcile episode audio into the cold archive (#1631, v2 #55). Local-first: for "
            "each episode, skip if already in cold; else if a local original exists (run media/ "
            "or the audio-cache) MOVE it into cold (byte-identical, no download); else DOWNLOAD "
            "from the publisher (bounded retry + backoff). A cleanup sweep then reclaims local "
            "media/ confirmed in cold. Idempotent: safe to interrupt and re-run. Episodes aged "
            "out of a publisher's feed window are reported 'rolled_off' — a normal outcome, not "
            "a failure. Downloaded audio is re-encoded by dynamic-ad feeds and is NOT "
            "byte-identical to the transcript's audio; harvested local originals ARE. Entries "
            "are stamped accordingly. Use --dry-run to see the move-vs-download split first."
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
    back.add_argument(
        "--max-retries",
        dest="max_retries",
        type=int,
        default=None,
        help=(
            "Download attempts per episode before giving up (default 3), with exponential "
            "backoff between tries. 404/410 (rolled off) are never retried. Harvested local "
            "originals and cold hits do no network I/O and are unaffected."
        ),
    )

    sweep = sub.add_parser(
        "sweep",
        help="Reclaim local episode audio that the cold archive already holds",
        description=(
            "Walk every run dir under the corpus and delete local media/ for episodes CONFIRMED "
            "in cold (same key AND same byte size). Anything unconfirmed is kept and reported. "
            "Idempotent and interruptible.\n\n"
            "Until 2026-08-21 this ran at the start of every pipeline run, where it cost a "
            "whole-corpus pass of rclone round trips before the run applied its episode "
            "work-list — a one-episode repair waited ~16 minutes for maintenance it never asked "
            "for. Use --dry-run first: it reports exactly what would be reclaimed and deletes "
            "nothing."
        ),
    )
    sweep.add_argument("--corpus", required=True, help="Corpus root (parent of feeds/).")
    _add_archive_source_args(sweep, label="source")
    sweep.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be reclaimed and how much. Deletes nothing.",
    )

    ns = parser.parse_args(argv)
    if getattr(ns, "archive_subcommand", None) == "backfill":
        from .backfill import DEFAULT_MAX_RETRIES, DEFAULT_RATE_LIMIT_S, DEFAULT_TIMEOUT_S

        if ns.rate_limit is None:
            ns.rate_limit = DEFAULT_RATE_LIMIT_S
        if ns.timeout is None:
            ns.timeout = DEFAULT_TIMEOUT_S
        if ns.max_retries is None:
            ns.max_retries = DEFAULT_MAX_RETRIES
    ns.command = "archive"
    return ns
