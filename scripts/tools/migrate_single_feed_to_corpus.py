#!/usr/bin/env python3
"""Migrate a legacy single-feed output_dir to the unified corpus layout (#644).

Legacy shape:
    <output_dir>/
    ├── run_<id1>/
    ├── run_<id2>/
    └── ...

Target shape (matches multi-feed corpus):
    <output_dir>/
    └── feeds/
        └── <feed_slug>/
            ├── run_<id1>/
            ├── run_<id2>/
            └── ...

The feed slug is deterministically derived from the RSS URL using the same
``feed_workspace_dirname`` function the multi-feed pipeline uses, so a future
multi-feed run against the same output_dir will land under the same sub-dir.

Usage:
    python scripts/tools/migrate_single_feed_to_corpus.py \\
        --output-dir /path/to/existing/single-feed/corpus \\
        --rss-url https://feeds.example.com/podcast.xml \\
        [--dry-run]

Safe to re-run: idempotent (detects already-migrated corpora and exits clean).
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from podcast_scraper.utils.filesystem import feed_workspace_dirname  # noqa: E402

logger = logging.getLogger(__name__)


_RUN_DIR_RE = re.compile(r"^run_\d{8}-\d{6}_[a-f0-9]+$")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", required=True, help="Legacy single-feed output_dir")
    parser.add_argument("--rss-url", required=True, help="RSS URL that was used for the runs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without executing them",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_dir():
        logger.error("Output dir does not exist: %s", output_dir)
        return 2

    # Idempotency check: if feeds/<slug>/ already exists AND there are no
    # legacy run_* siblings at the top, this is already migrated.
    slug = feed_workspace_dirname(args.rss_url)
    target_dir = output_dir / "feeds" / slug
    top_run_dirs = [p for p in output_dir.iterdir() if p.is_dir() and _RUN_DIR_RE.match(p.name)]
    if not top_run_dirs:
        logger.info(
            "No legacy run_* dirs at top of %s — nothing to migrate. " "(feeds/%s exists: %s)",
            output_dir,
            slug,
            target_dir.exists(),
        )
        return 0

    logger.info("Output dir:   %s", output_dir)
    logger.info("RSS URL:      %s", args.rss_url)
    logger.info("Target slug:  %s", slug)
    logger.info("Target dir:   %s", target_dir)
    logger.info("Run dirs to move: %d", len(top_run_dirs))
    for p in top_run_dirs:
        logger.info("  - %s", p.name)

    if args.dry_run:
        logger.info("--dry-run: no changes made.")
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    for p in top_run_dirs:
        dest = target_dir / p.name
        if dest.exists():
            logger.warning("Destination already exists, skipping: %s", dest)
            continue
        logger.info("Moving %s -> %s", p, dest)
        shutil.move(str(p), str(dest))

    logger.info(
        "Migration complete. Set single_feed_uses_corpus_layout: true in your "
        "profile or CLI for future runs against this output_dir."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
