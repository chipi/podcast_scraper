"""Child-process entry for scoring every episode's reel clips: ``python -m`` this.

Reads ``pickle((corpus_root, moments_config, catalog_rows))`` on stdin and writes
``pickle([(metadata_relative_path, token, clips), ...])`` on stdout. See
:func:`podcast_scraper.server.app_cache_warm._score_reel_clips_isolated` for why the scoring leaves
the api process. A plain ``python -m`` child, not ``multiprocessing``: a spawned worker
re-imports the parent's ``__main__``, which in the api is ``opentelemetry-instrument python -m
podcast_scraper.cli serve`` and from a test is pytest, so the child crashes.

Pickle is safe here: both ends are this package, and the bytes never leave the pipe between them.
"""

from __future__ import annotations

import pickle  # nosec B403 - parent and child are this package, over a private pipe
import sys
from pathlib import Path


def main() -> int:
    """Score the rows on stdin; write ``[(metadata_relative_path, token, clips)]`` on stdout."""
    from podcast_scraper.server.app_goal_reel import _episode_clips_uncached, _episode_token

    root_str, cfg, rows = pickle.loads(sys.stdin.buffer.read())  # nosec B301 - see module doc
    root = Path(root_str)
    out = []
    for row in rows:
        # The token BEFORE scoring: an episode rewritten meanwhile is then a miss later, never a
        # stale clip list filed under the new files.
        token = _episode_token(root, row)
        out.append((row.metadata_relative_path, token, _episode_clips_uncached(root, row, cfg)))
    sys.stdout.buffer.write(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
