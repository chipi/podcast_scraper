"""Operator entry point for S3.2 / D-43: withdraw a published language from discovery.

DRY RUN BY DEFAULT. Nothing is removed without `--apply`, because the thing being changed is what
listeners can find and the reversal — a reindex — costs real time on a large corpus.

    python scripts/tools/withdraw_language.py --corpus <dir> --language es
    python scripts/tools/withdraw_language.py --corpus <dir> --language es --apply

Reversal, in full:

    python -m podcast_scraper.cli index-two-tier --output-dir <corpus>

That works because withdrawal removes SEARCH INDEX rows only and the index is derived from the
corpus. See src/podcast_scraper/language_withdrawal.py for what a listener does and does not still
see afterwards — notably that direct links keep playing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from podcast_scraper.language_withdrawal import (  # noqa: E402
    apply_withdrawal,
    format_plan,
    plan_withdrawal,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path, help="Corpus root directory.")
    parser.add_argument("--language", required=True, help="Language tag to withdraw (e.g. `es`).")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove the rows. Without this the plan is printed and nothing changes.",
    )
    args = parser.parse_args(argv)

    corpus = args.corpus.expanduser()
    if not corpus.is_dir():
        print(f"no such corpus directory: {corpus}", file=sys.stderr)
        return 2

    plan = plan_withdrawal(corpus, args.language)
    print(format_plan(plan))
    if plan.error:
        return 2
    if not args.apply:
        print("\nDRY RUN — nothing was changed. Re-run with --apply to withdraw.")
        return 0
    if plan.empty:
        return 0

    result = apply_withdrawal(corpus, args.language)
    if result.error:
        print(f"\nFAILED: {result.error}", file=sys.stderr)
        return 1
    print(f"\nWithdrew {len(result.episodes)} episode(s), {result.total_rows} index row(s):")
    for tier, n in sorted(result.rows_removed.items()):
        print(f"  {tier}: {n}")
    print("\nUndo with:\n" f"  python -m podcast_scraper.cli index-two-tier --output-dir {corpus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
