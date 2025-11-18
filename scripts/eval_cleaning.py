#!/usr/bin/env python3
"""Evaluate transcript cleaning quality by comparing raw vs cleaned transcripts.

This script:
1. Loads raw and cleaned transcripts from data/eval/
2. Computes removal statistics (chars, words, percentages)
3. Checks if sponsors/outros were actually removed (pattern counts before/after)
4. Flags potential issues (too much removed, cleaning ineffective)
5. Shows diff snippets of what was removed

File structure expected:
- transcript.raw.txt (raw Whisper output)
- transcript.cleaned.txt (cleaned transcript)

Usage:
    python scripts/eval_cleaning.py
    python scripts/eval_cleaning.py --output results/cleaning_eval.json
    python scripts/eval_cleaning.py --episode ep01  # Evaluate single episode
"""

import argparse
import json
import logging
import re
import sys
import time
from difflib import unified_diff
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path to import podcast_scraper modules
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Sponsor/outro patterns to check
SPONSOR_PATTERNS = [
    r"this episode is brought to you by",
    r"today['']s episode is sponsored by",
    r"our sponsor(?:s)? (?:today|this week) (?:is|are)",
    r"thanks again to our (?:friends|sponsors) at",
    r"thanks to our sponsor",
    r"thanks to (?:our )?(?:friends at|sponsors at)",
    r"brought to you by",
    r"sponsored by",
]

# Brand names commonly found in podcast ads
BRAND_NAMES = [
    "figma",
    "justworks",
    "stripe",
    "vanta",
    "miro",
    "linear",
    "notion",
    "airtable",
    "zapier",
    "hubspot",
    "squarespace",
    "wix",
    "shopify",
    "mailchimp",
    "convertkit",
]

# Outro patterns
OUTRO_PATTERNS = [
    r"subscribe (?:to|on)",
    r"rate (?:and )?review",
    r"leave a (?:rating|review)",
    r"check out (?:our )?newsletter",
    r"sign up for (?:our )?newsletter",
    r"follow (?:us|me) (?:on|at)",
    r"find (?:us|me) (?:on|at)",
    r"visit (?:our )?website",
    r"go to",
    r"visit",
]


def load_text(path: Path) -> str:
    """Load text from file.

    Args:
        path: Path to text file

    Returns:
        Text content
    """
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return ""


def count_patterns(text: str, patterns: List[str], case_sensitive: bool = False) -> Dict[str, int]:
    """Count occurrences of patterns in text.

    Args:
        text: Text to search
        patterns: List of regex patterns
        case_sensitive: Whether to match case

    Returns:
        Dictionary mapping pattern to count
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    counts = {}
    for pattern in patterns:
        matches = len(re.findall(pattern, text, flags=flags))
        counts[pattern] = matches
    return counts


def count_brand_mentions(text: str, brand_names: List[str]) -> Dict[str, int]:
    """Count mentions of brand names in text.

    Args:
        text: Text to search
        brand_names: List of brand names to search for

    Returns:
        Dictionary mapping brand name to count
    """
    text_lower = text.lower()
    counts = {}
    for brand in brand_names:
        # Simple word boundary matching
        pattern = r"\b" + re.escape(brand) + r"\b"
        matches = len(re.findall(pattern, text_lower))
        if matches > 0:
            counts[brand] = matches
    return counts


def compute_removal_stats(raw_text: str, cleaned_text: str) -> Dict[str, Any]:
    """Compute removal statistics.

    Args:
        raw_text: Raw transcript text
        cleaned_text: Cleaned transcript text

    Returns:
        Dictionary with removal statistics
    """
    raw_chars = len(raw_text)
    cleaned_chars = len(cleaned_text)
    removed_chars = raw_chars - cleaned_chars
    removal_char_pct = (removed_chars / raw_chars * 100) if raw_chars > 0 else 0.0

    raw_words = len(raw_text.split())
    cleaned_words = len(cleaned_text.split())
    removed_words = raw_words - cleaned_words
    removal_word_pct = (removed_words / raw_words * 100) if raw_words > 0 else 0.0

    return {
        "raw_chars": raw_chars,
        "cleaned_chars": cleaned_chars,
        "removed_chars": removed_chars,
        "removal_char_pct": removal_char_pct,
        "raw_words": raw_words,
        "cleaned_words": cleaned_words,
        "removed_words": removed_words,
        "removal_word_pct": removal_word_pct,
    }


def get_diff_snippet(raw_text: str, cleaned_text: str, max_lines: int = 10) -> List[str]:
    """Get a diff snippet showing what was removed.

    Args:
        raw_text: Raw transcript text
        cleaned_text: Cleaned transcript text
        max_lines: Maximum number of diff lines to return

    Returns:
        List of diff lines
    """
    raw_lines = raw_text.splitlines()
    cleaned_lines = cleaned_text.splitlines()

    diff = list(
        unified_diff(
            raw_lines,
            cleaned_lines,
            fromfile="raw",
            tofile="cleaned",
            lineterm="",
            n=3,  # Context lines
        )
    )

    # Return first max_lines of diff
    return diff[:max_lines]


def evaluate_cleaning(episode_dir: Path) -> Dict[str, Any]:
    """Evaluate cleaning for a single episode.

    Args:
        episode_dir: Directory containing transcript.raw.txt and transcript.cleaned.txt

    Returns:
        Dictionary with evaluation results
    """
    episode_id = episode_dir.name
    raw_path = episode_dir / "transcript.raw.txt"
    cleaned_path = episode_dir / "transcript.cleaned.txt"

    if not raw_path.exists():
        logger.warning(f"[{episode_id}] transcript.raw.txt not found, skipping")
        return {"episode_id": episode_id, "error": "transcript.raw.txt not found"}

    if not cleaned_path.exists():
        logger.warning(f"[{episode_id}] transcript.cleaned.txt not found, skipping")
        return {"episode_id": episode_id, "error": "transcript.cleaned.txt not found"}

    # Load transcripts
    raw_text = load_text(raw_path)
    cleaned_text = load_text(cleaned_path)

    if not raw_text:
        return {"episode_id": episode_id, "error": "empty raw transcript"}
    if not cleaned_text:
        return {"episode_id": episode_id, "error": "empty cleaned transcript"}

    # Compute removal statistics
    stats = compute_removal_stats(raw_text, cleaned_text)

    # Count sponsor patterns before and after
    sponsor_counts_raw = count_patterns(raw_text, SPONSOR_PATTERNS)
    sponsor_counts_cleaned = count_patterns(cleaned_text, SPONSOR_PATTERNS)
    sponsor_total_raw = sum(sponsor_counts_raw.values())
    sponsor_total_cleaned = sum(sponsor_counts_cleaned.values())

    # Count brand mentions before and after
    brand_counts_raw = count_brand_mentions(raw_text, BRAND_NAMES)
    brand_counts_cleaned = count_brand_mentions(cleaned_text, BRAND_NAMES)
    brand_total_raw = sum(brand_counts_raw.values())
    brand_total_cleaned = sum(brand_counts_cleaned.values())

    # Count outro patterns before and after
    outro_counts_raw = count_patterns(raw_text, OUTRO_PATTERNS)
    outro_counts_cleaned = count_patterns(cleaned_text, OUTRO_PATTERNS)
    outro_total_raw = sum(outro_counts_raw.values())
    outro_total_cleaned = sum(outro_counts_cleaned.values())

    # Flags
    removal_ratio = stats["removal_char_pct"] / 100.0
    too_much_removed = removal_ratio > 0.6
    cleaning_ineffective = (
        sponsor_total_raw > 0 and sponsor_total_cleaned >= sponsor_total_raw * 0.8
    )  # Less than 20% reduction

    # Get diff snippet
    diff_snippet = get_diff_snippet(raw_text, cleaned_text, max_lines=15)

    result: Dict[str, Any] = {
        "episode_id": episode_id,
        "removal_stats": stats,
        "sponsor_patterns": {
            "raw": {
                "total": sponsor_total_raw,
                "by_pattern": sponsor_counts_raw,
            },
            "cleaned": {
                "total": sponsor_total_cleaned,
                "by_pattern": sponsor_counts_cleaned,
            },
            "removed": sponsor_total_raw - sponsor_total_cleaned,
            "removal_pct": (
                ((sponsor_total_raw - sponsor_total_cleaned) / sponsor_total_raw * 100)
                if sponsor_total_raw > 0
                else 0.0
            ),
        },
        "brand_mentions": {
            "raw": {
                "total": brand_total_raw,
                "by_brand": brand_counts_raw,
            },
            "cleaned": {
                "total": brand_total_cleaned,
                "by_brand": brand_counts_cleaned,
            },
            "removed": brand_total_raw - brand_total_cleaned,
            "removal_pct": (
                ((brand_total_raw - brand_total_cleaned) / brand_total_raw * 100)
                if brand_total_raw > 0
                else 0.0
            ),
        },
        "outro_patterns": {
            "raw": {
                "total": outro_total_raw,
                "by_pattern": outro_counts_raw,
            },
            "cleaned": {
                "total": outro_total_cleaned,
                "by_pattern": outro_counts_cleaned,
            },
            "removed": outro_total_raw - outro_total_cleaned,
            "removal_pct": (
                ((outro_total_raw - outro_total_cleaned) / outro_total_raw * 100)
                if outro_total_raw > 0
                else 0.0
            ),
        },
        "flags": {
            "too_much_removed": too_much_removed,
            "cleaning_ineffective": cleaning_ineffective,
        },
        "diff_snippet": diff_snippet,
    }

    return result


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate transcript cleaning quality")
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="data/eval",
        help="Directory containing evaluation episodes (default: data/eval)",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Evaluate single episode only (e.g., 'ep01')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: results/cleaning_eval_<timestamp>.json)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Find evaluation episodes
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        logger.error(f"Evaluation directory not found: {eval_dir}")
        sys.exit(1)

    if args.episode:
        # Single episode mode
        episode_dirs = [eval_dir / args.episode]
        if not episode_dirs[0].exists():
            logger.error(f"Episode directory not found: {episode_dirs[0]}")
            sys.exit(1)
    else:
        # All episodes
        episode_dirs = [d for d in eval_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not episode_dirs:
        logger.error(f"No episode directories found in {eval_dir}")
        sys.exit(1)

    logger.info(f"Found {len(episode_dirs)} episode(s) to evaluate")

    # Evaluate each episode
    results = []
    for episode_dir in sorted(episode_dirs):
        result = evaluate_cleaning(episode_dir)
        results.append(result)

    # Filter successful results
    successful_results = [r for r in results if "error" not in r]
    if not successful_results:
        logger.error("No successful evaluations")
        sys.exit(1)

    # Compute aggregate statistics
    total_episodes = len(successful_results)
    avg_char_removal_pct = (
        sum(r["removal_stats"]["removal_char_pct"] for r in successful_results) / total_episodes
    )
    avg_word_removal_pct = (
        sum(r["removal_stats"]["removal_word_pct"] for r in successful_results) / total_episodes
    )

    # Sponsor removal stats
    total_sponsors_raw = sum(r["sponsor_patterns"]["raw"]["total"] for r in successful_results)
    total_sponsors_cleaned = sum(
        r["sponsor_patterns"]["cleaned"]["total"] for r in successful_results
    )
    sponsor_removal_pct = (
        ((total_sponsors_raw - total_sponsors_cleaned) / total_sponsors_raw * 100)
        if total_sponsors_raw > 0
        else 0.0
    )

    # Brand mention stats
    total_brands_raw = sum(r["brand_mentions"]["raw"]["total"] for r in successful_results)
    total_brands_cleaned = sum(r["brand_mentions"]["cleaned"]["total"] for r in successful_results)
    brand_removal_pct = (
        ((total_brands_raw - total_brands_cleaned) / total_brands_raw * 100)
        if total_brands_raw > 0
        else 0.0
    )

    # Outro pattern stats
    total_outros_raw = sum(r["outro_patterns"]["raw"]["total"] for r in successful_results)
    total_outros_cleaned = sum(r["outro_patterns"]["cleaned"]["total"] for r in successful_results)
    outro_removal_pct = (
        ((total_outros_raw - total_outros_cleaned) / total_outros_raw * 100)
        if total_outros_raw > 0
        else 0.0
    )

    # Flag counts
    too_much_removed_count = sum(1 for r in successful_results if r["flags"]["too_much_removed"])
    cleaning_ineffective_count = sum(
        1 for r in successful_results if r["flags"]["cleaning_ineffective"]
    )

    # Build output
    output_data = {
        "summary": {
            "total_episodes": total_episodes,
            "avg_char_removal_pct": avg_char_removal_pct,
            "avg_word_removal_pct": avg_word_removal_pct,
            "sponsor_removal": {
                "raw_total": total_sponsors_raw,
                "cleaned_total": total_sponsors_cleaned,
                "removed": total_sponsors_raw - total_sponsors_cleaned,
                "removal_pct": sponsor_removal_pct,
            },
            "brand_removal": {
                "raw_total": total_brands_raw,
                "cleaned_total": total_brands_cleaned,
                "removed": total_brands_raw - total_brands_cleaned,
                "removal_pct": brand_removal_pct,
            },
            "outro_removal": {
                "raw_total": total_outros_raw,
                "cleaned_total": total_outros_cleaned,
                "removed": total_outros_raw - total_outros_cleaned,
                "removal_pct": outro_removal_pct,
            },
            "flags": {
                "too_much_removed_count": too_much_removed_count,
                "cleaning_ineffective_count": cleaning_ineffective_count,
            },
        },
        "episodes": results,
    }

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"cleaning_eval_{timestamp}.json"

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("CLEANING EVALUATION SUMMARY")
    print("=" * 70)

    print("\n📊 Removal Statistics:")
    print(f"  Episodes evaluated:        {total_episodes}")
    print(f"  Avg char removal:          {avg_char_removal_pct:.1f}%")
    print(f"  Avg word removal:          {avg_word_removal_pct:.1f}%")

    print("\n🎯 Sponsor/Ad Pattern Removal:")
    print(f"  Patterns found (raw):      {total_sponsors_raw}")
    print(f"  Patterns found (cleaned):  {total_sponsors_cleaned}")
    print(f"  Removal rate:               {sponsor_removal_pct:.1f}%")

    print("\n🏷️  Brand Mention Removal:")
    print(f"  Mentions found (raw):       {total_brands_raw}")
    print(f"  Mentions found (cleaned):   {total_brands_cleaned}")
    print(f"  Removal rate:               {brand_removal_pct:.1f}%")

    print("\n👋 Outro Pattern Removal:")
    print(f"  Patterns found (raw):       {total_outros_raw}")
    print(f"  Patterns found (cleaned):   {total_outros_cleaned}")
    print(f"  Removal rate:               {outro_removal_pct:.1f}%")

    print("\n⚠️  Flags:")
    if too_much_removed_count > 0:
        print(f"  ⚠️  Too much removed (>60%): {too_much_removed_count} episode(s)")
    if cleaning_ineffective_count > 0:
        print(f"  ⚠️  Cleaning ineffective:   {cleaning_ineffective_count} episode(s)")
    if too_much_removed_count == 0 and cleaning_ineffective_count == 0:
        print("  ✅ No issues detected")

    print("\n📋 Per-Episode Details:")
    for result in successful_results:
        ep_id = result["episode_id"]
        stats = result["removal_stats"]
        flags = result["flags"]
        sponsor = result["sponsor_patterns"]
        brand = result["brand_mentions"]
        outro = result["outro_patterns"]

        flag_icons = []
        if flags["too_much_removed"]:
            flag_icons.append("⚠️")
        if flags["cleaning_ineffective"]:
            flag_icons.append("❌")
        if not flag_icons:
            flag_icons.append("✅")

        print(f"\n  {ep_id} {' '.join(flag_icons)}")
        print(
            f"    Removal: {stats['removal_char_pct']:.1f}% chars, "
            f"{stats['removal_word_pct']:.1f}% words"
        )
        sponsor_msg = (
            f"    Sponsors: {sponsor['raw']['total']} → "
            f"{sponsor['cleaned']['total']} ({sponsor['removal_pct']:.1f}% removed)"
        )
        print(sponsor_msg)
        brand_msg = (
            f"    Brands: {brand['raw']['total']} → "
            f"{brand['cleaned']['total']} ({brand['removal_pct']:.1f}% removed)"
        )
        print(brand_msg)
        outro_msg = (
            f"    Outros: {outro['raw']['total']} → "
            f"{outro['cleaned']['total']} ({outro['removal_pct']:.1f}% removed)"
        )
        print(outro_msg)

    print(f"\n💾 Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
