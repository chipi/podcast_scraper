#!/usr/bin/env python3
"""Evaluate summarization quality using ROUGE metrics and reference-free checks.

This script:
1. Loads cleaned transcripts (transcript.cleaned.txt) and reference summaries
   (summary.gold.*.txt) from data/eval/
2. Generates summaries using the configured model
3. Computes ROUGE scores (if reference summaries exist)
4. Performs reference-free checks (compression ratio, repetition, keyword coverage)
5. Outputs results in JSON format for regression testing

File structure expected:
- transcript.cleaned.txt (input for summarization)
- summary.gold.long.txt (detailed reference, default)
- summary.gold.short.txt (optional concise reference)

Usage:
    # Use defaults (BART-large for MAP, LED/long-fast for REDUCE) with auto-generated filename
    python scripts/eval_summaries.py
    # Outputs to: results/eval_YYYYMMDD_HHMMSS.json

    # Specify custom output file
    python scripts/eval_summaries.py --output results/my_evaluation.json

    # Specify MAP model only (REDUCE defaults to LED)
    python scripts/eval_summaries.py --map-model bart-large

    # Specify both MAP and REDUCE models
    python scripts/eval_summaries.py --map-model bart-large --reduce-model long-fast

    # Use config file (overrides CLI arguments)
    python scripts/eval_summaries.py --config config.yaml

    # Use short reference summaries
    python scripts/eval_summaries.py --use-short-reference
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("ERROR: rouge-score library not installed. Install with: pip install rouge-score")
    sys.exit(1)

# Add parent directory to path to import podcast_scraper modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from podcast_scraper import config, summarizer

logger = logging.getLogger(__name__)

# Reference-free check thresholds
MIN_COMPRESSION_RATIO = 2.0  # Summary should be at least 2× shorter
MAX_COMPRESSION_RATIO = 50.0  # Summary shouldn't be more than 50× shorter
REPETITION_NGRAM_SIZE = 3  # Check for repeated 3-grams
REPETITION_THRESHOLD = 5  # Flag if any 3-gram appears more than 5 times
KEYWORD_COVERAGE_TOP_N = 20  # Extract top 20 keywords from transcript


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


def compute_rouge_scores(prediction: str, reference: str) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        prediction: Generated summary
        reference: Reference summary

    Returns:
        Dictionary with ROUGE scores (precision, recall, fmeasure for each metric)
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    scores = scorer.score(reference, prediction)

    # Convert to serializable format
    result = {}
    for metric_name, score_obj in scores.items():
        result[metric_name] = {
            "precision": score_obj.precision,
            "recall": score_obj.recall,
            "fmeasure": score_obj.fmeasure,
        }
    return result


def compute_compression_ratio(transcript_words: int, summary_words: int) -> float:
    """Compute compression ratio (transcript length / summary length).

    Args:
        transcript_words: Number of words in transcript
        summary_words: Number of words in summary

    Returns:
        Compression ratio (higher = more compressed)
    """
    if summary_words == 0:
        return float("inf")
    return transcript_words / summary_words


def check_repetition(
    text: str, ngram_size: int = REPETITION_NGRAM_SIZE, threshold: int = REPETITION_THRESHOLD
) -> Tuple[bool, List[str]]:
    """Check for repetitive n-grams in text.

    Args:
        text: Text to check
        ngram_size: Size of n-grams to check
        threshold: Threshold for flagging repetition

    Returns:
        Tuple of (is_repetitive, list_of_repeated_ngrams)
    """
    words = text.lower().split()
    if len(words) < ngram_size:
        return False, []

    ngrams = []
    for i in range(len(words) - ngram_size + 1):
        ngram = " ".join(words[i : i + ngram_size])
        ngrams.append(ngram)

    ngram_counts = Counter(ngrams)
    repeated = [ngram for ngram, count in ngram_counts.items() if count > threshold]

    return len(repeated) > 0, repeated


def extract_keywords(text: str, top_n: int = KEYWORD_COVERAGE_TOP_N) -> List[str]:
    """Extract top keywords from text using simple frequency (excluding stopwords).

    Args:
        text: Text to extract keywords from
        top_n: Number of top keywords to return

    Returns:
        List of top keywords
    """
    # Simple stopwords list (can be expanded)
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
    }

    words = re.findall(r"\b[a-z]+\b", text.lower())
    word_freq = Counter(word for word in words if word not in stopwords and len(word) > 2)

    return [word for word, _ in word_freq.most_common(top_n)]


def compute_keyword_coverage(transcript: str, summary: str) -> Tuple[float, List[str], List[str]]:
    """Compute keyword coverage (how many transcript keywords appear in summary).

    Args:
        transcript: Original transcript
        summary: Generated summary

    Returns:
        Tuple of (coverage_ratio, covered_keywords, missing_keywords)
    """
    transcript_keywords = set(extract_keywords(transcript))
    summary_words = set(re.findall(r"\b[a-z]+\b", summary.lower()))

    covered = transcript_keywords & summary_words
    missing = transcript_keywords - summary_words

    coverage_ratio = len(covered) / len(transcript_keywords) if transcript_keywords else 0.0

    return coverage_ratio, list(covered), list(missing)


def evaluate_episode(
    episode_dir: Path,
    model: summarizer.SummaryModel,
    reduce_model: Optional[summarizer.SummaryModel],
    cfg: config.Config,
    use_short_reference: bool = False,
) -> Dict[str, Any]:
    """Evaluate summarization for a single episode.

    Args:
        episode_dir: Directory containing transcript.cleaned.txt and summary.gold.*.txt
        model: Summary model for MAP phase
        reduce_model: Optional separate model for REDUCE phase
        cfg: Configuration object
        use_short_reference: If True, use summary.gold.short.txt instead of summary.gold.long.txt

    Returns:
        Dictionary with evaluation results
    """
    episode_id = episode_dir.name
    transcript_path = episode_dir / "transcript.cleaned.txt"

    # Try to find reference summary (prefer long, fallback to short)
    if use_short_reference:
        reference_path = episode_dir / "summary.gold.short.txt"
        if not reference_path.exists():
            reference_path = episode_dir / "summary.gold.long.txt"
    else:
        reference_path = episode_dir / "summary.gold.long.txt"
        if not reference_path.exists():
            reference_path = episode_dir / "summary.gold.short.txt"

    if not transcript_path.exists():
        logger.warning(f"[{episode_id}] transcript.cleaned.txt not found, skipping")
        return {"episode_id": episode_id, "error": "transcript.cleaned.txt not found"}

    # Load cleaned transcript (already cleaned, but we can validate if raw exists)
    cleaned_transcript = load_text(transcript_path)
    if not cleaned_transcript:
        return {"episode_id": episode_id, "error": "empty cleaned transcript"}

    # Optional: Validate cleaning if raw transcript exists
    raw_transcript_path = episode_dir / "transcript.raw.txt"
    if raw_transcript_path.exists():
        raw_transcript = load_text(raw_transcript_path)
        if raw_transcript:
            # Re-clean raw transcript to validate it matches cleaned version
            validated_cleaned = summarizer.clean_transcript(raw_transcript)
            if validated_cleaned.strip() != cleaned_transcript.strip():
                logger.warning(
                    f"[{episode_id}] Cleaned transcript doesn't match pipeline output. "
                    "Using provided cleaned transcript as-is."
                )

    # Generate summary
    logger.info(f"[{episode_id}] Generating summary...")
    start_time = time.time()

    try:
        # Use same default logic as pipeline: chunk_size defaults to
        # DEFAULT_SUMMARY_CHUNK_SIZE if not set
        chunk_size = cfg.summary_chunk_size or config.DEFAULT_SUMMARY_CHUNK_SIZE
        summary = summarizer.summarize_long_text(
            model=model,
            text=cleaned_transcript,
            chunk_size=chunk_size,
            max_length=cfg.summary_max_length,
            min_length=cfg.summary_min_length,
            prompt=cfg.summary_prompt,
            reduce_model=reduce_model,
        )
        generation_time = time.time() - start_time
    except Exception as e:
        logger.error(f"[{episode_id}] Summarization failed: {e}")
        return {"episode_id": episode_id, "error": str(e)}

    if not summary:
        return {"episode_id": episode_id, "error": "empty summary generated"}

    # Compute metrics
    transcript_words = len(cleaned_transcript.split())
    summary_words = len(summary.split())
    transcript_chars = len(cleaned_transcript)
    summary_chars = len(summary)

    compression_ratio = compute_compression_ratio(transcript_words, summary_words)
    is_repetitive, repeated_ngrams = check_repetition(summary)
    keyword_coverage, covered_keywords, missing_keywords = compute_keyword_coverage(
        cleaned_transcript, summary
    )

    result: Dict[str, Any] = {
        "episode_id": episode_id,
        "transcript_words": transcript_words,
        "transcript_chars": transcript_chars,
        "summary_words": summary_words,
        "summary_chars": summary_chars,
        "compression_ratio": compression_ratio,
        "generation_time_seconds": generation_time,
        "repetition": {
            "is_repetitive": is_repetitive,
            "repeated_ngrams": repeated_ngrams[:10],  # Limit to first 10
        },
        "keyword_coverage": {
            "ratio": keyword_coverage,
            "covered_count": len(covered_keywords),
            "missing_count": len(missing_keywords),
            "covered_keywords": covered_keywords[:10],  # Limit to first 10
            "missing_keywords": missing_keywords[:10],  # Limit to first 10
        },
    }

    # ROUGE scores (if reference exists)
    if reference_path.exists():
        reference = load_text(reference_path)
        if reference:
            rouge_scores = compute_rouge_scores(summary, reference)
            result["rouge"] = rouge_scores
            result["reference_summary_words"] = len(reference.split())
            result["reference_type"] = "long" if "long" in reference_path.name else "short"
        else:
            result["rouge"] = None
            result["rouge_error"] = "empty reference summary"
    else:
        result["rouge"] = None
        result["rouge_error"] = "summary.gold.long.txt or summary.gold.short.txt not found"

    # Reference-free checks
    result["checks"] = {
        "compression_ok": MIN_COMPRESSION_RATIO <= compression_ratio <= MAX_COMPRESSION_RATIO,
        "no_repetition": not is_repetitive,
        "keyword_coverage_ok": keyword_coverage >= 0.3,  # At least 30% keyword coverage
    }

    return result


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate summarization quality")
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="data/eval",
        help="Directory containing evaluation episodes (default: data/eval)",
    )
    parser.add_argument(
        "--map-model",
        type=str,
        default=None,
        help=(
            "MAP model name/key (e.g., 'bart-large', 'bart-small', 'pegasus') "
            "or HuggingFace model ID. Defaults to 'bart-large' (same as app default)"
        ),
    )
    parser.add_argument(
        "--reduce-model",
        type=str,
        default=None,
        help=(
            "REDUCE model name/key (e.g., 'long-fast', 'long', 'bart-large') "
            "or HuggingFace model ID. Defaults to 'long-fast' "
            "(LED-base, same as app default)"
        ),
    )
    # Backward compatibility: support --model as alias for --map-model
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="(Deprecated: use --map-model) Model name/key for MAP phase. Defaults to 'bart-large'",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (JSON or YAML) - overrides model arguments",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path for results (default: results/eval_<timestamp>.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'mps', 'cpu', or None for auto)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--use-short-reference",
        action="store_true",
        help="Use summary.gold.short.txt instead of summary.gold.long.txt for ROUGE scoring",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load config or create from arguments
    if args.config:
        cfg = config.load_config_file(args.config)
    else:
        # Use --map-model if provided, fallback to --model for backward compatibility
        map_model_arg = args.map_model or args.model

        # Create minimal config for evaluation
        # If models not provided, defaults will be applied by
        # select_summary_model/select_reduce_model
        cfg = config.Config(
            rss_url="",  # Not needed for evaluation
            generate_summaries=True,
            summary_model=map_model_arg,  # None means use default (bart-large)
            summary_reduce_model=args.reduce_model,  # None means use default (long-fast)
            summary_device=args.device,
        )

    # Select models (uses app defaults if not provided: BART-large for MAP, LED for REDUCE)
    map_model_name = summarizer.select_summary_model(cfg)
    reduce_model_name = summarizer.select_reduce_model(cfg, map_model_name)

    logger.info(
        f"MAP model: {map_model_name} {'(default)' if not (args.map_model or args.model) else ''}"
    )
    logger.info(f"REDUCE model: {reduce_model_name} {'(default)' if not args.reduce_model else ''}")

    # Load models
    logger.info("Loading MAP model...")
    map_model = summarizer.SummaryModel(
        model_name=map_model_name,
        device=cfg.summary_device,
        cache_dir=cfg.summary_cache_dir,
    )

    reduce_model = None
    if reduce_model_name != map_model_name:
        logger.info("Loading REDUCE model...")
        reduce_model = summarizer.SummaryModel(
            model_name=reduce_model_name,
            device=cfg.summary_device,
            cache_dir=cfg.summary_cache_dir,
        )
    else:
        logger.info("Using MAP model for REDUCE phase")

    # Find evaluation episodes
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        logger.error(f"Evaluation directory not found: {eval_dir}")
        sys.exit(1)

    episode_dirs = [d for d in eval_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not episode_dirs:
        logger.error(f"No episode directories found in {eval_dir}")
        sys.exit(1)

    logger.info(f"Found {len(episode_dirs)} episodes to evaluate")

    # Evaluate each episode
    results = []
    for episode_dir in sorted(episode_dirs):
        result = evaluate_episode(
            episode_dir, map_model, reduce_model, cfg, args.use_short_reference
        )
        results.append(result)

    # Compute aggregate statistics
    successful_results = [r for r in results if "error" not in r]
    if not successful_results:
        logger.error("No successful evaluations")
        sys.exit(1)

    # Aggregate metrics
    total_episodes = len(successful_results)
    avg_generation_time = (
        sum(r["generation_time_seconds"] for r in successful_results) / total_episodes
    )
    avg_compression_ratio = sum(r["compression_ratio"] for r in successful_results) / total_episodes
    avg_keyword_coverage = (
        sum(r["keyword_coverage"]["ratio"] for r in successful_results) / total_episodes
    )

    # ROUGE scores (only for episodes with references)
    rouge_results = [r for r in successful_results if r.get("rouge") is not None]
    avg_rouge_scores = None
    if rouge_results:
        rouge1_f = sum(r["rouge"]["rouge1"]["fmeasure"] for r in rouge_results) / len(rouge_results)
        rouge2_f = sum(r["rouge"]["rouge2"]["fmeasure"] for r in rouge_results) / len(rouge_results)
        rougeL_f = sum(r["rouge"]["rougeL"]["fmeasure"] for r in rouge_results) / len(rouge_results)
        avg_rouge_scores = {
            "rouge1_f": rouge1_f,
            "rouge2_f": rouge2_f,
            "rougeL_f": rougeL_f,
            "episodes_with_references": len(rouge_results),
        }

    # Check pass rates
    compression_ok_count = sum(1 for r in successful_results if r["checks"]["compression_ok"])
    no_repetition_count = sum(1 for r in successful_results if r["checks"]["no_repetition"])
    keyword_coverage_ok_count = sum(
        1 for r in successful_results if r["checks"]["keyword_coverage_ok"]
    )

    # Build output
    output_data = {
        "model": {
            "map": map_model_name,
            "reduce": reduce_model_name,
        },
        "config": {
            "chunk_size": cfg.summary_chunk_size or config.DEFAULT_SUMMARY_CHUNK_SIZE,
            "max_length": cfg.summary_max_length,
            "min_length": cfg.summary_min_length,
            "word_chunk_size": cfg.summary_word_chunk_size
            or config.DEFAULT_SUMMARY_WORD_CHUNK_SIZE,
            "word_overlap": cfg.summary_word_overlap or config.DEFAULT_SUMMARY_WORD_OVERLAP,
            "device": cfg.summary_device or "auto",
            "has_custom_prompt": cfg.summary_prompt is not None,
        },
        "summary": {
            "total_episodes": total_episodes,
            "episodes_with_references": len(rouge_results),
            "avg_generation_time_seconds": avg_generation_time,
            "avg_compression_ratio": avg_compression_ratio,
            "avg_keyword_coverage": avg_keyword_coverage,
            "checks": {
                "compression_ok_rate": compression_ok_count / total_episodes,
                "no_repetition_rate": no_repetition_count / total_episodes,
                "keyword_coverage_ok_rate": keyword_coverage_ok_count / total_episodes,
            },
        },
        "rouge": avg_rouge_scores,
        "episodes": results,
    }

    # Determine output path (default to results/eval_<timestamp>.json if not provided)
    if args.output:
        output_path = Path(args.output)
    else:
        # Generate default filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"eval_{timestamp}.json"

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # Models and Configuration
    print("\n📊 Models & Configuration:")
    print(f"  MAP Model:    {map_model_name}")
    print(f"  REDUCE Model: {reduce_model_name}")
    print(f"  Device:       {cfg.summary_device or 'auto'}")

    # Key parameters that impact quality
    chunk_size = cfg.summary_chunk_size or config.DEFAULT_SUMMARY_CHUNK_SIZE
    word_chunk_size = cfg.summary_word_chunk_size or config.DEFAULT_SUMMARY_WORD_CHUNK_SIZE
    word_overlap = cfg.summary_word_overlap or config.DEFAULT_SUMMARY_WORD_OVERLAP

    print("\n⚙️  Summarization Parameters:")
    print(f"  Max Length:        {cfg.summary_max_length} tokens")
    print(f"  Min Length:        {cfg.summary_min_length} tokens")
    print(f"  Chunk Size:        {chunk_size} tokens")
    print(f"  Word Chunk Size:   {word_chunk_size} words")
    print(f"  Word Overlap:      {word_overlap} words")
    if cfg.summary_prompt:
        prompt_preview = (
            cfg.summary_prompt[:60] + "..." if len(cfg.summary_prompt) > 60 else cfg.summary_prompt
        )
        print(f"  Custom Prompt:     {prompt_preview}")
    else:
        print("  Custom Prompt:     None (using default)")

    # Evaluation Results
    print("\n📈 Evaluation Results:")
    print(f"  Episodes evaluated:        {total_episodes}")
    print(f"  Episodes with references:  {len(rouge_results)}")
    print(f"  Average generation time:    {avg_generation_time:.1f}s")
    print(f"  Average compression ratio: {avg_compression_ratio:.1f}×")
    print(f"  Average keyword coverage:   {avg_keyword_coverage:.1%}")

    if avg_rouge_scores:
        print("\n🎯 ROUGE Scores (F-measure):")
        print(f"  ROUGE-1: {avg_rouge_scores['rouge1_f']:.3f}")
        print(f"  ROUGE-2: {avg_rouge_scores['rouge2_f']:.3f}")
        print(f"  ROUGE-L: {avg_rouge_scores['rougeL_f']:.3f}")

    print("\n✅ Quality Checks Pass Rates:")
    compression_pct = compression_ok_count / total_episodes
    print(
        f"  Compression OK:      {compression_ok_count}/{total_episodes} "
        f"({compression_pct:.1%})"
    )
    repetition_pct = no_repetition_count / total_episodes
    print(
        f"  No Repetition:       {no_repetition_count}/{total_episodes} " f"({repetition_pct:.1%})"
    )
    keyword_pct = keyword_coverage_ok_count / total_episodes
    print(
        f"  Keyword Coverage OK: {keyword_coverage_ok_count}/{total_episodes} "
        f"({keyword_pct:.1%})"
    )

    print(f"\n💾 Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
