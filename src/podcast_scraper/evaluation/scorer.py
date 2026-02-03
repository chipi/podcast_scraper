"""Metrics scorer for experiment evaluation.

This module computes metrics from predictions, including:
- Intrinsic metrics (gates, length, performance, cost) - always computed
- Extrinsic metrics (ROUGE, BLEU, WER, embedding similarity) - computed when references exist

The scorer is separate from the runner to keep roles clean.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from podcast_scraper.evaluation.ner_scorer import compute_ner_vs_reference_metrics
from podcast_scraper.evaluation.schema_validator import (
    validate_summarization_reference,
)

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    import jiwer
except ImportError:
    jiwer = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    # Ensure required NLTK data is available
    try:
        import nltk

        nltk.download("punkt", quiet=True)
    except Exception:
        # If download fails, word_tokenize will fail later - that's OK
        pass
except ImportError:
    sentence_bleu = None
    SmoothingFunction = None
    word_tokenize = None

logger = logging.getLogger(__name__)


def load_predictions(predictions_path: Path) -> List[Dict[str, Any]]:
    """Load predictions from JSONL file.

    Args:
        predictions_path: Path to predictions.jsonl

    Returns:
        List of prediction records
    """
    predictions = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: ~4 chars per token).

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def compute_intrinsic_metrics(  # noqa: C901
    predictions: List[Dict[str, Any]],
    dataset_id: str,
    run_id: str,
    metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute intrinsic metrics (no reference needed).

    These are computed from predictions alone:
    - Gates: boilerplate/speaker/truncation
    - Length stats
    - Performance: latency
    - Cost

    Args:
        predictions: List of prediction records
        dataset_id: Dataset identifier
        run_id: Run identifier
        metadata_map: Optional mapping of episode_id -> metadata dict (for speaker detection)

    Returns:
        Intrinsic metrics dictionary
    """
    episode_count = len(predictions)

    # Quality gates - detect common issues
    boilerplate_leaks = []
    speaker_label_leaks = []  # Labels like "Host:", "Guest:", "Speaker 1:" (FAIL gate)
    speaker_name_leaks = []  # Actual names like "Alice", "Bob" from metadata (WARN only)
    truncated = []
    failed_episodes = []
    # Per-episode gate breakdown (for debugging which gate failed)
    episode_gate_failures = {}  # episode_id -> list of gate names that failed

    # Patterns for detection
    boilerplate_patterns = [
        "subscribe to our newsletter",
        "follow us on",
        "rate and review",
        "sponsor message",
        "advertisement",
        "this episode is brought to you by",
        "thanks to our sponsor",
    ]

    # Speaker label patterns (labels like "Host:", "Guest:", "Speaker 1:")
    # These are FAIL gates - should never appear in summaries
    speaker_label_patterns = [
        r"\bHost:\s*",
        r"\bGuest:\s*",
        r"\bSpeaker\s+\d+:\s*",
        r"\bInterviewer:\s*",
        r"\bInterviewee:\s*",
    ]

    import re

    for pred in predictions:
        episode_id = pred.get("episode_id", "unknown")
        # Handle both old format (summary_long) and new format (summary_final)
        output = pred.get("output", {})
        summary = (
            output.get("summary_final")
            or output.get("summary_long")
            or (output if isinstance(output, str) else "")
        )
        if not summary or not isinstance(summary, str):
            continue

        summary_lower = summary.lower()

        # Check for boilerplate leaks (respect expectations)
        has_boilerplate = False
        if metadata_map and episode_id in metadata_map:
            expectations = metadata_map[episode_id].get("expectations", {})
            allow_sponsor_content = expectations.get("allow_sponsor_content", False)

            # Only check for boilerplate if sponsor content is not allowed
            if not allow_sponsor_content:
                has_boilerplate = any(pattern in summary_lower for pattern in boilerplate_patterns)
        else:
            # Fallback: check for boilerplate if no expectations specified
            has_boilerplate = any(pattern in summary_lower for pattern in boilerplate_patterns)

        # Track which gates failed for this episode
        episode_failures = []

        if has_boilerplate:
            boilerplate_leaks.append(episode_id)
            episode_failures.append("boilerplate_leak")

        # Check for speaker label leaks (FAIL gate: "Host:", "Guest:", "Speaker 1:")
        has_speaker_label_leak = False
        if metadata_map and episode_id in metadata_map:
            episode_metadata = metadata_map[episode_id]
            expectations = episode_metadata.get("expectations", {})
            allow_speaker_labels = expectations.get("allow_speaker_labels", False)

            # Only check for label leaks if labels are not allowed
            if not allow_speaker_labels:
                has_speaker_label_leak = any(
                    re.search(pattern, summary, re.IGNORECASE) for pattern in speaker_label_patterns
                )
        else:
            # Fallback: check for label leaks if no expectations specified
            has_speaker_label_leak = any(
                re.search(pattern, summary, re.IGNORECASE) for pattern in speaker_label_patterns
            )

        if has_speaker_label_leak:
            speaker_label_leaks.append(episode_id)
            episode_failures.append("speaker_label_leak")

        # Check for speaker name leaks (WARN only: actual names like "Alice", "Bob")
        has_speaker_name_leak = False
        if metadata_map and episode_id in metadata_map:
            episode_metadata = metadata_map[episode_id]
            speakers = episode_metadata.get("speakers", [])
            expectations = episode_metadata.get("expectations", {})
            allow_speaker_names = expectations.get("allow_speaker_names", False)

            # Only check for name leaks if names are not allowed
            if not allow_speaker_names and speakers:
                for speaker in speakers:
                    if isinstance(speaker, dict):
                        speaker_name = speaker.get("name", "").strip()
                        # Skip placeholder names
                        if not speaker_name or speaker_name.startswith("TODO:"):
                            continue
                    elif isinstance(speaker, str):
                        speaker_name = speaker.strip()
                        if not speaker_name or speaker_name.startswith("TODO:"):
                            continue
                    else:
                        continue

                    # Check for speaker name followed by colon (e.g., "Alice Johnson:")
                    # or in brackets (e.g., "[Alice Johnson: Right.]")
                    name_pattern = re.escape(speaker_name)
                    # Match: "Name:" or "[Name:" or "Name:" at start of line
                    leak_patterns = [
                        rf"\b{name_pattern}:\s*",  # "Alice Johnson:"
                        rf"\[{name_pattern}:\s*",  # "[Alice Johnson:"
                        rf"\[{name_pattern}\s+",  # "[Alice Johnson "
                    ]
                    if any(re.search(pattern, summary, re.IGNORECASE) for pattern in leak_patterns):
                        has_speaker_name_leak = True
                        break

        if has_speaker_name_leak:
            speaker_name_leaks.append(episode_id)

        # Check for truncation (output ends abruptly or with common truncation markers)
        truncation_markers = ["...", "…", "[TRUNCATED]", "[CUT OFF]"]
        ends_with_marker = any(summary.rstrip().endswith(marker) for marker in truncation_markers)
        # Also check if output is suspiciously short compared to input
        input_length = pred.get("metadata", {}).get("input_length_chars", 0)
        output_length = pred.get("metadata", {}).get("output_length_chars", len(summary))
        # If output is less than 1% of input, likely truncated (for summarization)
        is_truncated = False
        if input_length > 0 and output_length > 0:
            compression_ratio = output_length / input_length
            if compression_ratio < 0.01:  # Less than 1% - suspiciously short
                is_truncated = True
                truncated.append(episode_id)
        elif ends_with_marker:
            is_truncated = True
            truncated.append(episode_id)

        # Track truncation failures
        if is_truncated:
            episode_failures.append("truncation")

        # Collect all failed episodes (gates: boilerplate, speaker_label_leak, truncation)
        # Note: speaker_name_leak is WARN only, not a gate
        # Use is_truncated (not ends_with_marker) to match the truncation rate logic
        if has_boilerplate or has_speaker_label_leak or is_truncated:
            failed_episodes.append(episode_id)
            # Store which gates failed for this episode
            if episode_failures:
                episode_gate_failures[episode_id] = episode_failures

    gates = {
        "boilerplate_leak_rate": (
            len(boilerplate_leaks) / episode_count if episode_count > 0 else 0.0
        ),
        "speaker_label_leak_rate": (
            len(speaker_label_leaks) / episode_count if episode_count > 0 else 0.0
        ),
        "truncation_rate": len(truncated) / episode_count if episode_count > 0 else 0.0,
        "failed_episodes": list(set(failed_episodes)),  # Deduplicate
        "episode_gate_failures": episode_gate_failures,  # Per-episode breakdown
    }

    # Warnings (not gates, but tracked for monitoring)
    warnings = {
        "speaker_name_leak_rate": (
            len(speaker_name_leaks) / episode_count if episode_count > 0 else 0.0
        ),
    }

    # Length metrics
    token_counts = []
    for pred in predictions:
        # Handle both old format (summary_long) and new format (summary_final)
        output = pred.get("output", {})
        summary = (
            output.get("summary_final")
            or output.get("summary_long")
            or (output if isinstance(output, str) else "")
        )
        if summary and isinstance(summary, str):
            token_counts.append(estimate_tokens(summary))

    length = {
        "avg_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
    }

    # Performance metrics
    latencies = [
        pred.get("metadata", {}).get("processing_time_seconds", 0) * 1000  # Convert to ms
        for pred in predictions
        if pred.get("metadata", {}).get("processing_time_seconds") is not None
    ]
    performance = {
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
    }

    # Cost metrics - extract from metadata if available
    # OpenAI providers may include cost information in metadata or usage info
    costs = []
    for pred in predictions:
        metadata = pred.get("metadata", {})
        # Check for cost in metadata (OpenAI providers may include this)
        if "cost_usd" in metadata:
            costs.append(metadata["cost_usd"])
        # Also check for usage info that we can compute cost from
        elif "usage" in metadata:
            usage = metadata["usage"]
            # Extract token counts if available
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            model = metadata.get("model", "")
            # Compute cost based on model pricing (if we have token counts)
            if prompt_tokens > 0 or completion_tokens > 0:
                # Use OpenAI pricing (approximate - actual pricing may vary)
                # This is a fallback if cost_usd is not directly provided
                # GPT-4o-mini: $0.15/1M input, $0.60/1M output
                # GPT-4o: $2.50/1M input, $10.00/1M output
                if "gpt-4o-mini" in model.lower():
                    input_cost = (prompt_tokens / 1_000_000) * 0.15
                    output_cost = (completion_tokens / 1_000_000) * 0.60
                    costs.append(input_cost + output_cost)
                elif "gpt-4o" in model.lower() and "mini" not in model.lower():
                    input_cost = (prompt_tokens / 1_000_000) * 2.50
                    output_cost = (completion_tokens / 1_000_000) * 10.00
                    costs.append(input_cost + output_cost)
                # For other models, we'd need to add pricing - skip for now

    # Build result dict - only include cost if we have cost data (skip for ML models)
    result = {
        "gates": gates,
        "warnings": warnings,  # Warnings (not gates, but tracked for monitoring)
        "length": length,
        "performance": performance,
    }

    # Only include cost section if we have cost data (OpenAI runs)
    if costs:
        result["cost"] = {
            "avg_cost_usd": sum(costs) / len(costs),
            "total_cost_usd": sum(costs),
        }
    # For ML models, skip cost section entirely (no cost data available)

    return result


def compute_rouge_vs_reference(
    predictions: List[Dict[str, Any]],
    reference_predictions: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """Compute ROUGE scores vs a reference.

    Args:
        predictions: Experiment predictions
        reference_predictions: Reference predictions

    Returns:
        Dictionary with rouge1_f1, rouge2_f1, rougeL_f1

    Raises:
        ImportError: If rouge-score library is not installed
    """
    if rouge_scorer is None:
        raise ImportError(
            "rouge-score library is required for ROUGE computation. "
            "Install with: pip install 'rouge-score>=0.1.2'"
        )

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Match predictions by episode_id
    pred_by_id = {p.get("episode_id"): p for p in predictions}
    ref_by_id = {p.get("episode_id"): p for p in reference_predictions}

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for episode_id in pred_by_id.keys():
        if episode_id not in ref_by_id:
            logger.warning(f"Episode {episode_id} not found in reference, skipping ROUGE")
            continue

        # Handle both old format (summary_long) and new format (summary_final)
        pred_output = pred_by_id[episode_id].get("output", {})
        pred_text = (
            pred_output.get("summary_final")
            or pred_output.get("summary_long")
            or (pred_output if isinstance(pred_output, str) else "")
        )
        ref_output = ref_by_id[episode_id].get("output", {})
        ref_text = (
            ref_output.get("summary_final")
            or ref_output.get("summary_long")
            or (ref_output if isinstance(ref_output, str) else "")
        )

        if not pred_text or not ref_text:
            logger.warning(f"Empty text for episode {episode_id}, skipping ROUGE")
            continue

        scores = scorer.score(ref_text, pred_text)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    # Compute averages
    if not rouge1_scores:
        return {
            "rouge1_f1": None,
            "rouge2_f1": None,
            "rougeL_f1": None,
        }

    return {
        "rouge1_f1": sum(rouge1_scores) / len(rouge1_scores),
        "rouge2_f1": sum(rouge2_scores) / len(rouge2_scores),
        "rougeL_f1": sum(rougeL_scores) / len(rougeL_scores),
    }


def compute_bleu_vs_reference(
    predictions: List[Dict[str, Any]],
    reference_predictions: List[Dict[str, Any]],
) -> Optional[float]:
    """Compute BLEU score vs a reference.

    Args:
        predictions: Experiment predictions
        reference_predictions: Reference predictions

    Returns:
        Average BLEU score (0-1) or None if nltk is not available

    Raises:
        ImportError: If nltk library is not installed
    """
    if sentence_bleu is None or word_tokenize is None or SmoothingFunction is None:
        raise ImportError(
            "nltk library is required for BLEU computation. "
            "Install with: pip install 'nltk>=3.8.0'"
        )

    # Match predictions by episode_id
    pred_by_id = {p.get("episode_id"): p for p in predictions}
    ref_by_id = {p.get("episode_id"): p for p in reference_predictions}

    bleu_scores = []
    smoothing = SmoothingFunction().method1

    for episode_id in pred_by_id.keys():
        if episode_id not in ref_by_id:
            logger.warning(f"Episode {episode_id} not found in reference, skipping BLEU")
            continue

        # Handle both old format (summary_long) and new format (summary_final)
        pred_output = pred_by_id[episode_id].get("output", {})
        pred_text = (
            pred_output.get("summary_final")
            or pred_output.get("summary_long")
            or (pred_output if isinstance(pred_output, str) else "")
        )
        ref_output = ref_by_id[episode_id].get("output", {})
        ref_text = (
            ref_output.get("summary_final")
            or ref_output.get("summary_long")
            or (ref_output if isinstance(ref_output, str) else "")
        )

        if not pred_text or not ref_text:
            logger.warning(f"Empty text for episode {episode_id}, skipping BLEU")
            continue

        try:
            # Tokenize texts
            pred_tokens = word_tokenize(pred_text.lower())
            ref_tokens = word_tokenize(ref_text.lower())

            # Compute BLEU score with smoothing
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
            bleu_scores.append(bleu)
        except Exception as e:
            logger.warning(f"Error computing BLEU for episode {episode_id}: {e}")
            continue

    if not bleu_scores:
        return None

    return float(sum(bleu_scores) / len(bleu_scores))


def compute_wer_vs_reference(
    predictions: List[Dict[str, Any]],
    reference_predictions: List[Dict[str, Any]],
) -> Optional[float]:
    """Compute Word Error Rate (WER) vs a reference.

    WER is typically used for transcription tasks, but can also be useful
    for summarization to measure word-level differences.

    Args:
        predictions: Experiment predictions
        reference_predictions: Reference predictions

    Returns:
        Average WER (0-1, where 0 is perfect, 1 is completely different)
        or None if jiwer is not available

    Raises:
        ImportError: If jiwer library is not installed
    """
    if jiwer is None:
        raise ImportError(
            "jiwer library is required for WER computation. "
            "Install with: pip install 'jiwer>=3.0.0'"
        )

    # Match predictions by episode_id
    pred_by_id = {p.get("episode_id"): p for p in predictions}
    ref_by_id = {p.get("episode_id"): p for p in reference_predictions}

    wer_scores = []

    for episode_id in pred_by_id.keys():
        if episode_id not in ref_by_id:
            logger.warning(f"Episode {episode_id} not found in reference, skipping WER")
            continue

        # Handle both old format (summary_long) and new format (summary_final)
        pred_output = pred_by_id[episode_id].get("output", {})
        pred_text = (
            pred_output.get("summary_final")
            or pred_output.get("summary_long")
            or (pred_output if isinstance(pred_output, str) else "")
        )
        ref_output = ref_by_id[episode_id].get("output", {})
        ref_text = (
            ref_output.get("summary_final")
            or ref_output.get("summary_long")
            or (ref_output if isinstance(ref_output, str) else "")
        )

        if not pred_text or not ref_text:
            logger.warning(f"Empty text for episode {episode_id}, skipping WER")
            continue

        try:
            # Compute WER using jiwer
            wer = jiwer.wer(ref_text, pred_text)
            wer_scores.append(wer)
        except Exception as e:
            logger.warning(f"Error computing WER for episode {episode_id}: {e}")
            continue

    if not wer_scores:
        return None

    return float(sum(wer_scores) / len(wer_scores))


def compute_embedding_similarity(
    predictions: List[Dict[str, Any]],
    reference_predictions: List[Dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2",
) -> Optional[float]:
    """Compute average cosine similarity between embeddings.

    Uses sentence-transformers to compute semantic embeddings and then
    calculates cosine similarity between prediction and reference embeddings.

    Args:
        predictions: Experiment predictions
        reference_predictions: Reference predictions
        model_name: Name of the sentence-transformer model to use (default: all-MiniLM-L6-v2)

    Returns:
        Average cosine similarity (0-1, where 1 is identical)
        or None if sentence-transformers is not available

    Raises:
        ImportError: If sentence-transformers library is not installed
    """
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers library is required for embedding similarity computation. "
            "Install with: pip install 'sentence-transformers>=2.2.0'"
        )

    try:
        # Load model (will download on first use)
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Failed to load sentence-transformer model '{model_name}': {e}")
        return None

    # Match predictions by episode_id
    pred_by_id = {p.get("episode_id"): p for p in predictions}
    ref_by_id = {p.get("episode_id"): p for p in reference_predictions}

    similarities = []

    for episode_id in pred_by_id.keys():
        if episode_id not in ref_by_id:
            logger.warning(
                f"Episode {episode_id} not found in reference, skipping embedding similarity"
            )
            continue

        # Handle both old format (summary_long) and new format (summary_final)
        pred_output = pred_by_id[episode_id].get("output", {})
        pred_text = (
            pred_output.get("summary_final")
            or pred_output.get("summary_long")
            or (pred_output if isinstance(pred_output, str) else "")
        )
        ref_output = ref_by_id[episode_id].get("output", {})
        ref_text = (
            ref_output.get("summary_final")
            or ref_output.get("summary_long")
            or (ref_output if isinstance(ref_output, str) else "")
        )

        if not pred_text or not ref_text:
            logger.warning(f"Empty text for episode {episode_id}, skipping embedding similarity")
            continue

        try:
            # Compute embeddings
            pred_embedding = model.encode(pred_text, convert_to_tensor=True)
            ref_embedding = model.encode(ref_text, convert_to_tensor=True)

            # Compute cosine similarity using numpy (more portable than torch)
            import numpy as np

            # Convert to numpy if needed
            if hasattr(pred_embedding, "cpu"):
                pred_embedding = pred_embedding.cpu().numpy()
            if hasattr(ref_embedding, "cpu"):
                ref_embedding = ref_embedding.cpu().numpy()

            # Compute cosine similarity manually
            dot_product = np.dot(pred_embedding, ref_embedding)
            norm_pred = np.linalg.norm(pred_embedding)
            norm_ref = np.linalg.norm(ref_embedding)
            similarity = dot_product / (norm_pred * norm_ref) if (norm_pred * norm_ref) > 0 else 0.0
            similarities.append(float(similarity))
        except Exception as e:
            logger.warning(f"Error computing embedding similarity for episode {episode_id}: {e}")
            continue

    if not similarities:
        return None

    return sum(similarities) / len(similarities)


def compute_vs_reference_metrics(
    predictions: List[Dict[str, Any]],
    reference_id: str,
    reference_path: Path,
) -> Dict[str, Any]:
    """Compute metrics vs a reference.

    Args:
        predictions: Experiment predictions
        reference_id: Reference identifier
        reference_path: Path to reference directory

    Returns:
        Metrics vs reference dictionary
    """
    # Load reference predictions
    ref_predictions_path = reference_path / "predictions.jsonl"
    if not ref_predictions_path.exists():
        raise FileNotFoundError(f"Reference predictions not found: {ref_predictions_path}")

    reference_predictions = load_predictions(ref_predictions_path)

    # Validate each reference entry against schema
    for ref_entry in reference_predictions:
        try:
            validate_summarization_reference(ref_entry)
        except ValueError as e:
            raise ValueError(
                f"Reference entry validation failed for episode "
                f"{ref_entry.get('episode_id', 'unknown')}: {e}"
            ) from e

    # Load reference metadata
    ref_metadata_path = reference_path / "baseline.json"
    reference_quality = None
    if ref_metadata_path.exists():
        ref_metadata = json.loads(ref_metadata_path.read_text(encoding="utf-8"))
        reference_quality = ref_metadata.get("reference_quality")

    # Match predictions by episode_id
    pred_by_id = {p.get("episode_id"): p for p in predictions}
    ref_by_id = {p.get("episode_id"): p for p in reference_predictions}

    # Validate episode IDs match
    pred_ids = set(pred_by_id.keys())
    ref_ids = set(ref_by_id.keys())
    if pred_ids != ref_ids:
        missing = ref_ids - pred_ids
        extra = pred_ids - ref_ids
        raise ValueError(
            f"Episode ID mismatch for reference '{reference_id}': "
            f"missing={missing}, extra={extra}"
        )

    # Compute ROUGE
    try:
        rouge_scores = compute_rouge_vs_reference(predictions, reference_predictions)
    except ImportError as e:
        logger.warning(f"ROUGE computation skipped: {e}")
        rouge_scores = {"rouge1_f1": None, "rouge2_f1": None, "rougeL_f1": None}

    # Compute BLEU
    try:
        bleu_score = compute_bleu_vs_reference(predictions, reference_predictions)
    except ImportError as e:
        logger.warning(f"BLEU computation skipped: {e}")
        bleu_score = None

    # Compute WER
    try:
        wer_score = compute_wer_vs_reference(predictions, reference_predictions)
    except ImportError as e:
        logger.warning(f"WER computation skipped: {e}")
        wer_score = None

    # Compute embedding similarity
    try:
        embedding_similarity = compute_embedding_similarity(predictions, reference_predictions)
    except ImportError as e:
        logger.warning(f"Embedding similarity computation skipped: {e}")
        embedding_similarity = None

    # Compute coverage ratio (ML tokens / silver tokens)
    ml_token_counts = []
    silver_token_counts = []
    for episode_id in pred_ids:
        pred = pred_by_id[episode_id]
        ref = ref_by_id[episode_id]

        # Extract summary text from predictions (handle both old and new formats)
        pred_output = pred.get("output", {})
        pred_summary = (
            pred_output.get("summary_final")
            or pred_output.get("summary_long")
            or (pred_output if isinstance(pred_output, str) else "")
        )
        if pred_summary and isinstance(pred_summary, str):
            ml_token_counts.append(estimate_tokens(pred_summary))

        # Extract summary text from reference
        ref_output = ref.get("output", {})
        ref_summary = (
            ref_output.get("summary_final")
            or ref_output.get("summary_long")
            or (ref_output if isinstance(ref_output, str) else "")
        )
        if ref_summary and isinstance(ref_summary, str):
            silver_token_counts.append(estimate_tokens(ref_summary))

    # Calculate coverage ratio (average ML tokens / average silver tokens)
    coverage_ratio = None
    if ml_token_counts and silver_token_counts:
        avg_ml_tokens = sum(ml_token_counts) / len(ml_token_counts)
        avg_silver_tokens = sum(silver_token_counts) / len(silver_token_counts)
        if avg_silver_tokens > 0:
            coverage_ratio = avg_ml_tokens / avg_silver_tokens

    return {
        "reference_quality": reference_quality,
        "rouge1_f1": rouge_scores.get("rouge1_f1"),
        "rouge2_f1": rouge_scores.get("rouge2_f1"),
        "rougeL_f1": rouge_scores.get("rougeL_f1"),
        "bleu": bleu_score,
        "wer": wer_score,
        "embedding_cosine": embedding_similarity,
        "coverage_ratio": coverage_ratio,
        "numbers_retained": None,  # TODO: Implement numbers retention metric
    }


def score_run(
    predictions_path: Path,
    dataset_id: str,
    run_id: str,
    reference_paths: Optional[Dict[str, Path]] = None,
    metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    scoring_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score a run and compute all metrics.

    Args:
        predictions_path: Path to predictions.jsonl
        dataset_id: Dataset identifier
        run_id: Run identifier
        reference_paths: Optional dict of {reference_id: reference_path}
        metadata_map: Optional mapping of episode_id -> metadata dict (for speaker detection)
        scoring_params: Optional scoring parameters (e.g., {"match": ["exact", "overlap"]} for NER)

    Returns:
        Complete metrics dictionary with intrinsic and vs_reference sections
    """
    # Load predictions
    predictions = load_predictions(predictions_path)

    # Determine task type from predictions (check if entities or summary_final exists)
    task_type = None
    if predictions:
        first_pred = predictions[0]
        output = first_pred.get("output", {})
        if "entities" in output:
            task_type = "ner_entities"
        elif "summary_final" in output or "summary_long" in output:
            task_type = "summarization"

    # Compute intrinsic metrics (always)
    intrinsic = compute_intrinsic_metrics(
        predictions, dataset_id, run_id, metadata_map=metadata_map
    )

    # Compute vs_reference metrics (if references provided)
    vs_reference = {}
    if reference_paths:
        for ref_id, ref_path in reference_paths.items():
            try:
                if task_type == "ner_entities":
                    vs_reference[ref_id] = compute_ner_vs_reference_metrics(
                        predictions,
                        ref_id,
                        ref_path,
                        scoring_params=scoring_params,
                        dataset_id=dataset_id,
                        metadata_map=metadata_map,
                    )
                else:
                    vs_reference[ref_id] = compute_vs_reference_metrics(
                        predictions, ref_id, ref_path
                    )
            except Exception as e:
                logger.error(f"Failed to compute metrics vs reference '{ref_id}': {e}")
                # Continue with other references
                vs_reference[ref_id] = {"error": str(e)}

    # Build metrics dict with schema field
    metrics = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "episode_count": len(predictions),
        "intrinsic": intrinsic,
        "vs_reference": vs_reference if vs_reference else None,
    }

    # Add schema field based on task type
    if task_type == "ner_entities":
        metrics["schema"] = "metrics_ner_v1"
        metrics["task"] = "ner_entities"
    elif task_type == "summarization":
        metrics["schema"] = "metrics_summarization_v1"
        metrics["task"] = "summarization"

    return metrics
