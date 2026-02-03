"""NER-specific scoring functions for entity extraction evaluation.

This module computes precision, recall, and F1 for NER tasks using
exact and overlap matching strategies.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def hash_text(text: str) -> str:
    """Compute SHA256 hash of text.

    Args:
        text: Text to hash

    Returns:
        SHA256 hex digest
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_entity_text(text: str) -> str:
    """Normalize entity text for entity-set comparison.

    Normalization steps:
    - Trim whitespace
    - Strip trailing punctuation (:, ,, ., etc.)
    - Lowercase

    Args:
        text: Entity text to normalize

    Returns:
        Normalized text string
    """
    text = text.strip()
    # Strip trailing punctuation (common in transcripts: "Maya:", "Liam,")
    while text and text[-1] in ":,.;!?":
        text = text[:-1].strip()
    return text.lower()


def validate_gold_index(index_data: Dict[str, Any], schema_path: Path) -> None:
    """Validate gold index JSON against schema.

    Args:
        index_data: Index JSON data
        schema_path: Path to schema file

    Raises:
        ValueError: If validation fails
    """
    # Basic validation (can be enhanced with jsonschema library if available)
    required_fields = ["schema", "dataset_id", "episodes"]
    for field in required_fields:
        if field not in index_data:
            raise ValueError(f"Gold index missing required field: {field}")

    if index_data.get("schema") != "ner_entities_gold_index_v1":
        raise ValueError(
            f"Invalid schema in gold index: {index_data.get('schema')}, "
            f"expected 'ner_entities_gold_index_v1'"
        )

    if not isinstance(index_data.get("episodes"), list):
        raise ValueError("Gold index 'episodes' must be a list")


def validate_gold_episode(episode_data: Dict[str, Any], schema_path: Path) -> None:
    """Validate gold episode JSON against schema.

    Args:
        episode_data: Episode JSON data
        schema_path: Path to schema file

    Raises:
        ValueError: If validation fails
    """
    # Basic validation (can be enhanced with jsonschema library if available)
    required_fields = ["schema", "dataset_id", "episode_id", "text_fingerprint", "entities"]
    for field in required_fields:
        if field not in episode_data:
            raise ValueError(f"Gold episode missing required field: {field}")

    if episode_data.get("schema") != "ner_entities_gold_v1":
        raise ValueError(
            f"Invalid schema in gold episode: {episode_data.get('schema')}, "
            f"expected 'ner_entities_gold_v1'"
        )

    if not isinstance(episode_data.get("entities"), list):
        raise ValueError("Gold episode 'entities' must be a list")

    # Validate entity structure
    for entity in episode_data.get("entities", []):
        if not all(key in entity for key in ["start", "end", "text", "label"]):
            raise ValueError("Gold entity missing required fields: start, end, text, label")


def load_gold_reference(
    reference_path: Path, dataset_id: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Load gold reference entities from reference directory.

    High-level flow:
    1. Load index.json
    2. For each episode in index.episodes:
       - Load <episode_id>.json
       - Validate against schema
       - Verify text_fingerprint (if transcript available)
    3. Return only episodes with valid gold data

    Args:
        reference_path: Path to reference directory (contains index.json and episode JSON files)
        dataset_id: Dataset ID for text fingerprint verification (required for gold references)

    Returns:
        Dict mapping episode_id -> gold entities dict

    Raises:
        FileNotFoundError: If index.json is missing or materialized transcript not found
        ValueError: If schema validation fails or text fingerprint mismatch
    """
    # Resolve path to absolute to avoid path resolution issues
    reference_path = reference_path.resolve()

    gold_by_episode: Dict[str, Dict[str, Any]] = {}

    if not reference_path.is_dir():
        raise ValueError(f"Gold reference path must be a directory: {reference_path}")

    # Step 1: Load and validate index.json
    index_path = reference_path / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Gold reference index.json not found: {index_path}")

    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        index_schema_path = Path("data/eval/schemas/ner_entities_gold_index_v1.json")
        validate_gold_index(index_data, index_schema_path)
    except Exception as e:
        raise ValueError(f"Failed to load or validate gold index: {e}") from e

    # Step 2: Load each episode from index.episodes
    episode_ids = index_data.get("episodes", [])
    if not episode_ids:
        logger.warning(f"Gold index has no episodes: {index_path}")
        return gold_by_episode

    episode_schema_path = Path("data/eval/schemas/ner_entities_gold_v1.json")

    for episode_id in episode_ids:
        episode_file = reference_path / f"{episode_id}.json"
        if not episode_file.exists():
            raise FileNotFoundError(
                f"Gold episode file not found: {episode_file}. "
                "All episodes listed in index.json must exist."
            )

        # Load episode JSON
        episode_data = json.loads(episode_file.read_text(encoding="utf-8"))

        # Validate against schema
        validate_gold_episode(episode_data, episode_schema_path)

        # HARD GUARDRAIL: Verify text_fingerprint matches materialized transcript
        # This prevents scoring against mismatched transcripts (e.g., different preprocessing)
        if not dataset_id:
            raise ValueError(
                f"dataset_id is required for gold reference validation. "
                f"Cannot verify text fingerprint for {episode_id} without dataset_id."
            )

        text_fingerprint = episode_data.get("text_fingerprint", "")
        if not text_fingerprint:
            raise ValueError(
                f"Gold reference for {episode_id} is missing text_fingerprint. "
                "Cannot verify transcript integrity."
            )

        if not text_fingerprint.startswith("sha256:"):
            raise ValueError(
                f"Invalid text_fingerprint format for {episode_id}: "
                f"expected 'sha256:<hex>', got '{text_fingerprint[:50]}...'"
            )

        expected_hash = text_fingerprint[7:]  # Remove "sha256:" prefix

        # Load materialized transcript to verify fingerprint
        materialized_dir = Path("data/eval/materialized") / dataset_id
        transcript_path = materialized_dir / f"{episode_id}.txt"
        if not transcript_path.exists():
            raise FileNotFoundError(
                f"Materialized transcript not found for {episode_id} at {transcript_path}. "
                "Cannot verify text fingerprint. Ensure dataset is materialized."
            )

        transcript_text = transcript_path.read_text(encoding="utf-8").strip()
        actual_hash = hash_text(transcript_text)

        if actual_hash != expected_hash:
            raise ValueError(
                f"Text fingerprint mismatch for {episode_id}: "
                f"gold reference expects {expected_hash[:16]}..., "
                f"but materialized transcript has {actual_hash[:16]}... "
                f"This indicates the transcript has changed (different preprocessing, "
                f"different source, etc.). Scoring aborted to prevent confusion. "
                f"Gold reference: {episode_file}, "
                f"Materialized transcript: {transcript_path}"
            )

        gold_by_episode[episode_id] = episode_data
        logger.debug(f"Loaded gold reference for episode {episode_id} (fingerprint verified)")

    logger.info(f"Loaded {len(gold_by_episode)} gold episode(s) from {reference_path}")
    return gold_by_episode


def match_exact(pred_entity: Dict[str, Any], gold_entity: Dict[str, Any]) -> bool:
    """Check if predicted entity exactly matches gold entity.

    Exact match requires:
    - Same label
    - Same start position
    - Same end position

    Args:
        pred_entity: Predicted entity dict with start, end, text, label
        gold_entity: Gold entity dict with start, end, text, label

    Returns:
        True if exact match
    """
    return (
        pred_entity.get("label") == gold_entity.get("label")
        and pred_entity.get("start") == gold_entity.get("start")
        and pred_entity.get("end") == gold_entity.get("end")
    )


def match_overlap(pred_entity: Dict[str, Any], gold_entity: Dict[str, Any]) -> bool:
    """Check if predicted entity overlaps with gold entity.

    Overlap match requires:
    - Same label
    - Spans overlap (pred.start < gold.end AND gold.start < pred.end)

    Args:
        pred_entity: Predicted entity dict with start, end, text, label
        gold_entity: Gold entity dict with start, end, text, label

    Returns:
        True if overlap match
    """
    if pred_entity.get("label") != gold_entity.get("label"):
        return False

    pred_start = int(pred_entity.get("start", 0))
    pred_end = int(pred_entity.get("end", 0))
    gold_start = int(gold_entity.get("start", 0))
    gold_end = int(gold_entity.get("end", 0))

    # Check overlap: pred.start < gold.end AND gold.start < pred.end
    return bool(pred_start < gold_end and gold_start < pred_end)


def match_entities_one_to_one(
    pred_entities: List[Dict[str, Any]],
    gold_entities: List[Dict[str, Any]],
    match_fn,
) -> Tuple[List[int], List[int], List[int]]:
    """Match predicted entities to gold entities using one-to-one matching.

    Each gold entity can match at most one prediction, and each prediction
    can match at most one gold. Uses greedy matching by highest overlap.

    Args:
        pred_entities: List of predicted entity dicts
        gold_entities: List of gold entity dicts
        match_fn: Function to check if two entities match (exact or overlap)

    Returns:
        Tuple of (matched_pred_indices, matched_gold_indices, unmatched_pred_indices)
    """
    matched_pred: Set[int] = set()
    matched_gold: Set[int] = set()
    matches: List[Tuple[int, int]] = []  # (pred_idx, gold_idx)

    # Build all possible matches
    for pred_idx, pred_ent in enumerate(pred_entities):
        for gold_idx, gold_ent in enumerate(gold_entities):
            if match_fn(pred_ent, gold_ent):
                matches.append((pred_idx, gold_idx))

    # Sort by overlap (for overlap matching, prefer higher overlap)
    # For exact matching, all matches are equivalent
    def match_quality(pred_idx: int, gold_idx: int) -> float:
        pred_ent = pred_entities[pred_idx]
        gold_ent = gold_entities[gold_idx]
        # Compute overlap ratio
        pred_start = int(pred_ent.get("start", 0))
        pred_end = int(pred_ent.get("end", 0))
        gold_start = int(gold_ent.get("start", 0))
        gold_end = int(gold_ent.get("end", 0))

        overlap_start = max(pred_start, gold_start)
        overlap_end = min(pred_end, gold_end)
        overlap_len = max(0, overlap_end - overlap_start)

        pred_len = pred_end - pred_start
        gold_len = gold_end - gold_start

        if pred_len == 0 or gold_len == 0:
            return 0.0

        # Use F1-like score: 2 * overlap / (pred_len + gold_len)
        return float((2.0 * overlap_len) / (pred_len + gold_len))

    matches.sort(key=lambda m: match_quality(m[0], m[1]), reverse=True)

    # Greedy matching: assign each match if both entities are still available
    for pred_idx, gold_idx in matches:
        if pred_idx not in matched_pred and gold_idx not in matched_gold:
            matched_pred.add(pred_idx)
            matched_gold.add(gold_idx)

    unmatched_pred = [i for i in range(len(pred_entities)) if i not in matched_pred]
    matched_pred_list = sorted(matched_pred)
    matched_gold_list = sorted(matched_gold)

    return matched_pred_list, matched_gold_list, unmatched_pred


def derive_gold_scope(gold_entities: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Derive scope sets from gold entities.

    Gold-scoped NER scoring rule:
    - Only evaluate predictions that could possibly be part of gold
    - PERSON: only if text appears in gold PERSON entities
    - ORG: only if text appears in gold ORG entities
    - Everything else: ignored (not in scope)

    Args:
        gold_entities: List of gold entity dicts

    Returns:
        Dict with "person_texts" (set) and "org_texts" (set)
    """
    scope_person_texts: Set[str] = set()
    scope_org_texts: Set[str] = set()

    for gold_ent in gold_entities:
        label = gold_ent.get("label", "")
        text = gold_ent.get("text", "").strip().lower()
        if not text:
            continue

        if label == "PERSON":
            scope_person_texts.add(text)
        elif label == "ORG":
            scope_org_texts.add(text)

    return {
        "person_texts": scope_person_texts,
        "org_texts": scope_org_texts,
    }


def is_in_gold_scope(
    pred_entity: Dict[str, Any],
    gold_scope: Dict[str, Set[str]],
) -> bool:
    """Check if a predicted entity is within the gold policy scope.

    Args:
        pred_entity: Predicted entity dict with label, text, start, end
        gold_scope: Dict with "person_texts" (set) and "org_texts" (set) derived from gold

    Returns:
        True if entity is in scope and should be scored, False otherwise
    """
    label = pred_entity.get("label", "")
    text = pred_entity.get("text", "").strip().lower()

    if label == "PERSON":
        scope_person_texts = gold_scope.get("person_texts", set())
        return text in scope_person_texts

    if label == "ORG":
        scope_org_texts = gold_scope.get("org_texts", set())
        return text in scope_org_texts

    # All other labels are not in scope
    return False


def compute_ner_metrics_for_match_type(
    predictions: List[Dict[str, Any]],
    gold_by_episode: Dict[str, Dict[str, Any]],
    match_type: str,
    metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute NER metrics for a specific match type (exact or overlap) with scope-aware filtering.

    Args:
        predictions: List of prediction records
        gold_by_episode: Dict mapping episode_id -> gold entities dict
        match_type: "exact" or "overlap"
        metadata_map: Optional mapping of episode_id -> metadata dict (for scope filtering)

    Returns:
        Metrics dict with TP, FP, FN, precision, recall, F1, and per-label metrics
    """
    match_fn = match_exact if match_type == "exact" else match_overlap

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Per-label metrics
    label_metrics: Dict[str, Dict[str, int]] = {}

    episodes_scored = 0
    episodes_total = len(predictions)

    for pred_record in predictions:
        episode_id = pred_record.get("episode_id")
        if not episode_id:
            continue

        # Get gold entities for this episode
        gold_data = gold_by_episode.get(episode_id)
        if not gold_data:
            logger.debug(f"No gold reference for episode {episode_id}, skipping")
            continue

        episodes_scored += 1

        # Extract predicted entities
        pred_output = pred_record.get("output", {})
        pred_entities = pred_output.get("entities", [])

        # Extract gold entities
        gold_entities = gold_data.get("entities", [])

        # Step A: Derive scope sets from gold entities
        gold_scope = derive_gold_scope(gold_entities)

        # Step B: Filter predictions - only keep those in gold scope
        scoped_pred_entities = []
        scoped_pred_indices = []  # Original indices for tracking
        ignored_by_label: Dict[str, int] = {}

        for idx, pred_ent in enumerate(pred_entities):
            if is_in_gold_scope(pred_ent, gold_scope):
                scoped_pred_entities.append(pred_ent)
                scoped_pred_indices.append(idx)
            else:
                # Track ignored predictions by label
                label = pred_ent.get("label", "UNKNOWN")
                ignored_by_label[label] = ignored_by_label.get(label, 0) + 1

        # Log filtering stats
        ignored_total = len(pred_entities) - len(scoped_pred_entities)
        logger.info(
            f"Episode {episode_id}: {len(pred_entities)} predicted entities → "
            f"{len(scoped_pred_entities)} in scope ({ignored_total} ignored). "
            f"Gold scope: PERSON={len(gold_scope['person_texts'])} texts, "
            f"ORG={len(gold_scope['org_texts'])} texts. "
            f"Ignored by label: {ignored_by_label}"
        )

        # Match only scoped predictions to gold entities
        matched_pred, matched_gold, unmatched_pred = match_entities_one_to_one(
            scoped_pred_entities, gold_entities, match_fn
        )

        # Count TP, FP, FN
        # TP: scoped predictions that matched gold
        tp = len(matched_pred)
        # FP: scoped predictions that didn't match gold
        fp = len(unmatched_pred)
        # FN: gold entities that weren't matched (always counted)
        fn = len(gold_entities) - len(matched_gold)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per-label metrics (use scoped entities)
        for scoped_idx in matched_pred:
            label = scoped_pred_entities[scoped_idx].get("label", "UNKNOWN")
            if label not in label_metrics:
                label_metrics[label] = {"tp": 0, "fp": 0, "fn": 0}
            label_metrics[label]["tp"] += 1

        for scoped_idx in unmatched_pred:
            label = scoped_pred_entities[scoped_idx].get("label", "UNKNOWN")
            if label not in label_metrics:
                label_metrics[label] = {"tp": 0, "fp": 0, "fn": 0}
            label_metrics[label]["fp"] += 1

        for gold_idx in range(len(gold_entities)):
            if gold_idx not in matched_gold:
                label = gold_entities[gold_idx].get("label", "UNKNOWN")
                if label not in label_metrics:
                    label_metrics[label] = {"tp": 0, "fp": 0, "fn": 0}
                label_metrics[label]["fn"] += 1

    # Compute overall precision, recall, F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute per-label F1
    per_label_f1: Dict[str, float] = {}
    for label, counts in label_metrics.items():
        label_tp = counts["tp"]
        label_fp = counts["fp"]
        label_fn = counts["fn"]
        label_precision = label_tp / (label_tp + label_fp) if (label_tp + label_fp) > 0 else 0.0
        label_recall = label_tp / (label_tp + label_fn) if (label_tp + label_fn) > 0 else 0.0
        label_f1 = (
            2 * label_precision * label_recall / (label_precision + label_recall)
            if (label_precision + label_recall) > 0
            else 0.0
        )
        per_label_f1[label] = label_f1

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_label_f1": per_label_f1,
        "episodes_scored": episodes_scored,
        "episodes_total": episodes_total,
    }


def compute_entity_set_metrics(
    predictions: List[Dict[str, Any]],
    gold_by_episode: Dict[str, Dict[str, Any]],
    metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute entity-set metrics (position-agnostic, text-based comparison).

    Entity-set scoring answers: "Did we extract the correct entities for KG?"
    - Compares sets of unique, normalized entity texts per label
    - Ignores positions/offsets
    - Better for KG bootstrapping where entity identity matters more than mention count

    Args:
        predictions: List of prediction records
        gold_by_episode: Dict mapping episode_id -> gold entities dict
        metadata_map: Optional mapping of episode_id -> metadata dict
            (unused, kept for API consistency)

    Returns:
        Metrics dict with entity-set TP, FP, FN, precision, recall, F1, and per-label metrics
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Per-label metrics
    label_metrics: Dict[str, Dict[str, int]] = {}

    episodes_scored = 0
    episodes_total = len(predictions)

    for pred_record in predictions:
        episode_id = pred_record.get("episode_id")
        if not episode_id:
            continue

        # Get gold entities for this episode
        gold_data = gold_by_episode.get(episode_id)
        if not gold_data:
            logger.debug(f"No gold reference for episode {episode_id}, skipping")
            continue

        episodes_scored += 1

        # Extract predicted entities
        pred_output = pred_record.get("output", {})
        pred_entities = pred_output.get("entities", [])

        # Extract gold entities
        gold_entities = gold_data.get("entities", [])

        # Step A: Derive scope sets from gold entities
        gold_scope = derive_gold_scope(gold_entities)

        # Step B: Filter predictions - only keep those in gold scope
        scoped_pred_entities = []
        for pred_ent in pred_entities:
            if is_in_gold_scope(pred_ent, gold_scope):
                scoped_pred_entities.append(pred_ent)

        # Step C: Build entity sets (normalized, deduplicated)
        # Gold sets: unique normalized texts per label
        gold_sets: Dict[str, Set[str]] = {}
        for gold_ent in gold_entities:
            label = gold_ent.get("label", "")
            text = normalize_entity_text(gold_ent.get("text", ""))
            if not text:
                continue
            if label not in gold_sets:
                gold_sets[label] = set()
            gold_sets[label].add(text)

        # Pred sets: unique normalized texts per label (from scoped predictions only)
        pred_sets: Dict[str, Set[str]] = {}
        for pred_ent in scoped_pred_entities:
            label = pred_ent.get("label", "")
            text = normalize_entity_text(pred_ent.get("text", ""))
            if not text:
                continue
            if label not in pred_sets:
                pred_sets[label] = set()
            pred_sets[label].add(text)

        # Step D: Compare sets per label
        all_labels = set(gold_sets.keys()) | set(pred_sets.keys())
        for label in all_labels:
            gold_set = gold_sets.get(label, set())
            pred_set = pred_sets.get(label, set())

            # TP = intersection (entities found in both)
            tp = len(gold_set & pred_set)
            # FP = pred - gold (entities predicted but not in gold)
            fp = len(pred_set - gold_set)
            # FN = gold - pred (entities in gold but not predicted)
            fn = len(gold_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Per-label metrics
            if label not in label_metrics:
                label_metrics[label] = {"tp": 0, "fp": 0, "fn": 0}
            label_metrics[label]["tp"] += tp
            label_metrics[label]["fp"] += fp
            label_metrics[label]["fn"] += fn

    # Compute overall precision, recall, F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute per-label F1
    per_label_f1: Dict[str, float] = {}
    for label, counts in label_metrics.items():
        label_tp = counts["tp"]
        label_fp = counts["fp"]
        label_fn = counts["fn"]
        label_precision = label_tp / (label_tp + label_fp) if (label_tp + label_fp) > 0 else 0.0
        label_recall = label_tp / (label_tp + label_fn) if (label_tp + label_fn) > 0 else 0.0
        label_f1 = (
            2 * label_precision * label_recall / (label_precision + label_recall)
            if (label_precision + label_recall) > 0
            else 0.0
        )
        per_label_f1[label] = label_f1

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_label_f1": per_label_f1,
        "episodes_scored": episodes_scored,
        "episodes_total": episodes_total,
    }


def compute_ner_vs_reference_metrics(
    predictions: List[Dict[str, Any]],
    reference_id: str,
    reference_path: Path,
    scoring_params: Optional[Dict[str, Any]] = None,
    dataset_id: Optional[str] = None,
    metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute NER metrics against a gold reference with scope-aware filtering.

    Args:
        predictions: List of prediction records
        reference_id: Reference identifier
        reference_path: Path to reference directory or file
        scoring_params: Optional scoring parameters (e.g., {"match": ["exact", "overlap"]})
        dataset_id: Dataset ID for text fingerprint verification (required for gold references)
        metadata_map: Optional mapping of episode_id -> metadata dict (for scope filtering)

    Returns:
        Metrics dict with match type results
    """
    # Load gold reference (validates index.json and episode JSONs)
    gold_by_episode = load_gold_reference(reference_path, dataset_id=dataset_id)

    if not gold_by_episode:
        logger.warning(f"No gold reference data found at {reference_path}")
        return {"error": "No gold reference data found"}

    # Get match types from scoring params (default to ["exact", "overlap"])
    # Support both "match" (legacy) and "mode" (new) params
    match_types = ["exact", "overlap"]
    if scoring_params:
        if "mode" in scoring_params:
            match_types = scoring_params["mode"]
        elif "match" in scoring_params:
            match_types = scoring_params["match"]

    # Compute metrics for each match type
    results: Dict[str, Any] = {}

    for match_type in match_types:
        if match_type == "entity_set":
            # Entity-set scoring (position-agnostic, text-based)
            results["entity_set"] = compute_entity_set_metrics(
                predictions, gold_by_episode, metadata_map=metadata_map
            )
        elif match_type in ("exact", "overlap", "mention_exact", "mention_overlap"):
            # Mention-level scoring (position-based)
            # Map "mention_exact" -> "exact", "mention_overlap" -> "overlap"
            actual_match_type = match_type.replace("mention_", "")
            results[match_type] = compute_ner_metrics_for_match_type(
                predictions, gold_by_episode, actual_match_type, metadata_map=metadata_map
            )
        else:
            logger.warning(f"Unknown match type: {match_type}, skipping")
            continue

    # Add metadata and diagnostics
    results["reference_id"] = reference_id
    total_predicted = sum(len(pred.get("output", {}).get("entities", [])) for pred in predictions)
    results["entities_predicted"] = total_predicted
    results["entities_gold"] = sum(
        len(gold.get("entities", [])) for gold in gold_by_episode.values()
    )

    # Compute ignored predictions totals (diagnostics)
    total_scoped = 0
    ignored_by_label: Dict[str, int] = {}
    for pred_record in predictions:
        episode_id = pred_record.get("episode_id")
        if not episode_id:
            continue
        gold_data = gold_by_episode.get(episode_id)
        if not gold_data:
            continue

        gold_entities = gold_data.get("entities", [])
        gold_scope = derive_gold_scope(gold_entities)
        pred_entities = pred_record.get("output", {}).get("entities", [])

        for pred_ent in pred_entities:
            if is_in_gold_scope(pred_ent, gold_scope):
                total_scoped += 1
            else:
                label = pred_ent.get("label", "UNKNOWN")
                ignored_by_label[label] = ignored_by_label.get(label, 0) + 1

    results["ignored_predictions_total"] = total_predicted - total_scoped
    results["ignored_predictions_by_label"] = ignored_by_label

    return results
