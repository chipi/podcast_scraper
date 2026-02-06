#!/usr/bin/env python3
"""Experiment runner with contract enforcement (RFC-015).

This script:
- Loads an ExperimentConfig from YAML
- Enforces contract requirements (dataset_id, baseline_id, golden_required)
- Validates baseline exists and matches dataset_id
- Uses factory API with params to create providers (supports OpenAI and hf_local backends)
- Processes episodes and generates structured outputs with fingerprints
- Writes predictions and metadata to data/eval/runs/<run_id>/

Evaluation (ROUGE, etc.) is handled by existing eval scripts that consume predictions.jsonl.

Usage:
    python scripts/run_experiment.py experiments/my_experiment.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path to import podcast_scraper modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from podcast_scraper.evaluation.comparator import compare_vs_baseline
from podcast_scraper.evaluation.experiment_config import (
    discover_input_files,
    episode_id_from_path,
    ExperimentConfig,
    load_experiment_config,
)
from podcast_scraper.evaluation.regression import RegressionChecker
from podcast_scraper.evaluation.reporter import (
    generate_comparison_report,
    generate_metrics_report,
    print_report,
    save_report,
)
from podcast_scraper.evaluation.scorer import score_run
from podcast_scraper.providers.params import SummarizationParams
from podcast_scraper.summarization.factory import create_summarization_provider

logger = logging.getLogger(__name__)

# Import enhanced fingerprint generation
from scripts.eval.materialize_baseline import (
    generate_enhanced_fingerprint,
    get_git_status,
)

logger = logging.getLogger(__name__)


def ensure_experiment_models_cached(map_model: str, reduce_model: str) -> None:
    """Ensure required Transformers models are cached for experiment, downloading if needed.

    This function checks if the MAP and REDUCE models specified in the experiment
    config are cached, and if not, downloads them using the centralized preload
    script logic. This is the ONLY place where models can be downloaded during
    experiments - libraries are never allowed to download on their own.

    Args:
        map_model: MAP model identifier (may be alias like "bart-small" or full ID)
        reduce_model: REDUCE model identifier (may be alias like "long-fast" or full ID)

    Note:
        This only downloads models if they're not already cached. It uses the same
        preload logic as the preload script, ensuring all downloads go through our
        centralized mechanism. Models are then loaded with local_files_only=True
        to prevent libraries from attempting their own downloads.
    """
    try:
        from podcast_scraper.cache import get_transformers_cache_dir
        from podcast_scraper.providers.ml import summarizer
        from podcast_scraper.providers.ml.model_loader import (
            preload_transformers_models,
        )

        transformers_cache = get_transformers_cache_dir()
        models_to_download = []

        # Resolve MAP model alias to actual model ID
        resolved_map = summarizer.DEFAULT_SUMMARY_MODELS.get(map_model, map_model)
        model_cache_name = resolved_map.replace("/", "--")
        model_cache_path = transformers_cache / f"models--{model_cache_name}"
        if not model_cache_path.exists():
            models_to_download.append(resolved_map)
            logger.info(f"MAP model {map_model} ({resolved_map}) not cached, will download")

        # Resolve REDUCE model alias to actual model ID
        resolved_reduce = summarizer.DEFAULT_SUMMARY_MODELS.get(reduce_model, reduce_model)
        # Only check REDUCE if it's different from MAP
        if resolved_reduce != resolved_map:
            reduce_cache_name = resolved_reduce.replace("/", "--")
            reduce_cache_path = transformers_cache / f"models--{reduce_cache_name}"
            if not reduce_cache_path.exists():
                models_to_download.append(resolved_reduce)
                logger.info(
                    f"REDUCE model {reduce_model} ({resolved_reduce}) not cached, " "will download"
                )

        # Download missing models using centralized preload functions
        if models_to_download:
            logger.info(
                f"Downloading {len(models_to_download)} missing model(s) "
                "(this may take a few minutes)..."
            )
            try:
                preload_transformers_models(models_to_download)
                logger.info("Missing models downloaded and cached successfully")
            except Exception as e:
                logger.warning(
                    f"Could not automatically download models: {e}. "
                    "You may need to run 'make preload-ml-models' manually."
                )
                # Don't fail - let the normal loading process handle the error
        else:
            logger.debug("All required models are already cached")

    except ImportError:
        # Preload script not available - that's okay, we'll try to load anyway
        logger.debug("Model loader module not available, skipping cache check")
    except Exception as e:
        # Don't fail on cache check errors - let normal loading handle it
        logger.debug(f"Error checking model cache: {e}")


def hash_text(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_run_readme(
    run_path: Path,
    run_id: str,
    dataset_id: str,
    experiment_config_path: Optional[str] = None,
    dry_run: bool = False,
    score_only: bool = False,
) -> None:
    """Create README.md for a run directory using template.

    Args:
        run_path: Path to run directory
        run_id: Run identifier
        dataset_id: Dataset identifier
        experiment_config_path: Optional path to experiment config file
        dry_run: Whether this was a dry-run (predictions only, no metrics)
        score_only: Whether this was score-only mode (scoring only, no inference)
    """
    # Load template
    template_path = Path("data/eval/runs/RUN_README_TEMPLATE.md")
    if not template_path.exists():
        logger.warning(
            f"Run README template not found: {template_path}, skipping README generation"
        )
        return

    template_content = template_path.read_text(encoding="utf-8")

    # Determine config path (relative if possible)
    config_path_str = "N/A"
    if experiment_config_path:
        try:
            config_path = Path(experiment_config_path)
            if config_path.exists():
                try:
                    config_path_str = str(config_path.relative_to(Path.cwd()))
                except ValueError:
                    config_path_str = str(config_path)
        except Exception:
            config_path_str = str(experiment_config_path)

    # Build mode description
    if score_only:
        mode_description = "Score-only (scoring and comparison only, inference skipped)"
    elif dry_run:
        mode_description = "Dry-run (predictions only, metrics/comparison skipped)"
    else:
        mode_description = "Full experiment (with metrics and comparison)"

    # Render template
    content = template_content.format(
        run_id=run_id,
        dataset_id=dataset_id,
        config_path=config_path_str,
        mode_description=mode_description,
    )

    readme_path = run_path / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    logger.info(f"Created README.md: {readme_path}")


def find_reference_path(reference_id: str, dataset_id: str) -> Path:
    """Find reference path (checks baselines, references, and gold directories).

    Args:
        reference_id: Reference identifier
        dataset_id: Dataset identifier

    Returns:
        Path to reference directory

    Raises:
        FileNotFoundError: If reference not found
    """
    # Check silver references directory (references/silver/{reference_id})
    silver_path = Path("data/eval/references/silver") / reference_id
    if silver_path.exists():
        return silver_path

    # Check gold references directory (references/gold/{task}/{reference_id})
    # For NER tasks, check ner_entities subdirectory
    gold_ner_path = Path("data/eval/references/gold/ner_entities") / reference_id
    if gold_ner_path.exists():
        return gold_ner_path

    # Check gold summarization references
    gold_summarization_path = Path("data/eval/references/gold/summarization") / reference_id
    if gold_summarization_path.exists():
        return gold_summarization_path

    # Check old structure (references/{dataset_id}/{reference_id}) for backward compatibility
    ref_path = Path("data/eval/references") / dataset_id / reference_id
    if ref_path.exists():
        logger.warning(
            f"Found reference in old location: {ref_path}. "
            "Consider migrating to new structure: references/silver/{reference_id}/"
        )
        return ref_path

    # Check baselines directory (baselines/{reference_id})
    baseline_path = Path("data/eval/baselines") / reference_id
    if baseline_path.exists():
        return baseline_path

    # Check old benchmarks directory (fallback for older project structure)
    old_baseline_path = Path("benchmarks/baselines") / reference_id
    if old_baseline_path.exists():
        return old_baseline_path

    raise FileNotFoundError(
        f"Reference '{reference_id}' not found. "
        f"Checked: data/eval/references/{dataset_id}/{reference_id}, "
        f"data/eval/references/gold/ner_entities/{reference_id}, "
        f"data/eval/references/gold/{reference_id}, "
        f"data/eval/baselines/{reference_id}, benchmarks/baselines/{reference_id}"
    )


def validate_reference(reference_id: str, reference_path: Path, dataset_id: str) -> None:
    """Validate that a reference has required files.

    Args:
        reference_id: Reference identifier
        reference_path: Path to reference directory
        dataset_id: Dataset identifier

    Raises:
        ValueError: If reference is invalid (missing required files)
    """
    if not reference_path.exists():
        raise ValueError(f"Reference path does not exist: {reference_path}")

    # Check if this is a gold reference (has episode JSON files) or a
    # baseline/silver reference (has predictions.jsonl)
    predictions_path = reference_path / "predictions.jsonl"
    has_episode_jsons = any(reference_path.glob("*.json")) and not predictions_path.exists()
    has_predictions = predictions_path.exists()

    if not (has_episode_jsons or has_predictions):
        raise ValueError(
            f"Reference '{reference_id}' is missing required files. "
            f"Expected either predictions.jsonl (for baselines/silver) or "
            f"episode JSON files (for gold references). "
            f"Found at: {reference_path}"
        )


def run_experiment(  # noqa: C901
    cfg: ExperimentConfig,
    baseline_id: Optional[str] = None,
    reference_ids: Optional[List[str]] = None,
    dry_run: bool = False,
    smoke_inference_only: bool = False,
    score_only: bool = False,
    force: bool = False,
) -> None:
    """Run a single experiment described by ExperimentConfig.

    This implements the complete evaluation loop:
    1. Runner produces outputs (predictions + fingerprint + run metadata)
        - skipped if score_only=True
    2. Scorer computes metrics (gates, stability, cost/latency,
        and optionally "vs reference" metrics)
    3. Comparator computes deltas vs baseline

    Args:
        cfg: Experiment configuration
        baseline_id: Optional baseline ID for comparison
        reference_ids: Optional list of reference IDs for evaluation
        dry_run: If True, skip metrics and comparison (inference only)
        smoke_inference_only: If True, skip metrics and comparison
            (inference only, alias for dry_run)
        score_only: If True, skip inference and use existing
            predictions.jsonl (scoring only)
        force: If True, delete existing run directory before starting.
            Useful for re-running experiments during development.

    Raises:
        ValueError: If contract validation fails
        RuntimeError: If experiment execution fails
    """
    # Get dataset_id from config
    dataset_id = cfg.data.dataset_id
    if not dataset_id:
        raise ValueError("dataset_id is REQUIRED in data config")

    # Validate references if provided
    reference_paths = {}
    if reference_ids:
        for ref_id in reference_ids:
            ref_path = find_reference_path(ref_id, dataset_id)
            validate_reference(ref_id, ref_path, dataset_id)
            reference_paths[ref_id] = ref_path

    run_id = cfg.id
    # Store runs in data/eval/runs/ for consistency with baseline/reference structure
    results_dir = Path("data/eval/runs") / run_id

    # Handle existing directory
    if results_dir.exists():
        if force:
            logger.info(f"Removing existing run directory (--force): {results_dir}")
            shutil.rmtree(results_dir)
        elif not score_only:
            # In score_only mode, we expect the directory to exist
            logger.warning(
                f"Run directory already exists: {results_dir}. "
                "Use --force to delete and re-run, or --score-only to re-score existing results."
            )

    results_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = results_dir / "predictions.jsonl"
    baseline_json_path = results_dir / "baseline.json"  # For compatibility with baseline structure
    fingerprint_path = results_dir / "fingerprint.json"
    log_path = results_dir / "run.log"

    # Set up file logging to capture execution log
    # This complements fingerprint.json by recording what actually happened
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Capture all levels to file
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    # Add file handler to root logger (captures all module logs)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info(f"Starting experiment run: {run_id}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Execution log: {log_path}")
    logger.info(f"Dataset: {dataset_id}")
    logger.info("=" * 80)

    # If score_only mode, verify predictions.jsonl exists and skip inference
    if score_only:
        if not predictions_path.exists():
            raise FileNotFoundError(
                f"Score-only mode requires existing predictions.jsonl "
                f"at {predictions_path}\n"
                f"  Run the experiment normally first to generate predictions, "
                f"or use a different run_id."
            )
        logger.info(f"Score-only mode: Using existing predictions from {predictions_path}")
        logger.info("Skipping inference phase. Proceeding directly to scoring...")
    else:
        # Support both summarization and ner_entities tasks
        if cfg.task not in ("summarization", "ner_entities"):
            raise NotImplementedError(
                f"Only 'summarization' and 'ner_entities' tasks are supported. Got: {cfg.task}"
            )

        # Create provider based on task and backend type
        if cfg.task == "summarization":
            logger.info("Creating summarization provider...")
        if cfg.backend.type == "openai":
            # OpenAI backend: use params from config if provided
            # Extract params from config.params (if present)
            params_dict = cfg.params or {}
            max_length = params_dict.get("max_length", 800)
            min_length = params_dict.get("min_length", 200)
            temperature = params_dict.get("temperature", 0.0)
            summarization_params = SummarizationParams(
                model_name=cfg.backend.model,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
            )
            provider = create_summarization_provider("openai", summarization_params)
            provider.initialize()
            model_name = cfg.backend.model
            device = None
            # Create params dict for metadata
            params_dict = summarization_params.model_dump()
        elif cfg.backend.type == "hf_local":
            # Use HF models from experiment config
            from podcast_scraper import config

            # Build model name string for fingerprint
            map_model = cfg.backend.map_model or "bart-small"
            reduce_model = cfg.backend.reduce_model or "long-fast"
            model_name = f"{map_model}+{reduce_model}"

            # Ensure models are cached before initializing provider
            # This checks cache and downloads if needed, same as app workflow
            logger.info("Checking if required models are cached...")
            ensure_experiment_models_cached(map_model, reduce_model)

            # Use explicit map_params/reduce_params/tokenize (required for hf_local)
            map_max_length = cfg.map_params.max_new_tokens
            map_min_length = cfg.map_params.min_new_tokens
            reduce_max_length = cfg.reduce_params.max_new_tokens
            reduce_min_length = cfg.reduce_params.min_new_tokens

            # Build params dict for fingerprint (includes all resolved params)
            params_dict = {
                "map_model": map_model,
                "reduce_model": reduce_model,
                "map_params": cfg.map_params.model_dump(),
                "reduce_params": cfg.reduce_params.model_dump(),
                "tokenize": cfg.tokenize.model_dump(),
            }

            cfg_obj = config.Config(
                rss_url="",  # Not used for experiment
                summary_provider="transformers",
                summary_model=map_model,
                summary_reduce_model=reduce_model,
                generate_summaries=True,
                generate_metadata=True,  # Required when generate_summaries=True
                summary_map_params={
                    "max_new_tokens": map_max_length,
                    "min_new_tokens": map_min_length,
                    "num_beams": cfg.map_params.num_beams,
                    "no_repeat_ngram_size": cfg.map_params.no_repeat_ngram_size,
                    "length_penalty": cfg.map_params.length_penalty,
                    "early_stopping": cfg.map_params.early_stopping,
                    "repetition_penalty": cfg.map_params.repetition_penalty,
                },
                summary_reduce_params={
                    "max_new_tokens": reduce_max_length,
                    "min_new_tokens": reduce_min_length,
                    "num_beams": cfg.reduce_params.num_beams,
                    "no_repeat_ngram_size": cfg.reduce_params.no_repeat_ngram_size,
                    "length_penalty": cfg.reduce_params.length_penalty,
                    "early_stopping": cfg.reduce_params.early_stopping,
                    "repetition_penalty": cfg.reduce_params.repetition_penalty,
                },
                summary_tokenize={
                    "map_max_input_tokens": cfg.tokenize.map_max_input_tokens,
                    "reduce_max_input_tokens": cfg.tokenize.reduce_max_input_tokens,
                    "truncation": cfg.tokenize.truncation,
                },
                summary_word_chunk_size=(cfg.chunking.word_chunk_size if cfg.chunking else None),
                summary_word_overlap=(cfg.chunking.word_overlap if cfg.chunking else None),
                transcribe_missing=False,  # Don't initialize Whisper
                # for summarization-only experiments
            )
            provider = create_summarization_provider(cfg_obj)

            # Initialize provider (this will load models and log model loading)
            provider.initialize()

            # Log generation parameters after models are loaded (appears in first 20 lines)
            logger.info("=" * 80)
            logger.info("=== Generation Parameters ===")
            logger.info("Map stage:")
            logger.info(f"  Model: {map_model}")
            logger.info(f"  max_new_tokens: {cfg.map_params.max_new_tokens}")
            logger.info(f"  min_new_tokens: {cfg.map_params.min_new_tokens}")
            logger.info(f"  num_beams: {cfg.map_params.num_beams}")
            logger.info(f"  no_repeat_ngram_size: {cfg.map_params.no_repeat_ngram_size}")
            logger.info(f"  length_penalty: {cfg.map_params.length_penalty}")
            logger.info(f"  early_stopping: {cfg.map_params.early_stopping}")
            logger.info(f"  repetition_penalty: {cfg.map_params.repetition_penalty}")
            logger.info("Reduce stage:")
            logger.info(f"  Model: {reduce_model}")
            logger.info(f"  max_new_tokens: {cfg.reduce_params.max_new_tokens}")
            logger.info(f"  min_new_tokens: {cfg.reduce_params.min_new_tokens}")
            logger.info(f"  num_beams: {cfg.reduce_params.num_beams}")
            logger.info(f"  no_repeat_ngram_size: {cfg.reduce_params.no_repeat_ngram_size}")
            logger.info(f"  length_penalty: {cfg.reduce_params.length_penalty}")
            logger.info(f"  early_stopping: {cfg.reduce_params.early_stopping}")
            logger.info(f"  repetition_penalty: {cfg.reduce_params.repetition_penalty}")
            logger.info("Tokenization:")
            logger.info(f"  map_max_input_tokens: {cfg.tokenize.map_max_input_tokens}")
            logger.info(f"  reduce_max_input_tokens: {cfg.tokenize.reduce_max_input_tokens}")
            logger.info(f"  truncation: {cfg.tokenize.truncation}")
            if cfg.chunking:
                logger.info("Chunking:")
                logger.info(f"  strategy: {cfg.chunking.strategy}")
                logger.info(f"  word_chunk_size: {cfg.chunking.word_chunk_size}")
                logger.info(f"  word_overlap: {cfg.chunking.word_overlap}")
            logger.info("Preprocessing:")
            logger.info(f"  profile: {cfg.preprocessing_profile}")
            logger.info("=" * 80)
            # Get device from provider if available
            device = None
            map_model_obj = getattr(provider, "_map_model", None)
            if map_model_obj:
                device = getattr(map_model_obj, "device", None)
                if device:
                    device = str(device)
        elif cfg.task == "ner_entities":
            # NER task: load spaCy model
            logger.info("Creating NER provider...")
            if cfg.backend.type != "spacy_local":
                raise ValueError(
                    f"NER task requires 'spacy_local' backend. Got: {cfg.backend.type}"
                )

            # Load spaCy model
            from podcast_scraper.providers.ml.speaker_detection import _load_spacy_model

            model_name = cfg.backend.model
            logger.info(f"Loading spaCy model: {model_name}")
            nlp = _load_spacy_model(model_name)
            if nlp is None:
                raise RuntimeError(
                    f"Failed to load spaCy model: {model_name}. "
                    f"Install with: python -m spacy download {model_name}"
                )
            logger.info(f"✓ spaCy model loaded: {model_name}")

            # Store nlp model for later use
            provider = None  # NER doesn't use provider pattern
            device = None  # spaCy handles device internally
            params_dict = {
                "model": model_name,
                "labels": cfg.params.get("labels") if cfg.params else None,
            }
        else:
            raise ValueError(f"Unsupported backend type: {cfg.backend.type}")

        # Generate enhanced fingerprint with pipeline structure
        # Get git info
        git_info = get_git_status()

        # Get preprocessing profile from config (defaults to cleaning_v3)
        preprocessing_profile = cfg.preprocessing_profile

        # Generate enhanced fingerprint with pipeline structure
        # For NER tasks, provider is None, so we pass model_name directly
        fingerprint = generate_enhanced_fingerprint(
            baseline_id=run_id,  # Use run_id as baseline_id for compatibility
            dataset_id=dataset_id,
            experiment_config=cfg,
            provider=provider,
            model_name=model_name if cfg.task == "summarization" else cfg.backend.model,
            preprocessing_profile=preprocessing_profile,
            git_info=git_info,
        )

        # Discover input files
        input_files = discover_input_files(cfg.data)
        if not input_files:
            if cfg.data.dataset_id:
                raise RuntimeError(f"No input files found for dataset: {cfg.data.dataset_id}")
            else:
                raise RuntimeError(f"No input files found for glob: {cfg.data.episodes_glob}")

        logger.info(f"Found {len(input_files)} episode(s) to process")

        # Phase 1: Run inference and generate predictions
        logger.info("Phase 1: Running inference...")

        # Process episodes
        start_time = time.time()
        predictions = []
        total_chars_in = 0
        total_chars_out = 0

        with open(predictions_path, "w", encoding="utf-8") as pred_f:
            for path in input_files:
                episode_id = episode_id_from_path(path, cfg.data)
                logger.info(f"Processing episode: {episode_id}")

                # Read transcript
                text = path.read_text(encoding="utf-8").strip()
                if not text:
                    logger.warning(f"Skipping empty transcript: {path}")
                    continue

                # Apply preprocessing profile (for both summarization and NER tasks)
                preprocessing_profile = cfg.preprocessing_profile
                if preprocessing_profile:
                    from podcast_scraper.preprocessing.profiles import apply_profile_with_stats

                    cleaned_text, preprocess_stats = apply_profile_with_stats(
                        text, preprocessing_profile
                    )
                    if cleaned_text != text:
                        removed_chars = len(text) - len(cleaned_text)
                        removed_pct = (removed_chars / len(text) * 100) if len(text) else 0
                        logger.info(
                            f"[PREPROCESSING] Profile: {preprocessing_profile}, "
                            f"lines: {preprocess_stats['initial_lines']} → "
                            f"{preprocess_stats['final_lines']} "
                            f"({preprocess_stats['lines_removed']} removed), "
                            f"chars: {len(text):,} → {len(cleaned_text):,} "
                            f"({removed_chars:,} removed, {removed_pct:.1f}%)"
                        )
                        text = cleaned_text.strip()
                    else:
                        logger.info(
                            f"[PREPROCESSING] Profile: {preprocessing_profile}, "
                            f"lines: {preprocess_stats['initial_lines']} (no changes)"
                        )

                total_chars_in += len(text)
                input_hash = hash_text(text)

                # Process based on task type
                t0 = time.time()
                try:
                    if cfg.task == "summarization":
                        # Build summary params based on config structure
                        if cfg.backend.type == "hf_local" and cfg.map_params and cfg.reduce_params:
                            # New structure: pass explicit generation params
                            summary_params = {
                                "return_intermediates": True,
                                # Map stage params
                                "map_max_new_tokens": cfg.map_params.max_new_tokens,
                                "map_min_new_tokens": cfg.map_params.min_new_tokens,
                                "map_num_beams": cfg.map_params.num_beams,
                                "map_no_repeat_ngram_size": cfg.map_params.no_repeat_ngram_size,
                                "map_length_penalty": cfg.map_params.length_penalty,
                                "map_early_stopping": cfg.map_params.early_stopping,
                                "map_repetition_penalty": cfg.map_params.repetition_penalty,
                                "map_encoder_no_repeat_ngram_size": (
                                    cfg.map_params.encoder_no_repeat_ngram_size
                                ),
                                # Reduce stage params
                                "reduce_max_new_tokens": cfg.reduce_params.max_new_tokens,
                                "reduce_min_new_tokens": cfg.reduce_params.min_new_tokens,
                                "reduce_num_beams": cfg.reduce_params.num_beams,
                                "reduce_no_repeat_ngram_size": (
                                    cfg.reduce_params.no_repeat_ngram_size
                                ),
                                "reduce_length_penalty": cfg.reduce_params.length_penalty,
                                "reduce_early_stopping": cfg.reduce_params.early_stopping,
                                "reduce_repetition_penalty": cfg.reduce_params.repetition_penalty,
                                "reduce_encoder_no_repeat_ngram_size": (
                                    cfg.reduce_params.encoder_no_repeat_ngram_size
                                ),
                                # Tokenization params
                                "map_max_input_tokens": cfg.tokenize.map_max_input_tokens,
                                "reduce_max_input_tokens": cfg.tokenize.reduce_max_input_tokens,
                                "truncation": cfg.tokenize.truncation,
                                # Chunking params (if specified)
                                "use_word_chunking": (
                                    cfg.chunking.strategy == "word_chunking"
                                    if cfg.chunking
                                    else None
                                ),
                                "word_chunk_size": (
                                    cfg.chunking.word_chunk_size if cfg.chunking else None
                                ),
                                "word_overlap": (
                                    cfg.chunking.word_overlap if cfg.chunking else None
                                ),
                                # Preprocessing profile
                                "preprocessing_profile": cfg.preprocessing_profile,
                            }
                        else:
                            # OpenAI backend: no special params needed
                            summary_params = {
                                "preprocessing_profile": cfg.preprocessing_profile,
                            }
                        summary_result = provider.summarize(text, params=summary_params)
                        if isinstance(summary_result, dict):
                            summary = summary_result.get("summary", "")
                            intermediates = summary_result.get("intermediates")
                        else:
                            summary = str(summary_result)
                            intermediates = None

                        dt = time.time() - t0
                        total_chars_out += len(summary)
                        output_hash = hash_text(summary)

                        # Log map/reduce input sizes for diagnostics
                        if intermediates and "map_summaries" in intermediates:
                            map_summaries = intermediates["map_summaries"]
                            map_chunks_count = len(map_summaries)

                            # Calculate average map summary length
                            if map_summaries:
                                map_summary_texts = [
                                    item["text"] if isinstance(item, dict) else str(item)
                                    for item in map_summaries
                                ]
                                map_summary_lengths = [len(s) for s in map_summary_texts]
                                avg_map_summary_chars = sum(map_summary_lengths) / len(
                                    map_summary_lengths
                                )
                            else:
                                avg_map_summary_chars = 0

                            # Get reduce input size from intermediates
                            # (calculated in summarize_long_text)
                            reduce_input_chars = intermediates.get("reduce_input_chars", 0)

                            logger.info(
                                f"Episode {episode_id} map/reduce stats: "
                                f"map_chunks={map_chunks_count}, "
                                f"avg_map_summary={avg_map_summary_chars:.0f} chars, "
                                f"reduce_input={reduce_input_chars:,} chars"
                            )

                        logger.info(
                            f"Episode {episode_id} completed in {dt:.1f}s, "
                            f"summary length={len(summary)} chars"
                        )

                        # Write prediction record for summarization
                        record: Dict[str, Any] = {
                            "episode_id": episode_id,
                            "dataset_id": dataset_id,
                            "output": {
                                "summary_final": summary,
                            },
                            "fingerprint_ref": "fingerprint.json",
                            "metadata": {
                                "input_hash": f"sha256:{input_hash}",
                                "output_hash": f"sha256:{output_hash}",
                                "input_path": str(path),
                                "input_length_chars": len(text),
                                "output_length_chars": len(summary),
                                "processing_time_seconds": dt,
                            },
                        }

                        # Include intermediate outputs if available (for map/reduce pipelines)
                        if intermediates:
                            record["intermediate"] = intermediates

                    elif cfg.task == "ner_entities":
                        # Extract entities using spaCy
                        from podcast_scraper.providers.ml.ner_extraction import extract_all_entities

                        # Get labels filter from params if specified
                        labels = None
                        if cfg.params and "labels" in cfg.params:
                            labels = cfg.params["labels"]

                        entities = extract_all_entities(text, nlp, labels=labels)

                        dt = time.time() - t0
                        total_chars_out += len(str(entities))  # For compression stats
                        output_hash = hash_text(str(entities))

                        logger.info(
                            f"Episode {episode_id} completed in {dt:.1f}s, "
                            f"entities found={len(entities)}"
                        )

                        # Write prediction record for NER
                        record: Dict[str, Any] = {
                            "episode_id": episode_id,
                            "dataset_id": dataset_id,
                            "output": {
                                "entities": entities,
                            },
                            "fingerprint_ref": "fingerprint.json",
                            "metadata": {
                                "input_hash": f"sha256:{input_hash}",
                                "output_hash": f"sha256:{output_hash}",
                                "input_path": str(path),
                                "input_length_chars": len(text),
                                "output_length_chars": len(str(entities)),
                                "processing_time_seconds": dt,
                                "entities_count": len(entities),
                            },
                        }
                    else:
                        raise ValueError(f"Unsupported task: {cfg.task}")

                except Exception as e:
                    logger.error(f"Failed to process episode {episode_id}: {e}")
                    raise RuntimeError(f"Processing failed for episode {episode_id}: {e}") from e

                pred_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                predictions.append(record)

        total_time = time.time() - start_time
        avg_time = total_time / len(predictions) if predictions else 0
        avg_compression = (total_chars_in / total_chars_out) if total_chars_out > 0 else None

        logger.info(f"Phase 1 completed in {total_time:.1f}s")
        logger.info(f"Total input: {total_chars_in:,} chars, output: {total_chars_out:,} chars")
        if cfg.task == "summarization":
            logger.info(
                f"Average compression: {avg_compression:.1f}x" if avg_compression else "N/A"
            )
        logger.info(f"Predictions: {predictions_path}")

        # Cleanup provider if it exists (NER tasks don't use provider pattern)
        if provider is not None:
            try:
                provider.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up provider: {e}")

        # Save run metadata (using baseline.json format for compatibility)
        import os
        from datetime import datetime

        baseline_metadata = {
            "run_id": run_id,  # Primary identifier for runs
            "dataset_id": dataset_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "created_by": os.getenv("USER", "unknown"),
            "fingerprint_ref": "fingerprint.json",
            "task": cfg.task,
            "backend": {
                "type": cfg.backend.type,
                **({"model": cfg.backend.model} if cfg.backend.type == "openai" else {}),
                **(
                    {
                        "map_model": cfg.backend.map_model,
                        "reduce_model": cfg.backend.reduce_model,
                    }
                    if cfg.backend.type == "hf_local"
                    else {}
                ),
            },
            "params": params_dict,
            "stats": {
                "num_episodes": len(predictions),
                "total_time_seconds": total_time,
                "avg_time_seconds": avg_time,
                "total_chars_in": total_chars_in,
                "total_chars_out": total_chars_out,
                "avg_compression": avg_compression,
            },
        }

        baseline_json_path.write_text(
            json.dumps(baseline_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Save fingerprint (now a dict from generate_enhanced_fingerprint)
        fingerprint_path.write_text(
            json.dumps(fingerprint, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(f"Metadata: {baseline_json_path}")
        logger.info(f"Fingerprint: {fingerprint_path}")

        # Create README.md for the run
        # Get config path from the config file that was loaded
        config_path = None
        if hasattr(cfg, "_config_path"):
            config_path = cfg._config_path
        elif hasattr(cfg, "id"):
            # Try to find config file by ID
            config_candidates = [
                Path(f"data/eval/configs/{cfg.id}.yaml"),
                Path(f"data/eval/configs/{cfg.id}.yml"),
            ]
            for candidate in config_candidates:
                if candidate.exists():
                    config_path = str(candidate)
                    break

        create_run_readme(
            run_path=results_dir,
            run_id=run_id,
            dataset_id=dataset_id,
            experiment_config_path=config_path,
            dry_run=dry_run or smoke_inference_only,
            score_only=False,  # Normal inference mode
        )

    # Score-only mode: Load existing predictions to get metadata
    if score_only:
        logger.info("Loading existing predictions for metadata...")
        from podcast_scraper.evaluation.scorer import load_predictions

        existing_predictions = load_predictions(predictions_path)
        logger.info(f"Loaded {len(existing_predictions)} existing prediction(s)")

        # Try to load existing baseline.json for metadata
        if baseline_json_path.exists():
            try:
                existing_metadata = json.loads(baseline_json_path.read_text(encoding="utf-8"))
                dataset_id = existing_metadata.get("dataset_id", dataset_id)
                logger.debug(f"Loaded dataset_id from existing baseline.json: {dataset_id}")
            except Exception as e:
                logger.warning(f"Failed to load existing baseline.json: {e}")

        # Create README for score-only mode
        config_path = None
        if hasattr(cfg, "_config_path"):
            config_path = cfg._config_path
        elif hasattr(cfg, "id"):
            config_candidates = [
                Path(f"data/eval/configs/{cfg.id}.yaml"),
                Path(f"data/eval/configs/{cfg.id}.yml"),
            ]
            for candidate in config_candidates:
                if candidate.exists():
                    config_path = str(candidate)
                    break

        create_run_readme(
            run_path=results_dir,
            run_id=run_id,
            dataset_id=dataset_id,
            experiment_config_path=config_path,
            dry_run=False,
            score_only=True,
        )

    # Phase 2: Score the run (compute metrics)
    # Skip scoring if dry-run or smoke-inference-only mode
    if dry_run or smoke_inference_only:
        logger.info("Skipping metrics computation (dry-run or smoke-inference-only mode)")
        logger.info("Run completed with predictions only. Use full experiment-run for metrics.")
        return

    # Note: score_only mode always runs scoring (that's the point)

    # Load metadata for speaker detection
    from podcast_scraper.evaluation.metadata_validator import validate_episode_metadata

    metadata_map: Dict[str, Dict[str, Any]] = {}
    for path in input_files:
        episode_id = episode_id_from_path(path, cfg.data)
        # Try to find metadata file in same directory as transcript
        metadata_candidates = [
            path.parent / f"{path.stem}.metadata.json",  # Flat format: p01_e01.metadata.json
            path.parent / f"{path.stem}.meta.json",  # Materialized format: p01_e01.meta.json
            path.parent / "metadata.json",  # Old format: metadata.json in directory
        ]
        for candidate in metadata_candidates:
            if candidate.exists():
                try:
                    metadata_content = json.loads(candidate.read_text(encoding="utf-8"))
                    # Validate metadata structure with assertions
                    validate_episode_metadata(metadata_content, episode_id)
                    metadata_map[episode_id] = metadata_content
                    logger.debug(f"Loaded metadata for {episode_id} from {candidate}")
                    break
                except AssertionError as e:
                    logger.error(
                        f"Metadata validation failed for {episode_id} from {candidate}: {e}"
                    )
                    # Continue to next candidate or skip if all fail
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {candidate}: {e}")
                    continue

    logger.info("Computing metrics...")

    # Extract scoring params from config if present (for NER tasks)
    scoring_params = None
    if cfg.params and "scoring" in cfg.params:
        scoring_params = cfg.params["scoring"]

    metrics = score_run(
        predictions_path=predictions_path,
        dataset_id=dataset_id,
        run_id=run_id,
        reference_paths=reference_paths if reference_paths else None,
        metadata_map=metadata_map if metadata_map else None,
        scoring_params=scoring_params,
    )

    # Save metrics.json
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Metrics: {metrics_path}")

    # Validate metrics.json against schema (lenient - warns on mismatch)
    try:
        from podcast_scraper.evaluation.schema_validator import (
            validate_metrics_ner,
            validate_metrics_summarization,
        )

        task_type = metrics.get("task")
        if task_type == "ner_entities":
            validate_metrics_ner(metrics, strict=False)  # Lenient for now
            logger.info("✓ NER metrics validation completed")
        elif task_type == "summarization":
            validate_metrics_summarization(metrics, strict=False)  # Lenient for now
            logger.info("✓ Summarization metrics validation completed")
    except ValueError as e:
        # Only raise if strict validation was requested (not the case here)
        logger.warning(f"Metrics validation issue (non-fatal): {e}")
    except Exception as e:
        logger.warning(f"Metrics validation skipped (schema validator unavailable): {e}")

    # Generate and save human-readable metrics report
    metrics_report = generate_metrics_report(metrics)
    metrics_report_path = results_dir / "metrics_report.md"
    save_report(metrics_report, metrics_report_path)
    print_report(metrics_report)

    # Phase 3: Compare vs baseline (if provided)
    if baseline_id:
        logger.info(f"Computing comparison vs baseline: {baseline_id}")
        baseline_path = find_reference_path(baseline_id, dataset_id)
        baseline_metrics_path = baseline_path / "metrics.json"

        if not baseline_metrics_path.exists():
            logger.warning(
                f"Baseline metrics not found at {baseline_metrics_path}. "
                "Skipping comparison. Baseline may need to be rescored."
            )
        else:
            try:
                comparison = compare_vs_baseline(
                    experiment_metrics_path=metrics_path,
                    baseline_metrics_path=baseline_metrics_path,
                    baseline_id=baseline_id,
                    dataset_id=dataset_id,
                )

                # Save comparison
                comparisons_dir = results_dir / "comparisons"
                comparisons_dir.mkdir(exist_ok=True)
                comparison_path = comparisons_dir / f"vs_{baseline_id}.json"
                comparison_path.write_text(
                    json.dumps(comparison, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.info(f"Comparison: {comparison_path}")

                # Check NER invariants (hard guardrails) before regression rules
                task_type = metrics.get("task")
                if task_type == "ner_entities":
                    checker = RegressionChecker()
                    # Check invariants for all references in vs_reference
                    vs_reference = metrics.get("vs_reference", {})
                    invariant_violations = []
                    for ref_id in vs_reference.keys():
                        violations = checker.check_ner_invariants(metrics, reference_id=ref_id)
                        invariant_violations.extend(violations)

                    if invariant_violations:
                        logger.error("❌ NER INVARIANT VIOLATIONS (hard guardrails):")
                        for violation in invariant_violations:
                            logger.error(f"  {violation}")
                        raise ValueError(
                            f"NER invariant violations detected: {', '.join(invariant_violations)}"
                        )
                    logger.info("✅ NER invariants check passed")

                # Check for regressions
                checker = RegressionChecker()
                regressions = checker.check_metrics(
                    experiment_metrics=metrics,
                    baseline_metrics=json.loads(baseline_metrics_path.read_text(encoding="utf-8")),
                )

                # Save regression report
                if regressions:
                    regression_report = {
                        "baseline_id": baseline_id,
                        "experiment_run_id": run_id,
                        "dataset_id": dataset_id,
                        "regressions": regressions,
                        "should_block_ci": checker.should_block_ci(regressions),
                    }
                    regression_path = comparisons_dir / f"regressions_vs_{baseline_id}.json"
                    regression_path.write_text(
                        json.dumps(regression_report, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    logger.warning(
                        f"⚠️  {len(regressions)} regression(s) detected: {regression_path}"
                    )

                    # Log regressions
                    for reg in regressions:
                        severity = reg.get("severity", "unknown").upper()
                        logger.warning(f"  [{severity}] {reg.get('message', 'Unknown regression')}")

                    if checker.should_block_ci(regressions):
                        logger.error("❌ CI BLOCKING: Error-level regressions detected!")
                else:
                    logger.info("✅ No regressions detected")

                # Generate and save human-readable comparison report
                comparison_report = generate_comparison_report(comparison)
                comparison_report_path = comparisons_dir / f"vs_{baseline_id}_report.md"
                save_report(comparison_report, comparison_report_path)
                print_report(comparison_report)

                # Log summary
                deltas = comparison.get("deltas", {})
                logger.info("Comparison summary:")
                for key, value in deltas.items():
                    if isinstance(value, list):
                        logger.info(f"  {key}: {value}")
                    elif value is not None:
                        sign = "+" if value >= 0 else ""
                        logger.info(f"  {key}: {sign}{value}")
            except Exception as e:
                logger.error(f"Failed to compute comparison vs baseline: {e}", exc_info=True)

    logger.info(
        f"Episodes={len(predictions)}, avg_time={avg_time:.1f}s, "
        f"avg_compression={avg_compression:.1f}x"
        if avg_compression
        else f"Episodes={len(predictions)}, avg_time={avg_time:.1f}s"
    )

    # Cleanup provider if it exists (NER tasks don't use provider pattern)
    if provider is not None:
        provider.cleanup()

    # Remove file handler to close log file cleanly
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path):
            handler.close()
            root_logger.removeHandler(handler)

    logger.info(f"✓ Execution log saved: {log_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Run an AI experiment with complete evaluation loop " "(runner + scorer + comparator)."
        ),
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment YAML config file.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        dest="baseline_id",
        help="Baseline ID for comparison (optional but recommended for experiments).",
    )
    parser.add_argument(
        "--reference",
        type=str,
        action="append",
        dest="reference_ids",
        help="Reference ID for evaluation (can be specified multiple times). "
        "References can be silver/gold and are used for ROUGE computation.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: generate predictions only, skip metrics and comparison. "
        "Useful for quick inference testing without full evaluation.",
    )
    parser.add_argument(
        "--smoke-inference-only",
        action="store_true",
        help="Smoke test mode: generate predictions only, skip metrics and comparison. "
        "Alias for --dry-run. Useful for quick inference testing.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help=(
            "Score-only mode: skip inference, use existing predictions.jsonl, "
            "run scoring and comparison. "
            "Useful for re-scoring existing runs or testing scoring logic "
            "without re-running inference."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force mode: delete existing run directory before starting. "
            "Useful for re-running experiments during iterative development. "
            "Without this flag, existing directories are reused (which may "
            "cause stale files to persist)."
        ),
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load config
    try:
        cfg = load_experiment_config(args.config)
        # Store config path for README generation
        cfg._config_path = args.config
    except Exception as e:
        logger.error(f"Failed to load experiment config: {e}")
        sys.exit(1)

    # Validate API key for OpenAI backend
    if cfg.backend.type == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            logger.error(
                "OPENAI_API_KEY environment variable not set. "
                "Export it before running this script."
            )
            sys.exit(1)

    # Run experiment
    try:
        run_experiment(
            cfg,
            baseline_id=args.baseline_id,
            reference_ids=args.reference_ids,
            dry_run=args.dry_run or args.smoke_inference_only,
            smoke_inference_only=args.smoke_inference_only,
            score_only=args.score_only,
            force=args.force,
        )
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
