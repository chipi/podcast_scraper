#!/usr/bin/env python3
"""Baseline materialization script (RFC-041 Phase 0).

This script:
- Creates baseline artifacts from current main branch
- Freezes dataset definitions
- Generates baseline metadata with fingerprints
- Ensures baselines are immutable and reviewable

Usage:
    python scripts/materialize_baseline.py --baseline-id bart_led_baseline_v1
        --dataset-id indicator_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path to import podcast_scraper modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from podcast_scraper.evaluation.experiment_config import ExperimentConfig, load_experiment_config
from podcast_scraper.prompts.store import get_prompt_metadata
from podcast_scraper.providers.params import SummarizationParams
from podcast_scraper.summarization.factory import create_summarization_provider

logger = logging.getLogger(__name__)


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def hash_text(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Metrics computation is now handled by scorer.py (RFC-016 Phase 3)
# Import the scorer module to use its unified metrics computation
from podcast_scraper.evaluation.scorer import score_run


def get_provider_library_info(provider) -> Dict[str, str]:
    """Get provider library information.

    Args:
        provider: Provider instance

    Returns:
        Dictionary with provider_library and provider_library_version
    """
    provider_type = provider.__class__.__name__
    if provider_type == "OpenAIProvider":
        try:
            import openai

            return {
                "provider_library": "openai",
                "provider_library_version": getattr(openai, "__version__", "unknown"),
            }
        except ImportError:
            return {"provider_library": "openai", "provider_library_version": "unknown"}
    elif provider_type == "MLProvider":
        try:
            import transformers

            return {
                "provider_library": "transformers",
                "provider_library_version": getattr(transformers, "__version__", "unknown"),
            }
        except ImportError:
            return {"provider_library": "transformers", "provider_library_version": "unknown"}
    else:
        return {"provider_library": "unknown", "provider_library_version": "unknown"}


def get_model_details(
    provider, model_name: str, experiment_config: Optional[ExperimentConfig]
) -> Dict[str, Any]:
    """Extract detailed model information from provider.

    Args:
        provider: Provider instance
        model_name: Model name string
        experiment_config: Optional experiment configuration

    Returns:
        Dictionary with model details (model_name, model_revision, weights_format, etc.)
    """
    model_info: Dict[str, Any] = {
        "model_name": model_name,
        "model_revision": None,
        "weights_format": None,
        "tokenizer_name": None,
        "tokenizer_revision": None,
    }

    provider_type = provider.__class__.__name__

    if provider_type == "MLProvider":
        # For HuggingFace models, try to extract details from the provider
        try:
            # MLProvider has map_model and reduce_model properties (SummaryModel instances)
            map_model_obj = getattr(provider, "map_model", None)
            reduce_model_obj = getattr(provider, "reduce_model", None)

            # Extract info for map_model (primary model)
            if map_model_obj:
                map_model_name = map_model_obj.model_name
                map_model_revision = getattr(map_model_obj, "revision", None) or "main"
                map_tokenizer_name = map_model_name
                if hasattr(map_model_obj, "tokenizer") and map_model_obj.tokenizer is not None:
                    tokenizer = map_model_obj.tokenizer
                    if hasattr(tokenizer, "name_or_path"):
                        map_tokenizer_name = tokenizer.name_or_path
                model_info["map_model"] = {
                    "model_name": map_model_name,
                    "model_revision": map_model_revision,
                    "tokenizer_name": map_tokenizer_name,
                    "tokenizer_revision": map_model_revision,
                }
            else:
                # Fallback: try to parse from model_name string if it contains "+"
                if "+" in model_name:
                    parts = model_name.split("+", 1)
                    model_info["map_model"] = {
                        "model_name": parts[0].strip(),
                        "model_revision": "main",
                        "tokenizer_name": parts[0].strip(),
                        "tokenizer_revision": "main",
                    }

            # Extract info for reduce_model (if exists)
            if reduce_model_obj:
                reduce_model_name = reduce_model_obj.model_name
                reduce_model_revision = getattr(reduce_model_obj, "revision", None) or "main"
                reduce_tokenizer_name = reduce_model_name
                if (
                    hasattr(reduce_model_obj, "tokenizer")
                    and reduce_model_obj.tokenizer is not None
                ):
                    tokenizer = reduce_model_obj.tokenizer
                    if hasattr(tokenizer, "name_or_path"):
                        reduce_tokenizer_name = tokenizer.name_or_path
                model_info["reduce_model"] = {
                    "model_name": reduce_model_name,
                    "model_revision": reduce_model_revision,
                    "tokenizer_name": reduce_tokenizer_name,
                    "tokenizer_revision": reduce_model_revision,
                }
            else:
                # Fallback: try to parse from model_name string if it contains "+"
                if "+" in model_name:
                    parts = model_name.split("+", 1)
                    if len(parts) > 1:
                        model_info["reduce_model"] = {
                            "model_name": parts[1].strip(),
                            "model_revision": "main",
                            "tokenizer_name": parts[1].strip(),
                            "tokenizer_revision": "main",
                        }

            # Set primary model_name (use map_model or combined)
            if map_model_obj:
                model_info["model_name"] = map_model_obj.model_name
                model_info["model_revision"] = getattr(map_model_obj, "revision", None) or "main"
                model_info["tokenizer_name"] = model_info.get("map_model", {}).get(
                    "tokenizer_name", model_info["model_name"]
                )
                model_info["tokenizer_revision"] = model_info["model_revision"]
            elif "+" in model_name:
                # Use combined name
                model_info["model_name"] = model_name
                model_info["model_revision"] = "main"
                model_info["tokenizer_name"] = model_name
                model_info["tokenizer_revision"] = "main"
            else:
                # Single model case
                summary_model = reduce_model_obj
                if summary_model:
                    model_info["model_name"] = summary_model.model_name
                    model_info["model_revision"] = (
                        getattr(summary_model, "revision", None) or "main"
                    )
                    if hasattr(summary_model, "tokenizer") and summary_model.tokenizer is not None:
                        tokenizer = summary_model.tokenizer
                        if hasattr(tokenizer, "name_or_path"):
                            model_info["tokenizer_name"] = tokenizer.name_or_path
                        else:
                            model_info["tokenizer_name"] = summary_model.model_name
                    else:
                        model_info["tokenizer_name"] = summary_model.model_name
                    model_info["tokenizer_revision"] = model_info["model_revision"]

            # Set defaults if still None
            if model_info["tokenizer_name"] is None:
                model_info["tokenizer_name"] = model_info.get("model_name", model_name)
            if model_info["model_revision"] is None:
                model_info["model_revision"] = "main"
            if model_info["tokenizer_revision"] is None:
                model_info["tokenizer_revision"] = model_info["model_revision"]

            # Weights format (same for both models typically)
            model_info["weights_format"] = None

        except Exception as e:
            # If we can't extract details, use defaults
            logger.warning(f"Could not extract detailed model info: {e}")
            model_info["tokenizer_name"] = model_name
            model_info["model_revision"] = "main"
            model_info["tokenizer_revision"] = "main"

    elif provider_type == "OpenAIProvider":
        # OpenAI models don't have these details
        model_info["tokenizer_name"] = None
        model_info["model_revision"] = None
        model_info["tokenizer_revision"] = None
        model_info["weights_format"] = None

    return model_info


def get_runtime_info(provider) -> Dict[str, Any]:  # noqa: C901
    """Extract runtime execution information from provider.

    Args:
        provider: Provider instance

    Returns:
        Dictionary with runtime details (device, backend, versions, etc.)
    """
    runtime: Dict[str, Any] = {}
    provider_type = provider.__class__.__name__

    if provider_type == "MLProvider":
        try:
            # Get the model instance (map_model or reduce_model)
            map_model = getattr(provider, "map_model", None)
            reduce_model = getattr(provider, "reduce_model", None)
            summary_model = map_model or reduce_model

            if summary_model:
                device = summary_model.device
                runtime["device"] = device

                # Try to import torch to get detailed info
                try:
                    import torch

                    runtime["torch_version"] = getattr(torch, "__version__", None)

                    # Get device-specific information
                    if device == "mps":
                        # Apple Silicon MPS
                        runtime["device_name"] = platform.machine()  # e.g., "arm64"
                        # Try to get more specific device name
                        try:
                            import subprocess

                            result = subprocess.run(
                                ["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True,
                                text=True,
                                timeout=2,
                            )
                            if result.returncode == 0:
                                runtime["device_name"] = result.stdout.strip()
                        except Exception:
                            pass

                        runtime["metal_backend"] = "mps"
                        # Try to get Metal feature set
                        try:
                            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                                runtime["metal_enabled"] = True
                                # Metal feature set is not directly exposed, but we can infer
                                # macOS 14+ typically has GPUFamily2_v1 or higher
                                runtime["metal_feature_set"] = "macOS_GPUFamily2_v1"  # Best guess
                        except Exception:
                            runtime["metal_enabled"] = False

                        # Get dtype from model if available
                        if hasattr(summary_model, "model") and summary_model.model is not None:
                            model_dtype = (
                                str(summary_model.model.dtype)
                                if hasattr(summary_model.model, "dtype")
                                else None
                            )
                            if model_dtype:
                                # Convert torch dtype to string format
                                dtype_map = {
                                    "torch.float32": "float32",
                                    "torch.float16": "float16",
                                    "torch.bfloat16": "bfloat16",
                                }
                                runtime["dtype"] = dtype_map.get(
                                    model_dtype, model_dtype.replace("torch.", "")
                                )

                        runtime["inference_backend"] = "transformers_generate"

                        # Get number of threads (CPU threads used by PyTorch)
                        try:
                            num_threads = torch.get_num_threads()
                            runtime["num_threads"] = num_threads
                        except Exception:
                            pass

                        # Check for compilation (torch.compile)
                        runtime["compile"] = {
                            "enabled": False,
                            "mode": None,
                        }
                        # Note: torch.compile state is not easily accessible, so we default to False

                    elif device == "cuda":
                        # NVIDIA CUDA
                        try:
                            device_name = (
                                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                            )
                            if device_name:
                                runtime["device_name"] = device_name
                        except Exception:
                            pass

                        runtime["cuda_enabled"] = True
                        try:
                            runtime["cuda_version"] = torch.version.cuda
                        except Exception:
                            pass

                        # Get dtype from model
                        if hasattr(summary_model, "model") and summary_model.model is not None:
                            model_dtype = (
                                str(summary_model.model.dtype)
                                if hasattr(summary_model.model, "dtype")
                                else None
                            )
                            if model_dtype:
                                dtype_map = {
                                    "torch.float32": "float32",
                                    "torch.float16": "float16",
                                    "torch.bfloat16": "bfloat16",
                                }
                                runtime["dtype"] = dtype_map.get(
                                    model_dtype, model_dtype.replace("torch.", "")
                                )

                        runtime["inference_backend"] = "transformers_generate"

                        # Get number of threads
                        try:
                            num_threads = torch.get_num_threads()
                            runtime["num_threads"] = num_threads
                        except Exception:
                            pass

                        runtime["compile"] = {
                            "enabled": False,
                            "mode": None,
                        }

                    elif device == "cpu":
                        # CPU
                        runtime["device_name"] = platform.processor() or platform.machine()
                        runtime["inference_backend"] = "transformers_generate"

                        # Get dtype
                        if hasattr(summary_model, "model") and summary_model.model is not None:
                            model_dtype = (
                                str(summary_model.model.dtype)
                                if hasattr(summary_model.model, "dtype")
                                else None
                            )
                            if model_dtype:
                                dtype_map = {
                                    "torch.float32": "float32",
                                    "torch.float16": "float16",
                                    "torch.bfloat16": "bfloat16",
                                }
                                runtime["dtype"] = dtype_map.get(
                                    model_dtype, model_dtype.replace("torch.", "")
                                )

                        # Get number of threads
                        try:
                            num_threads = torch.get_num_threads()
                            runtime["num_threads"] = num_threads
                        except Exception:
                            pass

                        runtime["compile"] = {
                            "enabled": False,
                            "mode": None,
                        }

                except ImportError:
                    # torch not available
                    runtime["device"] = device
                    runtime["torch_version"] = None

        except Exception as e:
            logger.warning(f"Could not extract runtime info: {e}")
            runtime = {"device": "unknown"}

    elif provider_type == "OpenAIProvider":
        # OpenAI API - device is still the physical computer running the evaluation
        # Backend is the API, but device is where the code executes
        try:
            import torch

            # Detect the actual device where evaluation runs
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                device_name = platform.machine()  # e.g., "arm64"
                try:
                    import subprocess

                    result = subprocess.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    if result.returncode == 0:
                        device_name = result.stdout.strip()
                except Exception:
                    pass
                runtime = {
                    "device": device,
                    "device_name": device_name,
                    "backend": "openai_api",
                    "inference_backend": "openai_api",
                }
            elif torch.cuda.is_available():
                device = "cuda"
                try:
                    device_name = torch.cuda.get_device_name(0)
                except Exception:
                    device_name = None
                runtime = {
                    "device": device,
                    "device_name": device_name,
                    "backend": "openai_api",
                    "inference_backend": "openai_api",
                }
            else:
                device = "cpu"
                runtime = {
                    "device": device,
                    "device_name": platform.processor() or platform.machine(),
                    "backend": "openai_api",
                    "inference_backend": "openai_api",
                }
        except ImportError:
            # torch not available, use platform detection
            runtime = {
                "device": "cpu",  # Fallback
                "device_name": platform.processor() or platform.machine(),
                "backend": "openai_api",
                "inference_backend": "openai_api",
            }

    return runtime


def get_preprocessing_steps(profile_id: str) -> Dict[str, Any]:
    """Get preprocessing steps for a profile.

    Preprocessing defines what the model sees.
    Fingerprinting it defines why results changed.
    Preprocessing is deterministic, rule-based, versionable transformations
    applied before model runs.

    Args:
        profile_id: Preprocessing profile ID (e.g., "cleaning_v3", "summary_input_cleaning_v1")

    Returns:
        Dictionary of preprocessing steps with their configurations
    """
    # Map profile_id to known step configurations
    # This should match the actual implementation in preprocessing/profiles.py
    profile_steps = {
        "cleaning_v1": {
            "remove_timestamps": True,
            "normalize_speakers": True,
            "remove_sponsor_blocks": False,  # v1 didn't have this
            "collapse_blank_lines": True,
            "remove_fillers": False,
        },
        "cleaning_v2": {
            "remove_timestamps": True,
            "normalize_speakers": True,
            "remove_sponsor_blocks": True,  # Added in v2
            "collapse_blank_lines": True,
            "remove_fillers": False,
        },
        "cleaning_v3": {
            "remove_timestamps": True,
            "normalize_speakers": True,
            "remove_sponsor_blocks": True,
            "collapse_blank_lines": True,
            "remove_fillers": False,
            "remove_garbage_lines": True,  # Added in v3
            "remove_credit_blocks": True,  # Added in v3 (strips credits FIRST)
            "remove_outro_blocks": True,  # Added in v3
            "remove_artifacts": True,  # Added in v3 (summarization artifacts)
        },
        "cleaning_v4": {
            "remove_timestamps": True,
            "normalize_speakers": True,
            "remove_sponsor_blocks": True,
            "collapse_blank_lines": True,
            "remove_fillers": False,
            "remove_garbage_lines": True,
            "remove_credit_blocks": True,
            "remove_outro_blocks": True,
            "remove_artifacts": True,
            "strip_episode_header": True,  # Added in v4
            "anonymize_speakers": True,  # Added in v4
            "filter_junk_lines": True,  # Added in v4 (is_junk_line)
        },
        "cleaning_none": {
            "remove_timestamps": False,
            "normalize_speakers": False,
            "remove_sponsor_blocks": False,
            "collapse_blank_lines": False,
            "remove_fillers": False,
        },
    }

    # Return steps for known profile, or default for unknown
    if profile_id in profile_steps:
        return profile_steps[profile_id]
    else:
        # Unknown profile - return minimal info
        return {
            "remove_timestamps": None,  # Unknown
            "normalize_speakers": None,
            "remove_sponsor_blocks": None,
            "collapse_blank_lines": None,
        }


def get_preprocessing_profile_version(profile_id: str) -> str:
    """Get version number from profile_id.

    Args:
        profile_id: Profile ID (e.g., "cleaning_v3" -> "3.0")

    Returns:
        Version string (e.g., "3.0", "1.0", or "unknown")
    """
    # Extract version from profile_id patterns like "cleaning_v3", "summary_input_cleaning_v2"
    import re

    # Pattern: _v<number> at the end
    match = re.search(r"_v(\d+)$", profile_id)
    if match:
        version_num = match.group(1)
        return f"{version_num}.0"

    # Pattern: v<number> at the end
    match = re.search(r"v(\d+)$", profile_id)
    if match:
        version_num = match.group(1)
        return f"{version_num}.0"

    # Default version for profiles without explicit version
    return "1.0"


def generate_enhanced_fingerprint(
    baseline_id: str,
    dataset_id: str,
    experiment_config: Optional[ExperimentConfig],
    provider: Any,
    model_name: str,
    preprocessing_profile: str,
    git_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate enhanced fingerprint with structured sections.

    The fingerprint captures all variables that affect AI model outputs:
    - run_context: When and why this run happened (baseline_id, dataset_id, git state)
    - provider: Who did the work (OpenAI vs local ML, library versions)
    - model: What brain ran (model name, version, endpoint)
    - generation_params: How randomness was controlled (temperature, token limits)
    - preprocessing: What text the model actually saw (profile, steps)
    - chunking: How text was structured (strategy, sizes)
    - environment: Last-mile reproducibility (Python version, OS)

    Args:
        baseline_id: Baseline identifier
        dataset_id: Dataset identifier
        experiment_config: Optional experiment configuration
        provider: Provider instance
        model_name: Model name (e.g., "gpt-4o-mini", "bart-small+long-fast")
        preprocessing_profile: Preprocessing profile ID (e.g., "cleaning_v3")
        git_info: Git information dictionary with commit_sha, branch, is_dirty

    Returns:
        Enhanced fingerprint dictionary with all sections populated

    Note on model_version:
        - OpenAI: Extracted from model name if it contains a snapshot date
          (e.g., "gpt-4o-mini-2024-07-18" -> "2024-07-18")
        - HuggingFace: Git revision/commit hash (e.g., "main", "abc123def456")
          Currently None if not specified in config (TODO: extract from model)

    Note on endpoint:
        - OpenAI: API endpoint used ("chat.completions" for chat models, "completions" for older)
        - HuggingFace: None (local models don't use API endpoints)
    """
    run_id = datetime.utcnow().isoformat() + "Z"
    provider_type = provider.__class__.__name__
    provider_lib_info = get_provider_library_info(provider)

    # Extract detailed model information
    task = "summarization"  # Default for baselines
    endpoint = None

    # Get detailed model info from provider
    model_details = get_model_details(provider, model_name, experiment_config)

    if experiment_config:
        task = experiment_config.task
        if experiment_config.backend.type == "openai":
            # OpenAI API endpoint: "chat.completions" for chat models,
            # "completions" for older models
            # Most modern models (gpt-4o-mini, gpt-4, etc.) use chat.completions
            endpoint = "chat.completions"
            # OpenAI model version: Extract from model name if it contains a snapshot date
            # Example: "gpt-4o-mini-2024-07-18" -> "2024-07-18"
            # Standard names like "gpt-4o-mini" don't have explicit versions
            model_name_str = experiment_config.backend.model
            # Check if model name ends with a date pattern (YYYY-MM-DD)
            # Pattern: model-name-YYYY-MM-DD
            if len(model_name_str) >= 11:  # Minimum: "gpt-4-2024-01-01" (15 chars)
                # Try to extract date from end of string
                # Look for pattern: -YYYY-MM-DD at the end
                try:
                    # Check if last 10 characters form a valid date
                    potential_date = model_name_str[-10:]
                    datetime.strptime(potential_date, "%Y-%m-%d")
                    # Check that it's preceded by a dash
                    if model_name_str[-11] == "-":
                        model_details["model_revision"] = potential_date
                except (ValueError, IndexError):
                    # Not a date pattern, leave as None
                    pass
        elif experiment_config.backend.type == "hf_local":
            # For HuggingFace models, revision is already extracted in get_model_details
            endpoint = None  # Local models don't use API endpoints

    # Extract generation parameters
    # For baselines, use deterministic defaults (temperature=0.0, top_p=1.0, seed=42)
    # This ensures reproducibility and makes comparisons meaningful
    generation_params = {}
    map_generation_params = {}
    reduce_generation_params = {}

    if experiment_config:
        # Handle different task types
        if experiment_config.task == "ner_entities":
            # NER tasks don't have generation params
            generation_params = {}
            map_generation_params = {}
            reduce_generation_params = {}
        elif experiment_config.backend.type == "openai":
            # OpenAI API parameters (use defaults from SummarizationParams)
            generation_params = {
                "temperature": 0.0,  # Baseline: 0.0 for deterministic
                "top_p": 1.0,  # Baseline: 1.0 (disabled nucleus sampling)
                "max_tokens": 800,  # Default from SummarizationParams
                "seed": 42,  # Baseline: fixed seed for reproducibility
            }
            map_generation_params = {}
            reduce_generation_params = {}
        else:
            # HuggingFace/Transformers models - use explicit map_params/reduce_params
            if not experiment_config.map_params:
                raise ValueError("map_params required for hf_local backend with summarization task")
            map_generation_params = {
                "max_new_tokens": experiment_config.map_params.max_new_tokens,
                "min_new_tokens": experiment_config.map_params.min_new_tokens,
                "num_beams": experiment_config.map_params.num_beams,
                "no_repeat_ngram_size": experiment_config.map_params.no_repeat_ngram_size,
                "length_penalty": experiment_config.map_params.length_penalty,
                "early_stopping": experiment_config.map_params.early_stopping,
                "repetition_penalty": experiment_config.map_params.repetition_penalty,
                "temperature": 0.0,  # Deterministic
                "seed": 42,  # Fixed seed
            }
            # Extract reduce_params separately (CRITICAL: reduce stage uses different params)
            if experiment_config.reduce_params:
                reduce_generation_params = {
                    "max_new_tokens": experiment_config.reduce_params.max_new_tokens,
                    "min_new_tokens": experiment_config.reduce_params.min_new_tokens,
                    "num_beams": experiment_config.reduce_params.num_beams,
                    "no_repeat_ngram_size": experiment_config.reduce_params.no_repeat_ngram_size,
                    "length_penalty": experiment_config.reduce_params.length_penalty,
                    "early_stopping": experiment_config.reduce_params.early_stopping,
                    "repetition_penalty": experiment_config.reduce_params.repetition_penalty,
                    "temperature": 0.0,  # Deterministic
                    "seed": 42,  # Fixed seed
                }
            else:
                # Fallback: use map params if reduce_params not specified
                reduce_generation_params = map_generation_params.copy()
            generation_params = map_generation_params.copy()

    # Extract tokenization information from tokenize config
    tokenize_config = {}
    if experiment_config and experiment_config.tokenize:
        tokenize_config = {
            "map_max_input_tokens": experiment_config.tokenize.map_max_input_tokens,
            "reduce_max_input_tokens": experiment_config.tokenize.reduce_max_input_tokens,
            "truncation": experiment_config.tokenize.truncation,
        }

    # For NER tasks, extract params (currently unused but kept for future use)
    # ner_params = {}
    # if experiment_config and experiment_config.task == "ner_entities" and \
    #     experiment_config.params:
    #     ner_params = {
    #         "labels": experiment_config.params.get("labels"),
    #         "model": (
    #             experiment_config.backend.model
    #             if experiment_config.backend.type == "spacy_local"
    #             else None
    #         ),
    #     }

    # Extract chunking information
    # Chunking is one of the most important parts of the pipeline.
    # It affects quality, cost, latency, hallucination rate, and comparability.
    # Changing chunking can change outputs as much as changing models.
    chunking = {}
    if experiment_config:
        # Use explicit chunking config if provided, otherwise use defaults
        if experiment_config.chunking:
            # Explicit chunking config provided
            chunking_config = experiment_config.chunking
            strategy = chunking_config.strategy
            word_chunk_size = chunking_config.word_chunk_size
            word_overlap = chunking_config.word_overlap
            source = "explicit_config"
        else:
            # No explicit chunking config - use defaults (must be marked as such)
            strategy = "word_chunking"  # Default strategy
            word_chunk_size = 900  # Default from summarizer.DEFAULT_WORD_CHUNK_SIZE
            word_overlap = 150  # Default from summarizer.DEFAULT_WORD_OVERLAP
            source = "default"

        # Effective token chunk size is capped at 600 for encoder-decoder models
        # Overlap is calculated as 10% of effective chunk size
        # ENCODER_DECODER_TOKEN_CHUNK_SIZE
        effective_token_chunk_size = min(600, word_chunk_size)
        # CHUNK_OVERLAP_RATIO = 0.1
        token_overlap = max(1, int(effective_token_chunk_size * 0.1))

        chunking = {
            "strategy": strategy,
            "word_chunk_size": word_chunk_size,
            "word_overlap": word_overlap,
            "effective_token_chunk_size": effective_token_chunk_size,  # Actual token size used
            "token_overlap": token_overlap,  # Actual token overlap used
            "boundary_heuristic": "sentence_boundary_prefer",
            "source": source,  # "explicit_config" or "default" - governance tracking
            # Note: word chunking is converted to token chunking internally
            # for encoder-decoder models (BART, PEGASUS)
        }
    else:
        # No experiment config - use defaults
        # Try to determine from model name
        effective_token_chunk_size = None
        effective_token_overlap = None

        if "bart" in model_name.lower() or "pegasus" in model_name.lower():
            effective_token_chunk_size = 600
            effective_token_overlap = 60
        elif "led" in model_name.lower() or "longformer" in model_name.lower():
            effective_token_chunk_size = 16384
            effective_token_overlap = 1638

        chunking = {
            "strategy": "auto",
            "boundary_heuristic": "sentence_boundary_prefer",
        }

        if effective_token_chunk_size:
            chunking["effective_token_chunk_size"] = effective_token_chunk_size
            chunking["effective_token_overlap"] = effective_token_overlap
            chunking["overlap_ratio"] = 0.1

    # Get environment information
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    os_info = f"{platform.system()}-{platform.release()}-{platform.machine()}"

    # Determine pipeline structure
    map_model_info = model_details.get("map_model")
    reduce_model_info = model_details.get("reduce_model")
    is_map_reduce = map_model_info is not None and reduce_model_info is not None

    # Build pipeline structure
    if is_map_reduce:
        # Map-reduce pipeline with separate stages
        pipeline = {
            "type": "map_reduce",
            "stages": {
                "map": {
                    "stage_id": f"map_{map_model_info['model_name'].split('/')[-1]}",
                    "model": {
                        "provider_type": "local_ml",
                        "framework": "transformers",
                        "model_name": map_model_info["model_name"],
                        "model_revision": map_model_info.get("model_revision", "main"),
                        "tokenizer_name": map_model_info.get("tokenizer_name"),
                        "tokenizer_revision": map_model_info.get("tokenizer_revision"),
                    },
                    "generation_params": {
                        "max_new_tokens": generation_params.get(
                            "max_new_tokens", generation_params.get("max_length", 150)
                        ),
                        "min_new_tokens": generation_params.get(
                            "min_new_tokens", generation_params.get("min_length", 30)
                        ),
                        "temperature": generation_params.get("temperature", 0.0),
                        "top_p": generation_params.get("top_p", 1.0),
                        "repetition_penalty": generation_params.get("repetition_penalty", 1.05),
                        "seed": generation_params.get("seed", 42),
                    },
                },
                "reduce": {
                    "stage_id": f"reduce_{reduce_model_info['model_name'].split('/')[-1]}",
                    "model": {
                        "provider_type": "local_ml",
                        "framework": "transformers",
                        "model_name": reduce_model_info["model_name"],
                        "model_revision": reduce_model_info.get("model_revision", "main"),
                        "tokenizer_name": reduce_model_info.get("tokenizer_name"),
                        "tokenizer_revision": reduce_model_info.get("tokenizer_revision"),
                    },
                    "generation_params": {
                        # Reduce phase uses reduce_params (not map params!)
                        "max_new_tokens": reduce_generation_params.get(
                            "max_new_tokens",
                            generation_params.get(
                                "max_new_tokens", generation_params.get("max_length", 150)
                            ),
                        ),
                        "min_new_tokens": reduce_generation_params.get(
                            "min_new_tokens",
                            generation_params.get(
                                "min_new_tokens", generation_params.get("min_length", 30)
                            ),
                        ),
                        "num_beams": reduce_generation_params.get(
                            "num_beams", generation_params.get("num_beams", None)
                        ),
                        "no_repeat_ngram_size": reduce_generation_params.get(
                            "no_repeat_ngram_size",
                            generation_params.get("no_repeat_ngram_size", None),
                        ),
                        "length_penalty": reduce_generation_params.get(
                            "length_penalty", generation_params.get("length_penalty", None)
                        ),
                        "early_stopping": reduce_generation_params.get(
                            "early_stopping", generation_params.get("early_stopping", None)
                        ),
                        "temperature": reduce_generation_params.get(
                            "temperature", generation_params.get("temperature", 0.0)
                        ),
                        "top_p": reduce_generation_params.get(
                            "top_p", generation_params.get("top_p", 1.0)
                        ),
                        "repetition_penalty": reduce_generation_params.get(
                            "repetition_penalty", generation_params.get("repetition_penalty", 1.05)
                        ),
                        "seed": reduce_generation_params.get(
                            "seed", generation_params.get("seed", 42)
                        ),
                    },
                },
            },
        }
    else:
        # Single-stage pipeline (OpenAI or single HF model)
        pipeline = {
            "type": "single_stage",
            "stages": {
                "main": {
                    "stage_id": "main",
                    "model": {
                        "provider_type": (
                            "openai"
                            if experiment_config and experiment_config.backend.type == "openai"
                            else "local_ml"
                        ),
                        "framework": (
                            None
                            if experiment_config and experiment_config.backend.type == "openai"
                            else "transformers"
                        ),
                        "model_name": model_details["model_name"],
                        "model_revision": model_details.get("model_revision"),
                        "tokenizer_name": model_details.get("tokenizer_name"),
                        "tokenizer_revision": model_details.get("tokenizer_revision"),
                        "endpoint": endpoint,  # Only for OpenAI
                    },
                    "generation_params": generation_params,
                },
            },
        }

    fingerprint = {
        "fingerprint_version": "1.0",
        "task": task,
        "run_context": {
            "run_id": run_id,
            "baseline_id": baseline_id,
            "dataset_id": dataset_id,
            "git": {
                "commit": git_info.get("commit_sha"),
                "branch": git_info.get("branch"),
                "dirty": git_info.get("is_dirty", False),
            },
        },
        "provider": {
            "provider_type": provider_type.lower().replace("provider", ""),
            **provider_lib_info,
        },
        "pipeline": pipeline,
        "preprocessing": {
            "profile_id": preprocessing_profile,
            "profile_version": get_preprocessing_profile_version(preprocessing_profile),
            "steps": get_preprocessing_steps(preprocessing_profile),
            # Preprocessing defines what the model sees.
            # Fingerprinting it defines why results changed.
            # Preprocessing is deterministic, rule-based, versionable transformations
            # applied before model runs (not part of the model itself).
        },
        "tokenization": (
            tokenize_config if tokenize_config else {}
        ),  # New: explicit tokenization limits
        "chunking": chunking,
        "environment": {
            "python_version": python_version,
            "os": os_info,
        },
        "runtime": get_runtime_info(provider),
    }

    # Conditionally add prompts section (only for OpenAI backends that use prompts)
    prompt_info = get_prompt_info(experiment_config)
    if prompt_info is not None:
        fingerprint["prompts"] = prompt_info

    return fingerprint


def get_prompt_info(experiment_config: Optional[ExperimentConfig]) -> Optional[Dict[str, Any]]:
    """Extract prompt information from experiment config.

    Prompts are critical for reproducibility - different prompts produce different outputs.
    This section captures prompt names, hashes, and parameters.

    Note: Prompts are only used for OpenAI backends. For ML models (hf_local),
    this returns None to omit the prompts section entirely.

    Args:
        experiment_config: Optional experiment configuration

    Returns:
        Dictionary with prompt information, or None if prompts are not used
    """
    # Only include prompts for OpenAI backends (ML models don't use prompts)
    if not experiment_config or experiment_config.backend.type != "openai":
        return None

    prompts: Dict[str, Any] = {}

    if hasattr(experiment_config, "prompts"):
        try:
            prompt_config = experiment_config.prompts
            user_prompt_name = prompt_config.user if hasattr(prompt_config, "user") else None
            system_prompt_name = prompt_config.system if hasattr(prompt_config, "system") else None
            prompt_params = prompt_config.params if hasattr(prompt_config, "params") else {}

            # Get prompt metadata (includes hash)
            if user_prompt_name:
                try:
                    user_metadata = get_prompt_metadata(user_prompt_name, prompt_params)
                    prompts["user"] = {
                        "name": user_metadata["name"],
                        "file": user_metadata.get("file"),
                        "sha256": user_metadata.get("sha256"),
                        "params": user_metadata.get("params", {}),
                    }
                except Exception as e:
                    logger.warning(f"Could not get user prompt metadata: {e}")
                    prompts["user"] = {
                        "name": user_prompt_name,
                        "sha256": None,
                    }

            if system_prompt_name:
                try:
                    system_metadata = get_prompt_metadata(system_prompt_name, prompt_params)
                    prompts["system"] = {
                        "name": system_metadata["name"],
                        "file": system_metadata.get("file"),
                        "sha256": system_metadata.get("sha256"),
                        "params": system_metadata.get("params", {}),
                    }
                except Exception as e:
                    logger.warning(f"Could not get system prompt metadata: {e}")
                    prompts["system"] = {
                        "name": system_prompt_name,
                        "sha256": None,
                    }

            # Only return prompts dict if we have at least one prompt
            if prompts:
                return prompts
        except Exception as e:
            logger.warning(f"Could not extract prompt info: {e}")

    # No prompts configured or error - return None to omit section
    return None


def get_git_commit() -> Optional[str]:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_status() -> Dict[str, Any]:
    """Get git repository status."""
    try:
        commit_sha = get_git_commit()
        status_output = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        is_dirty = bool(status_output.stdout.strip())
        return {
            "commit_sha": commit_sha,
            "is_dirty": is_dirty,
            "branch": subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit_sha": None, "is_dirty": None, "branch": None}


def load_dataset_json(dataset_id: str) -> Dict[str, Any]:
    """Load dataset JSON definition.

    Args:
        dataset_id: Dataset identifier (e.g., "indicator_v1")

    Returns:
        Dataset dictionary

    Raises:
        FileNotFoundError: If dataset JSON doesn't exist
    """
    # Try data/eval/datasets first, then benchmarks/datasets
    dataset_path = Path("data/eval/datasets") / f"{dataset_id}.json"
    if not dataset_path.exists():
        dataset_path = Path("benchmarks/datasets") / f"{dataset_id}.json"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset definition not found: tried data/eval/datasets/{dataset_id}.json "
            f"and benchmarks/datasets/{dataset_id}.json"
        )
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def create_baseline_readme(
    baseline_path: Path,
    baseline_id: str,
    dataset_id: str,
    episode_count: int,
    fingerprint: Dict[str, Any],
) -> None:
    """Create README.md for a baseline directory using template.

    Args:
        baseline_path: Path to baseline directory
        baseline_id: Baseline identifier
        dataset_id: Dataset identifier
        episode_count: Number of episodes in baseline
        fingerprint: Fingerprint dictionary with model/config details
    """
    # Load template
    template_path = Path("data/eval/baselines/BASELINE_README_TEMPLATE.md")
    if not template_path.exists():
        raise FileNotFoundError(f"Baseline README template not found: {template_path}")
    template_content = template_path.read_text(encoding="utf-8")

    # Extract pipeline info from fingerprint
    pipeline = fingerprint.get("pipeline", {})
    pipeline_type = pipeline.get("type", "single_stage")
    stages = pipeline.get("stages", {})

    # Extract preprocessing info
    preprocessing_info = fingerprint.get("preprocessing", {})
    preprocessing_profile = preprocessing_info.get("profile_id", "Unknown")

    # Determine model description from pipeline structure
    if pipeline_type == "map_reduce":
        map_stage = stages.get("map", {})
        reduce_stage = stages.get("reduce", {})
        map_model = map_stage.get("model", {})
        reduce_model = reduce_stage.get("model", {})

        map_name = map_model.get("model_name", "Unknown")
        map_rev = map_model.get("model_revision", "main")
        reduce_name = reduce_model.get("model_name", "Unknown")
        reduce_rev = reduce_model.get("model_revision", "main")

        model_description = (
            f"- **MAP Model:** {map_name} (revision: {map_rev})\n"
            f"- **REDUCE Model:** {reduce_name} (revision: {reduce_rev})"
        )

        # Get generation params from map stage (they're typically the same)
        gen_params = map_stage.get("generation_params", {})
    else:
        # Single-stage pipeline
        main_stage = stages.get("main", {})
        main_model = main_stage.get("model", {})
        model_name = main_model.get("model_name", "Unknown")
        model_rev = main_model.get("model_revision", "main")
        model_description = f"- **Model:** {model_name} (revision: {model_rev})"

        gen_params = main_stage.get("generation_params", {})

    temperature = gen_params.get("temperature", "Unknown")
    seed = gen_params.get("seed", "Unknown")

    # Generate next version ID for replacement policy
    if "_v" in baseline_id:
        base_name = baseline_id.rsplit("_v", 1)[0]
        version_num = int(baseline_id.rsplit("_v", 1)[1])
        next_version = f"{base_name}_v{version_num + 1}"
    else:
        next_version = f"{baseline_id}_v2"

    # Render template
    content = template_content.format(
        baseline_id=baseline_id,
        dataset_id=dataset_id,
        episode_count=episode_count,
        next_version=next_version,
        model_description=model_description,
        preprocessing_profile=preprocessing_profile,
        temperature=temperature,
        seed=seed,
    )

    readme_path = baseline_path / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    logger.info(f"Created README.md: {readme_path}")


def find_reference_path(reference_id: str, dataset_id: str) -> Path:
    """Find reference path (checks baselines and references directories).

    Args:
        reference_id: Reference identifier
        dataset_id: Dataset identifier

    Returns:
        Path to reference directory

    Raises:
        FileNotFoundError: If reference not found
    """
    # Check references directory first (references/{dataset_id}/{reference_id})
    ref_path = Path("data/eval/references") / dataset_id / reference_id
    if ref_path.exists():
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
        f"data/eval/baselines/{reference_id}, benchmarks/baselines/{reference_id}"
    )


def materialize_baseline(
    baseline_id: str,
    dataset_id: str,
    experiment_config_path: Optional[str] = None,
    preprocessing_profile: Optional[str] = None,
    output_dir: Optional[str] = None,
    reference_ids: Optional[List[str]] = None,
) -> None:
    """Materialize a baseline from current system state.

    Args:
        baseline_id: Unique baseline identifier (e.g., "bart_led_baseline_v1")
        dataset_id: Dataset identifier (e.g., "indicator_v1")
        experiment_config_path: Optional path to experiment config YAML
        preprocessing_profile: Optional preprocessing profile ID
        output_dir: Optional output directory (default: benchmarks/baselines)
        reference_ids: Optional list of reference IDs for vs_reference metrics

    Raises:
        ValueError: If baseline already exists
        FileNotFoundError: If dataset or required files don't exist
        RuntimeError: If baseline generation fails
    """
    # Determine baseline output directory
    if output_dir:
        baseline_base = Path(output_dir)
    else:
        baseline_base = Path("benchmarks/baselines")

    # Check if baseline already exists (immutability)
    baseline_path = baseline_base / baseline_id
    if baseline_path.exists():
        raise ValueError(
            f"Baseline '{baseline_id}' already exists at {baseline_path}. "
            "Baselines are immutable. Create a new baseline with a different ID."
        )

    logger.info(f"Materializing baseline: {baseline_id} for dataset: {dataset_id}")

    # Load dataset definition
    dataset = load_dataset_json(dataset_id)

    # Create baseline directory structure
    baseline_path.mkdir(parents=True, exist_ok=True)
    artifacts_dir = baseline_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Get git info
    git_info = get_git_status()
    if git_info["is_dirty"]:
        logger.warning("Repository has uncommitted changes. Baseline will include dirty state.")

    # Load experiment config if provided
    experiment_config = None
    if experiment_config_path:
        experiment_config = load_experiment_config(experiment_config_path)
        # Save config to baseline
        config_path = baseline_path / "config.yaml"
        config_path.write_text(Path(experiment_config_path).read_text(encoding="utf-8"))

    # Process each episode in dataset
    predictions = []
    total_chars_in = 0
    total_chars_out = 0
    start_time = time.time()

    # Create provider (if experiment config provided, use it; otherwise use defaults)
    if experiment_config and experiment_config.backend.type == "openai":
        # OpenAI backend - use defaults from SummarizationParams
        summarization_params = SummarizationParams(
            model_name=experiment_config.backend.model,
        )
        provider = create_summarization_provider("openai", summarization_params)
        provider.initialize()
        model_name = experiment_config.backend.model
    elif experiment_config and experiment_config.backend.type == "hf_local":
        # Use HF models from experiment config
        from podcast_scraper import config

        # Build model name string for fingerprint
        map_model = experiment_config.backend.map_model or "bart-small"
        reduce_model = experiment_config.backend.reduce_model or "long-fast"
        model_name = f"{map_model}+{reduce_model}"

        # Use explicit map_params/reduce_params (required for hf_local)
        summary_map_params = {
            "max_new_tokens": experiment_config.map_params.max_new_tokens,
            "min_new_tokens": experiment_config.map_params.min_new_tokens,
        }
        summary_reduce_params = {
            "max_new_tokens": experiment_config.reduce_params.max_new_tokens,
            "min_new_tokens": experiment_config.reduce_params.min_new_tokens,
        }

        cfg = config.Config(
            rss_url="",  # Not used for baseline
            summary_provider="transformers",
            summary_model=map_model,
            summary_reduce_model=reduce_model,
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
            summary_map_params=summary_map_params,
            summary_reduce_params=summary_reduce_params,
        )
        provider = create_summarization_provider(cfg)
        provider.initialize()
    else:
        # Default: use transformers with BART
        from podcast_scraper import config

        cfg = config.Config(
            rss_url="",  # Not used for baseline
            summary_provider="transformers",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
        )
        provider = create_summarization_provider(cfg)
        provider.initialize()
        model_name = "facebook/bart-large-cnn"

    # Generate enhanced fingerprint structure
    fingerprint = generate_enhanced_fingerprint(
        baseline_id=baseline_id,
        dataset_id=dataset_id,
        experiment_config=experiment_config,
        provider=provider,
        model_name=model_name,
        preprocessing_profile=preprocessing_profile or "cleaning_v3",
        git_info=git_info,
    )

    for episode in dataset["episodes"]:
        episode_id = episode["episode_id"]
        logger.info(f"Processing episode: {episode_id}")

        # Load transcript
        transcript_path = Path(episode["transcript_path"])
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")

        transcript_text = transcript_path.read_text(encoding="utf-8").strip()
        total_chars_in += len(transcript_text)
        input_hash = hash_text(transcript_text)

        # Generate summary
        # For baselines, we can optionally include intermediates (but keep lean by default)
        # Set include_intermediates=True in environment or config if needed for debugging
        include_intermediates = (
            os.getenv("BASELINE_INCLUDE_INTERMEDIATES", "false").lower() == "true"
        )

        t0 = time.time()
        try:
            # Request intermediates only if explicitly enabled (baselines should be lean)
            summary_params = (
                {"return_intermediates": include_intermediates}
                if experiment_config and experiment_config.backend.type == "hf_local"
                else {}
            )
            summary_result = provider.summarize(transcript_text, params=summary_params)
            if isinstance(summary_result, dict):
                summary = summary_result.get("summary", "")
                intermediates = summary_result.get("intermediates")
            else:
                summary = str(summary_result)
                intermediates = None
        except Exception as e:
            logger.error(f"Failed to summarize episode {episode_id}: {e}")
            raise RuntimeError(f"Summarization failed for episode {episode_id}: {e}") from e

        dt = time.time() - t0
        total_chars_out += len(summary)
        output_hash = hash_text(summary)

        # Save prediction (will be written to JSONL after all episodes processed)
        prediction_record: Dict[str, Any] = {
            "episode_id": episode_id,
            "dataset_id": dataset_id,
            "baseline_id": baseline_id,
            "output": {
                "summary_final": summary,  # Renamed from summary_long for clarity
            },
            "fingerprint_ref": "fingerprint.json",
            "metadata": {
                "input_hash": f"sha256:{input_hash}",
                "output_hash": f"sha256:{output_hash}",
                "input_path": str(transcript_path),
                "input_length_chars": len(transcript_text),
                "output_length_chars": len(summary),
                "processing_time_seconds": dt,
            },
        }

        # For baselines: include intermediates only if explicitly enabled
        # Otherwise, include just counts/hashes to keep baselines lean
        if intermediates:
            if include_intermediates:
                # Full intermediate outputs (for debugging)
                prediction_record["intermediate"] = intermediates
            else:
                # Lean version: just counts and hashes
                map_summaries = intermediates.get("map_summaries", [])
                prediction_record["intermediate"] = {
                    "map_summaries": [
                        {
                            "chunk_id": item["chunk_id"],
                            "text_hash": f"sha256:{hash_text(item['text'])}",
                            "text_length_chars": len(item["text"]),
                        }
                        for item in map_summaries
                    ],
                    "map_summaries_count": len(map_summaries),
                }

        predictions.append(prediction_record)

        logger.info(
            f"Episode {episode_id} completed in {dt:.1f}s, " f"summary length={len(summary)} chars"
        )

    total_time = time.time() - start_time
    avg_time = total_time / len(predictions) if predictions else 0
    avg_compression = (total_chars_in / total_chars_out) if total_chars_out > 0 else None

    # Add run_id to all predictions (same timestamp for all episodes in this baseline)
    run_id = datetime.utcnow().isoformat() + "Z"
    for prediction in predictions:
        prediction["metadata"]["run_id"] = run_id

    # Note: We use baseline_id as run_id for metrics (for consistency)
    metrics_run_id = baseline_id

    # Save predictions to single JSONL file (needed for scorer.py)
    predictions_path = baseline_path / "predictions.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    logger.info(f"Predictions saved to: {predictions_path}")

    # Save baseline.json (renamed from metadata.json)
    baseline_metadata = {
        "baseline_id": baseline_id,
        "dataset_id": dataset_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "created_by": os.getenv("USER", "unknown"),
        "fingerprint_ref": "fingerprint.json",
        "stats": {
            "num_episodes": len(predictions),
            "total_time_seconds": total_time,
            "avg_time_seconds": avg_time,
            "total_chars_in": total_chars_in,
            "total_chars_out": total_chars_out,
            "avg_compression": avg_compression,
        },
    }

    baseline_path_file = baseline_path / "baseline.json"
    baseline_path_file.write_text(
        json.dumps(baseline_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save enhanced fingerprint
    fingerprint_path = baseline_path / "fingerprint.json"
    fingerprint_path.write_text(
        json.dumps(fingerprint, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Find reference paths if reference_ids provided
    reference_paths = None
    if reference_ids:
        reference_paths = {}
        for ref_id in reference_ids:
            try:
                ref_path = find_reference_path(ref_id, dataset_id)
                # Validate reference has predictions.jsonl
                ref_predictions_path = ref_path / "predictions.jsonl"
                if not ref_predictions_path.exists():
                    logger.warning(
                        f"Reference '{ref_id}' missing predictions.jsonl, "
                        "skipping vs_reference metrics"
                    )
                    continue
                reference_paths[ref_id] = ref_path
                logger.info(f"Found reference '{ref_id}' at {ref_path}")
            except FileNotFoundError as e:
                logger.warning(
                    f"Reference '{ref_id}' not found, skipping vs_reference metrics: {e}"
                )
                continue

    # Compute structured metrics from predictions using scorer.py
    # Use baseline_id as run_id for metrics (for consistency with experiment runs)
    # Use scorer.py to compute metrics (includes all implemented gates, cost tracking, etc.)
    metrics = score_run(
        predictions_path=predictions_path,
        dataset_id=dataset_id,
        run_id=metrics_run_id,
        reference_paths=reference_paths,
    )
    metrics_path = baseline_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Create README.md
    create_baseline_readme(
        baseline_path=baseline_path,
        baseline_id=baseline_id,
        dataset_id=dataset_id,
        episode_count=len(predictions),
        fingerprint=fingerprint,
    )

    logger.info(f"Baseline materialized: {baseline_path}")
    logger.info(f"Baseline metadata: {baseline_path_file}")
    logger.info(f"Fingerprint: {fingerprint_path}")
    logger.info(f"Metrics: {metrics_path}")

    # Cleanup
    provider.cleanup()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Materialize a baseline from current system state (RFC-041 Phase 0)."
    )
    parser.add_argument(
        "--baseline-id",
        type=str,
        required=True,
        help="Unique baseline identifier (e.g., 'bart_led_baseline_v1')",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Dataset identifier (e.g., 'indicator_v1')",
    )
    parser.add_argument(
        "--experiment-config",
        type=str,
        default=None,
        help="Optional path to experiment config YAML",
    )
    parser.add_argument(
        "--preprocessing-profile",
        type=str,
        default=None,
        help="Optional preprocessing profile ID (e.g., 'cleaning_v3')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for baseline (default: benchmarks/baselines)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        action="append",
        default=None,
        help="Reference ID for vs_reference metrics (can be specified multiple times)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Validate API key if needed
    if args.experiment_config:
        try:
            cfg = load_experiment_config(args.experiment_config)
            if cfg.backend.type == "openai" and not os.getenv("OPENAI_API_KEY"):
                logger.error(
                    "OPENAI_API_KEY environment variable not set. "
                    "Export it before running this script."
                )
                sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load experiment config: {e}")
            sys.exit(1)

    # Materialize baseline
    try:
        materialize_baseline(
            baseline_id=args.baseline_id,
            dataset_id=args.dataset_id,
            experiment_config_path=args.experiment_config,
            preprocessing_profile=args.preprocessing_profile,
            output_dir=args.output_dir,
            reference_ids=args.reference,
        )
    except Exception as e:
        logger.error(f"Baseline materialization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
