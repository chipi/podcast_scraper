"""Evaluation system for LLM experiments.

This package provides:
- Experiment configuration models
- Metrics computation and scoring
- Comparison and regression detection
- History tracking and reporting
- Provider fingerprinting for reproducibility
"""

# Re-export evaluation functions
from .comparator import compare_vs_baseline

# Re-export experiment config for convenience
from .config import (
    BackendConfig,
    DataConfig,
    ExperimentConfig,
    GenerationParams,
    HFBackendConfig,
    load_experiment_config,
    OpenAIBackendConfig,
    PromptConfig,
    TokenizeConfig,
)

# Re-export fingerprinting
from .fingerprint import generate_provider_fingerprint, ProviderFingerprint
from .regression import RegressionChecker, RegressionRule
from .scorer import score_run

__all__ = [
    # Config
    "BackendConfig",
    "DataConfig",
    "ExperimentConfig",
    "GenerationParams",
    "HFBackendConfig",
    "OpenAIBackendConfig",
    "PromptConfig",
    "TokenizeConfig",
    "load_experiment_config",
    # Evaluation
    "score_run",
    "compare_vs_baseline",
    "RegressionChecker",
    "RegressionRule",
    # Fingerprinting
    "ProviderFingerprint",
    "generate_provider_fingerprint",
]
