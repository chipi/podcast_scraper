"""Run manifest generation for reproducibility tracking.

This module creates run_manifest.json files that capture all information needed
to reproduce a pipeline run, including git SHA, config hash, environment details,
and model information.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.redaction import redact_secrets

logger = logging.getLogger(__name__)


@dataclass
class RunManifest:
    """Run manifest for reproducibility tracking."""

    # Run identification
    run_id: str
    created_at: str
    created_by: str

    # Version control
    git_commit_sha: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False

    # Configuration
    config_sha256: Optional[str] = None
    config_path: Optional[str] = None
    full_config_string: Optional[str] = None  # Full provider/model config string (for reference)

    # Environment
    python_version: str = ""
    os_name: str = ""
    os_version: str = ""
    cpu_info: Optional[str] = None
    gpu_info: Optional[str] = None

    # Dependencies
    torch_version: Optional[str] = None
    transformers_version: Optional[str] = None
    whisper_version: Optional[str] = None

    # Models used
    whisper_model: Optional[str] = None
    whisper_model_revision: Optional[str] = None
    summary_model: Optional[str] = None
    summary_model_revision: Optional[str] = None
    reduce_model: Optional[str] = None
    reduce_model_revision: Optional[str] = None

    # Device configuration
    whisper_device: Optional[str] = None
    summary_device: Optional[str] = None

    # Generation parameters
    temperature: Optional[float] = None
    seed: Optional[int] = None

    # Schema version (Issue #379) - must be last due to dataclass field ordering
    schema_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert manifest to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save_to_file(self, filepath: str | Path) -> None:
        """Save manifest to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        manifest_json = self.to_json()
        filepath.write_text(manifest_json, encoding="utf-8")
        logger.info(f"Run manifest saved to: {filepath}")


def _get_git_info() -> tuple[Optional[str], Optional[str], bool]:
    """Get git commit SHA, branch, and dirty status.

    Returns:
        Tuple of (commit_sha, branch, dirty)
    """
    try:
        # Get commit SHA
        commit_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )

        # Get branch name
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )

        # Check if working tree is dirty
        dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
            != ""
        )

        return commit_sha, branch, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Git not available or not in a git repo
        return None, None, False


def _get_config_hash(cfg: Any) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Calculate SHA256 hash of configuration.

    Args:
        cfg: Config object

    Returns:
        Tuple of (config_sha256, config_path, full_config_string)
    """
    try:
        # Get config as dict and redact secrets before serialization
        config_dict = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
        # Redact secrets recursively (key-based and pattern-based detection)
        config_dict_redacted = redact_secrets(config_dict, redact_patterns=True)
        config_json = json.dumps(config_dict_redacted, sort_keys=True, default=str)
        config_sha256 = hashlib.sha256(config_json.encode("utf-8")).hexdigest()

        # Get config path if available
        config_path = getattr(cfg, "config_path", None)

        return config_sha256, config_path, config_json
    except Exception as e:
        logger.warning(f"Failed to calculate config hash: {e}")
        return None, None, None


def _revision_for_summary_model(model_name: Optional[str]) -> Optional[str]:
    """Resolve pinned revision for a summary/reduce model (same logic as summarizer).

    Args:
        model_name: Hugging Face model identifier or None

    Returns:
        Pinned revision string from config_constants, or None
    """
    if not model_name:
        return None
    model_lower = model_name.lower()
    try:
        from podcast_scraper import config_constants
    except ImportError:
        return None
    if "pegasus" in model_lower:
        return getattr(config_constants, "PEGASUS_CNN_DAILYMAIL_REVISION", None)
    if "led-base-16384" in model_lower or model_name == "allenai/led-base-16384":
        return getattr(config_constants, "LED_BASE_16384_REVISION", None)
    if "led-large-16384" in model_lower or model_name == "allenai/led-large-16384":
        return getattr(config_constants, "LED_LARGE_16384_REVISION", None)
    return None


def _get_gpu_info() -> Optional[str]:
    """Get GPU information if available.

    Returns:
        GPU info string or None
    """
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            return f"CUDA: {', '.join(gpu_names)}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "MPS: Apple Silicon GPU"
    except ImportError:
        pass
    return None


def create_run_manifest(cfg: Any, output_dir: str, run_id: Optional[str] = None) -> RunManifest:
    """Create run manifest from configuration and environment.

    Args:
        cfg: Configuration object
        output_dir: Output directory path
        run_id: Optional run identifier

    Returns:
        RunManifest object
    """
    # Get git info
    git_commit_sha, git_branch, git_dirty = _get_git_info()

    # Get config hash
    config_sha256, config_path, full_config_string = _get_config_hash(cfg)

    # Get environment info
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    os_name = platform.system()
    os_version = platform.release()
    cpu_info = platform.processor() or platform.machine()

    # Get GPU info
    gpu_info = _get_gpu_info()

    # Get dependency versions
    torch_version = None
    transformers_version = None
    whisper_version = None

    try:
        import torch

        torch_version = getattr(torch, "__version__", None)
    except ImportError:
        pass

    try:
        import transformers

        transformers_version = getattr(transformers, "__version__", None)
    except ImportError:
        pass

    try:
        import whisper

        whisper_version = getattr(whisper, "__version__", None)
    except ImportError:
        pass

    # Get model information from config
    whisper_model = getattr(cfg, "whisper_model", None)
    summary_model = getattr(cfg, "summary_model", None)
    reduce_model = getattr(cfg, "summary_reduce_model", None)

    # Resolve model revisions (same pinning as summarizer; Issue #429)
    whisper_model_revision = None  # Whisper revisions not pinned in config_constants
    summary_model_revision = _revision_for_summary_model(summary_model)
    reduce_model_revision = _revision_for_summary_model(reduce_model)

    # Get device configuration
    whisper_device = getattr(cfg, "whisper_device", None)
    summary_device = getattr(cfg, "summary_device", None)

    # Get generation parameters
    temperature = getattr(cfg, "temperature", None)
    seed = getattr(cfg, "seed", None)

    # Get user info
    created_by = os.getenv("USER") or os.getenv("USERNAME") or "unknown"

    # Create manifest
    manifest = RunManifest(
        run_id=run_id or datetime.utcnow().isoformat() + "Z",
        created_at=datetime.utcnow().isoformat() + "Z",
        created_by=created_by,
        git_commit_sha=git_commit_sha,
        git_branch=git_branch,
        git_dirty=git_dirty,
        config_sha256=config_sha256,
        config_path=config_path,
        full_config_string=full_config_string,
        python_version=python_version,
        os_name=os_name,
        os_version=os_version,
        cpu_info=cpu_info,
        gpu_info=gpu_info,
        torch_version=torch_version,
        transformers_version=transformers_version,
        whisper_version=whisper_version,
        whisper_model=whisper_model,
        whisper_model_revision=whisper_model_revision,
        summary_model=summary_model,
        summary_model_revision=summary_model_revision,
        reduce_model=reduce_model,
        reduce_model_revision=reduce_model_revision,
        whisper_device=whisper_device,
        summary_device=summary_device,
        temperature=temperature,
        seed=seed,
    )

    return manifest
