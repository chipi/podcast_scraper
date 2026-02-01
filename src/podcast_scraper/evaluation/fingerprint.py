"""Provider fingerprinting for reproducibility and debugging.

This module implements deep provider fingerprinting (ADR-027) to capture
all environment variables that affect AI model outputs, enabling perfect
reproducibility and easier debugging of quality regressions.

See ADR-027 for design rationale.

Note: This module was moved from podcast_scraper.evaluation.fingerprint.py
because it's primarily used by evaluation scripts to track experimental runs,
not by provider implementations themselves.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProviderFingerprint:
    """Fingerprint of provider environment for reproducibility.

    Captures all variables that could affect AI model outputs:
    - Model details (names, versions, hashes)
    - Hardware (device, precision)
    - Software (library versions)
    - Git state (commit hash, dirty status)

    Attributes:
        model_name: Model identifier (e.g., "facebook/bart-large-cnn", "gpt-4o-mini")
        model_version: Model version/revision (if available)
        model_hash: SHA256 hash of model weights (for local models)
        device: Device name (e.g., "mps", "cuda", "cpu")
        device_name: Hardware device name (e.g., "M1 Max", "RTX 4090")
        precision: Numerical precision (e.g., "fp16", "fp32", "int8")
        package_version: Version of podcast_scraper package
        git_commit: Git commit hash (short, 7 characters)
        git_dirty: Whether repository has uncommitted changes
        library_versions: Dictionary of library versions (torch, transformers, etc.)
        preprocessing_profile: Preprocessing profile ID (e.g., "cleaning_v3")
        fingerprint_hash: SHA256 hash of the fingerprint (for quick comparison)

    Example:
        >>> fingerprint = generate_provider_fingerprint(
        ...     model_name="facebook/bart-large-cnn",
        ...     device="mps",
        ...     preprocessing_profile="cleaning_v3"
        ... )
        >>> print(fingerprint.fingerprint_hash)
    """

    model_name: str
    model_version: Optional[str] = None
    model_hash: Optional[str] = None
    device: Optional[str] = None
    device_name: Optional[str] = None
    precision: Optional[str] = None
    package_version: str = "unknown"
    git_commit: Optional[str] = None
    git_dirty: bool = False
    library_versions: Dict[str, str] = field(default_factory=dict)
    preprocessing_profile: Optional[str] = None
    fingerprint_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert fingerprint to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_hash": self.model_hash,
            "device": self.device,
            "device_name": self.device_name,
            "precision": self.precision,
            "package_version": self.package_version,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "library_versions": self.library_versions,
            "preprocessing_profile": self.preprocessing_profile,
            "fingerprint_hash": self.fingerprint_hash,
        }

    def compute_hash(self) -> str:
        """Compute SHA256 hash of fingerprint for quick comparison."""
        # Create deterministic string representation
        parts = [
            f"model:{self.model_name}",
            f"version:{self.model_version or 'none'}",
            f"hash:{self.model_hash or 'none'}",
            f"device:{self.device or 'none'}",
            f"device_name:{self.device_name or 'none'}",
            f"precision:{self.precision or 'none'}",
            f"package:{self.package_version}",
            f"commit:{self.git_commit or 'none'}",
            f"dirty:{self.git_dirty}",
            f"profile:{self.preprocessing_profile or 'none'}",
        ]
        # Add library versions in sorted order for determinism
        lib_versions = sorted(f"{lib}:{ver}" for lib, ver in self.library_versions.items())
        parts.extend(lib_versions)

        fingerprint_str = "|".join(parts)
        return hashlib.sha256(fingerprint_str.encode("utf-8")).hexdigest()[:16]


def _get_git_commit() -> tuple[Optional[str], bool]:
    """Get git commit hash and dirty status.

    Returns:
        Tuple of (commit_hash, is_dirty)
        Returns (None, False) if not in a git repository or git is unavailable.
    """
    try:
        # Get commit hash (short, 7 characters)
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode != 0:
            return None, False
        commit = result.stdout.strip()

        # Check if repository is dirty
        diff_result: subprocess.CompletedProcess[bytes] = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        is_dirty = diff_result.returncode != 0

        return commit, is_dirty
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        # Git not available or not in a git repository
        return None, False


def _get_package_version() -> str:
    """Get podcast_scraper package version."""
    try:
        from .. import __version__

        return __version__
    except ImportError:
        return "unknown"


def _get_library_versions() -> Dict[str, str]:
    """Get versions of key ML libraries.

    Returns:
        Dictionary mapping library names to versions.
    """
    versions: Dict[str, str] = {}

    # Core ML libraries
    libraries = [
        "torch",
        "transformers",
        "whisper",
        "spacy",
        "numpy",
        "accelerate",
    ]

    for lib_name in libraries:
        try:
            lib = __import__(lib_name)
            if hasattr(lib, "__version__"):
                versions[lib_name] = str(lib.__version__)
        except ImportError:
            # Library not installed, skip
            pass

    return versions


def _get_device_name(device: Optional[str]) -> Optional[str]:
    """Get human-readable device name.

    Args:
        device: Device identifier ("cpu", "cuda", "mps", etc.)

    Returns:
        Human-readable device name or None if unavailable.
    """
    if device is None:
        return None

    if device == "cpu":
        return platform.processor() or platform.machine()
    elif device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except ImportError:
            pass
        return "CUDA (unknown device)"
    elif device == "mps":
        # macOS Metal Performance Shaders
        return platform.machine()  # e.g., "arm64" for Apple Silicon
    else:
        return device


def _get_precision(device: Optional[str]) -> Optional[str]:
    """Get numerical precision used by models.

    Args:
        device: Device identifier

    Returns:
        Precision string ("fp32", "fp16", "int8", etc.) or None if unknown.
    """
    # For now, we default to fp32/fp16 based on device
    # This can be enhanced to actually detect precision from model
    if device == "cuda":
        # CUDA typically uses fp16 for performance
        return "fp16"
    elif device == "mps":
        # MPS typically uses fp32
        return "fp32"
    elif device == "cpu":
        return "fp32"
    else:
        return None


def generate_provider_fingerprint(
    model_name: str,
    model_version: Optional[str] = None,
    model_hash: Optional[str] = None,
    device: Optional[str] = None,
    preprocessing_profile: Optional[str] = None,
) -> ProviderFingerprint:
    """Generate provider fingerprint for reproducibility.

    Args:
        model_name: Model identifier (required)
        model_version: Model version/revision (optional)
        model_hash: SHA256 hash of model weights (optional, for local models)
        device: Device identifier ("cpu", "cuda", "mps", etc.)
        preprocessing_profile: Preprocessing profile ID (e.g., "cleaning_v3")

    Returns:
        ProviderFingerprint instance with all environment details captured.

    Example:
        >>> fingerprint = generate_provider_fingerprint(
        ...     model_name="facebook/bart-large-cnn",
        ...     device="mps",
        ...     preprocessing_profile="cleaning_v3"
        ... )
        >>> print(fingerprint.fingerprint_hash)
    """
    git_commit, git_dirty = _get_git_commit()
    package_version = _get_package_version()
    library_versions = _get_library_versions()
    device_name = _get_device_name(device)
    precision = _get_precision(device)

    fingerprint = ProviderFingerprint(
        model_name=model_name,
        model_version=model_version,
        model_hash=model_hash,
        device=device,
        device_name=device_name,
        precision=precision,
        package_version=package_version,
        git_commit=git_commit,
        git_dirty=git_dirty,
        library_versions=library_versions,
        preprocessing_profile=preprocessing_profile,
    )

    # Compute hash after all fields are set
    fingerprint.fingerprint_hash = fingerprint.compute_hash()

    return fingerprint
