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
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.log_redaction import format_exception_for_log
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
    #
    # A1 (2026-08-12): ``whisper_model`` is a MISNOMER retained for backward compatibility.
    # It holds the actual transcription model for whichever provider ran — so a Deepgram run
    # stamps ``whisper_model="nova-3"``. That mislabelling caused a false "wrong profile"
    # scare during a provenance audit, and makes it impossible to select one engine's
    # episodes for reprocessing in a mixed corpus.
    #
    # Prefer the provider-neutral pair below. Both are populated on every run; the legacy
    # field is kept so existing readers and on-disk manifests keep working. Do not add new
    # readers of ``whisper_model``.
    # 2026-08-28: WHICH profile produced this run, plus the diarization routing. The manifest
    # is the per-run reproducibility record and could not previously answer "what profile was
    # this?" at all — you inferred it from the run's argv and hoped the corpus config had not
    # been edited since. With per-feed and per-request profile overrides that inference is not
    # sound, so the resolved values are recorded here.
    profile: Optional[str] = None
    diarization_provider: Optional[str] = None
    diarization_model: Optional[str] = None
    transcription_provider: Optional[str] = None
    transcription_model: Optional[str] = None

    whisper_model: Optional[str] = None  # legacy alias of transcription_model — see above
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


#: Captured ONCE per process, on first use. See ``_get_git_info``.
_GIT_INFO: Optional[tuple[Optional[str], Optional[str], bool]] = None
_GIT_INFO_LOCK = threading.Lock()


def _get_git_info() -> tuple[Optional[str], Optional[str], bool]:
    """Get git commit SHA, branch, and dirty status — captured ONCE per process.

    The capture is process-wide on purpose. This probe answers "which code produced this
    artifact", and the code executing a run is whatever was on disk when the process started:
    Python imports its modules once at startup, so an edit or a deploy landing mid-run does not
    change the running code. Re-reading the working tree later answers a different question —
    what HEAD is right now — and silently labels the artifact with a commit that never touched
    it.

    Observed 2026-08-16 in a 14-episode acceptance run, where a commit landed mid-run::

        15:27:17  e055286   NVIDIA
        15:35:20  e055286   a16z
        15:40:03  2ceb653   Hard Fork     <- stamped with a commit made mid-run

    All four episodes were executed by e055286's code; the fourth manifest's SHA was false. In
    production the same thing happens whenever a deploy lands during a corpus pass, which takes
    hours. ADR-132 makes this THE field you consult when an artifact looks wrong, so it gets
    read exactly when something is already suspicious — a field that can disagree with reality
    is worse than no field at all.

    Caching here rather than in each caller also keeps the run manifest and the per-episode
    manifests consistent with one another; two independent caches could still straddle a commit.

    Thread-safe: episodes are processed concurrently, and racing threads would each shell out to
    git for a value that must be identical anyway.

    Returns:
        Tuple of (commit_sha, branch, dirty)
    """
    global _GIT_INFO
    if _GIT_INFO is None:
        with _GIT_INFO_LOCK:
            if _GIT_INFO is None:
                _GIT_INFO = _probe_git_info()
    return _GIT_INFO


def reset_git_info_cache() -> None:
    """Drop the captured git info (tests only).

    Production never calls this: re-capturing mid-run would reintroduce exactly the drift the
    cache exists to prevent.
    """
    global _GIT_INFO
    with _GIT_INFO_LOCK:
        _GIT_INFO = None


#: Baked into the pipeline image at build time. See ``_git_info_from_env``.
GIT_SHA_ENV = "PODCAST_GIT_SHA"
GIT_BRANCH_ENV = "PODCAST_GIT_BRANCH"
GIT_DIRTY_ENV = "PODCAST_GIT_DIRTY"


def _git_info_from_env() -> Optional[tuple[Optional[str], Optional[str], bool]]:
    """Provenance baked in at image build time, or ``None`` if this is not a built image.

    WHY THIS EXISTS: in the container that production actually runs, shelling out to git returns
    nothing. ``.dockerignore`` excludes ``.git/`` from the build context and the runtime stage
    never installs the git binary, so ``git rev-parse`` raises FileNotFoundError and the probe
    below reports ``(None, None, False)``. Every manifest written by the pipeline image recorded
    ``git_sha: null``.

    That is worse than it sounds. ADR-132 makes ``git_sha`` the exact-code backstop — THE field
    you consult when an artifact looks wrong and you need to know which code produced it. It has
    therefore been absent at exactly the moments it was designed for. The 2026-08-16 acceptance
    run showed a clean single SHA, which read as proof the provenance chain worked; that run
    executed from source, where ``.git`` is present, so it proved nothing about the image.

    Fixing it by un-ignoring ``.git/`` would bloat the build context AND still need the git
    binary in the runtime image. Passing the SHA in as a build arg costs neither.

    A source checkout sets none of these variables and falls through to the git probe, so dev
    boxes and CI keep working exactly as before.
    """
    sha = (os.environ.get(GIT_SHA_ENV) or "").strip()
    if not sha:
        return None
    branch = (os.environ.get(GIT_BRANCH_ENV) or "").strip() or None
    dirty = (os.environ.get(GIT_DIRTY_ENV) or "").strip().lower() in {"1", "true", "yes"}
    return sha, branch, dirty


def _probe_git_info() -> tuple[Optional[str], Optional[str], bool]:
    """Provenance for (commit_sha, branch, dirty). Call ``_get_git_info`` instead.

    Prefers the build-time environment (a built image), falls back to shelling out to git (a
    source checkout).
    """
    from_env = _git_info_from_env()
    if from_env is not None:
        return from_env
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
        logger.warning("Failed to calculate config hash: %s", format_exception_for_log(e))
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
    # LongT5 and FLAN-T5 (e.g. hybrid_ml map/reduce) use get_pinned_revision_for_model
    revision = getattr(config_constants, "get_pinned_revision_for_model", lambda _: None)(
        model_name
    )
    return revision


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


def _get_remote_accelerator_info(cfg: Any) -> Optional[str]:
    """Which REMOTE accelerator endpoints this run used, when the GPU is not local.

    2026-08-30: ``gpu_info`` was None on every prod run, including the DGX ones whose entire
    point was the GPU — because ``_get_gpu_info`` asks torch about the LOCAL machine, and the
    pipeline container runs on a VPS with no GPU. The accelerator is remote (tailnet DGX:
    faster-whisper, pyannote, vLLM), so the manifest recorded "no GPU" for runs that were
    almost entirely GPU work. Record the endpoints actually addressed, so a manifest can still
    answer "what hardware produced this episode".
    """
    parts = []
    host = getattr(cfg, "dgx_tailnet_host", None)
    if host:
        parts.append(f"dgx_host={host}")
    for label, field in (
        ("vllm", "vllm_api_base"),
        ("litellm", "litellm_api_base"),
    ):
        base = getattr(cfg, field, None)
        if base:
            parts.append(f"{label}={base}")
    for label, field in (
        ("asr", "transcription_provider"),
        ("diarization", "diarization_provider"),
    ):
        prov = getattr(cfg, field, None)
        if prov and "dgx" in str(prov):
            parts.append(f"{label}_remote={prov}")
    return "; ".join(parts) or None


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
    # Local GPU when there is one; otherwise the REMOTE accelerators this run addressed.
    gpu_info = _get_gpu_info() or _get_remote_accelerator_info(cfg)

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

    # Get model information from config. Resolve the ACTUAL transcription model for the configured
    # provider (dgx_whisper_model for DGX, moss_model, etc.) — reading cfg.whisper_model directly
    # stamped the unused local default (base.en) on every DGX run.
    from ..utils.provider_metrics import transcription_model_for_cfg

    # A1: resolve the ACTUAL provider + model, and record them under provider-neutral names.
    # ``whisper_model`` keeps receiving the same value purely for backward compatibility —
    # a Deepgram run has always stamped whisper_model="nova-3" there, which is what made
    # provenance audits misread the engine.
    transcription_model = transcription_model_for_cfg(cfg) or getattr(cfg, "whisper_model", None)
    transcription_provider = getattr(cfg, "transcription_provider", None)
    transcription_provider = str(transcription_provider) if transcription_provider else None
    _profile = getattr(cfg, "profile", None)
    _profile = str(_profile) if _profile else None
    _diar_provider = getattr(cfg, "diarization_provider", None)
    _diar_provider = str(_diar_provider) if _diar_provider else None
    _diar_model = getattr(cfg, "dgx_diarize_model", None) or getattr(
        cfg, "deepgram_diarization_model", None
    )
    _diar_model = str(_diar_model) if _diar_model else None
    whisper_model = transcription_model
    summary_model = getattr(cfg, "summary_model", None)
    reduce_model = getattr(cfg, "summary_reduce_model", None)

    # Resolve model revisions (same pinning as summarizer; Issue #429)
    whisper_model_revision = None  # Whisper revisions not pinned in config_constants
    summary_model_revision = _revision_for_summary_model(summary_model)
    reduce_model_revision = _revision_for_summary_model(reduce_model)

    # Get device configuration
    whisper_device = getattr(cfg, "whisper_device", None)
    summary_device = getattr(cfg, "summary_device", None)

    # Generation parameters, resolved for the provider that ACTUALLY RAN (2026-08-30).
    # These were read from ``cfg.temperature`` / ``cfg.seed``, which do not exist — every
    # temperature and seed field in Config is provider-namespaced (litellm_temperature,
    # vllm_summary_seed, ...). So both landed None in every manifest ever written, and a
    # manifest that claims to describe a reproducible run was silently missing its two
    # determinism knobs. Resolve from the active summary provider instead.
    _sp = str(getattr(cfg, "summary_provider", "") or "")
    temperature = getattr(cfg, f"{_sp}_temperature", None) if _sp else None
    seed = getattr(cfg, f"{_sp}_summary_seed", None) if _sp else None
    if seed is None:
        seed = getattr(cfg, "seed", None)

    # Who triggered this run. ``USER``/``USERNAME`` are unset inside the pipeline container,
    # so every containerised run recorded "unknown" — i.e. exactly the production runs whose
    # provenance matters most. Prefer an explicit trigger identity when the spawner supplies
    # one, then the OS user, and record the EXECUTION CONTEXT rather than giving up.
    created_by = (
        os.getenv("PODCAST_RUN_TRIGGERED_BY")
        or os.getenv("USER")
        or os.getenv("USERNAME")
        or ("container:" + os.getenv("HOSTNAME", "unknown"))
    )

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
        profile=_profile,
        diarization_provider=_diar_provider,
        diarization_model=_diar_model,
        transcription_provider=transcription_provider,
        transcription_model=transcription_model,
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
