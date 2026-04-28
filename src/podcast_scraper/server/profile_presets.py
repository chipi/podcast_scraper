"""List packaged pipeline profile names under ``config/profiles/``.

Discovery order matches ``Config._resolve_profile`` directory candidates: cwd-relative
``config/profiles`` first, then repo-root ``config/profiles`` (union, de-duplicated).

Filtering: when ``PODCAST_AVAILABLE_PROFILES`` is set (comma-separated allowlist of
profile names), the returned list is intersected with the allowlist. Used by the
pre-prod overlay (RFC-081) to restrict the operator UI to profiles whose backing
pipeline image is actually published to GHCR — Phase 1 publishes ``pipeline-llm``
only, so the env defaults to ``cloud_thin``. Unset env = no filtering (dev / CI).
"""

from __future__ import annotations

import os
from pathlib import Path


def _profile_directories() -> list[Path]:
    """Return resolved ``config/profiles`` dirs to scan (same roots as preset file load)."""
    dirs: list[Path] = []
    cwd_prof = Path("config/profiles")
    if cwd_prof.is_dir():
        dirs.append(cwd_prof.resolve())
    # src/podcast_scraper/server/profile_presets.py → parents[3] = repo root
    here = Path(__file__).resolve()
    root = here.parents[3]
    repo_prof = (root / "config" / "profiles").resolve()
    if repo_prof.is_dir() and repo_prof not in dirs:
        dirs.append(repo_prof)
    return dirs


def _stem_allowed(stem: str) -> bool:
    # Example stubs only (e.g. profile_freeze.example.yaml → stem profile_freeze.example).
    if stem.endswith(".example"):
        return False
    return True


def _env_allowlist() -> set[str] | None:
    """Parse ``PODCAST_AVAILABLE_PROFILES`` into an allowlist set, or None when unset.

    Empty / whitespace-only env value = treated as unset (no filtering). Profile
    names are stripped; empty entries (trailing commas, double commas) ignored.
    """
    raw = os.environ.get("PODCAST_AVAILABLE_PROFILES", "").strip()
    if not raw:
        return None
    allowed = {n.strip() for n in raw.split(",") if n.strip()}
    return allowed or None


def validate_operator_profile_allowed(operator_yaml: Path) -> str | None:
    """Reject job submission when operator's profile is outside the allowlist.

    Defense-in-depth for RFC-081 §Layer 1: even if a stale viewer bundle
    sends a profile that's hidden in the operator-config dropdown, the api
    must still refuse to enqueue a pipeline run that would crash several
    minutes in (no published image for that profile / wrong extras tier).

    Reads the operator yaml's ``profile:`` line; when non-empty AND not in
    the post-filter ``list_packaged_profile_names()``, raises ``ValueError``.
    Returns the validated profile name (or ``None`` if the operator file is
    missing or has no profile line — that's the no-op default that lets
    ``Config._resolve_profile`` fall back).

    The route handler catches ``ValueError`` and turns it into HTTP 400 —
    same pattern as ``validate_operator_pipeline_extras``.
    """
    # Local import — avoids a circular import at module load time
    # (operator_yaml_profile imports nothing from server, but defensive).
    from podcast_scraper.server.operator_yaml_profile import split_operator_yaml_profile

    try:
        text = operator_yaml.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    profile_name, _body = split_operator_yaml_profile(text)
    pn = profile_name.strip()
    if not pn:
        return None
    allowed = list_packaged_profile_names()
    if pn not in allowed:
        # Surface the allowlist so the operator can see what they should
        # have picked — silently 400'ing without telling them what's
        # available is a hostile UX.
        raise ValueError(
            f"profile {pn!r} is not in the available profiles for this environment "
            f"(allowed: {', '.join(allowed) or '(none)'}). "
            f"Re-pick from the operator Profile menu."
        )
    return pn


def list_packaged_profile_names() -> list[str]:
    """Sorted preset basenames (no ``.yaml``) for operator UI.

    Union of ``*.yaml`` stems from every existing ``config/profiles`` directory
    (cwd-relative then repo — mirrors ``Config._resolve_profile`` lookup roots).
    Excludes ``*.example.yaml`` (stem ends with ``.example``).

    When ``PODCAST_AVAILABLE_PROFILES`` is set, the list is intersected with that
    allowlist — used by pre-prod (RFC-081 §Layer 1) to hide profiles whose backing
    pipeline image isn't published. The intersection is taken **after** the on-disk
    discovery so we never advertise a profile that doesn't exist on disk, even if
    the operator typo'd a name into the env var.
    """
    names: set[str] = set()
    for d in _profile_directories():
        for p in d.glob("*.yaml"):
            stem = p.stem
            if _stem_allowed(stem):
                names.add(stem)
    allowed = _env_allowlist()
    if allowed is not None:
        names = names & allowed
    return sorted(names)
