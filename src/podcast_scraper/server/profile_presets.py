"""List packaged pipeline profile names under ``config/profiles/``.

Discovery order matches ``Config._resolve_profile`` directory candidates: cwd-relative
``config/profiles`` first, then repo-root ``config/profiles`` (union, de-duplicated).
"""

from __future__ import annotations

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


def list_packaged_profile_names() -> list[str]:
    """Sorted preset basenames (no ``.yaml``) for operator UI.

    Union of ``*.yaml`` stems from every existing ``config/profiles`` directory
    (cwd-relative then repo — mirrors ``Config._resolve_profile`` lookup roots).
    Excludes ``*.example.yaml`` (stem ends with ``.example``).
    """
    names: set[str] = set()
    for d in _profile_directories():
        for p in d.glob("*.yaml"):
            stem = p.stem
            if _stem_allowed(stem):
                names.add(stem)
    return sorted(names)
