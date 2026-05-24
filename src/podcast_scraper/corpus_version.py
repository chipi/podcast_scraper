"""Corpus / code version contract (GitHub #796)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from podcast_scraper import __version__

CORPUS_MANIFEST_FILE = "corpus_manifest.json"

# Warn when on-disk corpus was produced below this semver.
MIN_SUPPORTED_CORPUS_CODE_VERSION = "2.6.0"


def resolve_git_commit_sha() -> str:
    """Best-effort git HEAD for pipeline ``produced_by`` stamps."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    sha = proc.stdout.strip()
    return sha if sha else "unknown"


def build_produced_by(*, produced_at: str) -> dict[str, str]:
    """Stamp written into ``corpus_manifest.json`` on each pipeline finalize."""
    sha = resolve_git_commit_sha()
    short = sha[:7] if len(sha) >= 7 else sha
    return {
        "code_version": __version__,
        "git_sha": short,
        "produced_at": produced_at,
    }


def read_produced_by(corpus_root: Path) -> dict[str, Any] | None:
    """Load ``produced_by`` from corpus manifest, with legacy ``tool_version`` fallback."""
    manifest = corpus_root / CORPUS_MANIFEST_FILE
    if not manifest.is_file():
        return None
    try:
        doc = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    produced_by = doc.get("produced_by")
    if isinstance(produced_by, dict):
        return produced_by
    tool_version = doc.get("tool_version")
    if isinstance(tool_version, str) and tool_version.strip():
        return {
            "code_version": tool_version.strip(),
            "git_sha": "unknown",
            "produced_at": doc.get("updated_at", ""),
        }
    return None


def corpus_code_version(produced_by: dict[str, Any] | None) -> str | None:
    """Extract ``code_version`` from a ``produced_by`` dict, if present."""
    if not produced_by:
        return None
    raw = produced_by.get("code_version")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def assess_corpus_version_compat(
    produced_by: dict[str, Any] | None,
    *,
    server_version: str = __version__,
    min_supported: str = MIN_SUPPORTED_CORPUS_CODE_VERSION,
) -> tuple[str | None, str | None]:
    """Return ``(corpus_code_version, warning_message)``."""
    corpus_ver = corpus_code_version(produced_by)
    if corpus_ver is None:
        return None, (
            f"Corpus has no produced_by stamp; server {server_version} supports "
            f"corpora from {min_supported}+. Re-run pipeline or "
            f"``make reprocess-corpus-from-transcripts`` to refresh artifacts."
        )
    try:
        if Version(corpus_ver) < Version(min_supported):
            return corpus_ver, (
                f"Corpus produced by {corpus_ver} is below server minimum "
                f"{min_supported}. Reprocess from transcripts before relying on "
                f"Library / Digest / Graph / Search surfaces."
            )
    except InvalidVersion:
        return corpus_ver, (
            f"Corpus produced_by.code_version {corpus_ver!r} is not a valid semver; "
            f"verify compatibility or reprocess."
        )
    return corpus_ver, None
