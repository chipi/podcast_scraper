"""Resolve viewer operator YAML path (per-corpus vs ``serve --config-file`` override)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

VIEWER_OPERATOR_BASENAME = "viewer_operator.yaml"


def viewer_operator_yaml_path(app: Any, corpus_root: Path) -> Path:
    """Return operator file path for ``corpus_root``.

    When ``serve`` was started with ``--config-file`` / ``PODCAST_SERVE_CONFIG_FILE``, all
    corpora share that single resolved file. Otherwise the file lives next to
    ``feeds.spec.yaml``: ``<corpus>/viewer_operator.yaml``.
    """
    fixed = getattr(app.state, "operator_config_fixed_path", None)
    if fixed is not None:
        return Path(fixed)
    return corpus_root / VIEWER_OPERATOR_BASENAME


def packaged_viewer_operator_example_path() -> Path | None:
    """Path to ``config/examples/viewer_operator.example.yaml`` when present (repo / sdist tree)."""
    try:
        import podcast_scraper as _pkg
    except ImportError:
        return None
    # ``__file__`` is ``…/src/podcast_scraper/__init__.py`` (dev) or
    # ``…/site-packages/podcast_scraper/__init__.py``.
    here = Path(_pkg.__file__).resolve().parent
    candidate = here.parents[1] / "config" / "examples" / "viewer_operator.example.yaml"
    return candidate if candidate.is_file() else None
