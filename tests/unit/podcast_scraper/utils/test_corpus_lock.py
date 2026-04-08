"""Tests for corpus parent file lock (multi-feed)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from filelock import Timeout

from podcast_scraper.utils.corpus_lock import corpus_lock_enabled, corpus_parent_lock


@pytest.mark.unit
def test_corpus_lock_disabled_via_env() -> None:
    with patch.dict(os.environ, {"PODCAST_SCRAPER_CORPUS_LOCK": "0"}):
        assert corpus_lock_enabled() is False


@pytest.mark.unit
def test_corpus_parent_lock_acquire_release(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    with patch.dict(os.environ, {"PODCAST_SCRAPER_CORPUS_LOCK": "1"}):
        with corpus_parent_lock(root):
            pass
    assert root.is_dir()


@pytest.mark.unit
@patch("filelock.FileLock")
def test_corpus_parent_lock_timeout_raises(mock_filelock_class: MagicMock, tmp_path: Path) -> None:
    mock_lock = MagicMock()
    mock_filelock_class.return_value = mock_lock
    mock_lock.acquire.side_effect = Timeout(mock_lock)
    with patch.dict(os.environ, {"PODCAST_SCRAPER_CORPUS_LOCK": "1"}):
        with pytest.raises(RuntimeError, match="locked"):
            with corpus_parent_lock(tmp_path / "locked_corpus"):
                pass
