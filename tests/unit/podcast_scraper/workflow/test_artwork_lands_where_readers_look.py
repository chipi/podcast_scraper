# mypy: disable-error-code="call-arg"
# Deliberate: Config(rss_url=...) — alias="rss"; populate-by-name accepts either at runtime.
"""Downloaded artwork must land where the readers resolve it: the corpus root (#2204).

The pipeline wrote each image under the RUN dir, and every reader resolves the stored relpath
against the CORPUS root. On prod (2026-10-01) that left 0 of 2,002 served episodes with
resolvable local art while 1.49 GB of copies sat in run dirs. These tests drive the writer and
the catalog's reader against one corpus, built the way the pipeline builds it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper import config
from podcast_scraper.server.corpus_catalog import _verified_artwork_relpath
from podcast_scraper.utils import filesystem
from podcast_scraper.utils.corpus_artwork import download_podcast_artwork
from podcast_scraper.workflow.metadata_generation import artwork_store_root

pytestmark = [pytest.mark.unit]

FEED_URL = "https://feed-a.example/rss"
JPEG = b"\xff\xd8\xff\xe0" + b"cover-art" * 50


def _corpus_run(corpus: Path, run: str) -> tuple[config.Config, str]:
    """A per-feed cfg rebased the way the corpus loops do it, and that run's own dir."""
    feed_dir = filesystem.corpus_feed_output_dir(str(corpus), FEED_URL)
    cfg = config.Config(rss_url=FEED_URL, output_dir=feed_dir)
    run_dir = Path(feed_dir) / run
    run_dir.mkdir(parents=True)
    return cfg, str(run_dir)


def _download(root: Path) -> str:
    with patch("podcast_scraper.utils.corpus_artwork.http_get", return_value=(JPEG, "image/jpeg")):
        rel = download_podcast_artwork(
            "https://cdn.example/cover.jpg", root, user_agent="t", timeout=5
        )
    assert rel
    return rel


def test_a_corpus_run_stores_art_at_the_corpus_root(tmp_path: Path) -> None:
    cfg, run_dir = _corpus_run(tmp_path, "run_a")
    assert artwork_store_root(cfg, run_dir) == tmp_path.resolve()


def test_the_catalog_resolves_what_the_pipeline_wrote(tmp_path: Path) -> None:
    """THE SEAM. The catalog only offers local art it can find under the corpus root."""
    cfg, run_dir = _corpus_run(tmp_path, "run_a")
    rel = _download(artwork_store_root(cfg, run_dir))
    assert _verified_artwork_relpath(tmp_path.resolve(), rel) == rel


def test_a_second_run_reuses_the_stored_image(tmp_path: Path) -> None:
    """Re-runs used to add a full copy of every image each time."""
    cfg, run_a = _corpus_run(tmp_path, "run_a")
    _cfg_b, run_b = _corpus_run(tmp_path, "run_b")
    rel_a = _download(artwork_store_root(cfg, run_a))
    rel_b = _download(artwork_store_root(cfg, run_b))
    assert rel_a == rel_b
    stored = [p for p in tmp_path.rglob("*.jpg") if p.is_file()]
    assert len(stored) == 1, stored
    assert stored[0].read_bytes() == JPEG
    assert hashlib.sha256(JPEG).hexdigest() in stored[0].name


def test_a_run_outside_any_corpus_keeps_its_own_dir(tmp_path: Path) -> None:
    """No corpus layout → nothing to share; the run dir stays the root it always was."""
    cfg = config.Config(rss_url=FEED_URL, output_dir=str(tmp_path / "out"))
    run_dir = tmp_path / "out" / "run_x"
    assert artwork_store_root(cfg, str(run_dir)) == run_dir.resolve()
