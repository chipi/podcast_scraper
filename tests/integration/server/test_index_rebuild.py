"""POST /api/index/rebuild (GitHub #507 follow-up).

Requires ``fastapi`` and FAISS (``pip install -e '.[server]'``).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("faiss")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def test_index_rebuild_requires_corpus_path(tmp_path: Path) -> None:
    app = create_app(None, static_dir=False)
    client = TestClient(app)
    response = client.post("/api/index/rebuild")
    assert response.status_code == 400


def test_index_rebuild_202_when_faiss_available(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "a.metadata.json").write_text(
        '{"episode": {"episode_id": "e1"}, "feed": {}}',
        encoding="utf-8",
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    with patch("podcast_scraper.server.routes.index_rebuild.index_corpus"):
        response = client.post("/api/index/rebuild", params={"path": str(tmp_path)})
    assert response.status_code == 202
    body = response.json()
    assert body.get("accepted") is True
    assert body.get("rebuild") is False
    assert Path(body["corpus_path"]).resolve() == tmp_path.resolve()


def test_index_rebuild_409_when_already_running(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "a.metadata.json").write_text(
        '{"episode": {"episode_id": "e1"}, "feed": {}}',
        encoding="utf-8",
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    hold = threading.Event()
    released = threading.Event()

    def _block(*_a: object, **_k: object) -> None:
        hold.set()
        released.wait(timeout=5.0)

    with patch(
        "podcast_scraper.server.routes.index_rebuild.index_corpus",
        side_effect=_block,
    ):
        worker = threading.Thread(
            target=lambda: client.post("/api/index/rebuild", params={"path": str(tmp_path)}),
        )
        worker.start()
        assert hold.wait(timeout=5.0)
        time.sleep(0.05)
        second = client.post("/api/index/rebuild", params={"path": str(tmp_path)})
        assert second.status_code == 409
        released.set()
        worker.join(timeout=5.0)
