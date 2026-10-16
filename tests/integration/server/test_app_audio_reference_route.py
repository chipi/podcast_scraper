"""GET /api/app/episodes/{slug}/audio-reference (RFC-129, #2359)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from tests.integration.server.test_app_moments_route import _client, _episode, _slug  # noqa: E402

pytestmark = [pytest.mark.integration]

REFERENCE = {
    "version": 1,
    "sample_rate": 8000,
    "hop": 160,
    "nfft": 512,
    "bands": 16,
    "scale": 20.0,
    "window_s": 10.0,
    "duration_s": 1800.0,
    "guid": "g-1",
    "reach_s": 90.0,
    "probe_s": 12.0,
    "checkpoints": [
        {"t": 60.0, "kind": "talk_start", "frames": "AAEC", "long": "AAECAwQF"},
        {"t": 360.0, "kind": "interval", "frames": "BAUG"},
    ],
}


def test_serves_the_reference_next_to_the_transcript(tmp_path: Path) -> None:
    _episode(tmp_path, "0001-a", "ep1", insights=1, segments=True)
    (tmp_path / "transcripts" / "0001-a.audioref.json").write_text(
        json.dumps(REFERENCE), encoding="utf-8"
    )
    res = _client(tmp_path).get(f"/api/app/episodes/{_slug(tmp_path, 'ep1')}/audio-reference")
    assert res.status_code == 200
    body = res.json()
    # The long windows stay out of the reference; each checkpoint says whether it has one.
    assert body["checkpoints"] == [
        {"t": 60.0, "kind": "talk_start", "frames": "AAEC", "has_long": True},
        {"t": 360.0, "kind": "interval", "frames": "BAUG", "has_long": False},
    ]
    assert body["duration_s"] == 1800.0
    assert body["reach_s"] == 90.0
    assert "guid" not in body


def test_serves_one_long_window_on_request(tmp_path: Path) -> None:
    _episode(tmp_path, "0001-a", "ep1", insights=1, segments=True)
    (tmp_path / "transcripts" / "0001-a.audioref.json").write_text(
        json.dumps(REFERENCE), encoding="utf-8"
    )
    client = _client(tmp_path)
    base = f"/api/app/episodes/{_slug(tmp_path, 'ep1')}/audio-reference/long"
    res = client.get(f"{base}/0")
    assert res.status_code == 200
    assert res.json() == {
        "episode_slug": _slug(tmp_path, "ep1"),
        "index": 0,
        "t": 60.0,
        "frames": "AAECAwQF",
    }
    assert client.get(f"{base}/1").status_code == 404  # an interval checkpoint has none
    assert client.get(f"{base}/7").status_code == 404


def test_404_without_a_reference(tmp_path: Path) -> None:
    _episode(tmp_path, "0001-a", "ep1", insights=1, segments=True)
    res = _client(tmp_path).get(f"/api/app/episodes/{_slug(tmp_path, 'ep1')}/audio-reference")
    assert res.status_code == 404
