"""RFC-129: ``build-maps`` from the episode's reference and the stream as served now. No archive."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from podcast_scraper.audio_sync import drift, maps, references
from podcast_scraper.audio_sync.fingerprint import decode, SR

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]
CORPUS = REPO / "tests" / "fixtures" / "app-validation-corpus" / "v3"
AUDIO = REPO / "tests" / "fixtures" / "audio" / "v3"


def _encode(samples: np.ndarray, path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-"]
        + ["-b:a", "128k", str(path)],
        input=samples.astype(np.float32).tobytes(),
        check=True,
    )
    return path


def test_maps_a_drifted_episode_from_its_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "corpus"
    shutil.copytree(CORPUS / "feeds" / "p01", root / "feeds" / "p01")
    meta = next((root / "feeds" / "p01").glob("*/metadata/p01_e01.metadata.json"))
    rel = str(meta.relative_to(root))
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc["content"]["media_url"] = "https://host.example/p01_e01.mp3"
    meta.write_text(json.dumps(doc), encoding="utf-8")

    # The reference, as ingest writes it, from the audio we transcribed.
    from podcast_scraper.server.app_catalog_cache import cached_catalog

    row = next(r for r in cached_catalog(root) if r.metadata_relative_path == rel)
    assert references.write_for_episode(root, row, str(AUDIO / "p01_e01.mp3"), "g-1")

    # The stream as served now: a 30 s ad inserted at 120 s.
    original = decode(str(AUDIO / "p01_e01.mp3"))
    ad = decode(str(AUDIO / "p01_e02.mp3"))[10 * SR : 40 * SR]
    served = _encode(
        np.concatenate([original[: 120 * SR], ad, original[120 * SR :]]), tmp_path / "served.mp3"
    )
    monkeypatch.setattr(drift, "served_url", lambda url: str(served))

    assert maps.main(["--corpus", str(root), "--path", rel, "--write"]) == 0
    out = json.loads((root / "audio_time_maps.json").read_text(encoding="utf-8"))
    anchors = out[rel]["anchors"]
    assert out[rel]["method"] == "reference"
    assert anchors[0] == [0.0, pytest.approx(0.0, abs=0.04)]
    assert anchors[-1][1] == pytest.approx(30.0, abs=0.04)
    assert 120.0 <= anchors[-1][0] <= 300.0


def test_without_a_reference_there_is_no_map(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    shutil.copytree(CORPUS / "feeds" / "p01", root / "feeds" / "p01")
    meta = next((root / "feeds" / "p01").glob("*/metadata/p01_e01.metadata.json"))
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc["content"]["media_url"] = "https://host.example/p01_e01.mp3"
    meta.write_text(json.dumps(doc), encoding="utf-8")
    rel = str(meta.relative_to(root))
    assert maps.main(["--corpus", str(root), "--path", rel, "--write"]) == 0
    assert json.loads((root / "audio_time_maps.json").read_text(encoding="utf-8")) == {}
