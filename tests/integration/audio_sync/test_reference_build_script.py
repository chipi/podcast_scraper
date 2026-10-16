"""RFC-129: scripts/ops/audio_reference_build.py over a corpus and an archive.

The fixture corpus has no GUIDs, so a copy gets them. The archive is a local one laid out like the
real one (``sha256/<aa>/<bb>/<digest>.mp3``), holding the fixture audio.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from podcast_scraper.utils.audio_cache import rel_key_for_guid

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]
CORPUS = REPO / "tests" / "fixtures" / "app-validation-corpus" / "v3"
AUDIO = REPO / "tests" / "fixtures" / "audio" / "v3"

_spec = importlib.util.spec_from_file_location(
    "audio_reference_build", REPO / "scripts" / "ops" / "audio_reference_build.py"
)
assert _spec and _spec.loader
build_script = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = build_script
_spec.loader.exec_module(build_script)

EPISODES = {"p01_e01": "exact", "p01_e02": "refetch", "p01_e03": "missing"}


def _meta(root: Path, ep: str) -> Path:
    return next((root / "feeds" / "p01").glob(f"*/metadata/{ep}.metadata.json"))


@pytest.fixture()
def setup(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "corpus"
    shutil.copytree(CORPUS / "feeds" / "p01", root / "feeds" / "p01")
    archive = tmp_path / "archive"
    provenance = []
    for ep, kind in EPISODES.items():
        meta = _meta(root, ep)
        doc = json.loads(meta.read_text(encoding="utf-8"))
        guid = f"guid-{ep}"
        doc["episode"]["guid"] = guid
        meta.write_text(json.dumps(doc), encoding="utf-8")
        if kind != "missing":
            key = rel_key_for_guid(guid, ".mp3")
            assert key
            (archive / key).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(AUDIO / f"{ep}.mp3", archive / key)
        provenance.append(
            {
                "guid": guid,
                "origin": "pipeline_download" if kind != "refetch" else "backfill_refetch",
                "byte_identical_to_transcribed_audio": kind != "refetch",
            }
        )
    prov = root / ".podcast_scraper" / "audio-archive-provenance.jsonl"
    prov.parent.mkdir(parents=True)
    prov.write_text("\n".join(json.dumps(p) for p in provenance), encoding="utf-8")
    return root, archive


def _args(root: Path, archive: Path, *extra: str) -> list[str]:
    paths = []
    for ep in EPISODES:
        paths += ["--path", str(_meta(root, ep).relative_to(root))]
    return ["--corpus", str(root), "--local-root", str(archive), *paths, *extra]


def test_writes_a_reference_only_from_an_exact_original(
    setup: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    root, archive = setup
    assert build_script.main(_args(root, archive, "--write")) == 0
    out = capsys.readouterr().out
    assert "skip_no_exact_original" in out and "p01_e02" in out
    assert "skip_not_in_archive" in out and "p01_e03" in out
    refs = sorted(p.name for p in root.rglob("*.audioref.json"))
    assert refs == ["p01_e01.audioref.json"]
    doc = json.loads(next(root.rglob("p01_e01.audioref.json")).read_text(encoding="utf-8"))
    assert doc["guid"] == "guid-p01_e01"
    assert doc["checkpoints"] and doc["checkpoints"][0]["kind"] == "talk_start"
    assert doc["duration_s"] == pytest.approx(488.0, abs=1.0)


def test_without_write_nothing_is_written(
    setup: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    root, archive = setup
    assert build_script.main(_args(root, archive)) == 0
    assert "would_write" in capsys.readouterr().out
    assert not list(root.rglob("*.audioref.json"))
