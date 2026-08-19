"""End-to-end audio offload -> provenance -> evict against REAL rclone (#1787, #1789).

The unit tests use an in-memory fake backend; this exercises the whole subsystem through the
actual ``RcloneStorageBackend`` (rclone shelling out to a ``local`` remote, so no network / box
is needed) to prove the pieces compose:

    pipeline download -> store_via (real rclone upload) -> record_pipeline_provenance
      -> evict_run_dir (real rclone lsjson cold-check) -> audio re-fetchable from cold

The load-bearing property is the last step: eviction is only safe because the evicted audio can
be pulled back from cold for a reproducible reprocess.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

pytest.importorskip("podcast_scraper.utils.storage_backend")

if shutil.which("rclone") is None:
    pytest.skip("rclone binary not on PATH", allow_module_level=True)


def test_store_provenance_evict_refetch_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RCLONE_CONFIG_LOCALARCHIVE_TYPE", "local")

    from podcast_scraper.archive import backfill, offload
    from podcast_scraper.utils import audio_cache
    from podcast_scraper.utils.storage_backend import RcloneStorageBackend

    cold = tmp_path / "cold"
    cold.mkdir()
    run_dir = tmp_path / "feeds" / "feed_a" / "run_1"
    (run_dir / "metadata").mkdir(parents=True)
    (run_dir / "media").mkdir(parents=True)
    (run_dir / "transcripts").mkdir(parents=True)

    backend = RcloneStorageBackend(remote="localarchive", base_path=str(cold))

    def make_episode(guid: str, stem: str, *, in_cold: bool) -> Path:
        audio_rel = f"media/{stem}.mp3"
        media = run_dir / audio_rel
        media.write_bytes(b"REAL-AUDIO-" + guid.encode())
        doc = {"episode": {"guid": guid}, "content": {"audio_relpath": audio_rel}}
        (run_dir / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc))
        if in_cold:
            rel = audio_cache.store_via(backend, guid, str(media))  # real rclone upload
            assert rel is not None
            backfill.record_pipeline_provenance(
                str(run_dir), guid=guid, rel_key=rel, source_url=f"https://cdn/{stem}.mp3"
            )
        return media

    m_incold = make_episode("g-in", "0001 - Archived", in_cold=True)
    m_notcold = make_episode("g-out", "0002 - NotYet", in_cold=False)
    transcript = run_dir / "transcripts" / "0001 - Archived.txt"
    transcript.write_text("hello")
    prov_path = run_dir / ".podcast_scraper" / "audio-archive-provenance.jsonl"

    # Real rclone confirms the upload, and provenance stamped it as an original download.
    incold_key = audio_cache.rel_key_for_guid("g-in", ".mp3")
    assert incold_key is not None
    assert backend.exists(incold_key)
    prov = [json.loads(x) for x in prov_path.read_text().splitlines() if x.strip()]
    assert prov[0]["origin"] == "pipeline_download"
    assert prov[0]["byte_identical_to_transcribed_audio"] is True

    report = offload.evict_run_dir(str(run_dir), backend)  # real rclone lsjson cold-check

    assert report.evicted == 1
    assert not m_incold.exists()  # confirmed-in-cold -> evicted
    assert m_notcold.is_file()  # not in cold -> kept (safety)
    assert transcript.is_file()  # never a candidate
    assert prov_path.is_file()  # never a candidate

    # Reproducibility: the evicted audio is pullable back from cold.
    pulled = tmp_path / "pulled.mp3"
    assert audio_cache.fetch_into(backend, "g-in", str(pulled))
    assert pulled.read_bytes() == b"REAL-AUDIO-g-in"
