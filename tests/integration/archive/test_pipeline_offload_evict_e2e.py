"""Full-pipeline local harness for the audio cold-storage archive (#1787, epic #1788).

Runs the REAL ``run_pipeline`` end to end against:
  * a local HTTP server serving a fixture feed + audio (a real download over localhost), and
  * a local rclone remote (``type=local`` -> a temp dir) as the "cold storage service" — the
    exact ``RcloneStorageBackend`` code the Hetzner Storage Box uses, just pointed at disk.

Only the orthogonal ASR is mocked (whisper returns canned text instantly — no model, no cost,
deterministic). EVERYTHING the archive touches is real: the download choke point archives the
audio (``store_via``), ``media/`` is persisted, finalize evicts it once it is size-matched in
cold, provenance is stamped, and the evicted audio is re-fetched from cold.

This is the harness that would have caught advisor M1 (it asserts the real end-to-end flow, not a
fake backend). Run it verbosely with ``make audio-archive-local-e2e`` to watch each step.
"""

from __future__ import annotations

import functools
import http.server
import shutil
import socketserver
import threading
from pathlib import Path
from typing import Iterator, Tuple
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.integration]

if shutil.which("rclone") is None:
    pytest.skip("rclone binary not on PATH", allow_module_level=True)

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_FEED_XML = _FIXTURES / "rss" / "p01_fast.xml"  # no <podcast:transcript> -> the ASR/download path
_AUDIO = _FIXTURES / "audio" / "v3" / "p01_e01_fast.mp3"
_CANNED_TRANSCRIPT = "This is a locally-transcribed fixture episode about building trails."


def _log(step: str) -> None:
    print(f"[audio-archive-e2e] {step}", flush=True)


@pytest.fixture()
def local_site(tmp_path: Path) -> Iterator[Tuple[str, Path]]:
    """Serve ``feed.xml`` + ``audio/`` over localhost; yields (feed_url, site_dir)."""
    site = tmp_path / "site"
    (site / "audio").mkdir(parents=True)
    shutil.copy(_FEED_XML, site / "feed.xml")  # enclosure is /audio/p01_e01_fast.mp3 (relative)
    shutil.copy(_AUDIO, site / "audio" / "p01_e01_fast.mp3")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    _log(f"serving fixture feed + audio at http://127.0.0.1:{port}/")
    try:
        yield f"http://127.0.0.1:{port}/feed.xml", site
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_pipeline_downloads_archives_and_evicts(
    local_site: Tuple[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import podcast_scraper
    from podcast_scraper import config
    from podcast_scraper.utils import audio_cache
    from podcast_scraper.utils.storage_backend import RcloneStorageBackend

    feed_url, _ = local_site
    corpus = tmp_path / "corpus"
    cold = tmp_path / "cold"
    cold.mkdir()

    # Cold storage = a local rclone remote (real backend, local transport).
    monkeypatch.setenv("RCLONE_CONFIG_LOCALARCHIVE_TYPE", "local")

    cfg = config.Config(
        rss=feed_url,  # alias for rss_url (mypy's pydantic plugin wants the alias)
        output_dir=str(corpus),
        transcribe_missing=True,
        whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
        max_episodes=1,
        transcript_cache_enabled=False,
        persist_episode_media=True,
        single_feed_uses_corpus_layout=True,
        # The audio-archive wiring under test:
        audio_storage_backend="remote",
        audio_remote_rclone_remote="localarchive",
        audio_remote_base_path=str(cold),
        audio_evict_local_after_offload=True,
    )

    # Mock ONLY the ASR — everything the archive touches runs for real.
    with patch(
        "podcast_scraper.providers.ml.ml_provider.MLProvider._transcribe_with_whisper",
        return_value=({"text": _CANNED_TRANSCRIPT, "segments": []}, 1.0),
    ):
        _log("running the real pipeline (download -> archive -> transcribe(mock) -> evict)...")
        count, summary = podcast_scraper.run_pipeline(cfg)

    _log(f"pipeline returned count={count}")
    assert count > 0, "pipeline processed no episodes"

    backend = RcloneStorageBackend(remote="localarchive", base_path=str(cold))
    guid = "p01_e01_fast"
    cold_key = audio_cache.rel_key_for_guid(guid, ".mp3")
    assert cold_key is not None

    # 1. Audio was archived to cold, and its size matches the fixture we served.
    cold_size = backend.size(cold_key)
    _log(f"cold has the audio: key={cold_key} size={cold_size}")
    assert cold_size == _AUDIO.stat().st_size, "cold object size != served audio size"

    # 2. media/ was written locally, then EVICTED (no audio left under any media/ dir at rest).
    media_files = [
        p for p in corpus.glob("**/media/*") if p.suffix.lower() in {".mp3", ".m4a", ".wav"}
    ]
    _log(f"local media/ audio files remaining after eviction: {len(media_files)} (want 0)")
    assert media_files == [], f"eviction left local audio behind: {media_files}"

    # 3. Provenance stamped the pipeline download as an original (byte-identical).
    prov_path = corpus / ".podcast_scraper" / "audio-archive-provenance.jsonl"
    assert prov_path.is_file(), "no provenance file written at the corpus root"
    import json

    rows = [json.loads(x) for x in prov_path.read_text().splitlines() if x.strip()]
    pipeline_rows = [r for r in rows if r.get("guid") == guid]
    _log(f"provenance rows for {guid}: {[r['origin'] for r in pipeline_rows]}")
    assert pipeline_rows, "no provenance row for the ingested episode"
    assert pipeline_rows[0]["origin"] == "pipeline_download"
    assert pipeline_rows[0]["byte_identical_to_transcribed_audio"] is True

    # 4. Reproducibility: the evicted audio is pullable back from cold, byte-for-byte.
    pulled = tmp_path / "pulled.mp3"
    assert audio_cache.fetch_into(backend, guid, str(pulled)), "could not re-fetch from cold"
    assert pulled.read_bytes() == _AUDIO.read_bytes(), "re-fetched bytes != original"
    _log("re-fetched the evicted audio from cold — bytes match. E2E PASS.")
