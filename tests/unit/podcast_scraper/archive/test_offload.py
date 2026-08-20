"""End-of-run local-audio eviction after cold offload (#1787, epic #1788).

The archive is internal-only and never served, so a local ``media/`` copy is disposable once
its audio is in cold storage. These tests pin the properties that make it safe to point at a
live corpus — because it DELETES corpus audio:

* **confirmed-in-cold gate** — an episode whose audio is not in the backend is KEPT; eviction
  never destroys the only copy;
* **media/ only** — transcripts, art, and the provenance file are never candidates, and an
  ``audio_relpath`` that escapes ``media/`` is refused;
* **dry-run moves nothing** — it reports what it would do;
* **best-effort** — a backend that raises keeps the file rather than crashing the run.
"""

# mypy: disable-error-code="arg-type"
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper.archive import offload
from podcast_scraper.utils.audio_cache import rel_key_for_guid

pytestmark = [pytest.mark.unit]


# Local media bytes every fixture episode is written with. Cold copies must match this SIZE for
# the evict guard to fire — that is the whole point of the H1 size check.
_MEDIA_BYTES = b"AUDIO-BYTES"


class _FakeBackend:
    """In-memory StorageBackend stand-in keyed by rel_key, mirroring test_backfill."""

    def __init__(
        self, preloaded: Optional[Dict[str, bytes]] = None, *, raises: bool = False
    ) -> None:
        self.store: Dict[str, bytes] = dict(preloaded or {})
        self.exists_calls: List[str] = []
        self.raises = raises

    def exists(self, rel_key: str) -> bool:
        if self.raises:
            raise RuntimeError("backend down")
        self.exists_calls.append(rel_key)
        return rel_key in self.store

    def size(self, rel_key: str):
        if self.raises:
            raise RuntimeError("backend down")
        blob = self.store.get(rel_key)
        return len(blob) if blob else None

    def upload(self, src_path: str, rel_key: str) -> bool:  # pragma: no cover - unused here
        return True

    def download(self, rel_key: str, dest_path: str) -> bool:  # pragma: no cover - unused here
        return False


def _in_cold(guid: str, ext: str = ".mp3", *, content: bytes = _MEDIA_BYTES) -> Dict[str, bytes]:
    """A preloaded-backend dict marking ``guid`` present in cold with SIZE-matching bytes."""
    key = rel_key_for_guid(guid, ext)
    assert key is not None
    return {key: content}


def _make_run_dir(
    tmp_path: Path,
    episodes: List[Dict[str, Any]],
    *,
    write_media: bool = True,
) -> Path:
    """Build a run dir with ``metadata/*.metadata.json`` + ``media/`` for the given episodes.

    Each episode dict: ``{"guid": ..., "stem": ..., "ext": ".mp3"}``.
    """
    run_dir = tmp_path / "feeds" / "feed_a" / "run_1"
    (run_dir / "metadata").mkdir(parents=True)
    (run_dir / "media").mkdir(parents=True)
    for ep in episodes:
        stem = ep["stem"]
        ext = ep.get("ext", ".mp3")
        audio_relpath = f"media/{stem}{ext}"
        doc = {
            "feed": {"feed_id": "feed_a"},
            "episode": {"guid": ep["guid"], "episode_id": stem},
            "content": {"audio_relpath": audio_relpath},
        }
        (run_dir / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc))
        if write_media:
            (run_dir / "media" / f"{stem}{ext}").write_bytes(_MEDIA_BYTES)
    return run_dir


class TestEvictRunDir:
    def test_audio_confirmed_in_cold_is_evicted(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [{"guid": "g1", "stem": "0001 - Ep One"}])
        backend = _FakeBackend(_in_cold("g1"))
        media_file = run_dir / "media" / "0001 - Ep One.mp3"
        assert media_file.is_file()

        report = offload.evict_run_dir(str(run_dir), backend)

        assert report.evicted == 1
        assert report.bytes_freed == len(b"AUDIO-BYTES")
        assert not media_file.exists()  # local copy gone; cold retains it

    def test_audio_not_in_cold_is_kept(self, tmp_path: Path) -> None:
        # The safety-critical case: nothing in cold -> nothing deleted.
        run_dir = _make_run_dir(tmp_path, [{"guid": "g1", "stem": "0001 - Ep One"}])
        backend = _FakeBackend()  # empty cold
        media_file = run_dir / "media" / "0001 - Ep One.mp3"

        report = offload.evict_run_dir(str(run_dir), backend)

        assert report.evicted == 0
        assert report.kept_not_in_cold == 1
        assert media_file.is_file()  # only copy preserved

    def test_cold_size_mismatch_is_kept(self, tmp_path: Path) -> None:
        # H1: the archive holds an object under this GUID, but its size differs from the local
        # file (a dedupe against a different, re-encoded copy). The delete MUST be refused —
        # otherwise the bytes that produced this run's transcript are lost forever.
        run_dir = _make_run_dir(tmp_path, [{"guid": "g1", "stem": "0001 - Ep One"}])
        backend = _FakeBackend(_in_cold("g1", content=b"DIFFERENT-ENCODE-longer"))  # size != local
        media_file = run_dir / "media" / "0001 - Ep One.mp3"

        report = offload.evict_run_dir(str(run_dir), backend)

        assert report.evicted == 0
        assert report.kept_size_mismatch == 1
        assert media_file.is_file()  # only copy of the transcribed bytes preserved

    def test_cold_size_unknowable_is_kept(self, tmp_path: Path) -> None:
        # exists() says present but size() returns None (transport can't confirm) -> keep.
        run_dir = _make_run_dir(tmp_path, [{"guid": "g1", "stem": "0001 - Ep"}])

        class _NoSizeBackend(_FakeBackend):
            def size(self, rel_key: str):
                return None

        backend = _NoSizeBackend(_in_cold("g1"))
        report = offload.evict_run_dir(str(run_dir), backend)
        assert report.evicted == 0
        assert report.kept_size_mismatch == 1
        assert (run_dir / "media" / "0001 - Ep.mp3").is_file()

    def test_missing_guid_keeps_the_file(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "metadata").mkdir(parents=True)
        (run_dir / "media").mkdir(parents=True)
        doc = {"episode": {"guid": ""}, "content": {"audio_relpath": "media/x.mp3"}}
        (run_dir / "metadata" / "x.metadata.json").write_text(json.dumps(doc))
        (run_dir / "media" / "x.mp3").write_bytes(b"A")
        backend = _FakeBackend(_in_cold("anything"))

        report = offload.evict_run_dir(str(run_dir), backend)

        assert report.evicted == 0
        assert (run_dir / "media" / "x.mp3").is_file()

    def test_dry_run_deletes_nothing(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [{"guid": "g1", "stem": "0001 - Ep"}])
        backend = _FakeBackend(_in_cold("g1"))
        media_file = run_dir / "media" / "0001 - Ep.mp3"

        report = offload.evict_run_dir(str(run_dir), backend, dry_run=True)

        assert report.dry_run is True
        assert report.evicted == 1  # counted
        assert media_file.is_file()  # but present

    def test_backend_error_keeps_the_file(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [{"guid": "g1", "stem": "0001 - Ep"}])
        backend = _FakeBackend(raises=True)
        media_file = run_dir / "media" / "0001 - Ep.mp3"

        report = offload.evict_run_dir(str(run_dir), backend)

        assert report.evicted == 0
        assert report.kept_not_in_cold == 1
        assert media_file.is_file()

    def test_none_backend_is_a_noop(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [{"guid": "g1", "stem": "0001 - Ep"}])
        report = offload.evict_run_dir(str(run_dir), None)
        assert report.evicted == 0
        assert (run_dir / "media" / "0001 - Ep.mp3").is_file()

    def test_unreadable_metadata_is_skipped(self, tmp_path: Path) -> None:
        # A corrupt metadata json must be skipped (logged), never crash the pass.
        run_dir = tmp_path / "run"
        (run_dir / "metadata").mkdir(parents=True)
        (run_dir / "media").mkdir(parents=True)
        (run_dir / "metadata" / "bad.metadata.json").write_text("{ not json")
        (run_dir / "media" / "x.mp3").write_bytes(_MEDIA_BYTES)
        report = offload.evict_run_dir(str(run_dir), _FakeBackend(_in_cold("g1")))
        assert report.evicted == 0  # no usable record -> nothing considered

    def test_non_dict_metadata_is_skipped(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "metadata").mkdir(parents=True)
        (run_dir / "media").mkdir(parents=True)
        (run_dir / "metadata" / "list.metadata.json").write_text("[1, 2, 3]")
        report = offload.evict_run_dir(str(run_dir), _FakeBackend())
        assert report.evicted == 0

    def test_cold_size_probe_raising_keeps_the_file(self, tmp_path: Path) -> None:
        # already_archived returns a key, but size() raises -> cannot confirm -> keep.
        run_dir = _make_run_dir(tmp_path, [{"guid": "g1", "stem": "0001 - Ep"}])

        class _SizeRaises(_FakeBackend):
            def size(self, rel_key: str):
                raise RuntimeError("size probe down")

        report = offload.evict_run_dir(str(run_dir), _SizeRaises(_in_cold("g1")))
        assert report.evicted == 0
        assert report.kept_size_mismatch == 1
        assert (run_dir / "media" / "0001 - Ep.mp3").is_file()

    def test_no_metadata_dir_is_noop(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        report = offload.evict_run_dir(str(run_dir), _FakeBackend(_in_cold("g1")))
        assert report.evicted == 0

    def test_only_media_confirmed_episodes_evicted_mixed(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(
            tmp_path,
            [
                {"guid": "g1", "stem": "0001 - A"},  # in cold -> evict
                {"guid": "g2", "stem": "0002 - B"},  # not in cold -> keep
            ],
        )
        backend = _FakeBackend(_in_cold("g1"))

        report = offload.evict_run_dir(str(run_dir), backend)

        assert report.evicted == 1
        assert report.kept_not_in_cold == 1
        assert not (run_dir / "media" / "0001 - A.mp3").exists()
        assert (run_dir / "media" / "0002 - B.mp3").is_file()


class TestNeverTouchesNonMedia:
    def test_provenance_and_art_are_never_deleted(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [{"guid": "g1", "stem": "0001 - Ep"}])
        ps = run_dir / ".podcast_scraper"
        (ps / "corpus-art").mkdir(parents=True)
        art = ps / "corpus-art" / "cover.jpg"
        art.write_bytes(b"JPG")
        prov = ps / "audio-archive-provenance.jsonl"
        prov.write_text('{"guid":"g1"}\n')
        backend = _FakeBackend(_in_cold("g1"))

        offload.evict_run_dir(str(run_dir), backend)

        assert art.is_file()
        assert prov.is_file()

    def test_audio_relpath_escaping_media_is_refused(self, tmp_path: Path) -> None:
        # A crafted audio_relpath pointing outside media/ must never be followed.
        run_dir = tmp_path / "run"
        (run_dir / "metadata").mkdir(parents=True)
        (run_dir / "media").mkdir(parents=True)
        victim = run_dir / "transcripts"
        victim.mkdir()
        secret = victim / "keep.txt"
        secret.write_text("do not delete")
        doc = {
            "episode": {"guid": "g1"},
            "content": {"audio_relpath": "../transcripts/keep.txt"},
        }
        (run_dir / "metadata" / "x.metadata.json").write_text(json.dumps(doc))
        backend = _FakeBackend(_in_cold("g1"))

        report = offload.evict_run_dir(str(run_dir), backend)

        assert report.evicted == 0
        assert secret.is_file()  # traversal refused


class TestSweepCorpus:
    def test_sweeps_every_run_dir(self, tmp_path: Path) -> None:
        # Two run dirs under the corpus; both have a confirmed-in-cold episode.
        r1 = _make_run_dir(tmp_path, [{"guid": "g1", "stem": "0001 - A"}])
        # second feed/run
        r2 = tmp_path / "feeds" / "feed_b" / "run_2"
        (r2 / "metadata").mkdir(parents=True)
        (r2 / "media").mkdir(parents=True)
        doc = {"episode": {"guid": "g2"}, "content": {"audio_relpath": "media/0001 - C.mp3"}}
        (r2 / "metadata" / "c.metadata.json").write_text(json.dumps(doc))
        (r2 / "media" / "0001 - C.mp3").write_bytes(_MEDIA_BYTES)  # size must match cold

        backend = _FakeBackend({**_in_cold("g1"), **_in_cold("g2")})

        report = offload.sweep_corpus(str(tmp_path), backend)

        assert report.evicted == 2
        assert not (r1 / "media" / "0001 - A.mp3").exists()
        assert not (r2 / "media" / "0001 - C.mp3").exists()

    def test_sweep_none_backend_is_noop(self, tmp_path: Path) -> None:
        _make_run_dir(tmp_path, [{"guid": "g1", "stem": "0001 - A"}])
        report = offload.sweep_corpus(str(tmp_path), None)
        assert report.evicted == 0
