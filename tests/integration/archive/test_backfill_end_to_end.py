"""``archive backfill`` end to end: real corpus, real HTTP, real storage backend (#1631).

The unit tests inject a fake backend and a fake opener, which proves the classification logic
but not that the pieces fit. This drives the actual CLI entrypoint against a corpus laid out on
disk the way the pipeline writes one, a local HTTP server standing in for the publisher's CDN,
and the real :class:`LocalStorageBackend` — so a wrong archive key, a bad metadata traversal,
or a broken argv contract fails here rather than on the prod box.
"""

from __future__ import annotations

import json
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

from podcast_scraper.archive.cli_handlers import parse_archive_argv, run_archive
from podcast_scraper.utils.audio_cache import rel_key_for_guid

pytestmark = [pytest.mark.integration]

_AUDIO = b"ID3" + b"\x00" * 4096


@pytest.fixture()
def cdn(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A local HTTP server serving one real audio file (and 404 for anything else)."""
    root = tmp_path_factory.mktemp("cdn")
    (root / "ep1.mp3").write_bytes(_AUDIO)
    (root / "ep2.mp3").write_bytes(_AUDIO)
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _write_episode(corpus: Path, *, feed: str, guid: str, title: str, media_url: str) -> None:
    """Write a metadata file in the layout ``_iter_corpus_episodes`` actually globs for."""
    meta_dir = corpus / "feeds" / feed / "run_20260815T120000Z" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    doc: Dict[str, Any] = {
        "episode": {
            "guid": guid,
            "episode_id": guid,
            "title": title,
            "published_date": "2026-08-01",
        },
        "feed": {"title": feed, "feed_id": feed},
        "content": {"media_url": media_url, "media_type": "audio/mpeg"},
    }
    (meta_dir / f"0001 - {title}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture()
def corpus(tmp_path: Path, cdn: str) -> Path:
    root = tmp_path / "corpus"
    _write_episode(
        root, feed="Hard Fork", guid="guid-1", title="Episode One", media_url=f"{cdn}/ep1.mp3"
    )
    _write_episode(
        root, feed="Planet Money", guid="guid-2", title="Episode Two", media_url=f"{cdn}/ep2.mp3"
    )
    return root


def _run(corpus: Path, archive_root: Path, *extra: str) -> int:
    args = parse_archive_argv(
        [
            "backfill",
            "--corpus",
            str(corpus),
            "--local-root",
            str(archive_root),
            "--rate-limit",
            "0",
            *extra,
        ]
    )
    return run_archive(args)


class TestDryRun:
    def test_it_reports_without_fetching_anything(
        self, corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        archive = tmp_path / "archive"
        assert _run(corpus, archive, "--dry-run") == 0
        out = capsys.readouterr().out
        assert "dry-run" in out
        assert "Hard Fork" in out and "Planet Money" in out
        # Nothing was written: the whole point of a preview.
        assert not archive.exists() or not any(archive.rglob("*.mp3"))


class TestRealRun:
    def test_it_stores_both_episodes_under_the_pipeline_key(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        archive = tmp_path / "archive"
        assert _run(corpus, archive) == 0
        for guid in ("guid-1", "guid-2"):
            key = rel_key_for_guid(guid, ".mp3")
            assert key is not None, f"no archive key for {guid}"
            stored = archive / key
            assert stored.is_file(), f"missing {key}"
            assert stored.read_bytes() == _AUDIO

    def test_the_key_matches_what_the_pipeline_would_look_up(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """A backfill that writes a different key than ``fetch_into`` probes archives nothing
        usable — the audio would be present and permanently invisible."""
        from podcast_scraper.utils import audio_cache
        from podcast_scraper.utils.storage_backend import LocalStorageBackend

        archive = tmp_path / "archive"
        assert _run(corpus, archive) == 0
        dest = tmp_path / "pulled.mp3"
        assert audio_cache.fetch_into(LocalStorageBackend(archive), "guid-1", str(dest))
        assert dest.read_bytes() == _AUDIO

    def test_a_second_run_stores_nothing_new(self, corpus: Path, tmp_path: Path) -> None:
        archive = tmp_path / "archive"
        assert _run(corpus, archive) == 0
        before = {p: p.stat().st_mtime_ns for p in archive.rglob("*") if p.is_file()}
        assert _run(corpus, archive) == 0
        after = {p: p.stat().st_mtime_ns for p in archive.rglob("*") if p.is_file()}
        assert before == after, "re-run rewrote archived audio; backfill must be idempotent"

    def test_provenance_marks_the_audio_as_refetched(self, corpus: Path, tmp_path: Path) -> None:
        assert _run(corpus, tmp_path / "archive") == 0
        prov = corpus / ".podcast_scraper" / "audio-archive-provenance.jsonl"
        rows = [json.loads(line) for line in prov.read_text(encoding="utf-8").splitlines()]
        assert {r["guid"] for r in rows} == {"guid-1", "guid-2"}
        assert all(r["byte_identical_to_transcribed_audio"] is False for r in rows)

    def test_a_missing_enclosure_is_rolled_off_and_still_exits_zero(
        self, tmp_path: Path, cdn: str
    ) -> None:
        """An episode the publisher no longer serves is a normal outcome. Exiting non-zero
        would make a scheduled backfill look broken forever on a corpus with old episodes."""
        root = tmp_path / "corpus"
        _write_episode(
            root,
            feed="Old Show",
            guid="gone",
            title="Aged Out",
            media_url=f"{cdn}/does-not-exist.mp3",
        )
        assert _run(root, tmp_path / "archive") == 0

    def test_a_feed_selector_scopes_the_pass(self, corpus: Path, tmp_path: Path) -> None:
        archive = tmp_path / "archive"
        assert _run(corpus, archive, "--feed", "Hard Fork") == 0
        key1 = rel_key_for_guid("guid-1", ".mp3")
        key2 = rel_key_for_guid("guid-2", ".mp3")
        assert key1 is not None and key2 is not None
        assert (archive / key1).is_file()
        assert not (archive / key2).exists()


def _write_episode_with_local_audio(
    corpus: Path, *, feed: str, guid: str, title: str, media_url: str, payload: bytes
) -> Path:
    """Write an episode whose ORIGINAL audio already sits in the run's ``media/`` on disk.

    This is the world backfill v2 targets: prod already holds the original bytes, so backfill
    must MOVE them into cold (byte-identical) rather than re-download a re-encode. Returns the
    local media path so a test can assert the cleanup sweep later evicts it.
    """
    run_dir = corpus / "feeds" / feed / "run_20260815T120000Z"
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (run_dir / "media").mkdir(parents=True, exist_ok=True)
    stem = f"0001 - {title}"
    media_rel = f"media/{stem}.mp3"
    (run_dir / media_rel).write_bytes(payload)
    doc: Dict[str, Any] = {
        "episode": {
            "guid": guid,
            "episode_id": guid,
            "title": title,
            "published_date": "2026-08-01",
        },
        "feed": {"title": feed, "feed_id": feed},
        "content": {"media_url": media_url, "media_type": "audio/mpeg", "audio_relpath": media_rel},
    }
    (run_dir / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    return run_dir / media_rel


class TestHarvestAndSweep:
    """Local-first: an existing original is moved into cold, then the local copy is reclaimed."""

    def test_local_original_is_harvested_then_evicted(self, tmp_path: Path, cdn: str) -> None:
        root = tmp_path / "corpus"
        original = b"ID3" + b"\x11" * 8192  # distinct from the CDN's _AUDIO — proves no re-download
        local = _write_episode_with_local_audio(
            root,
            feed="Hard Fork",
            guid="have",
            title="Local One",
            media_url=f"{cdn}/ep1.mp3",
            payload=original,
        )
        archive = tmp_path / "archive"
        assert _run(root, archive) == 0

        # Harvested the ORIGINAL bytes into cold under the pipeline key (not the CDN re-encode).
        key = rel_key_for_guid("have", ".mp3")
        assert key is not None and (archive / key).read_bytes() == original
        # Provenance says byte-identical (harvest), not a refetch.
        prov = root / ".podcast_scraper" / "audio-archive-provenance.jsonl"
        rows = [json.loads(line) for line in prov.read_text(encoding="utf-8").splitlines()]
        assert rows[0]["origin"] == "backfill_harvest_local"
        assert rows[0]["byte_identical_to_transcribed_audio"] is True
        # Cleanup sweep reclaimed the now-redundant local copy (confirmed in cold + size-matched).
        assert not local.exists(), "local original should be evicted once safely in cold"

    def test_dry_run_reports_the_move_and_evicts_nothing(
        self, tmp_path: Path, cdn: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "corpus"
        local = _write_episode_with_local_audio(
            root,
            feed="Hard Fork",
            guid="have",
            title="Local One",
            media_url=f"{cdn}/ep1.mp3",
            payload=b"ID3" + b"\x22" * 4096,
        )
        assert _run(root, tmp_path / "archive", "--dry-run") == 0
        out = capsys.readouterr().out
        assert "move-local" in out and "to move from local" in out
        assert local.exists(), "dry-run must not evict anything"


class TestBackendContract:
    def test_no_backend_is_a_clear_error_not_a_traceback(self, corpus: Path) -> None:
        args = parse_archive_argv(["backfill", "--corpus", str(corpus)])
        with pytest.raises(SystemExit) as ei:
            run_archive(args)
        assert "--rclone-remote" in str(ei.value) and "--local-root" in str(ei.value)

    def test_pull_still_works(self, corpus: Path, tmp_path: Path) -> None:
        """Backfill was added beside pull; the regression that matters is breaking pull."""
        archive = tmp_path / "archive"
        assert _run(corpus, archive) == 0
        args = parse_archive_argv(
            [
                "pull",
                "--corpus",
                str(corpus),
                "--dest",
                str(tmp_path / "pulled"),
                "--local-root",
                str(archive),
            ]
        )
        assert run_archive(args) == 0
        assert list((tmp_path / "pulled").rglob("*.mp3"))
