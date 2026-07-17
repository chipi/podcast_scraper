"""#1199: ``archive pull`` CLI — corpus enumeration, selectors, local-backend pull."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.archive.cli_handlers import parse_archive_argv, run_archive
from podcast_scraper.utils import audio_cache
from podcast_scraper.utils.storage_backend import LocalStorageBackend

pytestmark = pytest.mark.unit


def _write_meta(corpus: Path, feed_dir: str, run: str, name: str, doc: dict) -> None:
    d = corpus / "feeds" / feed_dir / run / "metadata"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")


def _episode(guid: str, title: str, feed_title: str, date: str) -> dict:
    return {
        "feed": {"title": feed_title, "feed_id": feed_title.lower().replace(" ", "-")},
        "episode": {
            "guid": guid,
            "episode_id": guid,
            "title": title,
            "published_date": date,
            "episode_number": None,
        },
        "content": {"media_url": f"https://x/{guid}.mp3?size=1000", "media_type": "audio/mpeg"},
    }


@pytest.fixture()
def corpus_and_archive(tmp_path):
    corpus = tmp_path / "corpus"
    _write_meta(
        corpus,
        "feedA",
        "run_20260101-000000_x",
        "0001 - Ep One",
        _episode("guid-a1", "Ep One", "Show A", "2026-01-01T00:00:00+00:00"),
    )
    _write_meta(
        corpus,
        "feedA",
        "run_20260201-000000_x",
        "0002 - Ep Two",
        _episode("guid-a2", "Ep Two", "Show A", "2026-02-15T00:00:00+00:00"),
    )
    # A GUID stored across two runs must dedupe to one pulled file.
    _write_meta(
        corpus,
        "feedA",
        "run_20260301-000000_x",
        "0001 - Ep One",
        _episode("guid-a1", "Ep One", "Show A", "2026-01-01T00:00:00+00:00"),
    )

    archive_root = tmp_path / "archive"
    be = LocalStorageBackend(archive_root)
    for guid, payload in (("guid-a1", b"one"), ("guid-a2", b"two")):
        src = tmp_path / f"{guid}.mp3"
        src.write_bytes(payload)
        assert audio_cache.store_via(be, guid, str(src)) is not None
    return corpus, archive_root


def _args(corpus, archive_root, dest, extra=None):
    argv = ["pull", "--corpus", str(corpus), "--dest", str(dest), "--local-root", str(archive_root)]
    argv += extra or []
    return parse_archive_argv(argv)


class TestArchivePull:
    def test_pull_all_dedupes_and_names(self, corpus_and_archive, tmp_path):
        corpus, archive_root = corpus_and_archive
        dest = tmp_path / "out"
        rc = run_archive(_args(corpus, archive_root, dest))
        assert rc == 0
        pulled = sorted(p.name for p in dest.rglob("*.mp3"))
        assert pulled == ["0000 - Ep One.mp3", "0001 - Ep Two.mp3"]  # 2 unique guids
        assert (dest / "Show A" / "0000 - Ep One.mp3").read_bytes() == b"one"

    def test_dry_run_writes_nothing(self, corpus_and_archive, tmp_path, capsys):
        corpus, archive_root = corpus_and_archive
        dest = tmp_path / "out"
        rc = run_archive(_args(corpus, archive_root, dest, ["--dry-run"]))
        assert rc == 0
        assert not dest.exists()
        assert "dry-run" in capsys.readouterr().out

    def test_since_selector(self, corpus_and_archive, tmp_path):
        corpus, archive_root = corpus_and_archive
        dest = tmp_path / "out"
        run_archive(_args(corpus, archive_root, dest, ["--since", "2026-02-01"]))
        assert sorted(p.name for p in dest.rglob("*.mp3")) == ["0000 - Ep Two.mp3"]

    def test_skip_existing_unless_force(self, corpus_and_archive, tmp_path):
        corpus, archive_root = corpus_and_archive
        dest = tmp_path / "out"
        run_archive(_args(corpus, archive_root, dest))
        # Corrupt one file; a plain re-run skips it, --force re-pulls.
        f = dest / "Show A" / "0000 - Ep One.mp3"
        f.write_bytes(b"STALE")
        run_archive(_args(corpus, archive_root, dest))
        assert f.read_bytes() == b"STALE"
        run_archive(_args(corpus, archive_root, dest, ["--force"]))
        assert f.read_bytes() == b"one"

    def test_parse_sets_command(self, tmp_path):
        ns = _args(tmp_path, tmp_path, tmp_path)
        assert ns.command == "archive" and ns.archive_subcommand == "pull"
