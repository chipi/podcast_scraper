"""``archive backfill`` — recovering audio lost to an ephemeral container cache (#1631).

Audio caching was on all along, but ``audio_cache_in_corpus`` defaulted to False, so the cache
resolved to ``/app/.cache/audio`` — inside a ``docker compose run --rm`` container, not under
the mounted corpus volume. Every job downloaded audio, cached it, and destroyed the cache on
exit. ~473 episodes predate the fix.

These tests pin the properties that make the tool safe to point at live publishers:

* it is **idempotent** — the expensive, rude operation is the fetch, so an already-archived
  episode must not be re-downloaded;
* **rolled off is not a failure** — publishers truncate feeds, so an aged-out episode is a
  reported outcome, never an error and never a retry loop;
* it is a **polite client** — hundreds of episodes concentrated on a few CDNs;
* recovered audio is **stamped as re-fetched**, because dynamic-ad feeds re-encode per request
  and those bytes are not the ones that produced the existing transcript.
"""

# mypy: disable-error-code="arg-type"
# Deliberate in this file: lightweight duck-typed doubles passed where the production type is
# declared.
# Constructing the real types would pull in the machinery these tests isolate. The
# annotations on the helpers here are what make mypy check these bodies at all — most
# older test files are unannotated and therefore unchecked.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.error import HTTPError

import pytest

from podcast_scraper.archive import backfill as bf

pytestmark = [pytest.mark.unit]


def _key(guid: str, ext: str) -> str:
    """``rel_key_for_guid`` is Optional — a guid that will not digest returns None.

    None is a real outcome worth failing on rather than silently using as a dict key, which is
    what these tests did before: ``{None: b"..."}`` is a perfectly valid dict and the archive
    lookup would simply never match it, so the test would pass for the wrong reason.
    """
    from podcast_scraper.utils.audio_cache import rel_key_for_guid

    key = rel_key_for_guid(guid, ext)
    assert key is not None, f"no archive key for {guid!r}"
    return key


class _FakeBackend:
    """In-memory StorageBackend stand-in: rel_key -> bytes."""

    def __init__(self, preloaded: Dict[str, bytes] | None = None) -> None:
        self.store: Dict[str, bytes] = dict(preloaded or {})
        self.uploads: List[str] = []
        self.exists_calls: List[str] = []

    def exists(self, rel_key: str) -> bool:
        self.exists_calls.append(rel_key)
        return rel_key in self.store

    def upload(self, src_path: str, rel_key: str) -> bool:
        with open(src_path, "rb") as fh:
            self.store[rel_key] = fh.read()
        self.uploads.append(rel_key)
        return True

    def download(self, rel_key: str, dest_path: str) -> bool:  # pragma: no cover - unused here
        return False

    def describe(self) -> str:
        return "fake"


class _Resp:
    """Minimal urlopen-style response supporting the context-manager + read() protocol."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._sent = False

    def read(self, _n: int = -1) -> bytes:
        if self._sent:
            return b""
        self._sent = True
        return self._payload

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *_a: Any) -> None:
        return None


def _opener(payload: bytes = b"audio-bytes"):
    def _open(_req: Any, timeout: int = 0) -> _Resp:  # noqa: ARG001
        return _Resp(payload)

    return _open


def _raising_opener(exc: Exception):
    def _open(_req: Any, timeout: int = 0):  # noqa: ARG001
        raise exc

    return _open


def _ep(**over: Any) -> Dict[str, Any]:
    base = {
        "guid": "guid-abc",
        "title": "An Episode",
        "feed_title": "Hard Fork",
        "media_url": "https://cdn.example.com/ep1.mp3",
        "media_type": "audio/mpeg",
    }
    base.update(over)
    return base


class TestIdempotence:
    """Re-running must not re-download. The fetch is the expensive and impolite part."""

    def test_an_already_archived_episode_is_not_fetched(self, tmp_path: Path) -> None:
        key = _key("guid-abc", ".mp3")
        backend = _FakeBackend({key: b"already-here"})

        def _explode(*_a: Any, **_k: Any):
            raise AssertionError("fetched an episode the archive already holds")

        out = bf.backfill_episode(_ep(), backend, corpus_dir=str(tmp_path), opener=_explode)
        assert out.outcome == bf.ALREADY_PRESENT
        assert out.rel_key == key
        assert backend.uploads == []

    def test_a_second_pass_stores_nothing_new(self, tmp_path: Path) -> None:
        backend = _FakeBackend()
        first = bf.backfill_episode(_ep(), backend, corpus_dir=str(tmp_path), opener=_opener())
        assert first.outcome == bf.STORED
        second = bf.backfill_episode(_ep(), backend, corpus_dir=str(tmp_path), opener=_opener())
        assert second.outcome == bf.ALREADY_PRESENT
        assert len(backend.uploads) == 1

    def test_force_refetches_a_present_episode(self, tmp_path: Path) -> None:
        backend = _FakeBackend({_key("guid-abc", ".mp3"): b"old"})
        out = bf.backfill_episode(
            _ep(), backend, corpus_dir=str(tmp_path), force=True, opener=_opener(b"new")
        )
        assert out.outcome == bf.STORED
        assert out.rel_key is not None, "a STORED outcome must carry the key it stored under"
        assert backend.store[out.rel_key] == b"new"

    def test_the_archived_extension_is_honoured_not_guessed(self, tmp_path: Path) -> None:
        """Archived as .m4a; the URL says .mp3. Guessing from the URL would re-download it."""
        backend = _FakeBackend({_key("guid-abc", ".m4a"): b"present"})
        out = bf.backfill_episode(
            _ep(media_url="https://cdn.example.com/ep1.mp3"),
            backend,
            corpus_dir=str(tmp_path),
            opener=_raising_opener(AssertionError("should not fetch")),
        )
        assert out.outcome == bf.ALREADY_PRESENT


class TestRolledOffIsNotAFailure:
    """Publishers truncate feeds. An aged-out episode is expected, not broken."""

    @pytest.mark.parametrize("status", [404, 410])
    def test_gone_from_the_window_is_reported_not_raised(self, tmp_path: Path, status: int) -> None:
        backend = _FakeBackend()
        exc = HTTPError("https://cdn.example.com/ep1.mp3", status, "gone", None, None)
        out = bf.backfill_episode(
            _ep(), backend, corpus_dir=str(tmp_path), opener=_raising_opener(exc)
        )
        assert out.outcome == bf.ROLLED_OFF
        assert backend.uploads == []

    def test_a_server_error_is_a_retryable_failure_not_a_rolloff(self, tmp_path: Path) -> None:
        """503 means try later; conflating it with rolled_off would silently abandon episodes
        that are still perfectly available."""
        backend = _FakeBackend()
        exc = HTTPError("https://cdn.example.com/ep1.mp3", 503, "unavailable", None, None)
        out = bf.backfill_episode(
            _ep(), backend, corpus_dir=str(tmp_path), opener=_raising_opener(exc)
        )
        assert out.outcome == bf.FETCH_FAILED

    def test_a_transport_error_never_propagates(self, tmp_path: Path) -> None:
        """One dead host must not abort a several-hundred-episode pass."""
        backend = _FakeBackend()
        out = bf.backfill_episode(
            _ep(), backend, corpus_dir=str(tmp_path), opener=_raising_opener(OSError("reset"))
        )
        assert out.outcome == bf.FETCH_FAILED
        assert "reset" in (out.detail or "")

    def test_an_empty_body_is_not_treated_as_success(self, tmp_path: Path) -> None:
        backend = _FakeBackend()
        out = bf.backfill_episode(_ep(), backend, corpus_dir=str(tmp_path), opener=_opener(b""))
        assert out.outcome == bf.FETCH_FAILED
        assert backend.uploads == []

    def test_an_episode_without_a_media_url_is_classified_not_crashed(self, tmp_path: Path) -> None:
        out = bf.backfill_episode(
            _ep(media_url=""), _FakeBackend(), corpus_dir=str(tmp_path), opener=_opener()
        )
        assert out.outcome == bf.NO_MEDIA_URL


class TestProvenance:
    """Re-fetched audio is not the audio that produced the transcript. Say so, in the corpus."""

    def test_a_stored_episode_is_stamped_as_refetched(self, tmp_path: Path) -> None:
        backend = _FakeBackend()
        bf.backfill_episode(_ep(), backend, corpus_dir=str(tmp_path), opener=_opener())
        path = tmp_path / ".podcast_scraper" / "audio-archive-provenance.jsonl"
        assert path.is_file()
        row = json.loads(path.read_text(encoding="utf-8").strip())
        assert row["origin"] == "backfill_refetch"
        assert row["byte_identical_to_transcribed_audio"] is False
        assert row["guid"] == "guid-abc"
        assert row["source_url"] == "https://cdn.example.com/ep1.mp3"

    def test_nothing_is_stamped_when_nothing_was_fetched(self, tmp_path: Path) -> None:
        backend = _FakeBackend({_key("guid-abc", ".mp3"): b"x"})
        bf.backfill_episode(
            _ep(), backend, corpus_dir=str(tmp_path), opener=_raising_opener(AssertionError())
        )
        assert not (tmp_path / ".podcast_scraper" / "audio-archive-provenance.jsonl").exists()


class TestPoliteClient:
    def test_repeated_hits_on_one_host_are_spaced(self) -> None:
        slept: List[float] = []
        now = [0.0]

        def _record_sleep(seconds: float) -> None:
            """Record the sleep and advance the fake clock by it."""
            slept.append(seconds)
            now[0] += seconds

        lim = bf.HostRateLimiter(
            2.0,
            sleep=_record_sleep,
            clock=lambda: now[0],
        )
        lim.wait("https://cdn.example.com/a.mp3")
        lim.wait("https://cdn.example.com/b.mp3")
        assert slept and slept[0] == pytest.approx(2.0)

    def test_different_hosts_do_not_throttle_each_other(self) -> None:
        """A global limiter would crawl needlessly across distinct CDNs."""
        slept: List[float] = []
        lim = bf.HostRateLimiter(2.0, sleep=slept.append, clock=lambda: 0.0)
        lim.wait("https://a.example.com/1.mp3")
        lim.wait("https://b.example.com/1.mp3")
        assert slept == []

    def test_a_zero_interval_disables_waiting(self) -> None:
        slept: List[float] = []
        lim = bf.HostRateLimiter(0, sleep=slept.append, clock=lambda: 0.0)
        lim.wait("https://a.example.com/1.mp3")
        lim.wait("https://a.example.com/2.mp3")
        assert slept == []

    def test_requests_identify_themselves(self) -> None:
        seen: Dict[str, Any] = {}

        def _open(req: Any, timeout: int = 0):  # noqa: ARG001
            seen["ua"] = req.get_header("User-agent")
            return _Resp(b"x")

        bf._download("https://cdn.example.com/a.mp3", os.devnull, timeout_s=5, opener=_open)
        assert "podcast-scraper" in (seen["ua"] or "")


class TestDryRun:
    """The report must be trustworthy enough to authorise a several-hundred-file download."""

    def test_it_fetches_nothing(self, tmp_path: Path) -> None:
        backend = _FakeBackend()
        bf.plan_backfill([_ep(), _ep(guid="g2")], backend)
        assert backend.uploads == []

    def test_it_separates_already_archived_from_recoverable(self) -> None:
        backend = _FakeBackend({_key("have", ".mp3"): b"x"})
        report = bf.plan_backfill([_ep(guid="have"), _ep(guid="need")], backend)
        counts = report.counts()
        assert counts[bf.ALREADY_PRESENT] == 1
        assert counts[bf.STORED] == 1

    def test_it_estimates_size_from_enclosure_hints(self) -> None:
        report = bf.plan_backfill(
            [_ep(media_url="https://cdn.example.com/a.mp3?size=1000000")], _FakeBackend()
        )
        assert report.estimated_bytes == 1_000_000

    def test_the_estimate_is_presented_as_a_floor(self) -> None:
        """Only some publishers advertise size; calling a partial sum 'the total' would
        understate a download the operator is authorising."""
        report = bf.plan_backfill(
            [_ep(media_url="https://cdn.example.com/a.mp3?size=1000000"), _ep(guid="g2")],
            _FakeBackend(),
        )
        text = bf.format_dry_run(report)
        assert ">=" in text and "floor" in text

    def test_the_report_groups_by_feed(self) -> None:
        report = bf.plan_backfill(
            [_ep(guid="a", feed_title="Hard Fork"), _ep(guid="b", feed_title="Planet Money")],
            _FakeBackend(),
        )
        text = bf.format_dry_run(report)
        assert "Hard Fork" in text and "Planet Money" in text

    def test_it_says_rolloffs_cannot_be_predicted(self) -> None:
        """A dry run cannot know what a publisher still serves without asking, and must not
        imply otherwise."""
        text = bf.format_dry_run(bf.plan_backfill([_ep()], _FakeBackend()))
        assert "rolled_off" in text


def _write_local_original(corpus: Path, guid: str, payload: bytes = b"original-bytes") -> str:
    """Create a run ``media/`` file + its metadata for ``guid`` and return the media path.

    Mirrors the corpus layout ``offload._iter_run_episode_audio`` reads: a run dir with
    ``metadata/<stem>.metadata.json`` (carrying ``content.audio_relpath``) and the media file.
    """
    run_dir = corpus / "feeds" / "somefeed" / "run_20260101T000000Z"
    (run_dir / "media").mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    stem = "0001-an-episode"
    media_rel = f"media/{stem}.mp3"
    (run_dir / media_rel).write_bytes(payload)
    (run_dir / "metadata" / f"{stem}.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"guid": guid},
                "content": {"audio_relpath": media_rel},
            }
        ),
        encoding="utf-8",
    )
    return str(run_dir / media_rel)


def _write_cache_original(corpus: Path, guid: str, payload: bytes = b"cache-bytes") -> str:
    """Seed the GUID-keyed audio-cache with an original for ``guid``; return its path."""
    from podcast_scraper.utils.audio_cache import cache_path_for_guid, IN_CORPUS_AUDIO_CACHE_REL

    cache_root = corpus / IN_CORPUS_AUDIO_CACHE_REL
    dest = cache_path_for_guid(cache_root, guid, ".mp3")
    assert dest is not None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return str(dest)


class TestHarvestLocalFirst:
    """The v2 core: an existing local original is MOVED into cold, never re-downloaded."""

    def test_a_local_original_is_harvested_not_downloaded(self, tmp_path: Path) -> None:
        media = _write_local_original(tmp_path, "guid-abc", b"the-original")
        backend = _FakeBackend()
        lookup = bf.build_local_lookup(str(tmp_path))

        out = bf.backfill_episode(
            _ep(),
            backend,
            corpus_dir=str(tmp_path),
            local_lookup=lookup,
            opener=_raising_opener(AssertionError("harvest must not hit the network")),
        )
        assert out.outcome == bf.HARVESTED
        assert out.rel_key is not None
        assert backend.store[out.rel_key] == b"the-original"
        assert out.bytes_stored == os.path.getsize(media)

    def test_harvested_audio_is_stamped_byte_identical(self, tmp_path: Path) -> None:
        _write_local_original(tmp_path, "guid-abc")
        backend = _FakeBackend()
        lookup = bf.build_local_lookup(str(tmp_path))

        bf.backfill_episode(
            _ep(), backend, corpus_dir=str(tmp_path), local_lookup=lookup, opener=_opener()
        )
        row = json.loads(
            (tmp_path / ".podcast_scraper" / "audio-archive-provenance.jsonl")
            .read_text(encoding="utf-8")
            .strip()
        )
        assert row["origin"] == "backfill_harvest_local"
        assert row["byte_identical_to_transcribed_audio"] is True
        assert "source_path" in row

    def test_already_in_cold_skips_harvest_and_download(self, tmp_path: Path) -> None:
        _write_local_original(tmp_path, "guid-abc")
        backend = _FakeBackend({_key("guid-abc", ".mp3"): b"in-cold"})
        lookup = bf.build_local_lookup(str(tmp_path))

        out = bf.backfill_episode(
            _ep(),
            backend,
            corpus_dir=str(tmp_path),
            local_lookup=lookup,
            opener=_raising_opener(AssertionError("must not fetch")),
        )
        assert out.outcome == bf.ALREADY_PRESENT
        assert backend.uploads == []

    def test_the_audio_cache_is_the_fallback_source(self, tmp_path: Path) -> None:
        """No run media/, but the GUID-keyed cache holds the original — harvest from there."""
        _write_cache_original(tmp_path, "guid-abc", b"from-the-cache")
        backend = _FakeBackend()
        lookup = bf.build_local_lookup(str(tmp_path))

        out = bf.backfill_episode(
            _ep(),
            backend,
            corpus_dir=str(tmp_path),
            local_lookup=lookup,
            opener=_raising_opener(AssertionError("cache original should be harvested")),
        )
        assert out.outcome == bf.HARVESTED
        assert out.rel_key is not None
        assert backend.store[out.rel_key] == b"from-the-cache"

    def test_a_failed_harvest_upload_falls_through_to_download(self, tmp_path: Path) -> None:
        _write_local_original(tmp_path, "guid-abc")
        lookup = bf.build_local_lookup(str(tmp_path))

        class _RejectUpload(_FakeBackend):
            def upload(self, src_path: str, rel_key: str) -> bool:
                # First (harvest) upload fails; the download path then re-tries the upload.
                if not self.uploads:
                    self.uploads.append("REJECTED")
                    return False
                return super().upload(src_path, rel_key)

        backend = _RejectUpload()
        out = bf.backfill_episode(
            _ep(),
            backend,
            corpus_dir=str(tmp_path),
            local_lookup=lookup,
            opener=_opener(b"downloaded"),
        )
        assert out.outcome == bf.STORED
        assert out.rel_key is not None
        assert backend.store[out.rel_key] == b"downloaded"

    def test_no_local_original_falls_through_to_download(self, tmp_path: Path) -> None:
        backend = _FakeBackend()
        lookup = bf.build_local_lookup(str(tmp_path))  # empty corpus -> no originals
        out = bf.backfill_episode(
            _ep(), backend, corpus_dir=str(tmp_path), local_lookup=lookup, opener=_opener()
        )
        assert out.outcome == bf.STORED


class TestBuildLocalLookup:
    def test_media_is_preferred_over_cache(self, tmp_path: Path) -> None:
        media = _write_local_original(tmp_path, "guid-abc", b"media-copy")
        _write_cache_original(tmp_path, "guid-abc", b"cache-copy")
        lookup = bf.build_local_lookup(str(tmp_path))
        assert lookup("guid-abc") == media

    def test_a_guid_with_no_local_audio_returns_none(self, tmp_path: Path) -> None:
        lookup = bf.build_local_lookup(str(tmp_path))
        assert lookup("nope") is None


class TestResilientDownload:
    """Bounded retry + backoff, but 404/410 (gone) is never retried."""

    def _counting_opener(self, fail_times: int, exc: Exception, payload: bytes = b"ok"):
        calls = {"n": 0}

        def _open(_req: Any, timeout: int = 0):  # noqa: ARG001
            calls["n"] += 1
            if calls["n"] <= fail_times:
                raise exc
            return _Resp(payload)

        return _open, calls

    def test_a_transient_failure_is_retried_then_succeeds(self) -> None:
        opener, calls = self._counting_opener(2, OSError("reset"))
        slept: List[float] = []
        written = bf._download_with_retry(
            "https://cdn.example.com/a.mp3",
            os.devnull,
            timeout_s=5,
            opener=opener,
            max_attempts=3,
            backoff_base_s=1.0,
            sleep=slept.append,
        )
        assert written == 2  # len(b"ok")
        assert calls["n"] == 3
        assert slept == [1.0, 2.0]  # exponential backoff between the 3 attempts

    def test_a_rolled_off_status_is_not_retried(self) -> None:
        exc = HTTPError("https://cdn.example.com/a.mp3", 410, "gone", None, None)
        opener, calls = self._counting_opener(99, exc)
        slept: List[float] = []
        with pytest.raises(HTTPError):
            bf._download_with_retry(
                "https://cdn.example.com/a.mp3",
                os.devnull,
                timeout_s=5,
                opener=opener,
                max_attempts=3,
                sleep=slept.append,
            )
        assert calls["n"] == 1  # gave up immediately, no retry
        assert slept == []

    def test_it_gives_up_after_max_attempts(self) -> None:
        opener, calls = self._counting_opener(99, OSError("dead"))
        slept: List[float] = []
        with pytest.raises(OSError):
            bf._download_with_retry(
                "https://cdn.example.com/a.mp3",
                os.devnull,
                timeout_s=5,
                opener=opener,
                max_attempts=3,
                backoff_base_s=1.0,
                sleep=slept.append,
            )
        assert calls["n"] == 3
        assert slept == [1.0, 2.0]  # slept between attempts, not after the last

    def test_retry_exhaustion_classifies_as_fetch_failed(self, tmp_path: Path) -> None:
        backend = _FakeBackend()
        out = bf.backfill_episode(
            _ep(),
            backend,
            corpus_dir=str(tmp_path),
            opener=_raising_opener(OSError("dead")),
            max_attempts=2,
        )
        assert out.outcome == bf.FETCH_FAILED


class TestDryRunSplit:
    """The dry-run must report move-from-local vs download before anything moves."""

    def test_plan_separates_harvest_from_download(self, tmp_path: Path) -> None:
        _write_local_original(tmp_path, "have-local")
        backend = _FakeBackend({_key("in-cold", ".mp3"): b"x"})
        lookup = bf.build_local_lookup(str(tmp_path))
        report = bf.plan_backfill(
            [_ep(guid="in-cold"), _ep(guid="have-local"), _ep(guid="need-download")],
            backend,
            local_lookup=lookup,
        )
        counts = report.counts()
        assert counts[bf.ALREADY_PRESENT] == 1
        assert counts[bf.HARVESTED] == 1
        assert counts[bf.STORED] == 1

    def test_dry_run_text_shows_the_move_split(self, tmp_path: Path) -> None:
        _write_local_original(tmp_path, "have-local")
        lookup = bf.build_local_lookup(str(tmp_path))
        report = bf.plan_backfill([_ep(guid="have-local")], _FakeBackend(), local_lookup=lookup)
        text = bf.format_dry_run(report)
        assert "move-local" in text
        assert "to move from local" in text

    def test_result_reports_moved_from_local(self) -> None:
        r = bf.BackfillReport(
            outcomes=[
                bf.EpisodeOutcome("g1", "A", "F", bf.HARVESTED, bytes_stored=2_000_000_000),
                bf.EpisodeOutcome("g2", "B", "F", bf.STORED, bytes_stored=1_000_000_000),
            ]
        )
        text = bf.format_result(r)
        assert "moved from local" in text
        assert "2.00 GB" in text and "1.00 GB" in text


class TestResultSummary:
    def test_rolled_off_and_failed_are_reported_separately(self) -> None:
        r = bf.BackfillReport(
            outcomes=[
                bf.EpisodeOutcome("g1", "A", "F", bf.ROLLED_OFF),
                bf.EpisodeOutcome("g2", "B", "F", bf.FETCH_FAILED, detail="503"),
                bf.EpisodeOutcome("g3", "C", "F", bf.STORED, bytes_stored=10),
            ]
        )
        text = bf.format_result(r)
        assert "unrecoverable" in text
        assert "retryable" in text

    def test_stored_bytes_roll_up(self) -> None:
        r = bf.BackfillReport(
            outcomes=[
                bf.EpisodeOutcome("g1", "A", "F", bf.STORED, bytes_stored=2_000_000_000),
                bf.EpisodeOutcome("g2", "B", "F", bf.STORED, bytes_stored=1_000_000_000),
            ]
        )
        assert r.stored_bytes == 3_000_000_000
        assert "3.00 GB" in bf.format_result(r)
