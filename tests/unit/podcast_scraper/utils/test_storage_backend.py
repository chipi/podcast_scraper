"""#1199: pluggable local-vs-remote audio-archive backend.

The rclone backend is driven by an in-memory fake runner — CI never invokes the
real binary or touches a real remote (project rule: no paid/remote deps in CI).
"""

from __future__ import annotations

import json
import subprocess

import pytest

from podcast_scraper.utils.storage_backend import (
    LocalStorageBackend,
    RcloneStorageBackend,
    StorageBackendError,
)

pytestmark = pytest.mark.unit

KEY = "sha256/ab/cd/abcd1234.mp3"


class TestLocalStorageBackend:
    def test_upload_exists_download_roundtrip(self, tmp_path):
        src = tmp_path / "src.mp3"
        src.write_bytes(b"audio-bytes")
        be = LocalStorageBackend(tmp_path / "archive")

        assert be.exists(KEY) is False
        assert be.upload(str(src), KEY) is True
        assert be.exists(KEY) is True

        dest = tmp_path / "out" / "got.mp3"
        assert be.download(KEY, str(dest)) is True
        assert dest.read_bytes() == b"audio-bytes"

    def test_upload_dedupes_existing(self, tmp_path):
        src = tmp_path / "src.mp3"
        src.write_bytes(b"one")
        be = LocalStorageBackend(tmp_path / "archive")
        assert be.upload(str(src), KEY) is True
        # A second upload of different bytes is a no-op: the key already exists.
        src.write_bytes(b"two-different")
        assert be.upload(str(src), KEY) is True
        got = tmp_path / "g.mp3"
        be.download(KEY, str(got))
        assert got.read_bytes() == b"one"

    def test_download_miss_and_empty_upload(self, tmp_path):
        be = LocalStorageBackend(tmp_path / "archive")
        assert be.download(KEY, str(tmp_path / "x.mp3")) is False
        empty = tmp_path / "empty.mp3"
        empty.write_bytes(b"")
        assert be.upload(str(empty), KEY) is False


class FakeRclone:
    """In-memory rclone stand-in. Remote targets contain ':'; local paths don't."""

    def __init__(self):
        self.store: dict[str, bytes] = {}
        self.calls: list[list[str]] = []

    def __call__(self, args, timeout):
        args = list(args)
        self.calls.append(args)
        sub = args[1]
        if sub == "lsjson":
            target = args[-1]
            if target in self.store:
                out = json.dumps([{"Size": len(self.store[target]), "IsDir": False}])
            else:
                out = "[]"
            return subprocess.CompletedProcess(args, 0, out, "")
        if sub == "copyto":
            a, b = args[2], args[3]
            if ":" in b:  # upload: local a -> remote b
                with open(a, "rb") as fh:
                    self.store[b] = fh.read()
                return subprocess.CompletedProcess(args, 0, "", "")
            # download: remote a -> local b
            if a in self.store:
                with open(b, "wb") as fh:
                    fh.write(self.store[a])
                return subprocess.CompletedProcess(args, 0, "", "")
            return subprocess.CompletedProcess(args, 1, "", "not found")
        return subprocess.CompletedProcess(args, 1, "", "unknown")


class TestRcloneStorageBackend:
    def _be(self, runner, base="archive"):
        return RcloneStorageBackend("testremote", base, runner=runner)

    def test_roundtrip_via_fake_rclone(self, tmp_path):
        fake = FakeRclone()
        be = self._be(fake)
        src = tmp_path / "src.mp3"
        src.write_bytes(b"remote-audio")

        assert be.exists(KEY) is False
        assert be.upload(str(src), KEY) is True
        assert be.exists(KEY) is True
        # target keying: remote:base/key
        assert "testremote:archive/" + KEY in fake.store

        dest = tmp_path / "out.mp3"
        assert be.download(KEY, str(dest)) is True
        assert dest.read_bytes() == b"remote-audio"

    def test_upload_dedupe_skips_copy(self, tmp_path):
        fake = FakeRclone()
        be = self._be(fake)
        src = tmp_path / "src.mp3"
        src.write_bytes(b"x")
        assert be.upload(str(src), KEY) is True
        n_copyto = sum(1 for c in fake.calls if c[1] == "copyto")
        assert be.upload(str(src), KEY) is True  # already exists -> dedupe
        n_copyto_after = sum(1 for c in fake.calls if c[1] == "copyto")
        assert n_copyto_after == n_copyto  # no second copyto

    def test_upload_failure_is_false_not_raise(self, tmp_path):
        def failing(args, timeout):
            if args[1] == "lsjson":
                return subprocess.CompletedProcess(args, 0, "[]", "")
            return subprocess.CompletedProcess(args, 1, "", "boom")

        be = self._be(failing)
        src = tmp_path / "s.mp3"
        src.write_bytes(b"y")
        assert be.upload(str(src), KEY) is False

    def test_missing_binary_fails_loud(self):
        with pytest.raises(StorageBackendError, match="on PATH"):
            RcloneStorageBackend("r", "b", rclone_bin="rclone-does-not-exist-xyz")

    def test_empty_remote_fails_loud(self):
        with pytest.raises(StorageBackendError, match="rclone remote name"):
            RcloneStorageBackend("", runner=FakeRclone())

    def test_size_returns_stored_bytes_and_none_for_miss(self, tmp_path):
        # #1787 evict guard: size() must report the cold object's byte length, None if absent.
        fake = FakeRclone()
        be = self._be(fake)
        src = tmp_path / "s.mp3"
        src.write_bytes(b"twelve bytes")  # 12 bytes
        be.upload(str(src), KEY)
        assert be.size(KEY) == 12
        assert be.size("sha256/zz/zz/absent.mp3") is None  # cache miss -> None

    def test_lsjson_nonzero_rc_is_none(self):
        # A transport / parent-missing non-zero rc must fall to None (KEEP the file at evict).
        def failing(args, timeout):
            return subprocess.CompletedProcess(args, 1, "", "parent not found")

        be = self._be(failing)
        assert be.size(KEY) is None
        assert be.exists(KEY) is False

    def test_lsjson_unparsable_stdout_is_none(self):
        def garbled(args, timeout):
            return subprocess.CompletedProcess(args, 0, "not json", "")

        be = self._be(garbled)
        assert be.size(KEY) is None
        assert be.exists(KEY) is False

    def test_lsjson_timeout_is_none(self):
        def timing_out(args, timeout):
            raise subprocess.TimeoutExpired(cmd=list(args), timeout=timeout)

        be = self._be(timing_out)
        assert be.size(KEY) is None  # never raises

    def test_lsjson_ignores_dir_and_zero_size_entries(self):
        def dir_and_zero(args, timeout):
            out = json.dumps([{"Size": 0, "IsDir": True}, {"Size": 0, "IsDir": False}])
            return subprocess.CompletedProcess(args, 0, out, "")

        be = self._be(dir_and_zero)
        assert be.size(KEY) is None  # a dir or empty object is not a present object


class TestLocalSize:
    def test_local_size(self, tmp_path):
        be = LocalStorageBackend(tmp_path / "archive")
        src = tmp_path / "s.mp3"
        src.write_bytes(b"abcde")  # 5 bytes
        be.upload(str(src), KEY)
        assert be.size(KEY) == 5
        assert be.size("sha256/aa/bb/missing.mp3") is None

    def test_abc_default_size_is_none(self):
        # The base StorageBackend.size default is None ("cannot confirm") so an unknown backend
        # never lets the evict guard delete on an assumed match.
        from podcast_scraper.utils.storage_backend import StorageBackend

        class _Min(StorageBackend):
            def exists(self, rel_key):
                return True

            def upload(self, src_path, rel_key):
                return True

            def download(self, rel_key, dest_path):
                return True

        assert _Min().size("anything") is None


class TestAnAbsoluteBasePathStaysAbsolute:
    """A leading slash is meaning, not noise (#1802).

    `base_path` used to be `.strip("/")`, which removes BOTH ends — turning an absolute path
    into a relative one. Right for a remote whose paths are root-relative (SFTP, S3); wrong for
    an rclone remote of `type=local`, which has no configurable root. There `remote:/abs/path`
    is how you address an absolute location, and stripping the slash silently re-points every
    write at the process CWD.

    It was not hypothetical: the archive e2e tests pass an absolute pytest `tmp_path` against a
    `type=local` remote, so each run wrote cold-store objects into the repo working tree,
    rebuilding the tmp path as directories under the repo root. 2.7MB reached git before anyone
    noticed, and the mechanism recurred on every run.
    """

    @staticmethod
    def _target(base: str) -> str:
        from podcast_scraper.utils.storage_backend import RcloneStorageBackend

        # `runner=` matters: without it the constructor requires the rclone BINARY on PATH
        # (`shutil.which`), which CI's unit job does not have and my machine does. These tests
        # are about path construction only and must not need the real binary — the same
        # local-has-it / CI-does-not blindness that broke this suite twice already today.
        return RcloneStorageBackend(remote="r", base_path=base, runner=FakeRclone())._target(
            "sha256/aa/x.mp3"
        )

    def test_an_absolute_base_path_survives(self) -> None:
        assert self._target("/tmp/cold") == "r:/tmp/cold/sha256/aa/x.mp3"

    def test_a_relative_base_path_is_unchanged(self) -> None:
        """The shipped default. Behaviour here must be byte-identical to before the fix."""
        assert self._target("podcast-audio-archive") == "r:podcast-audio-archive/sha256/aa/x.mp3"

    def test_an_empty_base_path_is_unchanged(self) -> None:
        """What production actually sets (`cloud_balanced.yaml`: the jail root IS the archive)."""
        assert self._target("") == "r:sha256/aa/x.mp3"

    def test_a_trailing_slash_is_still_stripped(self) -> None:
        """Trailing slashes ARE noise — stripping them avoids a doubled separator."""
        assert self._target("archive/") == "r:archive/sha256/aa/x.mp3"
        assert self._target("/tmp/cold/") == "r:/tmp/cold/sha256/aa/x.mp3"

    def test_surrounding_whitespace_is_still_trimmed(self) -> None:
        assert self._target("  archive  ") == "r:archive/sha256/aa/x.mp3"
