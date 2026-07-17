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
