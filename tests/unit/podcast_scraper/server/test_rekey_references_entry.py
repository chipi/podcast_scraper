"""The re-key command must run as the app-data owner, not as the root of ``docker exec``."""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.server import rekey_references


def test_main_drops_root_to_the_appdata_owner(tmp_path: Path, monkeypatch) -> None:
    seen: list[Path] = []
    monkeypatch.setattr(rekey_references, "drop_root_to_owner", seen.append)
    # --undo with no receipts returns before touching the corpus
    assert rekey_references.main(["--appdata", str(tmp_path), "--undo"]) == 1
    assert seen == [tmp_path]
