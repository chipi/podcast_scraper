"""scripts/measure/m0043_candidates.py: rule A (stated only by the transcript-filled guest list and
absent from the publisher's text) and its agreement with m0016's own stated-name sources."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.upgrade.migrations import m0016_ad_reader_names_removed as m16

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "measure" / "m0043_candidates.py"
_spec = importlib.util.spec_from_file_location("m0043_candidates", _SCRIPT)
assert _spec and _spec.loader
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

AD = "Ann Reader"


def _episode(
    root: Path,
    *,
    content: Dict[str, Any] | None = None,
    diag: Dict[str, Any] | None = None,
    episode: Dict[str, Any] | None = None,
) -> tuple:
    run = root / "feeds" / "f" / "run_1"
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir()
    body = {"transcript_file_path": "transcripts/ep.txt", "detected_guests": [AD]}
    body.update(content or {})
    doc = {
        "feed": {"title": "The Show", "description": "A weekly show."},
        "episode": episode or {"title": "Episode 1", "description": "We talk."},
        "content": body,
    }
    (run / "metadata" / "ep.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (run / "transcripts" / "ep.speakers.diagnostics.json").write_text(
        json.dumps(diag or {"tried": {}, "summary": {}}), encoding="utf-8"
    )
    return ("feeds/f/run_1/metadata/ep.metadata.json", "SPEAKER_01", AD)


def test_a_name_stated_only_by_the_transcript_guest_list_is_rule_a(tmp_path: Path) -> None:
    r = mc.classify(tmp_path, _episode(tmp_path))
    assert r["verdict"] == "rule_a"
    assert r["sources"] == ["content.detected_guests"] and r["in_publisher_text"] is False


@pytest.mark.parametrize(
    "where",
    [
        {"title": f"Episode 1 with {AD}"},
        {"title": "Episode 1", "description": f"Guest: {AD.upper()}."},
    ],
)
def test_a_name_the_publisher_wrote_is_kept(tmp_path: Path, where: Dict[str, Any]) -> None:
    r = mc.classify(tmp_path, _episode(tmp_path, episode=where))
    assert r["verdict"] == "kept: the publisher's text names them"


@pytest.mark.parametrize(
    ("content", "diag", "source"),
    [
        ({"detected_hosts": [AD]}, None, "content.detected_hosts"),
        (None, {"tried": {"known_hosts": [AD]}}, "tried.known_hosts"),
        (None, {"tried": {"detected_guests": [AD]}}, "tried.detected_guests"),
        (None, {"tried": {"metadata_named": [AD]}}, "tried.metadata_named"),
        (None, {"summary": {"unbound_names": [AD]}}, "summary.unbound_names"),
        ({"speakers": [{"name": AD, "source": "feed"}]}, None, "speakers.stated_entry"),
        ({"speakers": [{"name": AD, "placed": False}]}, None, "speakers.stated_entry"),
    ],
)
def test_a_second_stated_source_keeps_the_name(
    tmp_path: Path, content: Any, diag: Any, source: str
) -> None:
    r = mc.classify(tmp_path, _episode(tmp_path, content=content, diag=diag))
    assert r["verdict"] == f"kept: stated by {source}"
    assert set(r["sources"]) == {"content.detected_guests", source}


def test_a_self_intro_roster_entry_is_not_a_stated_source(tmp_path: Path) -> None:
    content = {"speakers": [{"name": AD, "source": "self_intro", "voice": "SPEAKER_01"}]}
    assert mc.classify(tmp_path, _episode(tmp_path, content=content))["verdict"] == "rule_a"


@pytest.mark.parametrize(
    ("content", "diag"),
    [
        (None, None),
        ({"detected_guests": [], "detected_hosts": [AD]}, None),
        ({"detected_guests": []}, {"tried": {"metadata_named": [AD]}}),
        ({"detected_guests": [], "speakers": [{"name": AD, "source": "hint"}]}, None),
        ({"detected_guests": []}, None),
    ],
)
def test_the_sources_agree_with_m0016s_stated_names(
    tmp_path: Path, content: Any, diag: Any
) -> None:
    """Guardrail: rule A reads the same sources m0016 exempts by. If m0016 grows a source and this
    script does not, a name m0016 counts as stated would come out rule_a here."""
    rel, _voice, name = _episode(tmp_path, content=content, diag=diag)
    meta = tmp_path / rel
    body = json.loads(meta.read_text(encoding="utf-8"))["content"]
    ours = bool(mc.stated_by_source(meta, body, name))
    theirs = any(m16.same_person(name, s) for s in m16._stated_names(meta, body))
    assert ours == theirs


def test_main_without_signatures_refuses(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert mc.main(["--corpus", str(tmp_path)]) == 2
    assert "ad_signatures.json" in capsys.readouterr().out
