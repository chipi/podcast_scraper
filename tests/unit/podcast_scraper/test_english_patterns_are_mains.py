"""The English naming path is main's, byte for byte.

Operator rule (2026-10-05): "whatever was in English hard coded in some constants, we were
supposed to move to maps, add translations for other languages, but don't touch English."

The multilingual work broke that once without anyone noticing — the English self-introduction
words were widened ("im", "my name's") and twenty-odd English name patterns had their `[A-Z]`
replaced by an accented class — and every test stayed green, because each test asserted what its
author meant to change, not that nothing else changed. This asserts the second thing.

`tests/fixtures/naming/english_patterns.json` holds every module-level regex, word set and
pattern string of the four naming modules, captured from main by
`scripts/tools/capture_english_naming_patterns.py`. A failure here means English matching
changed. Regenerate the golden ONLY when an English change was explicitly asked for.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_GOLDEN = json.loads(
    (_ROOT / "tests" / "fixtures" / "naming" / "english_patterns.json").read_text(encoding="utf-8")
)
_spec = importlib.util.spec_from_file_location(
    "_capture_english", _ROOT / "scripts" / "tools" / "capture_english_naming_patterns.py"
)
assert _spec and _spec.loader
_capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_capture)

#: Names main had that now live elsewhere, with the identical value. A MOVE is fine; a change is
#: not, so each still has to equal its golden.
_MOVED: Dict[Tuple[str, str], Tuple[str, str, Any]] = {
    # The five cue bodies were imported into roster from hosts; they are still in hosts.
    **{
        ("podcast_scraper.providers.ml.diarization.roster", name): (
            "podcast_scraper.speaker_detectors.hosts",
            name,
            None,
        )
        for name in (
            "CUE_FIRST_BODY",
            "CUE_FIRST_PAST_BODY",
            "GREETED_TAIL",
            "NAME_FIRST_REPORT_TAIL",
            "NAME_FIRST_TAIL",
        )
    },
    # The roster read hosts' English host-introduction pattern directly; it now takes its row
    # from the per-language map (2026-10-09), so the English value is that map's "en" row.
    ("podcast_scraper.providers.ml.diarization.roster", "_GUEST_INTRODUCED_BY_HOST_RE"): (
        "podcast_scraper.speaker_detectors.hosts",
        "_GUEST_INTRODUCED_BY_HOST_BY_LANGUAGE",
        "en",
    ),
    # The stated-name possessive became the English row of a per-language map.
    ("podcast_scraper.speaker_detectors.hosts", "_STATED_POSSESSIVE_PREFIX"): (
        "podcast_scraper.speaker_detectors.hosts",
        "_POSSESSIVE_PREFIX_BY_LANGUAGE",
        "en",
    ),
}


def _cases() -> List[Tuple[str, str]]:
    return [(mod, name) for mod, rows in _GOLDEN["modules"].items() for name in sorted(rows)]


def test_the_golden_covers_every_naming_module() -> None:
    assert set(_GOLDEN["modules"]) == set(_capture.MODULES)
    # A golden that captured nothing would pass everything — the first version of the capture
    # script dropped every compiled regex. Assert the patterns are actually in it.
    patterns = sum(
        1
        for rows in _GOLDEN["modules"].values()
        for v in rows.values()
        if "re" in v or "patterns" in v
    )
    assert patterns >= 50, patterns


@pytest.mark.parametrize("module,name", _cases(), ids=lambda x: x.rsplit(".", 1)[-1])
def test_english_is_mains(module: str, name: str) -> None:
    golden = _GOLDEN["modules"][module][name]
    target_mod, target_name, key = _MOVED.get((module, name), (module, name, None))
    value = getattr(importlib.import_module(target_mod), target_name, None)
    assert value is not None, f"{module}.{name} is gone (and not a recorded move)"
    if key is not None:
        value = value[key]
    assert _capture.encode(value) == golden, (
        f"{module}.{name} no longer matches main's English. If this English change was asked for, "
        "regenerate tests/fixtures/naming/english_patterns.json; if not, revert it."
    )
