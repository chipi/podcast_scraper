"""Public modules the split keeps must import without loading the speaker stage (E04 S2).

Checked in a fresh interpreter: inside this test process other tests have long since imported
the diarization package, so `sys.modules` here proves nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[3] / "src"

#: Public modules whose import must not reach the speaker stage.
PUBLIC = [
    "podcast_scraper.providers.resilience.fallback",
    "podcast_scraper.providers.tailnet_dgx",
    "podcast_scraper.stage_hooks",
    "podcast_scraper.text_language",
]
#: Module prefixes that belong to the speaker stage.
SPEAKER_STAGE = (
    "podcast_scraper.providers.ml.diarization",
    "podcast_scraper.speaker_detectors",
    "podcast_scraper.speaker_stage",
)

_PROBE = """
import importlib, json, sys
importlib.import_module(sys.argv[1])
prefixes = tuple(sys.argv[2:])
print(json.dumps(sorted(m for m in sys.modules if m.startswith(prefixes))))
"""


@pytest.mark.parametrize("module", PUBLIC)
def test_importing_it_loads_no_speaker_stage_module(module: str) -> None:
    env = dict(os.environ, PYTHONPATH=str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, module, *SPEAKER_STAGE],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def test_the_dgx_diarization_provider_is_still_importable_from_the_package() -> None:
    from podcast_scraper.providers.tailnet_dgx import TailnetDgxDiarizationProvider
    from podcast_scraper.providers.tailnet_dgx.diarization_provider import (
        TailnetDgxDiarizationProvider as direct,
    )

    assert TailnetDgxDiarizationProvider is direct
