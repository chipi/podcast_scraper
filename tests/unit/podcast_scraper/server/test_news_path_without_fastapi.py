"""The cluster cards' news path runs without fastapi (2026-11-23 prod incident).

The Player's board summarizer is built from public's pipeline-llm image, which has no fastapi. Its
news storyline sweep calls ``relational_view._with_news``, which imported the news helpers from the
route module ``server.routes.corpus_news``, and that module imports fastapi at its top: every sweep
failed with ModuleNotFoundError (98 failures in the first hours; GlitchTip podcast-pipeline 1190).
The test runs the path in a child interpreter where ``import fastapi`` fails, as in that image.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SRC = Path(__file__).resolve().parents[4] / "src"

_CHILD = """
import sys
from pathlib import Path

for name in ("fastapi", "starlette"):
    sys.modules[name] = None  # ``import fastapi`` raises ImportError, as in the pipeline image

from podcast_scraper.server import relational_view as rv
from podcast_scraper.server.schemas import AppClusterCard

root = Path(sys.argv[1])
card = AppClusterCard(id="thc:none", label="None", member_topics=[], episode_count=0, episodes=[])
out = rv._with_news(root, card)
assert out.article_count == 0 and out.articles == [], out
print("NEWS-PATH-OK")
"""


def test_cluster_news_path_imports_no_fastapi(tmp_path: Path) -> None:
    child = subprocess.run(
        [sys.executable, "-c", _CHILD, str(tmp_path)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(_SRC), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert child.returncode == 0 and "NEWS-PATH-OK" in child.stdout, child.stderr[-2000:]
