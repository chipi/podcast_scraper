"""The board summarizer runs on the pipeline image, which has no FastAPI (RFC-127).

Found by running the sidecar in that image (2026-10-13): ``board_summaries`` reached
``app_corpus_access``, whose top-level ``from fastapi import ...`` failed the import, so every cycle
logged ``No module named 'fastapi'`` and summarised nothing. Each test suite here has FastAPI
installed, so nothing else would notice it coming back. This imports the module, and runs one
sweep, in a fresh interpreter where FastAPI cannot be imported.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[4] / "tests/fixtures/app-validation-corpus/v3"

_PROGRAM = f"""
import sys
sys.modules["fastapi"] = None  # ImportError on any `import fastapi`, as on the pipeline image
import tempfile
from pathlib import Path
from podcast_scraper.server import board_summaries as bs
result = bs.sweep(
    Path({str(CORPUS)!r}),
    Path(tempfile.mkdtemp()),
    summarize=lambda material: "text",
    price=lambda material: 0.0,
    model="m",
    prompt="p",
)
print("OK", result.stopped, dict(result.counts))
"""


def test_the_summarizer_imports_and_sweeps_without_fastapi() -> None:
    out = subprocess.run(
        [sys.executable, "-c", _PROGRAM], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.startswith("OK"), out.stdout
