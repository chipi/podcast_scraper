"""The reprocess workflow's inline id list never splits an id on its comma (#2187, 2026-10-13).

Tag-URI guids (``tag:soundcloud,2010:tracks/…``) always contain a comma; the old ``tr ',' '\\n'``
cut them into fragments that matched no episode, so 8 backfill episodes could not be sent.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "scripts" / "ops" / "split_episode_ids.sh"
WORKFLOW = REPO / ".github" / "workflows" / "reprocess-prod.yml"


def _split(text: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT)], input=text, capture_output=True, text=True, check=False
    )


def test_a_tag_uri_id_keeps_its_comma() -> None:
    out = _split("tag:soundcloud,2010:tracks/2067916744\n")
    assert out.returncode == 0
    assert out.stdout.splitlines() == ["tag:soundcloud,2010:tracks/2067916744"]


@pytest.mark.parametrize(
    "text",
    [
        "a1\ntag:blogger.com,1999:blog-17.post-5\nb2\n",  # the chain: one per line
        "a1 tag:blogger.com,1999:blog-17.post-5 b2",  # typed in the GitHub form
        "  a1\n\n\ttag:blogger.com,1999:blog-17.post-5  \r\nb2  \n",  # stray blanks / CRLF
    ],
)
def test_lines_and_spaces_separate_ids(text: str) -> None:
    out = _split(text)
    assert out.returncode == 0
    assert out.stdout.splitlines() == ["a1", "tag:blogger.com,1999:blog-17.post-5", "b2"]


@pytest.mark.parametrize("text", ["", "   \n\t\n"])
def test_no_id_left_fails(text: str) -> None:
    out = _split(text)
    assert out.returncode == 1
    assert out.stdout == ""


def test_the_workflow_uses_the_script_and_never_splits_on_commas() -> None:
    wf = WORKFLOW.read_text(encoding="utf-8")
    assert "scripts/ops/split_episode_ids.sh" in wf
    assert "tr ',' '\\n'" not in wf
