"""The DGX pyannote server returns one embedding per speaker (#2304, voice identity capture).

The server is a separate container that cannot import ``podcast_scraper``, so it carries its own
copy of the extraction. This checks that copy against the package's, on the same input, so the
two cannot drift: prod diarizes on the DGX, and a mismatch would store vectors under the wrong
voice. Imports FastAPI (the server module), hence integration rather than unit (U3).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, List

import pytest

from podcast_scraper.providers.ml.diarization.base import speaker_embeddings_from_output

pytestmark = pytest.mark.integration

_APP = Path(__file__).resolve().parents[3] / "infra" / "dgx" / "pyannote-server" / "app.py"


def _server() -> Any:
    spec = importlib.util.spec_from_file_location("pyannote_server_app", _APP)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Annotation:
    def __init__(self, labels: List[str]) -> None:
        self._labels = labels

    def labels(self) -> List[str]:
        return list(self._labels)


class _Output:
    def __init__(self, labels: List[str], rows: List[List[float]]) -> None:
        self.speaker_diarization = _Annotation(labels)
        self.speaker_embeddings = rows


def test_the_server_and_the_package_extract_the_same_vectors() -> None:
    output = _Output(
        ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        [[0.123456789, -1.0], [math.nan, 0.5], [0.25, 0.75]],
    )
    server = _server()._speaker_embeddings(output, output.speaker_diarization)
    assert server == speaker_embeddings_from_output(output)
    assert server == {"SPEAKER_00": [0.12346, -1.0], "SPEAKER_02": [0.25, 0.75]}


def test_a_pyannote_3_result_gives_no_vectors() -> None:
    annotation = _Annotation(["SPEAKER_00"])
    assert _server()._speaker_embeddings(annotation, annotation) == {}
