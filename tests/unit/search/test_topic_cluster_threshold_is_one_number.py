"""The topic-cluster threshold must have exactly ONE source of truth.

af6bed32 retuned the threshold 0.75 -> 0.70 on real-corpus evidence and changed
``config.py``, ``kg/pipeline.py``, ``providers/ml/model_registry.py`` and the
``build_topic_clusters_for_corpus`` keyword default. It did **not** change the two places that
actually rebuild ``search/topic_clusters.json``:

* ``parse_topic_clusters_argv`` carried its own literal ``default=0.75``, so the CLI passed 0.75
  explicitly on every run and the function default was unreachable.
* the ``topic-clusters`` make target passed ``--threshold "${THRESHOLD:-0.75}"``.

Net effect: the retune was inert on the only command an operator runs to regenerate the artifact.
A duplicated literal is what made that possible, so these tests assert there is no duplicate left
rather than asserting the value 0.70 in yet another place.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from podcast_scraper.search.cli_handlers import parse_topic_clusters_argv
from podcast_scraper.search.topic_clusters import DEFAULT_TOPIC_CLUSTER_THRESHOLD

pytestmark = pytest.mark.unit

_ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_the_cli_default_is_the_module_constant() -> None:
    """THE regression: not "is it 0.70", but "does the CLI track whatever the constant says"."""
    args = parse_topic_clusters_argv(["--output-dir", "/tmp/does-not-need-to-exist"])
    assert args.threshold == DEFAULT_TOPIC_CLUSTER_THRESHOLD


def test_an_explicit_threshold_still_wins() -> None:
    """Sweeps pass --threshold; the default must not become a floor or a clamp."""
    args = parse_topic_clusters_argv(["--output-dir", "/tmp/x", "--threshold", "0.82"])
    assert args.threshold == pytest.approx(0.82)


def test_the_config_knob_agrees_with_the_module_constant() -> None:
    """Two independent entry points (YAML config vs the CLI) must not disagree silently."""
    from podcast_scraper.config import Config

    field = Config.model_fields["topic_cluster_threshold"]
    assert field.default == pytest.approx(DEFAULT_TOPIC_CLUSTER_THRESHOLD), (
        "config.topic_cluster_threshold and search.topic_clusters."
        "DEFAULT_TOPIC_CLUSTER_THRESHOLD have drifted apart — the pipeline and the "
        "topic-clusters CLI would build different clusterings from the same corpus"
    )


def test_the_make_target_does_not_pin_a_literal_threshold() -> None:
    """The make target must pass --threshold only when the operator sets THRESHOLD.

    A ``${THRESHOLD:-0.75}`` style default here re-creates the exact bug: the CLI never sees its
    own default because make always supplies one.
    """
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    target = re.search(r"^topic-clusters:\n((?:\t.*\n)+)", makefile, re.M)
    assert target, "the topic-clusters make target moved — re-point this test"
    body = target.group(1)
    assert "--threshold" in body, "the target should still forward an explicit THRESHOLD"
    assert not re.search(r"THRESHOLD:-[0-9]", body), (
        f"the topic-clusters target pins a default threshold, which shadows "
        f"DEFAULT_TOPIC_CLUSTER_THRESHOLD:\n{body}"
    )
