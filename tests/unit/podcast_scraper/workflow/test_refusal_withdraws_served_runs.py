"""A language-refused episode is not served, whatever ran before it (#2375).

A refusal writes nothing new, and it used to leave earlier runs of the same episode in place: two
Turkish episodes transcribed before the language guard existed stayed served as English. Now every
served run of a refused episode is marked unusable, which the catalog and the indexer both honour.
"""

from __future__ import annotations

import json

import pytest

from podcast_scraper import config as config_mod
from podcast_scraper.languages import episode_is_unusable
from podcast_scraper.workflow import episode_processor as ep

pytestmark = pytest.mark.unit


def _run_meta(root, feed, run, name, episode_id):
    d = root / "feeds" / feed / run / "metadata"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.metadata.json"
    path.write_text(json.dumps({"episode": {"episode_id": episode_id, "title": name}}), "utf-8")
    return path


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "podcast_scraper.workflow.helpers.get_episode_id_from_episode",
        lambda episode, rss: (episode, None),
    )
    cfg = config_mod.Config(rss="https://example.com/f.xml").model_copy(
        update={"output_dir": str(tmp_path)}
    )
    return tmp_path, cfg


def _unusable(path):
    return episode_is_unusable(json.loads(path.read_text("utf-8")))


def test_every_served_run_of_the_refused_episode_is_withdrawn(corpus) -> None:
    root, cfg = corpus
    old = _run_meta(root, "show", "run_20260912", "turkish", "ep-tr")
    newer = _run_meta(root, "show", "run_20261002", "turkish", "ep-tr")
    other = _run_meta(root, "show", "run_20261002", "english", "ep-en")
    assert ep._withdraw_served_runs("ep-tr", cfg, "tr") == 2
    assert "'tr'" in (_unusable(old) or "") and _unusable(newer)
    assert _unusable(other) is None


def test_without_an_identified_language_the_reason_still_says_refused(corpus) -> None:
    root, cfg = corpus
    run = _run_meta(root, "show", "run_1", "x", "ep-x")
    assert ep._withdraw_served_runs("ep-x", cfg, None) == 1
    assert _unusable(run) == "language_refused"


def test_an_already_withdrawn_run_is_left_as_it_was(corpus) -> None:
    root, cfg = corpus
    _run_meta(root, "show", "run_1", "x", "ep-x")
    ep._withdraw_served_runs("ep-x", cfg, "tr")
    assert ep._withdraw_served_runs("ep-x", cfg, "tr") == 0


def test_a_broken_metadata_file_does_not_stop_the_others(corpus) -> None:
    root, cfg = corpus
    bad = root / "feeds" / "show" / "run_1" / "metadata"
    bad.mkdir(parents=True)
    (bad / "broken.metadata.json").write_text("{not json", "utf-8")
    good = _run_meta(root, "show", "run_2", "x", "ep-x")
    assert ep._withdraw_served_runs("ep-x", cfg, "tr") == 1 and _unusable(good)
