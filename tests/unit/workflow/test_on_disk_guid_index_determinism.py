"""``_on_disk_guid_index`` must resolve each guid to the NEWEST run's idx, deterministically.

Why this file is worth more than its size: the value it guards is an ``idx``, and a wrong idx
does not raise. ``_reprocess_existing_episodes`` feeds it to the ``{idx} - *.txt`` transcript
glob, so a stale idx makes ``relabel_only`` / ``rediarize_only`` re-derive a SUPERSEDED run's
transcript and write the result over the current one — a green run with the wrong content.

Two failure shapes, both invisible to a single green run:

1. Adding the ``feeds/<slug>/run_*/`` patterns put the same guid in several run dirs. The
   ``if guid in out: continue`` short-circuit then kept whichever ``Path.glob`` yielded first,
   which is filesystem order — so the answer could differ between two runs over identical bytes.
2. Even made deterministic by sorting, lexicographic order is not recency order once run ids
   carry a prefix. Only the central dedupe rule ranks by timestamp.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import podcast_scraper.workflow.stages.scraping as scraping_mod
from podcast_scraper.workflow.stages.scraping import _on_disk_guid_index


def _write(root, rel_dir, idx, guid, title="Ep"):
    d = root / rel_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{idx:04d} - {title}.metadata.json").write_text(
        json.dumps({"episode": {"guid": guid, "title": title}}), encoding="utf-8"
    )


@pytest.mark.parametrize(
    "rel_dir",
    [
        "run_20260814-055303/metadata",
        "metadata",
        "feeds/my_feed_abc123/run_20260814-055303/metadata",
        "feeds/my_feed_abc123/metadata",
    ],
    ids=["flat-run", "flat", "feed-nested-run", "feed-nested"],
)
def test_every_supported_layout_is_indexed(tmp_path, rel_dir):
    """The feed-nested pair is production's actual layout, not an extra.

    Without them ``reprocess_existing_only`` aborted with "no on-disk episode GUIDs were found"
    against a corpus that was sitting right there — which reads as a missing corpus.
    """
    _write(tmp_path, rel_dir, 7, "guid-1")
    assert _on_disk_guid_index(str(tmp_path))["guid-1"][0] == 7


@pytest.mark.parametrize(
    "prefix",
    ["feeds/my_feed_abc123/", ""],
    ids=["feed-nested", "flat-run"],
)
def test_newest_run_wins_when_an_episode_is_reprocessed(tmp_path, prefix):
    """The load-bearing case: same guid, two runs, different idx.

    Parametrised over BOTH layouts because the first version of this file only covered the
    feed-nested one, and that gap hid a live bug for a whole review cycle: the shared dedupe
    rule bails out on any path without a ``feeds/`` prefix
    (``corpus_scope.feed_dir_and_run_segment_from_relpath`` returns ``(None, None)``), so flat
    ``run_*/metadata/`` candidates passed through undeduped and the caller's first-wins loop
    picked the OLDEST run. The flat layout is not hypothetical — it is a single-feed output
    dir, which is what skip-existing reads during a run.
    """
    _write(tmp_path, f"{prefix}run_20260101-000000/metadata", 3, "guid-1")
    _write(tmp_path, f"{prefix}run_20260814-055303/metadata", 41, "guid-1")

    assert _on_disk_guid_index(str(tmp_path))["guid-1"][0] == 41


@pytest.mark.parametrize("prefix", ["feeds/my_feed_abc123/", ""], ids=["feed-nested", "flat-run"])
def test_uuid_prefixed_run_does_not_beat_a_newer_plain_run(tmp_path, prefix):
    """Recency must come from the parsed timestamp, not lexicographic order — in both layouts."""
    _write(tmp_path, f"{prefix}run_alpha_20260814-055303/metadata", 41, "guid-1")
    _write(tmp_path, f"{prefix}run_zulu_20260101-000000/metadata", 3, "guid-1")

    assert _on_disk_guid_index(str(tmp_path))["guid-1"][0] == 41


def test_newest_run_wins_when_lexicographic_order_disagrees(tmp_path):
    """Sorting alone is not enough — a run id may carry a prefix before the timestamp.

    ``run_zulu_20260101-000000`` sorts AFTER ``run_alpha_20260814-055303`` but is a year older.
    Only ranking by the parsed timestamp gets this right; a naive ``sorted()[-1]`` returns the
    stale run and the test above would still pass.
    """
    feed = "feeds/my_feed_abc123"
    _write(tmp_path, f"{feed}/run_alpha_20260814-055303/metadata", 41, "guid-1")
    _write(tmp_path, f"{feed}/run_zulu_20260101-000000/metadata", 3, "guid-1")

    assert _on_disk_guid_index(str(tmp_path))["guid-1"][0] == 41


def test_result_is_stable_across_repeated_calls(tmp_path):
    """Directly pins determinism: the pre-fix code could answer differently run to run.

    Repetition alone would not have caught it (glob order is stable within a process), so the
    duplicate-across-layouts arrangement is what makes an order-dependent answer observable —
    the same guid is reachable through more than one pattern.
    """
    feed = "feeds/my_feed_abc123"
    _write(tmp_path, f"{feed}/run_20260101-000000/metadata", 3, "guid-1")
    _write(tmp_path, f"{feed}/run_20260814-055303/metadata", 41, "guid-1")
    _write(tmp_path, "run_20260601-000000/metadata", 17, "guid-1")

    answers = {_on_disk_guid_index(str(tmp_path))["guid-1"][0] for _ in range(5)}
    assert answers == {41}


def test_distinct_episodes_are_all_kept(tmp_path):
    """Dedupe must collapse RUNS of one episode, never distinct episodes."""
    feed = "feeds/my_feed_abc123"
    _write(tmp_path, f"{feed}/run_20260814-055303/metadata", 1, "guid-1")
    _write(tmp_path, f"{feed}/run_20260814-055303/metadata", 2, "guid-2", title="Ep2")

    idx = _on_disk_guid_index(str(tmp_path))
    assert {k: v[0] for k, v in idx.items()} == {"guid-1": 1, "guid-2": 2}


def test_description_is_carried_for_speaker_detection(tmp_path):
    """The episode block is returned whole; ``_synthesize_feed_item`` needs ``description``.

    A synthesized item without it silently guts metadata-driven guest naming for every
    reconstructed episode, while live-served episodes in the same run keep it.
    """
    d = tmp_path / "run_20260814-055303" / "metadata"
    d.mkdir(parents=True)
    (d / "0001 - Ep.metadata.json").write_text(
        json.dumps({"episode": {"guid": "g", "title": "Ep", "description": "with Jane Roe"}}),
        encoding="utf-8",
    )
    assert _on_disk_guid_index(str(tmp_path))["g"][1]["description"] == "with Jane Roe"


def test_unreadable_and_idx_less_files_are_skipped_not_fatal(tmp_path):
    d = tmp_path / "run_20260814-055303" / "metadata"
    d.mkdir(parents=True)
    (d / "0001 - Good.metadata.json").write_text(
        json.dumps({"episode": {"guid": "g"}}), encoding="utf-8"
    )
    (d / "0002 - Bad.metadata.json").write_text("{not json", encoding="utf-8")
    (d / "NoIdx - Title.metadata.json").write_text(
        json.dumps({"episode": {"guid": "g2"}}), encoding="utf-8"
    )

    assert set(_on_disk_guid_index(str(tmp_path))) == {"g"}


def test_empty_corpus_returns_empty(tmp_path):
    assert _on_disk_guid_index(str(tmp_path)) == {}


class TestDedupeIsNotOptional:
    """A missing dedupe rule must be a hard failure, never a degrade.

    This class exists because writing it found a bug. The dedupe import was wrapped in
    ``except ImportError`` + a WARNING, which looked like prudent defence. Testing that branch
    showed the fallback was worse than the failure it handled: with no dedupe, the
    ``if guid in out: continue`` short-circuit keeps the FIRST candidate and the globs are
    sorted ascending, so a reprocessed episode resolved to its OLDEST run — deterministically,
    every time. Not "may resolve to a superseded run" as the WARNING claimed: always did.

    That idx drives the ``{idx} - *.txt`` transcript glob, so the degraded path would have had
    relabel_only / rediarize_only re-derive a superseded transcript and write it over the
    current one, behind a warning nobody reads and a zero exit.

    The import now sits at MODULE scope. It was inline under a comment asserting a cycle
    through ``workflow.metadata_generation``; an adversarial review disputed that, and hoisting
    it and importing the package five different ways (scraping-first, package-first,
    corpus_scope-first, cli entrypoint, indexer-first) showed no cycle exists. The
    justification was fabricated. A module-scope import cannot be silently swallowed at all,
    which is a stronger guarantee than the raise it replaces.
    """

    def test_the_import_is_at_module_scope_and_unguarded(self):
        """Pins the structural property: no try/except, no inline import to re-swallow.

        Asserted over the parsed AST, not the source text. A first version grepped the function
        body for "except ImportError" and failed by matching the COMMENT that explains why
        there is no longer one — a test that reads prose instead of code.
        """
        import ast

        tree = ast.parse(Path(scraping_mod.__file__).read_text(encoding="utf-8"))
        module_level = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and "corpus_scope" in (node.module or "")
            for alias in node.names
        }
        assert "dedupe_metadata_paths_newest_run_per_episode" in module_level, (
            "the dedupe import must stay at module scope; the cycle it was once said to avoid "
            "does not exist (verified under five import orders)"
        )

        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_on_disk_guid_index"
        )
        for node in ast.walk(fn):
            assert not isinstance(
                node, (ast.Import, ast.ImportFrom)
            ), "the dedupe import was moved back inline"
            if isinstance(node, ast.ExceptHandler) and node.type is not None:
                assert "ImportError" not in ast.dump(node.type), (
                    "the dedupe import was re-wrapped in except ImportError — that fallback "
                    "resolves a reprocessed episode to its OLDEST run, silently"
                )

    def test_the_swallowed_fallback_would_have_picked_the_oldest_run(self, tmp_path):
        """Pins WHY the swallow had to go, by reproducing exactly what it used to do.

        Kept as a live demonstration rather than a comment: if someone re-adds an
        ``except ImportError`` that falls through to plain first-wins, this documents the
        result they would be shipping.
        """
        feed = "feeds/my_feed_abc123"
        _write(tmp_path, f"{feed}/run_20260101-000000/metadata", 3, "guid-1")
        _write(tmp_path, f"{feed}/run_20260814-055303/metadata", 41, "guid-1")

        first_wins = {}
        for pattern in (
            "run_*/metadata/*.metadata.json",
            "metadata/*.metadata.json",
            "feeds/*/run_*/metadata/*.metadata.json",
            "feeds/*/metadata/*.metadata.json",
        ):
            for p in sorted(tmp_path.glob(pattern)):
                guid = json.loads(p.read_text())["episode"]["guid"]
                first_wins.setdefault(guid, int(p.name.split(" ")[0]))

        assert first_wins["guid-1"] == 3, "the un-deduped path resolves to the OLDEST run"
        assert _on_disk_guid_index(str(tmp_path))["guid-1"][0] == 41
