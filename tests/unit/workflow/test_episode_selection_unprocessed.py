"""``episode_selection=unprocessed`` — make ``max_episodes`` count WORK, not feed positions.

THE DRIFT (C3, measured on the 2026-08-31 batch). ``episode_offset`` counts positions in the
feed as it stands right now, and positions move as a feed publishes. "Give me the next 10 I do
not have" expressed as "skip the newest 10" is only equivalent while the feed is frozen. Feeds
finished at 8, 8, 8, 8, 7, 7, 7 of a requested 10 because each had published 2-3 episodes since
the previous run, so ``offset=10`` landed shallower than intended and re-selected episodes
already ingested.

Nothing was corrupted: ``skip_existing`` dropped the overlap, which is the safety net WORKING.
The defect is that the drop happens AFTER ``max_episodes`` has been spent on those items, so
the limit is consumed by work that will not happen. The shortfall grows with the gap between
runs — the failure mode is "quietly does less than you asked", indefinitely.

Filtering by GUID before the limit is immune to feed movement: the same principle
``corpus_metadata_index`` applies elsewhere — resolve by stable identity, never by position.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from podcast_scraper.workflow.stages import scraping


def _item(guid: str, title: str = "Ep") -> ET.Element:
    it = ET.Element("item")
    ET.SubElement(it, "guid").text = guid
    ET.SubElement(it, "title").text = title
    return it


def _corpus_with(tmp_path, *guids: str):
    """A feed output dir whose run_*/metadata declares *guids* as already ingested."""
    d = tmp_path / "run_20260814-055303" / "metadata"
    d.mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(guids, start=1):
        (d / f"{i:04d} - Ep.metadata.json").write_text(
            json.dumps({"episode": {"guid": g, "title": "Ep"}}), encoding="utf-8"
        )
    return tmp_path


def _cfg(tmp_path, **over):
    base = dict(output_dir=str(tmp_path), episode_selection="unprocessed", episode_offset=0)
    base.update(over)
    return SimpleNamespace(**base)


class TestFiltering:
    def test_already_ingested_items_are_dropped(self, tmp_path):
        _corpus_with(tmp_path, "g1", "g2")
        items = [_item("g1"), _item("g2"), _item("g3"), _item("g4")]

        kept = scraping._drop_already_ingested(items, _cfg(tmp_path))

        assert [scraping.extract_item_guid(i) for i in kept] == ["g3", "g4"]

    def test_empty_corpus_keeps_everything(self, tmp_path):
        items = [_item("g1"), _item("g2")]
        assert len(scraping._drop_already_ingested(items, _cfg(tmp_path))) == 2

    def test_an_item_with_no_guid_is_KEPT(self, tmp_path):
        """Unidentifiable must not mean discarded.

        Dropping it would silently shrink the reachable feed — the exact class of quiet loss
        this mode exists to remove — and ``skip_existing`` still guards the duplicate case
        downstream.
        """
        _corpus_with(tmp_path, "g1")
        anon = ET.Element("item")
        ET.SubElement(anon, "title").text = "No guid"
        kept = scraping._drop_already_ingested([_item("g1"), anon], _cfg(tmp_path))
        assert len(kept) == 1 and scraping.extract_item_guid(kept[0]) is None

    def test_it_reports_what_it_dropped(self, tmp_path, caplog):
        _corpus_with(tmp_path, "g1", "g2")
        caplog.set_level("INFO")
        scraping._drop_already_ingested([_item("g1"), _item("g2"), _item("g3")], _cfg(tmp_path))
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "already ingested" in msg and "2" in msg


class TestTheDriftItFixes:
    """The scenario, reproduced: a feed that grew between runs."""

    @staticmethod
    def _simulate_positional(items, offset, limit, ingested):
        """Today's behaviour: slice by position, THEN skip_existing drops the overlap."""
        window = items[offset:][:limit]
        return [i for i in window if scraping.extract_item_guid(i) not in ingested]

    def test_positional_selection_under_delivers_after_the_feed_grows(self, tmp_path):
        # Run 1 ingested the 10 newest: g10..g1 (newest first).
        ingested = {f"g{n}" for n in range(1, 11)}
        # Two new episodes published since; feed is now n1, n2, g1..g20 newest-first.
        feed = [_item("n1"), _item("n2")] + [_item(f"g{n}") for n in range(1, 21)]

        got = self._simulate_positional(feed, offset=10, limit=10, ingested=ingested)

        assert len(got) == 8, "offset=10 lands 2 items shallow, so 2 of the 10 are re-selections"

    def test_unprocessed_selection_delivers_the_full_ask(self, tmp_path):
        _corpus_with(tmp_path, *[f"g{n}" for n in range(1, 11)])
        feed = [_item("n1"), _item("n2")] + [_item(f"g{n}") for n in range(1, 21)]

        kept = scraping._drop_already_ingested(feed, _cfg(tmp_path))[:10]

        assert len(kept) == 10, "the limit must count episodes of WORK"
        assert all(
            scraping.extract_item_guid(i) not in {f"g{n}" for n in range(1, 11)} for i in kept
        )

    def test_it_also_picks_up_the_newly_published_ones(self, tmp_path):
        """A positional offset SKIPS new episodes at the head; identity-based selection does not."""
        _corpus_with(tmp_path, *[f"g{n}" for n in range(1, 11)])
        feed = [_item("n1"), _item("n2")] + [_item(f"g{n}") for n in range(1, 21)]

        kept = scraping._drop_already_ingested(feed, _cfg(tmp_path))[:10]
        guids = [scraping.extract_item_guid(i) for i in kept]

        assert guids[:2] == ["n1", "n2"]


class TestItIsActuallyWiredIn:
    """Drive ``prepare_episodes_from_feed``, not just the helper.

    Mutation-testing exposed this gap: disabling the ``episode_selection == 'unprocessed'``
    branch entirely left every other test in this file GREEN, because they all call
    ``_drop_already_ingested`` directly. A filter that is never reached is indistinguishable
    from no filter — and "the wiring, not the function, was broken" has been the shape of three
    separate defects in this change set.
    """

    @staticmethod
    def _feed(*guids):
        return SimpleNamespace(
            items=[_item(g) for g in guids], base_url="https://example.com/", title="F"
        )

    @staticmethod
    def _full_cfg(tmp_path, **over):
        from podcast_scraper import config

        base = {
            "rss_url": "https://example.com/f.xml",
            "output_dir": str(tmp_path),
            "max_episodes": 2,
            "episode_selection": "unprocessed",
        }
        base.update(over)
        return config.Config.model_validate(base)

    def test_unprocessed_mode_reaches_the_filter(self, tmp_path):
        _corpus_with(tmp_path, "g1", "g2")
        feed = self._feed("g1", "g2", "g3", "g4")

        eps = scraping.prepare_episodes_from_feed(feed, self._full_cfg(tmp_path))

        assert [scraping.extract_item_guid(e.item) for e in eps] == ["g3", "g4"], (
            "the limit of 2 must yield two UNINGESTED episodes; if g1/g2 appear the filter "
            "never ran and the limit was spent on work that skip_existing will discard"
        )

    def test_position_mode_still_spends_the_limit_positionally(self, tmp_path):
        """The default must be untouched — this is the behaviour #521 documents."""
        _corpus_with(tmp_path, "g1", "g2")
        feed = self._feed("g1", "g2", "g3", "g4")

        eps = scraping.prepare_episodes_from_feed(
            feed, self._full_cfg(tmp_path, episode_selection="position")
        )

        assert [scraping.extract_item_guid(e.item) for e in eps] == ["g1", "g2"]


class TestDefaultIsUnchanged:
    def test_position_is_the_default(self):
        from podcast_scraper import config

        cfg = config.Config.model_validate({"rss_url": "https://example.com/f.xml"})
        assert cfg.episode_selection == "position", (
            "episode_offset is documented positional behaviour (#521) with an E2E suite; "
            "changing the default would redefine it under existing callers"
        )

    @pytest.mark.parametrize("mode", ["position", "unprocessed"])
    def test_both_modes_are_accepted(self, mode):
        from podcast_scraper import config

        cfg = config.Config.model_validate(
            {"rss_url": "https://example.com/f.xml", "episode_selection": mode}
        )
        assert cfg.episode_selection == mode

    def test_offset_with_unprocessed_warns(self, caplog):
        """Coherent but almost always a leftover from a positional recipe."""
        from podcast_scraper import config

        caplog.set_level("WARNING")
        config.Config.model_validate(
            {
                "rss_url": "https://example.com/f.xml",
                "episode_selection": "unprocessed",
                "episode_offset": 10,
            }
        )
        assert any("POSITIONAL" in r.getMessage() for r in caplog.records)
