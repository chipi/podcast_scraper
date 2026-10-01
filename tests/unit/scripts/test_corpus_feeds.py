"""Guards that the corpus feeds actually describe the corpus.

The defect these exist for: ``build_app_validation_corpus.py`` builds episodes from
``tests/fixtures/transcripts/<ver>/`` and reads RSS only for feed-level metadata, so nothing ever
checked that the feeds listed the corpus's episodes. They didn't — between them the hand-authored
fixtures advertised 26 episodes, with p07 and p08 carrying exactly one each. A real pipeline run can
only process what a feed advertises, so the corpus was unbuildable by the pipeline that is supposed
to build it, and it took pointing the real pipeline at them to notice.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import xml.etree.ElementTree as ET  # nosec B405 — Element/ParseError typing only
from pathlib import Path

import pytest
from defusedxml.ElementTree import parse as safe_parse

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "build_corpus_feeds.py"
RSS_DIR = ROOT / "tests" / "fixtures" / "rss"
_ITUNES_NS = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"

#: Derived from the transcripts on disk, not written down.
#:
#: These were `CORPUS_SHOWS = 9` and `MIN_EPISODES_PER_SHOW = 4`, from a time when the corpus was
#: nine shows of four episodes. Both stopped being true: there are fourteen shows, and the five
#: non-English ones carry ONE episode each. The floor in particular encoded an assumption the
#: corpus had outgrown — "a show has at least four episodes" — and would have reported the new
#: languages as a defect when what they are is a corpus that grew in shows rather than episodes.
#:
#: What the original defect actually was: `p07` and `p08` advertised one episode each while
#: HOLDING four, so a pipeline run could not rebuild them. That is a FEED-vs-CORPUS disagreement,
#: not a minimum — and it is still caught below, now stated as the thing it is.


def _episodes_on_disk_by_show() -> dict[str, int]:
    """show id -> canonical `pNN_eNN` transcripts present, which is what a feed must advertise."""
    out: dict[str, int] = {}
    for txt in (ROOT / "tests" / "fixtures" / "transcripts" / _version()).glob("p*_e*.txt"):
        m = re.fullmatch(r"(p\d+)_e\d+", txt.stem)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


def _version() -> str:
    return (ROOT / "tests" / "fixtures" / "FIXTURES_VERSION").read_text(encoding="utf-8").strip()


def _load():
    spec = importlib.util.spec_from_file_location("_corpus_feeds", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_builder():
    """The corpus builder module, for identity checks against its APP_SHOWS."""
    spec = importlib.util.spec_from_file_location(
        "_app_corpus_builder", ROOT / "scripts" / "build_app_validation_corpus.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _corpus_feed_paths() -> list[Path]:
    return sorted(RSS_DIR.glob("p*_corpus.xml"))


def _items(path: Path) -> list[ET.Element]:
    channel = safe_parse(path).getroot().find("channel")
    return [] if channel is None else channel.findall("item")


class TestCoverage:
    def test_one_feed_per_show(self) -> None:
        paths = _corpus_feed_paths()
        expected = _episodes_on_disk_by_show()
        assert {p.name.split("_")[0] for p in paths} == set(expected), (
            f"a show on disk has no corpus feed, or vice versa: feeds={[p.name for p in paths]} "
            f"shows={sorted(expected)}"
        )

    def test_every_show_advertises_exactly_what_it_holds(self) -> None:
        """The original defect: p07 and p08 advertised ONE episode each while holding four.

        Compared against the TRANSCRIPTS, not against a floor. A floor of four would now fail the
        five non-English shows for having one episode — which is not a defect, it is what a show
        in a new language looks like when it starts. What makes a run unable to rebuild the
        corpus is the feed and the corpus DISAGREEING, at any depth.
        """
        expected = _episodes_on_disk_by_show()
        wrong = {
            path.name: (len(_items(path)), expected.get(path.name.split("_")[0]))
            for path in _corpus_feed_paths()
            if len(_items(path)) != expected.get(path.name.split("_")[0])
        }
        assert not wrong, (
            "these feeds disagree with the transcripts on disk (advertised, actual), so a "
            f"pipeline run cannot rebuild the corpus from them: {wrong}"
        )

    def test_every_episode_has_transcript_and_audio(self) -> None:
        transcripts = ROOT / "tests" / "fixtures" / "transcripts" / _version()
        audio = ROOT / "tests" / "fixtures" / "audio" / _version()
        missing: list[str] = []
        for path in _corpus_feed_paths():
            for item in _items(path):
                guid = (item.findtext("guid") or "").strip()
                if not (transcripts / f"{guid}.txt").is_file():
                    missing.append(f"{guid}: no transcript")
                if not (audio / f"{guid}.mp3").is_file():
                    missing.append(f"{guid}: no audio")
        assert not missing, f"feeds advertise episodes with no media: {missing}"


class TestMeasuredNotAsserted:
    def test_enclosure_length_matches_the_file(self) -> None:
        """Hand-authored feeds carried invented sizes; generated ones are read off disk."""
        audio = ROOT / "tests" / "fixtures" / "audio" / _version()
        wrong: list[str] = []
        for path in _corpus_feed_paths():
            for item in _items(path):
                guid = (item.findtext("guid") or "").strip()
                enclosure = item.find("enclosure")
                mp3 = audio / f"{guid}.mp3"
                if enclosure is None or not mp3.is_file():
                    continue
                declared = int(enclosure.get("length") or 0)
                actual = mp3.stat().st_size
                if declared != actual:
                    wrong.append(f"{guid}: feed says {declared}, file is {actual}")
        assert not wrong, f"enclosure length does not match the media: {wrong}"

    def test_every_item_has_a_nonzero_duration(self) -> None:
        bad: list[str] = []
        for path in _corpus_feed_paths():
            for item in _items(path):
                guid = (item.findtext("guid") or "").strip()
                duration = item.find(f"{_ITUNES_NS}duration")
                text = "" if duration is None else (duration.text or "").strip()
                if not re.match(r"^\d\d:\d\d:\d\d$", text) or text == "00:00:00":
                    bad.append(f"{guid}: {text!r}")
        assert not bad, f"items with missing or zero duration: {bad}"

    def test_cover_art_reference_resolves(self) -> None:
        images = ROOT / "tests" / "fixtures" / "images" / _version()
        missing: list[str] = []
        for path in _corpus_feed_paths():
            channel = safe_parse(path).getroot().find("channel")
            assert channel is not None
            image = channel.find(f"{_ITUNES_NS}image")
            href = "" if image is None else str(image.get("href") or "")
            name = href.rsplit("/", 1)[-1]
            if not name or not (images / name).is_file():
                missing.append(f"{path.name} -> {href!r}")
        assert not missing, f"corpus feeds point at cover art that does not exist: {missing}"


class TestGenerator:
    def test_committed_feeds_are_current(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--check"], cwd=ROOT, capture_output=True, text=True
        )
        assert (
            result.returncode == 0
        ), f"committed corpus feeds are stale:\n{result.stdout}\n{result.stderr}"

    def test_shows_match_the_corpus_builder(self) -> None:
        """If APP_SHOWS gains a show, these feeds must too — else it silently has no episodes.

        They cannot diverge any more: `build_corpus_feeds.SHOWS` IS `APP_SHOWS`, imported rather
        than copied beside it. It used to be a duplicate with a comment promising it "mirrors"
        the builder, and the promise broke the moment p10..p14 were added — the feeds described
        a 40-episode corpus while the builder built a 45-episode one, which is the exact failure
        this script exists to prevent. Asserting identity is what is left worth asserting.
        """
        mod = _load()
        builder_mod = _load_builder()
        assert list(mod.SHOWS) == list(builder_mod.APP_SHOWS), (
            "build_corpus_feeds.SHOWS is no longer the builder's APP_SHOWS — if the import was "
            "replaced by a copy, the two will drift again"
        )
        assert len(mod.SHOWS) == len(
            _episodes_on_disk_by_show()
        ), f"{len(mod.SHOWS)} shows wired, {len(_episodes_on_disk_by_show())} on disk"


_differs_beyond_decoder_rounding = _load()._differs_beyond_decoder_rounding


class TestStalenessToleratesDecoderRoundingOnly:
    """``--check`` must catch a forgotten regeneration without failing on a different ffmpeg.

    A lossy MP3's reported length depends on the decoder: ffmpeg 7.1 and 9.0.1 disagree by up to a
    frame on the same file, which rounds to a 1-second difference in ``HH:MM:SS``. Observed on
    p06/p08 purely by changing which ffmpeg was on PATH. The check compared the documents byte for
    byte, so it failed with no actionable difference — and regenerating to satisfy it would just
    move the failure to everyone on the other version.
    """

    ITEM = (
        "<item><title>Ep {n}</title>"
        "<itunes:duration>{h}:{m:02d}:{s:02d}</itunes:duration></item>"
    )

    def _feed(self, *durations: tuple[int, int, int], titles: list[str] | None = None) -> str:
        names = titles or [f"Ep {i}" for i in range(len(durations))]
        items = "".join(
            f"<item><title>{names[i]}</title>"
            f"<itunes:duration>{h}:{m:02d}:{s:02d}</itunes:duration></item>"
            for i, (h, m, s) in enumerate(durations)
        )
        return f"<rss>{items}</rss>"

    def test_identical_is_not_stale(self) -> None:
        a = self._feed((0, 4, 9), (0, 4, 16))
        assert _differs_beyond_decoder_rounding(a, a) is False

    def test_one_second_of_duration_jitter_is_not_stale(self) -> None:
        # The exact p06 case: 00:04:09 committed, 00:04:08 rendered.
        committed = self._feed((0, 4, 9), (0, 4, 16), (0, 4, 32))
        rendered = self._feed((0, 4, 8), (0, 4, 15), (0, 4, 31))
        assert _differs_beyond_decoder_rounding(committed, rendered) is False

    def test_a_real_duration_change_is_still_stale(self) -> None:
        """A re-recorded or re-encoded episode moves far more than a frame."""
        committed = self._feed((0, 4, 9))
        rendered = self._feed((0, 5, 30))
        assert _differs_beyond_decoder_rounding(committed, rendered) is True

    def test_an_edited_title_is_still_stale(self) -> None:
        """Tolerance applies to durations ONLY — everything else stays byte-exact."""
        committed = self._feed((0, 4, 9), titles=["Ep 0"])
        rendered = self._feed((0, 4, 9), titles=["Ep 0 (remastered)"])
        assert _differs_beyond_decoder_rounding(committed, rendered) is True

    def test_a_new_item_is_still_stale(self) -> None:
        committed = self._feed((0, 4, 9))
        rendered = self._feed((0, 4, 9), (0, 3, 1))
        assert _differs_beyond_decoder_rounding(committed, rendered) is True

    def test_jitter_does_not_mask_a_real_change_elsewhere(self) -> None:
        """The one that matters: rounding on item 1 must not excuse a real change on item 2."""
        committed = self._feed((0, 4, 9), (0, 4, 16))
        rendered = self._feed((0, 4, 8), (0, 9, 44))
        assert _differs_beyond_decoder_rounding(committed, rendered) is True
