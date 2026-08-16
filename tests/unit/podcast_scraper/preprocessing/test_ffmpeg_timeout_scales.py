"""The ffmpeg budget scales with input size (#1657 acceptance, GitHub #558/#561).

A flat 300s was an undeclared file-size limit. Measured on the acceptance corpus:

    idle host    30.8 MB -> 136.8s   91.5 MB -> 241.5s   105.6 MB -> 246.0s
    loaded host  30.8 MB -> 202.0s   75-121 MB -> TIMEOUT

The same 30.8 MB file took 48 % longer under load (Colima's QEMU was burning 3+ cores), and
91.5 MB already used 241s of the 300s budget on an idle box — so every large episode was one
busy moment away from silently falling back to UNPREPROCESSED audio. That fallback is not
cosmetic: the original file then really does exceed the 25 MB upload cap, costs more to
transcribe, and skips the mono/16 kHz/loudness normalisation the transcriber expects.

The budget is a hung-process guard, not a performance target: being too tight loses quality
silently, being too loose costs a few minutes of noticing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.preprocessing.audio import ffmpeg_processor as fp

pytestmark = [pytest.mark.unit]


def _file_of_mb(tmp_path: Path, mb: float, name: str = "a.mp3") -> str:
    p = tmp_path / name
    with open(p, "wb") as fh:
        fh.write(b"\0" * int(mb * 1024 * 1024))
    return str(p)


class TestItScalesWithSize:
    def test_a_small_file_gets_at_least_the_old_budget(self, tmp_path: Path) -> None:
        """This change may only widen the window — never narrow it for anyone."""
        assert fp.ffmpeg_timeout_for(_file_of_mb(tmp_path, 1)) >= 300.0

    def test_a_large_file_gets_more_than_the_old_flat_budget(self, tmp_path: Path) -> None:
        assert fp.ffmpeg_timeout_for(_file_of_mb(tmp_path, 120)) > 300.0

    def test_bigger_input_never_gets_a_smaller_budget(self, tmp_path: Path) -> None:
        small = fp.ffmpeg_timeout_for(_file_of_mb(tmp_path, 10, "s.mp3"))
        large = fp.ffmpeg_timeout_for(_file_of_mb(tmp_path, 100, "l.mp3"))
        assert large > small

    @pytest.mark.parametrize("mb,observed_seconds", [(30.8, 202.0), (91.5, 241.5), (105.6, 246.0)])
    def test_every_real_observation_fits_with_headroom(
        self, tmp_path: Path, mb: float, observed_seconds: float
    ) -> None:
        """The actual measurements from the acceptance run, including the loaded-host one that
        timed out under the flat ceiling. Each must now fit with room to spare."""
        budget = fp.ffmpeg_timeout_for(_file_of_mb(tmp_path, mb))
        assert budget > observed_seconds * 1.5, (
            f"{mb} MB took {observed_seconds}s in reality; budget {budget}s leaves too little "
            "room for a slow host"
        )

    def test_the_121mb_case_that_timed_out_now_fits(self, tmp_path: Path) -> None:
        """121.33 MB timed out at 300s in part 2. Extrapolating the idle-host fit
        (~92s + 1.5s/MB) and the 1.48x load factor gives ~400s; the budget must clear that."""
        assert fp.ffmpeg_timeout_for(_file_of_mb(tmp_path, 121.33)) > 400.0


class TestItStaysBounded:
    def test_it_is_capped(self, tmp_path: Path) -> None:
        """An unbounded budget turns a wedged ffmpeg into a wedged pipeline."""
        assert fp.ffmpeg_timeout_for(_file_of_mb(tmp_path, 5000)) <= fp.FFMPEG_TIMEOUT_MAX_SECONDS

    def test_a_missing_file_falls_back_to_the_base_budget(self, tmp_path: Path) -> None:
        """Never raise from a timeout calculation — that would fail the episode outright."""
        assert fp.ffmpeg_timeout_for(str(tmp_path / "nope.mp3")) == fp.FFMPEG_TIMEOUT_BASE_SECONDS

    def test_an_unreadable_path_falls_back(self) -> None:
        assert fp.ffmpeg_timeout_for("") == fp.FFMPEG_TIMEOUT_BASE_SECONDS


class TestTheCallSitesUseIt:
    """A scaled helper nothing calls is worse than no helper — it reads as fixed."""

    def test_no_hardcoded_300s_timeout_remains(self) -> None:
        src = Path(fp.__file__).read_text(encoding="utf-8")
        assert "timeout=300.0" not in src, "a call site still uses the flat budget"

    def test_both_encode_paths_use_the_scaled_budget(self) -> None:
        src = Path(fp.__file__).read_text(encoding="utf-8")
        assert src.count("timeout=ffmpeg_timeout_for(input_path)") == 2

    def test_the_timeout_message_does_not_hardcode_300(self) -> None:
        """The old message said "after 300s" regardless of the real budget — a log line that
        lies about the number it just enforced."""
        src = Path(fp.__file__).read_text(encoding="utf-8")
        assert "timed out after 300s" not in src
