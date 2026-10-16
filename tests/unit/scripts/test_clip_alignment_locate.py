"""``locate`` in scripts/eval/clip_alignment_check.py: where a clip's quote starts in what Whisper
heard, with the quote's words in order, and one misheard word on the way."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "eval" / "clip_alignment_check.py"
_spec = importlib.util.spec_from_file_location("clip_alignment_check", _SCRIPT)
assert _spec and _spec.loader
check = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = check  # its dataclass looks itself up there
_spec.loader.exec_module(check)

QUOTE = (
    "Socrates says these men are like... Glaucus says, well, they're peculiar prisoners. "
    "And Socrates says, no, no, they're like us."
)


def _heard(text: str, start: float = 100.0) -> list[dict]:
    return [{"word": w, "start": start + 0.3 * i} for i, w in enumerate(text.split())]


def test_the_quote_is_found_where_it_starts() -> None:
    words = _heard("the shadows on the wall " + QUOTE)
    assert check.locate(words, check.content_words(QUOTE)) == pytest.approx(101.5)


def test_one_misheard_name_does_not_lose_the_quote() -> None:
    """Whisper heard "Glaucus" as "glorkeren" on the real episode: the clip was right (+0.1 s), the
    checker said "not located"."""
    heard = QUOTE.replace("Glaucus", "glorkeren")
    words = _heard("the shadows on the wall " + heard)
    assert check.locate(words, check.content_words(QUOTE)) == pytest.approx(101.5)


def test_words_out_of_order_are_not_the_quote() -> None:
    words = _heard("prisoners peculiar they're well says glaucus like men these says socrates")
    assert check.locate(words, check.content_words(QUOTE)) is None
