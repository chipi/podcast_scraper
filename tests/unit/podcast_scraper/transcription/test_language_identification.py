"""Which language an episode is in, when it does not read as its show's (#2369)."""

from __future__ import annotations

import pytest

from podcast_scraper.transcription.language_identification import (
    agreed_language,
    CLIP_SECONDS,
    identification_clips,
)

pytestmark = pytest.mark.unit

ENGLISH = "This is the part where you and they talk about what was there, and that is not all."
TURKISH = (
    "Bu bölümde kitabın idari ve bürokratik boyutuna odaklanıyoruz ve bunu tartışıyoruz bugün."
)


def _segs(texts, seconds=10.0):
    out, t = [], 0.0
    for text in texts:
        out.append({"start": t, "end": t + seconds, "text": " ".join([text] * 3)})
        t += seconds
    return out


def test_clips_come_from_stretches_not_in_the_shows_language() -> None:
    texts = [ENGLISH] * 3 + [TURKISH] * 30  # an English sponsor read opens a Turkish episode
    clips = identification_clips(_segs(texts), "en")
    assert len(clips) == 3
    assert all(a >= 30.0 for a, _ in clips), "a clip landed on the English opening"


def test_an_ad_in_the_shows_language_mid_episode_is_never_chosen() -> None:
    texts = [TURKISH] * 15 + [ENGLISH] * 3 + [TURKISH] * 15
    clips = identification_clips(_segs(texts), "en")
    assert len(clips) == 3
    assert not any(150.0 <= a < 180.0 for a, _ in clips)


def test_clips_are_spread_early_middle_late() -> None:
    clips = identification_clips(_segs([TURKISH] * 30), "en")
    starts = [a for a, _ in clips]
    assert starts == sorted(starts) and starts[0] < 100.0 < starts[1] < 200.0 < starts[2]


def test_each_clip_is_about_thirty_seconds() -> None:
    for a, b in identification_clips(_segs([TURKISH] * 30), "en"):
        assert 0 < b - a <= CLIP_SECONDS


def test_an_episode_in_the_shows_language_gives_no_clips() -> None:
    assert identification_clips(_segs([ENGLISH] * 30), "en") == []


def test_too_few_foreign_stretches_give_no_clips() -> None:
    assert identification_clips(_segs([ENGLISH] * 27 + [TURKISH] * 6), "en") == []


def test_a_short_stretch_says_too_little() -> None:
    sparse = [{"start": 30.0 * i, "end": 30.0 * (i + 1), "text": "evet"} for i in range(9)]
    assert identification_clips(sparse, "en") == []


def test_a_show_language_with_no_word_list_gives_no_clips() -> None:
    assert identification_clips(_segs([TURKISH] * 30), "tr") == []


@pytest.mark.parametrize(
    "answers, expected",
    [
        (["tr", "tr", "tr"], "tr"),
        (["TR", " tr", "tr "], "tr"),
        (["tr", "tr", "az"], None),
        (["tr", "tr", None], None),
        (["tr", "tr"], None),
        ([None, None, None], None),
    ],
)
def test_all_three_clips_must_agree(answers, expected) -> None:
    assert agreed_language(answers) == expected


class _Provider:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def transcribe_clip(self, path, language="unset", prompt=None):
        self.calls.append(language)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return {"text": "", "segments": [], "language": answer}


def _identify(monkeypatch, answers, segments=None):
    from podcast_scraper import config as config_mod
    from podcast_scraper.workflow import episode_processor as ep

    monkeypatch.setattr("podcast_scraper.transcription.gap_recovery.cut_clip", lambda *a, **k: None)
    cfg = config_mod.Config(rss="https://example.com/f.xml").model_copy(
        update={"feed_declared_language": "en"}
    )
    provider = _Provider(answers)
    result = {"segments": segments if segments is not None else _segs([TURKISH] * 30)}
    return ep._identify_episode_language(result, cfg, "/tmp/episode.mp3", provider), provider


def test_three_agreeing_clips_name_the_language(monkeypatch) -> None:
    got, provider = _identify(monkeypatch, ["tr", "tr", "tr"])
    assert got == "tr"
    assert provider.calls == [None, None, None], "a clip was sent with a language"


def test_disagreeing_clips_name_nothing(monkeypatch) -> None:
    assert _identify(monkeypatch, ["tr", "tr", "az"])[0] is None


def test_a_failing_clip_names_nothing(monkeypatch) -> None:
    assert _identify(monkeypatch, ["tr", RuntimeError("dgx down"), "tr"])[0] is None


def test_nothing_is_sent_when_no_stretch_qualifies(monkeypatch) -> None:
    got, provider = _identify(monkeypatch, [], segments=_segs([ENGLISH] * 30))
    assert got is None and provider.calls == []


def test_no_media_or_no_clip_call_names_nothing(monkeypatch) -> None:
    from podcast_scraper import config as config_mod
    from podcast_scraper.workflow import episode_processor as ep

    cfg = config_mod.Config(rss="https://example.com/f.xml")
    result = {"segments": _segs([TURKISH] * 30)}
    assert ep._identify_episode_language(result, cfg, None, _Provider(["tr"] * 3)) is None
    assert ep._identify_episode_language(result, cfg, "/tmp/x.mp3", object()) is None
