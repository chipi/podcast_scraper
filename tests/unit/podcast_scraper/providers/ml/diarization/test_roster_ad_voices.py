"""An advertisement is not a person, and the roster used to disagree.

Hard Fork's pre-roll is two Athletic journalists introducing themselves and plugging their World Cup
app. It carries no sponsor language at all — no "brought to you by", no "dot com slash promo" — so
the ad-pattern list scored ZERO hits, `ad_intervals` came back empty, and every ad-aware guard in
this module was inert.

The ad then walked through the roster's most-trusted signal. `_self_intros_by_voice` holds that a
voice saying "I'm <First Last>" IS that person, and reading its own name aloud is the one thing an
ad narrator always does. So Paul Tenorio (soccer) and Amy Lawrence (football) were crowned the hosts
of a technology podcast in 10 of 10 episodes, taking the roster slots the real hosts needed and
leaving a voice cluster free for a hallucinated "Elon Musk" to claim.

The fix uses no keywords. An ad narrator opens the episode, speaks for 30 seconds, and is never
heard again; a host spans the hour. Measured over those 10 episodes:

    hosts          26-42% of talk, spanning 96-99% of the episode
    guests         11-22% of talk, spanning 18-42%
    ad narrators   0.3-0.4% of talk, spanning 1%

``test_a_short_episode_has_no_ads`` is the guard on the guard, and it is not hypothetical: the first
version of this rule was absolute ("under 90s of talk, only at the edges") and typed the entire cast
of a three-minute fixture as advertising, because in a short clip everyone is near an edge. The
share test is what makes it scale-free.
"""

from __future__ import annotations

from typing import List

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import (
    _canonicalize_to_known_host,
    _edge_ad_voices,
    resolve_speaker_roster,
    VOICE_COMMERCIAL,
)

HOSTS = ["Kevin Roose", "Casey Newton"]


def _seg(spk: str, start: float, end: float) -> DiarizationSegment:
    return DiarizationSegment(start=start, end=end, speaker=spk)


def _hardfork_shaped() -> DiarizationResult:
    """A real episode's shape: a 30s pre-roll read by two voices, then an hour of the show."""
    segs: List[DiarizationSegment] = [
        _seg("SPEAKER_AD1", 0.0, 15.0),  # "I'm Paul Tenorio. I cover soccer..."
        _seg("SPEAKER_AD2", 15.0, 30.0),  # "And I'm Amy Lawrence..."
    ]
    t = 40.0
    while t < 3600.0:  # the actual show: two hosts and a guest, for an hour
        segs.append(_seg("SPEAKER_HOST1", t, t + 20))
        segs.append(_seg("SPEAKER_HOST2", t + 20, t + 35))
        if t > 1800.0:
            segs.append(_seg("SPEAKER_GUEST", t + 35, t + 60))
        t += 60.0
    return DiarizationResult(segments=segs, num_speakers=4)


def _voice_texts() -> dict:
    return {
        "SPEAKER_AD1": "I'm Paul Tenorio. I cover soccer for The Athletic.",
        "SPEAKER_AD2": "And I'm Amy Lawrence. I cover football for The Athletic.",
        # The hosts self-introduce too, and the ASR mangles them — as it really does.
        "SPEAKER_HOST1": "I'm Kevin Russo and this is Hard Fork.",
        "SPEAKER_HOST2": "I'm Casey Noon, and I'm here in New York.",
        "SPEAKER_GUEST": "I'm Adam Rodman, I research clinical decision making.",
    }


def _roster():  # noqa: ANN202
    return resolve_speaker_roster(
        _hardfork_shaped(),
        "I'm Paul Tenorio. I cover soccer for The Athletic. And I'm Amy Lawrence.",
        detected_guests=["Dr. Adam Rodman"],
        known_hosts=HOSTS,
        voice_texts=_voice_texts(),
    )


def test_the_ad_narrators_are_found_without_any_ad_keywords() -> None:
    """No sponsor language exists in this ad. Structure alone has to find it."""
    assert _edge_ad_voices(_hardfork_shaped()) == {"SPEAKER_AD1", "SPEAKER_AD2"}


def test_an_ad_narrator_is_never_named_even_though_it_says_its_own_name() -> None:
    """THE BUG. Being named short-circuited the commercial check — and an ad is always named."""
    roster = _roster()
    for v in ("SPEAKER_AD1", "SPEAKER_AD2"):
        role = roster.by_voice[v]
        assert not role.named, f"{v} was named {role.name!r} — an advertisement is not a person"
        assert role.voice_type == VOICE_COMMERCIAL
        assert roster.display_label_for(v) == "Advertisement"

    named = {r.name for r in roster.by_voice.values() if r.named}
    assert "Paul Tenorio" not in named
    assert "Amy Lawrence" not in named


def test_the_ads_do_not_steal_the_hosts_slots() -> None:
    """The real hosts must still be found — and under their CONFIGURED spelling, not the ASR's.

    The transcript says "Kevin Russo" and "Casey Noon". A corpus that believes those are two more
    people ends up with three Kevins hosting the same show.
    """
    roster = _roster()
    assert roster.by_voice["SPEAKER_HOST1"].name == "Kevin Roose"
    assert roster.by_voice["SPEAKER_HOST1"].role == "host"
    assert roster.by_voice["SPEAKER_HOST2"].name == "Casey Newton"
    assert roster.by_voice["SPEAKER_HOST2"].role == "host"


def test_the_guest_keeps_the_cluster_the_impostor_used_to_take() -> None:
    roster = _roster()
    assert "Rodman" in roster.by_voice["SPEAKER_GUEST"].name


def test_the_hosts_come_from_the_feed_and_the_ads_do_not_take_their_slots() -> None:
    """THE SECOND BUG. The feed says who hosts the show — the roster only has to match the voices.

    Hard Fork's feed says it plainly: "journalists Kevin Roose and Casey Newton". So there are TWO
    host slots and both names are known. What went wrong is that an ADVERT took one of them: the
    pre-roll narrator escaped the ad test, opened the episode, was named from her own ad self-intro,
    and filled a host slot. The cap was then full, so the real co-host dropped through to GUEST
    naming and was published as "Dr. Adam Rodman" — a guest's name on a host, which this module's
    docstring promises never happens.

    Both hosts must be matched, under the FEED's spelling, and neither may be an advert.
    """
    roster = _roster()

    hosts = {v: r for v, r in roster.by_voice.items() if r.role == "host"}
    assert set(hosts) == {"SPEAKER_HOST1", "SPEAKER_HOST2"}, (
        f"the feed names two hosts; the roster matched {sorted(hosts)} — an advert or a guest is "
        "occupying a host slot"
    )
    assert {r.name for r in hosts.values()} == set(HOSTS)
    for r in hosts.values():
        assert "Rodman" not in r.name, (
            f"a HOST was given the guest's name {r.name!r} — the invariant this module promises "
            "('a guest's name is never assigned to a host voice') was broken"
        )
    assert "Rodman" in roster.by_voice["SPEAKER_GUEST"].name


def test_talk_share_never_makes_a_guest_into_a_host() -> None:
    """The rule that was tuned to one show, and the format that breaks it.

    A previous fix inferred hosts structurally — "a host talks a lot and is present from the first
    minute to the last". True of Hard Fork. FALSE of the interview format, where it inverts:

        Invest Like the Best   the GUEST talks 82%, the host 17%
        Latent Space           the GUEST talks 85%

    Here the guest holds ~80% of the episode and speaks from the opening minute to the last, and is
    still the guest — because the FEED names the host. A statistic may not overrule a stated fact.
    """
    segs = [_seg("HOST", 0.0, 60.0)]
    t = 60.0
    while t < 3600.0:  # long answers from the guest, short questions from the host
        segs.append(_seg("GUEST", t, t + 100))
        segs.append(_seg("HOST", t + 100, t + 120))
        t += 120.0
    diar = DiarizationResult(segments=segs, num_speakers=2)

    roster = resolve_speaker_roster(
        diar,
        "Welcome back to the show. My guest today is Brian Chesky.",
        detected_guests=["Brian Chesky"],
        known_hosts=["Patrick O'Shaughnessy"],  # the feed names ONE host
        voice_texts={},
    )

    assert roster.by_voice["HOST"].role == "host"
    assert roster.by_voice["HOST"].name == "Patrick O'Shaughnessy"
    assert (
        roster.by_voice["GUEST"].role == "guest"
    ), "the guest out-talks the host 4:1 and spans the whole episode — and is still the guest"
    assert roster.by_voice["GUEST"].name == "Brian Chesky"


def test_a_short_episode_has_no_ads() -> None:
    """GUARD ON THE GUARD. In a 3-minute clip every voice is near an edge and briefly spoken.

    The first version of this rule was absolute and typed the whole cast as advertising. Anything
    that reintroduces an absolute-only test fails here.
    """
    short = DiarizationResult(
        segments=[
            _seg("A", 0.0, 60.0),
            _seg("B", 60.0, 120.0),
            _seg("A", 120.0, 170.0),
        ],
        num_speakers=2,
    )
    assert _edge_ad_voices(short) == set()


def test_a_brief_mid_episode_speaker_is_not_an_ad() -> None:
    """A caller or a clip in the MIDDLE is not an ad, however brief — it fails the edge test."""
    diar = _hardfork_shaped()
    segs = list(diar.segments) + [_seg("SPEAKER_CLIP", 1800.0, 1815.0)]
    assert "SPEAKER_CLIP" not in _edge_ad_voices(DiarizationResult(segments=segs, num_speakers=5))


class TestCanonicalizingAnAsrMangledHost:
    """The self-intro is transcribed, so it carries the ASR's spelling, and the roster trusts it
    above ``known_hosts``. That is how one host became three people, none spelled correctly."""

    def test_soundex_catches_the_vowel_swap(self) -> None:
        assert _canonicalize_to_known_host("Kevin Russo", HOSTS) == "Kevin Roose"

    def test_edit_distance_catches_what_soundex_misses(self) -> None:
        # "Newton" -> "Noon" is not a soundex match; neither rule alone is enough.
        assert _canonicalize_to_known_host("Casey Noon", HOSTS) == "Casey Newton"

    def test_a_guest_sharing_a_hosts_first_name_is_left_alone(self) -> None:
        """The dangerous direction. Snapping on the first name alone would rename real people."""
        assert _canonicalize_to_known_host("Kevin Systrom", HOSTS) == "Kevin Systrom"
        assert _canonicalize_to_known_host("Casey Affleck", HOSTS) == "Casey Affleck"

    def test_an_unrelated_name_is_untouched(self) -> None:
        assert _canonicalize_to_known_host("Adam Rodman", HOSTS) == "Adam Rodman"
