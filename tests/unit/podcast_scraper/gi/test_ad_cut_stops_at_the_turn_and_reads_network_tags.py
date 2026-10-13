"""The ad cut ends at the turn that ends the ad (D30), and reads network promo tags (D15).

D30: the cut snaps to the next ". ", "! " or "? ". A screenplay turn ends ".\\n", which was not a
terminator, so an ad ending a turn took the next speaker's first sentence with it. Measured on El
Hilo (2026-10-10) and on the prod snapshot of 2026-10-09: 134 of 3,235 stored episodes move a
boundary, mostly keeping sign-offs and openers ("Thank you very much, Patrick.") the old cut ate.

D15: El Hilo and Radio Ambulante carry iHeart network cross-promos ("This is an iHeart Podcast",
"Download the iHeart Radio app", "on iHeart Radio, Apple Podcasts") that matched no English ad
pattern, so the ad-free English kept them: `chars_removed: 0`. The cues are the network's own tags,
not the generic "wherever you get your podcasts", which hosts say inside their own intros (Hard
Fork, The Rest Is Science) and which cut those intros whole in the replay. On the same prod
snapshot the tags change nothing.
"""

from __future__ import annotations

from podcast_scraper.gi.ad_regions import excise_ad_regions

_CONTENT = "SPEAKER_06: The region is one of the most biodiverse in the world. " + (
    "It is also where the institutions are absent, and that is the story today. " * 40
)


def test_a_cut_that_ends_a_turn_keeps_the_next_speakers_first_sentence() -> None:
    preroll = (
        "SPEAKER_00: This episode is brought to you by Acme. Visit acme.com to learn more.\n"
        "SPEAKER_01: Use code PODCAST for 20% off your first order.\n"
    )
    cleaned, _, meta = excise_ad_regions(preroll + _CONTENT)
    assert meta.excised_ranges
    assert cleaned.lstrip().startswith("SPEAKER_06: The region is one of the most biodiverse")


def test_an_iheart_network_preroll_is_cut() -> None:
    preroll = (
        "SPEAKER_00: This is an iHeart Podcast. Guaranteed human.\n"
        "SPEAKER_00: A new season of our sister show starts this week. Download the iHeart Radio "
        "app and search for it.\n"
        "SPEAKER_03: Listen on iHeart Radio, Apple Podcasts, or wherever you listen to podcasts.\n"
    )
    cleaned, _, meta = excise_ad_regions(preroll + _CONTENT)
    assert meta.excised_ranges
    assert "iHeart" not in cleaned
    assert cleaned.lstrip().startswith("SPEAKER_06: The region is one of the most biodiverse")


def test_a_host_asking_for_a_subscription_is_not_an_ad_block() -> None:
    intro = (
        "SPEAKER_00: Welcome to the show. Subscribe wherever you get your podcasts, follow us on "
        "Spotify, and listen on Apple Podcasts.\n"
    )
    cleaned, _, meta = excise_ad_regions(intro + _CONTENT)
    assert meta.excised_ranges == []
    assert cleaned.startswith("SPEAKER_00: Welcome to the show.")
