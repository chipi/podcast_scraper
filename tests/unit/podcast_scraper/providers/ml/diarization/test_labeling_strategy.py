"""Provider-specific speaker-labeling strategies (ADR-126).

community-1's finer clustering breaks assumptions the Deepgram-tuned heuristics baked in: it splits
hosts into their own clusters (garbled self-intros), leaves cold-open promo readers as their own
first-speaking clusters (sometimes with a stray content turn merged in), and splits guests. These
tests pin the community-1 strategy's fixes against those shapes, with synthetic names, while the
Deepgram base stays frozen at the v2.1.x behavior (covered by test_roster.py).
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.labeling_strategy import (
    _unique_first_name_host,
    Community1LabelingStrategy,
    DiarizationLabelingStrategy,
    labeling_strategy_for,
)
from podcast_scraper.speaker_detectors.boilerplate import recurring_shingles

pytestmark = pytest.mark.unit

_C1 = Community1LabelingStrategy()


def _shingles(*ad_sentences: str, episodes: int = 6):
    """Recurring shingles as if ``ad_sentences`` ran verbatim in every one of ``episodes`` eps."""
    return recurring_shingles([" ".join(ad_sentences)] * episodes)


def test_labeling_strategy_for_selects_community1_for_the_self_hosted_pyannote_path() -> None:
    assert labeling_strategy_for("tailnet_dgx").name == "pyannote_community1"
    assert labeling_strategy_for("deepgram").name == "deepgram"
    assert labeling_strategy_for(None).name == "deepgram"


def test_recorded_voices_robust_catches_an_ad_reader_with_a_merged_stray_content_turn() -> None:
    # community-1's ep0005 shape: a promo reader ("I'm Ada Vale…") whose recurring ad is split into
    # short turns AND has one long non-recurring interview sentence merged into the cluster. The
    # whole-cluster fraction dips below the bar; dropping the single non-recurring longest turn (the
    # merge artifact) recovers the ad. Synthetic recurring ad text.
    ad = "Download the Riverside app now for full coverage of every match in the tournament."
    sh = _shingles(ad)
    turns = [
        ("PROMO", "I cover football for Riverside."),  # short → un-scoreable on its own
        ("PROMO", "And I'm Ada Vale."),  # short self-intro inside the ad
        ("PROMO", ad),  # the recurring body
        ("PROMO", "In other words I am painting a portrait of a founder who reshaped an era here."),
        ("HOST", "Welcome back to the show, a huge episode today, so much for us to get through."),
    ]
    talk = {"PROMO": 22.0, "HOST": 3000.0}  # PROMO barely speaks (share ~0.7%)
    rec = _C1.recorded_voices(turns, None, talk, sh)
    assert "PROMO" in rec  # ad reader caught despite the merged interview sentence
    assert "HOST" not in rec  # the real host is not a recording


def test_recorded_voices_robust_keeps_a_single_turn_ad_and_spares_a_credit_host() -> None:
    ad = "Subscribe now for a special offer on all of our games at the website slash join today."
    sh = _shingles(ad)
    turns = [
        ("ADREAD", ad),  # one recurring turn — its longest turn is itself recurring, so not dropped
        ("COHOST", ad),  # reads the same credits BUT is a real co-host (high share)
        ("COHOST", "And that is the whole show, thanks for listening, see you next week folks."),
    ]
    talk = {"ADREAD": 12.0, "COHOST": 900.0}
    rec = _C1.recorded_voices(turns, None, talk, sh)
    assert "ADREAD" in rec  # single-turn ad still caught (longest turn is recurring → kept)
    assert "COHOST" not in rec  # a co-host who reads credits is spared by the share gate


def test_unique_first_name_host_snaps_a_garbled_surname_but_abstains_on_ambiguity() -> None:
    # "Casey Noonan" fails surname canonicalization for "Casey Newton" but resolves by first name.
    hosts = ["Kevin Roose", "Casey Newton"]
    assert _unique_first_name_host("Casey Noonan", hosts) == "Casey Newton"
    # a mononym has no host first name to match
    assert _unique_first_name_host("Casey", ["Casey Newton"]) is None
    # two hosts share the first name -> abstain (cannot tell which)
    assert _unique_first_name_host("Casey Noonan", ["Casey Newton", "Casey Adams"]) is None
    # the deepgram/base strategy never first-name-snaps
    assert DiarizationLabelingStrategy().snap_extra("Casey Noonan", ["Casey Newton"]) is None


def test_community1_host_candidates_exclude_clips_and_guests_keep_real_hosts() -> None:
    # first-speak order: a promo clip @0 and a cameo @3 open the show; the two real hosts speak at
    # 32/77s. "First to speak" would crown the clips; eligibility drops them (ad + sub-cameo) so the
    # two dominant real speakers take the host slots.
    first_start = {"PROMO": 0.0, "CAMEO": 3.0, "HOST_A": 32.0, "HOST_B": 77.0, "GUEST": 200.0}
    talk = {"PROMO": 24.0, "CAMEO": 8.0, "HOST_A": 1400.0, "HOST_B": 1500.0, "GUEST": 600.0}
    cand = _C1.host_candidate_voices(
        first_start=first_start,
        talk=talk,
        known_hosts=["Kevin Roose", "Casey Newton"],
        conv_guests={"GUEST"},
        montage_suppressed={"PROMO"},  # promo already flagged upstream (recorded/montage)
        cameo_floor=20.0,
    )
    assert cand == {"HOST_A", "HOST_B"}  # the real hosts, not the cold-open clips
