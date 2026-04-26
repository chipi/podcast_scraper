"""#656 Stage D — Metrics ad-region excision counters.

Direct unit tests for ``Metrics.record_ad_region_excision`` and
the ``Metrics.finish()`` dict surface that run-summary parsing reads.
"""

from podcast_scraper.workflow.metrics import Metrics


def test_initial_counters_are_zero() -> None:
    m = Metrics()
    assert m.ad_chars_excised_preroll == 0
    assert m.ad_chars_excised_postroll == 0
    assert m.ad_episodes_with_excision_count == 0


def test_record_ad_region_excision_sums_pre_and_post() -> None:
    m = Metrics()
    m.record_ad_region_excision(120, 45)
    m.record_ad_region_excision(30, 0)
    m.record_ad_region_excision(0, 200)
    assert m.ad_chars_excised_preroll == 150
    assert m.ad_chars_excised_postroll == 245
    # Three distinct episodes, all with at least one side > 0.
    assert m.ad_episodes_with_excision_count == 3


def test_record_ad_region_excision_skips_zero_zero() -> None:
    """A call with both sides zero is a no-op for the episode counter —
    GI runs excision on every episode but most have nothing to cut."""
    m = Metrics()
    m.record_ad_region_excision(0, 0)
    m.record_ad_region_excision(10, 0)
    assert m.ad_episodes_with_excision_count == 1
    assert m.ad_chars_excised_preroll == 10
    assert m.ad_chars_excised_postroll == 0


def test_record_ad_region_excision_clamps_negative() -> None:
    """Defensive: if postroll_cut_start somehow exceeds source_length
    upstream (e.g. mocked test data), we store zero not a negative."""
    m = Metrics()
    m.record_ad_region_excision(-5, -10)
    assert m.ad_chars_excised_preroll == 0
    assert m.ad_chars_excised_postroll == 0
    assert m.ad_episodes_with_excision_count == 0


def test_finish_surfaces_ad_excision_counters() -> None:
    """``Metrics.finish()`` is the dict surface that run-summary parsing
    reads via ``metrics.json``."""
    m = Metrics()
    m.record_ad_region_excision(100, 50)
    d = m.finish()
    assert d["ad_chars_excised_preroll"] == 100
    assert d["ad_chars_excised_postroll"] == 50
    assert d["ad_episodes_with_excision_count"] == 1
