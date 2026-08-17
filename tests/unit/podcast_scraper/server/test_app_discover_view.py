"""Unit tests for :mod:`podcast_scraper.server.app_discover_view` (#1098).

Covers the two pure-ish pieces of the personalized-discovery ranker:

* :func:`_significance` — the provisional content-depth weighting
  (``+2`` GI, ``+1`` KG, ``+0.2`` per summary bullet capped at five).
* :func:`rank_discover` — significance × interest-affinity re-ranking with a
  recency (``-idx``) tie-break, exercised against a tiny on-disk fixture corpus
  so the per-episode KG loads run through the real loader (direct row
  construction would not exercise ``_episode_features``).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from podcast_scraper.server.app_discover_view import (
    _newest_publish_date,
    _recency_boost,
    _significance,
    rank_discover,
)
from podcast_scraper.server.app_ranking_config import (
    DEFAULT_RANKING_CONFIG,
    SIGNAL_INTEREST_AFFINITY,
    SIGNAL_RECENCY,
)
from podcast_scraper.server.corpus_catalog import (
    build_catalog_rows_cumulative,
    CatalogEpisodeRow,
)

pytestmark = [pytest.mark.unit]


def _row(*, has_gi: bool, has_kg: bool, bullets: tuple[str, ...]) -> CatalogEpisodeRow:
    """Minimal catalog row carrying only the fields ``_significance`` reads."""
    return CatalogEpisodeRow(
        metadata_relative_path="metadata/x.metadata.json",
        feed_id="f",
        feed_title=None,
        episode_id="e",
        episode_title="E",
        publish_date=None,
        summary_title=None,
        summary_bullets=bullets,
        summary_text=None,
        gi_relative_path="metadata/x.gi.json",
        kg_relative_path="metadata/x.kg.json",
        bridge_relative_path="metadata/x.bridge.json",
        has_gi=has_gi,
        has_kg=has_kg,
        has_bridge=False,
    )


# --------------------------------------------------------------------------- #
# _significance
# --------------------------------------------------------------------------- #


def test_significance_baseline_is_one() -> None:
    assert _significance(_row(has_gi=False, has_kg=False, bullets=())) == 1.0


def test_significance_gi_adds_two() -> None:
    assert _significance(_row(has_gi=True, has_kg=False, bullets=())) == 3.0


def test_significance_kg_adds_one() -> None:
    assert _significance(_row(has_gi=False, has_kg=True, bullets=())) == 2.0


def test_significance_bullets_add_point_two_each() -> None:
    row = _row(has_gi=False, has_kg=False, bullets=("a", "b", "c"))
    assert _significance(row) == pytest.approx(1.0 + 3 * 0.2)


def test_significance_bullets_capped_at_five() -> None:
    row = _row(has_gi=False, has_kg=False, bullets=tuple("abcdefghij"))  # 10 bullets
    # capped at 5 → 1.0 + 5 * 0.2 == 2.0; the extra five bullets are inert.
    assert _significance(row) == pytest.approx(2.0)


def test_significance_combines_all_weights() -> None:
    row = _row(has_gi=True, has_kg=True, bullets=("a", "b"))
    # 1 (base) + 2 (gi) + 1 (kg) + 2 * 0.2 (bullets) == 4.4
    assert _significance(row) == pytest.approx(4.4)


# --------------------------------------------------------------------------- #
# rank_discover (small on-disk fixture corpus)
# --------------------------------------------------------------------------- #


def _write_episode(
    root: Path,
    *,
    stem: str,
    episode_id: str,
    topics: list[tuple[str, str]],
    published: str,
    with_gi: bool = False,
    persons: list[tuple[str, str]] | None = None,
) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "myfeed", "title": "My Show", "url": "https://pod.example/feed.xml"},
        "episode": {
            "episode_id": episode_id,
            "title": f"Episode {episode_id}",
            "published_date": published,
            "duration_seconds": 1000,
        },
        "summary": {"title": "Sum", "bullets": ["a"]},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hello", encoding="utf-8")
    nodes = [{"id": tid, "type": "Topic", "properties": {"label": label}} for tid, label in topics]
    nodes += [
        {"id": pid, "type": "Person", "properties": {"name": name}} for pid, name in (persons or [])
    ]
    (root / "metadata" / f"{stem}.kg.json").write_text(
        json.dumps({"episode_id": episode_id, "nodes": nodes}), encoding="utf-8"
    )
    if with_gi:
        gi = {"episode_id": episode_id, "nodes": [], "edges": []}
        (root / "metadata" / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")


def _corpus(root: Path) -> None:
    # epOld is older but about AI (+GI, +person Jane); epNew is newer but about Health.
    _write_episode(
        root,
        stem="0001-old",
        episode_id="old",
        topics=[("topic:ai", "AI")],
        published="2024-01-01T00:00:00",
        with_gi=True,
        persons=[("person:jane", "Jane")],
    )
    _write_episode(
        root,
        stem="0002-new",
        episode_id="new",
        topics=[("topic:health", "Health")],
        published="2024-06-01T00:00:00",
    )
    (root / "search").mkdir(parents=True, exist_ok=True)
    payload = {
        "clusters": [
            {
                "graph_compound_parent_id": "tc:ai",
                "canonical_label": "AI",
                "member_count": 3,
                "members": [{"topic_id": "topic:ai", "label": "AI"}],
            },
            {
                "graph_compound_parent_id": "tc:health",
                "canonical_label": "Health",
                "member_count": 1,
                "members": [{"topic_id": "topic:health", "label": "Health"}],
            },
        ]
    }
    (root / "search" / "topic_clusters.json").write_text(json.dumps(payload), encoding="utf-8")


def _rows_newest_first(root: Path) -> list[CatalogEpisodeRow]:
    rows = build_catalog_rows_cumulative(root)
    titles = [r.episode_title for r in rows]
    assert titles == ["Episode new", "Episode old"]  # catalog is recency (newest-first)
    return rows


def test_empty_interests_is_recency_passthrough(tmp_path: Path) -> None:
    _corpus(tmp_path)
    rows = _rows_newest_first(tmp_path)
    out = rank_discover(tmp_path, [], rows, limit=10)
    assert [s.title for s in out] == ["Episode new", "Episode old"]


def test_only_empty_string_tokens_collapse_to_recency(tmp_path: Path) -> None:
    # ``rank_discover`` keeps a token only when ``str(i)`` is truthy; an all-empty-string
    # interest list yields an empty set → the recency passthrough (no scoring at all).
    _corpus(tmp_path)
    rows = _rows_newest_first(tmp_path)
    out = rank_discover(tmp_path, ["", ""], rows, limit=10)
    assert [s.title for s in out] == ["Episode new", "Episode old"]


def test_cluster_interest_reranks_matching_episode_first(tmp_path: Path) -> None:
    _corpus(tmp_path)
    rows = _rows_newest_first(tmp_path)
    out = rank_discover(tmp_path, ["tc:ai"], rows, limit=10)
    # epOld matches the followed cluster AND has GI → leads despite being older.
    assert [s.title for s in out] == ["Episode old", "Episode new"]


def test_topic_interest_reranks(tmp_path: Path) -> None:
    _corpus(tmp_path)
    rows = _rows_newest_first(tmp_path)
    out = rank_discover(tmp_path, ["topic:ai"], rows, limit=10)
    assert [s.title for s in out] == ["Episode old", "Episode new"]


def test_person_interest_reranks(tmp_path: Path) -> None:
    _corpus(tmp_path)
    rows = _rows_newest_first(tmp_path)
    out = rank_discover(tmp_path, ["person:jane"], rows, limit=10)
    assert [s.title for s in out] == ["Episode old", "Episode new"]


def _write_theme_clusters(root: Path, thc_id: str, label: str, member_topic_ids: list[str]) -> None:
    """Write ``enrichments/topic_theme_clusters.json`` with one theme cluster (envelope-wrapped)."""
    (root / "enrichments").mkdir(parents=True, exist_ok=True)
    payload = {
        "data": {
            "clusters": [
                {
                    "cluster_type": "theme",
                    "canonical_label": label,
                    "graph_compound_parent_id": thc_id,
                    "member_count": len(member_topic_ids),
                    "members": [{"topic_id": t, "label": t} for t in member_topic_ids],
                }
            ]
        }
    }
    (root / "enrichments" / "topic_theme_clusters.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_theme_cluster_interest_reranks_matching_episode_first(tmp_path: Path) -> None:
    # A followed storyline (`thc:`) must re-rank exactly like a semantic cluster: the episode whose
    # topic is in the theme cluster leads. epOld (topic:ai, +GI) is in the storyline → it wins.
    _corpus(tmp_path)
    _write_theme_clusters(tmp_path, "thc:ai-safety", "AI safety", ["topic:ai"])
    rows = _rows_newest_first(tmp_path)
    out = rank_discover(tmp_path, ["thc:ai-safety"], rows, limit=10)
    assert [s.title for s in out] == ["Episode old", "Episode new"]


def test_theme_cluster_token_without_artifact_grants_no_affinity(tmp_path: Path) -> None:
    # With no theme-cluster artifact, a `thc:` token matches nothing (like any unknown prefix) —
    # zero affinity to both equal-depth episodes → recency order is preserved.
    _write_episode(
        tmp_path,
        stem="0001-old",
        episode_id="old",
        topics=[("topic:ai", "AI")],
        published="2024-01-01T00:00:00",
    )
    _write_episode(
        tmp_path,
        stem="0002-new",
        episode_id="new",
        topics=[("topic:ai", "AI")],
        published="2024-06-01T00:00:00",
    )
    rows = build_catalog_rows_cumulative(tmp_path)
    out = rank_discover(tmp_path, ["thc:nonexistent"], rows, limit=10)
    assert [s.title for s in out] == ["Episode new", "Episode old"]


def _write_velocity_envelope(root: Path, topic_velocities: dict[str, float]) -> None:
    (root / "enrichments").mkdir(parents=True, exist_ok=True)
    topics = [{"topic_id": t, "velocity_last_over_6mo": v} for t, v in topic_velocities.items()]
    (root / "enrichments" / "temporal_velocity.json").write_text(
        json.dumps({"data": {"topics": topics}}), encoding="utf-8"
    )


def test_trend_velocity_signal_boosts_hot_topic_episode(tmp_path: Path) -> None:
    from podcast_scraper.server.app_ranking_config import ranking_config_from_dict

    _corpus(tmp_path)
    # topic:health is hot (3× its 6-mo average), topic:ai is flat.
    _write_velocity_envelope(tmp_path, {"topic:health": 3.0, "topic:ai": 1.0})
    rows = _rows_newest_first(tmp_path)
    interests = ["topic:ai", "topic:health"]  # both episodes match affinity equally

    # Trend OFF (default): the deeper (+GI) older AI episode still leads.
    default_out = rank_discover(tmp_path, interests, rows, limit=10)
    assert [s.title for s in default_out] == ["Episode old", "Episode new"]

    # Trend ON with a strong weight: the hot-topic episode flips to the top.
    cfg = ranking_config_from_dict(
        {
            "signals": [
                {"name": "trend_velocity", "enabled": True, "weight": 5.0, "params": {"cap": 1.5}}
            ]
        }
    )
    trend_out = rank_discover(tmp_path, interests, rows, limit=10, config=cfg)
    assert [s.title for s in trend_out] == ["Episode new", "Episode old"]


def test_trend_velocity_disabled_ignores_envelope(tmp_path: Path) -> None:
    # Even with a very hot envelope present, the default (trend OFF) config must not apply it.
    _corpus(tmp_path)
    _write_velocity_envelope(tmp_path, {"topic:health": 9.0})
    rows = _rows_newest_first(tmp_path)
    out = rank_discover(tmp_path, ["topic:ai", "topic:health"], rows, limit=10)
    assert [s.title for s in out] == ["Episode old", "Episode new"]


def test_unknown_prefix_token_grants_no_affinity(tmp_path: Path) -> None:
    # An unknown-prefix token (lands in cluster_interests, matches nothing) gives zero
    # affinity to BOTH episodes — so the order is pure significance, not interest-driven.
    # epOld carries GI (+2) and so leads on depth alone; flip it via two equal-depth rows
    # to prove the unknown token added no per-episode boost (order would be recency then).
    _write_episode(
        tmp_path,
        stem="0001-old",
        episode_id="old",
        topics=[("topic:ai", "AI")],
        published="2024-01-01T00:00:00",
    )
    _write_episode(
        tmp_path,
        stem="0002-new",
        episode_id="new",
        topics=[("topic:ai", "AI")],
        published="2024-06-01T00:00:00",
    )
    rows = build_catalog_rows_cumulative(tmp_path)
    assert [r.episode_title for r in rows] == ["Episode new", "Episode old"]
    out = rank_discover(tmp_path, ["genre:jazz"], rows, limit=10)
    # Equal significance + zero affinity for both → recency tie-break preserved.
    assert [s.title for s in out] == ["Episode new", "Episode old"]


def test_recency_tie_break_keeps_equal_score_newest_first(tmp_path: Path) -> None:
    # Two episodes with identical depth (no GI, same bullets, both with KG) and an
    # interest that neither matches → equal score; the -idx tie-break must preserve
    # the incoming newest-first order.
    _write_episode(
        tmp_path,
        stem="0001-old",
        episode_id="old",
        topics=[("topic:ai", "AI")],
        published="2024-01-01T00:00:00",
    )
    _write_episode(
        tmp_path,
        stem="0002-new",
        episode_id="new",
        topics=[("topic:ai", "AI")],
        published="2024-06-01T00:00:00",
    )
    rows = build_catalog_rows_cumulative(tmp_path)
    assert [r.episode_title for r in rows] == ["Episode new", "Episode old"]
    # Interest no episode matches → both score == _significance (equal) → tie.
    out = rank_discover(tmp_path, ["person:nobody"], rows, limit=10)
    assert [s.title for s in out] == ["Episode new", "Episode old"]


def test_limit_truncates_after_ranking(tmp_path: Path) -> None:
    _corpus(tmp_path)
    rows = _rows_newest_first(tmp_path)
    out = rank_discover(tmp_path, ["tc:ai"], rows, limit=1)
    assert [s.title for s in out] == ["Episode old"]  # top-ranked survives the cap


def test_default_affinity_weight_is_two() -> None:
    # Guards the documented "fully on-interest episode → (1 + affinity_weight)x" contract, now
    # sourced from the tunable ranking-signal registry rather than a module constant.
    from podcast_scraper.server.app_ranking_config import (
        DEFAULT_RANKING_CONFIG,
        SIGNAL_INTEREST_AFFINITY,
    )

    assert DEFAULT_RANKING_CONFIG.weight_of(SIGNAL_INTEREST_AFFINITY) == 2.0


# --- recency as a graded signal, not just the tie-break (#22) ------------------------------------
#
# Any non-empty interest set used to sort the whole pool by score, with recency surviving only as
# the `-idx` tie-break and SIGNAL_RECENCY carrying weight 0.0. So following ONE topic reshuffled
# even the episodes that had nothing to do with it — newest-first became enrichment-depth-first.


class TestRecencyBoost:
    """The decay curve itself."""

    def test_the_newest_episode_gets_the_full_boost(self) -> None:
        assert _recency_boost("2026-07-16", date(2026, 7, 16), 365.0) == 1.0

    def test_one_half_life_halves_it(self) -> None:
        assert _recency_boost("2025-07-16", date(2026, 7, 16), 365.0) == pytest.approx(
            0.5, abs=0.01
        )

    def test_two_half_lives_quarter_it(self) -> None:
        assert _recency_boost("2024-07-16", date(2026, 7, 16), 365.0) == pytest.approx(
            0.25, abs=0.01
        )

    def test_it_decays_from_the_pool_not_the_wall_clock(self) -> None:
        """A corpus that stopped updating must still rank its own shelf.

        Against wall-clock `now`, every episode in an archive decays to ~0 together and the signal
        silently stops discriminating — exactly when "what is newest here" matters most. Relative
        decay keeps the newest thing available at 1.0 however old the archive is.
        """
        assert _recency_boost("1999-01-01", date(1999, 1, 1), 365.0) == 1.0

    @pytest.mark.parametrize(
        ("published", "newest", "half_life"),
        [
            (None, date(2026, 7, 16), 365.0),  # no date
            ("not-a-date", date(2026, 7, 16), 365.0),  # unparseable
            ("2026-07-16", None, 365.0),  # empty pool
            ("2026-07-16", date(2026, 7, 16), 0.0),  # half-life disabled
            ("2026-07-16", date(2026, 7, 16), -5.0),  # nonsense half-life
        ],
    )
    def test_degenerate_inputs_contribute_nothing(self, published, newest, half_life) -> None:
        """A missing or broken date must not become a BOOST — it must be inert."""
        assert _recency_boost(published, newest, half_life) == 0.0

    def test_an_episode_newer_than_the_newest_is_clamped(self) -> None:
        """Negative age would give a boost above 1.0 and let a stray future date dominate."""
        assert _recency_boost("2030-01-01", date(2026, 7, 16), 365.0) == 1.0


def _dated_row(publish_date: str | None) -> CatalogEpisodeRow:
    """A row carrying only the field ``_newest_publish_date`` reads."""
    return CatalogEpisodeRow(
        metadata_relative_path="metadata/x.metadata.json",
        feed_id="f",
        feed_title=None,
        episode_id="e",
        episode_title="E",
        publish_date=publish_date,
        summary_title=None,
        summary_bullets=(),
        summary_text=None,
        gi_relative_path=None,
        kg_relative_path=None,
        bridge_relative_path=None,
        has_gi=False,
        has_kg=False,
        has_bridge=False,
    )


class TestNewestPublishDate:
    def test_picks_the_maximum(self) -> None:
        rows = [_dated_row(d) for d in ("2024-01-01", "2026-07-16", "2025-05-05")]
        assert _newest_publish_date(rows) == date(2026, 7, 16)

    def test_ignores_unparseable_and_missing(self) -> None:
        rows = [_dated_row(d) for d in (None, "garbage", "2025-05-05")]
        assert _newest_publish_date(rows) == date(2025, 5, 5)

    def test_none_when_nothing_is_dated(self) -> None:
        assert _newest_publish_date([_dated_row(None)]) is None


class TestTheDefaultConfigShipsRecencyOn:
    """The whole point of #22 was that the slot existed and was switched off.

    Pinned so it cannot quietly go back to weight 0.0 — that would restore the original bug with
    no test failing, since every other assertion here passes a config explicitly.
    """

    def test_recency_is_enabled_with_a_real_weight(self) -> None:
        signal = DEFAULT_RANKING_CONFIG.get(SIGNAL_RECENCY)
        assert signal is not None
        assert signal.enabled is True
        assert signal.weight > 0.0, "recency is back to being a tie-break only"

    def test_the_half_life_is_measured_not_intuitive(self) -> None:
        """30 days is DEAD on a corpus this sparse (2nd-newest boost 0.014); the value must be
        chosen against the eval, not against intuition about when an episode feels stale."""
        half_life = DEFAULT_RANKING_CONFIG.params_of(SIGNAL_RECENCY).get("half_life_days")
        assert half_life is not None and float(half_life) >= 90.0, (
            "a short half-life silently does nothing on a sparsely-published corpus — re-measure "
            "against scripts/eval/score/rank_discover_v1.py before lowering this"
        )

    def test_affinity_still_outranks_freshness(self) -> None:
        """Following something must mean more than "this is new", or personalisation is
        cosmetic — a fresh episode you have no interest in should not beat one you follow."""
        affinity = DEFAULT_RANKING_CONFIG.weight_of(SIGNAL_INTEREST_AFFINITY)
        assert affinity > DEFAULT_RANKING_CONFIG.weight_of(SIGNAL_RECENCY)
