"""Unit tests for the user's heard∪captured episode set (P3 #1120, RFC-101 §1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server.app_user_corpus import derive_episode_set


def _recording(seen: list, per_episode: dict):
    """Stand in for ``_episode_entities``, recording which rows it was asked about.

    Was ``lambda root, row: seen.append(row) or per_episode[row]`` — that uses ``append``'s
    return value (always None) to sequence two statements inside an expression. It works by
    accident of ``or``, and it hides the recording behind a trick.
    """

    def _episode_entities(root, row):
        seen.append(row)
        return per_episode[row]

    return _episode_entities


def test_heard_requires_threshold_of_known_duration() -> None:
    playback = [
        {"slug": "ep-30pct", "position_seconds": 300},  # 300/1000 = 30% → heard
        {"slug": "ep-29pct", "position_seconds": 290},  # below 30% → not heard
        {"slug": "ep-nodur", "position_seconds": 999},  # unknown duration → not heard alone
    ]
    durations = {"ep-30pct": 1000.0, "ep-29pct": 1000.0}  # ep-nodur absent
    got = derive_episode_set(playback, [], durations)
    assert got == {"ep-30pct"}


def test_captured_always_qualifies_even_without_playback() -> None:
    got = derive_episode_set([], ["ep-hl", "ep-fav", ""], {})
    assert got == {"ep-hl", "ep-fav"}  # blanks dropped


def test_union_of_heard_and_captured() -> None:
    playback = [{"slug": "ep-heard", "position_seconds": 600}]
    durations = {"ep-heard": 1000.0}
    got = derive_episode_set(playback, ["ep-cap"], durations)
    assert got == {"ep-heard", "ep-cap"}


def test_custom_threshold() -> None:
    playback = [{"slug": "ep", "position_seconds": 100}]  # 10%
    durations = {"ep": 1000.0}
    assert derive_episode_set(playback, [], durations, threshold=0.05) == {"ep"}
    assert derive_episode_set(playback, [], durations, threshold=0.5) == set()


def test_malformed_playback_rows_are_skipped() -> None:
    playback: list[dict] = [
        {"position_seconds": 500},
        {"slug": "", "position_seconds": 5},
        {"slug": "ok"},
    ]
    durations = {"ok": 10.0}
    # 'ok' has no position_seconds → defaults 0 → not heard; no crash on missing slug/position
    assert derive_episode_set(playback, [], durations) == set()


def test_user_episode_set_heard_via_playback(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # The heard-via-listening path end-to-end: ≥30% played of a known-duration episode qualifies,
    # below-threshold does not — exercising user_episode_set + slug_durations over a real corpus.
    import json
    from pathlib import Path

    from podcast_scraper.server import app_user_state
    from podcast_scraper.server.app_slugs import slug_for_row
    from podcast_scraper.server.app_user_corpus import user_episode_set
    from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

    root = Path(tmp_path) / "corpus"
    (root / "metadata").mkdir(parents=True)
    for eid, dur in [("ep-heard", 1000), ("ep-skim", 1000)]:
        (root / "metadata" / f"{eid}.metadata.json").write_text(
            json.dumps(
                {
                    "feed": {"feed_id": "f", "title": "S", "url": "https://f.ex/f.xml"},
                    "episode": {
                        "episode_id": eid,
                        "title": eid,
                        "published_date": "2024-01-01T00:00:00",
                        "duration_seconds": dur,
                    },
                    "content": {},
                }
            ),
            encoding="utf-8",
        )
    slugs = {r.episode_id: slug_for_row(r) for r in build_catalog_rows_cumulative(root)}
    data_dir = Path(tmp_path) / "appdata"
    app_user_state.set_playback(data_dir, "u1", slugs["ep-heard"], 400.0, 1)  # 40% → heard
    app_user_state.set_playback(data_dir, "u1", slugs["ep-skim"], 100.0, 1)  # 10% → not heard
    assert user_episode_set(root, data_dir, "u1") == {slugs["ep-heard"]}


def test_derive_interests_aggregates_and_ranks_top_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # #1139: interest tokens are the topics/people of the user's episode set, ranked by
    # frequency (id asc as a stable tiebreak). Stubs isolate the aggregation from KG IO.
    import podcast_scraper.server.app_user_corpus as uc

    monkeypatch.setattr(uc, "user_episode_set", lambda *a, **k: {"s1", "s2", "s3"})
    monkeypatch.setattr(uc, "cached_catalog", lambda root: ["r_s1", "r_s2", "r_s3"])
    monkeypatch.setattr(uc, "slug_for_row", lambda r: r[2:])  # "r_s1" -> "s1"
    per_episode = {
        "r_s1": [("topic", "topic:ai", "AI"), ("person", "person:jane", "Jane")],
        "r_s2": [("topic", "topic:ai", "AI"), ("topic", "topic:vc", "VC")],
        "r_s3": [],  # no KG → contributes nothing
    }
    # (kind, id, label) now — one core produces the token, so the test speaks its input shape.
    monkeypatch.setattr(uc, "_episode_entities", lambda root, row: per_episode[row])

    got = uc.derive_interests(tmp_path, tmp_path, "u1", k=3)
    assert got[0] == "topic:ai"  # frequency 2 → leads
    assert set(got) == {"topic:ai", "person:jane", "topic:vc"}


def test_derive_interests_respects_k_and_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import podcast_scraper.server.app_user_corpus as uc

    # No episodes → no derived interests.
    monkeypatch.setattr(uc, "user_episode_set", lambda *a, **k: set())
    assert uc.derive_interests(tmp_path, tmp_path, "u1") == []

    # k caps the list; the most frequent survive.
    monkeypatch.setattr(uc, "user_episode_set", lambda *a, **k: {"s1"})
    monkeypatch.setattr(uc, "cached_catalog", lambda root: ["r_s1"])
    monkeypatch.setattr(uc, "slug_for_row", lambda r: r[2:])
    monkeypatch.setattr(
        uc,
        "_episode_entities",
        lambda root, row: [
            ("topic", "topic:a", "A"),
            ("topic", "topic:b", "B"),
            ("topic", "topic:c", "C"),
        ],
    )
    assert uc.derive_interests(tmp_path, tmp_path, "u1", k=2) == ["topic:a", "topic:b"]


# --- RFC-114 faceting: episode-favorites move to `saved`, out of `experienced` (the correction) ---


def test_experienced_excludes_whole_episode_favorites(tmp_path: Path) -> None:
    # A favorited-but-never-played episode must NOT be in `experienced` (recall) — RFC-114 §1.1.
    from podcast_scraper.server import app_user_corpus, app_user_state

    uid = "u_0123456789abcdef01234567"
    app_user_state.add_favorite(tmp_path, uid, {"kind": "episode", "ref": "ep-fav"})
    # no playback, no highlight for ep-fav
    experienced = app_user_corpus.experienced_episode_set(tmp_path, tmp_path, uid)
    saved = app_user_corpus.saved_episode_set(tmp_path, uid)
    assert "ep-fav" not in experienced  # the correction: not in recall
    assert saved == {"ep-fav"}  # it IS in the saved facet


def test_saved_insight_favorite_counts_as_experienced(tmp_path: Path) -> None:
    # A saved *insight* carries its episode slug and IS engagement → experienced.
    from podcast_scraper.server import app_user_corpus, app_user_state

    uid = "u_0123456789abcdef01234567"
    app_user_state.add_favorite(
        tmp_path, uid, {"kind": "insight", "ref": "ins-1", "slug": "ep-ins"}
    )
    experienced = app_user_corpus.experienced_episode_set(tmp_path, tmp_path, uid)
    assert "ep-ins" in experienced
    assert (
        app_user_corpus.saved_episode_set(tmp_path, uid) == set()
    )  # insight fav is not an episode


def test_highlight_and_note_are_experienced(tmp_path: Path) -> None:
    from podcast_scraper.server import app_user_corpus, app_user_state

    uid = "u_0123456789abcdef01234567"
    app_user_state.add_highlight(
        tmp_path, uid, {"id": "h1", "episode_slug": "ep-hl", "kind": "span", "created_at": 1}
    )
    experienced = app_user_corpus.experienced_episode_set(tmp_path, tmp_path, uid)
    assert "ep-hl" in experienced


# --- ONE definition of "what this user is into" (#28) -------------------------------------------
#
# There were three. All counted person/topic occurrences across the user's heard∪captured episodes,
# and each chose ITS OWN episodes:
#
#     derive_interests        recency-ranked, 40   (only after #18 fixed it)
#     /corpus  _top_entities  sorted(slugs)[:40]   the alphabetical freeze, still live there
#     /interests/derived      every episode        no bound at all
#
# So the same user was told they were into three different things depending on which screen they
# opened, #18's fix reached one of the three, and /interests/derived did an unbounded number of KG
# loads for a heavy listener. They had already drifted once on token FORMAT — the doubled
# `topic:topic:` prefix (d390f7b0).
#
# These tests exist to keep it at one. The token-format cases moved here from
# test_app_resurfacing.py when the duplicate was deleted.


class TestInterestToken:
    """The token must be exactly the id the ranker compares against — no more, no less."""

    def test_real_kg_ids_are_not_double_prefixed(self) -> None:
        from podcast_scraper.server.app_user_corpus import interest_token

        assert interest_token("topic", "topic:systems-thinking") == "topic:systems-thinking"
        assert interest_token("person", "person:sam") == "person:sam"

    def test_unprefixed_ids_still_get_their_kind(self) -> None:
        """Back-compat: a hand-written id without the prefix is still namespaced."""
        from podcast_scraper.server.app_user_corpus import interest_token

        assert interest_token("topic", "ai") == "topic:ai"
        assert interest_token("person", "jane") == "person:jane"


class TestOneDefinitionForEverySurface:
    """/discover, /corpus and /interests/derived must answer from the same counts.

    Not "the same numbers today" — the same FUNCTION. A test that re-implements the ranking would
    drift alongside the code, which is the failure being fixed. So: assert the projections agree
    with the core they project from.
    """

    @staticmethod
    def _stub(monkeypatch, per_episode: dict) -> None:
        import podcast_scraper.server.app_user_corpus as uc

        monkeypatch.setattr(uc, "user_episode_set", lambda *a, **k: set(per_episode))
        monkeypatch.setattr(uc, "cached_catalog", lambda root: list(per_episode))
        monkeypatch.setattr(uc, "slug_for_row", lambda r: r)
        # (slug, engagement_ts) since #24. Timestamp 0 everywhere means no decay applies, so these
        # projection assertions keep testing count order and nothing else.
        monkeypatch.setattr(
            uc, "_most_recently_engaged", lambda d, u, s, n: [(x, 0) for x in sorted(s)[:n]]
        )
        monkeypatch.setattr(uc, "_episode_entities", lambda root, row: per_episode[row])

    def test_derive_interests_is_exactly_the_top_k_tokens_of_the_counts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import podcast_scraper.server.app_user_corpus as uc

        # Count order must DISAGREE with alphabetical order, or this cannot tell a real projection
        # from a re-sort. `topic:zzz` is the most-heard and alphabetically last; `topic:aaa` is the
        # least-heard and alphabetically first. (The first version of this test used data where the
        # two orders coincided, so it passed against a deliberately broken projection.)
        self._stub(
            monkeypatch,
            {
                "s1": [("topic", "topic:zzz", "Z"), ("topic", "topic:aaa", "A")],
                "s2": [("topic", "topic:zzz", "Z"), ("topic", "topic:mmm", "M")],
                "s3": [("topic", "topic:zzz", "Z"), ("topic", "topic:mmm", "M")],
            },
        )
        counts = uc.derived_interest_counts(tmp_path, tmp_path, "u1")
        assert [r["token"] for r in counts] == [
            "topic:zzz",
            "topic:mmm",
            "topic:aaa",
        ], "the core is not ranking by count — the rest of this test proves nothing"
        for k in (1, 2, 3, 8):
            assert uc.derive_interests(tmp_path, tmp_path, "u1", k=k) == [
                row["token"] for row in counts[:k]
            ], f"the /discover projection diverged from the core at k={k}"

    def test_every_surface_sees_the_same_bound(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """One number, not three: /corpus had its own _MAX_ENTITY_SCAN, /interests/derived none."""
        import podcast_scraper.server.app_user_corpus as uc

        assert uc.DERIVED_MAX_EPISODES == 40
        assert uc.DERIVED_TOP_K == 8

        seen: list[str] = []
        per_episode = {f"s{i:03d}": [("topic", f"topic:t{i}", f"T{i}")] for i in range(60)}
        self._stub(monkeypatch, per_episode)
        monkeypatch.setattr(
            uc,
            "_episode_entities",
            _recording(seen, per_episode),
        )
        uc.derived_interest_counts(tmp_path, tmp_path, "u1")
        assert len(seen) == uc.DERIVED_MAX_EPISODES, (
            "the core loaded a different number of KGs than the shared bound — the unbounded "
            "/interests/derived scan is exactly what this prevents"
        )

    def test_counts_carry_what_the_dict_surfaces_need(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """/corpus and /interests/derived render {token, kind, label, count, weight}; the core must
        supply all five, or those two surfaces would need their own derivation again.

        Exact-shape rather than subset, because both routes do ``DerivedInterest(**row)`` — an
        unexpected key is a 500 on a page load, and a missing one is a silently defaulted field.
        """
        import podcast_scraper.server.app_user_corpus as uc

        self._stub(monkeypatch, {"s1": [("topic", "topic:long-form", "long form")]})
        row = uc.derived_interest_counts(tmp_path, tmp_path, "u1")[0]
        assert row == {
            "token": "topic:long-form",
            "kind": "topic",
            "label": "long form",
            "count": 1,
            # The stub engages every episode at ts 0, so nothing decays: one occurrence, weight 1.0.
            "weight": 1.0,
        }

    def test_ranking_is_deterministic_across_reads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Ties must not reorder between reads, or two surfaces rendered moments apart disagree."""
        import podcast_scraper.server.app_user_corpus as uc

        self._stub(
            monkeypatch,
            {
                "s1": [("topic", "topic:b", "B"), ("topic", "topic:a", "A")],
                "s2": [("topic", "topic:c", "C")],
            },
        )
        first = uc.derived_interest_counts(tmp_path, tmp_path, "u1")
        assert all(uc.derived_interest_counts(tmp_path, tmp_path, "u1") == first for _ in range(3))
        assert [r["token"] for r in first] == ["topic:a", "topic:b", "topic:c"]


# --- the profile must be able to FORGET (#24) -----------------------------------------------------
#
# derive_interests was a pure accumulator: every heard episode added 1 to its tokens, for ever,
# within the 40-episode window. So the only closed loop in the product — heard -> derived interests
# -> ranked higher -> heard — had no term that could ever shrink. A user whose taste had moved on
# kept being recommended the taste they left, and the more they had listened before the move, the
# longer it took to escape.


class TestDerivedInterestsDecay:
    """Time-decay over the engagement that produced each token."""

    @staticmethod
    def _stub_shifted(monkeypatch, tmp_path) -> None:
        """12 old episodes on taste A, 4 recent on taste B — the case decay exists for.

        Deliberately the WORSE case for taste B: it is outnumbered 3:1, so raw counting cannot
        surface it however recent it is.
        """
        import podcast_scraper.server.app_user_corpus as uc

        day = 86400
        now = 1_760_000_000
        episodes: dict[str, list[tuple[str, str, str]]] = {}
        engaged: list[tuple[str, int]] = []
        for i in range(12):
            slug = f"old{i:02d}"
            episodes[slug] = [("topic", "topic:a-old", "Old taste")]
            engaged.append((slug, now - (180 + i * 5) * day))
        for i in range(4):
            slug = f"new{i:02d}"
            episodes[slug] = [("topic", "topic:b-new", "New taste")]
            engaged.append((slug, now - i * 2 * day))

        monkeypatch.setattr(uc, "user_episode_set", lambda *a, **k: set(episodes))
        monkeypatch.setattr(uc, "cached_catalog", lambda root: list(episodes))
        monkeypatch.setattr(uc, "slug_for_row", lambda r: r)
        monkeypatch.setattr(uc, "_most_recently_engaged", lambda d, u, s, n: engaged[:n])
        monkeypatch.setattr(uc, "_episode_entities", lambda root, row: episodes[row])

    def test_a_recent_taste_outranks_a_larger_but_stale_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The headline. Four episodes from last week beat twelve from six months ago."""
        import podcast_scraper.server.app_user_corpus as uc

        self._stub_shifted(monkeypatch, tmp_path)
        rows = uc.derived_interest_counts(tmp_path, tmp_path, "u1")
        by_token = {r["token"]: r for r in rows}

        assert rows[0]["token"] == "topic:b-new", (
            "the taste the user has actually moved TO must lead the profile; ranked by raw count "
            f"it never can (12 vs 4). Got: {[(r['token'], r['count'], r['weight']) for r in rows]}"
        )
        # count stays the honest episode tally — decay changes the ORDER, not the reported number.
        assert by_token["topic:a-old"]["count"] == 12
        assert by_token["topic:b-new"]["count"] == 4
        assert by_token["topic:a-old"]["weight"] < by_token["topic:b-new"]["weight"]

    def test_the_stale_taste_is_demoted_not_deleted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Decay, not amnesia. Six months of listening still means something."""
        import podcast_scraper.server.app_user_corpus as uc

        self._stub_shifted(monkeypatch, tmp_path)
        stale = next(
            r
            for r in uc.derived_interest_counts(tmp_path, tmp_path, "u1")
            if r["token"] == "topic:a-old"
        )
        assert stale["weight"] > 0.0, "an old interest must fade, not vanish"

    def test_weight_never_exceeds_the_episode_count(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The contract the API field states: weight <= count, always."""
        import podcast_scraper.server.app_user_corpus as uc

        self._stub_shifted(monkeypatch, tmp_path)
        for row in uc.derived_interest_counts(tmp_path, tmp_path, "u1"):
            assert 0.0 < row["weight"] <= row["count"], row


class TestDecayDegradesSafely:
    """``_decayed`` runs on timestamps read from user files, which may simply not be there."""

    def test_no_timestamps_at_all_means_no_decay(self) -> None:
        """Not "everything is infinitely old". A corpus without engagement metadata must fall back
        to plain count ranking, which is what the code did before #24 — never to an empty profile.
        """
        import podcast_scraper.server.app_user_corpus as uc

        assert uc._decayed([("a", 0), ("b", 0)]) == [("a", 1.0), ("b", 1.0)]

    def test_an_untimed_episode_inherits_the_oldest_known_weight(self) -> None:
        """It sorts last but stays eligible — the promise `_most_recently_engaged` already makes.

        Read as epoch-0 it would be ~55 years old and weigh 0.0, silently deleting those episodes
        from the profile rather than ranking them last.
        """
        import podcast_scraper.server.app_user_corpus as uc

        day = 86400
        now = 1_760_000_000
        out = dict(uc._decayed([("new", now), ("old", now - 90 * day), ("untimed", 0)]))
        assert out["new"] == pytest.approx(1.0)
        assert out["old"] == pytest.approx(0.5)  # exactly one half-life
        assert out["untimed"] == pytest.approx(out["old"])
        assert out["untimed"] > 0.0

    def test_a_zero_half_life_disables_decay_rather_than_dividing_by_zero(self) -> None:
        import podcast_scraper.server.app_user_corpus as uc

        assert uc._decayed([("a", 100), ("b", 1)], half_life_days=0) == [("a", 1.0), ("b", 1.0)]

    def test_the_newest_engagement_always_weighs_one(self) -> None:
        """Ages run from the user's OWN newest engagement, not wall-clock — so someone returning
        after a year away finds the profile they left, not a uniformly flattened one."""
        import podcast_scraper.server.app_user_corpus as uc

        long_ago = 1_000_000_000
        day = 86400
        out = dict(uc._decayed([("x", long_ago), ("y", long_ago - 90 * day)]))
        assert out["x"] == pytest.approx(1.0)
        assert out["y"] == pytest.approx(0.5)


# --- the recency ordering itself, unstubbed (#18, found unguarded by the #70 sweep) ---------------
#
# Every other test in this file STUBS `_most_recently_engaged`, because they are testing what the
# callers do with its result. That left the function's own behaviour — the thing #18 actually fixed
# — covered by nothing: reverting it to `sorted(slugs)[:limit]` broke no test at all.
#
# The bug it fixes is subtle and permanent. Slugs are `{feed-slug}-{hash}`, so an alphabetical sort
# groups by SHOW; past the 40-episode bound the profile froze on whichever shows happened to be
# spelled first, and new listening stopped moving it. Exactly backwards for a signal whose whole
# job is to track what someone is into lately.


def test_most_recently_engaged_orders_by_recency_not_by_slug(tmp_path: Path) -> None:
    from podcast_scraper.server import app_user_state
    from podcast_scraper.server.app_user_corpus import _most_recently_engaged

    uid = "u_0123456789abcdef01234567"
    # Alphabetical order is the REVERSE of engagement order, so the two cannot be confused.
    # 'aaa' is alphabetically first and heard longest ago; 'zzz' is last and heard most recently.
    for slug, ts in (("aaa-ep", 1_000), ("mmm-ep", 2_000), ("zzz-ep", 3_000)):
        app_user_state.set_playback(tmp_path, uid, slug, 10_000.0, ts)

    got = _most_recently_engaged(tmp_path, uid, {"aaa-ep", "mmm-ep", "zzz-ep"}, 3)
    assert [slug for slug, _ts in got] == ["zzz-ep", "mmm-ep", "aaa-ep"], got
    assert [ts for _slug, ts in got] == [3_000, 2_000, 1_000], got


def test_the_bound_keeps_the_most_RECENT_episodes(tmp_path: Path) -> None:
    """The half that mattered in production: which episodes survive the 40-episode cap.

    Under the alphabetical sort the cap kept whichever shows sorted first — so a heavy listener's
    profile was decided by feed-id spelling and then never changed again.
    """
    from podcast_scraper.server import app_user_state
    from podcast_scraper.server.app_user_corpus import _most_recently_engaged

    uid = "u_0123456789abcdef01234567"
    slugs = {f"{chr(ord('a') + i)}-ep": 1_000 + i for i in range(10)}  # 'a' oldest ... 'j' newest
    for slug, ts in slugs.items():
        app_user_state.set_playback(tmp_path, uid, slug, 10_000.0, ts)

    kept = [slug for slug, _ts in _most_recently_engaged(tmp_path, uid, set(slugs), 3)]
    assert kept == ["j-ep", "i-ep", "h-ep"], kept


def test_an_episode_with_no_timestamp_sorts_last_but_stays_eligible(tmp_path: Path) -> None:
    """A corpus without engagement metadata degrades to "some bounded subset", never to none."""
    from podcast_scraper.server import app_user_state
    from podcast_scraper.server.app_user_corpus import _most_recently_engaged

    uid = "u_0123456789abcdef01234567"
    app_user_state.set_playback(tmp_path, uid, "timed-ep", 10_000.0, 5_000)
    got = _most_recently_engaged(tmp_path, uid, {"timed-ep", "untimed-ep"}, 5)
    assert [slug for slug, _ts in got] == ["timed-ep", "untimed-ep"], got
    assert dict(got)["untimed-ep"] == 0
