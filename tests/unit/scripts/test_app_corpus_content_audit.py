"""The build refuses to succeed on the defect shapes v3 actually shipped with.

v3 committed two defects that a file-existence check cannot see, and both survived a long time:

  * every one of the 36 episode summaries was the transcript's opening greeting;
  * every duration was the same hardcoded 1800s.

The first was fixed in ``metadata.json`` and left untouched in the GI Insight layer, where it stayed
at 36/36. The second was fixed in ``metadata.json`` and left untouched in BOTH the GI Episode node
and the insight-density sidecar. In each case the build printed a success line, because nothing
asserted anything about content — only about files.

These tests pin the audit that now runs at the end of every build. They are written against the
AUDIT rather than against the committed fixture, so they state the rule instead of re-deriving it
from whatever the corpus happens to contain today.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from build_app_validation_corpus import (  # noqa: E402
    _audit_built_corpus,
    _feed_parity_problems,
    _synthesized_fallback_problems,
    is_greeting_or_filler,
)

pytestmark = pytest.mark.unit

_REAL_SUMMARY = "Position sizing is the only risk control that survives an uncertain edge."
_REAL_INSIGHT = "Correlation between supposedly uncorrelated bets is the thing that ruins you."


# --- the shared rule --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Welcome back to the show, everyone, and thanks for tuning in again this week.",
        "welcome to another episode of the podcast where we get into the weeds",
        '  "Welcome back to the show" — with a leading quote and spaces',
        "You're listening to the show that takes risk seriously.",
        "I'm your host and today we have a fantastic guest lined up for you.",
        "Yeah, exactly. And it ties into what we're covering today.",
    ],
)
def test_openings_and_filler_are_recognised(text: str) -> None:
    assert is_greeting_or_filler(text) is True


@pytest.mark.parametrize(
    "text",
    [
        _REAL_SUMMARY,
        "The welcome mat problem in API design is that defaults become permanent.",
        "A great question to ask is whether the correlation survives a regime change.",
    ],
)
def test_real_content_is_not_mistaken_for_a_greeting(text: str) -> None:
    # "welcome" and "great question" MID-sentence must not trip the filter. A rule that eats real
    # content is a rule someone will disable, which is how the fixture ends up unchecked again.
    assert is_greeting_or_filler(text) is False


# --- the audit --------------------------------------------------------------------------------


def _corpus(
    tmp_path: Path,
    *,
    episodes: int = 24,
    summary: str = _REAL_SUMMARY,
    insight: str = _REAL_INSIGHT,
    duration_of: Callable[[int], float] | None = None,
    gi_ms_of: Callable[[int, float], float] | None = None,
    with_kg: bool = True,
) -> Path:
    """Write a small structurally-real corpus; every knob defaults to a HEALTHY value."""
    root = tmp_path / "v9"
    meta_dir = root / "feeds" / "p01" / "run_20260101-000000" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for i in range(episodes):
        stem = f"p01_e{i:02d}"
        seconds = duration_of(i) if duration_of else 300.0 + i * 17
        (meta_dir / f"{stem}.metadata.json").write_text(
            json.dumps(
                {
                    "episode": {"title": f"Episode {i}", "duration_seconds": seconds},
                    "summary": {"raw_text": summary},
                }
            ),
            encoding="utf-8",
        )
        (meta_dir / f"{stem}.gi.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "type": "Episode",
                            "properties": {
                                "duration_ms": (
                                    gi_ms_of(i, seconds) if gi_ms_of else seconds * 1000
                                )
                            },
                        },
                        {"type": "Insight", "properties": {"text": insight}},
                    ]
                }
            ),
            encoding="utf-8",
        )
        if with_kg:
            # A KG per episode is part of being HEALTHY, not an extra (#38). Without one, every
            # capture on that episode is withheld from the Revisit tab, Your Week and the digest —
            # so a corpus missing them is broken in a way no file count reveals. This fixture
            # claimed to be healthy while shipping none, and the new check caught it.
            (meta_dir / f"{stem}.kg.json").write_text(
                json.dumps(
                    {
                        "episode_id": stem,
                        "nodes": [
                            {
                                "id": "person:jane-doe",
                                "type": "Person",
                                "properties": {"name": "Jane"},
                            },
                            # One feed-wide topic AND one episode-specific one. A healthy corpus
                            # has both: shared topics are what make a show coherent, episode
                            # topics are what stop every show collapsing into a single theme
                            # cluster (#62.6). The fixture used to carry only the former.
                            {"id": "topic:ai", "type": "Topic", "properties": {"label": "AI"}},
                            {
                                "id": f"topic:ep-{i}",
                                "type": "Topic",
                                "properties": {"label": f"Theme {i}"},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
    return root


def test_a_healthy_corpus_passes(tmp_path: Path) -> None:
    assert _audit_built_corpus(_corpus(tmp_path)) == []


def test_one_duration_for_every_episode_is_caught(tmp_path: Path) -> None:
    """The v3 shape exactly: 1 distinct duration across the whole corpus."""
    problems = _audit_built_corpus(_corpus(tmp_path, duration_of=lambda _i: 1800.0))
    assert any("distinct value" in p for p in problems), problems


def test_a_duration_fixed_in_only_one_layer_is_caught(tmp_path: Path) -> None:
    """The specific way the v3 fix went wrong — metadata measured, GI still on the old constant.

    This is the check that matters most: the corpus reported itself fixed because the ONE layer
    anybody looked at was fixed.
    """
    problems = _audit_built_corpus(_corpus(tmp_path, gi_ms_of=lambda _i, _s: 1800.0 * 1000))
    assert any("gi Episode duration_ms" in p for p in problems), problems


def test_a_greeting_summary_is_caught(tmp_path: Path) -> None:
    problems = _audit_built_corpus(
        _corpus(tmp_path, summary="Welcome back to the show, everyone. Great to have you here.")
    )
    assert any("summary is a greeting" in p for p in problems), problems


def test_a_summary_that_only_restates_the_title_is_caught(tmp_path: Path) -> None:
    root = _corpus(tmp_path, episodes=1)
    path = next(root.rglob("*.metadata.json"))
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["summary"]["raw_text"] = doc["episode"]["title"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert any("restates the episode title" in p for p in _audit_built_corpus(root)), root


def test_a_greeting_INSIGHT_is_caught(tmp_path: Path) -> None:
    """The half the v3 summary fix never touched — 36/36 episodes led with the host's welcome.

    These are indexed and surface everywhere insights do, so a corpus like this cannot exercise an
    insight surface with anything an insight surface is for.
    """
    problems = _audit_built_corpus(
        _corpus(tmp_path, insight="Welcome back to the show — let's get into it.")
    )
    assert any("Insight is a greeting/filler" in p for p in problems), problems


def test_an_empty_corpus_is_a_failure_not_a_pass(tmp_path: Path) -> None:
    """A build that wrote nothing must not audit clean.

    That is the loudest possible false green.
    """
    assert _audit_built_corpus(tmp_path / "nothing-here") != []


def test_an_episode_with_no_knowledge_graph_is_caught(tmp_path: Path) -> None:
    """The KG is what a captured moment RESOLVES to (#38).

    Without one, `refs_for_highlight` returns [] and every revisit surface withholds the user's own
    capture — silently, because our pipeline missed a step. That is a build defect, and the loudest
    place to catch it is here rather than as an inexplicably empty Your Week weeks later.
    """
    problems = _audit_built_corpus(_corpus(tmp_path, with_kg=False))
    assert any("no knowledge graph" in p for p in problems), problems


def test_the_kg_check_counts_and_names_the_offenders(tmp_path: Path) -> None:
    """A bare "some episodes are broken" sends the reader back to the filesystem to work out
    WHICH — so the message carries the count and names the offenders."""
    root = _corpus(tmp_path, episodes=24)
    (root / "feeds" / "p01" / "run_20260101-000000" / "metadata" / "p01_e07.kg.json").unlink()
    problems = _audit_built_corpus(root)
    kg = next(p for p in problems if "no knowledge graph" in p)
    assert "1/24" in kg, kg
    assert "p01_e07" in kg, kg


# --- the summary must not be the transcript's opening (#58) --------------------------------------
#
# Check 2 pattern-matches greeting PHRASES — a guess about what junk looks like. This asks the
# structural question instead: is the summary just the top of the transcript? That is the shape v3
# shipped in 36/36 episodes, and it is what no client-side heuristic could catch: by the time the
# text reaches the player's reading dialog there is nothing left to compare it against.
#
# Honest scope: on the committed v3 corpus this finds no episode check 2 misses. Its value is that
# it does not depend on a phrase list — an echo that opens "So the thing about enduro racing is…"
# is invisible to check 2 and obvious to this one.


def _corpus_with_transcript(tmp_path: Path, *, summary: str, opening: str) -> Path:
    root = _corpus(tmp_path, episodes=1, summary=summary)
    meta_dir = root / "feeds" / "p01" / "run_20260101-000000" / "metadata"
    tx_dir = meta_dir.parent / "transcripts"
    tx_dir.mkdir(parents=True, exist_ok=True)
    (tx_dir / "p01_e00.segments.json").write_text(
        json.dumps([{"id": 0, "start": 0.0, "end": 6.0, "text": opening}]), encoding="utf-8"
    )
    return root


def test_a_summary_that_is_the_transcript_opening_is_caught(tmp_path: Path) -> None:
    opening = (
        "Welcome back to Singletrack Sessions. Today we're talking about enduro racing, and I'm "
        "joined by Sophie Lorenz."
    )
    problems = _audit_built_corpus(
        _corpus_with_transcript(tmp_path, summary=opening, opening=opening)
    )
    assert any("repeats the transcript's opening" in p for p in problems), problems


def test_an_echo_that_does_not_look_like_a_greeting_is_still_caught(tmp_path: Path) -> None:
    """The case check 2 cannot see. No "welcome", no "thanks for joining" — just the transcript."""
    opening = (
        "So the thing about enduro racing that nobody mentions is the descent, which is where the "
        "whole race is actually decided by tyre choice."
    )
    problems = _audit_built_corpus(
        _corpus_with_transcript(tmp_path, summary=opening, opening=opening)
    )
    assert any("repeats the transcript's opening" in p for p in problems), problems


def test_a_real_summary_beside_a_greeting_transcript_is_not_flagged(tmp_path: Path) -> None:
    """The false-positive guard. Every episode's transcript opens with a greeting; only a summary
    that COPIES it is a defect."""
    problems = _audit_built_corpus(
        _corpus_with_transcript(
            tmp_path,
            summary=_REAL_SUMMARY,
            opening="Welcome back to Singletrack Sessions. Today we talk enduro racing.",
        )
    )
    assert not any("repeats the transcript's opening" in p for p in problems), problems


def test_an_episode_with_no_transcript_is_not_flagged(tmp_path: Path) -> None:
    """Nothing to compare against is not evidence of junk."""
    problems = _audit_built_corpus(_corpus(tmp_path, episodes=1))
    assert not any("repeats the transcript's opening" in p for p in problems), problems


def test_an_echo_is_caught_through_case_and_whitespace_differences(tmp_path: Path) -> None:
    """Comparison is normalised, so a re-cased or re-spaced copy is still a copy.

    Without this the earlier echo tests could not tell: they use the SAME string for summary and
    transcript, so a raw `in` comparison passes them too. Sabotage confirmed it — dropping
    `_norm_text` from both sides left them green.
    """
    opening = "Welcome back to Singletrack Sessions. Today we're talking about enduro racing."
    reshaped = "welcome back    to Singletrack SESSIONS.\n Today we're talking about enduro racing."
    problems = _audit_built_corpus(
        _corpus_with_transcript(tmp_path, summary=reshaped, opening=opening)
    )
    assert any("repeats the transcript's opening" in p for p in problems), problems


def test_a_short_shared_phrase_is_not_enough_to_flag(tmp_path: Path) -> None:
    """The threshold does real work: a summary may legitimately open on the same few words.

    Pinned explicitly because the false-positive test above cannot see it — its summary happens to
    share no leading character with the transcript, so even a degenerate 1-character prefix fails
    to match it. That is an accident of the fixture text, not a property of the check.
    """
    opening = "Enduro racing is decided on the descent, and tyre choice is most of it."
    summary = "Enduro racing rewards preparation over talent — " + _REAL_SUMMARY
    problems = _audit_built_corpus(
        _corpus_with_transcript(tmp_path, summary=summary, opening=opening)
    )
    assert not any("repeats the transcript's opening" in p for p in problems), problems


# --- corpus-level checks: defects no per-episode assertion can see (#62) ------------------------
#
# Each of these lives in the RELATIONSHIP between artifacts or between episodes. Every single-file
# assertion passes on a corpus carrying all three, which is how v3 shipped with them.


def test_every_episode_carrying_only_feed_wide_topics_is_caught(tmp_path: Path) -> None:
    """v3's topics are per-FEED constants, so the corpus collapses to one theme cluster and
    Storylines degenerates to a single blob — while every episode does technically have topics."""
    root = _corpus(tmp_path, episodes=4)
    meta_dir = root / "feeds" / "p01" / "run_20260101-000000" / "metadata"
    for kg in meta_dir.glob("*.kg.json"):
        doc = json.loads(kg.read_text())
        doc["nodes"] = [n for n in doc["nodes"] if not str(n.get("id", "")).startswith("topic:ep-")]
        kg.write_text(json.dumps(doc), encoding="utf-8")
    problems = _audit_built_corpus(root)
    assert any("only feed-wide topics" in p for p in problems), problems


def test_a_publish_time_that_disagrees_across_layers_is_caught(tmp_path: Path) -> None:
    """v3 carried midnight in metadata and midday in the KG for the same episode.

    Nothing crashes on a 12-hour skew — "newest first" just means two different orders depending
    on which artifact a surface happens to read.
    """
    root = _corpus(tmp_path, episodes=1)
    meta_dir = root / "feeds" / "p01" / "run_20260101-000000" / "metadata"
    meta = json.loads((meta_dir / "p01_e00.metadata.json").read_text())
    meta.setdefault("episode", {})["published_date"] = "2024-10-21T00:00:00"
    (meta_dir / "p01_e00.metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    kg = json.loads((meta_dir / "p01_e00.kg.json").read_text())
    kg["publish_date"] = "2024-10-21T12:00:00"  # the skew, in the KG's own spelling
    (meta_dir / "p01_e00.kg.json").write_text(json.dumps(kg), encoding="utf-8")

    problems = _audit_built_corpus(root)
    assert any("publish time" in p and "disagrees" in p for p in problems), problems


def test_the_skew_check_finds_a_nested_stamp_under_any_publish_key(tmp_path: Path) -> None:
    """Searched by SHAPE, not by one known key — the layers disagree on the name AND the nesting.

    The first version looked for `published_datetime` at the top level, found it nowhere in the
    real corpus, and would have shipped as a permanently silent no-op. A check that cannot fail is
    worse than no check: it reads as coverage.
    """
    root = _corpus(tmp_path, episodes=1)
    meta_dir = root / "feeds" / "p01" / "run_20260101-000000" / "metadata"
    meta = json.loads((meta_dir / "p01_e00.metadata.json").read_text())
    meta.setdefault("episode", {})["published_date"] = "2024-10-21T00:00:00"
    (meta_dir / "p01_e00.metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    gi = json.loads((meta_dir / "p01_e00.gi.json").read_text())
    gi["data"] = {"episode": {"deeply": {"publish_datetime": "2024-10-21T12:00:00"}}}
    (meta_dir / "p01_e00.gi.json").write_text(json.dumps(gi), encoding="utf-8")

    assert any("publish time" in p for p in _audit_built_corpus(root))


def test_a_run_directory_the_search_layer_cannot_parse_is_caught(tmp_path: Path) -> None:
    """`corpus_scope._RUN_TS_RE` wants `run_YYYYMMDD-HHMMSS`. The builder writes an UNDERSCORE
    before the time, so nothing matches and run-recency ordering falls back to file mtime — a
    property of the checkout rather than of the corpus (#63)."""
    root = _corpus(tmp_path, episodes=1)
    bad = root / "feeds" / "p01" / "run_20260101_000000"
    (root / "feeds" / "p01" / "run_20260101-000000").rename(bad)
    problems = _audit_built_corpus(root)
    assert any("run-recency pattern" in p for p in problems), problems


def test_a_conforming_run_directory_is_not_flagged(tmp_path: Path) -> None:
    problems = _audit_built_corpus(_corpus(tmp_path, episodes=1))
    assert not any("run-recency pattern" in p for p in problems), problems


def test_a_feed_item_with_no_built_episode_is_caught(tmp_path: Path) -> None:
    """v3's feeds advertise 40 items against 36 built episodes (#62.5, #61).

    Nothing errors — the extra four simply describe episodes that do not exist, so anything reading
    the FEED sees a corpus four episodes larger than the one the app can open. Invisible to both
    sides alone, which is why the check has to hold them together.
    """
    root = _corpus(tmp_path, episodes=2)
    rss = tmp_path / "rss"
    rss.mkdir()
    (rss / "p01_corpus.xml").write_text(
        "<rss><channel>"
        "<item><guid>p01_e00</guid></item>"
        "<item><guid>p01_e01</guid></item>"
        "<item><guid>p01_e99</guid></item>"  # advertised, never built
        "</channel></rss>",
        encoding="utf-8",
    )
    problems = _feed_parity_problems(root, rss)
    assert any("never built" in p and "p01_e99" in p for p in problems), problems


def test_a_built_episode_in_no_feed_item_is_caught(tmp_path: Path) -> None:
    """The other direction: an episode the app can open that no feed advertises."""
    root = _corpus(tmp_path, episodes=2)
    rss = tmp_path / "rss"
    rss.mkdir()
    (rss / "p01_corpus.xml").write_text(
        "<rss><channel><item><guid>p01_e00</guid></item></channel></rss>", encoding="utf-8"
    )
    problems = _feed_parity_problems(root, rss)
    assert any("no feed item" in p and "p01_e01" in p for p in problems), problems


def test_a_matching_feed_is_not_flagged(tmp_path: Path) -> None:
    root = _corpus(tmp_path, episodes=2)
    rss = tmp_path / "rss"
    rss.mkdir()
    (rss / "p01_corpus.xml").write_text(
        "<rss><channel>"
        "<item><guid>https://example/p01_e00</guid></item>"  # guids may be URLs; the tail is the id
        "<item><guid>https://example/p01_e01</guid></item>"
        "</channel></rss>",
        encoding="utf-8",
    )
    assert _feed_parity_problems(root, rss) == []


def test_no_feed_directory_means_no_parity_claim(tmp_path: Path) -> None:
    """Absent input is not evidence of a defect — the feeds are a sibling fixture, not part of
    the corpus, so a corpus built without them is simply unaudited on this axis."""
    assert _feed_parity_problems(_corpus(tmp_path, episodes=1), tmp_path / "nope") == []


def test_a_pipeline_run_that_silently_fell_back_fails_the_build() -> None:
    """The report was always explicit and the build still exited 0 (#62.9).

    So "we ran the pipeline" stayed true on paper while a transcript opening line shipped as the
    summary — the v3 defect exactly, arriving through the door marked "we already report this".
    """
    problems = _synthesized_fallback_problems(
        ran_pipeline=True, synthesized=["p01_e00", "p01_e01"], total=4, allowed=False
    )
    assert problems and "fell back" in problems[0], problems


def test_the_fallback_can_be_accepted_deliberately() -> None:
    assert (
        _synthesized_fallback_problems(
            ran_pipeline=True, synthesized=["p01_e00"], total=4, allowed=True
        )
        == []
    )


def test_without_a_pipeline_run_the_stand_in_is_the_expected_output() -> None:
    """There is nothing to fall back FROM, so the stand-in is not a defect."""
    assert (
        _synthesized_fallback_problems(
            ran_pipeline=False, synthesized=["p01_e00"], total=1, allowed=False
        )
        == []
    )
