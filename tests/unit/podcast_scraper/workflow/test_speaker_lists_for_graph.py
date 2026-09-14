"""Merge rules for the host/guest lists handed to the graph (#2062).

The diarization roster listened to the episode; ``detected_hosts`` / ``detected_guests`` only read
the show notes, before any audio was processed. Host and guest are SPEAKING roles, so when a roster
exists it is the only source consulted; the hint is a fallback for episodes that have no roster,
not a supplement to one.

This file first pinned the opposite rule — an additive merge that kept hint-only names so a guest
the roster could not place would not be dropped. Production refuted it: on episodes where
diarization named every voice it heard, 19.4% of host nodes and 14.3% of guest nodes belong to
someone who did not speak — a co-host who sat the episode out, or the show's own name as a person.
(Counting every episode gives a bigger number, but a partial roster is silent about its anonymous
voices rather than denying them, so that is an upper bound.) The tests below pin the corrected
rule and the reason it changed.
"""

from __future__ import annotations

import pytest

from podcast_scraper.identity.roster_provenance import (
    build_content_speakers_block,
    roster_provenance,
    roster_source,
)
from podcast_scraper.workflow.metadata_generation import (
    _speaker_lists_for_graph,
    ContentMetadata,
    EpisodeMetadata,
    EpisodeMetadataDocument,
    FeedMetadata,
    ProcessingMetadata,
    PROVENANCE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SpeakerInfo,
)

pytestmark = pytest.mark.unit


def _sp(name: str, role: str, sid: str | None = None) -> SpeakerInfo:
    return SpeakerInfo(id=sid or role, name=name, role=role)


class TestTheRosterWins:
    def test_roster_guest_survives_an_empty_hint(self) -> None:
        # The prod shape: a network feed whose pre-diarization hint knows nobody.
        hosts, guests = _speaker_lists_for_graph(
            [_sp("Patrick O'Shaughnessy", "host"), _sp("Brian Chesky", "guest")], [], []
        )
        assert hosts == ["Patrick O'Shaughnessy"]
        assert guests == ["Brian Chesky"]

    def test_roster_guest_beats_a_hint_that_calls_them_a_host(self) -> None:
        hosts, guests = _speaker_lists_for_graph(
            [_sp("Brian Chesky", "guest")], ["Brian Chesky"], []
        )
        assert guests == ["Brian Chesky"]
        assert hosts == []

    def test_roster_host_beats_a_hint_that_calls_them_a_guest(self) -> None:
        hosts, guests = _speaker_lists_for_graph(
            [_sp("Patrick O'Shaughnessy", "host")], [], ["Patrick O'Shaughnessy"]
        )
        assert hosts == ["Patrick O'Shaughnessy"]
        assert guests == []


class TestTheHintIsOnlyUsedWhenThereIsNoRoster:
    def test_a_name_the_roster_never_heard_is_not_published_as_a_guest(self) -> None:
        # THE 40% CASE. The hint names someone the roster never heard. They did not speak, so they
        # are not a guest of this episode. Extraction still records them as `mentioned` if the
        # transcript mentions them — which is the truthful role.
        hosts, guests = _speaker_lists_for_graph(
            [_sp("Patrick O'Shaughnessy", "host")], [], ["Someone On Tape"]
        )
        assert guests == []
        assert hosts == ["Patrick O'Shaughnessy"]

    def test_a_co_host_who_sat_the_episode_out_is_not_a_host_of_it(self) -> None:
        # "Sarah Guo" on an episode where Elad Gil interviews Glenn Fogel — a real prod case.
        hosts, guests = _speaker_lists_for_graph(
            [_sp("Elad Gil", "host"), _sp("Glenn Fogel", "guest")], ["Sarah Guo"], []
        )
        assert hosts == ["Elad Gil"]
        assert guests == ["Glenn Fogel"]
        assert "Sarah Guo" not in hosts + guests

    def test_no_roster_at_all_falls_back_entirely_to_the_hint(self) -> None:
        hosts, guests = _speaker_lists_for_graph(None, ["A Host"], ["A Guest"])
        assert (hosts, guests) == (["A Host"], ["A Guest"])

    def test_everything_empty_yields_empty(self) -> None:
        assert _speaker_lists_for_graph(None, None, None) == ([], [])


class TestHygiene:
    def test_duplicates_collapse_case_insensitively_first_spelling_wins(self) -> None:
        hosts, guests = _speaker_lists_for_graph(
            [_sp("Brian Chesky", "guest")], [], ["brian chesky", "BRIAN CHESKY"]
        )
        assert guests == ["Brian Chesky"]

    def test_blank_names_are_dropped(self) -> None:
        hosts, guests = _speaker_lists_for_graph([_sp("  ", "guest")], ["  "], [""])
        assert (hosts, guests) == ([], [])

    def test_order_is_preserved(self) -> None:
        hosts, guests = _speaker_lists_for_graph(
            [
                _sp("G1", "guest", "guest_1"),
                _sp("G2", "guest", "guest_2"),
                _sp("G3", "guest", "guest_3"),
            ],
            [],
            [],
        )
        assert guests == ["G1", "G2", "G3"]

    def test_hint_order_is_preserved_when_there_is_no_roster(self) -> None:
        hosts, guests = _speaker_lists_for_graph(None, ["H1", "H2"], ["G1", "G2"])
        assert (hosts, guests) == (["H1", "H2"], ["G1", "G2"])

    def test_a_roster_voice_with_no_usable_role_is_treated_as_host(self) -> None:
        # Matches _build_speakers_from_diarized_segments' own fallback, so the two agree.
        hosts, guests = _speaker_lists_for_graph([_sp("Nobody Knows", "")], [], [])
        assert hosts == ["Nobody Knows"]


class TestEveryCallerPassesTheFeedTitle:
    """`_speaker_lists_for_graph` can only refuse a show name when it KNOWS the title (advisor).

    `names_the_show(candidate, feed_title)` returns False on an empty title by design — "no title
    means no opinion". So a caller that omits `feed_title` silently disables the guard it is
    calling the function for. The enrich-edges path in `search/cli_handlers.py` did exactly that.

    The review called it Low, reasoning a show name rarely appears as a literal `<Name>:` marker.
    Measured, the gap is wider than that: WITHOUT the title every show name is kept.

        name                      no title    with title
        Africa Tech Summit        KEPT        refused
        Machine Learning Street   KEPT        refused
        Latent.Space              KEPT        refused
        Kevin Roose (real host)   KEPT        KEPT

    Two code paths giving two different answers to "is this a person" is the drift this arc keeps
    paying for, so the caller now passes it.
    """

    @staticmethod
    def _roster(name: str, role: str = "host"):
        from types import SimpleNamespace

        return [SimpleNamespace(name=name, role=role)]

    def test_without_a_title_the_show_name_survives(self) -> None:
        hosts, _g = _speaker_lists_for_graph(self._roster("Africa Tech Summit"), [], [])
        assert "Africa Tech Summit" in hosts, "documents WHY the title is required, not optional"

    def test_with_the_title_the_show_name_is_refused(self) -> None:
        hosts, _g = _speaker_lists_for_graph(
            self._roster("Africa Tech Summit"), [], [], feed_title="Africa Tech Summit Podcast"
        )
        assert "Africa Tech Summit" not in hosts

    def test_a_real_host_is_kept_either_way(self) -> None:
        for kwargs in ({}, {"feed_title": "Hard Fork"}):
            hosts, _g = _speaker_lists_for_graph(self._roster("Kevin Roose"), [], [], **kwargs)
            assert "Kevin Roose" in hosts

    def test_the_enrich_edges_caller_passes_it(self) -> None:
        # Guards the CALL SITE. A unit test of the function cannot catch an omission at a caller,
        # and this omission is invisible: the function returns a plausible answer either way.
        import inspect

        from podcast_scraper.search import cli_handlers

        src = inspect.getsource(cli_handlers)
        idx = src.index("_speaker_lists_for_graph(")
        assert (
            "feed_title" in src[idx : idx + 500]
        ), "enrich-edges must pass feed_title or the show-name guard is inert on that path"


class TestTheArtifactRecordsWhereTheRosterCameFrom:
    """`content.speakers` must say whether it is a MEASUREMENT or a FALLBACK (#2070).

    `metadata_generation.py`::

        speakers = diarized_speakers or _build_speakers_from_detected_names(detected_hosts, ...)

    **This is #2065's root cause.** The real diarization roster and the pre-diarization HINT — read
    from the feed and the show notes before a second of audio is processed — land in the same field
    and are indistinguishable on disk. That is precisely why the graph could be fed the hint for
    months while nobody could tell by looking at an artifact: there was nothing in the artifact to
    look at.

    Every downstream reader then treats a guess as evidence. `_speaker_lists_for_graph`'s own
    docstring says "the roster is the ONLY source used when it heard the episode" — a rule it
    cannot actually enforce, because it cannot tell which it was handed.

    The fix is one field, written where the choice is made. A reader that cares can then refuse to
    treat a hint as a roster; one that does not is unaffected.
    """

    def test_a_diarized_roster_is_labelled_diarized(self) -> None:
        assert (
            roster_source(_speaker_infos_like([{"name": "A", "role": "host"}]), diarized=True)
            == "diarized"
        )

    def test_a_hint_fallback_is_labelled_hint(self) -> None:
        roster = _speaker_infos_like([{"name": "A", "role": "host"}])
        assert roster_source(roster, diarized=False) == "hint"

    def test_an_empty_roster_has_no_source_to_claim(self) -> None:
        assert roster_source([], diarized=True) is None

    def test_the_field_lands_on_the_artifact(self) -> None:
        # The property that matters: a reader can tell the two apart from the artifact ALONE,
        # which is exactly what does not hold today.
        content = build_content_speakers_block(
            speakers=[{"name": "A", "role": "host"}], diarized=True, num_speakers=2
        )
        assert content["speakers_source"] == "diarized"
        hint = build_content_speakers_block(
            speakers=[{"name": "A", "role": "host"}], diarized=False, num_speakers=None
        )
        assert hint["speakers_source"] == "hint"

    def test_an_unlabelled_artifact_reads_as_unknown_not_as_trusted(self) -> None:
        # Every artifact already on disk predates the field. Unknown must be its own answer —
        # treating it as "diarized" is the assumption that produced #2065.
        assert (
            roster_provenance({"content": {"speakers": [{"name": "A", "role": "host"}]}})
            == "unknown"
        )

    def test_a_labelled_artifact_reads_back(self) -> None:
        assert (
            roster_provenance({"content": {"speakers": [{"name": "A"}], "speakers_source": "hint"}})
            == "hint"
        )

    def test_no_speakers_at_all_is_its_own_answer(self) -> None:
        assert roster_provenance({"content": {"speakers": []}}) == "absent"


def _speaker_infos_like(rows):
    from types import SimpleNamespace

    return [SimpleNamespace(name=r["name"], role=r["role"]) for r in rows]


class TestTheWriterCannotEmitARosterWithoutSayingWhereItCameFrom:
    """The grandfather clause as a MECHANISM, not a comment (#2070).

    Writing `speakers_source` is easy; keeping it written is the hard part. A comment saying
    "always set this" rots, and #2070 exists precisely because nobody could tell a measurement
    from a fallback by reading an artifact.

    So the rule is pinned to the SCHEMA VERSION and enforced on the document model, which is the
    one place that holds both the version and the roster. The writer then cannot ship a versioned
    artifact without provenance, and the check is a test that fails rather than a convention
    someone has to remember.

    The grandfathered population is bounded and named: artifacts below `PROVENANCE_SCHEMA_VERSION`
    predate the field and are legal without it — that is every artifact currently on disk.
    `roster_provenance` reports those as `unknown`, its own answer, never upgraded to `diarized`.
    """

    @staticmethod
    def _doc(version: str, speakers, source=None):
        from datetime import datetime

        return EpisodeMetadataDocument(
            feed=FeedMetadata(feed_id="f", title="T", url="https://e.com/f.xml"),
            episode=EpisodeMetadata(title="E", episode_id="e1"),
            content=ContentMetadata(speakers=speakers, speakers_source=source),
            processing=ProcessingMetadata(
                processing_timestamp=datetime(2026, 9, 14),
                output_directory="/tmp",
                schema_version=version,
            ),
        )

    def test_a_versioned_roster_without_provenance_is_refused(self) -> None:
        with pytest.raises(ValueError, match="speakers_source"):
            self._doc(PROVENANCE_SCHEMA_VERSION, [SpeakerInfo(id="host", name="A", role="host")])

    def test_a_versioned_roster_with_provenance_is_accepted(self) -> None:
        doc = self._doc(
            PROVENANCE_SCHEMA_VERSION,
            [SpeakerInfo(id="host", name="A", role="host")],
            source="diarized",
        )
        assert doc.content.speakers_source == "diarized"

    def test_an_older_artifact_is_grandfathered(self) -> None:
        # Every artifact already on disk. Legal without the field — and `roster_provenance`
        # reports it as `unknown`, not as trusted.
        doc = self._doc("1.0.0", [SpeakerInfo(id="host", name="A", role="host")])
        assert doc.content.speakers_source is None

    def test_an_empty_roster_needs_no_provenance_at_any_version(self) -> None:
        # No roster, no origin to claim. Requiring one would invent provenance for a value that
        # does not exist.
        doc = self._doc(PROVENANCE_SCHEMA_VERSION, [])
        assert doc.content.speakers_source is None

    def test_an_unknown_source_value_is_refused(self) -> None:
        with pytest.raises(ValueError):
            ContentMetadata(
                speakers=[SpeakerInfo(id="host", name="A", role="host")],
                # deliberately invalid — the point is that pydantic refuses it at runtime,
                # which mypy cannot express as a passing call.
                speakers_source="probably-diarized",  # type: ignore[arg-type]
            )

    def test_the_live_writer_version_is_at_or_above_the_gate(self) -> None:
        # The gate is inert until SCHEMA_VERSION reaches it. This asserts the bump happened, so
        # the mechanism is ON rather than merely defined.
        assert SCHEMA_VERSION >= PROVENANCE_SCHEMA_VERSION

    @pytest.mark.parametrize(
        "version,gated",
        [
            ("1.0.0", False),
            ("1.1.0", True),
            ("1.2.0", True),
            # THE CASE A STRING COMPARE GETS WRONG. "1.9.0" >= "1.1.0" is True as strings and
            # also True as versions, so it proves nothing on its own — but "1.10.0" is the pair
            # that separates them in the other direction, and a future gate pinned at "1.9.0"
            # would silently stop firing at "1.10.0" under lexicographic ordering.
            ("1.9.0", True),
            ("1.10.0", True),
            ("2.0.0", True),
            # Not a version at all: grandfathered, not crashed. An artifact that cannot state its
            # schema must still load.
            ("", False),
            ("not-a-version", False),
        ],
    )
    def test_the_gate_uses_version_ordering_not_string_ordering(
        self, version: str, gated: bool
    ) -> None:
        speakers = [SpeakerInfo(id="host", name="A", role="host")]
        if gated:
            with pytest.raises(ValueError, match="speakers_source"):
                self._doc(version, speakers)
        else:
            assert self._doc(version, speakers).content.speakers_source is None

    def test_a_string_compare_would_disagree_with_this_gate(self) -> None:
        """THE REMOVAL PROOF for the parsed comparison.

        If someone "simplifies" the validator back to `version >= PROVENANCE_SCHEMA_VERSION`,
        this records what breaks: lexicographic ordering puts "1.9.0" above "1.10.0", so a gate
        pinned at a double-digit minor stops firing for every lower-numbered-but-newer release.
        """
        from packaging.version import Version

        assert "1.9.0" >= "1.10.0", "string ordering: 9 sorts above 1"
        assert Version("1.9.0") < Version("1.10.0"), "version ordering disagrees — that is the bug"
