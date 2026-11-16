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

from podcast_scraper.workflow.metadata_generation import (
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
