"""A host-spoken spelling the episode never states must not become a person (#2095, #876).

MEASURED ON A REAL FAILURE. Dwarkesh, "Grant Sanderson — AI and the future of maths": the host
says *"chatting with Drance Anderson, who runs the blue one Brown"* — the ASR's rendering of
Grant Sanderson and of 3Blue1Brown. The introduction reader bound the spoken spelling verbatim,
every downstream matcher then treated it as an extra person (``grant``/``drance`` is 3 edits,
past every tolerance this roster will defend), and the episode published 27 turns under a human
being who does not exist, while the correctly-spelled ``Grant Sanderson`` sat unbound in the
episode's own metadata.

The refusal is deliberately placed BEFORE the LLM merge. A voice absent from ``voice_intro`` is
one the resolver's closed-list answer can still reach, so refusing the reader's spelling does not
leave the voice nameless — it hands it back to the path that gets it right.
"""

import pytest

from podcast_scraper.providers.ml.diarization.roster import (
    _refuse_unstated_introductions,
    _resembles_stated,
)


class TestTheSpellingIsRecognisedAsAMangling:
    """`_resembles_stated` only ever REFUSES a name, so it is permissive on purpose — but it must
    still tell a mangled spelling from a different human being."""

    @pytest.mark.parametrize(
        "spoken,stated",
        [
            ("Drance Anderson", "Grant Sanderson"),  # the case this exists for
            ("Dara Khadrshahid", "Dara Khosrowshahi"),
            ("Danny Stockman", "Daniela Stockmann"),
            ("Anant Nageswaram", "Dr. V Anantha Nageswaran"),
        ],
    )
    def test_a_mangling_of_a_stated_person_is_recognised(self, spoken, stated) -> None:
        assert _resembles_stated(spoken, [stated]) == stated

    @pytest.mark.parametrize(
        "spoken,stated",
        [
            # A real person the metadata simply omits. No stated name resembles them, so the
            # binding stands — about half the corpus's 75 unstated reader bindings are this
            # shape, and refusing them would delete correct names for nothing.
            ("David Sanger", "Mackenzie Price"),
            ("Chase Harrison", "Alexander Stubb"),
            ("Mati Staniszewski", "Sarah Guo"),
        ],
    )
    def test_an_unrelated_name_is_not_called_a_mangling(self, spoken, stated) -> None:
        assert _resembles_stated(spoken, [stated]) is None

    def test_a_shared_surname_DOES_resemble_and_that_is_deliberate(self) -> None:
        """Documenting a place this is knowingly loose. "Robert Pape" and "Karen Pape" are two
        different people, and the BINDING matchers keep them apart (`_same_person` refuses). Here
        they resemble, so if an episode stated Karen and the host introduced a Robert, his name
        would be refused rather than published.

        That is the intended trade under #876. This fires only where the reader produced a name
        the episode never states while the stated person is still unplaced — evidence that is
        already weak — and the cost of refusing is an unnamed voice the LLM may still name, while
        the cost of publishing is a real person credited with words they never said.
        """
        assert _resembles_stated("Robert Pape", ["Karen Pape"]) == "Karen Pape"


class TestTheRefusal:
    def test_the_unstated_spelling_is_dropped_and_the_voice_reported(self) -> None:
        """Drance Anderson goes; the voice comes back so its guest ROLE can survive the refusal."""
        out = {"SPEAKER_01": "Drance Anderson"}
        refused = _refuse_unstated_introductions(
            out,
            stated_persons=["Grant Sanderson"],
            known_hosts=["Dwarkesh Patel"],
            voice_intro={},
        )
        assert out == {}, "a name the episode never states must not be published"
        assert refused == {"SPEAKER_01"}, "the host did introduce somebody — keep the role"

    def test_a_stated_name_is_never_refused(self) -> None:
        """The good case. The reader usually gets it right (741 of 1,458 bindings)."""
        out = {"SPEAKER_01": "Grant Sanderson"}
        refused = _refuse_unstated_introductions(out, ["Grant Sanderson"], ["Dwarkesh Patel"], {})
        assert out == {"SPEAKER_01": "Grant Sanderson"}
        assert refused == set()

    def test_a_show_that_states_nobody_is_left_alone(self) -> None:
        """549 of the reader's bindings are this shape — Planet Money and friends name no one in
        metadata, so there is nothing to resemble and the introduction is the only evidence there
        is. Refusing here would delete names with no better answer to replace them."""
        out = {"SPEAKER_01": "Robert Smith"}
        refused = _refuse_unstated_introductions(out, [], ["Planet Money"], {})
        assert out == {"SPEAKER_01": "Robert Smith"}
        assert refused == set()

    def test_nothing_is_refused_once_the_stated_guest_is_bound(self) -> None:
        """The refusal's whole warrant is that the stated person is still unplaced. Once some voice
        holds that name, an unstated extra name is a SECOND person — possibly a real one — and
        this must not touch it."""
        out = {"SPEAKER_02": "Drance Anderson"}
        refused = _refuse_unstated_introductions(
            out,
            stated_persons=["Grant Sanderson"],
            known_hosts=["Dwarkesh Patel"],
            voice_intro={"SPEAKER_01": "Grant Sanderson"},
        )
        assert out == {"SPEAKER_02": "Drance Anderson"}
        assert refused == set()

    def test_a_known_host_does_not_count_as_the_unbound_person(self) -> None:
        """Only a stated NON-host warrants the refusal. The host is seated by other evidence
        entirely, and letting an unbound host justify deleting a guest's name would re-open the
        N1 hole (a guest must never be renamed toward a host)."""
        out = {"SPEAKER_01": "Kevin Ross"}
        refused = _refuse_unstated_introductions(
            out, stated_persons=["Kevin Roose"], known_hosts=["Kevin Roose"], voice_intro={}
        )
        assert out == {"SPEAKER_01": "Kevin Ross"}, "no stated non-host is unbound here"
        assert refused == set()
