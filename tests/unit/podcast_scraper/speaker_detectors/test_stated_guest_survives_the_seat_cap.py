"""A stated guest must not be deleted by the SCREENPLAY's seat count (#2095, #2078).

MEASURED ON THE CONTROL CORPUS. ``_build_speaker_names_list`` applied
``cfg.screenplay_num_speakers`` — how many seats the screenplay renders, default 2 — to the list
of people the episode STATES:

    speaker_names = list(hosts)[:max_names] + guests[: max_names - len(hosts)]

On any show with two feed-stated hosts the guest slice is empty. The guest then reached neither
``detected_guests`` nor ``metadata_named``, and because the record builder reads only those, the
person was absent from ``content.speakers`` altogether — not even carried as ``placed: false``.

``Mackenzie Price`` appeared in NO record file in the entire control corpus, on a Hard Fork
episode whose own transcript says *"So let's bring in Alpha School cofounder, Mackenzie Price"*
followed by *"Thanks for having me"*. Same shape for ``Alexander Stubb`` and ``Glenn Fogel``.

The invariant this pins: a stated name may enter the record unplaced and may be bound by
introduction / self-intro / sign-off evidence. Only the ARITHMETIC path ("one spare name, one
spare voice") needs a bounded list, and the screenplay's seat count belongs to the screenplay.
"""

from podcast_scraper.speaker_detectors import detection


class TestTheScreenplaySeatCountDoesNotDeletePeople:
    def test_the_guest_survives_when_two_hosts_fill_every_seat(self) -> None:
        """The exact Hard Fork shape that lost her."""
        names, succeeded, _ = detection._build_speaker_names_list(
            {"Kevin Roose", "Casey Newton"}, ["Mackenzie Price"], 2
        )
        assert succeeded
        assert "Mackenzie Price" in names, (
            "two stated hosts filled the old cap and the guest slice was empty, so the person "
            "the episode introduces by name never reached the record at all"
        )
        for host in ("Kevin Roose", "Casey Newton"):
            assert host in names, "and the hosts must not be traded away for the guest either"

    def test_several_guests_all_survive(self) -> None:
        """The Rest Is Politics: Leading states two guests; both were lost, not one."""
        names, _, _ = detection._build_speaker_names_list(
            {"Rory Stewart", "Alastair Campbell"},
            ["Robert Malley", "Mark Williams"],
            2,
        )
        assert {"Robert Malley", "Mark Williams"} <= set(names)

    def test_a_single_host_show_is_unchanged(self) -> None:
        """The common shape must behave exactly as before — this is not a widening for its own
        sake, and one host plus one guest never hit the cap."""
        names, _, _ = detection._build_speaker_names_list(
            {"Dwarkesh Patel"}, ["Grant Sanderson"], 2
        )
        assert names == ["Dwarkesh Patel", "Grant Sanderson"]

    def test_detection_still_reports_failure_with_nobody(self) -> None:
        """Uncapping must not turn 'found nobody' into a success — the placeholder contract that
        #876 depends on is untouched."""
        names, succeeded, used_defaults = detection._build_speaker_names_list(set(), [], 2)
        assert not succeeded
        assert used_defaults
        assert names == detection.DEFAULT_SPEAKER_NAMES
